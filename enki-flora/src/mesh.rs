//! Branch mesh — Rust port of dryad `branchMesh.js` (`buildBranchGeometry`).
//!
//! Walks `graph.bones` into a single merged tapered-tube mesh: one ring of
//! vertices per skeleton node, parallel-transport (rotation-minimizing) frames
//! along each branch chain, ring-to-ring quads, and a collapsed apex at each
//! childless tip.
//!
//! Scope is deliberately the smallest thing that renders one lit tree:
//!
//! HIERARCHICAL SKELETAL WIND (1:1 dryad). The dryad source bakes per-vertex
//! `boneIndex` / `boneFraction` plus the whole `bones_wind` chain hierarchy +
//! `nodeToBone` table; all of that is ported here now. ONE wind bone per branch
//! chain (bone index == chain index). The bake is PURELY ADDITIVE — it writes
//! only into the new `bone_index` / `bone_fraction` arrays inside the existing
//! `emit_ring` / `write_vertex` calls and never perturbs positions / normals /
//! uvs / ao / radii / indices or their order, so the existing determinism test
//! (`seed_42_mesh_is_nonempty_finite_in_range_and_deterministic`) stays green.
//! ponytail: per-node `windWeight` is NOT baked per-vertex (the dryad shader's
//! final form does not read it — see windSkinGlsl.js); it is only meaned into
//! per-bone `stiffness`, so only that aggregate is computed here.
//!
//! ponytail: SDF / skin path dropped (legacy dryad material, not used by enki).
//!
//! ponytail: cross-section RIB/FLUTE modulation (`ringProfile`, `ribbing`,
//! `flatness`, `ribCount`) dropped — rings are perfect circles. This is the
//! identity case of dryad's `ringProfile`, so the geometry matches dryad's
//! default (ribbing=0, flatness=0) output exactly. Refine if bark relief needs
//! geometric ribs rather than shader-only relief.
//!
//! `radii` per-vertex attribute carries the TRUE local tube radius per vertex —
//! the dryad bark shader needs it as `barkFeatureScale` (furrows sized to actual
//! thickness, consistent around the trunk). The integration packs it into the uv
//! stream's `.z` (an otherwise-unused pad). See `BranchMesh.radii`.
//!
//! Math is done in `[f64; 3]` (mirroring the JS array math term-for-term) and
//! narrowed to `f32` only at output, so the buffers are deterministic.

use crate::skeleton::Graph;

type V3 = [f64; 3];

// ---------------------------------------------------------------------------
// Math helpers (mirror branchMesh.js v3* helpers term-for-term).
// ---------------------------------------------------------------------------

#[inline]
fn v3len(v: V3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn v3norm(v: V3) -> V3 {
    let len = v3len(v);
    if len < 1e-12 {
        return [1.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

#[inline]
fn v3dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn v3cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn v3sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Stable perpendicular to a unit vector (matches skeleton.js `perpTo`).
fn perp_to(dir: V3) -> V3 {
    if v3dot(dir, [1.0, 0.0, 0.0]).abs() < 0.9 {
        v3norm(v3cross(dir, [1.0, 0.0, 0.0]))
    } else {
        v3norm(v3cross(dir, [0.0, 0.0, 1.0]))
    }
}

/// Rotation-minimizing transport of frame basis `{u, v}` from `prev_tangent` to
/// `new_tangent` (Rodrigues rotation). Returns the updated `(u, v)`.
fn transport_frame(prev_tangent: V3, new_tangent: V3, u: V3, v: V3) -> (V3, V3) {
    let d = v3dot(prev_tangent, new_tangent).clamp(-1.0, 1.0);
    if d > 1.0 - 1e-10 {
        return (u, v); // already aligned
    }
    let axis_raw = v3cross(prev_tangent, new_tangent);
    let axis_len = v3len(axis_raw);
    if axis_len < 1e-12 {
        // anti-parallel: flip both basis vectors
        return ([-u[0], -u[1], -u[2]], [-v[0], -v[1], -v[2]]);
    }
    let ax = [axis_raw[0] / axis_len, axis_raw[1] / axis_len, axis_raw[2] / axis_len];
    let angle = d.acos();
    let c = angle.cos();
    let s = angle.sin();

    let rot = |vec: V3| -> V3 {
        let dt = v3dot(ax, vec);
        let cr = v3cross(ax, vec);
        [
            vec[0] * c + cr[0] * s + ax[0] * dt * (1.0 - c),
            vec[1] * c + cr[1] * s + ax[1] * dt * (1.0 - c),
            vec[2] * c + cr[2] * s + ax[2] * dt * (1.0 - c),
        ]
    };

    (v3norm(rot(u)), v3norm(rot(v)))
}

/// Default radial segment count by branch level (dryad `defaultRadialSegs`).
#[inline]
fn radial_segs_for(level: i32) -> usize {
    if level <= 0 {
        16
    } else if level == 1 {
        10
    } else if level == 2 {
        7
    } else {
        5
    }
}

/// Minimum ring radius in world units (dryad `opts.minRadius` default).
const MIN_RADIUS: f64 = 0.004;

/// Wind-bone budget ceiling (dryad `MAX_WIND_BONES`). The realistic worst case is
/// ~300-550 chains, well under this — it's the storage-buffer sizing cap.
pub const MAX_WIND_BONES: usize = 1024;

// ---------------------------------------------------------------------------
// Output type.
// ---------------------------------------------------------------------------

/// Axis-aligned bounds of the emitted mesh (degenerate at origin when empty).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

/// One wind bone (== one branch chain). Mirror of dryad `bones_wind` columns.
///
/// - `parent`: bone index of the chain that owns this chain's fork-attachment
///   node, or `-1` for a hierarchy root. INVARIANT `parent < self_index` (the
///   solver's top-down precondition; chains are collected in ascending node
///   order so a fork-child's bone index always exceeds its fork-parent's).
/// - `pivot`: the FIRST chain node's world pos — the fork-attachment point the
///   bone hinges about (kept pinned via the parent's composition).
/// - `axis`: unit tangent from the real start toward the next node (axis hint;
///   the solver uses a wind-derived axis, this is informational/parity).
/// - `stiffness`: mean per-node wind weight over the real chain nodes (`0` when
///   rigid). Scales the solver's per-bone rotation angle.
/// - `branch_level`: branch level of the real start node.
/// - `is_rigid`: `1` for the trunk base chain (index 0) and root-flare chains —
///   their local rotation is forced to identity so the tree doesn't swing about
///   the origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindBone {
    pub parent: i32,
    pub pivot: [f32; 3],
    pub axis: [f32; 3],
    pub stiffness: f32,
    pub branch_level: u32,
    pub is_rigid: u8,
}

/// Merged tapered-tube branch mesh. SoA, f32, ready for GPU upload.
///
/// - `positions`: `3 * vertex_count` (xyz per vertex)
/// - `normals`:   `3 * vertex_count` (outward tube normal per vertex)
/// - `uvs`:       `2 * vertex_count` (u = angle/segs, v = world arc length)
/// - `ao`:        `vertex_count` (ambient occlusion in [0.35, 1.0])
/// - `radii`:     `vertex_count` (true local tube radius, `MIN_RADIUS`-clamped —
///                bark feature-size hint; integration packs into uv.z)
/// - `indices`:   `3 * triangle_count` (u32, CCW from outside)
/// - `bone_index`: `vertex_count` (wind-bone index per vertex; chain index)
/// - `bone_fraction`: `vertex_count` (normalized arc from pivot [0..1]; 0 at the
///                pivot/prepended ring, 1 at the chain tip)
/// - `bones_wind`: the per-chain wind hierarchy (one [`WindBone`] per chain)
/// - `node_to_bone`: `n_nodes` (each graph node → its owning bone, or `-1`)
#[derive(Clone, Debug, PartialEq)]
pub struct BranchMesh {
    pub positions: Vec<f32>,
    pub normals: Vec<f32>,
    pub uvs: Vec<f32>,
    pub ao: Vec<f32>,
    pub radii: Vec<f32>,
    pub indices: Vec<u32>,
    pub vertex_count: u32,
    pub triangle_count: u32,
    pub bounds: Bounds,
    pub bone_index: Vec<f32>,
    pub bone_fraction: Vec<f32>,
    pub bones_wind: Vec<WindBone>,
    pub node_to_bone: Vec<i32>,
}

impl BranchMesh {
    fn empty() -> Self {
        BranchMesh {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            ao: Vec::new(),
            radii: Vec::new(),
            indices: Vec::new(),
            vertex_count: 0,
            triangle_count: 0,
            bounds: Bounds { min: [0.0; 3], max: [0.0; 3] },
            bone_index: Vec::new(),
            bone_fraction: Vec::new(),
            bones_wind: Vec::new(),
            node_to_bone: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Builder scratch state (mirrors the closures in branchMesh.js).
// ---------------------------------------------------------------------------

struct Builder {
    positions: Vec<f32>,
    normals: Vec<f32>,
    uvs: Vec<f32>,
    ao: Vec<f32>,
    radii: Vec<f32>,
    indices: Vec<u32>,
    // ── Wind bake (purely additive — written alongside every vertex push). ──
    bone_index: Vec<f32>,
    bone_fraction: Vec<f32>,
    v_cursor: u32,
    min: V3,
    max: V3,
}

impl Builder {
    fn new() -> Self {
        Builder {
            positions: Vec::new(),
            normals: Vec::new(),
            uvs: Vec::new(),
            ao: Vec::new(),
            radii: Vec::new(),
            indices: Vec::new(),
            bone_index: Vec::new(),
            bone_fraction: Vec::new(),
            v_cursor: 0,
            min: [f64::INFINITY; 3],
            max: [f64::NEG_INFINITY; 3],
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn write_vertex(
        &mut self,
        p: V3,
        n: V3,
        u: f64,
        v: f64,
        ao_val: f64,
        radius: f64,
        bone_idx: f64,
        bone_frac: f64,
    ) -> u32 {
        let vi = self.v_cursor;
        self.positions.push(p[0] as f32);
        self.positions.push(p[1] as f32);
        self.positions.push(p[2] as f32);
        self.normals.push(n[0] as f32);
        self.normals.push(n[1] as f32);
        self.normals.push(n[2] as f32);
        self.uvs.push(u as f32);
        self.uvs.push(v as f32);
        self.ao.push(ao_val as f32);
        self.radii.push(radius as f32);
        self.bone_index.push(bone_idx as f32);
        self.bone_fraction.push(bone_frac as f32);
        for k in 0..3 {
            if p[k] < self.min[k] {
                self.min[k] = p[k];
            }
            if p[k] > self.max[k] {
                self.max[k] = p[k];
            }
        }
        self.v_cursor += 1;
        vi
    }

    /// Emit a ring of `segs` vertices at `pos` with radius `r` and frame `{u, v}`.
    /// Returns the index of the first vertex of the ring.
    /// ponytail: ribbing/flatness identity case — perfect circle, analytic radial
    /// normal `cosθ·u + sinθ·v`.
    #[allow(clippy::too_many_arguments)]
    fn emit_ring(
        &mut self,
        pos: V3,
        r: f64,
        u: V3,
        v: V3,
        segs: usize,
        arc_len: f64,
        ao_val: f64,
        bone_idx: f64,
        bone_frac: f64,
    ) -> u32 {
        let first_vert = self.v_cursor;
        let two_pi = std::f64::consts::TAU;
        for j in 0..segs {
            let theta = (j as f64 / segs as f64) * two_pi;
            let ct = theta.cos();
            let st = theta.sin();

            let p = [
                pos[0] + r * (ct * u[0] + st * v[0]),
                pos[1] + r * (ct * u[1] + st * v[1]),
                pos[2] + r * (ct * u[2] + st * v[2]),
            ];

            let mut n = [
                ct * u[0] + st * v[0],
                ct * u[1] + st * v[1],
                ct * u[2] + st * v[2],
            ];
            let n_len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if n_len > 1e-12 {
                n = [n[0] / n_len, n[1] / n_len, n[2] / n_len];
            }

            let u_coord = j as f64 / segs as f64;
            self.write_vertex(p, n, u_coord, arc_len, ao_val, r, bone_idx, bone_frac);
        }
        first_vert
    }

    /// Quad strip between two consecutive rings. CCW from outside the tube.
    fn emit_quad_strip(&mut self, ring_a: u32, ring_b: u32, segs: usize) {
        for j in 0..segs as u32 {
            let j1 = (j + 1) % segs as u32;
            self.indices.push(ring_a + j);
            self.indices.push(ring_a + j1);
            self.indices.push(ring_b + j);
            self.indices.push(ring_b + j);
            self.indices.push(ring_a + j1);
            self.indices.push(ring_b + j1);
        }
    }

    /// Fan from a single apex vertex to a ring (tip collapse).
    fn emit_apex_fan(&mut self, apex_vert: u32, ring_base: u32, segs: usize) {
        for j in 0..segs as u32 {
            let j1 = (j + 1) % segs as u32;
            self.indices.push(ring_base + j);
            self.indices.push(ring_base + j1);
            self.indices.push(apex_vert);
        }
    }
}

// ---------------------------------------------------------------------------
// build_branch_mesh
// ---------------------------------------------------------------------------

/// Options for [`build_branch_mesh_with`]. `Default` == legacy behavior (no
/// cull), so the golden / determinism paths and existing callers are byte
/// identical to [`build_branch_mesh`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BranchMeshOpts {
    /// Twig-tip wood cull: when `Some(level)`, the TUBE GEOMETRY of leaf-bearing
    /// terminal twig chains whose start branch_level `>= level` is NOT emitted —
    /// the foliage pass clothes those twigs with leaf cards, so rendering bark
    /// there only produced the "bare reddish sticks poking through" artifact.
    ///
    /// Only TIP chains (no children) are eligible, so culling never orphans a
    /// thicker parent's connection: the parent tube + its fork-parent shared ring
    /// stay meshed; only the outermost leaf-covered twig stubs are dropped.
    ///
    /// The wind-bone hierarchy (`bones_wind`, `node_to_bone`) is built for ALL
    /// chains regardless of the cull, so bone indices/parents stay valid and the
    /// leaf pass's `nearest_bone` lookup is unchanged. The cull only suppresses
    /// per-vertex tube emission for the culled chains.
    pub twig_cull_level: Option<i32>,
}

/// Build a merged tapered-tube mesh from a solved skeleton `graph`.
///
/// Walks bone chains (runs of single-child nodes broken by forks/tips), emits a
/// ring per node with rotation-minimizing frames, bridges forks by prepending
/// the fork-parent node as a shared ring, and collapses childless tips to an
/// apex. Determinism: pure function of `graph`, no rng / time / global state.
pub fn build_branch_mesh(graph: &Graph) -> BranchMesh {
    build_branch_mesh_with(graph, BranchMeshOpts::default())
}

/// Like [`build_branch_mesh`], but with [`BranchMeshOpts`] (twig-tip wood cull).
/// Deterministic: pure function of `graph` + `opts`.
pub fn build_branch_mesh_with(graph: &Graph, opts: BranchMeshOpts) -> BranchMesh {
    let nodes = &graph.nodes;
    let bones = &graph.bones;

    if bones.is_empty() || nodes.is_empty() {
        return BranchMesh::empty();
    }

    let n_nodes = nodes.len();

    // ponytail: includeRoot is always true here (single-tree render) — root-flare
    // bones are kept; no filtering pass, so `bones` is used directly.

    // children_of[i] = child node indices (bone.a -> bone.b, a < b by contract).
    let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    let mut has_bone_parent = vec![false; n_nodes];
    for bone in bones {
        children_of[bone.a].push(bone.b);
        has_bone_parent[bone.b] = true;
    }

    // Chain starts: bone-graph roots, plus every child of a fork (multi-child node).
    let mut is_chain_start = vec![false; n_nodes];
    for bone in bones {
        if !has_bone_parent[bone.a] {
            is_chain_start[bone.a] = true;
        }
        if children_of[bone.a].len() > 1 {
            for &ci in &children_of[bone.a] {
                is_chain_start[ci] = true;
            }
        }
    }

    // Fork-parent of each chain-start child (for the bridging prepend), else -1.
    let mut parent_of_chain_start: Vec<isize> = vec![-1; n_nodes];
    for bone in bones {
        if children_of[bone.a].len() > 1 {
            for &ci in &children_of[bone.a] {
                parent_of_chain_start[ci] = bone.a as isize;
            }
        }
    }

    // Walk chains in ascending node-index order (deterministic). A chain that
    // starts at a fork-child is prepended with the fork-parent node so the
    // fork->child bone is bridged by a shared ring (no gap at the joint).
    // ponytail: simplified joint (shared ring), refine if seams show — no crotch
    // hull / web bridging is generated.
    let mut chains: Vec<Vec<usize>> = Vec::new();
    for start in 0..n_nodes {
        if !is_chain_start[start] {
            continue;
        }
        let fork_parent = parent_of_chain_start[start];
        let mut chain: Vec<usize> = if fork_parent >= 0 {
            vec![fork_parent as usize, start]
        } else {
            vec![start]
        };
        let mut cur = start;
        while children_of[cur].len() == 1 {
            cur = children_of[cur][0];
            chain.push(cur);
        }
        let last = *chain.last().unwrap();
        if chain.len() >= 2 || children_of[last].is_empty() {
            chains.push(chain);
        }
    }

    // ---- Per-node AO (deterministic, geometry-only). ----
    let mut max_radius = 0.0_f64;
    let mut min_radius_nodes = f64::INFINITY;
    for node in nodes {
        let r = node.radius.max(0.0);
        if r > max_radius {
            max_radius = r;
        }
        if r < min_radius_nodes {
            min_radius_nodes = r;
        }
    }
    if max_radius < 1e-8 {
        max_radius = 1.0;
    }
    // Guard: if all nodes share a radius, avoid div-by-zero in the thinness curve.
    if min_radius_nodes >= max_radius {
        min_radius_nodes = 0.0;
    }

    let mut is_fork = vec![false; n_nodes];
    let mut is_fork_child = vec![false; n_nodes];
    for bone in bones {
        if children_of[bone.a].len() >= 2 {
            is_fork[bone.a] = true;
            is_fork_child[bone.b] = true;
        }
    }

    let mut node_ao = vec![0.0_f64; n_nodes];
    for (ni, node) in nodes.iter().enumerate() {
        let r = node.radius.max(0.0);
        let thickness_dark = r / max_radius;
        let mut ao_val = 1.0 - thickness_dark * 0.65;
        if is_fork[ni] {
            ao_val -= 0.18;
        }
        if is_fork_child[ni] {
            ao_val -= 0.10;
        }
        node_ao[ni] = ao_val.clamp(0.35, 1.0);
    }

    // ---- Per-node wind weight (dryad branchMesh.js:522-538). 0 = rigid anchor,
    //      1 = max flex. Root nodes are pinned to 0; otherwise the sharpened
    //      thinness curve `pow((maxR-r)/(maxR-minR), 1.5)` — thick trunk ≈ 0,
    //      fine twigs ≈ 1. Used only to mean into per-bone stiffness below. ----
    let r_range = max_radius - min_radius_nodes;
    let mut node_wind_weight = vec![0.0_f64; n_nodes];
    for (ni, node) in nodes.iter().enumerate() {
        if node.is_root {
            node_wind_weight[ni] = 0.0;
            continue;
        }
        let r = node.radius.max(0.0);
        let thinness = if r_range > 1e-8 {
            ((max_radius - r) / r_range).clamp(0.0, 1.0)
        } else {
            0.0
        };
        node_wind_weight[ni] = thinness.powf(1.5);
    }

    // ---- WIND BONE hierarchy: one bone per chain (bone index == chain index).
    //      Mirror of dryad branchMesh.js:556-667. nodeToBone is first-claim-wins
    //      over chains in order; because chains are collected in ascending node
    //      order, a fork-child's bone index always exceeds its fork-parent's →
    //      parent[c] < c (the solver's top-down precondition). ----
    let bone_count = chains.len();

    // node_to_bone: -1 = not rendered. First-claim wins (the prepended fork-parent
    // ring is already owned by its own chain, processed earlier).
    let mut node_to_bone: Vec<i32> = vec![-1; n_nodes];
    for (ci, chain) in chains.iter().enumerate() {
        for &node_idx in chain {
            if node_to_bone[node_idx] == -1 {
                node_to_bone[node_idx] = ci as i32;
            }
        }
    }

    // Per-chain total arc from the pivot node (for boneFraction normalization).
    let mut chain_total_arc_from_pivot = vec![1.0_f64; chains.len()];
    for (ci, chain) in chains.iter().enumerate() {
        let n = chain.len();
        let has_prepend = n >= 2 && parent_of_chain_start[chain[1]] == chain[0] as isize;
        let pivot_ni = if has_prepend { 1 } else { 0 };
        let mut arc = 0.0_f64;
        for ni in (pivot_ni + 1)..n {
            arc += v3len(v3sub(nodes[chain[ni]].pos, nodes[chain[ni - 1]].pos));
        }
        chain_total_arc_from_pivot[ci] = if arc > 1e-12 { arc } else { 1.0 };
    }

    // Per-bone wind tables.
    let mut bones_wind: Vec<WindBone> = Vec::with_capacity(bone_count);
    for (ci, chain) in chains.iter().enumerate() {
        let n = chain.len();
        // Prepend detection (same rule as the chain walk + the arc table above).
        let has_prepend = n >= 2 && parent_of_chain_start[chain[1]] == chain[0] as isize;
        let real_start_idx = if has_prepend { 1 } else { 0 };
        let fork_parent_node_idx: i32 = if has_prepend { chain[0] as i32 } else { -1 };

        // Pivot = the FIRST chain node (the fork-attachment node when prepended).
        let pivot_node = &nodes[chain[0]];
        let pivot = [
            pivot_node.pos[0] as f32,
            pivot_node.pos[1] as f32,
            pivot_node.pos[2] as f32,
        ];

        // Axis hint: unit tangent from the real start toward the next node, else up.
        let axis = if chain.len() > real_start_idx + 1 {
            let a = v3norm(v3sub(
                nodes[chain[real_start_idx + 1]].pos,
                nodes[chain[real_start_idx]].pos,
            ));
            [a[0] as f32, a[1] as f32, a[2] as f32]
        } else {
            [0.0, 1.0, 0.0]
        };

        let first_real_node = &nodes[chain[real_start_idx]];
        let branch_level = first_real_node.branch_level.max(0) as u32;

        // isRigid: the trunk base chain (ci==0) and root-flare chains are anchors.
        let is_rigid: u8 = if ci == 0 || first_real_node.is_root { 1 } else { 0 };

        // Stiffness: rigid → 0; else mean node_wind_weight over the real nodes.
        let stiffness = if is_rigid == 1 {
            0.0
        } else {
            let mut sum = 0.0_f64;
            let mut count = 0usize;
            for ni in real_start_idx..chain.len() {
                sum += node_wind_weight[chain[ni]];
                count += 1;
            }
            if count > 0 { (sum / count as f64) as f32 } else { 0.0 }
        };

        // Parent bone: the chain that owns the fork-parent node. Guard against a
        // self/forward parent (origin-fork edge case) → fall back to root (-1).
        let parent = if fork_parent_node_idx >= 0 {
            let pb = node_to_bone[fork_parent_node_idx as usize];
            if pb >= 0 && (pb as usize) < ci { pb } else { -1 }
        } else {
            -1
        };

        bones_wind.push(WindBone {
            parent,
            pivot,
            axis,
            stiffness,
            branch_level,
            is_rigid,
        });
    }

    let mut b = Builder::new();

    // ---- Emit geometry per chain. ci == bone index for every vertex of the
    //      chain; boneFraction = arcFromPivot / chainTotalArcFromPivot, clamped
    //      [0,1] (0 for the prepended fork ring before the pivot). ----
    for (ci, chain) in chains.iter().enumerate() {
        let n = chain.len();
        let level = nodes[chain[0]].branch_level.max(0);
        let segs = radial_segs_for(level);
        let is_tip = children_of[*chain.last().unwrap()].is_empty();

        // ── Twig-tip wood cull (opt-in). The foliage pass clothes leaf-bearing
        //    twigs (branch_level >= cull_level) in leaf cards, so rendering bark
        //    there is the "bare sticks poking past the canopy" artifact. We strip
        //    the TUBE on the outer twig portion of TIP chains only:
        //      * Only TIP chains (no children) are eligible — truncating an
        //        interior chain would orphan its child geometry; tips have none.
        //      * Within a tip chain, branch_level rises monotonically toward the
        //        tip, so the twig nodes (>= cull_level) form a contiguous TAIL.
        //        We keep the leading rings (level < cull_level, plus ONE boundary
        //        node so the kept stub still meets its leafy continuation flush)
        //        and drop the rest. If the chain is twig all the way (the whole
        //        thing is >= cull_level — e.g. a fine fork-child), drop it whole.
        //    Bones/node_to_bone (built above) are UNAFFECTED, so the wind
        //    hierarchy and the leaf pass's nearest_bone lookup are untouched. ──
        let mut culled_tip = false; // dropped the real apex → no apex fan
        let mut cull_keep: Option<usize> = None; // #rings to keep when truncating
        if let Some(cull_level) = opts.twig_cull_level {
            if is_tip {
                // First node index whose branch_level reaches the twig level.
                let twig_start = chain
                    .iter()
                    .position(|&ni| nodes[ni].branch_level >= cull_level);
                if let Some(ts) = twig_start {
                    // Keep up to and including the boundary node (ts), so the
                    // surviving bark stub overlaps where leaves begin. Need >= 2
                    // kept rings for a tube; otherwise the chain is essentially
                    // all twig → drop it entirely.
                    let keep = ts + 1;
                    if keep < 2 {
                        continue; // whole chain is twig wood: cull it
                    }
                    cull_keep = Some(keep);
                    culled_tip = true; // the real tip (and its apex) is gone
                }
            }
        }
        let bone_idx = ci as f64;
        // Pivot node index within the chain (0 if no prepend, 1 if prepended).
        let has_prepend = n >= 2 && parent_of_chain_start[chain[1]] == chain[0] as isize;
        let pivot_ni = if has_prepend { 1 } else { 0 };
        let total_arc = chain_total_arc_from_pivot[ci];

        // Initial tangent + frame.
        let tangent = if n >= 2 {
            v3norm(v3sub(nodes[chain[1]].pos, nodes[chain[0]].pos))
        } else {
            [0.0, 1.0, 0.0]
        };
        let frame_u = perp_to(tangent);
        let frame_v = v3norm(v3cross(tangent, frame_u));

        // Single-node chain: minimal geometry, no triangles. boneFraction = 0.
        if n == 1 {
            let node = &nodes[chain[0]];
            let r = node.radius.max(MIN_RADIUS);
            let ao_val = node_ao[chain[0]];
            if is_tip {
                b.write_vertex(node.pos, [0.0, 1.0, 0.0], 0.0, 0.0, ao_val, r, bone_idx, 0.0);
            } else {
                b.emit_ring(node.pos, r, frame_u, frame_v, segs, 0.0, ao_val, bone_idx, 0.0);
            }
            continue;
        }

        // Multi-node chain: rings along the chain.
        let mut arc_len = 0.0_f64;
        let mut arc_from_pivot = 0.0_f64; // arc from the pivot node (for boneFraction)
        let mut ring_indices: Vec<u32> = Vec::new();
        let mut prev_tangent = tangent;
        let mut cur_u = frame_u;
        let mut cur_v = frame_v;

        // Normal tip chains collapse the last node to an apex (ring_count = n-1,
        // apex fan after). When the twig tail is culled we instead emit `keep`
        // closed rings (the boundary node gets a real ring, not an apex) and skip
        // the fan — `effective_is_tip` drives the apex block below.
        let effective_is_tip = is_tip && !culled_tip;
        let ring_count = match cull_keep {
            Some(keep) => keep,
            None => {
                if is_tip {
                    n - 1
                } else {
                    n
                }
            }
        };

        for ni in 0..ring_count {
            let node = &nodes[chain[ni]];
            let r = node.radius.max(MIN_RADIUS);

            if ni > 0 {
                let prev_pos = nodes[chain[ni - 1]].pos;
                let cur_pos = node.pos;
                let new_tangent = if ni < n - 1 {
                    let next_pos = nodes[chain[ni + 1]].pos;
                    let incoming = v3norm(v3sub(cur_pos, prev_pos));
                    let outgoing = v3norm(v3sub(next_pos, cur_pos));
                    let mut avg = v3norm([
                        incoming[0] + outgoing[0],
                        incoming[1] + outgoing[1],
                        incoming[2] + outgoing[2],
                    ]);
                    if v3len(avg) < 0.5 {
                        avg = incoming;
                    }
                    v3norm(avg)
                } else {
                    v3norm(v3sub(cur_pos, prev_pos))
                };

                let (nu, nv) = transport_frame(prev_tangent, new_tangent, cur_u, cur_v);
                cur_u = nu;
                cur_v = nv;
                prev_tangent = new_tangent;

                let seg_len = v3len(v3sub(cur_pos, prev_pos));
                arc_len += seg_len;
                if ni >= pivot_ni {
                    arc_from_pivot += seg_len;
                }
            }

            // boneFraction: 0 at/before the pivot, 1 at the chain end.
            let bone_frac = if ni < pivot_ni {
                0.0
            } else {
                (arc_from_pivot / total_arc).min(1.0)
            };

            let ring_start = b.emit_ring(
                node.pos,
                r,
                cur_u,
                cur_v,
                segs,
                arc_len,
                node_ao[chain[ni]],
                bone_idx,
                bone_frac,
            );
            ring_indices.push(ring_start);
        }

        for ri in 0..ring_indices.len().saturating_sub(1) {
            b.emit_quad_strip(ring_indices[ri], ring_indices[ri + 1], segs);
        }

        if effective_is_tip {
            let tip_node = &nodes[chain[n - 1]];
            let prev_pos = nodes[chain[n - 2]].pos;
            let tip_seg = v3len(v3sub(tip_node.pos, prev_pos));
            let tip_arc_len = arc_len + tip_seg;
            let tip_bone_frac = ((arc_from_pivot + tip_seg) / total_arc).min(1.0);
            let apex_normal = prev_tangent;
            let apex_vert = b.write_vertex(
                tip_node.pos,
                apex_normal,
                0.5,
                tip_arc_len,
                node_ao[chain[n - 1]],
                tip_node.radius.max(MIN_RADIUS),
                bone_idx,
                tip_bone_frac,
            );
            b.emit_apex_fan(apex_vert, *ring_indices.last().unwrap(), segs);
        }
    }

    let vertex_count = b.v_cursor;
    let triangle_count = (b.indices.len() / 3) as u32;

    let bounds = if vertex_count == 0 {
        Bounds { min: [0.0; 3], max: [0.0; 3] }
    } else {
        Bounds {
            min: [b.min[0] as f32, b.min[1] as f32, b.min[2] as f32],
            max: [b.max[0] as f32, b.max[1] as f32, b.max[2] as f32],
        }
    };

    BranchMesh {
        positions: b.positions,
        normals: b.normals,
        uvs: b.uvs,
        ao: b.ao,
        radii: b.radii,
        indices: b.indices,
        vertex_count,
        triangle_count,
        bounds,
        bone_index: b.bone_index,
        bone_fraction: b.bone_fraction,
        bones_wind,
        node_to_bone,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{random_genome, resolve, Env};

    #[test]
    fn seed_42_mesh_is_nonempty_finite_in_range_and_deterministic() {
        let env = Env::default();
        let genome = random_genome(&env, 42);
        let resolved = resolve(&genome, &env);

        let mesh = build_branch_mesh(&resolved.graph);

        // Non-empty.
        assert!(mesh.vertex_count > 0, "expected vertices");
        assert!(mesh.triangle_count > 0, "expected triangles");

        // SoA lengths match the vertex/triangle counts.
        assert_eq!(mesh.positions.len(), mesh.vertex_count as usize * 3);
        assert_eq!(mesh.normals.len(), mesh.vertex_count as usize * 3);
        assert_eq!(mesh.uvs.len(), mesh.vertex_count as usize * 2);
        assert_eq!(mesh.ao.len(), mesh.vertex_count as usize);
        assert_eq!(mesh.radii.len(), mesh.vertex_count as usize);
        assert_eq!(mesh.indices.len(), mesh.triangle_count as usize * 3);
        // Bone arrays are per-vertex (purely additive bake).
        assert_eq!(mesh.bone_index.len(), mesh.vertex_count as usize);
        assert_eq!(mesh.bone_fraction.len(), mesh.vertex_count as usize);

        // All floats finite; radii strictly positive (MIN_RADIUS-clamped).
        for &x in mesh.positions.iter().chain(&mesh.normals).chain(&mesh.uvs).chain(&mesh.ao).chain(&mesh.radii) {
            assert!(x.is_finite(), "non-finite float in buffers");
        }
        for &r in &mesh.radii {
            assert!(r > 0.0, "radius must be positive");
        }
        for &b in &mesh.bounds.min {
            assert!(b.is_finite());
        }
        for &b in &mesh.bounds.max {
            assert!(b.is_finite());
        }

        // Indices in range.
        for &idx in &mesh.indices {
            assert!(idx < mesh.vertex_count, "index {idx} out of range {}", mesh.vertex_count);
        }

        // Deterministic: same graph -> identical buffers.
        let mesh2 = build_branch_mesh(&resolved.graph);
        assert_eq!(mesh, mesh2, "build_branch_mesh is not deterministic");

        // Re-resolving the same genome+env -> identical graph -> identical mesh.
        let resolved2 = resolve(&genome, &env);
        let mesh3 = build_branch_mesh(&resolved2.graph);
        assert_eq!(mesh, mesh3, "mesh differs across identical resolves");
    }

    #[test]
    fn seed_42_twig_cull_mesh_is_nonempty_finite_in_range_deterministic_and_smaller() {
        let env = Env::default();
        let genome = random_genome(&env, 42);
        let resolved = resolve(&genome, &env);

        // Baseline (no cull) and the twig-tip-culled output share the SAME graph.
        let base = build_branch_mesh(&resolved.graph);
        let opts = BranchMeshOpts { twig_cull_level: Some(3) };
        let mesh = build_branch_mesh_with(&resolved.graph, opts);

        // Non-empty: the trunk + thick branches always survive the cull.
        assert!(mesh.vertex_count > 0, "expected vertices after twig cull");
        assert!(mesh.triangle_count > 0, "expected triangles after twig cull");

        // The cull removed leaf-bearing twig tubes → strictly fewer verts/tris.
        assert!(
            mesh.vertex_count < base.vertex_count,
            "twig cull should drop vertices ({} !< {})",
            mesh.vertex_count,
            base.vertex_count
        );
        assert!(
            mesh.triangle_count < base.triangle_count,
            "twig cull should drop triangles ({} !< {})",
            mesh.triangle_count,
            base.triangle_count
        );

        // SoA lengths still consistent.
        assert_eq!(mesh.positions.len(), mesh.vertex_count as usize * 3);
        assert_eq!(mesh.normals.len(), mesh.vertex_count as usize * 3);
        assert_eq!(mesh.uvs.len(), mesh.vertex_count as usize * 2);
        assert_eq!(mesh.ao.len(), mesh.vertex_count as usize);
        assert_eq!(mesh.radii.len(), mesh.vertex_count as usize);
        assert_eq!(mesh.indices.len(), mesh.triangle_count as usize * 3);
        assert_eq!(mesh.bone_index.len(), mesh.vertex_count as usize);
        assert_eq!(mesh.bone_fraction.len(), mesh.vertex_count as usize);

        // All floats finite; radii positive.
        for &x in mesh.positions.iter().chain(&mesh.normals).chain(&mesh.uvs).chain(&mesh.ao).chain(&mesh.radii) {
            assert!(x.is_finite(), "non-finite float after twig cull");
        }
        for &r in &mesh.radii {
            assert!(r > 0.0, "radius must be positive after twig cull");
        }

        // Indices in range against the culled vertex count.
        for &idx in &mesh.indices {
            assert!(idx < mesh.vertex_count, "index {idx} out of range {}", mesh.vertex_count);
        }

        // Bone hierarchy is UNAFFECTED by the cull (built for all chains).
        assert_eq!(mesh.bones_wind, base.bones_wind, "cull must not change wind bones");
        assert_eq!(mesh.node_to_bone, base.node_to_bone, "cull must not change node_to_bone");
        // Per-vertex bone index still valid against the (unchanged) bone count.
        let bc = mesh.bones_wind.len();
        for &bi in &mesh.bone_index {
            assert!(bi.is_finite() && bi >= 0.0 && (bi as usize) < bc, "bone_index {bi} out of range {bc}");
        }
        for &bf in &mesh.bone_fraction {
            assert!(bf.is_finite() && (0.0..=1.0).contains(&bf), "bone_fraction {bf} not in [0,1]");
        }

        // Deterministic on the NEW culled path: same graph + opts → identical buffers.
        let mesh2 = build_branch_mesh_with(&resolved.graph, opts);
        assert_eq!(mesh, mesh2, "twig-culled build_branch_mesh is not deterministic");
        let resolved2 = resolve(&genome, &env);
        let mesh3 = build_branch_mesh_with(&resolved2.graph, opts);
        assert_eq!(mesh, mesh3, "twig-culled mesh differs across identical resolves");
    }

    #[test]
    fn wind_bone_bake_is_valid_and_deterministic() {
        let env = Env::default();
        let genome = random_genome(&env, 42);
        let resolved = resolve(&genome, &env);
        let mesh = build_branch_mesh(&resolved.graph);

        // At least the trunk chain → at least one bone.
        assert!(!mesh.bones_wind.is_empty(), "expected wind bones");
        assert!(
            mesh.bones_wind.len() <= MAX_WIND_BONES,
            "bone count {} exceeds MAX_WIND_BONES",
            mesh.bones_wind.len()
        );
        let bc = mesh.bones_wind.len();

        // Solver precondition: parent strictly precedes child (top-down order).
        // Trunk base (bone 0) is a rigid anchor with stiffness 0.
        assert_eq!(mesh.bones_wind[0].is_rigid, 1, "trunk chain must be rigid");
        assert_eq!(mesh.bones_wind[0].stiffness, 0.0, "rigid bone stiffness must be 0");
        for (c, bone) in mesh.bones_wind.iter().enumerate() {
            assert!(bone.parent == -1 || ((bone.parent as usize) < c),
                "bone {c} parent {} violates parent < child", bone.parent);
            assert!(bone.stiffness.is_finite() && bone.stiffness >= 0.0);
            for &p in &bone.pivot { assert!(p.is_finite()); }
            for &a in &bone.axis { assert!(a.is_finite()); }
        }

        // Per-vertex bake: index in range, fraction in [0,1], all finite.
        for &bi in &mesh.bone_index {
            assert!(bi.is_finite() && bi >= 0.0 && (bi as usize) < bc,
                "bone_index {bi} out of range {bc}");
        }
        for &bf in &mesh.bone_fraction {
            assert!(bf.is_finite() && (0.0..=1.0).contains(&bf),
                "bone_fraction {bf} not in [0,1]");
        }

        // node_to_bone: every entry is -1 or a valid bone index.
        assert_eq!(mesh.node_to_bone.len(), resolved.graph.nodes.len());
        for &nb in &mesh.node_to_bone {
            assert!(nb == -1 || ((nb as usize) < bc), "node_to_bone {nb} out of range");
        }

        // Deterministic: identical graph → identical bone tables.
        let mesh2 = build_branch_mesh(&resolved.graph);
        assert_eq!(mesh.bone_index, mesh2.bone_index);
        assert_eq!(mesh.bone_fraction, mesh2.bone_fraction);
        assert_eq!(mesh.bones_wind, mesh2.bones_wind);
        assert_eq!(mesh.node_to_bone, mesh2.node_to_bone);
    }

    #[test]
    fn empty_graph_yields_empty_mesh() {
        let graph = Graph {
            nodes: Vec::new(),
            bones: Vec::new(),
            meta: crate::skeleton::Meta {
                body_axis: [0.0, 1.0, 0.0],
                light_dir: [0.0, 1.0, 0.0],
            },
        };
        let mesh = build_branch_mesh(&graph);
        assert_eq!(mesh.vertex_count, 0);
        assert_eq!(mesh.triangle_count, 0);
        assert!(mesh.positions.is_empty());
        assert!(mesh.indices.is_empty());
    }
}

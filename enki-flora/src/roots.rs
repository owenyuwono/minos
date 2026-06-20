//! Procedural root system growth — port of dryad `roots.js` (`growRootSystem`).
//!
//! Appends `isRoot` nodes and bones to an existing skeleton [`Graph`] IN PLACE:
//!   - One downward-curving taproot (geotropism toward −Y),
//!   - N major laterals (fractional-crossfade idiom) that spread then dive,
//!   - Recursive sub-branches off laterals (depthFrac crossfade at deepest level),
//!   - Above-ground buttress/flare wings at the trunk base.
//!
//! DETERMINISM IS LOAD-BEARING. ALL randomness comes from the passed `root_rng`,
//! an ISOLATED sub-stream `mulberry32((structuralSeed ^ ROOT_SALT) >>> 0)` built
//! by [`crate::genome::resolve`]. Every skip/budget path STILL consumes the draws
//! it would have consumed ([`consume_sub_branch_draws`] purity), so the rootRng
//! draw count is INDEPENDENT of budget — never let a budget gate skip a draw.
//!
//! Vector math uses plain `[f64; 3]` to mirror the JS array math term-for-term.
//! ponytail: trig (sin/cos) may differ ~1 ULP cross-language — accepted ceiling.

// `1.4142` in the curvature-axis sin-hash is a JS-ported magic literal (NOT
// `SQRT_2`) that MUST stay verbatim for bit-exactness; `manual_clamp` mirrors JS
// `Math.max(Math.min())` term-order. `should_implement_trait` is fine — these are
// not iterators. Hex literals mirror the JS salt spelling.
#![allow(
    clippy::approx_constant,
    clippy::manual_clamp,
    clippy::should_implement_trait,
    clippy::unusual_byte_groupings,
    clippy::needless_range_loop
)]

use crate::allometry::derive_traits;
use crate::genome::{Env, Genome};
use crate::rng::Mulberry32;
use crate::skeleton::{Bone, Graph, Node};

type V3 = [f64; 3];

// ---------------------------------------------------------------------------
// Exported constants
// ---------------------------------------------------------------------------

/// Dedicated root bone budget — independent of canopy `MAX_BONES`.
pub const ROOT_BONE_BUDGET: usize = 220;

/// Sub-stream salt — distinct from foliage's `0x1EAF1EAF`.
pub const ROOT_SALT: u32 = 0x900D_5EED;

// ---------------------------------------------------------------------------
// Math helpers (mirror roots.js per-module helpers)
// ---------------------------------------------------------------------------

#[inline]
fn len3(v: V3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn norm3(v: V3) -> V3 {
    let l = len3(v);
    if l < 1e-10 {
        return [1.0, 0.0, 0.0];
    }
    [v[0] / l, v[1] / l, v[2] / l]
}

#[inline]
fn dot3(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn cross3(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

/// Rodrigues rotation: rotate `v` around unit axis `ax` by `angle` (radians).
#[inline]
fn rotate_around(v: V3, ax: V3, angle: f64) -> V3 {
    let c = angle.cos();
    let s = angle.sin();
    let d = dot3(v, ax);
    let cr = cross3(ax, v);
    [
        v[0] * c + cr[0] * s + ax[0] * d * (1.0 - c),
        v[1] * c + cr[1] * s + ax[1] * d * (1.0 - c),
        v[2] * c + cr[2] * s + ax[2] * d * (1.0 - c),
    ]
}

/// Unit vector perpendicular to `dir` (arbitrary but stable).
#[inline]
fn perp_to(dir: V3) -> V3 {
    if dot3(dir, [1.0, 0.0, 0.0]).abs() < 0.9 {
        norm3(cross3(dir, [1.0, 0.0, 0.0]))
    } else {
        norm3(cross3(dir, [0.0, 0.0, 1.0]))
    }
}

// ---------------------------------------------------------------------------
// Deterministic curvature helpers (no rng draws consumed)
// ---------------------------------------------------------------------------

/// Deterministic arc angle for a root segment (radians).
#[inline]
fn root_curvature_arc(structural_seed: f64, lateral_index: f64, level: f64) -> f64 {
    let h = (structural_seed * 5.1234 + lateral_index * 1.8765 + level * 2.4567).sin();
    h * 0.18
}

/// Deterministic axis (perpendicular to `dir`) for the curvature arc.
#[inline]
fn root_curvature_axis(dir: V3, structural_seed: f64, lateral_index: f64, level: f64) -> V3 {
    let h = (structural_seed * 8.6543 + lateral_index * 3.1415 + level * 1.4142).sin();
    let base = perp_to(dir);
    let plane_angle = h * std::f64::consts::PI;
    norm3(rotate_around(base, dir, plane_angle))
}

// ---------------------------------------------------------------------------
// Node constructor for root nodes (skeleton::Node::new is private).
// ---------------------------------------------------------------------------

#[inline]
fn root_node(pos: V3, radius: f64, weight: f64, root_level: i32, parent_idx: i32) -> Node {
    // ROOT NODE CONTRACT: isRoot=true, branchLevel=0, parentIdx<own. Root nodes
    // MUST NOT set isStem/isWoody/isTerminal/attachPos. `root_level` is folded
    // into the node by reusing no extra field — dryad stores it on the node, but
    // no downstream Rust consumer reads rootLevel, so it is dropped here.
    // ponytail: rootLevel field dropped (no Rust reader); curvature hashing uses
    // the value locally before the node is built, so determinism is unaffected.
    let _ = root_level;
    Node {
        pos,
        radius,
        weight,
        branch_level: 0,
        parent_idx,
        is_stem: false,
        is_woody: false,
        is_terminal: false,
        is_root: true,
        attach_pos: None,
        flat_normal: None,
    }
}

// ---------------------------------------------------------------------------
// buildRootChain
// ---------------------------------------------------------------------------

struct ChainTip {
    tip_idx: usize,
    tip_pos: V3,
    tip_dir: V3,
}

/// Place `num_segments` nodes along a root chain, applying geotropism (bending
/// toward −Y) plus deterministic curvature. Mutates `nodes`/`bones`.
#[allow(clippy::too_many_arguments)]
fn build_root_chain(
    start_pos: V3,
    initial_dir: V3,
    seg_len: f64,
    num_segments: usize,
    geo_strength: f64,
    structural_seed: f64,
    lateral_index: f64,
    root_level: i32,
    weight: f64,
    parent_idx0: usize,
    nodes: &mut Vec<Node>,
    bones: &mut Vec<Bone>,
) -> ChainTip {
    let mut pos = start_pos;
    let mut dir = norm3(initial_dir);
    const WORLD_DOWN: V3 = [0.0, -1.0, 0.0];

    let arc_angle = root_curvature_arc(structural_seed, lateral_index, root_level as f64);
    let arc_axis = root_curvature_axis(dir, structural_seed, lateral_index, root_level as f64);
    let arc_per_seg = if num_segments > 1 {
        arc_angle / num_segments as f64
    } else {
        0.0
    };

    let mut parent_idx = parent_idx0;
    let mut tip_idx = parent_idx0;

    for _j in 0..num_segments {
        // Structural arc (deterministic — no rng).
        if arc_per_seg.abs() > 1e-6 {
            dir = norm3(rotate_around(dir, arc_axis, arc_per_seg));
        }

        // Geotropism: bend toward −Y (inverted vs canopy anti-gravity droop).
        if geo_strength > 1e-6 {
            let geo_ax = cross3(dir, WORLD_DOWN);
            let geo_ax_len = len3(geo_ax);
            if geo_ax_len > 1e-6 {
                let geo_ax_unit = [
                    geo_ax[0] / geo_ax_len,
                    geo_ax[1] / geo_ax_len,
                    geo_ax[2] / geo_ax_len,
                ];
                dir = norm3(rotate_around(dir, geo_ax_unit, geo_strength));
            }
        }

        let effective_len = seg_len * weight;
        pos = [
            pos[0] + dir[0] * effective_len,
            pos[1] + dir[1] * effective_len,
            pos[2] + dir[2] * effective_len,
        ];

        let idx = nodes.len();
        nodes.push(root_node(pos, 0.04, weight, root_level, parent_idx as i32));
        bones.push(Bone { a: parent_idx, b: idx });

        parent_idx = idx;
        tip_idx = idx;
    }

    ChainTip {
        tip_idx,
        tip_pos: pos,
        tip_dir: dir,
    }
}

// ---------------------------------------------------------------------------
// growRootSystem — public API
// ---------------------------------------------------------------------------

/// Append a full root system to an existing skeleton `graph` IN PLACE.
///
/// `root_rng` MUST be `mulberry32((genome.structural_seed ^ ROOT_SALT) >>> 0)`.
/// `env` is the optional environment envelope (aridity/wind bias). `max_root_bones`
/// defaults to [`ROOT_BONE_BUDGET`] when `None`.
pub fn grow_root_system(
    graph: &mut Graph,
    genome: &Genome,
    root_rng: &mut Mulberry32,
    env: Option<&Env>,
    max_root_bones: Option<usize>,
) {
    // Woodiness gate / continuous ramp (smoothstep over [0.12, 0.30]).
    let woodiness = genome.woodiness;
    let _wt = ((woodiness - 0.12) / (0.30 - 0.12)).clamp(0.0, 1.0);
    let woodiness_root_scale = _wt * _wt * (3.0 - 2.0 * _wt);
    if woodiness_root_scale <= 0.0 {
        return;
    }

    let root_count = genome.root_count;
    let root_depth = genome.root_depth;
    let root_spread = genome.root_spread;
    let root_flare = genome.root_flare;
    let root_buttress = genome.root_buttress;
    let root_branchiness = genome.root_branchiness;
    let _root_taper = genome.root_taper; // read by proportions, not geometry
    let structural_seed = genome.structural_seed as f64;

    // Optional env bias.
    let mut arid_bias = 0.0;
    let mut wind_bias = 0.0;
    if let Some(e) = env {
        arid_bias = e.aridity.clamp(0.0, 1.0);
        const CALM_WIND: f64 = 0.2;
        wind_bias = (e.wind - CALM_WIND).max(0.0);
    }

    let max_root_bones = max_root_bones.unwrap_or(ROOT_BONE_BUDGET);

    // Allometric stature scaling × woodiness ramp.
    let allometric_root_scale = derive_traits(genome.trunk_height).root_scale;
    let root_scale = allometric_root_scale * woodiness_root_scale;

    let mut root_bones_added: usize = 0;

    // --- Geometry parameters from genes ---
    let taproot_depth = (0.8 + root_depth * 1.2 + arid_bias * 0.6) * root_scale;
    let taproot_segs = ((4.0 + root_depth * 4.0).round()).max(3.0) as usize;
    let taproot_seg_len = taproot_depth / taproot_segs as f64;

    let fract_laterals = 2.0 + root_count * 4.0;
    let full_laterals = fract_laterals.floor() as i32;
    let lateral_frac = fract_laterals - full_laterals as f64;

    let spread_radius = (0.5 + root_spread * 0.8 + wind_bias * 0.3) * root_scale;
    let lateral_segs = ((3.0 + root_spread * 3.0).round()).max(3.0) as usize;
    let lateral_seg_len = spread_radius / lateral_segs as f64;

    let fract_sub_depth = root_branchiness * 3.0;
    let max_sub_depth = fract_sub_depth.floor() as i32;
    let sub_depth_frac = fract_sub_depth - max_sub_depth as f64;

    let taproot_geo = 0.12;
    let lateral_geo = 0.06 + root_depth * 0.08;

    let fract_buttress = root_buttress * 6.0;
    let full_buttress = fract_buttress.floor() as i32;
    let buttress_frac = fract_buttress - full_buttress as f64;
    let buttress_count = full_buttress + if buttress_frac > 0.0 { 1 } else { 0 };
    let buttress_len = (0.10 + root_flare * 0.35) * root_scale;
    let buttress_segs: usize = 2;

    // Origin: first trunk base node (index 0 by skeleton convention).
    let first_trunk_base_idx: usize = 0;
    let origin = graph.nodes[first_trunk_base_idx].pos;

    // --- 1. TAPROOT ---
    let taproot_dir: V3 = [0.0, -1.0, 0.0];
    build_root_chain(
        origin,
        taproot_dir,
        taproot_seg_len,
        taproot_segs,
        taproot_geo,
        structural_seed,
        -1.0, // lateralIndex = -1 for taproot
        0,
        1.0,
        first_trunk_base_idx,
        &mut graph.nodes,
        &mut graph.bones,
    );
    root_bones_added += taproot_segs;

    // --- 2. BUTTRESS / FLARE WINGS ---
    if buttress_count > 0
        && root_bones_added + buttress_count as usize * buttress_segs <= max_root_bones
    {
        for b in 0..buttress_count {
            let base_az = (b as f64 / buttress_count as f64) * std::f64::consts::PI * 2.0;
            let az_jitter = (root_rng.next() - 0.5) * 0.5;
            let az = base_az + az_jitter;

            let buttress_dir = norm3([az.cos() * 0.85, 0.15, az.sin() * 0.85]);
            let w_weight = if b == full_buttress && buttress_frac > 0.0 {
                buttress_frac
            } else {
                1.0
            };
            let buttress_seg_len = (buttress_len / buttress_segs as f64) * w_weight;

            let mut pos = origin;
            let mut parent_idx = first_trunk_base_idx;
            for _s in 0..buttress_segs {
                pos = [
                    pos[0] + buttress_dir[0] * buttress_seg_len,
                    pos[1] + buttress_dir[1] * buttress_seg_len,
                    pos[2] + buttress_dir[2] * buttress_seg_len,
                ];
                let idx = graph.nodes.len();
                graph.nodes.push(root_node(
                    pos,
                    0.04 + root_flare * 0.06,
                    w_weight,
                    0,
                    parent_idx as i32,
                ));
                graph.bones.push(Bone { a: parent_idx, b: idx });
                parent_idx = idx;
                root_bones_added += 1;
            }
        }
    } else if buttress_count > 0 {
        // Budget full — consume jitter draws for determinism anyway.
        for _b in 0..buttress_count {
            root_rng.next();
        }
    }

    // --- 3. MAJOR LATERALS + fractional crossfade ---
    let total_laterals = full_laterals + 1; // always include the crossfade slot

    for li in 0..total_laterals {
        let is_crossfade = li == full_laterals;
        let weight = if is_crossfade { lateral_frac } else { 1.0 };

        // Azimuth draw — always consumed (crossfade too).
        let base_az = (li as f64 / full_laterals as f64) * std::f64::consts::PI * 2.0;
        let az_jitter = (root_rng.next() - 0.5) * 0.8;
        let az = base_az + az_jitter;

        let lateral_init_dir = norm3([az.cos(), 0.0, az.sin()]);

        if is_crossfade && lateral_frac == 0.0 {
            consume_sub_branch_draws(root_rng, max_sub_depth);
            continue;
        }

        if root_bones_added >= max_root_bones {
            consume_sub_branch_draws(root_rng, max_sub_depth);
            continue;
        }

        let segs_avail = lateral_segs.min(max_root_bones - root_bones_added);
        if segs_avail < 1 {
            consume_sub_branch_draws(root_rng, max_sub_depth);
            continue;
        }

        let lateral = build_root_chain(
            origin,
            lateral_init_dir,
            lateral_seg_len,
            segs_avail,
            lateral_geo,
            structural_seed,
            li as f64,
            0,
            weight,
            first_trunk_base_idx,
            &mut graph.nodes,
            &mut graph.bones,
        );
        root_bones_added += segs_avail;

        let mut counter = root_bones_added;
        grow_sub_branches(
            lateral.tip_idx,
            lateral.tip_pos,
            lateral.tip_dir,
            li as f64,
            1, // rootLevel 1 for first sub-branch level
            max_sub_depth,
            sub_depth_frac,
            weight,
            structural_seed,
            lateral_geo,
            root_rng,
            &mut graph.nodes,
            &mut graph.bones,
            max_root_bones,
            &mut counter,
        );
        root_bones_added = count_root_bones(&graph.nodes, first_trunk_base_idx);
    }
}

// ---------------------------------------------------------------------------
// _growSubBranches — recursive sub-branch emitter.
//
// Always consumes 2 draws per call regardless of depth/budget (determinism).
// ---------------------------------------------------------------------------
#[allow(clippy::too_many_arguments)]
fn grow_sub_branches(
    parent_idx: usize,
    parent_pos: V3,
    parent_dir: V3,
    lateral_index: f64,
    depth: i32,
    max_sub_depth: i32,
    sub_depth_frac: f64,
    parent_weight: f64,
    structural_seed: f64,
    lateral_geo: f64,
    root_rng: &mut Mulberry32,
    nodes: &mut Vec<Node>,
    bones: &mut Vec<Bone>,
    max_root_bones: usize,
    counter: &mut usize,
) {
    // Always consume 2 draws for this level (determinism).
    let draw0 = root_rng.next();
    let draw1 = root_rng.next();

    if depth > max_sub_depth {
        return;
    }

    let depth_weight = if depth == max_sub_depth && sub_depth_frac < 1.0 {
        sub_depth_frac
    } else {
        1.0
    };
    let effective_weight = parent_weight * depth_weight;

    let sub_segs: usize = 3;

    for si in 0..2 {
        let draw = if si == 0 { draw0 } else { draw1 };

        if *counter >= max_root_bones {
            consume_sub_branch_draws_at(root_rng, depth + 1, max_sub_depth);
            continue;
        }
        if effective_weight < 0.02 {
            consume_sub_branch_draws_at(root_rng, depth + 1, max_sub_depth);
            continue;
        }

        let az_offset = if si == 0 { 0.6 } else { -0.6 };
        let spread_dir = norm3(rotate_around(
            parent_dir,
            [0.0, 1.0, 0.0],
            az_offset + (draw - 0.5) * 0.4,
        ));

        let sub_seg_len = 0.12 + draw * 0.10;

        let segs_avail = sub_segs.min(max_root_bones - *counter);
        if segs_avail < 1 {
            consume_sub_branch_draws_at(root_rng, depth + 1, max_sub_depth);
            continue;
        }

        let sub = build_root_chain(
            parent_pos,
            spread_dir,
            sub_seg_len,
            segs_avail,
            lateral_geo * 1.2,
            structural_seed,
            lateral_index * 10.0 + si as f64,
            depth,
            effective_weight,
            parent_idx,
            nodes,
            bones,
        );
        *counter += segs_avail;

        grow_sub_branches(
            sub.tip_idx,
            sub.tip_pos,
            sub.tip_dir,
            lateral_index,
            depth + 1,
            max_sub_depth,
            sub_depth_frac,
            effective_weight,
            structural_seed,
            lateral_geo,
            root_rng,
            nodes,
            bones,
            max_root_bones,
            counter,
        );
    }
}

// ---------------------------------------------------------------------------
// Draw-purity helpers — consume the same draws _growSubBranches would.
// ---------------------------------------------------------------------------

fn consume_sub_branch_draws(root_rng: &mut Mulberry32, max_sub_depth: i32) {
    consume_sub_branch_draws_at(root_rng, 1, max_sub_depth);
}

fn consume_sub_branch_draws_at(root_rng: &mut Mulberry32, depth: i32, max_sub_depth: i32) {
    root_rng.next(); // draw0
    root_rng.next(); // draw1

    if depth > max_sub_depth {
        return;
    }
    consume_sub_branch_draws_at(root_rng, depth + 1, max_sub_depth);
    consume_sub_branch_draws_at(root_rng, depth + 1, max_sub_depth);
}

// ---------------------------------------------------------------------------
// _countRootBones — count appended root nodes since the canopy finished.
// ---------------------------------------------------------------------------

fn count_root_bones(nodes: &[Node], first_trunk_base_idx: usize) -> usize {
    let mut count = 0;
    for n in &nodes[first_trunk_base_idx + 1..] {
        if n.is_root {
            count += 1;
        }
    }
    count
}

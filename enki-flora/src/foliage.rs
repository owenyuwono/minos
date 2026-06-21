//! Cluster-card foliage instance generation — port of dryad `foliage.js`
//! (`generateFoliage`).
//!
//! Produces a Structure-of-Arrays cluster set ([`FoliageSoA`]): each instance is
//! a CLUSTER card (a sprig of several leaves), so there are fewer, bigger
//! instances than a per-leaf scheme.
//!
//! DETERMINISM IS LOAD-BEARING. The leaf stream is
//! `mulberry32((structuralSeed ^ 0x1EAF1EAF) >>> 0)`, completely separate from
//! the skeleton/root streams. The optional spine post-pass uses its own
//! `mulberry32((structuralSeed ^ SPINE_SALT) >>> 0)`, gated on `spininess > 0`
//! (zero draws + zero instances when spininess == 0). Same graph+genome →
//! byte-identical SoA. The per-cluster draw cadence (9 leaf draws/cluster, the
//! gate + azBase draws, the spine draws) is preserved exactly.
//!
//! Render-only helpers (`expandClumpsToLeaves`, `expandClumpsToCrossedCards`)
//! and the optional `nodeToBone` wind-bone map are NOT ported: they are render
//! path, not generation, and `resolve` never supplies a `nodeToBone`.
//! ponytail: render-path leaf-fan expansion dropped; bone_index stays 0.
//!
//! Vector math uses plain `[f64; 3]` to mirror the JS array math term-for-term,
//! then narrows to `f32` only at SoA write time (the JS SoA is `Float32Array`).
//! ponytail: trig (sin/cos) may differ ~1 ULP cross-language — accepted ceiling.

// GOLDEN_ANGLE uses an inlined √5 literal (NOT `SQRT_2`/`PI`) that MUST stay
// verbatim for bit-exactness; `manual_clamp` mirrors JS `Math.max(Math.min())`.
// The index loops walk `graph.nodes` by `i` while collecting into a separate
// `Vec`, so a range loop is the clear form. Hex salts mirror the JS spelling.
#![allow(
    clippy::approx_constant,
    clippy::manual_clamp,
    clippy::needless_range_loop,
    clippy::unusual_byte_groupings
)]

use crate::allometry::derive_traits;
use crate::genome::Genome;
use crate::rng::Mulberry32;
use crate::skeleton::Graph;

type V3 = [f64; 3];

// ---------------------------------------------------------------------------
// Public constants — mirror foliage.js exports.
// ---------------------------------------------------------------------------

/// Salt XOR'd with structuralSeed for the isolated spine RNG stream.
pub const SPINE_SALT: u32 = 0x5B1E_5B1E;

/// Maximum cluster (anchor) instances allocated (hard budget).
pub const MAX_LEAVES: usize = 6000;

/// Base clusters-per-segment before density scaling.
pub const BASE_DENSITY: f64 = 12.0;

/// Tip-weight exponent: cluster density ∝ t^TIP_EXPONENT along each segment.
pub const TIP_EXPONENT: f64 = 1.5;

/// Number of extra apical clusters added to terminal node segments.
pub const APICAL_CLUSTER: f64 = 3.0;

/// Golden angle in radians — drives helical phyllotactic azimuth.
pub const GOLDEN_ANGLE: f64 = std::f64::consts::PI * (3.0 - 2.236_067_977_499_79); // π(3−√5)

/// Base cluster size in world units before genome modifiers.
pub const LEAF_BASE: f64 = 0.80;

/// Silhouette jitter: fraction of segment length for axial position jitter.
pub const JITTER_FRAC: f64 = 0.20;

/// Per-cluster scale jitter range (±SIZE_JITTER fraction).
pub const SIZE_JITTER: f64 = 0.30;

/// Phototropism strength: how strongly leaf tangent is lerped toward +Y.
pub const PHOTO_STRENGTH: f64 = 0.25;

/// Minimum branchLevel for full-density foliage.
pub const OUTER_LEVEL_THRESHOLD: i32 = 3;

/// Probability a non-terminal inner segment gets body clusters.
pub const INNER_SEGMENT_PROB: f64 = 0.15;

/// Probability a non-terminal mid-level segment gets body clusters.
pub const MID_SEGMENT_PROB: f64 = 0.55;

/// Base scale of a spine card before `spininess` scaling.
pub const SPINE_BASE_SCALE: f64 = 0.12;

/// Number of spines emitted per eligible node at spininess=1.
pub const SPINE_DENSITY: f64 = 6.0;

// ---------------------------------------------------------------------------
// Output type
// ---------------------------------------------------------------------------

/// Structure-of-Arrays foliage output. Only `count` instances are present; the
/// per-vector arrays (`position`/`normal`/`tangent`) hold `3 * count` floats and
/// the per-instance arrays hold `count` floats. (dryad over-allocates to
/// `MAX_LEAVES` and reports `count`; here we push exactly `count` entries —
/// `[0, count)` are the only valid slots either way.)
#[derive(Clone, Debug, PartialEq)]
pub struct FoliageSoA {
    pub count: usize,
    pub position: Vec<f32>,  // 3 * count
    pub normal: Vec<f32>,    // 3 * count
    pub tangent: Vec<f32>,   // 3 * count
    pub scale: Vec<f32>,     // count
    pub rotation: Vec<f32>,  // count
    pub age_color: Vec<f32>, // count
    pub exposure: Vec<f32>,  // count
    pub bone_index: Vec<f32>, // count (always 0 — no nodeToBone in resolve)
    pub shape: f32,          // = appendageBreadth
    // --- Render-side metadata (NOT part of resolve()/golden comparison) ---
    // These carry the source-node identity for each anchor so the viewer can
    // restrict leaves to fine twigs and re-seat the leaf base on the wood
    // surface. They add no rng draws and do not affect count/position/scale/
    // rotation/shape, so `golden_vector_matches_dryad` is unaffected.
    pub source_radius: Vec<f32>,       // count — mid_radius (wood surface radius) at the anchor
    pub source_branch_level: Vec<f32>, // count — node.branch_level
    pub source_is_terminal: Vec<f32>,  // count — 1.0 if source node is_terminal else 0.0
}

// ---------------------------------------------------------------------------
// Math helpers (mirror foliage.js)
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
fn cross3(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot3(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Rodrigues rotation: rotate `v` around unit axis `ax` by `angle` (radians).
#[inline]
fn rotate3(v: V3, ax: V3, angle: f64) -> V3 {
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

#[inline]
fn perp_to(dir: V3) -> V3 {
    if dot3(dir, [1.0, 0.0, 0.0]).abs() < 0.9 {
        norm3(cross3(dir, [1.0, 0.0, 0.0]))
    } else {
        norm3(cross3(dir, [0.0, 0.0, 1.0]))
    }
}

/// Proximal bare-zone fraction for this branchLevel.
#[inline]
fn bare_skip(branch_level: i32) -> f64 {
    if branch_level <= 1 {
        0.55
    } else if branch_level == 2 {
        0.40
    } else if branch_level == 3 {
        0.20
    } else {
        0.08
    }
}

/// Tip-weighted positions on `[skip, 1]` using `t^tip_exp` density.
fn tip_weighted_positions(n: usize, skip: f64, tip_exp: f64) -> Vec<f64> {
    let l = 1.0 - skip;
    let exp = 1.0 / (tip_exp + 1.0);
    let mut result = Vec::with_capacity(n);
    for k in 0..n {
        let normalized = (((k as f64) + 0.5) / n as f64).powf(exp);
        result.push(skip + normalized * l);
    }
    result
}

#[inline]
fn clamp01(v: f64) -> f64 {
    v.max(0.0).min(1.0)
}

// ---------------------------------------------------------------------------
// Eligible-segment record.
// ---------------------------------------------------------------------------

struct Segment {
    parent_idx: usize,
    node_idx: usize,
}

// ---------------------------------------------------------------------------
// generateFoliage
// ---------------------------------------------------------------------------

/// Generate cluster-card instances for the solved `graph` + `genome`.
pub fn generate_foliage(graph: &Graph, genome: &Genome) -> FoliageSoA {
    // SoA accumulators — pushed in `count` order (only valid slots).
    let mut position: Vec<f32> = Vec::new();
    let mut normal: Vec<f32> = Vec::new();
    let mut tangent: Vec<f32> = Vec::new();
    let mut scale: Vec<f32> = Vec::new();
    let mut rotation: Vec<f32> = Vec::new();
    let mut age_color: Vec<f32> = Vec::new();
    let mut exposure: Vec<f32> = Vec::new();
    let mut bone_index: Vec<f32> = Vec::new();
    // Render-side metadata (see FoliageSoA docs) — not compared by golden.
    let mut source_radius: Vec<f32> = Vec::new();
    let mut source_branch_level: Vec<f32> = Vec::new();
    let mut source_is_terminal: Vec<f32> = Vec::new();

    // --- Genome genes ---
    let appendage_density = genome.appendage_density;
    let leaf_size = genome.leaf_size;
    let appendage_breadth = genome.appendage_breadth;
    let radial_order = genome.radial_order;
    let weep = genome.weep;
    let tip_tuft = genome.tip_tuft;
    let leaf_division = genome.leaf_division;
    let rosette = genome.rosette;
    let leaf_width_g = genome.leaf_width;
    let narrow_factor = clamp01((0.30 - leaf_width_g) / 0.30);
    let frondyness = clamp01(leaf_division.max(rosette).max(narrow_factor));

    let leaf_area_scale = derive_traits(genome.trunk_height).leaf_area_scale;

    // --- Isolated leaf RNG stream ---
    let mut rng = Mulberry32::new(genome.structural_seed ^ 0x1EAF_1EAF);

    let nodes = &graph.nodes;

    // --- Collect eligible segments ---
    let mut segments: Vec<Segment> = Vec::new();
    for i in 0..nodes.len() {
        let node = &nodes[i];
        if node.is_root {
            continue;
        }
        if node.branch_level < 1 {
            continue;
        }
        let p_idx = node.parent_idx;
        if p_idx < 0 {
            continue;
        }
        if !node.is_woody && !node.is_terminal {
            continue;
        }
        segments.push(Segment {
            parent_idx: p_idx as usize,
            node_idx: i,
        });
    }

    // --- maxBranchLevel for ageColor normalization ---
    let mut max_branch_level = 1;
    for s in &segments {
        let bl = nodes[s.node_idx].branch_level;
        if bl > max_branch_level {
            max_branch_level = bl;
        }
    }

    // --- Crown bounding box (Y) for exposure normalization ---
    let mut crown_min_y = f64::INFINITY;
    let mut crown_max_y = f64::NEG_INFINITY;
    for s in &segments {
        let y = nodes[s.node_idx].pos[1];
        if y < crown_min_y {
            crown_min_y = y;
        }
        if y > crown_max_y {
            crown_max_y = y;
        }
    }
    let crown_y_range = (crown_max_y - crown_min_y).max(1e-6);

    let mut count: usize = 0;
    const UP: V3 = [0.0, 1.0, 0.0];

    // Apical reserve / body budget.
    let terminal_segment_count = segments.iter().filter(|s| nodes[s.node_idx].is_terminal).count();
    let apical_reserve = (terminal_segment_count as f64 * APICAL_CLUSTER).min(MAX_LEAVES as f64) as usize;
    let body_budget = MAX_LEAVES - apical_reserve;

    // Frondy even-coverage per-segment base.
    let mut frondy_per_seg_base = 0.0;
    if frondyness > 0.0 && !segments.is_empty() {
        let mut weight_sum = 0.0;
        for s in &segments {
            weight_sum += 1.0 + 0.5 * nodes[s.node_idx].branch_level as f64;
        }
        frondy_per_seg_base = (body_budget as f64 * 0.90) / weight_sum.max(1.0);
    }

    // -----------------------------------------------------------------------
    // writeCluster — consumes 9 rng draws (sizeJitter, jitter, azJitter,
    // outAngle, clusterRoll) — wait: dryad lists 9 but the no-op radialPush/
    // jPhi/jCosT/volJit draws were removed in the current source. The actual
    // draws are: sizeJitter, jitter, azJitter, outAngle, clusterRoll = 5 draws.
    // (The header comment in foliage.js is stale; the body is the contract.)
    // -----------------------------------------------------------------------
    #[allow(clippy::too_many_arguments)]
    fn write_cluster(
        // SoA out
        position: &mut Vec<f32>,
        normal: &mut Vec<f32>,
        tangent: &mut Vec<f32>,
        scale: &mut Vec<f32>,
        rotation: &mut Vec<f32>,
        age_color: &mut Vec<f32>,
        exposure: &mut Vec<f32>,
        bone_index: &mut Vec<f32>,
        // Render-side metadata out (not compared by golden).
        source_radius: &mut Vec<f32>,
        source_branch_level: &mut Vec<f32>,
        source_is_terminal: &mut Vec<f32>,
        count: &mut usize,
        rng: &mut Mulberry32,
        // params
        p_pos: V3,
        raw_axis: V3,
        axis: V3,
        seg_len: f64,
        mid_radius: f64,
        node_branch_level: i32,
        node_is_terminal: bool,
        t: f64,
        az_base: f64,
        k: usize,
        leaf_age: f64,
        crown_min_y: f64,
        crown_y_range: f64,
        max_branch_level: i32,
        // captured genome/derived
        leaf_size: f64,
        appendage_breadth: f64,
        leaf_area_scale: f64,
        radial_order: f64,
        weep: f64,
    ) {
        let seg_pos = [
            p_pos[0] + raw_axis[0] * t,
            p_pos[1] + raw_axis[1] * t,
            p_pos[2] + raw_axis[2] * t,
        ];

        // SIZE
        let size_jitter = 1.0 + (rng.next() - 0.5) * 2.0 * SIZE_JITTER;
        let cluster_scale =
            LEAF_BASE * leaf_size * (0.6 + 0.4 * appendage_breadth) * size_jitter * leaf_area_scale;

        // CANOPY ENVELOPE JITTER
        let jitter = (rng.next() - 0.5) * 2.0 * JITTER_FRAC * seg_len;
        let j_pos = [
            seg_pos[0] + axis[0] * jitter,
            seg_pos[1] + axis[1] * jitter,
            seg_pos[2] + axis[2] * jitter,
        ];

        // AZIMUTH
        let az_step = (1.0 - radial_order) * std::f64::consts::PI + radial_order * GOLDEN_ANGLE;
        let azimuth = az_base + k as f64 * az_step;
        let az_jitter = (rng.next() - 0.5) * 0.6;
        let az_final = azimuth + az_jitter;

        let perp = perp_to(axis);
        let radial_dir = rotate3(perp, axis, az_final);

        let cluster_base = [
            j_pos[0] + radial_dir[0] * mid_radius,
            j_pos[1] + radial_dir[1] * mid_radius,
            j_pos[2] + radial_dir[2] * mid_radius,
        ];

        // UPWARD/OUTWARD LEAN
        let out_angle = (std::f64::consts::PI / 4.0) + rng.next() * (std::f64::consts::PI / 4.0);
        let out_axis = norm3(cross3(axis, radial_dir));
        let mut cluster_tangent = rotate3(axis, out_axis, out_angle);
        cluster_tangent = norm3([
            cluster_tangent[0] * (1.0 - PHOTO_STRENGTH) + UP[0] * PHOTO_STRENGTH,
            cluster_tangent[1] * (1.0 - PHOTO_STRENGTH) + UP[1] * PHOTO_STRENGTH,
            cluster_tangent[2] * (1.0 - PHOTO_STRENGTH) + UP[2] * PHOTO_STRENGTH,
        ]);

        // LEAF-HANG (weep > 0 only; strict no-op at weep=0). No rng draws.
        let mut weep_normal: Option<V3> = None;
        if weep > 0.0 {
            let hang = weep.sqrt();
            const DOWN: V3 = [0.0, -1.0, 0.0];
            cluster_tangent = norm3([
                cluster_tangent[0] * (1.0 - hang) + DOWN[0] * hang,
                cluster_tangent[1] * (1.0 - hang) + DOWN[1] * hang,
                cluster_tangent[2] * (1.0 - hang) + DOWN[2] * hang,
            ]);

            let d_rad = dot3(radial_dir, cluster_tangent);
            let rej_x = radial_dir[0] - cluster_tangent[0] * d_rad;
            let rej_y = radial_dir[1] - cluster_tangent[1] * d_rad;
            let rej_z = radial_dir[2] - cluster_tangent[2] * d_rad;
            let rej_len = (rej_x * rej_x + rej_y * rej_y + rej_z * rej_z).sqrt();
            let outward_normal = if rej_len > 1e-6 {
                [rej_x / rej_len, rej_y / rej_len, rej_z / rej_len]
            } else {
                norm3(cross3(cluster_tangent, radial_dir))
            };

            let k_sky = 0.65;
            let sky_blend = hang * k_sky;
            weep_normal = Some(norm3([
                outward_normal[0] * (1.0 - sky_blend),
                outward_normal[1] * (1.0 - sky_blend) + sky_blend,
                outward_normal[2] * (1.0 - sky_blend),
            ]));
        }

        let cluster_normal = match weep_normal {
            Some(n) => n,
            None => norm3(cross3(cluster_tangent, radial_dir)),
        };
        let cluster_roll = (rng.next() - 0.5) * 0.8;

        // EXPOSURE
        let normalized_level = node_branch_level as f64 / max_branch_level as f64;
        let normalized_height = clamp01((cluster_base[1] - crown_min_y) / crown_y_range);
        let exp_val = clamp01(0.5 * normalized_level + 0.5 * normalized_height);

        position.push(cluster_base[0] as f32);
        position.push(cluster_base[1] as f32);
        position.push(cluster_base[2] as f32);
        normal.push(cluster_normal[0] as f32);
        normal.push(cluster_normal[1] as f32);
        normal.push(cluster_normal[2] as f32);
        tangent.push(cluster_tangent[0] as f32);
        tangent.push(cluster_tangent[1] as f32);
        tangent.push(cluster_tangent[2] as f32);
        scale.push(cluster_scale as f32);
        rotation.push(cluster_roll as f32);
        age_color.push(leaf_age as f32);
        exposure.push(exp_val as f32);
        bone_index.push(0.0); // no nodeToBone in resolve
        // Render-side metadata: carry the wood-surface radius + source-node
        // identity so the viewer can keep leaves on twigs and seat the base.
        source_radius.push(mid_radius as f32);
        source_branch_level.push(node_branch_level as f32);
        source_is_terminal.push(if node_is_terminal { 1.0 } else { 0.0 });
        *count += 1;
    }

    // -----------------------------------------------------------------------
    // PASS 1 — body clusters for all segments.
    // -----------------------------------------------------------------------
    for s in &segments {
        if count >= body_budget {
            break;
        }
        let parent_idx = s.parent_idx;
        let node_idx = s.node_idx;
        let node_branch_level = nodes[node_idx].branch_level;
        let node_is_terminal = nodes[node_idx].is_terminal;
        let node_weight = nodes[node_idx].weight;

        let p_pos = nodes[parent_idx].pos;
        let n_pos = nodes[node_idx].pos;

        let raw_axis = [
            n_pos[0] - p_pos[0],
            n_pos[1] - p_pos[1],
            n_pos[2] - p_pos[2],
        ];
        let seg_len = len3(raw_axis);
        if seg_len < 1e-10 {
            rng.next(); // consume azimuth draw on degenerate seg
            continue;
        }
        let axis = norm3(raw_axis);

        let tip_boost = 1.0 + 0.5 * node_branch_level as f64;
        let normal_per_seg = (BASE_DENSITY * appendage_density * tip_boost).round();
        let frondy_share = (frondy_per_seg_base * tip_boost).round().min(normal_per_seg).max(2.0);
        let clusters_per_seg_base = if frondyness <= 0.0 {
            normal_per_seg
        } else {
            (normal_per_seg + (frondy_share - normal_per_seg) * frondyness).round()
        };
        let seg_weight = node_weight;
        let clusters_per_seg = (clusters_per_seg_base * seg_weight).round();
        let body_to_place =
            (clusters_per_seg.max(0.0) as usize).min(body_budget - count);

        // Gate rng (always consumed).
        let gate_roll = rng.next();

        if body_to_place < 1 {
            rng.next(); // consume azBase draw for determinism
            continue;
        }

        let segment_is_active = if node_is_terminal {
            true
        } else {
            let broadleaf_prob = if node_branch_level < OUTER_LEVEL_THRESHOLD {
                INNER_SEGMENT_PROB
            } else {
                MID_SEGMENT_PROB
            };
            let active_prob = broadleaf_prob + (1.0 - broadleaf_prob) * frondyness;
            gate_roll < active_prob
        };

        let mid_radius = (nodes[parent_idx].radius + nodes[node_idx].radius) * 0.5;
        let leaf_age = node_branch_level as f64 / max_branch_level as f64;

        // azBase — always consumed (even if inactive).
        let az_base = rng.next() * std::f64::consts::PI * 2.0;

        if !segment_is_active {
            continue;
        }

        let base_skip = if node_is_terminal {
            0.0
        } else {
            bare_skip(node_branch_level)
        };
        let broad_tip_exp = TIP_EXPONENT + tip_tuft * 4.0;
        let broad_skip = base_skip + (1.0 - base_skip) * tip_tuft * 0.6;
        let frond_tip_exp = 1.0 + tip_tuft * 1.0;
        let frond_skip = if node_is_terminal { 0.0 } else { 0.12 };
        let effective_tip_exp = broad_tip_exp + (frond_tip_exp - broad_tip_exp) * frondyness;
        let effective_skip = broad_skip + (frond_skip - broad_skip) * frondyness;
        let t_values = tip_weighted_positions(body_to_place, effective_skip, effective_tip_exp);

        for k in 0..body_to_place {
            write_cluster(
                &mut position,
                &mut normal,
                &mut tangent,
                &mut scale,
                &mut rotation,
                &mut age_color,
                &mut exposure,
                &mut bone_index,
                &mut source_radius,
                &mut source_branch_level,
                &mut source_is_terminal,
                &mut count,
                &mut rng,
                p_pos,
                raw_axis,
                axis,
                seg_len,
                mid_radius,
                node_branch_level,
                node_is_terminal,
                t_values[k],
                az_base,
                k,
                leaf_age,
                crown_min_y,
                crown_y_range,
                max_branch_level,
                leaf_size,
                appendage_breadth,
                leaf_area_scale,
                radial_order,
                weep,
            );
        }
    }

    // -----------------------------------------------------------------------
    // PASS 2 — apical clusters for every terminal segment.
    // -----------------------------------------------------------------------
    for s in &segments {
        let node_idx = s.node_idx;
        if !nodes[node_idx].is_terminal {
            continue;
        }
        if count >= MAX_LEAVES {
            break;
        }
        let parent_idx = s.parent_idx;
        let node_branch_level = nodes[node_idx].branch_level;
        let node_weight = nodes[node_idx].weight;

        let p_pos = nodes[parent_idx].pos;
        let n_pos = nodes[node_idx].pos;
        let raw_axis = [
            n_pos[0] - p_pos[0],
            n_pos[1] - p_pos[1],
            n_pos[2] - p_pos[2],
        ];
        let seg_len = len3(raw_axis);
        if seg_len < 1e-10 {
            continue; // degenerate: skip, no rng consumed
        }
        let axis = norm3(raw_axis);
        let mid_radius = (nodes[parent_idx].radius + nodes[node_idx].radius) * 0.5;
        let leaf_age = node_branch_level as f64 / max_branch_level as f64;

        let seg_weight = node_weight;
        let dens_factor = appendage_density.min(1.0);
        let apical_to_place =
            ((APICAL_CLUSTER * dens_factor * seg_weight).round() as i64).min((MAX_LEAVES - count) as i64);
        if apical_to_place < 1 {
            continue;
        }
        let apical_to_place = apical_to_place as usize;

        let az_base = rng.next() * std::f64::consts::PI * 2.0;

        for idx in 0..apical_to_place {
            let t = if apical_to_place == 1 {
                1.02
            } else {
                1.0 + (idx as f64 / (apical_to_place as f64 - 1.0)) * 0.03
            };
            write_cluster(
                &mut position,
                &mut normal,
                &mut tangent,
                &mut scale,
                &mut rotation,
                &mut age_color,
                &mut exposure,
                &mut bone_index,
                &mut source_radius,
                &mut source_branch_level,
                &mut source_is_terminal,
                &mut count,
                &mut rng,
                p_pos,
                raw_axis,
                axis,
                seg_len,
                mid_radius,
                node_branch_level,
                true, // apical pass runs only for terminal segments
                t,
                az_base,
                idx,
                leaf_age,
                crown_min_y,
                crown_y_range,
                max_branch_level,
                leaf_size,
                appendage_breadth,
                leaf_area_scale,
                radial_order,
                weep,
            );
        }
    }

    // -----------------------------------------------------------------------
    // SPINE POST-PASS — gated on spininess > 0 (isolated SPINE_SALT stream).
    // -----------------------------------------------------------------------
    let spininess = genome.spininess;
    if spininess > 0.0 {
        let mut spine_rng = Mulberry32::new(genome.structural_seed ^ SPINE_SALT);

        let fract_spines = SPINE_DENSITY * spininess;
        let full_spines = fract_spines.floor() as i32;
        let spine_frac = fract_spines - full_spines as f64;
        let spine_count = full_spines + if spine_frac > 0.0 { 1 } else { 0 };
        let spine_scale = SPINE_BASE_SCALE * (0.5 + 0.5 * spininess);

        // Extended segment list: ALL non-root nodes with a real parent.
        let mut spine_segments: Vec<Segment> = Vec::new();
        for i in 0..nodes.len() {
            let node = &nodes[i];
            if node.is_root {
                continue;
            }
            let p_idx = node.parent_idx;
            if p_idx < 0 {
                continue;
            }
            spine_segments.push(Segment {
                parent_idx: p_idx as usize,
                node_idx: i,
            });
        }

        for s in &spine_segments {
            if count >= MAX_LEAVES {
                break;
            }
            let parent_idx = s.parent_idx;
            let node_idx = s.node_idx;
            let node_branch_level = nodes[node_idx].branch_level;
            let node_is_terminal = nodes[node_idx].is_terminal;
            let spine_mid_radius =
                (nodes[parent_idx].radius + nodes[node_idx].radius) * 0.5;

            let p_pos = nodes[parent_idx].pos;
            let n_pos = nodes[node_idx].pos;
            let raw_axis = [
                n_pos[0] - p_pos[0],
                n_pos[1] - p_pos[1],
                n_pos[2] - p_pos[2],
            ];
            let seg_len = len3(raw_axis);
            if seg_len < 1e-10 {
                spine_rng.next(); // consume azimuth draw for determinism
                continue;
            }
            let axis = norm3(raw_axis);

            let az_base = spine_rng.next() * std::f64::consts::PI * 2.0;

            let to_place = (spine_count as usize).min(MAX_LEAVES - count);

            for k in 0..to_place {
                if count >= MAX_LEAVES {
                    break;
                }
                let t = 0.05 + (k as f64 / (to_place as f64 - 1.0).max(1.0)) * 0.90;

                let spine_pos = [
                    p_pos[0] + raw_axis[0] * t,
                    p_pos[1] + raw_axis[1] * t,
                    p_pos[2] + raw_axis[2] * t,
                ];

                let azimuth = az_base + k as f64 * GOLDEN_ANGLE;
                let perp = perp_to(axis);
                let _radial_dir = rotate3(perp, axis, azimuth);

                let az_jitter = (spine_rng.next() - 0.5) * 0.4;
                let final_dir = rotate3(perp, axis, azimuth + az_jitter);

                let spine_tangent = norm3(final_dir);
                let spine_normal = norm3(cross3(axis, spine_tangent));

                let roll = (spine_rng.next() - 0.5) * 0.5;

                let normalized_height = clamp01((spine_pos[1] - crown_min_y) / crown_y_range);
                let exp_val = clamp01(
                    0.5 * (node_branch_level as f64 / max_branch_level as f64)
                        + 0.5 * normalized_height,
                );

                position.push(spine_pos[0] as f32);
                position.push(spine_pos[1] as f32);
                position.push(spine_pos[2] as f32);
                normal.push(spine_normal[0] as f32);
                normal.push(spine_normal[1] as f32);
                normal.push(spine_normal[2] as f32);
                tangent.push(spine_tangent[0] as f32);
                tangent.push(spine_tangent[1] as f32);
                tangent.push(spine_tangent[2] as f32);
                let s_scale = if k as i32 == full_spines && spine_frac > 0.0 {
                    spine_scale * spine_frac
                } else {
                    spine_scale
                };
                scale.push(s_scale as f32);
                rotation.push(roll as f32);
                age_color.push((node_branch_level as f64 / max_branch_level as f64) as f32);
                exposure.push(exp_val as f32);
                bone_index.push(0.0);
                source_radius.push(spine_mid_radius as f32);
                source_branch_level.push(node_branch_level as f32);
                source_is_terminal.push(if node_is_terminal { 1.0 } else { 0.0 });
                count += 1;
            }
        }
    }

    FoliageSoA {
        count,
        position,
        normal,
        tangent,
        scale,
        rotation,
        age_color,
        exposure,
        bone_index,
        shape: appendage_breadth as f32,
        source_radius,
        source_branch_level,
        source_is_terminal,
    }
}

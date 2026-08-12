//! Branch skeleton construction — port of dryad `skeleton.js` (`buildSkeleton`).
//!
//! Recursive tapering branch graph for procedural FLORA. CONTINUOUS MORPHOSPACE:
//! all structural parameters derive from continuous `[0,1]` genes so every gene
//! interpolation yields a valid, smoothly-changing organism.
//!
//! DETERMINISM IS LOAD-BEARING. [`build_skeleton`] reproduces dryad's EXACT rng
//! draw COUNT and ORDER. Any reordering reshuffles every downstream value. The
//! draw schedule is (see inline `// RNG draw` comments):
//!
//! - draw 1: `trunkBend` — lateral (X) arc.
//! - draw 2: `trunkLean` — fore-aft (Z) arc.
//! - draws 3+s: per basal stem, one azimuth jitter draw (always, even for the
//!   crossfade stem).
//! - BFS, leader-first order, per branching node: ALWAYS 3 draws per child
//!   (`drawA` length, `drawB` angle, `drawC` azimuth) — consumed even for
//!   zero-weight crossfade children.
//! - per terminal: 1 draw for `flatNormal` azimuth jitter (`markTerminal`).
//! - Curvature/whorl/rosette azimuths use the structuralSeed sin-hash — NO draws.
//! - Gated paths (whorl / rosette / crownStart / tillering crossfade) at identity
//!   draw ZERO extra rng (ported as draw-correct branches).
//!
//! INVARIANT: rng draw count/order is NEVER affected by `jitter_amp`.
//!
//! Vector math uses plain `[f64; 3]` (not glam) to mirror the JS array math
//! term-for-term — bit-faithful float reproduction is the contract.
//! ponytail: trig (sin/cos) may differ ~1 ULP cross-language — accepted ceiling.

// Determinism over clippy: the curvature/whorl sin-hash magic numbers (6.2831,
// 9.4248, 2.7183, ...) are JS-ported literals that MUST stay verbatim — they are
// NOT `TAU`/`E` and substituting the std constants would break bit-exactness.
// `manual_clamp` is allowed because the `.max(lo).min(hi)` form mirrors JS
// `Math.max(Math.min())` term-order (std `f64::clamp` reorders and panics if
// lo>hi); keeping the JS form is the faithful port.
#![allow(clippy::approx_constant, clippy::manual_clamp)]

use crate::allometry::{compute_size_factor, derive_traits, Traits};
use crate::genome::Genome;
use crate::rng::Mulberry32;

// ---------------------------------------------------------------------------
// Math helpers (port of skeleton.js normalize/cross/dot/rotateAround/perpTo/lerp)
// ---------------------------------------------------------------------------

type V3 = [f64; 3];

#[inline]
fn normalize(v: V3) -> V3 {
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if len < 1e-10 {
        return [1.0, 0.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

#[inline]
fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Rodrigues rotation: rotate vector `v` by `angle` around unit axis `ax`.
#[inline]
fn rotate_around(v: V3, ax: V3, angle: f64) -> V3 {
    let c = angle.cos();
    let s = angle.sin();
    let d = dot(v, ax);
    let cr = cross(ax, v);
    [
        v[0] * c + cr[0] * s + ax[0] * d * (1.0 - c),
        v[1] * c + cr[1] * s + ax[1] * d * (1.0 - c),
        v[2] * c + cr[2] * s + ax[2] * d * (1.0 - c),
    ]
}

/// Compute a perpendicular vector to `dir` (unit).
#[inline]
fn perp_to(dir: V3) -> V3 {
    if dot(dir, [1.0, 0.0, 0.0]).abs() < 0.9 {
        normalize(cross(dir, [1.0, 0.0, 0.0]))
    } else {
        normalize(cross(dir, [0.0, 0.0, 1.0]))
    }
}

#[inline]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ---------------------------------------------------------------------------
// Branch curvature helpers + tunable constants (skeleton.js constants)
// ---------------------------------------------------------------------------

/// Maximum arc a single branch chain can subtend (~31°).
const MAX_ARC_ANGLE: f64 = 0.55;
/// Radius multiplier per branch level (exponential tip thinning).
const TAPER_BASE: f64 = 0.72;
/// Terminal (deepest) branches are a touch longer.
const TIP_LENGTH_BOOST: f64 = 1.0;
/// Branch base radius.
const BRANCH_BASE_RADIUS: f64 = 0.04;

/// Exported bone budget ceiling (absolute upper bound on emitted bones).
pub const MAX_BONES: usize = 900;

/// World-space height of a full (weight=1) single trunk.
const TRUNK_HEIGHT: f64 = 2.0;
/// Multiplier on the initial branch length fed into the BFS.
const BASE_BRANCH_LENGTH: f64 = 1.0;
/// Extra angular spread added to `branchAngle` for laterals at level >= 1.
const LATERAL_SPREAD_BOOST: f64 = 0.15;

/// Deterministic arc angle for a branch (no rng). Result in
/// `[-MAX_ARC_ANGLE, +MAX_ARC_ANGLE]`.
#[inline]
fn compute_curvature_arc(structural_seed: f64, child_index: f64, level: f64) -> f64 {
    let h = (structural_seed * 6.2831 + child_index * 2.3999 + level * 1.6180).sin();
    h * MAX_ARC_ANGLE * 0.6
}

/// Deterministic unit axis (perpendicular to `dir`) for the branch arc (no rng).
#[inline]
fn compute_curvature_axis(dir: V3, structural_seed: f64, child_index: f64, level: f64) -> V3 {
    let h = (structural_seed * 9.4248 + child_index * 3.7699 + level * 2.7183).sin();
    let base = perp_to(dir);
    let plane_angle = h * std::f64::consts::PI;
    normalize(rotate_around(base, dir, plane_angle))
}

/// One placed chain node: position + the direction at that node.
struct ChainNode {
    pos: V3,
    dir: V3,
}

/// Place `int_nodes` segments along a curving branch. Returns one entry per
/// placed node; the last entry's `.dir` is the next child's initial dir.
#[allow(clippy::too_many_arguments)]
fn build_curved_chain(
    start_pos: V3,
    initial_dir: V3,
    seg_len: f64,
    int_nodes: usize,
    arc_angle: f64,
    arc_axis: V3,
    droop_per_seg: f64,
    wiggle_per_seg: f64,
    wiggle_axis: V3,
    radius_scale: f64,
) -> Vec<ChainNode> {
    let mut result = Vec::with_capacity(int_nodes);
    let mut pos = start_pos;
    let mut dir = initial_dir;
    let arc_per_seg = if int_nodes > 1 {
        arc_angle / int_nodes as f64
    } else {
        0.0
    };

    for j in 0..int_nodes {
        // Step 1: arc curvature.
        if arc_per_seg.abs() > 1e-6 {
            dir = normalize(rotate_around(dir, arc_axis, arc_per_seg));
        }
        // Step 2: gravitropic droop.
        if droop_per_seg.abs() > 1e-6 {
            dir = normalize(rotate_around(dir, perp_to(dir), droop_per_seg));
        }
        // Step 3: per-node wiggle.
        if wiggle_per_seg.abs() > 1e-6 {
            let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
            dir = normalize(rotate_around(dir, wiggle_axis, wiggle_per_seg * sign));
        }
        // Advance position.
        pos = [
            pos[0] + dir[0] * seg_len * radius_scale,
            pos[1] + dir[1] * seg_len * radius_scale,
            pos[2] + dir[2] * seg_len * radius_scale,
        ];
        result.push(ChainNode { pos, dir });
    }
    result
}

// ---------------------------------------------------------------------------
// applySymmetry — ONE general operator for continuous radialOrder.
//
// Returns a Vec of instances; each instance is a Vec of positions (one per
// appendage-spec offset). `_rng` in JS is passed for signature compat and NOT
// consumed — omitted here.
// ---------------------------------------------------------------------------

/// Distribute `appendage_spec` offsets around `axis` per the continuous
/// `radial_order` gene. Bit-faithful port of `applySymmetry`.
fn apply_symmetry(
    appendage_spec: &[V3],
    radial_order: f64,
    axis: V3,
    attach_site: V3,
) -> Vec<Vec<V3>> {
    let golden_angle = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
    let golden_advance = 0.6;

    let n = 2 + (radial_order * 5.0).floor() as i64;

    let radial_step = (std::f64::consts::PI * 2.0) / n as f64;
    let angle_step = if radial_order <= 0.5 {
        let t = radial_order * 2.0;
        lerp(std::f64::consts::PI, radial_step, t)
    } else {
        let t = (radial_order - 0.5) * 2.0;
        lerp(radial_step, golden_angle, t)
    };

    let advance_per_copy_total = if radial_order > 0.5 {
        lerp(0.0, golden_advance, (radial_order - 0.5) * 2.0)
    } else {
        0.0
    };

    let mut instances = Vec::with_capacity(n.max(0) as usize);
    for i in 0..n {
        let rot = i as f64 * angle_step;
        let advance = if n > 1 {
            (i as f64 / (n - 1) as f64) * advance_per_copy_total - advance_per_copy_total * 0.5
        } else {
            0.0
        };

        let mut limb_nodes = Vec::with_capacity(appendage_spec.len());
        for spec in appendage_spec {
            let rot_off = rotate_around(*spec, axis, rot);
            limb_nodes.push([
                attach_site[0] + rot_off[0] + axis[0] * advance,
                attach_site[1] + rot_off[1] + axis[1] * advance,
                attach_site[2] + rot_off[2] + axis[2] * advance,
            ]);
        }
        instances.push(limb_nodes);
    }
    instances
}

// ---------------------------------------------------------------------------
// Graph types — the skeleton output contract consumed by proportions/skin/roots.
// ---------------------------------------------------------------------------

/// One skeleton node. Mirrors the JS node object; optional JS fields become
/// `Option`/bool flags. `radius` is set by buildSkeleton and rewritten by
/// `solveProportions`.
#[derive(Clone, Debug)]
pub struct Node {
    pub pos: V3,
    pub radius: f64,
    pub weight: f64,
    pub branch_level: i32,
    /// `-1` for the trunk-base node (JS `parentIdx === -1`).
    pub parent_idx: i32,
    // Flags (JS: presence of a key / boolean).
    pub is_stem: bool,
    pub is_woody: bool,
    pub is_terminal: bool,
    pub is_root: bool,
    /// Set by `markTerminal` (terminal nodes only).
    pub attach_pos: Option<V3>,
    /// Set by `markTerminal` — geometry-derived normal axis for skin.
    pub flat_normal: Option<V3>,
}

impl Node {
    #[inline]
    fn new(pos: V3, radius: f64, weight: f64, branch_level: i32, parent_idx: i32) -> Self {
        Self {
            pos,
            radius,
            weight,
            branch_level,
            parent_idx,
            is_stem: false,
            is_woody: false,
            is_terminal: false,
            is_root: false,
            attach_pos: None,
            flat_normal: None,
        }
    }
}

/// A bone connecting two node indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bone {
    pub a: usize,
    pub b: usize,
}

/// Skeleton meta — body axis + the (placeholder) light dir (overridden by
/// `solveProportions` from `sunAngle`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Meta {
    pub body_axis: V3,
    pub light_dir: V3,
}

/// The skeleton graph: nodes + bones + meta.
#[derive(Clone, Debug)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub bones: Vec<Bone>,
    pub meta: Meta,
}

// ---------------------------------------------------------------------------
// BFS queue item
// ---------------------------------------------------------------------------

struct QueueItem {
    node_idx: usize,
    level: i32,
    dir: V3,
    len: f64,
    attach_pos: V3,
    radius_scale: f64,
    child_index: f64,
}

// ---------------------------------------------------------------------------
// buildSkeleton
// ---------------------------------------------------------------------------

/// Build the branch skeleton for `genome`, consuming `rng` in the exact dryad
/// draw order. `jitter_amp` scales ALL per-node jitter magnitudes uniformly and
/// NEVER affects the rng draw count/order.
pub fn build_skeleton(genome: &Genome, rng: &mut Mulberry32, jitter_amp: f64) -> Graph {
    let branchiness = genome.branchiness;
    let branch_factor_n = genome.branch_factor_n;
    let tillering = genome.tillering;
    let radial_order = genome.radial_order;
    let segmentation = genome.segmentation;
    let branch_angle = genome.branch_angle;
    let length_ratio = genome.length_ratio;
    let apical_bias = genome.apical_bias;
    let jitter = genome.jitter;
    let structural_seed_u = genome.structural_seed;
    let trunk_height = genome.trunk_height;

    // structuralSeed [0,1] in JS is the *gene*; in the genome it is stored as the
    // u32 draw. dryad's buildSkeleton reads `genome.structuralSeed` (the u32) and
    // uses it directly in the sin-hash. We mirror that: cast u32 → f64.
    let structural_seed = structural_seed_u as f64;

    let stem_spread = genome.stem_spread;
    let rosette = genome.rosette;
    let crown_start = genome.crown_start;

    // trunkHeightFactor + allometry traits (no rng).
    let trunk_height_factor = compute_size_factor(trunk_height);
    let traits: Traits = derive_traits(trunk_height);

    // Derive integer structural parameters from continuous genes.
    let int_nodes = (2.0 + segmentation * 1.0).floor().max(2.0).min(3.0) as usize;
    let trunk_segments_count = (5.0 + segmentation * 3.0).floor().max(5.0).min(8.0) as usize;

    // Fractional branch depth.
    let fract_depth = branchiness * 7.0;
    let base_max_depth = fract_depth.floor() as i32;
    let depth_frac = fract_depth - base_max_depth as f64;
    let max_depth = if base_max_depth > 0 {
        (base_max_depth + traits.branch_order_bonus).max(1)
    } else {
        0
    };

    // Fractional children.
    let fract_children = 1.0 + branch_factor_n * 3.0;
    let full_children = fract_children.floor() as i32;
    let children_frac = fract_children - full_children as f64;

    // Fractional tillering.
    let fract_stems = 1.0 + tillering * 6.0;
    let full_stems = fract_stems.floor() as i32;
    let stem_frac = fract_stems - full_stems as f64;

    // Combined jitter.
    let j_amp = jitter * jitter_amp;

    let mut nodes: Vec<Node> = Vec::new();
    let mut bones: Vec<Bone> = Vec::new();

    const BODY_AXIS: V3 = [0.0, 1.0, 0.0];
    let light_dir = normalize([0.4, 0.9, 0.3]);
    let soft_ceiling = MAX_BONES - 20;

    const MAX_FOOTPRINT: f64 = 0.6;
    const ROSETTE_MAX: f64 = 12.0;

    // RNG draw 1: trunkBend. RNG draw 2: trunkLean.
    let trunk_bend = (rng.next() - 0.5) * 0.35 * j_amp;
    let trunk_lean = (rng.next() - 0.5) * 0.25 * j_amp;

    // buildTrunk: create trunk nodes for one stem, returning all node indices.
    const MAX_STEM_FAN_RADIUS: f64 = 0.18;
    let build_trunk = |nodes: &mut Vec<Node>,
                       bones: &mut Vec<Bone>,
                       base_pos: V3,
                       weight: f64,
                       stem_az: f64|
     -> Vec<usize> {
        let mut trunk_node_indices = Vec::with_capacity(trunk_segments_count);
        for i in 0..trunk_segments_count {
            let t = if trunk_segments_count > 1 {
                i as f64 / (trunk_segments_count - 1) as f64
            } else {
                0.0
            };
            let trunk_h = TRUNK_HEIGHT * trunk_height_factor * weight;
            let bend_arc = trunk_bend * (t * std::f64::consts::PI).sin() * weight;
            let lean_arc = trunk_lean * (t * std::f64::consts::PI).sin() * weight;
            let fan_amp = MAX_STEM_FAN_RADIUS * (t * std::f64::consts::PI).sin() * weight;
            let x_off = base_pos[0] + bend_arc + fan_amp * stem_az.cos();
            let z_off = base_pos[2] + lean_arc + fan_amp * stem_az.sin();
            let y_pos = base_pos[1] + t * trunk_h;

            let idx = nodes.len();
            let mut node = Node::new(
                [x_off, y_pos, z_off],
                0.12 * weight,
                weight,
                0,
                if i == 0 { -1 } else { idx as i32 - 1 },
            );
            node.is_stem = true;
            nodes.push(node);
            trunk_node_indices.push(idx);

            if i > 0 {
                bones.push(Bone { a: idx - 1, b: idx });
            }
        }
        trunk_node_indices
    };

    // TILLERING: fractional basal stem count.
    let total_stems = full_stems + if stem_frac > 0.0 { 1 } else { 0 };

    // Draw one azimuth jitter per stem (always — draws 3..3+totalStems-1).
    let mut stem_azimuths = Vec::with_capacity(total_stems.max(0) as usize);
    for _ in 0..total_stems {
        stem_azimuths.push(rng.next() * std::f64::consts::PI * 2.0);
    }

    // { idx: nodeIdx, weight }
    let mut all_trunk_tops: Vec<(usize, f64)> = Vec::new();
    // { chain: nodeIdx[], weight } — full trunk node chains (for crownStart).
    let mut all_trunk_chains: Vec<(Vec<usize>, f64)> = Vec::new();

    if total_stems == 1 {
        let ti = build_trunk(&mut nodes, &mut bones, [0.0, 0.0, 0.0], 1.0, 0.0);
        all_trunk_tops.push((*ti.last().unwrap(), 1.0));
        all_trunk_chains.push((ti, 1.0));
    } else {
        for s in 0..total_stems {
            let az = stem_azimuths[s as usize];
            let is_crossfade = s == full_stems;
            let weight = if is_crossfade { stem_frac } else { 1.0 };
            let r = stem_spread * MAX_FOOTPRINT * weight;
            let base_pos = [az.cos() * r, 0.0, az.sin() * r];
            let ti = build_trunk(&mut nodes, &mut bones, base_pos, weight, az);
            all_trunk_tops.push((*ti.last().unwrap(), weight));
            all_trunk_chains.push((ti, weight));
        }
    }

    // RISK D MITIGATION: reserve the 3 ex-stub root bones for budget parity.
    const ROOT_STUB_RESERVED_BONES: usize = 3;
    let mut bone_counter = bones.len() + ROOT_STUB_RESERVED_BONES;

    // -------------------------------------------------------------------------
    // markTerminal — tag a node terminal; consumes 1 rng draw (flatNormal azimuth).
    // Kept as a closure-free local fn taking the data it needs (Rust borrow rules).
    // -------------------------------------------------------------------------
    fn mark_terminal(nodes: &mut [Node], rng: &mut Mulberry32, node_idx: usize, branch_dir: V3, attach_pos: V3) {
        {
            let node = &mut nodes[node_idx];
            node.is_woody = false; // JS `delete node.isWoody`
            node.is_terminal = true;
            node.attach_pos = Some(attach_pos);
        }
        let azimuth_jitter = rng.next() * std::f64::consts::PI * 2.0; // 1 draw, always

        let parallel_test = dot(branch_dir, BODY_AXIS).abs();
        let raw_normal = if parallel_test > 0.97 {
            [1.0, 0.0, 0.0]
        } else {
            normalize(cross(branch_dir, BODY_AXIS))
        };
        nodes[node_idx].flat_normal = Some(rotate_around(raw_normal, branch_dir, azimuth_jitter));
    }

    // -------------------------------------------------------------------------
    // Whorl tier constants.
    // -------------------------------------------------------------------------
    const WHORL_MAX_TIERS: f64 = 9.0;
    const WHORL_BRANCHES_PER_TIER: usize = 7;
    const WHORL_START_FRAC_MAX: f64 = 0.65;
    const WHORL_TRUNK_END_FRAC: f64 = 0.95;
    const WHORL_BASE_LEN_FRAC: f64 = 0.30;
    const WHORL_TOP_LEN_FRAC: f64 = 0.07;
    let inter_tier_offset = std::f64::consts::PI / WHORL_BRANCHES_PER_TIER as f64;
    const WHORL_CROWN_ANGLE: f64 = 0.25;
    const WHORL_BASE_ANGLE: f64 = 1.15;

    let mut queue: Vec<QueueItem> = Vec::new();

    let whorl = genome.whorl;

    if whorl > 0.0 && !all_trunk_tops.is_empty() {
        // tierBuckets[ti] = Vec<QueueItem>, bucketed for budget-aware enqueue order.
        let mut tier_buckets: Vec<Vec<QueueItem>> = Vec::new();

        // Collect trunk chain from dominant (first) stem, ascending-Y.
        let dominant_trunk_top_idx = all_trunk_tops[0].0;
        let mut trunk_chain: Vec<(usize, f64)> = Vec::new(); // (nodeIdx, y)
        {
            let mut cur: i32 = dominant_trunk_top_idx as i32;
            while cur != -1 {
                let n = &nodes[cur as usize];
                if !n.is_stem {
                    break;
                }
                trunk_chain.push((cur as usize, n.pos[1]));
                cur = n.parent_idx;
            }
            trunk_chain.reverse(); // base → top
        }

        if trunk_chain.len() >= 2 {
            let trunk_base_y = trunk_chain[0].1;
            let trunk_top_y = trunk_chain[trunk_chain.len() - 1].1;
            let trunk_span = trunk_top_y - trunk_base_y;

            let start_frac = lerp(0.0, WHORL_START_FRAC_MAX, whorl);

            const WHORL_TIER_HARD_MAX: f64 = 18.0;
            let tier_count_float =
                (whorl * WHORL_MAX_TIERS * traits.tier_count_scale).min(WHORL_TIER_HARD_MAX);
            let full_tiers = tier_count_float.floor() as i32;
            let whorl_tier_frac = tier_count_float - full_tiers as f64;

            let n_tiers = full_tiers + if whorl_tier_frac > 0.0 { 1 } else { 0 };

            for ti in 0..n_tiers {
                if bone_counter >= soft_ceiling {
                    break;
                }
                let is_crossfade_tier = ti == full_tiers && whorl_tier_frac > 0.0;
                let tier_radius_scale = if is_crossfade_tier { whorl_tier_frac } else { 1.0 };

                let tier_pos_frac = if n_tiers > 1 {
                    ti as f64 / (n_tiers - 1) as f64
                } else {
                    0.5
                };
                let frac = start_frac + (WHORL_TRUNK_END_FRAC - start_frac) * tier_pos_frac;
                let target_y = trunk_base_y + frac * trunk_span;

                // Snap to nearest trunk-chain node at or above target_y.
                let mut snap_idx: i32 = -1;
                let mut best_dist = f64::INFINITY;
                for &(node_idx, y) in &trunk_chain {
                    if y < target_y - 1e-6 {
                        continue;
                    }
                    let d = y - target_y;
                    if d < best_dist {
                        best_dist = d;
                        snap_idx = node_idx as i32;
                    }
                }
                if snap_idx == -1 {
                    snap_idx = trunk_chain[trunk_chain.len() - 1].0 as i32;
                }
                let snap_idx = snap_idx as usize;
                let snap_pos = nodes[snap_idx].pos;

                let height_ratio = 1.0 - tier_pos_frac;
                let conical_shape = 0.2 + 0.8 * height_ratio;
                let tier_branch_len = trunk_span
                    * lerp(WHORL_TOP_LEN_FRAC, WHORL_BASE_LEN_FRAC, height_ratio)
                    * conical_shape
                    * tier_radius_scale;

                let tier_polar_angle = lerp(WHORL_BASE_ANGLE, WHORL_CROWN_ANGLE, tier_pos_frac);
                let whorl_jitter = 0.4;
                let tier_rotation = ti as f64 * inter_tier_offset;
                let whorl_base_radius = BRANCH_BASE_RADIUS * TAPER_BASE.powi(1);

                for bi in 0..WHORL_BRANCHES_PER_TIER {
                    if bone_counter >= soft_ceiling {
                        break;
                    }
                    let base_az = (std::f64::consts::PI * 2.0 * bi as f64)
                        / WHORL_BRANCHES_PER_TIER as f64
                        + tier_rotation;
                    let jitter_hash = (structural_seed * 6.2831 + ti as f64 * 7.3 + bi as f64 * 2.9999)
                        .sin()
                        * whorl_jitter
                        * (1.0 - whorl);
                    let az = base_az + jitter_hash;

                    let branch_dir = normalize([
                        tier_polar_angle.sin() * az.cos(),
                        tier_polar_angle.cos(),
                        tier_polar_angle.sin() * az.sin(),
                    ]);

                    let child_key = (ti as usize * WHORL_BRANCHES_PER_TIER + bi) as f64;
                    let arc_angle = compute_curvature_arc(structural_seed, child_key, 1.0);
                    let arc_axis = compute_curvature_axis(branch_dir, structural_seed, child_key, 1.0);
                    let droop_per_seg = 0.04;
                    let seg_len = tier_branch_len / int_nodes as f64;

                    let chain_nodes = build_curved_chain(
                        snap_pos,
                        branch_dir,
                        seg_len,
                        int_nodes,
                        arc_angle,
                        arc_axis,
                        droop_per_seg,
                        0.0,
                        perp_to(branch_dir),
                        tier_radius_scale,
                    );

                    let whorl_radius = whorl_base_radius * tier_radius_scale;
                    let mut prev_idx = snap_idx;
                    for cn in &chain_nodes {
                        if bone_counter >= soft_ceiling {
                            break;
                        }
                        let new_idx = nodes.len();
                        let mut node = Node::new(cn.pos, whorl_radius, tier_radius_scale, 1, prev_idx as i32);
                        node.is_woody = true;
                        nodes.push(node);
                        bones.push(Bone { a: prev_idx, b: new_idx });
                        bone_counter += 1;
                        prev_idx = new_idx;
                    }

                    let tier_tail_idx = nodes.len() - 1;
                    if tier_tail_idx != snap_idx {
                        let tier_final_dir = chain_nodes[chain_nodes.len() - 1].dir;
                        let ti_us = ti as usize;
                        while tier_buckets.len() <= ti_us {
                            tier_buckets.push(Vec::new());
                        }
                        tier_buckets[ti_us].push(QueueItem {
                            node_idx: tier_tail_idx,
                            level: 1,
                            dir: tier_final_dir,
                            len: tier_branch_len * length_ratio,
                            attach_pos: nodes[tier_tail_idx].pos,
                            radius_scale: tier_radius_scale,
                            child_index: child_key,
                        });
                    }
                }
            }
        }

        // BFS ORDER: apex/leader first (short conifer spike), then round-robin
        // tiers TOP→bottom.
        const CONIFER_LEADER_FRAC: f64 = 0.22;
        for &(top_idx, weight) in &all_trunk_tops {
            queue.push(QueueItem {
                node_idx: top_idx,
                level: 0,
                dir: [0.0, 1.0, 0.0],
                len: BASE_BRANCH_LENGTH * TRUNK_HEIGHT * trunk_height_factor * CONIFER_LEADER_FRAC,
                attach_pos: nodes[top_idx].pos,
                radius_scale: weight,
                child_index: 0.0,
            });
        }
        let mut max_bucket = 0;
        for b in &tier_buckets {
            if b.len() > max_bucket {
                max_bucket = b.len();
            }
        }
        // Round-robin: k-th branch of every tier (top→bottom) before (k+1)-th.
        // We must move QueueItems out of the buckets; collect indices then drain.
        for k in 0..max_bucket {
            for ti in (0..tier_buckets.len()).rev() {
                if k < tier_buckets[ti].len() {
                    // Take by swap-removal would reorder; instead use Option slots.
                    // Simpler: rebuild via mem::take after the loop is hard, so we
                    // clone the item's scalar fields (QueueItem is small + Copy-able
                    // fields). Move out using std::mem::replace with a sentinel.
                    let item = std::mem::replace(
                        &mut tier_buckets[ti][k],
                        QueueItem {
                            node_idx: 0,
                            level: 0,
                            dir: [0.0, 0.0, 0.0],
                            len: 0.0,
                            attach_pos: [0.0, 0.0, 0.0],
                            radius_scale: 0.0,
                            child_index: 0.0,
                        },
                    );
                    queue.push(item);
                }
            }
        }
    } else {
        // whorl === 0: byte-identical path.
        for &(top_idx, weight) in &all_trunk_tops {
            queue.push(QueueItem {
                node_idx: top_idx,
                level: 0,
                dir: [0.0, 1.0, 0.0],
                len: BASE_BRANCH_LENGTH * TRUNK_HEIGHT * trunk_height_factor,
                attach_pos: nodes[top_idx].pos,
                radius_scale: weight,
                child_index: 0.0,
            });
        }

        // crownStart < 1: emit primary crown branches down to crownStart·H.
        if crown_start < 1.0 {
            let crown_golden = std::f64::consts::PI * (3.0 - 5.0_f64.sqrt());
            let mut lb_index: i64 = 0;
            for si in 0..all_trunk_chains.len() {
                let (chain, weight) = (&all_trunk_chains[si].0, all_trunk_chains[si].1);
                let n = chain.len();
                if n < 3 {
                    continue;
                }
                let base_az = if si < stem_azimuths.len() {
                    stem_azimuths[si]
                } else {
                    0.0
                };
                for k in 1..(n - 1) {
                    let t = k as f64 / (n - 1) as f64;
                    if t < crown_start {
                        continue;
                    }
                    let az = base_az + lb_index as f64 * crown_golden;
                    lb_index += 1;
                    let up: f64 = 0.55;
                    let s = (1.0 - up * up).max(0.0).sqrt();
                    let dir = normalize([az.cos() * s, up, az.sin() * s]);
                    let primary_len = BASE_BRANCH_LENGTH * TRUNK_HEIGHT * trunk_height_factor;
                    queue.push(QueueItem {
                        node_idx: chain[k],
                        level: 0,
                        dir,
                        len: primary_len * (0.55 + 0.35 * (1.0 - t)),
                        attach_pos: nodes[chain[k]].pos,
                        radius_scale: weight,
                        child_index: k as f64,
                    });
                }
            }
        }
    }

    // -------------------------------------------------------------------------
    // BFS BRANCHING
    // -------------------------------------------------------------------------
    let mut queue_head = 0;
    while queue_head < queue.len() {
        // Copy out the scalar item fields (no borrow held across mutation).
        let (node_idx, level, dir, len, attach_pos, radius_scale, child_index) = {
            let item = &queue[queue_head];
            (
                item.node_idx,
                item.level,
                item.dir,
                item.len,
                item.attach_pos,
                item.radius_scale,
                item.child_index,
            )
        };
        queue_head += 1;

        let next_level = level + 1;

        let is_deepest_level = next_level == max_depth;
        let tip_length_mult = if is_deepest_level { TIP_LENGTH_BOOST } else { 1.0 };
        let child_len = len * length_ratio * tip_length_mult;

        // Terminate: exceeded max depth or bone budget exhausted.
        if next_level > max_depth || bone_counter >= soft_ceiling {
            mark_terminal(&mut nodes, rng, node_idx, dir, attach_pos);
            continue;
        }

        let child_cost = int_nodes;
        let max_affordable_children = (soft_ceiling - bone_counter) / child_cost;

        if max_affordable_children < 1 {
            mark_terminal(&mut nodes, rng, node_idx, dir, attach_pos);
            continue;
        }

        let depth_weight = if next_level == max_depth && depth_frac < 1.0 {
            depth_frac
        } else {
            1.0
        };

        let level_taper = TAPER_BASE.powi(next_level);

        let total_children = full_children + if children_frac > 0.0 { 1 } else { 0 };
        let actual_children = (total_children as usize).min(max_affordable_children);

        let perp = perp_to(dir);

        // Consume rng draws: ALWAYS 3 per child (determinism — count fixed).
        struct Draws {
            draw_a: f64,
            draw_b: f64,
        }
        let mut draws = Vec::with_capacity(actual_children);
        for _ in 0..actual_children {
            let draw_a = (rng.next() - 0.5) * 0.16 * j_amp;
            let draw_b = (rng.next() - 0.5) * (std::f64::consts::PI / 18.0) * j_amp;
            // 3rd draw per child is consumed for determinism (count fixed) but unused.
            let _draw_c = (rng.next() - 0.5) * (std::f64::consts::PI / 9.0) * j_amp;
            draws.push(Draws { draw_a, draw_b });
        }

        let parent_pos = nodes[node_idx].pos;

        // Leader (child index 0).
        {
            let leader_len = child_len * (1.0 + apical_bias * 0.3) * (1.0 + draws[0].draw_a);
            let max_trunk_leader_divergence = std::f64::consts::PI / 36.0; // 5° cap
            let raw_leader_divergence = branch_angle * 0.15 + draws[0].draw_b;
            let leader_divergence = if level == 0 {
                raw_leader_divergence
                    .max(-max_trunk_leader_divergence)
                    .min(max_trunk_leader_divergence)
            } else {
                raw_leader_divergence
            };
            let leader_initial_dir = normalize(rotate_around(dir, perp, leader_divergence));

            let child_radius_scale = radius_scale * depth_weight;
            let node_radius = BRANCH_BASE_RADIUS * child_radius_scale * level_taper;

            let arc_angle = compute_curvature_arc(structural_seed, child_index, next_level as f64);
            let arc_axis =
                compute_curvature_axis(leader_initial_dir, structural_seed, child_index, next_level as f64);
            let droop_per_seg = 0.04 * (next_level as f64 / (max_depth as f64 + 1.0));
            let wiggle_per_seg = draws[0].draw_b * 0.35;
            let wiggle_axis = perp_to(leader_initial_dir);
            let seg_len = leader_len / int_nodes as f64;

            let chain_nodes = build_curved_chain(
                parent_pos,
                leader_initial_dir,
                seg_len,
                int_nodes,
                arc_angle,
                arc_axis,
                droop_per_seg,
                wiggle_per_seg,
                wiggle_axis,
                child_radius_scale,
            );

            let mut prev_idx = node_idx;
            for cn in &chain_nodes {
                let new_idx = nodes.len();
                let mut node = Node::new(cn.pos, node_radius, child_radius_scale, next_level, prev_idx as i32);
                node.is_woody = true;
                nodes.push(node);
                bones.push(Bone { a: prev_idx, b: new_idx });
                bone_counter += 1;
                prev_idx = new_idx;
            }

            let leader_tail_idx = nodes.len() - 1;
            let leader_final_dir = chain_nodes[chain_nodes.len() - 1].dir;
            queue.push(QueueItem {
                node_idx: leader_tail_idx,
                level: next_level,
                dir: leader_final_dir,
                len: child_len,
                attach_pos: nodes[leader_tail_idx].pos,
                radius_scale: child_radius_scale,
                child_index: 0.0,
            });
        }

        // Laterals (indices 1..actualChildren-1) via applySymmetry.
        let num_laterals = actual_children as i64 - 1;
        if num_laterals > 0 {
            let apical_suppress = 1.0 - apical_bias * 0.45;
            let spread_boost = if level > 0 { LATERAL_SPREAD_BOOST } else { 0.0 };
            let lateral_branch_angle =
                (branch_angle + spread_boost) * (1.0 - apical_bias * 0.40) + draws[1].draw_b;
            let lateral_len = child_len * (1.0 + draws[1].draw_a) * apical_suppress;
            let lateral_base_dir = normalize([
                dir[0] * lateral_branch_angle.cos() + perp[0] * lateral_branch_angle.sin(),
                dir[1] * lateral_branch_angle.cos() + perp[1] * lateral_branch_angle.sin(),
                dir[2] * lateral_branch_angle.cos() + perp[2] * lateral_branch_angle.sin(),
            ]);

            let lateral_spec_single_step = [lateral_base_dir];
            let lateral_instances =
                apply_symmetry(&lateral_spec_single_step, radial_order, dir, [0.0, 0.0, 0.0]);

            let used = (num_laterals as usize).min(lateral_instances.len());
            #[allow(clippy::needless_range_loop)]
            for li in 0..used {
                // Extra crossfade child (beyond fullChildren) scaled by childrenFrac.
                let is_xfade_child = (li as i64 + 2) > full_children as i64 && children_frac > 0.0;
                let lateral_crossfade_weight = if is_xfade_child { children_frac } else { 1.0 };
                let child_radius_scale = radius_scale * depth_weight * lateral_crossfade_weight;
                let node_radius = BRANCH_BASE_RADIUS * child_radius_scale * level_taper;

                let rotated_offset = lateral_instances[li][0];
                let lateral_initial_dir = normalize(rotated_offset);

                let lateral_child_idx = child_index + li as f64 + 1.0;
                let arc_angle =
                    compute_curvature_arc(structural_seed, lateral_child_idx, next_level as f64);
                let arc_axis = compute_curvature_axis(
                    lateral_initial_dir,
                    structural_seed,
                    lateral_child_idx,
                    next_level as f64,
                );
                let droop_per_seg = 0.04 * (next_level as f64 / (max_depth as f64 + 1.0));
                let wiggle_per_seg = draws[1].draw_b * 0.35;
                let wiggle_axis = perp_to(lateral_initial_dir);
                let seg_len = lateral_len / int_nodes as f64;

                let chain_nodes = build_curved_chain(
                    parent_pos,
                    lateral_initial_dir,
                    seg_len,
                    int_nodes,
                    arc_angle,
                    arc_axis,
                    droop_per_seg,
                    wiggle_per_seg,
                    wiggle_axis,
                    child_radius_scale,
                );

                let mut prev_idx = node_idx;
                for cn in &chain_nodes {
                    let new_idx = nodes.len();
                    let mut node =
                        Node::new(cn.pos, node_radius, child_radius_scale, next_level, prev_idx as i32);
                    node.is_woody = true;
                    nodes.push(node);
                    bones.push(Bone { a: prev_idx, b: new_idx });
                    bone_counter += 1;
                    prev_idx = new_idx;
                }

                let lateral_tail_idx = nodes.len() - 1;
                let lateral_final_dir = chain_nodes[chain_nodes.len() - 1].dir;
                queue.push(QueueItem {
                    node_idx: lateral_tail_idx,
                    level: next_level,
                    dir: lateral_final_dir,
                    len: child_len,
                    attach_pos: nodes[lateral_tail_idx].pos,
                    radius_scale: child_radius_scale,
                    child_index: lateral_child_idx,
                });
            }
        }
    }

    // -------------------------------------------------------------------------
    // ROSETTE POST-PASS — gated on rosette > 0 (zero draws otherwise).
    // -------------------------------------------------------------------------
    if rosette > 0.0 && !all_trunk_tops.is_empty() {
        let fract_fronds = rosette * ROSETTE_MAX;
        let full_fronds = fract_fronds.floor() as i32;
        let frond_frac = fract_fronds - full_fronds as f64;
        let n = (full_fronds + if frond_frac > 0.0 { 1 } else { 0 }).max(1);

        let trunk_top_idx = all_trunk_tops[0].0;
        let trunk_top_pos = nodes[trunk_top_idx].pos;

        let frond_len = BASE_BRANCH_LENGTH * TRUNK_HEIGHT * trunk_height_factor * length_ratio;
        let frond_segments = int_nodes.max(5);
        let frond_seg_len = frond_len / frond_segments as f64;
        let frond_radius = BRANCH_BASE_RADIUS * TAPER_BASE.powi(1);

        const FROND_POLAR_MIN: f64 = 0.35;
        const FROND_POLAR_MAX: f64 = 1.25;
        const FROND_ARCH_MIN: f64 = 0.70;
        const FROND_ARCH_MAX: f64 = 1.90;

        for fi in 0..n {
            if bone_counter >= soft_ceiling {
                break;
            }
            let az = (std::f64::consts::PI * 2.0 * fi as f64) / n as f64;
            let elev_hash = 0.5 + 0.5 * (structural_seed * 4.7 + fi as f64 * 1.873).sin();
            let frond_polar = lerp(FROND_POLAR_MIN, FROND_POLAR_MAX, elev_hash);
            let frond_arch = lerp(FROND_ARCH_MIN, FROND_ARCH_MAX, elev_hash);

            let frond_dir = normalize([
                frond_polar.sin() * az.cos(),
                frond_polar.cos(),
                frond_polar.sin() * az.sin(),
            ]);
            let arch_axis = normalize([az.sin(), 0.0, -az.cos()]);

            let frond_weight = if fi == full_fronds && frond_frac > 0.0 {
                frond_frac
            } else {
                1.0
            };

            let chain_nodes = build_curved_chain(
                trunk_top_pos,
                frond_dir,
                frond_seg_len,
                frond_segments,
                frond_arch,
                arch_axis,
                0.0,
                0.0,
                perp_to(frond_dir),
                frond_weight,
            );

            let mut prev_idx = trunk_top_idx;
            for cn in &chain_nodes {
                if bone_counter >= soft_ceiling {
                    break;
                }
                let new_idx = nodes.len();
                let mut node = Node::new(cn.pos, frond_radius, frond_weight, 1, prev_idx as i32);
                node.is_woody = true;
                nodes.push(node);
                bones.push(Bone { a: prev_idx, b: new_idx });
                bone_counter += 1;
                prev_idx = new_idx;
            }

            if prev_idx != trunk_top_idx {
                mark_terminal(&mut nodes, rng, prev_idx, frond_dir, trunk_top_pos);
            }
        }
    }

    Graph {
        nodes,
        bones,
        meta: Meta {
            body_axis: BODY_AXIS,
            light_dir,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{random_genome, Env};

    #[test]
    fn deterministic_same_seed() {
        let env = Env::default();
        let g = random_genome(&env, 42);
        let mut r1 = Mulberry32::new(g.structural_seed);
        let mut r2 = Mulberry32::new(g.structural_seed);
        let a = build_skeleton(&g, &mut r1, g.jitter);
        let b = build_skeleton(&g, &mut r2, g.jitter);
        assert_eq!(a.nodes.len(), b.nodes.len());
        assert_eq!(a.bones.len(), b.bones.len());
        for (na, nb) in a.nodes.iter().zip(b.nodes.iter()) {
            assert_eq!(na.pos, nb.pos);
            assert_eq!(na.radius, nb.radius);
        }
    }

    #[test]
    fn jitter_amp_invariant_draw_count() {
        // Same genome, two jitter_amp values → SAME node/bone count (draw count
        // is invariant to jitterAmp).
        let env = Env::default();
        let g = random_genome(&env, 7);
        let mut r1 = Mulberry32::new(g.structural_seed);
        let mut r2 = Mulberry32::new(g.structural_seed);
        let a = build_skeleton(&g, &mut r1, 1.0);
        let b = build_skeleton(&g, &mut r2, 0.0);
        assert_eq!(a.nodes.len(), b.nodes.len());
        assert_eq!(a.bones.len(), b.bones.len());
    }

    #[test]
    fn emits_trunk_and_branches() {
        let env = Env::default();
        let g = random_genome(&env, 1);
        let mut r = Mulberry32::new(g.structural_seed);
        let graph = build_skeleton(&g, &mut r, g.jitter);
        assert!(graph.nodes.len() >= 5, "expected at least a trunk chain");
        assert!(graph.bones.len() <= MAX_BONES, "bone budget respected");
        // parentIdx ordering invariant: parents precede children.
        for (i, n) in graph.nodes.iter().enumerate() {
            if n.parent_idx >= 0 {
                assert!((n.parent_idx as usize) < i, "parent must precede child");
            }
        }
        // No isRoot nodes emitted by the skeleton stage.
        assert!(graph.nodes.iter().all(|n| !n.is_root));
    }
}

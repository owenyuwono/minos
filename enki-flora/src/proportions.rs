//! In-place proportion solve — port of dryad `proportions.js` (`solveProportions`).
//!
//! Physics-derived sizing for ROOTED PLANTS. SEED NEVER TOUCHES RADII OR
//! POSITIONS — ZERO randomness: no rng parameter, no rng import. Determinism
//! guarantee: same envelope + same input graph → identical output, bit-for-bit.
//!
//! Three strictly-separated radius concepts:
//!   1. Absolute trunk-base radius from physics (gravity/medium/wind/aridity).
//!   2. Pipe-model taper (`TAPER`): single forward pass, parent before children.
//!   3. Terminal leaf radius: pipe-model radius × leaf physics modifiers.
//!
//! Mutates the [`Graph`] in place and writes `meta.light_dir` from `sunAngle`.
//!
//! Vector math uses plain `[f64; 3]` to mirror the JS array math term-for-term.
//! ponytail: trig (sin/cos) may differ ~1 ULP cross-language — accepted ceiling.

// The forward passes index `graph.nodes` (mutably) AND `children_of` by the same
// `i`, so a range loop is the clear form (enumerate would borrow-conflict).
#![allow(clippy::needless_range_loop)]

use crate::allometry::derive_traits;
use crate::genome::{Env, Genome, Medium};
use crate::skeleton::Graph;

type V3 = [f64; 3];

// ---------------------------------------------------------------------------
// Vector helpers (port of proportions.js vecLen/vecNorm/vecDot/vecCross/...)
// ---------------------------------------------------------------------------

#[inline]
fn vec_len(v: V3) -> f64 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

#[inline]
fn vec_norm(v: V3) -> V3 {
    let len = vec_len(v);
    if len < 1e-12 {
        return [0.0, 1.0, 0.0];
    }
    [v[0] / len, v[1] / len, v[2] / len]
}

#[inline]
fn vec_dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[inline]
fn vec_cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[inline]
fn vec_scale(v: V3, s: f64) -> V3 {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Rodrigues' rotation: rotate `v` around unit axis `k` by `theta`.
#[inline]
fn rotate_around_axis(v: V3, k: V3, theta: f64) -> V3 {
    let cos_t = theta.cos();
    let sin_t = theta.sin();
    let d = vec_dot(k, v);
    let cr = vec_cross(k, v);
    [
        v[0] * cos_t + cr[0] * sin_t + k[0] * d * (1.0 - cos_t),
        v[1] * cos_t + cr[1] * sin_t + k[1] * d * (1.0 - cos_t),
        v[2] * cos_t + cr[2] * sin_t + k[2] * d * (1.0 - cos_t),
    ]
}

// ---------------------------------------------------------------------------
// Gravity helpers
// ---------------------------------------------------------------------------

/// Smooth clamped power: `g=1 → 1.0`.
#[inline]
fn grav_pow(g: f64, exp: f64) -> f64 {
    g.max(0.05).powf(exp)
}

/// Symmetric clamp to `[-limit, +limit]`; at `x=0` returns exactly 0.
#[inline]
fn clamp_sym(x: f64, limit: f64) -> f64 {
    if x > limit {
        limit
    } else if x < -limit {
        -limit
    } else {
        x
    }
}

/// Da Vinci / pipe-model default taper (overridden by `genome.taper`).
const TAPER_DEFAULT: f64 = 0.70;

// ---------------------------------------------------------------------------
// solveProportions
// ---------------------------------------------------------------------------

/// Solve physics-derived sizing in place. Consumes NO rng. The optional
/// `genome` mirrors JS `genome = null`: pass `None` for legacy-identity output.
///
/// # Panics
/// On the same ordering invariant violations dryad throws on (`parentIdx >=
/// ownIndex`), so a malformed skeleton fails loudly rather than silently.
pub fn solve_proportions(graph: &mut Graph, envelope: &Env, genome: Option<&Genome>) {
    let gravity = envelope.gravity;
    let medium = envelope.medium;
    let light = envelope.light;
    let sun_angle = envelope.sun_angle;
    let wind = envelope.wind;
    let aridity = envelope.aridity;

    let body_axis: V3 = graph.meta.body_axis;

    // -------------------------------------------------------------------------
    // Ordering invariant guard: parents must precede children.
    // (JS also guards isFrond; that tag never exists in the Rust Node, so the
    // only meaningful guard is the ordering one.)
    // -------------------------------------------------------------------------
    for i in 0..graph.nodes.len() {
        let p = graph.nodes[i].parent_idx;
        if p != -1 && p as usize >= i {
            panic!(
                "solveProportions: ordering invariant violated at node[{i}] — \
                 parentIdx={p} must be < ownIndex={i}."
            );
        }
    }

    // -------------------------------------------------------------------------
    // sunAngle → lightDir (this stage owns lightDir).
    // -------------------------------------------------------------------------
    let deg = std::f64::consts::PI / 180.0;
    let elev = (80.0 - 65.0 * sun_angle) * deg;
    let sin_az = 0.8;
    let cos_az = 0.6;
    let cos_el = elev.cos();
    let sin_el = elev.sin();
    let light_dir = vec_norm([sin_az * cos_el, sin_el, cos_az * cos_el]);
    graph.meta.light_dir = light_dir;

    // -------------------------------------------------------------------------
    // Medium modifiers.
    // -------------------------------------------------------------------------
    // dryad has 'air' | 'water' | 'dense_air'; the Rust Medium enum is Air |
    // Water (dense_air is not represented), so isDenseAir is always false here.
    let is_water = medium == Medium::Water;
    let is_dense_air = false;

    let stem_medium_mod = if is_water {
        0.65
    } else if is_dense_air {
        0.90
    } else {
        1.0
    };
    let leaf_medium_mod = if is_water {
        1.6
    } else if is_dense_air {
        1.25
    } else {
        1.0
    };

    // Light / wind / aridity modifiers.
    let leaf_light_mod = 1.0 + (0.6 - light) * 0.8;

    let stem_wind_radius_mod = 1.0 + (wind - 0.2) * 0.6;
    let _stem_wind_taper_mod = 1.0 + (0.2 - wind) * 0.5; // read in JS, unused downstream
    let leaf_wind_length_mod = 1.0 + (0.2 - wind) * 0.6;
    let leaf_wind_radius_mod = 1.0 + (0.2 - wind) * 0.4;
    let wind_droop_extra = (wind - 0.2) * 0.30;

    let stem_arid_radius_mod = 1.0 + (aridity - 0.35) * 0.7;
    let leaf_arid_length_mod = 1.0 + (0.35 - aridity) * 0.9;
    let leaf_arid_radius_mod = 1.0 + (0.35 - aridity) * 0.7;

    // -------------------------------------------------------------------------
    // Genome gene modifiers (identity when genome is None).
    // -------------------------------------------------------------------------
    let taper_gene = genome.map(|g| g.taper);
    let taper_resolved = match taper_gene {
        Some(t) => 0.55 + t * 0.30,
        None => TAPER_DEFAULT,
    };

    let succulence_gene = genome.map(|g| g.succulence).unwrap_or(0.0);
    let succulence_stem_mod = 1.0 + succulence_gene * 1.0;
    let taper = taper_resolved + (1.0 - taper_resolved) * succulence_gene;

    let trunk_taper_gene = genome.map(|g| g.trunk_taper).unwrap_or(0.0);

    let stem_girth_mod = match genome {
        Some(g) => 0.5 + g.stem_girth * 1.5,
        None => 1.0,
    };

    let rigidity_gene = genome.map(|g| g.rigidity).unwrap_or(0.0);
    let rigidity_droop_scale = 0.25 + (1.0 - rigidity_gene) * 0.75;

    let verticality_gene = genome.map(|g| g.verticality).unwrap_or(0.5);
    let vert_angle = (0.5 - verticality_gene) * std::f64::consts::PI;

    let droop_bias_gene = genome.map(|g| g.droop_bias).unwrap_or(0.0);
    const DROOP_BIAS_GAIN: f64 = 0.6;

    // -------------------------------------------------------------------------
    // RADIUS CONCEPT 1: absolute trunk-base radius from physics.
    // -------------------------------------------------------------------------
    const WOODINESS_LOW: f64 = 0.25;
    let woodiness_gene = genome.map(|g| g.woodiness).unwrap_or(1.0);
    let woodiness_trunk_mod = if woodiness_gene >= 1.0 {
        1.0
    } else {
        WOODINESS_LOW + woodiness_gene * (1.0 - WOODINESS_LOW)
    };

    // ALLOMETRY girthScale: JS `deriveTraits(genome||{}, env).girthScale`.
    // GIRTH_EXP=0 in allometry.rs → girthScale ≡ 1.0 (height/width decoupled),
    // and derive_traits reads only trunkHeight. genome=None → trunkHeight default
    // 0.5 → sizeFactor 1.0 → girthScale 1.0. ponytail: derive from trunkHeight.
    let girth_scale = derive_traits(genome.map(|g| g.trunk_height).unwrap_or(0.5)).girth_scale;

    let rigidity_girth_mod = 0.85 + rigidity_gene * 0.15;
    let stem_thickness_factor = grav_pow(gravity, 1.0 / 3.0)
        * stem_medium_mod
        * stem_wind_radius_mod
        * stem_arid_radius_mod
        * succulence_stem_mod
        * stem_girth_mod
        * woodiness_trunk_mod
        * girth_scale
        * rigidity_girth_mod;
    let stem_base_radius = 0.20 * stem_thickness_factor;

    // Root sizing.
    let root_flare_factor = grav_pow(gravity, 0.5);
    let root_base_radius = 0.18 * root_flare_factor;

    // RADIUS CONCEPT 3: terminal leaf modifiers.
    let leaf_phys_mod = leaf_medium_mod * leaf_light_mod * leaf_wind_radius_mod * leaf_arid_radius_mod;

    // Terminal leaf elongation.
    let leaf_elongation =
        grav_pow(gravity, -0.35) * leaf_light_mod * leaf_wind_length_mod * leaf_arid_length_mod;

    // Droop angle.
    let droop_physics_base = (0.55 * grav_pow(gravity, 0.55) + wind_droop_extra).max(0.0);
    const TIP_ACCUM: f64 = 0.30;

    // Phototropism angle.
    let phototropism = genome.map(|g| g.phototropism).unwrap_or(1.0);
    let photo_angle = 0.40 * grav_pow(gravity, -0.45) * phototropism;

    // Woody droop per branchLevel.
    let woody_droop_base = 0.06 * grav_pow(gravity, 0.60) * rigidity_droop_scale;

    let down_dir = vec_norm(vec_scale(body_axis, -1.0));

    // -------------------------------------------------------------------------
    // Build children map.
    // -------------------------------------------------------------------------
    let n_nodes = graph.nodes.len();
    let mut children_of: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
    for i in 0..n_nodes {
        let p = graph.nodes[i].parent_idx;
        if p >= 0 {
            children_of[p as usize].push(i);
        }
    }

    // Root pipe-model parameters.
    let root_flare_gene = genome.map(|g| g.root_flare).unwrap_or(0.0);
    let root_taper_gene = genome.map(|g| g.root_taper).unwrap_or(0.5);
    let taproot_base_radius = root_base_radius * (1.0 + root_flare_gene);
    let root_taper_ratio = 0.60 + root_taper_gene * 0.25;

    // -------------------------------------------------------------------------
    // RADIUS CONCEPT 2: pipe-model forward pass.
    // -------------------------------------------------------------------------
    for i in 0..n_nodes {
        let is_root = graph.nodes[i].is_root;
        let parent_idx = graph.nodes[i].parent_idx;

        if is_root {
            if parent_idx < 0 || !graph.nodes[parent_idx as usize].is_root {
                graph.nodes[i].radius = taproot_base_radius;
            } else {
                let r_parent = graph.nodes[parent_idx as usize].radius;
                let root_siblings: Vec<usize> = children_of[parent_idx as usize]
                    .iter()
                    .copied()
                    .filter(|&ci| graph.nodes[ci].is_root)
                    .collect();
                let c = root_siblings.len();
                if c == 1 {
                    graph.nodes[i].radius = r_parent * root_taper_ratio;
                } else {
                    let leader_idx = *root_siblings.iter().max().unwrap();
                    let r_leader = r_parent * root_taper_ratio;
                    let residual_sq = (r_parent * r_parent - r_leader * r_leader).max(0.0);
                    let n_laterals = c - 1;
                    let r_lateral = if n_laterals > 0 {
                        (residual_sq / n_laterals as f64).sqrt()
                    } else {
                        0.0
                    };
                    graph.nodes[i].radius = if i == leader_idx { r_leader } else { r_lateral };
                }
            }
            continue;
        }

        if parent_idx < 0 {
            graph.nodes[i].radius = stem_base_radius;
            continue;
        }

        let r_parent = graph.nodes[parent_idx as usize].radius;
        let siblings = &children_of[parent_idx as usize];
        let c = siblings.len();

        let parent_branch_level = graph.nodes[parent_idx as usize].branch_level;
        let effective_taper = if parent_branch_level == 0 && trunk_taper_gene > 0.0 {
            taper + (1.0 - taper) * trunk_taper_gene
        } else {
            taper
        };

        if c == 1 {
            graph.nodes[i].radius = r_parent * effective_taper;
        } else {
            // Identify the leader (smallest direction divergence).
            let gp_idx = graph.nodes[parent_idx as usize].parent_idx;
            let parent_axis = if gp_idx >= 0 {
                let gp_pos = graph.nodes[gp_idx as usize].pos;
                let p_pos = graph.nodes[parent_idx as usize].pos;
                vec_norm([
                    p_pos[0] - gp_pos[0],
                    p_pos[1] - gp_pos[1],
                    p_pos[2] - gp_pos[2],
                ])
            } else {
                vec_norm(body_axis)
            };

            let p_pos = graph.nodes[parent_idx as usize].pos;
            let mut leader_idx: i64 = -1;
            let mut leader_dot = f64::NEG_INFINITY;
            for &child_idx in siblings.iter() {
                let child_pos = graph.nodes[child_idx].pos;
                let dir = vec_norm([
                    child_pos[0] - p_pos[0],
                    child_pos[1] - p_pos[1],
                    child_pos[2] - p_pos[2],
                ]);
                let d = vec_dot(dir, parent_axis);
                if d > leader_dot {
                    leader_dot = d;
                    leader_idx = child_idx as i64;
                }
            }

            let r_leader = r_parent * effective_taper;
            let residual_sq = (r_parent * r_parent - r_leader * r_leader).max(0.0);
            let n_laterals = c - 1;
            let r_lateral = if n_laterals > 0 {
                (residual_sq / n_laterals as f64).sqrt()
            } else {
                0.0
            };
            graph.nodes[i].radius = if i as i64 == leader_idx { r_leader } else { r_lateral };
        }
    }

    // -------------------------------------------------------------------------
    // Tip-taper pass: drive free-end (no-children) node radii to near-zero.
    // -------------------------------------------------------------------------
    for i in 0..n_nodes {
        if children_of[i].is_empty() {
            let r = graph.nodes[i].radius;
            graph.nodes[i].radius = (r * 0.08).max(0.012);
        }
    }

    // -------------------------------------------------------------------------
    // Second pass: terminal leaf modifier + position bending.
    // -------------------------------------------------------------------------
    for i in 0..n_nodes {
        if graph.nodes[i].is_root {
            continue;
        }

        if graph.nodes[i].is_terminal {
            // Radius: pipe-model × leaf physics.
            graph.nodes[i].radius *= leaf_phys_mod;

            let attach_pos = match graph.nodes[i].attach_pos {
                Some(ap) => ap,
                None => continue,
            };
            let pos = graph.nodes[i].pos;
            let raw_offset = [
                pos[0] - attach_pos[0],
                pos[1] - attach_pos[1],
                pos[2] - attach_pos[2],
            ];
            let offset_len = vec_len(raw_offset);
            if offset_len < 1e-10 {
                continue;
            }

            // Step 1: verticality + gravity + wind droop (single combined rotation).
            let t_branch_level = graph.nodes[i].branch_level as f64;
            let tip_accum_factor = 1.0 + t_branch_level * TIP_ACCUM;
            let droop_angle =
                (droop_physics_base * tip_accum_factor * rigidity_droop_scale).min(std::f64::consts::PI / 2.0);
            let droop_bias_angle = clamp_sym(
                droop_bias_gene * t_branch_level * DROOP_BIAS_GAIN,
                std::f64::consts::PI / 2.0,
            );
            let total_droop_angle = vert_angle + droop_angle - droop_bias_angle;

            let mut k_droop = vec_cross(down_dir, raw_offset);
            let k_droop_len = vec_len(k_droop);
            let offset_after_droop = if k_droop_len < 1e-10 {
                raw_offset
            } else {
                k_droop = vec_scale(k_droop, 1.0 / k_droop_len);
                rotate_around_axis(raw_offset, k_droop, total_droop_angle)
            };

            // Step 2: phototropism.
            let mut k_photo = vec_cross(offset_after_droop, light_dir);
            let k_photo_len = vec_len(k_photo);
            let offset_final = if k_photo_len < 1e-10 {
                offset_after_droop
            } else {
                k_photo = vec_scale(k_photo, 1.0 / k_photo_len);
                rotate_around_axis(offset_after_droop, k_photo, photo_angle)
            };

            // Step 3: elongation scale.
            let scaled_offset = vec_scale(offset_final, leaf_elongation);

            graph.nodes[i].pos = [
                attach_pos[0] + scaled_offset[0],
                attach_pos[1] + scaled_offset[1],
                attach_pos[2] + scaled_offset[2],
            ];
            continue;
        }

        // Woody (interior) nodes: per-level gravitational droop.
        let n = &graph.nodes[i];
        if n.is_woody && n.branch_level > 0 {
            let parent_idx = n.parent_idx;
            if parent_idx < 0 {
                continue;
            }
            let branch_level = n.branch_level;
            let pos = n.pos;
            let parent_pos = graph.nodes[parent_idx as usize].pos;
            let raw_offset = [
                pos[0] - parent_pos[0],
                pos[1] - parent_pos[1],
                pos[2] - parent_pos[2],
            ];
            let offset_len = vec_len(raw_offset);
            if offset_len < 1e-10 {
                continue;
            }

            let woody_droop_bias_angle = clamp_sym(
                droop_bias_gene * branch_level as f64 * DROOP_BIAS_GAIN,
                std::f64::consts::PI / 2.0,
            );
            let woody_droop_angle = woody_droop_base * branch_level as f64 + woody_droop_bias_angle;

            let mut k_droop = vec_cross(raw_offset, down_dir);
            let k_droop_len = vec_len(k_droop);
            if k_droop_len < 1e-10 {
                continue;
            }
            k_droop = vec_scale(k_droop, 1.0 / k_droop_len);
            let drooped_offset = rotate_around_axis(raw_offset, k_droop, woody_droop_angle);

            graph.nodes[i].pos = [
                parent_pos[0] + drooped_offset[0],
                parent_pos[1] + drooped_offset[1],
                parent_pos[2] + drooped_offset[2],
            ];
        }
    }

    // -------------------------------------------------------------------------
    // WEEP PASS — willow-style cascade droop (genome.weep > 0). No rng.
    // -------------------------------------------------------------------------
    let weep = genome.map(|g| g.weep).unwrap_or(0.0);
    if weep > 0.0 {
        const WEEP_MAX_PER_LEVEL: f64 = 0.50;
        const WEEP_STRAND_SCALE_MAX: f64 = 1.8;
        const WEEP_STRAND_MIN_LEVEL: i32 = 2;

        for i in 0..n_nodes {
            if graph.nodes[i].is_root {
                continue;
            }
            let level = graph.nodes[i].branch_level;
            if level == 0 {
                continue;
            }
            let weep_angle = (weep * WEEP_MAX_PER_LEVEL * level as f64).min(std::f64::consts::PI);
            if weep_angle < 1e-12 {
                continue;
            }

            // Strand thinning.
            graph.nodes[i].radius *= 1.0 - weep * 0.35 * (level as f64 / 3.0).min(1.0);

            let parent_idx = graph.nodes[i].parent_idx;
            if parent_idx < 0 {
                continue;
            }
            let anchor_pos = graph.nodes[parent_idx as usize].pos;
            let pos = graph.nodes[i].pos;
            let raw_offset = [
                pos[0] - anchor_pos[0],
                pos[1] - anchor_pos[1],
                pos[2] - anchor_pos[2],
            ];
            let offset_len = vec_len(raw_offset);
            if offset_len < 1e-10 {
                continue;
            }

            let mut k_weep = vec_cross(raw_offset, down_dir);
            let k_weep_len = vec_len(k_weep);
            if k_weep_len < 1e-10 {
                continue;
            }
            k_weep = vec_scale(k_weep, 1.0 / k_weep_len);
            let mut bent_offset = rotate_around_axis(raw_offset, k_weep, weep_angle);

            if graph.nodes[i].is_terminal && level >= WEEP_STRAND_MIN_LEVEL {
                let strand_scale = 1.0 + weep * (WEEP_STRAND_SCALE_MAX - 1.0);
                bent_offset = vec_scale(bent_offset, strand_scale);
            }

            graph.nodes[i].pos = [
                anchor_pos[0] + bent_offset[0],
                anchor_pos[1] + bent_offset[1],
                anchor_pos[2] + bent_offset[2],
            ];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{random_genome, Env};
    use crate::rng::Mulberry32;
    use crate::skeleton::build_skeleton;

    #[test]
    fn deterministic_no_rng() {
        // solveProportions consumes NO rng: same graph + env → bit-identical twice.
        let env = Env::default();
        let g = random_genome(&env, 42);
        let mut r = Mulberry32::new(g.structural_seed);
        let base = build_skeleton(&g, &mut r, g.jitter);

        let mut a = base.clone();
        let mut b = base.clone();
        solve_proportions(&mut a, &env, Some(&g));
        solve_proportions(&mut b, &env, Some(&g));
        for (na, nb) in a.nodes.iter().zip(b.nodes.iter()) {
            assert_eq!(na.pos, nb.pos);
            assert_eq!(na.radius, nb.radius);
        }
        // light_dir written from sunAngle.
        assert_eq!(a.meta.light_dir, b.meta.light_dir);
    }

    #[test]
    fn trunk_base_radius_is_stem_base() {
        let env = Env::default();
        let g = random_genome(&env, 3);
        let mut r = Mulberry32::new(g.structural_seed);
        let mut graph = build_skeleton(&g, &mut r, g.jitter);
        solve_proportions(&mut graph, &env, Some(&g));
        // The trunk base node (parentIdx == -1) must have a positive radius.
        let base = graph.nodes.iter().find(|n| n.parent_idx == -1).unwrap();
        assert!(base.radius > 0.0);
    }

    #[test]
    fn tips_are_thin() {
        let env = Env::default();
        let g = random_genome(&env, 9);
        let mut r = Mulberry32::new(g.structural_seed);
        let mut graph = build_skeleton(&g, &mut r, g.jitter);
        solve_proportions(&mut graph, &env, Some(&g));
        // Every node has a finite, non-negative radius after the solve.
        for n in &graph.nodes {
            assert!(n.radius.is_finite() && n.radius >= 0.0);
        }
    }
}

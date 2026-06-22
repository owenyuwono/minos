//! Deterministic surface scatter for procedural trees (Phase B, `--features flora`).
//!
//! World-deterministic: a tree's existence / position / yaw / scale is a pure hash
//! of its lat-lon grid cell, so walking around never reshuffles the grove — only
//! the visible window (a radius around the player) slides. No LOD, no instancing:
//! the caller draws ONE shared species mesh once per returned instance via
//! `FloraView::record_at`.
//!
//! ponytail: lat-lon cells distort near the poles (fixed angular `dlon` → cells
//! bunch up there) and there is NO hardware instancing (draw cost is ~N×2). Both
//! are fine for a walk-among-trees grove on the mid-latitude surface; revisit with
//! a cube-sphere cell id + instanced draw if trees ever need planet-wide coverage.

use enki_planet::height::HeightField;
use glam::{DVec3, Mat4, Vec3, Vec4};

use crate::controls::terrain_grid::ground_radius;
use crate::controls::PLANET_RADIUS;

/// One placed tree: where to stand it + how to spin/scale it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreeInstance {
    /// f64 world position on the terrain surface (rides `surface_radius`).
    pub origin: DVec3,
    /// Rotation about the radial up (radians).
    pub yaw: f32,
    /// Uniform size multiplier.
    pub scale: f32,
}

/// Draw trees within this distance of the player (m).
pub const RADIUS_M: f64 = 120.0;
/// Mean spacing between candidate cells (m) — the lat-lon cell edge.
pub const SPACING_M: f64 = 8.0;
/// Hard cap on drawn instances (the no-instancing draw-call ceiling). Density
/// above this is clamped to the nearest `MAX_TREES`.
pub const MAX_TREES: usize = 120;

/// splitmix64-style finalizer over a mixed (cell_i, cell_j, salt) key.
fn hash(cell_i: i64, cell_j: i64, salt: u64) -> u64 {
    let mut x = (cell_i as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (cell_j as u64).wrapping_mul(0xC2B2AE3D27D4EB4F)
        ^ salt.wrapping_mul(0x165667B19E3779F9);
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D049BB133111EB);
    x ^= x >> 31;
    x
}

/// hash → f32 in [0, 1).
fn unit(h: u64) -> f32 {
    ((h >> 40) as f32) / ((1u64 << 24) as f32)
}

/// Unit direction for a lat/lon (internal, self-consistent convention).
fn dir_of(lat: f64, lon: f64) -> DVec3 {
    let (sl, cl) = lat.sin_cos();
    DVec3::new(cl * lon.cos(), sl, cl * lon.sin())
}

/// All trees within `RADIUS_M` of `center` (the player's ground position), hashed
/// deterministically per lat-lon cell. `density` ∈ [0,1] is the fraction of cells
/// that get a tree. Each origin rides the **rendered** (grid-faceted) terrain via
/// [`ground_radius`] — the SAME single-source grounding the character's feet use —
/// so trees sit ON the drawn ground instead of diving into the analytic curve's
/// sub-cell detail the mesh never draws.
pub fn scatter(
    hf: &dyn HeightField,
    height_scale: f64,
    center: DVec3,
    density: f32,
) -> Vec<TreeInstance> {
    let mut out = Vec::new();
    if density <= 0.0 || center.length_squared() == 0.0 {
        return out;
    }
    let da = SPACING_M / PLANET_RADIUS; // angular cell size (radians)
    let c = center.normalize();
    let lat0 = c.y.clamp(-1.0, 1.0).asin();
    let lon0 = c.z.atan2(c.x);
    let li0 = (lat0 / da).round() as i64;
    let lj0 = (lon0 / da).round() as i64;
    let w = (RADIUS_M / SPACING_M).ceil() as i64 + 1;
    let half = std::f64::consts::FRAC_PI_2;

    for li in (li0 - w)..=(li0 + w) {
        let lat_c = li as f64 * da;
        if lat_c <= -half || lat_c >= half {
            continue; // skip the poles (lat-lon degenerates there)
        }
        for lj in (lj0 - w)..=(lj0 + w) {
            if unit(hash(li, lj, 1)) >= density {
                continue;
            }
            // Jitter within the cell (±0.4 cell) so the grid doesn't read as rows.
            let dlat = (unit(hash(li, lj, 2)) - 0.5) as f64 * 0.8 * da;
            let dlon = (unit(hash(li, lj, 3)) - 0.5) as f64 * 0.8 * da;
            let dir = dir_of(lat_c + dlat, lj as f64 * da + dlon);
            let r = ground_radius(hf, height_scale, dir);
            let origin = dir * r;
            if (origin - center).length() > RADIUS_M {
                continue;
            }
            out.push(TreeInstance {
                origin,
                yaw: unit(hash(li, lj, 4)) * std::f32::consts::TAU,
                scale: 0.85 + unit(hash(li, lj, 5)) * 0.4, // 0.85..1.25
            });
        }
    }
    // Cap: keep the nearest MAX_TREES (bounds the per-frame draw calls).
    if out.len() > MAX_TREES {
        out.sort_by(|a, b| {
            let da = (a.origin - center).length_squared();
            let db = (b.origin - center).length_squared();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        out.truncate(MAX_TREES);
    }
    out
}

/// Camera-relative model matrix for one scattered instance: orient tree-local +Y
/// to the radial surface normal at `origin`, spin by `yaw` about up, uniform
/// `scale`, and translate by `(origin - camera)` in f64 (cast to f32 last, so
/// planet-scale coords don't lose precision). Mirrors `FloraView::model` + a yaw.
/// The 3×3 stays a similarity (det > 0) so cull-BACK winding is preserved.
pub fn instance_model(origin: DVec3, yaw: f32, scale: f32, camera_world_pos: DVec3) -> Mat4 {
    let up = origin.normalize_or(DVec3::Y).as_vec3();
    let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
    let right = reference.cross(up).normalize();
    let forward = right.cross(up).normalize();
    // Yaw about up, in the tangent plane (preserves orthonormality + handedness).
    let (s, c) = yaw.sin_cos();
    let right_y = right * c + forward * s;
    let forward_y = forward * c - right * s;
    let mut m = Mat4::from_cols(
        (right_y * scale).extend(0.0),
        (up * scale).extend(0.0),
        (forward_y * scale).extend(0.0),
        Vec4::W,
    );
    let rel = (origin - camera_world_pos).as_vec3();
    m.w_axis.x = rel.x;
    m.w_axis.y = rel.y;
    m.w_axis.z = rel.z;
    m.w_axis.w = 1.0;
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FlatHf;
    impl HeightField for FlatHf {
        fn height(&self, _dir: DVec3, _level: u8) -> f64 {
            100.0
        }
    }

    #[test]
    fn scatter_is_deterministic_grounded_and_bounded() {
        let hf = FlatHf;
        let hs = 1.0;
        let center_dir = DVec3::new(0.3, 0.55, 0.2).normalize();
        let center = center_dir * ground_radius(&hf, hs, center_dir);

        let a = scatter(&hf, hs, center, 0.6);
        let b = scatter(&hf, hs, center, 0.6);
        assert!(!a.is_empty(), "expected some trees at density 0.6");
        assert_eq!(a, b, "same inputs must yield byte-identical scatter");
        assert!(a.len() <= MAX_TREES, "must respect the draw-call cap");

        for t in &a {
            // Rides the single-source ground: radius == ground_radius at dir.
            let dir = t.origin.normalize();
            let r = ground_radius(&hf, hs, dir);
            assert!((t.origin.length() - r).abs() < 1e-6, "tree not grounded on the mesh");
            assert!(
                (t.origin - center).length() <= RADIUS_M + 1e-6,
                "tree outside the scatter radius"
            );
            assert!((0.85..=1.25).contains(&t.scale), "scale out of range");
        }
    }

    #[test]
    fn density_zero_is_empty() {
        let hf = FlatHf;
        let center = DVec3::new(0.3, 0.55, 0.2).normalize() * (PLANET_RADIUS + 100.0);
        assert!(scatter(&hf, 1.0, center, 0.0).is_empty());
    }
}

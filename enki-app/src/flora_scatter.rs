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

use std::sync::OnceLock;

use enki_planet::height::HeightField;
use enki_planet::noise::{fbm, Noise3D};
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

/// Draw trees within this distance of the player (m). Sized so faraway trees
/// stay visible instead of popping out a few steps ahead — the count this
/// implies at the default density is the real cost (see `MAX_TREES`).
pub const RADIUS_M: f64 = 300.0;
/// Mean spacing between candidate cells (m) — the lat-lon cell edge. Smaller =
/// denser grove.
pub const SPACING_M: f64 = 5.0;
/// Safety ceiling on drawn instances — a hang-guard, NOT a routine LOD cull. At
/// the default `RADIUS_M`×density every tree in the window renders (no faraway
/// trees hidden); this only trips on a pathological density×radius, and when it
/// does we drop the FARTHEST and warn (so a cull is never silent).
/// ponytail: trees are plain per-instance draws (NOT Nanite-virtualized), so the
/// per-frame draw cost scales linearly with the count — this is the upper bound
/// until trees go through a virtualized/instanced path; then the radius can grow
/// without an fps cliff and this ceiling can lift.
pub const MAX_TREES: usize = 16000;

// ── Forest groves (approach B) ──────────────────────────────────────────────
// Climate decides WHERE forests can exist; a low-frequency noise breaks that
// region into groves with bald clearings, so trees don't carpet the whole biome.
// ponytail: tunables held as constants — eyeball them on the planet and nudge;
// promote to GUI sliders if they need live tuning.

/// Grove feature frequency on the unit sphere — bigger = smaller groves
/// (≈ `PLANET_RADIUS / GROVE_FREQ` metres per blob, so ~600 m here).
const GROVE_FREQ: f64 = 80.0;
/// Grove-mask cut on the fbm output (≈[-1, 1]); higher = less forest cover.
const GROVE_THRESH: f32 = 0.05;
/// Soft edge half-width around the cut (grove → clearing falloff).
const GROVE_EDGE: f32 = 0.20;

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

/// Scalar Hermite smoothstep on `[e0, e1]`.
#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Fixed-seed noise for the grove mask (built once, lazily).
fn grove_noise() -> &'static Noise3D {
    static N: OnceLock<Noise3D> = OnceLock::new();
    N.get_or_init(|| Noise3D::new(0xF0_4E_57)) // "FOrEST"
}

/// Forest suitability at unit `dir` ∈ [0, 1] (approach B). Zero off forest
/// ground; inside it a low-freq grove mask carves blobs + clearings so trees
/// don't carpet the whole biome. The climate gate mirrors
/// [`enki_planet::coloring::biome_color`]'s land rules on the SAME inputs
/// (`height(dir, 0)` for elevation + `climate(dir, h)` for temp/precip), so trees
/// land on the green, tree-bearing terrain — not desert / tundra / rock / ocean.
fn forest(hf: &dyn HeightField, dir: DVec3) -> f32 {
    // Elevation in biome units (level 0 = the rendered grid; see terrain_grid).
    let h = hf.height(dir, 0);
    const BEACH: f64 = 0.02; // above the waterline/beach band (biome_color e<0.012)
    if h <= BEACH {
        return 0.0; // ocean / shore
    }
    if hf.wetness(dir) > 0.5 || hf.lake_mask_at(dir) > 0.5 {
        return 0.0; // standing in a river / lake
    }
    let (temp_c, precip) = hf.climate(dir, h);
    // Vegetated-forest band: warm enough (tundra→forest by ~8 °C), wet enough
    // (desert/steppe→forest past ~0.45 precip), below the rock treeline (e~0.5).
    let warm = smoothstep(2.0, 8.0, temp_c);
    let wet = smoothstep(0.30, 0.45, precip);
    let below_treeline = 1.0 - smoothstep(0.40, 0.55, h as f32);
    let gate = warm * wet * below_treeline;
    if gate <= 0.0 {
        return 0.0;
    }
    // Grove mask → forest blobs with bald clearings between.
    let p = dir * GROVE_FREQ;
    let n = fbm(grove_noise(), p.x, p.y, p.z, 4, 1.0, 2.0, 0.5) as f32;
    let grove = smoothstep(GROVE_THRESH - GROVE_EDGE, GROVE_THRESH + GROVE_EDGE, n);
    gate * grove
}

/// All trees within `RADIUS_M` of `center` (the player's ground position), hashed
/// deterministically per lat-lon cell. `density` ∈ [0,1] is the *peak* per-cell
/// chance; the actual chance is `density * forest(dir)`, so trees only appear on
/// forest-suitable ground and clump into groves (see [`forest`]) instead of
/// carpeting the whole surface. Each origin rides the **rendered** (grid-faceted)
/// terrain via [`ground_radius`] — the SAME single-source grounding the
/// character's feet use — so trees sit ON the drawn ground instead of diving into
/// the analytic curve's sub-cell detail the mesh never draws.
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
            // Cheap reject first: `forest` only ever LOWERS the probability below
            // `density`, so a cell that fails the bare density draw can't survive —
            // skip the pricier dir + climate + grove eval for it.
            let u = unit(hash(li, lj, 1));
            if u >= density {
                continue;
            }
            // Jitter within the cell (±0.4 cell) so the grid doesn't read as rows.
            let dlat = (unit(hash(li, lj, 2)) - 0.5) as f64 * 0.8 * da;
            let dlon = (unit(hash(li, lj, 3)) - 0.5) as f64 * 0.8 * da;
            let dir = dir_of(lat_c + dlat, lj as f64 * da + dlon);
            // Forest gate (approach B): only forest ground, clumped into groves.
            if u >= density * forest(hf, dir) {
                continue;
            }
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
    // Safety ceiling: only trips on a pathological density×radius. Keep the
    // nearest MAX_TREES and warn — dropped faraway trees are never silent.
    if out.len() > MAX_TREES {
        log::warn!(
            "flora_scatter: {} trees in range exceeds MAX_TREES={}; dropping the farthest {}",
            out.len(),
            MAX_TREES,
            out.len() - MAX_TREES,
        );
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

    // Low forest land everywhere: h in (BEACH, treeline) + default climate
    // (15 °C, 0.5 precip) → temperate forest, so only the grove mask gates trees.
    struct FlatHf;
    impl HeightField for FlatHf {
        fn height(&self, _dir: DVec3, _level: u8) -> f64 {
            0.1
        }
    }

    /// Constant-climate region for exercising the `forest` biome gate.
    struct Region {
        h: f64,
        temp: f32,
        precip: f32,
    }
    impl HeightField for Region {
        fn height(&self, _dir: DVec3, _level: u8) -> f64 {
            self.h
        }
        fn climate(&self, _dir: DVec3, _height: f64) -> (f32, f32) {
            (self.temp, self.precip)
        }
    }

    #[test]
    fn forest_only_on_wet_temperate_land() {
        let dir = DVec3::new(0.3, 0.55, 0.2).normalize();
        // Hard-rejected biomes → exactly zero suitability.
        assert_eq!(forest(&Region { h: -0.1, temp: 15.0, precip: 0.6 }, dir), 0.0, "ocean");
        assert_eq!(forest(&Region { h: 0.1, temp: 32.0, precip: 0.05 }, dir), 0.0, "desert (dry)");
        assert_eq!(forest(&Region { h: 0.1, temp: -10.0, precip: 0.6 }, dir), 0.0, "frozen (cold)");
        assert_eq!(forest(&Region { h: 0.85, temp: 15.0, precip: 0.6 }, dir), 0.0, "above treeline");
        // Wet temperate land: climate gate is open, so SOME direction lands in a
        // grove (the mask varies over the sphere) — i.e. forests exist, but not
        // everywhere (clearings too).
        let land = Region { h: 0.1, temp: 15.0, precip: 0.6 };
        let mut in_grove = 0;
        let mut total = 0;
        for i in 0..4000 {
            let a = i as f64 * 0.013;
            let d = DVec3::new(a.cos(), (a * 0.7).sin() * 0.5, a.sin()).normalize();
            total += 1;
            if forest(&land, d) > 0.0 {
                in_grove += 1;
            }
        }
        assert!(in_grove > 0, "expected forest groves on wet temperate land");
        assert!(in_grove < total, "expected clearings too (not a full carpet)");
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

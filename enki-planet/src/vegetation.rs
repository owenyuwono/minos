//! Forest-suitability field — the single source of truth shared by tree scatter
//! (`enki-app::flora_scatter`) and the baked terrain canopy tint
//! (`enki-nanite::bake::tessellate`).
//!
//! [`vegetation_density`] returns `[0, 1]` forest suitability at a unit `dir`:
//! zero off forest ground (ocean / shore / standing water / desert / tundra /
//! rock / above the treeline), and inside the forest band a low-frequency grove
//! mask carves blobs + bald clearings so trees don't carpet the whole biome.
//!
//! Because BOTH the scatter and the tint read this one fn (same seed, frequency,
//! and constants), the green canopy patches baked into the terrain color line up
//! EXACTLY with where trees actually scatter — they can never desync.

use glam::DVec3;

use crate::height::HeightField;
use crate::noise::{fbm, Noise3D};

// ── Forest groves (approach B) ──────────────────────────────────────────────
// Climate decides WHERE forests can exist; a low-frequency noise breaks that
// region into groves with bald clearings. This module is the SINGLE source of
// truth for these constants — the tree scatter (`enki-app::flora_scatter`) and
// the baked canopy tint (`enki-nanite::bake::tessellate`) both read this fn, so
// the tint and the scatter line up to the texel by construction.

/// Above the waterline/beach band (`biome_color` e<0.012).
const BEACH: f64 = 0.02;
/// Grove feature frequency on the unit sphere — bigger = smaller groves.
const GROVE_FREQ: f64 = 80.0;
/// Grove-mask cut on the fbm output (≈[-1, 1]); higher = less forest cover.
const GROVE_THRESH: f32 = 0.05;
/// Soft edge half-width around the cut (grove → clearing falloff).
const GROVE_EDGE: f32 = 0.20;
/// Fixed grove-mask noise seed — "FOrEST".
const GROVE_SEED: u32 = 0xF0_4E_57;

/// Scalar Hermite smoothstep on `[e0, e1]`.
#[inline]
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Fixed-seed noise for the grove mask (built once, lazily).
fn grove_noise() -> &'static Noise3D {
    use std::sync::OnceLock;
    static N: OnceLock<Noise3D> = OnceLock::new();
    N.get_or_init(|| Noise3D::new(GROVE_SEED))
}

/// Forest suitability at unit `dir` ∈ [0, 1]. Zero off forest ground; inside it
/// a low-freq grove mask carves blobs + clearings so trees don't carpet the
/// whole biome. The climate gate mirrors [`crate::coloring::biome_color`]'s land
/// rules on the SAME inputs (`height(dir, 0)` for elevation + `climate(dir, h)`
/// for temp/precip), so the result lands on the green, tree-bearing terrain —
/// not desert / tundra / rock / ocean.
///
/// This is the single source of truth for "where do forests grow": the planet's
/// tree scatter and the terrain canopy tint both read it (same noise, same
/// constants → they line up exactly).
pub fn vegetation_density(hf: &dyn HeightField, dir: DVec3) -> f32 {
    // Elevation in biome units (level 0 = the rendered grid; see terrain_grid).
    let h = hf.height(dir, 0);
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Constant-climate region for exercising the climate gate.
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
    fn zero_off_forest_ground() {
        let dir = DVec3::new(0.3, 0.55, 0.2).normalize();
        assert_eq!(vegetation_density(&Region { h: -0.1, temp: 15.0, precip: 0.6 }, dir), 0.0, "ocean");
        assert_eq!(vegetation_density(&Region { h: 0.1, temp: 32.0, precip: 0.05 }, dir), 0.0, "desert");
        assert_eq!(vegetation_density(&Region { h: 0.1, temp: -10.0, precip: 0.6 }, dir), 0.0, "frozen");
        assert_eq!(vegetation_density(&Region { h: 0.85, temp: 15.0, precip: 0.6 }, dir), 0.0, "above treeline");
    }

    #[test]
    fn groves_and_clearings_on_wet_temperate_land() {
        let land = Region { h: 0.1, temp: 15.0, precip: 0.6 };
        let mut in_grove = 0;
        let mut total = 0;
        for i in 0..4000 {
            let a = i as f64 * 0.013;
            let d = DVec3::new(a.cos(), (a * 0.7).sin() * 0.5, a.sin()).normalize();
            total += 1;
            if vegetation_density(&land, d) > 0.0 {
                in_grove += 1;
            }
        }
        assert!(in_grove > 0, "expected forest groves on wet temperate land");
        assert!(in_grove < total, "expected clearings too (not a full carpet)");
    }

    #[test]
    fn output_in_unit_range() {
        let land = Region { h: 0.1, temp: 15.0, precip: 0.6 };
        for i in 0..2000 {
            let a = i as f64 * 0.017;
            let d = DVec3::new(a.cos(), (a * 0.3).sin() * 0.7, a.sin()).normalize();
            let v = vegetation_density(&land, d);
            assert!((0.0..=1.0).contains(&v), "vegetation_density out of range: {v}");
        }
    }
}

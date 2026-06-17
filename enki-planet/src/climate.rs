//! Stage B — climate and biome simulation.
//!
//! Pure port of `demiurge/src/planet/climate.ts`. All arithmetic is f64,
//! all randomness derives from splitmix32/deriveSeed — no system entropy.
//!
//! Temperature is ANALYTIC (cheap, per-call, no bake):
//!   latitude profile (warm equator → cold pole) scaled by atmosphere,
//!   minus altitude lapse. Axial tilt > 54° inverts toward pole-favouring.
//!
//! Moisture is BAKED once into a 128²×6 cube-map at construction:
//!   bandWetness · seaProximity · rainShadow · moistGain, then box-blurred.
//!   Sampled bilinearly via `cubemap::sample_smooth`.
//!
//! `biome_color` is NOT here — it belongs to the mesher (task 4.2b).

use glam::DVec3;
use crate::cubemap::{
    texel_to_dir, texel_index, neighbor_texel, sample_smooth,
};

// ---------------------------------------------------------------------------
// Tunable constants — generic, not Earth-hardcoded
// ---------------------------------------------------------------------------

/// Cube-map resolution for the baked moisture field. 128 → 6·128² = 98304 texels.
const MOIST_RES: usize = 128;

/// Equator→pole temperature drop, °C, at thin atmosphere (atmosphere→0).
const EQUATOR_POLE_DELTA: f64 = 55.0;
/// Centres the latitude profile so baseTemp ≈ area-mean (cosLat area-mean over sphere = 0.5).
const LAT_MEAN: f64 = 0.5;
/// Default lapse rate: °C lost over full normalised height, at full atmosphere.
const LAPSE_PER_HEIGHT: f64 = 50.0;

/// Axial tilt (rad) where insolation inversion begins (≈54°) and completes (≈75°).
const TILT_INVERT_LO: f64 = 54.0 * std::f64::consts::PI / 180.0;
const TILT_INVERT_HI: f64 = 75.0 * std::f64::consts::PI / 180.0;

/// Inland moisture falloff scale (radians of crustDist).
const MOIST_INLAND_SCALE: f64 = 0.7;
/// Floor on sea-proximity so deep interiors still get a little moisture.
const MOIST_SEAPROX_FLOOR: f64 = 0.21;
/// Floor on raw band wetness — descending bands aren't bone-dry.
const BAND_FLOOR: f64 = 0.075;
/// Rain-shadow strength: leeward drying per unit upwind height excess.
const SHADOW_K: f64 = 1.6;
/// Floor for the rain-shadow multiplier.
const MIN_SHADOW: f64 = 0.25;
/// Upwind march: steps and per-step angular distance (radians).
const UPWIND_STEPS: usize = 3;
const UPWIND_STEP_RAD: f64 = 0.02;
/// Coarse LOD level at which heightFn is sampled for rain shadow.
const RAINSHADOW_LEVEL: u32 = 8;
/// Overall moisture gain before clamp.
const MOIST_GAIN: f64 = 1.23;
/// Cold places hold less liquid moisture: below this temp (°C) moisture is attenuated.
const FROZEN_TEMP: f64 = -5.0;
/// Width (°C) over which frozen attenuation ramps in below FROZEN_TEMP.
const FROZEN_WIDTH: f64 = 12.0;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Input parameters for `Climate::new`.
pub struct ClimateParams {
    /// Master seed. Stochastic offsets derive from this via deriveSeed.
    pub seed: u32,
    /// Mean surface temperature, °C-ish (Earth ≈ 15).
    pub base_temp: f64,
    /// 0..1: thick (→1) gives uniform/small gradients, thin (→0) gives extremes.
    pub atmosphere: f64,
    /// Circulation cells per hemisphere (1 slow … ~7 fast).
    pub band_count: u32,
    /// Axial tilt in radians — drives the >54° insolation inversion.
    pub axial_tilt_rad: f64,
    /// Heat redistribution R ∈ [0,1]. Default 1 (fully climatological).
    pub redistribution: Option<f64>,
    /// Greenhouse offset in °C added to baseTemp. Default 0.
    pub greenhouse: Option<f64>,
    /// Lapse rate: °C lost over full normalised height. Default 50.
    pub lapse_rate: Option<f64>,
}

/// Output from `Climate::sample`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClimateSample {
    /// °C-ish.
    pub temperature: f32,
    /// 0..1.
    pub moisture: f32,
}

/// Serialisable snapshot for zero-copy worker sharing.
/// Transfer `moisture_field` as raw bytes across worker boundaries.
#[derive(Debug, Clone)]
pub struct ClimateBaked {
    /// Resolution used during the bake (= MOIST_RES = 128).
    pub moist_res: usize,
    /// Baked cube-map moisture field, length `6 * moist_res²`.
    pub moisture_field: Vec<f32>,
    /// Scalar params needed to reconstruct the sampler.
    pub base_temp: f64,
    pub atmosphere: f64,
    pub band_count: u32,
    pub axial_tilt_rad: f64,
    pub redistribution: f64,
    pub greenhouse: f64,
    pub lapse_rate: f64,
}

// ---------------------------------------------------------------------------
// Local PRNG helpers — mirrors climate.ts (kept local, not exported)
// ---------------------------------------------------------------------------

/// Single splitmix32 step. Mirrors `splitmix32Step` in climate.ts.
/// Uses the SAME hash as `noise.rs::splitmix32` but only one step here.
fn splitmix32_step(a: u32) -> u32 {
    let a = a.wrapping_add(0x9e37_79b9);
    let mut t = a ^ (a >> 16);
    t = t.wrapping_mul(0x21f0_aaad);
    t = t ^ (t >> 15);
    t = t.wrapping_mul(0x735a_2d97);
    t ^ (t >> 15)
}

/// Derive a child seed from master + stream id.
/// Climate uses stream id 200 (moisture-field jitter) per climate.ts.
fn derive_seed(master_seed: u32, stream: u32) -> u32 {
    let s = master_seed ^ (stream.wrapping_add(1).wrapping_mul(0xdead_beef));
    splitmix32_step(s)
}

// ---------------------------------------------------------------------------
// Small math helpers
// ---------------------------------------------------------------------------

#[inline(always)]
fn clamp01(x: f64) -> f64 {
    x.clamp(0.0, 1.0)
}

#[inline(always)]
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = clamp01((x - e0) / (e1 - e0));
    t * t * (3.0 - 2.0 * t)
}

// ---------------------------------------------------------------------------
// Climate struct
// ---------------------------------------------------------------------------

/// Baked climate field: analytic temperature + bilinearly-sampled moisture.
#[allow(dead_code)]
pub struct Climate {
    base_temp: f64,
    greenhouse: f64,
    lapse_rate: f64,
    /// Tilt-inversion blend (0 = normal gradient, 1 = fully pole-favouring).
    invert_blend: f64,
    /// Equator→pole temperature gradient after atmosphere shrink.
    gradient: f64,
    /// Atmosphere factor for the altitude lapse.
    lapse_factor: f64,
    /// Overall moisture scale — thin atmosphere → drier.
    moist_gain: f64,
    /// Band-contrast exponent — thin atmosphere → sharper dry bands.
    band_contrast: f64,
    /// Per-texel band count (integer, ≥1).
    band_count: u32,
    /// Baked moisture field — length `6 * MOIST_RES²`.
    moisture_field: Vec<f32>,
    // Stored for `to_baked` only:
    atmosphere: f64,
    axial_tilt_rad: f64,
    redistribution: f64,
}

impl Climate {
    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    /// Build a new Climate, baking the moisture cube-map.
    ///
    /// `height_fn(dir, level) -> f64` returns normalised terrain height in
    /// `[-1, 1]` at a coarse LOD level (same contract as tectonics heightFn).
    /// `crust_dist_at(dir) -> f64` returns signed distance to crust edge
    /// in radians (positive = inland). From tectonics Phase 2.
    pub fn new(
        params: ClimateParams,
        height_fn: impl Fn(DVec3, u32) -> f64,
        crust_dist_at: impl Fn(DVec3) -> f64,
    ) -> Self {
        let atmosphere = clamp01(params.atmosphere);
        let band_count = params.band_count.max(1);
        let axial_tilt_rad = params.axial_tilt_rad;
        let redistribution = clamp01(params.redistribution.unwrap_or(1.0));
        let greenhouse = params.greenhouse.unwrap_or(0.0);
        let lapse_rate = params.lapse_rate.unwrap_or(LAPSE_PER_HEIGHT);

        let invert_blend = smoothstep(TILT_INVERT_LO, TILT_INVERT_HI, axial_tilt_rad);
        let gradient = EQUATOR_POLE_DELTA * (1.0 - 0.7 * atmosphere);
        let lapse_factor = 0.3 + 0.7 * atmosphere;
        let moist_gain = MOIST_GAIN * (0.34 + 1.1 * atmosphere);
        let band_contrast = 1.30 - 0.55 * atmosphere;

        // Touch seed stream 200 for determinism (mirrors climate.ts comment).
        let _jitter_seed = derive_seed(params.seed, 200);

        let moisture_field = bake_moisture(
            MOIST_RES,
            band_count,
            band_contrast,
            moist_gain,
            params.base_temp,
            greenhouse,
            gradient,
            lapse_factor,
            lapse_rate,
            invert_blend,
            &height_fn,
            &crust_dist_at,
        );

        Climate {
            base_temp: params.base_temp,
            greenhouse,
            lapse_rate,
            invert_blend,
            gradient,
            lapse_factor,
            moist_gain,
            band_contrast,
            band_count,
            moisture_field,
            atmosphere,
            axial_tilt_rad,
            redistribution,
        }
    }

    // -----------------------------------------------------------------------
    // Sampling
    // -----------------------------------------------------------------------

    /// Sample temperature (analytic) + moisture (baked field bilinear lookup).
    ///
    /// `dir` must be a unit vector; `height` is normalised terrain height in
    /// `[-1, 1]` (positive = above sea level).
    ///
    /// Returns `(temp_c, moisture)`: temp in °C, moisture clamped to `[0, 1]`.
    pub fn sample(&self, dir: DVec3, height: f64) -> (f32, f32) {
        let temp = self.temperature_at(dir, height);
        let m_raw = sample_smooth(&self.moisture_field, dir, MOIST_RES);
        let moisture = m_raw.clamp(0.0, 1.0);
        (temp as f32, moisture as f32)
    }

    // -----------------------------------------------------------------------
    // Serialisation
    // -----------------------------------------------------------------------

    /// Snapshot this Climate for zero-copy worker sharing.
    /// The caller owns the returned `ClimateBaked` (moisture_field is cloned).
    pub fn to_baked(&self) -> ClimateBaked {
        ClimateBaked {
            moist_res: MOIST_RES,
            moisture_field: self.moisture_field.clone(),
            base_temp: self.base_temp,
            atmosphere: self.atmosphere,
            band_count: self.band_count,
            axial_tilt_rad: self.axial_tilt_rad,
            redistribution: self.redistribution,
            greenhouse: self.greenhouse,
            lapse_rate: self.lapse_rate,
        }
    }

    /// Reconstruct a sampler-only Climate from a baked snapshot.
    /// Does NOT call `bake_moisture` — uses the field from `b` directly.
    pub fn from_baked(b: ClimateBaked) -> Self {
        let atmosphere = b.atmosphere;
        let axial_tilt_rad = b.axial_tilt_rad;
        let invert_blend = smoothstep(TILT_INVERT_LO, TILT_INVERT_HI, axial_tilt_rad);
        let gradient = EQUATOR_POLE_DELTA * (1.0 - 0.7 * atmosphere);
        let lapse_factor = 0.3 + 0.7 * atmosphere;
        let moist_gain = MOIST_GAIN * (0.34 + 1.1 * atmosphere);
        let band_contrast = 1.30 - 0.55 * atmosphere;

        Climate {
            base_temp: b.base_temp,
            greenhouse: b.greenhouse,
            lapse_rate: b.lapse_rate,
            invert_blend,
            gradient,
            lapse_factor,
            moist_gain,
            band_contrast,
            band_count: b.band_count,
            moisture_field: b.moisture_field,
            atmosphere,
            axial_tilt_rad,
            redistribution: b.redistribution,
        }
    }

    // -----------------------------------------------------------------------
    // Temperature helpers (private)
    // -----------------------------------------------------------------------

    /// Climatological (latitude-average) insolation in [0,1].
    /// Normal tilt: cosLat (warm equator). Extreme tilt: warm pole (|dir.y|).
    #[inline]
    fn climatological_insolation(&self, abs_y: f64, cos_lat: f64) -> f64 {
        cos_lat * (1.0 - self.invert_blend) + abs_y * self.invert_blend
    }

    /// Climatological temperature at `dir` and normalised height.
    #[inline]
    fn temperature_at(&self, dir: DVec3, height: f64) -> f64 {
        let abs_y = dir.y.abs();
        let cos_lat = (1.0 - dir.y * dir.y).max(0.0).sqrt();
        let lat = self.climatological_insolation(abs_y, cos_lat);
        let lapse = self.lapse_rate * height.max(0.0) * self.lapse_factor;
        self.base_temp + self.greenhouse + self.gradient * (lat - LAT_MEAN) - lapse
    }
}

// ---------------------------------------------------------------------------
// Moisture bake — free function (keeps Climate::new readable)
// ---------------------------------------------------------------------------

/// Band circulation wetness in [0,1].
/// `cos(2·band_count·latAngle)`: equator → 1 (wet), first dry trough at
/// `latAngle = π/(2·band_count)`.
fn band_wetness(dir_y: f64, band_count: u32) -> f64 {
    let lat_angle = dir_y.clamp(-1.0, 1.0).asin();
    let raw = 0.5 + 0.5 * (2.0 * band_count as f64 * lat_angle).cos();
    BAND_FLOOR + (1.0 - BAND_FLOOR) * raw
}

/// Index of the circulation band a latitude falls in (0 at equator, increasing poleward).
fn band_index(dir_y: f64, band_count: u32) -> u32 {
    let lat_angle = dir_y.clamp(-1.0, 1.0).asin().abs();
    (lat_angle / (std::f64::consts::PI / (2.0 * band_count as f64))).floor() as u32
}

/// Bake the moisture cube-map. For each texel:
///   `moisture = clamp01(bandWetness · seaProximity · rainShadow · moistGain) · frozenAtten`
/// then one cross-face box blur (centre weight 4, four cardinal neighbours weight 1).
#[allow(clippy::too_many_arguments)]
fn bake_moisture(
    res: usize,
    band_count: u32,
    band_contrast: f64,
    moist_gain: f64,
    base_temp: f64,
    greenhouse: f64,
    gradient: f64,
    lapse_factor: f64,
    lapse_rate: f64,
    invert_blend: f64,
    height_fn: &impl Fn(DVec3, u32) -> f64,
    crust_dist_at: &impl Fn(DVec3) -> f64,
) -> Vec<f32> {
    let polar_axis = DVec3::Y;
    let total = 6 * res * res;
    let mut field = vec![0.0f32; total];

    for face in 0..6usize {
        for y in 0..res {
            for x in 0..res {
                let dir = texel_to_dir(face, x, y, res);

                // 1. Banded circulation wetness.
                let band = band_wetness(dir.y, band_count).powf(band_contrast);

                // 2. Sea proximity — drier inland, floor so interiors aren't bone-dry.
                let cd = crust_dist_at(dir);
                let sea_prox = MOIST_SEAPROX_FLOOR
                    + (1.0 - MOIST_SEAPROX_FLOOR) * (-cd.max(0.0) / MOIST_INLAND_SCALE).exp();

                // 3. Rain shadow — march UPWIND along this band's zonal wind.
                //    east tangent = normalize(polarAxis × dir); wind sign alternates per band.
                let east_raw = polar_axis.cross(dir);
                let e_len = east_raw.length();
                let shadow;
                if e_len > 1e-6 {
                    let east = east_raw / e_len;
                    let wind_sign = if (band_index(dir.y, band_count) & 1) == 0 { 1.0_f64 } else { -1.0_f64 };
                    // Upwind is opposite the wind travel direction.
                    let up = -wind_sign * east;
                    let this_h = height_fn(dir, RAINSHADOW_LEVEL);
                    let mut max_upwind = this_h;
                    for s in 1..=UPWIND_STEPS {
                        let d = s as f64 * UPWIND_STEP_RAD;
                        let march = (dir + up * d).normalize();
                        let h_up = height_fn(march, RAINSHADOW_LEVEL);
                        if h_up > max_upwind {
                            max_upwind = h_up;
                        }
                    }
                    let raw_shadow = 1.0 - SHADOW_K * (max_upwind - this_h).max(0.0);
                    shadow = raw_shadow.clamp(MIN_SHADOW, 1.0);
                } else {
                    shadow = 1.0;
                }

                let mut m = clamp01(band * sea_prox * shadow * moist_gain);

                // 4. Frozen attenuation — very cold regions hold less liquid moisture.
                let t_height = height_fn(dir, RAINSHADOW_LEVEL);
                let temp = {
                    let abs_y = dir.y.abs();
                    let cos_lat = (1.0 - dir.y * dir.y).max(0.0).sqrt();
                    let lat_ins = cos_lat * (1.0 - invert_blend) + abs_y * invert_blend;
                    let lapse = lapse_rate * t_height.max(0.0) * lapse_factor;
                    base_temp + greenhouse + gradient * (lat_ins - LAT_MEAN) - lapse
                };
                if temp < FROZEN_TEMP {
                    let f = clamp01((FROZEN_TEMP - temp) / FROZEN_WIDTH);
                    m *= 1.0 - 0.85 * f;
                }

                field[texel_index(face, x, y, res)] = m as f32;
            }
        }
    }

    blur_field(&field, res)
}

/// One cross-face box blur (centre weight 4, four cardinal neighbours weight 1).
/// Uses `neighbor_texel` for seam-correct sampling.
fn blur_field(src: &[f32], res: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    for face in 0..6usize {
        for y in 0..res {
            for x in 0..res {
                let centre = src[texel_index(face, x, y, res)] as f64;
                let mut sum = centre * 4.0;
                let mut w = 4.0_f64;
                for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                    let nb = neighbor_texel(face, x, y, dx, dy, res);
                    sum += src[texel_index(nb.face, nb.x, nb.y, res)] as f64;
                    w += 1.0;
                }
                out[texel_index(face, x, y, res)] = (sum / w) as f32;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: flat terrain at height 0, half the planet is "inland" (positive crustDist).
    fn flat_height(_dir: DVec3, _level: u32) -> f64 { 0.0 }
    fn half_inland(dir: DVec3) -> f64 {
        // Positive x hemisphere = inland, negative = sea.
        dir.x
    }

    fn earth_like_params() -> ClimateParams {
        ClimateParams {
            seed: 42,
            base_temp: 15.0,
            atmosphere: 0.6,
            band_count: 3,
            axial_tilt_rad: 23.0_f64.to_radians(),
            redistribution: None,
            greenhouse: None,
            lapse_rate: None,
        }
    }

    #[test]
    fn deterministic_moisture_field() {
        let c1 = Climate::new(earth_like_params(), flat_height, half_inland);
        let c2 = Climate::new(earth_like_params(), flat_height, half_inland);
        assert_eq!(c1.moisture_field.len(), c2.moisture_field.len());
        for (i, (a, b)) in c1.moisture_field.iter().zip(c2.moisture_field.iter()).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "moisture_field[{i}] diverged");
        }
    }

    #[test]
    fn moisture_field_length() {
        let c = Climate::new(earth_like_params(), flat_height, half_inland);
        assert_eq!(c.moisture_field.len(), 6 * MOIST_RES * MOIST_RES);
    }

    #[test]
    fn moisture_in_zero_one() {
        let c = Climate::new(earth_like_params(), flat_height, half_inland);
        for (i, &v) in c.moisture_field.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0,
                "moisture_field[{i}]={v} out of [0,1]");
        }
    }

    #[test]
    fn sample_temperature_plausible_range() {
        let c = Climate::new(earth_like_params(), flat_height, half_inland);
        // Sample at equator, poles, and mid-latitudes at height=0.
        let dirs = [
            DVec3::X,                        // equatorial
            DVec3::Y,                        // north pole
            DVec3::NEG_Y,                    // south pole
            DVec3::new(0.0, 0.5, 0.866).normalize(), // ~30° latitude
        ];
        for dir in dirs {
            let (temp, _) = c.sample(dir, 0.0);
            // Reasonable range: no colder than -100°C, no hotter than +100°C.
            assert!(temp > -100.0 && temp < 100.0,
                "temperature {temp} out of plausible range at {dir:?}");
        }
    }

    #[test]
    fn sample_moisture_in_zero_one() {
        let c = Climate::new(earth_like_params(), flat_height, half_inland);
        let dirs = [
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
            DVec3::new(1.0, 1.0, 1.0).normalize(),
            DVec3::NEG_X,
        ];
        for dir in dirs {
            let (_, moisture) = c.sample(dir, 0.0);
            assert!(moisture >= 0.0 && moisture <= 1.0,
                "moisture {moisture} out of [0,1] at {dir:?}");
        }
    }

    #[test]
    fn sample_deterministic() {
        let c = Climate::new(earth_like_params(), flat_height, half_inland);
        let dir = DVec3::new(0.6, 0.5, 0.3).normalize();
        let (t1, m1) = c.sample(dir, 0.1);
        let (t2, m2) = c.sample(dir, 0.1);
        assert_eq!(t1.to_bits(), t2.to_bits());
        assert_eq!(m1.to_bits(), m2.to_bits());
    }

    #[test]
    fn to_baked_from_baked_round_trip() {
        let c = Climate::new(earth_like_params(), flat_height, half_inland);
        let baked = c.to_baked();
        let c2 = Climate::from_baked(baked);

        // Both must produce identical sample outputs at several directions.
        let dirs = [
            DVec3::X,
            DVec3::Y,
            DVec3::Z,
            DVec3::new(0.3, -0.7, 0.6).normalize(),
            DVec3::new(-1.0, 0.0, 0.0),
        ];
        for dir in dirs {
            let (t1, m1) = c.sample(dir, 0.2);
            let (t2, m2) = c2.sample(dir, 0.2);
            assert_eq!(t1.to_bits(), t2.to_bits(),
                "temperature diverged after round-trip at {dir:?}");
            assert_eq!(m1.to_bits(), m2.to_bits(),
                "moisture diverged after round-trip at {dir:?}");
        }
    }

    #[test]
    fn equator_warmer_than_pole() {
        let c = Climate::new(earth_like_params(), flat_height, half_inland);
        let (equator_temp, _) = c.sample(DVec3::X, 0.0);
        let (pole_temp, _) = c.sample(DVec3::Y, 0.0);
        assert!(equator_temp > pole_temp,
            "equator ({equator_temp}) should be warmer than pole ({pole_temp})");
    }

    #[test]
    fn altitude_lapse_cools() {
        let c = Climate::new(earth_like_params(), flat_height, half_inland);
        let dir = DVec3::X;
        let (t_low, _) = c.sample(dir, 0.0);
        let (t_high, _) = c.sample(dir, 1.0);
        assert!(t_high < t_low,
            "high altitude ({t_high}) should be colder than sea level ({t_low})");
    }

    /// The moisture bake is currently fully analytic (seed reserved for future
    /// jitter, per climate.ts comment). Different *parameters* must produce
    /// different fields; different seeds with identical params produce the same
    /// field by design (the seed touches stream 200 but the current model is
    /// parameter-only).
    #[test]
    fn different_params_differ() {
        let c1 = Climate::new(earth_like_params(), flat_height, half_inland);
        let mut p2 = earth_like_params();
        // Change a parameter that meaningfully affects the bake.
        p2.band_count = 1;
        let c2 = Climate::new(p2, flat_height, half_inland);
        let any_diff = c1.moisture_field.iter().zip(c2.moisture_field.iter())
            .any(|(a, b)| a.to_bits() != b.to_bits());
        assert!(any_diff, "different band_count produced identical moisture fields");
    }
}

//! Stage B — tectonic height field.
//!
//! `TectonicHeightField` implements the full `heightFn` from
//! `demiurge/src/planet/terrainSampler.ts` as a Rust [`HeightField`].
//! It composes:
//!   - [`Tectonics`] — plate query, boundary relief, volcano elevation
//!   - [`Climate`]   — temperature + moisture (delegated from `HeightField::climate`)
//!   - [`Noise3D`]   — fine detail FBM / ridged layers
//!
//! Both `Tectonics` and `Climate` are stored behind `Arc` so the
//! `TectonicHeightField` can be cloned cheaply and shared across worker threads
//! without any data duplication (both types are `Send + Sync`).
//!
//! # LOD-additive-octave invariant
//!
//! Detail FBM and ridged layers use the same additive-octave normalisation as
//! `SimpleHeightField`: octaves 0..BASE are bit-identical across LOD levels;
//! extra octaves are additive on top. Normalization constants:
//!
//! ```text
//!   FBM_BASE_OCTAVES    = 6  → MAX_AMP_FBM6    ≈ 1.96875
//!   RIDGED_BASE_OCTAVES = 4  → MAX_AMP_RIDGED4 = 0.9375
//! ```

use glam::DVec3;
use std::sync::Arc;

use crate::height::HeightField;
use crate::noise::Noise3D;
use crate::tectonics::{Tectonics, boundary_relief};
use crate::climate::{Climate, ClimateParams};

// ---------------------------------------------------------------------------
// LOD schedule constants (mirror terrainSampler.ts)
// ---------------------------------------------------------------------------

/// Anchor octave count for the detail FBM layer.
const FBM_BASE_OCTAVES: u32 = 6;
/// Anchor octave count for the detail ridged layer.
const RIDGED_BASE_OCTAVES: u32 = 4;

const FBM_GAIN: f64 = 0.5;
const FBM_LAC:  f64 = 2.0;

/// Σ_{i=0}^{FBM_BASE_OCTAVES-1} 0.5^i = 63/32 ≈ 1.96875
const MAX_AMP_FBM6: f64 = {
    let mut a = 0.0f64;
    let mut g = 1.0f64;
    let mut i = 0u32;
    while i < FBM_BASE_OCTAVES {
        a += g;
        g *= FBM_GAIN;
        i += 1;
    }
    a
};

/// Σ_{i=0}^{RIDGED_BASE_OCTAVES-1} 0.5^i = 15/16 = 0.9375
const MAX_AMP_RIDGED4: f64 = {
    let mut a = 0.0f64;
    let mut g = 1.0f64;
    let mut i = 0u32;
    while i < RIDGED_BASE_OCTAVES {
        a += g;
        g *= FBM_GAIN;
        i += 1;
    }
    a
};

// ---------------------------------------------------------------------------
// TECT_* tuning constants (single source of truth — ported from terrainSampler.ts)
// ---------------------------------------------------------------------------

// Crust-keyed base height constants
const TECT_LAND_BASE_0:      f64 = 0.06;
const TECT_LAND_BASE_SS:     f64 = 0.17;
const TECT_LAND_PLATE_MOD:   f64 = 0.15;
const TECT_OCEAN_BASE_0:     f64 = -0.42;
const TECT_OCEAN_DEPTH_AMP:  f64 = 0.10;
const TECT_OCEAN_DEPTH_SAT:  f64 = 0.45;
const TECT_OCEAN_PLATE_MOD:  f64 = 0.10;
const TECT_SHELF_W_PASSIVE:  f64 = 0.12;
const TECT_SHELF_W_ACTIVE:   f64 = 0.045;
const TECT_SHELF_W_STAGNANT: f64 = 0.045;
const TECT_COAST_LERP_HI:    f64 = 0.018;

// Ruggedness cascade constants
const BROAD_W:               f64 = 0.15;
const PALEO_W:               f64 = 0.07;
const RUGGED_ACTIVE:         f64 = 1.0;
const RUGGED_PALEO:          f64 = 0.7;
const RUGGED_PALEO_STAGNANT: f64 = 1.1;

// Broad regional uplift
const HIGHLAND_AMP:          f64 = 0.22;

// CC-collision plateau
const PLATEAU_AMP:           f64 = 0.20;
const PLATEAU_LO:            f64 = 0.30;
const PLATEAU_HI:            f64 = 0.70;

// Hill dissection floor — retained to mirror terrainSampler.ts exports;
// the role is folded into the detail-FBM CRATON_FLOOR path.
#[allow(dead_code)]
const HILL_FLOOR: f64 = 0.80;

// Fine detail FBM
const CRATON_FLOOR:          f64 = 0.18;
const TECT_DETAIL_FBM_BASE:  f64 = 0.10;
const TECT_DETAIL_FBM_SCALE: f64 = 3.2;

// Coastal ridging
const TECT_DETAIL_RIDGE:       f64 = 0.025;
const TECT_DETAIL_RIDGE_SCALE: f64 = 7.0;

// Broad undulation
const UNDULATION_AMP:      f64 = 0.17;
const UNDULATION_FREQ:     f64 = 3.5;
const UNDULATION_OCT:      u32 = 4;
const UNDULATION_VAR_FREQ: f64 = 0.7;
const UNDULATION_VAR_OCT:  u32 = 3;
const CRATON_MASK_STR:     f64 = 0.70;

// Offshore skerries
const TECT_ISLAND_AMP:       f64 = 0.12;
const TECT_ISLAND_FREQ:      f64 = 14.0;
const TECT_ISLAND_OCT:       u32 = 4;
const TECT_ISLAND_THRESH:    f64 = 0.55;
const TECT_ISLAND_COAST_LO:  f64 = -0.05;
const TECT_ISLAND_COAST_HI:  f64 = -0.005;

// Volcano headroom attenuation
const TECT_VOLC_HEADROOM_LO: f64 = 0.15;
const TECT_VOLC_HEADROOM_HI: f64 = 0.50;
const TECT_VOLC_SUM_MAX:     f64 = 0.64;

// ---------------------------------------------------------------------------
// Input parameters for TectonicHeightField::new
// ---------------------------------------------------------------------------

/// Construction parameters for [`TectonicHeightField`].
pub struct TectonicHeightFieldParams {
    /// Master seed.
    pub seed: u32,
    /// Target number of tectonic plates (0–48; 0/1 = stagnant lid).
    pub plate_count: usize,
    /// Arc-volcano density multiplier (clamped 0.2–3.0).
    pub arc_density: f64,
    /// Number of mantle hotspots (clamped 0–20).
    pub hotspot_count: u32,
    /// Hotspot height multiplier (clamped 0–3).
    pub hotspot_intensity: f64,
    /// Climate params (forwarded to `Climate::new`).
    pub climate: ClimateParams,
}

// ---------------------------------------------------------------------------
// TectonicHeightField
// ---------------------------------------------------------------------------

/// Full tectonic terrain height field — the production implementation of
/// [`HeightField`].
///
/// Holds `Arc<Tectonics>` and `Arc<Climate>` so it can be cheaply cloned
/// and shared across mesher worker threads without data duplication.
pub struct TectonicHeightField {
    pub tectonics: Arc<Tectonics>,
    pub climate:   Arc<Climate>,
    pub noise:     Noise3D,
    is_stagnant_lid: bool,
    shelf_w_base:    f64,
}

impl TectonicHeightField {
    /// Construct a new `TectonicHeightField`.
    ///
    /// Builds `Tectonics` + `Climate` from the supplied parameters. Both are
    /// wrapped in `Arc` immediately; construction is the only expensive step.
    pub fn new(params: TectonicHeightFieldParams) -> Self {
        let seed = params.seed;

        let tectonics = Arc::new(Tectonics::new(
            seed,
            params.plate_count,
            params.arc_density,
            params.hotspot_count,
            params.hotspot_intensity,
        ));

        let is_stagnant_lid = tectonics.plates.len() == 1;
        let shelf_w_base = if is_stagnant_lid {
            TECT_SHELF_W_STAGNANT
        } else {
            TECT_SHELF_W_PASSIVE
        };

        let noise = Noise3D::new(seed);

        // Climate::new needs a height_fn and crust_dist_at.
        // We clone the Arc for the closures (cheap).
        let tect_for_climate = Arc::clone(&tectonics);
        let tect_for_crust   = Arc::clone(&tectonics);
        let noise_for_climate = Noise3D::new(seed);

        let is_stagnant_lid_c = is_stagnant_lid;
        let shelf_w_base_c    = shelf_w_base;

        let height_fn_for_climate = move |dir: DVec3, level: u32| -> f64 {
            tect_height(
                &tect_for_climate,
                &noise_for_climate,
                dir,
                level as u8,
                is_stagnant_lid_c,
                shelf_w_base_c,
            )
        };

        let crust_dist_at = move |dir: DVec3| -> f64 {
            tect_for_crust.query(dir).crust_dist
        };

        let climate = Arc::new(Climate::new(
            params.climate,
            height_fn_for_climate,
            crust_dist_at,
        ));

        TectonicHeightField {
            tectonics,
            climate,
            noise,
            is_stagnant_lid,
            shelf_w_base,
        }
    }
}

// ---------------------------------------------------------------------------
// HeightField implementation
// ---------------------------------------------------------------------------

impl HeightField for TectonicHeightField {
    fn height(&self, dir: DVec3, level: u8) -> f64 {
        tect_height(
            &self.tectonics,
            &self.noise,
            dir,
            level,
            self.is_stagnant_lid,
            self.shelf_w_base,
        )
    }

    fn plate_color(&self, dir: DVec3) -> [f32; 3] {
        let q = self.tectonics.query(dir);
        let plates = &self.tectonics.plates;
        let base = plates[q.plate_id].color;

        // Regime tint near boundaries (mirrors plateColorFn in terrainSampler.ts)
        let bd = q.boundary_dist;
        if bd < 0.035 {
            let t = 1.0 - smoothstep(0.0, 0.035, bd);
            let conv = q.convergence;
            let sh   = q.shear;
            let (tr, tg, tb) = if conv > 0.12 {
                (0.95f64, 0.18f64, 0.12f64)
            } else if conv < -0.12 {
                (0.15f64, 0.35f64, 0.95f64)
            } else if sh.abs() > 0.18 {
                (0.95f64, 0.85f64, 0.15f64)
            } else {
                // Quiet boundary — no tint
                return [base[0] as f32, base[1] as f32, base[2] as f32];
            };
            let blend = 0.65 * t;
            return [
                (base[0] * (1.0 - blend) + tr * blend) as f32,
                (base[1] * (1.0 - blend) + tg * blend) as f32,
                (base[2] * (1.0 - blend) + tb * blend) as f32,
            ];
        }
        [base[0] as f32, base[1] as f32, base[2] as f32]
    }

    fn climate(&self, dir: DVec3, height: f64) -> (f32, f32) {
        self.climate.sample(dir, height)
    }
}

// ---------------------------------------------------------------------------
// Core height function (free function so Climate::new can call it)
// ---------------------------------------------------------------------------

/// The full tectonic heightFn, ported faithfully from `terrainSampler.ts`.
///
/// Extracted as a free function so `TectonicHeightField::new` can pass a
/// closure to `Climate::new` without a partial borrow of `self`.
fn tect_height(
    tectonics: &Tectonics,
    noise: &Noise3D,
    dir: DVec3,
    level: u8,
    is_stagnant_lid: bool,
    shelf_w_base: f64,
) -> f64 {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    let q   = tectonics.query(dir);
    let own = &tectonics.plates[q.plate_id];
    let c   = q.crust_dist; // signed SDF: + = crust/land, - = ocean

    // --- Ocean base ---
    let ocean_base = TECT_OCEAN_BASE_0
        - TECT_OCEAN_DEPTH_AMP
            * ((-c).max(0.0) / TECT_OCEAN_DEPTH_SAT).sqrt().min(1.0)
        + own.base_elevation * TECT_OCEAN_PLATE_MOD;

    // --- Land base ---
    let land_base = TECT_LAND_BASE_0
        + TECT_LAND_BASE_SS * smoothstep(0.0, 0.30, c)
        + own.base_elevation * TECT_LAND_PLATE_MOD;

    // --- Ruggedness cascade ---
    let active_uplift = q.convergence.max(0.0) * (-q.boundary_dist / BROAD_W).exp();
    let paleo_uplift  = (-q.paleo_dist / PALEO_W).exp();
    let rugged_raw    = RUGGED_ACTIVE * active_uplift
        + (if is_stagnant_lid { RUGGED_PALEO_STAGNANT } else { RUGGED_PALEO }) * paleo_uplift;
    let rugged    = rugged_raw.clamp(0.0, 1.0);
    let land_gate = smoothstep(0.0, 0.02, c);

    // --- Broad undulation (signed fbm — raises AND lowers) ---
    let und_noise = fbm_fixed(noise, x * UNDULATION_FREQ, y * UNDULATION_FREQ, z * UNDULATION_FREQ, UNDULATION_OCT);
    let und_var_raw = fbm_fixed(noise, x * UNDULATION_VAR_FREQ, y * UNDULATION_VAR_FREQ, z * UNDULATION_VAR_FREQ, UNDULATION_VAR_OCT);
    let und_var   = 0.5 + 0.8 * (und_var_raw * 0.5 + 0.5); // [0.5, 1.3]
    let undulation = UNDULATION_AMP * und_var * und_noise * land_gate * (1.0 - CRATON_MASK_STR * rugged);

    // --- CC collision weight ---
    let w_mine  = smoothstep(-0.10, 0.10, c);
    let w_other = smoothstep(-0.10, 0.10, q.other_crust_dist);
    let cc_collision = w_mine * w_other;

    // --- Broad uplift (tectonically gated) ---
    let broad_uplift = HIGHLAND_AMP * rugged * land_gate;

    // --- CC-collision plateau ---
    let plateau_frac = smoothstep(PLATEAU_LO, PLATEAU_HI, rugged * cc_collision);
    let plateau      = PLATEAU_AMP * plateau_frac * land_gate;

    // --- Shelf width (narrows at active margins) ---
    let activeness = (1.0 - smoothstep(0.02, 0.06, q.boundary_dist))
        * smoothstep(0.08, 0.25, q.convergence.abs().max(q.shear.abs()));
    let shelf_w = shelf_w_base * (1.0 - activeness) + TECT_SHELF_W_ACTIVE * activeness;

    // --- Base elevation (ocean/land blend across the shelf) ---
    let combined_land = land_base + broad_uplift + plateau;
    let base = combined_land
        + (ocean_base - combined_land) * (1.0 - smoothstep(-shelf_w, TECT_COAST_LERP_HI, c));

    // --- Boundary relief ---
    let ridged_at = |d: DVec3, freq: f64, octaves: u32| -> f64 {
        ridged_fixed(noise, d.x * freq, d.y * freq, d.z * freq, octaves)
    };
    let relief = boundary_relief(&q, &tectonics.plates, dir, &ridged_at);

    // --- LOD-adaptive octave counts ---
    let fbm_octaves: u32    = (level as u32 + 2).clamp(FBM_BASE_OCTAVES, 14);
    let ridged_octaves: u32 = (level as i32 - 2).max(0) as u32;
    let ridged_octaves: u32 = ridged_octaves.clamp(RIDGED_BASE_OCTAVES, 6);

    // --- Detail FBM (additive-octave, LOD-consistent) ---
    let detail_amp = TECT_DETAIL_FBM_BASE * (CRATON_FLOOR + (1.0 - CRATON_FLOOR) * rugged);
    let fx = x * TECT_DETAIL_FBM_SCALE;
    let fy = y * TECT_DETAIL_FBM_SCALE;
    let fz = z * TECT_DETAIL_FBM_SCALE;
    let fbm_raw = fbm_additive(noise, fx, fy, fz, fbm_octaves);
    let detail_fbm = detail_amp * fbm_raw;

    // --- Detail ridged (coastal only) ---
    let rx = x * TECT_DETAIL_RIDGE_SCALE;
    let ry = y * TECT_DETAIL_RIDGE_SCALE;
    let rz = z * TECT_DETAIL_RIDGE_SCALE;
    let ridged_raw = ridged_additive(noise, rx, ry, rz, ridged_octaves);
    let detail_ridged = TECT_DETAIL_RIDGE
        * ridged_raw
        * smoothstep(0.0, 0.08, c)
        * (0.7 + 0.3 * (-q.boundary_dist / 0.18).exp());

    let detail = detail_fbm + detail_ridged;

    // --- Offshore skerries / archipelagos ---
    let mid_band = TECT_ISLAND_COAST_LO * 0.5 + TECT_ISLAND_COAST_HI * 0.5;
    let island_band = smoothstep(TECT_ISLAND_COAST_LO, mid_band, c)
        * (1.0 - smoothstep(mid_band, TECT_ISLAND_COAST_HI, c));
    let island_h = if island_band > 0.01 {
        let i_raw = fbm_fixed(
            noise,
            x * TECT_ISLAND_FREQ,
            y * TECT_ISLAND_FREQ,
            z * TECT_ISLAND_FREQ,
            TECT_ISLAND_OCT,
        );
        let i_excess = (i_raw - TECT_ISLAND_THRESH).max(0.0) / (1.0 - TECT_ISLAND_THRESH);
        (TECT_ISLAND_AMP * i_excess).min(TECT_ISLAND_AMP) * island_band
    } else {
        0.0
    };

    // --- Arc volcanoes (headroom-aware) ---
    let terrain_h = base + relief + detail + island_h + undulation;
    let volcano   = (tectonics.volcano_elevation(dir)).min(TECT_VOLC_SUM_MAX);
    let headroom  = 1.0 - smoothstep(TECT_VOLC_HEADROOM_LO, TECT_VOLC_HEADROOM_HI, terrain_h);

    (terrain_h + volcano * headroom).clamp(-1.0, 1.0)
}

// ---------------------------------------------------------------------------
// Local math helpers
// ---------------------------------------------------------------------------

/// GLSL smoothstep — mirrors `_ss3` in terrainSampler.ts.
#[inline(always)]
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Accumulate `octaves` fbm octaves; divide by FIXED `MAX_AMP_FBM6`.
///
/// This is the additive-invariant form: the result for octaves 0..FBM_BASE_OCTAVES
/// is bit-identical across different `octaves` values (extra octaves only add on top).
fn fbm_additive(n: &Noise3D, x: f64, y: f64, z: f64, octaves: u32) -> f64 {
    let mut value = 0.0f64;
    let mut amp   = 1.0f64;
    let mut f     = 1.0f64;
    for _ in 0..octaves {
        value += amp * n.sample(x * f, y * f, z * f);
        amp   *= FBM_GAIN;
        f     *= FBM_LAC;
    }
    value / MAX_AMP_FBM6
}

/// Accumulate `octaves` ridged octaves; divide by FIXED `MAX_AMP_RIDGED4`.
fn ridged_additive(n: &Noise3D, x: f64, y: f64, z: f64, octaves: u32) -> f64 {
    let mut value = 0.0f64;
    let mut amp   = 1.0f64;
    let mut f     = 1.0f64;
    for _ in 0..octaves {
        let s = n.sample(x * f, y * f, z * f);
        value += amp * (1.0 - s.abs()).powi(2);
        amp   *= FBM_GAIN;
        f     *= FBM_LAC;
    }
    value / MAX_AMP_RIDGED4
}

/// FBM with a self-normalising max-amp (used for non-detail layers where
/// LOD consistency is not required — undulation, islands, etc.).
/// Mirrors the standard `fbm()` from noise.rs but inlined here to avoid
/// a different function signature.
fn fbm_fixed(n: &Noise3D, x: f64, y: f64, z: f64, octaves: u32) -> f64 {
    let mut value   = 0.0f64;
    let mut amp     = 1.0f64;
    let mut max_amp = 0.0f64;
    let mut f       = 1.0f64;
    for _ in 0..octaves {
        value   += amp * n.sample(x * f, y * f, z * f);
        max_amp += amp;
        amp     *= FBM_GAIN;
        f       *= FBM_LAC;
    }
    if max_amp > 0.0 { value / max_amp } else { 0.0 }
}

/// Ridged with self-normalising max-amp (used in `boundary_relief` callback).
fn ridged_fixed(n: &Noise3D, x: f64, y: f64, z: f64, octaves: u32) -> f64 {
    let mut value   = 0.0f64;
    let mut amp     = 1.0f64;
    let mut max_amp = 0.0f64;
    let mut f       = 1.0f64;
    for _ in 0..octaves {
        let s = n.sample(x * f, y * f, z * f);
        value   += amp * (1.0 - s.abs()).powi(2);
        max_amp += amp;
        amp     *= FBM_GAIN;
        f       *= FBM_LAC;
    }
    if max_amp > 0.0 { value / max_amp } else { 0.0 }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    // Small plate count for test speed: 4 plates, no hotspots.
    fn make_params(seed: u32) -> TectonicHeightFieldParams {
        TectonicHeightFieldParams {
            seed,
            plate_count: 4,
            arc_density: 1.0,
            hotspot_count: 0,
            hotspot_intensity: 0.0,
            climate: ClimateParams {
                seed,
                base_temp: 15.0,
                atmosphere: 0.6,
                band_count: 3,
                axial_tilt_rad: 23.0_f64.to_radians(),
                redistribution: None,
                greenhouse: None,
                lapse_rate: None,
            },
        }
    }

    fn make_hf(seed: u32) -> TectonicHeightField {
        TectonicHeightField::new(make_params(seed))
    }

    fn sample_sphere_dirs(n: usize) -> Vec<DVec3> {
        let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
        (0..n)
            .map(|i| {
                let y = 1.0 - (i as f64 / (n as f64 - 1.0)) * 2.0;
                let r = (1.0 - y * y).max(0.0).sqrt();
                let theta = golden * i as f64;
                DVec3::new(r * theta.cos(), y, r * theta.sin()).normalize()
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // height() ∈ [-1, 1]
    // -----------------------------------------------------------------------

    #[test]
    fn height_in_range() {
        let hf = make_hf(42);
        let dirs = sample_sphere_dirs(200);
        for &d in &dirs {
            let v = hf.height(d, 0);
            assert!(v >= -1.0 && v <= 1.0, "height out of range: {v}");
        }
    }

    // -----------------------------------------------------------------------
    // Both ocean (<0) and land (>0) present over a sphere
    // -----------------------------------------------------------------------

    #[test]
    fn height_has_ocean_and_land() {
        let hf = make_hf(42);
        let dirs = sample_sphere_dirs(400);
        let has_ocean = dirs.iter().any(|&d| hf.height(d, 3) < 0.0);
        let has_land  = dirs.iter().any(|&d| hf.height(d, 3) > 0.0);
        assert!(has_ocean, "no ocean found (all heights >= 0)");
        assert!(has_land,  "no land found (all heights <= 0)");
    }

    // -----------------------------------------------------------------------
    // Determinism: same seed → identical results
    // -----------------------------------------------------------------------

    #[test]
    fn height_deterministic() {
        let hf1 = make_hf(77);
        let hf2 = make_hf(77);
        let dirs = sample_sphere_dirs(80);
        for level in [0u8, 3, 6] {
            for &d in &dirs {
                let a = hf1.height(d, level);
                let b = hf2.height(d, level);
                assert_eq!(
                    a.to_bits(), b.to_bits(),
                    "non-deterministic at level={level}, dir={d:?}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Additive-octave LOD invariant
    //
    // The detail FBM base (first FBM_BASE_OCTAVES octaves / MAX_AMP_FBM6)
    // must be bit-identical across LOD levels because MAX_AMP_FBM6 never
    // changes — extra octaves are only additive. We verify by checking that
    // the 6-octave accumulation (level=0) equals the embedded 6-oct prefix
    // of a higher-level call, and that the difference between two adjacent
    // levels is small (only one extra octave).
    // -----------------------------------------------------------------------

    #[test]
    fn additive_octave_invariant() {
        let n = Noise3D::new(55);
        let dirs = sample_sphere_dirs(50);

        for &dir in &dirs {
            let x = dir.x * TECT_DETAIL_FBM_SCALE;
            let y = dir.y * TECT_DETAIL_FBM_SCALE;
            let z = dir.z * TECT_DETAIL_FBM_SCALE;

            // 6 octaves (base): bit-identical regardless of how many more follow.
            let base6 = fbm_additive(&n, x, y, z, FBM_BASE_OCTAVES);
            // 7 octaves: first 6 must equal base6 / MAX_AMP_FBM6 up to the extra octave.
            let oct7 = fbm_additive(&n, x, y, z, FBM_BASE_OCTAVES + 1);

            // base6 * MAX_AMP_FBM6 == raw 6-oct sum == first-6-oct raw sum of oct7 * MAX_AMP_FBM6
            // ↔ oct7 - base6 == amplitude of octave 7 / MAX_AMP_FBM6 (small)
            let delta = (oct7 - base6).abs();
            // Max amplitude of octave 7 = 0.5^6 / MAX_AMP_FBM6 ≈ 0.016
            assert!(
                delta < 0.04,
                "delta between 6- and 7-octave fbm too large ({delta}); normalization broken"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Low-frequency component matches across adjacent LOD levels
    // (public HeightField API level)
    // -----------------------------------------------------------------------

    #[test]
    fn height_low_freq_matches_across_levels() {
        let hf = make_hf(88);
        let dirs = sample_sphere_dirs(30);

        for &dir in &dirs {
            let h0 = hf.height(dir, 0);
            let h1 = hf.height(dir, 1);
            // h1 has at most 1 extra fbm octave (amp ≈ 0.5^6/MAX_AMP_FBM6 ≈ 0.016)
            // plus 0 extra ridged octaves (both clamp to 4 for level < 6).
            let diff = (h1 - h0).abs();
            assert!(
                diff < 0.10,
                "height(dir, 0) and height(dir, 1) differ by {diff} — LOD invariant broken at {dir:?}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // plate_color delegates to tectonics and returns [0,1] values
    // -----------------------------------------------------------------------

    #[test]
    fn plate_color_delegates_correctly() {
        let hf = make_hf(42);
        let dirs = sample_sphere_dirs(50);
        for &d in &dirs {
            let c = hf.plate_color(d);
            for (i, &ch) in c.iter().enumerate() {
                assert!(
                    ch >= 0.0 && ch <= 1.0,
                    "plate_color channel {i} = {ch} out of [0,1]"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // climate delegates to Climate::sample
    // -----------------------------------------------------------------------

    #[test]
    fn climate_delegates_correctly() {
        let hf = make_hf(42);
        let dirs = sample_sphere_dirs(50);
        for &d in &dirs {
            let h = hf.height(d, 0);
            let (temp, moisture) = hf.climate(d, h);
            // Temperature: reasonable physical range
            assert!(
                temp > -150.0 && temp < 150.0,
                "temperature {temp} out of plausible range"
            );
            // Moisture: clamped [0,1]
            assert!(
                moisture >= 0.0 && moisture <= 1.0,
                "moisture {moisture} out of [0,1]"
            );
        }
    }
}

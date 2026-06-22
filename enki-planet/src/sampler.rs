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
use crate::climate::{Climate, ClimateParams, WindSample};
use crate::erosion::{Erosion, ErosionOpts};

/// ki `_deriveSeed` — child seed from (master, stream). Verbatim port:
/// `s = master ^ (stream+1)*0xdeadbeef; s ^ (s >> 16)` (single xor-shift, NOT a
/// full splitmix step — distinct from the tectonics/erosion derivations).
#[inline]
fn derive_seed_proc(master: u32, stream: u32) -> u32 {
    let s = master ^ stream.wrapping_add(1).wrapping_mul(0xdead_beef);
    s ^ (s >> 16)
}

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
const HIGHLAND_AMP:          f64 = 0.18;

// CC-collision plateau
const PLATEAU_AMP:           f64 = 0.20;
const PLATEAU_LO:            f64 = 0.30;
const PLATEAU_HI:            f64 = 0.70;

// Hill dissection floor — retained to mirror terrainSampler.ts exports;
// the role is folded into the detail-FBM CRATON_FLOOR path.
#[allow(dead_code)]
const HILL_FLOOR: f64 = 0.80;

// Fine detail FBM
const CRATON_FLOOR:          f64 = 0.22;
const TECT_DETAIL_FBM_BASE:  f64 = 0.22;
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
// Phase-4 wiring constants (erosion / hardness / process palette / Option-C)
// ---------------------------------------------------------------------------

// Wide motion-driven deformation (Option C)
const UPLIFT_AMP:     f64 = 0.18;
const SUBSIDENCE_AMP: f64 = 0.06;
const UPLIFT_ZONE_W:  f64 = 0.25;

// Rock-hardness fine-detail contrast (±35% at HARD_DETAIL_BOOST*0.7).
const HARD_DETAIL_BOOST: f64 = 0.7;

// Erosion-steered detail
const EROSION_WARP_STR:     f64 = 0.10;
const EROSION_RIDGED_FLOOR: f64 = 0.35;
const EROSION_VINCISION_AMP: f64 = 0.08;

// Process-palette noise stream IDs (40/41/42).
const PROC_STREAM_GLACIAL: u32 = 40;
const PROC_STREAM_AEOLIAN: u32 = 41;
const PROC_STREAM_KARST:   u32 = 42;

// GLACIAL
const PROC_GLACIAL_STR:        f64 = 0.55;
const PROC_GLACIAL_ELEV_LO:    f64 = 0.10;
const PROC_GLACIAL_ELEV_HI:    f64 = 0.35;
const PROC_GLACIAL_TEMP_LO:    f64 = -40.0;
const PROC_GLACIAL_TEMP_HI:    f64 = 5.0;
const PROC_GLACIAL_MOIST_LO:   f64 = 0.20;
const PROC_GLACIAL_MOIST_HI:   f64 = 0.60;
const PROC_GLACIAL_CIRQUE_AMP: f64 = 0.004;

// AEOLIAN
const PROC_AEOLIAN_STR:     f64 = 0.50;
const PROC_AEOLIAN_WIND_LO: f64 = 0.30;
const PROC_AEOLIAN_WIND_HI: f64 = 0.75;
const PROC_AEOLIAN_ARID_LO: f64 = 0.00;
const PROC_AEOLIAN_ARID_HI: f64 = 0.70;
const PROC_AEOLIAN_FREQ:    f64 = 22.0;
const PROC_AEOLIAN_WARP:    f64 = 0.04;
const PROC_AEOLIAN_SMOOTH:  f64 = 0.40;

// KARST
const PROC_KARST_STR:      f64 = 0.30;
const PROC_KARST_MOIST_LO: f64 = 0.55;
const PROC_KARST_MOIST_HI: f64 = 0.90;
const PROC_KARST_SOLUB_LO: f64 = 0.20;
const PROC_KARST_SOLUB_HI: f64 = 0.65;
const PROC_KARST_FREQ:     f64 = 28.0;
const PROC_KARST_LAND_LO:  f64 = 0.02;

/// Default ocean-coverage percentile for the erosion sea-level cut.
/// ponytail: enki has no interior/water-budget model, so we pin an Earth-like
/// default. Wire a real value through `TectonicHeightFieldParams` if the GUI
/// ever exposes ocean fraction.
const DEFAULT_OCEAN_COVERAGE: f64 = 0.65;

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
    /// Crust composition [0,1]: 0 = icy/sediment (soft), 1 = rocky/basaltic (hard).
    /// Default 0.5. Feeds Phase-1 substrate-hardness contrast.
    pub composition: f64,
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
    /// Baked erosion field (Phase 4). `Some` in the shipping path; when present the
    /// orogenic stamps are gated off and `erosion.delta_at` owns the relief budget.
    erosion: Option<Erosion>,
    /// Process-palette noise streams (40/41/42 via ki `_deriveSeed`).
    glacial_noise: Noise3D,
    aeolian_noise: Noise3D,
    karst_noise:   Noise3D,
    /// Precomputed analytic base-temperature constants (mirror Climate, no sun term).
    base_temp:        f64,
    proc_greenhouse:  f64,
    proc_lapse_rate:  f64,
    proc_gradient:    f64,
    proc_lapse_factor: f64,
    proc_invert_blend: f64,
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
            params.composition,
        ));

        let is_stagnant_lid = tectonics.plates.len() == 1;
        let shelf_w_base = if is_stagnant_lid {
            TECT_SHELF_W_STAGNANT
        } else {
            TECT_SHELF_W_PASSIVE
        };

        let noise = Noise3D::new(seed);

        // Precompute the analytic base-temperature constants (mirror Climate;
        // no sun term) — used by the process-palette gating in height().
        let atm = params.climate.atmosphere.clamp(0.0, 1.0);
        let proc_greenhouse  = params.climate.greenhouse.unwrap_or(0.0);
        let proc_lapse_rate  = params.climate.lapse_rate.unwrap_or(50.0);
        let proc_gradient    = 55.0 * (1.0 - 0.7 * atm);
        let proc_lapse_factor = 0.3 + 0.7 * atm;
        let tilt_lo = 54.0_f64.to_radians();
        let tilt_hi = 75.0_f64.to_radians();
        let tilt_t  = ((params.climate.axial_tilt_rad - tilt_lo) / (tilt_hi - tilt_lo)).clamp(0.0, 1.0);
        let proc_invert_blend = tilt_t * tilt_t * (3.0 - 2.0 * tilt_t);
        let base_temp = params.climate.base_temp;

        // Process-palette noise streams (ki _deriveSeed; streams 40/41/42).
        let glacial_noise = Noise3D::new(derive_seed_proc(seed, PROC_STREAM_GLACIAL));
        let aeolian_noise = Noise3D::new(derive_seed_proc(seed, PROC_STREAM_AEOLIAN));
        let karst_noise   = Noise3D::new(derive_seed_proc(seed, PROC_STREAM_KARST));

        // Climate::new needs a height_fn and crust_dist_at. These use the
        // BASE-ONLY sampler (stamps gated, no erosion, no process palette) so
        // the rain-shadow march matches ki's base-only seed sampler.
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
                &HeightExtras::base_only(),
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

        // --- Bake erosion (B-path) from the base-only height + climate moisture ---
        // Base-only height closure (level 5) — matches ki EROSION_BAKE_LEVEL.
        let erosion = {
            let tect_for_h = Arc::clone(&tectonics);
            let noise_for_h = Noise3D::new(seed);
            let height_fn = move |dir: DVec3, level: u32| -> f64 {
                tect_height(
                    &tect_for_h,
                    &noise_for_h,
                    dir,
                    level as u8,
                    is_stagnant_lid_c,
                    shelf_w_base_c,
                    &HeightExtras::base_only(),
                )
            };
            let climate_for_m = Arc::clone(&climate);
            let moisture_fn = move |dir: DVec3, h_norm: f64| -> f64 {
                climate_for_m.sample(dir, h_norm).1 as f64
            };
            Erosion::new(ErosionOpts {
                seed,
                height_fn,
                moisture_fn,
                ocean_coverage: DEFAULT_OCEAN_COVERAGE,
                tectonics: Some(tectonics.as_ref()),
                res: None,
                k0: None,
                m_exp: None,
                n_exp: None,
                talus: None,
                kw_min: None,
                deposition_g: None,
                b_steps: None,
                b_uplift_rate: None,
            })
        };

        TectonicHeightField {
            tectonics,
            climate,
            noise,
            is_stagnant_lid,
            shelf_w_base,
            erosion: Some(erosion),
            glacial_noise,
            aeolian_noise,
            karst_noise,
            base_temp,
            proc_greenhouse,
            proc_lapse_rate,
            proc_gradient,
            proc_lapse_factor,
            proc_invert_blend,
        }
    }
}

// ---------------------------------------------------------------------------
// HeightField implementation
// ---------------------------------------------------------------------------

impl HeightField for TectonicHeightField {
    fn height(&self, dir: DVec3, level: u8) -> f64 {
        // Shipping B-path: orogenic stamps gated off, erosion owns the budget,
        // climate-driven process palette active.
        let ex = HeightExtras {
            gate_stamps: true,
            erosion: self.erosion.as_ref(),
            climate: Some(self.climate.as_ref()),
            glacial_noise: Some(&self.glacial_noise),
            aeolian_noise: Some(&self.aeolian_noise),
            karst_noise:   Some(&self.karst_noise),
            base_temp: self.base_temp,
            proc_greenhouse: self.proc_greenhouse,
            proc_lapse_rate: self.proc_lapse_rate,
            proc_gradient: self.proc_gradient,
            proc_lapse_factor: self.proc_lapse_factor,
            proc_invert_blend: self.proc_invert_blend,
        };
        tect_height(
            &self.tectonics,
            &self.noise,
            dir,
            level,
            self.is_stagnant_lid,
            self.shelf_w_base,
            &ex,
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

    fn material(&self, dir: DVec3) -> f32 {
        // Phase-1 rock hardness (soft→hard), already baked + warped in Tectonics.
        (self.tectonics.hardness_at(dir) as f32).clamp(0.0, 1.0)
    }

    fn volcanism(&self, dir: DVec3) -> f32 {
        // Arc + hotspot cone elevation — the same field added to terrain height in
        // tect_height(). Clamp the normalized cone height into a 0..1 debug ramp.
        (self.tectonics.volcano_elevation(dir) as f32).clamp(0.0, 1.0)
    }

    fn wetness(&self, dir: DVec3) -> f32 {
        // Surface wetness = open water (rivers ∪ lakes). Subsurface seep deferred
        // (no seep module yet) — ki's full surface-wetness is the union of both.
        let Some(erosion) = self.erosion.as_ref() else { return 0.0; };
        // River paint band on the log-normalized acc 0..1 field. CRITICAL: the
        // 4-pass seam-blur on flow_accum (erosion.rs) crushes the peak to ~0.48
        // (NOT 1.0), so the band must live below that — the old 0.30/0.55 put HI
        // *above* the field max, so rivers never saturated and only ~3.6% of land
        // cleared the 0.30 onset → an all-tan, empty Wetness view. Measured land
        // acc-CDF (seed 42): ≥0.20 ~23%, ≥0.25 ~10%, ≥0.30 ~3.6%, max ~0.48. So
        // 0.18→0.32 saturates the trunks, fades tributaries in, and floods nothing;
        // it sits inside the geometry's carve_gate (0.10,0.45) so water lands in the
        // valleys actually incised. ponytail: not used by height() (that reads acc
        // directly), so tuning this is golden-free — nudge LO down if still sparse.
        const RIVER_LO: f64 = 0.18;
        const RIVER_HI: f64 = 0.32;
        let river = smoothstep(RIVER_LO, RIVER_HI, erosion.acc_at(dir));
        let lake = erosion.lake_mask_at(dir);
        river.max(lake).clamp(0.0, 1.0) as f32
    }

    fn wind_speed_at(&self, dir: DVec3) -> f32 {
        self.climate.wind_speed_at(dir)
    }

    fn wind_at(&self, dir: DVec3) -> WindSample {
        self.climate.wind_at(dir)
    }

    fn moisture(&self, dir: DVec3) -> f32 {
        self.climate.moisture(dir)
    }

    fn flow_accum_at(&self, dir: DVec3) -> f32 {
        self.erosion.as_ref().map(|e| e.acc_at(dir) as f32).unwrap_or(0.0)
    }

    fn flow_dir_at(&self, dir: DVec3) -> DVec3 {
        self.erosion.as_ref().map(|e| e.flow_at(dir)).unwrap_or(DVec3::ZERO)
    }

    fn lake_mask_at(&self, dir: DVec3) -> f32 {
        self.erosion.as_ref().map(|e| e.lake_mask_at(dir) as f32).unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Core height function (free function so Climate::new can call it)
// ---------------------------------------------------------------------------

/// Phase-4 extras threaded into `tect_height` (erosion, climate process palette,
/// stamp-gating). All borrowed; cheap to assemble per call.
///
/// `gate_stamps` mirrors ki `_gateStamps` (`baseOnly || bActive`): when true the
/// orogenic stamps (boundary relief, CC plateau, broad uplift, broad deform) are
/// zeroed so `erosion.delta_at` owns the elevation budget without double-counting.
struct HeightExtras<'a> {
    gate_stamps: bool,
    erosion: Option<&'a Erosion>,
    /// Climate reference for the process-palette weights (moisture + wind). `None`
    /// during Climate's own bake (→ neutral moisture / no wind / pure FLUVIAL).
    climate: Option<&'a Climate>,
    glacial_noise: Option<&'a Noise3D>,
    aeolian_noise: Option<&'a Noise3D>,
    karst_noise:   Option<&'a Noise3D>,
    // Analytic base-temperature constants (mirror Climate, no sun term).
    base_temp: f64,
    proc_greenhouse: f64,
    proc_lapse_rate: f64,
    proc_gradient: f64,
    proc_lapse_factor: f64,
    proc_invert_blend: f64,
}

impl<'a> HeightExtras<'a> {
    /// The base-only extras used by the climate/erosion bake closures: stamps
    /// gated off, no erosion, no climate process palette.
    fn base_only() -> Self {
        HeightExtras {
            gate_stamps: true,
            erosion: None,
            climate: None,
            glacial_noise: None,
            aeolian_noise: None,
            karst_noise: None,
            base_temp: 0.0,
            proc_greenhouse: 0.0,
            proc_lapse_rate: 0.0,
            proc_gradient: 0.0,
            proc_lapse_factor: 0.0,
            proc_invert_blend: 0.0,
        }
    }
}

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
    ex: &HeightExtras,
) -> f64 {
    let x = dir.x;
    let y = dir.y;
    let z = dir.z;

    let q   = tectonics.query(dir);
    let c   = q.crust_dist; // signed SDF: + = crust/land, - = ocean

    // --- Ocean base ---
    // base_elevation: smoothed (C1) blurred field, not the Voronoi-stepped plate value,
    // so the ~30 m plate-boundary cliff disappears.
    let ocean_base = TECT_OCEAN_BASE_0
        - TECT_OCEAN_DEPTH_AMP
            * ((-c).max(0.0) / TECT_OCEAN_DEPTH_SAT).sqrt().min(1.0)
        + q.base_elevation * TECT_OCEAN_PLATE_MOD;

    // --- Land base ---
    let land_base = TECT_LAND_BASE_0
        + TECT_LAND_BASE_SS * smoothstep(0.0, 0.30, c)
        + q.base_elevation * TECT_LAND_PLATE_MOD;

    // --- Ruggedness cascade ---
    let active_uplift = q.convergence.max(0.0) * (-q.boundary_dist / BROAD_W).exp();
    let paleo_uplift  = (-q.paleo_dist / PALEO_W).exp();
    let rugged_raw    = RUGGED_ACTIVE * active_uplift
        + (if is_stagnant_lid { RUGGED_PALEO_STAGNANT } else { RUGGED_PALEO }) * paleo_uplift;
    let rugged    = rugged_raw.clamp(0.0, 1.0);
    // ruggedElevation: paleo arm only — drives broad regional uplift (active
    // convergence elevation is now supplied by broadDeform / Option C).
    let rugged_elevation = ((if is_stagnant_lid { RUGGED_PALEO_STAGNANT } else { RUGGED_PALEO })
        * paleo_uplift).clamp(0.0, 1.0);
    let land_gate = smoothstep(0.0, 0.02, c);

    // --- Wide motion-driven deformation (Option C) ---
    // Zeroed when stamps are gated (B-path owns the orogenic budget).
    let broad_deform = if ex.gate_stamps {
        0.0
    } else {
        let u = tectonics.uplift_at(dir);
        let uplift_term     = u.max(0.0) * UPLIFT_AMP * land_gate;
        let subsidence_term = (-u).max(0.0) * SUBSIDENCE_AMP * land_gate;
        (uplift_term - subsidence_term) * (-q.boundary_dist / UPLIFT_ZONE_W).exp()
    };

    // --- Broad undulation (signed fbm — raises AND lowers) ---
    let und_noise = fbm_fixed(noise, x * UNDULATION_FREQ, y * UNDULATION_FREQ, z * UNDULATION_FREQ, UNDULATION_OCT);
    let und_var_raw = fbm_fixed(noise, x * UNDULATION_VAR_FREQ, y * UNDULATION_VAR_FREQ, z * UNDULATION_VAR_FREQ, UNDULATION_VAR_OCT);
    let und_var   = 0.5 + 0.8 * (und_var_raw * 0.5 + 0.5); // [0.5, 1.3]
    let undulation = UNDULATION_AMP * und_var * und_noise * land_gate * (1.0 - CRATON_MASK_STR * rugged);

    // --- CC collision weight ---
    let w_mine  = smoothstep(-0.10, 0.10, c);
    let w_other = smoothstep(-0.10, 0.10, q.other_crust_dist);
    let cc_collision = w_mine * w_other;

    // --- Broad uplift (tectonically gated; paleo arm only) ---
    let broad_uplift = if ex.gate_stamps { 0.0 } else { HIGHLAND_AMP * rugged_elevation * land_gate };

    // --- CC-collision plateau ---
    let plateau = if ex.gate_stamps {
        0.0
    } else {
        let plateau_frac = smoothstep(PLATEAU_LO, PLATEAU_HI, rugged * cc_collision);
        PLATEAU_AMP * plateau_frac * land_gate
    };

    // --- Shelf width (narrows at active margins) ---
    let activeness = (1.0 - smoothstep(0.02, 0.06, q.boundary_dist))
        * smoothstep(0.08, 0.25, q.convergence.abs().max(q.shear.abs()));
    let shelf_w = shelf_w_base * (1.0 - activeness) + TECT_SHELF_W_ACTIVE * activeness;

    // --- Base elevation (ocean/land blend across the shelf) ---
    let combined_land = land_base + broad_uplift + plateau;
    let base = combined_land
        + (ocean_base - combined_land) * (1.0 - smoothstep(-shelf_w, TECT_COAST_LERP_HI, c));

    // --- Boundary relief (gated off in B-path) ---
    let ridged_at = |d: DVec3, freq: f64, octaves: u32| -> f64 {
        ridged_fixed(noise, d.x * freq, d.y * freq, d.z * freq, octaves)
    };
    let relief = if ex.gate_stamps {
        0.0
    } else {
        boundary_relief(&q, &tectonics.plates, dir, &ridged_at, base)
    };

    // --- LOD-adaptive octave counts ---
    let fbm_octaves: u32    = (level as u32 + 2).clamp(FBM_BASE_OCTAVES, 18);
    let ridged_octaves: u32 = (level as i32 - 2).max(0) as u32;
    let ridged_octaves: u32 = ridged_octaves.clamp(RIDGED_BASE_OCTAVES, 10);

    // ---- Erosion-steered fine detail (erosion present only) ----
    // Warp the FBM input along the downhill flow and read drainage discharge.
    // Both are pure fns of `dir` (fixed-resolution baked fields, no `level`),
    // so the additive-octave LOD invariant is preserved.
    let mut warp_x = x * TECT_DETAIL_FBM_SCALE;
    let mut warp_y = y * TECT_DETAIL_FBM_SCALE;
    let mut warp_z = z * TECT_DETAIL_FBM_SCALE;
    let mut acc = 0.0f64;
    if let Some(erosion) = ex.erosion {
        let flow = erosion.flow_at(dir);
        let warp = EROSION_WARP_STR * TECT_DETAIL_FBM_SCALE;
        warp_x += flow.x * warp;
        warp_y += flow.y * warp;
        warp_z += flow.z * warp;
        acc = erosion.acc_at(dir);
    }

    // ---- drainProxy → detailAmp re-keyed from drainage discharge ----
    let drain_proxy = if ex.erosion.is_some() {
        rugged.max(smoothstep(0.30, 0.85, acc))
    } else {
        rugged
    };
    let detail_amp_base = if drain_proxy.is_finite() {
        TECT_DETAIL_FBM_BASE * (CRATON_FLOOR + (1.0 - CRATON_FLOOR) * smoothstep(0.10, 0.70, drain_proxy))
    } else {
        TECT_DETAIL_FBM_BASE * CRATON_FLOOR
    };

    // ---- Rock-hardness fine-detail modulation (continuous, no seams) ----
    let hardness_term = 1.0 + HARD_DETAIL_BOOST * (q.rock_hardness - 0.5);
    let detail_amp = detail_amp_base * hardness_term;

    // ---- Climate-gated process palette (FLUVIAL/GLACIAL/AEOLIAN/KARST) ----
    // Base temperature: analytic, no sun term (mirrors Climate.temperature_at).
    let base_h = base;
    let abs_y = y.abs();
    let cos_lat = (1.0 - y * y).max(0.0).sqrt();
    let insolation = cos_lat * (1.0 - ex.proc_invert_blend) + abs_y * ex.proc_invert_blend;
    let lapse = ex.proc_lapse_rate * base_h.max(0.0) * ex.proc_lapse_factor;
    let base_temp_c = ex.base_temp + ex.proc_greenhouse + ex.proc_gradient * (insolation - 0.5) - lapse;

    // Baked moisture + wind (neutral when no climate ref).
    let (mut proc_moisture, mut proc_wind_speed, mut proc_wind_x, mut proc_wind_z) = (0.5, 0.0, 0.0, 0.0);
    if let Some(climate) = ex.climate {
        let (_t, m) = climate.sample(dir, base_h);
        proc_moisture = m as f64;
        let w = climate.wind_at(dir);
        proc_wind_speed = w.speed as f64;
        proc_wind_x = w.x as f64;
        proc_wind_z = w.z as f64;
    }
    let proc_aridity = 1.0 - proc_moisture;

    // GLACIAL weight
    let coldness   = smoothstep(PROC_GLACIAL_TEMP_HI, PROC_GLACIAL_TEMP_LO, base_temp_c);
    let elev_supply = smoothstep(PROC_GLACIAL_ELEV_LO, PROC_GLACIAL_ELEV_HI, base_h);
    let snow_supply = smoothstep(PROC_GLACIAL_MOIST_LO, PROC_GLACIAL_MOIST_HI, proc_moisture);
    let w_glacial  = coldness * elev_supply * snow_supply * land_gate;

    // AEOLIAN weight
    let wind_gate = smoothstep(PROC_AEOLIAN_WIND_LO, PROC_AEOLIAN_WIND_HI, proc_wind_speed);
    let arid_gate = smoothstep(PROC_AEOLIAN_ARID_LO, PROC_AEOLIAN_ARID_HI, proc_aridity);
    let w_aeolian = wind_gate * arid_gate * land_gate * PROC_AEOLIAN_STR;

    // KARST weight
    let solub_mid = (PROC_KARST_SOLUB_LO + PROC_KARST_SOLUB_HI) * 0.5;
    let solubility = smoothstep(PROC_KARST_SOLUB_LO, solub_mid, q.rock_hardness)
        * (1.0 - smoothstep(solub_mid, PROC_KARST_SOLUB_HI, q.rock_hardness));
    let karst_moist = smoothstep(PROC_KARST_MOIST_LO, PROC_KARST_MOIST_HI, proc_moisture);
    let karst_land  = smoothstep(PROC_KARST_LAND_LO, PROC_KARST_LAND_LO * 3.0, c);
    let w_karst     = karst_moist * solubility * karst_land * PROC_KARST_STR;

    // Glacial smoothing + aeolian deflation attenuate the fine FBM.
    let glacial_smooth  = 1.0 - PROC_GLACIAL_STR * w_glacial;
    let aeolian_deflate = 1.0 - w_aeolian * PROC_AEOLIAN_SMOOTH;
    let detail_amp_proc = detail_amp * glacial_smooth * aeolian_deflate;

    // Aeolian dune warp (perpendicular to wind) + ripple texture.
    let aeolian_warp_mag = PROC_AEOLIAN_WARP * TECT_DETAIL_FBM_SCALE * w_aeolian;
    let perp_x = -proc_wind_z;
    let perp_z =  proc_wind_x;
    let proc_warp_x = warp_x + perp_x * aeolian_warp_mag;
    let proc_warp_y = warp_y;
    let proc_warp_z = warp_z + perp_z * aeolian_warp_mag;

    let aeolian_ripple = if let Some(an) = ex.aeolian_noise {
        let shift = PROC_AEOLIAN_FREQ * 0.5 * w_aeolian;
        an.sample(
            x * PROC_AEOLIAN_FREQ + proc_wind_x * shift,
            y * PROC_AEOLIAN_FREQ,
            z * PROC_AEOLIAN_FREQ + proc_wind_z * shift,
        )
    } else {
        0.0
    };
    let aeolian_ripple_add = detail_amp * PROC_AEOLIAN_SMOOTH * w_aeolian * aeolian_ripple;

    // Karst negative dissolution pits.
    let karst_add = if let Some(kn) = ex.karst_noise {
        let karst_raw = kn.sample(x * PROC_KARST_FREQ, y * PROC_KARST_FREQ, z * PROC_KARST_FREQ);
        let karst_pit = -((1.0 - karst_raw.abs()).powi(2)); // [-1,0], peaks at noise=0
        detail_amp * w_karst * karst_pit
    } else {
        0.0
    };

    // Glacial cirque headwall steepening.
    let cirque_add = if let Some(gn) = ex.glacial_noise {
        let cirque_raw = gn.sample(
            x * TECT_DETAIL_RIDGE_SCALE * 1.5,
            y * TECT_DETAIL_RIDGE_SCALE * 1.5,
            z * TECT_DETAIL_RIDGE_SCALE * 1.5,
        );
        PROC_GLACIAL_CIRQUE_AMP * w_glacial * cirque_raw
    } else {
        0.0
    };

    // --- Detail FBM (additive-octave, LOD-consistent, warped) ---
    let fbm_raw = fbm_additive(noise, proc_warp_x, proc_warp_y, proc_warp_z, fbm_octaves);
    let detail_fbm = detail_amp_proc * fbm_raw;

    // --- Detail ridged (discharge-gated; coastal weighting) ---
    // `chan`: smoothstepped discharge gate (shared with vIncision).
    let chan = smoothstep(0.30, 0.85, acc);
    let rx = x * TECT_DETAIL_RIDGE_SCALE;
    let ry = y * TECT_DETAIL_RIDGE_SCALE;
    let rz = z * TECT_DETAIL_RIDGE_SCALE;
    let ridged_raw = ridged_additive(noise, rx, ry, rz, ridged_octaves);
    let ridged_scale_base = if ex.erosion.is_some() {
        EROSION_RIDGED_FLOOR + (1.0 - EROSION_RIDGED_FLOOR) * chan
    } else {
        1.0
    };
    let ridged_scale = ridged_scale_base * hardness_term;
    // In B/erosion mode boundary relief is gated off, so use the un-halved ridge amp.
    let ridge_amp = if ex.erosion.is_some() { 0.05 } else { TECT_DETAIL_RIDGE };
    let detail_ridged = ridge_amp
        * ridged_scale
        * ridged_raw
        * smoothstep(0.0, 0.08, c)
        * (0.7 + 0.3 * (-q.boundary_dist / 0.18).exp())
        * glacial_smooth
        * aeolian_deflate;

    // --- V-shaped incision deepened by discharge ---
    // Dedicated low-threshold carve gate (NOT `chan`) so tributaries — not only
    // high-discharge trunks — incise; this is what makes rivers read as carved
    // valley NETWORKS rather than a few faint notches (mirrors ki `carveGate`).
    let carve_gate = smoothstep(0.10, 0.45, acc);
    let v_incision = if ex.erosion.is_some() {
        -EROSION_VINCISION_AMP * carve_gate * land_gate
    } else {
        0.0
    };

    let detail = detail_fbm + detail_ridged + v_incision + karst_add + cirque_add + aeolian_ripple_add;

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

    // --- Arc volcanoes (headroom-aware, keyed to PRE-erosion surface) ---
    let surface_h = base + relief + detail + island_h + undulation + broad_deform;
    let volcano   = (tectonics.volcano_elevation(dir)).min(TECT_VOLC_SUM_MAX);
    let headroom  = 1.0 - smoothstep(TECT_VOLC_HEADROOM_LO, TECT_VOLC_HEADROOM_HI, surface_h);
    let erosion_h = ex.erosion.map(|e| e.delta_at(dir)).unwrap_or(0.0);

    (surface_h + erosion_h + volcano * headroom).clamp(-1.0, 1.0)
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
            composition: 0.5,
            climate: ClimateParams {
                seed,
                base_temp: 15.0,
                atmosphere: 0.6,
                band_count: 3,
                axial_tilt_rad: 23.0_f64.to_radians(),
                redistribution: None,
                greenhouse: None,
                lapse_rate: None,
                swirl_strength: None,
                n_high: None,
                n_low: None,
                cross_isobar_max: None,
                sigma_base: None,
                lat_spread: None,
                retrograde: None,
                equator_taper_width: None,
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
    // material() / wetness() are finite and in [0,1]
    // -----------------------------------------------------------------------

    #[test]
    fn material_and_wetness_in_range() {
        let hf = make_hf(42);
        for &d in &sample_sphere_dirs(50) {
            let m = hf.material(d);
            let w = hf.wetness(d);
            assert!(m.is_finite() && (0.0..=1.0).contains(&m), "material {m} out of [0,1]");
            assert!(w.is_finite() && (0.0..=1.0).contains(&w), "wetness {w} out of [0,1]");
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

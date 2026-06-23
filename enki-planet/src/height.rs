//! Height-field trait — the single abstraction between terrain data and meshing.

use glam::DVec3;

use crate::climate::WindSample;

/// Provides elevation and biome data for any point on the planet surface.
///
/// Implementors must be `Send + Sync` so they can be shared across worker threads.
pub trait HeightField: Send + Sync {
    /// Signed height offset from the base radius along `dir` (unit vector).
    fn height(&self, dir: DVec3, level: u8) -> f64;

    /// RGB tectonic-plate color for the given surface direction.
    fn plate_color(&self, _dir: DVec3) -> [f32; 3] {
        [0.5, 0.5, 0.5]
    }

    /// `(temperature_celsius, precipitation_0_1)` climate tuple.
    fn climate(&self, _dir: DVec3, _height: f64) -> (f32, f32) {
        (15.0, 0.5)
    }

    /// Rock hardness 0..1 (soft→hard) — debug "material" view. Default 0.0.
    fn material(&self, _dir: DVec3) -> f32 {
        0.0
    }

    /// Surface wetness 0..1 (open water: rivers ∪ lakes) — debug "wetness" view. Default 0.0.
    fn wetness(&self, _dir: DVec3) -> f32 {
        0.0
    }

    /// Volcano-cone influence 0..1 (arc + hotspot) — debug "volcano" view. Default 0.0.
    fn volcanism(&self, _dir: DVec3) -> f32 {
        0.0
    }

    /// Baked wind speed 0..1 along `dir` (drives ocean wave height/intensity).
    /// Default 0.5 (neutral) for height fields without a climate bake.
    fn wind_speed_at(&self, _dir: DVec3) -> f32 {
        0.5
    }

    /// Baked wind velocity sample (tangent-plane direction + speed) along `dir`.
    /// Drives the wind-streakline overlay. Default neutral (zero) — no flow shown
    /// for height fields without a climate bake.
    fn wind_at(&self, _dir: DVec3) -> WindSample {
        WindSample::default()
    }

    /// Baked atmospheric moisture 0..1 along `dir` (humidity — drives cloud
    /// placement). NOTE distinct from [`Self::wetness`] (surface rivers/lakes).
    /// Default 0.5 (neutral) for height fields without a climate bake.
    fn moisture(&self, _dir: DVec3) -> f32 {
        0.5
    }

    /// Log-normalized drainage discharge 0..1 at `dir` (river-network tracer).
    /// Default 0 (no erosion field). See `erosion::Erosion::acc_at`.
    fn flow_accum_at(&self, _dir: DVec3) -> f32 {
        0.0
    }

    /// UNblurred log-discharge 0..1 at `dir` — the river-network EXTRACTION reads
    /// this (the blurred `flow_accum_at` smears channel peaks into fat gradients).
    /// Default 0. See `erosion::Erosion::acc_sharp_at`.
    fn flow_accum_sharp_at(&self, _dir: DVec3) -> f32 {
        0.0
    }

    /// World-space downhill flow tangent at `dir` (magnitude = coherence signal,
    /// NOT unit). Default zero. See `erosion::Erosion::flow_at`.
    fn flow_dir_at(&self, _dir: DVec3) -> DVec3 {
        DVec3::ZERO
    }

    /// UNwarped downhill flow tangent at `dir` — the river-network ROUTING reads
    /// this (the domain warp in `flow_dir_at` misaligns routing from the unwarped
    /// channel raster). Default zero. See `erosion::Erosion::flow_dir_raw_at`.
    fn flow_dir_sharp_at(&self, _dir: DVec3) -> DVec3 {
        DVec3::ZERO
    }

    /// Lake-basin presence mask 0..1 at `dir` (river tracer terminates in lakes).
    /// Default 0. See `erosion::Erosion::lake_mask_at`.
    fn lake_mask_at(&self, _dir: DVec3) -> f32 {
        0.0
    }
}

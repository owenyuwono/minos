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

    /// Surface wetness 0..1 (open water) — debug "wetness" view. Default 0.0. No source
    /// is wired post-river-removal; the Nanite bake still carries it as a vertex channel.
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

    /// Live "peak sharpness" control 0..1 (0 = generator default, 1 = cusped/pointy
    /// mountain crests). No-op by default — only height fields with a tunable detail
    /// ridge (e.g. `TectonicHeightField`) act on it. Read lock-free per mesh sample, so
    /// the caller remeshes after changing it.
    fn set_peak_sharpness(&self, _s: f64) {}

    /// Live "ground roughness" control 0..1 (0 = generator default smooth, 1 = strong
    /// human-scale surface relief). No-op by default — only height fields with a tunable
    /// detail spectrum (e.g. `TectonicHeightField`) act on it. Read lock-free per mesh
    /// sample, so the caller remeshes after changing it.
    fn set_ground_roughness(&self, _s: f64) {}

    /// Baked atmospheric moisture 0..1 along `dir` (humidity — drives cloud
    /// placement). NOTE distinct from [`Self::wetness`] (surface rivers/lakes).
    /// Default 0.5 (neutral) for height fields without a climate bake.
    fn moisture(&self, _dir: DVec3) -> f32 {
        0.5
    }
}

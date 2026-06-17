//! Height-field trait — the single abstraction between terrain data and meshing.

use glam::DVec3;

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
}

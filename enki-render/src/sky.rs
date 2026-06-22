//! `sky` — solar clock that drives the sun direction (day/night + seasons) and the
//! sun-body render parameters.
//!
//! The focused planet renders at the world origin, so "orbiting the sun" is modelled
//! as the sun DIRECTION sweeping in the planet's body frame. The orbit is kept in
//! true-scale physical units (real day length, year, axial tilt) so day/night and
//! seasons are real, but the renderer only ever consumes the (scale-free) sun
//! direction + angular size — the "planetarium dome" trick from the solar-system plan
//! keeps everything inside f32 range.

use glam::{DVec3, Vec3};
use std::f64::consts::TAU;

// ── True-scale constants (SI) ────────────────────────────────────────────────

/// 1 astronomical unit (m) — Earth's mean orbital radius.
pub const AU_M: f64 = 1.495_978_707e11;
/// Sun photosphere radius (m).
pub const SUN_RADIUS_M: f64 = 6.957e8;
/// Mean solar day (s).
pub const EARTH_DAY_S: f64 = 86_400.0;
/// Sidereal year (s).
pub const EARTH_YEAR_S: f64 = 365.256_363 * EARTH_DAY_S;
/// Earth's axial tilt (rad) = 23.4393°.
pub const EARTH_TILT_RAD: f64 = 0.409_092_6;

/// Distance to the render "sky shell" (m). The sun (and later other bodies) are
/// drawn here, sized by their true angular radius. MUST stay inside `Camera.far`
/// (750 km) and beyond peak terrain relief (~120 km) so reversed-Z occludes the
/// sun behind the planet limb for free. f32 never sees a coordinate larger than this.
pub const SKY_SHELL_M: f64 = 4.0e5;

/// Solar clock for the focused planet: a true-scale orbit + spin producing the sun
/// direction (lighting) and the sun's angular size (rendering).
#[derive(Debug, Clone)]
pub struct SkyModel {
    /// Simulated time (s).
    pub t_seconds: f64,
    /// real-seconds → sim-seconds multiplier (GUI-tunable). 1.0 = real time.
    pub time_scale: f64,
    pub paused: bool,
    /// Solar day length (s) — the sun's apparent daily period.
    pub day_len_s: f64,
    /// Orbital year length (s).
    pub year_len_s: f64,
    /// Axial tilt (rad) — drives the seasonal solar declination.
    pub axial_tilt_rad: f64,
    /// Orbit phase offset (rad) — sets the starting season.
    pub orbit_phase: f64,
    /// Spin phase offset (rad) — sets the starting time-of-day.
    pub spin_phase: f64,
}

impl SkyModel {
    /// True-scale Earth defaults. `time_scale` defaults so day/night is visible
    /// (~one day per 30 s of real time); the GUI exposes it as a slider.
    pub fn earth() -> Self {
        Self {
            t_seconds: 0.0,
            time_scale: EARTH_DAY_S / 30.0, // ≈ 1 day / 30 s real
            paused: false,
            day_len_s: EARTH_DAY_S,
            year_len_s: EARTH_YEAR_S,
            axial_tilt_rad: EARTH_TILT_RAD,
            orbit_phase: 0.0,
            spin_phase: 0.0,
        }
    }

    /// Advance the clock by one real-time frame.
    pub fn tick(&mut self, dt: f32) {
        if !self.paused {
            self.t_seconds += dt as f64 * self.time_scale;
        }
    }

    /// Sun direction in the planet's body frame (unit; surface → sun, i.e. the
    /// lighting direction). Body frame: +Y = north pole, XZ = equatorial plane. The
    /// subsolar point sits at latitude = declination (seasonal) and longitude = hour
    /// angle (daily), so day/night and seasons both fall out.
    pub fn sun_dir(&self) -> DVec3 {
        let decl = self.axial_tilt_rad
            * (TAU * self.t_seconds / self.year_len_s + self.orbit_phase).sin();
        let hour = TAU * self.t_seconds / self.day_len_s + self.spin_phase;
        let (sd, cd) = decl.sin_cos();
        let (sh, ch) = hour.sin_cos();
        DVec3::new(cd * ch, sd, cd * sh)
    }

    /// Warm-white directional-light color (premultiplied intensity). The sun DISK
    /// uses its own HDR over-drive in the shader; this is the surface lighting color.
    pub fn sun_light_color(&self) -> Vec3 {
        Vec3::new(1.0, 0.96, 0.90) * 1.3
    }

    /// Heliocentric spin axis (north pole), tilted from the orbital normal (+Y) by the
    /// axial tilt. Fixed in the heliocentric frame, so the sub-solar latitude migrates
    /// over the orbit → seasons.
    pub fn spin_axis(&self) -> DVec3 {
        DVec3::new(self.axial_tilt_rad.sin(), self.axial_tilt_rad.cos(), 0.0).normalize()
    }

    /// Rotation taking a heliocentric direction into the focused planet's body-fixed
    /// (rendered) frame. The planet spins +φ each day, so the sky + sun appear to rotate
    /// −φ around the spin axis (sunrise → sunset). Apply to every rendered body
    /// direction AND every light direction so the sky and lighting stay consistent.
    pub fn helio_to_body(&self) -> glam::Quat {
        let phi = (TAU * self.t_seconds / self.day_len_s) as f32;
        glam::Quat::from_axis_angle(self.spin_axis().as_vec3(), -phi)
    }
}

impl Default for SkyModel {
    fn default() -> Self {
        Self::earth()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_dir_is_unit_and_bounded_by_tilt() {
        let mut s = SkyModel::earth();
        for i in 0..1000 {
            s.t_seconds = i as f64 * (EARTH_YEAR_S / 1000.0) + 123.0;
            let d = s.sun_dir();
            assert!((d.length() - 1.0).abs() < 1e-12, "sun_dir must be unit");
            // |latitude| of the subsolar point never exceeds the axial tilt.
            assert!(d.y.abs() <= EARTH_TILT_RAD.sin() + 1e-9, "declination exceeds tilt");
        }
    }

    #[test]
    fn day_night_sweeps_over_a_day() {
        let mut s = SkyModel::earth();
        s.t_seconds = 0.0; // hour 0, decl ~0 → noon at +X longitude
        assert!((s.sun_dir() - DVec3::X).length() < 1e-9);
        s.t_seconds = EARTH_DAY_S / 4.0; // +90° around the equator
        // ~Z, modulo a tiny seasonal declination drift over the quarter day.
        assert!((s.sun_dir() - DVec3::Z).length() < 5e-3, "got {:?}", s.sun_dir());
        s.t_seconds = EARTH_DAY_S / 2.0; // opposite side → night at +X longitude
        assert!(s.sun_dir().x < -0.99, "got {:?}", s.sun_dir());
    }

    #[test]
    fn seasons_from_tilt() {
        let mut s = SkyModel::earth();
        s.t_seconds = EARTH_YEAR_S / 4.0; // solstice → declination peaks at +tilt
        assert!((s.sun_dir().y - EARTH_TILT_RAD.sin()).abs() < 1e-3);
    }

    #[test]
    fn pause_freezes_the_clock() {
        let mut s = SkyModel::earth();
        s.paused = true;
        s.tick(1.0);
        assert_eq!(s.t_seconds, 0.0);
    }
}

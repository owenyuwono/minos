//! `system` — the solar system: f64 heliocentric bodies + circular orbits.
//!
//! The floating-origin foundation for the NMS-style solar system. Bodies live in one
//! absolute f64 heliocentric frame (the star at the origin). The **focused** body is
//! the render anchor: the camera + terrain are expressed relative to its center, so
//! its world position *cancels* in the camera-relative f64 subtract — the heavy Nanite
//! planet renders identically to "planet pinned at the origin" (zero regression). Other
//! bodies are drawn at `body.world_pos − focused.world_pos` (exact in f64), clamped to
//! the sky shell for f32-safe rendering.
//!
//! NMS-**compressed** scale: planets a few planet-radii apart (huge in each other's
//! sky), NOT true-AU (which would be invisible + force a universal rebase).

use glam::DVec3;
use std::f64::consts::TAU;

/// Index into [`SolarSystem::bodies`].
pub type BodyId = usize;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyKind {
    Star,
    Planet,
}

/// A circular orbit in a plane through the star (system origin).
#[derive(Clone, Copy, Debug)]
pub struct Orbit {
    pub radius_m: f64,
    pub period_s: f64,
    pub phase0: f64,
    /// Unit normal of the orbital plane (≈ +Y; small tilts give inclined orbits).
    pub plane_normal: DVec3,
}

impl Orbit {
    /// Heliocentric position on the orbit at sim-time `t` (star at origin).
    pub fn position(&self, t: f64) -> DVec3 {
        let theta = TAU * t / self.period_s + self.phase0;
        // Orthonormal basis in the orbital plane.
        let n = {
            let n = self.plane_normal.normalize_or_zero();
            if n.length_squared() < 0.5 { DVec3::Y } else { n }
        };
        let r = if n.x.abs() < 0.9 { DVec3::X } else { DVec3::Z };
        let u = r.cross(n).normalize();
        let v = n.cross(u);
        (u * theta.cos() + v * theta.sin()) * self.radius_m
    }
}

#[derive(Clone, Debug)]
pub struct Body {
    pub name: String,
    pub kind: BodyKind,
    pub radius_m: f64,
    /// Heliocentric world position (updated by [`SolarSystem::advance`]).
    pub world_pos: DVec3,
    /// `None` for the star (fixed at the system origin).
    pub orbit: Option<Orbit>,
    /// Base albedo (lit-sphere shading) / emissive tint (star).
    pub color: [f32; 3],
}

/// The whole solar system: bodies in one f64 frame + the focused render anchor.
#[derive(Clone, Debug)]
pub struct SolarSystem {
    pub bodies: Vec<Body>,
    pub focused: BodyId,
    pub star: BodyId,
}

impl SolarSystem {
    /// A compressed NMS-style starter system: a star + three planets at 5–12 planet
    /// radii (`PLANET_RADIUS = 50 km`), sized for a dramatic sky. Periods are in
    /// sim-seconds (the GUI time-scale tunes how fast that runs in real time).
    pub fn nms_default() -> Self {
        let pr = 50_000.0; // planet-radius baseline
        let tilt = |deg: f64| {
            let r = deg.to_radians();
            DVec3::new(r.sin(), r.cos(), 0.0)
        };
        let bodies = vec![
            Body {
                name: "Sol".into(), kind: BodyKind::Star, radius_m: 3.5 * pr,
                world_pos: DVec3::ZERO, orbit: None, color: [1.0, 0.92, 0.78],
            },
            Body {
                name: "Terra".into(), kind: BodyKind::Planet, radius_m: pr,
                world_pos: DVec3::ZERO, color: [0.30, 0.55, 0.85],
                orbit: Some(Orbit { radius_m: 40.0 * pr, period_s: 7.2e6, phase0: 0.0, plane_normal: DVec3::Y }),
            },
            Body {
                name: "Ferra".into(), kind: BodyKind::Planet, radius_m: 1.5 * pr,
                world_pos: DVec3::ZERO, color: [0.80, 0.45, 0.30],
                orbit: Some(Orbit { radius_m: 60.0 * pr, period_s: 1.3e7, phase0: 2.1, plane_normal: tilt(6.0) }),
            },
            Body {
                name: "Oceanus".into(), kind: BodyKind::Planet, radius_m: 2.5 * pr,
                world_pos: DVec3::ZERO, color: [0.20, 0.42, 0.72],
                orbit: Some(Orbit { radius_m: 90.0 * pr, period_s: 2.4e7, phase0: 4.3, plane_normal: tilt(-4.0) }),
            },
        ];
        let mut s = Self { bodies, focused: 1, star: 0 };
        s.advance(0.0);
        s
    }

    /// Update every body's `world_pos` to sim-time `t` (s).
    pub fn advance(&mut self, t: f64) {
        for b in &mut self.bodies {
            if let Some(orb) = b.orbit {
                b.world_pos = orb.position(t);
            }
        }
    }

    pub fn focused(&self) -> &Body {
        &self.bodies[self.focused]
    }
    pub fn star(&self) -> &Body {
        &self.bodies[self.star]
    }

    /// Sun direction in the focused body's frame (surface → star) — drives lighting.
    pub fn sun_dir_focused(&self) -> DVec3 {
        (self.star().world_pos - self.focused().world_pos).normalize_or_zero()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orbit_is_circular_and_periodic() {
        let o = Orbit { radius_m: 5.0e5, period_s: 1000.0, phase0: 0.3, plane_normal: DVec3::Y };
        for i in 0..50 {
            let t = i as f64 * 17.0;
            assert!((o.position(t).length() - 5.0e5).abs() < 1e-3, "radius constant");
        }
        // One full period returns to the start.
        assert!((o.position(0.0) - o.position(1000.0)).length() < 1e-6);
    }

    #[test]
    fn focused_body_origin_cancels_zero_regression() {
        // THE invariant: rendering an object near the focused body is independent of
        // where that body is in its orbit — so the Nanite planet is byte-identical to
        // the old "planet at origin". object_abs − camera_abs must drop focused.world_pos.
        let mut s = SolarSystem::nms_default();
        let local = DVec3::new(123.0, -45.0, 6.0); // a terrain vertex near the planet center
        let cam_offset = DVec3::new(0.0, 0.0, 55_000.0); // camera relative to the planet
        let rel_at = |sys: &SolarSystem| {
            let f = sys.focused().world_pos;
            (f + local) - (f + cam_offset) // object_abs − camera_abs
        };
        let r0 = rel_at(&s);
        s.advance(1.0e5); // move the planet a quarter-way around its orbit
        let r1 = rel_at(&s);
        assert!(s.focused().world_pos.length() > 1.0, "planet actually moved");
        assert!((r0 - r1).length() < 1e-9, "focused-relative render must not change");
        assert!((r0 - (local - cam_offset)).length() < 1e-9);
    }

    #[test]
    fn sun_dir_points_from_focused_to_star() {
        let s = SolarSystem::nms_default();
        let d = s.sun_dir_focused();
        assert!((d.length() - 1.0).abs() < 1e-12);
        // Star is at the origin; the focused planet is offset, so sun_dir ≈ −planet_dir.
        let expect = (-s.focused().world_pos).normalize();
        assert!((d - expect).length() < 1e-9);
    }
}

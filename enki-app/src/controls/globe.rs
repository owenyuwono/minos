//! `globe` — orbit camera around a spherical planet.
#![allow(dead_code)]
//!
//! `GlobeControls` represents the camera as spherical coordinates (radius,
//! azimuth θ, polar angle φ measured from north pole) around the planet
//! centre.  Drag rotates the orbit with inertia; scroll zooms exponentially.
//! `update(dt)` decays inertia each frame and blends zoom smoothly.
//!
//! All camera positions are in f64 for planet-scale precision. Angles and
//! angular velocities use f32 — they are fine for input deltas and damping.

use enki_render::camera::Camera;
use glam::{DVec3, Quat, Vec3};

// ── Constants ─────────────────────────────────────────────────────────────

/// Physical radius of the planet (m).  Used as the zero-altitude baseline.
pub const PLANET_RADIUS: f64 = 50_000.0;

/// Minimum altitude above the surface (m).
const MIN_ALTITUDE: f64 = 30.0;
/// Maximum altitude above the surface (m).
const MAX_ALTITUDE: f64 = PLANET_RADIUS * 8.0;

/// Angular damping factor applied per second.  0.0 = no friction, 1.0 = instant stop.
const INERTIA_DAMPING: f32 = 8.0;
/// Zoom blend rate (exponential smoothing coefficient, per second).
const ZOOM_SPEED: f32 = 6.0;
/// How much each scroll unit multiplies the current radius (log-space nudge).
const SCROLL_SENSITIVITY: f32 = 0.15;
/// Radians per pixel of drag.
const DRAG_SENSITIVITY: f32 = 0.005;
/// Floor on the altitude/radius pan-speed factor. Near the surface altitude/radius → 0
/// (radius is dominated by the 50 km planet radius), which makes panning crawl; clamp it
/// so close-up pan stays usable for traversal. Raise toward 1.0 for faster close-up pan,
/// lower for slower. Orbit saturates the factor near ~0.89 regardless.
const PAN_MIN_ALT_FACTOR: f32 = 0.4;
/// Polar clamp — stay this many radians away from north/south poles.
const POLAR_CLAMP: f32 = 0.01_f32;

// ── GlobeControls ─────────────────────────────────────────────────────────

/// Orbit camera around the planet centre using spherical coordinates.
///
/// Coordinate conventions:
/// - φ (phi)   = polar angle from +Y (north) axis, range `[POLAR_CLAMP, π − POLAR_CLAMP]`
/// - θ (theta) = azimuth angle around +Y axis, unbounded (wraps naturally)
/// - radius    = distance from planet centre, clamped to `[PLANET_RADIUS + MIN_ALTITUDE, …]`
#[derive(Debug, Clone)]
pub struct GlobeControls {
    // Current spherical coordinates
    radius: f64,
    theta: f32, // azimuth
    phi: f32,   // polar

    // Inertia: angular velocities carried from the last drag
    vel_theta: f32,
    vel_phi: f32,

    // Smooth zoom: target radius (log space blend destination)
    target_radius: f64,

    // Set by on_drag each frame it is called; cleared by update after being read.
    dragged_this_frame: bool,
}

impl GlobeControls {
    /// Create controls with the camera at the given altitude directly above
    /// the north pole.
    pub fn new(altitude: f64) -> Self {
        let radius = (PLANET_RADIUS + altitude).clamp(
            PLANET_RADIUS + MIN_ALTITUDE,
            PLANET_RADIUS + MAX_ALTITUDE,
        );
        Self {
            radius,
            theta: 0.0,
            phi: std::f32::consts::FRAC_PI_4, // 45° from north pole
            vel_theta: 0.0,
            vel_phi: 0.0,
            target_radius: radius,
            dragged_this_frame: false,
        }
    }

    /// Drag input: rotate azimuth and polar angle.
    ///
    /// `dx` = horizontal pixel delta (positive = right), `dy` = vertical pixel
    /// delta (positive = down).  Both in screen pixels; sensitivity converts
    /// to radians.
    ///
    /// During a drag the angle tracks the input 1:1 (direct application).
    /// The velocity is SET (not accumulated) so that on release the coast
    /// starts at the last frame's drag speed without compounding.
    pub fn on_drag(&mut self, dx: f32, dy: f32) {
        // Scale rotation by altitude/radius so panning is slower near the surface (finer
        // control up close) and faster from orbit — same intent as the altitude-scaled
        // zoom. Floored at PAN_MIN_ALT_FACTOR so close-up pan stays usable (pure
        // altitude/radius → ~0 near the surface, which crawls).
        let alt_factor = (((self.radius - PLANET_RADIUS) / self.radius) as f32).max(PAN_MIN_ALT_FACTOR);
        let d_theta = -dx * DRAG_SENSITIVITY * alt_factor;
        let d_phi   = -dy * DRAG_SENSITIVITY * alt_factor; // negated so dragging up tilts the view up (natural)
        // Apply delta directly — 1:1 tracking during drag.
        self.theta += d_theta;
        self.phi = (self.phi + d_phi).clamp(POLAR_CLAMP, std::f32::consts::PI - POLAR_CLAMP);
        // Seed the coast velocity at the latest per-frame speed (set, not accumulate).
        self.vel_theta = d_theta;
        self.vel_phi   = d_phi;
        self.dragged_this_frame = true;
    }

    /// Scroll input: zoom in/out geometrically toward the planet.
    ///
    /// Positive `delta` zooms in (decreases radius), negative zooms out. The step is
    /// proportional to ALTITUDE (distance to the surface), not radius (distance to the
    /// centre): each notch changes altitude by a fixed fraction, so zoom slows as you
    /// approach the surface (fine control up close) and stays fast from orbit — and it
    /// can't overshoot through the surface. Altitude is ≥ MIN_ALTITUDE by the clamp, so
    /// the step never collapses to zero.
    pub fn on_scroll(&mut self, delta: f32) {
        let altitude = (self.target_radius - PLANET_RADIUS) as f32;
        let step = delta * SCROLL_SENSITIVITY * altitude;
        self.target_radius = (self.target_radius - step as f64).clamp(
            PLANET_RADIUS + MIN_ALTITUDE,
            PLANET_RADIUS + MAX_ALTITUDE,
        );
    }

    /// Advance simulation by `dt` seconds: apply inertial coast (only when not
    /// dragging this frame) and blend zoom.
    ///
    /// When `on_drag` was called this frame the angles already track 1:1; coast
    /// is skipped so there is no compounding on top of direct tracking.
    pub fn update(&mut self, dt: f32) {
        if self.dragged_this_frame {
            // Drag handled the angle update directly — skip coast application.
            self.dragged_this_frame = false;
        } else {
            // No drag this frame: apply decaying coast.
            self.theta += self.vel_theta * dt;
            self.phi = (self.phi + self.vel_phi * dt)
                .clamp(POLAR_CLAMP, std::f32::consts::PI - POLAR_CLAMP);
            let decay = (-INERTIA_DAMPING * dt).exp();
            self.vel_theta *= decay;
            self.vel_phi *= decay;
        }

        // Exponential zoom blend toward target_radius
        let t = (ZOOM_SPEED * dt).min(1.0) as f64;
        self.radius = self.radius + (self.target_radius - self.radius) * t;
        self.radius = self.radius.clamp(
            PLANET_RADIUS + MIN_ALTITUDE,
            PLANET_RADIUS + MAX_ALTITUDE,
        );
    }

    /// Compute the camera state for the current spherical position.
    ///
    /// Position = spherical → Cartesian (planet-centred).
    /// Orientation = looking at planet centre, up = world +Y projected onto
    /// the tangent plane (falls back to +Z near poles).
    pub fn camera(&self) -> Camera {
        let pos = spherical_to_cartesian(self.radius, self.theta, self.phi);

        // Looking from pos toward the origin
        let forward = (-pos).normalize().as_vec3();

        // Choose a world-up hint; near poles, +Y collapses — use +Z instead.
        let world_up = if self.phi < 0.1 || self.phi > std::f32::consts::PI - 0.1 {
            Vec3::Z
        } else {
            Vec3::Y
        };

        // Build right-handed frame: right = forward × world_up (then re-derive up)
        let right = forward.cross(world_up).normalize();
        let up = right.cross(forward).normalize();

        // Quaternion from the rotation matrix columns
        let orientation = Quat::from_mat3(&glam::Mat3::from_cols(right, up, -forward));

        Camera {
            position: pos,
            orientation,
            fov_y_radians: 60_f32.to_radians(),
            near: 0.5,
            far: 750_000.0,
        }
    }

    /// Current radius (distance from planet centre, m).
    pub fn radius(&self) -> f64 {
        self.radius
    }

    /// Current altitude above the surface (m).
    pub fn altitude(&self) -> f64 {
        self.radius - PLANET_RADIUS
    }

    /// Polar angle φ in radians.
    pub fn phi(&self) -> f32 {
        self.phi
    }

    /// Azimuth angle θ in radians.
    pub fn theta(&self) -> f32 {
        self.theta
    }

    /// Drag sensitivity (radians per pixel).
    pub fn sensitivity(&self) -> f32 {
        DRAG_SENSITIVITY
    }
}

impl Default for GlobeControls {
    fn default() -> Self {
        Self::new(PLANET_RADIUS * 2.0)
    }
}

// ── Coordinate helpers ────────────────────────────────────────────────────

/// Convert spherical (r, θ, φ) to Cartesian (x, y, z) where φ is the polar
/// angle from +Y and θ is azimuth around +Y.
///
/// ```text
///   x = r · sin(φ) · sin(θ)
///   y = r · cos(φ)
///   z = r · sin(φ) · cos(θ)
/// ```
pub fn spherical_to_cartesian(r: f64, theta: f32, phi: f32) -> DVec3 {
    let sin_phi = phi.sin() as f64;
    let cos_phi = phi.cos() as f64;
    let sin_theta = theta.sin() as f64;
    let cos_theta = theta.cos() as f64;
    DVec3::new(
        r * sin_phi * sin_theta,
        r * cos_phi,
        r * sin_phi * cos_theta,
    )
}

/// Convert Cartesian to spherical (r, θ, φ).  Returns `(r, theta, phi)`.
pub fn cartesian_to_spherical(v: DVec3) -> (f64, f32, f32) {
    let r = v.length();
    if r < 1e-12 {
        return (0.0, 0.0, 0.0);
    }
    let phi = (v.y / r).clamp(-1.0, 1.0).acos() as f32;
    let theta = (v.x as f32).atan2(v.z as f32);
    (r, theta, phi)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    const EPS_F32: f32 = 1e-5;
    const EPS_F64: f64 = 1e-6;

    // ── Spherical ↔ Cartesian round-trips ────────────────────────────────

    #[test]
    fn spherical_to_cartesian_north_pole() {
        // phi = 0 → north pole (+Y direction)
        let v = spherical_to_cartesian(1.0, 0.0, 0.0);
        assert!((v.x).abs() < EPS_F64);
        assert!((v.y - 1.0).abs() < EPS_F64);
        assert!((v.z).abs() < EPS_F64);
    }

    #[test]
    fn spherical_to_cartesian_equator() {
        // phi = π/2, theta = 0 → (0, 0, r)
        let v = spherical_to_cartesian(1.0, 0.0, FRAC_PI_2);
        assert!((v.x).abs() < EPS_F64);
        assert!((v.y).abs() < EPS_F64);
        assert!((v.z - 1.0).abs() < EPS_F64);
    }

    #[test]
    fn round_trip_spherical() {
        // r round-trip precision is limited by f32 trig in cartesian_to_spherical;
        // allow 1e-3 relative error (f32 mantissa is ~7 digits).
        let r0 = PLANET_RADIUS + 1000.0;
        let theta0: f32 = 1.2;
        let phi0: f32 = 0.8;
        let cart = spherical_to_cartesian(r0, theta0, phi0);
        let (r1, theta1, phi1) = cartesian_to_spherical(cart);
        assert!((r1 - r0).abs() < r0 * 1e-5, "r: {r1} vs {r0}");
        assert!((theta1 - theta0).abs() < EPS_F32, "theta: {theta1} vs {theta0}");
        assert!((phi1 - phi0).abs() < EPS_F32, "phi: {phi1} vs {phi0}");
    }

    #[test]
    fn round_trip_arbitrary_quadrant() {
        // Allow 1e-5 relative error for r (f32 trig precision).
        let r0 = 200_000.0_f64;
        let theta0: f32 = -2.5;
        let phi0: f32 = 2.0;
        let cart = spherical_to_cartesian(r0, theta0, phi0);
        let (r1, theta1, phi1) = cartesian_to_spherical(cart);
        assert!((r1 - r0).abs() < r0 * 1e-5, "r: {r1} vs {r0}");
        assert!((theta1 - theta0).abs() < EPS_F32, "theta: {theta1} vs {theta0}");
        assert!((phi1 - phi0).abs() < EPS_F32, "phi: {phi1} vs {phi0}");
    }

    // ── Altitude clamping ─────────────────────────────────────────────────

    #[test]
    fn altitude_clamped_to_min() {
        let mut ctrl = GlobeControls::new(1.0);
        // Try to zoom way in
        ctrl.on_scroll(10_000.0);
        ctrl.update(10.0); // large dt to ensure convergence
        assert!(ctrl.altitude() >= MIN_ALTITUDE - 1.0,
            "altitude {} below min {}", ctrl.altitude(), MIN_ALTITUDE);
    }

    #[test]
    fn altitude_clamped_to_max() {
        let mut ctrl = GlobeControls::new(MAX_ALTITUDE - 1.0);
        // Try to zoom way out
        ctrl.on_scroll(-100_000.0);
        ctrl.update(10.0);
        assert!(ctrl.altitude() <= MAX_ALTITUDE + 1.0,
            "altitude {} above max {}", ctrl.altitude(), MAX_ALTITUDE);
    }

    #[test]
    fn zoom_step_scales_with_altitude() {
        // One scroll notch should move a far SMALLER absolute distance near the surface
        // than from orbit (geometric zoom → slow, fine control up close).
        let mut near = GlobeControls::new(1_000.0); // 1 km up
        let near_before = near.altitude();
        near.on_scroll(1.0);
        near.update(10.0); // large dt converges the zoom blend
        let near_step = (near_before - near.altitude()).abs();

        let mut far = GlobeControls::new(MAX_ALTITUDE); // orbit
        let far_before = far.altitude();
        far.on_scroll(1.0);
        far.update(10.0);
        let far_step = (far_before - far.altitude()).abs();

        assert!(far_step > near_step * 50.0,
            "zoom step should scale with altitude: near={near_step}, far={far_step}");
    }

    #[test]
    fn pan_step_scales_with_altitude() {
        // One drag should rotate FAR less near the surface than from orbit (slow, fine
        // pan up close) — the surface point tracks the cursor instead of whipping past.
        let mut near = GlobeControls::new(1_000.0);
        let near_before = near.theta();
        near.on_drag(50.0, 0.0);
        let near_step = (near_before - near.theta()).abs();

        let mut far = GlobeControls::new(MAX_ALTITUDE);
        let far_before = far.theta();
        far.on_drag(50.0, 0.0);
        let far_step = (far_before - far.theta()).abs();

        assert!(far_step > near_step * 1.5,
            "pan step should scale with altitude: near={near_step}, far={far_step}");
    }

    #[test]
    fn initial_altitude_within_bounds() {
        let ctrl = GlobeControls::new(500.0);
        assert!(ctrl.altitude() >= MIN_ALTITUDE);
        assert!(ctrl.altitude() <= MAX_ALTITUDE);
    }

    // ── Polar clamp ───────────────────────────────────────────────────────

    #[test]
    fn drag_does_not_flip_over_north_pole() {
        let mut ctrl = GlobeControls::new(10_000.0);
        ctrl.phi = 0.05; // near north pole
        // Drag strongly toward +Y (phi decreasing)
        for _ in 0..100 {
            ctrl.on_drag(0.0, -500.0);
        }
        assert!(ctrl.phi() >= POLAR_CLAMP,
            "phi {} below clamp {}", ctrl.phi(), POLAR_CLAMP);
    }

    #[test]
    fn drag_does_not_flip_over_south_pole() {
        let mut ctrl = GlobeControls::new(10_000.0);
        ctrl.phi = PI - 0.05; // near south pole
        for _ in 0..100 {
            ctrl.on_drag(0.0, 500.0);
        }
        assert!(ctrl.phi() <= PI - POLAR_CLAMP,
            "phi {} above clamp {}", ctrl.phi(), PI - POLAR_CLAMP);
    }

    // ── Camera output sanity ──────────────────────────────────────────────

    #[test]
    fn camera_is_finite() {
        let ctrl = GlobeControls::new(10_000.0);
        let cam = ctrl.camera();
        assert!(cam.position.is_finite());
        assert!(cam.orientation.is_finite());
    }

    #[test]
    fn camera_view_matrix_is_finite() {
        let ctrl = GlobeControls::new(10_000.0);
        let cam = ctrl.camera();
        let vm = cam.view_matrix();
        // All 16 elements must be finite
        for col in 0..4 {
            for row in 0..4 {
                assert!(vm.col(col)[row].is_finite(),
                    "view_matrix[{col}][{row}] is not finite");
            }
        }
    }

    #[test]
    fn camera_looks_roughly_at_origin() {
        let ctrl = GlobeControls::new(10_000.0);
        let cam = ctrl.camera();
        // Forward vector (−Z in camera space) should point from position toward origin
        let forward = cam.orientation * glam::Vec3::NEG_Z;
        let to_origin = (-cam.position).normalize().as_vec3();
        let dot = forward.dot(to_origin);
        assert!(dot > 0.99, "forward.dot(to_origin) = {dot}, expected > 0.99");
    }

    #[test]
    fn inertia_decays_to_zero() {
        let mut ctrl = GlobeControls::new(10_000.0);
        // Simulate: one drag frame sets velocity, then release (no more on_drag).
        ctrl.on_drag(100.0, 50.0);
        // First update clears dragged_this_frame; subsequent updates apply + decay coast.
        ctrl.update(0.016);
        ctrl.update(10.0); // large dt decays velocity to near zero
        assert!(ctrl.vel_theta.abs() < 1e-6, "vel_theta did not decay: {}", ctrl.vel_theta);
        assert!(ctrl.vel_phi.abs() < 1e-6, "vel_phi did not decay: {}", ctrl.vel_phi);
    }

    #[test]
    fn continuous_drag_tracking_and_coast() {
        let mut globe = GlobeControls::new(100_000.0);
        let dx = 5.0f32;
        let initial_theta = globe.theta();
        let alt_factor = (((globe.radius() - PLANET_RADIUS) / globe.radius()) as f32).max(PAN_MIN_ALT_FACTOR);
        let expected_step = dx * DRAG_SENSITIVITY * alt_factor; // radians per frame (altitude-scaled)

        // Phase 1: sustained drag for 10 frames — 1:1 tracking, monotonic, bounded.
        let mut prev_theta = initial_theta;
        for frame in 0..10 {
            globe.on_drag(dx, 0.0);
            globe.update(0.016);
            let t = globe.theta();
            assert!(
                t < prev_theta,
                "frame {frame}: theta {t} should decrease (positive dx → negative d_theta)"
            );
            let step = (prev_theta - t).abs();
            assert!(
                step > expected_step * 0.5 && step < expected_step * 1.5,
                "frame {frame}: step {step} out of [{}..{}]",
                expected_step * 0.5, expected_step * 1.5,
            );
            prev_theta = t;
        }

        // Phase 2: release — coast for 30 frames, same direction, decaying to ~0.
        let theta_at_release = prev_theta;
        let mut prev_increment = f32::MAX;
        for frame in 0..30 {
            globe.update(0.016);
            let t = globe.theta();
            assert!(
                t <= theta_at_release + 1e-6,
                "frame {frame}: theta {t} reversed past release point {theta_at_release}"
            );
            let increment = (prev_theta - t).abs();
            assert!(
                increment <= prev_increment + 1e-6,
                "coast should decay, frame {frame}: increment={increment} prev={prev_increment}"
            );
            prev_increment = increment;
            prev_theta = t;
        }
        // After 30 coast frames the motion should have nearly stopped.
        assert!(
            prev_increment < expected_step * 0.01,
            "coast did not decay to near-zero: last increment={prev_increment}"
        );
    }

    #[test]
    fn on_drag_applies_single_delta_not_accumulated_velocity() {
        let mut ctrl = GlobeControls::new(10_000.0);
        let initial_theta = ctrl.theta();

        // Single drag event from rest — vel was 0, so only this frame's delta should apply.
        let alt_factor = (((ctrl.radius() - PLANET_RADIUS) / ctrl.radius()) as f32).max(PAN_MIN_ALT_FACTOR);
        ctrl.on_drag(10.0, 0.0);

        let expected_delta = -10.0 * DRAG_SENSITIVITY * alt_factor;
        let actual_delta = ctrl.theta() - initial_theta;

        assert!(
            (actual_delta - expected_delta).abs() < 1e-6,
            "on_drag applied {actual_delta} but expected exactly {expected_delta} (single-frame delta only)"
        );

        // update(dt) afterward adds only the decayed inertial term, not another full delta.
        let theta_after_drag = ctrl.theta();
        ctrl.update(0.016); // one frame at ~60fps
        let inertial_addition = ctrl.theta() - theta_after_drag;

        // The inertial addition should be small (vel decayed by exp(-INERTIA_DAMPING * dt))
        // and strictly less than the full delta — NOT equal to it (which would indicate double-apply).
        assert!(
            inertial_addition.abs() < expected_delta.abs(),
            "update() added {inertial_addition} which is >= single-frame delta {expected_delta} — suggests double-apply"
        );
    }
}

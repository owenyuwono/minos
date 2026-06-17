//! `first_person` — surface walker on a spherical planet.
#![allow(dead_code)]
//!
//! `FirstPersonController` maintains a local tangent frame at the player's
//! feet on the sphere surface.  Up = normalised position vector; forward is
//! kept perpendicular to up by projecting onto the tangent plane.
//!
//! Mouse-look rotates in the tangent frame: yaw spins the heading around the
//! local up; pitch tilts the view.  Movement advances the player along the
//! sphere surface at a configurable speed.

use enki_render::camera::Camera;
use glam::{DVec3, Quat, Vec3};

// ── MoveInput ─────────────────────────────────────────────────────────────

/// Per-frame WASD movement request (plain booleans — no winit dependency).
#[derive(Debug, Clone, Copy, Default)]
pub struct MoveInput {
    /// Move forward along the local heading.
    pub forward: bool,
    /// Move backward along the local heading.
    pub backward: bool,
    /// Strafe left (perpendicular to heading, in the tangent plane).
    pub left: bool,
    /// Strafe right.
    pub right: bool,
    /// Sprint: multiply speed by [`SPRINT_MULTIPLIER`].
    pub sprint: bool,
}

// ── Constants ─────────────────────────────────────────────────────────────

/// Base walk speed (m/s).
const BASE_SPEED: f64 = 8.0;
/// Sprint multiplier applied when `MoveInput::sprint` is true.
pub const SPRINT_MULTIPLIER: f64 = 5.0;
/// Eye height above the surface (m).
const EYE_HEIGHT: f64 = 1.7;
/// Maximum pitch away from horizontal (radians), applied symmetrically.
const MAX_PITCH: f32 = 85_f32 * std::f32::consts::PI / 180.0;
/// Mouse-look sensitivity (radians per pixel).
const MOUSE_SENSITIVITY: f32 = 0.002;

// ── FirstPersonController ─────────────────────────────────────────────────

/// Surface-walker camera controller for first-person mode.
///
/// The player's feet are on the sphere at radius `surface_radius`.  The
/// camera eye is positioned `EYE_HEIGHT` metres above the feet along the
/// local up direction.
///
/// **Coordinate frame** (all in world space):
/// - `up`      = normalised position of the feet (outward sphere normal).
/// - `forward` = player heading, kept ⟂ `up` at all times.
/// - `right`   = `forward × up` (left-handed tangent plane basis; we
///   normalise in camera() to absorb floating-point drift).
#[derive(Debug, Clone)]
pub struct FirstPersonController {
    /// Feet position on the sphere surface (world, f64).
    feet: DVec3,
    /// Player heading direction in the tangent plane (f32, unit vector in world space).
    /// Always perpendicular to `up`.
    heading: Vec3,
    /// Current pitch in radians, clamped to ±MAX_PITCH.
    pitch: f32,
    /// Surface radius of the planet (m).  Treat as constant (terrain query hook later).
    surface_radius: f64,
}

impl FirstPersonController {
    /// Create a new controller.
    ///
    /// - `feet_pos`       — initial feet position; will be renormalised to `surface_radius`.
    /// - `surface_radius` — planet (or terrain) radius at this point (m).
    /// - `initial_heading`— preferred forward direction; must not be parallel to `feet_pos`.
    ///   Projected onto the tangent plane and normalised.  Falls back to a safe default if
    ///   the projection is degenerate.
    pub fn new(feet_pos: DVec3, surface_radius: f64, initial_heading: Vec3) -> Self {
        let up = feet_pos.normalize();
        let feet = up * surface_radius;
        let heading = project_onto_tangent_plane(initial_heading, up.as_vec3());
        Self {
            feet,
            heading,
            pitch: 0.0,
            surface_radius,
        }
    }

    /// Mouse-look: yaw in the tangent plane, pitch clamped to ±85°.
    ///
    /// `dx` positive = look right; `dy` positive = look down.
    pub fn on_mouse_look(&mut self, dx: f32, dy: f32) {
        let up = self.local_up().as_vec3();

        // Yaw: rotate heading around local up
        let yaw_delta = -dx * MOUSE_SENSITIVITY;
        let yaw_rot = Quat::from_axis_angle(up, yaw_delta);
        self.heading = yaw_rot * self.heading;
        // Re-project to guard against numeric drift accumulating over many frames
        self.heading = project_onto_tangent_plane(self.heading, up);

        // Pitch: clamp, no wrap
        self.pitch = (self.pitch - dy * MOUSE_SENSITIVITY).clamp(-MAX_PITCH, MAX_PITCH);
    }

    /// Move the player along the sphere surface.
    ///
    /// Advances the feet position on the sphere surface by integrating the
    /// desired velocity, then re-projects to keep feet on the sphere.
    pub fn on_move(&mut self, input: MoveInput, dt: f32) {
        let speed = if input.sprint {
            BASE_SPEED * SPRINT_MULTIPLIER
        } else {
            BASE_SPEED
        };

        let up = self.local_up().as_vec3();
        let right = self.heading.cross(up).normalize();

        let mut delta = Vec3::ZERO;
        if input.forward  { delta += self.heading; }
        if input.backward { delta -= self.heading; }
        if input.right    { delta += right; }
        if input.left     { delta -= right; }

        if delta.length_squared() < 1e-12 {
            return;
        }

        let dir = delta.normalize();
        let move_dist = speed * dt as f64;

        // Move in world space along the tangent direction, then snap back to sphere
        let new_feet_f64 = self.feet + dir.as_dvec3() * move_dist;
        self.feet = new_feet_f64.normalize() * self.surface_radius;

        // Reproject heading onto the new tangent plane (the up direction rotated)
        let new_up = self.local_up().as_vec3();
        self.heading = project_onto_tangent_plane(self.heading, new_up);
    }

    /// Build the `Camera` for the current player state.
    ///
    /// Eye is `EYE_HEIGHT` metres above `feet` along the local up.
    /// Orientation combines yaw (heading) and pitch.
    pub fn camera(&self) -> Camera {
        let up = self.local_up().as_vec3();
        let eye = self.feet + self.local_up() * EYE_HEIGHT;

        // Right = heading × up  (left-hand tangent basis, then normalise)
        let right = self.heading.cross(up).normalize();
        // Orthogonal up in the camera frame (= world up when pitch == 0)
        let cam_up = right.cross(self.heading).normalize();

        // Apply pitch: tilt forward/cam_up by self.pitch
        let pitch_rot = Quat::from_axis_angle(right, -self.pitch);
        let forward = pitch_rot * self.heading;
        let view_up = pitch_rot * cam_up;

        // Build orientation from axes
        let orientation = Quat::from_mat3(&glam::Mat3::from_cols(
            right,
            view_up,
            -forward,
        ));

        Camera {
            position: eye,
            orientation,
            fov_y_radians: 75_f32.to_radians(),
            near: 0.1,
            far: 100_000.0,
        }
    }

    /// Current feet position on the sphere surface.
    pub fn feet_position(&self) -> DVec3 {
        self.feet
    }

    /// Current eye position (feet + eye_height along up).
    pub fn eye_position(&self) -> DVec3 {
        self.feet + self.local_up() * EYE_HEIGHT
    }

    /// Current pitch angle (radians), clamped to ±MAX_PITCH.
    pub fn pitch(&self) -> f32 {
        self.pitch
    }

    /// Local up direction (outward surface normal at the feet position).
    fn local_up(&self) -> DVec3 {
        self.feet.normalize()
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Project `v` onto the tangent plane defined by unit normal `up`, then
/// normalise.  Falls back to a safe perpendicular to `up` if degenerate.
fn project_onto_tangent_plane(v: Vec3, up: Vec3) -> Vec3 {
    let projected = v - up * v.dot(up);
    if projected.length_squared() < 1e-10 {
        // v was parallel to up — pick an arbitrary perpendicular
        let alt = if up.abs().x < 0.9 { Vec3::X } else { Vec3::Z };
        (alt - up * alt.dot(up)).normalize()
    } else {
        projected.normalize()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::globe::PLANET_RADIUS;

    const SURFACE_R: f64 = PLANET_RADIUS;

    fn north_pole_controller() -> FirstPersonController {
        // Feet at north pole (+Y)
        FirstPersonController::new(
            DVec3::new(0.0, SURFACE_R, 0.0),
            SURFACE_R,
            Vec3::new(0.0, 0.0, -1.0), // initial heading toward -Z
        )
    }

    // ── Pitch clamping ────────────────────────────────────────────────────

    #[test]
    fn pitch_clamped_at_positive_max() {
        let mut ctrl = north_pole_controller();
        // Large downward drag
        ctrl.on_mouse_look(0.0, 1_000_000.0);
        let p = ctrl.pitch();
        assert!((-MAX_PITCH - 1e-5..=MAX_PITCH + 1e-5).contains(&p),
            "pitch {p} out of clamp range");
        assert!((p + MAX_PITCH).abs() < 1e-4,
            "pitch should be at -MAX_PITCH ({}) but is {p}", -MAX_PITCH);
    }

    #[test]
    fn pitch_clamped_at_negative_max() {
        let mut ctrl = north_pole_controller();
        // Large upward drag
        ctrl.on_mouse_look(0.0, -1_000_000.0);
        let p = ctrl.pitch();
        assert!((p - MAX_PITCH).abs() < 1e-4,
            "pitch should be at +MAX_PITCH ({}) but is {p}", MAX_PITCH);
    }

    #[test]
    fn pitch_85_degrees_boundary() {
        let mut ctrl = north_pole_controller();
        let deg85 = 85_f32.to_radians();
        ctrl.on_mouse_look(0.0, -(deg85 / MOUSE_SENSITIVITY));
        assert!((ctrl.pitch() - deg85).abs() < 1e-4);
    }

    // ── Surface grounding ─────────────────────────────────────────────────

    #[test]
    fn feet_stay_on_sphere_after_move() {
        let mut ctrl = north_pole_controller();
        let input = MoveInput { forward: true, ..Default::default() };
        for _ in 0..100 {
            ctrl.on_move(input, 0.016);
        }
        let r = ctrl.feet_position().length();
        assert!((r - SURFACE_R).abs() < 1.0,
            "feet radius {r} drifted from surface {SURFACE_R}");
    }

    #[test]
    fn eye_is_above_feet() {
        let ctrl = north_pole_controller();
        let feet = ctrl.feet_position().length();
        let eye = ctrl.eye_position().length();
        assert!((eye - feet - EYE_HEIGHT).abs() < 1e-6,
            "eye height {} != EYE_HEIGHT {}", eye - feet, EYE_HEIGHT);
    }

    // ── Camera output ─────────────────────────────────────────────────────

    #[test]
    fn camera_is_finite() {
        let ctrl = north_pole_controller();
        let cam = ctrl.camera();
        assert!(cam.position.is_finite());
        assert!(cam.orientation.is_finite());
    }

    #[test]
    fn camera_view_matrix_is_finite() {
        let ctrl = north_pole_controller();
        let cam = ctrl.camera();
        let vm = cam.view_matrix();
        for col in 0..4 {
            for row in 0..4 {
                assert!(vm.col(col)[row].is_finite(),
                    "view_matrix[{col}][{row}] is not finite");
            }
        }
    }

    #[test]
    fn no_movement_when_no_input() {
        let mut ctrl = north_pole_controller();
        let before = ctrl.feet_position();
        ctrl.on_move(MoveInput::default(), 0.016);
        let after = ctrl.feet_position();
        assert!((before - after).length() < 1e-9, "position changed with no input");
    }

    #[test]
    fn sprint_moves_faster_than_walk() {
        let input_walk   = MoveInput { forward: true, sprint: false, ..Default::default() };
        let input_sprint = MoveInput { forward: true, sprint: true,  ..Default::default() };
        let dt = 1.0_f32;

        let mut walk = north_pole_controller();
        let start = walk.feet_position();
        walk.on_move(input_walk, dt);
        let walk_dist = (walk.feet_position() - start).length();

        let mut sprint = north_pole_controller();
        sprint.on_move(input_sprint, dt);
        let sprint_dist = (sprint.feet_position() - start).length();

        assert!(sprint_dist > walk_dist,
            "sprint_dist {sprint_dist} should > walk_dist {walk_dist}");
    }

    // ── Tangent-plane projection ──────────────────────────────────────────

    #[test]
    fn project_parallel_to_up_falls_back_safely() {
        let up = Vec3::Y;
        // v == up should not produce NaN
        let result = super::project_onto_tangent_plane(Vec3::Y, up);
        assert!(result.is_finite());
        assert!((result.length() - 1.0).abs() < 1e-5);
        // Must be perpendicular to up
        assert!(result.dot(up).abs() < 1e-5);
    }
}

//! `third_person` — surface walker with an orbiting chase camera.
#![allow(dead_code)]
//!
//! The character's feet ride the terrain surface (same grounding as
//! [`super::first_person`]). Movement is **camera-relative**: WASD moves in the
//! camera's horizontal frame and the body turns to face its travel direction
//! (Zelda / modern-3rd-person feel). The mouse orbits the camera around the
//! character; scroll changes the boom length. A view toggle drops the camera to
//! the character's eye for a first-person view (the mesh is hidden by the caller
//! in that view).
//!
//! Coordinate frame (all world-space): `up` = feet direction (outward sphere
//! normal); `cam_dir` = camera azimuth kept ⟂ `up`; `facing` = body forward kept
//! ⟂ `up`.

use std::sync::Arc;

use enki_planet::height::HeightField;
use enki_render::camera::Camera;
use glam::{DVec3, Mat3, Quat, Vec3};

use super::first_person::{project_onto_tangent_plane, surface_radius, MoveInput, SPRINT_MULTIPLIER};
use super::terrain_grid::grid_surface_radius;

// ── Constants ─────────────────────────────────────────────────────────────

/// Base walk speed (m/s). Sprint multiplies by [`SPRINT_MULTIPLIER`].
const BASE_SPEED: f64 = 8.0;
/// Small ground-contact offset (m) so the boots rest on the surface instead of
/// z-fighting the ground triangle. Grounding is grid-accurate (rides the drawn
/// mesh), so this is just a contact bias, not a clip band-aid.
const FOOT_CLEARANCE: f64 = 0.05;
/// Camera target height above the feet (m) — roughly the shoulders.
const ANCHOR_HEIGHT: f64 = 1.6;
/// First-person eye height above the feet (m).
const EYE_HEIGHT: f64 = 1.7;
/// Orbit sensitivity (radians per pixel) for both yaw and pitch.
const ORBIT_SENSITIVITY: f32 = 0.005;
/// Camera pitch clamp (radians). Positive = camera elevated, looking down.
const MIN_PITCH: f32 = -30_f32 * std::f32::consts::PI / 180.0;
const MAX_PITCH: f32 =  80_f32 * std::f32::consts::PI / 180.0;
/// Boom length clamps (m) and the default on spawn.
const MIN_DIST: f32 = 2.0;
const MAX_DIST: f32 = 20.0;
const DEFAULT_DIST: f32 = 6.0;
/// Metres of boom shortening per scroll notch.
const ZOOM_STEP: f32 = 0.8;
/// How fast the body yaws to face its travel direction (radians/second).
const TURN_RATE: f32 = 12.0;
/// Keep the third-person camera this far off the terrain (m) — acts as the
/// camera's collision radius so the near plane clears the surface.
const CAM_MARGIN: f64 = 0.5;
/// Samples in the camera-collision march from the character outward.
const COLLISION_STEPS: u32 = 16;
/// Closest the collision may pull the boom (m) when fully obstructed.
const COLLISION_MIN: f32 = 0.4;
/// Boom smoothing rates (per second): snap IN fast when blocked, ease OUT slowly
/// when clear so the camera doesn't pop out from behind cover.
const BOOM_PULL_IN_RATE: f32 = 30.0;
const BOOM_PUSH_OUT_RATE: f32 = 6.0;

// ── View ────────────────────────────────────────────────────────────────────

/// Which camera the controller produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum View {
    /// Chase camera behind/above the character; the mesh is drawn.
    Third,
    /// Camera at the character's eye; the caller hides the mesh.
    First,
}

// ── ThirdPersonController ─────────────────────────────────────────────────

/// Surface-walking character controller with an orbiting chase camera.
#[derive(Clone)]
pub struct ThirdPersonController {
    /// Feet position on the terrain surface (world, f64).
    feet: DVec3,
    /// Body forward in the tangent plane (unit, ⟂ up).
    facing: Vec3,
    /// Camera azimuth direction in the tangent plane (unit, ⟂ up).
    cam_dir: Vec3,
    /// Camera pitch (radians), clamped to `[MIN_PITCH, MAX_PITCH]`.
    cam_pitch: f32,
    /// Desired chase-camera boom length (m) — what zoom sets.
    cam_dist: f32,
    /// Smoothed *effective* boom after collision (lerps toward the collided
    /// target; fast in, slow out). This is what `camera()` actually uses.
    eff_dist: f32,
    view: View,

    hf: Arc<dyn HeightField>,
    base_radius: f64,
    height_scale: f64,
    /// Quads per cube-face side of the rendered terrain grid (Nanite bake res).
    /// Feet ride this faceted grid, not the full analytic curve.
    grid_res: u32,
}

impl ThirdPersonController {
    /// Spawn the character grounded on the terrain.
    ///
    /// Only the *direction* of `feet_pos` is used; the radius is recomputed from
    /// the height field so the feet sit on the visible ground. `initial_heading`
    /// is projected onto the tangent plane (the camera starts behind it).
    pub fn new(
        feet_pos: DVec3,
        hf: Arc<dyn HeightField>,
        base_radius: f64,
        height_scale: f64,
        grid_res: u32,
        initial_heading: Vec3,
    ) -> Self {
        let dir = feet_pos.normalize();
        let facing = project_onto_tangent_plane(initial_heading, dir.as_vec3());
        let mut ctrl = Self {
            feet: DVec3::ZERO,
            facing,
            cam_dir: facing,
            cam_pitch: 0.2,
            cam_dist: DEFAULT_DIST,
            eff_dist: DEFAULT_DIST,
            view: View::Third,
            hf,
            base_radius,
            height_scale,
            grid_res,
        };
        ctrl.feet = ctrl.grounded(dir);
        ctrl
    }

    /// Analytic point sample of the surface radius — used for the camera-occlusion
    /// march (a conservative "is this above ground" probe), not for the feet.
    fn ground_radius(&self, dir: DVec3) -> f64 {
        surface_radius(self.hf.as_ref(), self.base_radius, self.height_scale, dir)
    }

    /// Feet world position for unit `dir`: the **rendered** (grid-faceted) surface
    /// under `dir`, lifted by [`FOOT_CLEARANCE`]. Riding the grid (not the analytic
    /// curve) is what keeps the body on the drawn ground instead of diving.
    fn grounded(&self, dir: DVec3) -> DVec3 {
        let r = grid_surface_radius(
            self.hf.as_ref(), self.base_radius, self.height_scale, self.grid_res, dir,
        );
        dir * (r + FOOT_CLEARANCE)
    }

    fn local_up(&self) -> DVec3 {
        self.feet.normalize()
    }

    /// Orbit the camera. `dx` positive = drag right; `dy` positive = drag down.
    pub fn on_orbit(&mut self, dx: f32, dy: f32) {
        let up = self.local_up().as_vec3();
        // Yaw the azimuth around local up; re-project to absorb numeric drift.
        let yaw = Quat::from_axis_angle(up, -dx * ORBIT_SENSITIVITY);
        self.cam_dir = project_onto_tangent_plane(yaw * self.cam_dir, up);
        // Non-inverted: mouse up (dy<0) tilts the view up (camera drops below the
        // character); mouse down raises the camera to look down. Positive pitch =
        // camera elevated, looking down.
        self.cam_pitch = (self.cam_pitch + dy * ORBIT_SENSITIVITY).clamp(MIN_PITCH, MAX_PITCH);
    }

    /// Scroll to change the boom length. Positive `delta` zooms in.
    pub fn on_zoom(&mut self, delta: f32) {
        self.cam_dist = (self.cam_dist - delta * ZOOM_STEP).clamp(MIN_DIST, MAX_DIST);
    }

    /// Toggle between third- and first-person camera.
    pub fn toggle_view(&mut self) {
        self.view = match self.view {
            View::Third => View::First,
            View::First => View::Third,
        };
    }

    pub fn view(&self) -> View {
        self.view
    }

    /// True when the character mesh should be drawn (third-person view).
    pub fn show_character(&self) -> bool {
        self.view == View::Third
    }

    /// Move the character (camera-relative) and turn the body toward its travel
    /// direction. Returns the current ground speed in m/s (0 = idle), which the
    /// animator maps to gait cadence + amplitude.
    pub fn on_move(&mut self, input: MoveInput, dt: f32) -> f32 {
        let up = self.local_up().as_vec3();
        let right = self.cam_dir.cross(up).normalize();

        let mut delta = Vec3::ZERO;
        if input.forward  { delta += self.cam_dir; }
        if input.backward { delta -= self.cam_dir; }
        if input.right    { delta += right; }
        if input.left     { delta -= right; }

        if delta.length_squared() < 1e-12 {
            return 0.0; // idle
        }

        let speed = if input.sprint { BASE_SPEED * SPRINT_MULTIPLIER } else { BASE_SPEED };
        let move_dir = delta.normalize();
        let move_dist = speed * dt as f64;

        // Advance along the tangent, then re-ground at the new direction.
        let new_dir = (self.feet + move_dir.as_dvec3() * move_dist).normalize();
        self.feet = self.grounded(new_dir);

        // Re-project onto the new tangent plane (up rotated as we moved).
        let new_up = self.local_up().as_vec3();
        self.cam_dir = project_onto_tangent_plane(self.cam_dir, new_up);

        // Turn the body toward the travel direction at a bounded rate.
        let desired = project_onto_tangent_plane(move_dir, new_up);
        self.facing = rotate_toward(self.facing, desired, TURN_RATE * dt, new_up);

        speed as f32
    }

    /// Camera target (look-at) point — shoulders height above the feet.
    fn anchor(&self) -> DVec3 {
        self.feet + self.local_up() * ANCHOR_HEIGHT
    }

    /// Camera look direction: azimuth tilted by pitch. Positive pitch tilts
    /// downward (the chase camera is elevated and looks down on the character).
    fn look_dir(&self) -> Vec3 {
        let upf = self.local_up().as_vec3();
        let (sp, cp) = (self.cam_pitch.sin(), self.cam_pitch.cos());
        (self.cam_dir * cp - upf * sp).normalize()
    }

    /// Per-frame tick: advance the smoothed collision boom. Call once per frame
    /// (with the active surface controller) so `camera()` reflects collision.
    pub fn update(&mut self, dt: f32) {
        let target = self.collide_boom(self.anchor(), self.look_dir());
        // Snap in fast when newly blocked; ease out slowly when the view clears.
        let rate = if target < self.eff_dist { BOOM_PULL_IN_RATE } else { BOOM_PUSH_OUT_RATE };
        let t = (rate * dt).clamp(0.0, 1.0);
        self.eff_dist += (target - self.eff_dist) * t;
    }

    /// Build the camera for the current view.
    pub fn camera(&self) -> Camera {
        let up = self.local_up();
        let upf = up.as_vec3();
        let look = self.look_dir();

        let position = match self.view {
            View::First => self.feet + up * EYE_HEIGHT,
            View::Third => self.anchor() - look.as_dvec3() * self.eff_dist as f64,
        };

        // Orientation from (forward = look, world up = surface up).
        let right = look.cross(upf).normalize();
        let cam_up = right.cross(look).normalize();
        let orientation = Quat::from_mat3(&Mat3::from_cols(right, cam_up, -look));

        Camera {
            position,
            orientation,
            fov_y_radians: 70_f32.to_radians(),
            near: 0.1,
            far: 100_000.0,
        }
    }

    /// Largest collision-free boom length: march from the anchor outward along the
    /// boom and stop at the first sample that enters terrain, so terrain *between*
    /// the character and the camera pulls the camera in (line-of-sight occlusion).
    ///
    /// Tested against the **drawn** grid surface (same as the feet) so it agrees
    /// with what's on screen — not the analytic curve. `CAM_MARGIN` is the camera's
    /// collision radius.
    fn collide_boom(&self, anchor: DVec3, look: Vec3) -> f32 {
        let out = -look.as_dvec3(); // anchor → desired camera position
        let mut safe = COLLISION_MIN.min(self.cam_dist);
        for i in 1..=COLLISION_STEPS {
            let d = self.cam_dist * (i as f32 / COLLISION_STEPS as f32);
            let pos = anchor + out * d as f64;
            let ground = grid_surface_radius(
                self.hf.as_ref(), self.base_radius, self.height_scale, self.grid_res,
                pos.normalize(),
            ) + CAM_MARGIN;
            if pos.length() < ground {
                break; // blocked here → keep the last clear distance
            }
            safe = d;
        }
        safe.clamp(COLLISION_MIN.min(self.cam_dist), self.cam_dist)
    }

    /// Current feet position on the surface.
    pub fn feet_position(&self) -> DVec3 {
        self.feet
    }

    /// Body forward direction (unit, tangent).
    pub fn facing(&self) -> Vec3 {
        self.facing
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// Rotate unit vector `from` toward unit vector `to` by at most `max_step`
/// radians (around `up`), then re-project onto the tangent plane of `up`.
fn rotate_toward(from: Vec3, to: Vec3, max_step: f32, up: Vec3) -> Vec3 {
    let angle = from.angle_between(to);
    if angle < 1e-4 {
        return to;
    }
    let t = (max_step / angle).min(1.0);
    let blended = from.lerp(to, t);
    project_onto_tangent_plane(blended, up)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::globe::PLANET_RADIUS;

    const R: f64 = PLANET_RADIUS;

    struct FlatHf;
    impl HeightField for FlatHf {
        fn height(&self, _dir: DVec3, _level: u8) -> f64 {
            0.0
        }
    }

    fn spawn() -> ThirdPersonController {
        ThirdPersonController::new(
            DVec3::new(0.0, R, 0.0),
            Arc::new(FlatHf),
            R,
            1.0,
            256, // grid_res
            Vec3::new(0.0, 0.0, -1.0),
        )
    }

    #[test]
    fn spawns_on_surface() {
        let c = spawn();
        assert!((c.feet_position().length() - R).abs() < 1.0);
    }

    #[test]
    fn feet_stay_on_sphere_after_move() {
        let mut c = spawn();
        let input = MoveInput { forward: true, ..Default::default() };
        for _ in 0..100 {
            c.on_move(input, 0.016);
        }
        assert!((c.feet_position().length() - R).abs() < 1.0);
    }

    #[test]
    fn no_input_is_idle() {
        let mut c = spawn();
        let before = c.feet_position();
        let speed = c.on_move(MoveInput::default(), 0.016);
        assert_eq!(speed, 0.0);
        assert!((c.feet_position() - before).length() < 1e-9);
    }

    #[test]
    fn sprint_reports_higher_speed() {
        let mut walk = spawn();
        let mut run = spawn();
        let w = walk.on_move(MoveInput { forward: true, ..Default::default() }, 0.016);
        let r = run.on_move(MoveInput { forward: true, sprint: true, ..Default::default() }, 0.016);
        assert!(r > w && w > 0.0);
    }

    #[test]
    fn body_turns_toward_travel_direction() {
        // Move along +right repeatedly; facing should converge to the strafe dir.
        let mut c = spawn();
        let up = c.local_up().as_vec3();
        let right = c.cam_dir.cross(up).normalize();
        for _ in 0..120 {
            c.on_move(MoveInput { right: true, ..Default::default() }, 0.05);
        }
        assert!(c.facing().dot(right) > 0.95, "facing did not turn toward travel dir");
    }

    #[test]
    fn cameras_are_finite_both_views() {
        let mut c = spawn();
        for _ in 0..2 {
            let cam = c.camera();
            assert!(cam.position.is_finite());
            assert!(cam.orientation.is_finite());
            c.toggle_view();
        }
    }

    #[test]
    fn first_person_eye_is_near_feet_third_is_far() {
        let mut c = spawn();
        // Third-person: camera is several metres from the feet.
        let third = (c.camera().position - c.feet_position()).length();
        c.toggle_view();
        // First-person: camera is at the eye, ~EYE_HEIGHT from the feet.
        let first = (c.camera().position - c.feet_position()).length();
        assert!(first < 2.5, "first-person camera too far from feet: {first}");
        assert!(third > first + 1.0, "third-person camera should be farther: {third} vs {first}");
    }

    #[test]
    fn chase_camera_stays_above_terrain() {
        // Even at the lowest pitch, collision must keep the camera above the surface.
        let mut c = spawn();
        for _ in 0..50 {
            c.on_orbit(0.0, -10_000.0); // drag up hard → minimum elevation (look up)
        }
        // Collision is applied by update(); converge the smoothed boom.
        for _ in 0..10 {
            c.update(0.1);
        }
        let cam = c.camera();
        let dir = cam.position.normalize();
        let min_r = c.ground_radius(dir);
        assert!(cam.position.length() >= min_r - 1e-3,
            "camera sank below terrain: r={} min={}", cam.position.length(), min_r);
    }

    #[test]
    fn collision_pulls_camera_in_when_blocked() {
        // Terrain that climbs steeply away from the spawn pole — the camera, sitting
        // a few metres off-pole, runs into the rising ground and the boom shortens.
        // Small test planet so a few metres of boom is a meaningful arc.
        struct RampHf;
        impl HeightField for RampHf {
            fn height(&self, dir: DVec3, _l: u8) -> f64 {
                1.0 - dir.normalize().y // 0 at the +Y pole, rising away from it
            }
        }
        let mut c = ThirdPersonController::new(
            DVec3::new(0.0, 100.0, 0.0), Arc::new(RampHf), 100.0, 2000.0, 256,
            Vec3::new(0.0, 0.0, -1.0),
        );
        for _ in 0..30 { c.update(0.1); }
        assert!(c.eff_dist < DEFAULT_DIST - 0.5,
            "boom should pull in against the rising terrain, got {}", c.eff_dist);
        assert!(c.eff_dist > 0.0 && c.eff_dist.is_finite());
    }

    #[test]
    fn zoom_clamps_to_range() {
        let mut c = spawn();
        for _ in 0..100 { c.on_zoom(1.0); }   // zoom all the way in
        assert!(c.cam_dist >= MIN_DIST - 1e-4);
        for _ in 0..100 { c.on_zoom(-1.0); }  // zoom all the way out
        assert!(c.cam_dist <= MAX_DIST + 1e-4);
    }
}

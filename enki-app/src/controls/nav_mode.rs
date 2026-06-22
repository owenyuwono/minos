//! `nav_mode` — navigation mode state machine.
//!
//! Tracks which camera controller is active and mediates transitions between
//! them.  The three modes form a linear state machine:
//!
//! ```text
//!  Globe  ──begin_placement──▶  Placement  ──point_picked──▶  Surface
//!    ▲                                                            │
//!    └────────────────────exit_surface───────────────────────────┘
//! ```
//!
//! `NavState` also holds a snapshot of the globe `Camera` so that returning
//! from the surface walker restores the previous orbit view.

use enki_render::camera::Camera;

// ── Mode ──────────────────────────────────────────────────────────────────

/// Active navigation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavMode {
    /// Orbiting the planet from space or high altitude.
    Globe,
    /// User is picking a spawn point on the surface (ray-cast placement phase).
    Placement,
    /// Walking on the planet surface (third-person chase cam, with a 1st-person
    /// view toggle). Hosts the `ThirdPersonController`.
    Surface,
    /// Free-flight through interplanetary space (`FreeCam`). Toggled from Globe.
    Space,
}

// ── State machine ─────────────────────────────────────────────────────────

/// Holds the current [`NavMode`] and any state that must survive mode
/// transitions (e.g. the saved globe camera for restore on exit).
#[derive(Debug, Clone)]
pub struct NavState {
    mode: NavMode,
    /// Snapshot of the globe `Camera` taken when we enter `Placement` or
    /// `Surface`, so we can restore it when the user exits the surface walker.
    saved_globe_camera: Option<Camera>,
}

impl NavState {
    /// Create a new state machine starting in [`NavMode::Globe`].
    pub fn new() -> Self {
        Self {
            mode: NavMode::Globe,
            saved_globe_camera: None,
        }
    }

    /// Current navigation mode.
    pub fn mode(&self) -> NavMode {
        self.mode
    }

    /// `Globe → Placement`: user has begun choosing a spawn point.
    ///
    /// Snapshots `globe_camera` so it can be restored later.
    /// Has no effect if the current mode is not [`NavMode::Globe`].
    pub fn begin_placement(&mut self, globe_camera: &Camera) {
        if self.mode == NavMode::Globe {
            self.saved_globe_camera = Some(globe_camera.clone());
            self.mode = NavMode::Placement;
        }
    }

    /// `Placement → Surface`: a surface point was successfully picked.
    ///
    /// Has no effect if the current mode is not [`NavMode::Placement`].
    pub fn point_picked(&mut self) {
        if self.mode == NavMode::Placement {
            self.mode = NavMode::Surface;
        }
    }

    /// `Surface → Globe`: user exits the surface walker.
    ///
    /// Returns the previously saved globe `Camera` if one was stored.
    /// Has no effect (and returns `None`) if the current mode is not
    /// [`NavMode::Surface`].
    pub fn exit_surface(&mut self) -> Option<Camera> {
        if self.mode == NavMode::Surface {
            self.mode = NavMode::Globe;
            self.saved_globe_camera.take()
        } else {
            None
        }
    }

    /// `Globe → Space` / `Space → Globe`: toggle free-flight.
    pub fn enter_space(&mut self) {
        if self.mode == NavMode::Globe {
            self.mode = NavMode::Space;
        }
    }
    pub fn exit_space(&mut self) {
        if self.mode == NavMode::Space {
            self.mode = NavMode::Globe;
        }
    }
}

impl Default for NavState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    fn test_camera() -> Camera {
        Camera {
            position: DVec3::new(0.0, 0.0, 100_000.0),
            orientation: glam::Quat::IDENTITY,
            fov_y_radians: 60_f32.to_radians(),
            near: 0.5,
            far: 750_000.0,
        }
    }

    #[test]
    fn starts_in_globe_mode() {
        let state = NavState::new();
        assert_eq!(state.mode(), NavMode::Globe);
    }

    #[test]
    fn globe_to_placement_transitions() {
        let mut state = NavState::new();
        let cam = test_camera();
        state.begin_placement(&cam);
        assert_eq!(state.mode(), NavMode::Placement);
    }

    #[test]
    fn placement_to_surface() {
        let mut state = NavState::new();
        state.begin_placement(&test_camera());
        state.point_picked();
        assert_eq!(state.mode(), NavMode::Surface);
    }

    #[test]
    fn surface_to_globe_restores_camera() {
        let mut state = NavState::new();
        let cam = test_camera();
        let expected_z = cam.position.z;
        state.begin_placement(&cam);
        state.point_picked();
        let restored = state.exit_surface().expect("should restore camera");
        assert_eq!(state.mode(), NavMode::Globe);
        assert!((restored.position.z - expected_z).abs() < 1e-6);
    }

    #[test]
    fn exit_from_wrong_mode_is_noop() {
        let mut state = NavState::new();
        // Calling exit_surface from Globe should do nothing
        let result = state.exit_surface();
        assert!(result.is_none());
        assert_eq!(state.mode(), NavMode::Globe);
    }

    #[test]
    fn begin_placement_from_wrong_mode_is_noop() {
        let mut state = NavState::new();
        state.begin_placement(&test_camera());
        state.point_picked();
        // In Surface — begin_placement should not regress mode
        state.begin_placement(&test_camera());
        assert_eq!(state.mode(), NavMode::Surface);
    }

    #[test]
    fn full_cycle_twice() {
        let mut state = NavState::new();
        for _ in 0..2 {
            assert_eq!(state.mode(), NavMode::Globe);
            state.begin_placement(&test_camera());
            state.point_picked();
            state.exit_surface();
        }
        assert_eq!(state.mode(), NavMode::Globe);
    }
}

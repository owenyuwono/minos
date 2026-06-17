//! `nav_mode` — navigation mode state machine.
#![allow(dead_code)]
//!
//! Tracks which camera controller is active and mediates transitions between
//! them.  The three modes form a linear state machine:
//!
//! ```text
//!  Globe  ──begin_placement──▶  Placement  ──point_picked──▶  FirstPerson
//!    ▲                                                              │
//!    └────────────────────exit_first_person────────────────────────┘
//! ```
//!
//! `NavState` also holds a snapshot of the globe `Camera` so that returning
//! from first-person restores the previous orbit view.

use enki_render::camera::Camera;

// ── Mode ──────────────────────────────────────────────────────────────────

/// Active navigation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavMode {
    /// Orbiting the planet from space or high altitude.
    Globe,
    /// User is picking a spawn point on the surface (ray-cast placement phase).
    Placement,
    /// Walking on the planet surface in first-person.
    FirstPerson,
}

// ── State machine ─────────────────────────────────────────────────────────

/// Holds the current [`NavMode`] and any state that must survive mode
/// transitions (e.g. the saved globe camera for restore on exit).
#[derive(Debug, Clone)]
pub struct NavState {
    mode: NavMode,
    /// Snapshot of the globe `Camera` taken when we enter `Placement` or
    /// `FirstPerson`, so we can restore it when the user exits first-person.
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

    /// `Placement → FirstPerson`: a surface point was successfully picked.
    ///
    /// Has no effect if the current mode is not [`NavMode::Placement`].
    pub fn point_picked(&mut self) {
        if self.mode == NavMode::Placement {
            self.mode = NavMode::FirstPerson;
        }
    }

    /// `FirstPerson → Globe`: user exits first-person view.
    ///
    /// Returns the previously saved globe `Camera` if one was stored.
    /// Has no effect (and returns `None`) if the current mode is not
    /// [`NavMode::FirstPerson`].
    pub fn exit_first_person(&mut self) -> Option<Camera> {
        if self.mode == NavMode::FirstPerson {
            self.mode = NavMode::Globe;
            self.saved_globe_camera.take()
        } else {
            None
        }
    }

    /// True when a globe camera snapshot is stored (i.e. we entered
    /// `Placement` or `FirstPerson` from `Globe`).
    pub fn has_saved_globe_camera(&self) -> bool {
        self.saved_globe_camera.is_some()
    }

    /// Borrow the saved globe camera snapshot without consuming it.
    pub fn saved_globe_camera(&self) -> Option<&Camera> {
        self.saved_globe_camera.as_ref()
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
        assert!(!state.has_saved_globe_camera());
    }

    #[test]
    fn globe_to_placement_snapshots_camera() {
        let mut state = NavState::new();
        let cam = test_camera();
        state.begin_placement(&cam);
        assert_eq!(state.mode(), NavMode::Placement);
        assert!(state.has_saved_globe_camera());
    }

    #[test]
    fn placement_to_first_person() {
        let mut state = NavState::new();
        state.begin_placement(&test_camera());
        state.point_picked();
        assert_eq!(state.mode(), NavMode::FirstPerson);
    }

    #[test]
    fn first_person_to_globe_restores_camera() {
        let mut state = NavState::new();
        let cam = test_camera();
        let expected_z = cam.position.z;
        state.begin_placement(&cam);
        state.point_picked();
        let restored = state.exit_first_person().expect("should restore camera");
        assert_eq!(state.mode(), NavMode::Globe);
        assert!((restored.position.z - expected_z).abs() < 1e-6);
    }

    #[test]
    fn exit_from_wrong_mode_is_noop() {
        let mut state = NavState::new();
        // Calling exit_first_person from Globe should do nothing
        let result = state.exit_first_person();
        assert!(result.is_none());
        assert_eq!(state.mode(), NavMode::Globe);
    }

    #[test]
    fn begin_placement_from_wrong_mode_is_noop() {
        let mut state = NavState::new();
        state.begin_placement(&test_camera());
        state.point_picked();
        // In FirstPerson — begin_placement should not regress mode
        state.begin_placement(&test_camera());
        assert_eq!(state.mode(), NavMode::FirstPerson);
    }

    #[test]
    fn full_cycle_twice() {
        let mut state = NavState::new();
        for _ in 0..2 {
            assert_eq!(state.mode(), NavMode::Globe);
            state.begin_placement(&test_camera());
            state.point_picked();
            state.exit_first_person();
        }
        assert_eq!(state.mode(), NavMode::Globe);
    }
}

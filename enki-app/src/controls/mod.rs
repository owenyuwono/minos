//! `controls` — headless navigation controller layer.
//!
//! All input is taken as plain primitive types (f32 deltas, bool flags) so the
//! controllers are fully testable without a display or winit dependency.
//!
//! # Intended wiring (not done here)
//!
//! The application layer (e.g. `main.rs` via winit) translates OS events into
//! the plain primitive inputs below, then each frame it:
//!
//! 1. Calls `active_controller.update(dt)` with the elapsed time.
//! 2. Reads the resulting `camera()` to obtain the new `Camera` for rendering.
//! 3. Consults `NavState` to decide which controller is active and to perform
//!    mode transitions (e.g. `begin_placement`, `point_picked`, `exit_first_person`).
//!
//! Example event routing (sketch only):
//! ```text
//!   winit MouseMotion { delta: (dx, dy) }
//!       → GlobeControls::on_drag(dx, dy)          // in Globe mode
//!       → FirstPersonController::on_mouse_look(dx, dy)  // in FirstPerson mode
//!
//!   winit MouseWheel { delta }
//!       → GlobeControls::on_scroll(delta)
//!
//!   winit KeyboardInput { key: W/A/S/D/Shift }
//!       → MoveInput flags → FirstPersonController::on_move(input, dt)
//!
//!   winit MouseInput (left-click in Placement mode)
//!       → camera_ray(camera, ndc_x, ndc_y, aspect)
//!       → pick(ray_origin, ray_dir, PLANET_CENTRE, PLANET_RADIUS)
//!       → NavState::point_picked() on success
//! ```
//!
//! NO winit types appear in this module.

pub mod globe;
pub mod first_person;
pub mod third_person;
pub mod terrain_grid;
pub mod surface_picker;
pub mod nav_mode;

// Re-export the main public surface.
// `#[allow(unused_imports)]` silences warnings when main.rs has not yet wired
// these controllers — the exports are intentional API surface, not dead code.
#[allow(unused_imports)]
pub use globe::{GlobeControls, spherical_to_cartesian, cartesian_to_spherical, PLANET_RADIUS};
#[allow(unused_imports)]
pub use first_person::{FirstPersonController, MoveInput, SPRINT_MULTIPLIER};
#[allow(unused_imports)]
pub use third_person::{ThirdPersonController, View};
#[allow(unused_imports)]
pub use surface_picker::{SurfaceHit, pick, camera_ray};
#[allow(unused_imports)]
pub use nav_mode::{NavMode, NavState};

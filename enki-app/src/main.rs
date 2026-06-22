// enki-app — winit 0.30 application shell with interactive navigation,
// material/view hot-swap, egui panel, and HUD.
//
// # Render loop overview
//   resumed()  → create window → Rhi::new → Scene::new [default]
//             OR StressHarness::new [stress]
//             → EguiState::new → Hud::new
//
//   RedrawRequested:
//     Default: begin_frame() → begin_rendering() → Scene::record_frame() →
//              EguiState::render() → end_frame()
//     Stress:  begin_frame() → streaming_upload*() → begin_rendering() →
//              StressHarness::record_frame() → EguiState::render() → end_frame()
//
// # Navigation mode cycle (Tab: Globe → Placement → Surface → Globe)
//   Globe       — orbit camera; LMB drag rotates, scroll zooms.
//   Placement   — shows orbit view; click to ray-cast a spawn point on the planet.
//   Surface     — third-person character: camera-relative WASD, mouse orbits the
//                 chase cam, scroll zooms the boom, V toggles first-person.
//   Escape      — return from Surface → Globe at any time.
//
// # View hot-swap keys (not consumed by egui)
//   M    — cycle view_mode (0–10; skips Nanite-only 3–5 on the classic path)
//   W    — toggle wireframe (Globe/Placement modes)
//   V    — toggle 1st/3rd-person camera (Surface mode)
//   Tab  — advance nav mode
//   Esc  — exit the surface walker to Globe
//
// # Event routing
//   WindowEvent → egui FIRST.  If egui consumed it (wants_pointer / wants_keyboard),
//   do NOT route it to navigation.
//
// # Vulkan frame ordering
//   vkCmdCopyBuffer is illegal inside a dynamic-rendering instance.
//   Egui texture uploads (set_textures) perform a one-shot submit internally —
//   they must happen in `build_frame`, outside begin_rendering/end_frame.
//   Egui draw commands (`cmd_draw`) must be inside begin_rendering, before end_frame.

mod character;
mod controls;
mod gui;
mod fps;
mod loading;
mod ocean;
mod wind;
mod atmosphere;
mod markers;
mod solar;
mod planet_view;
mod scene;
mod stress;
#[cfg(feature = "flora")]
mod flora_scatter;

use std::sync::Arc;
use std::time::Instant;

use enki_planet::climate::ClimateParams;
use enki_planet::height::HeightField;
use enki_planet::lod::{LodCamera, LodConfig};
use enki_render::{camera::Camera, frame::FrameUniforms, lights::Lights, projection::reversed_z_perspective, sky::SkyModel, system::SolarSystem};
use enki_rhi::{Rhi, RhiConfig, RhiError};
use glam::DVec3;
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{CursorGrabMode, Window, WindowAttributes, WindowId},
};

use character::Character;
#[cfg(feature = "flora")]
use enki_app::{
    flora_render::{FloraRenderer, ShadowUniforms},
    flora_view::{FloraView, LeafMode},
};
use controls::{
    first_person::MoveInput,
    globe::GlobeControls,
    nav_mode::{NavMode, NavState},
    third_person::ThirdPersonController,
    surface_picker,
    PLANET_RADIUS,
};
use gui::EguiState;
use fps::FpsMeter;
use ocean::{Ocean, WaveSurface};
use wind::WindOverlay;
use atmosphere::{Atmosphere, AtmoParams};
use markers::Markers;
use solar::BodyRenderer;
use controls::space::FreeCam;
use planet_view::{PlanetConfig, PlanetView, PlanetViewStats, RhiUploaderDummy, planet_view_from_hf};
use scene::Scene;
use stress::StressHarness;

/// Quads per cube-face side of the Nanite terrain bake. Single-sourced so the
/// third-person character grounds against the *same* grid the renderer draws
/// (see `controls::terrain_grid`). Must match the bake's `nanite_resolution`.
const NANITE_BAKE_RES: u32 = 1024;

// ── Render mode ───────────────────────────────────────────────────────────────

enum RenderMode {
    /// Default: live LOD planet rendered via `PlanetView`.
    Planet,
    /// `--markers`: the old depth-proof marker scene (precision regression test).
    Markers,
    /// `--stress-streaming`: streaming budget stress test.
    Stress,
    /// `--soak-terrain` / `ENKI_SOAK_TERRAIN=1`: scripted fly-through that validates
    /// dynamic LOD + streaming under real motion (split/merge churn, hide-while-pending,
    /// streaming graveyard lifecycle).
    SoakTerrain,
}

// ── App state ─────────────────────────────────────────────────────────────────

struct App {
    // Fields drop in declaration order.  Vulkan resources that hold device
    // handles MUST drop before `rhi` (which owns the VkDevice).  Order:
    //   egui  — holds pipeline/textures on the device; must free while device is alive.
    //   planet_view / scene / stress — hold only opaque u64 BufferHandles freed by
    //                    rhi's stores; their order relative to rhi is fine, but
    //                    keeping them before rhi is clearest.
    //   rhi   — LAST: wait_idle + destroys the VkDevice.
    window:       Option<Window>,
    egui:         Option<EguiState>,
    planet_view:  Option<PlanetView<RhiUploaderDummy>>,
    /// Translucent sea-level shell for the live Planet path (built once at startup).
    ocean:        Option<Ocean>,
    /// Solar clock: focused-planet spin → day/night + seasons.
    sky:          SkyModel,
    /// The solar system: f64 heliocentric bodies + orbits.
    system:       SolarSystem,
    /// Solar-system body renderer (sun + distant planets as lit spheres).
    bodies:       Option<BodyRenderer>,
    /// Free-flight space camera (Some only in NavMode::Space).
    freecam:      Option<FreeCam>,
    /// FFT spectral wave surface (camera-anchored), drawn near the surface.
    wave:         Option<WaveSurface>,
    /// Wind streakline overlay (particles riding the baked wind field).
    wind:         Option<WindOverlay>,
    /// Translucent atmosphere shell (halo + the wind's home altitude).
    atmosphere:   Option<Atmosphere>,
    /// Equator + pole reference markers.
    markers:      Option<Markers>,
    scene:        Option<Scene>,
    stress:       Option<StressHarness>,
    /// CPU-skinned humanoid drawn in third-person surface mode (built at startup
    /// in Planet mode; only drawn while walking the surface in third-person view).
    character:    Option<Character>,
    #[cfg(feature = "nanite")]
    nanite:       Option<enki_nanite::render::NaniteRenderer>,
    // Procedural trees (flora), behind `--features flora`. Phase A: the
    // flora-owned sub-renderer (drawing into the main 3D pass) + one walk-up tree
    // spawned near the player in Surface mode. Scatter/LOD is a later slice.
    #[cfg(feature = "flora")]
    flora_renderer: Option<FloraRenderer>,
    /// The single shared species mesh (the prototype default specimen). Drawn at
    /// many camera-relative models — one per scattered instance — via `record_at`.
    #[cfg(feature = "flora")]
    flora_tree:   Option<FloraView>,
    /// Deterministic scatter of trees within `RADIUS_M` of the player. Rebuilt
    /// only when the player moves past a threshold or the density changes.
    #[cfg(feature = "flora")]
    flora_instances: Vec<flora_scatter::TreeInstance>,
    /// Player ground position at the last scatter rebuild (the throttle anchor).
    #[cfg(feature = "flora")]
    flora_scatter_center: Option<DVec3>,
    /// Density used at the last scatter rebuild (rebuild on change).
    #[cfg(feature = "flora")]
    flora_last_density: f32,
    /// Wind animation clock (seconds), advanced each Surface frame trees draw.
    #[cfg(feature = "flora")]
    flora_clock:  f32,
    /// Show trees on the surface (GUI toggle; feature-agnostic so the panel
    /// compiles without `flora` — the draw is what's gated).
    flora_enabled: bool,
    /// Fraction of candidate cells that get a tree (GUI slider, 0..1).
    flora_density: f32,
    /// Stage-2 cluster streaming: the deep per-face DAGs (held in RAM) + the
    /// residency selector that streams only the near-cut subset to the GPU pool.
    #[cfg(feature = "nanite")]
    nanite_assets: Vec<enki_nanite::cluster::ClusterAsset>,
    #[cfg(feature = "nanite")]
    nanite_streamer: Option<enki_nanite::residency::ClusterStreamer>,
    /// Planet radius (for the streamer's altitude-relative reselection threshold).
    #[cfg(feature = "nanite")]
    nanite_radius: f64,
    /// Frame index of the last streaming reselection (rate-limits the re-pack so
    /// fast zoom can't trigger a full re-pack every frame → no sustained fps drop).
    #[cfg(feature = "nanite")]
    nanite_last_resel: u64,
    /// In-progress async startup load (heightfield + Nanite bake); `None` once done.
    loader:       Option<loading::Loader>,
    /// Planet config held until the async load completes (then `PlanetView` is built).
    pending_planet_cfg: Option<PlanetConfig>,
    /// Per-stage load timing, captured when the loader finishes (for the popup).
    load_timings: Option<loading::LoadTimings>,
    /// Show the one-time load-stats popup until the user dismisses it.
    show_load_stats: bool,
    fps:          FpsMeter,
    mode:         RenderMode,
    last_tick:    Instant,
    minimized:    bool,
    /// Monotonic frame counter for LOD age tracking.
    frame_counter: u64,

    // ── Navigation ────────────────────────────────────────────────────────
    nav:          NavState,
    globe:        GlobeControls,
    /// Surface walker + chase camera (1st/3rd-person view toggle). Spawned by a
    /// placement click; `None` while orbiting.
    surface:      Option<ThirdPersonController>,
    /// Ground speed (m/s) the surface walker reported this frame; drives the gait.
    surface_speed: f32,
    /// Last cursor NDC (x right, y up, -1..1) for Placement ray-cast.
    cursor_ndc:   (f32, f32),
    /// Shared terrain height source (clone of the loader's) so the FPS controller
    /// can stand/walk on the surface. `None` until the async load completes.
    height_field: Option<Arc<dyn HeightField>>,
    /// Elevation scale (m) the renderer baked with — pairs with `height_field`.
    terrain_height_scale: f64,

    // ── View hot-swap ─────────────────────────────────────────────────────
    /// Unified view mode 0–10 (View: 0 Lit,1 Unlit,2 Normal,3 Triangle,4 Cluster,
    /// 5 LOD; Planet: 6 Plate,7 Height,8 Material,9 Wetness,10 Volcano).
    view_mode:     u32,
    wireframe:     bool,
    /// TAA toggle (off by default; the proven non-TAA path stays the default).
    taa_enabled:   bool,
    /// Sub-pixel jitter + view-proj history for the TAA resolve.
    taa_jitter:    enki_render::taa::TaaJitter,
    /// Nanite UI state.
    nanite_enabled:    bool,
    /// LOD pixel-error threshold (lower = finer/smoother, heavier).
    nanite_tau:        f32,
    /// Draw the translucent ocean shell over the planet.
    ocean_enabled:     bool,
    /// Draw the wind streakline overlay.
    wind_enabled:      bool,
    /// Draw the atmosphere shell.
    atmo_enabled:      bool,
    /// Atmosphere tunables (height + density).
    atmo_params:       AtmoParams,
    /// Reference marker toggles (pole spikes / equator ring).
    markers_poles:     bool,
    markers_equator:   bool,
    /// Draw the FFT spectral wave surface (camera-anchored detail).
    wave_enabled:      bool,
    /// Wave choppiness (horizontal displacement gain) — GUI-tunable.
    wave_choppiness:   f32,
    /// Foam threshold (Jacobian below which whitecaps form) — GUI-tunable.
    wave_foam:         f32,
    /// Sea level as a metre offset from the terrain's `e = 0` datum (= PLANET_RADIUS).
    sea_level_m:       f64,

    // ── Raw input state ───────────────────────────────────────────────────
    /// Accumulated mouse-motion delta between frames (pixels, CursorMoved).
    mouse_delta:   (f32, f32),
    /// Left mouse button currently held.
    lmb_held:      bool,
    /// Left-click fired this frame (rising edge, for Placement pick).
    lmb_click:     bool,
    /// True while the mouse button is held and the drag started on the 3D scene.
    scene_drag_active: bool,
    /// Previous cursor position in physical pixels; used to compute CursorMoved deltas.
    cursor_pos_prev: Option<(f64, f64)>,
    /// Scroll wheel delta accumulated this frame.
    scroll:        f32,
    /// WASD + Shift held state for surface-walk movement.
    move_keys:     MoveInput,

    // ── Soak-terrain state ────────────────────────────────────────────────
    /// Monotonic elapsed time since the soak started (seconds).  Drives the
    /// deterministic scripted camera path in `SoakTerrain` mode.
    soak_elapsed: f64,
    /// Wall-clock instant when the soak mode started, for fps measurement.
    soak_start: Instant,
    /// Rolling frame count used to compute average fps between log ticks.
    soak_frames_since_log: u64,
    /// Elapsed seconds at the last periodic log tick.
    soak_last_log_elapsed: f64,

    // rhi MUST be declared last: Rust drops fields in declaration order, and
    // rhi owns the VkDevice.  egui (above) holds Vulkan pipelines/textures
    // that must be freed before the device is destroyed.
    rhi: Option<Rhi>,
}

impl App {
    fn new(mode: RenderMode) -> Self {
        Self {
            window:        None,
            egui:          None,
            planet_view:   None,
            ocean:         None,
            sky:           SkyModel::earth(),
            system:        SolarSystem::nms_default(),
            bodies:        None,
            freecam:       None,
            wave:          None,
            wind:          None,
            atmosphere:    None,
            markers:       None,
            scene:         None,
            stress:        None,
            character:     None,
            #[cfg(feature = "flora")]
            flora_renderer: None,
            #[cfg(feature = "flora")]
            flora_tree:    None,
            #[cfg(feature = "flora")]
            flora_instances: Vec::new(),
            #[cfg(feature = "flora")]
            flora_scatter_center: None,
            #[cfg(feature = "flora")]
            flora_last_density: -1.0,
            #[cfg(feature = "flora")]
            flora_clock:   0.0,
            flora_enabled: true,
            flora_density: 0.5,
            #[cfg(feature = "nanite")]
            nanite:        None,
            #[cfg(feature = "nanite")]
            nanite_assets: Vec::new(),
            #[cfg(feature = "nanite")]
            nanite_streamer: None,
            #[cfg(feature = "nanite")]
            nanite_radius: 0.0,
            #[cfg(feature = "nanite")]
            nanite_last_resel: 0,
            loader:        None,
            pending_planet_cfg: None,
            load_timings:  None,
            show_load_stats: false,
            fps:           FpsMeter::new(),
            mode,
            last_tick:     Instant::now(),
            minimized:     false,
            frame_counter: 0,
            nav:           NavState::new(),
            globe:         GlobeControls::new(100_000.0),
            surface:       None,
            surface_speed: 0.0,
            cursor_ndc:    (0.0, 0.0),
            height_field:  None,
            terrain_height_scale: 0.0,
            view_mode:     0,
            wireframe:     false,
            taa_enabled:   true,
            taa_jitter:    enki_render::taa::TaaJitter::new(),
            nanite_enabled:    true,
            nanite_tau:        1.0,
            ocean_enabled:     true,
            wind_enabled:      true,
            atmo_enabled:      true,
            atmo_params:       AtmoParams::default(),
            markers_poles:     false,
            markers_equator:   false,
            wave_enabled:      true,
            wave_choppiness:   1.3,
            wave_foam:         0.4,
            sea_level_m:       0.0,
            mouse_delta:   (0.0, 0.0),
            lmb_held:      false,
            lmb_click:     false,
            scene_drag_active: false,
            cursor_pos_prev:   None,
            scroll:        0.0,
            move_keys:     MoveInput::default(),
            soak_elapsed:           0.0,
            soak_start:             Instant::now(),
            soak_frames_since_log:  0,
            soak_last_log_elapsed:  0.0,
            rhi:           None,
        }
    }

    // ── Camera and altitude helpers ───────────────────────────────────────

    fn active_camera(&self) -> Camera {
        match self.nav.mode() {
            NavMode::Globe | NavMode::Placement => self.globe.camera(),
            NavMode::Surface => self
                .surface
                .as_ref()
                .map(|tpc| tpc.camera())
                .unwrap_or_else(|| self.globe.camera()),
            NavMode::Space => self
                .freecam
                .as_ref()
                .map(|f| f.camera(self.system.focused().world_pos))
                .unwrap_or_else(|| self.globe.camera()),
        }
    }

    fn altitude_m(&self) -> f64 {
        match self.nav.mode() {
            NavMode::Globe | NavMode::Placement => self.globe.altitude(),
            NavMode::Surface => self
                .surface
                .as_ref()
                .map(|tpc| tpc.feet_position().length() - PLANET_RADIUS)
                .unwrap_or_else(|| self.globe.altitude()),
            NavMode::Space => self
                .freecam
                .as_ref()
                .map(|f| (f.position - self.system.focused().world_pos).length() - PLANET_RADIUS)
                .unwrap_or_else(|| self.globe.altitude()),
        }
    }

    // ── Navigation mode transitions ───────────────────────────────────────

    /// Tab: Globe→Placement, Placement→Globe (cancel), Surface→Globe.
    fn cycle_nav_mode(&mut self) {
        match self.nav.mode() {
            NavMode::Globe => {
                self.nav.begin_placement(&self.globe.camera());
                log::info!("Nav: Globe → Placement (click to drop the character)");
            }
            NavMode::Placement => {
                // Cancel placement: reset state machine to Globe.
                self.nav = NavState::new();
                log::info!("Nav: Placement cancelled → Globe");
            }
            NavMode::Surface => {
                self.exit_surface("Tab");
            }
            NavMode::Space => {
                self.nav.exit_space();
                self.freecam = None;
                self.release_cursor();
                log::info!("Nav: Space → Globe");
            }
        }
    }

    /// Grab and hide the cursor for first-person mouse-look.
    ///
    /// Tries `CursorGrabMode::Locked` first (preferred — pointer is confined and
    /// reports raw deltas on Windows/Linux).  Falls back to `Confined` if the
    /// platform does not support Locked (macOS).  Silently ignores errors if
    /// neither mode is available.
    fn grab_cursor(&self) {
        if let Some(window) = &self.window {
            let locked = window.set_cursor_grab(CursorGrabMode::Locked);
            if locked.is_err() {
                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
            }
            window.set_cursor_visible(false);
        }
    }

    /// Release the cursor and make it visible again.
    fn release_cursor(&self) {
        if let Some(window) = &self.window {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
            window.set_cursor_visible(true);
        }
    }

    /// Common exit-surface logic: drop the walker, restore the orbit camera.
    fn exit_surface(&mut self, reason: &str) {
        if let Some(restored) = self.nav.exit_surface() {
            let alt = (restored.position.length() - PLANET_RADIUS).max(0.0);
            self.globe = GlobeControls::new(alt);
        }
        self.surface = None;
        self.surface_speed = 0.0;
        self.release_cursor();
        log::info!("Nav: Surface → Globe ({})", reason);
    }

    /// Placement mode: ray-cast and spawn the surface (third-person) controller.
    fn try_place_surface(&mut self) {
        // Compute aspect ratio from window size.
        let aspect = self
            .window
            .as_ref()
            .map(|w| {
                let s = w.inner_size();
                if s.height > 0 { s.width as f32 / s.height as f32 } else { 16.0 / 9.0 }
            })
            .unwrap_or(16.0 / 9.0);

        let camera   = self.active_camera();
        let (nx, ny) = self.cursor_ndc;
        let (origin, dir) = surface_picker::camera_ray(&camera, nx, ny, aspect);

        if let Some(hit) = surface_picker::pick(origin, dir, DVec3::ZERO, PLANET_RADIUS) {
            log::info!(
                "Nav: surface pick ({:.0}, {:.0}, {:.0}) — spawning character",
                hit.point.x, hit.point.y, hit.point.z
            );
            let cam_fwd  = (camera.orientation * glam::Vec3::NEG_Z).as_dvec3();
            let up       = hit.normal;
            let projected = cam_fwd - up * cam_fwd.dot(up);
            let heading  = if projected.length_squared() > 1e-10 {
                projected.normalize().as_vec3()
            } else {
                glam::Vec3::Z
            };

            let Some(hf) = self.height_field.clone() else {
                log::warn!("Nav: terrain not loaded yet — cannot place character");
                return;
            };
            self.surface = Some(ThirdPersonController::new(
                hit.point,
                hf,
                PLANET_RADIUS,
                self.terrain_height_scale,
                NANITE_BAKE_RES,
                heading,
            ));
            self.surface_speed = 0.0;
            self.nav.point_picked();
            self.grab_cursor();
            log::info!("Nav: Placement → Surface (third-person; V toggles first-person)");
        } else {
            log::info!("Nav: placement click missed the planet");
        }
    }

    // ── Per-frame update ──────────────────────────────────────────────────

    /// Compute the scripted soak-terrain camera for the current elapsed time.
    ///
    /// # Camera path
    ///
    /// The path loops with period `SOAK_PERIOD_S` and has three phases:
    ///
    ///   0 → DESCENT_END   — exponential descent from 100 km orbit to 500 m skim altitude.
    ///   DESCENT_END → ASCENT_START — skim at low altitude with slow azimuth rotation.
    ///   ASCENT_START → SOAK_PERIOD_S — ascent back to 100 km orbit.
    ///
    /// A slow continuous azimuth drift (one revolution per two full periods) ensures
    /// the LOD tree sees genuinely different surface regions each loop.
    ///
    /// The polar angle is fixed at 45° from the north pole to keep camera-relative
    /// rendering stable (avoids the pole singularity in look_at).
    fn soak_camera(&self, elapsed: f64) -> Camera {
        use controls::globe::spherical_to_cartesian;
        use glam::{Quat, Vec3};

        const PLANET_RADIUS_M: f64 = controls::PLANET_RADIUS;

        // ── Altitude profile ─────────────────────────────────────────────
        const HIGH_ALT: f64   = 100_000.0; // 100 km orbit
        const LOW_ALT:  f64   =     500.0; // 500 m skim
        const SOAK_PERIOD_S: f64  = 40.0;  // one full loop in seconds
        const DESCENT_FRAC:  f64  = 0.35;  // fraction of period spent descending
        const SKIM_FRAC:     f64  = 0.25;  // fraction skimming at low alt
        // ascent occupies the remaining (1 - DESCENT_FRAC - SKIM_FRAC) of the period

        let phase = (elapsed % SOAK_PERIOD_S) / SOAK_PERIOD_S; // 0..1 within the period

        let altitude = if phase < DESCENT_FRAC {
            // Descent: exponential ease-in so LOD splits happen gradually then rapidly.
            let t = phase / DESCENT_FRAC; // 0..1
            let log_high = HIGH_ALT.ln();
            let log_low  = LOW_ALT.ln();
            (log_high + (log_low - log_high) * t).exp()
        } else if phase < DESCENT_FRAC + SKIM_FRAC {
            // Skim: constant low altitude.
            LOW_ALT
        } else {
            // Ascent: exponential ease-out back to orbit.
            let t = (phase - DESCENT_FRAC - SKIM_FRAC) / (1.0 - DESCENT_FRAC - SKIM_FRAC);
            let log_low  = LOW_ALT.ln();
            let log_high = HIGH_ALT.ln();
            (log_low + (log_high - log_low) * t).exp()
        };

        let radius = PLANET_RADIUS_M + altitude;

        // ── Azimuth: one full revolution every two periods ───────────────
        let theta = (elapsed / (SOAK_PERIOD_S * 2.0) * std::f64::consts::TAU) as f32;

        // ── Fixed polar angle: 45° from north pole ───────────────────────
        let phi = std::f32::consts::FRAC_PI_4;

        let pos = spherical_to_cartesian(radius, theta, phi);

        let forward = (-pos).normalize().as_vec3();
        let world_up = Vec3::Y;
        let right    = forward.cross(world_up).normalize();
        let up       = right.cross(forward).normalize();
        let orientation = Quat::from_mat3(&glam::Mat3::from_cols(right, up, -forward));

        Camera {
            position: pos,
            orientation,
            fov_y_radians: 60_f32.to_radians(),
            near: 0.5,
            far: 750_000.0,
        }
    }

    fn update_nav(&mut self, dt: f32) {
        let (dx, dy) = self.mouse_delta;
        match self.nav.mode() {
            NavMode::Globe | NavMode::Placement => {
                // Globe drag sourced from CursorMoved deltas, gated by scene_drag_active.
                if self.scene_drag_active && (dx != 0.0 || dy != 0.0) {
                    self.globe.on_drag(dx, dy);
                }
                if self.scroll.abs() > 0.01 {
                    self.globe.on_scroll(self.scroll);
                }
                self.globe.update(dt);

                // Placement: left-click fires a ray-cast.
                if self.nav.mode() == NavMode::Placement && self.lmb_click {
                    self.try_place_surface();
                }
            }
            NavMode::Surface => {
                if let Some(tpc) = self.surface.as_mut() {
                    // Raw mouse motion orbits the chase camera.
                    if dx.abs() > 0.01 || dy.abs() > 0.01 {
                        tpc.on_orbit(dx, dy);
                    }
                    if self.scroll.abs() > 0.01 {
                        tpc.on_zoom(self.scroll);
                    }
                    // Camera-relative WASD; remember the speed for the gait.
                    self.surface_speed = tpc.on_move(self.move_keys, dt);
                    // Advance the smoothed camera collision for this frame.
                    tpc.update(dt);
                }
            }
            NavMode::Space => {
                if let Some(fc) = self.freecam.as_mut() {
                    if dx.abs() > 0.01 || dy.abs() > 0.01 {
                        fc.look(dx, dy);
                    }
                    if self.scroll.abs() > 0.01 {
                        fc.adjust_speed(self.scroll);
                    }
                    let fwd = (self.move_keys.forward as i32 - self.move_keys.backward as i32) as f32;
                    let rgt = (self.move_keys.right as i32 - self.move_keys.left as i32) as f32;
                    let boost = if self.move_keys.sprint { 6.0 } else { 1.0 };
                    fc.fly(fwd, rgt, boost, dt);
                }
            }
        }

        // Clear per-frame accumulators.
        self.mouse_delta = (0.0, 0.0);
        self.lmb_click   = false;
        self.scroll      = 0.0;
    }
}

// ── ApplicationHandler ────────────────────────────────────────────────────────

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let title = match self.mode {
            RenderMode::Planet      => "enki",
            RenderMode::Markers     => "enki — markers",
            RenderMode::Stress      => "enki — stress-streaming",
            RenderMode::SoakTerrain => "enki — soak-terrain",
        };

        let attrs = WindowAttributes::default()
            .with_title(title)
            .with_inner_size(LogicalSize::new(1280u32, 720u32))
            .with_resizable(true);

        let window = match event_loop.create_window(attrs) {
            Ok(w) => w,
            Err(e) => {
                log::error!("Failed to create window: {e}");
                event_loop.exit();
                return;
            }
        };

        log::info!("Window created (1280x720)");

        let config = RhiConfig {
            uniform_buffer_size: std::mem::size_of::<FrameUniforms>() as u64,
            ..RhiConfig::default()
        };

        let mut rhi = match Rhi::new(&window, config) {
            Ok(r) => r,
            Err(e) => {
                log::error!("Rhi::new failed: {e}");
                event_loop.exit();
                return;
            }
        };

        log::info!("RHI initialised");

        let color_format = rhi.swapchain_format();
        let samples      = rhi.msaa_samples();

        match self.mode {
            RenderMode::Planet | RenderMode::SoakTerrain => {
                // Build the terrain pipelines for PlanetView.
                use enki_render::{terrain_pass::TERRAIN_WGSL, material::ChunkPush};
                use enki_rhi::{GraphicsPipelineDesc, ShaderModule};
                let push_size = std::mem::size_of::<ChunkPush>() as u32;
                let set0 = rhi.set0_layout();

                let terrain_shader: ShaderModule = match rhi.create_shader_module(TERRAIN_WGSL) {
                    Ok(s) => s,
                    Err(e) => {
                        log::error!("Failed to compile terrain shader: {e}");
                        event_loop.exit();
                        return;
                    }
                };

                let terrain_pipeline = match rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
                    shader: terrain_shader,
                    vs_entry: "vs_main",
                    fs_entry: "fs_main",
                    push_constant_size: push_size,
                    set0_layout: set0,
                    color_format,
                    depth_format: enki_rhi::vk::Format::D32_SFLOAT,
                    samples,
                    blend: false,
                    fill: true,
                }) {
                    Ok(p) => p,
                    Err(e) => {
                        log::error!("Failed to create terrain pipeline: {e}");
                        event_loop.exit();
                        return;
                    }
                };

                let terrain_wireframe_pipeline = match rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
                    shader: terrain_shader,
                    vs_entry: "vs_main",
                    fs_entry: "fs_main",
                    push_constant_size: push_size,
                    set0_layout: set0,
                    color_format,
                    depth_format: enki_rhi::vk::Format::D32_SFLOAT,
                    samples,
                    blend: false,
                    fill: false,
                }) {
                    Ok(p) => p,
                    Err(e) => {
                        log::error!("Failed to create terrain wireframe pipeline: {e}");
                        event_loop.exit();
                        return;
                    }
                };

                rhi.destroy_shader_module(terrain_shader);

                // Ocean shell at the terrain's sea-level datum (e = 0 = PLANET_RADIUS).
                match Ocean::new(&mut rhi, color_format, samples, PLANET_RADIUS) {
                    Ok(o)  => self.ocean = Some(o),
                    Err(e) => log::error!("Ocean::new failed: {e}"),
                }
                // FFT spectral wave surface (camera-anchored detail patch).
                match WaveSurface::new(&mut rhi, color_format, samples, PLANET_RADIUS) {
                    Ok(w)  => self.wave = Some(w),
                    Err(e) => log::error!("WaveSurface::new failed: {e}"),
                }
                // Wind streakline overlay (rides the baked wind field).
                match WindOverlay::new(&mut rhi, color_format, samples, PLANET_RADIUS) {
                    Ok(w)  => self.wind = Some(w),
                    Err(e) => log::error!("WindOverlay::new failed: {e}"),
                }
                // Atmosphere shell (halo + the altitude the wind rides in).
                match Atmosphere::new(&mut rhi, color_format, samples, PLANET_RADIUS) {
                    Ok(a)  => self.atmosphere = Some(a),
                    Err(e) => log::error!("Atmosphere::new failed: {e}"),
                }
                // Equator + pole reference markers.
                match Markers::new(&mut rhi, color_format, samples, PLANET_RADIUS) {
                    Ok(m)  => self.markers = Some(m),
                    Err(e) => log::error!("Markers::new failed: {e}"),
                }
                // Solar-system bodies (sun + distant planets as lit spheres).
                match BodyRenderer::new(&mut rhi, color_format, samples) {
                    Ok(b)  => self.bodies = Some(b),
                    Err(e) => log::error!("BodyRenderer::new failed: {e}"),
                }
                // Third-person character (drawn only while walking the surface).
                match Character::new(&mut rhi, color_format, samples) {
                    Ok(c)  => self.character = Some(c),
                    Err(e) => log::error!("Character::new failed: {e}"),
                }
                // Procedural trees: flora-owned sub-renderer drawing INTO the app's
                // 3D pass (in_scene = true → swapchain format + ACES-in-shader so it
                // matches terrain). Trees are spawned lazily once on the surface.
                #[cfg(feature = "flora")]
                match FloraRenderer::new(&mut rhi, true) {
                    Ok(r)  => self.flora_renderer = Some(r),
                    Err(e) => log::error!("FloraRenderer::new failed: {e}"),
                }

                let cfg = PlanetConfig {
                    lod: LodConfig {
                        radius:        50_000.0,
                        height_scale:  1_200.0,
                        resolution:    32,
                        max_depth:     12,
                        target_tri_px: 2.0,
                        hysteresis:    0.15,
                        lru_capacity:  1024,
                    },
                    seed:      42,
                    n_workers: 4,
                    terrain_pipeline,
                    terrain_wireframe_pipeline,
                    // Tectonic terrain — matches demiurge defaults.
                    use_tectonics:     true,
                    plate_count:       12,
                    arc_density:       1.0,
                    hotspot_count:     8,
                    hotspot_intensity: 1.0,
                };

                // Climate params are baked into TectonicHeightField at startup
                // and are not retained in PlanetView after construction.
                let climate = ClimateParams {
                    seed:           42,
                    base_temp:      15.0,
                    atmosphere:     0.6,
                    band_count:     3,
                    axial_tilt_rad: 23.0_f64.to_radians(),
                    redistribution: None,
                    greenhouse:     None,
                    lapse_rate:     None,
                    swirl_strength: None,
                    n_high:         None,
                    n_low:          None,
                    cross_isobar_max: None,
                    sigma_base:     None,
                    lat_spread:     None,
                    retrograde:     None,
                    equator_taper_width: None,
                };

                // Kick off the async load (heightfield + 6 deep per-face Nanite
                // DAGs) on a worker thread. The main loop renders a progress bar
                // until it completes, then builds PlanetView + uploads the resident
                // Nanite DAGs on the main thread. The heightfield is built ONCE.
                let params = loading::LoadParams {
                    seed: cfg.seed,
                    use_tectonics: cfg.use_tectonics,
                    plate_count: cfg.plate_count,
                    arc_density: cfg.arc_density,
                    hotspot_count: cfg.hotspot_count,
                    hotspot_intensity: cfg.hotspot_intensity,
                    climate,
                    radius: cfg.lod.radius,
                    height_scale: cfg.lod.height_scale,
                    // Stage 2: bake DEEP (DAG exceeds the GPU pool); only the
                    // near-cut subset is streamed resident per frame.
                    nanite_resolution: NANITE_BAKE_RES,
                };
                if matches!(self.mode, RenderMode::SoakTerrain) {
                    self.soak_start = Instant::now();
                }
                self.loader = Some(loading::Loader::spawn(params));
                self.pending_planet_cfg = Some(cfg);
                log::info!("Async load started (heightfield + Nanite bake on a worker thread)");
            }
            RenderMode::Markers => {
                match Scene::new(&mut rhi, color_format, samples) {
                    Ok(s) => {
                        log::info!("Scene built (markers mode)");
                        self.scene = Some(s);
                    }
                    Err(e) => {
                        log::error!("Scene::new failed: {e}");
                        event_loop.exit();
                        return;
                    }
                }
            }
            RenderMode::Stress => {
                match StressHarness::new(&mut rhi, color_format, samples) {
                    Ok(h) => {
                        log::info!("Stress harness built");
                        self.stress = Some(h);
                    }
                    Err(e) => {
                        log::error!("StressHarness::new failed: {e}");
                        event_loop.exit();
                        return;
                    }
                }
            }
        }

        // Build egui state — needs the window for HasDisplayHandle (clipboard).
        self.egui = Some(EguiState::new(&rhi, &window));
        // egui draws at TYPE_1 in its own 1-sample pass on the resolved swapchain image.
        log::info!("egui initialised (1-sample UI pass, swapchain format {:?})", rhi.swapchain_format());

        // Initialise the TAA resolve resources (toggle starts OFF; the proven
        // non-TAA path remains the default until the user enables it).
        match rhi.create_shader_module(enki_render::taa::TAA_RESOLVE_WGSL) {
            Ok(module) => {
                let ubo_size = std::mem::size_of::<enki_render::taa::ResolveParams>() as u64;
                if let Err(e) = rhi.init_taa(&module, "vs_fullscreen", "fs_resolve", ubo_size) {
                    log::error!("TAA init failed: {e}");
                }
                rhi.destroy_shader_module(module);
            }
            Err(e) => log::error!("TAA resolve shader compile failed: {e}"),
        }

        self.last_tick = Instant::now();
        self.window    = Some(window);
        self.rhi       = Some(rhi);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        // ── 1. Feed to egui FIRST ─────────────────────────────────────────────
        // We need both window and egui; extract them carefully.
        let egui_consumed = match (self.egui.as_mut(), self.window.as_ref()) {
            (Some(eg), Some(w)) => eg.on_window_event(w, &event),
            _ => false,
        };

        // ── 2. Non-render events (always handled, egui consumed irrelevant) ────
        match &event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested");
                if let Some(rhi) = &self.rhi {
                    if let Err(e) = rhi.wait_idle() {
                        log::error!("wait_idle failed: {e}");
                    }
                    // Free caller-owned GPU resources before RHI teardown.
                    if let Some(wave) = &self.wave {
                        wave.destroy(rhi);
                    }
                }
                event_loop.exit();
                return;
            }

            WindowEvent::Resized(size) => {
                self.minimized = size.width == 0 || size.height == 0;
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
                return;
            }

            WindowEvent::RedrawRequested => {
                // Handled below — fall through.
            }

            _ => {}
        }

        // ── 3. Nav input (only when egui did not capture) ─────────────────────
        if !egui_consumed {
            self.handle_nav_window_event(&event);
        }

        // ── 4. Render ─────────────────────────────────────────────────────────
        if matches!(event, WindowEvent::RedrawRequested) {
            self.render_frame();
        }
    }

    /// Raw device events — mouse motion lives here in winit 0.30.
    ///
    /// Globe/Placement drag now sources from `WindowEvent::CursorMoved` (reliable
    /// on WSLg/Xwayland).  This path is kept alive for `NavMode::Surface`
    /// chase-cam orbit, which uses locked-cursor raw deltas that don't emit CursorMoved.
    ///
    /// NOTE: Surface raw-input may have the same WSLg reliability issue —
    /// tracked as a separate follow-up.
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id:  DeviceId,
        event:       DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta: (dx, dy) } = event {
            if matches!(self.nav.mode(), NavMode::Surface | NavMode::Space) {
                self.mouse_delta.0 += dx as f32;
                self.mouse_delta.1 += dy as f32;
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

// ── Navigation input (window events only — MouseMotion is in device_event) ───

impl App {
    fn handle_nav_window_event(&mut self, event: &WindowEvent) {
        match event {
            // Track cursor position as NDC for Placement ray-cast, and accumulate
            // CursorMoved deltas for orbit drag while scene_drag_active.
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(w) = &self.window {
                    let s = w.inner_size();
                    if s.width > 0 && s.height > 0 {
                        let nx =  (position.x as f32 / s.width  as f32) * 2.0 - 1.0;
                        let ny = -((position.y as f32 / s.height as f32) * 2.0 - 1.0);
                        self.cursor_ndc = (nx, ny);
                    }
                }
                if self.scene_drag_active {
                    if let Some((px, py)) = self.cursor_pos_prev {
                        self.mouse_delta.0 += (position.x - px) as f32;
                        self.mouse_delta.1 += (position.y - py) as f32;
                    }
                }
                self.cursor_pos_prev = Some((position.x, position.y));
            }

            // LMB state (drag and placement click).
            WindowEvent::MouseInput { button: MouseButton::Left, state, .. } => {
                let pressed = *state == ElementState::Pressed;
                if pressed && !self.lmb_held {
                    self.lmb_click = true; // rising edge
                    // Press landed on the 3D scene (egui consumed check is upstream).
                    self.scene_drag_active = true;
                    self.cursor_pos_prev = None; // fresh baseline; first CursorMoved sets it
                }
                if !pressed {
                    self.scene_drag_active = false;
                }
                self.lmb_held = pressed;
            }

            // Scroll wheel → Globe zoom.
            WindowEvent::MouseWheel { delta, .. } => {
                self.scroll += match delta {
                    MouseScrollDelta::LineDelta(_, y)   => *y,
                    MouseScrollDelta::PixelDelta(p)     => p.y as f32 * 0.01,
                };
            }

            // Keyboard input.
            WindowEvent::KeyboardInput {
                event: KeyEvent {
                    physical_key: PhysicalKey::Code(code),
                    state,
                    repeat: false,
                    ..
                },
                ..
            } => {
                let pressed = *state == ElementState::Pressed;
                match code {
                    // ── Nav mode cycle ────────────────────────────────────────
                    KeyCode::Tab if pressed => {
                        self.cycle_nav_mode();
                    }

                    // ── Exit surface walker ───────────────────────────────────
                    KeyCode::Escape if pressed => {
                        if self.nav.mode() == NavMode::Surface {
                            self.exit_surface("Escape");
                        }
                    }

                    // ── Toggle 1st/3rd-person camera on the surface walker ─────
                    KeyCode::KeyV if pressed && self.nav.mode() == NavMode::Surface => {
                        if let Some(tpc) = self.surface.as_mut() {
                            tpc.toggle_view();
                        }
                    }

                    // ── Toggle free-flight space camera (Globe ↔ Space) ───────
                    KeyCode::KeyF if pressed => match self.nav.mode() {
                        NavMode::Globe => {
                            let terra = self.system.focused().world_pos;
                            let helio = terra + self.globe.camera().position;
                            self.freecam = Some(FreeCam::looking_at(helio, terra));
                            self.nav.enter_space();
                            self.grab_cursor();
                            log::info!("Nav: Globe → Space (free flight — WASD + mouse, scroll = speed)");
                        }
                        NavMode::Space => {
                            self.nav.exit_space();
                            self.freecam = None;
                            self.release_cursor();
                            log::info!("Nav: Space → Globe");
                        }
                        _ => {}
                    },

                    // ── View mode cycle ───────────────────────────────────────
                    KeyCode::KeyM if pressed => {
                        #[cfg(feature = "nanite")]
                        let nanite_on = self.nanite_enabled && self.nanite.is_some();
                        #[cfg(not(feature = "nanite"))]
                        let nanite_on = false;
                        loop {
                            // 0–10 View/Planet, 11–12 Ocean (Surface/Intensity).
                            self.view_mode = (self.view_mode + 1) % 13;
                            // Skip the Nanite-only geometry views (3–5) on the classic path.
                            if nanite_on || !(3..=5).contains(&self.view_mode) {
                                break;
                            }
                        }
                        log::info!("View mode → {}", self.view_mode);
                    }

                    // ── Wireframe toggle ──────────────────────────────────────
                    // Use W in Globe/Placement mode for wireframe; on the surface
                    // W is forward movement.
                    KeyCode::KeyW if pressed && matches!(self.nav.mode(), NavMode::Globe | NavMode::Placement) => {
                        self.wireframe = !self.wireframe;
                        log::info!("Wireframe → {}", self.wireframe);
                    }

                    // ── Surface-walk WASD ─────────────────────────────────────
                    KeyCode::KeyW => { self.move_keys.forward  = pressed; }
                    KeyCode::KeyA => { self.move_keys.left     = pressed; }
                    KeyCode::KeyS => { self.move_keys.backward = pressed; }
                    KeyCode::KeyD => { self.move_keys.right    = pressed; }
                    KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                        self.move_keys.sprint = pressed;
                    }

                    _ => {}
                }
            }

            _ => {}
        }
    }
}

// ── Render frame ──────────────────────────────────────────────────────────────

impl App {
    /// Render the loading screen: clear + a centered egui progress bar.
    fn render_loading(&mut self, prog: &loading::LoadProgress) {
        let rhi = match self.rhi.as_mut() {
            Some(r) => r,
            None => return,
        };
        let fi = match rhi.begin_frame() {
            Ok(i) => i,
            Err(e) => {
                log::debug!("begin_frame (loading) skipped: {e}");
                return;
            }
        };
        if fi == u32::MAX {
            let _ = rhi.end_frame(fi);
            return;
        }

        if let (Some(egui), Some(window)) = (self.egui.as_mut(), self.window.as_ref()) {
            egui.loading_frame(window, rhi, prog.fraction, &prog.message);
        }

        rhi.begin_rendering(fi);
        rhi.set_viewport_scissor_full(fi);
        rhi.begin_ui_pass(fi);
        if let Some(egui) = self.egui.as_mut() {
            egui.render(rhi, fi);
        }
        if let Err(e) = rhi.end_frame(fi) {
            log::error!("end_frame (loading) error: {e}");
        }
    }

    fn render_frame(&mut self) {
        if self.minimized {
            return;
        }

        // ── Delta time ────────────────────────────────────────────────────
        let now = Instant::now();
        let dt  = now.duration_since(self.last_tick).as_secs_f32().min(0.1);
        // Smoothed/throttled frame time for the egui FPS readout (raw dt drives motion).
        self.fps.tick(dt);
        self.sky.tick(dt);
        self.system.advance(self.sky.t_seconds);
        let frame_time = self.fps.frame_time_s();
        self.last_tick = now;

        // ── Async loading: show a progress bar until the worker finishes ──
        if self.loader.is_some() {
            match self.loader.as_ref().and_then(|l| l.poll()) {
                Some(out) => {
                    // Worker finished — assemble on the main thread (GPU uploads
                    // need the RHI). Then fall through to normal rendering.
                    self.load_timings = Some(out.timings);
                    self.show_load_stats = true;
                    if let Some(cfg) = self.pending_planet_cfg.take() {
                        #[cfg(feature = "nanite")]
                        {
                            self.nanite_radius = cfg.lod.radius;
                        }
                        // Keep a handle to the terrain so the FPS controller can ride it.
                        self.height_field = Some(Arc::clone(&out.hf));
                        // Feed the planet's wind field to the ocean (wave height/intensity).
                        if let Some(w) = self.wave.as_mut() {
                            w.set_wind_source(Arc::clone(&out.hf));
                        }
                        if let Some(w) = self.wind.as_mut() {
                            w.set_wind_source(Arc::clone(&out.hf));
                        }
                        self.terrain_height_scale = cfg.lod.height_scale;
                        self.planet_view = Some(planet_view_from_hf(cfg, out.hf));
                    }


                    // Nanite (Stage 2): the deep per-face DAGs live in RAM; the GPU
                    // pool starts empty and the streamer uploads only the near-cut
                    // cluster subset each frame, so the DAG can exceed GPU memory.
                    #[cfg(feature = "nanite")]
                    if let Some(nodes) = out.nanite_asset {
                        if let Some(rhi) = self.rhi.as_mut() {
                            match enki_nanite::render::NaniteRenderer::new(rhi, &[]) {
                                Ok(r) => {
                                    let clusters: usize =
                                        nodes.iter().map(|a| a.cluster_count()).sum();
                                    log::info!(
                                        "Nanite Stage-2 ready ({} faces, {} clusters in DAG; streaming near-cut subset)",
                                        nodes.len(),
                                        clusters,
                                    );
                                    self.nanite_streamer = Some(
                                        enki_nanite::residency::ClusterStreamer::new(&nodes),
                                    );
                                    self.nanite_assets = nodes;
                                    self.nanite = Some(r);
                                }
                                Err(e) => log::error!("Nanite renderer init failed: {e}"),
                            }
                        }
                    }
                    self.loader = None;
                    log::info!("Async load complete.");
                }
                None => {
                    let prog = self.loader.as_ref().unwrap().progress();
                    self.render_loading(&prog);
                    return;
                }
            }
        }

        // ── Navigation update (skipped in soak mode — camera is scripted) ──
        if !matches!(self.mode, RenderMode::SoakTerrain) {
            self.update_nav(dt);
        } else {
            self.soak_elapsed += dt as f64;
            self.soak_frames_since_log += 1;
        }

        // ── Snapshot state for this frame ─────────────────────────────────
        let (camera, altitude) = if matches!(self.mode, RenderMode::SoakTerrain) {
            let cam = self.soak_camera(self.soak_elapsed);
            let alt = (cam.position.length() - controls::PLANET_RADIUS).max(0.0);
            (cam, alt)
        } else {
            (self.active_camera(), self.altitude_m())
        };
        let nav_mode      = self.nav.mode();
        let view_mode     = self.view_mode;
        // Ocean view modes (11–12) don't recolor the terrain — show it lit.
        let terrain_view  = if view_mode >= 11 { 0 } else { view_mode };
        let wireframe     = self.wireframe;
        let taa_on        = self.taa_enabled;
        // This frame's sub-pixel jitter (for rasterization); advance() rolls it
        // forward after the 3D draws so the resolve uses the matching history.
        let jitter_px     = if taa_on { self.taa_jitter.jitter_px() } else { glam::Vec2::ZERO };

        // ── Planet LOD stats (for HUD + egui) ────────────────────────────
        let planet_stats: Option<PlanetViewStats> = self
            .planet_view
            .as_ref()
            .map(|pv| pv.stats());

        // ── Build egui frame (OUTSIDE rendering instance) ─────────────────
        // Egui texture uploads happen here via one-shot submit.
        // This must run before begin_frame → begin_rendering.
        let ui_out = if self.egui.is_some() && self.window.is_some() && self.rhi.is_some() {
            let nanite_enabled    = self.nanite_enabled;
            let nanite_tau        = self.nanite_tau;
            let ocean_enabled     = self.ocean_enabled;
            let wind_enabled      = self.wind_enabled;
            let atmo_enabled      = self.atmo_enabled;
            let atmo_params       = self.atmo_params;
            let markers_poles     = self.markers_poles;
            let markers_equator   = self.markers_equator;
            let wind_params       = self.wind.as_ref().map(|w| w.params).unwrap_or_default();
            let sea_level_m       = self.sea_level_m;
            let wave_enabled      = self.wave_enabled;
            let wave_choppiness   = self.wave_choppiness;
            let wave_foam         = self.wave_foam;
            let flora_enabled     = self.flora_enabled;
            let flora_density      = self.flora_density;
            // LoadTimings is Copy — snapshot it out so the popup can borrow it while
            // egui mutably borrows self. Only passed while the popup is showing.
            let load_stats        = if self.show_load_stats { self.load_timings } else { None };
            // egui, window, and rhi are independent fields — split-borrow.
            let egui   = self.egui.as_mut().unwrap();
            let window = self.window.as_ref().unwrap();
            let rhi    = self.rhi.as_ref().unwrap();

            Some(egui.build_frame(
                window, rhi, nav_mode, altitude, frame_time, view_mode, wireframe, taa_on,
                nanite_enabled, nanite_tau, cfg!(feature = "nanite"),
                ocean_enabled, sea_level_m, wave_enabled, wave_choppiness, wave_foam,
                cfg!(feature = "flora"), flora_enabled, flora_density,
                self.sky.time_scale, self.sky.paused,
                wind_enabled, wind_params,
                atmo_enabled, atmo_params, markers_poles, markers_equator,
                None, planet_stats.as_ref(), load_stats.as_ref(),
            ))
        } else {
            None
        };

        // Apply the debug panel's control changes back to app state.  The 3D view
        // this frame still uses the pre-build snapshot (1-frame latency on toggles,
        // imperceptible); nav actions take effect via the &mut self calls below.
        if let Some(out) = ui_out {
            self.view_mode         = out.view_mode;
            self.wireframe         = out.wireframe;
            self.taa_enabled       = out.taa;
            self.nanite_enabled    = out.nanite_enabled;
            self.nanite_tau        = out.nanite_tau;
            // Nanite-only geometry views (3–5) collapse to Lit when the classic path is active.
            #[cfg(feature = "nanite")]
            let nanite_on = self.nanite_enabled && self.nanite.is_some();
            #[cfg(not(feature = "nanite"))]
            let nanite_on = false;
            if !nanite_on && (3..=5).contains(&self.view_mode) {
                self.view_mode = 0;
            }
            self.ocean_enabled     = out.ocean_enabled;
            self.sea_level_m       = out.sea_level_m;
            self.wave_enabled      = out.wave_enabled;
            self.wave_choppiness   = out.wave_choppiness;
            self.wave_foam         = out.wave_foam;
            self.flora_enabled     = out.flora_enabled;
            self.flora_density     = out.flora_density;
            self.wind_enabled      = out.wind_enabled;
            if let Some(w) = self.wind.as_mut() {
                w.params = out.wind;
            }
            self.atmo_enabled      = out.atmo_enabled;
            self.atmo_params       = out.atmo;
            self.markers_poles     = out.markers_poles;
            self.markers_equator   = out.markers_equator;
            self.sky.time_scale    = out.time_scale;
            self.sky.paused        = out.paused;
            if out.cycle_nav {
                self.cycle_nav_mode();
            }
            if out.exit_surface && self.nav.mode() == NavMode::Surface {
                self.exit_surface("UI button");
            }
            if out.dismiss_load_stats {
                self.show_load_stats = false;
            }
        }


        // ── Acquire aspect + screen height from RHI ───────────────────────
        let (aspect, screen_h_px) = self.rhi.as_ref().map(|r| {
            let ext = r.extent();
            let h = ext.height;
            let a = if h > 0 { ext.width as f32 / h as f32 } else { 16.0 / 9.0 };
            (a, h as f32)
        }).unwrap_or((16.0 / 9.0, 720.0));

        // ── Frame begin ───────────────────────────────────────────────────
        let rhi = match self.rhi.as_mut() { Some(r) => r, None => return };

        // Select the TAA path for this frame (must be set before begin_frame, which
        // branches its image-layout barriers on it). No-op until init_taa ran.
        rhi.set_taa_enabled(taa_on);

        let fi = match rhi.begin_frame() {
            Ok(idx) => idx,
            Err(RhiError::Vulkan(vk_result)) => {
                log::debug!("begin_frame swapchain event ({vk_result:?}) — skipping");
                if let Some(w) = &self.window { w.request_redraw(); }
                return;
            }
            Err(e) => {
                log::error!("begin_frame error: {e}");
                return;
            }
        };

        if fi == u32::MAX {
            if let Err(e) = rhi.end_frame(fi) {
                log::error!("end_frame error: {e}");
            }
            return;
        }

        // ── 3D rendering ──────────────────────────────────────────────────
        match self.mode {
            RenderMode::Planet => {
                // Build frame uniforms with rotation-only camera-relative view,
                // as documented in FrameUniforms::new. With TAA on, jitter the
                // projection sub-pixel (terrain + Nanite both read this view-proj).
                // Sun direction from the focused body's heliocentric position. The
                // planet-spin (day/night) applies ONLY when on/orbiting the planet; in
                // free space the sun is FIXED (heliocentric), so spin = identity.
                let spin = if nav_mode == NavMode::Space {
                    glam::Quat::IDENTITY
                } else {
                    self.sky.helio_to_body()
                };
                let sun_dir_body = spin
                    .mul_vec3(self.system.sun_dir_focused().as_vec3())
                    .normalize_or_zero();
                let lights = Lights::from_sun(sun_dir_body, self.sky.sun_light_color());
                let fu     = if taa_on {
                    FrameUniforms::new_jittered(
                        &camera, aspect, &lights, jitter_px, aspect * screen_h_px, screen_h_px,
                    )
                } else {
                    FrameUniforms::new(&camera, aspect, &lights)
                };

                // Build the LOD camera with full world view-proj for frustum culling.
                let proj = reversed_z_perspective(
                    camera.fov_y_radians, aspect, camera.near, camera.far,
                );
                let view_world      = camera.view_matrix();
                let view_proj_local = proj * view_world;

                let lod_cam = LodCamera {
                    local_pos:       camera.position,
                    v_fov_rad:       camera.fov_y_radians,
                    screen_h_px,
                    view_proj_local,
                };
                let camera_world_pos = camera.position;

                // Mutual exclusivity: Nanite is an LOD system that REPLACES the
                // quadtree terrain. When it's active, the quadtree is not rendered.
                #[cfg(feature = "nanite")]
                let nanite_active = self.nanite_enabled && self.nanite.is_some();
                #[cfg(not(feature = "nanite"))]
                let nanite_active = false;

                // PlanetView::update — streaming uploads (outside rendering instance).
                if !nanite_active {
                    if let Some(pv) = self.planet_view.as_mut() {
                        pv.update(rhi, fi, self.frame_counter, &lod_cam, camera_world_pos);
                    }
                }

                // Nanite (Stage 2): stream the near-cut cluster subset, then per-
                // frame uniforms + cull dispatch (MUST be outside the rendering
                // instance — compute dispatch is illegal inside it).
                #[cfg(feature = "nanite")]
                if nanite_active {
                    // 1. Reselect the resident set (throttled), then INCREMENTALLY
                    //    reconcile the GPU page pool: update_residency uploads only the
                    //    newly-added clusters and frees the removed ones (stable slots,
                    //    no whole-set re-pack) → no fast-zoom hitch.
                    {
                        let cot = 1.0 / (camera.fov_y_radians * 0.5).tan();
                        let altitude =
                            (camera_world_pos.length() - self.nanite_radius).max(1.0);
                        // Wide margin keeps the resident set valid between reselections.
                        let move_thresh = (altitude * 0.25).max(25.0);
                        // Rate-limit how often we recompute the (whole-DAG) selection.
                        const MIN_RESEL_FRAMES: u64 = 4;
                        let eligible = self.frame_counter.wrapping_sub(self.nanite_last_resel)
                            >= MIN_RESEL_FRAMES;
                        let changed = if eligible {
                            if let Some(s) = self.nanite_streamer.as_mut() {
                                s.update(
                                    camera_world_pos, self.nanite_tau, screen_h_px, cot, 2.5,
                                    move_thresh,
                                )
                                .is_some()
                            } else {
                                false
                            }
                        } else {
                            false
                        };
                        if changed {
                            self.nanite_last_resel = self.frame_counter;
                            let fif = rhi.frames_in_flight() as u64;
                            let completed = self.frame_counter;
                            let retire_at = self.frame_counter + fif;
                            if let (Some(n), Some(s)) =
                                (self.nanite.as_mut(), self.nanite_streamer.as_ref())
                            {
                                let _ = n.update_residency(
                                    rhi, completed, retire_at, &self.nanite_assets,
                                    s.entries(), s.selection(),
                                );
                            }
                        }
                    }
                    // 2. Per-frame uniforms (incl. this frame's active-slot list) + cull.
                    //    FrameUniforms gives the rotation-only camera-relative view-proj
                    //    AND the terrain's lights.
                    if let Some(n) = self.nanite.as_mut() {
                        // Dithered LOD cross-fade is only useful with TAA on (it
                        // resolves the per-frame stipple into a smooth blend).
                        let _ = n.update(
                            rhi, fi, camera_world_pos, &fu, screen_h_px,
                            camera.fov_y_radians, self.nanite_tau, terrain_view,
                            self.taa_enabled, self.frame_counter as u32,
                        );
                        let _ = n.record_cull(rhi, fi);
                    }
                }

                // Third-person character: advance the gait + skin + upload this
                // frame's verts (host-visible memcpy, fine outside the instance).
                // Drawn only while walking the surface in third-person view.
                let draw_character = nav_mode == NavMode::Surface
                    && self.surface.as_ref().is_some_and(|t| t.show_character());
                if draw_character {
                    if let Some(ch) = self.character.as_mut() {
                        if let Err(e) = ch.update(rhi, fi, self.surface_speed, dt) {
                            log::error!("Character::update error: {e}");
                        }
                    }
                }

                // Flora (procedural trees): build the shared species mesh once,
                // (re)scatter a grove around the player, then upload frame/wind
                // uniforms + keep the shadow map valid. All OUTSIDE begin_rendering
                // (host-visible uploads + a depth-only pass), mirroring
                // Character::update + the Nanite cull dispatch.
                #[cfg(feature = "flora")]
                let draw_trees =
                    nav_mode == NavMode::Surface && self.flora_enabled && self.flora_renderer.is_some();
                #[cfg(feature = "flora")]
                if draw_trees {
                    // 1. Build the species mesh once from the CANONICAL TreeSpec —
                    //    the SAME default the standalone flora_viewer starts from, so
                    //    the planet tree and the viewer tree are identical. Drawn at
                    //    many models — its stored origin is unused.
                    if self.flora_tree.is_none() {
                        if let (Some(tpc), Some(hf)) =
                            (self.surface.as_ref(), self.height_field.as_ref())
                        {
                            let dir = tpc.feet_position().normalize();
                            let r = controls::terrain_grid::ground_radius(
                                hf.as_ref(), self.terrain_height_scale, dir,
                            );
                            match FloraView::default_specimen(rhi, dir * r) {
                                Ok(tree) => {
                                    let single = matches!(tree.leaf_mode(), LeafMode::Single);
                                    if let Some(rend) = self.flora_renderer.as_mut() {
                                        let _ = rend.update_leaf_texture(
                                            rhi, &tree.leaf_genes(), single,
                                        );
                                    }
                                    self.flora_tree = Some(tree);
                                }
                                Err(e) => log::error!("FloraView::default_single failed: {e}"),
                            }
                        }
                    }
                    // 2. (Re)scatter when the player moved past half a cell or the
                    //    density changed (deterministic, so this never reshuffles).
                    if let (Some(tpc), Some(hf)) =
                        (self.surface.as_ref(), self.height_field.as_ref())
                    {
                        let center = tpc.feet_position();
                        let moved = self
                            .flora_scatter_center
                            .map_or(true, |p| {
                                (p - center).length() > flora_scatter::SPACING_M * 0.5
                            });
                        let density_changed =
                            (self.flora_density - self.flora_last_density).abs() > 1e-3;
                        if moved || density_changed {
                            self.flora_instances = flora_scatter::scatter(
                                hf.as_ref(), self.terrain_height_scale, center, self.flora_density,
                            );
                            self.flora_scatter_center = Some(center);
                            self.flora_last_density = self.flora_density;
                        }
                    }
                    // 3. Frame uniforms + ONE wind solve (shared by every instance —
                    //    same genome → same bones) + shadow-map validity (shadows off
                    //    in-scene: enabled (params.z) = 0 → fs skips the sample; the
                    //    empty depth pass just keeps set1's shadow map in SHADER_READ).
                    //    ponytail: the 2048² clear each frame is cheap; init once if
                    //    it ever shows on a profile.
                    self.flora_clock += dt;
                    if let (Some(rend), Some(tree)) =
                        (self.flora_renderer.as_mut(), self.flora_tree.as_mut())
                    {
                        let wind = [self.flora_clock, 0.6, 1.0, 0.0];
                        let _ = rend.set_frame_uniforms(rhi, fi, &fu);
                        let _ = rend.set_bone_matrices(rhi, fi, tree.solve_wind(wind));
                        let su = ShadowUniforms {
                            light_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
                            params: [1.0 / rend.shadow_map_size() as f32, 0.0, 0.0, 0.0],
                        };
                        let _ = rend.set_shadow_uniforms(rhi, fi, &su);
                        rend.begin_shadow_pass(rhi, fi);
                        rend.end_shadow_pass(rhi, fi);
                    }
                }

                // 3D opaque pass.
                rhi.begin_rendering(fi);
                rhi.set_viewport_scissor_full(fi);
                if !nanite_active {
                    if let Some(pv) = self.planet_view.as_mut() {
                        if let Err(e) = pv.record(rhi, fi, &fu, camera_world_pos, terrain_view, wireframe) {
                            log::error!("PlanetView::record error: {e}");
                        }
                    }
                }

                // Nanite: indirect, vertex-pulling draw (inside the rendering instance).
                #[cfg(feature = "nanite")]
                if nanite_active {
                    if let Some(n) = self.nanite.as_ref() {
                        let _ = n.record_draw(rhi, fi);
                    }
                }

                // Solar-system bodies (sun + distant planets) as lit spheres on the sky
                // shell; drawn after opaque terrain so reversed-Z lets the focused planet
                // limb eclipse them, before the near translucent layers.
                if let Some(bodies) = self.bodies.as_ref() {
                    if let Err(e) = bodies.record(rhi, fi, &fu, &camera, &self.system, spin) {
                        log::error!("BodyRenderer::record error: {e}");
                    }
                }

                // Atmosphere shell — after bodies, behind the foreground (the limb
                // halo against space + a soft silhouette).
                if self.atmo_enabled {
                    if let Some(a) = self.atmosphere.as_ref() {
                        if let Err(e) = a.record(rhi, fi, &fu, camera_world_pos, self.atmo_params) {
                            log::error!("Atmosphere::record error: {e}");
                        }
                    }
                }

                // Reference markers (equator ring + pole spikes), depth-tested so
                // the globe occludes the far side.
                if self.markers_poles || self.markers_equator {
                    let (poles, eq, sea) =
                        (self.markers_poles, self.markers_equator, self.sea_level_m);
                    if let Some(m) = self.markers.as_mut() {
                        if let Err(e) = m.record(rhi, fi, &fu, camera_world_pos, sea, poles, eq) {
                            log::error!("Markers::record error: {e}");
                        }
                    }
                }

                // Character (after terrain/Nanite, before the translucent ocean).
                if draw_character {
                    if let (Some(ch), Some(tpc)) =
                        (self.character.as_mut(), self.surface.as_ref())
                    {
                        let feet = tpc.feet_position();
                        let facing = tpc.facing();
                        if let Err(e) = ch.draw(rhi, fi, &fu, &camera, feet, facing) {
                            log::error!("Character::draw error: {e}");
                        }
                    }
                }

                // Procedural trees: drawn inside the opaque pass, after the
                // character, before the translucent ocean (shares reversed-Z depth).
                // One shared species mesh, one draw pair per scattered instance.
                #[cfg(feature = "flora")]
                if draw_trees {
                    if let (Some(rend), Some(tree)) =
                        (self.flora_renderer.as_ref(), self.flora_tree.as_ref())
                    {
                        let wind = [self.flora_clock, 0.6, 1.0, 0.0];
                        for inst in &self.flora_instances {
                            let model = flora_scatter::instance_model(
                                inst.origin, inst.yaw, inst.scale, camera_world_pos,
                            );
                            if let Err(e) = tree.record_at(rhi, rend, fi, model, wind) {
                                log::error!("FloraView::record_at error: {e}");
                                break;
                            }
                        }
                    }
                }

                // Ocean. With TAA on, begin_water_pass splits the 3D pass and the
                // shell + FFT waves render in one refractive 1× pass (matching colour,
                // depth-darkening, and refraction). Without TAA, fall back to the
                // simple alpha-blended shell in the MSAA pass.
                let mut water_split = false;
                if self.ocean_enabled {
                    let sea = self.sea_level_m;
                    // The expensive FFT patch is drawn only near the surface; the shell
                    // covers the rest. begin_water_pass returns false when TAA is off.
                    let draw_waves = self.wave_enabled && altitude < 20_000.0;
                    let split = self.wave.is_some() && rhi.begin_water_pass(fi).unwrap_or(false);
                    water_split = split;
                    if split {
                        let scene_view = rhi.refraction_src_view();
                        let scene_depth = rhi.refraction_depth_view();
                        let ext = rhi.extent();
                        let w = self.wave.as_mut().unwrap();
                        w.params.choppiness = self.wave_choppiness;
                        w.params.foam_threshold = self.wave_foam;
                        w.debug_intensity = view_mode == 12; // Ocean: Intensity
                        if let Err(e) = w.record(
                            rhi, fi, &fu, &camera, sea,
                            scene_view, scene_depth, (ext.width, ext.height), draw_waves,
                        ) {
                            log::error!("WaveSurface::record error: {e}");
                        }
                    } else if let Some(o) = self.ocean.as_ref() {
                        if let Err(e) = o.record(rhi, fi, &fu, camera_world_pos, sea) {
                            log::error!("Ocean::record error: {e}");
                        }
                    }
                }

                // Wind streakline overlay — drawn LAST so the opaque sea can't paint
                // over it (the streaks ride ~1.5 km up, the top atmosphere layer). In
                // the water-split 1× instance when TAA is on, else the MSAA pass.
                if self.wind_enabled {
                    let sea = self.sea_level_m;
                    if let Some(w) = self.wind.as_mut() {
                        if let Err(e) =
                            w.record(rhi, fi, &fu, camera_world_pos, sea, water_split)
                        {
                            log::error!("WindOverlay::record error: {e}");
                        }
                    }
                }
            }
            RenderMode::Markers => {
                rhi.begin_rendering(fi);
                if let Some(scene) = self.scene.as_mut() {
                    let fu = FrameUniforms::new(&camera, aspect, &scene.lights);
                    if let Err(e) = scene.record_frame(rhi, fi, &fu, &camera, view_mode, wireframe) {
                        log::error!("record_frame error: {e}");
                    }
                }
            }
            RenderMode::Stress => {
                let lights = Lights::demiurge_default();
                let fu     = FrameUniforms::new(&camera, aspect, &lights);
                if let Some(harness) = self.stress.as_mut() {
                    if let Err(e) = harness.upload_phase(rhi, fi) {
                        log::error!("stress upload_phase error: {e}");
                    }
                }
                rhi.begin_rendering(fi);
                if let Some(harness) = self.stress.as_mut() {
                    if let Err(e) = harness.draw_phase(rhi, fi, &fu) {
                        log::error!("stress draw_phase error: {e}");
                    }
                }
            }
            RenderMode::SoakTerrain => {
                // Scripted fly-through: builds LodCamera + FrameUniforms exactly
                // as Planet mode does, but with a deterministic soak camera position.
                let lights = Lights::demiurge_default();
                let fu     = FrameUniforms::new(&camera, aspect, &lights);

                let proj = reversed_z_perspective(
                    camera.fov_y_radians, aspect, camera.near, camera.far,
                );
                let view_world      = camera.view_matrix();
                let view_proj_local = proj * view_world;

                let lod_cam = LodCamera {
                    local_pos:       camera.position,
                    v_fov_rad:       camera.fov_y_radians,
                    screen_h_px,
                    view_proj_local,
                };
                let camera_world_pos = camera.position;

                if let Some(pv) = self.planet_view.as_mut() {
                    pv.update(rhi, fi, self.frame_counter, &lod_cam, camera_world_pos);
                }

                rhi.begin_rendering(fi);
                rhi.set_viewport_scissor_full(fi);
                if let Some(pv) = self.planet_view.as_mut() {
                    if let Err(e) = pv.record(rhi, fi, &fu, camera_world_pos, terrain_view, wireframe) {
                        log::error!("PlanetView::record error (soak): {e}");
                    }
                }
            }
        }

        self.frame_counter = self.frame_counter.wrapping_add(1);

        // ── Periodic planet LOD stats log (every ~5 s) ───────────────────
        // Logs resident chunk count and build queue to confirm the planet is
        // initializing in headless test runs.
        if matches!(self.mode, RenderMode::Planet) {
            // ~300 frames at 60 fps ≈ 5 s; use wrapping so it fires even on slow hw.
            if self.frame_counter % 300 == 1 {
                if let Some(pv) = self.planet_view.as_ref() {
                    let s = pv.stats();
                    log::info!(
                        "[planet] frame={} resident_chunks={} build_queue={} lod_levels={}-{}",
                        self.frame_counter, s.resident_count, s.build_queue_depth,
                        s.min_lod_level, s.max_lod_level,
                    );
                }
                #[cfg(feature = "nanite")]
                if self.nanite_enabled {
                    if let Some(n) = self.nanite.as_ref() {
                        log::info!(
                            "[nanite] enabled — visible clusters (prev frame): {}",
                            n.last_visible_clusters()
                        );
                    }
                }
            }
        }

        // ── Periodic soak stats log (every ~5 s, time-based) ─────────────
        if matches!(self.mode, RenderMode::SoakTerrain) {
            const SOAK_LOG_INTERVAL_S: f64 = 5.0;
            if self.soak_elapsed - self.soak_last_log_elapsed >= SOAK_LOG_INTERVAL_S {
                let wall_elapsed = self.soak_start.elapsed().as_secs_f64();
                let dt_since_log = (self.soak_elapsed - self.soak_last_log_elapsed).max(1e-6);
                let fps = self.soak_frames_since_log as f64 / dt_since_log;
                let graveyard_size = rhi.streaming_graveyard_size();

                if let Some(pv) = self.planet_view.as_ref() {
                    let s = pv.stats();
                    log::info!(
                        "[soak] elapsed={:.1}s wall={:.1}s frame={} fps={:.1} \
                         resident_chunks={} build_queue={} pending_upload={} \
                         lod_levels={}-{} graveyard={}",
                        self.soak_elapsed, wall_elapsed, self.frame_counter, fps,
                        s.resident_count, s.build_queue_depth, s.pending_upload_count,
                        s.min_lod_level, s.max_lod_level,
                        graveyard_size,
                    );
                }

                self.soak_last_log_elapsed = self.soak_elapsed;
                self.soak_frames_since_log = 0;
            }
        }

        // ── TAA resolve ──────────────────────────────────────────────────
        // Must run AFTER all 3D draws and BEFORE begin_ui_pass. Rolls the view-
        // proj/camera history forward, then reprojects + neighborhood-clamps +
        // blends the current frame against history and blits to the swapchain.
        // No-op when TAA is off (begin_rendering rendered straight to the swapchain).
        if taa_on {
            let proj = reversed_z_perspective(
                camera.fov_y_radians, aspect, camera.near, camera.far,
            );
            let unjittered_vp = proj * camera.view_matrix_rotation_only();
            self.taa_jitter.advance(unjittered_vp, camera.position);
            let params = self.taa_jitter.resolve_params(
                aspect * screen_h_px,
                screen_h_px,
                enki_render::taa::DEFAULT_HISTORY_BLEND,
            );
            if let Err(e) = rhi.taa_resolve(fi, params.as_bytes()) {
                log::error!("taa_resolve error: {e}");
            }
        }

        // ── Transition to 1-sample UI pass ───────────────────────────────
        // begin_ui_pass closes the MSAA 3D instance (non-TAA path), inserts a
        // resolve→load barrier on the swapchain image, and opens a 1-sample
        // rendering instance (loadOp=LOAD, no depth attachment). egui-ash-renderer
        // 0.11 hardcodes TYPE_1 samples, so it MUST draw in this separate pass.
        // (On the TAA path the 3D instance + swapchain were already handled by
        // taa_resolve; begin_ui_pass just opens the UI instance.)
        //
        // This call is always made (all render modes) so that end_frame always
        // closes a UI instance — maintaining the begin/end bracket invariant.
        rhi.begin_ui_pass(fi);

        // ── egui render (inside 1-sample UI pass) ─────────────────────────
        if let Some(egui) = self.egui.as_mut() {
            egui.render(rhi, fi);
        }

        // ── End frame ─────────────────────────────────────────────────────
        // end_frame closes the UI rendering instance opened by begin_ui_pass,
        // then submits and presents.
        if let Err(e) = rhi.end_frame(fi) {
            log::error!("end_frame error: {e}");
        }
    }
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("enki starting");

    let args: Vec<String> = std::env::args().collect();
    let stress_flag  = args.iter().any(|a| a == "--stress-streaming")
        || std::env::var("ENKI_STRESS_STREAMING").ok().as_deref() == Some("1");
    let markers_flag = args.iter().any(|a| a == "--markers");
    let soak_flag    = args.iter().any(|a| a == "--soak-terrain")
        || std::env::var("ENKI_SOAK_TERRAIN").ok().as_deref() == Some("1");

    let mode = if soak_flag {
        log::info!("Mode: soak-terrain (scripted fly-through, deterministic camera)");
        RenderMode::SoakTerrain
    } else if stress_flag {
        log::info!("Mode: stress-streaming");
        RenderMode::Stress
    } else if markers_flag {
        log::info!("Mode: markers (depth-proof scene)");
        RenderMode::Markers
    } else {
        log::info!("Mode: planet (default)");
        RenderMode::Planet
    };

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(mode);
    if let Err(e) = event_loop.run_app(&mut app) {
        log::error!("Event loop exited with error: {e}");
        std::process::exit(1);
    }

    log::info!("enki shut down");
}

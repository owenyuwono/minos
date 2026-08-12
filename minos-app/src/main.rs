// minos-app — winit 0.30 application shell with interactive navigation,
// material/view hot-swap, egui panel, and HUD.
//
// # Render loop overview
//   resumed()  → create window → Rhi::new → terrain pipelines + overlays
//             → async load (heightfield) → EguiState::new
//
//   RedrawRequested:
//     begin_frame() → terrain update → begin_rendering() →
//     terrain + overlays draw → EguiState::render() → end_frame()
//
// # Navigation mode cycle (Tab: Globe → Placement → Surface → Globe)
//   Globe       — orbit camera; LMB drag rotates, scroll zooms.
//   Placement   — shows orbit view; click to ray-cast a spawn point on the planet.
//   Surface     — third-person character: camera-relative WASD, mouse orbits the
//                 chase cam, scroll zooms the boom, V toggles first-person.
//   Escape      — return from Surface → Globe at any time.
//
// # View hot-swap keys (not consumed by egui)
//   M    — cycle view_mode (0–10; skips geometry-debug 3–5 on the classic path)
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
mod aerial;
mod clouds;
mod clouds_advect;
mod markers;
mod solar;
mod planet_view;
#[cfg(feature = "voxel")]
mod voxel_view;
#[cfg(feature = "flora")]
mod flora_scatter;

use std::sync::Arc;
use std::time::Instant;

use minos_planet::climate::ClimateParams;
use minos_planet::height::HeightField;
use minos_planet::lod::{LodCamera, LodConfig};
use minos_render::{camera::Camera, frame::FrameUniforms, lights::Lights, projection::{reversed_z_perspective, reversed_z_orthographic}, sky::SkyModel, system::SolarSystem};
use minos_rhi::{Rhi, RhiConfig, RhiError};
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
use minos_app::{
    flora_render::{FloraRenderer, ShadowUniforms},
    flora_view::{FloraView, LeafLod, LeafMode, TreeSpec},
};
use controls::{
    tangent::MoveInput,
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
use aerial::{Aerial, AerialParams};
use clouds::{Clouds, CloudParams};
use markers::Markers;
use solar::BodyRenderer;
use controls::space::FreeCam;
use planet_view::{PlanetConfig, PlanetView, PlanetViewStats, planet_view_from_hf};

/// Sun shadow cascades — MUST match `character.rs` SHADOW_CASCADES, the flora
/// `ShadowUniforms` cascade count, and the receiver shaders.
#[cfg_attr(not(feature = "voxel"), allow(dead_code))]
const SHADOW_CASCADES: u32 = 3;


// ── App state ─────────────────────────────────────────────────────────────────

struct App {
    // Fields drop in declaration order.  Vulkan resources that hold device
    // handles MUST drop before `rhi` (which owns the VkDevice).  Order:
    //   egui  — holds pipeline/textures on the device; must free while device is alive.
    //   planet_view — holds only opaque u64 BufferHandles freed by rhi's stores;
    //                    its order relative to rhi is fine, but keeping it before
    //                    rhi is clearest.
    //   rhi   — LAST: wait_idle + destroys the VkDevice.
    window:       Option<Window>,
    egui:         Option<EguiState>,
    planet_view:  Option<PlanetView>,
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
    /// Depth-aware atmospheric scattering (aerial perspective; water split, TAA-on only).
    aerial:       Option<Aerial>,
    /// Wind-driven volumetric clouds (drawn in the water split; TAA-on only).
    clouds:       Option<Clouds>,
    /// Equator + pole reference markers.
    markers:      Option<Markers>,
    /// CPU-skinned humanoid drawn in third-person surface mode (built at startup
    /// in Planet mode; only drawn while walking the surface in third-person view).
    character:    Option<Character>,
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
    /// Tree draw radius around the player (m, GUI slider). Far trees in this
    /// window collapse to a cheap impostor billboard (Tier-2 LOD). Feature-agnostic
    /// so the panel compiles without `flora`.
    flora_radius_m: f64,
    /// Radius used at the last scatter rebuild (rebuild on change).
    #[cfg(feature = "flora")]
    flora_last_radius_m: f64,
    /// Draw flora when the player is within this altitude of the surface (m, GUI
    /// slider) — so trees fade in during descent, not only in Surface mode.
    flora_alt_threshold_m: f64,
    /// Phase 2 voxel terrain: the on-demand transvoxel quadtree. When the `voxel`
    /// feature is on this REPLACES the classic quadtree (PlanetView) as the terrain.
    #[cfg(feature = "voxel")]
    voxel_view: Option<voxel_view::VoxelView>,
    /// In-progress async startup load (heightfield build); `None` once done.
    loader:       Option<loading::Loader>,
    /// Planet config held until the async load completes (then `PlanetView` is built).
    pending_planet_cfg: Option<PlanetConfig>,
    /// Per-stage load timing, captured when the loader finishes (for the popup).
    load_timings: Option<loading::LoadTimings>,
    /// Show the one-time load-stats popup until the user dismisses it.
    show_load_stats: bool,
    fps:          FpsMeter,
    /// CPU time recording the previous frame's 3D pass (ms) — shown in the Stats panel.
    cpu_ms:       f32,
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
    /// This frame's sun direction (body frame, toward the sun) — stashed so the TAA
    /// resolve's screen-space sun shadow can read it (it's built in the Planet arm).
    sun_dir_body: glam::Vec3,
    /// Sun shadow map: on/off + tuning. The light ortho is built
    /// camera-relative each frame; the map is allocated once with `shadow_size`.
    shadow_map_enabled: bool,
    shadow_size: u32,
    /// Cascade radius (m) — small = sharp (fewer metres over the same texels), at the
    /// cost of reach; and the light-space depth span (m).
    shadow_half_extent: f32,
    shadow_depth: f32,
    /// Reversed-Z ADD bias (kills acne) + along-normal offset (metres, helps slopes).
    shadow_depth_bias: f32,
    shadow_normal_bias: f32,
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
    /// Post-process anti-aliasing mode (chosen in the Settings panel).
    aa_mode:       crate::gui::AaMode,
    /// Voxel caves on/off + carve strength (GUI; feature-agnostic so the panel
    /// compiles without the `voxel` feature — the effect is what's gated).
    voxel_caves:       bool,
    voxel_cave_strength: f32,
    /// CDLOD geomorph region (fraction of each leaf's LOD range the morph spans).
    voxel_morph_region: f32,
    /// Mountain peak sharpness (0 rounded → 1 cusped/pointy).
    voxel_peak_sharpen: f32,
    /// Ground detail-normal knobs (terrain_csm.wgsl): strength (0 = off) + feature scale (m).
    voxel_detail_strength: f32,
    voxel_detail_scale: f32,
    /// POM parallax depth (m): 0 = flat normal-map only, higher = more apparent height.
    voxel_detail_depth: f32,
    /// Heightfield ground-roughness knob 0..1 (0 = smooth, 1 = strong human-scale relief).
    voxel_ground_rough: f64,
    /// Draw the translucent ocean shell over the planet.
    ocean_enabled:     bool,
    /// Draw the wind streakline overlay.
    wind_enabled:      bool,
    /// Draw the atmosphere shell.
    atmo_enabled:      bool,
    /// Atmosphere tunables (height + density).
    atmo_params:       AtmoParams,
    /// Depth-aware atmospheric scattering (aerial perspective + sky dome).
    aerial_enabled:    bool,
    aerial_params:     AerialParams,
    /// Draw the volumetric clouds.
    clouds_enabled:    bool,
    /// Cloud tunables (coverage / density / altitude / wind speed / …).
    cloud_params:      CloudParams,
    /// Real-elapsed seconds for cloud wind-advection (independent of sim time_scale).
    cloud_time:        f32,
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

    // rhi MUST be declared last: Rust drops fields in declaration order, and
    // rhi owns the VkDevice.  egui (above) holds Vulkan pipelines/textures
    // that must be freed before the device is destroyed.
    rhi: Option<Rhi>,
}

impl App {
    fn new() -> Self {
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
            aerial:        None,
            clouds:        None,
            markers:       None,
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
            flora_density: 0.65,
            // Default tree draw radius. flora_scatter (the source of RADIUS_M) is
            // only compiled under `flora`; the field itself is feature-agnostic so
            // the GUI compiles without it, hence the cfg split on the default.
            #[cfg(feature = "flora")]
            flora_radius_m: flora_scatter::RADIUS_M,
            #[cfg(not(feature = "flora"))]
            flora_radius_m: 250.0,
            #[cfg(feature = "flora")]
            flora_last_radius_m: -1.0,
            flora_alt_threshold_m: 4000.0,
            #[cfg(feature = "voxel")]
            voxel_view:    None,
            loader:        None,
            pending_planet_cfg: None,
            load_timings:  None,
            show_load_stats: false,
            fps:           FpsMeter::new(),
            cpu_ms:        0.0,
            last_tick:     Instant::now(),
            minimized:     false,
            frame_counter: 0,
            nav:           NavState::new(),
            globe:         GlobeControls::new(100_000.0),
            surface:       None,
            surface_speed: 0.0,
            sun_dir_body:  glam::Vec3::Y,
            shadow_map_enabled: true,
            shadow_size:   4096, // per-cascade; 3×4096²×FiF ≈ 400 MB (halve-on-VRAM-fail)
            shadow_half_extent: 20.0,  // BASE (cascade 0) half-extent → 20/40/80 m cascades
            shadow_depth:       200.0,
            shadow_depth_bias:  0.001,
            shadow_normal_bias: 0.03,  // base; scaled ×2^cascade in the shader
            cursor_ndc:    (0.0, 0.0),
            height_field:  None,
            terrain_height_scale: 0.0,
            view_mode:     0,
            wireframe:     false,
            aa_mode:       crate::gui::AaMode::Fxaa,
            voxel_caves:       true,
            voxel_cave_strength: 90.0,
            voxel_morph_region: 1.0,
            voxel_peak_sharpen: 0.5,
            voxel_detail_strength: 0.6,
            voxel_detail_scale: 1.5,
            voxel_detail_depth: 0.25,
            voxel_ground_rough: 0.6,
            ocean_enabled:     true,
            wind_enabled:      false,
            atmo_enabled:      true,
            atmo_params:       AtmoParams::default(),
            // Aerial + clouds are HIDDEN by default (they were the TAA water-split
            // passes; with TAA removed the atmosphere shell provides the sky). The GUI
            // toggles still exist to re-enable them for testing.
            aerial_enabled:    false,
            aerial_params:     AerialParams::default(),
            clouds_enabled:    false,
            cloud_params:      CloudParams::default(),
            cloud_time:        0.0,
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
                    // Camera-relative WASD; remember the speed for the gait. Hand the
                    // controller the live voxel collision surface FIRST, so feet AND the
                    // camera-occlusion march ride the SAME rendered triangles (no neck) — and
                    // the feet sit on the DRAWN mesh, not the analytic that sinks below the
                    // still-streaming coarse surface. Falls back to analytic until resident.
                    #[cfg(feature = "voxel")]
                    {
                        let collider = self.voxel_view.as_ref().map(|vv| vv.collider_dyn());
                        tpc.set_collider(collider);
                    }
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

        let attrs = WindowAttributes::default()
            .with_title("minos")
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

        {
            {
                // Build the terrain pipelines for PlanetView.
                use minos_render::{terrain_pass::TERRAIN_WGSL, material::ChunkPush};
                use minos_rhi::{GraphicsPipelineDesc, ShaderModule};
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
                    depth_format: minos_rhi::vk::Format::D32_SFLOAT,
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
                    depth_format: minos_rhi::vk::Format::D32_SFLOAT,
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
                // Atmospheric scattering (depth-aware aerial perspective + sky dome).
                match Aerial::new(&mut rhi, color_format, PLANET_RADIUS) {
                    Ok(a)  => self.aerial = Some(a),
                    Err(e) => log::error!("Aerial::new failed: {e}"),
                }
                // Volumetric clouds (fullscreen raymarch in the water split).
                match Clouds::new(&mut rhi, color_format, PLANET_RADIUS) {
                    Ok(c)  => self.clouds = Some(c),
                    Err(e) => log::error!("Clouds::new failed: {e}"),
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
                        // Stable LOD threshold (split at 64px). Finer (1.5/1.0) CHURNS even
                        // with a static camera + more mesher cores: the LodTree's wanted set
                        // oscillates faster than meshing settles, so leaves never converge
                        // (resident stuck ~600 with ~700 perpetually in-flight, hide≈30/frame).
                        // Higher terrain resolution needs an LOD-stability fix (hysteresis /
                        // GPU meshing) first — see Known open items. Don't drop this blindly.
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

                // Kick off the async heightfield build on a worker thread. The main
                // loop renders a progress bar until it completes; the terrain is
                // meshed on demand by the voxel renderer. The heightfield is built ONCE.
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
                };
                self.loader = Some(loading::Loader::spawn(params));
                self.pending_planet_cfg = Some(cfg);
                log::info!("Async load started (heightfield on a worker thread)");
            }
        }

        // Build egui state — needs the window for HasDisplayHandle (clipboard).
        self.egui = Some(EguiState::new(&rhi, &window));
        // egui draws at TYPE_1 in its own 1-sample pass on the resolved swapchain image.
        log::info!("egui initialised (1-sample UI pass, swapchain format {:?})", rhi.swapchain_format());

        // Initialise the scene-composite offscreen targets (MSAA + resolved
        // current/depth + the refraction copies). The 3D pass renders into them and
        // `present_composite` blits the result to the swapchain; the refractive ocean
        // samples the opaque copies. No temporal AA.
        if let Err(e) = rhi.init_scene_targets() {
            log::error!("scene targets init failed: {e}");
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
                    if let Some(aerial) = &self.aerial {
                        aerial.destroy(rhi);
                    }
                    if let Some(clouds) = &self.clouds {
                        clouds.destroy(rhi);
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
                        // Geometry views (3–5) work on the Nanite path AND the voxel
                        // terrain; only the bare classic quadtree lacks them.
                        let geom_views = cfg!(feature = "voxel");
                        loop {
                            // 0–10 View/Planet, 11–12 Ocean (Surface/Intensity).
                            self.view_mode = (self.view_mode + 1) % 13;
                            if geom_views || !(3..=5).contains(&self.view_mode) {
                                break;
                            }
                        }
                        log::info!("View mode → {}", self.view_mode);
                    }

                    // ── Voxel terrain edit: G dig / H fill at the player's feet ──
                    #[cfg(feature = "voxel")]
                    KeyCode::KeyG if pressed && self.nav.mode() == NavMode::Surface => {
                        if let (Some(vv), Some(tpc)) = (self.voxel_view.as_mut(), self.surface.as_ref()) {
                            vv.queue_edit(tpc.feet_position(), 6.0, true);
                            log::info!("voxel dig at feet");
                        }
                    }
                    #[cfg(feature = "voxel")]
                    KeyCode::KeyH if pressed && self.nav.mode() == NavMode::Surface => {
                        if let (Some(vv), Some(tpc)) = (self.voxel_view.as_mut(), self.surface.as_ref()) {
                            vv.queue_edit(tpc.feet_position(), 6.0, false);
                            log::info!("voxel fill at feet");
                        }
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
        // Clouds drift in real time (weather), independent of the sim time_scale.
        self.cloud_time += dt;

        // ── Async loading: show a progress bar until the worker finishes ──
        if self.loader.is_some() {
            match self.loader.as_ref().and_then(|l| l.poll()) {
                Some(out) => {
                    // Worker finished — assemble on the main thread (GPU uploads
                    // need the RHI). Then fall through to normal rendering.
                    self.load_timings = Some(out.timings);
                    self.show_load_stats = true;
                    if let Some(cfg) = self.pending_planet_cfg.take() {
                        // Keep a handle to the terrain so the FPS controller can ride it.
                        self.height_field = Some(Arc::clone(&out.hf));
                        // Feed the planet's wind field to the ocean (wave height/intensity).
                        if let Some(w) = self.wave.as_mut() {
                            w.set_wind_source(Arc::clone(&out.hf));
                        }
                        if let Some(w) = self.wind.as_mut() {
                            w.set_wind_source(Arc::clone(&out.hf));
                        }
                        if let Some(c) = self.clouds.as_mut() {
                            c.set_source(Arc::clone(&out.hf));
                        }
                        self.terrain_height_scale = cfg.lod.height_scale;
                        // Phase 2: stand up the on-demand voxel terrain (transvoxel
                        // quadtree). It drives the same LOD selection as PlanetView but
                        // meshes each leaf as a transvoxel block. subdiv=32 MATCHES the LOD
                        // metric (resolution=32 → ~2 px/cell at target_tri_px); at subdiv=16
                        // leaves rendered at half the targeted triangle density (visible LOD
                        // coarseness). The transvoxel CacheCentralBlockOnly switch (mesh_leaf)
                        // keeps per-leaf mesh cost ~flat despite the higher subdiv.
                        #[cfg(feature = "voxel")]
                        {
                            let vlod = LodConfig {
                                radius:        cfg.lod.radius,
                                height_scale:  cfg.lod.height_scale,
                                resolution:    cfg.lod.resolution,
                                max_depth:     cfg.lod.max_depth,
                                target_tri_px: cfg.lod.target_tri_px,
                                hysteresis:    cfg.lod.hysteresis,
                                lru_capacity:  cfg.lod.lru_capacity,
                            };
                            let shadow_size = self.shadow_size;
                            if let Some(rhi) = self.rhi.as_mut() {
                                // Re-home the sun shadow map onto the voxel terrain
                                // (it was created in the now-deleted Nanite loader).
                                if let Err(e) = rhi.create_shadow_map(shadow_size, SHADOW_CASCADES) {
                                    log::error!("create_shadow_map failed: {e}");
                                }
                                let vv = voxel_view::VoxelView::new(
                                    rhi,
                                    Arc::clone(&out.hf),
                                    vlod,
                                    32, // subdiv — matches resolution=32 (see comment above)
                                    cfg.terrain_pipeline,
                                );
                                self.voxel_view = Some(vv);
                            }
                            log::info!("voxel terrain (VoxelView) initialised");
                        }
                        self.planet_view = Some(planet_view_from_hf(cfg, out.hf));
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

        // ── Navigation update ──────────────────────────────────────────────
        self.update_nav(dt);

        // ── Snapshot state for this frame ─────────────────────────────────
        let (camera, altitude) = (self.active_camera(), self.altitude_m());
        let nav_mode      = self.nav.mode();
        let view_mode     = self.view_mode;
        // Ocean/cloud view modes (11–13) don't recolor the terrain — show it lit.
        let terrain_view  = if view_mode >= 11 { 0 } else { view_mode };
        let wireframe     = self.wireframe;
        let aa_mode       = self.aa_mode;

        // ── Planet LOD stats (for HUD + egui) ────────────────────────────
        let planet_stats: Option<PlanetViewStats> = self
            .planet_view
            .as_ref()
            .map(|pv| pv.stats());

        // ── Build egui frame (OUTSIDE rendering instance) ─────────────────
        // Egui texture uploads happen here via one-shot submit.
        // This must run before begin_frame → begin_rendering.
        let ui_out = if self.egui.is_some() && self.window.is_some() && self.rhi.is_some() {
            let voxel_caves          = self.voxel_caves;
            let voxel_cave_strength  = self.voxel_cave_strength;
            let voxel_morph_region   = self.voxel_morph_region;
            let voxel_peak_sharpen   = self.voxel_peak_sharpen;
            let voxel_detail_strength = self.voxel_detail_strength;
            let voxel_detail_scale    = self.voxel_detail_scale;
            let voxel_detail_depth    = self.voxel_detail_depth;
            let voxel_ground_rough    = self.voxel_ground_rough;
            let shadow_map_enabled = self.shadow_map_enabled;
            let shadow_half_extent = self.shadow_half_extent;
            let shadow_depth       = self.shadow_depth;
            let shadow_depth_bias  = self.shadow_depth_bias;
            let shadow_normal_bias = self.shadow_normal_bias;
            let ocean_enabled     = self.ocean_enabled;
            let wind_enabled      = self.wind_enabled;
            let atmo_enabled      = self.atmo_enabled;
            let atmo_params       = self.atmo_params;
            let aerial_enabled    = self.aerial_enabled;
            let aerial_params     = self.aerial_params;
            let clouds_enabled    = self.clouds_enabled;
            let cloud_params      = self.cloud_params;
            let markers_poles     = self.markers_poles;
            let markers_equator   = self.markers_equator;
            let wind_params       = self.wind.as_ref().map(|w| w.params).unwrap_or_default();
            let sea_level_m       = self.sea_level_m;
            let wave_enabled      = self.wave_enabled;
            let wave_choppiness   = self.wave_choppiness;
            let wave_foam         = self.wave_foam;
            let flora_enabled     = self.flora_enabled;
            let flora_density      = self.flora_density;
            let flora_radius_m     = self.flora_radius_m;
            let flora_alt_threshold_m = self.flora_alt_threshold_m;
            // LoadTimings is Copy — snapshot it out so the popup can borrow it while
            // egui mutably borrows self. Only passed while the popup is showing.
            let load_stats        = if self.show_load_stats { self.load_timings } else { None };

            // Profiler counters for the top-right Stats panel (all ~1–2 frames stale).
            let gpu_ms = self.rhi.as_ref().map(|r| r.gpu_time_ms()).unwrap_or(0.0);
            #[allow(unused_mut)]
            let mut triangles = 0u64;
            // Real terrain geometry (voxel leaves + any classic quadtree chunks).
            if let Some(pv) = self.planet_view.as_ref() {
                triangles += pv.triangle_count() as u64;
            }
            #[cfg(feature = "voxel")]
            if let Some(vv) = self.voxel_view.as_ref() {
                triangles += vv.triangle_count() as u64;
            }
            let profiler = crate::gui::Profiler {
                cpu_ms: self.cpu_ms, gpu_ms, triangles,
            };

            // egui, window, and rhi are independent fields — split-borrow.
            let egui   = self.egui.as_mut().unwrap();
            let window = self.window.as_ref().unwrap();
            let rhi    = self.rhi.as_ref().unwrap();

            Some(egui.build_frame(
                window, rhi, nav_mode, altitude, frame_time, view_mode, wireframe, aa_mode,
                shadow_map_enabled, shadow_half_extent, shadow_depth, shadow_depth_bias, shadow_normal_bias,
                cfg!(feature = "voxel"), voxel_caves, voxel_cave_strength, voxel_morph_region,
                voxel_peak_sharpen, voxel_detail_strength, voxel_detail_scale, voxel_detail_depth,
                voxel_ground_rough,
                ocean_enabled, sea_level_m, wave_enabled, wave_choppiness, wave_foam,
                cfg!(feature = "flora"), flora_enabled, flora_density,
                flora_radius_m, flora_alt_threshold_m,
                self.sky.time_scale, self.sky.paused,
                wind_enabled, wind_params,
                atmo_enabled, atmo_params,
                aerial_enabled, aerial_params,
                clouds_enabled, cloud_params,
                markers_poles, markers_equator,
                planet_stats.as_ref(), load_stats.as_ref(),
                profiler,
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
            self.aa_mode           = out.aa_mode;
            self.shadow_map_enabled  = out.shadow_map_enabled;
            self.shadow_half_extent  = out.shadow_half_extent;
            self.shadow_depth        = out.shadow_depth;
            self.shadow_depth_bias   = out.shadow_depth_bias;
            self.shadow_normal_bias  = out.shadow_normal_bias;
            self.voxel_caves          = out.voxel_caves;
            self.voxel_cave_strength  = out.voxel_cave_strength;
            self.voxel_morph_region   = out.voxel_morph_region;
            self.voxel_peak_sharpen   = out.voxel_peak_sharpen;
            self.voxel_detail_strength = out.voxel_detail_strength;
            self.voxel_detail_scale    = out.voxel_detail_scale;
            self.voxel_detail_depth    = out.voxel_detail_depth;
            self.voxel_ground_rough    = out.voxel_ground_rough;
            #[cfg(feature = "voxel")]
            if let Some(vv) = self.voxel_view.as_mut() {
                vv.set_caves_enabled(out.voxel_caves);
                vv.set_cave_strength(out.voxel_cave_strength as f64);
                vv.set_morph_region(out.voxel_morph_region);
                vv.set_peak_sharpen(out.voxel_peak_sharpen as f64);
                vv.set_detail(out.voxel_detail_strength, out.voxel_detail_scale, out.voxel_detail_depth);
                vv.set_ground_roughness(out.voxel_ground_rough);
            }
            // Geometry views (3–5) work on the voxel terrain; only the bare classic
            // quadtree lacks them → collapse to Lit there.
            let geom_views = cfg!(feature = "voxel");
            if !geom_views && (3..=5).contains(&self.view_mode) {
                self.view_mode = 0;
            }
            self.ocean_enabled     = out.ocean_enabled;
            self.sea_level_m       = out.sea_level_m;
            self.wave_enabled      = out.wave_enabled;
            self.wave_choppiness   = out.wave_choppiness;
            self.wave_foam         = out.wave_foam;
            self.flora_enabled     = out.flora_enabled;
            self.flora_density     = out.flora_density;
            self.flora_radius_m    = out.flora_radius_m;
            self.flora_alt_threshold_m = out.flora_alt_threshold_m;
            self.wind_enabled      = out.wind_enabled;
            if let Some(w) = self.wind.as_mut() {
                w.params = out.wind;
            }
            self.atmo_enabled      = out.atmo_enabled;
            self.atmo_params       = out.atmo;
            self.aerial_enabled    = out.aerial_enabled;
            self.aerial_params     = out.aerial;
            self.clouds_enabled    = out.clouds_enabled;
            self.cloud_params      = out.clouds;
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

        // SSAA: apply the selected mode's supersample factor before begin_frame (it may
        // rebuild the scene targets). Cheap no-op when unchanged.
        rhi.set_render_scale(aa_mode.render_scale());

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

        // Free streaming buffers the GPU has finished with (retired terrain leaves).
        // MUST run once per frame — without it the deferred-destruction graveyard grows
        // unbounded as leaves churn (LOD/camera), VRAM fills, and uploads start failing,
        // so leaves can't refine and degrade into stale/coarse blocks over time.
        rhi.collect_streaming_garbage();

        if fi == u32::MAX {
            if let Err(e) = rhi.end_frame(fi) {
                log::error!("end_frame error: {e}");
            }
            return;
        }

        // CPU recording time: from here (past begin_frame's fence wait) to just
        // before end_frame — so it reflects real CPU cost, not vsync idle.
        let cpu_t0 = Instant::now();

        // ── 3D rendering ──────────────────────────────────────────────────
        {
            {
                // Build frame uniforms with rotation-only camera-relative view,
                // as documented in FrameUniforms::new.
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
                self.sun_dir_body = sun_dir_body; // stashed for shadow/light reuse
                let lights = Lights::from_sun(sun_dir_body, self.sky.sun_light_color());
                let fu     = FrameUniforms::new(&camera, aspect, &lights);

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

                // Sun shadow map: 3 cascade light matrices, built in the SAME
                // camera-relative world space the Nanite draw + casters use (so caster
                // and receiver agree). Centred on the player's feet in Surface mode so
                // the small near cascade reaches the wedge ahead; camera otherwise.
                // `shadow_params = [depth_bias, normal_bias, strength, enabled]`.
                // Hoisted here so cull/update AND the caster pass below share them.
                let shadow_focus = if nav_mode == NavMode::Surface {
                    self.surface.as_ref().map(|tpc| tpc.feet_position()).unwrap_or(camera_world_pos)
                } else {
                    camera_world_pos
                };
                // Cascade light matrices, built whenever the shadow map exists (the
                // voxel terrain re-homed it — see the loader). Receivers fall back to
                // the screen-space shadow when it's off, so dummies are harmless.
                let (cascade_mvps, shadow_params): ([glam::Mat4; 3], [f32; 4]) =
                    if self.shadow_map_enabled && rhi.has_shadow_map() {
                        (
                            sun_cascade_matrices(
                                self.sun_dir_body, camera_world_pos, shadow_focus,
                                self.shadow_half_extent, self.shadow_depth, self.shadow_size,
                            ),
                            [self.shadow_depth_bias, self.shadow_normal_bias, 1.0, 1.0],
                        )
                    } else {
                        ([glam::Mat4::IDENTITY; 3], [0.0; 4])
                    };

                // Voxel terrain (Phase 2) REPLACES the quadtree as the terrain when present.
                #[cfg(feature = "voxel")]
                let voxel_on = self.voxel_view.is_some();
                #[cfg(not(feature = "voxel"))]
                let voxel_on = false;

                // PlanetView::update — streaming uploads (outside rendering instance).
                if !voxel_on {
                    if let Some(pv) = self.planet_view.as_mut() {
                        pv.update(rhi, fi, self.frame_counter, &lod_cam, camera_world_pos);
                    }
                }
                #[cfg(feature = "voxel")]
                if let Some(vv) = self.voxel_view.as_mut() {
                    vv.set_view_mode(terrain_view); // bake the selected data field; remeshes on change
                    vv.update(rhi, fi, &lod_cam);
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
                // Tier-2: flora draws in ANY nav mode the opaque bracket supports
                // (it's mode-invariant), gated on a fade-in altitude so trees resolve
                // during descent — not only once on the surface. The scatter window
                // (RADIUS_M) only matters within a few hundred m of the ground; the
                // threshold sits well above it so impostors appear before landing.
                #[cfg(feature = "flora")]
                let draw_trees = self.flora_enabled
                    && self.flora_renderer.is_some()
                    && altitude < self.flora_alt_threshold_m;
                // Scatter/build center: the player's feet in Surface mode; otherwise
                // (Globe / Placement / Space) `self.surface` is None, so use the
                // SUB-CAMERA ground point — project the camera radially to the planet
                // and ground it on the rendered mesh. f64 throughout (camera-relative
                // precision); None until the heightfield has loaded.
                #[cfg(feature = "flora")]
                let flora_center: Option<DVec3> = if draw_trees {
                    if let Some(tpc) = self.surface.as_ref() {
                        Some(tpc.feet_position())
                    } else if let Some(hf) = self.height_field.as_ref() {
                        let dir = camera_world_pos.normalize_or_zero();
                        if dir.length_squared() > 0.0 {
                            let r = controls::terrain_grid::ground_radius(
                                hf.as_ref(), self.terrain_height_scale, dir,
                            );
                            Some(dir * r)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                #[cfg(feature = "flora")]
                if let (true, Some(center)) = (draw_trees, flora_center) {
                    // 1. Build the species mesh once from the canonical TreeSpec,
                    //    overridden to SINGLE-leaf cards (the shared `default_specimen`
                    //    is Cluster; the planet wants individual leaves). Drawn at
                    //    many models — its stored origin is unused.
                    if self.flora_tree.is_none() {
                        if let Some(hf) = self.height_field.as_ref() {
                            // The species mesh is placement-agnostic (drawn at many
                            // per-instance models); seed it at the current center.
                            let dir = center.normalize_or_zero();
                            let r = controls::terrain_grid::ground_radius(
                                hf.as_ref(), self.terrain_height_scale, dir,
                            );
                            let spec = TreeSpec {
                                leaf_mode: LeafMode::Single,
                                ..TreeSpec::default_specimen()
                            };
                            match FloraView::from_spec(rhi, &spec, dir * r) {
                                Ok(tree) => {
                                    let single = matches!(tree.leaf_mode(), LeafMode::Single);
                                    if let Some(rend) = self.flora_renderer.as_mut() {
                                        let _ = rend.update_leaf_texture(
                                            rhi, &tree.leaf_genes(), single,
                                        );
                                        // Bake the TOP/SIDE impostor atlas from the
                                        // species mesh (one-shot fenced, like the leaf
                                        // texture upload above) so far trees show a
                                        // textured billboard, not a flat green disc.
                                        if let Err(e) = rend.bake_impostor(rhi, &tree) {
                                            log::error!("FloraRenderer::bake_impostor failed: {e}");
                                        }
                                    }
                                    self.flora_tree = Some(tree);
                                }
                                Err(e) => log::error!("FloraView::from_spec failed: {e}"),
                            }
                        }
                    }
                    // 2. (Re)scatter when the player moved a fraction of the WINDOW,
                    //    the density changed, OR the radius changed (deterministic, so
                    //    this never reshuffles). NOT per frame — throttled here. A small
                    //    move only changes a thin far edge ring, so rebuilding the whole
                    //    250 m disc every ~1 s of walking (the old fixed 2.5 m gate) was
                    //    the walk-time spike; the floor keeps sub-cell moves from
                    //    rebuilding the cell-quantized scatter.
                    if let Some(hf) = self.height_field.as_ref() {
                        let rescatter_move = (self.flora_radius_m
                            * flora_scatter::RESCATTER_MOVE_FRAC)
                            .max(flora_scatter::SPACING_M);
                        let moved = self
                            .flora_scatter_center
                            .map_or(true, |p| (p - center).length() > rescatter_move);
                        let density_changed =
                            (self.flora_density - self.flora_last_density).abs() > 1e-3;
                        let radius_changed =
                            (self.flora_radius_m - self.flora_last_radius_m).abs() > 1e-3;
                        if moved || density_changed || radius_changed {
                            self.flora_instances = flora_scatter::scatter(
                                hf.as_ref(), self.terrain_height_scale, center,
                                self.flora_density, self.flora_radius_m,
                            );
                            self.flora_scatter_center = Some(center);
                            self.flora_last_density = self.flora_density;
                            self.flora_last_radius_m = self.flora_radius_m;
                        }
                    }
                    // Trees ride the analytic CDLOD-MORPHED surface — the radius
                    // terrain_csm.wgsl actually draws at each direction/distance, from
                    // `surface_radius` alone (no mesh raycast, no residency). This is what keeps
                    // FAR trees on the coarse far mesh instead of floating at the fine level-12
                    // height, AND near trees on the morphed surface — one path, every distance.
                    // Cheap (~2 height samples/tree), so all trees every frame, no spike.
                    // ponytail: dug/edited terrain isn't in `surface_radius` (density-only) — a
                    //   tree over a hole won't drop into it; re-add a targeted mesh raycast
                    //   (`vv.collider_dyn().ground_radius(dir)`) for edited regions when digging
                    //   leaves WIP.
                    #[cfg(feature = "voxel")]
                    if let (Some(vv), Some(hf)) =
                        (self.voxel_view.as_ref(), self.height_field.as_ref())
                    {
                        for inst in self.flora_instances.iter_mut() {
                            let dir = inst.origin.normalize_or_zero();
                            if dir.length_squared() > 0.5 {
                                let r = vv.morphed_ground_radius(
                                    hf.as_ref(),
                                    self.terrain_height_scale,
                                    camera_world_pos,
                                    dir,
                                );
                                inst.origin = dir * r;
                            }
                        }
                    }
                    // 3. Frame uniforms + ONE wind solve (shared by every instance —
                    //    same genome → same bones) + make the trees RECEIVE the
                    //    planet's 3-cascade sun CSM (the same maps the terrain samples):
                    //    point the flora shadow set at the cascade depth views and
                    //    upload the cascade matrices (camera-relative → use_view_pos=1).
                    //    The empty self-shadow pass below keeps flora's own map in
                    //    SHADER_READ as the no-CSM fallback (enabled=0 then).
                    self.flora_clock += dt;
                    let recv_shadows = self.shadow_map_enabled && rhi.has_shadow_map();
                    let cascade_views = if recv_shadows {
                        Some([
                            rhi.shadow_map_view(fi, 0),
                            rhi.shadow_map_view(fi, 1),
                            rhi.shadow_map_view(fi, 2),
                        ])
                    } else {
                        None
                    };
                    if let (Some(rend), Some(tree)) =
                        (self.flora_renderer.as_mut(), self.flora_tree.as_mut())
                    {
                        let wind = [self.flora_clock, 0.6, 1.0, 0.0];
                        let _ = rend.set_frame_uniforms(rhi, fi, &fu);
                        let _ = rend.set_bone_matrices(rhi, fi, tree.solve_wind(wind));
                        let su = ShadowUniforms {
                            light_view_proj: cascade_mvps[0].to_cols_array_2d(),
                            // (1/size [unused — shader uses textureDimensions], normal_bias,
                            //  enabled, dappled off on the planet)
                            params: [
                                1.0 / rend.shadow_map_size() as f32,
                                shadow_params[1],
                                if recv_shadows { 1.0 } else { 0.0 },
                                0.0,
                            ],
                            light_view_proj1: cascade_mvps[1].to_cols_array_2d(),
                            light_view_proj2: cascade_mvps[2].to_cols_array_2d(),
                            // (cascade_count=3, use_view_pos=1, depth_bias, _)
                            params2: [3.0, 1.0, shadow_params[0], 0.0],
                        };
                        let _ = rend.set_shadow_uniforms(rhi, fi, &su);
                        if let Some(views) = cascade_views {
                            rend.set_shadow_cascade_views(fi, views);
                        }
                        // Keep flora's own self-shadow map in SHADER_READ (the fallback
                        // bound when recv_shadows is false). Empty when the CSM is used.
                        rend.begin_shadow_pass(rhi, fi);
                        rend.end_shadow_pass(rhi, fi);
                    }
                }

                // Sun shadow caster pass (depth-only, before the main instance): one
                // pass PER CASCADE — render the casters (voxel terrain cut, character,
                // nearby trees) into each cascade map from the light. The terrain then
                // PCF-samples the tightest covering cascade in fs_color.
                if self.shadow_map_enabled && rhi.has_shadow_map() {
                    for c in 0..rhi.shadow_cascade_count() {
                        let lvp = cascade_mvps[c as usize];
                        // Cascade c covers ±(base·2^c) around the focus; cast within that
                        // + a margin for shadows reaching inward from just outside.
                        #[cfg(feature = "flora")]
                        let cast_radius =
                            (self.shadow_half_extent * (1u32 << c) as f32 * 1.5) as f64;
                        rhi.begin_shadow_pass(fi, c);
                        // Voxel terrain caster (resident leaves → this cascade's depth).
                        #[cfg(feature = "voxel")]
                        if let Some(vv) = self.voxel_view.as_ref() {
                            let _ = vv.record_shadow(rhi, fi, lvp, camera_world_pos);
                        }
                        // Character caster (real mesh silhouette) — always near the focus.
                        if draw_character {
                            if let (Some(ch), Some(tpc)) =
                                (self.character.as_ref(), self.surface.as_ref())
                            {
                                let _ = ch.record_shadow(
                                    rhi, fi, &camera, tpc.feet_position(), tpc.facing(), lvp,
                                );
                            }
                        }
                        // Tree casters within this cascade's radius (smaller cascade →
                        // fewer trees). ponytail: distance gate, not a sun-frustum cull.
                        #[cfg(feature = "flora")]
                        if draw_trees {
                            if let (Some(rend), Some(tree)) =
                                (self.flora_renderer.as_ref(), self.flora_tree.as_ref())
                            {
                                let wind = [self.flora_clock, 0.6, 1.0, 0.0];
                                let light_planes = frustum_side_planes(lvp);
                                let base_r = tree.cull_radius();
                                for inst in &self.flora_instances {
                                    if (inst.origin - shadow_focus).length() > cast_radius {
                                        continue;
                                    }
                                    // Sun-frustum (cascade footprint) cull: a tree
                                    // outside this cascade's light box can't shadow into
                                    // it. Tighter than the distance gate at the box
                                    // corners; the sun-depth axis is NOT tested (a tree
                                    // up-sun still casts into the footprint).
                                    let center = (inst.origin - camera_world_pos).as_vec3();
                                    if !sphere_in_side_planes(&light_planes, center, base_r * inst.scale)
                                    {
                                        continue;
                                    }
                                    let model = flora_scatter::instance_model(
                                        inst.origin, inst.yaw, inst.scale, camera_world_pos,
                                    );
                                    let _ = tree.record_shadow_model(
                                        rhi, rend, fi, model, lvp, wind, LeafLod::FULL,
                                    );
                                }
                            }
                        }
                        rhi.end_shadow_pass(fi, c);
                    }
                }

                // 3D opaque pass.
                rhi.begin_rendering(fi);
                rhi.set_viewport_scissor_full(fi);
                if !voxel_on {
                    if let Some(pv) = self.planet_view.as_mut() {
                        if let Err(e) = pv.record(rhi, fi, &fu, camera_world_pos, terrain_view, wireframe) {
                            log::error!("PlanetView::record error: {e}");
                        }
                    }
                }

                // Voxel terrain (Phase 2): draw all resident transvoxel leaves.
                #[cfg(feature = "voxel")]
                if let Some(vv) = self.voxel_view.as_ref() {
                    if let Err(e) = vv.record(
                        rhi, fi, &fu, camera_world_pos, terrain_view, cascade_mvps, shadow_params,
                    ) {
                        log::error!("VoxelView::record error: {e}");
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

                // Atmosphere shell — the sky/halo whenever the depth-aware `Aerial`
                // pass is OFF (the default now). When aerial is on it IS the scattering,
                // the sky dome AND the limb halo, so drawing the shell too would
                // double-tint the planet blue — mutually exclusive (matches aerial.rs).
                if self.atmo_enabled && !self.aerial_enabled {
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
                        if let Err(e) =
                            ch.draw(rhi, fi, &fu, &camera, feet, facing, cascade_mvps, shadow_params)
                        {
                            log::error!("Character::draw error: {e}");
                        }
                    }
                }

                // Procedural trees: drawn inside the opaque pass, after the character,
                // before the translucent ocean (shares reversed-Z depth).
                // GEOMETRY NEAR, FLAT GREEN DISC MID, TINTED GROUND FAR: within
                // TREE_GEOM_M the real tree draws; past it each tree collapses to ONE
                // flat green disc (the impostor billboard) — top-down / mid-range bare
                // branches read as red sticks, so a flat canopy dot looks far better and
                // is cheap. Past the scatter ring there are no instances at all — the
                // baked forest canopy tint (vegetation_density in tessellate.rs) IS the
                // forest from a distance / orbit, at zero per-frame cost.
                // ponytail: per-instance draws (NOT Nanite-virtualized); instance the
                // discs if the count ever costs. Disc colour + crossover are consts —
                // promote to GUI sliders if tuned often.
                #[cfg(feature = "flora")]
                if draw_trees {
                    if let (Some(rend), Some(tree)) =
                        (self.flora_renderer.as_ref(), self.flora_tree.as_ref())
                    {
                        // Geometry within TREE_GEOM_M; flat green disc beyond, opaque by
                        // the end of the TREE_FADE_M crossfade band.
                        const TREE_GEOM_M: f32 = 40.0;
                        const TREE_FADE_M: f32 = 25.0;
                        const DISC_GREEN: [f32; 3] = [0.10, 0.22, 0.08];
                        let disc_green = glam::Vec3::from(DISC_GREEN);
                        let wind = [self.flora_clock, 0.6, 1.0, 0.0];
                        // Frustum + behind-camera cull. Trees are plain per-instance
                        // draws and the scatter ring wraps the whole sphere around the
                        // player, so ~half sit behind the camera — skipping the
                        // off-screen ones is a direct draw-call saving.
                        let cam_planes =
                            frustum_side_planes(glam::Mat4::from_cols_array_2d(&fu.view_proj));
                        let cam_fwd = camera.orientation * glam::Vec3::NEG_Z;
                        let cam_right = camera.orientation * glam::Vec3::X;
                        let cam_up = camera.orientation * glam::Vec3::Y;
                        let base_r = tree.cull_radius();
                        for inst in &self.flora_instances {
                            let center = (inst.origin - camera_world_pos).as_vec3();
                            let radius = base_r * inst.scale;
                            if center.dot(cam_fwd) < -radius
                                || !sphere_in_side_planes(&cam_planes, center, radius)
                            {
                                continue;
                            }
                            let disc =
                                ((center.length() - TREE_GEOM_M) / TREE_FADE_M).clamp(0.0, 1.0);
                            let model = flora_scatter::instance_model(
                                inst.origin, inst.yaw, inst.scale, camera_world_pos,
                            );
                            // Geometry (leaves thin as the disc fades in); skipped once
                            // fully a disc.
                            if disc < 1.0 {
                                let lod = LeafLod {
                                    density_frac: 1.0 - disc,
                                    cast_leaf_shadows: true,
                                    impostor_blend: 0.0,
                                };
                                if let Err(e) = tree.record_at_lod(rhi, rend, fi, model, wind, lod)
                                {
                                    log::error!("FloraView::record_at_lod error: {e}");
                                    break;
                                }
                            }
                            // Flat green disc (alpha-blended) ON TOP, once the fade has
                            // started; camera-facing billboard.
                            if disc > 0.0 {
                                // Tree radial up = the surface normal at the
                                // instance (the FS blends TOP vs SIDE atlas tiles).
                                let tree_up = inst.origin.normalize().as_vec3();
                                if let Err(e) = tree.record_impostor_at(
                                    rhi, rend, fi, model, cam_right, cam_up, disc_green,
                                    disc, tree_up,
                                ) {
                                    log::error!("FloraView::record_impostor_at error: {e}");
                                    break;
                                }
                            }
                        }
                    }
                }

                // Ocean. With TAA on, begin_water_pass splits the 3D pass and the
                // shell + FFT waves render in one refractive 1× pass (matching colour,
                // depth-darkening, and refraction). Without TAA, fall back to the
                // simple alpha-blended shell in the MSAA pass.
                let mut water_split = false;
                // Drive the cloud worker's active state every frame (two-way idle gate)
                // so it sleeps when clouds are toggled off rather than spinning forever.
                // record() (the only path that touches active) isn't reached while
                // clouds are disabled or TAA is off, so latch it here unconditionally.
                if let Some(c) = self.clouds.as_ref() {
                    c.set_advect_active(self.clouds_enabled);
                }
                // Open the refraction split (1× instance + resolved opaque colour/depth)
                // if the ocean, aerial, or clouds want it. The split needs the scene
                // targets allocated (always, except at zero extent). Clouds/aerial are
                // OFF by default now (TAA gone); the FFT ocean drives the split.
                let want_split = (self.ocean_enabled && self.wave.is_some())
                    || (self.aerial_enabled && self.aerial.is_some())
                    || (self.clouds_enabled && self.clouds.is_some());
                if want_split || self.ocean_enabled {
                    let sea = self.sea_level_m;
                    // The expensive FFT patch is drawn only near the surface; the shell
                    // covers the rest. begin_water_pass returns false only at zero extent.
                    let draw_waves = self.wave_enabled && altitude < 20_000.0;
                    let split = want_split && rhi.begin_water_pass(fi).unwrap_or(false);
                    water_split = split;
                    if split {
                        let scene_depth = rhi.refraction_depth_view();
                        let ext = rhi.extent();
                        // Ocean first (opaque sea writes into `current`)...
                        if self.ocean_enabled {
                            let scene_view = rhi.refraction_src_view();
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
                        }
                        // ...then atmospheric scattering over the opaque scene + ocean
                        // (reads the opaque pre-ocean depth → distance-graded aerial
                        // perspective on terrain/sea + a sky dome on the cleared pixels).
                        // Before clouds so it doesn't tint the cloud tops.
                        if self.aerial_enabled {
                            if let Some(a) = self.aerial.as_mut() {
                                if let Err(e) = a.record(
                                    rhi, fi, &fu, &camera, scene_depth,
                                    (ext.width, ext.height), self.aerial_params, false,
                                ) {
                                    log::error!("Aerial::record error: {e}");
                                }
                            }
                        }
                        // ...then clouds on top: a shell above sea level, so they
                        // composite over both land and sea. The march still clamps to
                        // the OPAQUE depth (seabed/terrain, captured before the ocean),
                        // so terrain occlusion is unchanged.
                        if self.clouds_enabled {
                            if let Some(c) = self.clouds.as_mut() {
                                if let Err(e) = c.record(
                                    rhi, fi, &fu, &camera, scene_depth,
                                    (ext.width, ext.height), self.cloud_time,
                                    self.cloud_params, view_mode == 13,
                                ) {
                                    log::error!("Clouds::record error: {e}");
                                }
                            }
                        }
                    } else if self.ocean_enabled {
                        if let Some(o) = self.ocean.as_ref() {
                            if let Err(e) = o.record(rhi, fi, &fu, camera_world_pos, sea) {
                                log::error!("Ocean::record error: {e}");
                            }
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
        }

        self.frame_counter = self.frame_counter.wrapping_add(1);

        // ── Periodic planet LOD stats log (every ~5 s) ───────────────────
        // Logs resident chunk count and build queue to confirm the planet is
        // initializing in headless test runs.
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
        }

        // ── Composite to the swapchain ───────────────────────────────────
        // Close the 3D / water instance and blit the composited `current` to the
        // swapchain. Must run AFTER all 3D draws and BEFORE begin_ui_pass. No-op at
        // zero extent (begin_rendering rendered straight to the swapchain).
        if let Err(e) = rhi.present_composite(fi, self.aa_mode.into()) {
            log::error!("present_composite error: {e}");
        }

        // ── Transition to 1-sample UI pass ───────────────────────────────
        // begin_ui_pass closes the MSAA 3D instance (zero-extent path), inserts a
        // resolve→load barrier on the swapchain image, and opens a 1-sample
        // rendering instance (loadOp=LOAD, no depth attachment). egui-ash-renderer
        // 0.11 hardcodes TYPE_1 samples, so it MUST draw in this separate pass.
        // (On the composite path the 3D instance + swapchain were already handled by
        // present_composite; begin_ui_pass just opens the UI instance.)
        //
        // This call is always made (all render modes) so that end_frame always
        // closes a UI instance — maintaining the begin/end bracket invariant.
        rhi.begin_ui_pass(fi);

        // ── egui render (inside 1-sample UI pass) ─────────────────────────
        if let Some(egui) = self.egui.as_mut() {
            egui.render(rhi, fi);
        }

        // Measure CPU recording cost now (before submit/present); stored after the
        // `rhi` borrow ends below, for next frame's Stats panel.
        let cpu_ms = cpu_t0.elapsed().as_secs_f32() * 1000.0;

        // ── End frame ─────────────────────────────────────────────────────
        // end_frame closes the UI rendering instance opened by begin_ui_pass,
        // then submits and presents.
        if let Err(e) = rhi.end_frame(fi) {
            log::error!("end_frame error: {e}");
        }
        self.cpu_ms = cpu_ms;
    }
}

/// Build the sun shadow map's light view-projection in **camera-relative** world
/// space (camera at the origin, world axes — the same space the Nanite draw's
/// `world_rel` lives in). Reversed-Z ortho fit to a fixed region around the camera.
/// ponytail: a single cascade centred on the camera; a player-centred / multi-
/// cascade fit (and texel-snap for crawl-free edges) is the upgrade.
/// `focus_world` = the world point the cascade is centred on (the player's feet in
/// third-person; the camera otherwise). `half_extent` = cascade radius in metres —
/// SMALL is sharp: texel size = `2*half_extent/shadow_size`, so 60 m @ 4096² ≈
/// 0.03 m/texel (vs 180 m ≈ 0.09 m). An ortho's density is uniform + translation-
/// invariant, so centring is purely a *reach* choice (a small box must sit on the
/// player to cover the wedge ahead); shrinking `half_extent` is the density lever.
/// The four SIDE frustum planes (left, right, bottom, top) of a view-projection
/// matrix, normalized (Gribb–Hartmann). A point `p` is inside a plane iff
/// `dot(plane.xyz, p) + plane.w >= 0`. Near/far are omitted on purpose: for the
/// camera we reject behind-the-eye points with an explicit forward dot (sidesteps
/// the reversed-Z near/far sign subtlety); for a sun cascade the depth axis must
/// NOT cull a caster (a tree up-sun still shadows the footprint).
#[cfg(feature = "flora")]
fn frustum_side_planes(vp: glam::Mat4) -> [glam::Vec4; 4] {
    let (r0, r1, r3) = (vp.row(0), vp.row(1), vp.row(3));
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1].map(|p| {
        let n = p.truncate().length();
        if n > 1e-8 { p / n } else { p }
    })
}

/// Conservative sphere-vs-side-planes test (`center` in the matrix's space).
#[cfg(feature = "flora")]
fn sphere_in_side_planes(planes: &[glam::Vec4; 4], center: glam::Vec3, radius: f32) -> bool {
    planes.iter().all(|pl| pl.truncate().dot(center) + pl.w >= -radius)
}

#[allow(dead_code)] // used by the voxel shadow caster (and dead under --no-default-features)
fn sun_light_view_proj(
    sun_dir: glam::Vec3,
    cam_world: glam::DVec3,
    focus_world: glam::DVec3,
    half_extent: f32,
    depth: f32,
    shadow_size: u32,
) -> glam::Mat4 {
    let mut sun = sun_dir.normalize_or_zero();
    if sun.length_squared() < 0.5 {
        sun = glam::Vec3::Y;
    }
    // Radial "up" at the camera (the planet centre is the world origin).
    let mut up = cam_world.as_vec3().normalize_or_zero();
    if up.length_squared() < 0.5 {
        up = glam::Vec3::Y;
    }
    // Avoid a degenerate look-at when the sun is near the local vertical.
    if up.dot(sun).abs() > 0.99 {
        up = if sun.x.abs() < 0.9 { glam::Vec3::X } else { glam::Vec3::Z };
    }
    let center = (focus_world - cam_world).as_vec3(); // camera-relative cascade centre
    let eye = center + sun * (depth * 0.5);
    let view = glam::Mat4::look_at_rh(eye, center, up);

    // Texel-snap on the FOCUS point's light-space offset (f64 against the true world
    // position) so the box steps in whole texels as the player walks → no crawl.
    let texel = (2.0 * half_extent / shadow_size.max(1) as f32) as f64;
    let right = view.row(0).truncate().as_dvec3();
    let upv = view.row(1).truncate().as_dvec3();
    let sx = (focus_world.dot(right) - (focus_world.dot(right) / texel).round() * texel) as f32;
    let sy = (focus_world.dot(upv) - (focus_world.dot(upv) / texel).round() * texel) as f32;
    let snap = glam::Mat4::from_translation(glam::Vec3::new(sx, sy, 0.0));

    let proj = reversed_z_orthographic(-half_extent, half_extent, -half_extent, half_extent, 1.0, depth);
    proj * snap * view
}

/// Build the 3 cascade light matrices — concentric player-centred boxes with
/// geometrically-growing half-extents (`base`, `base*2`, `base*4`): cascade 0 sharp
/// + near, cascade 2 coarse + far. The `3` MUST match `SHADOW_CASCADES` (= 3) used
/// by the voxel terrain + character CSM receivers.
#[allow(dead_code)] // used by the voxel shadow caster (dead under --no-default-features)
fn sun_cascade_matrices(
    sun_dir: glam::Vec3,
    cam_world: glam::DVec3,
    focus_world: glam::DVec3,
    base_half_extent: f32,
    depth: f32,
    shadow_size: u32,
) -> [glam::Mat4; 3] {
    std::array::from_fn(|c| {
        let half = base_half_extent * (1u32 << c) as f32;
        sun_light_view_proj(sun_dir, cam_world, focus_world, half, depth, shadow_size)
    })
}

// ── Entry point ───────────────────────────────────────────────────────────────

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    log::info!("minos starting");

    let event_loop = EventLoop::new().expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new();
    if let Err(e) = event_loop.run_app(&mut app) {
        log::error!("Event loop exited with error: {e}");
        std::process::exit(1);
    }

    log::info!("minos shut down");
}

#[cfg(all(test, feature = "flora"))]
mod cull_tests {
    use super::{frustum_side_planes, sphere_in_side_planes};
    use minos_render::projection::reversed_z_orthographic;
    use glam::Vec3;

    #[test]
    fn side_planes_accept_inside_reject_outside() {
        // Ortho box ±2 in x/y; the matrix's own space is the test space. The side
        // planes ignore depth, so any z works.
        let vp = reversed_z_orthographic(-2.0, 2.0, -2.0, 2.0, 1.0, 100.0);
        let planes = frustum_side_planes(vp);
        assert!(sphere_in_side_planes(&planes, Vec3::ZERO, 0.1), "centre is inside");
        assert!(!sphere_in_side_planes(&planes, Vec3::new(10.0, 0.0, 0.0), 0.1), "far +x is outside");
        assert!(!sphere_in_side_planes(&planes, Vec3::new(0.0, 10.0, 0.0), 0.1), "far +y is outside");
        // A big enough bounding sphere widens acceptance (conservative — never drops
        // geometry that pokes into the frustum).
        assert!(sphere_in_side_planes(&planes, Vec3::new(2.5, 0.0, 0.0), 1.0), "straddler kept");
    }
}

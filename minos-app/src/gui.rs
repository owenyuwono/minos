//! `gui` — interactive egui debug panel for minos-app.
//!
//! `EguiState` wraps:
//!  - an `egui::Context` — the egui runtime.
//!  - an `egui_winit::State` — translates winit events to egui input.
//!  - an `egui_ash_renderer::Renderer` — records draw commands into a Vulkan
//!    command buffer using the `dynamic-rendering` feature.
//!
//! # Integration with the frame loop
//!
//! Per frame:
//!
//! 1. Feed winit events: call `EguiState::on_window_event` before routing to nav.
//!    If `egui_ctx.wants_pointer_input()` or `wants_keyboard_input()` returns `true`,
//!    the event should NOT be forwarded to navigation.
//!
//! 2. Build the UI: call `EguiState::build_frame(window, rhi, …)` to run the egui
//!    frame and upload any changed textures.  This must happen OUTSIDE the
//!    dynamic-rendering instance (texture uploads use a one-shot submit internally).
//!    It returns a [`UiOutput`] of the control changes the caller applies back.
//!
//! 3. Render: after all 3D draw calls, call `rhi.begin_ui_pass(fi)` to close the
//!    MSAA 3D instance and open a 1-sample UI pass, then call
//!    `EguiState::render(rhi, fi)` to record egui commands into that pass,
//!    BEFORE `rhi.end_frame(fi)`.  The separate 1-sample pass is required because
//!    `egui-ash-renderer` 0.11 hardcodes `rasterizationSamples = TYPE_1`.

use egui::{Context, FullOutput, ViewportId};
use egui_ash_renderer::{DynamicRendering, Options, Renderer};
use egui_winit::State as WinitState;

use minos_rhi::Rhi;

use crate::controls::nav_mode::NavMode;
use crate::loading::LoadTimings;
use crate::planet_view::PlanetViewStats;

// ── Anti-aliasing ─────────────────────────────────────────────────────────────

/// Anti-aliasing mode (chosen in the Settings panel). MSAA 4× is always on
/// underneath; this picks the post-process AA and/or SSAA supersample factor. Each
/// mode maps to a `(render_scale, minos_rhi::PostAa)` pair the app applies per frame.
/// To add a method: add a variant + its mappings here, and (for a post-process pass)
/// the matching RHI pipeline (see `present_composite`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AaMode {
    Off,
    Fxaa,
    Ssaa15,
    Ssaa2,
}

impl AaMode {
    /// All selectable modes, in display order (for the Settings combo).
    pub const ALL: [AaMode; 4] = [AaMode::Off, AaMode::Fxaa, AaMode::Ssaa15, AaMode::Ssaa2];

    pub fn label(self) -> &'static str {
        match self {
            AaMode::Off => "Off (MSAA only)",
            AaMode::Fxaa => "FXAA",
            AaMode::Ssaa15 => "SSAA 1.5×",
            AaMode::Ssaa2 => "SSAA 2×",
        }
    }

    /// SSAA supersample factor for the 3D pass (1.0 = native).
    pub fn render_scale(self) -> f32 {
        match self {
            AaMode::Off | AaMode::Fxaa => 1.0,
            AaMode::Ssaa15 => 1.5,
            AaMode::Ssaa2 => 2.0,
        }
    }
}

impl From<AaMode> for minos_rhi::PostAa {
    fn from(m: AaMode) -> Self {
        match m {
            // SSAA needs no post filter — the downsampling blit IS the AA.
            AaMode::Off | AaMode::Ssaa15 | AaMode::Ssaa2 => minos_rhi::PostAa::None,
            AaMode::Fxaa => minos_rhi::PostAa::Fxaa,
        }
    }
}

// ── UiOutput ────────────────────────────────────────────────────────────────

/// Control changes produced by the debug panel in one frame.
///
/// The caller applies these to app state after `build_frame` returns: the
/// settings fields are the new values; the action flags are one-shot edges.
#[derive(Debug, Clone, Copy)]
pub struct UiOutput {
    /// Unified view mode 0–10 — View: 0 Lit, 1 Unlit, 2 Normal, 3 Triangle,
    /// 4 Cluster, 5 LOD (3–5 geometry-debug); Planet: 6 Plate, 7 Height, 8 Material,
    /// 9 Wetness, 10 Volcano.
    pub view_mode: u32,
    pub wireframe: bool,
    /// Post-process anti-aliasing mode (Settings panel).
    pub aa_mode: AaMode,
    /// Sun shadow map: on/off + cascade + bias tuning.
    pub shadow_map_enabled: bool,
    /// Cascade radius (m) — smaller = sharper, less reach — and depth span (m).
    pub shadow_half_extent: f32,
    pub shadow_depth: f32,
    /// Reversed-Z ADD bias (kills acne) + along-normal offset (m, helps slopes).
    pub shadow_depth_bias: f32,
    pub shadow_normal_bias: f32,
    /// Carve dig-able caves into the voxel terrain.
    pub voxel_caves: bool,
    /// Cave carve strength (m).
    pub voxel_cave_strength: f32,
    /// CDLOD geomorph region: fraction of each leaf's LOD distance range over which the
    /// morph happens (1.0 = smoothest; lower = later/snappier).
    pub voxel_morph_region: f32,
    /// Mountain peak sharpness: 0 = rounded (generator default), 1 = cusped/pointy crests.
    pub voxel_peak_sharpen: f32,
    /// Ground detail-normal: strength (0 = off) + feature wavelength (m). terrain_csm.wgsl.
    pub voxel_detail_strength: f32,
    pub voxel_detail_scale: f32,
    /// POM parallax depth (m): 0 = flat normal-map only, higher = more apparent height.
    pub voxel_detail_depth: f32,
    /// Heightfield ground roughness 0..1 (0 = smooth, 1 = strong human-scale relief).
    pub voxel_ground_rough: f64,
    /// Draw the translucent ocean shell over the planet.
    pub ocean_enabled: bool,
    /// Sea level as a metre offset from the terrain's `e = 0` datum.
    pub sea_level_m: f64,
    /// Draw the FFT spectral wave surface (near-surface refractive detail).
    pub wave_enabled: bool,
    /// Wave horizontal displacement gain ("choppiness").
    pub wave_choppiness: f32,
    /// Jacobian value below which whitecap foam forms.
    pub wave_foam: f32,
    /// Show procedural trees on the surface (Phase B flora).
    pub flora_enabled: bool,
    /// Fraction of candidate cells that get a tree (0..1).
    pub flora_density: f32,
    /// Tree draw radius around the player (m) — far trees collapse to impostors.
    pub flora_radius_m: f64,
    /// Draw flora within this altitude of the surface (m) — fade in on descent.
    pub flora_alt_threshold_m: f64,
    /// Cycle nav mode (same as Tab).
    pub cycle_nav: bool,
    /// Exit the surface walker back to orbit (same as Esc).
    pub exit_surface: bool,
    /// User dismissed the load-stats popup this frame.
    pub dismiss_load_stats: bool,
    /// Sim time scale (sim-seconds per real-second) — drives orbits + day/night.
    pub time_scale: f64,
    /// Pause sim time (orbits + day/night freeze).
    pub paused: bool,
    /// Draw the wind streakline overlay.
    pub wind_enabled: bool,
    /// Wind overlay tunables (speed / width / altitude / gust / intensity).
    pub wind: crate::wind::WindParams,
    /// Draw the atmosphere shell.
    pub atmo_enabled: bool,
    /// Atmosphere tunables (height + density).
    pub atmo: crate::atmosphere::AtmoParams,
    /// Draw the depth-aware atmospheric scattering (aerial perspective + sky dome).
    pub aerial_enabled: bool,
    /// Aerial-scattering tunables (height + strength + intensity + sky brightness).
    pub aerial: crate::aerial::AerialParams,
    /// Draw the volumetric clouds.
    pub clouds_enabled: bool,
    /// Cloud tunables (coverage / density / altitude / wind speed / …).
    pub clouds: crate::clouds::CloudParams,
    /// Reference markers: pole spikes / equator ring.
    pub markers_poles: bool,
    pub markers_equator: bool,
}

// ── Profiler ──────────────────────────────────────────────────────────────

/// Per-frame perf counters for the top-right Stats panel. All ~1–2 frames stale
/// (read back from the GPU / measured on the prior frame) — fine for a HUD.
#[derive(Debug, Clone, Copy, Default)]
pub struct Profiler {
    /// CPU time spent recording the 3D frame (ms) — excludes the begin_frame
    /// fence wait + vsync, so it reflects real CPU cost even when GPU/vsync-bound.
    pub cpu_ms: f32,
    /// Whole-frame GPU time from timestamp queries (ms); 0.0 if unsupported.
    pub gpu_ms: f32,
    /// Triangles drawn by the terrain (voxel leaves / quadtree chunks) last frame.
    pub triangles: u64,
}

/// Compact count: `2.41M` / `18.3K` / `742`.
fn fmt_count(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.2}M", n as f64 / 1.0e6)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1.0e3)
    } else {
        n.to_string()
    }
}

// ── EguiState ─────────────────────────────────────────────────────────────

/// All egui + renderer state for one window.
pub struct EguiState {
    ctx:      Context,
    winit:    WinitState,
    renderer: Renderer,
    /// Cached FullOutput from the last `build_frame`.
    output:   Option<FullOutput>,
    /// Settings window open/closed (toggled by the ⚙ Settings button).
    show_settings: bool,
}

impl EguiState {
    /// Create egui state.
    ///
    /// `window` is used to initialise clipboard support (egui-winit needs a
    /// `HasDisplayHandle` implementor at construction time).
    pub fn new(rhi: &Rhi, window: &winit::window::Window) -> Self {
        let ctx = Context::default();

        // WinitState manages clipboard, cursor icon, and event translation.
        let winit = WinitState::new(
            ctx.clone(),
            ViewportId::ROOT,
            window,            // &dyn HasDisplayHandle
            None,              // native_pixels_per_point — auto-detect
            None,              // theme
            None,              // max_texture_side
        );

        let device   = rhi.device_handle();
        let instance = rhi.instance_handle();
        let physical = rhi.physical_device();

        let dynamic_rendering = DynamicRendering {
            color_attachment_format: rhi.swapchain_format(),
            // The UI pass has no depth attachment (egui has depth test/write off).
            // Pass None so the renderer's pipeline is created without a depth format.
            depth_attachment_format: None,
        };

        let options = Options {
            in_flight_frames:    2, // must match Rhi frames_in_flight
            srgb_framebuffer:    false,
            enable_depth_test:   false, // UI draws on top; no depth test
            enable_depth_write:  false,
        };

        let renderer = Renderer::with_default_allocator(
            instance,
            physical,
            device,
            dynamic_rendering,
            options,
        )
        .expect("failed to create egui-ash renderer");

        Self {
            ctx,
            winit,
            renderer,
            output: None,
            show_settings: false,
        }
    }

    /// Feed a winit window event to egui.
    ///
    /// Returns `true` if egui consumed the event (pointer or keyboard captured).
    /// When `true`, the caller should NOT forward the event to navigation.
    pub fn on_window_event(
        &mut self,
        window: &winit::window::Window,
        event: &winit::event::WindowEvent,
    ) -> bool {
        let resp = self.winit.on_window_event(window, event);
        resp.consumed
    }

    /// Build the egui debug panel and return the user's control changes.
    ///
    /// Must be called once per rendered frame, BEFORE `render` and BEFORE
    /// `rhi.begin_rendering(fi)`.  Texture uploads are submitted inside this
    /// call via a one-shot command buffer (outside the dynamic-rendering instance).
    ///
    /// Interactive widgets bind to the *current* values passed in, so the panel
    /// always reflects live state; the returned [`UiOutput`] carries the new
    /// values + one-shot actions for the caller to apply.
    #[allow(clippy::too_many_arguments)]
    pub fn build_frame(
        &mut self,
        window:            &winit::window::Window,
        rhi:               &Rhi,
        nav_mode:          NavMode,
        altitude_m:        f64,
        frame_time_s:      f32,
        view_mode:         u32,
        wireframe:         bool,
        aa_mode:           AaMode,
        shadow_map_enabled: bool,
        shadow_half_extent: f32,
        shadow_depth:       f32,
        shadow_depth_bias:  f32,
        shadow_normal_bias: f32,
        voxel_available:   bool,
        voxel_caves:       bool,
        voxel_cave_strength: f32,
        voxel_morph_region: f32,
        voxel_peak_sharpen: f32,
        voxel_detail_strength: f32,
        voxel_detail_scale: f32,
        voxel_detail_depth: f32,
        voxel_ground_rough: f64,
        ocean_enabled:     bool,
        sea_level_m:       f64,
        wave_enabled:      bool,
        wave_choppiness:   f32,
        wave_foam:         f32,
        flora_available:   bool,
        flora_enabled:     bool,
        flora_density:     f32,
        flora_radius_m:    f64,
        flora_alt_threshold_m: f64,
        time_scale:        f64,
        paused:            bool,
        wind_enabled:      bool,
        wind:              crate::wind::WindParams,
        atmo_enabled:      bool,
        atmo:              crate::atmosphere::AtmoParams,
        aerial_enabled:    bool,
        aerial:            crate::aerial::AerialParams,
        clouds_enabled:    bool,
        clouds:            crate::clouds::CloudParams,
        markers_poles:     bool,
        markers_equator:   bool,
        planet_stats:      Option<&PlanetViewStats>,
        load_stats:        Option<&LoadTimings>,
        profiler:          Profiler,
    ) -> UiOutput {
        let raw_input = self.winit.take_egui_input(window);

        let mut out = UiOutput {
            view_mode,
            wireframe,
            aa_mode,
            shadow_map_enabled,
            shadow_half_extent,
            shadow_depth,
            shadow_depth_bias,
            shadow_normal_bias,
            voxel_caves,
            voxel_cave_strength,
            voxel_morph_region,
            voxel_peak_sharpen,
            voxel_detail_strength,
            voxel_detail_scale,
            voxel_detail_depth,
            voxel_ground_rough,
            ocean_enabled,
            sea_level_m,
            wave_enabled,
            wave_choppiness,
            wave_foam,
            flora_enabled,
            flora_density,
            flora_radius_m,
            flora_alt_threshold_m,
            time_scale,
            paused,
            wind_enabled,
            wind,
            atmo_enabled,
            atmo,
            aerial_enabled,
            aerial,
            clouds_enabled,
            clouds,
            markers_poles,
            markers_equator,
            cycle_nav: false,
            exit_surface: false,
            dismiss_load_stats: false,
        };

        let mut show_settings = self.show_settings;
        let full_output = self.ctx.run(raw_input, |ctx| {
            // Standalone ⚙ button — a top-right screen overlay, always visible
            // (independent of the debug panel) — toggles the Settings window.
            egui::Area::new("settings_btn".into())
                .anchor(egui::Align2::RIGHT_TOP, egui::vec2(-8.0, 8.0))
                .show(ctx, |ui| {
                    if ui.button("⚙").clicked() {
                        show_settings = !show_settings;
                    }
                });

            // Settings window — toggled by the ⚙ overlay above. Order-independent in
            // egui; more graphics settings slot in here as the engine grows.
            egui::Window::new("⚙ Settings")
                .open(&mut show_settings)
                .resizable(true)
                .default_width(260.0)
                .show(ctx, |ui| {
                    ui.heading("Graphics");
                    egui::ComboBox::from_label("Anti-aliasing")
                        .selected_text(out.aa_mode.label())
                        .show_ui(ui, |ui| {
                            for m in AaMode::ALL {
                                ui.selectable_value(&mut out.aa_mode, m, m.label());
                            }
                        });
                    ui.label("MSAA 4× is always on; this picks the post-process / supersampling AA.");
                });

            egui::Window::new("minos · debug")
                .resizable(true)
                .default_pos([8.0, 8.0])
                .default_width(220.0)
                .show(ctx, |ui| {
                    use egui::CollapsingHeader;

                    // ── Navigation ───────────────────────────────────────────
                    CollapsingHeader::new("Navigation").default_open(true).show(ui, |ui| {
                        ui.label(format!("Mode: {nav_mode:?}"));
                        ui.label(format!("Altitude: {:.1} km", altitude_m / 1000.0));
                        ui.horizontal(|ui| {
                            if ui.button("Cycle mode").clicked() {
                                out.cycle_nav = true;
                            }
                            if nav_mode == NavMode::Surface
                                && ui.button("Exit to orbit").clicked()
                            {
                                out.exit_surface = true;
                            }
                        });
                    });

                    // (Performance / FPS moved to the top-right "Stats" panel.)

                    // ── Time (orbits + day/night speed) ──────────────────────
                    CollapsingHeader::new("Time").default_open(false).show(ui, |ui| {
                        ui.checkbox(&mut out.paused, "Pause");
                        ui.add(
                            egui::Slider::new(&mut out.time_scale, 1.0..=300_000.0)
                                .logarithmic(true)
                                .text("Sim speed (×real)"),
                        );
                    });

                    // ── View ─────────────────────────────────────────────────
                    CollapsingHeader::new("View").default_open(true).show(ui, |ui| {
                        // Triangle/Cluster/LOD work on the voxel terrain (modes 3/4/5
                        // in terrain.wgsl / terrain_csm.wgsl); shown when voxel is active.
                        ui.horizontal(|ui| {
                            ui.label("View:");
                            ui.selectable_value(&mut out.view_mode, 0, "Lit");
                            ui.selectable_value(&mut out.view_mode, 1, "Unlit");
                            ui.selectable_value(&mut out.view_mode, 2, "Normal");
                            if voxel_available {
                                ui.selectable_value(&mut out.view_mode, 3, "Triangle");
                                ui.selectable_value(&mut out.view_mode, 4, "Cluster");
                                ui.selectable_value(&mut out.view_mode, 5, "LOD");
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Planet:");
                            ui.selectable_value(&mut out.view_mode, 6, "Plate");
                            // Height/Material/Wetness/Volcano need per-vertex data the
                            // 4×vec3 format lacks. The voxel path bakes the selected field
                            // into vertex color on demand (remesh on switch). Greyed only
                            // on the classic quadtree path.
                            ui.add_enabled_ui(voxel_available, |ui| {
                                ui.selectable_value(&mut out.view_mode, 7, "Height");
                                ui.selectable_value(&mut out.view_mode, 8, "Material");
                                ui.selectable_value(&mut out.view_mode, 9, "Wetness");
                                ui.selectable_value(&mut out.view_mode, 10, "Volcano");
                            });
                        });
                        ui.horizontal(|ui| {
                            ui.label("Ocean:");
                            ui.selectable_value(&mut out.view_mode, 11, "Surface");
                            ui.selectable_value(&mut out.view_mode, 12, "Intensity");
                        });
                        ui.horizontal(|ui| {
                            ui.label("Clouds:");
                            ui.selectable_value(&mut out.view_mode, 13, "Density");
                        });
                        ui.checkbox(&mut out.wireframe, "Wireframe");
                        // Sun shadow map: re-homed onto the voxel terrain (Phase 3), so
                        // the controls show for a `voxel` build. Tunes the CSM that the
                        // character + trees receive on top of the SS shadow.
                        if voxel_available {
                            ui.add_enabled_ui(voxel_available, |ui| {
                                ui.checkbox(&mut out.shadow_map_enabled, "Sun shadows (map)");
                                if out.shadow_map_enabled {
                                    ui.add(egui::Slider::new(&mut out.shadow_half_extent, 8.0..=64.0)
                                        .text("Shadow base radius (m) — cascade 0; ×2 per cascade"));
                                    ui.add(egui::Slider::new(&mut out.shadow_depth, 60.0..=600.0)
                                        .text("Shadow depth span (m)"));
                                    ui.add(egui::Slider::new(&mut out.shadow_depth_bias, 0.0..=0.005)
                                        .text("Shadow depth bias"));
                                    ui.add(egui::Slider::new(&mut out.shadow_normal_bias, 0.0..=0.3)
                                        .text("Shadow normal bias (m)"));
                                }
                            });
                        }
                    });

                    // ── Ocean ────────────────────────────────────────────────
                    CollapsingHeader::new("Ocean").default_open(true).show(ui, |ui| {
                        ui.checkbox(&mut out.ocean_enabled, "Show ocean");
                        ui.add_enabled(
                            out.ocean_enabled,
                            egui::Slider::new(&mut out.sea_level_m, -1200.0..=1200.0)
                                .text("Sea level (m)")
                                .step_by(5.0),
                        );
                        ui.separator();
                        ui.checkbox(&mut out.wave_enabled, "FFT waves (near surface)");
                        ui.add_enabled(
                            out.wave_enabled,
                            egui::Slider::new(&mut out.wave_choppiness, 0.0..=2.5).text("Choppiness"),
                        );
                        ui.add_enabled(
                            out.wave_enabled,
                            egui::Slider::new(&mut out.wave_foam, 0.0..=1.0).text("Foam amount"),
                        );
                    });

                    // ── Wind ─────────────────────────────────────────────────
                    CollapsingHeader::new("Wind").default_open(true).show(ui, |ui| {
                        ui.checkbox(&mut out.wind_enabled, "Show wind streaks");
                        ui.add_enabled(out.wind_enabled,
                            egui::Slider::new(&mut out.wind.speed, 0.0..=0.4).text("Speed"));
                        ui.add_enabled(out.wind_enabled,
                            egui::Slider::new(&mut out.wind.width, 0.0005..=0.006).text("Streak width"));
                        ui.add_enabled(out.wind_enabled,
                            egui::Slider::new(&mut out.wind.altitude, 0.0..=8000.0).text("Altitude (m)"));
                        ui.add_enabled(out.wind_enabled,
                            egui::Slider::new(&mut out.wind.gust, 0.0..=1.5).text("Gust"));
                        ui.add_enabled(out.wind_enabled,
                            egui::Slider::new(&mut out.wind.intensity, 0.0..=2.0).text("Intensity"));
                    });

                    // ── Atmosphere ───────────────────────────────────────────
                    CollapsingHeader::new("Atmosphere").default_open(true).show(ui, |ui| {
                        ui.checkbox(&mut out.atmo_enabled, "Show atmosphere");
                        ui.add_enabled(out.atmo_enabled,
                            egui::Slider::new(&mut out.atmo.height, 200.0..=8000.0).text("Height (m)"));
                        ui.add_enabled(out.atmo_enabled,
                            egui::Slider::new(&mut out.atmo.density, 0.0..=2.0).text("Density"));
                    });

                    // ── Aerial scattering (depth-aware; OFF by default) ──────
                    CollapsingHeader::new("Aerial (scattering)").default_open(true).show(ui, |ui| {
                        ui.checkbox(&mut out.aerial_enabled, "Show aerial perspective (experimental)");
                        let on = out.aerial_enabled;
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.aerial.height, 1000.0..=40000.0).text("Top (m)"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.aerial.falloff, 200.0..=15000.0).text("Falloff (m)"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.aerial.beta, 1.0e-5..=3.0e-4)
                                .logarithmic(true).text("Strength (β 1/m)"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.aerial.intensity, 0.0..=2.0).text("Intensity"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.aerial.sky_strength, 0.0..=6.0).text("Sky brightness"));
                    });

                    // ── Clouds (OFF by default) ──────────────────────────────
                    CollapsingHeader::new("Clouds").default_open(true).show(ui, |ui| {
                        ui.checkbox(&mut out.clouds_enabled, "Show clouds (experimental)");
                        let on = out.clouds_enabled;
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.coverage, 0.0..=1.0).text("Coverage"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.density, 0.0..=0.08).text("Density"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.base_alt_m, 200.0..=8000.0).text("Base alt (m)"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.thickness_m, 500.0..=8000.0).text("Thickness (m)"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.wind_speed, 0.0..=6000.0).text("Wind speed (m/s)"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.noise_scale, 500.0..=8000.0).text("Feature size (m)"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.cloud_type, 0.0..=1.0).text("Type (stratus→cumulus)"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.moisture_influence, 0.0..=1.0).text("Climate influence"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.hg_g, 0.0..=0.95).text("Forward scatter"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.powder, 0.0..=1.0).text("Powder (dark edges)"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.curl, 0.0..=1.0).text("Curl turbulence"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.form_rate, 0.0..=0.5).text("Form rate (re-form)"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.decay_rate, 0.0..=0.2).text("Decay rate"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.steps, 16.0..=96.0).text("March steps"));
                    });

                    // ── Reference markers ────────────────────────────────────
                    CollapsingHeader::new("Reference").default_open(false).show(ui, |ui| {
                        ui.checkbox(&mut out.markers_equator, "Equator");
                        ui.checkbox(&mut out.markers_poles, "Poles + axis");
                    });

                    // ── Trees (flora) ────────────────────────────────────────
                    CollapsingHeader::new("Trees").default_open(true).show(ui, |ui| {
                        ui.add_enabled_ui(flora_available, |ui| {
                            ui.checkbox(&mut out.flora_enabled, "Show trees");
                            ui.add_enabled(
                                out.flora_enabled,
                                egui::Slider::new(&mut out.flora_density, 0.0..=1.0)
                                    .text("Density"),
                            );
                            // ponytail: max is bounded by the O(radius²) per-rebuild
                            // scatter SCAN (~60k climate+raycast evals at 2 km / 5 m
                            // spacing) — Tier-2 impostors make far DRAWS cheap but do
                            // nothing for the scan; a spatial-hash / cube-cell-id
                            // windowed rebuild is the path to a wider horizon.
                            ui.add_enabled(
                                out.flora_enabled,
                                egui::Slider::new(&mut out.flora_radius_m, 100.0..=2000.0)
                                    .text("Radius (m)"),
                            );
                            ui.add_enabled(
                                out.flora_enabled,
                                egui::Slider::new(&mut out.flora_alt_threshold_m, 0.0..=8000.0)
                                    .text("Fade-in altitude (m)"),
                            );
                        });
                        if !flora_available {
                            ui.small("Built without the `flora` feature.");
                        }
                    });

                    // ── Voxel terrain (only in a `--features voxel` build) ────
                    if voxel_available {
                        CollapsingHeader::new("Voxel").default_open(true).show(ui, |ui| {
                            ui.checkbox(&mut out.voxel_caves, "Caves (carve below surface)");
                            ui.add_enabled_ui(out.voxel_caves, |ui| {
                                ui.add(
                                    egui::Slider::new(&mut out.voxel_cave_strength, 0.0..=200.0)
                                        .text("Cave strength (m)"),
                                );
                            });
                            ui.add(
                                egui::Slider::new(&mut out.voxel_morph_region, 0.05..=1.0)
                                    .text("LOD morph region"),
                            );
                            ui.add(
                                egui::Slider::new(&mut out.voxel_peak_sharpen, 0.0..=1.0)
                                    .text("Peak sharpness"),
                            );
                            ui.add(
                                egui::Slider::new(&mut out.voxel_ground_rough, 0.0..=1.0)
                                    .text("Ground roughness (relief)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut out.voxel_detail_strength, 0.0..=2.0)
                                    .text("Ground detail"),
                            );
                            ui.add(
                                egui::Slider::new(&mut out.voxel_detail_scale, 0.3..=6.0)
                                    .text("Detail scale (m)"),
                            );
                            ui.add(
                                egui::Slider::new(&mut out.voxel_detail_depth, 0.0..=1.0)
                                    .text("Detail depth (m, POM)"),
                            );
                        });
                    }

                    // ── Planet LOD stats ─────────────────────────────────────
                    if let Some(ps) = planet_stats {
                        CollapsingHeader::new("Planet LOD").default_open(true).show(ui, |ui| {
                            ui.label(format!("Resident chunks: {}", ps.resident_count));
                            ui.label(format!("Build queue: {}", ps.build_queue_depth));
                            ui.label(format!(
                                "LOD levels: {}-{}",
                                ps.min_lod_level, ps.max_lod_level
                            ));
                        });
                    }

                    CollapsingHeader::new("Key hints").show(ui, |ui| {
                        ui.label("[M] cycle material");
                        ui.label("[W] toggle wireframe");
                        ui.label("[Tab] cycle nav mode");
                        ui.label("[V] 1st/3rd-person (surface)");
                        ui.label("[Esc] exit to orbit");
                    });
                });

            // ── Top-right Stats / profiler panel ─────────────────────────────
            // y=44 clears the ⚙ settings overlay button (top-right, ~26 px) above it.
            egui::Window::new("Stats")
                .anchor(egui::Align2::RIGHT_TOP, [-8.0, 44.0])
                .resizable(false)
                .default_width(150.0)
                .show(ctx, |ui| {
                    let fps = if frame_time_s > 0.0 { 1.0 / frame_time_s } else { 0.0 };
                    let fps_color = if fps >= 50.0 {
                        egui::Color32::from_rgb(120, 230, 120)
                    } else if fps >= 30.0 {
                        egui::Color32::from_rgb(230, 220, 120)
                    } else {
                        egui::Color32::from_rgb(230, 120, 120)
                    };
                    let ext = rhi.extent();
                    let gpu = if profiler.gpu_ms > 0.0 {
                        format!("{:.2} ms", profiler.gpu_ms)
                    } else {
                        "—".to_string()
                    };
                    egui::Grid::new("stats_grid").num_columns(2).striped(true).show(ui, |ui| {
                        ui.label("FPS");
                        ui.colored_label(fps_color, format!("{fps:.0}  ({:.2} ms)", frame_time_s * 1000.0));
                        ui.end_row();
                        ui.label("CPU");
                        ui.label(format!("{:.2} ms", profiler.cpu_ms));
                        ui.end_row();
                        ui.label("GPU");
                        ui.label(gpu);
                        ui.end_row();
                        ui.label("Triangles");
                        ui.label(fmt_count(profiler.triangles));
                        ui.end_row();
                        ui.label("Resolution");
                        ui.label(format!("{}×{}", ext.width, ext.height));
                        ui.end_row();
                    });
                });

            // One-time load-stats popup: total + per-stage breakdown. Shown until
            // the user clicks OK or closes it (caller stops passing `load_stats`).
            if let Some(t) = load_stats {
                let total = t.total_ms.max(1.0);
                let mut open = true;
                egui::Window::new("Load complete")
                    .collapsible(false)
                    .resizable(false)
                    .anchor(egui::Align2::CENTER_TOP, [0.0, 48.0])
                    .open(&mut open)
                    .show(ctx, |ui| {
                        ui.heading(format!("Loaded in {:.1} s", t.total_ms / 1000.0));
                        ui.add_space(6.0);
                        egui::Grid::new("load_stats").num_columns(3).striped(true).show(ui, |ui| {
                            let row = |ui: &mut egui::Ui, name: &str, ms: f64, pct: bool| {
                                ui.label(name);
                                ui.label(format!("{:.2} s", ms / 1000.0));
                                ui.label(if pct { format!("{:.0}%", 100.0 * ms / total) } else { String::new() });
                                ui.end_row();
                            };
                            row(ui, "Heightfield", t.heightfield_ms, true);
                        });
                        ui.add_space(8.0);
                        if ui.button("OK").clicked() {
                            out.dismiss_load_stats = true;
                        }
                    });
                if !open {
                    out.dismiss_load_stats = true;
                }
            }
        });
        self.show_settings = show_settings;

        // Upload any new/changed textures (font atlas, etc.) to the GPU.
        // `set_textures` does an immediate one-shot submit internally.
        // This MUST happen outside the dynamic-rendering instance.
        if !full_output.textures_delta.set.is_empty() {
            let queue = rhi.queue_handle();
            // Use frame-slot 0's command pool for texture uploads.  This pool
            // is not recording between end_frame and begin_frame, so it is
            // safe to use here (outside any frame recording session).
            let pool = rhi.command_pool(0);
            if let Err(e) = self
                .renderer
                .set_textures(queue, pool, full_output.textures_delta.set.as_slice())
            {
                log::error!("egui set_textures failed: {e}");
            }
        }

        // Handle platform output (clipboard writes, cursor shape changes, etc.)
        self.winit
            .handle_platform_output(window, full_output.platform_output.clone());

        self.output = Some(full_output);

        out
    }

    /// Build a centered loading screen (progress bar + spinner) for frame slot
    /// `fi`. Mirrors `build_frame`'s texture-upload handling; call it in place of
    /// `build_frame` while the async loader is still running, then `render` as usual.
    pub fn loading_frame(
        &mut self,
        window: &winit::window::Window,
        rhi: &Rhi,
        fraction: f32,
        message: &str,
    ) {
        let raw_input = self.winit.take_egui_input(window);

        let full_output = self.ctx.run(raw_input, |ctx| {
            egui::Area::new(egui::Id::new("loading"))
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.heading("Loading minos…");
                        ui.add_space(10.0);
                        ui.label(message);
                        ui.add_space(10.0);
                        ui.add(
                            egui::ProgressBar::new(fraction)
                                .desired_width(300.0)
                                .show_percentage(),
                        );
                        ui.add_space(10.0);
                        ui.spinner();
                    });
                });
        });

        if !full_output.textures_delta.set.is_empty() {
            let queue = rhi.queue_handle();
            let pool = rhi.command_pool(0);
            if let Err(e) =
                self.renderer
                    .set_textures(queue, pool, full_output.textures_delta.set.as_slice())
            {
                log::error!("egui set_textures (loading) failed: {e}");
            }
        }

        self.winit
            .handle_platform_output(window, full_output.platform_output.clone());
        self.output = Some(full_output);
    }

    /// Record egui draw commands into the current frame's command buffer.
    ///
    /// Must be called AFTER `rhi.begin_ui_pass(fi)` and BEFORE `rhi.end_frame(fi)`.
    /// The commands land inside the 1-sample UI rendering instance opened by
    /// `begin_ui_pass`, NOT the MSAA 3D instance.  `egui-ash-renderer` 0.11
    /// hardcodes `rasterizationSamples = TYPE_1` and requires a 1-sample target.
    pub fn render(&mut self, rhi: &Rhi, fi: u32) {
        let output = match self.output.take() {
            Some(o) => o,
            None    => return,
        };

        let pixels_per_point = output.pixels_per_point;
        let primitives = self.ctx.tessellate(output.shapes, pixels_per_point);

        if !primitives.is_empty() {
            let cmd    = rhi.current_command_buffer(fi);
            let extent = rhi.extent();

            if let Err(e) = self.renderer.cmd_draw(cmd, extent, pixels_per_point, &primitives) {
                log::error!("egui cmd_draw failed: {e}");
            }
        }

        // Free any textures egui is done with.
        if !output.textures_delta.free.is_empty() {
            if let Err(e) = self.renderer.free_textures(&output.textures_delta.free) {
                log::error!("egui free_textures failed: {e}");
            }
        }
    }

    /// `true` if egui wants to capture pointer input this frame.
    ///
    /// The primary guard is `on_window_event().consumed`, but callers can also
    /// query this directly for pointer-lock decisions.
    #[allow(dead_code)]
    pub fn wants_pointer(&self) -> bool {
        self.ctx.wants_pointer_input()
    }

    /// `true` if egui wants to capture keyboard input this frame.
    #[allow(dead_code)]
    pub fn wants_keyboard(&self) -> bool {
        self.ctx.wants_keyboard_input()
    }
}

//! `gui` — interactive egui debug panel for enki-app.
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

use enki_rhi::Rhi;

use crate::controls::nav_mode::NavMode;
use crate::loading::LoadTimings;
use crate::planet_view::PlanetViewStats;

// ── UiOutput ────────────────────────────────────────────────────────────────

/// Control changes produced by the debug panel in one frame.
///
/// The caller applies these to app state after `build_frame` returns: the
/// settings fields are the new values; the action flags are one-shot edges.
#[derive(Debug, Clone, Copy)]
pub struct UiOutput {
    /// Unified view mode 0–10 — View: 0 Lit, 1 Unlit, 2 Normal, 3 Triangle,
    /// 4 Cluster, 5 LOD (3–5 Nanite-only); Planet: 6 Plate, 7 Height, 8 Material,
    /// 9 Wetness, 10 Volcano.
    pub view_mode: u32,
    pub wireframe: bool,
    /// Temporal anti-aliasing (hides discrete-LOD shimmer + edge aliasing).
    pub taa: bool,
    pub nanite_enabled: bool,
    /// LOD pixel-error threshold (lower = finer / smoother LOD, heavier).
    pub nanite_tau: f32,
    /// Draw the translucent ocean shell over the planet.
    pub ocean_enabled: bool,
    /// Sea level as a metre offset from the terrain's `e = 0` datum.
    pub sea_level_m: f64,
    /// Draw the FFT spectral wave surface (near-surface detail; needs TAA on).
    pub wave_enabled: bool,
    /// Wave horizontal displacement gain ("choppiness").
    pub wave_choppiness: f32,
    /// Jacobian value below which whitecap foam forms.
    pub wave_foam: f32,
    /// Show procedural trees on the surface (Phase B flora).
    pub flora_enabled: bool,
    /// Fraction of candidate cells that get a tree (0..1).
    pub flora_density: f32,
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
    /// Draw the volumetric clouds.
    pub clouds_enabled: bool,
    /// Cloud tunables (coverage / density / altitude / wind speed / …).
    pub clouds: crate::clouds::CloudParams,
    /// Reference markers: pole spikes / equator ring.
    pub markers_poles: bool,
    pub markers_equator: bool,
    /// Draw the traced river network.
    pub rivers_enabled: bool,
}

// ── EguiState ─────────────────────────────────────────────────────────────

/// All egui + renderer state for one window.
pub struct EguiState {
    ctx:      Context,
    winit:    WinitState,
    renderer: Renderer,
    /// Cached FullOutput from the last `build_frame`.
    output:   Option<FullOutput>,
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
        taa:               bool,
        nanite_enabled:    bool,
        nanite_tau:        f32,
        nanite_available:  bool,
        ocean_enabled:     bool,
        sea_level_m:       f64,
        wave_enabled:      bool,
        wave_choppiness:   f32,
        wave_foam:         f32,
        flora_available:   bool,
        flora_enabled:     bool,
        flora_density:     f32,
        time_scale:        f64,
        paused:            bool,
        wind_enabled:      bool,
        wind:              crate::wind::WindParams,
        atmo_enabled:      bool,
        atmo:              crate::atmosphere::AtmoParams,
        clouds_enabled:    bool,
        clouds:            crate::clouds::CloudParams,
        markers_poles:     bool,
        markers_equator:   bool,
        rivers_enabled:    bool,
        stress_stats:      Option<&str>,
        planet_stats:      Option<&PlanetViewStats>,
        load_stats:        Option<&LoadTimings>,
    ) -> UiOutput {
        let raw_input = self.winit.take_egui_input(window);

        let mut out = UiOutput {
            view_mode,
            wireframe,
            taa,
            nanite_enabled,
            nanite_tau,
            ocean_enabled,
            sea_level_m,
            wave_enabled,
            wave_choppiness,
            wave_foam,
            flora_enabled,
            flora_density,
            time_scale,
            paused,
            wind_enabled,
            wind,
            atmo_enabled,
            atmo,
            clouds_enabled,
            clouds,
            markers_poles,
            markers_equator,
            rivers_enabled,
            cycle_nav: false,
            exit_surface: false,
            dismiss_load_stats: false,
        };

        let full_output = self.ctx.run(raw_input, |ctx| {
            egui::Window::new("enki · debug")
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

                    // ── Performance ──────────────────────────────────────────
                    CollapsingHeader::new("Performance").default_open(true).show(ui, |ui| {
                        let fps = if frame_time_s > 0.0 { 1.0 / frame_time_s } else { 0.0 };
                        ui.label(format!("FPS: {fps:.0}   ({:.2} ms)", frame_time_s * 1000.0));
                    });

                    // ── Time (orbits + day/night speed) ──────────────────────
                    CollapsingHeader::new("Time").default_open(false).show(ui, |ui| {
                        ui.checkbox(&mut out.paused, "Pause");
                        ui.add(
                            egui::Slider::new(&mut out.time_scale, 60.0..=300_000.0)
                                .logarithmic(true)
                                .text("Sim speed (×real)"),
                        );
                    });

                    // ── View ─────────────────────────────────────────────────
                    CollapsingHeader::new("View").default_open(true).show(ui, |ui| {
                        // Geometry-debug views (Triangle/Cluster/LOD) need the Nanite path.
                        let nanite_active = nanite_available && out.nanite_enabled;
                        ui.horizontal(|ui| {
                            ui.label("View:");
                            ui.selectable_value(&mut out.view_mode, 0, "Lit");
                            ui.selectable_value(&mut out.view_mode, 1, "Unlit");
                            ui.selectable_value(&mut out.view_mode, 2, "Normal");
                            ui.add_enabled_ui(nanite_active, |ui| {
                                ui.selectable_value(&mut out.view_mode, 3, "Triangle");
                                ui.selectable_value(&mut out.view_mode, 4, "Cluster");
                                ui.selectable_value(&mut out.view_mode, 5, "LOD");
                            });
                        });
                        ui.horizontal(|ui| {
                            ui.label("Planet:");
                            ui.selectable_value(&mut out.view_mode, 6, "Plate");
                            ui.selectable_value(&mut out.view_mode, 7, "Height");
                            ui.selectable_value(&mut out.view_mode, 8, "Material");
                            ui.selectable_value(&mut out.view_mode, 9, "Wetness");
                            ui.selectable_value(&mut out.view_mode, 10, "Volcano");
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
                        ui.checkbox(&mut out.taa, "TAA (temporal AA)");
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

                    // ── Clouds (needs TAA on) ────────────────────────────────
                    CollapsingHeader::new("Clouds").default_open(true).show(ui, |ui| {
                        ui.checkbox(&mut out.clouds_enabled, "Show clouds");
                        let on = out.clouds_enabled;
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.coverage, 0.0..=1.0).text("Coverage"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.density, 0.0..=0.25).text("Density"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.base_alt_m, 200.0..=8000.0).text("Base alt (m)"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.thickness_m, 500.0..=8000.0).text("Thickness (m)"));
                        ui.add_enabled(on,
                            egui::Slider::new(&mut out.clouds.wind_speed, 0.0..=300.0).text("Wind speed (m/s)"));
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
                            egui::Slider::new(&mut out.clouds.steps, 16.0..=96.0).text("March steps"));
                    });

                    // ── Reference markers ────────────────────────────────────
                    CollapsingHeader::new("Reference").default_open(false).show(ui, |ui| {
                        ui.checkbox(&mut out.markers_equator, "Equator");
                        ui.checkbox(&mut out.markers_poles, "Poles + axis");
                        ui.checkbox(&mut out.rivers_enabled, "Rivers");
                    });

                    // ── Trees (flora) ────────────────────────────────────────
                    CollapsingHeader::new("Trees").default_open(true).show(ui, |ui| {
                        ui.add_enabled_ui(flora_available, |ui| {
                            ui.checkbox(&mut out.flora_enabled, "Show trees (surface)");
                            ui.add_enabled(
                                out.flora_enabled,
                                egui::Slider::new(&mut out.flora_density, 0.0..=1.0)
                                    .text("Density"),
                            );
                        });
                        if !flora_available {
                            ui.small("Built without the `flora` feature.");
                        }
                    });

                    // ── Nanite ───────────────────────────────────────────────
                    CollapsingHeader::new("Nanite").default_open(true).show(ui, |ui| {
                        ui.add_enabled_ui(nanite_available, |ui| {
                            ui.checkbox(&mut out.nanite_enabled, "Enable Nanite");
                            ui.add(
                                egui::Slider::new(&mut out.nanite_tau, 0.25..=8.0)
                                    .text("LOD threshold (px)"),
                            );
                        });
                        if !nanite_available {
                            ui.small("Built without the `nanite` feature.");
                        }
                    });

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

                    // ── Stress stats ─────────────────────────────────────────
                    if let Some(stats) = stress_stats {
                        CollapsingHeader::new("Stress").default_open(true).show(ui, |ui| {
                            ui.label(stats);
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
                            // Bake fields are 0 without the `nanite` feature — hide them then.
                            if t.bake_wall_ms > 0.0 {
                                row(ui, "Nanite bake", t.bake_wall_ms, true);
                                row(ui, "    tessellate", t.tessellate_ms, false);
                                row(ui, "    clusters", t.clusters_ms, false);
                                row(ui, "    dag", t.dag_ms, false);
                            }
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
                        ui.heading("Loading enki…");
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

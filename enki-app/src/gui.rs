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
use crate::planet_view::PlanetViewStats;

// ── UiOutput ────────────────────────────────────────────────────────────────

/// Control changes produced by the debug panel in one frame.
///
/// The caller applies these to app state after `build_frame` returns: the
/// settings fields are the new values; the action flags are one-shot edges.
#[derive(Debug, Clone, Copy)]
pub struct UiOutput {
    pub material_mode: u32,
    pub wireframe: bool,
    /// Temporal anti-aliasing (hides discrete-LOD shimmer + edge aliasing).
    pub taa: bool,
    pub nanite_enabled: bool,
    /// 0 = Off, 1 = Triangle, 2 = Cluster, 3 = LOD.
    pub nanite_debug_mode: u32,
    /// LOD pixel-error threshold (lower = finer / smoother LOD, heavier).
    pub nanite_tau: f32,
    /// Cycle nav mode (same as Tab).
    pub cycle_nav: bool,
    /// Exit first-person (same as Esc).
    pub exit_first_person: bool,
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
        material_mode:     u32,
        wireframe:         bool,
        taa:               bool,
        nanite_enabled:    bool,
        nanite_debug_mode: u32,
        nanite_tau:        f32,
        nanite_available:  bool,
        stress_stats:      Option<&str>,
        planet_stats:      Option<&PlanetViewStats>,
    ) -> UiOutput {
        let raw_input = self.winit.take_egui_input(window);

        let mut out = UiOutput {
            material_mode,
            wireframe,
            taa,
            nanite_enabled,
            nanite_debug_mode,
            nanite_tau,
            cycle_nav: false,
            exit_first_person: false,
        };

        let full_output = self.ctx.run(raw_input, |ctx| {
            egui::Window::new("enki · debug")
                .resizable(true)
                .default_pos([8.0, 8.0])
                .default_width(220.0)
                .show(ctx, |ui| {
                    // ── Navigation ───────────────────────────────────────────
                    ui.heading("Navigation");
                    ui.label(format!("Mode: {nav_mode:?}"));
                    ui.label(format!("Altitude: {:.1} km", altitude_m / 1000.0));
                    ui.horizontal(|ui| {
                        if ui.button("Cycle mode").clicked() {
                            out.cycle_nav = true;
                        }
                        if nav_mode == NavMode::FirstPerson
                            && ui.button("Exit first-person").clicked()
                        {
                            out.exit_first_person = true;
                        }
                    });

                    ui.separator();

                    // ── Performance ──────────────────────────────────────────
                    ui.heading("Performance");
                    let fps = if frame_time_s > 0.0 { 1.0 / frame_time_s } else { 0.0 };
                    ui.label(format!("FPS: {fps:.0}   ({:.2} ms)", frame_time_s * 1000.0));

                    ui.separator();

                    // ── View ─────────────────────────────────────────────────
                    ui.heading("View");
                    ui.horizontal(|ui| {
                        ui.label("Material:");
                        for m in 0..4u32 {
                            ui.selectable_value(&mut out.material_mode, m, m.to_string());
                        }
                    });
                    ui.checkbox(&mut out.wireframe, "Wireframe");
                    ui.checkbox(&mut out.taa, "TAA (temporal AA)");

                    ui.separator();

                    // ── Nanite (debug) ───────────────────────────────────────
                    // Present now so the panel layout is visible; these controls
                    // go live when N2 (the runtime cluster renderer) lands.
                    ui.heading("Nanite (debug)");
                    ui.add_enabled_ui(nanite_available, |ui| {
                        ui.checkbox(&mut out.nanite_enabled, "Enable Nanite view");
                        ui.horizontal(|ui| {
                            ui.label("Color:");
                            ui.selectable_value(&mut out.nanite_debug_mode, 0, "Off");
                            ui.selectable_value(&mut out.nanite_debug_mode, 1, "Triangle");
                            ui.selectable_value(&mut out.nanite_debug_mode, 2, "Cluster");
                            ui.selectable_value(&mut out.nanite_debug_mode, 3, "LOD");
                        });
                        ui.add(
                            egui::Slider::new(&mut out.nanite_tau, 0.25..=8.0)
                                .text("LOD threshold (px)"),
                        );
                    });
                    if !nanite_available {
                        ui.small("Build with the `nanite` feature to enable.");
                    }

                    // ── Planet LOD stats ─────────────────────────────────────
                    if let Some(ps) = planet_stats {
                        ui.separator();
                        ui.heading("Planet LOD");
                        ui.label(format!("Resident chunks: {}", ps.resident_count));
                        ui.label(format!("Build queue: {}", ps.build_queue_depth));
                        ui.label(format!(
                            "LOD levels: {}-{}",
                            ps.min_lod_level, ps.max_lod_level
                        ));
                    }

                    // ── Stress stats ─────────────────────────────────────────
                    if let Some(stats) = stress_stats {
                        ui.separator();
                        ui.heading("Stress");
                        ui.label(stats);
                    }

                    ui.separator();
                    ui.collapsing("Key hints", |ui| {
                        ui.label("[M] cycle material");
                        ui.label("[W] toggle wireframe");
                        ui.label("[Tab] cycle nav mode");
                        ui.label("[Esc] exit first-person");
                    });
                });
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

// flora_viewer — standalone enki-flora debug viewer with a dryad-CAS egui panel.
//
// ponytail: the smallest host that lets us LOOK at one tree, reseed, AND DRIVE
// the morphospace live. It owns its own winit window + Rhi + swapchain + frame
// loop, builds ONE FloraView at the world origin, and auto-orbits a camera
// framed to the branch-mesh AABB. NO planet / nanite / TAA / nav — just the
// 6-call frame loop from the bootstrap recon, plus an egui panel.
//
// The egui panel is a SCHEMA-DRIVEN copy of enki's EguiState plumbing
// (gui.rs new/on_window_event/render are copy-identical; only build_frame is
// flora-specific): gene sliders grouped by tier, climate sliders, presets,
// generate/reseed, and stats. Editing any gene/env LIVE-REBUILDS the tree via
// FloraView::from_genome — the core dryad CAS loop.
//
// Run:  cargo run -p enki-app --bin flora_viewer --features flora
//
// Keys (all // ponytail: debug shortcuts):
//   Esc        quit
//   Space / N  next tree (random_genome with seed+1)

use std::time::Instant;

use enki_app::flora_render::{FloraRenderer, ShadowUniforms};
use enki_app::flora_view::{FloraView, LeafMode, RenderMode};
#[cfg(all(feature = "flora", feature = "nanite"))]
use enki_app::flora_nanite::FloraNanite;
use enki_app::staging::Staging;
use enki_flora::genome::{random_genome, Env, Genome, Medium};
use enki_render::{camera::Camera, frame::FrameUniforms, lights::Lights};
use enki_rhi::Rhi;
use enki_rhi::RhiError;
use enki_rhi::RhiConfig;
use glam::{DVec3, Mat4, Quat, Vec3};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::{ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowAttributes, WindowId},
};

// ponytail: presets live in a subdir (NOT a sibling `bin/presets.rs`) so cargo's
// auto-bin discovery doesn't pick them up as a second, main-less binary target.
#[path = "flora_viewer/presets.rs"]
mod presets;

// ── Render mode ──────────────────────────────────────────────────────────────
// The `RenderMode` enum lives in flora_view (it owns the pipeline selection +
// debug_mode push lane); the viewer just drives the selector and forwards it via
// `FloraView::set_render_mode`.

/// Selectable render modes + their labels, in panel order. The Triangle/Cluster/
/// LOD trio drives the enki-nanite virtualized-geometry debug views and is only
/// compiled in when BOTH `flora` and `nanite` are on (the Nanite branch path).
#[cfg(all(feature = "flora", feature = "nanite"))]
const RENDER_MODES: [(RenderMode, &str); 8] = [
    (RenderMode::Lit, "Lit"),
    (RenderMode::Unlit, "Unlit"),
    (RenderMode::Wireframe, "Wireframe"),
    (RenderMode::Normals, "Normals"),
    (RenderMode::Ao, "AO"),
    (RenderMode::Triangle, "Triangle"),
    (RenderMode::Cluster, "Cluster"),
    (RenderMode::Lod, "LOD"),
];
#[cfg(not(all(feature = "flora", feature = "nanite")))]
const RENDER_MODES: [(RenderMode, &str); 5] = [
    (RenderMode::Lit, "Lit"),
    (RenderMode::Unlit, "Unlit"),
    (RenderMode::Wireframe, "Wireframe"),
    (RenderMode::Normals, "Normals"),
    (RenderMode::Ao, "AO"),
];

// ── CAS tabs ─────────────────────────────────────────────────────────────────
// dryad's Sims-CAS left panel groups the genes by BODY PART. The tab strip mirrors
// dryad's ACTUAL index.html tab row EXACTLY — five tabs: Climate, Trunk, Branches,
// Leaves, Root. Climate holds the env sliders; the other four are curated, ordered
// gene lists driven by `GENE_TABS` below (label + slider range per dryad's UI feel,
// which differs from the schema clamp ranges — so we drive the sliders from an
// explicit table, NOT from raw `fields_mut()` ordering).

/// The active left-panel tab. Held in `App` and passed `&mut` into `build_frame`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Climate,
    Trunk,
    Branches,
    Leaves,
    Root,
}

/// Tab strip order + labels — dryad's exact five-tab row.
const TABS: [(Tab, &str); 5] = [
    (Tab::Climate, "Climate"),
    (Tab::Trunk, "Trunk"),
    (Tab::Branches, "Branches"),
    (Tab::Leaves, "Leaves"),
    (Tab::Root, "Root"),
];

/// One curated gene slider: the genome's JS field name (matches
/// `Genome::fields_mut()`), the dryad UI label, and the dryad slider range
/// (min, max). The range is the UI feel range from dryad's index.html — it can
/// differ from the schema clamp range, so we drive the slider extents from here.
struct GeneSlider {
    /// camelCase JS name, used to find the live `&mut f64` in `fields_mut()`.
    name: &'static str,
    /// dryad's human label.
    label: &'static str,
    min: f64,
    max: f64,
}

/// `GeneSlider` shorthand.
const fn g(name: &'static str, label: &'static str, min: f64, max: f64) -> GeneSlider {
    GeneSlider { name, label, min, max }
}

// Per-tab curated, ordered gene slider lists — dryad's exact membership +
// labels + ranges for the four gene tabs (Climate is env, handled inline). Any
// gene NOT listed is intentionally omitted (these are dryad's exact tab sets).
// The slider's live value comes from the genome via `fields_mut()`; these tables
// only fix ORDER, LABEL, and RANGE. They are `const` so `gene_tab_sliders` can
// return a `'static` slice.

const TRUNK_SLIDERS: &[GeneSlider] = &[
    g("stemGirth", "Stem girth", 0.0, 1.0),
    g("taper", "Taper", 0.0, 1.0),
    g("trunkHeight", "Trunk height", 0.0, 1.0),
    g("trunkTaper", "Trunk taper", 0.0, 1.0),
    g("succulence", "Succulence", 0.0, 1.0),
    g("ribbing", "Ribbing", 0.0, 1.0),
    g("segmentation", "Segmentation", 0.0, 1.0),
    g("spininess", "Spininess", 0.0, 1.0),
    g("woodiness", "Woodiness", 0.0, 1.0),
    g("barkHue", "Bark hue", 0.0, 1.0),
    g("barkLightness", "Bark lightness", 0.0, 1.0),
    g("barkRelief", "Bark relief", 0.0, 1.0),
    g("barkLenticels", "Bark lenticels", 0.0, 1.0),
    g("barkScale", "Bark scale", 0.0, 1.0),
    g("barkOrient", "Bark orient", 0.0, 1.0),
    g("barkPlates", "Bark plates", 0.0, 1.0),
    g("barkShed", "Bark shed", 0.0, 1.0),
    g("barkUnderHue", "Bark under-hue", 0.0, 1.0),
];

const BRANCHES_SLIDERS: &[GeneSlider] = &[
    g("branchiness", "Branchiness", 0.0, 1.0),
    g("branchFactorN", "Branch factor N", 0.0, 1.0),
    g("tillering", "Tillering", 0.0, 1.0),
    g("radialOrder", "Radial order", 0.0, 1.0),
    g("whorl", "Whorl", 0.0, 1.0),
    g("crownStart", "Crown start", 0.0, 1.0),
    g("branchAngle", "Branch angle", 0.35, 0.80),
    g("lengthRatio", "Length ratio", 0.55, 0.85),
    g("apicalBias", "Apical bias", 0.0, 1.0),
    g("droopBias", "Droop bias", -0.2, 0.4),
    g("rigidity", "Rigidity", 0.0, 1.0),
    g("verticality", "Verticality", 0.0, 1.0),
    g("weep", "Weep", 0.0, 1.0),
];

const LEAVES_SLIDERS: &[GeneSlider] = &[
    g("leafSize", "Leaf size", 0.6, 1.6),
    g("leafWidth", "Leaf breadth", 0.0, 1.0),
    g("leafLength", "Leaf length", 0.0, 1.0),
    g("leafTip", "Leaf tip", 0.0, 1.0),
    g("leafSerration", "Leaf serration", 0.0, 1.0),
    g("leafLobing", "Leaf lobing", 0.0, 1.0),
    g("leafSkew", "Leaf skew", 0.0, 1.0),
    g("leafDivision", "Leaf division", 0.0, 1.0),
    g("frondFan", "Frond fan", 0.0, 1.0),
    g("tipTuft", "Tip tuft", 0.0, 1.0),
    g("appendageBreadth", "Appendage breadth", 0.0, 1.0),
    g("appendageDensity", "Leaf density", 0.0, 1.5),
    g("pigment", "Pigment", 0.0, 1.0),
    g("jitter", "Jitter", 0.0, 1.5),
];

const ROOT_SLIDERS: &[GeneSlider] = &[
    g("rootCount", "Root count", 0.0, 1.0),
    g("rootDepth", "Root depth", 0.0, 1.0),
    g("rootSpread", "Root spread", 0.0, 1.0),
    g("rootFlare", "Root flare", 0.0, 1.0),
    g("rootButtress", "Buttress", 0.0, 1.0),
    g("rootBranchiness", "Root branchiness", 0.0, 1.0),
    g("rootTaper", "Root taper", 0.0, 1.0),
];

/// Curated slider list for a gene tab (Climate is env-driven, handled inline).
fn gene_tab_sliders(tab: Tab) -> &'static [GeneSlider] {
    match tab {
        Tab::Trunk => TRUNK_SLIDERS,
        Tab::Branches => BRANCHES_SLIDERS,
        Tab::Leaves => LEAVES_SLIDERS,
        Tab::Root => ROOT_SLIDERS,
        Tab::Climate => &[],
    }
}

// ── egui harness (flora-specific build_frame; rest copied from gui.rs) ───────

/// All egui + renderer state for the viewer window. Mirrors `gui::EguiState`;
/// `new`/`on_window_event`/`render` are copy-identical (they only touch Rhi +
/// winit + egui). Only `build_frame` differs (flora panel, not the planet HUD).
struct FloraGui {
    ctx: egui::Context,
    winit: egui_winit::State,
    renderer: egui_ash_renderer::Renderer,
    output: Option<egui::FullOutput>,
}

/// What the panel changed this frame, for the viewer to apply after build.
#[derive(Default)]
struct PanelOut {
    /// A gene/env slider or preset changed → rebuild the tree from the genome.
    rebuild: bool,
    /// 'Generate' clicked → reroll seed + random_genome(env), then rebuild.
    generate: bool,
    /// Render-mode selector changed → re-apply via FloraView::set_render_mode
    /// (no tree rebuild; just a pipeline/push-lane switch).
    mode_changed: bool,
}

impl FloraGui {
    fn new(rhi: &Rhi, window: &Window) -> Self {
        let ctx = egui::Context::default();
        let winit = egui_winit::State::new(
            ctx.clone(),
            egui::ViewportId::ROOT,
            window,
            None,
            None,
            None,
        );

        let dynamic_rendering = egui_ash_renderer::DynamicRendering {
            color_attachment_format: rhi.swapchain_format(),
            depth_attachment_format: None,
        };
        let options = egui_ash_renderer::Options {
            in_flight_frames: 2,
            srgb_framebuffer: false,
            enable_depth_test: false,
            enable_depth_write: false,
        };
        let renderer = egui_ash_renderer::Renderer::with_default_allocator(
            rhi.instance_handle(),
            rhi.physical_device(),
            rhi.device_handle(),
            dynamic_rendering,
            options,
        )
        .expect("failed to create egui-ash renderer");

        Self { ctx, winit, renderer, output: None }
    }

    fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.winit.on_window_event(window, event).consumed
    }

    fn wants_keyboard(&self) -> bool {
        self.ctx.wants_keyboard_input()
    }

    /// Build the flora CAS panel. Binds sliders to the live `genome`/`env`, so
    /// the panel always reflects on-screen state; returns the actions the viewer
    /// applies (rebuild / generate). Must run OUTSIDE the rendering instance
    /// (texture upload does a one-shot submit). Mirrors `gui::build_frame`.
    #[allow(clippy::too_many_arguments)]
    fn build_frame(
        &mut self,
        window: &Window,
        rhi: &Rhi,
        genome: &mut Genome,
        env: &mut Env,
        mode: &mut RenderMode,
        seed: &mut u32,
        tab: &mut Tab,
        wind_on: &mut bool,
        wind_strength: &mut f32,
        wind_dir: &mut Vec3,
        bloom_on: &mut bool,
        bloom_strength: &mut f32,
        stats: &Stats,
    ) -> PanelOut {
        let raw_input = self.winit.take_egui_input(window);
        let mut out = PanelOut::default();

        let full_output = self.ctx.run(raw_input, |ctx| {
            // ── (A) LEFT controls SidePanel: header (presets + generate/reroll/
            //    seed), a tab strip, then ONLY the active tab's sliders. ─────────
            // Widened modestly (300→320) so the gene/env slider labels (e.g.
            // "appendageDensity", "temperature") sit fully inside the panel.
            const LEFT_PANEL_W: f32 = 320.0;
            egui::SidePanel::left("controls")
                .resizable(true)
                .default_width(LEFT_PANEL_W)
                .show(ctx, |ui| {
                    ui.heading("flora · CAS");

                    // Header: Preset dropdown + Generate / Reroll.
                    ui.horizontal_wrapped(|ui| {
                        egui::ComboBox::from_id_salt("preset_combo")
                            .selected_text("Preset…")
                            .show_ui(ui, |ui| {
                                for (name, g) in presets::ALL {
                                    if ui.selectable_label(false, *name).clicked() {
                                        *genome = **g;
                                        out.rebuild = true;
                                    }
                                }
                            });
                        if ui.button("Generate").clicked() {
                            out.generate = true;
                        }
                        if ui.button("Reroll").clicked() {
                            *seed = seed.wrapping_add(1);
                            out.generate = true;
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Seed:");
                        if ui.add(egui::DragValue::new(seed)).changed() {
                            out.generate = true;
                        }
                        ui.label("Struct:");
                        if ui
                            .add(egui::DragValue::new(&mut genome.structural_seed))
                            .changed()
                        {
                            out.rebuild = true;
                        }
                    });

                    ui.separator();

                    // Tab strip — direct analog of dryad's `.tab-btn` row.
                    ui.horizontal_wrapped(|ui| {
                        for (t, label) in TABS {
                            ui.selectable_value(tab, t, label);
                        }
                    });

                    ui.separator();

                    egui::ScrollArea::vertical().show(ui, |ui| {
                        match *tab {
                            // Climate = the env sliders, in dryad's order: Gravity,
                            // Medium, Light, Sun angle, Wind, Aridity, Temperature.
                            // Env only reaches the tree through resolve(genome, env),
                            // so any edit needs a rebuild.
                            Tab::Climate => {
                                let mut c = false;
                                c |= ui.add(egui::Slider::new(&mut env.gravity, 0.1..=3.0).text("Gravity")).changed();
                                // Medium combo (dryad: air / water / dense_air). The
                                // Rust `Medium` enum is Air|Water only — "dense_air"
                                // has no variant, so it is omitted (noted in report).
                                egui::ComboBox::from_label("Medium")
                                    .selected_text(match env.medium {
                                        Medium::Air => "Air",
                                        Medium::Water => "Water",
                                    })
                                    .show_ui(ui, |ui| {
                                        c |= ui.selectable_value(&mut env.medium, Medium::Air, "Air").changed();
                                        c |= ui.selectable_value(&mut env.medium, Medium::Water, "Water").changed();
                                    });
                                c |= ui.add(egui::Slider::new(&mut env.light, 0.0..=1.0).text("Light")).changed();
                                c |= ui.add(egui::Slider::new(&mut env.sun_angle, 0.0..=1.0).text("Sun angle")).changed();
                                c |= ui.add(egui::Slider::new(&mut env.wind, 0.0..=1.0).text("Wind")).changed();
                                c |= ui.add(egui::Slider::new(&mut env.aridity, 0.0..=1.0).text("Aridity")).changed();
                                c |= ui.add(egui::Slider::new(&mut env.temperature, 0.0..=1.0).text("Temperature")).changed();
                                if c {
                                    out.rebuild = true;
                                }
                            }
                            // Gene tabs: render the CURATED, ordered slider list for
                            // this tab (label + dryad range), each bound to the live
                            // genome field looked up by JS name in fields_mut().
                            active => {
                                let mut fields = genome.fields_mut();
                                for slider in gene_tab_sliders(active) {
                                    // Find the live &mut f64 for this gene by name.
                                    let Some((_, _, _, val)) = fields
                                        .iter_mut()
                                        .find(|(n, _, _, _)| *n == slider.name)
                                    else {
                                        // Gene listed in the tab table but absent from
                                        // the Rust Genome — skip it (logged once).
                                        log::warn!(
                                            "flora CAS: gene '{}' not in Genome — skipped",
                                            slider.name
                                        );
                                        continue;
                                    };
                                    if ui
                                        .add(
                                            egui::Slider::new(*val, slider.min..=slider.max)
                                                .text(slider.label),
                                        )
                                        .changed()
                                    {
                                        out.rebuild = true;
                                    }
                                }
                            }
                        }
                    });
                });

            // ── (B) TOP-RIGHT Stats window (dryad field set). ─────────────────
            egui::Window::new("Stats")
                .anchor(egui::Align2::RIGHT_TOP, [-8.0, 8.0])
                .resizable(false)
                .collapsible(false)
                .show(ctx, |ui| {
                    // FPS color-coded per dryad (≥50 good, ≥30 ok, else bad).
                    let fps_color = if stats.fps >= 50.0 {
                        egui::Color32::from_rgb(0x4c, 0xd9, 0x64)
                    } else if stats.fps >= 30.0 {
                        egui::Color32::from_rgb(0xe6, 0xc4, 0x4c)
                    } else {
                        egui::Color32::from_rgb(0xe0, 0x5c, 0x5c)
                    };
                    ui.horizontal(|ui| {
                        ui.label("FPS:");
                        ui.colored_label(fps_color, format!("{:.0}  ({:.2} ms)", stats.fps, stats.frame_ms));
                    });
                    ui.label(format!("Triangles: {}", stats.triangles));
                    ui.label(format!("Draw calls: {}", stats.draw_calls));
                    ui.label(format!("Leaf clusters: {}", stats.leaves));
                    ui.label(format!("Bones: {}", stats.bones));
                    ui.label(format!("Resolution: {}x{}", stats.resolution.0, stats.resolution.1));
                });

            // ── (C) View window: render-mode buttons + Wind + Bloom. Placed just
            //    to the RIGHT of the left controls SidePanel (dryad puts the
            //    render-mode panel beside the controls, NOT over them). Uses
            //    `default_pos` (a one-time initial position) rather than `anchor`
            //    so it clears the SidePanel yet stays draggable. x = panel width +
            //    a small margin; y = top. ──
            egui::Window::new("View")
                .default_pos([LEFT_PANEL_W + 8.0, 8.0])
                .resizable(false)
                .collapsible(false)
                .show(ctx, |ui| {
                    ui.horizontal_wrapped(|ui| {
                        for (m, label) in RENDER_MODES {
                            if ui.selectable_value(mode, m, label).changed() {
                                out.mode_changed = true;
                            }
                        }
                    });
                    ui.separator();
                    // Wind (animation only — NOT a rebuild; read live in render()).
                    // `env.wind` (Climate tab) is the MORPHOLOGY gene; this is the
                    // procedural global-field sway.
                    ui.checkbox(wind_on, "Wind");
                    ui.add(egui::Slider::new(wind_strength, 0.0..=1.5).text("strength"));
                    ui.add(egui::Slider::new(&mut wind_dir.x, -1.0..=1.0).text("dir x"));
                    ui.add(egui::Slider::new(&mut wind_dir.z, -1.0..=1.0).text("dir z"));
                    ui.separator();
                    // Bloom (UnrealBloom POST — animation only, NOT a rebuild).
                    ui.checkbox(bloom_on, "Bloom");
                    ui.add_enabled(
                        *bloom_on,
                        egui::Slider::new(bloom_strength, 0.0..=1.0).text("strength"),
                    );
                });
        });

        if !full_output.textures_delta.set.is_empty() {
            let queue = rhi.queue_handle();
            let pool = rhi.command_pool(0);
            if let Err(e) =
                self.renderer
                    .set_textures(queue, pool, full_output.textures_delta.set.as_slice())
            {
                log::error!("egui set_textures failed: {e}");
            }
        }
        self.winit
            .handle_platform_output(window, full_output.platform_output.clone());
        self.output = Some(full_output);
        out
    }

    fn render(&mut self, rhi: &Rhi, fi: u32) {
        let output = match self.output.take() {
            Some(o) => o,
            None => return,
        };
        let ppp = output.pixels_per_point;
        let primitives = self.ctx.tessellate(output.shapes, ppp);
        if !primitives.is_empty() {
            let cmd = rhi.current_command_buffer(fi);
            let extent = rhi.extent();
            if let Err(e) = self.renderer.cmd_draw(cmd, extent, ppp, &primitives) {
                log::error!("egui cmd_draw failed: {e}");
            }
        }
        if !output.textures_delta.free.is_empty() {
            if let Err(e) = self.renderer.free_textures(&output.textures_delta.free) {
                log::error!("egui free_textures failed: {e}");
            }
        }
    }
}

/// Per-frame stats snapshot for the panel.
#[derive(Default, Clone, Copy)]
struct Stats {
    fps: f32,
    frame_ms: f32,
    triangles: u32,
    /// dryad "Draw calls": branch + leaf passes (+ sky + ground from Staging).
    draw_calls: u32,
    /// dryad "Leaf clusters" == enki foliage card count.
    leaves: u32,
    bones: u32,
    /// Swapchain resolution (width, height).
    resolution: (u32, u32),
}

/// Auto-fit framing derived from the branch-mesh AABB.
#[derive(Clone, Copy)]
struct Fit {
    center: Vec3,
    distance: f32,
}

impl Fit {
    fn from_bounds(min: [f32; 3], max: [f32; 3], fov_y: f32) -> Self {
        let mn = Vec3::from(min);
        let mx = Vec3::from(max);
        let center = (mn + mx) * 0.5;
        let radius = (mx - center).length().max(0.5);
        let distance = (radius / (fov_y * 0.5).sin()) * 1.4;
        Fit { center, distance }
    }
}

/// Build the sun's light-space view-projection (world/tree-local → light clip),
/// fitting an orthographic frustum to the tree+ground bounds along `sun_dir`.
///
/// 1:1 with dryad's `updateShadowCamera` (viewer.js:286-302), adapted to enki's
/// reversed-Z:
///   - square ortho `halfExtent = max(halfWidthX, halfDepthZ) + 5` (margin +5),
///   - light view = lookAt(eye = sun_dir·50, target = bounds center, up = +Y),
///   - depth range mapped reversed-Z (near→1, far→0) so the GREATER compare and
///     the depth-0 clear match the rest of the engine.
///
/// `sun_dir` points TOWARD the sun (FrameUniforms.sun0_dir convention). The tree
/// sits at the world origin in the viewer, so the ground (also at origin) shares
/// the frame; the bounds are clamped so the above-ground tree fits tightly.
fn light_view_proj(bounds_min: Vec3, bounds_max: Vec3, sun_dir: Vec3) -> Mat4 {
    let center = (bounds_min + bounds_max) * 0.5;
    let half_x = (bounds_max.x - bounds_min.x) * 0.5;
    let half_z = (bounds_max.z - bounds_min.z) * 0.5;
    // dryad margin +5 world units; clamp small trees to a sane minimum.
    let half_extent = half_x.max(half_z).max(0.5) + 5.0;

    // Light view: eye on the sun side, looking at the tree center, +Y up. Guard
    // the degenerate up‖dir case (sun straight overhead) by tilting up to +Z.
    let dir = sun_dir.normalize_or(Vec3::Y);
    let eye = center + dir * 50.0;
    let up = if dir.abs_diff_eq(Vec3::Y, 1e-3) || dir.abs_diff_eq(Vec3::NEG_Y, 1e-3) {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let view = Mat4::look_at_rh(eye, center, up);

    // dryad near = -50, far = bounds.max.y + 50 (along the light axis from the
    // eye). With the eye 50 units out, the tree spans roughly [0, 100] in view-z;
    // use a generous symmetric range so nothing clips. view-space z is negative
    // forward (RH), so depths run near..far as -near_d..-far_d.
    let near_d = 1.0_f32;
    let far_d = 100.0 + (bounds_max.y - bounds_min.y) + 10.0;

    // Reversed-Z orthographic (Vulkan, Y-down clip, depth [0,1], near→1 far→0).
    let ortho = ortho_reversed_z(
        -half_extent,
        half_extent,
        -half_extent,
        half_extent,
        near_d,
        far_d,
    );
    ortho * view
}

/// Reversed-Z orthographic projection (RH, Vulkan clip: Y-down, depth [0,1],
/// near→1.0, far→0.0). Mirrors `reversed_z_perspective`'s convention for ortho.
fn ortho_reversed_z(l: f32, r: f32, b: f32, t: f32, near: f32, far: f32) -> Mat4 {
    // x,y: standard ortho scale/offset. Y flipped for Vulkan.
    let sx = 2.0 / (r - l);
    let sy = -2.0 / (t - b); // flip Y
    let tx = -(r + l) / (r - l);
    let ty = (t + b) / (t - b); // sign follows the flipped Y
    // z: map view-z (RH, forward = -z) [-near, -far] → NDC [1, 0].
    //   NDC.z = a*z_e + c, with z_e=-near → 1, z_e=-far → 0.
    //   a*(-near)+c = 1 ; a*(-far)+c = 0  →  a = 1/(far-near), c = far/(far-near).
    let a = 1.0 / (far - near);
    let c = far / (far - near);
    Mat4::from_cols(
        glam::Vec4::new(sx, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, sy, 0.0, 0.0),
        glam::Vec4::new(0.0, 0.0, a, 0.0),
        glam::Vec4::new(tx, ty, c, 1.0),
    )
}

struct App {
    window: Option<Window>,
    rhi: Option<Rhi>,
    /// The flora-OWNED Vulkan sub-renderer (pipelines + descriptor sets + UBO
    /// ring). Built once when the Rhi comes up; FloraView + Staging record
    /// through it. Mirrors how `egui` is held alongside the Rhi.
    flora_renderer: Option<FloraRenderer>,
    flora: Option<FloraView>,
    /// The Nanite branch path: the current tree's branch mesh baked into a
    /// ClusterAsset + the cull/draw renderer. Rebuilt with every tree (build_tree).
    /// Drives ONLY the Triangle/Cluster/LOD branch-meshlet debug views (Lit and the
    /// other flora views use flora's own full render path, not Nanite).
    #[cfg(all(feature = "flora", feature = "nanite"))]
    flora_nanite: Option<FloraNanite>,
    staging: Option<Staging>,
    egui: Option<FloraGui>,
    fit: Fit,
    seed: u32,
    /// The single source of truth for what's on screen — reseed and edits both
    /// flow through this genome, so the panel always reflects the live tree.
    genome: Genome,
    env: Env,
    mode: RenderMode,
    /// Active left-panel CAS tab (Climate / Trunk / Branches / Leaves / Root).
    tab: Tab,
    start: Instant,
    last_frame: Instant,
    minimized: bool,
    /// Triangle/leaf/bone/draw-call counts from the last rebuild (cheap to cache).
    last_stats: (u32, u32, u32, u32),
    /// Wind animation state (NOT a genome gene — `env.wind` drives morphology via
    /// resolve()). `wind_on` toggles sway; `wind_strength`/`wind_dir` shape it.
    /// `wind_clock` is an elapsed-seconds accumulator advanced by frame dt (no
    /// wall-clock fn), so a paused/zero-strength tree is exactly static.
    wind_on: bool,
    wind_strength: f32,
    wind_dir: Vec3, // xz used as the horizontal wind direction
    wind_clock: f32,
    /// UnrealBloom POST controls (dryad: strength 0.15). `bloom_on` gates the
    /// effect; the effective strength fed to `run_bloom` is 0 when off, so OFF or
    /// strength 0 is byte-identical to the offscreen-stage parity image.
    bloom_on: bool,
    bloom_strength: f32,
    /// ponytail: HEADLESS SCREENSHOT mode. `Some(path)` when `FLORA_SCREENSHOT` is
    /// set → hide the egui panel, render a few warm-up frames, then capture the
    /// final composited frame into a PNG at `path` and exit 0. `frame_count` ticks
    /// each rendered frame so we know when the scene has settled (camera at initial
    /// framing, pipelines/swapchain warm).
    screenshot: Option<String>,
    /// ponytail: when `FLORA_SCREENSHOT_UI=1` is set ALONGSIDE `FLORA_SCREENSHOT`,
    /// RENDER the egui panels into the captured frame (so the GUI layout shows up
    /// in the PNG). Normal `FLORA_SCREENSHOT` without `_UI` still hides the panel
    /// for a clean render.
    screenshot_ui: bool,
    frame_count: u32,
    /// Set once the screenshot PNG has been written → `about_to_wait` shuts down
    /// and exits the event loop (exit 0).
    screenshot_done: bool,
}

/// ponytail: warm-up frames before the capture (per the brief: ~45) — enough for
/// the swapchain/pipelines to be warm and the auto-orbit camera at its initial
/// framing (t≈0.7s of orbit, still essentially front-on).
const SCREENSHOT_WARMUP_FRAMES: u32 = 45;

impl App {
    fn new() -> Self {
        let env = Env::default();
        // ponytail: FLORA_SEED (when set) picks a varied specimen via the SAME
        // random_genome(env, seed) path the panel/Space use, so a captured seed
        // matches the interactive view. With NO FLORA_SEED we instead start from
        // dryad's curated TREE_DEFAULT (brown furrowed bark + full green canopy)
        // rather than random_genome(0) — whose raw `pigment` draw rolls a RANDOM
        // hue (often blue/cyan) and density, giving a sparse oddly-colored default.
        let seed_env = std::env::var("FLORA_SEED")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok());
        let seed = seed_env.unwrap_or(0);
        let genome = match seed_env {
            Some(s) => random_genome(&env, s),
            None => presets::TREE_DEFAULT,
        };
        // Screenshot mode: FLORA_SCREENSHOT=<output path> → headless capture + exit.
        let screenshot = std::env::var("FLORA_SCREENSHOT")
            .ok()
            .filter(|p| !p.trim().is_empty());
        let screenshot_ui = std::env::var("FLORA_SCREENSHOT_UI")
            .ok()
            .map(|v| v.trim() == "1")
            .unwrap_or(false);
        // ponytail: FLORA_VIEW (or its legacy alias FLORA_MODE) picks the INITIAL
        // render mode by name (case-insensitive) so a headless capture can target
        // any view, incl. the Nanite Triangle/Cluster/LOD debug views, without
        // driving the egui panel. Defaults to Lit.
        let mode_env = std::env::var("FLORA_VIEW")
            .or_else(|_| std::env::var("FLORA_MODE"))
            .ok();
        let mode = match mode_env.as_deref().map(str::trim) {
            Some(s) if s.eq_ignore_ascii_case("unlit") => RenderMode::Unlit,
            Some(s) if s.eq_ignore_ascii_case("wireframe") => RenderMode::Wireframe,
            Some(s) if s.eq_ignore_ascii_case("normals") => RenderMode::Normals,
            Some(s) if s.eq_ignore_ascii_case("ao") => RenderMode::Ao,
            #[cfg(all(feature = "flora", feature = "nanite"))]
            Some(s) if s.eq_ignore_ascii_case("triangle") => RenderMode::Triangle,
            #[cfg(all(feature = "flora", feature = "nanite"))]
            Some(s) if s.eq_ignore_ascii_case("cluster") => RenderMode::Cluster,
            #[cfg(all(feature = "flora", feature = "nanite"))]
            Some(s) if s.eq_ignore_ascii_case("lod") => RenderMode::Lod,
            _ => RenderMode::Lit,
        };
        App {
            window: None,
            rhi: None,
            flora_renderer: None,
            flora: None,
            #[cfg(all(feature = "flora", feature = "nanite"))]
            flora_nanite: None,
            staging: None,
            egui: None,
            fit: Fit { center: Vec3::ZERO, distance: 30.0 },
            seed,
            genome,
            env,
            mode,
            tab: Tab::Climate,
            start: Instant::now(),
            last_frame: Instant::now(),
            minimized: false,
            last_stats: (0, 0, 0, 0),
            // Wind ON by default at a gentle strength (strength 0 = exact static).
            wind_on: true,
            wind_strength: 0.6,
            wind_dir: Vec3::new(1.0, 0.0, 0.0),
            wind_clock: 0.0,
            // Bloom ON at dryad's default strength (0 / OFF = exact parity).
            bloom_on: true,
            bloom_strength: 0.15,
            screenshot,
            screenshot_ui,
            frame_count: 0,
            screenshot_done: false,
        }
    }

    /// (Re)build the tree at the world origin from the CURRENT genome + env and
    /// refresh the auto-fit framing from its branch-mesh AABB.
    fn build_tree(&mut self) {
        let rhi = match self.rhi.as_mut() {
            Some(r) => r,
            None => return,
        };
        // ponytail: rebuilding leaks the prior FloraView's GPU buffers until
        // shutdown (FloraView has no Drop) — acceptable for a debug tool. The
        // flora-owned pipelines are NOT rebuilt here (they live in
        // FloraRenderer, built once); only the per-tree meshes are re-uploaded.
        match FloraView::from_genome_with_mode(
            rhi,
            &self.genome,
            &self.env,
            DVec3::ZERO,
            self.leaf_mode,
        ) {
            Ok(mut f) => {
                // A fresh FloraView defaults to Lit; carry the live mode forward
                // so a rebuild (slider/preset/generate) doesn't reset the view.
                f.set_render_mode(self.mode);
                // ── (Re)bake + upload this tree's leaf textures (color + normal)
                //    into the FloraRenderer's set1 leaf bindings. Leaf genes vary
                //    per genome, so this runs on every build/reseed (the GPU is
                //    already idle — `rebuild` waits before build). The Single leaf
                //    mode bakes the one-leaf sprite; Cluster bakes the 5-8 fan. ──
                if let Some(renderer) = self.flora_renderer.as_mut() {
                    let single = f.leaf_mode() == LeafMode::Single;
                    if let Err(e) = renderer.update_leaf_texture(rhi, &f.leaf_genes(), single) {
                        log::error!("FloraRenderer::update_leaf_texture failed: {e}");
                    }
                }
                let fov_y = 60_f32.to_radians();
                let (min, max) = f.local_bounds();
                self.fit = Fit::from_bounds(min, max, fov_y);
                // Draw calls: the tree's branch+leaf passes plus the 2 Staging
                // draws (sky + ground) the viewer always records.
                self.last_stats = (
                    f.triangle_count(),
                    f.leaf_count(),
                    f.bone_count(),
                    f.draw_call_count() + 2,
                );

                let foot = ((max[0] - min[0]).abs()).max((max[2] - min[2]).abs());
                let shadow_radius = (foot * 0.6).max(2.0);
                match Staging::new(rhi, shadow_radius) {
                    Ok(s) => self.staging = Some(s),
                    Err(e) => log::error!("Staging::new failed: {e}"),
                }

                // ── Nanite branch bake: feed THIS tree's branch mesh through
                //    enki-nanite's cluster_build + dag, then build the cull/draw
                //    renderer against the flora SCENE pass color format (so its draw
                //    records inside begin_scene_pass). Rebuilt every tree (the mesh
                //    changes per genome). ponytail: all clusters permanently resident
                //    — one static tree, no streamer. The old renderer's GPU buffers
                //    leak until shutdown (like FloraView), acceptable for a debug
                //    tool; we wait_idle in rebuild() before replacing it.
                #[cfg(all(feature = "flora", feature = "nanite"))]
                {
                    let resolved = enki_flora::genome::resolve(&self.genome, &self.env);
                    let bm = enki_flora::mesh::build_branch_mesh(&resolved.graph);
                    let scene_fmt = self
                        .flora_renderer
                        .as_ref()
                        .map(|r| r.scene_color_format());
                    if let Some(scene_fmt) = scene_fmt {
                        match FloraNanite::new(rhi, &bm, DVec3::ZERO, scene_fmt) {
                            Ok(n) => {
                                log::info!(
                                    "FloraNanite: {} clusters resident (branch mesh → nanite)",
                                    n.cluster_count()
                                );
                                self.flora_nanite = Some(n);
                            }
                            Err(e) => log::error!("FloraNanite::new failed: {e}"),
                        }
                    }
                }

                self.flora = Some(f);
                log::info!(
                    "Tree seed {} — fit center {:?} dist {:.1}m",
                    self.seed, self.fit.center, self.fit.distance
                );
            }
            Err(e) => log::error!("FloraView::from_genome failed: {e}"),
        }
    }

    /// Wait for the GPU to idle (safe: no in-flight use of the old tree), then
    /// rebuild from the current genome and request a redraw. The single rebuild
    /// path for slider edits, presets, generate, and Space/N reseed.
    fn rebuild(&mut self) {
        if let Some(rhi) = self.rhi.as_ref() {
            let _ = rhi.wait_idle();
        }
        self.build_tree();
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }

    /// Idle the GPU, then free the flora-owned sub-renderer's Vulkan objects
    /// (deletable-flora contract). Must run before the Rhi is dropped. Idempotent:
    /// `Option::take` ensures a double-call (CloseRequested then drop) is a no-op.
    fn shutdown(&mut self) {
        if let Some(rhi) = self.rhi.as_ref() {
            let _ = rhi.wait_idle();
        }
        if let Some(mut renderer) = self.flora_renderer.take() {
            renderer.destroy();
        }
        // ponytail: drop the egui renderer (and its Vulkan objects) HERE, while the
        // device is still alive — `App`'s field order drops `rhi` (→ device) before
        // `egui`, so relying on `App`'s Drop would tear egui down against a dead
        // device (VUID-vkDestroyDevice-05137 + an access violation). Dropping it now
        // (idempotent: Option::take) sidesteps that on both Esc and the screenshot
        // exit. egui_ash_renderer frees its pipeline/pool/layout in its own Drop.
        self.egui = None;
    }

    fn camera(&self) -> Camera {
        let t = self.start.elapsed().as_secs_f32();
        let yaw = t * 0.35;
        let elev = 18_f32.to_radians();
        let (sy, cy) = yaw.sin_cos();
        let dir = Vec3::new(cy * elev.cos(), elev.sin(), sy * elev.cos());
        let eye = self.fit.center + dir * self.fit.distance;
        let forward = (self.fit.center - eye).normalize_or(Vec3::NEG_Z);
        let orientation =
            Quat::from_mat4(&glam::Mat4::look_to_rh(Vec3::ZERO, forward, Vec3::Y)).inverse();
        Camera {
            position: eye.as_dvec3(),
            orientation,
            fov_y_radians: 60_f32.to_radians(),
            near: 0.1,
            far: 10_000.0,
        }
    }

    fn render(&mut self) {
        if self.minimized {
            return;
        }

        // Frame timing for the FPS stat.
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;

        // Advance the wind clock from frame dt (accumulator, NOT wall-clock). Only
        // ticks while wind is animating, so toggling off freezes the phase too.
        if self.wind_on {
            self.wind_clock += dt;
        }
        let resolution = self
            .rhi
            .as_ref()
            .map(|r| {
                let e = r.extent();
                (e.width, e.height)
            })
            .unwrap_or((0, 0));
        let stats = Stats {
            fps: if dt > 0.0 { 1.0 / dt } else { 0.0 },
            frame_ms: dt * 1000.0,
            triangles: self.last_stats.0,
            draw_calls: self.last_stats.3,
            leaves: self.last_stats.1,
            bones: self.last_stats.2,
            resolution,
        };

        // ── Build the egui panel BEFORE begin_frame (texture upload one-shots
        //    a submit; must be outside the rendering instance). Split-borrow
        //    egui/window/rhi/genome/env — all disjoint App fields. ──
        // ponytail: in screenshot mode SKIP the panel so the captured PNG is a
        // clean render (no UI chrome) — UNLESS FLORA_SCREENSHOT_UI=1, which builds
        // the panels so the GUI layout shows up in the PNG for inspection.
        let build_panel = self.screenshot.is_none() || self.screenshot_ui;
        let mut panel = PanelOut::default();
        if build_panel {
        if let (Some(egui), Some(window), Some(rhi)) =
            (self.egui.as_mut(), self.window.as_ref(), self.rhi.as_ref())
        {
            panel = egui.build_frame(
                window,
                rhi,
                &mut self.genome,
                &mut self.env,
                &mut self.mode,
                &mut self.seed,
                &mut self.tab,
                &mut self.wind_on,
                &mut self.wind_strength,
                &mut self.wind_dir,
                &mut self.bloom_on,
                &mut self.bloom_strength,
                &stats,
            );
        }
        } // end panel-build gate
        // Apply panel actions (each leads to a rebuild). 'generate' rerolls the
        // genome from the current env; 'rebuild' uses the edited genome as-is.
        // `rebuild()` re-applies `self.mode` (a fresh FloraView defaults to Lit).
        if panel.generate {
            self.genome = random_genome(&self.env, self.seed);
            self.rebuild();
        } else if panel.rebuild {
            self.rebuild();
        } else if panel.mode_changed {
            // Mode-only change: no tree rebuild, just flip the pipeline/push lane.
            if let Some(flora) = self.flora.as_mut() {
                flora.set_render_mode(self.mode);
            }
        }

        let camera = self.camera();
        // Flora viewer uses dryad's EXACT lights (sun #fff4e0 ×3.0, hemi
        // #88aacc/#443322 ×0.3, no fill, no white ambient). The shared planet
        // app keeps Lights::demiurge_default — this is a flora-local choice.
        let lights = Lights::dryad_default();
        let aspect = self
            .rhi
            .as_ref()
            .map(|r| {
                let e = r.extent();
                if e.height > 0 { e.width as f32 / e.height as f32 } else { 16.0 / 9.0 }
            })
            .unwrap_or(16.0 / 9.0);
        let fu = FrameUniforms::new(&camera, aspect, &lights);
        let camera_world_pos = camera.position;

        // ── Wind vec4 (shared by the shadow caster + the lit pass so the cast
        //    silhouette sways in lockstep). strength==0 → exact static tree. ──
        let wind_strength = if self.wind_on { self.wind_strength } else { 0.0 };
        let wind_d = Vec3::new(self.wind_dir.x, 0.0, self.wind_dir.z).normalize_or(Vec3::X);
        let wind = [self.wind_clock, wind_strength, wind_d.x, wind_d.z];

        // ── Sun shadow light matrix + uniforms. The sun is FrameUniforms.sun0
        //    (dryad's shadow-casting DirectionalLight); fit the ortho frustum to
        //    the tree's WORLD bounds (the ground is the receiver, not a caster). ──
        let sun_dir = Vec3::from_slice(&fu.sun0_dir[..3]);
        let (shadow_uniforms, light_vp) = match self.flora.as_ref() {
            Some(flora) => {
                let (bmin, bmax) = flora.world_bounds();
                // Include the ground contact (y=0) so the trunk base casts onto it.
                let bmin = bmin.min(Vec3::ZERO);
                let lvp = light_view_proj(bmin, bmax, sun_dir);
                let texel = 1.0
                    / self
                        .flora_renderer
                        .as_ref()
                        .map(|r| r.shadow_map_size())
                        .unwrap_or(2048) as f32;
                let su = ShadowUniforms {
                    light_view_proj: lvp.to_cols_array_2d(),
                    // (1/size, normalBias 0.02 [dryad], enabled, 0)
                    params: [texel, 0.02, 1.0, 0.0],
                };
                (su, Some(lvp))
            }
            None => (
                ShadowUniforms {
                    light_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                    params: [1.0 / 2048.0, 0.02, 0.0, 0.0], // disabled
                },
                None,
            ),
        };

        let rhi = match self.rhi.as_mut() {
            Some(r) => r,
            None => return,
        };

        let fi = match rhi.begin_frame() {
            Ok(idx) => idx,
            Err(RhiError::Vulkan(_)) => {
                if let Some(w) = &self.window { w.request_redraw(); }
                return;
            }
            Err(e) => {
                log::error!("begin_frame error: {e}");
                return;
            }
        };
        if fi == u32::MAX {
            let _ = rhi.end_frame(fi);
            return;
        }

        // ── Resize the flora-owned POST targets to match the (possibly recreated)
        //    swapchain extent. enki recreates its swapchain internally; flora's HDR
        //    + bloom targets are swapchain-sized so they must follow. wait_idle is
        //    cheap here (only fires on an actual extent change). ──
        let cur_extent = rhi.extent();
        if let Some(renderer) = self.flora_renderer.as_mut() {
            if renderer.post_extent() != cur_extent {
                let _ = rhi.wait_idle();
                if let Err(e) = renderer.resize_post(cur_extent) {
                    log::error!("flora resize_post error: {e}");
                }
            }
        }

        // Upload this frame's FrameUniforms + ShadowUniforms into the flora-owned
        // ring slots (host-visible writes; cheap memcpy, no command recording).
        // Both staging + flora bind set0 (frame) + set1 (shadow) from these slots.
        if let Some(renderer) = self.flora_renderer.as_ref() {
            if let Err(e) = renderer.set_frame_uniforms(rhi, fi, &fu) {
                log::error!("flora set_frame_uniforms error: {e}");
            }
            if let Err(e) = renderer.set_shadow_uniforms(rhi, fi, &shadow_uniforms) {
                log::error!("flora set_shadow_uniforms error: {e}");
            }
        }

        // ── HIERARCHICAL WIND: solve this frame's per-bone matrices (CPU) and
        //    upload them into the flora-owned set1/binding 6 storage buffer for
        //    this frame-in-flight. Same `wind` vec4 the lit + shadow passes use,
        //    so the bone matrices (and thus the swaying silhouette) are consistent
        //    across the depth caster and the main pass. strength==0 → identities
        //    → exact static rest pose. ──
        if let (Some(renderer), Some(flora)) =
            (self.flora_renderer.as_ref(), self.flora.as_mut())
        {
            let mats = flora.solve_wind(wind);
            if let Err(e) = renderer.set_bone_matrices(rhi, fi, mats) {
                log::error!("flora set_bone_matrices error: {e}");
            }
        }

        // ── SUN SHADOW depth pass — recorded in the begin_frame→begin_rendering
        //    gap (outside any dynamic-rendering instance; Vulkan forbids nesting).
        //    Renders the tree's branch+leaf depth into the flora-owned shadow map,
        //    then barriers it to SHADER_READ for the main pass to sample. The
        //    ground is a receiver only (dryad), so it is NOT drawn here. ──
        if let (Some(renderer), Some(flora), Some(lvp)) =
            (self.flora_renderer.as_mut(), self.flora.as_ref(), light_vp)
        {
            renderer.begin_shadow_pass(rhi, fi);
            if let Err(e) = flora.record_shadow(rhi, renderer, fi, lvp, wind) {
                log::error!("FloraView::record_shadow error: {e}");
            }
            renderer.end_shadow_pass(rhi, fi);
        }

        // ── NANITE branch path: per-frame update + cull. The cull is a COMPUTE
        //    dispatch, so it MUST be recorded OUTSIDE any rendering instance — here,
        //    in the begin_frame→begin_rendering gap, after the (now-closed) shadow
        //    pass and BEFORE begin_scene_pass. The matching record_draw runs INSIDE
        //    the scene pass below. Only run it for the Nanite debug views
        //    (Triangle / Cluster / LOD); every other mode (Lit/Unlit/Wireframe/
        //    Normals/Ao) uses flora's own branch pipeline and skips the cull. ──
        #[cfg(all(feature = "flora", feature = "nanite"))]
        let nanite_branch = self.mode.uses_nanite() && self.flora_nanite.is_some();
        #[cfg(all(feature = "flora", feature = "nanite"))]
        if nanite_branch {
            let screen_h = cur_extent.height as f32;
            // tau: screen-space error cut in pixels. ~1px gives near-LOD0 detail at
            // the viewer's tight framing; the small static tree never starves.
            const NANITE_TAU_PX: f32 = 1.0;
            if let Some(n) = self.flora_nanite.as_mut() {
                if let Err(e) = n.update_and_cull(
                    rhi,
                    fi,
                    camera_world_pos,
                    &fu,
                    screen_h,
                    camera.fov_y_radians,
                    NANITE_TAU_PX,
                    self.mode.nanite_debug_mode(),
                    self.frame_count,
                ) {
                    log::error!("FloraNanite update_and_cull error: {e}");
                }
            }
        }

        // ── OFFSCREEN SCENE pass — sky/ground/tree into the flora-owned LINEAR-HDR
        //    MSAA target (RGBA16F), recorded in the begin_frame→begin_rendering gap
        //    (like the shadow pass). Resolves to a single-sample HDR texture, then
        //    runs the UnrealBloom chain. Both are flora-owned vkCmdBeginRendering on
        //    flora-owned targets, OUTSIDE enki's swapchain instance. ──
        if let Some(renderer) = self.flora_renderer.as_mut() {
            renderer.begin_scene_pass(rhi, fi);
        }
        if let (Some(renderer), Some(staging)) =
            (self.flora_renderer.as_ref(), self.staging.as_mut())
        {
            if let Err(e) = staging.record(rhi, renderer, fi, camera_world_pos) {
                log::error!("Staging::record error: {e}");
            }
        }
        // ── Tree: branches + leaves into the scene pass. The Nanite debug views
        //    (Triangle / Cluster / LOD) draw ONLY the branch meshlets through the
        //    Nanite vertex-pulling indirect draw (its cull already ran in the gap
        //    above), in the chosen per-tri/cluster/LOD color mode, and DELIBERATELY
        //    SKIP the leaf cards — a geometry/cluster debug view is about the branch
        //    meshlets, not foliage (and Nanite's generic draw miscolors leaf cards).
        //    Every other view (Lit/Unlit/Wireframe/Normals/Ao) runs flora's FULL
        //    branch+leaf path: textured PBR bark + IBL + PCF shadows + green leaf
        //    cards — the photoreal render, never routed through Nanite. ──
        #[cfg(all(feature = "flora", feature = "nanite"))]
        if nanite_branch {
            // Branches ONLY, via Nanite (indirect, vertex-pulling) — inside the
            // scene pass. No leaf draw: these are branch-meshlet debug views.
            if let Some(n) = self.flora_nanite.as_ref() {
                if let Err(e) = n.record_draw(rhi, fi) {
                    log::error!("FloraNanite::record_draw error: {e}");
                }
            }
        } else if let (Some(renderer), Some(flora)) =
            (self.flora_renderer.as_ref(), self.flora.as_mut())
        {
            // Same `wind` vec4 the shadow caster used → cast silhouette matches.
            if let Err(e) = flora.record(rhi, renderer, fi, camera_world_pos, wind) {
                log::error!("FloraView::record error: {e}");
            }
        }
        #[cfg(not(all(feature = "flora", feature = "nanite")))]
        if let (Some(renderer), Some(flora)) =
            (self.flora_renderer.as_ref(), self.flora.as_mut())
        {
            // Same `wind` vec4 the shadow caster used → cast silhouette matches.
            if let Err(e) = flora.record(rhi, renderer, fi, camera_world_pos, wind) {
                log::error!("FloraView::record error: {e}");
            }
        }
        // Effective bloom strength: 0 when toggled OFF (composite zeros out →
        // output == scene, the offscreen-stage parity image).
        let bloom_strength = if self.bloom_on { self.bloom_strength } else { 0.0 };
        if let Some(renderer) = self.flora_renderer.as_ref() {
            renderer.end_scene_pass(rhi, fi);
            renderer.run_bloom(rhi, fi, bloom_strength);
        }

        // ── SCREENSHOT capture: on the warm-up'd capture frame, render the SAME
        //    fs_output composite into a flora-owned TRANSFER_SRC image + copy to a
        //    host-visible buffer. Recorded in the begin_frame→begin_rendering gap
        //    (its own rendering instance), BEFORE the normal swapchain composite —
        //    so the on-screen present is unaffected. ──
        self.frame_count = self.frame_count.wrapping_add(1);
        let do_capture = self.screenshot.is_some() && self.frame_count == SCREENSHOT_WARMUP_FRAMES;
        if do_capture {
            if self.screenshot_ui {
                // FLORA_SCREENSHOT_UI: draw the egui panels INTO the captured frame
                // so the GUI layout is inspectable in the PNG. Open the capture
                // pass, draw the composite, draw egui on top, then close + copy.
                // (Consumes egui's tessellated output, so the swapchain UI pass
                // below no-ops — fine: this is a one-shot headless capture+exit.)
                if let Some(renderer) = self.flora_renderer.as_mut() {
                    if let Err(e) = renderer.begin_capture(rhi, fi) {
                        log::error!("flora begin_capture error: {e}");
                    }
                }
                if let Some(egui) = self.egui.as_mut() {
                    egui.render(rhi, fi);
                }
                if let Some(renderer) = self.flora_renderer.as_mut() {
                    if let Err(e) = renderer.end_capture(rhi, fi) {
                        log::error!("flora end_capture error: {e}");
                    }
                }
            } else if let Some(renderer) = self.flora_renderer.as_mut() {
                if let Err(e) = renderer.record_capture(rhi, fi) {
                    log::error!("flora record_capture error: {e}");
                }
            }
        }

        // ── enki's swapchain MSAA instance: the COMPOSITE/OUTPUT fullscreen pass
        //    (scene_HDR + bloom → ACES+sRGB ONCE → swapchain), then egui on top. ──
        rhi.begin_rendering(fi);
        rhi.set_viewport_scissor_full(fi);
        if let Some(renderer) = self.flora_renderer.as_ref() {
            renderer.record_composite(rhi, fi);
        }

        // Close the MSAA 3D instance, open the 1-sample UI pass, record egui.
        // ponytail: skip egui in clean screenshot mode (no UI chrome in the PNG);
        // the panel was never built this frame, so render() would no-op anyway.
        // In FLORA_SCREENSHOT_UI mode the panel WAS built but its output was
        // already consumed by the capture pass above, so this render() no-ops too.
        rhi.begin_ui_pass(fi);
        if self.screenshot.is_none() {
            if let Some(egui) = self.egui.as_mut() {
                egui.render(rhi, fi);
            }
        }

        if let Err(e) = rhi.end_frame(fi) {
            log::error!("end_frame error: {e}");
        }

        // ── After the capture frame is submitted, idle the GPU, read the buffer
        //    back (BGRA→RGBA), write the PNG, and request shutdown. ──
        if do_capture {
            self.write_screenshot();
        }
    }

    /// Read the captured frame back and write it to the `FLORA_SCREENSHOT` PNG
    /// path, then flag the app to exit. Runs once, after the capture frame's
    /// submit. // ponytail: a wait_idle + a single image crate save — the simplest
    /// correct readback; no async, this is a one-shot headless path.
    fn write_screenshot(&mut self) {
        let path = match self.screenshot.clone() {
            Some(p) => p,
            None => return,
        };
        if let Some(rhi) = self.rhi.as_ref() {
            let _ = rhi.wait_idle();
        }
        let result = match (self.flora_renderer.as_ref(), self.rhi.as_ref()) {
            (Some(renderer), Some(rhi)) => renderer.read_capture_rgba(rhi),
            _ => Err(enki_rhi::RhiError::Other("no renderer/rhi for screenshot".into())),
        };
        match result {
            Ok((w, h, rgba)) => {
                match image::RgbaImage::from_raw(w, h, rgba) {
                    Some(img) => match img.save(&path) {
                        Ok(()) => log::info!("FLORA_SCREENSHOT wrote {w}x{h} → {path}"),
                        Err(e) => log::error!("FLORA_SCREENSHOT save {path} failed: {e}"),
                    },
                    None => log::error!("FLORA_SCREENSHOT: RGBA buffer too small for {w}x{h}"),
                }
            }
            Err(e) => log::error!("FLORA_SCREENSHOT read_capture failed: {e}"),
        }
        // Signal the event loop to exit cleanly (handled in about_to_wait).
        self.screenshot_done = true;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = WindowAttributes::default()
            .with_title("enki — flora viewer")
            .with_inner_size(LogicalSize::new(1280u32, 720u32))
            .with_resizable(true);
        let window = match event_loop.create_window(attrs) {
            Ok(w) => w,
            Err(e) => {
                log::error!("create_window failed: {e}");
                event_loop.exit();
                return;
            }
        };

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

        // Build the flora-owned sub-renderer (pipelines + descriptor sets + UBO
        // ring) once, from the Rhi's raw handles. Mirrors FloraGui::new.
        match FloraRenderer::new(&mut rhi) {
            Ok(r) => self.flora_renderer = Some(r),
            Err(e) => {
                log::error!("FloraRenderer::new failed: {e}");
                event_loop.exit();
                return;
            }
        }

        // egui needs &Rhi + &Window; both exist now.
        self.egui = Some(FloraGui::new(&rhi, &window));

        self.window = Some(window);
        self.rhi = Some(rhi);
        self.build_tree();
        log::info!("flora viewer ready — Esc quit, Space/N next tree, panel drives the morphospace");
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        // Feed egui first; gate keyboard shortcuts on what it didn't consume.
        let egui_consumed = match (self.egui.as_mut(), self.window.as_ref()) {
            (Some(eg), Some(w)) => eg.on_window_event(w, &event),
            _ => false,
        };
        let wants_kb = self.egui.as_ref().map(|e| e.wants_keyboard()).unwrap_or(false);

        match event {
            WindowEvent::CloseRequested => {
                self.shutdown();
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                self.minimized = size.width == 0 || size.height == 0;
                if let Some(w) = &self.window { w.request_redraw(); }
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } if !egui_consumed && !wants_kb => match code {
                KeyCode::Escape => {
                    self.shutdown();
                    event_loop.exit();
                }
                // Space / N = next specimen: reroll genome from the current env,
                // keeping the genome the single source of truth (panel updates).
                KeyCode::Space | KeyCode::KeyN => {
                    self.seed = self.seed.wrapping_add(1);
                    self.genome = random_genome(&self.env, self.seed);
                    self.rebuild();
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => self.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Screenshot written → clean shutdown (wait_idle + free flora objects) and
        // exit the event loop (process exit 0).
        if self.screenshot_done {
            self.shutdown();
            event_loop.exit();
            return;
        }
        if let Some(w) = &self.window {
            w.request_redraw();
        }
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new().expect("event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new();
    event_loop.run_app(&mut app).expect("run_app");
}

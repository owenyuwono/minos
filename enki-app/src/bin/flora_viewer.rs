// flora_viewer — standalone enki-flora debug viewer with a dryad-CAS egui panel.
//
// ponytail: the smallest host that lets us LOOK at one tree, reseed, AND DRIVE
// the morphospace live. It owns its own winit window + Rhi + swapchain + frame
// loop, builds ONE FloraView at the world origin, and drives an interactive
// orbit/zoom/pan camera framed to the branch-mesh AABB. NO planet / nanite /
// TAA — just the 6-call frame loop from the bootstrap recon, plus an egui panel.
//
// The egui panel is a SCHEMA-DRIVEN copy of enki's EguiState plumbing
// (gui.rs new/on_window_event/render are copy-identical; only build_frame is
// flora-specific): gene sliders grouped by tier, climate sliders, presets,
// generate/reseed, and stats. Editing any gene/env LIVE-REBUILDS the tree via
// FloraView::from_genome_with_mode_tuned — the core dryad CAS loop.
//
// Run:  cargo run -p enki-app --bin flora_viewer --features flora
//
// Keys (all // ponytail: debug shortcuts):
//   Esc        quit
//   Space / N  next tree (random_genome with seed+1)
//   LMB drag   orbit (yaw/pitch)
//   wheel      zoom (distance)
//   RMB / MMB  pan (move look-at target in the camera plane)

use std::time::Instant;

use enki_app::flora_render::{FloraRenderer, ShadowUniforms};
use enki_app::flora_view::{FloraView, LeafLod, LeafMode, LeafTuning, RenderMode, TreeSpec};
use enki_app::fps::FpsMeter;
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
    event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent},
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

/// Selectable render modes + their labels, in panel order.
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
    // appendageDensity feeds round(12·d·tipBoost) clusters/seg, so the usable band
    // is ~0.02 (first leaves) .. ~0.3 (very dense); below ~0.02 it rounds to bare.
    // The old 0.0..1.5 range crammed that whole band into the bottom ~13% (and
    // saturated the MAX_LEAVES budget well before 1.5). 0.0..0.3 puts the realistic
    // range across the full slider so the cliff at ~0.02 is easy to navigate.
    g("appendageDensity", "Leaf density", 0.0, 0.3),
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
    /// Debug INSPECTOR dock state (dryad inspectorPanels port). Holds the cached
    /// egui texture handles for the baked leaf COLOR / NORMAL sprites + the CPU
    /// bark swatch, plus a change-key so they're only re-baked when the genome
    /// changes. Pigment swatch/ramp + cross-section are painter-drawn every frame
    /// (cheap, always current) and need no caching.
    inspector: Inspector,
}

/// Cached Inspector previews. The leaf/bark images are expensive CPU bakes, so
/// they're keyed on a cheap change-stamp (`leaf_genes` is `Copy`; bark genes are
/// the push-vec4s) and only rebuilt when that stamp changes.
#[derive(Default)]
struct Inspector {
    leaf_color: Option<egui::TextureHandle>,
    leaf_normal: Option<egui::TextureHandle>,
    bark: Option<egui::TextureHandle>,
    /// Stamp the cached images were baked for: (leaf genes, leaf mode flag, bark
    /// vec4s). Compared each frame; a mismatch triggers a re-bake.
    stamp: Option<InspectorStamp>,
}

/// Cheap equality stamp for the Inspector's cached bakes. Built from the live
/// `FloraView` each frame; when it differs from the cached one the leaf/bark
/// images are re-baked.
#[derive(Clone, Copy, PartialEq)]
struct InspectorStamp {
    /// Leaf-shape genes + pigment + seed (the bake inputs). Stored as the bit
    /// patterns so it's `Eq` without `f64: Eq`.
    leaf: [u64; 8],
    single: bool,
    bark: [u32; 12],
}

impl InspectorStamp {
    fn new(flora: &FloraView) -> Self {
        let lg = flora.leaf_genes();
        let leaf = [
            lg.pigment.to_bits(),
            lg.leaf_width.to_bits(),
            lg.leaf_length.to_bits(),
            lg.leaf_tip.to_bits(),
            lg.leaf_serration.to_bits(),
            lg.leaf_lobing.to_bits(),
            lg.leaf_skew.to_bits(),
            lg.seed as u64,
        ];
        let (b0, b1, b2) = flora.bark_genes();
        let bark = [
            b0[0].to_bits(), b0[1].to_bits(), b0[2].to_bits(), b0[3].to_bits(),
            b1[0].to_bits(), b1[1].to_bits(), b1[2].to_bits(), b1[3].to_bits(),
            b2[0].to_bits(), b2[1].to_bits(), b2[2].to_bits(), b2[3].to_bits(),
        ];
        InspectorStamp {
            leaf,
            single: flora.leaf_mode() == LeafMode::Single,
            bark,
        }
    }
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

// ── dryad-look egui theme ────────────────────────────────────────────────────
// Approximate dryad's HTML/CSS CAS panel (C:\Work\Prototype\dryad\index.html) in
// egui 0.33. dryad mapping (hex = flattened from the CSS rgba, since egui has no
// per-panel `backdrop-filter: blur` + translucency, so fills are OPAQUE approx):
//   panel/window bg   rgba(8,10,14,0.88) #080A0E  → #0A0D12 (nudged up so it
//                                                    doesn't read pure-black w/o blur)
//   panel border      rgba(80,120,160,.35)        → #3A526B
//   body text         #9ab                        → #99AABB
//   strong/heading    #cde                        → #CCDDEE
//   dim caption       rgba(120,160,200,~.7)       → #6E8CA8
//   accent (sel/slider/active-tab) #5af           → #55AAFF
//   accent2 (active text / hovered)#8cf           → #88CCFF
//   slider track / input bg rgba(20,30,40,.9)     → #141E28  (extreme_bg_color)
//   button bg         rgba(20,40,60,.9)           → #14283C  (widgets.inactive)
//   hover bg          rgba(30,60,100,.9)          → #1E3C64  (widgets.hovered)
//   active/sel bg     rgba(20,50,90,.85)          → #14325A  (widgets.active)
//   radius: panels 6px, buttons/inputs 3px; font: body 11px, heading 13px.
// APPROXIMATIONS (egui can't match CSS exactly): no blur/translucency (opaque
// fills); no 2px bottom-border active-tab underline (egui `selectable_value`
// fills the selected pill with the accent instead — closest available); no
// per-side padding or uppercase letter-spacing on section labels.
fn apply_dryad_style(ctx: &egui::Context) {
    use egui::{Color32, CornerRadius, FontFamily, FontId, Stroke, TextStyle, Vec2};

    let panel = Color32::from_rgb(0x0A, 0x0D, 0x12); // dryad #080A0E (nudged up)
    let border = Color32::from_rgb(0x3A, 0x52, 0x6B); // rgba(80,120,160,.35)
    let text = Color32::from_rgb(0x99, 0xAA, 0xBB); // #9ab body
    let dim = Color32::from_rgb(0x6E, 0x8C, 0xA8); // rgba(120,160,200,~.7) caption
    let accent = Color32::from_rgb(0x55, 0xAA, 0xFF); // #5af  selection / slider / tab
    let accent2 = Color32::from_rgb(0x88, 0xCC, 0xFF); // #8cf  active text / hovered
    let track = Color32::from_rgb(0x14, 0x1E, 0x28); // input + slider-track bg
    let widget = Color32::from_rgb(0x14, 0x28, 0x3C); // button bg rgba(20,40,60)
    let hovered = Color32::from_rgb(0x1E, 0x3C, 0x64); // hover bg rgba(30,60,100)
    let active = Color32::from_rgb(0x14, 0x32, 0x5A); // selected/active rgba(20,50,90)

    let mut v = egui::Visuals::dark();
    v.panel_fill = panel; // SidePanel  (= dryad #ui)
    v.window_fill = panel; // Stats / View windows (= #stats-panel)
    v.window_stroke = Stroke::new(1.0, border);
    v.window_corner_radius = CornerRadius::same(6); // dryad 6px panels
    v.faint_bg_color = Color32::from_rgb(0x10, 0x18, 0x22); // alt-row / faint fill
    v.extreme_bg_color = track; // SLIDER TRACK + text-edit bg
    v.selection.bg_fill = accent; // slider filled portion + selected text (#5af)
    v.selection.stroke = Stroke::new(1.0, accent2);
    v.hyperlink_color = accent2;
    v.override_text_color = Some(text); // global #9ab; ui.strong()/colored for #cde/#8cf

    let r = CornerRadius::same(3); // dryad button/input 3px

    v.widgets.noninteractive.bg_fill = panel;
    v.widgets.noninteractive.weak_bg_fill = panel;
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, dim); // separators/captions
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, border);

    v.widgets.inactive.bg_fill = widget; // button rest #14283C
    v.widgets.inactive.weak_bg_fill = widget;
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, accent2); // button label #8cf
    v.widgets.inactive.bg_stroke = Stroke::new(1.0, Color32::from_rgb(0x35, 0x55, 0x70));
    v.widgets.inactive.corner_radius = r;

    v.widgets.hovered.bg_fill = hovered; // #1E3C64
    v.widgets.hovered.weak_bg_fill = hovered;
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::from_rgb(0xAA, 0xDD, 0xFF)); // #adf
    v.widgets.hovered.bg_stroke = Stroke::new(1.0, accent2);
    v.widgets.hovered.corner_radius = r;

    v.widgets.active.bg_fill = active; // pressed/selected #14325A
    v.widgets.active.weak_bg_fill = active;
    v.widgets.active.fg_stroke = Stroke::new(1.0, accent2);
    v.widgets.active.bg_stroke = Stroke::new(1.0, accent);
    v.widgets.active.corner_radius = r;

    // `open` (combo popups, expanded menus) — keep them on-theme too.
    v.widgets.open.bg_fill = active;
    v.widgets.open.weak_bg_fill = active;
    v.widgets.open.fg_stroke = Stroke::new(1.0, accent2);
    v.widgets.open.bg_stroke = Stroke::new(1.0, accent);
    v.widgets.open.corner_radius = r;

    ctx.set_visuals(v);

    // Spacing / sizing — dryad is compact: 11px font, ~6px gaps, 4px button pad.
    ctx.style_mut(|s| {
        s.spacing.item_spacing = Vec2::new(6.0, 6.0); // dryad gap:6 / margin-top:6
        s.spacing.button_padding = Vec2::new(8.0, 4.0); // dryad padding 4px 10px
        s.spacing.slider_width = 180.0; // fills the 320px panel
        s.spacing.interact_size.y = 20.0; // tighter rows
        s.visuals.clip_rect_margin = 3.0;
        // Font sizes ≈ dryad: body 11px, heading 13px (egui pt ≈ px here).
        s.text_styles
            .insert(TextStyle::Heading, FontId::new(13.0, FontFamily::Proportional));
        s.text_styles
            .insert(TextStyle::Body, FontId::new(11.0, FontFamily::Proportional));
        s.text_styles
            .insert(TextStyle::Button, FontId::new(11.0, FontFamily::Proportional));
        s.text_styles
            .insert(TextStyle::Monospace, FontId::new(11.0, FontFamily::Monospace));
    });
}

/// Draw the stem CROSS-SECTION outline (dryad inspectorPanels `crossSectionPoints`
/// + branchMesh `ringProfile`) into `rect` with the egui painter. Filled brown,
/// dark stroke, fitted to the rect. ribCount is derived from `segmentation` per
/// dryad: `clamp(8 + round(segmentation*8), 8, 16)`.
fn draw_cross_section(
    painter: &egui::Painter,
    rect: egui::Rect,
    ribbing: f32,
    flatness: f32,
    segmentation: f32,
) {
    // dryad branchMesh.js constants.
    const RIB_DEPTH: f32 = 0.35;
    const FLAT_MAX: f32 = 0.80;
    let rib_count = (8.0 + (segmentation * 8.0).round()).clamp(8.0, 16.0);

    const STEPS: usize = 180;
    let mut pts: Vec<(f32, f32)> = Vec::with_capacity(STEPS);
    let mut max_r = 1e-6f32;
    for i in 0..STEPS {
        let theta = std::f32::consts::TAU * i as f32 / STEPS as f32;
        // ringProfile: rScale = 1 - ribbing*RIB_DEPTH*0.5*(1 + cos(ribCount*θ)).
        let r_scale = 1.0 - ribbing * RIB_DEPTH * 0.5 * (1.0 + (rib_count * theta).cos());
        let u_scale = 1.0 - flatness * FLAT_MAX;
        let x = r_scale * theta.cos() * u_scale;
        let y = r_scale * theta.sin();
        pts.push((x, y));
        max_r = max_r.max((x * x + y * y).sqrt());
    }
    // Normalize so max radius = 1, then fit into the rect (with padding, +Y up).
    let pad = 12.0;
    let half = (rect.width().min(rect.height()) * 0.5 - pad).max(1.0);
    let center = rect.center();
    let poly: Vec<egui::Pos2> = pts
        .iter()
        .map(|(x, y)| {
            let nx = x / max_r;
            let ny = y / max_r;
            egui::pos2(center.x + nx * half, center.y - ny * half) // flip Y
        })
        .collect();

    painter.add(egui::Shape::convex_polygon(
        poly,
        egui::Color32::from_rgb(0x78, 0x56, 0x34), // dryad rgba(120,86,52)
        egui::Stroke::new(1.25, egui::Color32::from_rgb(0x3C, 0x28, 0x18)),
    ));
}

impl FloraGui {
    fn new(rhi: &Rhi, window: &Window) -> Self {
        let ctx = egui::Context::default();
        apply_dryad_style(&ctx);
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

        Self { ctx, winit, renderer, output: None, inspector: Inspector::default() }
    }

    fn on_window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.winit.on_window_event(window, event).consumed
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
        leaf_mode: &mut LeafMode,
        leaf_tuning: &mut LeafTuning,
        seed: &mut u32,
        tab: &mut Tab,
        wind_on: &mut bool,
        wind_strength: &mut f32,
        wind_dir: &mut Vec3,
        gizmo_on: &mut bool,
        bloom_on: &mut bool,
        bloom_strength: &mut f32,
        dappled: &mut bool,
        leaf_opt: &mut bool,
        flora: Option<&FloraView>,
        stats: &Stats,
    ) -> PanelOut {
        let raw_input = self.winit.take_egui_input(window);
        let mut out = PanelOut::default();

        // ── INSPECTOR bakes (BEFORE ctx.run, so load_texture's one-shot upload
        //    submit lands outside the rendering instance — same constraint as the
        //    panel build itself). Only re-bake the expensive leaf/bark images when
        //    the genome changed (stamp mismatch); otherwise reuse the handles. ──
        if let Some(flora) = flora {
            let stamp = InspectorStamp::new(flora);
            if self.inspector.stamp != Some(stamp) {
                let lg = flora.leaf_genes();
                let single = stamp.single;
                let leaf = if single {
                    enki_flora::leaf_texture::bake_leaf_single(&lg)
                } else {
                    enki_flora::leaf_texture::bake_leaf_cluster(&lg)
                };
                let sz = [leaf.size as usize, leaf.size as usize];
                let color_img = egui::ColorImage::from_rgba_unmultiplied(sz, &leaf.color);
                let normal_img = egui::ColorImage::from_rgba_unmultiplied(sz, &leaf.normal);
                self.inspector.leaf_color = Some(self.ctx.load_texture(
                    "insp_leaf_color",
                    color_img,
                    egui::TextureOptions::LINEAR,
                ));
                self.inspector.leaf_normal = Some(self.ctx.load_texture(
                    "insp_leaf_normal",
                    normal_img,
                    egui::TextureOptions::LINEAR,
                ));
                // Bark swatch (CPU-eval of barkAlbedo — labeled APPROX).
                let (b0, b1, b2) = flora.bark_genes();
                let bg = enki_flora::bark_swatch::BarkGenes::from_vec4s(b0, b1, b2);
                const BARK_SZ: u32 = 192;
                let bark_px = enki_flora::bark_swatch::bake_bark_swatch(&bg, BARK_SZ);
                let bark_img = egui::ColorImage::from_rgba_unmultiplied(
                    [BARK_SZ as usize, BARK_SZ as usize],
                    &bark_px,
                );
                self.inspector.bark = Some(self.ctx.load_texture(
                    "insp_bark",
                    bark_img,
                    egui::TextureOptions::LINEAR,
                ));
                self.inspector.stamp = Some(stamp);
            }
        }
        // Snapshot the pigment/cross genes for the painter-drawn panels.
        let pigment = genome.pigment;
        let ribbing = genome.ribbing;
        let flatness = genome.flatness;
        let segmentation = genome.segmentation;
        let single_leaf = *leaf_mode == LeafMode::Single;
        // TextureHandle is cheap to clone (Arc); clone out so the ctx.run closure
        // doesn't borrow `self` (ctx is borrowed for the whole run).
        let insp_leaf_color = self.inspector.leaf_color.clone();
        let insp_leaf_normal = self.inspector.leaf_normal.clone();
        let insp_bark = self.inspector.bark.clone();

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
                    // dryad <h1> is 13px/500 #cde (heading TextStyle = 13px above).
                    ui.heading(
                        egui::RichText::new("flora · CAS")
                            .color(egui::Color32::from_rgb(0xCC, 0xDD, 0xEE)),
                    );

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

                    // Tab strip — dryad's `.tab-btn` row: transparent text that is
                    // dim (#9ab-ish) when inactive and accent (#8cf) with a 2px #5af
                    // bottom-border when active. egui has no CSS underline, so we
                    // hand-roll `selectable_label`s (transparent, not the accent pill)
                    // + paint an accent hline under the active one (closest match).
                    let accent = egui::Color32::from_rgb(0x55, 0xAA, 0xFF); // #5af
                    let accent2 = egui::Color32::from_rgb(0x88, 0xCC, 0xFF); // #8cf
                    let tab_dim = egui::Color32::from_rgb(0x6E, 0x8C, 0xA8);
                    ui.horizontal_wrapped(|ui| {
                        for (t, label) in TABS {
                            let on = *tab == t;
                            let txt = egui::RichText::new(label)
                                .color(if on { accent2 } else { tab_dim });
                            // Transparent button (override the accent selection pill)
                            // so only the text colour + underline read as "active".
                            let resp = ui.add(
                                egui::Button::new(txt)
                                    .fill(egui::Color32::TRANSPARENT)
                                    .stroke(egui::Stroke::NONE),
                            );
                            if on {
                                let r = resp.rect;
                                ui.painter().hline(
                                    r.x_range(),
                                    r.bottom() - 1.0,
                                    egui::Stroke::new(2.0, accent), // dryad 2px #5af
                                );
                            }
                            if resp.clicked() {
                                *tab = t;
                            }
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
                    // dryad stat row: dim label LEFT, monospace #8cf value RIGHT.
                    let dim = egui::Color32::from_rgb(0x6E, 0x8C, 0xA8);
                    let val_col = egui::Color32::from_rgb(0x88, 0xCC, 0xFF); // #8cf
                    // One dryad-style row: label (dim) left, mono value (color) right.
                    let row = |ui: &mut egui::Ui, label: &str, value: String, col: egui::Color32| {
                        ui.horizontal(|ui| {
                            ui.label(egui::RichText::new(label).color(dim));
                            ui.with_layout(
                                egui::Layout::right_to_left(egui::Align::Center),
                                |ui| {
                                    ui.monospace(egui::RichText::new(value).color(col));
                                },
                            );
                        });
                    };
                    // FPS color-coded per dryad (≥50 good, ≥30 ok, else bad).
                    let fps_color = if stats.fps >= 50.0 {
                        egui::Color32::from_rgb(0x4c, 0xd9, 0x64)
                    } else if stats.fps >= 30.0 {
                        egui::Color32::from_rgb(0xe6, 0xc4, 0x4c)
                    } else {
                        egui::Color32::from_rgb(0xe0, 0x5c, 0x5c)
                    };
                    row(ui, "FPS", format!("{:.0}  ({:.2} ms)", stats.fps, stats.frame_ms), fps_color);
                    row(ui, "Triangles", format!("{}", stats.triangles), val_col);
                    row(ui, "Draw calls", format!("{}", stats.draw_calls), val_col);
                    row(ui, "Leaf clusters", format!("{}", stats.leaves), val_col);
                    row(ui, "Bones", format!("{}", stats.bones), val_col);
                    row(ui, "Resolution", format!("{}x{}", stats.resolution.0, stats.resolution.1), val_col);
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
                    // Leaf mode: Cluster (one card per cluster, fan sprite) vs
                    // Single (each cluster fanned into its individual leaves, each
                    // a one-leaf sprite). Flipping it changes BOTH the mesh
                    // (expansion) and the baked sprite, so it forces a full rebuild.
                    ui.label("Leaf mode:");
                    ui.horizontal_wrapped(|ui| {
                        let mut c = ui
                            .selectable_value(leaf_mode, LeafMode::Cluster, "Cluster")
                            .changed();
                        c |= ui
                            .selectable_value(leaf_mode, LeafMode::Single, "Single")
                            .changed();
                        if c {
                            out.rebuild = true;
                        }
                    });
                    ui.separator();
                    // ── Leaf placement (phyllotaxis) tuning — RENDER-SIDE knobs. ──
                    // These adjust the leaf STRIP orientation/droop/size in
                    // build_leaf_mesh on top of the gen-core SoA; they are NOT
                    // genome genes, so editing them rebuilds ONLY the leaf mesh and
                    // never touches the golden gen-core path. Identity defaults
                    // (lift/up_bias 0, droop/size ×1.0) reproduce the current look.
                    egui::CollapsingHeader::new("Leaf placement")
                        .default_open(false)
                        .show(ui, |ui| {
                            // Divergence angle readout (the golden angle the gen-
                            // core azimuth blends toward — reference only).
                            let golden_deg =
                                (enki_flora::foliage::GOLDEN_ANGLE * 180.0
                                    / std::f64::consts::PI) as f32;
                            ui.label(
                                egui::RichText::new(format!(
                                    "Divergence (golden): {golden_deg:.3}°"
                                ))
                                .color(egui::Color32::from_rgb(0x6E, 0x8C, 0xA8)),
                            );
                            let mut c = false;
                            c |= ui
                                .add(
                                    egui::Slider::new(&mut leaf_tuning.lift, -1.0..=1.0)
                                        .text("Insertion / lift"),
                                )
                                .changed();
                            c |= ui
                                .add(
                                    egui::Slider::new(&mut leaf_tuning.up_bias, 0.0..=1.0)
                                        .text("Up-bias"),
                                )
                                .changed();
                            c |= ui
                                .add(
                                    egui::Slider::new(&mut leaf_tuning.droop, 0.0..=3.0)
                                        .text("Droop"),
                                )
                                .changed();
                            c |= ui
                                .add(
                                    egui::Slider::new(&mut leaf_tuning.size, 0.2..=2.0)
                                        .text("Leaf size"),
                                )
                                .changed();
                            // ponytail: render-side leaf DENSITY (smooth subsample,
                            // sidesteps the cliffy appendageDensity gene). 0 = bare,
                            // 1 = every twig anchor; rebuilds the leaf mesh on edit.
                            c |= ui
                                .add(
                                    egui::Slider::new(&mut leaf_tuning.density, 0.0..=1.0)
                                        .text("Leaf density"),
                                )
                                .changed();
                            if ui.button("Reset").clicked() {
                                *leaf_tuning = LeafTuning::default();
                                c = true;
                            }
                            // Editing any knob rebuilds the leaf mesh (the existing
                            // rebuild path re-runs build_leaf_mesh).
                            if c {
                                out.rebuild = true;
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
                    // Wind-direction debug arrow gizmo (flora-owned, scene pass).
                    ui.checkbox(gizmo_on, "Wind gizmo");
                    ui.separator();
                    // Bloom (UnrealBloom POST — animation only, NOT a rebuild).
                    ui.checkbox(bloom_on, "Bloom");
                    ui.add_enabled(
                        *bloom_on,
                        egui::Slider::new(bloom_strength, 0.0..=1.0).text("strength"),
                    );
                    ui.separator();
                    // DAPPLED sun-through-canopy self-shadow (render-side, NOT a
                    // rebuild — a checkbox alone leaves `out.rebuild` unset; it just
                    // flips ShadowUniforms.params.w next frame).
                    ui.checkbox(dappled, "Dappled shadows");
                    ui.separator();
                    // LEAF OPTIMIZATION master toggle (distance LOD: card subsample +
                    // leaf-shadow drop + impostor far-tier). Render-side, no rebuild.
                    // At the default close framing it's a no-op (NEAR band), so on/off
                    // here is identical — ZOOM OUT to see the live readout below move.
                    ui.checkbox(leaf_opt, "Leaf optimization");
                    // LIVE readout so the LOD is observable (this is why it can look
                    // like "nothing" up close — it's full-quality until you zoom out).
                    let dim = egui::Color32::from_rgb(150, 160, 175);
                    if *leaf_opt {
                        ui.label(
                            egui::RichText::new(format!(
                                "  cards {}/{} ({:.0}%) · shadows {} · impostor {:.0}%",
                                stats.leaf_cards_drawn,
                                stats.leaf_cards_total,
                                stats.leaf_density * 100.0,
                                if stats.leaf_shadows { "on" } else { "off" },
                                stats.leaf_impostor * 100.0,
                            ))
                            .small()
                            .color(dim),
                        );
                        ui.label(
                            egui::RichText::new("  (zoom out — thins as the tree shrinks)")
                                .small()
                                .italics()
                                .color(dim),
                        );
                    } else {
                        ui.label(
                            egui::RichText::new(format!("  full: {} cards, every frame", stats.leaf_cards_total))
                                .small()
                                .color(dim),
                        );
                    }
                });

            // ── (D) INSPECTOR dock — dryad inspectorPanels port. A scrollable
            //    window of live previews of the generated components, each LABELED,
            //    updating when the genome/sliders change. Placed bottom-right by
            //    default (clear of Stats/View), draggable. ──
            let dim = egui::Color32::from_rgb(0x6E, 0x8C, 0xA8);
            let heading = egui::Color32::from_rgb(0xCC, 0xDD, 0xEE);
            egui::Window::new("Inspector")
                .anchor(egui::Align2::RIGHT_BOTTOM, [-8.0, -8.0])
                .resizable(true)
                .default_width(180.0)
                .show(ctx, |ui| {
                    egui::ScrollArea::vertical()
                        // Fit the scroll region to the actual (DPI-scaled) viewport
                        // so the dock never overflows the screen — scroll internally.
                        .max_height((ui.ctx().screen_rect().height() - 120.0).max(240.0))
                        .show(ui, |ui| {
                            let img_w = (ui.available_width() - 8.0).clamp(80.0, 128.0);

                            // (1) LEAF TEXTURE — baked COLOR sprite.
                            ui.label(
                                egui::RichText::new(if single_leaf {
                                    "LEAF TEXTURE (single)"
                                } else {
                                    "LEAF TEXTURE (cluster)"
                                })
                                .color(heading),
                            );
                            if let Some(t) = &insp_leaf_color {
                                ui.image((t.id(), egui::vec2(img_w, img_w)));
                            } else {
                                ui.label(egui::RichText::new("(no tree)").color(dim));
                            }

                            // LEAF NORMAL map.
                            ui.add_space(4.0);
                            ui.label(egui::RichText::new("LEAF NORMAL").color(heading));
                            if let Some(t) = &insp_leaf_normal {
                                ui.image((t.id(), egui::vec2(img_w, img_w)));
                            }

                            ui.separator();

                            // (2) PIGMENT swatch + readout + ramp (painter-drawn).
                            ui.label(egui::RichText::new("PIGMENT").color(heading));
                            let [pr, pg, pb] = enki_flora::color::pigment_to_color(pigment);
                            let swcol = egui::Color32::from_rgb(
                                (pr * 255.0).round() as u8,
                                (pg * 255.0).round() as u8,
                                (pb * 255.0).round() as u8,
                            );
                            let (rect, _) = ui.allocate_exact_size(
                                egui::vec2(img_w, 22.0),
                                egui::Sense::hover(),
                            );
                            ui.painter().rect_filled(rect, 3.0, swcol);
                            // hex + p= readout, luminance-aware text color.
                            let hex = format!(
                                "#{:02X}{:02X}{:02X}  p={:.2}",
                                (pr * 255.0).round() as u8,
                                (pg * 255.0).round() as u8,
                                (pb * 255.0).round() as u8,
                                pigment
                            );
                            let lum = 0.299 * pr + 0.587 * pg + 0.114 * pb;
                            let tcol = if lum > 0.5 {
                                egui::Color32::from_rgba_unmultiplied(0, 0, 0, 200)
                            } else {
                                egui::Color32::from_rgba_unmultiplied(255, 255, 255, 220)
                            };
                            ui.painter().text(
                                rect.left_center() + egui::vec2(6.0, 0.0),
                                egui::Align2::LEFT_CENTER,
                                hex,
                                egui::FontId::monospace(10.0),
                                tcol,
                            );

                            // Pigment RAMP (full hue sweep) + current marker.
                            ui.add_space(2.0);
                            let (ramp, _) = ui.allocate_exact_size(
                                egui::vec2(img_w, 12.0),
                                egui::Sense::hover(),
                            );
                            let cols = 48usize;
                            for c in 0..cols {
                                let t0 = c as f32 / cols as f32;
                                let t1 = (c + 1) as f32 / cols as f32;
                                let [rr, gg, bb] =
                                    enki_flora::color::pigment_to_color(t0 as f64);
                                let slice = egui::Rect::from_min_max(
                                    egui::pos2(ramp.left() + ramp.width() * t0, ramp.top()),
                                    egui::pos2(ramp.left() + ramp.width() * t1, ramp.bottom()),
                                );
                                ui.painter().rect_filled(
                                    slice,
                                    0.0,
                                    egui::Color32::from_rgb(
                                        (rr * 255.0).round() as u8,
                                        (gg * 255.0).round() as u8,
                                        (bb * 255.0).round() as u8,
                                    ),
                                );
                            }
                            // Marker at the current pigment.
                            let mx = ramp.left() + ramp.width() * pigment as f32;
                            ui.painter().vline(
                                mx,
                                ramp.y_range(),
                                egui::Stroke::new(2.0, egui::Color32::WHITE),
                            );
                            ui.painter().vline(
                                mx + 1.5,
                                ramp.y_range(),
                                egui::Stroke::new(1.0, egui::Color32::BLACK),
                            );

                            ui.separator();

                            // (3) BARK SWATCH — CPU-eval of barkAlbedo (APPROX).
                            ui.label(egui::RichText::new("BARK SWATCH").color(heading));
                            ui.label(
                                egui::RichText::new(
                                    "CPU approx of in-shader barkAlbedo; V = twig→trunk",
                                )
                                .color(dim),
                            );
                            if let Some(t) = &insp_bark {
                                ui.image((t.id(), egui::vec2(img_w, img_w)));
                            }

                            ui.separator();

                            // (5) CROSS-SECTION — stem ring outline (painter-drawn).
                            ui.label(egui::RichText::new("STEM CROSS-SECTION").color(heading));
                            let (cs, _) = ui.allocate_exact_size(
                                egui::vec2(img_w, img_w * 0.8),
                                egui::Sense::hover(),
                            );
                            draw_cross_section(
                                ui.painter(),
                                cs,
                                ribbing as f32,
                                flatness as f32,
                                segmentation as f32,
                            );
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
    /// Leaf-optimization live readout (so the LOD is OBSERVABLE — it's a no-op at
    /// close framing by design). `opt_on` = master toggle; `cards_drawn/total` =
    /// the density-LOD prefix; `density`/`impostor` are the current LOD fractions;
    /// `shadows` = whether leaves still cast.
    leaf_cards_drawn: u32,
    leaf_cards_total: u32,
    leaf_density: f32,
    leaf_impostor: f32,
    leaf_shadows: bool,
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

/// Interactive orbit/zoom/pan camera (replaces the time-driven auto-spin).
///
/// State is the usual turntable quartet: a `target` look-at point (starts at the
/// auto-fit tree center), an eye `distance` from it (starts at the fit distance),
/// and `yaw`/`pitch` spherical angles. The eye sits at
/// `target + dir(yaw,pitch) * distance`, looking at `target`, +Y up — the same
/// quaternion convention the old `camera()` used (`look_to_rh(ZERO, fwd, +Y)`
/// inverted), so framing/lighting are unchanged at the initial pose.
///
/// Mappings (driven from `window_event`, only when egui didn't grab the pointer):
///   - LMB drag  → `yaw -= dx*ORBIT_SENS`, `pitch += dy*ORBIT_SENS` (clamped ±~89°)
///   - wheel     → `distance *= (1 - scroll*ZOOM_RATE)`, clamped to [min,max]
///   - RMB/MMB   → pan `target` in the camera's right/up plane, scaled by distance
#[derive(Clone, Copy)]
struct OrbitCamera {
    target: Vec3,
    distance: f32,
    yaw: f32,
    pitch: f32,
}

/// Drag sensitivity (radians per pixel) for orbit yaw/pitch.
const ORBIT_SENS: f32 = 0.005;
/// Geometric zoom step per wheel notch (fraction of current distance).
const ZOOM_RATE: f32 = 0.1;
/// Pitch clamp (~±89°) so the camera never flips through the pole.
const PITCH_LIMIT: f32 = 1.55;
/// Pan speed: world units per pixel, per unit of distance.
const PAN_RATE: f32 = 0.0015;

impl OrbitCamera {
    /// Frame from the auto-fit AABB. Initial yaw 0 / pitch 18° reproduces the old
    /// auto-spin's starting elevation, so the first frame matches the prior view.
    fn from_fit(fit: Fit) -> Self {
        OrbitCamera {
            target: fit.center,
            distance: fit.distance.max(0.05),
            yaw: 0.0,
            pitch: 18_f32.to_radians(),
        }
    }

    fn orbit(&mut self, dx: f32, dy: f32) {
        self.yaw -= dx * ORBIT_SENS;
        self.pitch = (self.pitch + dy * ORBIT_SENS).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// `scroll` > 0 zooms IN (wheel up). Distance stays in a sane range so the
    /// tree can't be lost behind the near plane or shrunk to a dot.
    fn zoom(&mut self, scroll: f32) {
        self.distance = (self.distance * (1.0 - scroll * ZOOM_RATE)).clamp(0.05, 10_000.0);
    }

    /// Pan the look-at target in the camera's right/up plane (RMB/MMB drag). Drag
    /// magnitude scales with distance so the world tracks the cursor at any zoom.
    fn pan(&mut self, dx: f32, dy: f32) {
        let (right, up, _) = self.basis();
        let k = self.distance * PAN_RATE;
        self.target += right * (-dx * k) + up * (dy * k);
    }

    /// Orthonormal camera basis (right, up, forward). `forward` points from the
    /// eye TOWARD the target. Pole-safe via `normalize_or`.
    fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        // dir = eye-from-target direction; forward (eye→target) is its negation.
        let dir = Vec3::new(cy * cp, sp, sy * cp);
        let forward = (-dir).normalize_or(Vec3::NEG_Z);
        let right = forward.cross(Vec3::Y).normalize_or(Vec3::X);
        let up = right.cross(forward);
        (right, up, forward)
    }

    fn camera(&self) -> Camera {
        let (_, _, forward) = self.basis();
        let eye = self.target - forward * self.distance;
        let orientation =
            Quat::from_mat4(&Mat4::look_to_rh(Vec3::ZERO, forward, Vec3::Y)).inverse();
        Camera {
            position: eye.as_dvec3(),
            orientation,
            fov_y_radians: 60_f32.to_radians(),
            near: 0.1,
            far: 10_000.0,
        }
    }
}

/// Parse the optional `FLORA_CAM=yaw,pitch,dist` env var (degrees, degrees, world
/// units) for headless capture framing. Returns `None` if unset/malformed so the
/// default auto-fit pose is used (existing FLORA_SCREENSHOT captures unchanged).
fn cam_override() -> Option<(f32, f32, f32)> {
    let raw = std::env::var("FLORA_CAM").ok()?;
    let parts: Vec<f32> = raw
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    if parts.len() == 3 {
        Some((parts[0], parts[1], parts[2]))
    } else {
        None
    }
}

/// Parse the optional `FLORA_WIND=strength,dirx,dirz[,time]` env var for headless
/// capture. Forces wind ON at the given strength/direction and (4th value, else a
/// non-zero default) seeds the wind clock so a STILL shows the canopy displaced
/// along the wind vector. Returns `(strength, dir_x, dir_z, time)` or `None` when
/// unset/malformed (existing captures keep the panel defaults).
fn wind_override() -> Option<(f32, f32, f32, f32)> {
    let raw = std::env::var("FLORA_WIND").ok()?;
    let parts: Vec<f32> = raw
        .split(',')
        .filter_map(|s| s.trim().parse::<f32>().ok())
        .collect();
    match parts.len() {
        // Default to a non-zero phase so a single frame is mid-gust, not at rest.
        3 => Some((parts[0], parts[1], parts[2], 2.0)),
        4 => Some((parts[0], parts[1], parts[2], parts[3])),
        _ => None,
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
    staging: Option<Staging>,
    egui: Option<FloraGui>,
    /// Interactive orbit/zoom/pan camera (replaces the auto-spin). Re-framed from
    /// the auto-fit AABB on every build/reseed; the screenshot path drives this
    /// directly so captures keep the deterministic auto-fit framing.
    orbit: OrbitCamera,
    /// Mouse drag state for the orbit controller. `last_cursor` is the previous
    /// CursorMoved position (for deltas); the button flags select orbit vs pan.
    lmb_down: bool,
    rmb_down: bool,
    mmb_down: bool,
    last_cursor: Option<(f32, f32)>,
    /// Wind-gizmo interaction state. `gizmo_hover` is set each mouse-move when the
    /// cursor's ground-plane (y=0) point lands on the arrow footprint and egui
    /// isn't using the pointer (drives the highlight). `gizmo_grabbed` is latched
    /// on LMB-down over the arrow: while true, the orbit camera is SUPPRESSED
    /// (its dispatch keys off `lmb_down`, which we leave false during a grab) and
    /// CursorMoved instead ray-casts to y=0 and rewrites `wind_dir`.
    gizmo_hover: bool,
    gizmo_grabbed: bool,
    /// Last known cursor position regardless of button state — needed so a hover
    /// hit-test (and the grab decision at press time) has a position even when no
    /// drag is in flight. `last_cursor` is reset between drags, so it can't serve.
    cursor_pos: Option<(f32, f32)>,
    seed: u32,
    /// The single source of truth for what's on screen — reseed and edits both
    /// flow through this genome, so the panel always reflects the live tree.
    genome: Genome,
    env: Env,
    mode: RenderMode,
    /// Leaf rendering mode (Cluster = one card/cluster, Single = one card/leaf).
    /// Flipping it forces a tree rebuild (mesh expansion + sprite re-bake).
    leaf_mode: LeafMode,
    /// Render-side leaf-placement tuning (lift/up_bias/droop/size). RENDER-ONLY
    /// (not a genome gene) — editing it rebuilds the leaf mesh but never touches
    /// the golden gen-core path. Defaults are identity (current look unchanged).
    leaf_tuning: LeafTuning,
    /// Active left-panel CAS tab (Climate / Trunk / Branches / Leaves / Root).
    tab: Tab,
    last_frame: Instant,
    /// Smoothed/throttled FPS meter (≈2 Hz display refresh) — same as the main
    /// `enki` app, so the readout doesn't flicker every frame.
    fps: FpsMeter,
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
    /// Wind-direction debug GIZMO toggle (default ON): a flora-owned unlit arrow
    /// at the tree base pointing along `wind_dir`, length/color scaled by strength.
    /// Updates LIVE as the dir x/z + strength sliders move. Independent of
    /// `wind_on` — it's a direction indicator, drawn whenever the toggle is set.
    gizmo_on: bool,
    /// UnrealBloom POST controls (dryad: strength 0.15). `bloom_on` gates the
    /// effect; the effective strength fed to `run_bloom` is 0 when off, so OFF or
    /// strength 0 is byte-identical to the offscreen-stage parity image.
    bloom_on: bool,
    bloom_strength: f32,
    /// DAPPLED sun-through-canopy self-shadow mode (render-side, NOT a rebuild).
    /// When on, the high-res shadow map's sun-spots are let to dominate: the
    /// shader softens the canopy-normal fake + the exposure floor + the ambient/
    /// IBL fill so the dappling reads on the trunk + interior leaves. Carried in
    /// `ShadowUniforms.params.w` and uploaded per-frame — OFF is byte-identical to
    /// today's look. `FLORA_DAPPLED=1` seeds it on for headless captures.
    dappled: bool,
    /// "Leaf optimization" master toggle (default ON): the distance-driven leaf
    /// LOD stack — card-count subsample, leaf-shadow drop, instanced draw, and the
    /// billboard impostor far-tier. OFF forces the full per-card path at every
    /// distance (the reference look). At the default close framing the toggle is a
    /// no-op (NEAR band), so on/off there is pixel-identical. `FLORA_LEAFOPT=0|1`
    /// seeds it for headless captures.
    leaf_opt: bool,
    /// ponytail: HEADLESS SCREENSHOT mode. `Some(path)` when `FLORA_SCREENSHOT` is
    /// set → hide the egui panel, render a few warm-up frames, then capture the
    /// final composited frame into a PNG at `path` and exit 0. `frame_count` ticks
    /// each rendered frame so we know when the scene has settled (camera at initial
    /// framing, pipelines/swapchain warm).
    screenshot: Option<String>,
    frame_count: u32,
    /// Set once the screenshot PNG has been written → `about_to_wait` shuts down
    /// and exits the event loop (exit 0).
    screenshot_done: bool,
}

/// ponytail: warm-up frames before the capture (per the brief: ~45) — enough for
/// the swapchain/pipelines to be warm. The camera is the fixed auto-fit orbit pose
/// (no input fed in headless mode), so framing is deterministic front-on.
const SCREENSHOT_WARMUP_FRAMES: u32 = 45;

impl App {
    fn new() -> Self {
        // The viewer's startup tree is the CANONICAL default — the SAME [`TreeSpec`]
        // the in-planet app builds from — so the viewer and the planet agree. The
        // FLORA_* env vars below only OVERRIDE this canonical base for headless
        // capture; the interactive defaults all come from the spec.
        let spec = TreeSpec::default_specimen();
        let env = spec.env;
        // ponytail: FLORA_SEED (when set) picks a varied specimen via the SAME
        // random_genome(env, seed) path the panel/Space use, so a captured seed
        // matches the interactive view. With NO FLORA_SEED we start from the
        // canonical spec genome (dryad's curated TREE_DEFAULT) rather than
        // random_genome(0) — whose raw `pigment` draw rolls a random hue/density.
        let seed_env = std::env::var("FLORA_SEED")
            .ok()
            .and_then(|s| s.trim().parse::<u32>().ok());
        let seed = seed_env.unwrap_or(0);
        let genome = match seed_env {
            Some(s) => random_genome(&env, s),
            None => spec.genome,
        };
        // Screenshot mode: FLORA_SCREENSHOT=<output path> → headless capture + exit.
        let screenshot = std::env::var("FLORA_SCREENSHOT")
            .ok()
            .filter(|p| !p.trim().is_empty());
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
            _ => RenderMode::Lit,
        };
        // FLORA_WIND override (parsed once) seeds the initial wind strength/dir/clock.
        let wo = wind_override();
        App {
            window: None,
            rhi: None,
            flora_renderer: None,
            flora: None,
            staging: None,
            egui: None,
            // Real framing is set by the first build_tree(); this is a placeholder.
            orbit: OrbitCamera {
                target: Vec3::ZERO,
                distance: 30.0,
                yaw: 0.0,
                pitch: 18_f32.to_radians(),
            },
            lmb_down: false,
            rmb_down: false,
            mmb_down: false,
            last_cursor: None,
            gizmo_hover: false,
            gizmo_grabbed: false,
            cursor_pos: None,
            seed,
            genome,
            env,
            mode,
            // Initial leaf mode = the canonical spec default (FLORA_LEAFMODE
            // overrides it for headless capture). Tracks TreeSpec, not a hardcode.
            leaf_mode: LeafMode::from_env_or(spec.leaf_mode),
            // Render tuning from the canonical spec; FLORA_LEAFDENSITY overrides the
            // density for headless capture (slider still drives it interactively).
            leaf_tuning: {
                let mut t = spec.leaf_tuning;
                t.density = LeafTuning::density_from_env(t.density);
                t
            },
            tab: Tab::Climate,
            last_frame: Instant::now(),
            fps: FpsMeter::new(),
            minimized: false,
            last_stats: (0, 0, 0, 0),
            // Wind ON by default at a gentle strength (strength 0 = exact static).
            // FLORA_WIND=strength,dirx,dirz[,time] overrides for headless capture.
            wind_on: true,
            wind_strength: wo.map(|w| w.0).unwrap_or(0.6),
            wind_dir: wo
                .map(|w| Vec3::new(w.1, 0.0, w.2))
                .unwrap_or(Vec3::new(1.0, 0.0, 0.0)),
            wind_clock: wo.map(|w| w.3).unwrap_or(0.0),
            // Wind-direction gizmo ON by default (per the brief) so the set wind
            // direction is always visible; the FLORA_WIND capture shows it too.
            gizmo_on: true,
            // Bloom ON at dryad's default strength (0 / OFF = exact parity).
            bloom_on: true,
            bloom_strength: 0.15,
            // Dappled OFF by default (byte-identical to the parity look). Set
            // FLORA_DAPPLED=1 to seed it on for captures.
            dappled: std::env::var("FLORA_DAPPLED")
                .map(|v| v != "0")
                .unwrap_or(false),
            // Leaf optimization ON by default; FLORA_LEAFOPT=0 forces the reference
            // (full per-card) path for A/B capture.
            leaf_opt: std::env::var("FLORA_LEAFOPT")
                .map(|v| v != "0")
                .unwrap_or(true),
            screenshot,
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
        match FloraView::from_genome_with_mode_tuned(
            rhi,
            &self.genome,
            &self.env,
            DVec3::ZERO,
            self.leaf_mode,
            self.leaf_tuning,
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
                    // Bake the TOP/SIDE impostor atlas from this tree (leaf textures
                    // are now SHADER_READ + IBL ready + GPU idle — rebuild wait_idles).
                    if let Err(e) = renderer.bake_impostor(rhi, &f) {
                        log::error!("FloraRenderer::bake_impostor failed: {e}");
                    }
                }
                let fov_y = 60_f32.to_radians();
                let (min, max) = f.local_bounds();
                let fit = Fit::from_bounds(min, max, fov_y);
                // Re-frame the orbit camera on each build/reseed (simplest: reset
                // to the auto-fit pose). An optional FLORA_CAM=yaw,pitch,dist (degs
                // + world units) overrides the starting pose for headless capture.
                self.orbit = OrbitCamera::from_fit(fit);
                if let Some((yaw, pitch, dist)) = cam_override() {
                    self.orbit.yaw = yaw.to_radians();
                    self.orbit.pitch = (pitch.to_radians()).clamp(-PITCH_LIMIT, PITCH_LIMIT);
                    self.orbit.distance = dist.max(0.05);
                }
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

                self.flora = Some(f);
                log::info!(
                    "Tree seed {} — fit center {:?} dist {:.1}m",
                    self.seed, fit.center, fit.distance
                );
            }
            Err(e) => log::error!("FloraView::from_genome_with_mode_tuned failed: {e}"),
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

    /// The view camera. Interactive runs use the orbit controller (LMB orbit,
    /// wheel zoom, RMB/MMB pan — all driven from `window_event`). Headless
    /// screenshot mode renders the orbit pose too, but since `build_tree` resets
    /// it to the auto-fit pose (optionally `FLORA_CAM`-overridden) and no input is
    /// fed, captures keep the deterministic auto-fit framing.
    fn camera(&self) -> Camera {
        self.orbit.camera()
    }

    /// Current viewport size in pixels (0,0 if the RHI isn't up yet).
    fn viewport(&self) -> (f32, f32) {
        self.rhi
            .as_ref()
            .map(|r| {
                let e = r.extent();
                (e.width as f32, e.height as f32)
            })
            .unwrap_or((0.0, 0.0))
    }

    /// Cast a ray from the orbit eye through the cursor pixel `(px, py)` onto the
    /// ground plane y=0 (tree base = world origin). Returns the world hit (x,_,z)
    /// or None if the viewport is degenerate or the ray is parallel/behind.
    ///
    /// Convention-free (Option B from recon): builds the ray straight from the
    /// orbit camera BASIS + fov, so it sidesteps the reversed-Z projection that
    /// would trip up a generic unproject. `forward` = eye→target, +Y up.
    fn cursor_ground_hit(&self, px: f32, py: f32) -> Option<Vec3> {
        let (w, h) = self.viewport();
        if w <= 0.0 || h <= 0.0 {
            return None;
        }
        let (right, up, forward) = self.orbit.basis();
        let cam = self.orbit.camera();
        let eye = cam.position.as_vec3();
        // Pixel → NDC (winit y is top-down; flip so screen-up maps to +up).
        let ndc_x = 2.0 * px / w - 1.0;
        let ndc_y = 1.0 - 2.0 * py / h;
        let ty = (cam.fov_y_radians * 0.5).tan();
        let tx = ty * (w / h);
        let dir = (forward + right * (ndc_x * tx) + up * (ndc_y * ty)).normalize_or(forward);
        if dir.y.abs() < 1e-6 {
            return None;
        }
        let t = -eye.y / dir.y;
        if t <= 0.0 {
            return None;
        }
        Some(eye + dir * t)
    }

    /// True if the ground point under cursor pixel `(px,py)` lands on the arrow's
    /// flat footprint (a fattened segment from the origin along `wind_dir`). Uses
    /// the SAME constants the arrow mesh is built from (staging.rs) plus a small
    /// grab margin. Robust at any zoom/angle (no screen-space projection).
    fn gizmo_hit(&self, px: f32, py: f32) -> bool {
        let Some(hit) = self.cursor_ground_hit(px, py) else {
            return false;
        };
        let dir = Vec3::new(self.wind_dir.x, 0.0, self.wind_dir.z).normalize_or(Vec3::X);
        let dir2 = glam::Vec2::new(dir.x, dir.z);
        // World arrow length = (shaft + head) * per-frame strength scale.
        let len_scale = 0.35 + self.wind_strength.clamp(0.0, 1.5);
        let l = enki_app::staging::ARROW_TOTAL_LEN * len_scale;
        let p = glam::Vec2::new(hit.x, hit.z);
        let along = p.dot(dir2);
        let perp = (p - dir2 * along).length();
        along >= -0.5 && along <= l + 0.5 && perp <= enki_app::staging::ARROW_GRAB_HALF
    }

    /// Drive the wind direction from a gizmo drag: ray-cast the cursor to y=0 and
    /// set `wind_dir = normalize((groundHit - base).xz)` (base = origin), so the
    /// arrow points from the tree base toward the cursor's ground point. Because
    /// the panel's "dir x"/"dir z" sliders bind directly to `wind_dir.x/.z` and are
    /// re-read every frame, they stay synced automatically (no extra plumbing).
    ///
    /// Optional secondary mapping (direction stays primary): the cursor's DISTANCE
    /// from the base scales strength, clamped to the slider's 0..=1.5 range — so
    /// dragging the arrow further out also strengthens the wind. strength=0 is only
    /// reachable when the cursor is at the base, keeping a hard-static tree possible.
    fn apply_gizmo_drag(&mut self, px: f32, py: f32) {
        let Some(hit) = self.cursor_ground_hit(px, py) else {
            return;
        };
        let xz = Vec3::new(hit.x, 0.0, hit.z);
        // DIRECTION (primary): normalize toward the ground point; keep the old dir
        // if the cursor is right on the base (degenerate), so the arrow never snaps.
        let dir = xz.normalize_or(Vec3::new(self.wind_dir.x, 0.0, self.wind_dir.z));
        if dir.length_squared() > 0.0 {
            self.wind_dir = Vec3::new(dir.x, 0.0, dir.z);
        }
        // STRENGTH (secondary): map ground distance → strength. The full arrow
        // length at strength 1.0 is ARROW_TOTAL_LEN; use it as the unit scale so a
        // drag of ~one arrow-length ≈ strength 1.0. Clamped to the slider range.
        let dist = xz.length();
        self.wind_strength =
            (dist / enki_app::staging::ARROW_TOTAL_LEN).clamp(0.0, 1.5);
    }

    fn render(&mut self) {
        if self.minimized {
            return;
        }

        // Frame timing. Raw `dt` drives motion/wind; the FPS readout is fed through
        // the throttled FpsMeter (≈2 Hz avg) so it doesn't flicker every frame —
        // same as the main `enki` app.
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32();
        self.last_frame = now;
        self.fps.tick(dt);
        let ft = self.fps.frame_time_s();
        let (fps_val, frame_ms_val) = if ft > 0.0 {
            (1.0 / ft, ft * 1000.0)
        } else if dt > 0.0 {
            (1.0 / dt, dt * 1000.0) // pre-first-window fallback
        } else {
            (0.0, 0.0)
        };

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
        // Leaf-optimization LIVE readout so the LOD is observable (it's a no-op at
        // close framing by design → looks like "nothing" without this). Recompute
        // the same LeafLod the render uses, from the orbit camera.
        let (leaf_cards_total, leaf_lod_now) = match self.flora.as_ref() {
            Some(f) => (
                f.leaf_card_count(),
                f.leaf_lod(self.orbit.camera().position, self.leaf_opt),
            ),
            None => (0, LeafLod::FULL),
        };
        let leaf_cards_drawn =
            (leaf_lod_now.density_frac.clamp(0.0, 1.0) * leaf_cards_total as f32).round() as u32;
        let stats = Stats {
            fps: fps_val,
            frame_ms: frame_ms_val,
            triangles: self.last_stats.0,
            draw_calls: self.last_stats.3,
            leaves: self.last_stats.1,
            bones: self.last_stats.2,
            resolution,
            leaf_cards_drawn,
            leaf_cards_total,
            leaf_density: leaf_lod_now.density_frac,
            leaf_impostor: leaf_lod_now.impostor_blend,
            leaf_shadows: leaf_lod_now.cast_leaf_shadows,
        };

        // ── Build the egui panel BEFORE begin_frame (texture upload one-shots
        //    a submit; must be outside the rendering instance). Split-borrow
        //    egui/window/rhi/genome/env — all disjoint App fields. ──
        // ponytail: in screenshot mode SKIP the panel so the captured PNG is a
        // clean render (no UI chrome).
        let build_panel = self.screenshot.is_none();
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
                &mut self.leaf_mode,
                &mut self.leaf_tuning,
                &mut self.seed,
                &mut self.tab,
                &mut self.wind_on,
                &mut self.wind_strength,
                &mut self.wind_dir,
                &mut self.gizmo_on,
                &mut self.bloom_on,
                &mut self.bloom_strength,
                &mut self.dappled,
                &mut self.leaf_opt,
                self.flora.as_ref(),
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

        // Distance-driven leaf LOD for this frame (the "Leaf optimization" toggle).
        // Computed once and shared by the shadow caster + lit pass so the cast
        // silhouette tracks the drawn cards. FULL when there's no tree / toggle off.
        let leaf_lod = self
            .flora
            .as_ref()
            .map(|f| f.leaf_lod(camera_world_pos, self.leaf_opt))
            .unwrap_or(LeafLod::FULL);

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
                    // (1/size, normalBias 0.02 [dryad], enabled, dappled)
                    params: [texel, 0.02, 1.0, if self.dappled { 1.0 } else { 0.0 }],
                    // Viewer = single tree-local cascade: count=1, use_view_pos=0
                    // (sample tree-local shadow_pos), dryad depth bias 0.0005.
                    light_view_proj1: Mat4::IDENTITY.to_cols_array_2d(),
                    light_view_proj2: Mat4::IDENTITY.to_cols_array_2d(),
                    params2: [1.0, 0.0, 0.0005, 0.0],
                };
                (su, Some(lvp))
            }
            None => (
                ShadowUniforms {
                    light_view_proj: Mat4::IDENTITY.to_cols_array_2d(),
                    params: [1.0 / 2048.0, 0.02, 0.0, 0.0], // disabled
                    light_view_proj1: Mat4::IDENTITY.to_cols_array_2d(),
                    light_view_proj2: Mat4::IDENTITY.to_cols_array_2d(),
                    params2: [1.0, 0.0, 0.0005, 0.0],
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
            if let Err(e) = flora.record_shadow(rhi, renderer, fi, lvp, wind, leaf_lod) {
                log::error!("FloraView::record_shadow error: {e}");
            }
            renderer.end_shadow_pass(rhi, fi);
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
        // ── Tree: branches + leaves into the scene pass (flora's full render path:
        //    textured PBR bark + IBL + PCF shadows + green leaf cards). ──
        if let (Some(renderer), Some(flora)) =
            (self.flora_renderer.as_ref(), self.flora.as_mut())
        {
            // Same `wind` vec4 the shadow caster used → cast silhouette matches.
            if let Err(e) = flora.record(rhi, renderer, fi, camera_world_pos, wind, leaf_lod) {
                log::error!("FloraView::record error: {e}");
            }
        }
        // ── LEAF IMPOSTOR far-tier (#3): a camera-facing billboard that crossfades
        //    in as the cards thin out with distance (LeafLod::impostor_blend).
        //    Alpha-blended, drawn AFTER the tree so it composites over the (now
        //    sparse) canopy. Camera basis from the orbit so the quad faces us. ──
        if leaf_lod.impostor_blend > 0.0 {
            if let (Some(renderer), Some(flora)) =
                (self.flora_renderer.as_ref(), self.flora.as_ref())
            {
                let (cam_right, cam_up, _) = self.orbit.basis();
                if let Err(e) = flora.record_impostor(
                    rhi, renderer, fi, camera_world_pos, cam_right, cam_up, leaf_lod.impostor_blend,
                ) {
                    log::error!("FloraView::record_impostor error: {e}");
                }
            }
        }
        // ── WIND GIZMO: a flora-owned unlit arrow at the tree base pointing along
        //    `wind_dir`, length/color scaled by the wind strength. Drawn AFTER the
        //    tree so it depth-tests against ground+tree; gated on the panel toggle
        //    (default ON). It reads the SAME `wind_dir` the solver/vertex push use,
        //    so the arrow always points where the canopy leans, and updates live as
        //    the dir x/z + strength sliders move. Uses the panel strength (not the
        //    `wind_on`-gated `wind_strength`) so the arrow still shows direction
        //    when Wind is toggled off. ──
        if self.gizmo_on {
            if let (Some(renderer), Some(staging)) =
                (self.flora_renderer.as_ref(), self.staging.as_mut())
            {
                if let Err(e) = staging.record_arrow(
                    rhi,
                    renderer,
                    fi,
                    camera_world_pos,
                    self.wind_dir,
                    self.wind_strength,
                    self.gizmo_hover || self.gizmo_grabbed,
                ) {
                    log::error!("Staging::record_arrow error: {e}");
                }
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
            // Open the capture pass (draws the fs_output composite into it), then
            // close + copy the captured frame into the host-visible readback buffer.
            if let Some(renderer) = self.flora_renderer.as_mut() {
                if let Err(e) = renderer.begin_capture(rhi, fi) {
                    log::error!("flora begin_capture error: {e}");
                }
                if let Err(e) = renderer.end_capture(rhi, fi) {
                    log::error!("flora end_capture error: {e}");
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
        // in_scene = false: the viewer keeps its own offscreen HDR + bloom pass.
        match FloraRenderer::new(&mut rhi, false) {
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
        let wants_kb = self
            .egui
            .as_ref()
            .map(|e| e.ctx.wants_keyboard_input())
            .unwrap_or(false);
        // egui pointer gate for nav: true when the cursor is over a panel. Used
        // for the orbit mouse arms so dragging on a panel never moves the camera.
        let wants_ptr = self
            .egui
            .as_ref()
            .map(|e| e.ctx.wants_pointer_input() || e.ctx.is_pointer_over_area())
            .unwrap_or(false);

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
            // ── Orbit controller mouse input (only when egui isn't using the
            //    pointer). Button-down latches drag state; CursorMoved applies the
            //    delta (LMB→orbit, RMB/MMB→pan); wheel zooms. egui consumes pointer
            //    events first (event-first philosophy), so panels stay interactive.
            WindowEvent::MouseInput { state, button, .. } => {
                let down = state == ElementState::Pressed;
                // Only START a drag when egui isn't using the pointer (so a click
                // on a panel doesn't grab the camera). ALWAYS honor a release,
                // even over a panel, so a button can't stay latched down.
                if down && (egui_consumed || wants_ptr) {
                    // press swallowed by egui — ignore for nav
                } else {
                    match button {
                        MouseButton::Left => {
                            if down {
                                // GIZMO PRIORITY: decide a grab BEFORE orbit latches.
                                // If the LMB press lands on the arrow footprint, grab
                                // the gizmo and DON'T set `lmb_down` — that structurally
                                // suppresses orbit (its CursorMoved dispatch keys off
                                // `lmb_down`). Otherwise fall through to a normal orbit.
                                let over = self.gizmo_on
                                    && self
                                        .cursor_pos
                                        .map(|(x, y)| self.gizmo_hit(x, y))
                                        .unwrap_or(false);
                                if over {
                                    self.gizmo_grabbed = true;
                                    // Apply the direction immediately on press so a
                                    // click (no move) still snaps toward the cursor.
                                    if let Some((x, y)) = self.cursor_pos {
                                        self.apply_gizmo_drag(x, y);
                                    }
                                } else {
                                    self.lmb_down = true;
                                }
                            } else {
                                // Release ALWAYS clears both, so neither can stick.
                                self.lmb_down = false;
                                self.gizmo_grabbed = false;
                            }
                        }
                        MouseButton::Right => self.rmb_down = down,
                        MouseButton::Middle => self.mmb_down = down,
                        _ => {}
                    }
                }
                if !self.lmb_down && !self.rmb_down && !self.mmb_down && !self.gizmo_grabbed {
                    self.last_cursor = None;
                }
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            // Cursor motion: (a) GRABBED gizmo → ray-cast to y=0 and steer wind_dir;
            // (b) a latched button → orbit/pan as before; (c) no button → just track
            // position + update the hover highlight. The gizmo branch comes FIRST so
            // a grab beats orbit; orbit can't fire during a grab anyway since the
            // grab leaves `lmb_down` false.
            WindowEvent::CursorMoved { position, .. } => {
                let p = (position.x as f32, position.y as f32);
                self.cursor_pos = Some(p);
                if self.gizmo_grabbed {
                    // Live re-orient: arrow + sliders + canopy all follow this.
                    self.apply_gizmo_drag(p.0, p.1);
                    self.last_cursor = Some(p);
                    if let Some(w) = &self.window {
                        w.request_redraw();
                    }
                } else if self.lmb_down || self.rmb_down || self.mmb_down {
                    if let Some((lx, ly)) = self.last_cursor {
                        let (dx, dy) = (p.0 - lx, p.1 - ly);
                        if self.lmb_down {
                            self.orbit.orbit(dx, dy);
                        } else if self.rmb_down || self.mmb_down {
                            self.orbit.pan(dx, dy);
                        }
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                    }
                    self.last_cursor = Some(p);
                } else {
                    // No drag: refresh the hover highlight (only when egui isn't
                    // using the pointer, so hovering a panel doesn't light the arrow).
                    let hover = self.gizmo_on
                        && !wants_ptr
                        && !egui_consumed
                        && self.gizmo_hit(p.0, p.1);
                    if hover != self.gizmo_hover {
                        self.gizmo_hover = hover;
                        if let Some(w) = &self.window {
                            w.request_redraw();
                        }
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed && !wants_ptr => {
                let s = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 * 0.01,
                };
                self.orbit.zoom(s);
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
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

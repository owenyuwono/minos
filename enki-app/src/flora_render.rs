// flora_render.rs — flora-OWNED Vulkan sub-renderer (behind `--features flora`).
//
// ponytail: this is the SMALLEST self-contained sub-renderer that re-renders the
// CURRENT tree+ground+sky through pipelines flora OWNS, mirroring the
// `egui-ash-renderer` precedent (gui.rs): it manages its OWN VkShaderModules,
// VkDescriptorSetLayouts, VkPipelineLayout, VkPipelines, descriptor pool, and a
// per-frame uniform ring, using ONLY raw handles from enki-rhi
// (device_handle/instance_handle/physical_device/swapchain_format/msaa_samples/
// current_command_buffer/...). It records into the viewer's begin_rendering
// instance via the raw command buffer — exactly like egui's `cmd_draw`.
//
// WHY this exists (the enabler for shadows/IBL/bloom/wind): enki-rhi's
// `create_graphics_pipeline` HARDCODES the pipeline layout to a SINGLE set
// (set0 = FrameUniforms; pipeline.rs:132) and `bind_pipeline` binds only set0.
// That blocks flora pipelines from sampling a shadow map / IBL cubes or reading
// bone buffers. This module builds its OWN VkPipelineLayout = [set0, set1] so
// set1 is AVAILABLE for those resources. THIS STAGE only proves visual parity:
// set1 is created (a shadow-map sampler slot) but bound to nothing and unused by
// the shaders yet. The next stages fill set1 + add the shadow pass.
//
// DELETABLE-FLORA CONTRACT: every Vulkan object here is owned by `FloraRenderer`
// and freed in `destroy()` (called by the viewer before Rhi teardown). Deleting
// flora = dropping this file + flora_view.rs + staging.rs + the
// `#[cfg(feature="flora")]` wiring. enki-rhi's terrain/nanite path is untouched.
//
// The pipeline state below is a BYTE-FOR-BYTE copy of enki's main-pass pipeline
// (pipeline.rs): reversed-Z depth GREATER, depth clear 0, D32_SFLOAT depth,
// color = swapchain format, MSAA = rhi.msaa_samples(), the fixed 4×vec3 vertex
// layout, cull BACK / front-face CCW, dynamic viewport+scissor — so the tree
// renders identically to today, just through flora-owned objects.

use std::ffi::CString;

use ash::vk;
use enki_render::frame::FrameUniforms;
use enki_rhi::{Rhi, RhiError};
use naga::{
    back::spv,
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};

const FLORA_WGSL: &str = include_str!("../../enki-flora/shaders/flora.wgsl");
const STAGING_WGSL: &str = include_str!("../../enki-flora/shaders/staging.wgsl");
const POST_WGSL: &str = include_str!("../../enki-flora/shaders/post.wgsl");

/// Square shadow-map resolution. Bumped from dryad's 2048 to 4096 so the gaps
/// between the ~thousands of small leaf cutouts resolve into discrete sun-spots
/// for the DAPPLED self-shadow mode (a finer map = each PCF tap covers half the
/// world-space radius → sharper, more-resolved canopy gaps). It is one tree, so
/// a 4096² D32 map (64 MiB) + its depth pass is trivial here; we keep it always
/// high-res rather than recreating the image on toggle (the dappled toggle is a
/// pure per-frame uniform-lane write — see `ShadowUniforms.params.w`).
pub const SHADOW_MAP_SIZE: u32 = 4096;

/// The LINEAR-HDR offscreen scene format (dryad's composer targets HalfFloatType).
const HDR_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

/// IMPOSTOR ATLAS — one RGBA16F image holding TWO square tiles side-by-side:
///   SIDE = u ∈ [0, TILE) (horizon-level view), TOP = u ∈ [TILE, 2·TILE) (top-down).
/// `fs_impostor` blends them by view elevation. Baked ONCE at startup from the real
/// tree (linear HDR, alpha = silhouette coverage). One tile is plenty at impostor
/// distance (a tree is a few px); 512² keeps a crisp silhouette without VRAM cost.
const IMPOSTOR_TILE: u32 = 512;
/// Full atlas extent = 2 tiles wide × 1 tile tall (1024 × 512).
const IMPOSTOR_ATLAS_W: u32 = IMPOSTOR_TILE * 2;
const IMPOSTOR_ATLAS_H: u32 = IMPOSTOR_TILE;

/// UnrealBloom mip count (nMips = 5). The bloom chain starts at half-res and
/// halves each mip: res/2, res/4, res/8, res/16, res/32.
const BLOOM_MIPS: usize = 5;

/// Per-mip separable-gaussian kernel radii (UnrealBloomPass kernelSizeArray).
const BLOOM_KERNELS: [f32; BLOOM_MIPS] = [3.0, 5.0, 7.0, 9.0, 11.0];

/// 96-byte push for the post passes (mirrors `PostPush` in post.wgsl: 2×vec4).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct PostPush {
    params: [f32; 4],
    params2: [f32; 4],
}

/// set1 / binding 2 — the light matrices + PCF params the main pass reads. Mirrors
/// `ShadowUniforms` in flora.wgsl (full) / staging.wgsl (the first {mat4,vec4}
/// subset). 224 bytes (3×mat4 + 2×vec4).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShadowUniforms {
    /// Cascade 0: world → light clip (the depth-pass matrix). In the viewer this is
    /// the single tree-local self-shadow matrix; on the planet it's CSM cascade 0.
    pub light_view_proj: [[f32; 4]; 4],
    /// (1/SHADOW_MAP_SIZE, normalBias, enabled, dappled).
    /// `.w` (dappled): 0 = today's look (real shadow + the soft canopy fakes);
    /// 1 = let the real high-res shadow dominate (soften the canopy-normal blend,
    /// lower the exposure floor, trim the ambient/IBL fill, extra PCF taps).
    pub params: [f32; 4],
    /// Cascade 1 & 2 (planet 3-cascade CSM; identity/unused in the viewer).
    /// APPENDED after `params` so the viewer-ground `staging.wgsl`, which declares
    /// only `{light_view_proj, params}`, keeps its byte offsets unchanged.
    pub light_view_proj1: [[f32; 4]; 4],
    pub light_view_proj2: [[f32; 4]; 4],
    /// (cascade_count, use_view_pos, depth_bias, _). cascade_count = 1 (viewer) or
    /// 3 (planet); use_view_pos = 1 → the receiver samples camera-relative view_pos.
    pub params2: [f32; 4],
}

/// Which owned pipeline to bind for a draw. Branch/leaf have wireframe twins;
/// staging pipelines (ground/sky/shadow) do not. BranchDepth/LeafDepth are the
/// depth-only sun-shadow casters (bound only inside the shadow pass).
#[derive(Clone, Copy)]
pub enum FloraPipeline {
    Branch,
    BranchWire,
    Leaf,
    LeafWire,
    Ground,
    Sky,
    Shadow,
    /// Wind-direction debug gizmo (unlit solid-color arrow). Uses `ArrowPush`.
    Arrow,
    /// Far-distance leaf billboard (the LOD impostor). Uses `ImpostorPush`,
    /// alpha-blended, drawn in the scene pass after the tree.
    Impostor,
    BranchDepth,
    LeafDepth,
}

/// `BranchDepthPush` mirror (flora.wgsl) — 96B: light matrix + rot quat + wind.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct BranchDepthPush {
    pub light_view_proj: [[f32; 4]; 4],
    pub rot: [f32; 4],
    pub wind: [f32; 4],
}

/// `LeafDepthPush` mirror (flora.wgsl) — 128B: light matrix + rot quat + wind +
/// leaf shape (for the alpha cutout) + skew.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LeafDepthPush {
    pub light_view_proj: [[f32; 4]; 4],
    pub rot: [f32; 4],
    pub wind: [f32; 4],
    pub leaf_params: [f32; 4],
    pub leaf_params2: [f32; 4],
}

/// A flora-owned VkPipeline + its (shared) layout. Kept together so `bind` can
/// record both the pipeline and the push-constant layout off one handle.
struct OwnedPipeline {
    pipeline: vk::Pipeline,
    layout: vk::PipelineLayout,
}

/// A flora-owned color image + its backing memory + a full-chain sampling view +
/// one render (attachment) view per mip. Raw-allocated like the shadow/IBL images
/// (no gpu-allocator dep). Used for the HDR scene resolve and the bloom mips.
struct OwnedImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    /// View spanning ALL mips (for `textureSampleLevel` reads).
    sample_view: vk::ImageView,
    /// One single-mip view per level (color attachment targets for blur passes).
    mip_views: Vec<vk::ImageView>,
}

impl OwnedImage {
    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.sample_view, None);
            for v in &self.mip_views {
                device.destroy_image_view(*v, None);
            }
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
    }
}

/// The swapchain-extent-dependent POST targets + a per-frame post descriptor set
/// ring. Recreated on resize (`recreate`); the pipelines/layout/sampler are
/// resolution-independent and live on `Post` directly.
///
/// ponytail: the scene MSAA color is the ONLY multisample post target; everything
/// downstream (resolve, bloom, composite source) is single-sample, mirroring
/// dryad's HalfFloat composer chain.
struct PostTargets {
    /// Scene HDR MSAA color (RGBA16F, msaa_samples) — the sky/ground/tree target.
    scene_msaa_image: vk::Image,
    scene_msaa_memory: vk::DeviceMemory,
    scene_msaa_view: vk::ImageView,
    /// Scene depth (D32, msaa_samples) — flora now owns the scene depth buffer.
    scene_depth_image: vk::Image,
    scene_depth_memory: vk::DeviceMemory,
    scene_depth_view: vk::ImageView,
    /// Resolved single-sample HDR (RGBA16F, SAMPLED) — bright-pass + output read it.
    scene_resolve: OwnedImage,
    /// Bright-pass output (half-res, 1 mip).
    bloom_bright: OwnedImage,
    /// Horizontal-blur mip chain (5 mips, half-res base).
    bloom_h: OwnedImage,
    /// Vertical-blur mip chain (5 mips, half-res base) — composite samples its LODs.
    bloom_v: OwnedImage,
    /// Composite (combined bloom) output, full-res 1 mip — output pass reads it.
    bloom_composite: OwnedImage,
    extent: vk::Extent2D,
    /// First-frame layout init guard for the scene MSAA/depth images.
    initialized: bool,
}

/// The flora-OWNED POST pipeline (dryad RenderPass→UnrealBloom→OutputPass). Holds
/// the resolution-independent pipelines/layout/sampler/descriptor pool, plus the
/// resize-dependent `targets`. All single-set (set0 = 2 textures + 2 samplers).
struct Post {
    module: vk::ShaderModule,
    /// set0: src tex (b0) + src sampler (b1) + aux tex (b2) + aux sampler (b3).
    set_layout: vk::DescriptorSetLayout,
    /// Layout = [set0] + a 32-byte (2×vec4) FS|VS push range.
    pipeline_layout: vk::PipelineLayout,
    /// Linear CLAMP sampler used by every post tap.
    sampler: vk::Sampler,
    bright: OwnedPipeline,
    blur: OwnedPipeline,
    composite: OwnedPipeline,
    /// Output pass — MSAA sample count + swapchain format (runs in enki's instance).
    output: OwnedPipeline,
    /// A generously-sized transient-set pool PER frame-in-flight. We allocate
    /// fresh sets each frame and RESET this frame's OWN pool at frame start, so
    /// one pool serves all bloom mips + composite + output for that frame without
    /// racing the other FIF's in-flight submission (which the frame-acquire fence
    /// guarantees has completed before this frame's pool is reset/re-recorded).
    pools: Vec<vk::DescriptorPool>,
    targets: PostTargets,
}

/// ponytail: HEADLESS SCREENSHOT capture target (only allocated when
/// `FLORA_SCREENSHOT` is set). A flora-owned single-sample B8G8R8A8_SRGB image
/// (COLOR_ATTACHMENT | TRANSFER_SRC) that the `fs_output` pass renders into —
/// byte-identical to the swapchain (same _SRGB format → hardware sRGB OETF on
/// store, same fs_output shader), so we never need the RHI's private swapchain
/// image handle. After the draw it's copied into `readback` (host-visible,
/// TRANSFER_DST) via vkCmdCopyImageToBuffer; the viewer reads it after wait_idle.
struct Capture {
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
    /// Host-visible TRANSFER_DST readback buffer (extent.w*h*4 bytes, BGRA).
    readback: enki_rhi::BufferHandle,
    /// SINGLE-SAMPLE `fs_output` pipeline (B8G8R8A8_SRGB, no depth) for the capture
    /// instance. The shared `post.output` is built at MSAA `samples` (it runs in
    /// enki's MSAA swapchain instance) — using it on a 1-sample capture image is a
    /// sample-count mismatch, so the capture owns its own 1-sample twin.
    pipeline: vk::Pipeline,
    extent: vk::Extent2D,
}

/// The flora-owned sub-renderer. Mirrors `egui_ash_renderer::Renderer`: owns all
/// its Vulkan objects, records into a caller-supplied command buffer.
pub struct FloraRenderer {
    device: ash::Device,

    /// True when this renderer draws INTO the app's main 3D pass (swapchain
    /// format, ACES applied in `fs_branch`/`fs_leaf`) rather than its own
    /// offscreen HDR + bloom pass. Read by `FloraView` to set the tonemap lane.
    in_scene: bool,

    /// ponytail: present only in headless screenshot mode (`FLORA_SCREENSHOT`).
    /// Lazily built on the first capture frame, sized to the swapchain extent.
    capture: Option<Capture>,

    // Shader modules (one per WGSL source; multiple entry points each).
    flora_module: vk::ShaderModule,
    staging_module: vk::ShaderModule,

    // Descriptor set layouts.
    //   set0 = FrameUniforms UBO (binding 0, UNIFORM_BUFFER, VERTEX|FRAGMENT) —
    //          byte-identical to enki's set0 so the WGSL `@group(0)@binding(0)`
    //          matches. Flora owns its OWN copy (not rhi.set0_layout()) to stay
    //          self-contained / deletable, exactly like egui owns its own.
    //   set1 (the shadow set: SAMPLED depth + comparison sampler + ShadowUniforms
    //          UBO) is declared further down alongside the shadow resources.
    set0_layout: vk::DescriptorSetLayout,

    // Pipeline layout = [set0, set1] + a 128-byte VS|FS push range. This is the
    // object enki-rhi's create_graphics_pipeline CANNOT build (it hardcodes one
    // set). Both branch (BranchPush) and leaf (LeafPush) pushes are ≤128B, and
    // staging's StagePush is 64B, so ONE 128-byte layout serves every pipeline.
    pipeline_layout: vk::PipelineLayout,

    // The owned graphics pipelines (main pass + depth casters).
    branch: OwnedPipeline,
    branch_wire: OwnedPipeline,
    leaf: OwnedPipeline,
    leaf_wire: OwnedPipeline,
    ground: OwnedPipeline,
    sky: OwnedPipeline,
    shadow: OwnedPipeline,
    arrow: OwnedPipeline,
    impostor: OwnedPipeline,
    branch_depth: OwnedPipeline,
    leaf_depth: OwnedPipeline,

    // A separate pipeline layout for the depth casters: [set0, set1] (so the
    // bind point matches) but the casters only use push constants. We reuse the
    // SAME layout as the main pass (its 128B push range covers LeafDepthPush).
    // (No extra field — depth pipelines reference `pipeline_layout`.)

    // Descriptor pool for set0 (per-frame UBO sets) + the shadow set1.
    pool: vk::DescriptorPool,

    // Per-frame uniform ring: one host-visible UBO + one set0 descriptor per
    // frame-in-flight, so writing this frame's uniforms never races the GPU
    // reading last frame's. Mirrors enki's own descriptor ring.
    ubos: Vec<enki_rhi::BufferHandle>,
    set0: Vec<vk::DescriptorSet>,

    // ── SUN SHADOWS ──────────────────────────────────────────────────────────
    // The flora-OWNED depth shadow map (D32, SHADOW_MAP_SIZE²), allocated once on
    // the raw device with raw vkAllocateMemory (no gpu-allocator dep). Sampled by
    // the main pass via set1; rendered into by the depth casters in the shadow
    // pass. // ponytail: ONE persistent image → a single raw vkAllocateMemory is
    // shorter than pulling the gpu-allocator crate into enki-app for one alloc.
    shadow_image: vk::Image,
    shadow_memory: vk::DeviceMemory,
    shadow_view: vk::ImageView,
    /// Comparison sampler (reversed-Z: GREATER_OR_EQUAL), CLAMP_TO_EDGE.
    shadow_sampler: vk::Sampler,
    /// Has the shadow image been transitioned out of UNDEFINED yet? Tracks the
    /// first-frame layout init so we barrier UNDEFINED→ATTACHMENT only once.
    shadow_initialized: bool,

    // set1: shadow map (binding0 SAMPLED_IMAGE depth) + comparison sampler
    // (binding1) + ShadowUniforms UBO ring (binding2). One descriptor per frame
    // so the light matrix can change per frame without racing the GPU.
    //
    // set1 ALSO carries the IBL resources (bindings 3/4/5) — extended here rather
    // than as a new set because the pipeline layout already passes set1 and the
    // WGSL @group(1) is free to grow. The env texture + SH UBO are STATIC per
    // HDRI (baked once at `new()`), so they are SHARED across every frame's set1
    // (one image, one sampler, one SH UBO — not a per-frame ring like the shadow
    // UBO). // ponytail: static resources need no ring; writing them into all
    // `frames` set1 copies keeps the single write_*_descriptors path.
    set1_layout: vk::DescriptorSetLayout,
    set1: Vec<vk::DescriptorSet>,
    shadow_ubos: Vec<enki_rhi::BufferHandle>,

    // ── HIERARCHICAL WIND bone matrices (set1/binding 6) ─────────────────────
    // One host-visible STORAGE buffer per frame-in-flight, sized MAX_WIND_BONES
    // mat4s (1024 × 64 B = 64 KiB — over the UBO guaranteed-minimum, so a STORAGE
    // buffer, not a UBO). The wind solver writes the per-bone world matrices into
    // this frame's slot each frame (before the shadow + scene passes); both the
    // depth caster (VS) and the main branch/leaf VS read it via set1. // ponytail:
    // mirrors the shadow_ubos per-frame ring exactly — same FIF-safety argument.
    bone_ubos: Vec<enki_rhi::BufferHandle>,

    // ── IBL (image-based lighting) — flora-owned, baked from dryad's equirect
    //    HDRI on the CPU (flora_ibl), uploaded once. The equirect MIP CHAIN is one
    //    RGBA16F 2D image (roughness→LOD specular); the SH9 diffuse irradiance is
    //    one UBO. Both raw-allocated like the shadow image (no gpu-allocator dep).
    ibl_image: vk::Image,
    ibl_memory: vk::DeviceMemory,
    ibl_view: vk::ImageView,
    /// LINEAR trilinear sampler (REPEAT u / CLAMP v) for the equirect mip lookup.
    ibl_sampler: vk::Sampler,

    // ── LEAF CLUSTER TEXTURES (set1/binding 7,8,9) — CPU-baked per genome from
    //    the resolved leaf genes (enki_flora::leaf_texture::bake_leaf_cluster),
    //    uploaded like the IBL image. Color = RGBA8 sRGB (alpha=coverage), normal
    //    = RGBA8 UNORM (tangent-space). Raw-allocated like the shadow/IBL images.
    //    REBUILT on reseed/genome change via `update_leaf_texture` (the IBL is
    //    static, but leaf genes vary per tree). `None` until the first tree is
    //    built; `fs_leaf` is only ever recorded after a tree exists.
    leaf_color_image: vk::Image,
    leaf_color_memory: vk::DeviceMemory,
    leaf_color_view: vk::ImageView,
    leaf_normal_image: vk::Image,
    leaf_normal_memory: vk::DeviceMemory,
    leaf_normal_view: vk::ImageView,
    /// LINEAR CLAMP sampler shared by the leaf color + normal textures.
    leaf_tex_sampler: vk::Sampler,
    /// Have the leaf textures been baked at least once? Guards the first
    /// `update_leaf_texture` (which must NOT destroy the placeholder twice).
    leaf_tex_built: bool,

    // ── IMPOSTOR ATLAS (set1/binding 12) — the TOP+SIDE billboard texture baked
    //    ONCE from the real tree in `bake_impostor`. RGBA16F linear HDR, alpha =
    //    silhouette coverage. Sampled by `fs_impostor` with the IBL linear sampler
    //    (binding 4). Created (cleared) at `new()`, filled at the first
    //    `bake_impostor`; the VIEW is stable so the per-frame set1 write is
    //    one-time (like the IBL/leaf handles). A dedicated D32 bake-depth (small,
    //    transient-state-free) is kept only for the bake pass.
    impostor_atlas: OwnedImage,
    impostor_depth_image: vk::Image,
    impostor_depth_memory: vk::DeviceMemory,
    impostor_depth_view: vk::ImageView,
    /// Branch/leaf BAKE pipelines (built at the atlas sample count = 1, into
    /// HDR_FORMAT, clear-to-zero alpha). Separate from the scene pipelines because
    /// those are MSAA / may be swapchain-format (in-scene); the bake target is a
    /// single-sample HDR tile.
    impostor_bake_branch: OwnedPipeline,
    impostor_bake_leaf: OwnedPipeline,
    /// Has the atlas been baked yet? Guards the one-shot bake + the UNDEFINED→
    /// SHADER_READ first-transition (the cleared image starts UNDEFINED).
    impostor_atlas_baked: bool,

    // ── POST (dryad RenderPass → UnrealBloomPass → OutputPass) ────────────────
    // The flora-owned offscreen HDR + bloom chain + composite/output pipelines.
    // Raw handles needed to recreate the resize-dependent targets are cached.
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    post: Post,
    // The SH9 coeffs UBO (set1/binding5) is an RHI BufferHandle, created in
    // `new()`, written once, and bound into every frame's set1. It's static (no
    // per-frame update), and RHI-owned (reclaimed at RHI shutdown), so we don't
    // keep its handle on the struct — nothing here touches it after construction.
}

/// set1/binding5 mirror — the 9 RGB SH irradiance coeffs (cosine-convolved +
/// ×environmentIntensity on the CPU) plus the equirect mip max-LOD. 160 bytes.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ShCoeffs {
    /// 9 vec4 (xyz = SH coeff, w pad).
    pub c: [[f32; 4]; 9],
    /// (max_lod, 0, 0, 0) — roughness=1 maps to this LOD in the shader.
    pub params: [f32; 4],
}

impl FloraRenderer {
    /// Build every flora-owned Vulkan object from the RHI's raw handles. Mirrors
    /// `EguiState::new` → `Renderer::with_default_allocator`: pull raw handles,
    /// build our own pipeline/descriptor objects, return a self-contained renderer.
    pub fn new(rhi: &mut Rhi, in_scene: bool) -> Result<Self, RhiError> {
        let device = rhi.device_handle();
        // Viewer (in_scene = false): branch/leaf/staging render into the
        // flora-owned LINEAR-HDR offscreen target (RGBA16F), and the composite
        // OutputPass tonemaps to the swapchain. In-scene (true): they draw straight
        // into the app's 3D pass, so build them at the SWAPCHAIN format and tonemap
        // in-shader (the `aces_filmic` lane in flora.wgsl, driven by FloraView).
        let color_format = if in_scene { rhi.swapchain_format() } else { HDR_FORMAT };
        let samples = rhi.msaa_samples();
        let frames = rhi.frames_in_flight();

        // ── Shaders: compile WGSL → SPIR-V ourselves (like egui ships its own
        //    SPIR-V), then create our OWN VkShaderModules. We can't borrow the
        //    RHI's ShaderModule (its vk handle is pub(crate)), so we own these. ──
        let flora_module = compile_module(&device, FLORA_WGSL)?;
        let staging_module = compile_module(&device, STAGING_WGSL)?;

        // ── Descriptor set layouts (set0 UBO + set1 shadow set). ──
        let set0_layout = create_set0_layout(&device)?;
        let set1_layout = create_set1_layout(&device)?;

        // ── Pipeline layout: [set0, set1] + 128B VS|FS push range. ──
        let pipeline_layout =
            create_pipeline_layout(&device, &[set0_layout, set1_layout])?;

        // ── Pipelines (replicate enki's main-pass state exactly). ──
        let mk = |module, vs, fs, blend, fill, depth_test| {
            build_pipeline(
                &device,
                pipeline_layout,
                module,
                vs,
                fs,
                color_format,
                samples,
                blend,
                fill,
                depth_test,
            )
        };
        // BRANCH uses a 6-stream vertex layout (pos/nrm/uv/attr + aTangent/aFrameU
        // for the seamless bark frame); everything else stays 4-stream. Built via
        // build_pipeline_culled directly (BACK cull, like build_pipeline's default)
        // to pass n_streams = 6.
        let branch = build_pipeline_culled(
            &device, pipeline_layout, flora_module, "vs_branch", "fs_branch",
            color_format, samples, false, true, true, vk::CullModeFlags::BACK, 6,
        )?;
        let branch_wire = build_pipeline_culled(
            &device, pipeline_layout, flora_module, "vs_branch", "fs_branch",
            color_format, samples, false, false, true, vk::CullModeFlags::BACK, 6,
        )?;
        let leaf = mk(flora_module, "vs_leaf", "fs_leaf", false, true, true)?;
        let leaf_wire = mk(flora_module, "vs_leaf", "fs_leaf", false, false, true)?;
        // GROUND: a single y=0 world quad. Build it CULL-NONE (double-sided) so it
        // shows regardless of which way the orbit camera reads its winding — a
        // floor plane should never vanish to back-face culling. // ponytail.
        let ground = build_pipeline_culled(
            &device, pipeline_layout, staging_module, "vs_ground", "fs_ground",
            color_format, samples, false, true, true, vk::CullModeFlags::NONE, 4,
        )?;
        // SKY: depth_test=false (see build_pipeline) — fills the background instead
        // of being depth-rejected by the reversed-Z cleared-to-0 GREATER buffer.
        let sky = mk(staging_module, "vs_sky", "fs_sky", false, true, false)?;
        // Shadow disc is alpha-blended (blend:true → depth-write OFF), same as the
        // old staging shadow pipeline.
        let shadow = mk(staging_module, "vs_shadow", "fs_shadow", true, true, true)?;
        // ARROW gizmo: unlit, OPAQUE (writes depth), CULL-NONE so the thin shaft
        // box reads from any orbit angle. Depth-tested against ground+tree.
        let arrow = build_pipeline_culled(
            &device, pipeline_layout, staging_module, "vs_arrow", "fs_arrow",
            color_format, samples, false, true, true, vk::CullModeFlags::NONE, 4,
        )?;
        // IMPOSTOR: alpha-blended (blend:true → depth-write OFF) camera-facing
        // billboard, CULL-NONE (built CPU-side facing the camera, but a flat quad
        // can present either winding). Depth-TESTED so nearer geometry occludes it.
        let impostor = build_pipeline_culled(
            &device, pipeline_layout, staging_module, "vs_impostor", "fs_impostor",
            color_format, samples, true, true, true, vk::CullModeFlags::NONE, 4,
        )?;

        // ── DEPTH CASTER pipelines: depth-only (no color attachment), single
        //    sample (the shadow map is 1×), reversed-Z GREATER depth-write ON.
        //    Branch caster has no FS (depth-only); leaf caster has an alpha-test
        //    FS (it `discard`s outside the cutout). ──
        let branch_depth = build_depth_pipeline(
            &device,
            pipeline_layout,
            flora_module,
            "vs_branch_depth",
            None,
        )?;
        let leaf_depth = build_depth_pipeline(
            &device,
            pipeline_layout,
            flora_module,
            "vs_leaf_depth",
            Some("fs_leaf_depth"),
        )?;

        // ── Shadow map image (D32, SHADOW_MAP_SIZE²) + comparison sampler. ──
        let (shadow_image, shadow_memory, shadow_view) = create_shadow_image(
            &device,
            rhi.instance_handle(),
            rhi.physical_device(),
        )?;
        let shadow_sampler = create_shadow_sampler(&device)?;

        // ── Descriptor pool: `frames` set0 sets + `frames` set1 sets. ──
        let pool = create_pool(&device, frames as u32)?;

        // ── Per-frame UBO ring + set0 descriptors. ──
        let ubo_size = std::mem::size_of::<FrameUniforms>() as u64;
        let mut ubos = Vec::with_capacity(frames);
        let mut set0 = Vec::with_capacity(frames);
        for _ in 0..frames {
            let ubo = rhi.create_gpu_buffer(
                ubo_size,
                true, // host-visible: we write the uniforms CPU-side each frame
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            let set = allocate_set(&device, pool, set0_layout)?;
            // Point set0/binding0 at this frame's UBO.
            let buf = rhi.vk_buffer(ubo)?;
            write_ubo_descriptor(&device, set, buf, ubo_size);
            ubos.push(ubo);
            set0.push(set);
        }

        // ── IBL bake (CPU) + upload (GPU), ONCE. Decode dryad's equirect HDRI →
        //    SH9 diffuse coeffs + a roughness mip chain (flora_ibl), create the
        //    RGBA16F mip image + linear sampler + SH UBO, and stage-copy the mips
        //    in. These are STATIC per HDRI (one image/sampler/UBO shared across
        //    all frames' set1). ~7 mips maps roughness 0→1 onto LOD 0→6. ──
        let ibl = crate::flora_ibl::bake(7)
            .map_err(|e| RhiError::Other(format!("flora IBL bake: {e}").into()))?;
        let (ibl_w, ibl_h) = (ibl.mips[0].w as u32, ibl.mips[0].h as u32);
        let ibl_mip_levels = ibl.mips.len() as u32;
        let (ibl_image, ibl_memory, ibl_view) = create_ibl_image(
            &device,
            rhi.instance_handle(),
            rhi.physical_device(),
            ibl_w,
            ibl_h,
            ibl_mip_levels,
        )?;
        let ibl_sampler = create_ibl_sampler(&device, ibl.max_lod())?;
        upload_ibl_mips(rhi, ibl_image, &ibl.mips)?;

        // SH9 coeffs UBO (static): the 9 cosine-convolved RGB coeffs + max LOD.
        let sh = ShCoeffs {
            c: ibl.sh9,
            params: [ibl.max_lod(), 0.0, 0.0, 0.0],
        };
        let sh_ubo_size = std::mem::size_of::<ShCoeffs>() as u64;
        let sh_ubo =
            rhi.create_gpu_buffer(sh_ubo_size, true, vk::BufferUsageFlags::UNIFORM_BUFFER)?;
        rhi.write_storage_bytes(sh_ubo, bytemuck::bytes_of(&sh))?;
        let sh_ubo_buf = rhi.vk_buffer(sh_ubo)?;

        // ── LEAF CLUSTER TEXTURES (set1/binding 7,8,9): create the color (sRGB)
        //    + normal (UNORM) images + the shared LINEAR/CLAMP sampler at fixed
        //    LEAF_TEX_SIZE, and seed them with a TRANSPARENT placeholder (the real
        //    per-genome bake happens in `update_leaf_texture` once a tree exists).
        //    The image VIEWS are stable across reseed — only the pixel data is
        //    re-uploaded into these same images — so the descriptor write below is
        //    one-time, exactly like the IBL handles. ──
        let leaf_tex_size = enki_flora::leaf_texture::LEAF_TEX_SIZE;
        let (leaf_color_image, leaf_color_memory, leaf_color_view) = create_leaf_image(
            &device,
            rhi.instance_handle(),
            rhi.physical_device(),
            leaf_tex_size,
            leaf_tex_size,
            vk::Format::R8G8B8A8_SRGB,
        )?;
        let (leaf_normal_image, leaf_normal_memory, leaf_normal_view) = create_leaf_image(
            &device,
            rhi.instance_handle(),
            rhi.physical_device(),
            leaf_tex_size,
            leaf_tex_size,
            vk::Format::R8G8B8A8_UNORM,
        )?;
        let leaf_tex_sampler = create_leaf_sampler(&device)?;
        // Transparent placeholder (alpha 0 → fully discarded) so the images are in
        // SHADER_READ_ONLY_OPTIMAL with valid contents before the first tree bake.
        let placeholder = vec![0u8; (leaf_tex_size * leaf_tex_size * 4) as usize];
        upload_leaf_image(
            rhi,
            leaf_color_image,
            leaf_tex_size,
            leaf_tex_size,
            &placeholder,
            vk::ImageLayout::UNDEFINED,
        )?;
        upload_leaf_image(
            rhi,
            leaf_normal_image,
            leaf_tex_size,
            leaf_tex_size,
            &placeholder,
            vk::ImageLayout::UNDEFINED,
        )?;

        // ── IMPOSTOR ATLAS (set1/binding 12): a 1024×512 RGBA16F image, two square
        //    tiles (SIDE | TOP). COLOR_ATTACHMENT (the bake renders into it) |
        //    SAMPLED (fs_impostor reads it). Cleared/UNDEFINED until `bake_impostor`
        //    fills it; the view is stable so the set1 write is one-time. A small
        //    dedicated D32 bake-depth backs the tile rendering (own state, no fight
        //    with the swapchain-sized scene depth). ──
        let mem_props =
            unsafe { rhi.instance_handle().get_physical_device_memory_properties(rhi.physical_device()) };
        let impostor_atlas = create_color_image(
            &device,
            &mem_props,
            vk::Extent2D { width: IMPOSTOR_ATLAS_W, height: IMPOSTOR_ATLAS_H },
            1,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
        )?;
        let (impostor_depth_image, impostor_depth_memory, impostor_depth_view) =
            create_bake_depth(&device, &mem_props, IMPOSTOR_ATLAS_W, IMPOSTOR_ATLAS_H)?;
        // BAKE pipelines: single-sample, HDR_FORMAT, clear-to-zero-alpha tiles. The
        // branch is 6-stream cull-BACK (same as the lit branch); the leaf 4-stream.
        let impostor_bake_branch = build_pipeline_culled(
            &device, pipeline_layout, flora_module, "vs_branch", "fs_branch",
            HDR_FORMAT, vk::SampleCountFlags::TYPE_1, false, true, true,
            vk::CullModeFlags::BACK, 6,
        )?;
        // Match the lit leaf pipeline's cull (BACK) so the baked canopy looks like
        // the scene canopy (the leaf mesh emits both windings → still double-sided).
        let impostor_bake_leaf = build_pipeline_culled(
            &device, pipeline_layout, flora_module, "vs_leaf", "fs_leaf",
            HDR_FORMAT, vk::SampleCountFlags::TYPE_1, false, true, true,
            vk::CullModeFlags::BACK, 4,
        )?;

        // ── Per-frame set1 (shadow) ring: each set points binding0→shadow view,
        //    binding1→compare sampler, binding2→this frame's ShadowUniforms UBO,
        //    plus the SHARED IBL bindings 3/4/5 (env mip image, linear sampler,
        //    SH UBO) written into every frame copy. ──
        let shadow_ubo_size = std::mem::size_of::<ShadowUniforms>() as u64;
        // Per-frame bone-matrix storage buffer: MAX_WIND_BONES mat4s (64 KiB).
        let bone_buf_size =
            (enki_flora::mesh::MAX_WIND_BONES * 16 * std::mem::size_of::<f32>()) as u64;
        let mut set1 = Vec::with_capacity(frames);
        let mut shadow_ubos = Vec::with_capacity(frames);
        let mut bone_ubos = Vec::with_capacity(frames);
        for _ in 0..frames {
            let ubo = rhi.create_gpu_buffer(
                shadow_ubo_size,
                true,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            // Wind bone-matrix storage buffer (host-visible: solved + written
            // CPU-side each frame, like the shadow UBO).
            let bone_ubo = rhi.create_gpu_buffer(
                bone_buf_size,
                true,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            )?;
            let set = allocate_set(&device, pool, set1_layout)?;
            let buf = rhi.vk_buffer(ubo)?;
            write_shadow_descriptors(
                &device,
                set,
                shadow_view,
                shadow_sampler,
                buf,
                shadow_ubo_size,
            );
            write_ibl_descriptors(
                &device,
                set,
                ibl_view,
                ibl_sampler,
                sh_ubo_buf,
                sh_ubo_size,
            );
            // bindings 7/8/9 ← the shared (per-tree) leaf color/normal/sampler.
            write_leaf_tex_descriptors(
                &device,
                set,
                leaf_color_view,
                leaf_normal_view,
                leaf_tex_sampler,
            );
            // binding 12 ← the impostor atlas (static after the one-shot bake; the
            // image is cleared-UNDEFINED until then, but the view is valid to bind).
            write_impostor_atlas_descriptor(&device, set, impostor_atlas.sample_view);
            // binding 6 ← this frame's bone-matrix storage buffer.
            let bone_buf = rhi.vk_buffer(bone_ubo)?;
            write_bone_descriptor(&device, set, bone_buf, bone_buf_size);
            shadow_ubos.push(ubo);
            bone_ubos.push(bone_ubo);
            set1.push(set);
        }

        // ── POST: build the dryad RenderPass→UnrealBloom→OutputPass machinery.
        //    The OutputPass writes the swapchain format at MSAA sample count (it
        //    runs INSIDE enki's begin_rendering MSAA instance); the bright/blur/
        //    composite passes are single-sample HDR fullscreen. ──
        let instance = rhi.instance_handle().clone();
        let physical_device = rhi.physical_device();
        let post = build_post(
            &device,
            &instance,
            physical_device,
            rhi.swapchain_format(),
            samples,
            rhi.extent(),
            frames,
        )?;

        Ok(Self {
            device,
            in_scene,
            capture: None,
            flora_module,
            staging_module,
            set0_layout,
            pipeline_layout,
            branch,
            branch_wire,
            leaf,
            leaf_wire,
            ground,
            sky,
            shadow,
            arrow,
            impostor,
            branch_depth,
            leaf_depth,
            pool,
            ubos,
            set0,
            shadow_image,
            shadow_memory,
            shadow_view,
            shadow_sampler,
            shadow_initialized: false,
            set1_layout,
            set1,
            shadow_ubos,
            bone_ubos,
            ibl_image,
            ibl_memory,
            ibl_view,
            ibl_sampler,
            leaf_color_image,
            leaf_color_memory,
            leaf_color_view,
            leaf_normal_image,
            leaf_normal_memory,
            leaf_normal_view,
            leaf_tex_sampler,
            leaf_tex_built: false,
            impostor_atlas,
            impostor_depth_image,
            impostor_depth_memory,
            impostor_depth_view,
            impostor_bake_branch,
            impostor_bake_leaf,
            impostor_atlas_baked: false,
            instance,
            physical_device,
            post,
        })
    }

    /// (Re)bake the leaf cluster textures from the resolved genome leaf genes and
    /// re-upload them into the persistent leaf color/normal images. Called at tree
    /// build (and on reseed/regen, since leaf genes vary per tree). The image
    /// VIEWS are stable, so set1 never needs rewriting — only the pixel data
    /// changes. Idempotent w.r.t. layout: the first call uploads from UNDEFINED,
    /// later calls from SHADER_READ_ONLY_OPTIMAL.
    ///
    /// The caller MUST `wait_idle` first (the viewer's `rebuild` already does), so
    /// no in-flight frame is sampling the textures while they're re-uploaded.
    /// `single == true` bakes the SINGLE-leaf sprite (one centered leaf,
    /// `bake_leaf_single`) used by the viewer's Single leaf mode; otherwise the
    /// 5-8 leaf CLUSTER sprite (`bake_leaf_cluster`). Both upload the same two
    /// `set1` images (color + normal), so `fs_leaf` samples either transparently.
    pub fn update_leaf_texture(
        &mut self,
        rhi: &mut Rhi,
        genes: &enki_flora::leaf_texture::LeafGenes,
        single: bool,
    ) -> Result<(), RhiError> {
        let tex = if single {
            enki_flora::leaf_texture::bake_leaf_single(genes)
        } else {
            enki_flora::leaf_texture::bake_leaf_cluster(genes)
        };
        let old = if self.leaf_tex_built {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::UNDEFINED
        };
        upload_leaf_image(rhi, self.leaf_color_image, tex.size, tex.size, &tex.color, old)?;
        upload_leaf_image(rhi, self.leaf_normal_image, tex.size, tex.size, &tex.normal, old)?;
        self.leaf_tex_built = true;
        Ok(())
    }

    /// Bake the TWO-TILE impostor atlas (SIDE | TOP) from `view`'s real tree mesh,
    /// ONCE at startup. Must run AFTER `update_leaf_texture` (the leaf textures must
    /// be SHADER_READ) and after IBL (always ready post-`new()`). Records into a
    /// transient one-shot fenced command buffer (like `upload_leaf_image`), OUTSIDE
    /// the frame loop, so no `begin_frame`/`begin_rendering` is needed. The caller
    /// MUST `wait_idle` first (the viewer's `rebuild` does).
    ///
    /// Each tile is a single-sample HDR rendering instance cleared to `[0,0,0,0]`
    /// (so the leaf-card alpha cutout becomes the impostor silhouette), with an
    /// ortho fit to the tree bounds along the tile's view dir (SIDE = +Z, TOP =
    /// +Y). The branch + leaves draw at the LINEAR-HDR lane (no in-shader ACES), so
    /// the atlas stores pre-ACES color the `fs_impostor` output ACES-es once. After
    /// both tiles the atlas is barriered COLOR→SHADER_READ for the impostor sampler.
    pub fn bake_impostor(
        &mut self,
        rhi: &mut Rhi,
        view: &crate::flora_view::FloraView,
    ) -> Result<(), RhiError> {
        use enki_render::frame::FrameUniforms;
        use enki_render::lights::Lights;

        // ── Seed set0[0] base lighting (dryad rig) + set1[0] neutral shadow +
        //    identity bone matrices (static rest pose). view_proj is overwritten
        //    per tile below. ──
        let base_fu = FrameUniforms::new(
            &enki_render::camera::Camera::default_orbit(),
            1.0,
            &Lights::dryad_default(),
        );
        let neutral_shadow = ShadowUniforms {
            light_view_proj: glam::Mat4::IDENTITY.to_cols_array_2d(),
            params: [1.0 / SHADOW_MAP_SIZE as f32, 0.02, 0.0 /* DISABLED */, 0.0],
            light_view_proj1: glam::Mat4::IDENTITY.to_cols_array_2d(),
            light_view_proj2: glam::Mat4::IDENTITY.to_cols_array_2d(),
            params2: [1.0, 0.0, 0.0, 0.0],
        };
        self.set_shadow_uniforms(rhi, 0, &neutral_shadow)?;
        let identity_bones: Vec<f32> = {
            let mut v = vec![0.0f32; enki_flora::mesh::MAX_WIND_BONES * 16];
            for b in 0..enki_flora::mesh::MAX_WIND_BONES {
                let m = glam::Mat4::IDENTITY.to_cols_array();
                v[b * 16..b * 16 + 16].copy_from_slice(&m);
            }
            v
        };
        self.set_bone_matrices(rhi, 0, &identity_bones)?;

        // Ortho fit to the tree bounds (mirrors flora_viewer::light_view_proj). A
        // uniform cube fit (max extent) so all view dirs frame the whole tree.
        let (center, half) = view.impostor_frame();
        let tile_vp = |dir: glam::Vec3| -> [[f32; 4]; 4] {
            let d = dir.normalize_or(glam::Vec3::Y);
            let eye = center + d * 50.0;
            let up = if d.abs_diff_eq(glam::Vec3::Y, 1e-3) || d.abs_diff_eq(glam::Vec3::NEG_Y, 1e-3) {
                glam::Vec3::Z
            } else {
                glam::Vec3::Y
            };
            let view_m = glam::Mat4::look_at_rh(eye, center, up);
            let near_d = 1.0_f32;
            let far_d = 100.0 + 2.0 * half;
            let ortho = enki_render::projection::reversed_z_orthographic(
                -half, half, -half, half, near_d, far_d,
            );
            (ortho * view_m).to_cols_array_2d()
        };
        // SIDE = look along +Z toward the tree; TOP = straight down (+Y).
        let side_vp = tile_vp(glam::Vec3::Z);
        let top_vp = tile_vp(glam::Vec3::Y);
        // The bake eye (for the lit pass's view vector) — use the SIDE eye; a far
        // billboard's specular is invisible, so a single representative eye is fine.
        let se = center + glam::Vec3::Z * 50.0;
        let side_eye = [se.x, se.y, se.z, 1.0];

        // ── One-shot fenced command buffer (frame-0 pool, valid at startup). ──
        let device = self.device.clone();
        let pool = rhi.command_pool(0);
        let queue = rhi.queue_handle();
        let cmd = unsafe {
            device.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
        }
        .map_err(RhiError::Vulkan)?[0];
        unsafe {
            device.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(RhiError::Vulkan)?;

        // Barrier atlas UNDEFINED→COLOR_ATTACHMENT + depth UNDEFINED→DEPTH_ATTACHMENT
        // (whole image; both tiles share one image, cleared once at instance start
        // — but we open ONE instance per tile so the clear is scoped by the tile
        // viewport+scissor; clearing the whole image twice is fine, the scissor
        // confines the draw, and load_op CLEAR clears the whole render_area, which
        // we set to the tile rect).
        barrier_image(
            &device, cmd, self.impostor_atlas.image, vk::ImageAspectFlags::COLOR, 0, 1,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::empty(), vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );
        barrier_image(
            &device, cmd, self.impostor_depth_image, vk::ImageAspectFlags::DEPTH, 0, 1,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::empty(), vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        );

        // Record both tiles. Tile 0 = SIDE (x 0..TILE); tile 1 = TOP (x TILE..2·TILE).
        for (tile_x, vp) in [(0u32, side_vp), (IMPOSTOR_TILE, top_vp)] {
            // Per-tile FrameUniforms: dryad lights + this tile's ortho view_proj.
            let mut fu = base_fu;
            fu.view_proj = vp;
            fu.camera_pos = side_eye;
            self.set_frame_uniforms(rhi, 0, &fu)?;

            let tile_extent = vk::Extent2D { width: IMPOSTOR_TILE, height: IMPOSTOR_TILE };
            let color_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.impostor_atlas.sample_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::STORE)
                // *** alpha 0 → leaf-card coverage IS the impostor silhouette ***
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 0.0] },
                });
            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.impostor_depth_view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::CLEAR)
                .store_op(vk::AttachmentStoreOp::DONT_CARE)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
                });
            // render_area = the tile rect (CLEAR clears only this rect).
            let rendering_info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: tile_x as i32, y: 0 },
                    extent: tile_extent,
                })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_attachment))
                .depth_attachment(&depth_attachment);
            unsafe {
                device.cmd_begin_rendering(cmd, &rendering_info);
                // TILE-sized viewport+scissor (NOT set_viewport_scissor_full).
                let viewport = vk::Viewport {
                    x: tile_x as f32,
                    y: 0.0,
                    width: IMPOSTOR_TILE as f32,
                    height: IMPOSTOR_TILE as f32,
                    min_depth: 0.0,
                    max_depth: 1.0,
                };
                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x: tile_x as i32, y: 0 },
                    extent: tile_extent,
                };
                device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport));
                device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
            }
            // Branch + leaves (raw-ash binds into THIS transient cmd buffer).
            view.record_bake_draws(
                rhi,
                &device,
                cmd,
                self.pipeline_layout,
                self.set0[0],
                self.set1[0],
                self.impostor_bake_branch.pipeline,
                self.impostor_bake_leaf.pipeline,
            )?;
            unsafe { device.cmd_end_rendering(cmd) };
        }

        // Atlas COLOR→SHADER_READ so `fs_impostor` can sample it.
        barrier_image(
            &device, cmd, self.impostor_atlas.image, vk::ImageAspectFlags::COLOR, 0, 1,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE, vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::FRAGMENT_SHADER,
        );

        unsafe { device.end_command_buffer(cmd) }.map_err(RhiError::Vulkan)?;
        let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(RhiError::Vulkan)?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        unsafe {
            device
                .queue_submit(queue, std::slice::from_ref(&submit), fence)
                .map_err(RhiError::Vulkan)?;
            device
                .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
                .map_err(RhiError::Vulkan)?;
            device.destroy_fence(fence, None);
            device.free_command_buffers(pool, std::slice::from_ref(&cmd));
        }
        self.impostor_atlas_baked = true;
        Ok(())
    }

    /// The shadow-map resolution (square), for the viewer's PCF texel-size param.
    pub fn shadow_map_size(&self) -> u32 {
        SHADOW_MAP_SIZE
    }

    /// The offscreen SCENE pass color format (LINEAR-HDR RGBA16F). A renderer that
    /// records its OWN draw into the scene pass (e.g. the Nanite branch path) must
    /// build its color pipeline against this format — the dynamic-rendering color
    /// attachment format must match the bound pipeline's. Depth is `D32_SFLOAT` at
    /// `rhi.msaa_samples()`, both already the Nanite draw pipeline's defaults.
    pub fn scene_color_format(&self) -> vk::Format {
        HDR_FORMAT
    }

    /// Whether this renderer draws into the app's main 3D pass (vs its own
    /// offscreen HDR + bloom pass). Drives the ACES-in-shader push lane in
    /// `FloraView::record_branch`/`record_leaves`.
    pub fn in_scene(&self) -> bool {
        self.in_scene
    }

    /// Upload this frame's `ShadowUniforms` (light matrix + PCF params) into the
    /// set1 ring slot for `fi`. Call once per frame before the shadow pass.
    pub fn set_shadow_uniforms(
        &self,
        rhi: &mut Rhi,
        fi: u32,
        su: &ShadowUniforms,
    ) -> Result<(), RhiError> {
        rhi.write_storage_bytes(self.shadow_ubos[fi as usize], bytemuck::bytes_of(su))
    }

    /// Re-point this frame's set1 shadow textures (cascade 0/1/2 = bindings 0/10/11)
    /// at EXTERNAL depth views — the planet's 3-cascade CSM maps — so the trees
    /// receive the SAME shadows as the ground instead of their own self-shadow map.
    /// Call once per frame before the scene draw (with `cascade_count = 3`,
    /// `use_view_pos = 1` in the `ShadowUniforms`). The views must already be in
    /// `SHADER_READ_ONLY_OPTIMAL` (the CSM caster pass leaves them so).
    pub fn set_shadow_cascade_views(&self, fi: u32, views: [vk::ImageView; 3]) {
        let set = self.set1[fi as usize];
        let infos: [vk::DescriptorImageInfo; 3] = std::array::from_fn(|i| {
            vk::DescriptorImageInfo::default()
                .image_view(views[i])
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        });
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(std::slice::from_ref(&infos[0])),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(10)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(std::slice::from_ref(&infos[1])),
            vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(11)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(std::slice::from_ref(&infos[2])),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
    }

    /// Upload this frame's hierarchical-wind bone matrices into the set1/binding 6
    /// storage buffer for `fi`. `mats` is the solver output: a flat `[f32]` of
    /// per-bone column-major mat4s (`len == bone_count * 16`, `<= MAX_WIND_BONES *
    /// 16`). Call once per frame before the shadow + scene passes (same point as
    /// `set_shadow_uniforms`). When the solver runs at strength 0 every matrix is
    /// identity → the branch/leaf VS reproduce the exact static rest pose.
    pub fn set_bone_matrices(&self, rhi: &mut Rhi, fi: u32, mats: &[f32]) -> Result<(), RhiError> {
        rhi.write_storage_bytes(self.bone_ubos[fi as usize], bytemuck::cast_slice(mats))
    }

    /// Upload this frame's `FrameUniforms` into the ring slot for `fi`. Call once
    /// per frame before any flora draws (cheap host-visible memcpy).
    pub fn set_frame_uniforms(
        &self,
        rhi: &mut Rhi,
        fi: u32,
        fu: &FrameUniforms,
    ) -> Result<(), RhiError> {
        rhi.write_storage_bytes(self.ubos[fi as usize], bytemuck::bytes_of(fu))
    }

    /// Open the SUN SHADOW depth pass on the raw command buffer. MUST be called
    /// BETWEEN `rhi.begin_frame(fi)` and `rhi.begin_rendering(fi)` — that gap is
    /// outside any dynamic-rendering instance (Vulkan forbids nesting), and the
    /// streaming-upload precedent records there too. Barriers the shadow image
    /// into DEPTH_ATTACHMENT, begins a depth-only rendering instance clearing to
    /// reversed-Z far (0.0), and sets the shadow-sized viewport+scissor.
    ///
    /// Records `bind_depth` + draws (via `rhi.bind_vertex_buffers` etc., which hit
    /// the same cmd buffer) BETWEEN this and `end_shadow_pass`.
    pub fn begin_shadow_pass(&mut self, rhi: &Rhi, fi: u32) {
        let cmd = rhi.current_command_buffer(fi);
        let extent = vk::Extent2D {
            width: SHADOW_MAP_SIZE,
            height: SHADOW_MAP_SIZE,
        };

        // Barrier: first frame UNDEFINED→DEPTH_ATTACHMENT (discard); thereafter
        // SHADER_READ_ONLY→DEPTH_ATTACHMENT (last frame's sampling completed).
        let (old_layout, src_access, src_stage) = if self.shadow_initialized {
            (
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            )
        } else {
            self.shadow_initialized = true;
            (
                vk::ImageLayout::UNDEFINED,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
            )
        };
        self.barrier_shadow(
            cmd,
            old_layout,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            src_access,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            src_stage,
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        );

        // Depth-only dynamic-rendering instance, clear to reversed-Z far = 0.0.
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.shadow_view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .layer_count(1)
            .depth_attachment(&depth_attachment);
        unsafe {
            self.device.cmd_begin_rendering(cmd, &rendering_info);
            // Shadow-sized viewport+scissor (NOT the swapchain-sized full set).
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: SHADOW_MAP_SIZE as f32,
                height: SHADOW_MAP_SIZE as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            self.device
                .cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport));
            self.device
                .cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
        }
    }

    /// Bind a depth-caster pipeline + set0/set1 in the shadow pass. (set1 is bound
    /// only to satisfy the shared layout; the depth shaders don't sample it.)
    pub fn bind_depth(&self, rhi: &Rhi, fi: u32, which: FloraPipeline) {
        // Same as `bind`, but the caller has already set the shadow viewport, so
        // we don't touch viewport state here.
        self.bind(rhi, fi, which);
    }

    /// Close the shadow depth pass and barrier the shadow image to
    /// SHADER_READ_ONLY so the main pass can sample it. Call after the caster
    /// draws and BEFORE `rhi.begin_rendering(fi)`.
    pub fn end_shadow_pass(&self, rhi: &Rhi, fi: u32) {
        let cmd = rhi.current_command_buffer(fi);
        unsafe {
            self.device.cmd_end_rendering(cmd);
        }
        self.barrier_shadow(
            cmd,
            vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
    }

    /// Record a depth-aspect image barrier on the shadow image.
    #[allow(clippy::too_many_arguments)]
    fn barrier_shadow(
        &self,
        cmd: vk::CommandBuffer,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
        src_access: vk::AccessFlags,
        dst_access: vk::AccessFlags,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) {
        let barrier = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(new_layout)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(self.shadow_image)
            .src_access_mask(src_access)
            .dst_access_mask(dst_access)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        unsafe {
            self.device.cmd_pipeline_barrier(
                cmd,
                src_stage,
                dst_stage,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                std::slice::from_ref(&barrier),
            );
        }
    }

    /// Bind a flora-owned pipeline + this frame's set0 (FrameUniforms) and set1
    /// (shadow map + compare sampler + ShadowUniforms), into the raw command
    /// buffer. Replaces enki's `bind_pipeline`; flora binds its OWN layout/sets so
    /// the shadow set is present and the main shaders can sample the shadow map.
    pub fn bind(&self, rhi: &Rhi, fi: u32, which: FloraPipeline) {
        let cmd = rhi.current_command_buffer(fi);
        let op = self.pipeline_for(which);
        let set0 = self.set0[fi as usize];
        let set1 = self.set1[fi as usize];
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, op.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                op.layout,
                0,
                &[set0, set1],
                &[],
            );
        }
    }

    /// Record push constants against flora's OWN pipeline layout (replaces enki's
    /// `push_constants`, which uses enki's remembered layout). `bytes` ≤ 128.
    pub fn push(&self, rhi: &Rhi, fi: u32, bytes: &[u8]) {
        let cmd = rhi.current_command_buffer(fi);
        unsafe {
            self.device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytes,
            );
        }
    }

    /// The current POST target extent (for the viewer's resize check).
    pub fn post_extent(&self) -> vk::Extent2D {
        self.post.targets.extent
    }

    /// Recreate the resize-dependent POST targets (scene HDR MSAA/depth/resolve +
    /// bloom chain) at the new swapchain extent. The caller MUST `wait_idle` first
    /// (the old targets may still be in flight). The pipelines/sampler are
    /// resolution-independent and survive. No-op if the extent is unchanged.
    pub fn resize_post(&mut self, extent: vk::Extent2D) -> Result<(), RhiError> {
        if self.post.targets.extent == extent || extent.width == 0 || extent.height == 0 {
            return Ok(());
        }
        let old = std::mem::replace(
            &mut self.post.targets,
            create_post_targets(&self.device, &self.instance, self.physical_device, extent)?,
        );
        old.destroy(&self.device);
        Ok(())
    }

    /// Open the OFFSCREEN SCENE pass on the flora-owned LINEAR-HDR MSAA target +
    /// depth. Recorded in the begin_frame→begin_rendering gap (like the shadow
    /// pass). Barriers the MSAA color/depth + the resolve target into attachment
    /// layout, begins a dynamic-rendering instance (color→resolve, reversed-Z
    /// depth clear 0.0), and sets a full-extent viewport. Sky/ground/tree draw
    /// between this and `end_scene_pass`.
    pub fn begin_scene_pass(&mut self, rhi: &Rhi, fi: u32) {
        let cmd = rhi.current_command_buffer(fi);
        let t = &self.post.targets;
        let extent = t.extent;
        // Capture the handles we need so the later mutation of `initialized`
        // doesn't conflict with the (now-dropped) immutable borrow of `t`.
        let scene_msaa_image = t.scene_msaa_image;
        let scene_depth_image = t.scene_depth_image;
        let scene_resolve_image = t.scene_resolve.image;
        let scene_msaa_view = t.scene_msaa_view;
        let scene_resolve_view = t.scene_resolve.sample_view;
        let scene_depth_view = t.scene_depth_view;
        let initialized = t.initialized;

        // MSAA color + depth: UNDEFINED on first use (discard), else from the prior
        // frame's COLOR/DEPTH attachment usage (no read between frames → re-use as
        // attachment is the simplest correct ordering). Resolve target: UNDEFINED→
        // COLOR (it's written fresh, then read by the bright pass).
        let (col_old, col_src_access, col_src_stage) = if initialized {
            (
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            )
        } else {
            (
                vk::ImageLayout::UNDEFINED,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
            )
        };
        let (dep_old, dep_src_access, dep_src_stage) = if initialized {
            (
                vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            )
        } else {
            (
                vk::ImageLayout::UNDEFINED,
                vk::AccessFlags::empty(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
            )
        };

        barrier_image(
            &self.device, cmd, scene_msaa_image, vk::ImageAspectFlags::COLOR, 0, 1,
            col_old, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            col_src_access, vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            col_src_stage, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );
        barrier_image(
            &self.device, cmd, scene_depth_image, vk::ImageAspectFlags::DEPTH, 0, 1,
            dep_old, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            dep_src_access, vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dep_src_stage, vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        );
        // Resolve target → COLOR_ATTACHMENT (resolve write). Prior layout is
        // SHADER_READ (last frame's bright/output sampled it) or UNDEFINED on init.
        let res_old = if initialized {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        } else {
            vk::ImageLayout::UNDEFINED
        };
        barrier_image(
            &self.device, cmd, scene_resolve_image, vk::ImageAspectFlags::COLOR, 0, 1,
            res_old, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::SHADER_READ, vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::FRAGMENT_SHADER, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );
        self.post.targets.initialized = true;

        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0, 0.0, 0.0, 1.0] },
        };
        let clear_depth = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
        };
        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(scene_msaa_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
            .resolve_image_view(scene_resolve_view)
            .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(clear_color);
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(scene_depth_view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(clear_depth);
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment))
            .depth_attachment(&depth_attachment);
        unsafe {
            self.device.cmd_begin_rendering(cmd, &rendering_info);
            set_viewport_scissor(&self.device, cmd, extent);
        }
    }

    /// Close the offscreen scene pass and barrier the resolved HDR target to
    /// SHADER_READ so the bloom + output passes can sample it.
    pub fn end_scene_pass(&self, rhi: &Rhi, fi: u32) {
        let cmd = rhi.current_command_buffer(fi);
        let t = &self.post.targets;
        unsafe { self.device.cmd_end_rendering(cmd) };
        barrier_image(
            &self.device, cmd, t.scene_resolve.image, vk::ImageAspectFlags::COLOR, 0, 1,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE, vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::FRAGMENT_SHADER,
        );
    }

    /// Run the UnrealBloom chain on the resolved HDR scene: bright-pass → 5×
    /// {H blur, V blur} progressive mip chain → composite. Recorded in the
    /// begin_frame→begin_rendering gap (after `end_scene_pass`). Leaves
    /// `bloom_composite` in SHADER_READ for the output pass.
    ///
    /// `strength` is the UI-driven bloom strength (dryad default 0.15). The
    /// composite pre-multiplies by it, so `strength == 0.0` ⇒ the composite is
    /// all-zero ⇒ the output pass reads `scene + 0` ⇒ EXACTLY the offscreen-stage
    /// parity image. The bloom chain still RUNS at strength 0 (the bright/blur
    /// targets just feed a zeroed composite) — cheap, and avoids a layout-state
    /// branch in the output pass (it always samples a valid SHADER_READ composite).
    pub fn run_bloom(&self, rhi: &Rhi, fi: u32, strength: f32) {
        let cmd = rhi.current_command_buffer(fi);
        // Reset THIS frame's own transient-set pool (per-FIF) so each pass
        // allocates fresh sets. Resetting only `pools[fi]` is safe: the
        // frame-acquire fence already guaranteed this FIF's prior submission
        // completed before its command buffer was re-recorded. (Resetting a pool
        // shared across both FIFs would hit VUID-vkResetDescriptorPool-00313.)
        unsafe {
            self.device
                .reset_descriptor_pool(
                    self.post.pools[fi as usize],
                    vk::DescriptorPoolResetFlags::empty(),
                )
                .expect("reset post pool");
        }
        let t = &self.post.targets;
        let half = vk::Extent2D {
            width: (t.extent.width / 2).max(1),
            height: (t.extent.height / 2).max(1),
        };

        // ── Bright pass: scene_resolve → bloom_bright (half-res). scene_resolve is
        //    single-mip and fully transitioned to SHADER_READ by end_scene_pass. ──
        self.post_pass(
            cmd,
            fi,
            self.post.bright.pipeline,
            t.scene_resolve.sample_view,
            None,
            t.bloom_bright.mip_views[0],
            half,
            PostPush {
                // (threshold 1.1, smoothWidth 0.01, _, _)
                params: [1.1, 0.01, 0.0, 0.0],
                params2: [0.0; 4],
            },
            &[(t.bloom_bright.image, 0)],
        );

        // ── Progressive blur mip chain. Mip 0 input = bloom_bright; mip i>0 input
        //    = bloom_v mip (i-1). Each mip: H (bloom_h[i]) then V (bloom_v[i]). ──
        //
        // The blur source is bound as a SINGLE-MIP view (the exact mip that was
        // transitioned to SHADER_READ by the prior post_pass), NOT the full mip
        // chain — a full-chain sample view would let the sampler reach mips still
        // in UNDEFINED layout (VUID-vkCmdDraw-None-09600). With a single-mip view
        // the only reachable LOD is 0, so every blur tap samples LOD 0.0.
        for i in 0..BLOOM_MIPS {
            let mip_w = (half.width >> i).max(1);
            let mip_h = (half.height >> i).max(1);
            let mip_ext = vk::Extent2D { width: mip_w, height: mip_h };
            let inv = [1.0 / mip_w as f32, 1.0 / mip_h as f32];
            let kr = BLOOM_KERNELS[i];

            // H source: mip 0 reads bloom_bright[0]; mip i>0 reads the previous
            // mip's V result bloom_v[i-1] (both single-mip views, sampled at LOD 0).
            let h_input = if i == 0 {
                t.bloom_bright.mip_views[0]
            } else {
                t.bloom_v.mip_views[i - 1]
            };

            // Horizontal: direction (1,0). Input → bloom_h[i]. LOD lane = 0.0.
            self.post_pass(
                cmd,
                fi,
                self.post.blur.pipeline,
                h_input,
                None,
                t.bloom_h.mip_views[i],
                mip_ext,
                PostPush {
                    params: [1.0, 0.0, inv[0], inv[1]],
                    params2: [kr, 0.0, 0.0, 0.0],
                },
                &[(t.bloom_h.image, i as u32)],
            );
            // Vertical: direction (0,1). bloom_h[i] (single-mip view) → bloom_v[i].
            // LOD lane = 0.0 (the bound view's only level).
            self.post_pass(
                cmd,
                fi,
                self.post.blur.pipeline,
                t.bloom_h.mip_views[i],
                None,
                t.bloom_v.mip_views[i],
                mip_ext,
                PostPush {
                    params: [0.0, 1.0, inv[0], inv[1]],
                    params2: [kr, 0.0, 0.0, 0.0],
                },
                &[(t.bloom_v.image, i as u32)],
            );
        }

        // ── Composite: combine the 5 bloom_v mips → bloom_composite (half-res),
        //    pre-multiplied by bloomStrength so strength=0 ⇒ all-zero. This pass
        //    legitimately needs the FULL bloom_v mip chain (it samples LODs 0..4),
        //    and runs AFTER all 5 bloom_v mips are SHADER_READ — so the full-chain
        //    sample view is valid here. ──
        self.post_pass(
            cmd,
            fi,
            self.post.composite.pipeline,
            t.bloom_v.sample_view,
            None,
            t.bloom_composite.mip_views[0],
            half,
            PostPush {
                // (bloomStrength [UI, dryad default 0.15], bloomRadius 0.4, _, _).
                // strength=0 ⇒ composite all-zero ⇒ output == scene (parity).
                params: [strength, 0.4, 0.0, 0.0],
                params2: [0.0; 4],
            },
            &[(t.bloom_composite.image, 0)],
        );
    }

    /// Record one fullscreen post pass: barrier the target mip → COLOR, begin a
    /// 1-sample rendering instance on `dst_view`, bind the pipeline + a transient
    /// descriptor set (src + optional aux), push `push`, draw 3 verts, end, then
    /// barrier the just-written image(s) → SHADER_READ for the next pass.
    #[allow(clippy::too_many_arguments)]
    fn post_pass(
        &self,
        cmd: vk::CommandBuffer,
        fi: u32,
        pipeline: vk::Pipeline,
        src_view: vk::ImageView,
        aux_view: Option<vk::ImageView>,
        dst_view: vk::ImageView,
        extent: vk::Extent2D,
        push: PostPush,
        // (image, mip) pairs written by this pass → barriered to SHADER_READ after.
        outputs: &[(vk::Image, u32)],
    ) {
        // Barrier each output mip UNDEFINED→COLOR (we overwrite the whole target).
        for &(img, mip) in outputs {
            barrier_image(
                &self.device, cmd, img, vk::ImageAspectFlags::COLOR, mip, 1,
                vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::AccessFlags::empty(), vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            );
        }
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(dst_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE);
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&attachment));
        let set = self.post_set(fi, src_view, aux_view);
        unsafe {
            self.device.cmd_begin_rendering(cmd, &info);
            set_viewport_scissor(&self.device, cmd, extent);
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.post.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.post.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            self.device.cmd_draw(cmd, 3, 1, 0, 0);
            self.device.cmd_end_rendering(cmd);
        }
        for &(img, mip) in outputs {
            barrier_image(
                &self.device, cmd, img, vk::ImageAspectFlags::COLOR, mip, 1,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE, vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::FRAGMENT_SHADER,
            );
        }
    }

    /// Allocate + write a transient post descriptor set: src tex (b0) + sampler
    /// (b1) + aux tex (b2, defaults to src if none) + sampler (b3). From this
    /// frame-in-flight's own frame-reset pool (`pools[fi]`).
    fn post_set(&self, fi: u32, src_view: vk::ImageView, aux_view: Option<vk::ImageView>) -> vk::DescriptorSet {
        let set = allocate_set(&self.device, self.post.pools[fi as usize], self.post.set_layout)
            .expect("alloc post set");
        let aux = aux_view.unwrap_or(src_view);
        let src_info = vk::DescriptorImageInfo::default()
            .image_view(src_view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let aux_info = vk::DescriptorImageInfo::default()
            .image_view(aux)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let samp_info = vk::DescriptorImageInfo::default().sampler(self.post.sampler);
        let writes = [
            vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(0)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(std::slice::from_ref(&src_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(1)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(std::slice::from_ref(&samp_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(2)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .image_info(std::slice::from_ref(&aux_info)),
            vk::WriteDescriptorSet::default()
                .dst_set(set).dst_binding(3)
                .descriptor_type(vk::DescriptorType::SAMPLER)
                .image_info(std::slice::from_ref(&samp_info)),
        ];
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        set
    }

    /// Record the COMPOSITE/OUTPUT fullscreen pass INSIDE enki's begin_rendering
    /// (the swapchain MSAA instance). Samples scene_resolve (b0) + bloom_composite
    /// (b2), does scene+bloom → ACES+sRGB ONCE, writes the swapchain. The caller
    /// must have set the full swapchain viewport already (set_viewport_scissor_full).
    pub fn record_composite(&self, rhi: &Rhi, fi: u32) {
        let cmd = rhi.current_command_buffer(fi);
        let t = &self.post.targets;
        let set = self.post_set(fi, t.scene_resolve.sample_view, Some(t.bloom_composite.sample_view));
        let push = PostPush { params: [0.0; 4], params2: [0.0; 4] };
        unsafe {
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.post.output.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.post.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.post.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            self.device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }

    // ── HEADLESS SCREENSHOT capture (only used when FLORA_SCREENSHOT is set) ──

    /// Record the screenshot capture for frame `fi`: render the SAME `fs_output`
    /// composite (scene_resolve + bloom_composite → ACES → _SRGB) into a flora-owned
    /// single-sample B8G8R8A8_SRGB image, then vkCmdCopyImageToBuffer it into a
    /// host-visible readback buffer. Recorded in the begin_frame→begin_rendering gap
    /// (outside any dynamic-rendering instance), like the shadow/scene passes — so
    /// it runs BEFORE (and independent of) the normal swapchain composite + egui.
    /// The caller `wait_idle`s, then `read_capture_rgba`.
    ///
    /// `begin_capture`: (re)allocate the capture target, barrier it to COLOR, OPEN
    /// a single-sample rendering instance on it, and draw the `fs_output` composite
    /// triangle. Paired with `end_capture`, which closes the instance + copies to
    /// the readback buffer. // ponytail: reusing `fs_output` + the existing post
    /// descriptor set means a clean render with zero new shader/pipe.
    pub fn begin_capture(&mut self, rhi: &mut Rhi, fi: u32) -> Result<(), RhiError> {
        let extent = self.post.targets.extent;
        // (Re)allocate the capture target if missing or resized.
        let need_new = match &self.capture {
            Some(c) => c.extent != extent,
            None => true,
        };
        if need_new {
            if let Some(old) = self.capture.take() {
                let _ = rhi.wait_idle();
                old.destroy(&self.device, rhi);
            }
            self.capture = Some(create_capture(
                &self.device,
                &self.instance,
                self.physical_device,
                rhi,
                extent,
                self.post.module,
                self.post.pipeline_layout,
            )?);
        }
        let cap = self.capture.as_ref().unwrap();
        let cmd = rhi.current_command_buffer(fi);
        let t = &self.post.targets;

        // Capture image: UNDEFINED → COLOR (we overwrite the whole thing).
        barrier_image(
            &self.device, cmd, cap.image, vk::ImageAspectFlags::COLOR, 0, 1,
            vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::AccessFlags::empty(), vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            vk::PipelineStageFlags::TOP_OF_PIPE, vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        );

        // Single-sample fs_output pass into the capture image (same set as
        // record_composite: scene_resolve b0 + bloom_composite b2).
        let set = self.post_set(fi, t.scene_resolve.sample_view, Some(t.bloom_composite.sample_view));
        let push = PostPush { params: [0.0; 4], params2: [0.0; 4] };
        let attachment = vk::RenderingAttachmentInfo::default()
            .image_view(cap.view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::STORE);
        let info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&attachment));
        unsafe {
            self.device.cmd_begin_rendering(cmd, &info);
            set_viewport_scissor(&self.device, cmd, extent);
            self.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, cap.pipeline);
            self.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.post.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_push_constants(
                cmd,
                self.post.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(&push),
            );
            self.device.cmd_draw(cmd, 3, 1, 0, 0);
        }
        Ok(())
    }

    /// CLOSE the capture rendering instance, then barrier the image to TRANSFER_SRC
    /// and copy it into the readback buffer. Call after `begin_capture`.
    pub fn end_capture(&mut self, rhi: &mut Rhi, fi: u32) -> Result<(), RhiError> {
        let extent = self.post.targets.extent;
        let cap = self.capture.as_ref().unwrap();
        let cmd = rhi.current_command_buffer(fi);
        unsafe {
            self.device.cmd_end_rendering(cmd);
        }

        // COLOR → TRANSFER_SRC, then copy the whole image into the readback buffer.
        barrier_image(
            &self.device, cmd, cap.image, vk::ImageAspectFlags::COLOR, 0, 1,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::AccessFlags::COLOR_ATTACHMENT_WRITE, vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT, vk::PipelineStageFlags::TRANSFER,
        );
        let readback_buf = rhi.vk_buffer(cap.readback)?;
        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0) // tightly packed
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 });
        unsafe {
            self.device.cmd_copy_image_to_buffer(
                cmd,
                cap.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                readback_buf,
                std::slice::from_ref(&region),
            );
            // Make the copy visible to a host read (after the submit + wait_idle).
            let buf_barrier = vk::BufferMemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
                .dst_access_mask(vk::AccessFlags::HOST_READ)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .buffer(readback_buf)
                .offset(0)
                .size(vk::WHOLE_SIZE);
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[],
                std::slice::from_ref(&buf_barrier),
                &[],
            );
        }
        Ok(())
    }

    /// Read the captured frame back as tightly-packed RGBA8 bytes (BGRA→RGBA
    /// swizzled). MUST be called after `begin_capture`/`end_capture` + a frame
    /// submit + a `rhi.wait_idle()`. Returns (width, height, rgba). B8G8R8A8_SRGB
    /// so the bytes are already display-ready; we only reorder B↔R.
    pub fn read_capture_rgba(&self, rhi: &Rhi) -> Result<(u32, u32, Vec<u8>), RhiError> {
        let cap = self
            .capture
            .as_ref()
            .ok_or_else(|| RhiError::Other("no capture target (begin_capture not called)".into()))?;
        let mut bytes = rhi.read_buffer(cap.readback)?;
        let needed = (cap.extent.width * cap.extent.height * 4) as usize;
        bytes.truncate(needed);
        // BGRA → RGBA in place (swap B and R of each 4-byte pixel).
        for px in bytes.chunks_exact_mut(4) {
            px.swap(0, 2);
        }
        Ok((cap.extent.width, cap.extent.height, bytes))
    }

    fn pipeline_for(&self, which: FloraPipeline) -> &OwnedPipeline {
        match which {
            FloraPipeline::Branch => &self.branch,
            FloraPipeline::BranchWire => &self.branch_wire,
            FloraPipeline::Leaf => &self.leaf,
            FloraPipeline::LeafWire => &self.leaf_wire,
            FloraPipeline::Ground => &self.ground,
            FloraPipeline::Sky => &self.sky,
            FloraPipeline::Shadow => &self.shadow,
            FloraPipeline::Arrow => &self.arrow,
            FloraPipeline::Impostor => &self.impostor,
            FloraPipeline::BranchDepth => &self.branch_depth,
            FloraPipeline::LeafDepth => &self.leaf_depth,
        }
    }

    /// Free every owned Vulkan object. The caller MUST `rhi.wait_idle()` first and
    /// call this before Rhi teardown (the device handle outlives us but the queue
    /// must be idle). The per-frame UBOs are RHI BufferHandles, reclaimed by the
    /// RHI at its own shutdown — we only free what WE created on the raw device.
    pub fn destroy(&mut self) {
        unsafe {
            for op in [
                &self.branch,
                &self.branch_wire,
                &self.leaf,
                &self.leaf_wire,
                &self.ground,
                &self.sky,
                &self.shadow,
                &self.arrow,
                &self.impostor,
                &self.branch_depth,
                &self.leaf_depth,
            ] {
                self.device.destroy_pipeline(op.pipeline, None);
            }
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_descriptor_set_layout(self.set0_layout, None);
            self.device.destroy_descriptor_set_layout(self.set1_layout, None);
            // Pool destruction frees all sets allocated from it.
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.destroy_shader_module(self.flora_module, None);
            self.device.destroy_shader_module(self.staging_module, None);
            // ── Shadow resources (raw-allocated, so freed by hand). ──
            self.device.destroy_sampler(self.shadow_sampler, None);
            self.device.destroy_image_view(self.shadow_view, None);
            self.device.destroy_image(self.shadow_image, None);
            self.device.free_memory(self.shadow_memory, None);
            // ── IBL resources (raw-allocated mip image + sampler). The SH UBO is
            //    an RHI BufferHandle, reclaimed at RHI shutdown like the other
            //    flora UBOs — we only free what WE created on the raw device. ──
            self.device.destroy_sampler(self.ibl_sampler, None);
            self.device.destroy_image_view(self.ibl_view, None);
            self.device.destroy_image(self.ibl_image, None);
            self.device.free_memory(self.ibl_memory, None);
            // ── Leaf cluster textures (raw-allocated color + normal + sampler). ──
            self.device.destroy_sampler(self.leaf_tex_sampler, None);
            self.device.destroy_image_view(self.leaf_color_view, None);
            self.device.destroy_image(self.leaf_color_image, None);
            self.device.free_memory(self.leaf_color_memory, None);
            self.device.destroy_image_view(self.leaf_normal_view, None);
            self.device.destroy_image(self.leaf_normal_image, None);
            self.device.free_memory(self.leaf_normal_memory, None);
            // ── Impostor atlas + bake depth + bake pipelines. ──
            self.device.destroy_pipeline(self.impostor_bake_branch.pipeline, None);
            self.device.destroy_pipeline(self.impostor_bake_leaf.pipeline, None);
            self.impostor_atlas.destroy(&self.device);
            self.device.destroy_image_view(self.impostor_depth_view, None);
            self.device.destroy_image(self.impostor_depth_image, None);
            self.device.free_memory(self.impostor_depth_memory, None);
            // ── POST resources (pipelines + sampler + pool + resize targets). ──
            for op in [
                &self.post.bright,
                &self.post.blur,
                &self.post.composite,
                &self.post.output,
            ] {
                self.device.destroy_pipeline(op.pipeline, None);
            }
            self.device.destroy_pipeline_layout(self.post.pipeline_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.post.set_layout, None);
            for &pool in &self.post.pools {
                self.device.destroy_descriptor_pool(pool, None);
            }
            self.device.destroy_sampler(self.post.sampler, None);
            self.device.destroy_shader_module(self.post.module, None);
            self.post.targets.destroy(&self.device);
            // ── Screenshot capture target (image/view/memory/pipeline; the
            //    readback buffer is an RHI BufferHandle, reclaimed at RHI shutdown). ──
            if let Some(cap) = &self.capture {
                self.device.destroy_pipeline(cap.pipeline, None);
                self.device.destroy_image_view(cap.view, None);
                self.device.destroy_image(cap.image, None);
                self.device.free_memory(cap.memory, None);
            }
        }
    }
}

impl Capture {
    /// Free the capture image/view/memory + pipeline + reclaim the RHI readback buffer.
    fn destroy(&self, device: &ash::Device, rhi: &mut Rhi) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
        rhi.destroy_buffer(self.readback);
    }
}

/// Allocate a screenshot capture target at `extent`: a single-sample
/// B8G8R8A8_SRGB image (COLOR_ATTACHMENT | TRANSFER_SRC) + a host-visible
/// TRANSFER_DST readback buffer (w*h*4 bytes) + a 1-sample `fs_output` pipeline.
/// // ponytail: B8G8R8A8_SRGB matches the swapchain, so the hardware sRGB OETF on
/// store makes the bytes display-ready — no in-shader encode, no float→byte
/// conversion, just a BGRA→RGBA reorder.
fn create_capture(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    rhi: &mut Rhi,
    extent: vk::Extent2D,
    post_module: vk::ShaderModule,
    post_layout: vk::PipelineLayout,
) -> Result<Capture, RhiError> {
    let format = vk::Format::B8G8R8A8_SRGB;
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&image_info, None) }.map_err(RhiError::Vulkan)?;

    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let mem_type = find_memory_type(&mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or_else(|| RhiError::Other("no device-local memory for capture image".into()))?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc, None) }.map_err(RhiError::Vulkan)?;
    unsafe { device.bind_image_memory(image, memory, 0) }.map_err(RhiError::Vulkan)?;

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    let view = unsafe { device.create_image_view(&view_info, None) }.map_err(RhiError::Vulkan)?;

    let readback_size = (extent.width * extent.height * 4) as u64;
    let readback = rhi.create_gpu_buffer(readback_size, true, vk::BufferUsageFlags::TRANSFER_DST)?;

    // 1-sample fs_output pipeline into B8G8R8A8_SRGB (no depth attachment, since
    // `samples == TYPE_1` → build_fullscreen_pipeline omits the depth format).
    let pipeline = build_fullscreen_pipeline(
        device, post_layout, post_module, "fs_output", format, vk::SampleCountFlags::TYPE_1,
    )?
    .pipeline;

    Ok(Capture { image, memory, view, readback, pipeline, extent })
}

// ── Build helpers (all on the raw device — the egui-style self-contained path) ──

/// Compile WGSL → SPIR-V (naga, same frontend/version as enki-rhi) and create a
/// VkShaderModule. Identity binding map preserves @group/@binding.
fn compile_module(device: &ash::Device, src: &str) -> Result<vk::ShaderModule, RhiError> {
    let module = wgsl::parse_str(src)
        .map_err(|e| RhiError::Other(format!("flora WGSL parse: {}", e.emit_to_string(src)).into()))?;
    let info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| RhiError::Other(format!("flora WGSL validate: {e:?}").into()))?;
    let opts = spv::Options {
        lang_version: (1, 3),
        flags: spv::WriterFlags::DEBUG,
        binding_map: Default::default(),
        ..Default::default()
    };
    let words = spv::write_vec(&module, &info, &opts, None)
        .map_err(|e| RhiError::Other(format!("flora SPIR-V emit: {e:?}").into()))?;
    let ci = vk::ShaderModuleCreateInfo::default().code(&words);
    unsafe { device.create_shader_module(&ci, None) }.map_err(RhiError::Vulkan)
}

/// set0: FrameUniforms UBO at binding 0, VERTEX|FRAGMENT — identical to enki's.
fn create_set0_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout, RhiError> {
    let binding = vk::DescriptorSetLayoutBinding::default()
        .binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
    let ci = vk::DescriptorSetLayoutCreateInfo::default()
        .bindings(std::slice::from_ref(&binding));
    unsafe { device.create_descriptor_set_layout(&ci, None) }.map_err(RhiError::Vulkan)
}

/// set1: the SUN SHADOW set + the IBL resources, all FRAGMENT-visible —
///   binding 0 : SAMPLED_IMAGE  — the D32 shadow depth map (`texture_depth_2d`)
///   binding 1 : SAMPLER        — the comparison sampler (`sampler_comparison`)
///   binding 2 : UNIFORM_BUFFER — `ShadowUniforms` (light matrix + PCF params)
///   binding 3 : SAMPLED_IMAGE  — the equirect IBL mip chain (`texture_2d<f32>`)
///   binding 4 : SAMPLER        — LINEAR trilinear sampler (the env mip lookup)
///   binding 5 : UNIFORM_BUFFER — `ShCoeffs` (9 SH irradiance coeffs + max LOD)
///
/// Separate SAMPLED_IMAGE + SAMPLER (not COMBINED_IMAGE_SAMPLER) so the WGSL
/// `texture_depth_2d` + `sampler_comparison` split (and the IBL `texture_2d` +
/// `sampler` split) map 1:1.
fn create_set1_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout, RhiError> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // ── IBL ──
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(4)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(5)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // ── HIERARCHICAL WIND bone matrices ──
        //   binding 6 : STORAGE_BUFFER — `array<mat4x4<f32>>` of per-bone world
        //   matrices, read in the VERTEX stage (vs_branch/vs_leaf + the depth
        //   casters skin against it), NOT the fragment stage like 0-5.
        vk::DescriptorSetLayoutBinding::default()
            .binding(6)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX),
        // ── LEAF CLUSTER TEXTURES (CPU-baked per genome, flora_view leaf genes) ──
        //   binding 7 : SAMPLED_IMAGE — leaf color sprite (RGBA8 sRGB, A=coverage)
        //   binding 8 : SAMPLED_IMAGE — leaf tangent-space normal map (RGBA8 UNORM)
        //   binding 9 : SAMPLER       — LINEAR CLAMP sampler shared by both.
        // FRAGMENT-visible: fs_leaf samples color (alpha-cutout) + normal; the leaf
        // DEPTH caster also reaches binding 7/9 via the shared layout (alpha test).
        vk::DescriptorSetLayoutBinding::default()
            .binding(7)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(8)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(9)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // ── PLANET CSM cascades 1 & 2 (high bindings → viewer's 0-9 layout intact).
        //   binding 10/11 : SAMPLED_IMAGE — cascade-1/2 depth textures. In the viewer
        //   these point at the single self-shadow map (unused; cascade_count = 1);
        //   on the planet `set_shadow_cascade_views` re-points them at the CSM maps.
        vk::DescriptorSetLayoutBinding::default()
            .binding(10)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(11)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        // ── IMPOSTOR ATLAS (binding 12) — the baked TOP+SIDE billboard texture,
        //   sampled by `fs_impostor` (with the IBL linear sampler at binding 4). ──
        vk::DescriptorSetLayoutBinding::default()
            .binding(12)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&ci, None) }.map_err(RhiError::Vulkan)
}

/// Pipeline layout = the given sets + a single 128-byte VS|FS push range.
fn create_pipeline_layout(
    device: &ash::Device,
    sets: &[vk::DescriptorSetLayout],
) -> Result<vk::PipelineLayout, RhiError> {
    let push = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(128);
    let ci = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(sets)
        .push_constant_ranges(std::slice::from_ref(&push));
    unsafe { device.create_pipeline_layout(&ci, None) }.map_err(RhiError::Vulkan)
}

/// Build one graphics pipeline replicating enki's main-pass state (pipeline.rs):
/// 4×vec3 vertex layout, reversed-Z GREATER, dynamic rendering, dynamic
/// viewport/scissor, cull BACK / front-face CCW, MSAA `samples`.
#[allow(clippy::too_many_arguments)]
fn build_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    vs_entry: &str,
    fs_entry: &str,
    color_format: vk::Format,
    samples: vk::SampleCountFlags,
    blend: bool,
    fill: bool,
    // depth_test=false → the SKY: a fullscreen quad at clip z=0 (reversed-Z FAR).
    // With the scene depth cleared to 0.0 and the GREATER compare, a z=0 sky
    // fragment fails `0 > 0` and is rejected → black background. Disabling the
    // depth test (and depth write) lets the sky fill the frame; it's still drawn
    // FIRST so the ground/tree (depth>0) overdraw it. // ponytail.
    depth_test: bool,
) -> Result<OwnedPipeline, RhiError> {
    // Default cull: BACK for normal geometry; NONE for the (untransformed) sky,
    // whose NDC winding flips under Vulkan's y-down framebuffer space.
    let default_cull = if depth_test {
        vk::CullModeFlags::BACK
    } else {
        vk::CullModeFlags::NONE
    };
    build_pipeline_culled(
        device, layout, module, vs_entry, fs_entry, color_format, samples, blend,
        fill, depth_test, default_cull, 4,
    )
}

/// As `build_pipeline`, but with an explicit cull mode (the ground passes NONE so
/// a floor plane never vanishes to back-face culling regardless of orbit winding).
#[allow(clippy::too_many_arguments)]
fn build_pipeline_culled(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    vs_entry: &str,
    fs_entry: &str,
    color_format: vk::Format,
    samples: vk::SampleCountFlags,
    blend: bool,
    fill: bool,
    depth_test: bool,
    cull_mode: vk::CullModeFlags,
    // Number of separate vec3<f32> vertex streams. 4 for everything (pos/nrm/uv/
    // attr); 6 for the branch lit pipeline (+ aTangent/aFrameU for the seamless
    // bark frame). Each stream is its own binding=location, stride 12, vertex-rate.
    n_streams: u32,
) -> Result<OwnedPipeline, RhiError> {
    let vs_name = CString::new(vs_entry).unwrap();
    let fs_name = CString::new(fs_entry).unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(module)
            .name(&vs_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(module)
            .name(&fs_name),
    ];

    // `n_streams` separate vec3<f32> bindings (pos / normal / uv / attr [+ tangent /
    // frameU]), stride 12, one binding == one location.
    let bindings: Vec<vk::VertexInputBindingDescription> = (0..n_streams)
        .map(|i| {
            vk::VertexInputBindingDescription::default()
                .binding(i)
                .stride(12)
                .input_rate(vk::VertexInputRate::VERTEX)
        })
        .collect();
    let attributes: Vec<vk::VertexInputAttributeDescription> = (0..n_streams)
        .map(|i| {
            vk::VertexInputAttributeDescription::default()
                .location(i)
                .binding(i)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0)
        })
        .collect();
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    let polygon_mode = if fill {
        vk::PolygonMode::FILL
    } else {
        vk::PolygonMode::LINE
    };
    // `cull_mode` is supplied by the caller: BACK for normal geometry, NONE for
    // the sky (its NDC winding flips under Vulkan's y-down framebuffer space →
    // BACK culling would reject it, leaving a black background) and for the ground
    // floor plane (always visible regardless of the orbit camera's winding).
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(polygon_mode)
        .cull_mode(cull_mode)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .line_width(1.0);

    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(samples)
        .sample_shading_enable(false);

    // Reversed-Z: GREATER, depth-write off for blended (shadow disc) passes.
    // The SKY (depth_test=false) disables the test entirely AND depth write, so
    // its z=0 fragments aren't rejected by the cleared-to-0 GREATER buffer and it
    // never clobbers the depth the ground/tree test against.
    let depth_write = !blend && depth_test;
    let depth_compare = if depth_test {
        vk::CompareOp::GREATER
    } else {
        vk::CompareOp::ALWAYS
    };
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(depth_test)
        .depth_write_enable(depth_write)
        .depth_compare_op(depth_compare)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    let blend_attachment = if blend {
        vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .alpha_blend_op(vk::BlendOp::ADD)
            .color_write_mask(vk::ColorComponentFlags::RGBA)
    } else {
        vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA)
    };
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(std::slice::from_ref(&blend_attachment));

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let depth_format = vk::Format::D32_SFLOAT;
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(std::slice::from_ref(&color_format))
        .depth_attachment_format(depth_format);

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .push_next(&mut rendering_info);

    let pipelines = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, e)| RhiError::Vulkan(e))?;

    Ok(OwnedPipeline {
        pipeline: pipelines[0],
        layout,
    })
}

/// A descriptor pool for `frames` set0 sets (one UBO each) + `frames` set1 sets.
/// Each set1 now has THREE UBOs-worth (set0 frame UBO is separate): the shadow
/// UBO (per-frame) + the shared SH UBO (written into every frame copy), TWO
/// SAMPLED_IMAGEs (shadow depth + IBL mip chain) and TWO SAMPLERs (compare + IBL
/// linear). `frames` = frames_in_flight.
fn create_pool(device: &ash::Device, frames: u32) -> Result<vk::DescriptorPool, RhiError> {
    let sizes = [
        // set0 UBO (×frames) + set1 ShadowUniforms UBO (×frames) + set1 SH UBO
        // (×frames — the same SH buffer is bound into every frame's set1).
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(frames * 3),
        // shadow depth (×3 cascades) + IBL mip + leaf color + leaf normal +
        // impostor atlas images, per frame (= 7).
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(frames * 7),
        // compare sampler + IBL linear sampler + leaf linear sampler, per frame.
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLER)
            .descriptor_count(frames * 3),
        // wind bone-matrix storage buffer, per frame (set1/binding 6).
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(frames),
    ];
    let ci = vk::DescriptorPoolCreateInfo::default()
        .max_sets(frames * 2) // `frames` set0 + `frames` set1
        .pool_sizes(&sizes);
    unsafe { device.create_descriptor_pool(&ci, None) }.map_err(RhiError::Vulkan)
}

fn allocate_set(
    device: &ash::Device,
    pool: vk::DescriptorPool,
    layout: vk::DescriptorSetLayout,
) -> Result<vk::DescriptorSet, RhiError> {
    let layouts = [layout];
    let ai = vk::DescriptorSetAllocateInfo::default()
        .descriptor_pool(pool)
        .set_layouts(&layouts);
    let sets = unsafe { device.allocate_descriptor_sets(&ai) }.map_err(RhiError::Vulkan)?;
    Ok(sets[0])
}

fn write_ubo_descriptor(
    device: &ash::Device,
    set: vk::DescriptorSet,
    buffer: vk::Buffer,
    size: u64,
) {
    let info = vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(0)
        .range(size);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(0)
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

// ── SUN SHADOW build helpers ──────────────────────────────────────────────────

/// Build a depth-ONLY caster pipeline: no color attachment, single sample,
/// reversed-Z GREATER depth-write ON, 4×vec3 vertex layout (same meshes as the
/// main pass), dynamic viewport+scissor. `fs_entry` is `None` for the branch
/// caster (pure depth) and `Some(..)` for the leaf caster (alpha-test discard).
fn build_depth_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    vs_entry: &str,
    fs_entry: Option<&str>,
) -> Result<OwnedPipeline, RhiError> {
    let vs_name = CString::new(vs_entry).unwrap();
    let fs_name = fs_entry.map(|s| CString::new(s).unwrap());
    let mut stages = vec![vk::PipelineShaderStageCreateInfo::default()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(module)
        .name(&vs_name)];
    if let Some(name) = &fs_name {
        stages.push(
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(module)
                .name(name),
        );
    }

    // Same 4×vec3 layout as the main pass (the caster reads the SAME meshes).
    let bindings: [vk::VertexInputBindingDescription; 4] = std::array::from_fn(|i| {
        vk::VertexInputBindingDescription::default()
            .binding(i as u32)
            .stride(12)
            .input_rate(vk::VertexInputRate::VERTEX)
    });
    let attributes: [vk::VertexInputAttributeDescription; 4] = std::array::from_fn(|i| {
        vk::VertexInputAttributeDescription::default()
            .location(i as u32)
            .binding(i as u32)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(0)
    });
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default()
        .vertex_binding_descriptions(&bindings)
        .vertex_attribute_descriptions(&attributes);

    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // ponytail: NO front/back cull in the shadow pass — leaves are DoubleSide
    // (alpha cutout) and thin branch tubes shadow better front-and-back, matching
    // dryad's depth materials which don't cull. Avoids peter-panning on thin geo.
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .depth_clamp_enable(false)
        .rasterizer_discard_enable(false)
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .depth_bias_enable(false)
        .line_width(1.0);

    // Shadow map is single-sample.
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(vk::SampleCountFlags::TYPE_1)
        .sample_shading_enable(false);

    // Reversed-Z: GREATER, depth-WRITE on (this IS the depth we sample later).
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::GREATER)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);

    // No color attachment.
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(&[]);

    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    let depth_format = vk::Format::D32_SFLOAT;
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .depth_attachment_format(depth_format); // no color formats

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .push_next(&mut rendering_info);

    let pipelines = unsafe {
        device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            std::slice::from_ref(&pipeline_info),
            None,
        )
    }
    .map_err(|(_, e)| RhiError::Vulkan(e))?;

    Ok(OwnedPipeline {
        pipeline: pipelines[0],
        layout,
    })
}

/// Create the flora-OWNED shadow depth image (D32, SHADOW_MAP_SIZE²,
/// DEPTH_STENCIL_ATTACHMENT | SAMPLED) + a DEPTH view, backing it with a single
/// raw device allocation. Returns (image, memory, view).
///
/// ponytail: one persistent image → raw vkAllocateMemory with a manual
/// device-local memory-type search is shorter than threading the gpu-allocator
/// crate into enki-app. The RHI exposes no public image-create, so flora owns it.
fn create_shadow_image(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), RhiError> {
    let format = vk::Format::D32_SFLOAT;
    let extent = vk::Extent3D {
        width: SHADOW_MAP_SIZE,
        height: SHADOW_MAP_SIZE,
        depth: 1,
    };
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&image_info, None) }.map_err(RhiError::Vulkan)?;

    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let mem_type = find_memory_type(
        &mem_props,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| RhiError::Other("no device-local memory for shadow map".into()))?;

    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }.map_err(RhiError::Vulkan)?;
    unsafe { device.bind_image_memory(image, memory, 0) }.map_err(RhiError::Vulkan)?;

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    let view = unsafe { device.create_image_view(&view_info, None) }.map_err(RhiError::Vulkan)?;

    Ok((image, memory, view))
}

fn find_memory_type(
    props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    (0..props.memory_type_count).find(|&i| {
        (type_bits & (1 << i)) != 0
            && props.memory_types[i as usize]
                .property_flags
                .contains(flags)
    })
}

/// Create the COMPARISON sampler the PCF lookup uses. Reversed-Z: the depth map
/// stores GREATER-closer, so the lit test is "receiver depth >= caster depth" →
/// `GREATER_OR_EQUAL`. LINEAR filtering enables hardware 2×2 PCF per tap.
/// CLAMP_TO_EDGE so off-map samples read the border depth (outside the frustum
/// the shader already early-outs to lit, so the clamp only matters at the rim).
fn create_shadow_sampler(device: &ash::Device) -> Result<vk::Sampler, RhiError> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .compare_enable(true)
        .compare_op(vk::CompareOp::GREATER_OR_EQUAL)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_WHITE);
    unsafe { device.create_sampler(&info, None) }.map_err(RhiError::Vulkan)
}

/// Write the three set1 bindings: shadow depth view (SAMPLED_IMAGE), comparison
/// sampler (SAMPLER), and this frame's ShadowUniforms UBO.
fn write_shadow_descriptors(
    device: &ash::Device,
    set: vk::DescriptorSet,
    view: vk::ImageView,
    sampler: vk::Sampler,
    ubo: vk::Buffer,
    ubo_size: u64,
) {
    let image_info = vk::DescriptorImageInfo::default()
        .image_view(view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let sampler_info = vk::DescriptorImageInfo::default().sampler(sampler);
    let buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(ubo)
        .offset(0)
        .range(ubo_size);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&image_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(std::slice::from_ref(&sampler_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(2)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info)),
        // Cascade-1/2 slots default to the same (self-shadow) view so they are valid
        // for the viewer (cascade_count = 1 → never sampled). The planet re-points
        // them at the real CSM maps via `set_shadow_cascade_views`.
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(10)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&image_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(11)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&image_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

/// Write set1/binding 6: the per-frame wind bone-matrix STORAGE buffer.
fn write_bone_descriptor(
    device: &ash::Device,
    set: vk::DescriptorSet,
    buffer: vk::Buffer,
    size: u64,
) {
    let info = vk::DescriptorBufferInfo::default()
        .buffer(buffer)
        .offset(0)
        .range(size);
    let write = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(6)
        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
        .buffer_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };
}

// ── IBL (image-based lighting) build helpers ─────────────────────────────────

/// Pack an f32 into an IEEE-754 binary16 (half) bit pattern. // ponytail: a tiny
/// inline packer avoids adding the `half` crate as a direct dep just to fill an
/// RGBA16F upload; round-to-nearest-even isn't needed for an env map (truncation
/// is imperceptible), so this is the short truncating form with overflow→inf.
fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mant = bits & 0x007f_ffff;
    if exp <= 0 {
        // Subnormal/underflow → flush to signed zero (env radiance this small is 0).
        sign
    } else if exp >= 0x1f {
        // Overflow / inf / nan → half-inf (keeps bright pixels finite-ish).
        sign | 0x7c00
    } else {
        sign | ((exp as u16) << 10) | ((mant >> 13) as u16)
    }
}

/// Create the IBL equirect mip-chain image (R16G16B16A16_SFLOAT, `mip_levels`)
/// + a 2D view spanning all mips, backed by one raw device allocation. Mirrors
/// `create_shadow_image` but color/sampled with mips. Returns (image, mem, view).
fn create_ibl_image(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), RhiError> {
    let format = vk::Format::R16G16B16A16_SFLOAT;
    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(extent)
        .mip_levels(mip_levels)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&image_info, None) }.map_err(RhiError::Vulkan)?;

    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let mem_type = find_memory_type(
        &mem_props,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| RhiError::Other("no device-local memory for IBL image".into()))?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }.map_err(RhiError::Vulkan)?;
    unsafe { device.bind_image_memory(image, memory, 0) }.map_err(RhiError::Vulkan)?;

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(mip_levels)
                .base_array_layer(0)
                .layer_count(1),
        );
    let view = unsafe { device.create_image_view(&view_info, None) }.map_err(RhiError::Vulkan)?;
    Ok((image, memory, view))
}

/// LINEAR trilinear sampler for the equirect mip lookup. REPEAT in U (longitude
/// wraps) + CLAMP in V (the poles don't), mip LOD across the full chain. NOT a
/// comparison sampler (the IBL texture is color radiance, not depth).
fn create_ibl_sampler(device: &ash::Device, max_lod: f32) -> Result<vk::Sampler, RhiError> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .min_lod(0.0)
        .max_lod(max_lod)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK);
    unsafe { device.create_sampler(&info, None) }.map_err(RhiError::Vulkan)
}

/// Upload the CPU-baked equirect mip chain into the IBL image: one host-visible
/// staging buffer holding every mip as RGBA16F, then a one-shot command buffer
/// that barriers UNDEFINED→TRANSFER_DST, copies each mip, then barriers the whole
/// chain →SHADER_READ_ONLY. Mirrors the egui texture-upload pattern (a transient
/// command buffer from the frame-0 pool + a fenced submit on the graphics queue).
///
/// // ponytail: a single staging buffer + one fenced submit is the shortest
/// upload that's correct; no persistent transfer queue, no async — this runs once
/// at startup so a blocking wait is fine.
fn upload_ibl_mips(
    rhi: &mut Rhi,
    image: vk::Image,
    mips: &[crate::flora_ibl::Mip],
) -> Result<(), RhiError> {
    let device = rhi.device_handle();

    // Pack every mip's f32 RGBA into contiguous f16, recording each mip's byte
    // offset for the per-level copy.
    let mut staging_bytes: Vec<u8> = Vec::new();
    let mut offsets: Vec<u64> = Vec::with_capacity(mips.len());
    for mip in mips {
        offsets.push(staging_bytes.len() as u64);
        for &f in &mip.rgba {
            staging_bytes.extend_from_slice(&f32_to_f16_bits(f).to_le_bytes());
        }
    }

    // Host-visible staging buffer (TRANSFER_SRC), filled with the f16 mip data.
    let staging = rhi.create_gpu_buffer(
        staging_bytes.len() as u64,
        true,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    rhi.write_storage_bytes(staging, &staging_bytes)?;
    let staging_buf = rhi.vk_buffer(staging)?;

    // A transient command buffer from frame slot 0's pool (valid at startup; the
    // frame loop hasn't begun). Mirrors how egui's set_textures records uploads.
    let pool = rhi.command_pool(0);
    let queue = rhi.queue_handle();
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc) }
        .map_err(RhiError::Vulkan)?[0];

    let mip_levels = mips.len() as u32;
    let begin = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .begin_command_buffer(cmd, &begin)
            .map_err(RhiError::Vulkan)?;

        // UNDEFINED → TRANSFER_DST_OPTIMAL for every mip.
        let to_dst = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&to_dst),
        );

        // Copy each mip from its staging offset.
        for (level, mip) in mips.iter().enumerate() {
            let region = vk::BufferImageCopy::default()
                .buffer_offset(offsets[level])
                .buffer_row_length(0) // tightly packed
                .buffer_image_height(0)
                .image_subresource(
                    vk::ImageSubresourceLayers::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .mip_level(level as u32)
                        .base_array_layer(0)
                        .layer_count(1),
                )
                .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                .image_extent(vk::Extent3D {
                    width: mip.w as u32,
                    height: mip.h as u32,
                    depth: 1,
                });
            device.cmd_copy_buffer_to_image(
                cmd,
                staging_buf,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
            );
        }

        // TRANSFER_DST → SHADER_READ_ONLY for the whole chain.
        let to_read = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mip_levels)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&to_read),
        );

        device.end_command_buffer(cmd).map_err(RhiError::Vulkan)?;

        // Fenced submit + wait (startup one-shot; blocking is fine).
        let fence_ci = vk::FenceCreateInfo::default();
        let fence = device.create_fence(&fence_ci, None).map_err(RhiError::Vulkan)?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        device
            .queue_submit(queue, std::slice::from_ref(&submit), fence)
            .map_err(RhiError::Vulkan)?;
        device
            .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
            .map_err(RhiError::Vulkan)?;
        device.destroy_fence(fence, None);
        device.free_command_buffers(pool, std::slice::from_ref(&cmd));
    }

    // Reclaim the staging buffer (RHI-owned; freed at RHI shutdown otherwise, but
    // this one-shot upload buffer is large, so drop it now).
    rhi.destroy_buffer(staging);
    Ok(())
}

/// Create a single-mip 2D color image (`format`) + view, raw DEVICE_LOCAL alloc,
/// SAMPLED | TRANSFER_DST. Used for the CPU-baked leaf color (sRGB) + normal
/// (UNORM) cluster textures. Clone of `create_ibl_image` with `mip_levels=1`.
fn create_leaf_image(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    width: u32,
    height: u32,
    format: vk::Format,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), RhiError> {
    let extent = vk::Extent3D {
        width,
        height,
        depth: 1,
    };
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(extent)
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&image_info, None) }.map_err(RhiError::Vulkan)?;

    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let mem_type = find_memory_type(
        &mem_props,
        reqs.memory_type_bits,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )
    .ok_or_else(|| RhiError::Other("no device-local memory for leaf image".into()))?;
    let alloc_info = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc_info, None) }.map_err(RhiError::Vulkan)?;
    unsafe { device.bind_image_memory(image, memory, 0) }.map_err(RhiError::Vulkan)?;

    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    let view = unsafe { device.create_image_view(&view_info, None) }.map_err(RhiError::Vulkan)?;
    Ok((image, memory, view))
}

/// LINEAR / CLAMP_TO_EDGE sampler (single LOD) shared by the leaf color + normal
/// textures. dryad samples the leaf sprite as a clamped alpha-cutout sprite.
fn create_leaf_sampler(device: &ash::Device) -> Result<vk::Sampler, RhiError> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .min_lod(0.0)
        .max_lod(0.0)
        .border_color(vk::BorderColor::FLOAT_TRANSPARENT_BLACK);
    unsafe { device.create_sampler(&info, None) }.map_err(RhiError::Vulkan)
}

/// Upload a single-level RGBA8 buffer into `image` (width×height): one host-
/// visible TRANSFER_SRC staging buffer, a one-shot fenced command buffer doing
/// UNDEFINED→TRANSFER_DST, copy, →SHADER_READ_ONLY. The RGBA8 bytes go straight
/// in (no f16 packing). Mirrors `upload_ibl_mips` for a single mip.
///
/// `old_layout` is UNDEFINED on the first upload and SHADER_READ_ONLY_OPTIMAL on
/// a re-upload (reseed), so the barrier source matches the image's current state.
fn upload_leaf_image(
    rhi: &mut Rhi,
    image: vk::Image,
    width: u32,
    height: u32,
    rgba: &[u8],
    old_layout: vk::ImageLayout,
) -> Result<(), RhiError> {
    let device = rhi.device_handle();

    let staging = rhi.create_gpu_buffer(
        rgba.len() as u64,
        true,
        vk::BufferUsageFlags::TRANSFER_SRC,
    )?;
    rhi.write_storage_bytes(staging, rgba)?;
    let staging_buf = rhi.vk_buffer(staging)?;

    let pool = rhi.command_pool(0);
    let queue = rhi.queue_handle();
    let alloc = vk::CommandBufferAllocateInfo::default()
        .command_pool(pool)
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_buffer_count(1);
    let cmd = unsafe { device.allocate_command_buffers(&alloc) }.map_err(RhiError::Vulkan)?[0];

    let range = vk::ImageSubresourceRange::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_mip_level(0)
        .level_count(1)
        .base_array_layer(0)
        .layer_count(1);

    let begin = vk::CommandBufferBeginInfo::default()
        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
    unsafe {
        device
            .begin_command_buffer(cmd, &begin)
            .map_err(RhiError::Vulkan)?;

        let src_access = if old_layout == vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL {
            vk::AccessFlags::SHADER_READ
        } else {
            vk::AccessFlags::empty()
        };
        let to_dst = vk::ImageMemoryBarrier::default()
            .old_layout(old_layout)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .src_access_mask(src_access)
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .subresource_range(range);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&to_dst),
        );

        let region = vk::BufferImageCopy::default()
            .buffer_offset(0)
            .buffer_row_length(0)
            .buffer_image_height(0)
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .mip_level(0)
                    .base_array_layer(0)
                    .layer_count(1),
            )
            .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
            .image_extent(vk::Extent3D {
                width,
                height,
                depth: 1,
            });
        device.cmd_copy_buffer_to_image(
            cmd,
            staging_buf,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            std::slice::from_ref(&region),
        );

        let to_read = vk::ImageMemoryBarrier::default()
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(image)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .subresource_range(range);
        device.cmd_pipeline_barrier(
            cmd,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&to_read),
        );

        device.end_command_buffer(cmd).map_err(RhiError::Vulkan)?;

        let fence_ci = vk::FenceCreateInfo::default();
        let fence = device.create_fence(&fence_ci, None).map_err(RhiError::Vulkan)?;
        let submit = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        device
            .queue_submit(queue, std::slice::from_ref(&submit), fence)
            .map_err(RhiError::Vulkan)?;
        device
            .wait_for_fences(std::slice::from_ref(&fence), true, u64::MAX)
            .map_err(RhiError::Vulkan)?;
        device.destroy_fence(fence, None);
        device.free_command_buffers(pool, std::slice::from_ref(&cmd));
    }

    rhi.destroy_buffer(staging);
    Ok(())
}

/// Add the leaf-texture bindings (7 = color image, 8 = normal image, 9 = linear
/// sampler) to an already-written set1. The leaf textures are STATIC PER TREE —
/// the same handles (image views are stable across reseed; only the pixel data is
/// re-uploaded into the SAME images) go into every frame's set1.
fn write_leaf_tex_descriptors(
    device: &ash::Device,
    set: vk::DescriptorSet,
    color_view: vk::ImageView,
    normal_view: vk::ImageView,
    sampler: vk::Sampler,
) {
    let color_info = vk::DescriptorImageInfo::default()
        .image_view(color_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let normal_info = vk::DescriptorImageInfo::default()
        .image_view(normal_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let sampler_info = vk::DescriptorImageInfo::default().sampler(sampler);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(7)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&color_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(8)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&normal_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(9)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(std::slice::from_ref(&sampler_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

/// Write the impostor atlas (set1/binding 12, SAMPLED_IMAGE). The view is bound in
/// SHADER_READ_ONLY layout; the image is cleared-UNDEFINED until the one-shot bake
/// transitions it (the descriptor is valid to write before the image is filled —
/// it's only SAMPLED after the bake's COLOR→SHADER_READ barrier).
fn write_impostor_atlas_descriptor(
    device: &ash::Device,
    set: vk::DescriptorSet,
    atlas_view: vk::ImageView,
) {
    let info = vk::DescriptorImageInfo::default()
        .image_view(atlas_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let w = vk::WriteDescriptorSet::default()
        .dst_set(set)
        .dst_binding(12)
        .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
        .image_info(std::slice::from_ref(&info));
    unsafe { device.update_descriptor_sets(std::slice::from_ref(&w), &[]) };
}

/// Allocate a small dedicated D32 depth image for the impostor bake pass (its own
/// state, no contention with the swapchain-sized scene depth). Mirrors
/// `create_shadow_image`'s raw allocation but sized to the atlas.
fn create_bake_depth(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    width: u32,
    height: u32,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), RhiError> {
    let format = vk::Format::D32_SFLOAT;
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width, height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&image_info, None) }.map_err(RhiError::Vulkan)?;
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_type = find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or_else(|| RhiError::Other("no device-local memory for bake depth".into()))?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc, None) }.map_err(RhiError::Vulkan)?;
    unsafe { device.bind_image_memory(image, memory, 0) }.map_err(RhiError::Vulkan)?;
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    let view = unsafe { device.create_image_view(&view_info, None) }.map_err(RhiError::Vulkan)?;
    Ok((image, memory, view))
}

/// Add the IBL bindings (3 = env mip image, 4 = linear sampler, 5 = SH UBO) to an
/// already-shadow-written set1. The IBL resources are STATIC, so the same handles
/// go into every frame's set1.
fn write_ibl_descriptors(
    device: &ash::Device,
    set: vk::DescriptorSet,
    env_view: vk::ImageView,
    env_sampler: vk::Sampler,
    sh_ubo: vk::Buffer,
    sh_ubo_size: u64,
) {
    let image_info = vk::DescriptorImageInfo::default()
        .image_view(env_view)
        .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
    let sampler_info = vk::DescriptorImageInfo::default().sampler(env_sampler);
    let buffer_info = vk::DescriptorBufferInfo::default()
        .buffer(sh_ubo)
        .offset(0)
        .range(sh_ubo_size);
    let writes = [
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(3)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&image_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(4)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(std::slice::from_ref(&sampler_info)),
        vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(5)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .buffer_info(std::slice::from_ref(&buffer_info)),
    ];
    unsafe { device.update_descriptor_sets(&writes, &[]) };
}

// ── POST build helpers (offscreen HDR + bloom chain + fullscreen pipelines) ───

impl PostTargets {
    fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_image_view(self.scene_msaa_view, None);
            device.destroy_image(self.scene_msaa_image, None);
            device.free_memory(self.scene_msaa_memory, None);
            device.destroy_image_view(self.scene_depth_view, None);
            device.destroy_image(self.scene_depth_image, None);
            device.free_memory(self.scene_depth_memory, None);
        }
        self.scene_resolve.destroy(device);
        self.bloom_bright.destroy(device);
        self.bloom_h.destroy(device);
        self.bloom_v.destroy(device);
        self.bloom_composite.destroy(device);
    }
}

/// Build the resolution-independent POST machinery + the first set of targets.
fn build_post(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    swapchain_format: vk::Format,
    samples: vk::SampleCountFlags,
    extent: vk::Extent2D,
    frames: usize,
) -> Result<Post, RhiError> {
    // Capture the scene MSAA sample count for resize-time target recreation.
    SCENE_SAMPLES.with(|s| s.set(samples));
    let module = compile_module(device, POST_WGSL)?;
    let set_layout = create_post_set_layout(device)?;
    let pipeline_layout = create_post_pipeline_layout(device, set_layout)?;
    let sampler = create_post_sampler(device)?;

    // bright/blur/composite: single-sample RGBA16F, no depth.
    let bright = build_fullscreen_pipeline(
        device, pipeline_layout, module, "fs_bright", HDR_FORMAT, vk::SampleCountFlags::TYPE_1,
    )?;
    let blur = build_fullscreen_pipeline(
        device, pipeline_layout, module, "fs_blur", HDR_FORMAT, vk::SampleCountFlags::TYPE_1,
    )?;
    let composite = build_fullscreen_pipeline(
        device, pipeline_layout, module, "fs_composite", HDR_FORMAT, vk::SampleCountFlags::TYPE_1,
    )?;
    // output: runs INSIDE enki's MSAA instance → swapchain format + msaa_samples.
    let output = build_fullscreen_pipeline(
        device, pipeline_layout, module, "fs_output", swapchain_format, samples,
    )?;

    // One transient-set pool PER frame-in-flight (the pool is reset+reused each
    // frame; sharing one across FIFs would reset it while the other frame's
    // submission is still executing — VUID-vkResetDescriptorPool-00313).
    let mut pools = Vec::with_capacity(frames);
    for _ in 0..frames {
        pools.push(create_post_pool(device)?);
    }
    let targets = create_post_targets(device, instance, physical_device, extent)?;

    Ok(Post {
        module,
        set_layout,
        pipeline_layout,
        sampler,
        bright,
        blur,
        composite,
        output,
        pools,
        targets,
    })
}

/// set0 for the post passes: src tex (b0) + src sampler (b1) + aux tex (b2) +
/// aux sampler (b3), all FRAGMENT-visible.
fn create_post_set_layout(device: &ash::Device) -> Result<vk::DescriptorSetLayout, RhiError> {
    let bindings = [
        vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(2)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
        vk::DescriptorSetLayoutBinding::default()
            .binding(3)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT),
    ];
    let ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
    unsafe { device.create_descriptor_set_layout(&ci, None) }.map_err(RhiError::Vulkan)
}

/// Post pipeline layout = [set0] + a 32-byte (2×vec4) VS|FS push range.
fn create_post_pipeline_layout(
    device: &ash::Device,
    set_layout: vk::DescriptorSetLayout,
) -> Result<vk::PipelineLayout, RhiError> {
    let push = vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
        .offset(0)
        .size(std::mem::size_of::<PostPush>() as u32);
    let sets = [set_layout];
    let ci = vk::PipelineLayoutCreateInfo::default()
        .set_layouts(&sets)
        .push_constant_ranges(std::slice::from_ref(&push));
    unsafe { device.create_pipeline_layout(&ci, None) }.map_err(RhiError::Vulkan)
}

/// LINEAR CLAMP sampler for the post taps (upscale + neighbor reads). Mip LOD
/// spans the bloom chain so the composite can read explicit LODs 0..4.
fn create_post_sampler(device: &ash::Device) -> Result<vk::Sampler, RhiError> {
    let info = vk::SamplerCreateInfo::default()
        .mag_filter(vk::Filter::LINEAR)
        .min_filter(vk::Filter::LINEAR)
        .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
        .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        .min_lod(0.0)
        .max_lod(BLOOM_MIPS as f32)
        .border_color(vk::BorderColor::FLOAT_OPAQUE_BLACK);
    unsafe { device.create_sampler(&info, None) }.map_err(RhiError::Vulkan)
}

/// Pool sized for the per-frame transient post sets: bright(1) + 5 mips×2 blur(10)
/// + composite(1) + output(1) = 13 sets/frame; round up generously. Reset each
/// frame in `run_bloom`. // ponytail: a single oversized pool + per-frame reset is
/// shorter than a precise per-pass ring; the sets are write-once-use-once.
fn create_post_pool(device: &ash::Device) -> Result<vk::DescriptorPool, RhiError> {
    const MAX_SETS: u32 = 32;
    let sizes = [
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLED_IMAGE)
            .descriptor_count(MAX_SETS * 2),
        vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::SAMPLER)
            .descriptor_count(MAX_SETS * 2),
    ];
    let ci = vk::DescriptorPoolCreateInfo::default()
        .max_sets(MAX_SETS)
        .pool_sizes(&sizes);
    unsafe { device.create_descriptor_pool(&ci, None) }.map_err(RhiError::Vulkan)
}

/// Allocate the resize-dependent POST targets at `extent`.
fn create_post_targets(
    device: &ash::Device,
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
    extent: vk::Extent2D,
) -> Result<PostTargets, RhiError> {
    let mem_props = unsafe { instance.get_physical_device_memory_properties(physical_device) };
    let samples = scene_msaa_samples();

    // Scene HDR MSAA color (RGBA16F, msaa_samples) — sky/ground/tree target.
    let (scene_msaa_image, scene_msaa_memory, scene_msaa_view) = create_attachment_image(
        device, &mem_props, HDR_FORMAT, extent, samples,
        vk::ImageUsageFlags::COLOR_ATTACHMENT, vk::ImageAspectFlags::COLOR,
    )?;
    // Scene depth (D32, msaa_samples).
    let (scene_depth_image, scene_depth_memory, scene_depth_view) = create_attachment_image(
        device, &mem_props, vk::Format::D32_SFLOAT, extent, samples,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT, vk::ImageAspectFlags::DEPTH,
    )?;
    // Resolved single-sample HDR (COLOR resolve target + SAMPLED).
    let scene_resolve = create_color_image(
        device, &mem_props, extent, 1,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED,
    )?;

    let half = vk::Extent2D {
        width: (extent.width / 2).max(1),
        height: (extent.height / 2).max(1),
    };
    let bloom_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
    let bloom_bright = create_color_image(device, &mem_props, half, 1, bloom_usage)?;
    let bloom_h = create_color_image(device, &mem_props, half, BLOOM_MIPS as u32, bloom_usage)?;
    let bloom_v = create_color_image(device, &mem_props, half, BLOOM_MIPS as u32, bloom_usage)?;
    let bloom_composite = create_color_image(device, &mem_props, half, 1, bloom_usage)?;

    Ok(PostTargets {
        scene_msaa_image,
        scene_msaa_memory,
        scene_msaa_view,
        scene_depth_image,
        scene_depth_memory,
        scene_depth_view,
        scene_resolve,
        bloom_bright,
        bloom_h,
        bloom_v,
        bloom_composite,
        extent,
        initialized: false,
    })
}

/// The scene MSAA sample count. flora's scene pipelines are built at
/// `rhi.msaa_samples()`; the offscreen MSAA target must match. We thread the
/// value through targets creation via this single source (set in build_post via
/// the pipelines), but since `create_post_targets` is also called on resize
/// without the Rhi, we re-read it from a module-level cell set at build time.
fn scene_msaa_samples() -> vk::SampleCountFlags {
    SCENE_SAMPLES.with(|s| s.get())
}

thread_local! {
    /// The scene MSAA sample count, captured once at `build_post`. // ponytail:
    /// a thread-local avoids threading `samples` through `resize_post`/the targets
    /// struct for a value that never changes after construction.
    static SCENE_SAMPLES: std::cell::Cell<vk::SampleCountFlags> =
        const { std::cell::Cell::new(vk::SampleCountFlags::TYPE_1) };
}

/// Create a single-mip MSAA/depth attachment image (no SAMPLED). Returns
/// (image, memory, full view).
#[allow(clippy::too_many_arguments)]
fn create_attachment_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    format: vk::Format,
    extent: vk::Extent2D,
    samples: vk::SampleCountFlags,
    usage: vk::ImageUsageFlags,
    aspect: vk::ImageAspectFlags,
) -> Result<(vk::Image, vk::DeviceMemory, vk::ImageView), RhiError> {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(format)
        .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 })
        .mip_levels(1)
        .array_layers(1)
        .samples(samples)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&image_info, None) }.map_err(RhiError::Vulkan)?;
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_type = find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or_else(|| RhiError::Other("no device-local memory for post attachment".into()))?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc, None) }.map_err(RhiError::Vulkan)?;
    unsafe { device.bind_image_memory(image, memory, 0) }.map_err(RhiError::Vulkan)?;
    let view_info = vk::ImageViewCreateInfo::default()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(format)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );
    let view = unsafe { device.create_image_view(&view_info, None) }.map_err(RhiError::Vulkan)?;
    Ok((image, memory, view))
}

/// Create a 1-sample RGBA16F color image with `mips` levels (COLOR+SAMPLED),
/// returning a full-chain sample view + one render view per mip.
fn create_color_image(
    device: &ash::Device,
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    base: vk::Extent2D,
    mips: u32,
    usage: vk::ImageUsageFlags,
) -> Result<OwnedImage, RhiError> {
    let image_info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(HDR_FORMAT)
        .extent(vk::Extent3D { width: base.width, height: base.height, depth: 1 })
        .mip_levels(mips)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = unsafe { device.create_image(&image_info, None) }.map_err(RhiError::Vulkan)?;
    let reqs = unsafe { device.get_image_memory_requirements(image) };
    let mem_type = find_memory_type(mem_props, reqs.memory_type_bits, vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .ok_or_else(|| RhiError::Other("no device-local memory for post color image".into()))?;
    let alloc = vk::MemoryAllocateInfo::default()
        .allocation_size(reqs.size)
        .memory_type_index(mem_type);
    let memory = unsafe { device.allocate_memory(&alloc, None) }.map_err(RhiError::Vulkan)?;
    unsafe { device.bind_image_memory(image, memory, 0) }.map_err(RhiError::Vulkan)?;

    let sample_view = {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(HDR_FORMAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(0)
                    .level_count(mips)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        unsafe { device.create_image_view(&info, None) }.map_err(RhiError::Vulkan)?
    };
    let mut mip_views = Vec::with_capacity(mips as usize);
    for level in 0..mips {
        let info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(HDR_FORMAT)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .base_mip_level(level)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        mip_views.push(unsafe { device.create_image_view(&info, None) }.map_err(RhiError::Vulkan)?);
    }
    Ok(OwnedImage {
        image,
        memory,
        sample_view,
        mip_views,
    })
}

/// Build a fullscreen-triangle pipeline (NO vertex input, depth test OFF) for the
/// post passes. `vs_fullscreen` + the given FS, into `color_format` at `samples`.
/// Distinct from `build_pipeline` (which hardcodes the 4×vec3 scene vertex layout
/// + reversed-Z depth). The output pass runs in enki's MSAA depth instance, so it
/// declares the D32 depth format (test/write OFF) to stay render-pass-compatible.
fn build_fullscreen_pipeline(
    device: &ash::Device,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    fs_entry: &str,
    color_format: vk::Format,
    samples: vk::SampleCountFlags,
) -> Result<OwnedPipeline, RhiError> {
    let vs_name = CString::new("vs_fullscreen").unwrap();
    let fs_name = CString::new(fs_entry).unwrap();
    let stages = [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(module)
            .name(&vs_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(module)
            .name(&fs_name),
    ];
    // No vertex input — the VS synthesizes the triangle from vertex_index.
    let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
    let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
    let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
        .polygon_mode(vk::PolygonMode::FILL)
        .cull_mode(vk::CullModeFlags::NONE)
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
        .line_width(1.0);
    let multisample = vk::PipelineMultisampleStateCreateInfo::default()
        .rasterization_samples(samples)
        .sample_shading_enable(false);
    let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
        .depth_test_enable(false)
        .depth_write_enable(false)
        .depth_compare_op(vk::CompareOp::ALWAYS)
        .depth_bounds_test_enable(false)
        .stencil_test_enable(false);
    let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
        .blend_enable(false)
        .color_write_mask(vk::ColorComponentFlags::RGBA);
    let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
        .logic_op_enable(false)
        .attachments(std::slice::from_ref(&blend_attachment));
    let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
    let viewport_state = vk::PipelineViewportStateCreateInfo::default()
        .viewport_count(1)
        .scissor_count(1);

    // The output pass runs inside enki's begin_rendering (which binds a D32 depth
    // attachment), so it must declare the same depth format for compatibility. The
    // single-sample bloom passes have no depth attachment → UNDEFINED depth format.
    let has_depth = samples != vk::SampleCountFlags::TYPE_1;
    let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
        .color_attachment_formats(std::slice::from_ref(&color_format));
    if has_depth {
        rendering_info = rendering_info.depth_attachment_format(vk::Format::D32_SFLOAT);
    }

    let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
        .stages(&stages)
        .vertex_input_state(&vertex_input)
        .input_assembly_state(&input_assembly)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization)
        .multisample_state(&multisample)
        .depth_stencil_state(&depth_stencil)
        .color_blend_state(&color_blend)
        .dynamic_state(&dynamic_state)
        .layout(layout)
        .push_next(&mut rendering_info);
    let pipelines = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), std::slice::from_ref(&pipeline_info), None)
    }
    .map_err(|(_, e)| RhiError::Vulkan(e))?;
    Ok(OwnedPipeline { pipeline: pipelines[0], layout })
}

/// Set a full-extent viewport + scissor on the raw command buffer.
fn set_viewport_scissor(device: &ash::Device, cmd: vk::CommandBuffer, extent: vk::Extent2D) {
    let viewport = vk::Viewport {
        x: 0.0,
        y: 0.0,
        width: extent.width as f32,
        height: extent.height as f32,
        min_depth: 0.0,
        max_depth: 1.0,
    };
    let scissor = vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent };
    unsafe {
        device.cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport));
        device.cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
    }
}

/// Record a generic single-subresource image memory barrier.
#[allow(clippy::too_many_arguments)]
fn barrier_image(
    device: &ash::Device,
    cmd: vk::CommandBuffer,
    image: vk::Image,
    aspect: vk::ImageAspectFlags,
    base_mip: u32,
    mip_count: u32,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_access: vk::AccessFlags,
    dst_access: vk::AccessFlags,
    src_stage: vk::PipelineStageFlags,
    dst_stage: vk::PipelineStageFlags,
) {
    let barrier = vk::ImageMemoryBarrier::default()
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .src_access_mask(src_access)
        .dst_access_mask(dst_access)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .base_mip_level(base_mip)
                .level_count(mip_count)
                .base_array_layer(0)
                .layer_count(1),
        );
    unsafe {
        device.cmd_pipeline_barrier(
            cmd,
            src_stage,
            dst_stage,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            std::slice::from_ref(&barrier),
        );
    }
}

#[cfg(test)]
mod tests {
    //! SPIR-V emission gate: the enki-flora `tests/shader.rs` test only does
    //! parse+validate (its naga has no `spv-out`). enki-app's naga DOES, so we
    //! run the full parse→validate→SPIR-V path here — the same path
    //! `compile_module` takes at runtime — to catch backend errors (e.g. a
    //! depth-compare texture sample the SPIR-V writer rejects) without a GPU.

    fn emit_spv(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name} parse:\n{}", e.emit_to_string(src)));
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{name} validate: {e:?}"));
        let opts = naga::back::spv::Options {
            lang_version: (1, 3),
            flags: naga::back::spv::WriterFlags::DEBUG,
            ..Default::default()
        };
        let words = naga::back::spv::write_vec(&module, &info, &opts, None)
            .unwrap_or_else(|e| panic!("{name} spv emit: {e:?}"));
        assert!(words.len() > 16, "{name} SPIR-V too small");
    }

    #[test]
    fn flora_shaders_emit_spirv() {
        emit_spv("flora", super::FLORA_WGSL);
        emit_spv("staging", super::STAGING_WGSL);
        emit_spv("post", super::POST_WGSL);
    }
}

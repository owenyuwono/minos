// terrain.wgsl — opaque terrain pass
//
// Pipeline contract (integration engineer):
//   - Descriptor set 0, binding 0: FrameUniforms uniform buffer
//   - Push constants: ChunkPush (80 bytes, stages VERTEX | FRAGMENT)
//   - Vertex buffers:
//       slot 0 : positions    vec3<f32>   @location(0)
//       slot 1 : normals      vec3<f32>   @location(1)
//       slot 2 : colors       vec3<f32>   @location(2)
//       slot 3 : plate_colors vec3<f32>   @location(3)  (may duplicate colors when unused)
//   - Index buffer: u32, triangle-list, CCW winding viewed from outside
//   - Depth: D32_SFLOAT, cleared to 0.0, compare GREATER (reversed-Z)
//   - Render mode: opaque, FILL, cull BACK, no alpha blend
//   - Swapchain format: *_SRGB  — ACES tonemap applied in shader for mode 0;
//     debug modes 1–3 output raw data values without ACES (they are data
//     visualizations, not HDR radiance). The hardware sRGB OETF still applies to
//     all modes via the _SRGB surface — no manual encoding needed in any mode.
//
// ── Material modes ─────────────────────────────────────────────────────────────
//   0 — Normal-lit   : ambient + hemisphere + 2 directional, vertex-color albedo,
//                      ACES filmic tonemap. (Mode 0 is the only lit path.)
//   1 — Debug normals: world-space normal → RGB visualisation (n * 0.5 + 0.5).
//                      No ACES — raw [0,1] value written directly.
//   2 — Plate-color  : plate_color vertex attribute output directly.
//                      No ACES — raw sRGB tint written directly.
//   3 — Heightmap    : elevation above PLANET_RADIUS remapped to grayscale [0,1].
//                      No ACES — raw scalar written directly.

// ── Uniforms ────────────────────────────────────────────────────────────────

struct FrameUniforms {
    view_proj  : mat4x4<f32>,
    camera_pos : vec4<f32>,   // w unused
    sun0_dir   : vec4<f32>,   // w unused
    sun0_color : vec4<f32>,   // w unused
    sun1_dir   : vec4<f32>,   // w unused
    sun1_color : vec4<f32>,   // w unused
    hemi_sky   : vec4<f32>,   // w unused
    hemi_ground: vec4<f32>,   // w unused
    ambient    : vec4<f32>,   // x=y=z=intensity, w unused
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

struct ChunkPush {
    model         : mat4x4<f32>,
    material_mode : u32,
    _pad0         : u32,
    _pad1         : u32,
    _pad2         : u32,
}

var<immediate> pc: ChunkPush;

// ── Vertex stage ─────────────────────────────────────────────────────────────

struct VertexIn {
    @location(0) position   : vec3<f32>,
    @location(1) normal     : vec3<f32>,
    @location(2) color      : vec3<f32>,
    @location(3) plate_color: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos    : vec4<f32>,
    @location(0)       world_normal: vec3<f32>,
    @location(1)       albedo      : vec3<f32>,
    // world_pos and plate_color are forwarded for debug modes 2 and 3.
    // They are zero-cost when mode 0 is compiled by a real driver that
    // dead-strips unused interpolants, and harmless otherwise.
    @location(2)       world_pos   : vec3<f32>,
    @location(3)       plate_color : vec3<f32>,
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    // Camera-relative rendering: pc.model's translation column is
    // (object_origin_f64 − camera_pos_f64) cast to f32 (computed on the CPU
    // in ChunkPush::camera_relative).  frame.view_proj is proj × view_rotation_only
    // with the camera at the origin.  So this product runs entirely on small
    // near-origin coordinates — no f32 cancellation of large planetary coords.
    // Reversed-Z is handled entirely by the projection matrix (near→1, far→0)
    // and the GREATER depth compare op — no per-vertex depth math needed.
    let world_pos = pc.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = frame.view_proj * world_pos;

    // Transform normal by the normal matrix (transpose-inverse of model).
    // For uniform-scale models, the upper-left 3×3 of the model matrix suffices.
    // We normalise in the fragment shader.
    let m3 = mat3x3<f32>(
        pc.model[0].xyz,
        pc.model[1].xyz,
        pc.model[2].xyz,
    );
    out.world_normal = m3 * v.normal;

    // Material mode 0 uses vertex color as albedo.
    out.albedo = v.color;

    // Forward camera-relative position (xyz) and plate_color for debug modes.
    // world_pos.xyz here means "position relative to the camera", not absolute
    // world-space — consistent with the camera-relative rendering contract.
    out.world_pos   = world_pos.xyz;
    out.plate_color = v.plate_color;

    return out;
}

// ── Fragment stage ────────────────────────────────────────────────────────────

// Hemisphere ambient: linearly interpolates between ground and sky colors
// based on how much the surface normal points up (+Y).
fn hemisphere_ambient(n: vec3<f32>, sky: vec3<f32>, ground: vec3<f32>) -> vec3<f32> {
    let t = n.y * 0.5 + 0.5; // [0..1], 0=down, 1=up
    return mix(ground, sky, t);
}

// Lambert diffuse for one directional light.
fn directional_diffuse(
    n        : vec3<f32>,
    light_dir: vec3<f32>,  // world-space, pointing toward the light
    color    : vec3<f32>,
) -> vec3<f32> {
    let n_dot_l = max(dot(n, light_dir), 0.0);
    return color * n_dot_l;
}

// ACES filmic tonemapping (Narkowicz 2015 approximation).
// Input: linear HDR color. Output: linear [0..1] (hardware encodes to sRGB).
// Only used for mode 0 (normal-lit). Debug modes bypass this intentionally —
// they output data values that must not be remapped by a tone curve.
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Heightmap constants — must match the prototype planet parameters.
const PLANET_RADIUS: f32 = 50000.0;
const HEIGHT_SCALE : f32 = 1200.0;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);

    // ── Mode 0: Normal-lit ─────────────────────────────────────────────────────
    // Ambient isotropic + hemisphere + two directional lights, vertex-color
    // albedo, ACES filmic tonemap. This is the only mode that touches lighting
    // or the tonemap. The swapchain is _SRGB so the hardware applies the sRGB
    // OETF on write — no manual encoding here.
    if pc.material_mode == 0u {
        let albedo = in.albedo;

        let ambient_term = frame.ambient.xyz * albedo;
        let hemi_term    = hemisphere_ambient(n, frame.hemi_sky.xyz, frame.hemi_ground.xyz) * albedo;
        let diff0        = directional_diffuse(n, frame.sun0_dir.xyz, frame.sun0_color.xyz) * albedo;
        let diff1        = directional_diffuse(n, frame.sun1_dir.xyz, frame.sun1_color.xyz) * albedo;

        var lit = ambient_term + hemi_term + diff0 + diff1;
        lit = aces_filmic(lit);

        return vec4<f32>(lit, 1.0);
    }

    // ── Debug modes 1–3: no lighting, no ACES ─────────────────────────────────
    // These modes output raw data values that must read as-is from the image.
    // Applying ACES (or any tone curve) to a data visualization distorts the
    // encoded value, so it is intentionally omitted. The _SRGB swapchain still
    // applies its hardware OETF on write, which is acceptable for visual
    // debugging.

    // ── Mode 1: Debug normals ──────────────────────────────────────────────────
    // Map world-space normal to [0,1] RGB so every component is visible.
    // Output is independent of lighting and valid on any mesh geometry.
    // TODO: with real terrain this slot becomes LOD-level coloring.
    if pc.material_mode == 1u {
        let vis = n * 0.5 + vec3<f32>(0.5);
        return vec4<f32>(vis, 1.0);
    }

    // ── Mode 2: Plate-color ────────────────────────────────────────────────────
    // Output the tectonic plate tint directly. On placeholder geometry
    // plate_color equals the vertex color; with real terrain it carries the
    // per-plate tint from the generator.
    if pc.material_mode == 2u {
        return vec4<f32>(in.plate_color, 1.0);
    }

    // ── Mode 3: Heightmap (grayscale) ─────────────────────────────────────────
    // Elevation = distance from planet centre minus PLANET_RADIUS, divided by
    // HEIGHT_SCALE. This maps the nominal surface (radius = PLANET_RADIUS) to
    // ~0.0 and the maximum terrain height (~PLANET_RADIUS + HEIGHT_SCALE) to
    // ~1.0. The placeholder sphere has radius ≈ PLANET_RADIUS, so all fragments
    // land near 0.0 and the ramp shows as near-black — that is correct.
    // Clamp to [0,1] so any out-of-range geometry doesn't wrap or NaN.
    if pc.material_mode == 3u {
        let elevation = (length(in.world_pos) - PLANET_RADIUS) / HEIGHT_SCALE;
        let g = clamp(elevation, 0.0, 1.0);
        return vec4<f32>(g, g, g, 1.0);
    }

    // ── Fallback: out-of-range mode → bright magenta for easy identification ──
    return vec4<f32>(1.0, 0.0, 1.0, 1.0);
}

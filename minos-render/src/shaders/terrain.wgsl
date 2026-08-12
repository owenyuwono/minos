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
// ── View modes (unified with the Nanite nanite_draw.wgsl scheme) ───────────────
//   0  Lit      : biome albedo + ambient/hemisphere/2 suns + ACES (the only lit path).
//   1  Unlit    : biome albedo, flat (no lighting, no ACES).
//   2  Normal   : world-space normal → RGB (n * 0.5 + 0.5).
//   6  Plate    : per-plate tectonic tint (plate_color attribute), raw.
// Nanite-only modes 3/4/5 (triangle/cluster/LOD) and the data-channel modes
// 7/8/9/10 (height/material/wetness/volcano) are NOT yet supported on the classic
// quadtree path — its 4×vec3 vertex format (pos/normal/color/plate, hardcoded in
// minos-rhi) has no room for the elevation/material/wetness/volcanism scalars. They
// fall back to Lit here; full parity needs a 5th vertex channel through the minos-rhi
// streaming/pipeline layout. The Nanite path already supports all of them.

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
    dbg_id        : u32,   // per-mesh-unit id (leaf/cluster) — modes 3/4
    dbg_level     : u32,   // LOD level — mode 5
    _pad2         : u32,
}

var<immediate> pc: ChunkPush;

// ── Vertex stage ─────────────────────────────────────────────────────────────

struct VertexIn {
    @builtin(vertex_index) vid: u32,
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
    // Provoking-vertex index (flat) → a per-triangle seed for the Triangle view.
    @location(4) @interpolate(flat) vid : u32,
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
    out.vid         = v.vid;

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

// ── Geometry-debug colouring (modes 3/4/5) ────────────────────────────────────
// Distinct, well-separated colours per integer id. Copied from nanite_draw.wgsl so
// the voxel path's Triangle/Cluster/LOD views read the same as the old Nanite ones.
fn hash_u32(x: u32) -> u32 {
    var h = x * 0x9E3779B1u;
    h = h ^ (h >> 16u);
    h = h * 0x85EBCA77u;
    h = h ^ (h >> 13u);
    return h;
}
fn hash2(a: u32, b: u32) -> u32 {
    var h = a * 0x9E3779B1u ^ b * 0x85EBCA77u;
    h = h ^ (h >> 15u);
    return h;
}
fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - abs(hp % 2.0 - 1.0));
    var rgb: vec3<f32>;
    let hi = i32(hp);
    if (hi == 0) { rgb = vec3<f32>(c, x, 0.0); }
    else if (hi == 1) { rgb = vec3<f32>(x, c, 0.0); }
    else if (hi == 2) { rgb = vec3<f32>(0.0, c, x); }
    else if (hi == 3) { rgb = vec3<f32>(0.0, x, c); }
    else if (hi == 4) { rgb = vec3<f32>(x, 0.0, c); }
    else { rgb = vec3<f32>(c, 0.0, x); }
    return rgb + vec3<f32>(v - c);
}
fn id_color(id: u32) -> vec3<f32> {
    let h = hash_u32(id);
    let hue = f32(h % 3600u) / 10.0;
    let sat = 0.55 + f32((h >> 8u) & 0xFFu) / 255.0 * 0.30;
    let val = 0.80 + f32((h >> 16u) & 0xFFu) / 255.0 * 0.20;
    return hsv2rgb(hue, sat, val);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);

    // ── Mode 1: Unlit ──────────────────────────────────────────────────────────
    // Biome albedo, flat — no lighting, no ACES (the _SRGB swapchain still applies
    // its hardware OETF on write).
    // Modes 7–10 (Height/Material/Wetness/Volcano) bake their field into vertex color
    // on the voxel path (the mesher) and show it raw here, exactly like Unlit.
    if pc.material_mode == 1u || (pc.material_mode >= 7u && pc.material_mode <= 10u) {
        return vec4<f32>(in.albedo, 1.0);
    }

    // ── Mode 2: Normal ─────────────────────────────────────────────────────────
    // World-space normal → [0,1] RGB. Independent of lighting.
    if pc.material_mode == 2u {
        return vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
    }

    // ── Mode 6: Plate ──────────────────────────────────────────────────────────
    // Per-plate tectonic tint, raw. (Populated by the mesher via hf.plate_color.)
    if pc.material_mode == 6u {
        return vec4<f32>(in.plate_color, 1.0);
    }

    // ── Modes 3/4/5: geometry debug (per-mesh-unit ids pushed in ChunkPush) ──────
    // 3 Triangle: hash the provoking-vertex index with the leaf id → a unique colour
    //   per triangle (denser = finer LOD, same as the old Nanite triangle view).
    // 4 Cluster : one colour per resident leaf (shows the cube-sphere LOD tiling).
    // 5 LOD     : one colour per octree level → concentric LOD rings reveal whether
    //   selection is splitting near the camera / merging far away.
    if pc.material_mode == 3u {
        return vec4<f32>(id_color(hash2(pc.dbg_id, in.vid)), 1.0);
    }
    if pc.material_mode == 4u {
        return vec4<f32>(id_color(pc.dbg_id), 1.0);
    }
    if pc.material_mode == 5u {
        return vec4<f32>(id_color(pc.dbg_level + 1u), 1.0);
    }

    // ── Mode 0 Lit + fallback ──────────────────────────────────────────────────
    // Ambient isotropic + hemisphere + two directional lights, biome-color albedo,
    // ACES filmic tonemap. (3/4/5 are Nanite-only; on the classic quadtree path 7–10
    // have no baked data and fall back to Lit here — the voxel path handles them above.)
    let albedo = in.albedo;
    let ambient_term = frame.ambient.xyz * albedo;
    let hemi_term    = hemisphere_ambient(n, frame.hemi_sky.xyz, frame.hemi_ground.xyz) * albedo;
    let diff0        = directional_diffuse(n, frame.sun0_dir.xyz, frame.sun0_color.xyz) * albedo;
    let diff1        = directional_diffuse(n, frame.sun1_dir.xyz, frame.sun1_color.xyz) * albedo;
    var lit = ambient_term + hemi_term + diff0 + diff1;
    lit = aces_filmic(lit);
    return vec4<f32>(lit, 1.0);
}

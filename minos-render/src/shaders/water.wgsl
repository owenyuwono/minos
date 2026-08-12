// water.wgsl — translucent water sphere pass
//
// Pipeline contract (integration engineer):
//   - Descriptor set 0, binding 0: FrameUniforms uniform buffer
//   - Push constants: ChunkPush (80 bytes, stages VERTEX | FRAGMENT)
//   - Vertex buffers: same slots as terrain (positions, normals, colors, plate_colors)
//     colors from the mesh are used as a tint multiplier; intrinsic water color
//     comes from the material constant defined in this shader.
//   - Index buffer: u32, triangle-list, CCW winding
//   - Depth: D32_SFLOAT, cleared to 0.0, compare GREATER (reversed-Z)
//   - Render mode: alpha-blend, depth-write OFF, cull BACK
//     Blend equation (straight alpha):
//       dst.rgb = src.rgb * src.a + dst.rgb * (1 - src.a)
//       dst.a   = src.a + dst.a * (1 - src.a)
//   - Swapchain format: *_SRGB — ACES tonemap applied in shader; hardware encodes sRGB

// ── Uniforms ────────────────────────────────────────────────────────────────

struct FrameUniforms {
    view_proj  : mat4x4<f32>,
    camera_pos : vec4<f32>,
    sun0_dir   : vec4<f32>,
    sun0_color : vec4<f32>,
    sun1_dir   : vec4<f32>,
    sun1_color : vec4<f32>,
    hemi_sky   : vec4<f32>,
    hemi_ground: vec4<f32>,
    ambient    : vec4<f32>,
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

// ── Constants ────────────────────────────────────────────────────────────────

// Water intrinsic color (linear sRGB, #2e6b8f ≈ (0.180, 0.420, 0.560))
const WATER_COLOR: vec3<f32> = vec3<f32>(0.180, 0.420, 0.560);
const WATER_ALPHA: f32 = 0.72;
const WATER_ROUGHNESS: f32 = 0.15;

// ── Vertex stage ─────────────────────────────────────────────────────────────

struct VertexIn {
    @location(0) position   : vec3<f32>,
    @location(1) normal     : vec3<f32>,
    @location(2) color      : vec3<f32>,
    @location(3) plate_color: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos    : vec4<f32>,
    @location(0)       world_pos   : vec3<f32>,
    @location(1)       world_normal: vec3<f32>,
    @location(2)       vert_color  : vec3<f32>,
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    // Camera-relative rendering: pc.model's translation column encodes
    // (object_origin_f64 − camera_pos_f64) as f32.  frame.view_proj is
    // proj × view_rotation_only (camera at the origin).  All arithmetic
    // is on small near-origin values — no f32 cancellation at planet scale.
    let world_pos = pc.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = frame.view_proj * world_pos;
    // world_pos.xyz is camera-relative; store it for the view-vector below.
    out.world_pos = world_pos.xyz;

    let m3 = mat3x3<f32>(
        pc.model[0].xyz,
        pc.model[1].xyz,
        pc.model[2].xyz,
    );
    out.world_normal = m3 * v.normal;
    out.vert_color = v.color;
    return out;
}

// ── Fragment stage ────────────────────────────────────────────────────────────

fn hemisphere_ambient(n: vec3<f32>, sky: vec3<f32>, ground: vec3<f32>) -> vec3<f32> {
    let t = n.y * 0.5 + 0.5;
    return mix(ground, sky, t);
}

fn directional_diffuse(n: vec3<f32>, light_dir: vec3<f32>, color: vec3<f32>) -> vec3<f32> {
    return color * max(dot(n, light_dir), 0.0);
}

// Blinn-Phong specular lobe (cheap spec term for water glint).
// roughness → shininess conversion: shininess = 2 / roughness^4 - 2
fn blinn_phong_spec(
    n        : vec3<f32>,
    view_dir : vec3<f32>,
    light_dir: vec3<f32>,
    roughness: f32,
    color    : vec3<f32>,
) -> vec3<f32> {
    let h = normalize(light_dir + view_dir);
    let shininess = 2.0 / (roughness * roughness * roughness * roughness) - 2.0;
    let spec = pow(max(dot(n, h), 0.0), shininess);
    return color * spec;
}

fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);
    // Camera-relative view vector: frame.camera_pos.xyz == (0,0,0) in
    // camera-relative space, so this equals normalize(-in.world_pos).
    // Kept as camera_pos.xyz - world_pos for clarity and correct sign.
    let view_dir = normalize(frame.camera_pos.xyz - in.world_pos);

    // Albedo: intrinsic water color × vertex tint (vertex color used as multiplier)
    let albedo = WATER_COLOR * in.vert_color;

    // Day/night: fade the sky-fill (ambient + hemisphere) by local sun elevation so
    // the night hemisphere darkens and the terminator reads from orbit (matches the
    // terrain + wave-ocean fade). `n` is the outward sphere normal = local radial up;
    // the directional terms already fade via N·L. 0.05 = night floor (vs space).
    let day = mix(0.05, 1.0, smoothstep(-0.2, 0.0, dot(n, frame.sun0_dir.xyz)));

    // Diffuse lighting
    let ambient_term = frame.ambient.xyz * albedo * day;
    let hemi_term    = hemisphere_ambient(n, frame.hemi_sky.xyz, frame.hemi_ground.xyz) * albedo * day;
    let diff0        = directional_diffuse(n, frame.sun0_dir.xyz, frame.sun0_color.xyz) * albedo;
    let diff1        = directional_diffuse(n, frame.sun1_dir.xyz, frame.sun1_color.xyz) * albedo;

    // Specular glint (low roughness → tight highlight)
    let spec0 = blinn_phong_spec(n, view_dir, frame.sun0_dir.xyz, WATER_ROUGHNESS, frame.sun0_color.xyz * 0.5);
    let spec1 = blinn_phong_spec(n, view_dir, frame.sun1_dir.xyz, WATER_ROUGHNESS, frame.sun1_color.xyz * 0.3);

    var color = ambient_term + hemi_term + diff0 + diff1 + spec0 + spec1;

    // ACES tonemap (hardware encodes sRGB on write to _SRGB target)
    color = aces_filmic(color);

    // Output straight alpha; pipeline blend state handles compositing
    return vec4<f32>(color, WATER_ALPHA);
}

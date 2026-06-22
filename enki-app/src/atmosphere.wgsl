// atmosphere.wgsl — translucent atmosphere shell.
//
// A sky-tinted shell sphere at R + height, alpha-blended over the planet. Alpha
// is densest at the limb (grazing view = longest air column) and on the day side;
// a forward sun-glow brightens the sunward limb. Analytic — no scattering
// integral (ponytail: a single shell; port ki Atmosphere.ts for true Rayleigh/Mie).
//
// Pipeline contract: set0 binding 0 = FrameUniforms; ChunkPush push constants
// (model = uniform scale; pad0 reused as `density`). 4×vec3 vertex slots — only
// position (0) + normal (1) are read. Depth GREATER, depth-write OFF, alpha-blend.

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
    density       : f32, // reused pad slot
    _p1           : f32,
    _p2           : f32,
}

var<immediate> pc: ChunkPush;

struct VertexIn {
    @location(0) position   : vec3<f32>,
    @location(1) normal     : vec3<f32>,
    @location(2) color      : vec3<f32>,
    @location(3) plate_color: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
    @location(1)       normal   : vec3<f32>,
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    let world = pc.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = frame.view_proj * world;
    out.world_pos = world.xyz;            // camera-relative
    out.normal = normalize(v.normal);     // uniform scale → radial normal preserved
    return out;
}

// Base atmosphere tint (linear). Cool sky blue.
const ATMO_COLOR: vec3<f32> = vec3<f32>(0.33, 0.55, 1.00);

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let view = normalize(-in.world_pos);            // shell → camera (camera at origin)

    // Limb density: 0 at the sub-camera point, →1 at the grazing silhouette.
    let grazing = 1.0 - max(dot(n, view), 0.0);
    let rim = pow(clamp(grazing, 0.0, 1.0), 2.5);

    // Day/night: 1 on the sunlit side, 0 in shadow.
    let sun = clamp(dot(n, frame.sun0_dir.xyz) * 0.5 + 0.5, 0.0, 1.0);

    // Forward scattering: bright glow where you look toward the sun through air.
    let glow = pow(max(dot(view, frame.sun0_dir.xyz), 0.0), 8.0);

    let col = ATMO_COLOR * (0.45 + 0.55 * sun) + frame.sun0_color.xyz * glow * 0.6;
    let alpha = clamp(rim * pc.density * (0.25 + 0.75 * sun), 0.0, 1.0);
    return vec4<f32>(col, alpha);
}

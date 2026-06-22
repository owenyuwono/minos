// rivers.wgsl — drainage-network water ribbons.
//
// Pipeline contract mirrors markers.wgsl / wind.wgsl: set0 binding 0 =
// FrameUniforms, ChunkPush push constants (identity model — positions are
// camera-relative), 4×vec3 vertex slots (position 0, normal 1, color 2 read).
// Translucent (alpha blend, depth-write OFF, depth-test GREATER) so terrain in
// front occludes the ribbon and it reads as water on the surface.

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

struct VertexIn {
    @location(0) position   : vec3<f32>,
    @location(1) normal     : vec3<f32>,
    @location(2) color      : vec3<f32>,
    @location(3) p3         : vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0)       normal   : vec3<f32>,
    @location(1)       color    : vec3<f32>,
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    let world_pos = pc.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = frame.view_proj * world_pos;
    out.normal = v.normal;
    out.color = v.color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    // Diffuse from the primary sun + ambient/hemisphere fill — water reads as a
    // lit blue sheet. A sharpened sun term fakes a specular glint on the surface.
    let d0 = max(dot(n, frame.sun0_dir.xyz), 0.0);
    let d1 = max(dot(n, frame.sun1_dir.xyz), 0.0);
    let lit = in.color
        * (frame.ambient.xyz
            + frame.hemi_sky.xyz * (0.5 + 0.5 * n.y)
            + frame.sun0_color.xyz * d0
            + frame.sun1_color.xyz * d1);
    let glint = pow(d0, 48.0) * frame.sun0_color.xyz;
    return vec4<f32>(lit + glint, 0.85);
}

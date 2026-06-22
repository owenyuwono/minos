// wind.wgsl — wind streakline overlay.
//
// Pipeline contract (mirrors water.wgsl):
//   - Descriptor set 0, binding 0: FrameUniforms uniform buffer
//   - Push constants: ChunkPush (80 bytes, stages VERTEX | FRAGMENT)
//   - Vertex buffers: same 4×vec3 slots as terrain. Only slot 0 (position) and
//     slot 2 (color) are read; slots 1/3 are bound to the position buffer and
//     ignored. Positions are camera-relative; pc.model is identity (translation 0).
//   - color.x = fade (0..1, head→tail), color.y = wind speed 0..1, color.z unused.
//   - Depth: D32_SFLOAT, compare GREATER (reversed-Z), depth-write OFF.
//   - Render mode: straight-alpha blend, cull BACK. Target *_SRGB (linear out).

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
    @location(3) plate_color: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos  : vec4<f32>,
    @location(0)       fade_speed: vec2<f32>,
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    let world_pos = pc.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = frame.view_proj * world_pos;
    out.fade_speed = v.color.xy; // x = fade, y = speed01
    return out;
}

// Slow (calm) → fast (gusty) streak color ramp.
const SLOW_COL: vec3<f32> = vec3<f32>(0.28, 0.52, 0.95);
const FAST_COL: vec3<f32> = vec3<f32>(0.92, 0.97, 1.00);
const BRIGHT: f32 = 1.35;

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let fade  = clamp(in.fade_speed.x, 0.0, 1.0);
    let speed = clamp(in.fade_speed.y, 0.0, 1.0);
    let rgb = mix(SLOW_COL, FAST_COL, speed) * BRIGHT;
    return vec4<f32>(rgb, fade);
}

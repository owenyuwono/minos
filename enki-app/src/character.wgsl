// character.wgsl — lit humanoid that RECEIVES the sun CSM.
//
// Same lit model as terrain.wgsl mode 0 (ambient + hemisphere + 2 suns + ACES),
// but with its own descriptor set so it can sample the 3-cascade sun shadow map
// (the shared terrain pipeline has only set0 = FrameUniforms and can't). The
// cascade matrices project CAMERA-RELATIVE world positions → light clip, exactly
// like the Nanite receiver, so caster and receiver agree by construction.
//
// set0 (character-owned):
//   0 : CharFrame UBO (frame lighting + 3 cascade matrices + shadow params)
//   1,2,3 : per-cascade depth textures (re-pointed per frame-in-flight)
//   4 : comparison sampler (GREATER_OR_EQUAL → 2×2 hardware PCF per tap)

const SHADOW_CASCADES: u32 = 3u; // MUST match character.rs / voxel_view.rs / flora.wgsl

struct CharFrame {
    view_proj   : mat4x4<f32>,
    sun0_dir    : vec4<f32>,
    sun0_color  : vec4<f32>,
    sun1_dir    : vec4<f32>,
    sun1_color  : vec4<f32>,
    hemi_sky    : vec4<f32>,
    hemi_ground : vec4<f32>,
    ambient     : vec4<f32>,
    cascade_vp  : array<mat4x4<f32>, 3>, // camera-relative world → light clip
    shadow_params : vec4<f32>,           // [depth_bias, normal_bias, strength, enabled]
}

@group(0) @binding(0) var<uniform> frame : CharFrame;
@group(0) @binding(1) var shadow_map0 : texture_depth_2d;
@group(0) @binding(2) var shadow_map1 : texture_depth_2d;
@group(0) @binding(3) var shadow_map2 : texture_depth_2d;
@group(0) @binding(4) var shadow_samp : sampler_comparison;

struct ChunkPush {
    model         : mat4x4<f32>,
    material_mode : u32,
    _pad0         : u32,
    _pad1         : u32,
    _pad2         : u32,
}
var<immediate> pc : ChunkPush;

struct VsOut {
    @builtin(position) clip      : vec4<f32>,
    @location(0)       wnormal   : vec3<f32>,
    @location(1)       albedo    : vec3<f32>,
    @location(2)       world_rel : vec3<f32>, // camera-relative world pos (for shadows)
}

@vertex
fn vs_main(
    @location(0) position : vec3<f32>,
    @location(1) normal   : vec3<f32>,
    @location(2) color    : vec3<f32>,
    @location(3) plate    : vec3<f32>,
) -> VsOut {
    var out : VsOut;
    let world = pc.model * vec4<f32>(position, 1.0); // camera-relative
    out.clip = frame.view_proj * world;
    let m3 = mat3x3<f32>(pc.model[0].xyz, pc.model[1].xyz, pc.model[2].xyz);
    out.wnormal = m3 * normal;
    out.albedo = color;
    out.world_rel = world.xyz;
    return out;
}

fn hemisphere_ambient(n: vec3<f32>, sky: vec3<f32>, ground: vec3<f32>) -> vec3<f32> {
    return mix(ground, sky, n.y * 0.5 + 0.5);
}

fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Pick the tightest cascade whose light-space uv contains the point (offset along
// `n` by the cascade-scaled normal bias). Returns the cascade index (or
// SHADOW_CASCADES = "outside all") and writes uv + reversed-Z reference depth.
// 1:1 with nanite_draw.wgsl::select_cascade.
fn select_cascade(world_rel: vec3<f32>, n: vec3<f32>, uv_out: ptr<function, vec2<f32>>, ref_out: ptr<function, f32>) -> u32 {
    for (var i = 0u; i < SHADOW_CASCADES; i = i + 1u) {
        let scale = f32(1u << i);
        let p = world_rel + n * (frame.shadow_params.y * scale);
        let clip = frame.cascade_vp[i] * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) { continue; }
        let ndc = clip.xyz / clip.w;
        let u = ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);
        let m = 0.02;
        if (u.x > m && u.x < 1.0 - m && u.y > m && u.y < 1.0 - m && ndc.z > 0.0 && ndc.z < 1.0) {
            *uv_out = u;
            *ref_out = ndc.z + frame.shadow_params.x * scale;
            return i;
        }
    }
    return SHADOW_CASCADES;
}

// Cascaded sun shadow (3×3 PCF). 1 = lit, 0 = fully shadowed. 1:1 with
// nanite_draw.wgsl::sample_shadow so the character sits in the same shadows.
fn sample_shadow(world_rel: vec3<f32>, n: vec3<f32>) -> f32 {
    if (frame.shadow_params.w < 0.5) { return 1.0; }
    var uv = vec2<f32>(0.0, 0.0);
    var ref_depth = 0.0;
    let c = select_cascade(world_rel, n, &uv, &ref_depth);
    if (c >= SHADOW_CASCADES) { return 1.0; }
    let texel = 1.0 / f32(textureDimensions(shadow_map0).x);
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let o = uv + vec2<f32>(f32(dx), f32(dy)) * texel;
            switch (c) {
                case 0u: { sum = sum + textureSampleCompareLevel(shadow_map0, shadow_samp, o, ref_depth); }
                case 1u: { sum = sum + textureSampleCompareLevel(shadow_map1, shadow_samp, o, ref_depth); }
                default: { sum = sum + textureSampleCompareLevel(shadow_map2, shadow_samp, o, ref_depth); }
            }
        }
    }
    return sum / 9.0;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.wnormal);
    let albedo = in.albedo;
    let ambient_term = frame.ambient.xyz * albedo;
    let hemi = hemisphere_ambient(n, frame.hemi_sky.xyz, frame.hemi_ground.xyz) * albedo;
    let d0 = max(dot(n, frame.sun0_dir.xyz), 0.0) * frame.sun0_color.xyz * albedo;
    let d1 = max(dot(n, frame.sun1_dir.xyz), 0.0) * frame.sun1_color.xyz * albedo;
    let sh = sample_shadow(in.world_rel, n); // sun shadow map (terrain + trees + self)
    let lit = aces_filmic(ambient_term + hemi + d0 * sh + d1);
    return vec4<f32>(lit, 1.0);
}

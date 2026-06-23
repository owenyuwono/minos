// TAA resolve pass.
//
// Fullscreen triangle. For each pixel: reconstruct world position from depth +
// the inverse current view-proj, reproject with the previous view-proj to find
// where this surface was last frame, sample the history there, clamp it to the
// 3x3 neighborhood of the current frame (anti-ghosting), and blend.
//
// Output goes to the next history image (a single color attachment); the RHI then
// blits it to the swapchain. Matrices/params arrive via a uniform block (push
// constants can't hold two mat4s within the 128-byte guarantee).
//
// Entry points: vs_fullscreen, fs_resolve.

struct ResolveParams {
    cur_inv_view_proj: mat4x4<f32>, // clip -> world (current, un-jittered)
    prev_view_proj:    mat4x4<f32>, // world -> prev clip (un-jittered)
    cur_view_proj:     mat4x4<f32>, // world -> clip (current, un-jittered) — shadow march
    texel:             vec4<f32>,   // 1/w, 1/h, w, h
    misc:              vec4<f32>,   // alpha, history_valid(0|1), _, _
    sun:               vec4<f32>,   // xyz = sun dir (toward sun), w = shadow strength (0=off)
};

@group(0) @binding(0) var current_tex: texture_2d<f32>;
@group(0) @binding(1) var depth_tex:   texture_2d<f32>;
@group(0) @binding(2) var history_tex: texture_2d<f32>;
@group(0) @binding(3) var samp:        sampler;
@group(0) @binding(4) var<uniform> params: ResolveParams;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle from 3 vertices, no vertex buffers.
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u)); // (0,0)(2,0)(0,2)
    out.uv = uv;
    out.pos = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// Reconstruct world-space position from a UV + depth sample.
// Vulkan clip: NDC.xy in [-1,1] (y down), depth in [0,1] (reversed-Z here).
fn reconstruct_world(uv: vec2<f32>, depth: f32) -> vec3<f32> {
    let ndc = vec4<f32>(uv.x * 2.0 - 1.0, uv.y * 2.0 - 1.0, depth, 1.0);
    let world = params.cur_inv_view_proj * ndc;
    return world.xyz / world.w;
}

// ── Screen-space sun shadows (folded into the resolve) ───────────────────────
// March from the shaded point toward the sun; if a stored depth sample is nearer
// to the camera than the marched point (within a thickness slab), the sun is
// occluded → shadow. Reuses this pass's depth + matrices, so it costs no extra
// pass and shadows EVERYTHING in the depth buffer (character, trees, terrain).
// Limitations: screen-space (off-screen casters miss; reach = SUN_SHADOW_LEN).
const SUN_SHADOW_LEN: f32 = 12.0;      // metres marched toward the sun
const SUN_SHADOW_BIAS: f32 = 0.05;     // metres — avoid self-shadow acne
const SUN_SHADOW_THICK: f32 = 2.0;     // metres — occluder must be within this slab
const SUN_SHADOW_DARKEN: f32 = 0.55;   // 1 = black core, 0 = none
const SUN_SHADOW_PX_STRIDE: f32 = 2.0; // on-screen spacing between march samples
const SUN_SHADOW_MAX_STEPS: i32 = 48;  // perf cap on the adaptive step count

// `uv` = this pixel's screen UV (the march start on screen). The step COUNT scales
// with the march's on-screen length so samples stay ~SUN_SHADOW_PX_STRIDE px apart
// at any distance/grazing angle — a fixed count spreads thin over long shadows and
// steps over the (thin) occluder, which is what combed the shadow into slats.
fn sun_shadow(world: vec3<f32>, uv: vec2<f32>) -> f32 {
    if (params.sun.w <= 0.0) { return 1.0; }
    let sun = params.sun.xyz;

    // Project the far end of the march; its on-screen distance sets the sample count.
    let ce = params.cur_view_proj * vec4<f32>(world + sun * SUN_SHADOW_LEN, 1.0);
    if (ce.w <= 0.0) { return 1.0; }
    let end_uv = (ce.xy / ce.w) * 0.5 + 0.5;
    let span_px = length((end_uv - uv) * params.texel.zw);
    let steps = clamp(i32(span_px / SUN_SHADOW_PX_STRIDE), 4, SUN_SHADOW_MAX_STEPS);

    for (var i: i32 = 1; i <= steps; i = i + 1) {
        let t = SUN_SHADOW_LEN * f32(i) / f32(steps);
        let p = world + sun * t;
        let clip = params.cur_view_proj * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) { continue; }
        let suv = (clip.xy / clip.w) * 0.5 + 0.5;
        if (suv.x < 0.0 || suv.x > 1.0 || suv.y < 0.0 || suv.y > 1.0) { continue; }
        // textureSampleLevel (not textureSample): the march is non-uniform control flow.
        let sd = textureSampleLevel(depth_tex, samp, suv, 0.0).r;
        if (sd <= 0.0) { continue; } // sky
        let stored = reconstruct_world(suv, sd);
        let dp = length(p);          // camera at origin (camera-relative)
        let ds = length(stored);
        if (ds < dp - SUN_SHADOW_BIAS && ds > dp - SUN_SHADOW_THICK) {
            return 1.0 - params.sun.w * SUN_SHADOW_DARKEN;
        }
    }
    return 1.0;
}

@fragment
fn fs_resolve(in: VsOut) -> @location(0) vec4<f32> {
    let uv = in.uv;
    let texel = params.texel.xy;

    let cur = textureSample(current_tex, samp, uv).rgb;

    // 3x3 neighborhood AABB of the current frame (history is clamped into this
    // to suppress ghosting on disocclusion / motion).
    var nmin = cur;
    var nmax = cur;
    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            if (dx == 0 && dy == 0) { continue; }
            let o = vec2<f32>(f32(dx), f32(dy)) * texel;
            let c = textureSample(current_tex, samp, uv + o).rgb;
            nmin = min(nmin, c);
            nmax = max(nmax, c);
        }
    }

    var outc = cur;
    let valid = params.misc.y > 0.5;
    if (valid) {
        let depth = textureSample(depth_tex, samp, uv).r;
        // Skip sky / cleared depth (reversed-Z: far plane = 0.0).
        if (depth > 0.0) {
            let world = reconstruct_world(uv, depth);
            let prev_clip = params.prev_view_proj * vec4<f32>(world, 1.0);
            if (prev_clip.w > 0.0) {
                let prev_ndc = prev_clip.xy / prev_clip.w;
                let prev_uv = prev_ndc * 0.5 + 0.5;
                if (prev_uv.x >= 0.0 && prev_uv.x <= 1.0 && prev_uv.y >= 0.0 && prev_uv.y <= 1.0) {
                    var hist = textureSample(history_tex, samp, prev_uv).rgb;
                    hist = clamp(hist, nmin, nmax);
                    // Reduce the blend weight as on-screen motion grows.
                    let motion_px = length((prev_uv - uv) * params.texel.zw);
                    let alpha = clamp(params.misc.x - motion_px * 0.02, 0.0, params.misc.x);
                    outc = mix(cur, hist, alpha);
                }
            }
        }
    }

    // Screen-space sun shadow on the resolved color (after the history blend, so it
    // recomputes fresh each frame on the current depth — stable, no ghosting).
    let d_self = textureSampleLevel(depth_tex, samp, uv, 0.0).r;
    if (d_self > 0.0) {
        outc = outc * sun_shadow(reconstruct_world(uv, d_self), uv);
    }

    return vec4<f32>(outc, 1.0);
}

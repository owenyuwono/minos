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
    texel:             vec4<f32>,   // 1/w, 1/h, w, h
    misc:              vec4<f32>,   // alpha, history_valid(0|1), _, _
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

    return vec4<f32>(outc, 1.0);
}

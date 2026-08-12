// FXAA — fast approximate anti-aliasing (luma edge detect + directional blend).
//
// Post-process pass: samples the resolved scene colour (`src`) and writes the
// anti-aliased result straight to the swapchain. The reciprocal frame size is
// derived from `textureDimensions` (no push constant / UBO needed). A compact
// variant of Timothy Lottes' FXAA — quality is "good enough", cost ~1 fullscreen
// pass. Uses `textureSampleLevel` (explicit LOD 0) so it needs no derivatives.

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

struct VsOut {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle from the vertex index; uv spans [0,1] across the screen.
@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let uv = vec2<f32>(f32((vid << 1u) & 2u), f32(vid & 2u));
    out.uv = uv;
    out.pos = vec4<f32>(uv * 2.0 - vec2<f32>(1.0), 0.0, 1.0);
    return out;
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3<f32>(0.299, 0.587, 0.114));
}

@fragment
fn fs_fxaa(in: VsOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(src, 0));
    let inv = vec2<f32>(1.0) / dims;

    let SPAN_MAX = 8.0;
    let REDUCE_MUL = 1.0 / 8.0;
    let REDUCE_MIN = 1.0 / 128.0;

    let uv = in.uv;
    let rgbNW = textureSampleLevel(src, samp, uv + vec2<f32>(-1.0, -1.0) * inv, 0.0).rgb;
    let rgbNE = textureSampleLevel(src, samp, uv + vec2<f32>( 1.0, -1.0) * inv, 0.0).rgb;
    let rgbSW = textureSampleLevel(src, samp, uv + vec2<f32>(-1.0,  1.0) * inv, 0.0).rgb;
    let rgbSE = textureSampleLevel(src, samp, uv + vec2<f32>( 1.0,  1.0) * inv, 0.0).rgb;
    let rgbM  = textureSampleLevel(src, samp, uv, 0.0).rgb;

    let lNW = luma(rgbNW);
    let lNE = luma(rgbNE);
    let lSW = luma(rgbSW);
    let lSE = luma(rgbSE);
    let lM  = luma(rgbM);

    let lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    let lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));

    // Edge direction = gradient of luma across the 4 corners.
    var dir: vec2<f32>;
    dir.x = -((lNW + lNE) - (lSW + lSE));
    dir.y =  ((lNW + lSW) - (lNE + lSE));

    let dirReduce = max((lNW + lNE + lSW + lSE) * 0.25 * REDUCE_MUL, REDUCE_MIN);
    let rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2<f32>(-SPAN_MAX), vec2<f32>(SPAN_MAX)) * inv;

    let rgbA = 0.5 * (
        textureSampleLevel(src, samp, uv + dir * (1.0 / 3.0 - 0.5), 0.0).rgb +
        textureSampleLevel(src, samp, uv + dir * (2.0 / 3.0 - 0.5), 0.0).rgb);
    let rgbB = rgbA * 0.5 + 0.25 * (
        textureSampleLevel(src, samp, uv + dir * -0.5, 0.0).rgb +
        textureSampleLevel(src, samp, uv + dir *  0.5, 0.0).rgb);

    // If the second tap stepped past the local luma range, it crossed an edge —
    // fall back to the safer 2-tap average.
    let lB = luma(rgbB);
    var col = rgbB;
    if (lB < lMin || lB > lMax) {
        col = rgbA;
    }
    return vec4<f32>(col, 1.0);
}

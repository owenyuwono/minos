// post.wgsl — flora-OWNED post pipeline: a 1:1 port of dryad's three.js POST
// chain (RenderPass → UnrealBloomPass → OutputPass). All passes are fullscreen
// triangles with NO vertex input (positions from @builtin(vertex_index)).
//
// The chain, faithful to dryad (viewer.js bloom strength=0.15, radius=0.4,
// threshold=1.1; UnrealBloomPass nMips=5, kernels [3,5,7,9,11], smoothWidth 0.01;
// OutputPass = ACESFilmicToneMapping + sRGB OETF):
//
//   fs_bright    — LuminosityHighPassShader: smoothstep luma gate at threshold.
//   fs_blur      — separable gaussian (H or V via push.direction), per-mip radius.
//   fs_composite — combine the 5 blurred mips with the lerp'd bloom factors,
//                  pre-multiplied by bloomStrength (so strength=0 ⇒ all-zero).
//   fs_output    — scene + bloom → ACES (three.js matrix form, /0.6 exposure
//                  prescale) → LINEAR out (the _SRGB swapchain does the OETF).
//
// ponytail: ONE blur pipeline with a literal 11-tap loop (max KERNEL_RADIUS=11)
// and per-mip coefficients zeroed past the active radius is naga-safe (literal
// loop bound) and shorter than 5 specialized pipelines. The composite samples
// all 5 mips from a single 2D mip-chain image via explicit LOD.

// ── Fullscreen vertex stage (CCW triangle covering the viewport) ─────────────
struct VsOut {
    @builtin(position) pos : vec4<f32>,
    @location(0)       uv  : vec2<f32>,
}

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> VsOut {
    // Big triangle: (-1,-1),(3,-1),(-1,3) in clip → uv 0..1 across the viewport.
    var out: VsOut;
    let x = f32((vid << 1u) & 2u);   // 0,2,0
    let y = f32(vid & 2u);           // 0,0,2
    out.uv = vec2<f32>(x, y);        // 0..2 (covers 0..1 within the screen)
    out.pos = vec4<f32>(x * 2.0 - 1.0, y * 2.0 - 1.0, 0.0, 1.0);
    return out;
}

// ── set0: a single sampled texture + a linear-clamp sampler. The composite +
//    output passes that need a SECOND texture put it at binding 2/3. ──
@group(0) @binding(0) var src_tex     : texture_2d<f32>;
@group(0) @binding(1) var src_sampler : sampler;
// Composite/output second input (bloom-mip chain / bloom result).
@group(0) @binding(2) var aux_tex     : texture_2d<f32>;
@group(0) @binding(3) var aux_sampler : sampler;

// ── Push constants (≤128B). Reused across passes; each pass reads its lane. ──
struct PostPush {
    // bright: (threshold, smoothWidth, _, _)
    // blur:   (dirX, dirY, invSizeX, invSizeY)   + params2.x = kernelRadius
    // composite/output: (bloomStrength, bloomRadius, _, _)
    params  : vec4<f32>,
    // blur: (kernelRadius, mipLod, _, _)
    // composite: (mipLod0unused — mips sampled per-tap, _, _, _)
    params2 : vec4<f32>,
}
var<immediate> pc : PostPush;

// ─────────────────────────────────────────────────────────────────────────────
// BRIGHT PASS — three.js LuminosityHighPassShader (UnrealBloomPass highPassFS).
//   luma = dot(rgb, [0.299,0.587,0.114]); alpha = smoothstep(t, t+smoothWidth, luma)
//   out  = mix(vec4(0), texel, alpha)
// ─────────────────────────────────────────────────────────────────────────────
@fragment
fn fs_bright(in: VsOut) -> @location(0) vec4<f32> {
    let texel = textureSampleLevel(src_tex, src_sampler, in.uv, 0.0);
    let luma = dot(texel.rgb, vec3<f32>(0.299, 0.587, 0.114));
    let threshold = pc.params.x;
    let smooth_width = pc.params.y;
    let a = smoothstep(threshold, threshold + smooth_width, luma);
    return mix(vec4<f32>(0.0), texel, a);
}

// ─────────────────────────────────────────────────────────────────────────────
// BLUR PASS — separable gaussian (UnrealBloomPass getSeperableBlurMaterial).
//   coeff[i] = 0.39894 * exp(-0.5*i*i / r*r) / r,  i = 0 .. r-1
//   sum  = coeff[0]*center; for i=1..r-1 sum += coeff[i]*(t(+i)+t(-i))
//   wsum = coeff[0]; for i=1..r-1 wsum += 2*coeff[i]
//   out  = sum / wsum
// ponytail: literal 11-tap loop (max radius). Coefficients past the active
// kernelRadius contribute 0 because exp() is gated by `i < r` → those taps add
// nothing and wsum excludes them.
// ─────────────────────────────────────────────────────────────────────────────
@fragment
fn fs_blur(in: VsOut) -> @location(0) vec4<f32> {
    let direction = pc.params.xy;
    let inv_size = pc.params.zw;
    let r = pc.params2.x;             // kernelRadius ∈ {3,5,7,9,11}
    let lod = pc.params2.y;

    let inv_r = 1.0 / r;
    let c0 = 0.39894 * inv_r;         // i=0: exp(0)=1
    var sum = textureSampleLevel(src_tex, src_sampler, in.uv, lod).rgb * c0;
    var wsum = c0;
    // Max KERNEL_RADIUS = 11 → i runs 1..10 (i < r gates the active taps).
    for (var i: i32 = 1; i < 11; i = i + 1) {
        let fi = f32(i);
        if (fi >= r) { break; }       // literal bound above keeps naga happy
        let w = 0.39894 * exp(-0.5 * fi * fi * inv_r * inv_r) * inv_r;
        let off = direction * inv_size * fi;
        let sp = textureSampleLevel(src_tex, src_sampler, in.uv + off, lod).rgb;
        let sn = textureSampleLevel(src_tex, src_sampler, in.uv - off, lod).rgb;
        sum = sum + (sp + sn) * w;
        wsum = wsum + 2.0 * w;
    }
    return vec4<f32>(sum / wsum, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// COMPOSITE PASS — UnrealBloomPass getCompositeMaterial, pre-multiplied by
// bloomStrength so strength=0 ⇒ exactly zero (parity guarantee).
//   factor[i] = mix(bloomFactors[i], 1.2 - bloomFactors[i], bloomRadius)
//   result    = bloomStrength * Σ factor[i] * mip[i]
// bloomFactors = [1.0, 0.8, 0.6, 0.4, 0.2]; tint all white.
// The 5 mips are LODs 0..4 of a single mip-chain texture (src_tex).
// ─────────────────────────────────────────────────────────────────────────────
fn lerp_factor(f: f32, radius: f32) -> f32 {
    return mix(f, 1.2 - f, radius);
}

@fragment
fn fs_composite(in: VsOut) -> @location(0) vec4<f32> {
    let strength = pc.params.x;
    let radius = pc.params.y;
    let bf = array<f32, 5>(1.0, 0.8, 0.6, 0.4, 0.2);
    var acc = vec3<f32>(0.0);
    // ponytail: literal-unrolled 5 taps (one per mip LOD) — naga-safe, matches
    // UnrealBloomPass' fixed 5-mip blurTexture[] uniforms.
    acc = acc + lerp_factor(bf[0], radius) * textureSampleLevel(src_tex, src_sampler, in.uv, 0.0).rgb;
    acc = acc + lerp_factor(bf[1], radius) * textureSampleLevel(src_tex, src_sampler, in.uv, 1.0).rgb;
    acc = acc + lerp_factor(bf[2], radius) * textureSampleLevel(src_tex, src_sampler, in.uv, 2.0).rgb;
    acc = acc + lerp_factor(bf[3], radius) * textureSampleLevel(src_tex, src_sampler, in.uv, 3.0).rgb;
    acc = acc + lerp_factor(bf[4], radius) * textureSampleLevel(src_tex, src_sampler, in.uv, 4.0).rgb;
    return vec4<f32>(strength * acc, 1.0);
}

// ─────────────────────────────────────────────────────────────────────────────
// OUTPUT PASS — scene + bloom → ACES → LINEAR (the _SRGB swapchain does the OETF).
//
// 1:1 with three.js OutputPass: ACESFilmicToneMapping (tonemapping_pars_fragment)
// with the built-in `color *= toneMappingExposure / 0.6` prescale (exposure 1.0
// → ×1.6667), then sRGBTransferOETF. Because the flora swapchain is B8G8R8A8_SRGB,
// the hardware applies the sRGB OETF on store — so we output LINEAR here and do
// NOT encode sRGB in-shader (avoids a double encode).
//
// scene = src_tex (resolved HDR), bloom = aux_tex (composite result), added
// LINEAR (three.js additively blends bloom over the read buffer pre-OutputPass).
// ─────────────────────────────────────────────────────────────────────────────

// three.js ACESFilmicToneMapping (RRT+ODT matrix fit).
fn aces_three(color_in: vec3<f32>) -> vec3<f32> {
    // input  → ACEScg-ish working space
    let m1 = mat3x3<f32>(
        vec3<f32>(0.59719, 0.07600, 0.02840),
        vec3<f32>(0.35458, 0.90834, 0.13383),
        vec3<f32>(0.04823, 0.01566, 0.83777),
    );
    // working → sRGB/Rec.709
    let m2 = mat3x3<f32>(
        vec3<f32>( 1.60475, -0.10208, -0.00327),
        vec3<f32>(-0.53108,  1.10813, -0.07276),
        vec3<f32>(-0.07367, -0.00605,  1.07602),
    );
    let v = m1 * color_in;
    let a = v * (v + 0.0245786) - 0.000090537;
    let b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return clamp(m2 * (a / b), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_output(in: VsOut) -> @location(0) vec4<f32> {
    let scene = textureSampleLevel(src_tex, src_sampler, in.uv, 0.0).rgb;
    let bloom = textureSampleLevel(aux_tex, aux_sampler, in.uv, 0.0).rgb;
    var hdr = scene + bloom;
    // ponytail: three.js ACES carries a built-in color *= toneMappingExposure/0.6
    // prescale (×1.667 at exposure 1.0). With our HDR Preetham sky (linear luma
    // ~0.8–2.1) and the intensity-3.0 key sun, that ×1.667 over-exposed the WHOLE
    // frame to near-white (sky blown out, tree/ground washed away). Folding the
    // dryad scene exposure (0.7) INTO the prescale → effective 0.7/0.6 ≈ 1.167
    // keeps the sky a believable light blue and the bark/leaves bright-but-lit.
    hdr = hdr * (0.7 / 0.6);
    let mapped = aces_three(hdr);
    // LINEAR out — the B8G8R8A8_SRGB swapchain applies the sRGB OETF on store.
    return vec4<f32>(mapped, 1.0);
}

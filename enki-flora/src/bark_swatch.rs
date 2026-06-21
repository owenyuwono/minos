//! CPU bark-swatch rasterizer — a representative, debug-only port of
//! `flora.wgsl`'s `barkAlbedo` (the bark is otherwise FULLY procedural
//! in-shader, with no CPU bake). Used by the flora-viewer Inspector dock to show
//! a labeled bark preview that updates with the bark genes.
//!
//! # ponytail SHORTCUTS (this is an APPROXIMATION, not a pixel-match of the GPU)
//! - The real `fs_branch` evaluates bark in 3D object space on a tapered tube,
//!   with screen-space-derivative AA (`fwidth`) per octave and a normal-perturb
//!   relief pass. Here we rasterize a flat 2D swatch: we sample the albedo math
//!   on a synthetic object-space coordinate where the **V axis is mapped to the
//!   twig→trunk featureScale range** (0.14..0.55, the same clamp `fs_branch`
//!   uses) so the swatch shows the feature-scale sweep like dryad's tapered
//!   barkSwatch cylinder. `fw` (the AA footprint) is fed a small constant — no
//!   per-octave screen-space fade — so fine octaves are NOT dropped.
//! - We port the ALBEDO (`barkAlbedo` + `barkHeightField` for the height term it
//!   consumes), NOT the relief normal map. The swatch is flat-lit (albedo only),
//!   so furrow/plate RELIEF reads only through the albedo's height-coupled
//!   furrow/cavity tones, not through shading. The voronoi PLATE cracks (a normal
//!   feature) are therefore not visible in the swatch — acceptable for a debug
//!   color preview.
//! - Lenticels, lichen, blotch, oxide, micro-variation, the final warm-brown
//!   tint, and the woodiness→herbaceous blend ARE ported, so every bark gene
//!   still visibly drives the swatch.
//!
//! All math mirrors `flora.wgsl` constant-for-constant where ported.

/// Bark genes the swatch consumes — the SAME push-constant vec4s `FloraView`
/// exposes via `bark_genes()` and `fs_branch` reads.
#[derive(Clone, Copy, Debug)]
pub struct BarkGenes {
    pub hue: f32,
    pub lightness: f32,
    pub relief: f32,
    pub lenticels: f32,
    pub scale: f32,
    pub orient: f32,
    pub plates: f32,
    pub shed: f32,
    pub under_hue: f32,
    pub woodiness: f32,
}

impl BarkGenes {
    /// Build from the `(bark0, bark1, bark2)` vec4 triple `FloraView::bark_genes`
    /// returns.
    pub fn from_vec4s(bark0: [f32; 4], bark1: [f32; 4], bark2: [f32; 4]) -> Self {
        BarkGenes {
            hue: bark0[0],
            lightness: bark0[1],
            relief: bark0[2],
            lenticels: bark0[3],
            scale: bark1[0],
            orient: bark1[1],
            plates: bark1[2],
            shed: bark1[3],
            under_hue: bark2[0],
            woodiness: bark2[1],
        }
    }
}

// ── flora.wgsl bark constants (verbatim) ─────────────────────────────────────
const BARK_FISSURE_FREQ_SCALE: f32 = 0.42;
const BARK_FISSURE_WEIGHT: f32 = 0.55;
const BARK_RIDGE_FREQ_SCALE: f32 = 0.5;
const BARK_ORIENT_DEFAULT: f32 = 0.70;
const BARK_ANISO_SPREAD: f32 = 1.20;
const BARK_SHED_FREQ: f32 = 0.6;
const BARK_WARP_Y_DAMP: f32 = 0.30;
const BARK_Y_STRETCH: f32 = 3.0;

fn mix(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}
fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [mix(a[0], b[0], t), mix(a[1], b[1], t), mix(a[2], b[2], t)]
}
fn clamp(x: f32, lo: f32, hi: f32) -> f32 {
    x.max(lo).min(hi)
}
fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = clamp((x - e0) / (e1 - e0).max(1e-9), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
fn fract(x: f32) -> f32 {
    x - x.floor()
}

// hsl2rgb — flora.wgsl's branchless variant (matches the shader exactly).
fn hsl2rgb(h: f32, s: f32, l: f32) -> [f32; 3] {
    let m = |k: f32| -> f32 {
        // mod6 of (h*6 + k), then |.-3| - 1, clamped 0..1
        let v = h * 6.0 + k;
        let m6 = v - 6.0 * (v / 6.0).floor();
        clamp((m6 - 3.0).abs() - 1.0, 0.0, 1.0)
    };
    let r = m(0.0);
    let g = m(4.0);
    let b = m(2.0);
    let c = (1.0 - (2.0 * l - 1.0).abs()) * s;
    [l + c * (r - 0.5), l + c * (g - 0.5), l + c * (b - 0.5)]
}

// Value-noise hash in [-1,1] for a vec3 seed (flora.wgsl barkHash).
fn bark_hash(p: [f32; 3]) -> f32 {
    let mut x = [
        fract(p[0] * 127.1),
        fract(p[1] * 311.7),
        fract(p[2] * 74.7),
    ];
    let dot = x[0] * (x[1] + 19.19) + x[1] * (x[2] + 19.19) + x[2] * (x[0] + 19.19);
    x = [x[0] + dot, x[1] + dot, x[2] + dot];
    fract((x[0] + x[1]) * x[2]) * 2.0 - 1.0
}

// 3D value noise (flora.wgsl barkNoise).
fn bark_noise(p: [f32; 3]) -> f32 {
    let i = [p[0].floor(), p[1].floor(), p[2].floor()];
    let f = [p[0] - i[0], p[1] - i[1], p[2] - i[2]];
    let u = [
        f[0] * f[0] * (3.0 - 2.0 * f[0]),
        f[1] * f[1] * (3.0 - 2.0 * f[1]),
        f[2] * f[2] * (3.0 - 2.0 * f[2]),
    ];
    let h = |dx: f32, dy: f32, dz: f32| bark_hash([i[0] + dx, i[1] + dy, i[2] + dz]);
    let n000 = h(0.0, 0.0, 0.0);
    let n100 = h(1.0, 0.0, 0.0);
    let n010 = h(0.0, 1.0, 0.0);
    let n110 = h(1.0, 1.0, 0.0);
    let n001 = h(0.0, 0.0, 1.0);
    let n101 = h(1.0, 0.0, 1.0);
    let n011 = h(0.0, 1.0, 1.0);
    let n111 = h(1.0, 1.0, 1.0);
    let x00 = mix(n000, n100, u[0]);
    let x10 = mix(n010, n110, u[0]);
    let x01 = mix(n001, n101, u[0]);
    let x11 = mix(n011, n111, u[0]);
    let y0 = mix(x00, x10, u[1]);
    let y1 = mix(x01, x11, u[1]);
    mix(y0, y1, u[2])
}

fn ridge_octave(p: [f32; 3]) -> f32 {
    let n = 1.0 - bark_noise(p).abs();
    n * n
}

// 2-octave domain-warp vector (flora.wgsl barkWarpFBM).
fn bark_warp_fbm(p: [f32; 3]) -> [f32; 3] {
    let a = bark_noise(p);
    let b = bark_noise([p[0] * 2.07 + 3.17, p[1] * 2.07 + 1.43, p[2] * 2.07 + 2.71]);
    [
        (a + 0.5 * b) * 0.5,
        (b + 0.5 * a) * 0.5,
        (a * b) * 0.5,
    ]
}

// Ridged FBM, 8 octaves. fw is a small constant in the swatch (no screen-space AA).
fn ridged_fbm(pw: [f32; 3], freq_in: f32, fw: f32) -> f32 {
    let mut amp = 1.0f32;
    let mut total = 0.0f32;
    let mut norm = 0.0f32;
    let mut freq = freq_in;
    for _ in 0..8 {
        let oct_fw = fw * freq;
        let aa_fade = 1.0 - smoothstep(0.35, 1.3, oct_fw);
        total += ridge_octave([pw[0] * freq, pw[1] * freq, pw[2] * freq]) * amp * aa_fade;
        norm += amp;
        freq *= 2.13;
        amp *= 0.52;
    }
    let h = clamp(total / norm.max(1e-9), 0.0, 1.0);
    h.powf(0.65)
}

fn bark_height_field(p_grain: [f32; 3], feature_scale: f32, fw: f32, bark_scale: f32) -> f32 {
    let mut warp = bark_warp_fbm([p_grain[0] * 0.45, p_grain[1] * 0.45, p_grain[2] * 0.45]);
    warp = [warp[0] * 0.55, warp[1] * 0.55 * BARK_WARP_Y_DAMP, warp[2] * 0.55];
    let pw = [p_grain[0] + warp[0], p_grain[1] + warp[1], p_grain[2] + warp[2]];
    let base_freq = (BARK_RIDGE_FREQ_SCALE * mix(0.6, 1.8, bark_scale)) / feature_scale.max(0.01);
    let coarse = ridged_fbm(pw, base_freq * BARK_FISSURE_FREQ_SCALE, fw);
    let fine = ridged_fbm(pw, base_freq, fw);
    let mut combined = mix(fine, coarse, BARK_FISSURE_WEIGHT);
    combined *= 0.4 + 0.6 * coarse;
    clamp(combined, 0.0, 1.0)
}

fn bark_aniso_gain(orient: f32) -> f32 {
    let span = if orient >= BARK_ORIENT_DEFAULT {
        1.0 - BARK_ORIENT_DEFAULT
    } else {
        BARK_ORIENT_DEFAULT
    };
    let u = (orient - BARK_ORIENT_DEFAULT) / span.max(1e-3);
    (u * BARK_ANISO_SPREAD).exp2()
}

fn bark_shed_field(p: [f32; 3], bark_shed: f32) -> f32 {
    if bark_shed < 0.001 {
        return 0.0;
    }
    let patch = bark_noise([
        p[0] * BARK_SHED_FREQ + 13.7,
        p[1] * BARK_SHED_FREQ + 4.2,
        p[2] * BARK_SHED_FREQ + 8.1,
    ]) * 0.5
        + 0.5;
    let shed_t = mix(1.05, 0.02, bark_shed);
    smoothstep(shed_t, shed_t + 0.20, patch)
}

// Horizontal lenticel dashes (flora.wgsl barkLenticel).
fn bark_lenticel(p_grain: [f32; 3], prominence: f32) -> f32 {
    if prominence < 0.001 {
        return 0.0;
    }
    let band_y = p_grain[1] / 3.0;
    let band_index = (band_y * 5.0).floor();
    let band_frac = fract(band_y * 5.0);
    let angle = p_grain[2].atan2(p_grain[0]);
    let band_hash = fract((band_index * 127.1 + 311.7).sin() * 43758.5);
    let sectors = 5.0f32;
    let sector_angle = std::f32::consts::TAU / sectors;
    let sector_frac = fract((angle / sector_angle) + band_hash);
    let dash_w = 0.35f32;
    let dash_h = 0.18f32;
    let dc_h = smoothstep(0.5 - dash_w, 0.5 - dash_w * 0.3, sector_frac)
        * (1.0 - smoothstep(0.5 + dash_w * 0.3, 0.5 + dash_w, sector_frac));
    let dc_v = smoothstep(0.5 - dash_h, 0.5 - dash_h * 0.3, band_frac)
        * (1.0 - smoothstep(0.5 + dash_h * 0.3, 0.5 + dash_h, band_frac));
    let mask = dc_h * dc_v;
    let presence = fract(
        ((band_index * 73.13 + (sector_frac * sectors + band_hash).floor() * 57.3).sin()) * 91027.3,
    );
    let present = if presence >= 0.45 { 1.0 } else { 0.0 };
    mask * present * prominence
}

// Bark albedo — port of flora.wgsl barkAlbedo (verbatim blends + constants).
fn bark_albedo(p_grain: [f32; 3], h: f32, world_y: f32, g: &BarkGenes) -> [f32; 3] {
    let b_l = mix(0.10, 0.92, g.lightness);
    let b_s = g.hue * 0.55;
    let b_h = mix(0.09, 0.02, g.hue);

    let palette_ridge = hsl2rgb(b_h, b_s, clamp(b_l + 0.07, 0.0, 1.0));
    let palette_furrow = hsl2rgb(b_h, b_s * 1.15, clamp(b_l - 0.20, 0.0, 1.0));
    let palette_lichen = hsl2rgb(0.26, 0.28, clamp(0.32 + b_l * 0.18, 0.0, 1.0));
    let blotch_warm = hsl2rgb(clamp(b_h - 0.015, 0.0, 1.0), b_s, clamp(b_l - 0.05, 0.0, 1.0));
    let blotch_cool = hsl2rgb(clamp(b_h + 0.05, 0.0, 1.0), b_s * 0.7, clamp(b_l - 0.07, 0.0, 1.0));

    let base_sat = b_s * 0.58;
    let base_hue = mix(b_h, 0.065, 0.35);
    let mut base = hsl2rgb(base_hue, base_sat, b_l);

    let furrow_t = clamp((0.45 - h) * 2.0, 0.0, 1.0);
    let furrow_col = hsl2rgb(clamp(b_h - 0.005, 0.0, 1.0), b_s * 0.72, clamp(b_l - 0.20, 0.0, 1.0));
    base = mix3(base, furrow_col, furrow_t * 0.55);

    let lichen_crevice = clamp((0.22 - h) * 6.0, 0.0, 1.0);
    let lichen_base = clamp(1.0 - world_y * 0.55, 0.0, 1.0);
    let lichen_spatter = clamp(
        bark_noise([p_grain[0] * 3.7 + 7.3, p_grain[1] * 3.7 + 2.9, p_grain[2] * 3.7 + 5.1]) * 0.5
            + 0.5,
        0.0,
        1.0,
    );
    base = mix3(base, palette_lichen, lichen_crevice * lichen_base * lichen_spatter * 0.70);

    let blotch =
        bark_noise([p_grain[0] * 0.18 + 5.5, p_grain[1] * 0.18 + 1.3, p_grain[2] * 0.18 + 3.7]) * 0.5
            + 0.5;
    base = mix3(base, mix3(blotch_cool, blotch_warm, blotch), 0.22);

    let mut oxide =
        bark_noise([p_grain[0] * 0.22 + 9.2, p_grain[1] * 0.22 + 0.8, p_grain[2] * 0.22 + 6.4]) * 0.5
            + 0.5;
    oxide = smoothstep(0.40, 0.78, oxide);
    let oxide_col = hsl2rgb(clamp(b_h - 0.01, 0.0, 1.0), clamp(b_s * 1.4, 0.0, 1.0), clamp(b_l - 0.05, 0.0, 1.0));
    base = mix3(base, oxide_col, oxide * 0.40);

    let lenticel_mask = bark_lenticel(p_grain, clamp(g.lenticels * 1.2, 0.0, 1.0));
    let lenticel_col = mix3(palette_ridge, [palette_furrow[0] * 0.55, palette_furrow[1] * 0.55, palette_furrow[2] * 0.55], 0.75);
    base = mix3(base, lenticel_col, lenticel_mask);

    let cavity = mix(0.62, 1.0, clamp(h * 1.7, 0.0, 1.0));
    base = [base[0] * cavity, base[1] * cavity, base[2] * cavity];

    let micro = bark_noise([p_grain[0] * 8.3 + 1.1, p_grain[1] * 8.3 + 4.4, p_grain[2] * 8.3 + 2.2]) * 0.06;
    base = [
        clamp(base[0] + micro, 0.0, 1.0),
        clamp(base[1] + micro, 0.0, 1.0),
        clamp(base[2] + micro, 0.0, 1.0),
    ];

    // Final warm-brown darkening tint (flora.wgsl).
    [base[0] * 0.27, base[1] * 0.23, base[2] * 0.19]
}

/// Linear→sRGB gamma encode for an 8-bit channel (the GPU samples bark as a
/// linear albedo then the tonemap/sRGB write happens downstream; for a flat
/// preview we apply a plain sRGB encode so the swatch isn't gamma-dark).
fn linear_to_srgb_u8(c: f32) -> u8 {
    let c = clamp(c, 0.0, 1.0);
    let s = if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    };
    (s * 255.0).round().clamp(0.0, 255.0) as u8
}

/// Bake a `size`×`size` RGBA8 bark swatch. The U axis wraps the bark circumference
/// (x grain), the V axis sweeps the twig→trunk **featureScale** range (top = thin
/// twig, bottom = thick trunk) AND a world-Y proxy (so base-biased lichen shows).
/// Output is sRGB-encoded and fully opaque.
pub fn bake_bark_swatch(g: &BarkGenes, size: u32) -> Vec<u8> {
    let n = size as usize;
    let mut out = vec![0u8; n * n * 4];
    let orient_gain = bark_aniso_gain(g.orient);
    // Constant AA footprint — no screen-space derivatives in a flat swatch.
    const SWATCH_FW: f32 = 0.02;
    // Object-space extent the swatch maps across (a representative trunk patch).
    const X_SPAN: f32 = 1.6; // circumference grain in object units
    const Y_SPAN: f32 = 4.0; // vertical object-space run (twig→trunk + worldY)

    let herb_lo = [0.16f32, 0.34, 0.12];
    let herb_hi = [0.30f32, 0.52, 0.20];

    for py in 0..n {
        let v = py as f32 / (n - 1).max(1) as f32; // 0 top .. 1 bottom
        // featureScale: thin twig (0.14) at top → thick trunk (0.55) at bottom.
        let feature_scale = mix(0.14, 0.55, v);
        // world-Y proxy: trunk base (low Y) at the bottom so lichen pools there.
        let world_y = (1.0 - v) * 1.4;
        let obj_y = v * Y_SPAN;
        for px in 0..n {
            let u = px as f32 / (n - 1).max(1) as f32;
            let obj_x = (u - 0.5) * X_SPAN;
            // Match fs_branch's pGrain: (x, y*BARK_Y_STRETCH, z) tilted by orient.
            let p = [obj_x, obj_y * BARK_Y_STRETCH, 0.0];
            let p_grain = [p[0] / orient_gain, p[1] * orient_gain, p[2] / orient_gain];

            let mut bark_h = bark_height_field(p_grain, feature_scale, SWATCH_FW, g.scale);
            bark_h = mix(0.55, bark_h, mix(0.25, 1.0, g.relief));

            let mut albedo = bark_albedo(p_grain, bark_h, world_y, g);
            // woodiness → herbaceous-green blend (fs_branch).
            let herb = mix3(herb_lo, herb_hi, bark_h);
            albedo = mix3(herb, albedo, g.woodiness);
            // shed under-bark overlay, gated by woodiness.
            let shed = bark_shed_field(p_grain, g.shed);
            let ub_l = clamp(mix(0.10, 0.92, g.lightness) + 0.12, 0.0, 1.0);
            let ub_s = g.under_hue * 0.55;
            let ub_h = mix(0.09, 0.02, g.under_hue);
            let under = hsl2rgb(ub_h, ub_s, ub_l);
            albedo = mix3(albedo, under, shed * g.woodiness);

            let i = (py * n + px) * 4;
            out[i] = linear_to_srgb_u8(albedo[0]);
            out[i + 1] = linear_to_srgb_u8(albedo[1]);
            out[i + 2] = linear_to_srgb_u8(albedo[2]);
            out[i + 3] = 255;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_genes() -> BarkGenes {
        BarkGenes {
            hue: 0.4,
            lightness: 0.45,
            relief: 0.6,
            lenticels: 0.1,
            scale: 0.5,
            orient: 0.70,
            plates: 0.4,
            shed: 0.1,
            under_hue: 0.4,
            woodiness: 0.9,
        }
    }

    #[test]
    fn bakes_opaque_nonempty_swatch() {
        let g = default_genes();
        let img = bake_bark_swatch(&g, 32);
        assert_eq!(img.len(), 32 * 32 * 4);
        // Fully opaque.
        for px in img.chunks_exact(4) {
            assert_eq!(px[3], 255);
        }
        // Not a single flat color (the feature sweep must vary the image).
        let first = &img[0..3];
        let varies = img
            .chunks_exact(4)
            .any(|px| px[0] != first[0] || px[1] != first[1] || px[2] != first[2]);
        assert!(varies, "bark swatch should not be a single flat color");
    }

    #[test]
    fn deterministic() {
        let g = default_genes();
        assert_eq!(bake_bark_swatch(&g, 24), bake_bark_swatch(&g, 24));
    }

    #[test]
    fn hsl2rgb_matches_pigment_reference_shape() {
        // hsl2rgb red at h=0,s=1,l=0.5 → pure red.
        let c = hsl2rgb(0.0, 1.0, 0.5);
        assert!(c[0] > 0.99 && c[1] < 0.01 && c[2] < 0.01);
    }
}

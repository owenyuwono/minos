// flora_ibl.rs — CPU prefilter for flora's lazy image-based lighting (IBL).
//
// ponytail: this is the LAZY-but-faithful IBL bake the recon calls for — no
// render-to-cubemap, no PMREM, no BRDF LUT. From dryad's equirect HDRI
// (kloofendal_43d_clear_1k.hdr) we compute, ENTIRELY on the CPU at load time:
//
//   (1) 9 spherical-harmonic coefficients of the diffuse IRRADIANCE (the cosine-
//       convolved environment), pre-scaled by dryad's scene.environmentIntensity
//       = 0.6 — so the shader evaluates diffuse IBL with a single 9-term dot,
//       NO irradiance cubemap (flora_ibl::sh9).
//   (2) a roughness-indexed MIP CHAIN of the equirect itself, kept as a 2D
//       equirect image (NOT a cubemap): mip 0 = full-res linear radiance, each
//       coarser mip = a cosine/box downsample so rougher reflections read blurrier
//       (flora_ibl::mip_chain). The shader maps roughness→LOD and samples it as a
//       dir→uv equirect lookup. // ponytail: box/cos downsample, NOT a true
//       GGX-importance prefilter — close enough for the soft photoreal look and
//       an order of magnitude less code than PMREM.
//
// This module is PURE CPU + std + the `image` crate (hdr decode). It produces
// plain Vec<f32> data; flora_render.rs uploads it (the SH9 into a UBO, the mip
// chain into one RGBA16F 2D image). Deletable with the rest of flora.

use std::io::Cursor;

/// dryad scene.environmentIntensity (viewer.js:320) — baked into the SH coeffs
/// and applied to the specular sample in-shader so the WGSL stays clean.
pub const ENV_INTENSITY: f32 = 0.6;

/// The bundled HDRI (dryad's Poly Haven CC0 `kloofendal_43d_clear_1k.hdr`),
/// embedded so flora stays self-contained (no runtime asset path). 1k equirect.
pub const HDRI_BYTES: &[u8] =
    include_bytes!("../assets/flora/kloofendal_43d_clear_1k.hdr");

/// A decoded linear-radiance equirect (Radiance .hdr is already linear). RGB
/// stored row-major, `w`×`h`, +Y up. `pixels.len() == w*h*3`.
pub struct Equirect {
    pub w: usize,
    pub h: usize,
    /// Linear RGB radiance, 3 floats per texel, row 0 = top (v=0, +Y pole).
    pub pixels: Vec<f32>,
}

impl Equirect {
    #[inline]
    fn texel(&self, x: usize, y: usize) -> [f32; 3] {
        let i = (y * self.w + x) * 3;
        [self.pixels[i], self.pixels[i + 1], self.pixels[i + 2]]
    }
}

/// One uploadable mip level of the equirect chain (linear RGBA radiance, the A
/// channel = 1.0 padding so it maps to an RGBA16F image binding).
pub struct Mip {
    pub w: usize,
    pub h: usize,
    /// RGBA f32, row-major. Converted to f16 at upload time.
    pub rgba: Vec<f32>,
}

/// The full CPU-baked IBL set: SH9 diffuse coeffs + the equirect mip chain.
pub struct IblData {
    /// 9 RGB SH irradiance coefficients, cosine-convolved + ×ENV_INTENSITY, each
    /// padded to vec4 (w=0) for a std140 UBO array<vec4,9>.
    pub sh9: [[f32; 4]; 9],
    /// Roughness mip chain (mip 0 = full-res). `mips.len()-1` == max LOD.
    pub mips: Vec<Mip>,
}

impl IblData {
    /// Max LOD = last mip index, the value roughness=1.0 maps to in-shader.
    pub fn max_lod(&self) -> f32 {
        (self.mips.len().saturating_sub(1)) as f32
    }
}

/// Decode the bundled Radiance .hdr into a linear-radiance equirect via the
/// `image` crate (HdrDecoder → Rgb<f32>). // ponytail: the `image` crate's hdr
/// feature is the shortest correct Radiance parser; rolling our own RLE decoder
/// would be ~150 lines for no gain.
pub fn decode_hdri() -> Result<Equirect, String> {
    use image::codecs::hdr::HdrDecoder;
    use image::ImageDecoder;

    let decoder = HdrDecoder::new(Cursor::new(HDRI_BYTES))
        .map_err(|e| format!("flora IBL: HDR decode init: {e}"))?;
    let (w, h) = decoder.dimensions();
    let (w, h) = (w as usize, h as usize);
    // total_bytes = w*h*3*4 (Rgb32F). Read straight into f32 radiance.
    let mut buf = vec![0u8; w * h * 3 * 4];
    decoder
        .read_image(&mut buf)
        .map_err(|e| format!("flora IBL: HDR read: {e}"))?;
    // Reinterpret the byte buffer as f32 (native-endian, as `image` writes it).
    let mut pixels = vec![0f32; w * h * 3];
    for (dst, chunk) in pixels.iter_mut().zip(buf.chunks_exact(4)) {
        *dst = f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(Equirect { w, h, pixels })
}

// ── (1) SH9 irradiance projection ───────────────────────────────────────────
//
// Project the equirect to 9 RGB SH coefficients, cosine-convolve (Ramamoorthi A_l
// factors) so a single in-shader dot returns diffuse radiance, and ×ENV_INTENSITY.
// Equirect convention MUST match the shader's `equirect_uv`:
//   u = atan2(z, x)/2π + 0.5,  v = acos(y)/π   (Y-up).
// So for texel (x,y): phi = (u-0.5)*2π = atan2(z,x); theta = v*π = acos(dir.y).

/// Real SH basis Y_lm(dir) for l=0..2, 9 bands. Standard (Sloan) polynomial form.
#[inline]
fn sh_basis(d: [f32; 3]) -> [f32; 9] {
    let (x, y, z) = (d[0], d[1], d[2]);
    [
        0.282_095,             // Y00
        0.488_603 * y,         // Y1-1
        0.488_603 * z,         // Y10
        0.488_603 * x,         // Y11
        1.092_548 * x * y,     // Y2-2
        1.092_548 * y * z,     // Y2-1
        0.315_392 * (3.0 * z * z - 1.0), // Y20
        1.092_548 * x * z,     // Y21
        0.546_274 * (x * x - y * y),     // Y22
    ]
}

/// Compute the 9 cosine-convolved, intensity-scaled RGB SH coefficients.
pub fn sh9(env: &Equirect) -> [[f32; 4]; 9] {
    // Raw projection: ∫ L(dir) Y_lm(dir) dω over the sphere.
    let mut coeffs = [[0f64; 3]; 9];
    let mut weight_sum = 0f64;

    let inv_w = 1.0 / env.w as f64;
    let inv_h = 1.0 / env.h as f64;
    // dω for an equirect texel = (2π/w)(π/h)·sin(theta). The 2π·π/(w·h) constant
    // factors out of the per-texel loop (folded into the final normalize below).
    for y in 0..env.h {
        // v in (0,1): texel center. theta = v*π. sin(theta) is the area weight.
        let v = (y as f64 + 0.5) * inv_h;
        let theta = v * std::f64::consts::PI;
        let sin_t = theta.sin();
        let (st, ct) = (theta.sin() as f32, theta.cos() as f32);
        for x in 0..env.w {
            let u = (x as f64 + 0.5) * inv_w;
            let phi = (u - 0.5) * 2.0 * std::f64::consts::PI; // atan2(z,x)
            // dir from (u,v): y = cos(theta); x = sin(theta)cos(phi); z = sin(theta)sin(phi).
            let dir = [
                st * (phi as f32).cos(),
                ct,
                st * (phi as f32).sin(),
            ];
            let basis = sh_basis(dir);
            let c = env.texel(x, y);
            for (b, band) in basis.iter().enumerate() {
                let bw = (*band as f64) * sin_t;
                coeffs[b][0] += c[0] as f64 * bw;
                coeffs[b][1] += c[1] as f64 * bw;
                coeffs[b][2] += c[2] as f64 * bw;
            }
            weight_sum += sin_t;
        }
    }

    // Normalize: the discrete sum approximates ∫·dω where the true sphere solid
    // angle is 4π. weight_sum ≈ Σ sin(theta); scale so Σ(sin)·k = 4π.
    let norm = (4.0 * std::f64::consts::PI) / weight_sum;

    // Ramamoorthi cosine-lobe convolution factors A_l (irradiance = E(n) =
    // Σ A_l L_lm Y_lm(n)); A0=π, A1=2π/3, A2=π/4. Dividing by π converts
    // irradiance→diffuse radiance (Lambert albedo/π folds the π), so the shader
    // multiplies straight by albedo. Net per-band factor = A_l/π.
    let a_over_pi = [
        1.0,             // A0/π = 1
        2.0 / 3.0,       // A1/π
        2.0 / 3.0,
        2.0 / 3.0,
        1.0 / 4.0,       // A2/π
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
        1.0 / 4.0,
    ];

    let mut out = [[0f32; 4]; 9];
    for b in 0..9 {
        let scale = norm * a_over_pi[b] * ENV_INTENSITY as f64;
        out[b][0] = (coeffs[b][0] * scale) as f32;
        out[b][1] = (coeffs[b][1] * scale) as f32;
        out[b][2] = (coeffs[b][2] * scale) as f32;
        out[b][3] = 0.0;
    }
    out
}

// ── (2) Roughness mip chain ─────────────────────────────────────────────────
//
// Build a 2D equirect mip pyramid: mip 0 = full-res linear radiance, each coarser
// mip = a 2×2 box downsample of the previous. Coarser mips read as rougher
// reflections in-shader (roughness→LOD). // ponytail: a plain box pyramid, NOT a
// per-roughness GGX-importance prefilter. The wider footprint of the lower mips
// approximates the rougher lobe well enough for the soft IBL look; the proper
// PMREM prefilter is the deferred upgrade.

/// Build the equirect mip chain. `max_mips` caps the pyramid (~7 for a 1k image →
/// roughness 0→1 over LOD 0→6). Mip 0 is the full-res equirect as RGBA.
pub fn mip_chain(env: &Equirect, max_mips: usize) -> Vec<Mip> {
    // Mip 0: pack the RGB equirect into RGBA (A=1 pad) for the RGBA16F image.
    let mut mip0 = Vec::with_capacity(env.w * env.h * 4);
    for i in 0..env.w * env.h {
        let s = i * 3;
        mip0.push(env.pixels[s]);
        mip0.push(env.pixels[s + 1]);
        mip0.push(env.pixels[s + 2]);
        mip0.push(1.0);
    }
    let mut mips = vec![Mip {
        w: env.w,
        h: env.h,
        rgba: mip0,
    }];

    while mips.len() < max_mips {
        let prev = mips.last().unwrap();
        if prev.w <= 1 || prev.h <= 1 {
            break;
        }
        let nw = (prev.w / 2).max(1);
        let nh = (prev.h / 2).max(1);
        let mut rgba = vec![0f32; nw * nh * 4];
        for y in 0..nh {
            for x in 0..nw {
                // 2×2 box average. Wrap U (equirect is periodic in longitude);
                // clamp V (poles). // ponytail: a box tap is the laziest blur;
                // the cumulative pyramid still widens correctly per LOD.
                let mut acc = [0f32; 4];
                for dy in 0..2usize {
                    for dx in 0..2usize {
                        let sx = (x * 2 + dx) % prev.w;
                        let sy = (y * 2 + dy).min(prev.h - 1);
                        let si = (sy * prev.w + sx) * 4;
                        acc[0] += prev.rgba[si];
                        acc[1] += prev.rgba[si + 1];
                        acc[2] += prev.rgba[si + 2];
                        acc[3] += prev.rgba[si + 3];
                    }
                }
                let di = (y * nw + x) * 4;
                rgba[di] = acc[0] * 0.25;
                rgba[di + 1] = acc[1] * 0.25;
                rgba[di + 2] = acc[2] * 0.25;
                rgba[di + 3] = acc[3] * 0.25;
            }
        }
        mips.push(Mip { w: nw, h: nh, rgba });
    }
    mips
}

/// Decode + bake the full IBL set (SH9 + mip chain). Called once at renderer
/// construction. `max_mips` ~7 maps roughness 0→1 onto LOD 0→6.
pub fn bake(max_mips: usize) -> Result<IblData, String> {
    let env = decode_hdri()?;
    let sh9 = sh9(&env);
    let mips = mip_chain(&env, max_mips);
    Ok(IblData { sh9, mips })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hdri_decodes_and_bakes() {
        let env = decode_hdri().expect("decode");
        assert!(env.w >= 512 && env.h >= 256, "1k equirect expected");
        assert_eq!(env.pixels.len(), env.w * env.h * 3);

        let sh = sh9(&env);
        // The DC term (band 0) is the average irradiance; for a daylight HDRI it
        // must be positive and finite in every channel.
        for c in 0..3 {
            assert!(sh[0][c].is_finite() && sh[0][c] > 0.0, "DC band positive");
        }

        let mips = mip_chain(&env, 7);
        assert_eq!(mips[0].w, env.w);
        assert!(mips.len() >= 2);
        // Each mip halves until the cap.
        for w in mips.windows(2) {
            assert!(w[1].w <= w[0].w && w[1].h <= w[0].h);
        }
    }
}

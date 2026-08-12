//! Ocean wave spectrum + initial frequency-domain field `h0(k)`.
//!
//! Ported from poseidon (`src/ocean/spectrum.js`), itself from gasgiant/FFT-Ocean
//! (Tessendorf 2001 + Horvath 2015): a JONSWAP amplitude × TMA depth correction ×
//! Donelan-Banner directional spread × short-wave fade, summed over a wind-sea and
//! a swell spectrum. Computed once on the CPU; the per-frame sim ([`super::sim`])
//! evolves it in time and inverse-FFTs it.

use super::fft::{Cx, CZERO};

/// One directional spectrum (a wind-sea or a swell component).
#[derive(Clone, Copy, Debug)]
pub struct Spectrum {
    pub scale: f32,
    pub wind_speed: f32,
    pub wind_dir_rad: f32,
    pub fetch: f32,
    pub spread_blend: f32,
    pub swell: f32,
    pub gamma: f32,
    pub short_waves_fade: f32,
}

/// All physical params for one FFT cascade.
#[derive(Clone, Copy, Debug)]
pub struct CascadeParams {
    pub n: usize,
    pub length_scale: f32,
    pub cutoff_low: f32,
    pub cutoff_high: f32,
    pub g: f32,
    pub depth: f32,
    pub local: Spectrum,
    pub swell: Spectrum,
}

/// Per-texel constants for the time evolution (`(kx, 1/|k|, kz, omega)`).
#[derive(Clone, Copy, Default)]
pub struct Wave {
    pub kx: f32,
    pub inv_k: f32,
    pub kz: f32,
    pub omega: f32,
}

/// Initial spectrum for one cascade — computed once.
pub struct CascadeInit {
    pub length_scale: f32,
    /// `h0(k)`, row-major `n*n`.
    pub h0: Vec<Cx>,
    /// `conj(h0(-k))`, row-major `n*n`.
    pub h0_conj: Vec<Cx>,
    pub wave: Vec<Wave>,
}

// ── Dispersion (poseidon spectrum.js:15-24) ────────────────────────────────────

fn frequency(k: f32, g: f32, depth: f32) -> f32 {
    (g * k * (k * depth).min(20.0).tanh()).sqrt()
}

fn frequency_derivative(k: f32, g: f32, depth: f32) -> f32 {
    let th = (k * depth).min(20.0).tanh();
    let ch = (k * depth).cosh();
    let w = frequency(k, g, depth).max(1e-6);
    g * (depth * k / (ch * ch) + th) / (2.0 * w)
}

// ── Directional spread (Donelan-Banner cosine-2s) ──────────────────────────────

fn normalisation_factor(s: f32) -> f32 {
    let s2 = s * s;
    let s3 = s2 * s;
    let s4 = s3 * s;
    if s < 5.0 {
        -0.000564 * s4 + 0.00776 * s3 - 0.044 * s2 + 0.192 * s + 0.163
    } else {
        -4.80e-08 * s4 + 1.07e-05 * s3 - 9.53e-04 * s2 + 5.93e-02 * s + 3.93e-01
    }
}

fn cosine_2s(theta: f32, s: f32) -> f32 {
    normalisation_factor(s) * (theta * 0.5).cos().abs().powf(2.0 * s)
}

fn spread_power(omega: f32, peak: f32) -> f32 {
    if omega <= peak {
        6.97 * (omega / peak).powf(5.0)
    } else {
        9.77 * (omega / peak).powf(-2.5)
    }
}

/// Directional spreading `D(θ)` for one spectrum.
fn direction_spectrum(theta: f32, omega: f32, sp: &Spectrum, peak: f32) -> f32 {
    let swell = sp.swell.clamp(0.01, 1.0);
    let s = spread_power(omega, peak) + 16.0 * (omega / peak).min(20.0).tanh() * swell * swell;
    let base = (2.0 / std::f32::consts::PI) * theta.cos() * theta.cos();
    let cos2s = cosine_2s(theta - sp.wind_dir_rad, s);
    base + (cos2s - base) * sp.spread_blend // mix(base, cos2s, spread_blend)
}

// ── JONSWAP × TMA (poseidon spectrum.js:53-77) ─────────────────────────────────

fn tma_correction(omega: f32, g: f32, depth: f32) -> f32 {
    let omega_h = omega * (depth / g).sqrt();
    if omega_h <= 1.0 {
        0.5 * omega_h * omega_h
    } else if omega_h < 2.0 {
        1.0 - 0.5 * (2.0 - omega_h) * (2.0 - omega_h)
    } else {
        1.0
    }
}

fn jonswap(omega: f32, sp: &Spectrum, peak: f32, g: f32, depth: f32) -> f32 {
    let alpha = 0.076 * ((g * sp.fetch) / (sp.wind_speed * sp.wind_speed)).powf(-0.22);
    let sigma = if omega <= peak { 0.07 } else { 0.09 };
    let r = (-(omega - peak) * (omega - peak) / (2.0 * sigma * sigma * peak * peak)).exp();
    sp.scale
        * tma_correction(omega, g, depth)
        * alpha
        * g
        * g
        * (1.0 / omega.powf(5.0))
        * (-1.25 * (peak / omega).powf(4.0)).exp()
        * sp.gamma.powf(r)
}

fn peak_omega(sp: &Spectrum, g: f32) -> f32 {
    22.0 * ((sp.wind_speed * sp.fetch) / (g * g)).powf(-0.33)
}

/// Energy of one spectrum at angular frequency `omega` and direction `theta`.
fn spectrum_energy(omega: f32, theta: f32, k: f32, sp: &Spectrum, g: f32, depth: f32) -> f32 {
    let peak = peak_omega(sp, g);
    let s = jonswap(omega, sp, peak, g, depth);
    let d = direction_spectrum(theta, omega, sp, peak);
    let fade = (-sp.short_waves_fade * sp.short_waves_fade * k * k).exp();
    (s * d * fade).max(0.0)
}

// ── h0(k) build (poseidon spectrum.js:119-163) ─────────────────────────────────

/// Build the initial spectrum for a cascade from a shared complex-gaussian noise
/// field (`noise[i] = (re, im)`, both N(0,1)), of length `n*n`.
pub fn build_cascade(p: &CascadeParams, noise: &[Cx]) -> CascadeInit {
    let n = p.n;
    assert_eq!(noise.len(), n * n);
    let delta_k = 2.0 * std::f32::consts::PI / p.length_scale;
    let half = (n / 2) as i32;

    let mut h0k = vec![CZERO; n * n];
    let mut wave = vec![Wave::default(); n * n];

    for y in 0..n {
        for x in 0..n {
            let id = y * n + x;
            let nx = x as i32 - half;
            let nz = y as i32 - half;
            let kx = nx as f32 * delta_k;
            let kz = nz as f32 * delta_k;
            let k_len = (kx * kx + kz * kz).sqrt();
            let k_safe = k_len.max(p.cutoff_low);
            let theta = kz.atan2(kx);

            let omega = frequency(k_safe, p.g, p.depth);
            let d_omega = frequency_derivative(k_safe, p.g, p.depth);

            let energy = spectrum_energy(omega, theta, k_safe, &p.local, p.g, p.depth)
                + spectrum_energy(omega, theta, k_safe, &p.swell, p.g, p.depth);

            let in_band = if k_len >= p.cutoff_low && k_len <= p.cutoff_high { 1.0 } else { 0.0 };
            let amplitude = (energy * 2.0 * d_omega.abs() / k_safe * delta_k * delta_k)
                .max(0.0)
                .sqrt();

            h0k[id] = noise[id].scale(amplitude * in_band);
            wave[id] = Wave { kx, inv_k: 1.0 / k_safe, kz, omega };
        }
    }

    // Conjugate pack: h0_conj[id] = conj( h0k[ (N-x)%N, (N-y)%N ] ).
    let mut h0 = vec![CZERO; n * n];
    let mut h0_conj = vec![CZERO; n * n];
    for y in 0..n {
        for x in 0..n {
            let id = y * n + x;
            let xc = (n - x) % n;
            let yc = (n - y) % n;
            let idc = yc * n + xc;
            h0[id] = h0k[id];
            h0_conj[id] = h0k[idc].conj();
        }
    }

    CascadeInit { length_scale: p.length_scale, h0, h0_conj, wave }
}

// ── Shared gaussian noise (Box-Muller from a seeded PCG) ───────────────────────

/// Deterministic complex-gaussian noise field, `n*n` samples (re,im ~ N(0,1)).
pub fn gaussian_noise(n: usize, seed: u64) -> Vec<Cx> {
    let mut state = seed ^ 0x9e37_79b9_7f4a_7c15;
    let mut next_u01 = || -> f32 {
        // PCG-ish xorshift; good enough for noise.
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // map to (0,1], avoid 0 for the log.
        ((state >> 11) as f32 + 1.0) / (((1u64 << 53) + 1) as f32)
    };
    (0..n * n)
        .map(|_| {
            let u1 = next_u01();
            let u2 = next_u01();
            let mag = (-2.0 * u1.ln()).sqrt();
            Cx::new(mag * (2.0 * std::f32::consts::PI * u2).cos(), mag * (2.0 * std::f32::consts::PI * u2).sin())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_params(n: usize, length_scale: f32) -> CascadeParams {
        CascadeParams {
            n,
            length_scale,
            cutoff_low: 1e-4,
            cutoff_high: 9999.0,
            g: 9.81,
            depth: 500.0,
            local: Spectrum {
                scale: 1.0, wind_speed: 16.0, wind_dir_rad: 45f32.to_radians(),
                fetch: 100_000.0, spread_blend: 1.0, swell: 0.2, gamma: 3.3, short_waves_fade: 0.02,
            },
            swell: Spectrum {
                scale: 0.8, wind_speed: 2.0, wind_dir_rad: 70f32.to_radians(),
                fetch: 300_000.0, spread_blend: 1.0, swell: 1.0, gamma: 3.3, short_waves_fade: 0.01,
            },
        }
    }

    #[test]
    fn noise_is_roughly_unit_gaussian() {
        let noise = gaussian_noise(64, 42);
        let n = noise.len() as f32 * 2.0;
        let mean: f32 = noise.iter().map(|c| c.re + c.im).sum::<f32>() / n;
        let var: f32 = noise.iter().map(|c| c.re * c.re + c.im * c.im).sum::<f32>() / n;
        assert!(mean.abs() < 0.1, "mean {mean}");
        assert!((var - 1.0).abs() < 0.15, "var {var}");
    }

    #[test]
    fn h0_is_finite_and_has_energy() {
        let n = 64;
        let p = test_params(n, 250.0);
        let noise = gaussian_noise(n, 7);
        let c = build_cascade(&p, &noise);
        assert_eq!(c.h0.len(), n * n);
        let mut energy = 0.0f32;
        for h in &c.h0 {
            assert!(h.re.is_finite() && h.im.is_finite(), "h0 must be finite");
            energy += h.re * h.re + h.im * h.im;
        }
        assert!(energy > 0.0, "spectrum must carry energy");
    }

    #[test]
    fn band_mask_limits_cascade() {
        // A narrow band should zero most of the field.
        let n = 64;
        let mut p = test_params(n, 250.0);
        p.cutoff_low = 0.1;
        p.cutoff_high = 0.2;
        let noise = gaussian_noise(n, 1);
        let c = build_cascade(&p, &noise);
        let nonzero = c.h0.iter().filter(|h| h.re != 0.0 || h.im != 0.0).count();
        assert!(nonzero > 0 && nonzero < n * n, "band should be a subset, got {nonzero}");
    }
}

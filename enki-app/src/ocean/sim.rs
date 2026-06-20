//! Per-frame spectral ocean evolution: `h0(k)` → `h(k,t)` → inverse FFT →
//! displacement + normal + Jacobian foam, packed for GPU upload.
//!
//! Ported from poseidon (`spectrum.js buildTimeDependent`, `maps.js`,
//! `oceanSurfaceMaterial.js`). Runs on the CPU; one [`OceanTexel`] per grid point
//! is uploaded to a storage buffer the wave shader samples (tiled by `length_scale`).

use bytemuck::{Pod, Zeroable};

use super::fft::{ifft2, Cx};
use super::spectrum::{build_cascade, gaussian_noise, CascadeInit, CascadeParams};

/// One ocean grid sample, std430-friendly (two `vec4`, 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct OceanTexel {
    /// xyz = displacement (m, choppy x/z + height y), w = foam (0 none .. 1 full).
    pub disp: [f32; 4],
    /// xyz = surface normal (unit), w unused.
    pub normal: [f32; 4],
}

/// Tunable wave params the sim reads each step (live-editable from the GUI).
#[derive(Clone, Copy, Debug)]
pub struct WaveParams {
    /// Horizontal displacement gain ("choppiness").
    pub choppiness: f32,
    /// Jacobian value below which foam forms.
    pub foam_threshold: f32,
    /// Foam recovery rate (lower = foam lingers).
    pub foam_decay: f32,
    /// Overall vertical amplitude multiplier (artistic; 1.0 = physical).
    pub amplitude: f32,
}

impl Default for WaveParams {
    fn default() -> Self {
        Self { choppiness: 1.3, foam_threshold: 0.4, foam_decay: 0.4, amplitude: 1.0 }
    }
}

/// One FFT cascade's evolving state.
struct Cascade {
    init: CascadeInit,
    /// Persistent foam/turbulence accumulator (1.0 = un-foamed).
    turbulence: Vec<f32>,
}

/// CPU spectral ocean. Holds the cascades and produces a packed field each frame.
pub struct OceanSim {
    pub n: usize,
    cascades: Vec<Cascade>,
    /// Combined output (single cascade for now), `n*n` texels.
    out: Vec<OceanTexel>,
    // Reused scratch (4 packed complex IFFT inputs).
    f_xz: Vec<Cx>,
    f_ydxz: Vec<Cx>,
    f_yxyz: Vec<Cx>,
    f_xxzz: Vec<Cx>,
}

impl OceanSim {
    /// Build the sim from a set of cascade params (share one gaussian noise field).
    pub fn new(params: &[CascadeParams], seed: u64) -> Self {
        let n = params[0].n;
        let noise = gaussian_noise(n, seed);
        let cascades: Vec<Cascade> = params
            .iter()
            .map(|p| Cascade { init: build_cascade(p, &noise), turbulence: vec![1.0; n * n] })
            .collect();
        Self {
            n,
            cascades,
            out: vec![OceanTexel::zeroed(); n * n],
            f_xz: vec![Cx::ZERO; n * n],
            f_ydxz: vec![Cx::ZERO; n * n],
            f_yxyz: vec![Cx::ZERO; n * n],
            f_xxzz: vec![Cx::ZERO; n * n],
        }
    }

    /// The (single) cascade's spatial tile size in metres — the shader tiles by it.
    pub fn length_scale(&self) -> f32 {
        self.cascades[0].init.length_scale
    }

    /// Evolve to absolute `time` seconds (dt for foam accumulation), returning the
    /// packed per-texel field. Single-cascade for now (sums trivially over one).
    pub fn step(&mut self, time: f32, dt: f32, wp: &WaveParams) -> &[OceanTexel] {
        let n = self.n;
        // Clear combined output.
        for t in &mut self.out {
            *t = OceanTexel::zeroed();
        }

        let lambda = wp.choppiness;
        for c in self.cascades.iter_mut() {
            // ── 1. Time-dependent spectrum → 4 packed complex IFFT inputs ──────
            for id in 0..n * n {
                let w = c.init.wave[id];
                let phase = w.omega * time;
                let ex = Cx::new(phase.cos(), phase.sin());
                // h(k,t) = h0(k)·e^{iωt} + conj(h0(-k))·e^{-iωt}
                let h = c.init.h0[id].mul(ex).add(c.init.h0_conj[id].mul(ex.conj()));
                let ih = Cx::new(-h.im, h.re); // i·h
                let kx = w.kx;
                let kz = w.kz;
                let ik = w.inv_k;

                let disp_x = ih.scale(kx * ik);
                let disp_y = h;
                let disp_z = ih.scale(kz * ik);
                let disp_xdx = h.scale(-kx * kx * ik);
                let disp_ydx = ih.scale(kx);
                let disp_ydz = ih.scale(kz);
                let disp_zdx = h.scale(-kx * kz * ik);
                let disp_zdz = h.scale(-kz * kz * ik);

                // Pack two real fields per complex IFFT (A + iB trick).
                self.f_xz[id] = Cx::new(disp_x.re - disp_z.im, disp_x.im + disp_z.re);
                self.f_ydxz[id] = Cx::new(disp_y.re - disp_zdx.im, disp_y.im + disp_zdx.re);
                self.f_yxyz[id] = Cx::new(disp_ydx.re - disp_ydz.im, disp_ydx.im + disp_ydz.re);
                self.f_xxzz[id] = Cx::new(disp_xdx.re - disp_zdz.im, disp_xdx.im + disp_zdz.re);
            }

            // ── 2. Inverse FFT each packed field ──────────────────────────────
            ifft2(&mut self.f_xz, n);
            ifft2(&mut self.f_ydxz, n);
            ifft2(&mut self.f_yxyz, n);
            ifft2(&mut self.f_xxzz, n);

            // ── 3. Decode, assemble displacement / normal / foam ──────────────
            for y in 0..n {
                for x in 0..n {
                    let id = y * n + x;
                    // (-1)^(x+y) reconciles the centered (fftshifted) spectrum.
                    let sign = if (x + y) & 1 == 0 { 1.0 } else { -1.0 };

                    let dx = self.f_xz[id].re * sign;
                    let dz = self.f_xz[id].im * sign;
                    let dy = self.f_ydxz[id].re * sign; // height
                    let ddz_dx = self.f_ydxz[id].im * sign;
                    let ddy_dx = self.f_yxyz[id].re * sign;
                    let ddy_dz = self.f_yxyz[id].im * sign;
                    let ddx_dx = self.f_xxzz[id].re * sign;
                    let ddz_dz = self.f_xxzz[id].im * sign;

                    // Jacobian of horizontal displacement → foam.
                    let jxx = 1.0 + lambda * ddx_dx;
                    let jzz = 1.0 + lambda * ddz_dz;
                    let jxz = lambda * ddz_dx;
                    let jacobian = jxx * jzz - jxz * jxz;

                    // Accumulate turbulence: snap down on a fold, recover slowly.
                    let prev = c.turbulence[id];
                    let turb = jacobian.min(prev + dt * wp.foam_decay / jacobian.max(0.5));
                    c.turbulence[id] = turb;
                    let foam = ((wp.foam_threshold - turb) * 2.5).clamp(0.0, 1.0);

                    // Fold-aware slope → normal.
                    let slope_x = ddy_dx / (1.0 + lambda * ddx_dx);
                    let slope_z = ddy_dz / (1.0 + lambda * ddz_dz);
                    let inv_len = 1.0 / (slope_x * slope_x + 1.0 + slope_z * slope_z).sqrt();
                    let nrm = [-slope_x * inv_len, inv_len, -slope_z * inv_len];

                    let o = &mut self.out[id];
                    o.disp[0] += lambda * dx;
                    o.disp[1] += dy * wp.amplitude;
                    o.disp[2] += lambda * dz;
                    o.disp[3] = o.disp[3].max(foam);
                    // Single cascade: normal is this cascade's (sum/renormalize once multi).
                    o.normal = [nrm[0], nrm[1], nrm[2], 0.0];
                }
            }
        }
        &self.out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocean::spectrum::{CascadeParams, Spectrum};

    fn params(n: usize) -> CascadeParams {
        CascadeParams {
            n,
            length_scale: 250.0,
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
    fn step_produces_finite_waves_and_moves() {
        let mut sim = OceanSim::new(&[params(64)], 123);
        let wp = WaveParams::default();
        let a = sim.step(0.0, 0.016, &wp).to_vec();
        // All finite, and the height field is non-flat (real waves).
        let mut min_h = f32::INFINITY;
        let mut max_h = f32::NEG_INFINITY;
        for t in &a {
            for v in t.disp.iter().chain(t.normal.iter()) {
                assert!(v.is_finite(), "ocean field must be finite");
            }
            min_h = min_h.min(t.disp[1]);
            max_h = max_h.max(t.disp[1]);
        }
        assert!(max_h - min_h > 1e-4, "height field should vary, range {}", max_h - min_h);

        // Surface evolves with time (waves animate).
        let b = sim.step(2.0, 0.016, &wp).to_vec();
        let moved: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x.disp[1] - y.disp[1]).abs()).sum();
        assert!(moved > 1e-3, "waves should move over time, delta {moved}");
    }

    #[test]
    fn normals_are_unit_length() {
        let mut sim = OceanSim::new(&[params(32)], 9);
        let f = sim.step(1.0, 0.016, &WaveParams::default());
        for t in f {
            let n = t.normal;
            let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-3, "normal not unit: {len}");
        }
    }
}

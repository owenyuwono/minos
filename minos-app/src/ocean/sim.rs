//! Per-frame spectral ocean evolution: `h0(k)` → `h(k,t)` → inverse FFT →
//! displacement + slope derivatives + Jacobian turbulence, packed for GPU upload.
//!
//! Ported from poseidon (`spectrum.js buildTimeDependent`, `maps.js`). Runs on the
//! CPU. Multiple **cascades** (disjoint wavenumber bands at different tile sizes)
//! are written one after another into a single buffer (`cascade-major`); the wave
//! shader samples each at its own world tile size and sums them — this is what
//! removes the single-tile repetition. Per cascade, per grid point, an
//! [`OceanTexel`] carries the raw displacement + derivatives + turbulence; the
//! shader combines them (normal + foam), so cascades sum correctly.

use bytemuck::{Pod, Zeroable};

use super::fft::{Cx, Ifft2, CZERO};
use super::spectrum::{build_cascade, gaussian_noise, CascadeInit, CascadeParams};

/// One ocean grid sample, std430-friendly (two `vec4`, 32 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct OceanTexel {
    /// xyz = displacement (m, choppy x/z + height y), w = raw Jacobian turbulence
    /// (≈1 calm, <`foam_threshold` = whitecap). Shader thresholds + sums for foam.
    pub disp: [f32; 4],
    /// Slope derivatives: (dDy/dx, dDy/dz, λ·dDx/dx, λ·dDz/dz). Shader sums across
    /// cascades then forms the fold-aware normal.
    pub deriv: [f32; 4],
}

/// Tunable wave params the sim reads each step (live-editable from the GUI).
#[derive(Clone, Copy, Debug)]
pub struct WaveParams {
    /// Horizontal displacement gain ("choppiness").
    pub choppiness: f32,
    /// Jacobian value below which foam forms (consumed in the shader).
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
    /// Per-cascade output, cascade-major: cascade `c` occupies `out[c*n*n .. (c+1)*n*n]`.
    out: Vec<OceanTexel>,
    // Reused scratch (4 packed complex IFFT inputs).
    f_xz: Vec<Cx>,
    f_ydxz: Vec<Cx>,
    f_yxyz: Vec<Cx>,
    f_xxzz: Vec<Cx>,
    ifft: Ifft2,
}

impl OceanSim {
    /// Build the sim from a set of cascade params (share one gaussian noise field).
    pub fn new(params: &[CascadeParams], seed: u64) -> Self {
        let n = params[0].n;
        let noise = gaussian_noise(n, seed);
        let cascades: Vec<Cascade> = params
            .iter()
            .map(|p| {
                debug_assert_eq!(p.n, n, "all cascades must share the FFT resolution");
                Cascade { init: build_cascade(p, &noise), turbulence: vec![1.0; n * n] }
            })
            .collect();
        let count = cascades.len();
        Self {
            n,
            cascades,
            out: vec![OceanTexel::zeroed(); n * n * count],
            f_xz: vec![CZERO; n * n],
            f_ydxz: vec![CZERO; n * n],
            f_yxyz: vec![CZERO; n * n],
            f_xxzz: vec![CZERO; n * n],
            ifft: Ifft2::new(n),
        }
    }

    pub fn cascade_count(&self) -> usize {
        self.cascades.len()
    }

    /// Per-cascade spatial tile sizes (metres) — the shader tiles each by its own.
    pub fn length_scales(&self) -> Vec<f32> {
        self.cascades.iter().map(|c| c.init.length_scale).collect()
    }

    /// Evolve to absolute `time` seconds (dt for foam accumulation), returning the
    /// packed per-cascade field (cascade-major, `cascade_count * n*n` texels).
    pub fn step(&mut self, time: f32, dt: f32, wp: &WaveParams) -> &[OceanTexel] {
        let n = self.n;
        let lambda = wp.choppiness;

        for (ci, c) in self.cascades.iter_mut().enumerate() {
            // ── 1. Time-dependent spectrum → 4 packed complex IFFT inputs ──────
            for id in 0..n * n {
                let w = c.init.wave[id];
                let phase = w.omega * time;
                let ex = Cx::new(phase.cos(), phase.sin());
                let h = c.init.h0[id] * ex + c.init.h0_conj[id] * ex.conj();
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

                self.f_xz[id] = Cx::new(disp_x.re - disp_z.im, disp_x.im + disp_z.re);
                self.f_ydxz[id] = Cx::new(disp_y.re - disp_zdx.im, disp_y.im + disp_zdx.re);
                self.f_yxyz[id] = Cx::new(disp_ydx.re - disp_ydz.im, disp_ydx.im + disp_ydz.re);
                self.f_xxzz[id] = Cx::new(disp_xdx.re - disp_zdz.im, disp_xdx.im + disp_zdz.re);
            }

            // ── 2. Inverse FFT each packed field ──────────────────────────────
            self.ifft.run(&mut self.f_xz);
            self.ifft.run(&mut self.f_ydxz);
            self.ifft.run(&mut self.f_yxyz);
            self.ifft.run(&mut self.f_xxzz);

            // ── 3. Decode, store per-cascade displacement + derivatives + turb ─
            let base = ci * n * n;
            for y in 0..n {
                for x in 0..n {
                    let id = y * n + x;
                    let sign = if (x + y) & 1 == 0 { 1.0 } else { -1.0 };

                    let dx = self.f_xz[id].re * sign;
                    let dz = self.f_xz[id].im * sign;
                    let dy = self.f_ydxz[id].re * sign;
                    let ddz_dx = self.f_ydxz[id].im * sign;
                    let ddy_dx = self.f_yxyz[id].re * sign;
                    let ddy_dz = self.f_yxyz[id].im * sign;
                    let ddx_dx = self.f_xxzz[id].re * sign;
                    let ddz_dz = self.f_xxzz[id].im * sign;

                    // Jacobian of horizontal displacement → turbulence accumulation.
                    let jxx = 1.0 + lambda * ddx_dx;
                    let jzz = 1.0 + lambda * ddz_dz;
                    let jxz = lambda * ddz_dx;
                    let jacobian = jxx * jzz - jxz * jxz;
                    let prev = c.turbulence[id];
                    let turb = jacobian.min(prev + dt * wp.foam_decay / jacobian.max(0.5));
                    c.turbulence[id] = turb;

                    self.out[base + id] = OceanTexel {
                        disp: [lambda * dx, dy * wp.amplitude, lambda * dz, turb],
                        deriv: [ddy_dx, ddy_dz, lambda * ddx_dx, lambda * ddz_dz],
                    };
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

    fn params(n: usize, length_scale: f32) -> CascadeParams {
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
    fn multi_cascade_is_cascade_major_and_finite() {
        let n = 32;
        let mut sim = OceanSim::new(&[params(n, 800.0), params(n, 200.0)], 1);
        assert_eq!(sim.cascade_count(), 2);
        let f = sim.step(1.0, 0.016, &WaveParams::default());
        assert_eq!(f.len(), 2 * n * n);
        for t in f {
            for v in t.disp.iter().chain(t.deriv.iter()) {
                assert!(v.is_finite());
            }
        }
        // Different cascades carry different fields (disjoint bands).
        let c0_h: f32 = (0..n * n).map(|i| f[i].disp[1].abs()).sum();
        let c1_h: f32 = (0..n * n).map(|i| f[n * n + i].disp[1].abs()).sum();
        assert!(c0_h > 0.0 && c1_h > 0.0);
    }

    #[test]
    fn waves_animate() {
        let n = 32;
        let mut sim = OceanSim::new(&[params(n, 250.0)], 7);
        let a = sim.step(0.0, 0.016, &WaveParams::default()).to_vec();
        let b = sim.step(2.0, 0.016, &WaveParams::default()).to_vec();
        let moved: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x.disp[1] - y.disp[1]).abs()).sum();
        assert!(moved > 1e-3, "waves should move over time, delta {moved}");
    }
}

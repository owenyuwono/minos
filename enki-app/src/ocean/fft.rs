//! FFT for the spectral ocean — backed by `rustfft`.
//!
//! The hand-rolled radix-2 was only viable at N≤64; poseidon runs at **N=256**,
//! and rustfft (SIMD, planned) makes N=256 × 3 cascades affordable on the CPU.
//! The transform is the **unnormalized inverse** (no 1/N²), matching the h0
//! amplitude calibration; the centered-spectrum `(-1)^(x+y)` reconciliation is
//! applied at decode time in [`super::sim`].

pub use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};
use std::sync::Arc;

/// Complex sample (alias so the sim/spectrum read naturally).
pub type Cx = Complex<f32>;
pub const CZERO: Cx = Complex::new(0.0, 0.0);

/// Cached inverse 2D FFT (rows then columns), unnormalized. Reuses scratch +
/// column buffers across calls so per-frame evolution allocates nothing.
pub struct Ifft2 {
    n: usize,
    fft: Arc<dyn Fft<f32>>,
    col: Vec<Cx>,
    scratch: Vec<Cx>,
}

impl Ifft2 {
    pub fn new(n: usize) -> Self {
        let fft = FftPlanner::new().plan_fft_inverse(n);
        let slen = fft.get_inplace_scratch_len();
        Self { n, fft, col: vec![CZERO; n], scratch: vec![CZERO; slen] }
    }

    /// In-place unnormalized inverse 2D FFT of an `n×n` row-major grid.
    pub fn run(&mut self, grid: &mut [Cx]) {
        let n = self.n;
        debug_assert_eq!(grid.len(), n * n);
        for r in 0..n {
            self.fft.process_with_scratch(&mut grid[r * n..r * n + n], &mut self.scratch);
        }
        for c in 0..n {
            for r in 0..n {
                self.col[r] = grid[r * n + c];
            }
            self.fft.process_with_scratch(&mut self.col, &mut self.scratch);
            for r in 0..n {
                grid[r * n + c] = self.col[r];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ifft2_of_dc_is_constant() {
        // Inverse (unnormalized) of a single DC impulse → 1 everywhere.
        let n = 8;
        let mut grid = vec![CZERO; n * n];
        grid[0] = Complex::new(1.0, 0.0);
        Ifft2::new(n).run(&mut grid);
        for g in &grid {
            assert!((g.re - 1.0).abs() < 1e-4 && g.im.abs() < 1e-4, "got {g:?}");
        }
    }

    #[test]
    fn forward_then_ifft2_roundtrips() {
        let n = 16;
        let orig: Vec<Cx> = (0..n * n)
            .map(|i| Complex::new((i as f32).sin(), (i as f32 * 0.3).cos()))
            .collect();
        let mut grid = orig.clone();
        let fwd = FftPlanner::new().plan_fft_forward(n);
        for r in 0..n {
            fwd.process(&mut grid[r * n..r * n + n]);
        }
        let mut col = vec![CZERO; n];
        for c in 0..n {
            for r in 0..n {
                col[r] = grid[r * n + c];
            }
            fwd.process(&mut col);
            for r in 0..n {
                grid[r * n + c] = col[r];
            }
        }
        Ifft2::new(n).run(&mut grid);
        let nn = (n * n) as f32;
        for (g, o) in grid.iter().zip(orig.iter()) {
            assert!((g.re / nn - o.re).abs() < 1e-3);
            assert!((g.im / nn - o.im).abs() < 1e-3);
        }
    }
}

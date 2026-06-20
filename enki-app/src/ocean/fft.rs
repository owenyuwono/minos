//! Minimal complex FFT for the spectral ocean (radix-2, iterative, in-place).
//!
//! ponytail: hand-rolled radix-2 Cooley-Tukey rather than pulling in `rustfft` —
//! the ocean only needs power-of-two square transforms and this is ~40 lines we
//! control + can unit-test. Upgrade path: move the whole sim to GPU compute
//! (storage buffers; enki-rhi already supports it) if the CPU cost bites.

use std::f32::consts::PI;

/// A complex number (`re + i·im`).
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub struct Cx {
    pub re: f32,
    pub im: f32,
}

impl Cx {
    pub const ZERO: Cx = Cx { re: 0.0, im: 0.0 };

    #[inline]
    pub fn new(re: f32, im: f32) -> Cx {
        Cx { re, im }
    }
    #[inline]
    pub fn add(self, o: Cx) -> Cx {
        Cx { re: self.re + o.re, im: self.im + o.im }
    }
    #[inline]
    pub fn sub(self, o: Cx) -> Cx {
        Cx { re: self.re - o.re, im: self.im - o.im }
    }
    #[inline]
    pub fn mul(self, o: Cx) -> Cx {
        Cx {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    #[inline]
    pub fn scale(self, s: f32) -> Cx {
        Cx { re: self.re * s, im: self.im * s }
    }
    #[inline]
    pub fn conj(self) -> Cx {
        Cx { re: self.re, im: -self.im }
    }
}

/// In-place radix-2 FFT. `len` must be a power of two. `inverse=false` is the
/// forward transform (e^{-i}), `true` the inverse (e^{+i}); neither normalizes —
/// divide by `len` yourself for a round-trip.
pub fn fft_inplace(a: &mut [Cx], inverse: bool) {
    let n = a.len();
    debug_assert!(n.is_power_of_two(), "FFT length must be a power of two");
    if n <= 1 {
        return;
    }

    // Bit-reversal permutation.
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            a.swap(i, j);
        }
    }

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let ang = sign * 2.0 * PI / len as f32;
        let wlen = Cx::new(ang.cos(), ang.sin());
        let half = len / 2;
        let mut i = 0;
        while i < n {
            let mut w = Cx::new(1.0, 0.0);
            for k in 0..half {
                let u = a[i + k];
                let v = a[i + k + half].mul(w);
                a[i + k] = u.add(v);
                a[i + k + half] = u.sub(v);
                w = w.mul(wlen);
            }
            i += len;
        }
        len <<= 1;
    }
}

/// In-place inverse 2D FFT of an `n×n` row-major grid (rows then columns),
/// **unnormalized** (the ocean's h0 amplitude is pre-calibrated, matching the
/// unnormalized inverse used in poseidon's GPU butterfly).
pub fn ifft2(grid: &mut [Cx], n: usize) {
    debug_assert_eq!(grid.len(), n * n);
    // Rows.
    for r in 0..n {
        fft_inplace(&mut grid[r * n..r * n + n], true);
    }
    // Columns.
    let mut col = vec![Cx::ZERO; n];
    for c in 0..n {
        for r in 0..n {
            col[r] = grid[r * n + c];
        }
        fft_inplace(&mut col, true);
        for r in 0..n {
            grid[r * n + c] = col[r];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naive DFT for cross-checking the FFT on small sizes.
    fn dft(a: &[Cx], inverse: bool) -> Vec<Cx> {
        let n = a.len();
        let sign = if inverse { 1.0 } else { -1.0 };
        (0..n)
            .map(|k| {
                let mut acc = Cx::ZERO;
                for (m, &x) in a.iter().enumerate() {
                    let ang = sign * 2.0 * PI * (k * m) as f32 / n as f32;
                    acc = acc.add(x.mul(Cx::new(ang.cos(), ang.sin())));
                }
                acc
            })
            .collect()
    }

    #[test]
    fn fft_matches_dft() {
        // Deterministic pseudo-random input.
        let n = 16;
        let mut a: Vec<Cx> = (0..n)
            .map(|i| Cx::new(((i * 7 + 3) % 11) as f32 - 5.0, ((i * 5 + 1) % 9) as f32 - 4.0))
            .collect();
        let reference = dft(&a, false);
        fft_inplace(&mut a, false);
        for (got, exp) in a.iter().zip(reference.iter()) {
            assert!((got.re - exp.re).abs() < 1e-3, "re {got:?} vs {exp:?}");
            assert!((got.im - exp.im).abs() < 1e-3, "im {got:?} vs {exp:?}");
        }
    }

    #[test]
    fn forward_inverse_roundtrips() {
        let n = 32;
        let orig: Vec<Cx> = (0..n).map(|i| Cx::new((i as f32).sin(), (i as f32 * 0.3).cos())).collect();
        let mut a = orig.clone();
        fft_inplace(&mut a, false);
        fft_inplace(&mut a, true);
        for (got, exp) in a.iter().zip(orig.iter()) {
            // inverse is unnormalized → divide by n.
            assert!((got.re / n as f32 - exp.re).abs() < 1e-4);
            assert!((got.im / n as f32 - exp.im).abs() < 1e-4);
        }
    }

    #[test]
    fn ifft2_roundtrips_a_delta() {
        // A 2D delta transforms (forward) to all-ones; inverse of all-ones → delta·N².
        let n = 8;
        let mut grid = vec![Cx::ZERO; n * n];
        grid[0] = Cx::new(1.0, 0.0);
        // forward 2D
        for r in 0..n {
            fft_inplace(&mut grid[r * n..r * n + n], false);
        }
        let mut col = vec![Cx::ZERO; n];
        for c in 0..n {
            for r in 0..n {
                col[r] = grid[r * n + c];
            }
            fft_inplace(&mut col, false);
            for r in 0..n {
                grid[r * n + c] = col[r];
            }
        }
        for g in &grid {
            assert!((g.re - 1.0).abs() < 1e-3 && g.im.abs() < 1e-3);
        }
        // inverse 2D → back to delta × N²
        ifft2(&mut grid, n);
        assert!((grid[0].re - (n * n) as f32).abs() < 1e-2);
        for g in &grid[1..] {
            assert!(g.re.abs() < 1e-2 && g.im.abs() < 1e-2);
        }
    }
}

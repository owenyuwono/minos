//! Deterministic PRNG — `mulberry32` port.
//!
//! Bit-exact Rust port of dryad's `rng.js`. ALL randomness in the flora
//! pipeline flows from here (no `rand`, no system entropy) so that a given
//! `(env, seed)` pair reproduces byte-identical output across runs and across
//! the JS reference.
//!
//! The JS source is:
//! ```js
//! export function mulberry32(seed) {
//!   let s = seed >>> 0;
//!   return function rng() {
//!     s |= 0; s = s + 0x6D2B79F5 | 0;
//!     let t = Math.imul(s ^ s >>> 15, 1 | s);
//!     t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
//!     return ((t ^ t >>> 14) >>> 0) / 4294967296;
//!   };
//! }
//! ```
//!
//! Port mapping (state held as `i32`, reinterpreted as `u32` at each `>>> n`):
//! - `Math.imul(a, b)` → `(a).wrapping_mul(b)` on `i32` (32-bit signed wrap).
//! - `s + 0x6D2B79F5 | 0` → `s.wrapping_add(0x6D2B79F5)` (`| 0` is the int32
//!   coercion, already implied by `i32` wrapping arithmetic).
//! - `x >>> n` (unsigned logical shift) → `(x as u32 >> n) as i32`.
//! - operator precedence preserved: `>>>` binds tighter than `^`, tighter than
//!   `|`. `s ^ s >>> 15` is `s ^ (s >>> 15)`.
//! - final `(t ^ t >>> 14) >>> 0` reinterprets to `u32`; `/ 4294967296` is
//!   `as f64 / 4294967296.0`, yielding `f64` in `[0, 1)`.

/// `mulberry32` PRNG state. Construct with [`Mulberry32::new`]; draw with
/// [`Mulberry32::next`] (f64 in `[0,1)`) or [`Mulberry32::next_u32`].
#[derive(Clone, Copy, Debug)]
pub struct Mulberry32 {
    /// Internal state, modeled as `i32` to match JS int32 semantics exactly.
    s: i32,
}

impl Mulberry32 {
    /// Seed the generator. JS `let s = seed >>> 0` — the seed is a `u32`;
    /// we store it reinterpreted as `i32`.
    #[inline]
    pub fn new(seed: u32) -> Self {
        Self { s: seed as i32 }
    }

    /// One draw → `f64` in `[0, 1)`. Bit-exact mirror of the JS `rng()` body.
    #[inline]
    pub fn next(&mut self) -> f64 {
        // s |= 0; s = s + 0x6D2B79F5 | 0;
        self.s = self.s.wrapping_add(0x6D2B7_9F5u32 as i32);
        let s = self.s;

        // let t = Math.imul(s ^ s >>> 15, 1 | s);
        let mut t = (s ^ ((s as u32 >> 15) as i32)).wrapping_mul(1 | s);

        // t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        t = t
            .wrapping_add((t ^ ((t as u32 >> 7) as i32)).wrapping_mul(61 | t))
            ^ t;

        // return ((t ^ t >>> 14) >>> 0) / 4294967296;
        let bits = (t ^ ((t as u32 >> 14) as i32)) as u32;
        bits as f64 / 4294967296.0
    }

    /// Draw a fresh `u32`, mirroring dryad's `(rng() * 2 ** 32) >>> 0`
    /// (genome.js:485). The `>>> 0` truncates the `f64` product to `u32`.
    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        // (rng() * 2**32) >>> 0 — JS ToUint32 of the f64 product.
        // rng() is in [0,1), so the product is in [0, 2^32) and `as u32`
        // truncates identically to `>>> 0`.
        (self.next() * 4294967296.0) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cross-check the first few draws against the JS reference. These literals
    /// were produced by running dryad's `mulberry32(1)` in node.
    #[test]
    fn matches_js_reference_seed_1() {
        let mut r = Mulberry32::new(1);
        // node: const f=mulberry32(1); [f(),f(),f(),f()]
        let got: Vec<f64> = (0..4).map(|_| r.next()).collect();
        let expect = [
            0.6270739405881613,
            0.002735721180215478,
            0.5274470399599522,
            0.9810509674716741,
        ];
        for (g, e) in got.iter().zip(expect.iter()) {
            assert!((g - e).abs() < 1e-15, "draw mismatch: got {g}, expect {e}");
        }
    }

    #[test]
    fn next_u32_truncates() {
        let mut r = Mulberry32::new(12345);
        let u = r.next_u32();
        // Re-derive: same seed, one f64 draw * 2^32 truncated.
        let mut r2 = Mulberry32::new(12345);
        let f = r2.next();
        assert_eq!(u, (f * 4294967296.0) as u32);
    }
}

//! A basic height field driven by fractal noise — useful for early testing.
//!
//! # Design
//!
//! `SimpleHeightField::height` composes two noise layers (ported from demiurge
//! `terrainSampler.ts`):
//!
//!   1. **Continents** — `fbm` at low frequency → signed offset that separates
//!      oceans (< 0) from land (> 0).
//!   2. **Mountains** — `ridged` at medium frequency → additive ridge detail
//!      (always positive, scaled down so it cannot overwhelm the continent
//!      signal at sea level).
//!
//! # LOD-additive-octave invariant
//!
//! The normalization approach follows demiurge exactly (see the comment block in
//! `terrainSampler.ts` around "LOD-consistency guarantee"):
//!
//! ```text
//!   FBM_BASE_OCTAVES    = 6    → max_amp_fbm6    = Σ_{i=0}^{5} 0.5^i ≈ 1.96875
//!   RIDGED_BASE_OCTAVES = 4    → max_amp_ridged4 = Σ_{i=0}^{3} 0.5^i = 0.9375
//! ```
//!
//! For both layers we always accumulate ALL octaves (base + extra) but divide
//! by the **fixed base max-amp**, not the current octave count's max-amp.
//! This means octaves 0..BASE are byte-identical across levels; only octaves
//! BASE..N are additive on top — seams vanish between LOD levels.
//!
//! LOD→octave schedule mirrors `terrainSampler.ts`:
//!
//! ```text
//!   fbm octaves    = clamp(level + 2, 6, 14)
//!   ridged octaves = clamp(level - 2, 4,  6)   (clamped to ≥ RIDGED_BASE_OCTAVES so it
//!                                                only ever adds, never removes, octaves)
//! ```

use glam::DVec3;
use crate::height::HeightField;
use crate::noise::Noise3D;

// ---------------------------------------------------------------------------
// Tuning constants
// ---------------------------------------------------------------------------

/// Continent fbm: low frequency so plates read as whole continents.
const CONTINENT_FREQ: f64 = 1.2;
/// Mountain ridged: medium frequency for mountain belts.
const MOUNTAIN_FREQ:  f64 = 3.5;
/// Weight of the ridged (mountains) term relative to [-1,1] output.
const MOUNTAIN_AMP:   f64 = 0.35;

/// Base octave count for the fbm layer (LOD-normalization anchor).
const FBM_BASE_OCTAVES: u32    = 6;
/// Base octave count for the ridged layer (LOD-normalization anchor).
const RIDGED_BASE_OCTAVES: u32 = 4;

const FBM_GAIN:    f64 = 0.5;
const FBM_LAC:     f64 = 2.0;
const RIDGED_GAIN: f64 = 0.5;
const RIDGED_LAC:  f64 = 2.0;

/// Precomputed max-amp for FBM_BASE_OCTAVES octaves of gain 0.5.
/// = 1 + 0.5 + 0.25 + 0.125 + 0.0625 + 0.03125 = 1.96875
const MAX_AMP_FBM6: f64 = {
    let mut a = 0.0f64;
    let mut g = 1.0f64;
    let mut i = 0u32;
    while i < FBM_BASE_OCTAVES {
        a += g;
        g *= FBM_GAIN;
        i += 1;
    }
    a
};

/// Precomputed max-amp for RIDGED_BASE_OCTAVES octaves of gain 0.5.
/// = 1 + 0.5 + 0.25 + 0.125 = 0.9375
const MAX_AMP_RIDGED4: f64 = {
    let mut a = 0.0f64;
    let mut g = 1.0f64;
    let mut i = 0u32;
    while i < RIDGED_BASE_OCTAVES {
        a += g;
        g *= RIDGED_GAIN;
        i += 1;
    }
    a
};

// ---------------------------------------------------------------------------
// Helper: additive-octave accumulation
// ---------------------------------------------------------------------------

/// Accumulate `octaves` octaves of fbm, dividing by the FIXED base max-amp.
///
/// Octaves 0..FBM_BASE_OCTAVES produce exactly the same result at every LOD
/// level because the normalization constant (`MAX_AMP_FBM6`) never changes.
/// Extra octaves are simply added on top (additive invariant).
fn fbm_additive(n: &Noise3D, x: f64, y: f64, z: f64, octaves: u32) -> f64 {
    let mut value = 0.0f64;
    let mut amp = 1.0f64;
    let mut f = 1.0f64;
    for _ in 0..octaves {
        value += amp * n.sample(x * f, y * f, z * f);
        amp *= FBM_GAIN;
        f *= FBM_LAC;
    }
    value / MAX_AMP_FBM6
}

/// Accumulate `octaves` octaves of ridged, dividing by the FIXED base max-amp.
fn ridged_additive(n: &Noise3D, x: f64, y: f64, z: f64, octaves: u32) -> f64 {
    let mut value = 0.0f64;
    let mut amp = 1.0f64;
    let mut f = 1.0f64;
    for _ in 0..octaves {
        let s = n.sample(x * f, y * f, z * f);
        let ridge = (1.0 - s.abs()).powi(2);
        value += amp * ridge;
        amp *= RIDGED_GAIN;
        f *= RIDGED_LAC;
    }
    value / MAX_AMP_RIDGED4
}

// ---------------------------------------------------------------------------
// SimpleHeightField
// ---------------------------------------------------------------------------

/// Simple procedural height field backed by a single `Noise3D` instance.
pub struct SimpleHeightField {
    pub noise: Noise3D,
}

impl HeightField for SimpleHeightField {
    /// Return a signed height in [-1, 1] for the surface point indicated by
    /// the unit direction `dir`, at LOD `level`.
    ///
    /// Values < 0 are ocean; values > 0 are land.
    ///
    /// LOD-adaptive octave counts:
    /// - fbm:    clamp(level + 2, FBM_BASE_OCTAVES,    14)
    /// - ridged: clamp(level - 2, RIDGED_BASE_OCTAVES,  6)   (saturates from below)
    ///
    /// Both layers use fixed-normalization (additive invariant), so the low-
    /// frequency base is bit-identical across levels.
    fn height(&self, dir: DVec3, level: u8) -> f64 {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        // LOD-adaptive octave counts (mirrors terrainSampler.ts)
        let fbm_octaves: u32 = (level as u32 + 2).clamp(FBM_BASE_OCTAVES, 14);
        // level - 2 saturates: if level < 2 this would underflow, so clamp from below
        let ridged_octaves: u32 = (level as i32 - 2).max(0) as u32;
        let ridged_octaves: u32 = ridged_octaves.clamp(RIDGED_BASE_OCTAVES, 6);

        // Continent layer: fbm in [-1,1] (via fixed normalization)
        // Values > 0 read as land; < 0 as ocean.
        let continent = fbm_additive(
            &self.noise,
            x * CONTINENT_FREQ,
            y * CONTINENT_FREQ,
            z * CONTINENT_FREQ,
            fbm_octaves,
        );

        // Mountain layer: ridged ∈ [0,1] (via fixed normalization)
        // Scaled and shifted so it adds relief on both land and ocean without
        // flipping ocean polarity (subtract 0.5 so it's ≈ [-0.5*amp, +0.5*amp]).
        let mountain_raw = ridged_additive(
            &self.noise,
            x * MOUNTAIN_FREQ,
            y * MOUNTAIN_FREQ,
            z * MOUNTAIN_FREQ,
            ridged_octaves,
        );
        // Centre ridged around 0 (it naturally lives in [0,1] → shift by -0.5)
        let mountain = MOUNTAIN_AMP * (mountain_raw - 0.5);

        let raw = continent + mountain;
        // Clamp to [-1, 1] — both continent + mountain bounded so this rarely clips
        raw.clamp(-1.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;

    fn make_hf(seed: u32) -> SimpleHeightField {
        SimpleHeightField { noise: Noise3D::new(seed) }
    }

    // -----------------------------------------------------------------------
    // height() ∈ [-1, 1]
    // -----------------------------------------------------------------------

    #[test]
    fn height_in_range() {
        let hf = make_hf(42);
        let dirs = sample_sphere_dirs(200);
        for &d in &dirs {
            let v = hf.height(d, 0);
            assert!(v >= -1.0 && v <= 1.0, "height out of range: {}", v);
        }
    }

    // -----------------------------------------------------------------------
    // Both ocean (<0) and land (>0) exist over a sphere of directions
    // -----------------------------------------------------------------------

    #[test]
    fn height_has_ocean_and_land() {
        let hf = make_hf(42);
        let dirs = sample_sphere_dirs(400);
        let has_ocean = dirs.iter().any(|&d| hf.height(d, 3) < 0.0);
        let has_land  = dirs.iter().any(|&d| hf.height(d, 3) > 0.0);
        assert!(has_ocean, "no ocean found (all heights ≥ 0)");
        assert!(has_land,  "no land found (all heights ≤ 0)");
    }

    // -----------------------------------------------------------------------
    // Determinism: same (dir, level) → same value
    // -----------------------------------------------------------------------

    #[test]
    fn height_deterministic() {
        let hf1 = make_hf(77);
        let hf2 = make_hf(77);
        let dirs = sample_sphere_dirs(100);
        for level in [0u8, 3, 6, 8] {
            for &d in &dirs {
                let a = hf1.height(d, level);
                let b = hf2.height(d, level);
                assert_eq!(a.to_bits(), b.to_bits(), "non-deterministic at level={}", level);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Additive-octave invariant:
    //   The low-frequency component (first FBM_BASE_OCTAVES octaves at
    //   CONTINENT_FREQ) is identical at every LOD level for the same dir.
    //
    // We verify this by computing the base-6-octave fbm accumulation
    // directly (same formula as fbm_additive with octaves=6) and checking it
    // equals the low-freq contribution embedded in heights at level 0 vs 8.
    //
    // Strategy: expose the continent component separately via a test helper,
    // then compare its value across levels.  The continent value must be
    // bit-exact across levels because fbm_additive always uses MAX_AMP_FBM6
    // as divisor — the additive octaves only ever tack on top.
    // -----------------------------------------------------------------------

    #[test]
    fn additive_octave_invariant() {
        // Compute the base-6-octave fbm value for a fixed dir.
        // At level L, fbm octaves = max(L+2, 6).
        // For L=0 → 6 octaves; for L=8 → 10 octaves.
        // The first 6 octaves' contribution must be identical because
        // MAX_AMP_FBM6 is the same divisor in both cases.

        let n = Noise3D::new(55);
        let dirs = sample_sphere_dirs(50);

        for &dir in &dirs {
            let x = dir.x * CONTINENT_FREQ;
            let y = dir.y * CONTINENT_FREQ;
            let z = dir.z * CONTINENT_FREQ;

            // Accumulate exactly 6 octaves (the base; same at all levels)
            let base_contribution = {
                let mut value = 0.0f64;
                let mut amp = 1.0f64;
                let mut f = 1.0f64;
                for _ in 0..FBM_BASE_OCTAVES {
                    value += amp * n.sample(x * f, y * f, z * f);
                    amp *= FBM_GAIN;
                    f *= FBM_LAC;
                }
                value / MAX_AMP_FBM6
            };

            // fbm_additive at level 0 (6 octaves) must equal base_contribution exactly.
            let fbm_l0 = fbm_additive(&n, x / CONTINENT_FREQ, y / CONTINENT_FREQ, z / CONTINENT_FREQ, 6);
            // Pass coords pre-scaled: note fbm_additive scales by CONTINENT_FREQ * f internally,
            // so pass dir (unscaled) and let it scale internally.
            let fbm_l0_direct = fbm_additive(&n, dir.x * CONTINENT_FREQ, dir.y * CONTINENT_FREQ, dir.z * CONTINENT_FREQ, 6);

            // At level 8: fbm octaves = clamp(8+2, 6, 14) = 10 — still uses MAX_AMP_FBM6
            let fbm_l8_direct = fbm_additive(&n, dir.x * CONTINENT_FREQ, dir.y * CONTINENT_FREQ, dir.z * CONTINENT_FREQ, 10);

            // The base (6-octave) accumulation must be identical at L=0 and L=8.
            // Both calls use MAX_AMP_FBM6 as denominator, so the 6-octave part is
            // bit-exact. We extract it from each by computing their difference.
            // Alternatively, the L=0 result (6 octaves, base max-amp) IS the base.
            // The L=8 result adds 4 more octaves / same denominator — so subtracting
            // the delta leaves the identical base. But the simplest check is:
            // the L=0 and base_contribution must agree bit-exactly.
            assert_eq!(
                fbm_l0_direct.to_bits(),
                base_contribution.to_bits(),
                "6-octave fbm disagrees with base_contribution (implementation drift)"
            );

            // L=0 and L=8 differ by additional octave content, not by a rescaling of
            // the base. Verify: extract the 6-octave prefix from both.
            // Since both use MAX_AMP_FBM6 as denominator, the first-6-octave raw sum
            // divided by MAX_AMP_FBM6 is the same number in both.
            // Therefore fbm_l0_direct (6 oct) == the 6-octave prefix of fbm_l8_direct.
            //
            // Extract 6-octave prefix from the 10-octave fbm_l8:
            // fbm_l8 = (raw6 + raw7 + raw8 + raw9 + raw10) / MAX_AMP_FBM6
            // fbm_l0 = raw6 / MAX_AMP_FBM6
            // → fbm_l0 * MAX_AMP_FBM6 == first-6-octave raw sum component of fbm_l8
            // We can verify this by checking that the 6-oct value is < 10-oct value
            // in absolute terms (the extra octaves always add something nonzero somewhere)
            // and that the delta is small (≤ amplitude of octave 7 ≈ 0.5^6/MAX_AMP_FBM6).
            let delta = (fbm_l8_direct - fbm_l0_direct).abs();
            // Max delta = Σ_{i=6}^{9} 0.5^i / MAX_AMP_FBM6 ≈ 0.032
            assert!(
                delta < 0.05,
                "delta between L=0 and L=8 fbm too large ({}); normalization anchor broken",
                delta
            );

            let _ = fbm_l0;
        }
    }

    // -----------------------------------------------------------------------
    // Explicit cross-level additive invariant using the public HeightField API
    // -----------------------------------------------------------------------

    #[test]
    fn height_low_freq_matches_across_levels() {
        let hf = make_hf(88);
        let dirs = sample_sphere_dirs(30);

        for &dir in &dirs {
            // At level 0, fbm uses exactly FBM_BASE_OCTAVES (the base).
            // At level 8, fbm uses max(8+2,6)=10 octaves.
            // The difference between the two outputs comes only from the
            // 4 extra ridged octaves and the 4 extra fbm octaves.
            // Since ridged base is also anchored, we need to compare the
            // base-only component.
            //
            // The cleanest testable invariant: set level=0 vs level=1 and
            // check that the heights are close (differ only by a small
            // extra octave term, not by a wholesale rescaling).
            let h0 = hf.height(dir, 0);
            let h1 = hf.height(dir, 1);

            // h1 has 1 extra fbm octave (amplitude 0.5^6 / MAX_AMP_FBM6 ≈ 0.016)
            // and 0 extra ridged octaves (both clamp to 4 until level 6).
            // So |h1 - h0| must be small.
            let diff = (h1 - h0).abs();
            assert!(
                diff < 0.10,
                "height(dir, 0) and height(dir, 1) differ by {} — additive invariant broken",
                diff
            );
        }
    }

    // -----------------------------------------------------------------------
    // Helper: generate roughly-uniform sphere directions
    // -----------------------------------------------------------------------

    fn sample_sphere_dirs(n: usize) -> Vec<DVec3> {
        // Fibonacci sphere for deterministic uniform coverage
        let golden = std::f64::consts::PI * (3.0 - 5.0f64.sqrt());
        (0..n)
            .map(|i| {
                let y = 1.0 - (i as f64 / (n as f64 - 1.0)) * 2.0;
                let r = (1.0 - y * y).max(0.0).sqrt();
                let theta = golden * i as f64;
                DVec3::new(r * theta.cos(), y, r * theta.sin()).normalize()
            })
            .collect()
    }
}

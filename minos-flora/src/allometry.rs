//! Allometric / proportion scaling — port of dryad `allometry.js`.
//!
//! Continuous scaling LAWS (not an archetype system). Turns the single
//! `trunkHeight` size slider into a coherent set of coupled scalars. Everything
//! is identity (1.0, or 0) at `trunkHeight = 0.5` so a default-stature plant is
//! byte-identical to the pre-allometry pipeline.

/// Named allometric exponents (mirror the JS module constants).
///
/// `GIRTH_EXP = 0` — height and width are DECOUPLED (girthScale ≡ 1).
pub const GIRTH_EXP: f64 = 0.0;
/// Leaf-cluster card size ∝ sizeFactor^0.5 (sub-linear).
pub const LEAF_AREA_EXP: f64 = 0.5;
/// Whorl tier count ∝ sizeFactor^0.7.
pub const TIER_EXP: f64 = 0.7;
/// Root depth/spread/flare ∝ sizeFactor^1.0 (linear).
pub const ROOT_EXP: f64 = 1.0;
/// Extra recursion levels per decade of size.
pub const BRANCH_ORDER_GAIN: f64 = 2.0;

/// Master vertical-stature multiplier. Piecewise so `f(0.5) == 1.0` exactly:
/// - `[0, 0.5]`: `0.45 + t*1.10`     → f(0)=0.45, f(0.5)=1.00
/// - `(0.5, 1]`: `10^(2*(t-0.5))`    → f(0.5)=1.00, f(1)=10.0
pub fn compute_size_factor(trunk_height: f64) -> f64 {
    if trunk_height <= 0.5 {
        0.45 + trunk_height * 1.10
    } else {
        // ponytail: 10f64.powf matches Math.pow within ~1 ULP cross-language —
        // accepted ceiling, do not reimplement.
        10f64.powf(2.0 * (trunk_height - 0.5))
    }
}

/// Realized coupled scalars consumed by the geometry stage (skeleton /
/// proportions / foliage / roots). All identity at sizeFactor = 1.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Traits {
    pub size_factor: f64,
    pub girth_scale: f64,
    pub leaf_area_scale: f64,
    pub tier_count_scale: f64,
    /// Additive branch-recursion-depth bonus (can be negative for small plants).
    pub branch_order_bonus: i32,
    pub root_scale: f64,
}

/// `deriveTraits(genome, env)` — `env` is accepted in JS for future laws but
/// unused, so it is omitted here. `trunk_height` defaults to 0.5 in JS via
/// `genome.trunkHeight ?? 0.5`; pass the genome's value (defaulting upstream).
pub fn derive_traits(trunk_height: f64) -> Traits {
    let size_factor = compute_size_factor(trunk_height);
    // JS: Math.round(log10(sizeFactor)*GAIN) || 0  — the `|| 0` normalizes -0.
    // Math.round is round-half-up toward +inf; for the values here (log10 of a
    // positive factor) Rust's round-half-away-from-zero only differs on exact
    // .5 ties, which don't arise at default stature. ponytail: accepted.
    let bonus = (size_factor.log10() * BRANCH_ORDER_GAIN).round();
    let branch_order_bonus = if bonus == 0.0 { 0 } else { bonus as i32 };
    Traits {
        size_factor,
        girth_scale: size_factor.powf(GIRTH_EXP),
        leaf_area_scale: size_factor.powf(LEAF_AREA_EXP),
        tier_count_scale: size_factor.powf(TIER_EXP),
        branch_order_bonus,
        root_scale: size_factor.powf(ROOT_EXP),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_at_half() {
        let t = derive_traits(0.5);
        assert_eq!(t.size_factor, 1.0);
        assert_eq!(t.girth_scale, 1.0);
        assert_eq!(t.leaf_area_scale, 1.0);
        assert_eq!(t.tier_count_scale, 1.0);
        assert_eq!(t.branch_order_bonus, 0);
        assert_eq!(t.root_scale, 1.0);
    }

    #[test]
    fn piecewise_endpoints() {
        assert!((compute_size_factor(0.0) - 0.45).abs() < 1e-12);
        assert!((compute_size_factor(1.0) - 10.0).abs() < 1e-9);
    }
}

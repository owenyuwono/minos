//! `FLORA_SCHEMA` gene ranges + `clamp_field`.
//!
//! Port of the parts of dryad's `genomeSchema.js` that `randomGenome`/`resolve`
//! need: every continuous gene's `[lo, hi]` range, the `structuralSeed` `seed`
//! kind, and `clampField`. `genomeDistance` is intentionally NOT ported (not on
//! the DATA spine's hot path).
//!
//! In JS the schema is a frozen object map keyed by gene name. Here it's a
//! `match` on `&str` returning a [`GeneKind`] — same semantics, no allocation.

/// Gene domain. Continuous genes clamp to `[lo, hi]`; the lone `Seed` gene
/// (`structuralSeed`) coerces to `u32`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GeneKind {
    Continuous { lo: f64, hi: f64 },
    Seed,
}

/// Look up a gene's kind by name. Returns `None` for unknown fields,
/// matching JS `schema[field] === undefined`.
///
/// Source of truth is dryad `genomeSchema.js` (`FLORA_SCHEMA`).
pub fn gene(name: &str) -> Option<GeneKind> {
    use GeneKind::Continuous as C;
    let c = |lo, hi| C { lo, hi };
    Some(match name {
        // --- STRUCTURAL ---
        "branchiness" => c(0.0, 1.0),
        "branchFactorN" => c(0.0, 1.0),
        "tillering" => c(0.0, 1.0),
        "radialOrder" => c(0.0, 1.0),
        "appendageBreadth" => c(0.0, 1.0),
        "appendageDensity" => c(0.0, 1.5),
        "segmentation" => c(0.0, 1.0),
        "trunkHeight" => c(0.0, 1.0),
        "stemSpread" => c(0.0, 1.0),
        "rosette" => c(0.0, 1.0),
        "whorl" => c(0.0, 1.0),
        "crownStart" => c(0.0, 1.0),
        // --- PROPORTIONS ---
        "succulence" => c(0.0, 1.0),
        "stemGirth" => c(0.0, 1.0),
        "taper" => c(0.0, 1.0),
        "rigidity" => c(0.0, 1.0),
        "verticality" => c(0.0, 1.0),
        "ribbing" => c(0.0, 1.0),
        "spininess" => c(0.0, 1.0),
        "branchAngle" => c(0.35, 0.80),
        "lengthRatio" => c(0.55, 0.85),
        "apicalBias" => c(0.0, 1.0),
        "droopBias" => c(-0.2, 0.4),
        "weep" => c(0.0, 1.0),
        "flatness" => c(0.0, 1.0),
        "woodiness" => c(0.0, 1.0),
        "tipTuft" => c(0.0, 1.0),
        "phototropism" => c(0.0, 1.0),
        "trunkTaper" => c(0.0, 1.0),
        // --- COSMETIC ---
        "pigment" => c(0.0, 1.0),
        "leafSize" => c(0.6, 4.0),
        "jitter" => c(0.0, 1.5),
        "leafWidth" => c(0.0, 1.0),
        "leafLength" => c(0.0, 1.0),
        "leafTip" => c(0.0, 1.0),
        "leafSerration" => c(0.0, 1.0),
        "leafLobing" => c(0.0, 1.0),
        "leafSkew" => c(0.0, 1.0),
        "barkHue" => c(0.0, 1.0),
        "barkLightness" => c(0.0, 1.0),
        "barkRelief" => c(0.0, 1.0),
        "barkLenticels" => c(0.0, 1.0),
        "barkScale" => c(0.0, 1.0),
        "barkOrient" => c(0.0, 1.0),
        "barkPlates" => c(0.0, 1.0),
        "barkShed" => c(0.0, 1.0),
        "barkUnderHue" => c(0.0, 1.0),
        "leafDivision" => c(0.0, 1.0),
        "frondFan" => c(0.0, 1.0),
        "structuralSeed" => GeneKind::Seed,
        // --- ROOT SYSTEM (structural) ---
        "rootCount" => c(0.0, 1.0),
        "rootDepth" => c(0.0, 1.0),
        "rootSpread" => c(0.0, 1.0),
        "rootFlare" => c(0.0, 1.0),
        "rootButtress" => c(0.0, 1.0),
        "rootBranchiness" => c(0.0, 1.0),
        "rootTaper" => c(0.0, 1.0),
        _ => return None,
    })
}

/// `[lo, hi]` for a continuous gene. Panics for `Seed`/unknown genes — callers
/// in the draw schedule only request continuous genes.
#[inline]
pub fn gene_range(name: &str) -> (f64, f64) {
    match gene(name) {
        Some(GeneKind::Continuous { lo, hi }) => (lo, hi),
        _ => panic!("gene_range: '{name}' is not a continuous gene"),
    }
}

/// Coerce `value` into a gene's valid domain — port of `clampField`.
///
/// - continuous → clamp to `[lo, hi]`.
/// - seed → reinterpret as `u32` (a no-op in Rust since we already hold a `u32`).
/// - unknown field → return unchanged (JS `gene === undefined` branch).
#[inline]
pub fn clamp_field(name: &str, value: f64) -> f64 {
    match gene(name) {
        Some(GeneKind::Continuous { lo, hi }) => {
            // JS: value < lo ? lo : value > hi ? hi : value
            if value < lo {
                lo
            } else if value > hi {
                hi
            } else {
                value
            }
        }
        // Seed genes don't flow through the f64 clamp path; unknown → unchanged.
        _ => value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clamp_continuous() {
        assert_eq!(clamp_field("branchAngle", 0.1), 0.35);
        assert_eq!(clamp_field("branchAngle", 0.9), 0.80);
        assert_eq!(clamp_field("branchAngle", 0.5), 0.5);
        assert_eq!(clamp_field("droopBias", -0.5), -0.2);
    }

    #[test]
    fn unknown_field_passthrough() {
        assert_eq!(clamp_field("nope", 42.0), 42.0);
    }

    #[test]
    fn seed_is_not_continuous() {
        assert_eq!(gene("structuralSeed").unwrap(), GeneKind::Seed);
    }
}

//! `FLORA_SCHEMA` gene ranges + `clamp_field` + tier arrays.
//!
//! Port of the parts of dryad's `genomeSchema.js` that `randomGenome`/`resolve`
//! need: every continuous gene's `[lo, hi]` range, the `structuralSeed` `seed`
//! kind, `clampField`, and the tier-partitioned name arrays. `genomeDistance`
//! is intentionally NOT ported (not on the DATA spine's hot path).
//!
//! In JS the schema is a frozen object map keyed by gene name. Here it's a
//! `match` on `&str` returning a [`GeneKind`] — same semantics, no allocation.

/// Mutation/divergence tier. Mirrors `tier: 'structural'|'proportions'|'cosmetic'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Structural,
    Proportions,
    Cosmetic,
}

/// Gene domain. Continuous genes clamp to `[lo, hi]`; the lone `Seed` gene
/// (`structuralSeed`) coerces to `u32`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GeneKind {
    Continuous { lo: f64, hi: f64 },
    Seed,
}

/// Look up a gene's tier + kind by name. Returns `None` for unknown fields,
/// matching JS `schema[field] === undefined`.
///
/// Source of truth is dryad `genomeSchema.js` (`FLORA_SCHEMA`).
pub fn gene(name: &str) -> Option<(Tier, GeneKind)> {
    use GeneKind::Continuous as C;
    use Tier::*;
    let c = |lo, hi| C { lo, hi };
    Some(match name {
        // --- STRUCTURAL ---
        "branchiness" => (Structural, c(0.0, 1.0)),
        "branchFactorN" => (Structural, c(0.0, 1.0)),
        "tillering" => (Structural, c(0.0, 1.0)),
        "radialOrder" => (Structural, c(0.0, 1.0)),
        "appendageBreadth" => (Structural, c(0.0, 1.0)),
        "appendageDensity" => (Structural, c(0.0, 1.5)),
        "segmentation" => (Structural, c(0.0, 1.0)),
        "trunkHeight" => (Structural, c(0.0, 1.0)),
        "stemSpread" => (Structural, c(0.0, 1.0)),
        "rosette" => (Structural, c(0.0, 1.0)),
        "whorl" => (Structural, c(0.0, 1.0)),
        "crownStart" => (Structural, c(0.0, 1.0)),
        // --- PROPORTIONS ---
        "succulence" => (Proportions, c(0.0, 1.0)),
        "stemGirth" => (Proportions, c(0.0, 1.0)),
        "taper" => (Proportions, c(0.0, 1.0)),
        "rigidity" => (Proportions, c(0.0, 1.0)),
        "verticality" => (Proportions, c(0.0, 1.0)),
        "ribbing" => (Proportions, c(0.0, 1.0)),
        "spininess" => (Proportions, c(0.0, 1.0)),
        "branchAngle" => (Proportions, c(0.35, 0.80)),
        "lengthRatio" => (Proportions, c(0.55, 0.85)),
        "apicalBias" => (Proportions, c(0.0, 1.0)),
        "droopBias" => (Proportions, c(-0.2, 0.4)),
        "weep" => (Proportions, c(0.0, 1.0)),
        "flatness" => (Proportions, c(0.0, 1.0)),
        "woodiness" => (Proportions, c(0.0, 1.0)),
        "tipTuft" => (Proportions, c(0.0, 1.0)),
        "phototropism" => (Proportions, c(0.0, 1.0)),
        "trunkTaper" => (Proportions, c(0.0, 1.0)),
        // --- COSMETIC ---
        "pigment" => (Cosmetic, c(0.0, 1.0)),
        "leafSize" => (Cosmetic, c(0.6, 4.0)),
        "jitter" => (Cosmetic, c(0.0, 1.5)),
        "leafWidth" => (Cosmetic, c(0.0, 1.0)),
        "leafLength" => (Cosmetic, c(0.0, 1.0)),
        "leafTip" => (Cosmetic, c(0.0, 1.0)),
        "leafSerration" => (Cosmetic, c(0.0, 1.0)),
        "leafLobing" => (Cosmetic, c(0.0, 1.0)),
        "leafSkew" => (Cosmetic, c(0.0, 1.0)),
        "barkHue" => (Cosmetic, c(0.0, 1.0)),
        "barkLightness" => (Cosmetic, c(0.0, 1.0)),
        "barkRelief" => (Cosmetic, c(0.0, 1.0)),
        "barkLenticels" => (Cosmetic, c(0.0, 1.0)),
        "barkScale" => (Cosmetic, c(0.0, 1.0)),
        "barkOrient" => (Cosmetic, c(0.0, 1.0)),
        "barkPlates" => (Cosmetic, c(0.0, 1.0)),
        "barkShed" => (Cosmetic, c(0.0, 1.0)),
        "barkUnderHue" => (Cosmetic, c(0.0, 1.0)),
        "leafDivision" => (Cosmetic, c(0.0, 1.0)),
        "frondFan" => (Cosmetic, c(0.0, 1.0)),
        "structuralSeed" => (Cosmetic, GeneKind::Seed),
        // --- ROOT SYSTEM (structural) ---
        "rootCount" => (Structural, c(0.0, 1.0)),
        "rootDepth" => (Structural, c(0.0, 1.0)),
        "rootSpread" => (Structural, c(0.0, 1.0)),
        "rootFlare" => (Structural, c(0.0, 1.0)),
        "rootButtress" => (Structural, c(0.0, 1.0)),
        "rootBranchiness" => (Structural, c(0.0, 1.0)),
        "rootTaper" => (Structural, c(0.0, 1.0)),
        _ => return None,
    })
}

/// `[lo, hi]` for a continuous gene. Panics for `Seed`/unknown genes — callers
/// in the draw schedule only request continuous genes.
#[inline]
pub fn gene_range(name: &str) -> (f64, f64) {
    match gene(name) {
        Some((_, GeneKind::Continuous { lo, hi })) => (lo, hi),
        _ => panic!("gene_range: '{name}' is not a continuous gene"),
    }
}

/// Coerce `value` into a gene's valid domain — port of `clampField`.
///
/// - continuous → clamp to `[lo, hi]`.
/// - seed → reinterpret as `u32` (here a no-op since we already hold a `u32`;
///   exposed via [`clamp_field_seed`]).
/// - unknown field → return unchanged (JS `gene === undefined` branch).
#[inline]
pub fn clamp_field(name: &str, value: f64) -> f64 {
    match gene(name) {
        Some((_, GeneKind::Continuous { lo, hi })) => {
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

/// Seed coercion arm of `clampField` (`value >>> 0`). Trivial in Rust because
/// the value is already a `u32`; provided for parity/clarity.
#[inline]
pub fn clamp_field_seed(value: u32) -> u32 {
    value
}

/// All gene names with `tier === 'structural'`, in schema declaration order.
pub const STRUCTURAL_FIELDS: &[&str] = &[
    "branchiness",
    "branchFactorN",
    "tillering",
    "radialOrder",
    "appendageBreadth",
    "appendageDensity",
    "segmentation",
    "trunkHeight",
    "stemSpread",
    "rosette",
    "whorl",
    "crownStart",
    "rootCount",
    "rootDepth",
    "rootSpread",
    "rootFlare",
    "rootButtress",
    "rootBranchiness",
    "rootTaper",
];

/// All gene names with `tier === 'proportions'`, in schema declaration order.
pub const PROPORTIONS_FIELDS: &[&str] = &[
    "succulence",
    "stemGirth",
    "taper",
    "rigidity",
    "verticality",
    "ribbing",
    "spininess",
    "branchAngle",
    "lengthRatio",
    "apicalBias",
    "droopBias",
    "weep",
    "flatness",
    "woodiness",
    "tipTuft",
    "phototropism",
    "trunkTaper",
];

/// All gene names with `tier === 'cosmetic'`, in schema declaration order.
/// Includes `structuralSeed` (the lone `Seed` gene lives in this tier in dryad).
pub const COSMETIC_FIELDS: &[&str] = &[
    "pigment",
    "leafSize",
    "jitter",
    "leafWidth",
    "leafLength",
    "leafTip",
    "leafSerration",
    "leafLobing",
    "leafSkew",
    "barkHue",
    "barkLightness",
    "barkRelief",
    "barkLenticels",
    "barkScale",
    "barkOrient",
    "barkPlates",
    "barkShed",
    "barkUnderHue",
    "leafDivision",
    "frondFan",
    "structuralSeed",
];

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
        assert_eq!(gene("structuralSeed").unwrap().1, GeneKind::Seed);
    }
}

// =============================================================================
// golden.rs — GOLDEN-VECTOR determinism gate.
//
// Loads tests/golden.json (produced by tools/dump_golden.mjs from the REAL dryad
// generator, genome.js randomGenome+resolve) and asserts the Rust port matches:
//
//   * rng / genome gene values — EXACT (|Δ| < 1e-12, incl. structuralSeed exact).
//   * graph node positions     — relative 1e-6 (cross-language trig ULP ceiling).
//   * graph node radii         — absolute 1e-9.
//   * bone indices             — EXACT.
//   * foliage count            — EXACT.
//   * sampled foliage positions — relative 1e-6.
//
// ponytail: the 1e-6 geometry epsilon is the cross-language libm trig (sin/cos/
// atan2) ULP ceiling — an accepted divergence, NOT a bug. The rng and the genes
// are bit-exact (1e-12). If the Rust output diverges beyond these tolerances the
// test FAILS LOUDLY (no masking, no skips) and names the case + field + values.
//
// REGEN the golden file:  node enki-flora/tools/dump_golden.mjs
// =============================================================================

use enki_flora::genome::{random_genome, resolve, Env, Genome, Medium};
use serde_json::Value;

// --- Tolerances ---
const GENE_EPS: f64 = 1e-12; // rng/genes are bit-exact
const POS_REL_EPS: f64 = 1e-6; // node/leaf positions (trig ULP ceiling)
const RADIUS_ABS_EPS: f64 = 1e-9; // radii (no trig — tighter)

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

fn f(v: &Value) -> f64 {
    v.as_f64()
        .unwrap_or_else(|| panic!("expected number, got {v:?}"))
}

/// Relative-or-absolute closeness: |a-b| <= eps * max(1, |a|, |b|). Robust near
/// zero (falls back to absolute eps) and scales with magnitude otherwise.
fn close_rel(a: f64, b: f64, eps: f64) -> bool {
    let scale = 1.0_f64.max(a.abs()).max(b.abs());
    (a - b).abs() <= eps * scale
}

/// Reconstruct the Rust `Env` from the JSON case env so the Rust pipeline is fed
/// the byte-identical environment dryad was given (see dump_golden.mjs header on
/// why the envs are fully specified rather than `{}`).
fn env_from_json(e: &Value) -> Env {
    let medium = match e["medium"].as_str().unwrap_or("air") {
        "water" => Medium::Water,
        _ => Medium::Air,
    };
    let get = |k: &str, d: f64| e.get(k).and_then(|v| v.as_f64()).unwrap_or(d);
    Env {
        gravity: get("gravity", 1.0),
        medium,
        // energy/biochem are not part of the dumped env and not read by the
        // genome layer; keep the neutral defaults.
        energy: 0.5,
        biochem: 0.5,
        temperature: get("temperature", 0.5),
        light: get("light", 0.6),
        sun_angle: get("sunAngle", 0.25),
        wind: get("wind", 0.2),
        aridity: get("aridity", 0.35),
    }
}

/// Map every JS gene name to its value on the Rust `Genome`. This is the column
/// the gene assertions check against the golden `genes` object. structuralSeed
/// is handled separately (it is a u32, asserted exact).
fn gene_value(g: &Genome, js_name: &str) -> f64 {
    match js_name {
        "branchiness" => g.branchiness,
        "branchFactorN" => g.branch_factor_n,
        "tillering" => g.tillering,
        "radialOrder" => g.radial_order,
        "appendageBreadth" => g.appendage_breadth,
        "appendageDensity" => g.appendage_density,
        "segmentation" => g.segmentation,
        "succulence" => g.succulence,
        "stemGirth" => g.stem_girth,
        "taper" => g.taper,
        "rigidity" => g.rigidity,
        "verticality" => g.verticality,
        "ribbing" => g.ribbing,
        "spininess" => g.spininess,
        "branchAngle" => g.branch_angle,
        "lengthRatio" => g.length_ratio,
        "apicalBias" => g.apical_bias,
        "droopBias" => g.droop_bias,
        "pigment" => g.pigment,
        "leafSize" => g.leaf_size,
        "jitter" => g.jitter,
        "leafWidth" => g.leaf_width,
        "leafLength" => g.leaf_length,
        "leafTip" => g.leaf_tip,
        "leafSerration" => g.leaf_serration,
        "leafLobing" => g.leaf_lobing,
        "leafSkew" => g.leaf_skew,
        "barkHue" => g.bark_hue,
        "barkLightness" => g.bark_lightness,
        "barkRelief" => g.bark_relief,
        "barkLenticels" => g.bark_lenticels,
        "barkScale" => g.bark_scale,
        "barkOrient" => g.bark_orient,
        "barkPlates" => g.bark_plates,
        "barkShed" => g.bark_shed,
        "barkUnderHue" => g.bark_under_hue,
        "weep" => g.weep,
        "trunkHeight" => g.trunk_height,
        "flatness" => g.flatness,
        "stemSpread" => g.stem_spread,
        "rosette" => g.rosette,
        "woodiness" => g.woodiness,
        "whorl" => g.whorl,
        "crownStart" => g.crown_start,
        "tipTuft" => g.tip_tuft,
        "leafDivision" => g.leaf_division,
        "frondFan" => g.frond_fan,
        "phototropism" => g.phototropism,
        "trunkTaper" => g.trunk_taper,
        "rootCount" => g.root_count,
        "rootDepth" => g.root_depth,
        "rootSpread" => g.root_spread,
        "rootFlare" => g.root_flare,
        "rootButtress" => g.root_buttress,
        "rootBranchiness" => g.root_branchiness,
        "rootTaper" => g.root_taper,
        other => panic!("gene_value: unknown JS gene name '{other}'"),
    }
}

fn load_golden() -> Value {
    // CARGO_MANIFEST_DIR = enki-flora/. golden.json sits in tests/.
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden.json");
    let raw = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("cannot read {path}: {e}. Regenerate: `node enki-flora/tools/dump_golden.mjs`")
    });
    serde_json::from_str(&raw).expect("golden.json is not valid JSON")
}

// -----------------------------------------------------------------------------
// The gate.
// -----------------------------------------------------------------------------

#[test]
fn golden_vector_matches_dryad() {
    let golden = load_golden();
    let cases = golden["cases"].as_array().expect("golden.cases array");
    assert!(!cases.is_empty(), "golden.json has no cases");

    for case in cases {
        let seed = case["seed"].as_u64().expect("case.seed") as u32;
        let env = env_from_json(&case["env"]);
        let tag = format!("seed={seed} gravity={} aridity={}", env.gravity, env.aridity);

        // ---- Run the Rust port ----
        let genome = random_genome(&env, seed);

        // ---- (1) GENES — bit-exact (1e-12) ----
        let genes = case["genes"].as_object().expect("case.genes object");
        for (name, want) in genes {
            if name == "structuralSeed" {
                let got = genome.structural_seed as u64;
                let exp = want.as_u64().expect("structuralSeed must be a uint");
                assert_eq!(
                    got, exp,
                    "[{tag}] structuralSeed mismatch: rust={got} dryad={exp}"
                );
                continue;
            }
            let exp = f(want);
            let got = gene_value(&genome, name);
            assert!(
                (got - exp).abs() < GENE_EPS,
                "[{tag}] gene {name}: rust={got:.17} dryad={exp:.17} (|Δ|={:.3e} ≥ {GENE_EPS:.0e})",
                (got - exp).abs()
            );
        }

        // ---- Resolve for geometry ----
        let r = resolve(&genome, &env);
        let g = &case["graph"];

        // ---- (2) GRAPH node/bone counts — exact ----
        let exp_nodes = g["nodeCount"].as_u64().unwrap() as usize;
        let exp_bones = g["boneCount"].as_u64().unwrap() as usize;
        assert_eq!(
            r.graph.nodes.len(),
            exp_nodes,
            "[{tag}] node count: rust={} dryad={exp_nodes}",
            r.graph.nodes.len()
        );
        assert_eq!(
            r.graph.bones.len(),
            exp_bones,
            "[{tag}] bone count: rust={} dryad={exp_bones}",
            r.graph.bones.len()
        );
        assert_eq!(
            r.bone_count, exp_bones,
            "[{tag}] resolve.bone_count: rust={} dryad={exp_bones}",
            r.bone_count
        );

        // ---- (3) lightDir — within position epsilon ----
        let ld = g["lightDir"].as_array().unwrap();
        for k in 0..3 {
            let exp = f(&ld[k]);
            let got = r.light_dir[k];
            assert!(
                close_rel(got, exp, POS_REL_EPS),
                "[{tag}] lightDir[{k}]: rust={got} dryad={exp}"
            );
        }

        // ---- (4) node positions (rel 1e-6) + radii (abs 1e-9) ----
        let jnodes = g["nodes"].as_array().unwrap();
        for (i, jn) in jnodes.iter().enumerate() {
            let rn = &r.graph.nodes[i];
            let jp = jn["pos"].as_array().unwrap();
            for k in 0..3 {
                let exp = f(&jp[k]);
                let got = rn.pos[k];
                assert!(
                    close_rel(got, exp, POS_REL_EPS),
                    "[{tag}] node[{i}].pos[{k}]: rust={got} dryad={exp} (|Δ|={:.3e})",
                    (got - exp).abs()
                );
            }
            let exp_r = f(&jn["radius"]);
            let got_r = rn.radius;
            assert!(
                (got_r - exp_r).abs() <= RADIUS_ABS_EPS,
                "[{tag}] node[{i}].radius: rust={got_r} dryad={exp_r} (|Δ|={:.3e} > {RADIUS_ABS_EPS:.0e})",
                (got_r - exp_r).abs()
            );
        }

        // ---- (5) bone indices — exact ----
        let jbones = g["bones"].as_array().unwrap();
        for (i, jb) in jbones.iter().enumerate() {
            let a = jb["a"].as_u64().unwrap() as usize;
            let b = jb["b"].as_u64().unwrap() as usize;
            assert_eq!(
                (r.graph.bones[i].a, r.graph.bones[i].b),
                (a, b),
                "[{tag}] bone[{i}]: rust=({},{}) dryad=({a},{b})",
                r.graph.bones[i].a,
                r.graph.bones[i].b
            );
        }

        // ---- (6) foliage count exact + sampled positions (rel 1e-6) ----
        let fol = &case["foliage"];
        let exp_count = fol["count"].as_u64().unwrap() as usize;
        assert_eq!(
            r.foliage.count, exp_count,
            "[{tag}] foliage.count: rust={} dryad={exp_count}",
            r.foliage.count
        );

        let cap = fol["cap"].as_u64().unwrap() as usize;
        // The Rust SoA stores exactly `count` instances; the golden file captured
        // the first `cap` (≤ count) of them. Compare position/scale/rotation over
        // [0, cap). JS stored these as Float32Array → f32; Rust stores f32 too, so
        // values are already f32-rounded on both sides — compare as f64.
        let jpos = fol["position"].as_array().unwrap();
        let jscale = fol["scale"].as_array().unwrap();
        let jrot = fol["rotation"].as_array().unwrap();
        assert_eq!(jpos.len(), cap * 3, "[{tag}] golden foliage position len");
        assert!(r.foliage.position.len() >= cap * 3, "[{tag}] rust foliage too short");

        for i in 0..cap {
            for k in 0..3 {
                let exp = f(&jpos[i * 3 + k]);
                let got = r.foliage.position[i * 3 + k] as f64;
                assert!(
                    close_rel(got, exp, POS_REL_EPS),
                    "[{tag}] foliage[{i}].position[{k}]: rust={got} dryad={exp} (|Δ|={:.3e})",
                    (got - exp).abs()
                );
            }
            let exp_s = f(&jscale[i]);
            let got_s = r.foliage.scale[i] as f64;
            assert!(
                close_rel(got_s, exp_s, POS_REL_EPS),
                "[{tag}] foliage[{i}].scale: rust={got_s} dryad={exp_s}"
            );
            let exp_rot = f(&jrot[i]);
            let got_rot = r.foliage.rotation[i] as f64;
            assert!(
                close_rel(got_rot, exp_rot, POS_REL_EPS),
                "[{tag}] foliage[{i}].rotation: rust={got_rot} dryad={exp_rot}"
            );
        }

        // shape == appendageBreadth (f32 in JS); compare with a loose f32 epsilon.
        let exp_shape = f(&fol["shape"]);
        let got_shape = r.foliage.shape as f64;
        assert!(
            close_rel(got_shape, exp_shape, 1e-6),
            "[{tag}] foliage.shape: rust={got_shape} dryad={exp_shape}"
        );
    }
}

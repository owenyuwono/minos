//! Genome draw schedule + `resolve` orchestration — port of dryad `genome.js`.
//!
//! DETERMINISM IS LOAD-BEARING. The whole generator is pure + seeded
//! (`mulberry32`). [`random_genome`] reproduces dryad's EXACT rng draw COUNT and
//! ORDER — including the two RESERVED no-op draws and the raw `pigment` draw —
//! ported from the `genome.js` body (NOT the stale header-comment draw numbers).
//! Any reordering reshuffles every downstream value.

use crate::foliage::{generate_foliage, FoliageSoA};
use crate::genome_schema::{clamp_field, gene_range};
use crate::proportions::solve_proportions;
use crate::rng::Mulberry32;
use crate::roots::{grow_root_system, ROOT_SALT};
use crate::skeleton::{build_skeleton, Graph};

// ---------------------------------------------------------------------------
// Env — environment envelope. Defaults are dryad's neutral env.
// ---------------------------------------------------------------------------

/// Growth medium. JS uses the string `'air'` | `'water'`; only `'water'` is
/// branched on, so an enum is the faithful (and cheaper) representation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Medium {
    Air,
    Water,
}

/// Environment envelope passed to [`random_genome`] / [`resolve`].
///
/// `gravity, medium, light, wind, aridity, temperature` drive
/// [`compute_env_offset`]. `energy, biochem, sun_angle` are carried for parity
/// with the broader env contract but are NOT read by the genome layer
/// (`sunAngle` is documented as part of the neutral env yet unused in dryad's
/// `computeEnvOffset`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Env {
    pub gravity: f64,
    pub medium: Medium,
    pub energy: f64,
    pub biochem: f64,
    pub temperature: f64,
    pub light: f64,
    pub sun_angle: f64,
    pub wind: f64,
    pub aridity: f64,
}

impl Default for Env {
    /// dryad's neutral env: gravity=1, medium='air', light=0.6, sunAngle=0.25,
    /// wind=0.2, aridity=0.35, temperature=0.5. energy/biochem are neutral 0.5.
    fn default() -> Self {
        Self {
            gravity: 1.0,
            medium: Medium::Air,
            energy: 0.5,
            biochem: 0.5,
            temperature: 0.5,
            light: 0.6,
            sun_angle: 0.25,
            wind: 0.2,
            aridity: 0.35,
        }
    }
}

// ---------------------------------------------------------------------------
// Genome — one numeric gene per FLORA_SCHEMA field + the uint32 structuralSeed.
// ---------------------------------------------------------------------------

/// A flora genome: every continuous gene as `f64`, plus the `u32`
/// `structural_seed`. Field set mirrors `randomGenome`'s return object exactly
/// (leafDensity was folded into appendageDensity and is intentionally absent).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Genome {
    // Structural
    pub branchiness: f64,
    pub branch_factor_n: f64,
    pub tillering: f64,
    pub radial_order: f64,
    pub appendage_breadth: f64,
    pub appendage_density: f64,
    pub segmentation: f64,
    // Proportions
    pub succulence: f64,
    pub stem_girth: f64,
    pub taper: f64,
    pub rigidity: f64,
    pub verticality: f64,
    pub ribbing: f64,
    pub spininess: f64,
    pub branch_angle: f64,
    pub length_ratio: f64,
    pub apical_bias: f64,
    pub droop_bias: f64,
    // Cosmetic
    pub pigment: f64,
    pub leaf_size: f64,
    pub jitter: f64,
    pub leaf_width: f64,
    pub leaf_length: f64,
    pub leaf_tip: f64,
    pub leaf_serration: f64,
    pub leaf_lobing: f64,
    pub leaf_skew: f64,
    pub bark_hue: f64,
    pub bark_lightness: f64,
    pub bark_relief: f64,
    pub bark_lenticels: f64,
    pub bark_scale: f64,
    pub bark_orient: f64,
    pub bark_plates: f64,
    pub bark_shed: f64,
    pub bark_under_hue: f64,
    pub weep: f64,
    pub trunk_height: f64,
    pub flatness: f64,
    pub stem_spread: f64,
    pub rosette: f64,
    pub woodiness: f64,
    pub whorl: f64,
    pub crown_start: f64,
    pub tip_tuft: f64,
    pub leaf_division: f64,
    pub frond_fan: f64,
    pub phototropism: f64,
    pub trunk_taper: f64,
    pub structural_seed: u32,
    // Root system
    pub root_count: f64,
    pub root_depth: f64,
    pub root_spread: f64,
    pub root_flare: f64,
    pub root_buttress: f64,
    pub root_branchiness: f64,
    pub root_taper: f64,
}

impl Genome {
    /// Schema-driven editable-gene accessor: one entry per CONTINUOUS gene as
    /// `(js_name, lo, hi, &mut f64)`, borrowing the live field. The tuple's
    /// `(lo, hi)` come from the schema (`gene_range`), so a UI can build sliders
    /// by iterating this Vec instead of hand-writing 55 widgets.
    ///
    /// `structural_seed` (the lone `Seed` gene) is intentionally excluded — it is
    /// a `u32`, not a continuous `f64`, and `gene_range` panics on it. Edit it
    /// directly via the field. Every `&mut self.$field` targets a distinct field,
    /// so the `Vec` of disjoint mutable borrows is sound.
    pub fn fields_mut(&mut self) -> Vec<(&'static str, f64, f64, &mut f64)> {
        let mut v: Vec<(&'static str, f64, f64, &mut f64)> = Vec::new();
        macro_rules! f {
            ($js:literal, $field:ident) => {{
                let (lo, hi) = gene_range($js);
                v.push(($js, lo, hi, &mut self.$field));
            }};
        }
        // Structural
        f!("branchiness", branchiness);
        f!("branchFactorN", branch_factor_n);
        f!("tillering", tillering);
        f!("radialOrder", radial_order);
        f!("appendageBreadth", appendage_breadth);
        f!("appendageDensity", appendage_density);
        f!("segmentation", segmentation);
        f!("trunkHeight", trunk_height);
        f!("stemSpread", stem_spread);
        f!("rosette", rosette);
        f!("whorl", whorl);
        f!("crownStart", crown_start);
        f!("rootCount", root_count);
        f!("rootDepth", root_depth);
        f!("rootSpread", root_spread);
        f!("rootFlare", root_flare);
        f!("rootButtress", root_buttress);
        f!("rootBranchiness", root_branchiness);
        f!("rootTaper", root_taper);
        // Proportions
        f!("succulence", succulence);
        f!("stemGirth", stem_girth);
        f!("taper", taper);
        f!("rigidity", rigidity);
        f!("verticality", verticality);
        f!("ribbing", ribbing);
        f!("spininess", spininess);
        f!("branchAngle", branch_angle);
        f!("lengthRatio", length_ratio);
        f!("apicalBias", apical_bias);
        f!("droopBias", droop_bias);
        f!("weep", weep);
        f!("flatness", flatness);
        f!("woodiness", woodiness);
        f!("tipTuft", tip_tuft);
        f!("phototropism", phototropism);
        f!("trunkTaper", trunk_taper);
        // Cosmetic
        f!("pigment", pigment);
        f!("leafSize", leaf_size);
        f!("jitter", jitter);
        f!("leafWidth", leaf_width);
        f!("leafLength", leaf_length);
        f!("leafTip", leaf_tip);
        f!("leafSerration", leaf_serration);
        f!("leafLobing", leaf_lobing);
        f!("leafSkew", leaf_skew);
        f!("barkHue", bark_hue);
        f!("barkLightness", bark_lightness);
        f!("barkRelief", bark_relief);
        f!("barkLenticels", bark_lenticels);
        f!("barkScale", bark_scale);
        f!("barkOrient", bark_orient);
        f!("barkPlates", bark_plates);
        f!("barkShed", bark_shed);
        f!("barkUnderHue", bark_under_hue);
        f!("leafDivision", leaf_division);
        f!("frondFan", frond_fan);
        v
    }
}

// ---------------------------------------------------------------------------
// NEUTRAL — the "sensible average plant" gene vector (genome.js:179-258).
// Keyed by the same JS gene names used in the draw schedule. structuralSeed has
// no neutral (it is derived from a draw), so it is absent here.
// ---------------------------------------------------------------------------

/// Look up a gene's NEUTRAL default by JS gene name. Panics on unknown name —
/// every name used in the draw schedule is present.
fn neutral(name: &str) -> f64 {
    match name {
        // Structural
        "branchiness" => 0.50,
        "branchFactorN" => 0.33,
        "tillering" => 0.10,
        "radialOrder" => 0.25,
        "appendageBreadth" => 0.45,
        "appendageDensity" => 0.40,
        "segmentation" => 0.20,
        // Proportions
        "succulence" => 0.20,
        "stemGirth" => 0.40,
        "taper" => 0.50,
        "rigidity" => 0.60,
        "verticality" => 0.50,
        "ribbing" => 0.05,
        "spininess" => 0.05,
        "branchAngle" => 0.575,
        "lengthRatio" => 0.70,
        "apicalBias" => 0.50,
        "droopBias" => 0.00,
        // Cosmetic
        "pigment" => 0.45,
        "leafSize" => 1.10,
        "jitter" => 1.00,
        "leafWidth" => 0.50,
        "leafLength" => 0.45,
        "leafTip" => 0.40,
        "leafSerration" => 0.00,
        "leafLobing" => 0.00,
        "leafSkew" => 0.50,
        "barkHue" => 0.75,
        "barkLightness" => 0.30,
        "barkRelief" => 0.85,
        "barkLenticels" => 0.00,
        "barkScale" => 0.50,
        "barkOrient" => 0.70,
        "barkPlates" => 0.45,
        "barkShed" => 0.00,
        "barkUnderHue" => 0.75,
        "weep" => 0.00,
        "trunkHeight" => 0.50,
        "flatness" => 0.00,
        "stemSpread" => 0.00,
        "rosette" => 0.00,
        "woodiness" => 1.00,
        "whorl" => 0.00,
        "crownStart" => 1.00,
        "tipTuft" => 0.00,
        "leafDivision" => 0.00,
        "frondFan" => 0.00,
        "phototropism" => 1.00,
        "trunkTaper" => 0.00,
        // Root system
        "rootCount" => 0.45,
        "rootDepth" => 0.45,
        "rootSpread" => 0.50,
        "rootFlare" => 0.30,
        "rootButtress" => 0.15,
        "rootBranchiness" => 0.45,
        "rootTaper" => 0.50,
        _ => panic!("neutral: unknown gene '{name}'"),
    }
}

// ---------------------------------------------------------------------------
// EnvOffset — env-driven additive gene biases (computeEnvOffset, genome.js:279).
// ---------------------------------------------------------------------------

#[inline]
fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// Per-gene additive offset, keyed by JS gene name. Genes absent from the map
/// have offset 0 (matching the JS initializer where every gene starts at 0).
/// Only the genes that ever receive a nonzero offset are tracked; lookups for
/// any other gene return 0.0.
pub struct EnvOffsetTable {
    map: std::collections::HashMap<&'static str, f64>,
}

impl EnvOffsetTable {
    /// Additive offset for `name` (0.0 if the gene is unaffected by env).
    #[inline]
    pub fn get(&self, name: &str) -> f64 {
        self.map.get(name).copied().unwrap_or(0.0)
    }
}

/// Compute env-driven gene offsets — bit-faithful port of `computeEnvOffset`.
/// All offsets are 0 at the neutral env. Accumulation order matches the JS
/// (left-to-right `+=`/`-=`) for f64 reproducibility.
pub fn compute_env_offset(env: &Env) -> EnvOffsetTable {
    let gravity = env.gravity;
    let medium = env.medium;
    let light = env.light;
    let wind = env.wind;
    let aridity = env.aridity;
    let temperature = env.temperature;

    let mut o = EnvOffsetTable {
        map: std::collections::HashMap::new(),
    };
    let mut add = |name: &'static str, delta: f64| {
        *o.map.entry(name).or_insert(0.0) += delta;
    };

    // weep is the one nonzero initializer (genome.js:335).
    if medium == Medium::Water {
        add("weep", 0.05);
    }

    // --- aridity + temperature → dry/hot = cactus region ---
    let arid_frac = clamp((aridity - 0.35) / 0.65, -1.0, 1.0);
    let temp_frac = clamp((temperature - 0.5) / 0.5, -1.0, 1.0);
    let arid_heat = clamp(arid_frac * 0.5 + temp_frac * 0.5, -1.0, 1.0);

    add("succulence", 0.30 * arid_heat);
    add("spininess", 0.20 * arid_heat);
    add("ribbing", 0.15 * arid_heat);
    add("appendageBreadth", -0.20 * arid_heat);

    // --- medium === 'water' → kelp/holdfast ---
    if medium == Medium::Water {
        add("appendageBreadth", 0.25);
        add("rigidity", -0.25);
        add("verticality", 0.15);
        add("tillering", 0.20);
    }

    // --- gravity → heavy-planet squat/thick ---
    let grav_delta = clamp((gravity - 1.0) / 2.0, -0.5, 0.5);
    add("stemGirth", 0.20 * grav_delta);
    add("succulence", 0.10 * grav_delta);
    add("branchiness", -0.20 * grav_delta);

    // --- wind → low/streamlined ---
    let wind_delta = clamp((wind - 0.2) / 0.8, 0.0, 1.0);
    add("verticality", -0.25 * wind_delta);
    add("rigidity", 0.25 * wind_delta);
    add("appendageBreadth", -0.20 * wind_delta);

    // --- light → dim = more branching/broader for capture ---
    let light_delta = clamp((0.6 - light) / 0.6, -1.0, 1.0);
    add("branchiness", 0.20 * light_delta);
    add("appendageBreadth", 0.20 * light_delta);
    add("appendageDensity", 0.15 * light_delta);

    // --- root system env biases ---
    add("rootDepth", 0.30 * arid_heat);
    add("rootTaper", 0.10 * arid_heat);
    add("rootFlare", 0.30 * wind_delta);
    add("rootButtress", 0.20 * wind_delta);

    if medium == Medium::Water || aridity < 0.15 {
        add("rootSpread", 0.20);
        add("rootDepth", -0.20);
    }

    o
}

// ---------------------------------------------------------------------------
// randomGenome(env, seed) -> Genome
// ---------------------------------------------------------------------------

/// Standard seed spread for most genes (±12% of range).
const STD: f64 = 0.12;

/// `seedVar(draw, spread, span) = (draw - 0.5) * 2 * spread * span`.
/// Evaluate in this exact term order for f64 reproducibility.
#[inline]
fn seed_var(draw: f64, spread: f64, span: f64) -> f64 {
    (draw - 0.5) * 2.0 * spread * span
}

/// Construct a deterministic genome from `(env, seed)`.
///
/// The rng draws are consumed in the EXACT order of `genome.js`'s body
/// (draws 1..59 below, with raw `pigment`, the seed draw, and two RESERVED
/// no-op draws). Repeating with the same `(env, seed)` yields bit-identical
/// output.
pub fn random_genome(env: &Env, seed: u32) -> Genome {
    let mut rng = Mulberry32::new(seed);
    let offset = compute_env_offset(env);

    // One continuous gene = clampField(NEUTRAL + offset + seedVar(rng(), ...)).
    // Closure captures rng + offset; called in draw order.
    let gene = |rng: &mut Mulberry32, name: &str, spread: f64| -> f64 {
        let (lo, hi) = gene_range(name);
        let span = hi - lo;
        let raw = neutral(name) + offset.get(name) + seed_var(rng.next(), spread, span);
        clamp_field(name, raw)
    };

    // --- Structural (draws 01–07) ---
    let branchiness = gene(&mut rng, "branchiness", STD);
    let branch_factor_n = gene(&mut rng, "branchFactorN", STD);
    let tillering = gene(&mut rng, "tillering", STD);
    let radial_order = gene(&mut rng, "radialOrder", STD);
    let appendage_breadth = gene(&mut rng, "appendageBreadth", STD);
    let appendage_density = gene(&mut rng, "appendageDensity", STD);
    let segmentation = gene(&mut rng, "segmentation", STD);

    // --- Proportions (draws 08–18) ---
    let succulence = gene(&mut rng, "succulence", STD);
    let stem_girth = gene(&mut rng, "stemGirth", STD);
    let taper = gene(&mut rng, "taper", STD);
    let rigidity = gene(&mut rng, "rigidity", STD);
    let verticality = gene(&mut rng, "verticality", STD);
    let ribbing = gene(&mut rng, "ribbing", STD);
    let spininess = gene(&mut rng, "spininess", STD);
    let branch_angle = gene(&mut rng, "branchAngle", STD);
    let length_ratio = gene(&mut rng, "lengthRatio", STD);
    let apical_bias = gene(&mut rng, "apicalBias", STD);
    let droop_bias = gene(&mut rng, "droopBias", STD);

    // --- Cosmetic (draws 19–21 + seed draw 22) ---
    // draw 19: pigment is the ONLY raw draw — no neutral, no offset, no seedVar.
    let pigment = clamp_field("pigment", rng.next());
    let leaf_size = gene(&mut rng, "leafSize", 0.20); // draw 20 (spread 0.20, not STD)
    let jitter = gene(&mut rng, "jitter", 0.20); // draw 21 (spread 0.20, not STD)
    // draw 22: structuralSeed — (rng()*2^32)>>>0, bypasses clampField.
    let structural_seed = rng.next_u32();

    // --- Root system (draws 23–29) — appended AFTER structuralSeed ---
    let root_count = gene(&mut rng, "rootCount", STD);
    let root_depth = gene(&mut rng, "rootDepth", STD);
    let root_spread = gene(&mut rng, "rootSpread", STD);
    let root_flare = gene(&mut rng, "rootFlare", STD);
    let root_buttress = gene(&mut rng, "rootButtress", STD);
    let root_branchiness = gene(&mut rng, "rootBranchiness", STD);
    let root_taper = gene(&mut rng, "rootTaper", STD);

    // --- Leaf-texture cosmetics (draws 30–34) ---
    let leaf_width = gene(&mut rng, "leafWidth", STD);
    let leaf_length = gene(&mut rng, "leafLength", STD);
    let leaf_tip = gene(&mut rng, "leafTip", STD);
    let leaf_serration = gene(&mut rng, "leafSerration", STD);
    let leaf_lobing = gene(&mut rng, "leafLobing", STD);

    // --- weep (draw 35) ---
    let weep = gene(&mut rng, "weep", STD);

    // --- trunkHeight (draw 36) ---
    let trunk_height = gene(&mut rng, "trunkHeight", STD);

    // --- inert genes (draws 37–39) ---
    let flatness = gene(&mut rng, "flatness", STD);
    let stem_spread = gene(&mut rng, "stemSpread", STD);
    let rosette = gene(&mut rng, "rosette", STD);

    // --- inert genes (draws 40–42) ---
    let woodiness = gene(&mut rng, "woodiness", STD);
    let whorl = gene(&mut rng, "whorl", STD);
    let tip_tuft = gene(&mut rng, "tipTuft", STD);

    // --- leafDivision (draw 43) + RESERVED no-op (draw 44, ex-frondLeaf) ---
    let leaf_division = gene(&mut rng, "leafDivision", STD);
    rng.next(); // draw 44 RESERVED — consume + discard (frondLeaf merged in)

    // --- phototropism (draw 45) ---
    let phototropism = gene(&mut rng, "phototropism", STD);

    // --- trunkTaper (draw 46) ---
    let trunk_taper = gene(&mut rng, "trunkTaper", STD);

    // --- RESERVED no-op (draw 47, ex-trunkRings) ---
    rng.next(); // draw 47 RESERVED — consume + discard (trunkRings folded into barkOrient)

    // --- Bark axes (draws 48–52) ---
    let bark_hue = gene(&mut rng, "barkHue", STD);
    let bark_lightness = gene(&mut rng, "barkLightness", STD);
    let bark_relief = gene(&mut rng, "barkRelief", STD);
    let bark_lenticels = gene(&mut rng, "barkLenticels", STD);
    let bark_scale = gene(&mut rng, "barkScale", STD);

    // --- frondFan / leafSkew / crownStart (draws 53–55) ---
    let frond_fan = gene(&mut rng, "frondFan", STD);
    let leaf_skew = gene(&mut rng, "leafSkew", STD);
    let crown_start = gene(&mut rng, "crownStart", STD);

    // --- barkOrient / barkPlates / barkShed / barkUnderHue (draws 56–59) ---
    let bark_orient = gene(&mut rng, "barkOrient", STD);
    let bark_plates = gene(&mut rng, "barkPlates", STD);
    let bark_shed = gene(&mut rng, "barkShed", STD);
    let bark_under_hue = gene(&mut rng, "barkUnderHue", STD);

    Genome {
        branchiness,
        branch_factor_n,
        tillering,
        radial_order,
        appendage_breadth,
        appendage_density,
        segmentation,
        succulence,
        stem_girth,
        taper,
        rigidity,
        verticality,
        ribbing,
        spininess,
        branch_angle,
        length_ratio,
        apical_bias,
        droop_bias,
        pigment,
        leaf_size,
        jitter,
        leaf_width,
        leaf_length,
        leaf_tip,
        leaf_serration,
        leaf_lobing,
        leaf_skew,
        bark_hue,
        bark_lightness,
        bark_relief,
        bark_lenticels,
        bark_scale,
        bark_orient,
        bark_plates,
        bark_shed,
        bark_under_hue,
        weep,
        trunk_height,
        flatness,
        stem_spread,
        rosette,
        woodiness,
        whorl,
        crown_start,
        tip_tuft,
        leaf_division,
        frond_fan,
        phototropism,
        trunk_taper,
        structural_seed,
        root_count,
        root_depth,
        root_spread,
        root_flare,
        root_buttress,
        root_branchiness,
        root_taper,
    }
}

// ---------------------------------------------------------------------------
// resolve(genome, env) -> render data  [STUB — filled by the geometry stage]
// ---------------------------------------------------------------------------

/// Render-data output of [`resolve`] — the downstream contract. Holds the solved
/// skeleton `graph`, the generated `foliage` SoA, and the cosmetic/derived
/// scalars `resolve` exposes (pigment, leaf*/bark*, ribbing, flatness, ribCount,
/// lightFlux, woodiness).
///
/// ponytail: skin() SDF payload dropped — the mesh path ignores it; re-add the
/// 64-bone arrays (`boneAData`/`boneBData`/`blendK`/`materialMode`/uRibbing/…)
/// here if a consumer needs the legacy skin material.
///
/// No `PartialEq`: the embedded [`Graph`] / [`FoliageSoA`] carry float geometry
/// that is not meaningfully equatable; scalar parity is asserted field-wise.
#[derive(Clone, Debug)]
pub struct Resolved {
    /// The solved branch + root skeleton.
    pub graph: Graph,
    /// Generated cluster-card foliage (leaves + optional spines).
    pub foliage: FoliageSoA,
    /// `graph.bones.len()` (overrides any skin boneCount in dryad).
    pub bone_count: usize,
    /// `graph.meta.light_dir`.
    pub light_dir: [f64; 3],
    pub pigment: f64,
    pub leaf_size: f64,
    pub leaf_width: f64,
    pub leaf_length: f64,
    pub leaf_tip: f64,
    pub leaf_serration: f64,
    pub leaf_lobing: f64,
    pub leaf_skew: f64,
    pub bark_hue: f64,
    pub bark_lightness: f64,
    pub bark_relief: f64,
    pub bark_lenticels: f64,
    pub bark_scale: f64,
    pub bark_orient: f64,
    pub bark_plates: f64,
    pub bark_shed: f64,
    pub bark_under_hue: f64,
    pub leaf_division: f64,
    pub frond_fan: f64,
    pub ribbing: f64,
    pub flatness: f64,
    /// `max(8, min(16, 8 + round(segmentation * 8)))`.
    pub rib_count: u32,
    pub light_flux: f64,
    pub woodiness: f64,
    /// Willow droop gene (genome.weep, [0,1]). Drives the per-leaf gravity-droop
    /// scale + the canopy-normal lighting blend (dryad uWeep). 0 = no extra droop.
    pub weep: f64,
}

/// Pure render-data pipeline from a genome + env — port of dryad `resolve`.
///
/// Orchestration (genome.js:687–753), with the SAME stream construction + draw
/// cadence as dryad:
/// 1. `rng = mulberry32(structuralSeed)` — the skeleton stream.
/// 2. `buildSkeleton(genome, rng, genome.jitter)` — consumes the skeleton stream.
/// 3. `rootRng = mulberry32((structuralSeed ^ ROOT_SALT) >>> 0)` — ISOLATED.
/// 4. `growRootSystem(graph, genome, rootRng, { env })`.
/// 5. `solveProportions(graph, env, genome)` — in-place, no rng.
/// 6. `generateFoliage(graph, genome)` — owns its OWN streams (`^0x1EAF1EAF`,
///    and the gated `^SPINE_SALT`).
///
/// Stream isolation via distinct salts is the determinism mechanism: each is a
/// fresh independent `mulberry32`, so roots/foliage never perturb the skeleton
/// draw sequence. The legacy `skin()` SDF pass is dropped (see [`Resolved`]).
pub fn resolve(genome: &Genome, env: &Env) -> Resolved {
    // (1) skeleton stream + (2) build skeleton (jitter_amp = genome.jitter).
    let mut rng = Mulberry32::new(genome.structural_seed);
    let mut graph = build_skeleton(genome, &mut rng, genome.jitter);

    // (3) isolated root stream + (4) grow roots in place.
    let mut root_rng = Mulberry32::new(genome.structural_seed ^ ROOT_SALT);
    grow_root_system(&mut graph, genome, &mut root_rng, Some(env), None);

    // (5) proportions solve in place (writes meta.light_dir from sunAngle).
    solve_proportions(&mut graph, env, Some(genome));

    // (6) foliage — owns its leaf + (gated) spine streams internally.
    let foliage = generate_foliage(&graph, genome);

    let bone_count = graph.bones.len();
    let light_dir = graph.meta.light_dir;

    // ribCount = max(8, min(16, 8 + round(segmentation * 8))).
    // JS Math.round is round-half-up; segmentation ≥ 0 so the half-away-from-zero
    // difference cannot bite here. ponytail: documented, not reimplemented.
    let rib = 8.0 + (genome.segmentation * 8.0).round();
    let rib_count = rib.clamp(8.0, 16.0) as u32;

    Resolved {
        graph,
        foliage,
        bone_count,
        light_dir,
        pigment: genome.pigment,
        leaf_size: genome.leaf_size,
        leaf_width: genome.leaf_width,
        leaf_length: genome.leaf_length,
        leaf_tip: genome.leaf_tip,
        leaf_serration: genome.leaf_serration,
        leaf_lobing: genome.leaf_lobing,
        leaf_skew: genome.leaf_skew,
        bark_hue: genome.bark_hue,
        bark_lightness: genome.bark_lightness,
        bark_relief: genome.bark_relief,
        bark_lenticels: genome.bark_lenticels,
        bark_scale: genome.bark_scale,
        bark_orient: genome.bark_orient,
        bark_plates: genome.bark_plates,
        bark_shed: genome.bark_shed,
        bark_under_hue: genome.bark_under_hue,
        leaf_division: genome.leaf_division,
        frond_fan: genome.frond_fan,
        ribbing: genome.ribbing,
        flatness: genome.flatness,
        rib_count,
        light_flux: env.light,
        woodiness: genome.woodiness,
        weep: genome.weep,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_seed() {
        let env = Env::default();
        let a = random_genome(&env, 42);
        let b = random_genome(&env, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn neutral_env_zero_offset() {
        // At the neutral env every offset is 0 (so only seedVar applies).
        let off = compute_env_offset(&Env::default());
        for name in [
            "succulence",
            "spininess",
            "branchiness",
            "appendageBreadth",
            "rootDepth",
            "weep",
        ] {
            assert_eq!(off.get(name), 0.0, "offset for {name} should be 0 at neutral env");
        }
    }

    /// Cross-check the full draw schedule against dryad's `randomGenome({}, 42)`
    /// (neutral env). These literals come from running the JS reference in node;
    /// matching them proves the draw COUNT + ORDER (incl. raw pigment, the seed
    /// draw, and both RESERVED no-op draws) is bit-faithful.
    #[test]
    fn matches_js_random_genome_seed_42() {
        let g = random_genome(&Env::default(), 42);
        let eq = |got: f64, exp: f64, name: &str| {
            assert!(
                (got - exp).abs() < 1e-12,
                "{name}: got {got}, expect {exp}"
            );
        };
        eq(g.branchiness, 0.5242649004608393, "branchiness");
        eq(g.appendage_density, 0.4095733151864261, "appendageDensity");
        eq(g.succulence, 0.2299387169443071, "succulence");
        eq(g.branch_angle, 0.5423034144984558, "branchAngle");
        eq(g.droop_bias, 0.01592940937355161, "droopBias");
        eq(g.pigment, 0.003842951962724328, "pigment");
        eq(g.leaf_size, 1.0602634162828326, "leafSize");
        eq(g.jitter, 1.2024024555459618, "jitter");
        assert_eq!(g.structural_seed, 219942124, "structuralSeed");
        eq(g.root_count, 0.4721577577292919, "rootCount");
        eq(g.root_taper, 0.507280545514077, "rootTaper");
        eq(g.leaf_width, 0.3865096662379801, "leafWidth");
        eq(g.weep, 0.0, "weep");
        eq(g.trunk_height, 0.48797497486695646, "trunkHeight");
        eq(g.leaf_division, 0.0, "leafDivision");
        eq(g.phototropism, 1.0, "phototropism");
        eq(g.trunk_taper, 0.08609062062576413, "trunkTaper");
        eq(g.bark_hue, 0.6790056235156954, "barkHue");
        eq(g.frond_fan, 0.04338454373180866, "frondFan");
        eq(g.leaf_skew, 0.5463411479815841, "leafSkew");
        eq(g.crown_start, 1.0, "crownStart");
        eq(g.bark_orient, 0.6011123261414468, "barkOrient");
        eq(g.bark_under_hue, 0.8561506921984255, "barkUnderHue");
    }

    #[test]
    fn resolve_rib_count_neutral() {
        let env = Env::default();
        let g = random_genome(&env, 7);
        let r = resolve(&g, &env);
        assert!(r.rib_count >= 8 && r.rib_count <= 16);
        assert_eq!(r.light_flux, env.light);
    }
}

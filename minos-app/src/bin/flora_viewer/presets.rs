// flora_viewer presets — const Genomes ported from dryad/src/presets.js.
//
// ponytail: dryad spreads each preset from TREE_DEFAULT (`{...TREE_DEFAULT,
// ...overrides}`). Rust's functional struct update (`Genome { field: x,
// ..TREE_DEFAULT }`) is the exact same semantics, so each preset is one literal
// over the base. JS camelCase → Rust snake_case; `0xNN >>> 0` → `0xNNu32`.
// These are viewer-local debug data, so they live in the bin, not minos-flora.

use minos_flora::genome::Genome;

// The base every preset spreads from (dryad `TREE_DEFAULT`) now lives in the lib
// (`flora_view`) so the in-planet app draws the exact same default specimen.
// Re-exported here unchanged, so the presets below + the panel keep using it.
pub use minos_app::flora_view::TREE_DEFAULT;

/// Broad dome (dryad OAK).
const OAK: Genome = Genome {
    branchiness: 0.85,
    branch_factor_n: 0.62,
    branch_angle: 0.72,
    apical_bias: 0.55,
    stem_girth: 0.80,
    taper: 0.65,
    rigidity: 0.60,
    length_ratio: 0.65,
    leaf_lobing: 0.45,
    leaf_serration: 0.10,
    leaf_width: 0.60,
    leaf_length: 0.50,
    leaf_tip: 0.55,
    leaf_size: 1.10,
    appendage_density: 1.035,
    pigment: 0.30,
    trunk_height: 0.5147,
    bark_hue: 0.90,
    bark_lightness: 0.29,
    bark_relief: 0.90,
    bark_lenticels: 0.14,
    root_count: 0.60,
    root_spread: 0.70,
    root_flare: 0.45,
    root_buttress: 0.20,
    structural_seed: 0x4C2F7A11,
    ..TREE_DEFAULT
};

/// Whorled conifer spire (dryad PINE).
const PINE: Genome = Genome {
    branchiness: 0.55,
    branch_factor_n: 0.48,
    branch_angle: 0.50,
    apical_bias: 0.95,
    rigidity: 0.78,
    verticality: 0.70,
    stem_girth: 0.30,
    taper: 0.75,
    length_ratio: 0.72,
    appendage_breadth: 0.20,
    appendage_density: 1.425,
    leaf_width: 0.04,
    leaf_length: 0.82,
    leaf_tip: 0.12,
    leaf_size: 1.40,
    pigment: 0.26,
    bark_hue: 0.80,
    bark_lightness: 0.36,
    bark_relief: 0.92,
    bark_lenticels: 0.11,
    trunk_height: 0.72,
    root_depth: 0.60,
    root_spread: 0.55,
    root_flare: 0.25,
    whorl: 0.65,
    tip_tuft: 0.55,
    structural_seed: 0x3A1CF890,
    ..TREE_DEFAULT
};

/// Cascade (dryad WEEPING WILLOW).
const WILLOW: Genome = Genome {
    branchiness: 0.68,
    branch_factor_n: 0.58,
    verticality: 0.65,
    rigidity: 0.25,
    apical_bias: 0.50,
    branch_angle: 0.55,
    length_ratio: 0.80,
    taper: 0.80,
    stem_girth: 0.40,
    leaf_width: 0.18,
    leaf_length: 0.95,
    leaf_tip: 0.80,
    leaf_serration: 0.20,
    leaf_size: 1.20,
    appendage_density: 1.215,
    pigment: 0.34,
    root_count: 0.55,
    root_spread: 0.65,
    weep: 0.55,
    trunk_height: 0.5414,
    bark_hue: 0.80,
    bark_lightness: 0.36,
    bark_relief: 0.65,
    bark_lenticels: 0.49,
    structural_seed: 0xE5D2A0F7,
    ..TREE_DEFAULT
};

/// Slender columnar, white bark (dryad BIRCH).
const BIRCH: Genome = Genome {
    branchiness: 0.55,
    branch_factor_n: 0.50,
    verticality: 0.70,
    stem_girth: 0.30,
    taper: 0.80,
    rigidity: 0.35,
    branch_angle: 0.55,
    apical_bias: 0.65,
    appendage_density: 1.305,
    leaf_size: 0.85,
    leaf_serration: 0.55,
    leaf_width: 0.70,
    leaf_length: 0.20,
    leaf_skew: 0.30,
    leaf_tip: 0.30,
    pigment: 0.30,
    bark_hue: 0.05,
    bark_lightness: 0.87,
    bark_relief: 0.10,
    bark_lenticels: 1.00,
    trunk_height: 0.5147,
    crown_start: 0.45,
    structural_seed: 0x7F3A1E05,
    ..TREE_DEFAULT
};

/// All presets, in panel order.
pub const ALL: &[(&str, &Genome)] = &[
    ("Default", &TREE_DEFAULT),
    ("Oak", &OAK),
    ("Pine", &PINE),
    ("Willow", &WILLOW),
    ("Birch", &BIRCH),
];

//! `enki-flora` — self-contained, plug-and-play deterministic procedural plant
//! generation (a Rust port of the `dryad` flora pipeline) for enki.
//!
//! # Modularity contract
//! ALL flora logic, types, and data live in this crate. The lower crates
//! (`enki-rhi`, `enki-render`, `enki-planet`) never reference it; `enki-app` is
//! the only consumer, behind a `flora` Cargo feature. Removing flora is:
//! delete this folder, drop the workspace member, drop the optional dep +
//! feature, delete the `#[cfg(feature = "flora")]` app wiring.
//!
//! # Modules (mirroring the dryad source layout)
//! - [`rng`]           — deterministic `mulberry32` PRNG (bit-exact port).
//! - [`genome`]        — `randomGenome` draw schedule + `resolve` orchestration.
//! - [`genome_schema`] — `FLORA_SCHEMA` gene ranges + `clampField`.
//! - [`allometry`]     — proportion / allometric solving.
//! - [`color`]         — pigment / bark color derivation.
//! - [`skeleton`]      — branch skeleton construction.
//! - [`proportions`]   — in-place proportion solve.
//! - [`roots`]         — root system growth.
//! - [`foliage`]       — leaf / frond / spine generation.

pub mod rng;
pub mod genome;
pub mod genome_schema;
pub mod allometry;
pub mod color;
pub mod skeleton;
pub mod proportions;
pub mod roots;
pub mod foliage;
// ponytail: the deletable-crate contract is satisfied at the crate boundary —
// `enki-flora` itself is the unit `enki-app` gates behind `#[cfg(feature="flora")]`.
// There is no in-crate `flora` feature to gate against, so `mesh` is a plain
// module like its siblings; deleting flora still means deleting this folder.
pub mod mesh;
/// Hierarchical skeletal wind solver (Rust port of dryad `windSolver.js`).
pub mod wind_solver;
/// CPU leaf-cluster texture rasterizer (Rust port of dryad `leafTexture.js`
/// `makeLeafClusterTexture` cluster branch + `buildLeafNormalMap`).
pub mod leaf_texture;

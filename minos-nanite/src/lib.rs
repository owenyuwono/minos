//! `minos-nanite` — self-contained, plug-and-play Nanite-style virtualized
//! geometry for minos terrain.
//!
//! # Modularity contract
//! ALL Nanite logic, types, and shaders live in this crate. The lower crates
//! (`minos-rhi`, `minos-render`, `minos-planet`) never reference it; `minos-app` is
//! the only consumer, behind a `nanite` Cargo feature. Removing Nanite is:
//! delete this folder, drop the workspace member, drop the optional dep +
//! feature, delete the `#[cfg(feature = "nanite")]` app wiring. See
//! `docs/nanite-plan.md`.
//!
//! # Modules
//! - [`cluster`] — data-only cluster-DAG types (no GPU, no `ash`).
//! - [`bake`]    — offline DAG builder: tessellate a heightfield patch and build
//!   the cluster DAG. CPU-only, headless-testable. The runtime GPU pass is added
//!   in a later milestone (N1).

pub mod bake;
pub mod cluster;
pub mod render;
pub mod residency;
pub mod stream;

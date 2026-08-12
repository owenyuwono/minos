//! `minos-render` — high-level rendering layer on top of `minos-rhi`.
//!
//! Provides render passes, material definitions, camera, projection, and
//! geometry contracts. Depends on `minos-rhi` for all GPU operations.
//! Does NOT depend on `minos-app`.

pub mod frame;
pub mod camera;
pub mod projection;
pub mod material;
pub mod lights;
pub mod sky;
pub mod system;
pub mod terrain_pass;
pub mod water_pass;
pub mod body_pass;
pub mod geometry;

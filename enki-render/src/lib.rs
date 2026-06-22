//! `enki-render` — high-level rendering layer on top of `enki-rhi`.
//!
//! Provides render passes, material definitions, camera, projection, and
//! geometry contracts. Depends on `enki-rhi` for all GPU operations.
//! Does NOT depend on `enki-app`.

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
pub mod tonemap;
pub mod geometry;
pub mod taa;

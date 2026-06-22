// enki-app library surface.
//
// enki-app is primarily the `enki` binary (src/main.rs). This thin lib exists
// only to SHARE the flora GPU glue with the standalone `flora_viewer` binary
// (src/bin/flora_viewer.rs) without duplicating flora_view.rs.
//
// Deletable-flora contract holds: with the `flora` feature off, this lib is
// empty and references nothing flora-related.

// Flora-OWNED Vulkan sub-renderer (egui-ash-renderer precedent): owns its OWN
// pipelines + descriptor set layouts (set0 UBO + reserved set1) + UBO ring, the
// enabler for shadows/IBL/bloom/wind. flora_view + staging record through it.
#[cfg(feature = "flora")]
pub mod flora_render;

// CPU prefilter for flora's lazy IBL: decode dryad's equirect HDRI, project SH9
// diffuse irradiance + build a roughness mip chain. flora_render uploads it.
#[cfg(feature = "flora")]
pub mod flora_ibl;

#[cfg(feature = "flora")]
pub mod flora_view;

// Flora branch mesh → enki-nanite virtualized-geometry bake. Composes the two
// otherwise-independent features: present ONLY when BOTH are on. Default
// (nanite, no flora) and flora-only (no nanite) builds skip it cleanly.
#[cfg(all(feature = "flora", feature = "nanite"))]
pub mod flora_nanite;

// SCENE staging (sky/ground/contact-shadow) for the standalone flora viewer.
// Lives behind `flora` alongside flora_view; deleting flora drops both.
#[cfg(feature = "flora")]
pub mod staging;

// Smoothed/throttled FPS meter, shared with the `flora_viewer` binary so its
// Stats readout refreshes ~2 Hz instead of flickering every frame (same as the
// main `enki` app). Generic (no flora dep).
pub mod fps;

//! Water render pass — shader source and pipeline description.
//!
//! # Pipeline contract
//! The integration engineer must create one alpha-blend pipeline with:
//! - Topology: triangle-list
//! - Polygon mode: FILL
//! - Cull mode: BACK, CCW front face
//! - Depth state: D32_SFLOAT, write **OFF**, compare GREATER (reversed-Z)
//! - Blend: straight alpha over opaque geometry
//!   - `src_color_blend_factor = SRC_ALPHA`
//!   - `dst_color_blend_factor = ONE_MINUS_SRC_ALPHA`
//!   - `color_blend_op = ADD`
//!   - `src_alpha_blend_factor = ONE`
//!   - `dst_alpha_blend_factor = ONE_MINUS_SRC_ALPHA`
//!   - `alpha_blend_op = ADD`
//! - Descriptor set 0: `FrameUniforms` at binding 0 (`UNIFORM_BUFFER`)
//!   (same descriptor set layout as terrain — reuse the same set in the same frame)
//! - Push constants: `ChunkPush` (80 bytes) at offset 0, `VERTEX | FRAGMENT`
//! - Vertex inputs: same four float-3 buffer layout as terrain_pass
//!
//! # Draw order
//! Draw water **after** all opaque terrain draws so alpha blending composites
//! correctly over the finished opaque scene.
//!
//! # Call contract per draw
//! ```ignore
//! cmd.bind_pipeline(water_pipeline);
//! cmd.bind_descriptor_sets(0, &[frame_descriptor_set]); // same set as terrain
//! cmd.push_constants(ChunkPush::new(model_matrix, 0));
//! cmd.bind_vertex_buffers(&[positions, normals, colors, plate_colors]);
//! cmd.bind_index_buffer(indices, u32);
//! cmd.draw_indexed(index_count, 1, 0, 0, 0);
//! ```

/// WGSL source for the water vertex + fragment shader.
///
/// Entry points: `vs_main` (vertex), `fs_main` (fragment).
/// Group/binding: `@group(0) @binding(0)` = `FrameUniforms` uniform.
/// Push constant: `var<immediate> pc: ChunkPush` (naga 29 WGSL syntax for Vulkan push constants).
/// Alpha: fixed 0.72, straight (not premultiplied) — pipeline blend state handles compositing.
pub const WATER_WGSL: &str = include_str!("shaders/water.wgsl");

/// WGSL for the FFT wave surface (Tessendorf spectral ocean) — `vs_main`/`fs_main`.
///
/// Custom set 0: binding 0 `Frame` (uniform, mirrors `FrameUniforms`), binding 1
/// `field` (storage `array<OceanTexel>` — the CPU sim output), binding 2 `Ocean`
/// (uniform — tangent frame + tiling + colors). Bound via `cmd_bind_descriptor_set`
/// (NOT the rhi set0 ring). Same blend/depth contract as `WATER_WGSL`.
pub const OCEAN_SURFACE_WGSL: &str = include_str!("shaders/ocean_surface.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_wgsl_parses_with_naga() {
        let module = naga::front::wgsl::parse_str(WATER_WGSL);
        assert!(
            module.is_ok(),
            "water.wgsl failed to parse: {:?}",
            module.err()
        );
    }

    #[test]
    fn ocean_surface_wgsl_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(OCEAN_SURFACE_WGSL)
            .unwrap_or_else(|e| panic!("ocean_surface.wgsl failed to parse: {e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("ocean_surface.wgsl failed to validate: {e:?}"));
    }
}

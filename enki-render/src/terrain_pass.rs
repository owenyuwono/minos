//! Terrain render pass — shader source and pipeline description.
//!
//! # Pipeline contract
//! The integration engineer must create one opaque pipeline with:
//! - Topology: triangle-list
//! - Polygon mode: FILL
//! - Cull mode: BACK, CCW front face
//! - Depth state: D32_SFLOAT, write ON, compare GREATER (reversed-Z)
//! - Blend: disabled (fully opaque)
//! - Descriptor set 0: `FrameUniforms` at binding 0 (`UNIFORM_BUFFER`)
//! - Push constants: `ChunkPush` (80 bytes) at offset 0, `VERTEX | FRAGMENT`
//! - Vertex inputs (four separate float-3 buffers):
//!   - slot 0 `@location(0)`: positions
//!   - slot 1 `@location(1)`: normals
//!   - slot 2 `@location(2)`: colors
//!   - slot 3 `@location(3)`: plate_colors (may be same buffer as colors when unused)
//!
//! # Call contract per draw
//! ```ignore
//! cmd.bind_descriptor_sets(0, &[frame_descriptor_set]);
//! cmd.push_constants(ChunkPush::new(model_matrix, 0));
//! cmd.bind_vertex_buffers(&[positions, normals, colors, plate_colors]);
//! cmd.bind_index_buffer(indices, u32);
//! cmd.draw_indexed(index_count, 1, 0, 0, 0);
//! ```

/// WGSL source for the terrain vertex + fragment shader.
///
/// Entry points: `vs_main` (vertex), `fs_main` (fragment).
/// Group/binding: `@group(0) @binding(0)` = `FrameUniforms` uniform.
/// Push constant: `var<immediate> pc: ChunkPush` (naga 29 WGSL syntax for Vulkan push constants).
pub const TERRAIN_WGSL: &str = include_str!("shaders/terrain.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terrain_wgsl_parses_with_naga() {
        let module = naga::front::wgsl::parse_str(TERRAIN_WGSL);
        assert!(
            module.is_ok(),
            "terrain.wgsl failed to parse: {:?}",
            module.err()
        );
    }
}

//! Material push-constant data.
//!
//! `ChunkPush` is sent as a Vulkan push constant per-draw-call. It carries the
//! per-chunk model matrix and a material mode discriminant.
//!
//! # Push constant layout (≤ 128 bytes, std430)
//! ```text
//! offset   field            size
//!      0   model           64  (mat4)
//!     64   material_mode    4  (u32)
//!     68   _pad[0]          4
//!     72   _pad[1]          4
//!     76   _pad[2]          4
//! total: 80 bytes
//! ```
//!
//! # Material modes
//! | Value | Name          | Status              |
//! |-------|---------------|---------------------|
//! |   0   | Normal-lit    | Implemented (M1)    |
//! |   1   | Debug normals | Implemented (M3)    |
//! |   2   | Plate-color   | Implemented (M3)    |
//! |   3   | Heightmap     | Implemented (M3)    |
//!
//! # Pipeline contract (integration note)
//! Push `ChunkPush` via `vkCmdPushConstants` with stage flags
//! `VERTEX | FRAGMENT`, offset 0, size `size_of::<ChunkPush>()` (80 bytes).
//! The shaders declare `var<push_constant> pc: ChunkPush;`.

use bytemuck::{Pod, Zeroable};

/// Per-chunk push constant — one per `vkCmdDrawIndexed` call.
///
/// Carries the chunk's model matrix and a discriminant selecting the active
/// material code path. Must be ≤ 128 bytes (minimum Vulkan push-constant guarantee).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct ChunkPush {
    /// Model-to-world matrix for this chunk. Combined with `FrameUniforms::view_proj`
    /// in the vertex shader: `clip = frame.view_proj * pc.model * vec4(pos, 1.0)`.
    pub model: [[f32; 4]; 4],
    /// Material mode discriminant. 0 = normal-lit, 1 = debug normals, 2 = plate-color, 3 = heightmap.
    pub material_mode: u32,
    /// Explicit padding to keep the struct at a multiple of 16 bytes.
    pub _pad: [u32; 3],
}

impl ChunkPush {
    /// Identity push constant: model = identity matrix, mode = 0 (normal-lit).
    pub fn identity() -> Self {
        Self {
            model: glam::Mat4::IDENTITY.to_cols_array_2d(),
            material_mode: 0,
            _pad: [0; 3],
        }
    }

    /// Build from a `glam::Mat4` model matrix and the given material mode.
    pub fn new(model: glam::Mat4, material_mode: u32) -> Self {
        Self {
            model: model.to_cols_array_2d(),
            material_mode,
            _pad: [0; 3],
        }
    }

    /// Build a camera-relative push constant for precision rendering.
    ///
    /// # Camera-relative rendering
    ///
    /// The translation component of `model` is computed as
    /// `(object_origin_f64 − camera_pos_f64) as f32` — the subtraction happens
    /// in f64 on the CPU so only a small near-origin residual is cast to f32.
    /// Combined with `FrameUniforms`'s rotation-only view (camera at origin),
    /// the shader sees `view_proj · model · pos` entirely in near-origin
    /// coordinates — no f32 cancellation of large planetary coordinates.
    ///
    /// `object_origin` — world-space f64 origin of the object (chunk centre,
    ///                   marker position, etc.)
    /// `camera_pos`    — world-space f64 camera position (from `Camera::position`)
    /// `rotation_scale` — optional rotation+scale component; pass
    ///                    `Mat4::IDENTITY` for axis-aligned chunks.
    /// `material_mode`  — material discriminant (0–3).
    pub fn camera_relative(
        object_origin: glam::DVec3,
        camera_pos: glam::DVec3,
        rotation_scale: glam::Mat4,
        material_mode: u32,
    ) -> Self {
        // f64 subtraction eliminates the ~50 km coordinate; the residual is
        // typically O(1)…O(10 000) m — well within f32 precision.
        let rel_f64 = object_origin - camera_pos;
        let rel_f32 = rel_f64.as_vec3();

        // Inject the camera-relative translation into the rotation/scale matrix.
        // The w column encodes the translation; other columns carry rotation/scale.
        let mut model = rotation_scale;
        model.w_axis.x = rel_f32.x;
        model.w_axis.y = rel_f32.y;
        model.w_axis.z = rel_f32.z;
        // w_axis.w must stay 1.0 for affine transforms.
        model.w_axis.w = 1.0;

        Self {
            model: model.to_cols_array_2d(),
            material_mode,
            _pad: [0; 3],
        }
    }

    /// Stamp per-draw geometry-debug ids into the spare pad slots (no size change):
    /// `_pad[0]` = a stable per-mesh-unit id (leaf/cluster colour), `_pad[1]` = LOD
    /// level. Read by `terrain.wgsl` view modes 3 (triangle) / 4 (cluster) / 5 (LOD).
    /// No-op for the lit/data paths (they ignore the pads).
    pub fn with_dbg(mut self, id: u32, level: u32) -> Self {
        self._pad[0] = id;
        self._pad[1] = level;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunk_push_is_pod_and_size() {
        assert_eq!(
            std::mem::size_of::<ChunkPush>(),
            80,
            "ChunkPush must be 80 bytes (mat4 + u32 + 3×u32 pad)"
        );
        assert!(
            std::mem::size_of::<ChunkPush>() <= 128,
            "ChunkPush must fit in 128-byte push-constant limit"
        );
        assert_eq!(std::mem::align_of::<ChunkPush>(), 4);
    }

    #[test]
    fn chunk_push_identity_mode_zero() {
        let cp = ChunkPush::identity();
        assert_eq!(cp.material_mode, 0);
        // Identity mat4 has 1s on the diagonal.
        assert_eq!(cp.model[0][0], 1.0);
        assert_eq!(cp.model[1][1], 1.0);
        assert_eq!(cp.model[2][2], 1.0);
        assert_eq!(cp.model[3][3], 1.0);
    }
}

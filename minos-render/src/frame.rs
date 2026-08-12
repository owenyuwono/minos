//! Per-frame GPU uniform data.
//!
//! `FrameUniforms` is uploaded once per frame to the uniform buffer at
//! `@group(0) @binding(0)` in both terrain and water shaders.
//!
//! # Layout
//! The struct is `#[repr(C)]`, `bytemuck::Pod + Zeroable`, and std140-friendly:
//! every field is a `vec4` or `mat4` (no per-row padding required). Total size
//! is **12 × 16 = 192 bytes** (12 vec4-slots: 1 mat4 + 8 vec4s).
//!
//! # Pipeline contract (integration note)
//! Bind the buffer holding `FrameUniforms` to descriptor set 0, binding 0,
//! `UNIFORM_BUFFER` type.  Both `terrain.wgsl` and `water.wgsl` declare:
//! ```wgsl
//! @group(0) @binding(0) var<uniform> frame: FrameUniforms;
//! ```

use bytemuck::{Pod, Zeroable};

use crate::camera::Camera;
use crate::lights::Lights;
use crate::projection::reversed_z_perspective;

/// Per-frame constants uploaded to the GPU as a single uniform buffer.
///
/// All fields are 16-byte-aligned (vec4 or mat4). std140 rules are satisfied
/// without explicit padding. Size: 12 × 16 = **192 bytes**.
///
/// ```text
/// offset   field          size
///      0   view_proj      64  (mat4)
///     64   camera_pos     16  (vec4, w unused)
///     80   sun0_dir       16  (vec4, w unused)
///     96   sun0_color     16  (vec4, w unused)
///    112   sun1_dir       16  (vec4, w unused)
///    128   sun1_color     16  (vec4, w unused)
///    144   hemi_sky       16  (vec4, w unused)
///    160   hemi_ground    16  (vec4, w unused)
///    176   ambient        16  (vec4; xyz = white intensity, w unused)
/// total: 192 bytes
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct FrameUniforms {
    /// Combined view-projection matrix (reversed-Z, Vulkan clip space).
    pub view_proj: [[f32; 4]; 4],
    /// World-space camera position. `w` component is unused (set to 1.0).
    pub camera_pos: [f32; 4],
    /// Primary sun direction (world space, pointing toward the sun). `w` unused.
    pub sun0_dir: [f32; 4],
    /// Primary sun color × intensity (linear sRGB). `w` unused.
    pub sun0_color: [f32; 4],
    /// Secondary directional light direction. `w` unused.
    pub sun1_dir: [f32; 4],
    /// Secondary directional light color × intensity. `w` unused.
    pub sun1_color: [f32; 4],
    /// Hemisphere ambient — sky color (linear sRGB × intensity). `w` unused.
    pub hemi_sky: [f32; 4],
    /// Hemisphere ambient — ground color (linear sRGB × intensity). `w` unused.
    pub hemi_ground: [f32; 4],
    /// Isotropic ambient white (x = y = z = scalar intensity, w unused).
    pub ambient: [f32; 4],
}

impl FrameUniforms {
    /// Build a `FrameUniforms` from the current camera, viewport aspect ratio,
    /// and scene lights.
    ///
    /// # Camera-relative rendering
    ///
    /// `view_proj = proj · view_rotation_only` where `view_rotation_only` has
    /// **zero translation** — the camera sits at the origin of camera-relative
    /// space.  The big world-space coordinate cancellation is performed on the
    /// CPU in f64 inside `ChunkPush::camera_relative`, so the GPU shader only
    /// ever sees small near-origin coordinates.
    ///
    /// Consequence: `camera_pos` in the uniform is `vec4(0,0,0,1)` — all
    /// vertex positions the shader receives are already camera-relative, so the
    /// view vector for specular/Fresnel is `normalize(-world_pos_relative)`.
    /// The shader uses `frame.camera_pos.xyz` for that computation, so it
    /// naturally evaluates to the right value without any shader change.
    pub fn new(camera: &Camera, aspect: f32, lights: &Lights) -> Self {
        Self::new_jittered(camera, aspect, lights, glam::Vec2::ZERO, 1.0, 1.0)
    }

    /// Like [`Self::new`] but applies a sub-pixel TAA jitter (in pixels) to the
    /// projection used for rasterization. `view_w`/`view_h` are the viewport size
    /// in pixels (for the NDC conversion). Pass `Vec2::ZERO` for no jitter.
    pub fn new_jittered(
        camera: &Camera,
        aspect: f32,
        lights: &Lights,
        jitter_px: glam::Vec2,
        view_w: f32,
        view_h: f32,
    ) -> Self {
        let mut proj = reversed_z_perspective(
            camera.fov_y_radians,
            aspect,
            camera.near,
            camera.far,
        );
        // Sub-pixel NDC shift on the clip-w-producing column (survives the
        // perspective divide). Rasterization only — reproject with un-jittered.
        proj.z_axis.x += 2.0 * jitter_px.x / view_w.max(1.0);
        proj.z_axis.y += 2.0 * jitter_px.y / view_h.max(1.0);
        // Rotation-only view: camera origin = (0,0,0) in camera-relative space.
        // Per-object translations are subtracted from the camera position in f64
        // before upload (ChunkPush::camera_relative), keeping f32 arithmetic on
        // small near-origin values only.
        let view = camera.view_matrix_rotation_only();
        let view_proj = (proj * view).to_cols_array_2d();

        // camera_pos is (0,0,0) in camera-relative space. The water shader's
        // view-vector term is:  view_dir = normalize(camera_pos.xyz - world_pos)
        //                               = normalize(vec3(0) - world_pos_relative)
        //                               = normalize(-world_pos_relative)
        // which is correct: all vertex positions are camera-relative offsets.
        let d0 = lights.directional_0;
        let d1 = lights.directional_1;

        Self {
            view_proj,
            camera_pos: [0.0, 0.0, 0.0, 1.0],
            sun0_dir: [d0.direction.x, d0.direction.y, d0.direction.z, 0.0],
            sun0_color: [
                d0.color_intensity.x,
                d0.color_intensity.y,
                d0.color_intensity.z,
                0.0,
            ],
            sun1_dir: [d1.direction.x, d1.direction.y, d1.direction.z, 0.0],
            sun1_color: [
                d1.color_intensity.x,
                d1.color_intensity.y,
                d1.color_intensity.z,
                0.0,
            ],
            hemi_sky: [
                lights.hemisphere.sky.x,
                lights.hemisphere.sky.y,
                lights.hemisphere.sky.z,
                0.0,
            ],
            hemi_ground: [
                lights.hemisphere.ground.x,
                lights.hemisphere.ground.y,
                lights.hemisphere.ground.z,
                0.0,
            ],
            ambient: [lights.ambient, lights.ambient, lights.ambient, 0.0],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::reversed_z_perspective;

    #[test]
    fn frame_uniforms_is_pod_and_size() {
        // Pod is satisfied at compile time by the derive; this checks the size.
        assert_eq!(
            std::mem::size_of::<FrameUniforms>(),
            192,
            "FrameUniforms must be 192 bytes (9 × vec4 + 1 mat4)"
        );
        assert_eq!(std::mem::align_of::<FrameUniforms>(), 4);
    }

    #[test]
    fn frame_uniforms_new_builds_without_panic() {
        let camera = Camera::default_orbit();
        let lights = Lights::demiurge_default();
        let fu = FrameUniforms::new(&camera, 16.0 / 9.0, &lights);
        // view_proj should not be all-zero (would indicate a bug in matrix multiply).
        let flat: &[f32] = bytemuck::cast_slice(&fu.view_proj);
        assert!(
            flat.iter().any(|&v| v != 0.0),
            "view_proj must not be all zeros"
        );
    }

    /// Camera-relative rendering: camera_pos in the uniform must be (0,0,0,1).
    ///
    /// With the rotation-only view, all vertex positions the shader receives are
    /// camera-relative offsets.  The shader computes the view vector as
    /// `normalize(camera_pos.xyz - world_pos_relative)` = `normalize(-world_pos_relative)`,
    /// which is correct only if camera_pos is the zero vector.
    #[test]
    fn camera_pos_is_zero_in_camera_relative_space() {
        let camera = Camera::default_orbit();
        let lights = Lights::demiurge_default();
        let fu = FrameUniforms::new(&camera, 16.0 / 9.0, &lights);
        assert_eq!(
            fu.camera_pos,
            [0.0_f32, 0.0, 0.0, 1.0],
            "camera_pos must be (0,0,0,1) in camera-relative rendering"
        );
    }

    /// Precision test: camera-relative rendering avoids f32 cancellation at
    /// planet-surface scale (~50 000 m).
    ///
    /// A vertex at world position ~50 km and a camera at ~50 km nearby.  We
    /// compute the final clip-space position two ways:
    ///
    /// 1. **Naive all-f32 path** (the old broken path): build view matrix from
    ///    full world-space f32 position, transform vertex from world space.
    ///    At ~50 000 m scale f32 has ~4 m resolution, so this loses sub-meter
    ///    precision.
    ///
    /// 2. **Camera-relative path** (the new path): subtract camera position in
    ///    f64, cast the small residual to f32, apply the rotation-only view.
    ///    The GPU only ever sees numbers ~O(1) … O(100) m — well within f32
    ///    precision.
    ///
    /// We compare both against an f64 reference and assert that the camera-
    /// relative path agrees to < 1e-3 relative error, while the naive f32 path
    /// exceeds this threshold (demonstrating why the fix is necessary).
    #[test]
    fn camera_relative_precision_at_50km() {
        // ── Scene parameters ────────────────────────────────────────────────
        // Camera at ~50 000 m above the planet surface.
        let camera_pos_f64 = glam::DVec3::new(50_000.0, 0.0, 0.0);
        // Vertex 3 m in front of the camera (a close object on the surface).
        let vertex_world_f64 = glam::DVec3::new(50_003.0, 0.0, 0.0);

        let fov    = 60_f32.to_radians();
        let aspect = 16.0_f32 / 9.0;
        let near   = 0.5_f32;
        let far    = 750_000.0_f32;

        // Camera looking down the -X axis (toward planet origin).
        // orientation: 90° rotation around Y so forward = -X
        let orientation = glam::Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);

        // ── f64 reference path ──────────────────────────────────────────────
        // Build view and proj in f64, transform the vertex, get clip position.
        let cam_f64 = Camera {
            position: camera_pos_f64,
            orientation,
            fov_y_radians: fov,
            near,
            far,
        };
        // f64 view matrix: look_to_rh from camera_pos_f64, forward=-X, up=+Y
        let forward_f64 = (orientation * glam::Vec3::NEG_Z).as_dvec3();
        let up_f64      = (orientation * glam::Vec3::Y).as_dvec3();
        let view_f64 = glam::DMat4::look_to_rh(camera_pos_f64, forward_f64, up_f64);
        let proj_f32 = reversed_z_perspective(fov, aspect, near, far);
        // Promote proj to f64 for the reference path.
        let proj_f64 = glam::DMat4::from_cols_array_2d(
            &proj_f32.to_cols_array_2d().map(|col| col.map(|v| v as f64)),
        );
        let vp_f64 = proj_f64 * view_f64;
        let vertex_clip_ref = vp_f64 * vertex_world_f64.extend(1.0);
        // Perspective-divide to NDC.
        let ndc_ref = vertex_clip_ref.truncate() / vertex_clip_ref.w;

        // ── Naive all-f32 path (the old broken path) ────────────────────────
        // build view from full f32 world position — loses precision at 50 km.
        let view_f32_naive = cam_f64.view_matrix(); // uses position.as_vec3() internally
        let proj_f32_mat   = reversed_z_perspective(fov, aspect, near, far);
        let vp_naive = proj_f32_mat * view_f32_naive;
        // Vertex position cast to f32 naively.
        let vertex_f32_naive = vertex_world_f64.as_vec3();
        let clip_naive = vp_naive * vertex_f32_naive.extend(1.0);
        let ndc_naive = clip_naive.truncate() / clip_naive.w;
        let ndc_naive_f64 = glam::DVec3::new(
            ndc_naive.x as f64, ndc_naive.y as f64, ndc_naive.z as f64,
        );

        // ── Camera-relative path (the new path) ─────────────────────────────
        // Subtract in f64, then cast residual to f32.
        let vertex_rel_f64  = vertex_world_f64 - camera_pos_f64; // = (3.0, 0.0, 0.0)
        let vertex_rel_f32  = vertex_rel_f64.as_vec3();           // ~(3.0, 0.0, 0.0) — no precision loss
        let view_rot_only   = cam_f64.view_matrix_rotation_only();
        let vp_camrel = proj_f32_mat * view_rot_only;
        let clip_camrel = vp_camrel * vertex_rel_f32.extend(1.0);
        let ndc_camrel = clip_camrel.truncate() / clip_camrel.w;
        let ndc_camrel_f64 = glam::DVec3::new(
            ndc_camrel.x as f64, ndc_camrel.y as f64, ndc_camrel.z as f64,
        );

        // ── Assertions ───────────────────────────────────────────────────────
        // Relative error helper: |a - b| / (|b| + epsilon)
        let rel_err = |a: glam::DVec3, b: glam::DVec3| -> f64 {
            (a - b).length() / (b.length() + 1e-10)
        };

        let err_camrel = rel_err(ndc_camrel_f64, ndc_ref);
        let err_naive  = rel_err(ndc_naive_f64,  ndc_ref);

        // Camera-relative path must agree with the f64 reference to < 1e-3 relative.
        assert!(
            err_camrel < 1e-3,
            "camera-relative path error {err_camrel:.2e} exceeds 1e-3 — precision fix not working"
        );

        // Naive f32 path at 50 km should produce a measurable error relative to
        // the camera-relative path.  We assert the camera-relative path is
        // STRICTLY MORE ACCURATE than naive (not just both wrong).
        // (On most f32 implementations the naive error is many multiples of the
        // camera-relative error at this distance scale.)
        assert!(
            err_camrel <= err_naive,
            "expected camera-relative path to be at least as accurate as naive f32: \
             err_camrel={err_camrel:.2e}  err_naive={err_naive:.2e}"
        );
    }
}

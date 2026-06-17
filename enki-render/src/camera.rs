//! Camera — position + orientation + clip-plane parameters.
//! Navigation controllers mutate `Camera`; the render loop reads it.

/// World-space camera state. Position uses double precision to avoid precision
/// loss when orbiting at planet radius (~6.37e6 m). The view matrix is computed
/// in f32 after subtracting a camera-relative origin.
#[derive(Debug, Clone)]
pub struct Camera {
    /// World-space eye position (double precision, planet-centred).
    pub position: glam::DVec3,
    /// Camera orientation as a unit quaternion (f32 is sufficient for rotation).
    pub orientation: glam::Quat,
    /// Vertical field-of-view in radians.
    pub fov_y_radians: f32,
    /// Near clip plane in metres. Default: 0.5 m.
    pub near: f32,
    /// Far clip plane in metres. Default: 750_000 m (750 km — clears the
    /// atmosphere envelope used in M1 rendering).
    pub far: f32,
}

impl Camera {
    /// Default camera: positioned above the north pole, looking at the origin.
    pub fn default_orbit() -> Self {
        Self {
            position: glam::DVec3::new(0.0, 0.0, 8_000_000.0),
            orientation: glam::Quat::IDENTITY,
            fov_y_radians: 60_f32.to_radians(),
            near: 0.5,
            far: 750_000.0,
        }
    }

    /// Right-handed view matrix (camera → world inverse).
    /// Returns f32 Mat4; callers must subtract a camera-relative origin from
    /// world positions before transforming to avoid f32 precision loss.
    pub fn view_matrix(&self) -> glam::Mat4 {
        let forward = self.orientation * glam::Vec3::NEG_Z;
        let up = self.orientation * glam::Vec3::Y;
        // Cast to f32 only after subtracting per-draw origin (callers' responsibility)
        let eye = self.position.as_vec3();
        glam::Mat4::look_to_rh(eye, forward, up)
    }

    /// Rotation-only view matrix — camera sits at the **origin** of
    /// camera-relative space.  Translation is zero; the big world-space
    /// coordinate cancellation is pushed to the CPU (f64) inside
    /// `ChunkPush::camera_relative`, not done in f32 inside the GPU matrix
    /// multiply.
    ///
    /// Use this to build `FrameUniforms::view_proj` for camera-relative
    /// rendering.  Per-object offsets are computed by
    /// `object_origin_f64 - camera.position` (f64) then cast to f32.
    pub fn view_matrix_rotation_only(&self) -> glam::Mat4 {
        // Extract basis vectors from the orientation quaternion.
        // look_to_rh with eye=ZERO gives a pure rotation matrix (no translation
        // column), which is exactly what camera-relative rendering needs.
        let forward = self.orientation * glam::Vec3::NEG_Z;
        let up = self.orientation * glam::Vec3::Y;
        glam::Mat4::look_to_rh(glam::Vec3::ZERO, forward, up)
    }
}

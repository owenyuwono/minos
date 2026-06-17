//! Reversed-Z projection.
//!
//! # Convention (MUST be consistent across the entire engine)
//! - Depth buffer: `D32_SFLOAT`, cleared to **0.0** (not 1.0).
//! - Depth compare op: `GREATER` (not `LESS`).
//! - Near maps to NDC depth **1.0**; far maps to **0.0**.
//! - This gives uniform floating-point precision from near 0.5 m to far 750 km,
//!   eliminating z-fighting over the full planet-scale range.
//! - No per-shader log-z hack required (that was a WebGL2 fallback).

/// Build a reversed-Z perspective projection matrix.
///
/// # Parameters
/// - `fov_y_radians`: vertical field of view.
/// - `aspect`: viewport width / height.
/// - `near`: near clip plane (metres). Must be > 0.
/// - `far`: far clip plane (metres). Must be > near.
///
/// # Vulkan clip space
/// Returns a matrix in Vulkan clip space (Y-down, depth [0, 1]).
/// Callers pass this directly to the vertex-shader push constant or UBO.
pub fn reversed_z_perspective(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> glam::Mat4 {
    // Standard reversed-Z derivation: swap near/far roles.
    // tan(fov/2) scale factor:
    let f = 1.0 / (fov_y_radians / 2.0).tan();
    // Vulkan: Y is negated relative to OpenGL, handled by glam's RH Vulkan variant.
    // We manually build reversed-Z to be explicit about the convention.
    // Reversed-Z derivation (RH, Vulkan [0,1] depth, near→1.0, far→0.0):
    //   NDC.z = (A * z_e + B) / (-z_e)  where clip.w = -z_e
    //   At z_e=-near: NDC.z=1  →  A = near / (far - near)
    //   At z_e=-far:  NDC.z=0  →  B = near * far / (far - near)
    let a = near / (far - near);
    let b = near * far / (far - near);
    glam::Mat4::from_cols(
        glam::Vec4::new(f / aspect, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, -f, 0.0, 0.0), // flip Y for Vulkan
        glam::Vec4::new(0.0, 0.0, a, -1.0),
        glam::Vec4::new(0.0, 0.0, b, 0.0),
    )
}

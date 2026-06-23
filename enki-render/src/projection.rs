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

/// Build a reversed-Z orthographic projection (Vulkan clip: Y-down, depth [0,1],
/// **near→1.0, far→0.0** — same convention as [`reversed_z_perspective`]). Used for
/// the sun shadow map's light projection. `w` stays 1 (no perspective divide), so
/// depth is linear in view-space z; a closer caster stores a *greater* depth, which
/// pairs with a `GREATER_OR_EQUAL` comparison sampler + an ADD depth bias on the
/// receiver (see the Nanite/flora `sample_shadow`).
pub fn reversed_z_orthographic(
    l: f32,
    r: f32,
    b: f32,
    t: f32,
    near: f32,
    far: f32,
) -> glam::Mat4 {
    let sx = 2.0 / (r - l);
    let sy = 2.0 / (t - b);
    // ndc.z = z_e / (far - near) + far / (far - near): at z_e=-near → 1, z_e=-far → 0.
    let a = 1.0 / (far - near);
    let bz = far / (far - near);
    let tx = -(r + l) / (r - l);
    let ty = (t + b) / (t - b); // Vulkan Y-flip folds the OpenGL sign in
    glam::Mat4::from_cols(
        glam::Vec4::new(sx, 0.0, 0.0, 0.0),
        glam::Vec4::new(0.0, -sy, 0.0, 0.0), // flip Y for Vulkan
        glam::Vec4::new(0.0, 0.0, a, 0.0),
        glam::Vec4::new(tx, ty, bz, 1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reversed_z_ortho_depth_convention() {
        // The #1 silent-inversion trap: prove near→1, far→0, monotonic (closer =
        // greater), and Vulkan Y-flip, before any GPU shadow code relies on it.
        let m = reversed_z_orthographic(-10.0, 10.0, -10.0, 10.0, 1.0, 100.0);
        let d = |z: f32| {
            let c = m * glam::Vec4::new(0.0, 0.0, z, 1.0);
            c.z / c.w
        };
        assert!((d(-1.0) - 1.0).abs() < 1e-5, "near plane must map to depth 1");
        assert!(d(-100.0).abs() < 1e-5, "far plane must map to depth 0");
        assert!(d(-1.0) > d(-50.0) && d(-50.0) > d(-100.0), "reversed-Z: closer = greater depth");
        // Right/top edges → +1 / −1 (Vulkan Y-down).
        let corner = m * glam::Vec4::new(10.0, 10.0, -1.0, 1.0);
        assert!((corner.x / corner.w - 1.0).abs() < 1e-5, "right edge → +1");
        assert!((corner.y / corner.w + 1.0).abs() < 1e-5, "top edge → −1 (Vulkan Y-down)");
    }
}

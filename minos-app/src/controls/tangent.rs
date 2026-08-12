//! `tangent` — shared surface-walker primitives on a spherical planet.
//!
//! Small headless helpers (no winit/GPU) used by the surface controllers: the
//! per-frame WASD request ([`MoveInput`]), the sprint multiplier, the analytic
//! terrain radius probe ([`surface_radius`]), and the tangent-plane projection
//! ([`project_onto_tangent_plane`]) that keeps a heading perpendicular to the
//! local up as the player moves around the sphere.

use minos_planet::height::HeightField;
use glam::{DVec3, Vec3};

// ── MoveInput ─────────────────────────────────────────────────────────────

/// Per-frame WASD movement request (plain booleans — no winit dependency).
#[derive(Debug, Clone, Copy, Default)]
pub struct MoveInput {
    /// Move forward along the local heading.
    pub forward: bool,
    /// Move backward along the local heading.
    pub backward: bool,
    /// Strafe left (perpendicular to heading, in the tangent plane).
    pub left: bool,
    /// Strafe right.
    pub right: bool,
    /// Sprint: multiply speed by [`SPRINT_MULTIPLIER`].
    pub sprint: bool,
}

// ── Constants ─────────────────────────────────────────────────────────────

/// Sprint multiplier applied when `MoveInput::sprint` is true.
pub const SPRINT_MULTIPLIER: f64 = 5.0;

// ── Helpers ───────────────────────────────────────────────────────────────

/// **Analytic** terrain radius (m) along unit `dir`: `base_radius + hf.height·scale`
/// — the smooth procedural curve at full detail.
///
/// This is NOT the rendered surface. The drawn mesh is flat triangles between grid
/// vertices, and this curve carries sub-cell detail those triangles never show, so
/// grounding a visible object on it makes the object dive into bumps that aren't
/// drawn. Do NOT use it for grounding — use
/// [`crate::controls::terrain_grid::ground_radius`] (the single source of truth)
/// for anything that sits on the ground. Kept as a conservative "is this above
/// ground" probe. `dir` must be unit length.
pub(crate) fn surface_radius(
    hf: &dyn HeightField,
    base_radius: f64,
    height_scale: f64,
    dir: DVec3,
) -> f64 {
    base_radius + hf.height(dir, 0) * height_scale
}

/// Project `v` onto the tangent plane defined by unit normal `up`, then
/// normalise.  Falls back to a safe perpendicular to `up` if degenerate.
pub(crate) fn project_onto_tangent_plane(v: Vec3, up: Vec3) -> Vec3 {
    let projected = v - up * v.dot(up);
    if projected.length_squared() < 1e-10 {
        // v was parallel to up — pick an arbitrary perpendicular
        let alt = if up.abs().x < 0.9 { Vec3::X } else { Vec3::Z };
        (alt - up * alt.dot(up)).normalize()
    } else {
        projected.normalize()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_parallel_to_up_falls_back_safely() {
        let up = Vec3::Y;
        // v == up should not produce NaN
        let result = project_onto_tangent_plane(Vec3::Y, up);
        assert!(result.is_finite());
        assert!((result.length() - 1.0).abs() < 1e-5);
        // Must be perpendicular to up
        assert!(result.dot(up).abs() < 1e-5);
    }
}

//! `terrain_grid` — grounding interface + helpers.
#![allow(dead_code)]
//!
//! Grounding is "ride the mesh the eye sees, not the analytic curve" (which carries
//! sub-cell octave detail the mesh never draws → point samples sink into bumps that
//! aren't there). The rendered surface IS the collider: the voxel terrain's
//! `VoxelCollider` raycasts its actual triangles, behind the [`SurfaceCollider`]
//! trait — the character feet, scattered flora, and any future prop query that.
//!
//! This module now provides only: that trait, the cube-sphere inverse
//! [`dir_to_face_uv`] (used to find the leaf under a direction), and the analytic
//! [`ground_radius`] used as a FALLBACK where no rendered leaf is resident.

use minos_planet::face_bases::FACE_BASES;
use minos_planet::height::HeightField;
use glam::DVec3;

use super::PLANET_RADIUS;

/// A queryable ground-collision surface — the SINGLE source of truth for where the
/// rendered terrain sits along a direction. Implemented by the voxel terrain's
/// `VoxelCollider`, which raycasts the ACTUAL drawn triangles. The character feet +
/// scattered flora ground against this so they ride the exact mesh, not a parallel
/// analytic approximation (which always sinks/floats by the approximation error).
pub trait SurfaceCollider: Send + Sync {
    /// Radius (m from the planet centre) where the ray (centre → `dir`) hits the
    /// rendered surface, or `None` if no leaf is resident along `dir` (caller falls
    /// back to the analytic [`ground_radius`]).
    fn ground_radius(&self, dir: DVec3) -> Option<f64>;
}

/// Heightfield LOD the grounding samples. The voxel terrain renders
/// `surface_radius(dir, leaf_level)`, and near the player the LOD reaches its finest
/// (≈ the voxel `max_depth` = 12). Sampling height at this level here makes grounding
/// agree with the drawn voxel surface; the old value (0) was the Nanite bake level,
/// so the LOD-additive octaves 0→12 showed up as trees/character floating.
pub const GROUND_LEVEL: u8 = 12;

/// Analytic FALLBACK grounding — the radius (m from the planet centre) of the
/// heightfield surface along `dir`. This is the true continuous surface the voxel mesh
/// approximates; it is used ONLY where no rendered leaf is resident: flora placed from
/// orbit (no player) and the brief pre-streaming gap. The primary grounding is the
/// rendered mesh itself, via [`SurfaceCollider`] (the voxel terrain's `VoxelCollider`).
pub fn ground_radius(hf: &dyn HeightField, height_scale: f64, dir: DVec3) -> f64 {
    PLANET_RADIUS + hf.height(dir, GROUND_LEVEL) * height_scale
}

/// Invert the cube-sphere map: unit `dir` → `(face, cu, cv)` with `cu, cv ∈ [-1, 1]`
/// the face-local cube coords such that `cube_to_sphere(n + u·cu + v·cv) == dir`.
///
/// Face = dominant signed axis. On a face the Everitt–Zucker warp (with the normal
/// component fixed at 1) reduces to
///   `dir·u = cu·√(0.5 − cv²/6)`,  `dir·v = cv·√(0.5 − cu²/6)`,
/// a mild coupled pair solved by fixed-point iteration (a contraction; the corner
/// is the slowest case and still converges in a handful of steps).
pub fn dir_to_face_uv(dir: DVec3) -> (usize, f64, f64) {
    let (ax, ay, az) = (dir.x.abs(), dir.y.abs(), dir.z.abs());
    let face = if ax >= ay && ax >= az {
        if dir.x >= 0.0 { 0 } else { 1 }
    } else if ay >= az {
        if dir.y >= 0.0 { 2 } else { 3 }
    } else if dir.z >= 0.0 {
        4
    } else {
        5
    };
    let b = &FACE_BASES[face];
    let bu = dir.dot(b.u);
    let bv = dir.dot(b.v);

    let inv_root = 0.5_f64.sqrt();
    let mut cu = bu / inv_root;
    let mut cv = bv / inv_root;
    for _ in 0..32 {
        let ncu = bu / (0.5 - cv * cv / 6.0).max(1e-9).sqrt();
        let ncv = bv / (0.5 - cu * cu / 6.0).max(1e-9).sqrt();
        let done = (ncu - cu).abs() < 1e-12 && (ncv - cv).abs() < 1e-12;
        cu = ncu;
        cv = ncv;
        if done {
            break;
        }
    }
    (face, cu.clamp(-1.0, 1.0), cv.clamp(-1.0, 1.0))
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use minos_planet::face_bases::cube_to_sphere;

    /// Forward map: face-local cube coords (cu, cv) → unit sphere direction.
    fn face_dir(face: usize, cu: f64, cv: f64) -> DVec3 {
        let b = &FACE_BASES[face];
        cube_to_sphere(b.n + b.u * cu + b.v * cv).normalize()
    }

    #[test]
    fn inverse_warp_round_trips() {
        for &(cu, cv) in &[(0.0, 0.0), (0.3, -0.6), (-0.8, 0.2), (0.95, 0.95)] {
            for face in 0..6 {
                let dir = face_dir(face, cu, cv);
                let (f, rcu, rcv) = dir_to_face_uv(dir);
                assert_eq!(f, face, "face mismatch at ({cu},{cv})");
                assert!((rcu - cu).abs() < 1e-6 && (rcv - cv).abs() < 1e-6,
                    "uv round-trip face {face}: ({cu},{cv}) -> ({rcu},{rcv})");
            }
        }
    }

}

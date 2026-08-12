//! Cube-face basis vectors and sphere-projection utilities.

use glam::DVec3;

/// Orthonormal basis for one of the 6 cube faces.
pub struct FaceBasis {
    pub n: DVec3,
    pub u: DVec3,
    pub v: DVec3,
}

/// The 6 cube-face bases: `[+X, −X, +Y, −Y, +Z, −Z]`.
///
/// Each face is described by a unit normal `n` and two tangent axes `u`, `v`
/// such that a point on the face in `[-1,1]²` is `n + u·fu + v·fv`.
/// Conventions match `demiurge/src/planet/faceBases.ts`.
pub const FACE_BASES: [FaceBasis; 6] = [
    // Face 0: +X — normal=+X, u=+Z, v=+Y
    FaceBasis { n: DVec3::X,     u: DVec3::Z,     v: DVec3::Y },
    // Face 1: −X — normal=−X, u=−Z, v=+Y
    FaceBasis { n: DVec3::NEG_X, u: DVec3::NEG_Z, v: DVec3::Y },
    // Face 2: +Y — normal=+Y, u=+X, v=+Z
    FaceBasis { n: DVec3::Y,     u: DVec3::X,     v: DVec3::Z },
    // Face 3: −Y — normal=−Y, u=+X, v=−Z
    FaceBasis { n: DVec3::NEG_Y, u: DVec3::X,     v: DVec3::NEG_Z },
    // Face 4: +Z — normal=+Z, u=−X, v=+Y
    FaceBasis { n: DVec3::Z,     u: DVec3::NEG_X, v: DVec3::Y },
    // Face 5: −Z — normal=−Z, u=+X, v=+Y
    FaceBasis { n: DVec3::NEG_Z, u: DVec3::X,     v: DVec3::Y },
];

/// Project a cube-surface point onto the unit sphere using the Everitt-Zucker
/// equal-area warp.
///
/// For each axis component the warp is:
/// ```text
///   sx = x · √max(0, 1 − y²/2 − z²/2 + y²·z²/3)   (cyclic for sy, sz)
/// ```
/// Input `c` should be a point in `[−1,1]³` on a cube face. The output is
/// already unit-length for any cube-face point, but callers may normalize to
/// guard against floating-point drift.
pub fn cube_to_sphere(c: DVec3) -> DVec3 {
    let x2 = c.x * c.x;
    let y2 = c.y * c.y;
    let z2 = c.z * c.z;
    let sx = c.x * f64::sqrt(f64::max(0.0, 1.0 - y2 * 0.5 - z2 * 0.5 + y2 * z2 / 3.0));
    let sy = c.y * f64::sqrt(f64::max(0.0, 1.0 - z2 * 0.5 - x2 * 0.5 + z2 * x2 / 3.0));
    let sz = c.z * f64::sqrt(f64::max(0.0, 1.0 - x2 * 0.5 - y2 * 0.5 + x2 * y2 / 3.0));
    DVec3::new(sx, sy, sz)
}

/// Return the unit direction to the center of quadtree patch `(face, level, ix, iy)`.
///
/// The patch tile covers `1/2^level` of the `[−1,1]²` face in each axis. The
/// center in face-local `[-1,1]²` coords is:
/// ```text
///   fu = (ix + 0.5) / 2^level * 2 − 1
///   fv = (iy + 0.5) / 2^level * 2 − 1
/// ```
/// That maps to the cube point `n + u·fu + v·fv`, which is then warped and
/// normalized.
pub fn patch_center_dir(face: u8, level: u8, ix: u32, iy: u32) -> DVec3 {
    let scale = 1.0 / (1u64 << level) as f64;
    let fu = ((ix as f64 + 0.5) * scale) * 2.0 - 1.0;
    let fv = ((iy as f64 + 0.5) * scale) * 2.0 - 1.0;

    let b = &FACE_BASES[face as usize];
    let cube_pt = b.n + b.u * fu + b.v * fv;
    cube_to_sphere(cube_pt).normalize()
}

/// Tangent-frame basis for horizon/azimuth queries at unit surface direction
/// `dir` (the radial up). Returns `(tan_a, tan_b)` — two orthonormal tangents in
/// the plane ⊥ `dir`, defining "azimuth 0" (`tan_a`) and "azimuth +90°" (`tan_b`).
///
/// **This is the single source of truth for azimuth 0**, shared by the horizon
/// bake and the runtime shader (which mirrors this formula). It derives from
/// `dir` ALONE — body +Y projected onto the tangent plane, falling back to +X at
/// the poles — so bake and runtime agree regardless of cube face. (Cube-face u/v
/// axes flip across faces and must NOT be used here.)
pub fn horizon_basis(dir: DVec3) -> (DVec3, DVec3) {
    let up = dir;
    let r = if up.y.abs() < 0.999 { DVec3::Y } else { DVec3::X };
    let tan_a = (r - up * r.dot(up)).normalize();
    let tan_b = up.cross(tan_a); // unit, ⊥ both (right-handed)
    (tan_a, tan_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-12;

    /// `cube_to_sphere` must produce a unit vector for any face point in [-1,1]³.
    #[test]
    fn cube_to_sphere_is_unit_length() {
        // Sample points on all 6 faces at several positions.
        let samples: &[(f64, f64, f64)] = &[
            // Face centers
            ( 1.0,  0.0,  0.0),
            (-1.0,  0.0,  0.0),
            ( 0.0,  1.0,  0.0),
            ( 0.0, -1.0,  0.0),
            ( 0.0,  0.0,  1.0),
            ( 0.0,  0.0, -1.0),
            // Off-center face points
            ( 1.0,  0.7, -0.3),
            (-1.0, -0.5,  0.8),
            ( 0.4,  1.0,  0.9),
            (-0.9, -1.0,  0.2),
            ( 0.6, -0.1,  1.0),
            ( 0.3,  0.8, -1.0),
            // Corners
            ( 1.0,  1.0,  1.0),
            ( 1.0, -1.0,  1.0),
            (-1.0,  1.0, -1.0),
        ];
        for &(x, y, z) in samples {
            let v = cube_to_sphere(DVec3::new(x, y, z));
            let len = v.length();
            assert!(
                (len - 1.0).abs() < EPS,
                "cube_to_sphere({x},{y},{z}) length={len}, expected 1.0 within {EPS}"
            );
        }
    }

    /// Level-0 patch (0,0) on each face must point to the face normal.
    #[test]
    fn patch_center_dir_level0_is_face_normal() {
        let expected = [
            DVec3::X,
            DVec3::NEG_X,
            DVec3::Y,
            DVec3::NEG_Y,
            DVec3::Z,
            DVec3::NEG_Z,
        ];
        for face in 0u8..6 {
            let dir = patch_center_dir(face, 0, 0, 0);
            let exp = expected[face as usize];
            let diff = (dir - exp).length();
            assert!(
                diff < EPS,
                "face {face}: patch_center_dir got {dir:?}, expected {exp:?}, diff={diff}"
            );
        }
    }

    /// node_size formula: (PI/2 · radius) / 2^level.
    #[test]
    fn node_size_formula() {
        use crate::quadtree::QuadtreeNode;
        let radius = 6_371_000.0_f64;
        for level in 0u8..=8 {
            let node = QuadtreeNode::new(0, level, 0, 0, radius);
            let expected = (PI / 2.0 * radius) / (1u64 << level) as f64;
            assert!(
                (node.node_size - expected).abs() < 1e-6,
                "level {level}: node_size={} expected={expected}",
                node.node_size
            );
        }
    }

    /// `horizon_basis` returns an orthonormal tangent pair ⊥ `dir`, stable at poles.
    #[test]
    fn horizon_basis_orthonormal_and_tangent() {
        let dirs = [
            DVec3::new(0.3, 0.2, 0.9).normalize(),
            DVec3::Y,            // pole — +Y fallback path
            DVec3::NEG_Y,        // other pole
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(-0.5, 0.7, -0.5).normalize(),
        ];
        for d in dirs {
            let (a, b) = horizon_basis(d);
            assert!((a.length() - 1.0).abs() < 1e-9, "tan_a not unit for {d:?}");
            assert!((b.length() - 1.0).abs() < 1e-9, "tan_b not unit for {d:?}");
            assert!(a.dot(d).abs() < 1e-9, "tan_a not ⊥ dir for {d:?}");
            assert!(b.dot(d).abs() < 1e-9, "tan_b not ⊥ dir for {d:?}");
            assert!(a.dot(b).abs() < 1e-9, "tan_a not ⊥ tan_b for {d:?}");
            // Right-handed: tan_a × tan_b == dir.
            assert!((a.cross(b) - d).length() < 1e-9, "frame not right-handed for {d:?}");
        }
    }

    /// patch_center_dir produces unit-length output across many faces/levels.
    #[test]
    fn patch_center_dir_unit_length() {
        for face in 0u8..6 {
            for level in 0u8..=5 {
                let tiles = 1u32 << level;
                for ix in [0, tiles / 2, tiles - 1] {
                    for iy in [0, tiles / 2, tiles - 1] {
                        let dir = patch_center_dir(face, level, ix, iy);
                        let len = dir.length();
                        assert!(
                            (len - 1.0).abs() < EPS,
                            "face={face} level={level} ix={ix} iy={iy}: length={len}"
                        );
                    }
                }
            }
        }
    }
}

//! Cube-map utilities: texel ↔ direction projection, bilinear sampling.
//!
//! Face order: 0=+X, 1=−X, 2=+Y, 3=−Y, 4=+Z, 5=−Z
//! Same convention as `face_bases.rs` / `cubemap.ts`.
//!
//! Cross-face neighbours use the dir-roundtrip trick (same as cubemap.ts):
//!   1. Compute texel-centre direction.
//!   2. Nudge by ~1.2 texel-angles along the appropriate face tangent.
//!   3. Re-project with `dir_to_texel` → correct face/texel automatically.

use glam::DVec3;
use crate::face_bases::FACE_BASES;

// ---------------------------------------------------------------------------
// Public: Texel coordinate
// ---------------------------------------------------------------------------

/// (face, x, y) integer texel coordinate.
#[derive(Debug, Clone, Copy, Default)]
pub struct Texel {
    pub face: usize,
    pub x: usize,
    pub y: usize,
}

// ---------------------------------------------------------------------------
// dirToTexel — max-axis cube projection, nearest texel
// ---------------------------------------------------------------------------

/// Project a unit direction onto the cube face with the closest normal.
/// Returns the nearest integer texel at the given resolution.
/// Mirrors `dirToTexel` in `cubemap.ts`.
pub fn dir_to_texel(dir: DVec3, res: usize) -> Texel {
    let ax = dir.x.abs();
    let ay = dir.y.abs();
    let az = dir.z.abs();

    let (face, sc, tc): (usize, f64, f64);

    if ax >= ay && ax >= az {
        if dir.x > 0.0 {
            // Face 0: +X — u=+Z, v=+Y
            face = 0; sc = dir.z; tc = dir.y;
        } else {
            // Face 1: −X — u=−Z, v=+Y
            face = 1; sc = -dir.z; tc = dir.y;
        }
    } else if ay >= ax && ay >= az {
        if dir.y > 0.0 {
            // Face 2: +Y — u=+X, v=+Z
            face = 2; sc = dir.x; tc = dir.z;
        } else {
            // Face 3: −Y — u=+X, v=−Z
            face = 3; sc = dir.x; tc = -dir.z;
        }
    } else if dir.z > 0.0 {
        // Face 4: +Z — u=−X, v=+Y
        face = 4; sc = -dir.x; tc = dir.y;
    } else {
        // Face 5: −Z — u=+X, v=+Y
        face = 5; sc = dir.x; tc = dir.y;
    }

    let mc = if ax >= ay && ax >= az { ax } else if ay >= ax && ay >= az { ay } else { az };
    let half = res as f64 * 0.5;
    let xi = ((sc / mc + 1.0) * half).floor() as isize;
    let yi = ((tc / mc + 1.0) * half).floor() as isize;

    Texel {
        face,
        x: xi.clamp(0, (res as isize) - 1) as usize,
        y: yi.clamp(0, (res as isize) - 1) as usize,
    }
}

// ---------------------------------------------------------------------------
// texelToDir — texel-centre direction (unit vector)
// ---------------------------------------------------------------------------

/// Compute the unit-sphere direction at the CENTRE of texel `(face, x, y)`.
/// Uses `FACE_BASES` identical to `texelToDir` in `cubemap.ts`.
pub fn texel_to_dir(face: usize, x: usize, y: usize, res: usize) -> DVec3 {
    let b = &FACE_BASES[face];
    let half = res as f64 * 0.5;
    let u = (x as f64 + 0.5) / half - 1.0;
    let v = (y as f64 + 0.5) / half - 1.0;
    // cube point = n + u·tangentU + v·tangentV
    (b.n + b.u * u + b.v * v).normalize()
}

// ---------------------------------------------------------------------------
// texelIndex — flat array address
// ---------------------------------------------------------------------------

/// `face * res * res + y * res + x`
#[inline(always)]
pub fn texel_index(face: usize, x: usize, y: usize, res: usize) -> usize {
    face * res * res + y * res + x
}

// ---------------------------------------------------------------------------
// texAng — per-texel angular pitch
// ---------------------------------------------------------------------------

/// Angular pitch per texel at the face centre.
/// `(π/2) / res` radians per texel — a quarter circle divided by `res`.
#[inline(always)]
pub fn tex_ang(res: usize) -> f64 {
    std::f64::consts::FRAC_PI_2 / res as f64
}

// ---------------------------------------------------------------------------
// neighborTexel — cross-face neighbour via dir-roundtrip
// ---------------------------------------------------------------------------

/// Return the texel one grid step `(dx, dy)` from `(face, x, y)`.
///
/// In-bounds: trivial integer offset.
/// Out-of-bounds: dir-roundtrip through the face tangent (no adjacency table).
/// Mirrors `neighborTexel` in `cubemap.ts`.
pub fn neighbor_texel(face: usize, x: usize, y: usize, dx: i32, dy: i32, res: usize) -> Texel {
    let nx = x as i32 + dx;
    let ny = y as i32 + dy;

    if nx >= 0 && nx < res as i32 && ny >= 0 && ny < res as i32 {
        return Texel { face, x: nx as usize, y: ny as usize };
    }

    // Out-of-bounds: nudge the texel-centre direction along face tangents.
    let mut dir = texel_to_dir(face, x, y, res);
    let ang = tex_ang(res) * 1.2;
    let b = &FACE_BASES[face];

    if dx != 0 {
        dir += b.u * (dx as f64 * ang);
    }
    if dy != 0 {
        dir += b.v * (dy as f64 * ang);
    }

    dir_to_texel(dir.normalize(), res)
}

// ---------------------------------------------------------------------------
// sampleSmooth — bilinear interpolation across faces
// ---------------------------------------------------------------------------

/// Sample a per-texel `f32` slice smoothly at direction `dir`.
///
/// TRUE BILINEAR interpolation. Interior fast path (whole 2×2 cell on same
/// face) avoids cross-face neighbor calls. Edge path uses `neighbor_texel`.
/// Mirrors `sampleSmooth` in `cubemap.ts`.
pub fn sample_smooth(data: &[f32], dir: DVec3, res: usize) -> f64 {
    let ax = dir.x.abs();
    let ay = dir.y.abs();
    let az = dir.z.abs();

    let (face, sc, tc, mc): (usize, f64, f64, f64);

    if ax >= ay && ax >= az {
        mc = ax;
        if dir.x > 0.0 { face = 0; sc =  dir.z; tc = dir.y; }
        else            { face = 1; sc = -dir.z; tc = dir.y; }
    } else if ay >= ax && ay >= az {
        mc = ay;
        if dir.y > 0.0 { face = 2; sc = dir.x; tc =  dir.z; }
        else            { face = 3; sc = dir.x; tc = -dir.z; }
    } else {
        mc = az;
        if dir.z > 0.0 { face = 4; sc = -dir.x; tc = dir.y; }
        else            { face = 5; sc =  dir.x; tc = dir.y; }
    }

    // Continuous texel coords (integer = texel centre).
    let half = res as f64 * 0.5;
    let fx = (sc / mc + 1.0) * half - 0.5;
    let fy = (tc / mc + 1.0) * half - 0.5;

    let x0i = fx.floor() as isize;
    let y0i = fy.floor() as isize;
    let fu = fx - x0i as f64;
    let fv = fy - y0i as f64;
    let x1i = x0i + 1;
    let y1i = y0i + 1;

    let (t00, t10, t01, t11): (f64, f64, f64, f64);

    // Interior fast path: whole 2×2 cell on this face, in-bounds.
    if x0i >= 0 && y0i >= 0 && x1i < res as isize && y1i < res as isize {
        let x0 = x0i as usize;
        let y0 = y0i as usize;
        let x1 = x1i as usize;
        let y1 = y1i as usize;
        let base = face * res * res;
        t00 = data[base + y0 * res + x0] as f64;
        t10 = data[base + y0 * res + x1] as f64;
        t01 = data[base + y1 * res + x0] as f64;
        t11 = data[base + y1 * res + x1] as f64;
    } else {
        // Edge path: at least one corner crosses a face boundary.
        // Anchor = nearest in-bounds texel of the 2×2 cell.
        let ax_ = x0i.clamp(0, (res as isize) - 1) as usize;
        let ay_ = y0i.clamp(0, (res as isize) - 1) as usize;
        let ox0 = (x0i - ax_ as isize) as i32;
        let ox1 = (x1i - ax_ as isize) as i32;
        let oy0 = (y0i - ay_ as isize) as i32;
        let oy1 = (y1i - ay_ as isize) as i32;

        let n00 = neighbor_texel(face, ax_, ay_, ox0, oy0, res);
        let n10 = neighbor_texel(face, ax_, ay_, ox1, oy0, res);
        let n01 = neighbor_texel(face, ax_, ay_, ox0, oy1, res);
        let n11 = neighbor_texel(face, ax_, ay_, ox1, oy1, res);

        t00 = data[texel_index(n00.face, n00.x, n00.y, res)] as f64;
        t10 = data[texel_index(n10.face, n10.x, n10.y, res)] as f64;
        t01 = data[texel_index(n01.face, n01.x, n01.y, res)] as f64;
        t11 = data[texel_index(n11.face, n11.x, n11.y, res)] as f64;
    }

    let top = t00 + (t10 - t00) * fu;
    let bot = t01 + (t11 - t01) * fu;
    top + (bot - top) * fv
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const RES: usize = 128;

    /// Round-trip: `texel_to_dir` then `dir_to_texel` must return the same texel.
    #[test]
    fn texel_roundtrip() {
        for face in 0..6usize {
            for y in [0, 32, 63, 64, 127] {
                for x in [0, 32, 63, 64, 127] {
                    let dir = texel_to_dir(face, x, y, RES);
                    let t = dir_to_texel(dir, RES);
                    assert_eq!(t.face, face, "face mismatch at ({face},{x},{y})");
                    assert_eq!(t.x, x, "x mismatch at ({face},{x},{y})");
                    assert_eq!(t.y, y, "y mismatch at ({face},{x},{y})");
                }
            }
        }
    }

    /// `texel_to_dir` outputs must be unit length.
    #[test]
    fn texel_to_dir_unit_length() {
        for face in 0..6usize {
            for y in 0..RES {
                for x in 0..RES {
                    let d = texel_to_dir(face, x, y, RES);
                    let len = d.length();
                    assert!((len - 1.0).abs() < 1e-12,
                        "face={face} x={x} y={y}: length={len}");
                }
            }
        }
    }

    /// Face-centre texels must point along the face normal (within rounding).
    #[test]
    fn face_centre_points_along_normal() {
        let normals = [
            DVec3::X, DVec3::NEG_X, DVec3::Y, DVec3::NEG_Y, DVec3::Z, DVec3::NEG_Z,
        ];
        let mid = RES / 2 - 1; // center-ish texel
        for face in 0..6usize {
            let dir = texel_to_dir(face, mid, mid, RES);
            let dot = dir.dot(normals[face]);
            assert!(dot > 0.99, "face {face}: dot with normal={dot}");
        }
    }

    /// `tex_ang` for res=128 ≈ π/(2·128).
    #[test]
    fn tex_ang_value() {
        let expected = PI / (2.0 * RES as f64);
        let got = tex_ang(RES);
        assert!((got - expected).abs() < 1e-15,
            "tex_ang: got={got} expected={expected}");
    }

    /// In-bounds neighbors are trivial offsets.
    #[test]
    fn neighbor_texel_inbounds() {
        let t = neighbor_texel(0, 10, 20, 1, -1, RES);
        assert_eq!(t.face, 0);
        assert_eq!(t.x, 11);
        assert_eq!(t.y, 19);
    }

    /// `sample_smooth` on a constant field returns that constant everywhere.
    #[test]
    fn sample_smooth_constant_field() {
        let data = vec![0.5f32; 6 * RES * RES];
        let dirs = [DVec3::X, DVec3::NEG_Y, DVec3::Z, DVec3::new(1.0, 1.0, 1.0).normalize()];
        for dir in dirs {
            let v = sample_smooth(&data, dir, RES);
            assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
        }
    }

    /// `sample_smooth` is deterministic.
    #[test]
    fn sample_smooth_deterministic() {
        let data: Vec<f32> = (0..6 * RES * RES).map(|i| (i % 256) as f32 / 255.0).collect();
        let dir = DVec3::new(0.6, 0.5, 0.3).normalize();
        let v1 = sample_smooth(&data, dir, RES);
        let v2 = sample_smooth(&data, dir, RES);
        assert_eq!(v1.to_bits(), v2.to_bits());
    }
}

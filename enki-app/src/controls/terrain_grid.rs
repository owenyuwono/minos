//! `terrain_grid` — sample the *rendered* terrain surface under a direction.
#![allow(dead_code)]
//!
//! The character must stand on the mesh the eye sees, not on the full analytic
//! height field. The Nanite terrain is a cube-sphere grid of `grid_res` quads per
//! face side at level 0 (`enki-nanite/bake/tessellate.rs`): each grid vertex is
//! `cube_to_sphere(n + u·cu + v·cv)` displaced by `hf.height(dir, 0)`, and the
//! quads are split along the `b–c` diagonal (winding `a,c,b / b,c,d`).
//!
//! [`grid_surface_radius`] reproduces that exactly: it finds the grid cell under a
//! direction (inverting the Everitt–Zucker warp), then ray-casts the planet-centre
//! ray against the triangle the direction falls in. The result is the radius of
//! the **faceted** surface — so a point between vertices rides the flat triangle,
//! not the analytic curve. That kills the "diving": the analytic curve carries
//! sub-cell octave detail the mesh never draws, so a point sample of it sinks into
//! bumps that aren't there.
//!
//! ponytail: targets the Nanite renderer (the default). The classic quadtree
//! tessellates the same field at a different per-chunk resolution; grounding stays
//! close but isn't triangle-exact there.

use enki_planet::face_bases::{cube_to_sphere, FACE_BASES};
use enki_planet::height::HeightField;
use glam::DVec3;

use super::PLANET_RADIUS;

/// **The single source of truth for grounding.** The radius of the *rendered*
/// (grid-faceted) terrain along `dir`, at the canonical planet radius + Nanite
/// bake resolution. Everything that sits on the ground — the character's feet,
/// scattered trees, future props — must ride THIS, so they agree with each other
/// and with the drawn mesh. Wraps [`grid_surface_radius`] (the parameterized impl;
/// tests/tools that need a custom radius or resolution call that directly).
///
/// Grounding against the analytic field instead (`tangent::surface_radius`)
/// makes objects dive into sub-cell detail the mesh never draws — see module docs.
pub fn ground_radius(hf: &dyn HeightField, height_scale: f64, dir: DVec3) -> f64 {
    grid_surface_radius(hf, PLANET_RADIUS, height_scale, crate::NANITE_BAKE_RES, dir)
}

/// Radius (m) of the rendered terrain mesh along unit `dir`, reproducing the
/// Nanite tessellator at `grid_res` quads per face side (level 0).
pub fn grid_surface_radius(
    hf: &dyn HeightField,
    base_radius: f64,
    height_scale: f64,
    grid_res: u32,
    dir: DVec3,
) -> f64 {
    let (face, cu, cv) = dir_to_face_uv(dir);
    let basis = &FACE_BASES[face];
    let res = grid_res as f64;

    // Grid float coords; the cell is (gi, gj), fraction (su, sv) within it.
    let fu = ((cu + 1.0) * 0.5 * res).clamp(0.0, res);
    let fv = ((cv + 1.0) * 0.5 * res).clamp(0.0, res);
    let gi = (fu.floor() as i64).clamp(0, grid_res as i64 - 1);
    let gj = (fv.floor() as i64).clamp(0, grid_res as i64 - 1);
    let su = fu - gi as f64;
    let sv = fv - gj as f64;

    // World position of grid vertex (i, j) — the forward map, exactly as the baker.
    let corner = |i: i64, j: i64| -> DVec3 {
        let ccu = (i as f64 / res) * 2.0 - 1.0;
        let ccv = (j as f64 / res) * 2.0 - 1.0;
        let d = cube_to_sphere(basis.n + basis.u * ccu + basis.v * ccv).normalize();
        d * (base_radius + hf.height(d, 0) * height_scale)
    };

    // Pick the triangle the point falls in. The quad (a,c,b / b,c,d) splits along
    // the b–c diagonal `su + sv = 1`: a=(gi,gj), b=(gi+1,gj), c=(gi,gj+1), d=(gi+1,gj+1).
    let (w0, w1, w2) = if su + sv <= 1.0 {
        (corner(gi, gj), corner(gi, gj + 1), corner(gi + 1, gj)) // a, c, b
    } else {
        (corner(gi + 1, gj), corner(gi, gj + 1), corner(gi + 1, gj + 1)) // b, c, d
    };

    // Radius where the planet-centre ray (origin, dir) meets the triangle plane:
    // n·(dir·t) = n·w0  ⇒  t = n·w0 / n·dir   (t is the radius since dir is unit).
    let n = (w1 - w0).cross(w2 - w0);
    let denom = n.dot(dir);
    if denom.abs() < 1e-12 {
        return base_radius + hf.height(dir, 0) * height_scale; // degenerate fallback
    }
    n.dot(w0) / denom
}

/// Invert the cube-sphere map: unit `dir` → `(face, cu, cv)` with `cu, cv ∈ [-1, 1]`
/// the face-local cube coords such that `cube_to_sphere(n + u·cu + v·cv) == dir`.
///
/// Face = dominant signed axis. On a face the Everitt–Zucker warp (with the normal
/// component fixed at 1) reduces to
///   `dir·u = cu·√(0.5 − cv²/6)`,  `dir·v = cv·√(0.5 − cu²/6)`,
/// a mild coupled pair solved by fixed-point iteration (a contraction; the corner
/// is the slowest case and still converges in a handful of steps).
fn dir_to_face_uv(dir: DVec3) -> (usize, f64, f64) {
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

    const BASE: f64 = 50_000.0;
    const SCALE: f64 = 1_000.0;

    /// Smooth height — the grid should track it closely.
    struct SmoothHf;
    impl HeightField for SmoothHf {
        fn height(&self, dir: DVec3, _l: u8) -> f64 {
            0.1 * dir.normalize().y
        }
    }

    /// A spike confined to a tiny solid angle — present in a point sample but NOT
    /// at any grid vertex, so the mesh (and thus grid grounding) must ignore it.
    struct SpikeHf {
        at: DVec3,
    }
    impl HeightField for SpikeHf {
        fn height(&self, dir: DVec3, _l: u8) -> f64 {
            if (dir.normalize() - self.at).length() < 1e-4 { 1.0 } else { 0.0 }
        }
    }

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

    #[test]
    fn grid_vertex_returns_exact_height() {
        // At a grid vertex the ray hits a triangle corner → the exact vertex radius.
        let res = 256;
        let (gi, gj) = (100i64, 173i64);
        let dir = face_dir(0, (gi as f64 / res as f64) * 2.0 - 1.0,
                              (gj as f64 / res as f64) * 2.0 - 1.0);
        let got = grid_surface_radius(&SmoothHf, BASE, SCALE, res, dir);
        let analytic = BASE + SmoothHf.height(dir, 0) * SCALE;
        assert!((got - analytic).abs() < 1e-3, "vertex radius {got} vs analytic {analytic}");
    }

    #[test]
    fn ignores_subcell_spike() {
        // Spike at a cell *centre* (never a vertex). The point sample sees +SCALE m;
        // the grid must ride the flat (zero-height) triangle ≈ BASE instead.
        let res = 256;
        let (gi, gj) = (100.5, 173.5);
        let spike_dir = face_dir(0, (gi / res as f64) * 2.0 - 1.0, (gj / res as f64) * 2.0 - 1.0);
        let hf = SpikeHf { at: spike_dir };

        let analytic = BASE + hf.height(spike_dir, 0) * SCALE; // = BASE + 1000
        let grid = grid_surface_radius(&hf, BASE, SCALE, res, spike_dir);

        assert!((analytic - (BASE + SCALE)).abs() < 1e-6, "point sample should see the spike");
        assert!(grid < BASE + 1.0, "grid must ignore the sub-cell spike, got {grid}");
        assert!(grid > BASE - 5.0, "grid radius unreasonably low: {grid}");
    }
}

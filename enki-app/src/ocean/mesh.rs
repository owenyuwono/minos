//! Ocean wave-mesh generators for the three [`super::OceanMesh`] modes.
//!
//! - [`grid_mesh`]    — **Warped (A)**: one camera-anchored tangent grid, radially
//!   warped so cells bunch toward the center (= under the camera).
//! - [`clipmap_mesh`] — **Clipmap (D)**: concentric power-of-two rings (dense
//!   center → coarse rim), crack-free via per-cell transition fans on the inner
//!   edge (the shared mid-vertices are deduped against the finer ring).
//! - [`ndc_grid`] + [`project_to_sphere`] — **Projected (C)**: a screen-space grid
//!   whose vertices are ray-cast onto the sea sphere each frame (CPU f64).
//!
//! A/D meshes are tangent-space `[gx, cell, gz]` (shared `vs_main` path): `.xz` is
//! the tangent position, `.y` the local world cell size for the Nyquist cascade fade.

use std::collections::HashMap;

use glam::DVec3;

// ── A: warped grid ─────────────────────────────────────────────────────────────

/// Center-cell fraction of the radial warp (avoids a degenerate zero-size center)
/// and the falloff steepness. The outer ring stays at ±half so `edge_fade` + the
/// shell seam are unaffected. Tune `WARP_MIN` down for a denser center.
const WARP_MIN: f32 = 0.2;
const WARP_POW: f32 = 2.5;

/// A `(res+1)²` grid in the XZ plane spanning `[-half, +half]`, **radially warped**
/// (small cells at center, large at rim). Positions `[gx, cell, gz]`.
pub fn grid_mesh(res: u32, half: f32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let n = (res + 1) as usize;

    // 1. Warped XZ (chebyshev radius keeps square rings aligned with edge_fade).
    let mut xz = vec![[0.0f32; 2]; n * n];
    for j in 0..n {
        for i in 0..n {
            let u = i as f32 / res as f32 * 2.0 - 1.0;
            let v = j as f32 / res as f32 * 2.0 - 1.0;
            let r = u.abs().max(v.abs());
            let fr = WARP_MIN * r + (1.0 - WARP_MIN) * r.powf(WARP_POW);
            let s = if r > 1e-6 { fr / r } else { WARP_MIN };
            xz[j * n + i] = [u * s * half, v * s * half];
        }
    }

    // 2. Pack local cell size (distance to +i/+j neighbour) into y.
    let mut positions = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            let p = xz[j * n + i];
            let pi = xz[j * n + (i + 1).min(n - 1)];
            let pj = xz[(j + 1).min(n - 1) * n + i];
            let di = ((pi[0] - p[0]).powi(2) + (pi[1] - p[1]).powi(2)).sqrt();
            let dj = ((pj[0] - p[0]).powi(2) + (pj[1] - p[1]).powi(2)).sqrt();
            positions.push([p[0], di.max(dj), p[1]]);
        }
    }

    // 3. Indices (CCW from above: east×north = up).
    let nu = res + 1;
    let mut indices = Vec::with_capacity((res * res * 6) as usize);
    for j in 0..res {
        for i in 0..res {
            let a = j * nu + i;
            let b = a + 1;
            let c = a + nu;
            let d = c + 1;
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    (positions, indices)
}

// ── D: clipmap rings ───────────────────────────────────────────────────────────

/// Concentric nested grid: a center block of `2·block` cells at the finest size
/// `c0`, then `levels` rings each doubling the cell size and extent. Inner ring
/// edges are stitched to the finer ring with transition fans (shared mid-vertices
/// deduped), so the mesh is watertight (no T-junctions). `block` must be even.
/// Positions `[gx, cell, gz]`; total extent = `block · c0 · 2^levels`.
pub fn clipmap_mesh(block: i32, c0: f32, levels: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
    assert!(block % 2 == 0, "block must be even so ring boundaries align");
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut map: HashMap<(i32, i32), u32> = HashMap::new();
    let mut indices: Vec<u32> = Vec::new();

    // Vertex at lattice (ix,iz); `cell` recorded only on first creation (finer
    // rings are generated first → the finer cell size wins at shared boundaries).
    let vert = |positions: &mut Vec<[f32; 3]>,
                map: &mut HashMap<(i32, i32), u32>,
                ix: i32,
                iz: i32,
                cell: f32|
     -> u32 {
        *map.entry((ix, iz)).or_insert_with(|| {
            positions.push([ix as f32 * c0, cell, iz as f32 * c0]);
            (positions.len() - 1) as u32
        })
    };

    // Fan-triangulate a CCW perimeter from its corner farthest from the origin
    // (so the subdivided, hole-facing inner edges never make a degenerate fan).
    let emit = |positions: &mut Vec<[f32; 3]>, idx: &mut Vec<u32>, peri: &[u32]| {
        let mut o = 0usize;
        let mut best = -1.0f32;
        for (k, &vi) in peri.iter().enumerate() {
            let p = positions[vi as usize];
            let d2 = p[0] * p[0] + p[2] * p[2];
            if d2 > best {
                best = d2;
                o = k;
            }
        }
        let n = peri.len();
        for i in 1..n - 1 {
            idx.extend_from_slice(&[peri[o], peri[(o + i) % n], peri[(o + i + 1) % n]]);
        }
    };

    // Center block: full grid of step-1 cells over [-block, block].
    for iz in -block..block {
        for ix in -block..block {
            let bl = vert(&mut positions, &mut map, ix, iz, c0);
            let br = vert(&mut positions, &mut map, ix + 1, iz, c0);
            let tr = vert(&mut positions, &mut map, ix + 1, iz + 1, c0);
            let tl = vert(&mut positions, &mut map, ix, iz + 1, c0);
            emit(&mut positions, &mut indices, &[bl, br, tr, tl]);
        }
    }

    // Rings: step s = 2^k, annulus [inner, outer] = [block·2^(k-1), block·2^k].
    for k in 1..=levels {
        let s = 1i32 << k;
        let half = s / 2;
        let inner = block * (1 << (k - 1));
        let outer = block * (1 << k);
        let cell = c0 * s as f32;
        let mut iz = -outer;
        while iz < outer {
            let mut ix = -outer;
            while ix < outer {
                // Skip cells fully inside the hole (covered by the finer ring).
                let in_hole =
                    ix >= -inner && ix + s <= inner && iz >= -inner && iz + s <= inner;
                if in_hole {
                    ix += s;
                    continue;
                }
                // Which edges abut the hole (need a mid vertex from the finer ring)?
                let x_in = ix >= -inner && ix + s <= inner;
                let z_in = iz >= -inner && iz + s <= inner;
                let bottom = iz == inner && x_in; // edge z=iz on the hole's top
                let top = iz + s == -inner && x_in; // edge z=iz+s on the hole's bottom
                let left = ix == inner && z_in; // edge x=ix on the hole's right
                let right = ix + s == -inner && z_in; // edge x=ix+s on the hole's left

                // CCW perimeter: bl, [mid-b], br, [mid-r], tr, [mid-t], tl, [mid-l].
                let mut peri = Vec::with_capacity(8);
                peri.push(vert(&mut positions, &mut map, ix, iz, cell));
                if bottom {
                    peri.push(vert(&mut positions, &mut map, ix + half, iz, cell));
                }
                peri.push(vert(&mut positions, &mut map, ix + s, iz, cell));
                if right {
                    peri.push(vert(&mut positions, &mut map, ix + s, iz + half, cell));
                }
                peri.push(vert(&mut positions, &mut map, ix + s, iz + s, cell));
                if top {
                    peri.push(vert(&mut positions, &mut map, ix + half, iz + s, cell));
                }
                peri.push(vert(&mut positions, &mut map, ix, iz + s, cell));
                if left {
                    peri.push(vert(&mut positions, &mut map, ix, iz + half, cell));
                }
                emit(&mut positions, &mut indices, &peri);
                ix += s;
            }
            iz += s;
        }
    }

    (positions, indices)
}

// ── C: projected grid ──────────────────────────────────────────────────────────

/// A uniform `(res+1)²` grid of NDC positions in `[-1, 1]²` (the screen-space
/// lattice projected onto the sea sphere each frame) + its triangle indices.
pub fn ndc_grid(res: u32) -> (Vec<[f32; 2]>, Vec<u32>) {
    let n = res + 1;
    let mut ndc = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            ndc.push([i as f32 / res as f32 * 2.0 - 1.0, j as f32 / res as f32 * 2.0 - 1.0]);
        }
    }
    let mut indices = Vec::with_capacity((res * res * 6) as usize);
    for j in 0..res {
        for i in 0..res {
            let a = j * n + i;
            let b = a + 1;
            let c = a + n;
            let d = c + 1;
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    (ndc, indices)
}

/// Ray-cast a camera ray onto the sea-level sphere, all camera-relative (camera at
/// the origin). `center_rel` = planet centre − camera; `radius` = sea radius.
/// Returns the near hit; on a miss (ray above the horizon) returns the nearest
/// point ON the sphere (the limb silhouette) so the mesh's top edge sits on the
/// horizon with no NaNs. `dir` need not be normalized.
pub fn project_to_sphere(dir: DVec3, center_rel: DVec3, radius: f64, near: f64) -> DVec3 {
    let d = dir.normalize();
    let b = d.dot(center_rel);
    let c = center_rel.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc >= 0.0 {
        let t = b - disc.sqrt();
        if t >= near {
            return d * t;
        }
    }
    // Miss (or sphere behind near plane) → snap to the sphere point nearest the ray.
    let closest = d * b;
    let to = closest - center_rel;
    let len = to.length();
    if len > 1e-6 {
        center_rel + to / len * radius
    } else {
        d * near
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Watertight: every edge is shared by 1 or 2 triangles, and every count-1
    /// (boundary) edge lies on the outermost ring — i.e. no interior T-junctions.
    #[test]
    fn clipmap_is_watertight() {
        let block = 8;
        let c0 = 4.0;
        let levels = 4;
        let (pos, idx) = clipmap_mesh(block, c0, levels);
        assert!(idx.len() % 3 == 0);
        let outer = (block * (1 << levels)) as f32 * c0;

        let mut edges: HashMap<(u32, u32), u32> = HashMap::new();
        for tri in idx.chunks_exact(3) {
            // No degenerate triangles.
            let [a, b, c] = [tri[0], tri[1], tri[2]];
            assert!(a != b && b != c && a != c, "degenerate index triangle");
            let area = tri_area(&pos, a, b, c);
            assert!(area > 1e-3, "near-degenerate triangle area {area}");
            for &(u, v) in &[(a, b), (b, c), (c, a)] {
                let key = if u < v { (u, v) } else { (v, u) };
                *edges.entry(key).or_insert(0) += 1;
            }
        }
        for (&(u, v), &count) in &edges {
            assert!(count <= 2, "non-manifold edge shared {count}× ");
            if count == 1 {
                // Boundary edges must be on the outer rim (|x| or |z| == outer).
                let on_rim = |i: u32| {
                    let p = pos[i as usize];
                    (p[0].abs() - outer).abs() < 0.5 || (p[2].abs() - outer).abs() < 0.5
                };
                assert!(on_rim(u) && on_rim(v), "interior crack / T-junction edge");
            }
        }
    }

    fn tri_area(pos: &[[f32; 3]], a: u32, b: u32, c: u32) -> f32 {
        let pa = pos[a as usize];
        let pb = pos[b as usize];
        let pc = pos[c as usize];
        let ux = pb[0] - pa[0];
        let uz = pb[2] - pa[2];
        let vx = pc[0] - pa[0];
        let vz = pc[2] - pa[2];
        0.5 * (ux * vz - uz * vx).abs()
    }

    #[test]
    fn project_hits_sea_below_camera() {
        // Camera 1000 m above the sea, looking straight down → hit 1000 m below.
        let r = 50_000.0;
        let camera = DVec3::new(0.0, 0.0, r + 1000.0);
        let center_rel = -camera;
        let hit = project_to_sphere(DVec3::new(0.0, 0.0, -1.0), center_rel, r, 0.5);
        assert!((hit - DVec3::new(0.0, 0.0, -1000.0)).length() < 1e-3, "got {hit:?}");
    }

    #[test]
    fn project_miss_lands_on_sphere() {
        // A ray pointing away from the planet → snapped onto the sphere (limb).
        let r = 50_000.0;
        let camera = DVec3::new(0.0, 0.0, r + 1000.0);
        let center_rel = -camera;
        let hit = project_to_sphere(DVec3::new(0.0, 1.0, 1.0), center_rel, r, 0.5);
        // |hit_world| == radius (on the sphere): hit is camera-relative, world = hit + camera.
        let world = hit + camera;
        assert!((world.length() - r).abs() < 1.0, "limb point off sphere: {}", world.length());
    }
}

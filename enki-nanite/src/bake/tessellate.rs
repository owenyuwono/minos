//! High-resolution patch tessellator — the LOD0 source mesh for the baker.
//!
//! Produces ONE watertight, indexed triangle mesh for a cube-sphere quadtree
//! patch `(face, level, ix, iy)` sampled at `resolution` quads per side from a
//! [`HeightField`]. Unlike `enki-planet`'s `build_chunk`, it emits **no skirts**
//! (the Nanite DAG handles cracks via locked boundaries) and additionally
//! returns a per-vertex `boundary` mask — the 4 outer patch edges — which the DAG
//! builder locks at every simplify level so the patch can tile seamlessly.
//!
//! Positions are stored **patch-origin-relative** (f32): `origin` is the displaced
//! patch center in planet space (f64), and each position is `(world_f64 - origin)`.

use enki_planet::face_bases::{cube_to_sphere, horizon_basis, FACE_BASES};
use enki_planet::height::HeightField;
use glam::DVec3;

// ── Horizon self-shadow bake ─────────────────────────────────────────────────
// Per-vertex terrain self-shadow data: for each surface point, the SINE of the
// max terrain elevation angle in 4 azimuth directions (the `horizon_basis`
// tangents: tan_a, tan_b, -tan_a, -tan_b). Sun-direction-agnostic → baked once,
// the moving sun is compared against it at runtime (nanite_draw.wgsl). Stored as
// a SINE so the runtime test is a pure dot-product (no trig).
//
// ponytail: a COARSE per-patch grid is marched (cost decoupled from the ~1024²
// mesh), then each vertex bilinearly samples it. Upgrade to a per-face horizon
// *texture* (needs an enki-rhi set1) only if crisper edges are needed.
const HORIZON_AZIMUTHS: usize = 4;
/// March steps per azimuth (quadratic-spaced: dense near, sparse far).
const HORIZON_STEPS: u32 = 16;
/// Max great-circle arc to march (rad) ≈ 12 km on a 50 km-radius planet; beyond
/// this, curvature drops terrain below the local horizon anyway.
const HORIZON_MAX_ARC: f64 = 0.24;
/// Cap on the coarse grid side (texels = cap+1). Real bake res 1024 → 256; small
/// test meshes use their own (smaller) resolution so tests stay fast.
const HORIZON_GRID_CAP: usize = 256;

/// Max elevation-angle SINE of terrain seen from the surface point at unit `dir`
/// looking along tangent `t`, by marching the height field along the great circle.
/// Curvature is implicit (real 3-D points), so terrain that curves away yields ≤ 0.
fn march_horizon_sine(dir: DVec3, t: DVec3, p: &PatchParams, hf: &dyn HeightField) -> f32 {
    let p0 = dir * (p.radius + hf.height(dir, p.level) * p.height_scale);
    let mut max_s = 0.0f64;
    for step in 1..=HORIZON_STEPS {
        let f = step as f64 / HORIZON_STEPS as f64;
        let arc = HORIZON_MAX_ARC * f * f; // quadratic: dense near the receiver
        let di = (dir * arc.cos() + t * arc.sin()).normalize();
        let pi = di * (p.radius + hf.height(di, p.level) * p.height_scale);
        let los = pi - p0;
        let len = los.length();
        if len > 1e-6 {
            let s = los.dot(dir) / len; // sine of elevation above p0's tangent plane
            if s > max_s {
                max_s = s;
            }
        }
    }
    max_s.max(0.0) as f32
}

/// Build a coarse `(hg+1)²` horizon grid over the patch's parameter square; each
/// texel stores 4 horizon-elevation sines (the 4 `horizon_basis` azimuths).
fn build_horizon_grid(p: &PatchParams, hg: usize, hf: &dyn HeightField) -> Vec<[f32; 4]> {
    let basis = &FACE_BASES[p.face as usize];
    let scale = 1.0 / (1u64 << p.level) as f64;
    let g = hg + 1;
    let mut grid = vec![[0.0f32; 4]; g * g];
    for gy in 0..g {
        for gx in 0..g {
            let u = (p.ix as f64 + gx as f64 / hg as f64) * scale;
            let v = (p.iy as f64 + gy as f64 / hg as f64) * scale;
            let dir = cube_to_sphere(basis.n + basis.u * (u * 2.0 - 1.0) + basis.v * (v * 2.0 - 1.0))
                .normalize();
            let (ta, tb) = horizon_basis(dir);
            grid[gy * g + gx] = [
                march_horizon_sine(dir, ta, p, hf),
                march_horizon_sine(dir, tb, p, hf),
                march_horizon_sine(dir, -ta, p, hf),
                march_horizon_sine(dir, -tb, p, hf),
            ];
        }
    }
    grid
}

/// Bilinearly sample the horizon grid at vertex grid coord `(gi, gj)` (∈ 0..=res).
fn sample_horizon(grid: &[[f32; 4]], hg: usize, res: u32, gi: usize, gj: usize) -> [f32; 4] {
    let g = hg + 1;
    let fx = (gi as f64 / res as f64) * hg as f64;
    let fy = (gj as f64 / res as f64) * hg as f64;
    let x0 = (fx.floor() as usize).min(hg - 1);
    let y0 = (fy.floor() as usize).min(hg - 1);
    let tx = (fx - x0 as f64) as f32;
    let ty = (fy - y0 as f64) as f32;
    let (c00, c10) = (grid[y0 * g + x0], grid[y0 * g + x0 + 1]);
    let (c01, c11) = (grid[(y0 + 1) * g + x0], grid[(y0 + 1) * g + x0 + 1]);
    let mut out = [0.0f32; 4];
    for k in 0..HORIZON_AZIMUTHS {
        let a = c00[k] * (1.0 - tx) + c10[k] * tx;
        let b = c01[k] * (1.0 - tx) + c11[k] * tx;
        out[k] = a * (1.0 - ty) + b * ty;
    }
    out
}

/// Parameters describing one patch to tessellate.
#[derive(Debug, Clone, Copy)]
pub struct PatchParams {
    pub face: u8,
    pub level: u8,
    pub ix: u32,
    pub iy: u32,
    /// Quads per side. The vertex grid is `(resolution + 1)^2`.
    pub resolution: u32,
    pub radius: f64,
    pub height_scale: f64,
}

/// A tessellated patch: parallel vertex arrays + triangle indices + lock mask.
#[derive(Debug, Clone)]
pub struct PatchMesh {
    /// Patch-origin-relative positions (f32).
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 3]>,
    /// Per-vertex rock hardness 0..1 (debug "material" view), parallel to `colors`.
    pub material: Vec<f32>,
    /// Per-vertex surface wetness 0..1 (debug "wetness" view), parallel to `colors`.
    pub wetness: Vec<f32>,
    /// Per-vertex volcano-cone influence 0..1 (debug "volcano" view), parallel to `colors`.
    pub volcanism: Vec<f32>,
    /// Per-vertex signed normalized elevation h (~[-1,1]) (debug "height" view).
    pub elevation: Vec<f32>,
    /// Per-vertex per-plate tint (debug "plate" view), parallel to `colors`.
    pub plate: Vec<[f32; 3]>,
    /// Per-vertex terrain self-shadow: SINE of the max terrain elevation angle in
    /// 4 azimuths (`horizon_basis` tangents). Sun-direction-agnostic; runtime
    /// compares the sun elevation against it. Rides the DAG like the other scalars.
    pub horizon: Vec<[f32; 4]>,
    /// Triangle-list indices, CCW from outside, 32-bit.
    pub indices: Vec<u32>,
    /// Per-vertex: `true` if the vertex lies on an outer patch edge (lock for DAG).
    pub boundary: Vec<bool>,
    /// Absolute planet-space origin (f64); positions are relative to this.
    pub origin: DVec3,
}

/// Map grid coordinate `(gi, gj)` — possibly outside `[0, res]` for ghost
/// vertices used in normal estimation — to `(dir, world, height)`.
fn eval_vertex(p: &PatchParams, gi: i64, gj: i64, hf: &dyn HeightField) -> (DVec3, DVec3, f64) {
    let basis = &FACE_BASES[p.face as usize];
    let res = p.resolution as f64;
    let scale = 1.0 / (1u64 << p.level) as f64;
    let u = (p.ix as f64 + gi as f64 / res) * scale;
    let v = (p.iy as f64 + gj as f64 / res) * scale;
    let cu = u * 2.0 - 1.0;
    let cv = v * 2.0 - 1.0;
    let dir = cube_to_sphere(basis.n + basis.u * cu + basis.v * cv).normalize();
    let h = hf.height(dir, p.level);
    (dir, dir * (p.radius + h * p.height_scale), h)
}

/// Surface normal at sphere direction `dir`, computed from a **face-independent**
/// tangent basis so adjacent cube faces produce the *identical* normal at a shared
/// edge — eliminating the cross-face shading seam.
///
/// The old per-patch central difference used ghost vertices extrapolated in
/// face-parameter space (`gi = -1` lands off the cube face), so each face invented
/// a different edge normal. Here the tangent basis derives purely from `dir`, and
/// the height gradient is sampled in the sphere tangent plane — a function of the
/// world direction alone, hence continuous across every face boundary.
fn surface_normal(dir: DVec3, p: &PatchParams, hf: &dyn HeightField) -> DVec3 {
    // Tangent basis from the direction only (the basis *orientation* doesn't
    // affect the resulting surface normal, only the finite-difference sampling).
    let up_ref = if dir.y.abs() < 0.99 { DVec3::Y } else { DVec3::X };
    let t1 = dir.cross(up_ref).normalize();
    let t2 = dir.cross(t1).normalize();
    // One grid step in angle, so the normal captures detail at the mesh scale.
    let scale = 1.0 / (1u64 << p.level) as f64;
    let eps = std::f64::consts::FRAC_PI_2 * scale / p.resolution as f64;
    let disp = |d: DVec3| -> DVec3 {
        let dn = d.normalize();
        dn * (p.radius + hf.height(dn, p.level) * p.height_scale)
    };
    let pr = disp(dir + t1 * eps);
    let pl = disp(dir - t1 * eps);
    let pu = disp(dir + t2 * eps);
    let pd = disp(dir - t2 * eps);
    let mut n = (pr - pl).cross(pu - pd).normalize();
    if n.dot(dir) < 0.0 {
        n = -n;
    }
    n
}

/// Tessellate one quadtree patch into a watertight indexed mesh + lock mask.
pub fn tessellate_patch(p: &PatchParams, hf: &dyn HeightField) -> PatchMesh {
    let res = p.resolution;
    let grid = (res + 1) as usize;
    let vcount = grid * grid;

    // Origin = displaced patch center (f64), mirroring enki-planet's mesher so the
    // camera-relative precision convention matches the rest of the engine.
    let center_dir = {
        let basis = &FACE_BASES[p.face as usize];
        let scale = 1.0 / (1u64 << p.level) as f64;
        let cu = (p.ix as f64 + 0.5) * scale * 2.0 - 1.0;
        let cv = (p.iy as f64 + 0.5) * scale * 2.0 - 1.0;
        cube_to_sphere(basis.n + basis.u * cu + basis.v * cv).normalize()
    };
    let origin = center_dir * (p.radius + hf.height(center_dir, p.level) * p.height_scale);

    let mut positions = vec![[0.0f32; 3]; vcount];
    let mut normals = vec![[0.0f32; 3]; vcount];
    let mut colors = vec![[0.0f32; 3]; vcount];
    let mut material = vec![0.0f32; vcount];
    let mut wetness = vec![0.0f32; vcount];
    let mut volcanism = vec![0.0f32; vcount];
    let mut elevation = vec![0.0f32; vcount];
    let mut plate = vec![[0.0f32; 3]; vcount];
    let mut horizon = vec![[0.0f32; 4]; vcount];
    let mut boundary = vec![false; vcount];
    let mut dir_cache = vec![DVec3::ZERO; vcount];
    let mut h_cache = vec![0.0f64; vcount];

    // Coarse per-patch horizon grid (cost decoupled from the vertex count); each
    // vertex bilinearly samples it below. Cap the grid res so small test meshes
    // stay fast (the real bake at res 1024 uses HORIZON_GRID_CAP).
    let hg = (res as usize).min(HORIZON_GRID_CAP).max(8);
    let horizon_grid = build_horizon_grid(p, hg, hf);

    // Pass 1: positions + caches.
    for gj in 0..grid {
        for gi in 0..grid {
            let vi = gj * grid + gi;
            let (dir, world, h) = eval_vertex(p, gi as i64, gj as i64, hf);
            let rel = world - origin;
            positions[vi] = [rel.x as f32, rel.y as f32, rel.z as f32];
            dir_cache[vi] = dir;
            h_cache[vi] = h;
            boundary[vi] = gi == 0 || gi == grid - 1 || gj == 0 || gj == grid - 1;
        }
    }

    // Pass 2: face-independent analytic normals (continuous across cube edges →
    // no shading seam) + a simple height/slope color.
    for gj in 0..grid {
        for gi in 0..grid {
            let vi = gj * grid + gi;
            let dir = dir_cache[vi];

            let n = surface_normal(dir, p, hf);
            normals[vi] = [n.x as f32, n.y as f32, n.z as f32];

            // Biome coloring — identical to enki-planet's mesher, so the Nanite
            // view's albedo matches the quadtree terrain exactly.
            let slope = (1.0 - n.dot(dir)).clamp(0.0, 1.0) as f32;
            let (temp, moisture) = hf.climate(dir, h_cache[vi]);
            colors[vi] = enki_planet::coloring::biome_color(temp, moisture, h_cache[vi] as f32, slope);

            // Debug-view scalars, sampled at the same `dir` as color (default hf → 0).
            material[vi] = hf.material(dir);
            wetness[vi] = hf.wetness(dir);
            volcanism[vi] = hf.volcanism(dir);
            elevation[vi] = h_cache[vi] as f32;
            plate[vi] = hf.plate_color(dir);
            horizon[vi] = sample_horizon(&horizon_grid, hg, res, gi, gj);
        }
    }

    // Interior grid triangles, CCW viewed from outside (matches mesher winding).
    let mut indices = Vec::with_capacity((res * res * 6) as usize);
    for gj in 0..res as usize {
        for gi in 0..res as usize {
            let a = (gj * grid + gi) as u32;
            let b = a + 1;
            let c = a + grid as u32;
            let d = c + 1;
            indices.extend_from_slice(&[a, c, b, b, c, d]);
        }
    }

    PatchMesh { positions, normals, colors, material, wetness, volcanism, elevation, plate, horizon, indices, boundary, origin }
}

#[cfg(test)]
mod tests {
    use super::*;
    use enki_planet::height::HeightField;

    struct SineHf;
    impl HeightField for SineHf {
        fn height(&self, dir: DVec3, _l: u8) -> f64 {
            0.01 * (dir.x * 7.0 + dir.y * 5.0 + dir.z * 3.0).sin()
        }
    }

    /// Flat except a narrow Gaussian ridge centred at `bump_dir`.
    struct RidgeHf {
        bump_dir: DVec3,
        height: f64,
    }
    impl HeightField for RidgeHf {
        fn height(&self, dir: DVec3, _l: u8) -> f64 {
            let d = (dir.normalize() - self.bump_dir).length();
            self.height * (-d * d * 2000.0).exp()
        }
    }

    /// Frame-consistency: the horizon march must see a ridge placed in a given
    /// azimuth ONLY in that azimuth's slot — guards against a bake↔runtime axis
    /// swap/sign flip (the runtime mirrors `horizon_basis` + this sine convention).
    #[test]
    fn horizon_sees_ridge_only_in_its_azimuth() {
        use enki_planet::face_bases::horizon_basis;
        let p = PatchParams {
            face: 4, level: 0, ix: 0, iy: 0, resolution: 64,
            radius: 50_000.0, height_scale: 1_000.0,
        };
        let dir = DVec3::Z; // +Z face centre
        let (ta, tb) = horizon_basis(dir);
        // Ridge ~2.5 km away (arc 0.05 rad) along +tan_a, peak ~500 m.
        let arc: f64 = 0.05;
        let bump_dir = (dir * arc.cos() + ta * arc.sin()).normalize();
        let hf = RidgeHf { bump_dir, height: 0.5 };

        let toward = march_horizon_sine(dir, ta, &p, &hf);
        let perp = march_horizon_sine(dir, tb, &p, &hf);
        let away = march_horizon_sine(dir, -ta, &p, &hf);
        assert!(toward > 0.05, "ridge azimuth should be occluded: sine={toward}");
        assert!(perp < 0.02, "perpendicular azimuth should be clear: sine={perp}");
        assert!(away < 0.02, "opposite azimuth should be clear: sine={away}");
    }

    /// Flat terrain → ~0 horizon everywhere (nothing self-shadows).
    #[test]
    fn flat_terrain_has_no_horizon() {
        struct FlatHf;
        impl HeightField for FlatHf {
            fn height(&self, _d: DVec3, _l: u8) -> f64 { 0.0 }
        }
        let m = tessellate_patch(&params(32), &FlatHf);
        for h in &m.horizon {
            for &s in h {
                assert!(s < 1e-3, "flat terrain produced horizon sine {s}");
            }
        }
    }

    fn params(res: u32) -> PatchParams {
        PatchParams {
            face: 0,
            level: 2,
            ix: 1,
            iy: 1,
            resolution: res,
            radius: 6_371_000.0,
            height_scale: 8848.0,
        }
    }

    #[test]
    fn vertex_and_index_counts() {
        let res = 32;
        let m = tessellate_patch(&params(res), &SineHf);
        let expect_v = ((res + 1) * (res + 1)) as usize;
        assert_eq!(m.positions.len(), expect_v);
        assert_eq!(m.normals.len(), expect_v);
        assert_eq!(m.colors.len(), expect_v);
        assert_eq!(m.boundary.len(), expect_v);
        assert_eq!(m.indices.len(), (res * res * 6) as usize);
    }

    #[test]
    fn boundary_is_perimeter() {
        let res = 16;
        let m = tessellate_patch(&params(res), &SineHf);
        let count = m.boundary.iter().filter(|&&b| b).count();
        // Perimeter of a (res+1)^2 grid = 4*res vertices.
        assert_eq!(count, (4 * res) as usize);
    }

    #[test]
    fn positions_finite_indices_in_range() {
        let m = tessellate_patch(&params(16), &SineHf);
        let n = m.positions.len() as u32;
        for p in &m.positions {
            assert!(p[0].is_finite() && p[1].is_finite() && p[2].is_finite());
        }
        for &i in &m.indices {
            assert!(i < n, "index {i} out of range (n={n})");
        }
    }

    #[test]
    fn normals_unit_and_outward() {
        let m = tessellate_patch(&params(16), &SineHf);
        for (i, (&p, &nrm)) in m.positions.iter().zip(m.normals.iter()).enumerate() {
            let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
            assert!((len - 1.0).abs() < 1e-4, "normal[{i}] length {len}");
            // Outward: dot(normal, world_pos) > 0 where world = pos + origin.
            let w = (
                p[0] as f64 + m.origin.x,
                p[1] as f64 + m.origin.y,
                p[2] as f64 + m.origin.z,
            );
            let dot = nrm[0] as f64 * w.0 + nrm[1] as f64 * w.1 + nrm[2] as f64 * w.2;
            assert!(dot > 0.0, "normal[{i}] not outward (dot={dot})");
        }
    }
}

//! Converts a quadtree patch description into a `ChunkMeshArrays`.

use crate::face_bases::{cube_to_sphere, FACE_BASES};
use crate::height::HeightField;
use minos_render::geometry::ChunkMeshArrays;
use glam::DVec3;

/// Parameters fully describing one chunk build job.
pub struct ChunkBuildParams {
    pub face: u8,
    pub level: u8,
    pub ix: u32,
    pub iy: u32,
    pub resolution: u32,
    pub radius: f64,
    pub height_scale: f64,
}

/// Map grid coordinate (gi, gj) — which may be outside [0, res] for ghost
/// vertices — to a unit sphere direction, world position, and raw height.
fn eval_vertex(
    p: &ChunkBuildParams,
    gi: i64,
    gj: i64,
    hf: &dyn HeightField,
) -> (DVec3, DVec3, f64) {
    let basis = &FACE_BASES[p.face as usize];
    let res = p.resolution as f64;
    let scale = 1.0 / (1u64 << p.level) as f64;

    let u = (p.ix as f64 + gi as f64 / res) * scale;
    let v = (p.iy as f64 + gj as f64 / res) * scale;
    let cu = u * 2.0 - 1.0;
    let cv = v * 2.0 - 1.0;

    let cube_pt = basis.n + basis.u * cu + basis.v * cv;
    let dir = cube_to_sphere(cube_pt).normalize();

    let h = hf.height(dir, p.level);
    let r = p.radius + h * p.height_scale;
    let world = dir * r;

    (dir, world, h)
}

/// Surface normal at sphere direction `dir`, from a **face-independent** tangent
/// basis built from `dir` alone — so adjacent cube faces compute the *identical*
/// normal at a shared edge (cross-face shading seam → 0).
///
// ponytail: reuse the Nanite dir-based normal so quadtree edges match. The old
// per-vertex face-local central difference used ghost vertices extrapolated in
// face-parameter space (`gi = -1` lands off the cube face), so each face invented
// a different edge normal (~3.3° cross-face step). This is a pure function of the
// world direction, hence continuous across every face boundary (matches
// minos-nanite tessellate::surface_normal).
fn surface_normal(dir: DVec3, p: &ChunkBuildParams, hf: &dyn HeightField) -> DVec3 {
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

/// Tessellate one quadtree patch into vertex/index data.
pub fn build_chunk(p: &ChunkBuildParams, hf: &dyn HeightField) -> ChunkMeshArrays {
    let res = p.resolution;
    let grid_size = (res + 1) as usize;
    let grid_verts = grid_size * grid_size;
    let skirt_verts_per_edge = (res + 1) as usize;
    let skirt_verts = 4 * skirt_verts_per_edge;
    let total_verts = grid_verts + skirt_verts;

    let grid_index_count = (res * res * 6) as usize;
    let skirt_index_count = (4 * res * 6) as usize;
    let total_indices = grid_index_count + skirt_index_count;

    let mut positions = vec![[0.0f32; 3]; total_verts];
    let mut normals = vec![[0.0f32; 3]; total_verts];
    let mut colors = vec![[0.0f32; 3]; total_verts];
    // Per-vertex tectonic plate tint for the "Plate" debug view (parallel to colors).
    let mut plate = vec![[0.0f32; 3]; total_verts];
    let mut indices = vec![0u32; total_indices];

    // -- Origin: chunk center in world f64 -----------------------------------------
    // W3.0 camera-relative precision convention: origin = chunk centre so that
    //   (origin − camera_pos) is computed in f64 by ChunkPush::camera_relative,
    //   and every vertex position stored here is (world_f64 − origin) as f32.
    // At surface scale (~50 km) world_f64 and origin are both ~50 km; the
    // subtraction cancels to sub-metre magnitude — well within f32 precision.
    // The shader then adds two small f32 numbers instead of differencing two large ones.
    let center_dir = {
        let basis = &FACE_BASES[p.face as usize];
        let scale = 1.0 / (1u64 << p.level) as f64;
        let u = (p.ix as f64 + 0.5) * scale;
        let v = (p.iy as f64 + 0.5) * scale;
        let cu = u * 2.0 - 1.0;
        let cv = v * 2.0 - 1.0;
        let cube_pt = basis.n + basis.u * cu + basis.v * cv;
        cube_to_sphere(cube_pt).normalize()
    };
    let h_center = hf.height(center_dir, p.level);
    let origin: DVec3 = center_dir * (p.radius + h_center * p.height_scale);

    // -- Pass 1: positions + caches -------------------------------------------
    // dir_cache stores unit sphere directions for the dir-based normal,
    // outward-normal check, and coloring slope computation; h_cache the heights.
    let mut dir_cache = vec![DVec3::ZERO; grid_verts];
    let mut h_cache = vec![0.0f64; grid_verts];
    let mut min_h = f64::INFINITY;
    let mut max_h = f64::NEG_INFINITY;

    for gj in 0..grid_size {
        for gi in 0..grid_size {
            let vi = gj * grid_size + gi;
            let (dir, world, h) = eval_vertex(p, gi as i64, gj as i64, hf);

            // Origin-relative: (world_f64 - origin_f64) cast to f32.
            // Magnitude is O(node_size) — small enough for f32 precision.
            let rel = world - origin;
            positions[vi] = [rel.x as f32, rel.y as f32, rel.z as f32];
            dir_cache[vi] = dir;
            h_cache[vi] = h;

            if h < min_h { min_h = h; }
            if h > max_h { max_h = h; }
        }
    }

    // -- Skirt depth ----------------------------------------------------------
    // Size the skirt to the chunk's own height relief so flat plains don't
    // grow kilometre-deep flanges. See TS reference for the full rationale.
    let relief_world = (max_h - min_h) * p.height_scale;
    let skirt_depth = (1.5 * relief_world + 2.0).clamp(2.0, p.height_scale);

    // -- Pass 2: normals + colors ---------------------------------------------
    // ponytail: dir-based analytic normal (see `surface_normal`) instead of
    // face-local central differences. The normal is a pure function of `dir`, so
    // adjacent cube faces produce the IDENTICAL normal at a shared edge — no
    // cross-face shading crease — and no ghost-vertex heightfield evals on borders.
    for gj in 0..grid_size {
        for gi in 0..grid_size {
            let vi = gj * grid_size + gi;
            let dir = dir_cache[vi];
            let h = h_cache[vi];

            let normal = surface_normal(dir, p, hf);
            normals[vi] = [normal.x as f32, normal.y as f32, normal.z as f32];

            let slope = (1.0 - normal.dot(dir)) as f32;
            let (temp, moisture) = hf.climate(dir, h);
            colors[vi] = crate::coloring::biome_color(temp, moisture, h as f32, slope);
            plate[vi] = hf.plate_color(dir);
        }
    }

    // -- Interior grid indices ------------------------------------------------
    let mut ii = 0usize;
    for gj in 0..res as usize {
        for gi in 0..res as usize {
            let a = (gj * grid_size + gi) as u32;
            let b = a + 1;
            let c = a + grid_size as u32;
            let d = c + 1;
            // Two CCW triangles viewed from outside the sphere: (a,c,b) and (b,c,d)
            indices[ii]     = a;
            indices[ii + 1] = c;
            indices[ii + 2] = b;
            indices[ii + 3] = b;
            indices[ii + 4] = c;
            indices[ii + 5] = d;
            ii += 6;
        }
    }

    // -- Skirt vertices + indices ---------------------------------------------
    // For each of 4 border edges, emit (res+1) skirt vertices — the border
    // vertex pulled toward the planet center by skirt_depth. Skirt quads
    // seal the LOD crack at T-junctions between adjacent-LOD tiles.
    //
    // Vertex layout (relative to skirt_base = grid_verts):
    //   edge 0 (bottom, gj=0):  offsets  0..(res)
    //   edge 1 (right, gi=res): offsets  (res+1)..(2res+1)
    //   edge 2 (top, gj=res):   offsets  (2res+2)..(3res+2)  — gi traversed right→left
    //   edge 3 (left, gi=0):    offsets  (3res+3)..(4res+3)  — gj traversed top→bottom

    let skirt_base = grid_verts;
    let mut sv = skirt_base;

    // Emit one skirt vertex: pull border vertex toward planet center.
    // Copies normal and color from the corresponding border vertex.
    let emit_skirt_vert = |sv: &mut usize,
                                border_vi: usize,
                                positions: &mut Vec<[f32; 3]>,
                                normals: &mut Vec<[f32; 3]>,
                                colors: &mut Vec<[f32; 3]>,
                                plate: &mut Vec<[f32; 3]>| {
        // positions[border_vi] is origin-relative; reconstruct world f64 to compute skirt pull.
        let bx = positions[border_vi][0] as f64 + origin.x;
        let by = positions[border_vi][1] as f64 + origin.y;
        let bz = positions[border_vi][2] as f64 + origin.z;
        let len = (bx * bx + by * by + bz * bz).sqrt();
        let pull_scale = (len - skirt_depth) / len;

        // Store skirt vertex also origin-relative.
        positions[*sv] = [
            (bx * pull_scale - origin.x) as f32,
            (by * pull_scale - origin.y) as f32,
            (bz * pull_scale - origin.z) as f32,
        ];
        normals[*sv] = normals[border_vi];
        colors[*sv] = colors[border_vi];
        plate[*sv] = plate[border_vi];
        *sv += 1;
    };

    // Emit one skirt quad (2 triangles).
    // Winding: (border0, border1, skirt0) and (border1, skirt1, skirt0).
    let mut emit_skirt_quad = |ii: &mut usize,
                                border0: u32,
                                border1: u32,
                                skirt0: u32,
                                skirt1: u32| {
        indices[*ii]     = border0;
        indices[*ii + 1] = border1;
        indices[*ii + 2] = skirt0;
        indices[*ii + 3] = border1;
        indices[*ii + 4] = skirt1;
        indices[*ii + 5] = skirt0;
        *ii += 6;
    };

    // Edge 0: bottom row (gj=0), gi = 0..=res (left to right)
    let e0_start = skirt_base as u32;
    for gi in 0..=res as usize {
        let border_vi = gi; // gj=0
        emit_skirt_vert(&mut sv, border_vi, &mut positions, &mut normals, &mut colors, &mut plate);
    }
    for qi in 0..res as usize {
        let border0 = qi as u32;
        let border1 = border0 + 1;
        let skirt0 = e0_start + qi as u32;
        let skirt1 = skirt0 + 1;
        emit_skirt_quad(&mut ii, border0, border1, skirt0, skirt1);
    }

    // Edge 1: right column (gi=res), gj = 0..=res (bottom to top)
    let e1_start = e0_start + (res + 1);
    for gj in 0..=res as usize {
        let border_vi = gj * grid_size + res as usize;
        emit_skirt_vert(&mut sv, border_vi, &mut positions, &mut normals, &mut colors, &mut plate);
    }
    for qi in 0..res as usize {
        let border0 = (qi * grid_size + res as usize) as u32;
        let border1 = ((qi + 1) * grid_size + res as usize) as u32;
        let skirt0 = e1_start + qi as u32;
        let skirt1 = skirt0 + 1;
        emit_skirt_quad(&mut ii, border0, border1, skirt0, skirt1);
    }

    // Edge 2: top row (gj=res), gi = res..=0 (right to left, reversed)
    let e2_start = e1_start + (res + 1);
    for gi in (0..=res as usize).rev() {
        let border_vi = res as usize * grid_size + gi;
        emit_skirt_vert(&mut sv, border_vi, &mut positions, &mut normals, &mut colors, &mut plate);
    }
    for qi in 0..res as usize {
        let gi0 = res as usize - qi;
        let gi1 = res as usize - qi - 1;
        let border0 = (res as usize * grid_size + gi0) as u32;
        let border1 = (res as usize * grid_size + gi1) as u32;
        let skirt0 = e2_start + qi as u32;
        let skirt1 = skirt0 + 1;
        emit_skirt_quad(&mut ii, border0, border1, skirt0, skirt1);
    }

    // Edge 3: left column (gi=0), gj = res..=0 (top to bottom, reversed)
    let e3_start = e2_start + (res + 1);
    for gj in (0..=res as usize).rev() {
        let border_vi = gj * grid_size; // gi=0
        emit_skirt_vert(&mut sv, border_vi, &mut positions, &mut normals, &mut colors, &mut plate);
    }
    for qi in 0..res as usize {
        let gj0 = res as usize - qi;
        let gj1 = res as usize - qi - 1;
        let border0 = (gj0 * grid_size) as u32;
        let border1 = (gj1 * grid_size) as u32;
        let skirt0 = e3_start + qi as u32;
        let skirt1 = skirt0 + 1;
        emit_skirt_quad(&mut ii, border0, border1, skirt0, skirt1);
    }

    debug_assert_eq!(sv, total_verts, "skirt vertex count mismatch");
    debug_assert_eq!(ii, total_indices, "index count mismatch");

    ChunkMeshArrays {
        positions,
        normals,
        colors,
        plate_colors: Some(plate),
        indices,
        origin,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::height::HeightField;

    struct SineHf;
    impl HeightField for SineHf {
        fn height(&self, dir: DVec3, _level: u8) -> f64 {
            0.01 * (dir.x * 7.0 + dir.y * 5.0 + dir.z * 3.0).sin()
        }
    }

    /// Steeper, higher-frequency height (still a pure function of `dir`, so it is
    /// continuous across cube edges). The face-local estimator's seam grows with
    /// local relief steepness, so this gives the cross-face-normal regression a
    /// clear margin over the f32 normal-storage floor.
    struct SteepHf;
    impl HeightField for SteepHf {
        fn height(&self, dir: DVec3, _level: u8) -> f64 {
            0.05 * (dir.x * 23.0).sin() * (dir.y * 19.0).cos() + 0.03 * (dir.z * 31.0).sin()
        }
    }

    fn make_params(res: u32) -> ChunkBuildParams {
        ChunkBuildParams {
            face: 0,
            level: 1,
            ix: 0,
            iy: 0,
            resolution: res,
            radius: 6_371_000.0,
            height_scale: 8848.0,
        }
    }

    #[test]
    fn vertex_and_index_counts() {
        let res = 32u32;
        let p = make_params(res);
        let mesh = build_chunk(&p, &SineHf);
        let expected_verts = ((res + 1) * (res + 1) + 4 * (res + 1)) as usize;
        let expected_indices = (res * res * 6 + 4 * res * 6) as usize;
        assert_eq!(
            mesh.positions.len(),
            expected_verts,
            "vertex count: expected {expected_verts}"
        );
        assert_eq!(
            mesh.indices.len(),
            expected_indices,
            "index count: expected {expected_indices}"
        );
    }

    #[test]
    fn interior_normals_unit_and_outward() {
        let res = 16u32;
        let p = make_params(res);
        let mesh = build_chunk(&p, &SineHf);
        let grid_verts = ((res + 1) * (res + 1)) as usize;
        for (i, (&pos, &nrm)) in mesh.positions[..grid_verts]
            .iter()
            .zip(mesh.normals[..grid_verts].iter())
            .enumerate()
        {
            // Unit length
            let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "normal[{i}] length={len}"
            );
            // Outward: dot(normal, world_pos+origin) > 0
            let world_x = pos[0] as f64 + mesh.origin.x;
            let world_y = pos[1] as f64 + mesh.origin.y;
            let world_z = pos[2] as f64 + mesh.origin.z;
            let dot = nrm[0] as f64 * world_x
                + nrm[1] as f64 * world_y
                + nrm[2] as f64 * world_z;
            assert!(dot > 0.0, "normal[{i}] not outward: dot={dot}");
        }
    }

    #[test]
    fn all_positions_and_origin_finite() {
        let p = make_params(16);
        let mesh = build_chunk(&p, &SineHf);
        assert!(
            mesh.origin.x.is_finite()
                && mesh.origin.y.is_finite()
                && mesh.origin.z.is_finite()
        );
        for (i, &pos) in mesh.positions.iter().enumerate() {
            assert!(
                pos[0].is_finite() && pos[1].is_finite() && pos[2].is_finite(),
                "pos[{i}] not finite: {pos:?}"
            );
        }
    }

    #[test]
    fn colors_in_range() {
        let p = make_params(16);
        let mesh = build_chunk(&p, &SineHf);
        for (i, &col) in mesh.colors.iter().enumerate() {
            assert!(col[0] >= 0.0 && col[0] <= 1.0, "color[{i}].r out of range");
            assert!(col[1] >= 0.0 && col[1] <= 1.0, "color[{i}].g out of range");
            assert!(col[2] >= 0.0 && col[2] <= 1.0, "color[{i}].b out of range");
        }
    }

    #[test]
    fn adjacent_chunks_share_border_world_positions() {
        // Two horizontally adjacent chunks (same face, level, ix=0 and ix=1)
        // must evaluate the same heightfield at their shared border (u,v) —
        // so the reconstructed world positions agree within f32 quantization error.
        //
        // Convention (W3.0 camera-relative precision):
        //   Each chunk stores positions as (world_f64 − chunk_origin_f64) as f32.
        //   Reconstructing world = origin_f64 + rel_f32 as f64 introduces f32
        //   truncation O(rel_magnitude × 2^-24) ≈ O(node_size × 1e-7). We assert
        //   agreement within node_size × 1e-6 to absorb this and avoid brittle
        //   absolute tolerances that break at different LODs or planet radii.
        let res = 8u32;
        let grid_size = (res + 1) as usize;
        let level = 1u8;
        let radius = 6_371_000.0_f64;

        let pa = ChunkBuildParams {
            face: 0,
            level,
            ix: 0,
            iy: 0,
            resolution: res,
            radius,
            height_scale: 8848.0,
        };
        let pb = ChunkBuildParams {
            face: 0,
            level,
            ix: 1,
            iy: 0,
            resolution: res,
            radius,
            height_scale: 8848.0,
        };

        let ma = build_chunk(&pa, &SineHf);
        let mb = build_chunk(&pb, &SineHf);

        // node_size ≈ arc length of one tile side at this level.
        // At level L a face has 2^L tiles; tile arc ≈ radius * π/2 / 2^L.
        let node_size = radius * std::f64::consts::FRAC_PI_2 / (1u64 << level) as f64;
        let tol = node_size * 1e-6;

        for gj in 0..=res as usize {
            // Chunk A right edge: vertex index = gj*grid_size + res
            let ia = gj * grid_size + res as usize;
            let wa = [
                ma.positions[ia][0] as f64 + ma.origin.x,
                ma.positions[ia][1] as f64 + ma.origin.y,
                ma.positions[ia][2] as f64 + ma.origin.z,
            ];
            // Chunk B left edge: vertex index = gj*grid_size + 0
            let ib = gj * grid_size;
            let wb = [
                mb.positions[ib][0] as f64 + mb.origin.x,
                mb.positions[ib][1] as f64 + mb.origin.y,
                mb.positions[ib][2] as f64 + mb.origin.z,
            ];
            let dist = ((wa[0] - wb[0]).powi(2)
                + (wa[1] - wb[1]).powi(2)
                + (wa[2] - wb[2]).powi(2))
            .sqrt();
            assert!(
                dist < tol,
                "border mismatch at gj={gj}: dist={dist:.6} (tol={tol:.6}) wa={wa:?} wb={wb:?}"
            );
        }
    }

    /// Cross-face SHADING (normal) continuity along a shared cube edge.
    ///
    /// Regression guard for the quadtree mesher's cube-seam crease: before the
    /// dir-based-normal fix, each face estimated the border normal from
    /// face-local ghost vertices (`gi=-1` lands off the cube face), so the SAME
    /// physical edge vertex was shaded two different ways depending on which face
    /// owned it — a measured cross-face normal step up to ~3.33° (mean ~0.23°).
    /// The dir-based `surface_normal` is a pure function of the world direction,
    /// so both faces compute the IDENTICAL normal at a shared edge.
    ///
    /// We build two whole-face chunks (face 0 = +X, face 4 = +Z), which meet at
    /// the cube edge x=z (|x|=|z| largest). For each face-0 border vertex we find
    /// the coincident face-4 border vertex (vertex coincidence is exact — 0 m),
    /// then assert the angle between their stored normals is well under the
    /// pre-fix value. `SteepHf` is a pure function of `dir`, hence continuous
    /// across the edge, so any residual angle is purely the estimator seam.
    #[test]
    fn cross_face_normals_continuous_at_cube_edge() {
        let res = 64u32;
        let radius = 50_000.0_f64;
        let height_scale = 1200.0_f64;
        let mk = |face: u8| ChunkBuildParams {
            face,
            level: 0,
            ix: 0,
            iy: 0,
            resolution: res,
            radius,
            height_scale,
        };
        let ma = build_chunk(&mk(0), &SteepHf); // +X
        let mb = build_chunk(&mk(4), &SteepHf); // +Z
        let grid = (res + 1) as usize;
        let grid_verts = grid * grid;

        // Reconstruct absolute world position of grid vertex `vi` in mesh `m`.
        let world = |m: &minos_render::geometry::ChunkMeshArrays, vi: usize| -> DVec3 {
            DVec3::new(
                m.positions[vi][0] as f64 + m.origin.x,
                m.positions[vi][1] as f64 + m.origin.y,
                m.positions[vi][2] as f64 + m.origin.z,
            )
        };
        let normal = |m: &minos_render::geometry::ChunkMeshArrays, vi: usize| -> DVec3 {
            DVec3::new(
                m.normals[vi][0] as f64,
                m.normals[vi][1] as f64,
                m.normals[vi][2] as f64,
            )
        };

        // Collect face-0 perimeter vertices, then match each to the nearest
        // face-4 perimeter vertex. Pairs that actually coincide (≈0 m) are the
        // shared cube-edge vertices; we assert normal continuity on those.
        let is_border = |vi: usize| {
            let gi = vi % grid;
            let gj = vi / grid;
            gi == 0 || gi == grid - 1 || gj == 0 || gj == grid - 1
        };
        let b_border: Vec<usize> = (0..grid_verts).filter(|&vi| is_border(vi)).collect();

        let mut matched = 0usize;
        let mut max_deg = 0.0_f64;
        let mut sum_deg = 0.0_f64;
        for vi_a in (0..grid_verts).filter(|&vi| is_border(vi)) {
            let wa = world(&ma, vi_a);
            // nearest face-4 border vertex
            let mut best = f64::INFINITY;
            let mut best_vi = b_border[0];
            for &vi_b in &b_border {
                let d = (world(&mb, vi_b) - wa).length();
                if d < best {
                    best = d;
                    best_vi = vi_b;
                }
            }
            // Only treat as a shared edge vertex if it truly coincides.
            if best > radius * 1e-6 {
                continue;
            }
            matched += 1;
            let na = normal(&ma, vi_a);
            let nb = normal(&mb, best_vi);
            let ang = na.dot(nb).clamp(-1.0, 1.0).acos().to_degrees();
            max_deg = max_deg.max(ang);
            sum_deg += ang;
        }

        assert!(matched >= res as usize, "too few shared-edge vertices matched: {matched}");
        let mean_deg = sum_deg / matched as f64;
        // Measured on this exact SteepHf/res=64 case: the OLD face-local estimator
        // seams at max ~0.097° (mean ~0.060°); the NEW dir-based normal computes
        // the IDENTICAL f64 normal on both faces, so the only residual is the f32
        // normal-storage floor (max ~0.020°, mean ~0.005° — two coincident unit
        // vectors rounded to f32 independently). (On steep tectonic relief at
        // res=1024 the pre-fix seam reached ~3.33°.) 0.05° sits cleanly above the
        // f32 floor and below the pre-fix seam — real teeth, not brittle.
        assert!(
            max_deg < 0.05,
            "cross-face normal seam: max={max_deg:.4}° mean={mean_deg:.4}° over {matched} edge verts (expected at f32 floor ~0.02°)"
        );
    }
}

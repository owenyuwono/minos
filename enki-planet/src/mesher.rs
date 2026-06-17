//! Converts a quadtree patch description into a `ChunkMeshArrays`.

use crate::face_bases::{cube_to_sphere, FACE_BASES};
use crate::height::HeightField;
use enki_render::geometry::ChunkMeshArrays;
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
    // world_cache stores f64 absolute world positions for neighbour lookups in
    // Pass 2 (normal computation). dir_cache stores unit sphere directions for
    // the outward-normal check and coloring slope computation.
    let mut world_cache = vec![DVec3::ZERO; grid_verts];
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
            world_cache[vi] = world;
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

    // -- Ghost position helper ------------------------------------------------
    // Returns the absolute world position for an out-of-bounds grid index
    // (gi or gj outside [0, res]). Uses the same continuous heightfield so
    // the value matches exactly what the adjacent tile stores at its real
    // border vertex — eliminating same-level seams.
    let ghost_world = |gi: i64, gj: i64| -> DVec3 {
        let (_, world, _) = eval_vertex(p, gi, gj, hf);
        world
    };

    // -- Pass 2: normals + colors ---------------------------------------------
    for gj in 0..grid_size {
        for gi in 0..grid_size {
            let vi = gj * grid_size + gi;
            let dir = dir_cache[vi];
            let h = h_cache[vi];

            // Fetch neighbour world positions (absolute f64).
            // In-bounds: read from world_cache (0 extra heightfield evals).
            // Out-of-bounds: ghost vertex via one extra heightfield eval.
            let world_l = if gi > 0 {
                world_cache[gj * grid_size + gi - 1]
            } else {
                ghost_world(gi as i64 - 1, gj as i64)
            };
            let world_r = if gi < grid_size - 1 {
                world_cache[gj * grid_size + gi + 1]
            } else {
                ghost_world(gi as i64 + 1, gj as i64)
            };
            let world_d = if gj > 0 {
                world_cache[(gj - 1) * grid_size + gi]
            } else {
                ghost_world(gi as i64, gj as i64 - 1)
            };
            let world_u = if gj < grid_size - 1 {
                world_cache[(gj + 1) * grid_size + gi]
            } else {
                ghost_world(gi as i64, gj as i64 + 1)
            };

            // Central-difference normal: cross(∂pos/∂gi, ∂pos/∂gj).
            // Origin offsets cancel in the subtraction so absolute world
            // positions work identically to origin-relative ones here.
            let tan_u = world_r - world_l;
            let tan_v = world_u - world_d;

            let mut normal = tan_u.cross(tan_v).normalize();
            if normal.dot(dir) < 0.0 {
                normal = -normal;
            }

            normals[vi] = [normal.x as f32, normal.y as f32, normal.z as f32];

            let slope = (1.0 - normal.dot(dir)) as f32;
            let (temp, moisture) = hf.climate(dir, h);
            colors[vi] = crate::coloring::biome_color(temp, moisture, h as f32, slope);
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
                                colors: &mut Vec<[f32; 3]>| {
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
        emit_skirt_vert(&mut sv, border_vi, &mut positions, &mut normals, &mut colors);
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
        emit_skirt_vert(&mut sv, border_vi, &mut positions, &mut normals, &mut colors);
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
        emit_skirt_vert(&mut sv, border_vi, &mut positions, &mut normals, &mut colors);
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
        emit_skirt_vert(&mut sv, border_vi, &mut positions, &mut normals, &mut colors);
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
        plate_colors: None,
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
}

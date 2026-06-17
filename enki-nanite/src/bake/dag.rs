//! Cluster-DAG construction — the crack-free correctness core of N0.
//!
//! Bottom-up loop, starting from LOD0 base clusters ([`build_base_clusters`]):
//! 1. **Group** the level's clusters (~[`TARGET_GROUP_SIZE`]) via METIS k-way on
//!    the cluster adjacency graph (edge weight = shared boundary edges).
//! 2. **Merge + lock**: weld each group's clusters into one mesh; lock every
//!    vertex on a *boundary edge* of that merged mesh (an edge used by exactly one
//!    triangle). This locks both the patch's outer rim and inter-group seams, so
//!    neighbouring groups stay watertight across the LOD transition.
//! 3. **Simplify** the merged mesh to ~50% triangles with the locks held.
//! 4. **Error + bounds**: group error = `simplify_error·scale + max(child error)`
//!    (monotonic); group sphere = a sphere enclosing all child spheres (nested).
//! 5. **Re-split** the simplified mesh into parent clusters (level+1); back-fill
//!    each child's `parent_error`/`parent_bounds` to the group's.
//! Recurse until ≤1 cluster, no further reduction, or [`MAX_LEVELS`].
//!
//! Invariants this guarantees (checked in tests): `parent_error >= self_error`
//! for every cluster; parent bounds enclose child bounds; locked patch-boundary
//! vertices survive bit-identical to the coarsest level.

use std::collections::{BTreeMap, HashMap};

use glam::Vec3;

use crate::cluster::{BoundingSphere, Cluster, ClusterAsset, ClusterVertex, NormalCone, ROOT_PARENT_ERROR};
use enki_planet::height::HeightField;

use super::cluster_build::{build_base_clusters, MAX_CLUSTER_TRIS, MAX_CLUSTER_VERTS};
use super::tessellate::{tessellate_patch, PatchParams};

/// Target clusters per group (Nanite cites 8–32; 8 halves cluster count/level).
pub const TARGET_GROUP_SIZE: usize = 8;
/// Hard cap on DAG levels (loop backstop; convergence normally stops first).
pub const MAX_LEVELS: usize = 25;

/// Bit-exact position key for welding duplicated cluster vertices.
fn pos_key(p: [f32; 3]) -> [u32; 3] {
    [p[0].to_bits(), p[1].to_bits(), p[2].to_bits()]
}

// ── Bounding-sphere enclosure (conservative, not minimal) ────────────────────

fn merge_two(a: BoundingSphere, b: BoundingSphere) -> BoundingSphere {
    let delta = b.center - a.center;
    let d = delta.length();
    if d + b.radius <= a.radius {
        return a;
    }
    if d + a.radius <= b.radius {
        return b;
    }
    let new_r = (a.radius + b.radius + d) * 0.5;
    let dir = if d > 1e-6 { delta / d } else { Vec3::ZERO };
    BoundingSphere {
        center: a.center + dir * (new_r - a.radius),
        radius: new_r,
    }
}

/// Smallest-ish sphere enclosing all input spheres (iterative merge). Conservative
/// (encloses every input), which is all the nested-bounds invariant requires.
fn enclose_spheres(spheres: &[BoundingSphere]) -> BoundingSphere {
    let mut acc = spheres[0];
    for &s in &spheres[1..] {
        acc = merge_two(acc, s);
    }
    acc
}

// ── Group merge + boundary lock ──────────────────────────────────────────────

struct MergedMesh {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 3]>,
    indices: Vec<u32>,
    lock: Vec<bool>,
}

/// Weld a group's clusters into one indexed mesh and mark boundary-edge vertices
/// (used by exactly one triangle) as locked.
fn merge_group(clusters: &[Cluster], group: &[usize]) -> MergedMesh {
    let mut map: HashMap<[u32; 3], u32> = HashMap::new();
    let mut positions = Vec::new();
    let mut normals = Vec::new();
    let mut colors = Vec::new();
    let mut indices = Vec::new();

    for &ci in group {
        let c = &clusters[ci];
        let mut local = Vec::with_capacity(c.vertices.len());
        for v in &c.vertices {
            let id = *map.entry(pos_key(v.position)).or_insert_with(|| {
                let id = positions.len() as u32;
                positions.push(v.position);
                normals.push(v.normal);
                colors.push(v.color);
                id
            });
            local.push(id);
        }
        for t in &c.triangles {
            indices.push(local[t[0] as usize]);
            indices.push(local[t[1] as usize]);
            indices.push(local[t[2] as usize]);
        }
    }

    // Boundary edges = edges used by exactly one triangle in the merged mesh.
    let mut edge_count: HashMap<(u32, u32), u32> = HashMap::new();
    for tri in indices.chunks_exact(3) {
        for &(a, b) in &[(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            let e = if a < b { (a, b) } else { (b, a) };
            *edge_count.entry(e).or_insert(0) += 1;
        }
    }
    let mut lock = vec![false; positions.len()];
    for (&(a, b), &cnt) in &edge_count {
        if cnt == 1 {
            lock[a as usize] = true;
            lock[b as usize] = true;
        }
    }

    MergedMesh { positions, normals, colors, indices, lock }
}

// ── METIS grouping ───────────────────────────────────────────────────────────

/// Partition `clusters` into groups of ~`group_size` by shared-edge adjacency.
/// Deterministic: adjacency is accumulated symmetrically and emitted in sorted
/// order, and groups are returned ordered by partition id.
fn partition_groups(clusters: &[Cluster], group_size: usize) -> Vec<Vec<usize>> {
    let n = clusters.len();
    if n <= group_size {
        return vec![(0..n).collect()];
    }

    // Weld all level vertices to global ids so shared edges are detectable.
    let mut map: HashMap<[u32; 3], u32> = HashMap::new();
    let mut next_id = 0u32;
    let mut cluster_edges: Vec<Vec<(u32, u32)>> = Vec::with_capacity(n);
    for c in clusters {
        let mut local = Vec::with_capacity(c.vertices.len());
        for v in &c.vertices {
            let id = *map.entry(pos_key(v.position)).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            local.push(id);
        }
        let mut eset: std::collections::BTreeSet<(u32, u32)> = std::collections::BTreeSet::new();
        for t in &c.triangles {
            let vs = [local[t[0] as usize], local[t[1] as usize], local[t[2] as usize]];
            for &(a, b) in &[(vs[0], vs[1]), (vs[1], vs[2]), (vs[2], vs[0])] {
                eset.insert(if a < b { (a, b) } else { (b, a) });
            }
        }
        cluster_edges.push(eset.into_iter().collect());
    }

    let mut edge_clusters: HashMap<(u32, u32), Vec<u32>> = HashMap::new();
    for (ci, edges) in cluster_edges.iter().enumerate() {
        for &e in edges {
            edge_clusters.entry(e).or_default().push(ci as u32);
        }
    }

    let mut adj: Vec<BTreeMap<u32, i32>> = vec![BTreeMap::new(); n];
    for clist in edge_clusters.values() {
        for i in 0..clist.len() {
            for j in (i + 1)..clist.len() {
                let (a, b) = (clist[i] as usize, clist[j] as usize);
                *adj[a].entry(b as u32).or_insert(0) += 1;
                *adj[b].entry(a as u32).or_insert(0) += 1;
            }
        }
    }

    let chunk_fallback = || -> Vec<Vec<usize>> {
        let mut groups = Vec::new();
        let mut i = 0;
        while i < n {
            groups.push((i..(i + group_size).min(n)).collect());
            i += group_size;
        }
        groups
    };

    // CSR for METIS.
    let mut xadj: Vec<i32> = Vec::with_capacity(n + 1);
    let mut adjncy: Vec<i32> = Vec::new();
    let mut adjwgt: Vec<i32> = Vec::new();
    xadj.push(0);
    for a in &adj {
        for (&nb, &w) in a {
            adjncy.push(nb as i32);
            adjwgt.push(w);
        }
        xadj.push(adjncy.len() as i32);
    }
    if adjncy.is_empty() {
        return chunk_fallback();
    }

    let nparts = ((n + group_size - 1) / group_size).max(2) as i32;
    let mut part = vec![0i32; n];
    let res = match metis::Graph::new(1, nparts, &xadj, &adjncy) {
        Ok(g) => g.set_adjwgt(&adjwgt).part_kway(&mut part),
        Err(_) => return chunk_fallback(),
    };
    if res.is_err() {
        return chunk_fallback();
    }

    let mut buckets: BTreeMap<i32, Vec<usize>> = BTreeMap::new();
    for (ci, &p) in part.iter().enumerate() {
        buckets.entry(p).or_default().push(ci);
    }
    buckets.into_values().filter(|g| !g.is_empty()).collect()
}

// ── DAG loop ─────────────────────────────────────────────────────────────────

/// Build the full cluster DAG from LOD0 base clusters. Returns every cluster
/// across all levels with `parent_*`/`self_*` fields finalized.
pub fn build_dag(base: Vec<Cluster>) -> Vec<Cluster> {
    let mut all: Vec<Cluster> = Vec::new();
    let mut current = base;
    let mut lod: u8 = 0;

    for _ in 0..MAX_LEVELS {
        let n = current.len();
        if n <= 1 {
            break;
        }

        let groups = partition_groups(&current, TARGET_GROUP_SIZE);
        let mut next: Vec<Cluster> = Vec::new();
        let mut group_counter: u32 = 0;

        for group in &groups {
            let merged = merge_group(&current, group);
            if merged.indices.len() < 6 {
                group_counter += 1;
                continue; // too small to simplify meaningfully; children stay roots
            }

            let vbytes: &[u8] = bytemuck::cast_slice(merged.positions.as_slice());
            let adapter = meshopt::VertexDataAdapter::new(vbytes, 12, 0)
                .expect("merged vertex adapter");

            let target_index_count = (merged.indices.len() / 2).max(3);
            let mut rel_err = 0.0f32;
            let simplified = meshopt::simplify_with_locks(
                &merged.indices,
                &adapter,
                &merged.lock,
                target_index_count,
                1.0,
                meshopt::SimplifyOptions::empty(),
                Some(&mut rel_err),
            );
            let scale = meshopt::simplify_scale(&adapter);

            let max_child_err = group
                .iter()
                .map(|&ci| current[ci].self_error)
                .fold(0.0f32, f32::max);
            let group_error = rel_err * scale + max_child_err;

            let child_spheres: Vec<BoundingSphere> =
                group.iter().map(|&ci| current[ci].bounds).collect();
            let group_bounds = enclose_spheres(&child_spheres);

            // Re-split the simplified group into parent clusters.
            let mut parents: Vec<Cluster> = Vec::new();
            if simplified.len() >= 3 {
                let pmesh =
                    meshopt::build_meshlets(&simplified, &adapter, MAX_CLUSTER_VERTS, MAX_CLUSTER_TRIS, 0.0);
                for i in 0..pmesh.len() {
                    let m = pmesh.get(i);
                    let b = meshopt::compute_meshlet_bounds(m, &adapter);
                    let vertices: Vec<ClusterVertex> = m
                        .vertices
                        .iter()
                        .map(|&gv| {
                            let gv = gv as usize;
                            ClusterVertex {
                                position: merged.positions[gv],
                                normal: merged.normals[gv],
                                color: merged.colors[gv],
                            }
                        })
                        .collect();
                    let triangles: Vec<[u8; 3]> = m
                        .triangles
                        .chunks_exact(3)
                        .map(|c| [c[0], c[1], c[2]])
                        .collect();
                    parents.push(Cluster {
                        vertices,
                        triangles,
                        bounds: group_bounds,
                        normal_cone: NormalCone {
                            apex: Vec3::from_array(b.cone_apex),
                            axis: Vec3::from_array(b.cone_axis),
                            cutoff: b.cone_cutoff,
                        },
                        self_error: group_error,
                        parent_error: ROOT_PARENT_ERROR,
                        parent_bounds: group_bounds,
                        group: group_counter,
                        lod: lod + 1,
                    });
                }
            }

            // Only link children to a parent if one was actually produced.
            if !parents.is_empty() {
                for &ci in group {
                    current[ci].parent_error = group_error;
                    current[ci].parent_bounds = group_bounds;
                }
                next.append(&mut parents);
            }
            group_counter += 1;
        }

        let made_progress = !next.is_empty() && next.len() < n;
        all.append(&mut current); // children: parent links now finalized
        if !made_progress {
            current = next; // becomes the (possibly empty) root level
            break;
        }
        current = next;
        lod += 1;
    }

    all.append(&mut current); // top level: roots (parent_error = +inf)
    all
}

/// Full bake: tessellate the patch, build base clusters, build the DAG.
pub fn bake_patch(params: &PatchParams, hf: &dyn HeightField) -> ClusterAsset {
    let mesh = tessellate_patch(params, hf);
    let origin = mesh.origin;
    let clusters = build_dag(build_base_clusters(&mesh));
    ClusterAsset {
        clusters,
        patch_origin: origin,
        face: params.face,
        level: params.level,
        ix: params.ix,
        iy: params.iy,
    }
}

/// Bake the **whole planet** — all 6 cube faces at level 0 — into ONE resident
/// cluster asset, re-based to a common origin (planet centre, `DVec3::ZERO`).
///
/// Faces bake in parallel. At runtime the cluster-LOD adapts per-distance across
/// the entire globe (this is Nanite's job and *replaces* the quadtree LOD). Max
/// detail is capped by `resolution` because v1 keeps everything resident — finer-
/// than-bake detail needs streaming (N5).
///
/// Cube-face seam vertices coincide (the cube→sphere map is continuous across
/// edges) and are boundary-locked, so neighbouring faces stay crack-free.
pub fn bake_planet(
    hf: &dyn HeightField,
    radius: f64,
    height_scale: f64,
    resolution: u32,
) -> Vec<ClusterAsset> {
    // Bake the 6 faces in parallel. Each face is a node with its OWN origin — no
    // re-basing to the planet centre. The renderer applies a per-node camera-
    // relative translation (computed in f64), so f32 stays precise at any scale.
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..6u8)
            .map(|face| {
                s.spawn(move || {
                    bake_patch(
                        &PatchParams {
                            face,
                            level: 0,
                            ix: 0,
                            iy: 0,
                            resolution,
                            radius,
                            height_scale,
                        },
                        hf,
                    )
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().expect("face bake panicked")).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use enki_planet::simple_height::SimpleHeightField;
    use enki_planet::noise::Noise3D;
    use std::collections::HashSet;

    fn hf() -> SimpleHeightField {
        SimpleHeightField { noise: Noise3D::new(1337) }
    }

    fn params(res: u32) -> PatchParams {
        PatchParams {
            face: 2,
            level: 4,
            ix: 5,
            iy: 6,
            resolution: res,
            radius: 6_371_000.0,
            height_scale: 8_848.0,
        }
    }

    #[test]
    fn dag_converges_to_multiple_levels() {
        let asset = bake_patch(&params(64), &hf());
        assert!(asset.cluster_count() > 0);
        assert!(
            asset.lod_levels() >= 2,
            "DAG should build >=2 LOD levels, got {}",
            asset.lod_levels()
        );
        // Coarsest level must have fewer clusters than the finest.
        let max_lod = asset.clusters.iter().map(|c| c.lod).max().unwrap();
        let finest = asset.clusters.iter().filter(|c| c.lod == 0).count();
        let coarsest = asset.clusters.iter().filter(|c| c.lod == max_lod).count();
        assert!(coarsest < finest, "coarsest {coarsest} !< finest {finest}");
    }

    #[test]
    fn error_is_monotonic() {
        let asset = bake_patch(&params(64), &hf());
        for (i, c) in asset.clusters.iter().enumerate() {
            assert!(
                c.parent_error >= c.self_error,
                "cluster {i}: parent_error {} < self_error {} (non-monotonic)",
                c.parent_error,
                c.self_error
            );
        }
    }

    #[test]
    fn parent_bounds_enclose_self_bounds() {
        let asset = bake_patch(&params(64), &hf());
        for (i, c) in asset.clusters.iter().enumerate() {
            if c.is_root() {
                continue;
            }
            let d = (c.bounds.center - c.parent_bounds.center).length();
            let tol = c.parent_bounds.radius * 1e-3 + 1.0;
            assert!(
                d + c.bounds.radius <= c.parent_bounds.radius + tol,
                "cluster {i}: parent bounds do not enclose self (d={d}, self_r={}, parent_r={})",
                c.bounds.radius,
                c.parent_bounds.radius
            );
        }
    }

    #[test]
    fn patch_boundary_survives_to_coarsest_lod() {
        // Crack-free proxy: every locked patch-boundary vertex (LOD0) must still
        // exist bit-identical at the coarsest LOD — i.e. boundaries are never
        // collapsed, so adjacent patches/tiles stay watertight.
        let p = params(64);
        let mesh = tessellate_patch(&p, &hf());
        let boundary: HashSet<[u32; 3]> = mesh
            .positions
            .iter()
            .zip(mesh.boundary.iter())
            .filter(|(_, &b)| b)
            .map(|(pos, _)| pos_key(*pos))
            .collect();
        assert!(!boundary.is_empty());

        let asset = bake_patch(&p, &hf());
        let max_lod = asset.clusters.iter().map(|c| c.lod).max().unwrap();
        let top: HashSet<[u32; 3]> = asset
            .clusters
            .iter()
            .filter(|c| c.lod == max_lod)
            .flat_map(|c| c.vertices.iter().map(|v| pos_key(v.position)))
            .collect();

        let lost = boundary.iter().filter(|b| !top.contains(*b)).count();
        assert_eq!(lost, 0, "{lost} patch-boundary vertices lost by coarsest LOD (crack risk)");
    }

    /// On-demand diagnostic: print the DAG shape on real terrain.
    /// Run with: `cargo test -p enki-nanite report_dag_shape -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn report_dag_shape() {
        let asset = bake_patch(&params(128), &hf());
        let max_lod = asset.clusters.iter().map(|c| c.lod).max().unwrap();
        println!(
            "DAG: {} clusters, {} triangles, {} LOD levels",
            asset.cluster_count(),
            asset.total_triangles(),
            asset.lod_levels()
        );
        for l in 0..=max_lod {
            let cs: Vec<_> = asset.clusters.iter().filter(|c| c.lod == l).collect();
            let tris: usize = cs.iter().map(|c| c.triangles.len()).sum();
            let max_err = cs.iter().map(|c| c.self_error).fold(0.0f32, f32::max);
            println!(
                "  lod {l}: {:>4} clusters, {:>6} tris, max self_error {:.3} m",
                cs.len(),
                tris,
                max_err
            );
        }
    }

    #[test]
    fn bake_planet_covers_six_faces() {
        let hf = SimpleHeightField { noise: Noise3D::new(5) };
        let assets = bake_planet(&hf, 50_000.0, 1_200.0, 24);
        assert_eq!(assets.len(), 6, "one node per cube face");
        for a in &assets {
            assert!(a.cluster_count() > 0, "each face has clusters");
            assert!(a.lod_levels() >= 1);
            // Each face keeps its OWN origin near the planet surface (not centre).
            assert!(a.patch_origin.length() > 40_000.0, "face origin ~ planet radius");
            // Invariants hold per node.
            for c in &a.clusters {
                assert!(c.parent_error >= c.self_error, "monotonic error broken");
                assert!(c.bounds.center.is_finite() && c.bounds.radius > 0.0);
                // Positions are face-LOCAL (small), not re-based to planet scale.
            }
        }
    }

    #[test]
    fn bake_is_deterministic() {
        let a = bake_patch(&params(48), &hf());
        let b = bake_patch(&params(48), &hf());
        assert_eq!(a.cluster_count(), b.cluster_count());
        assert_eq!(a.total_triangles(), b.total_triangles());
        for (x, y) in a.clusters.iter().zip(b.clusters.iter()) {
            assert_eq!(x.self_error.to_bits(), y.self_error.to_bits());
            assert_eq!(x.lod, y.lod);
            assert_eq!(x.triangles.len(), y.triangles.len());
        }
    }
}

//! Base clustering — turn a tessellated patch (LOD0 mesh) into the finest level
//! of [`Cluster`]s via `meshopt::build_meshlets`.
//!
//! These LOD0 clusters have `self_error = 0` (they are the source geometry) and,
//! until the DAG loop links parents above them, `parent_error = +inf` (so they
//! are treated as roots / always-finest). The DAG builder (next N0 step) groups
//! these, simplifies, and fills in real parent links + monotonic errors.

use crate::cluster::{BoundingSphere, Cluster, ClusterVertex, ROOT_PARENT_ERROR};
use glam::Vec3;

use super::tessellate::PatchMesh;

/// Cluster size caps (Nanite convention; meshopt requires tris divisible by 4).
pub const MAX_CLUSTER_VERTS: usize = 64;
pub const MAX_CLUSTER_TRIS: usize = 128;

/// Partition `mesh`'s triangles into ≤128-tri / ≤64-vert clusters.
///
/// Every triangle of the input lands in exactly one cluster (meshlets are a
/// partition), so the returned clusters' triangle counts sum to the input count.
pub fn build_base_clusters(mesh: &PatchMesh) -> Vec<Cluster> {
    let vbytes: &[u8] = bytemuck::cast_slice(mesh.positions.as_slice());
    let adapter = meshopt::VertexDataAdapter::new(vbytes, 12, 0)
        .expect("vertex adapter: positions are [f32;3], stride 12, offset 0");

    // cone_weight = 0.0: bias meshlet formation toward spatial compactness (good
    // for LOD grouping) rather than normal-cone tightness (backface culling).
    let meshlets = meshopt::build_meshlets(
        &mesh.indices,
        &adapter,
        MAX_CLUSTER_VERTS,
        MAX_CLUSTER_TRIS,
        0.0,
    );

    let mut clusters = Vec::with_capacity(meshlets.len());
    for i in 0..meshlets.len() {
        let m = meshlets.get(i);
        let b = meshopt::compute_meshlet_bounds(m, &adapter);

        let vertices: Vec<ClusterVertex> = m
            .vertices
            .iter()
            .map(|&gv| {
                let gv = gv as usize;
                ClusterVertex {
                    position: mesh.positions[gv],
                    normal: mesh.normals[gv],
                    color: mesh.colors[gv],
                    material: mesh.material[gv],
                    wetness: mesh.wetness[gv],
                    volcanism: mesh.volcanism[gv],
                    elevation: mesh.elevation[gv],
                    plate: mesh.plate[gv],
                    horizon: mesh.horizon[gv],
                }
            })
            .collect();

        let triangles: Vec<[u8; 3]> = m
            .triangles
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect();

        let bounds = BoundingSphere {
            center: Vec3::from_array(b.center),
            radius: b.radius,
        };
        clusters.push(Cluster {
            vertices,
            triangles,
            bounds,
            self_error: 0.0,
            parent_error: ROOT_PARENT_ERROR,
            parent_bounds: bounds,
            group: 0,
            lod: 0,
        });
    }

    clusters
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::tessellate::{tessellate_patch, PatchParams};
    use enki_planet::height::HeightField;
    use glam::DVec3;

    struct SineHf;
    impl HeightField for SineHf {
        fn height(&self, dir: DVec3, _l: u8) -> f64 {
            0.01 * (dir.x * 7.0 + dir.y * 5.0 + dir.z * 3.0).sin()
        }
    }

    fn mesh(res: u32) -> super::PatchMesh {
        tessellate_patch(
            &PatchParams {
                face: 0,
                level: 2,
                ix: 1,
                iy: 1,
                resolution: res,
                radius: 6_371_000.0,
                height_scale: 8848.0,
            },
            &SineHf,
        )
    }

    #[test]
    fn base_clusters_partition_all_triangles() {
        let m = mesh(48);
        let clusters = build_base_clusters(&m);
        assert!(!clusters.is_empty(), "expected at least one cluster");

        let total: usize = clusters.iter().map(|c| c.triangles.len()).sum();
        assert_eq!(
            total,
            m.indices.len() / 3,
            "clusters must cover every triangle exactly once"
        );
    }

    #[test]
    fn base_clusters_respect_caps_and_validity() {
        let m = mesh(48);
        for c in &build_base_clusters(&m) {
            assert!(c.triangles.len() <= MAX_CLUSTER_TRIS, "tri cap exceeded");
            assert!(c.vertices.len() <= MAX_CLUSTER_VERTS, "vert cap exceeded");
            assert!(c.bounds.radius > 0.0, "degenerate bounds");
            assert_eq!(c.lod, 0);
            assert_eq!(c.self_error, 0.0);
            assert!(c.is_root(), "base clusters have no parent yet");
            for t in &c.triangles {
                for &idx in t {
                    assert!((idx as usize) < c.vertices.len(), "local index out of range");
                }
            }
        }
    }
}

//! Offline cluster-DAG baker (CPU-only, headless-testable).
//!
//! N0 builds this up in stages:
//! - [`tessellate`] — heightfield patch → watertight LOD0 mesh + boundary mask.
//! - (next) base clustering via `meshopt::build_meshlets`, METIS group partition,
//!   locked-boundary simplify, monotonic-error DAG.

pub mod cluster_build;
pub mod dag;
pub mod tessellate;

pub use cluster_build::build_base_clusters;
pub use dag::{bake_patch, bake_patch_timed, bake_planet, build_dag, BakeTimings};
pub use tessellate::{tessellate_patch, PatchMesh, PatchParams};

#[cfg(test)]
mod c_dep_smoke {
    //! N0.0 gating check: prove the C-backed deps (meshopt, METIS) build and link
    //! on this platform — independent of DAG logic. If these pass, the vendored
    //! `meshopt-sys` and `metis-sys` toolchain is good and the baker can proceed.
    use super::tessellate::*;
    use enki_planet::height::HeightField;
    use glam::DVec3;

    struct SineHf;
    impl HeightField for SineHf {
        fn height(&self, dir: DVec3, _l: u8) -> f64 {
            0.01 * (dir.x * 7.0 + dir.y * 5.0 + dir.z * 3.0).sin()
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
    fn meshopt_build_meshlets_links() {
        let mesh = tessellate_patch(&params(32), &SineHf);
        let vbytes: &[u8] = bytemuck::cast_slice(mesh.positions.as_slice());
        let adapter =
            meshopt::VertexDataAdapter::new(vbytes, 12, 0).expect("vertex adapter");
        let meshlets = meshopt::build_meshlets(&mesh.indices, &adapter, 64, 128, 0.0);
        assert!(meshlets.len() > 0, "expected at least one meshlet");
        for i in 0..meshlets.len() {
            let m = meshlets.get(i);
            assert!(m.triangles.len() / 3 <= 128, "meshlet exceeds 128 triangles");
            assert!(m.vertices.len() <= 64, "meshlet exceeds 64 vertices");
        }
    }

    #[test]
    fn metis_part_kway_links() {
        // 4-cycle graph (CSR), partition into 2 parts.
        let xadj: &[i32] = &[0, 2, 4, 6, 8];
        let adjncy: &[i32] = &[1, 3, 0, 2, 1, 3, 2, 0];
        let mut part = vec![0i32; 4];
        let _edgecut = metis::Graph::new(1, 2, xadj, adjncy)
            .expect("metis graph")
            .part_kway(&mut part)
            .expect("part_kway");
        assert!(part.iter().all(|&p| (0..2).contains(&p)), "invalid partition ids");
    }
}

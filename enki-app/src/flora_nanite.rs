// Flora branch mesh → enki-nanite ClusterAssets.
//
// Gated on BOTH `flora` and `nanite`: this is the only spot where the two
// otherwise-independent features compose. With either off, none of this
// compiles (deletable-flora + deletable-nanite contracts both hold).
//
// We feed flora's ARBITRARY tree-branch triangle mesh (build_branch_mesh →
// positions + indices) straight into the generic-mesh bake path:
//   build_base_clusters(&PatchMesh)  → LOD0 meshlets (≤64v / ≤128t)
//   build_dag(base)                  → full LOD DAG (meshopt simplify + METIS)
// This BYPASSES the heightfield `tessellate_patch` entry point — the public
// `bake_patch`/`bake_planet` are hard-wired to a HeightField, but
// `build_base_clusters`/`build_dag` only read `PatchMesh.{positions,normals,
// indices, per-vertex attrs}`, so a tree mesh is a drop-in.
//
// ponytail: one static tree → no streaming/residency/page churn. The single
// baked ClusterAsset is handed to NaniteRenderer::new as `initial`, which seeds
// it permanently resident; the viewer just runs cull + draw every frame.

use enki_flora::mesh::BranchMesh;
use enki_nanite::bake::{build_base_clusters, build_dag, PatchMesh};
use enki_nanite::cluster::ClusterAsset;
use glam::{DVec3, Quat, Vec3};

/// Bake a flora [`BranchMesh`] (flat `positions`/`normals` xyz + u32 `indices`)
/// into a single enki-nanite [`ClusterAsset`] — a full LOD cluster DAG.
///
/// `model_rot` is the SAME tree-local→world rotation `FloraView::model` applies in
/// the plain branch path (at the viewer's world origin it aligns local +Y to the
/// surface normal). The Nanite draw shader has NO model matrix — it only does
/// `view_proj * (pos + node_xlat)` — so we PRE-ROTATE the baked positions/normals
/// here. Then, with the asset's `patch_origin == tree origin`, Nanite's per-node
/// `node_xlat = origin - camera` reproduces flora's camera-relative translation
/// exactly, and the virtualized branch lands in the same place/orientation as the
/// plain branch. Pass [`Quat::IDENTITY`] to bake raw tree-local space.
///
/// The per-vertex Nanite "attribute lanes" (material/wetness/volcanism/
/// elevation/plate) are terrain-only debug channels; a tree has no meaning for
/// them, so they are zero-filled. The Triangle/Cluster/LOD debug views key on
/// cluster/triangle/lod ids, not these lanes, so they render correctly. `color`
/// is given a flat bark albedo for the Lit (mode 0) view.
pub fn bake_branch_mesh(bm: &BranchMesh, model_rot: Quat) -> ClusterAsset {
    let n = bm.positions.len() / 3;

    // Flat &[f32] xyz → Vec<[f32;3]>, pre-rotated by the model rotation so the
    // Nanite draw (no model matrix) matches the plain branch placement.
    // `chunks_exact(3)` drops any ragged tail (there is none — positions is
    // exactly 3*vertex_count, mesh.rs).
    let positions: Vec<[f32; 3]> = bm
        .positions
        .chunks_exact(3)
        .map(|c| (model_rot * Vec3::new(c[0], c[1], c[2])).to_array())
        .collect();

    // Normals are parallel to positions (3*vertex_count) — rotate them too (the
    // rotation is orthonormal, so no renormalize needed). If a degenerate/empty
    // mesh ever loses normals, fall back to +Y so the array shapes still match.
    let normals: Vec<[f32; 3]> = if bm.normals.len() == bm.positions.len() {
        bm.normals
            .chunks_exact(3)
            .map(|c| (model_rot * Vec3::new(c[0], c[1], c[2])).to_array())
            .collect()
    } else {
        vec![[0.0, 1.0, 0.0]; n]
    };

    let pm = PatchMesh {
        positions,
        normals,
        // Flat bark albedo (only the Lit view samples it).
        colors: vec![[0.45, 0.30, 0.18]; n],
        // Terrain-only debug lanes — irrelevant to a tree; zero-fill.
        material: vec![0.0; n],
        wetness: vec![0.0; n],
        volcanism: vec![0.0; n],
        elevation: vec![0.0; n],
        plate: vec![[0.0; 3]; n],
        indices: bm.indices.clone(),
        // build_base_clusters ignores `boundary`; the DAG re-derives lock edges
        // from the welded merged-group mesh, so an all-false mask is correct.
        boundary: vec![false; n],
        // Tree-local space — positions are already model-relative.
        origin: DVec3::ZERO,
    };

    let clusters = build_dag(build_base_clusters(&pm));

    ClusterAsset {
        clusters,
        patch_origin: DVec3::ZERO,
        // Provenance fields are terrain quadtree coords; meaningless for a tree.
        face: 0,
        level: 0,
        ix: 0,
        iy: 0,
    }
}

/// The tree-local→world model ROTATION the viewer applies to the branch at world
/// `origin` — IDENTICAL to `FloraView::model`/`rotation_quat`'s basis (right,
/// up=surface-normal, forward), so a Nanite bake pre-rotated by this matches the
/// plain branch placement. At the viewer's `origin == ZERO` this is the fixed
/// basis (right=+Z, up=+Y, forward=-X). Kept here (not imported from flora_view)
/// so the bake is self-contained.
pub fn model_rotation(origin: DVec3) -> Quat {
    let up = origin.normalize_or(DVec3::Y).as_vec3();
    let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
    let right = reference.cross(up).normalize();
    let forward = right.cross(up).normalize();
    Quat::from_mat3(&glam::Mat3::from_cols(right, up, forward))
}

/// The flora-viewer Nanite branch path: a single static tree's branch mesh baked
/// into a [`ClusterAsset`] and rendered through enki-nanite's cull+draw, inside
/// the flora HDR scene pass, with the Triangle/Cluster/LOD/Lit debug color modes.
///
/// ponytail: SIMPLEST residency — one small static tree, so EVERY cluster is
/// seeded permanently-resident via `NaniteRenderer::new_with_color_format(rhi,
/// &[asset], ...)` (the `initial` seed loop). No `ClusterStreamer`, no
/// `update_residency`, no page churn: every frame just re-uploads the (constant)
/// active-slot list, culls, and draws. The full LOD DAG is still present, so the
/// LOD-cut + cross-fade in the cull/draw shaders work; we simply never EVICT.
pub struct FloraNanite {
    renderer: enki_nanite::render::NaniteRenderer,
    /// The model rotation baked into the asset (origin-fixed); the per-frame
    /// `node_xlat` handles only translation, so this is constant.
    origin: DVec3,
    /// Total cluster count across the DAG (stats / diagnostics).
    cluster_count: usize,
}

impl FloraNanite {
    /// Bake `bm` (pre-rotated for `origin`) and build the Nanite renderer against
    /// the flora SCENE pass color format (so `record_draw` can be recorded INSIDE
    /// `begin_scene_pass`/`end_scene_pass`). All clusters are permanently resident.
    pub fn new(
        rhi: &mut enki_rhi::Rhi,
        bm: &BranchMesh,
        origin: DVec3,
        scene_color_format: ash::vk::Format,
    ) -> Result<Self, enki_rhi::RhiError> {
        let asset = bake_branch_mesh(bm, model_rotation(origin));
        let cluster_count = asset.clusters.len();
        // `initial = &[asset]` → the seed loop in `new_with_color_format` makes
        // every cluster resident and active. patch_origin == ZERO == the tree
        // origin, so node_xlat = origin - camera reproduces the camera-relative
        // translation flora's plain path uses.
        let renderer = enki_nanite::render::NaniteRenderer::new_with_color_format(
            rhi,
            std::slice::from_ref(&asset),
            scene_color_format,
        )?;
        Ok(Self { renderer, origin, cluster_count })
    }

    /// Total clusters across the LOD DAG (all resident).
    pub fn cluster_count(&self) -> usize {
        self.cluster_count
    }

    /// Visible cluster count from the last completed cull (≈2 frames stale).
    pub fn last_visible_clusters(&self) -> u32 {
        self.renderer.last_visible_clusters()
    }

    /// Per-frame update + cull. Call in the begin_frame→begin_rendering gap (the
    /// cull is a compute dispatch — illegal inside a rendering instance), BEFORE
    /// `begin_scene_pass`. `debug_mode` is nanite_draw's color mode (0 lit / 3 tri
    /// / 4 cluster / 5 LOD); `camera`/`fu` come straight from the viewer. `screen_h`
    /// is the framebuffer height in pixels (for screen-space error). No dither (no
    /// TAA in the viewer) → hard LOD swaps.
    #[allow(clippy::too_many_arguments)]
    pub fn update_and_cull(
        &mut self,
        rhi: &mut enki_rhi::Rhi,
        fi: u32,
        camera_world: DVec3,
        fu: &enki_render::frame::FrameUniforms,
        screen_h: f32,
        fov_y: f32,
        tau_px: f32,
        debug_mode: u32,
        frame_index: u32,
    ) -> Result<(), enki_rhi::RhiError> {
        self.renderer.update(
            rhi, fi, camera_world, fu, screen_h, fov_y, tau_px, debug_mode,
            false, // dither off (no TAA)
            frame_index,
        )?;
        self.renderer.record_cull(rhi, fi)
    }

    /// Record the indirect vertex-pulling branch draw. Call INSIDE the flora scene
    /// pass (between `begin_scene_pass` and `end_scene_pass`), in place of the plain
    /// branch draw — leaves still draw as cards after this.
    pub fn record_draw(&self, rhi: &enki_rhi::Rhi, fi: u32) -> Result<(), enki_rhi::RhiError> {
        self.renderer.record_draw(rhi, fi)
    }

    /// The tree origin the asset was baked for (rotation is origin-derived).
    pub fn origin(&self) -> DVec3 {
        self.origin
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use enki_flora::genome::{random_genome, resolve, Env};
    use enki_flora::mesh::build_branch_mesh;
    use enki_nanite::bake::cluster_build::{MAX_CLUSTER_TRIS, MAX_CLUSTER_VERTS};

    /// Build the seed-42 branch mesh through the SAME genome→resolve→graph path
    /// the viewer uses (random_genome → resolve → build_branch_mesh).
    fn seed42_branch_mesh() -> BranchMesh {
        let env = Env::default();
        let genome = random_genome(&env, 42);
        let resolved = resolve(&genome, &env);
        build_branch_mesh(&resolved.graph)
    }

    #[test]
    fn bake_seed42_yields_valid_clusters() {
        let bm = seed42_branch_mesh();
        assert!(bm.vertex_count > 0, "seed-42 branch mesh should be non-empty");

        let asset = bake_branch_mesh(&bm, Quat::IDENTITY);
        assert!(
            asset.cluster_count() > 0,
            "expected >0 clusters, got {}",
            asset.cluster_count()
        );

        // Every cluster must respect the Nanite meshlet caps.
        for (i, c) in asset.clusters.iter().enumerate() {
            assert!(
                c.vertices.len() <= MAX_CLUSTER_VERTS,
                "cluster {i} has {} verts > {MAX_CLUSTER_VERTS}",
                c.vertices.len()
            );
            assert!(
                c.triangles.len() <= MAX_CLUSTER_TRIS,
                "cluster {i} has {} tris > {MAX_CLUSTER_TRIS}",
                c.triangles.len()
            );
            // Cluster-local indices must be in range.
            for t in &c.triangles {
                for &v in t {
                    assert!(
                        (v as usize) < c.vertices.len(),
                        "cluster {i} triangle index {v} out of range"
                    );
                }
            }
        }
    }

    #[test]
    fn bake_seed42_is_deterministic() {
        let a = bake_branch_mesh(&seed42_branch_mesh(), Quat::IDENTITY);
        let b = bake_branch_mesh(&seed42_branch_mesh(), Quat::IDENTITY);
        assert_eq!(
            a.cluster_count(),
            b.cluster_count(),
            "cluster count must be deterministic"
        );
        assert_eq!(
            a.total_triangles(),
            b.total_triangles(),
            "total triangle count must be deterministic"
        );
        assert_eq!(
            a.lod_levels(),
            b.lod_levels(),
            "LOD level count must be deterministic"
        );
        // Spot-check full structural equality of the first cluster's geometry.
        if let (Some(ca), Some(cb)) = (a.clusters.first(), b.clusters.first()) {
            assert_eq!(ca.vertices.len(), cb.vertices.len());
            assert_eq!(ca.triangles, cb.triangles);
        }
    }
}

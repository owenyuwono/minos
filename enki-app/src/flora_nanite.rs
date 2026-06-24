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
use glam::{DVec3, Mat3, Quat, Vec3};

/// One placed tree for the forest-patch aggregator: where to stand it + how to
/// spin/scale it. Structurally identical to (and convertible from)
/// `crate::flora_scatter::TreeInstance` — but that type lives in the `enki` BIN
/// (`main.rs`), not this LIB crate, so it can't be imported here. Kept local so
/// the bake is self-contained and `enki-app`'s lib stays buildable on its own.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreeInstance {
    /// f64 world position on the terrain surface.
    pub origin: DVec3,
    /// Rotation about the radial up (radians).
    pub yaw: f32,
    /// Uniform size multiplier.
    pub scale: f32,
}

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
        // No terrain self-shadow horizon for a tree patch (zero = nothing occludes).
        horizon: vec![[0.0f32; 4]; n],
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
    }
}

/// The per-instance bake transform for one [`TreeInstance`], REBASED onto a shared
/// `patch_origin` instead of the camera. This is `flora_scatter::instance_model`'s
/// basis math (orient tree-local +Y to the radial surface normal, yaw about up,
/// uniform scale) with `camera_world_pos` replaced by `patch_origin` — so the
/// translate is `(origin - patch_origin)` in f64, cast to f32 LAST.
///
/// Returns `(linear, translation)`:
/// - `linear` (`Mat3`) is the similarity 3×3 applied to baked POSITIONS (rotation
///   + uniform scale + yaw, det > 0 → cull-BACK winding preserved).
/// - `translation` (`Vec3`) is `(origin - patch_origin).as_vec3()`.
///
/// Normals transform by the ROTATION part only; since `linear` is a uniform-scale ×
/// rotation, `linear / scale` is that pure rotation (we renormalize defensively).
///
/// Mirrors `flora_scatter::instance_model` exactly so a forest-patch tree lands in
/// the SAME place/orientation as the plain per-instance flora draw — only the
/// rebasing origin differs (patch centroid vs. camera).
fn instance_bake_transform(origin: DVec3, yaw: f32, scale: f32, patch_origin: DVec3) -> (Mat3, Vec3) {
    let up = origin.normalize_or(DVec3::Y).as_vec3();
    let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
    let right = reference.cross(up).normalize();
    let forward = right.cross(up).normalize();
    // Yaw about up, in the tangent plane (preserves orthonormality + handedness).
    let (s, c) = yaw.sin_cos();
    let right_y = right * c + forward * s;
    let forward_y = forward * c - right * s;
    // Columns match instance_model's (right, up, forward) basis, uniformly scaled.
    let linear = Mat3::from_cols(right_y * scale, up * scale, forward_y * scale);
    // Translate is computed in f64 (planet-scale precise) and cast to f32 LAST.
    let translation = (origin - patch_origin).as_vec3();
    (linear, translation)
}

/// Weld N pre-transformed flora trees into ONE [`ClusterAsset`] — a "forest patch":
/// one asset, one Nanite node, one indirect draw for a whole grove.
///
/// `patch_origin` is the caller-chosen SHARED f64 origin (the patch centroid on the
/// sphere, e.g. `dir_of(patch_center) * ground_radius(...)`). Every instance is
/// emitted RELATIVE to this origin (NOT each tree's own origin), so the runtime
/// `node_xlat = patch_origin - camera` places the whole patch with one translate.
///
/// The Nanite draw has NO model matrix and `node_xlat` is per-NODE (one patch =
/// one node), so each instance's yaw / scale / sphere-orientation is baked INTO the
/// f32 vertices via [`instance_bake_transform`] (the same basis
/// `flora_scatter::instance_model` uses, rebased onto `patch_origin`). Trees are
/// welded in slice index order into a single growing [`PatchMesh`]; positions /
/// normals are the only per-instance deltas (orientation+placement), and `origin`
/// is `patch_origin` — every other channel is the SAME flat/zero fill
/// [`bake_branch_mesh`] uses (the channels are tree-meaningless), so the resulting
/// asset is byte-shape-identical to a terrain one (render.rs `SLOT_FLOATS` and
/// `write_cluster` need ZERO changes).
///
/// No vertex welding ACROSS trees: disjoint trees share no coincident verts, and
/// the DAG's `merge_group` welds by BIT-EXACT position key, so a cross-tree weld
/// would no-op anyway. The cost: the merge is a PACKAGING convenience (one asset /
/// node / draw), NOT a triangle-collapse win — disjoint branchy tubes each stall at
/// their own per-tree simplification floor (≈653 tris / 8 clusters for the seed-42
/// specimen), so the coarsest LOD is ≈ `N × that floor`. Genuine collapse needs
/// CONNECTING geometry (a ground / canopy skirt joining trunks into one manifold);
/// see the test's collapse note and the Tier-3 roadmap.
///
/// Precision: keep the patch radius modest (≈ the scatter `RADIUS_M` window) so
/// `origin - patch_origin` stays small in f32 — large patches at planet scale
/// (~6.37e6 m) lose mm precision vs. a per-tree origin (fine for bark, but argues
/// for `patch_origin = patch centroid`, not the planet centre).
///
/// Deterministic given `(bm, instances, patch_origin)`: `bm` is the golden-gated
/// shared species mesh, `instances` come from the deterministic `flora_scatter`,
/// and the weld iterates the slice in index order; f32 glam transforms are
/// bit-stable for a fixed glam version, and `build_base_clusters`/`build_dag` are
/// already determinism-proven (see `dag::bake_is_deterministic`).
pub fn bake_forest_patch(
    bm: &BranchMesh,
    instances: &[TreeInstance],
    patch_origin: DVec3,
) -> ClusterAsset {
    let src_n = bm.positions.len() / 3;
    let has_normals = bm.normals.len() == bm.positions.len();

    // Source vertices/normals as [f32;3], read once and reused per instance.
    let src_pos: Vec<Vec3> = bm
        .positions
        .chunks_exact(3)
        .map(|c| Vec3::new(c[0], c[1], c[2]))
        .collect();
    let src_nrm: Vec<Vec3> = if has_normals {
        bm.normals
            .chunks_exact(3)
            .map(|c| Vec3::new(c[0], c[1], c[2]))
            .collect()
    } else {
        vec![Vec3::Y; src_n]
    };

    // One growing welded mesh in patch-local space.
    let total = src_n * instances.len();
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(total);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(total);
    let mut indices: Vec<u32> = Vec::with_capacity(bm.indices.len() * instances.len());

    for inst in instances {
        // Append this tree's vertices at `base`; rebase its indices by `base`.
        let base = positions.len() as u32;
        let (linear, t) = instance_bake_transform(inst.origin, inst.yaw, inst.scale, patch_origin);
        // Normals transform by rotation only: linear / scale = pure rotation
        // (renormalize defensively; the basis is orthonormal anyway).
        let inv_scale = if inst.scale.abs() > f32::EPSILON { 1.0 / inst.scale } else { 1.0 };
        let rot = linear * inv_scale;
        for i in 0..src_n {
            positions.push((linear * src_pos[i] + t).to_array());
            normals.push((rot * src_nrm[i]).normalize_or(Vec3::Y).to_array());
        }
        for &e in &bm.indices {
            indices.push(base + e);
        }
    }

    let n = positions.len();
    let pm = PatchMesh {
        positions,
        normals,
        // Flat bark albedo — IDENTICAL to bake_branch_mesh (only Lit samples it).
        colors: vec![[0.45, 0.30, 0.18]; n],
        // Terrain-only debug lanes — tree-meaningless; zero-fill (survive the DAG
        // bit-identical by vertex identity, so coarse-LOD bark stays correct).
        material: vec![0.0; n],
        wetness: vec![0.0; n],
        volcanism: vec![0.0; n],
        elevation: vec![0.0; n],
        plate: vec![[0.0; 3]; n],
        horizon: vec![[0.0f32; 4]; n],
        indices,
        // build_base_clusters ignores `boundary`; the DAG re-derives lock edges
        // from the welded merged-group mesh, so an all-false mask is correct.
        boundary: vec![false; n],
        // The SHARED patch origin — positions are emitted relative to it.
        origin: patch_origin,
    };

    let clusters = build_dag(build_base_clusters(&pm));

    ClusterAsset {
        clusters,
        patch_origin,
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
        // The Nanite draw set statically references the sun shadow map (bindings
        // 8/9), so even this debug path must bind a valid (cleared) map. The viewer
        // has no sun shadow; `update_and_cull` keeps it cleared with an empty pass.
        if !rhi.has_shadow_map() {
            rhi.create_shadow_map(1024, 3)?; // 3 = enki_nanite::render::SHADOW_CASCADES
        }
        renderer.bind_shadow_map(rhi);
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
            [glam::Mat4::IDENTITY; 3], // no sun shadow in the flora debug bake (per cascade)
            [0.0; 4],                  // shadow disabled (strength/enabled = 0)
        )?;
        self.renderer.record_cull(rhi, fi)?;
        // Keep the (unused) per-cascade shadow maps cleared + in SHADER_READ_ONLY so
        // the draw's statically-referenced bindings stay valid. Empty passes, no caster.
        for c in 0..3u32 {
            rhi.begin_shadow_pass(fi, c);
            rhi.end_shadow_pass(fi, c);
        }
        Ok(())
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

    // ── Forest-patch aggregator (Tier-3 CORE) ────────────────────────────────

    const PATCH_TREES: usize = 20;

    /// Coarsest-LOD tri count of a cluster set at its top DAG level. Mirrors the
    /// `dag.rs` coarsest-LOD probes (max `lod` → sum of those clusters' tris).
    fn coarsest_lod_tris(asset: &ClusterAsset) -> usize {
        let max_lod = asset.clusters.iter().map(|c| c.lod).max().unwrap_or(0);
        asset
            .clusters
            .iter()
            .filter(|c| c.lod == max_lod)
            .map(|c| c.triangles.len())
            .sum()
    }

    /// 20 synthetic [`TreeInstance`]s on a SMALL sphere cap (radius ~50 km, a few m
    /// apart), each with a distinct yaw/scale — a deterministic stand-in for a
    /// `flora_scatter` window. The patch_origin is the cap centre on the sphere.
    /// Spacing is wide enough that the welded trees are DISJOINT (no shared verts),
    /// exercising the honest "no cross-tree collapse" path.
    fn synthetic_patch() -> (Vec<TreeInstance>, DVec3) {
        const R: f64 = 50_000.0; // a modest test planet radius
        let center_dir = DVec3::new(0.3, 0.55, 0.2).normalize();
        let patch_origin = center_dir * R;
        // Two tangents at the cap centre to spread instances in the tangent plane.
        let up = center_dir;
        let t0 = DVec3::X.cross(up).normalize();
        let t1 = up.cross(t0).normalize();
        let mut out = Vec::with_capacity(PATCH_TREES);
        for k in 0..PATCH_TREES {
            // Lay the trees on a small grid ~6 m apart in the tangent plane.
            let gx = (k % 5) as f64 - 2.0;
            let gy = (k / 5) as f64 - 1.5;
            let local = t0 * (gx * 6.0) + t1 * (gy * 6.0);
            // Re-project onto the sphere so each origin is a real surface point.
            let dir = (up * R + local).normalize();
            out.push(TreeInstance {
                origin: dir * R,
                yaw: k as f32 * 0.37, // distinct, non-trivial yaws
                scale: 0.9 + (k as f32 % 4.0) * 0.05,
            });
        }
        (out, patch_origin)
    }

    #[test]
    fn forest_patch_yields_valid_clusters() {
        let bm = seed42_branch_mesh();
        let (instances, patch_origin) = synthetic_patch();
        let asset = bake_forest_patch(&bm, &instances, patch_origin);
        assert!(asset.cluster_count() > 0, "expected >0 clusters");
        // patch_origin rides through onto the asset (drives node_xlat).
        assert_eq!(asset.patch_origin, patch_origin);

        // Meshlet caps + cluster-local index validity (mirrors the single-tree test).
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
    fn forest_patch_weld_is_lossless() {
        // LOD0 tri count must equal N × the single-tree LOD0 tri count — the weld
        // appends every instance's geometry verbatim, dropping no triangles.
        let bm = seed42_branch_mesh();
        let (instances, patch_origin) = synthetic_patch();

        let single = bake_branch_mesh(&bm, Quat::IDENTITY);
        let single_lod0: usize = single
            .clusters
            .iter()
            .filter(|c| c.lod == 0)
            .map(|c| c.triangles.len())
            .sum();
        // Sanity on the documented seed-42 floor (LOD0 = 3240 tris).
        assert_eq!(single_lod0, 3240, "seed-42 single-tree LOD0 tri count drifted");

        let patch = bake_forest_patch(&bm, &instances, patch_origin);
        let patch_lod0: usize = patch
            .clusters
            .iter()
            .filter(|c| c.lod == 0)
            .map(|c| c.triangles.len())
            .sum();
        assert_eq!(
            patch_lod0,
            PATCH_TREES * single_lod0,
            "welded LOD0 tri count must be exactly N × single-tree (lossless weld)"
        );
    }

    #[test]
    fn forest_patch_error_is_monotonic_and_bounds_nest() {
        let bm = seed42_branch_mesh();
        let (instances, patch_origin) = synthetic_patch();
        let asset = bake_forest_patch(&bm, &instances, patch_origin);
        assert!(asset.lod_levels() >= 2, "expected a multi-level DAG");

        for (i, c) in asset.clusters.iter().enumerate() {
            // Monotonic error (mirrors dag::error_is_monotonic).
            assert!(
                c.parent_error >= c.self_error,
                "cluster {i}: parent_error {} < self_error {} (non-monotonic)",
                c.parent_error,
                c.self_error
            );
            // Nested parent bounds (mirrors dag::parent_bounds_enclose_self_bounds).
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
    fn forest_patch_is_deterministic() {
        let bm = seed42_branch_mesh();
        let (instances, patch_origin) = synthetic_patch();
        let a = bake_forest_patch(&bm, &instances, patch_origin);
        let b = bake_forest_patch(&bm, &instances, patch_origin);
        assert_eq!(a.cluster_count(), b.cluster_count(), "cluster count must be deterministic");
        assert_eq!(a.total_triangles(), b.total_triangles(), "total tris must be deterministic");
        assert_eq!(a.lod_levels(), b.lod_levels(), "LOD level count must be deterministic");
        if let (Some(ca), Some(cb)) = (a.clusters.first(), b.clusters.first()) {
            assert_eq!(ca.vertices.len(), cb.vertices.len());
            assert_eq!(ca.triangles, cb.triangles, "first cluster geometry must be deterministic");
        }
    }

    #[test]
    fn forest_patch_bark_channels_survive_to_coarsest_lod() {
        // The flat-fill channels must ride the DAG by vertex identity, bit-identical
        // to the coarsest LOD (meshopt simplifies on POSITION only; every other
        // channel is copied by the surviving vertex's index, never interpolated).
        let bm = seed42_branch_mesh();
        let (instances, patch_origin) = synthetic_patch();
        let asset = bake_forest_patch(&bm, &instances, patch_origin);
        let max_lod = asset.clusters.iter().map(|c| c.lod).max().unwrap();
        let coarse = asset
            .clusters
            .iter()
            .find(|c| c.lod == max_lod && !c.vertices.is_empty())
            .expect("a non-empty coarsest-LOD cluster");
        let v = &coarse.vertices[0];
        assert_eq!(v.color, [0.45, 0.30, 0.18], "coarse-LOD bark albedo drifted");
        assert_eq!(v.horizon, [0.0; 4], "coarse-LOD horizon must stay zero");
        assert_eq!(v.material, 0.0);
        assert_eq!(v.wetness, 0.0);
        assert_eq!(v.volcanism, 0.0);
        assert_eq!(v.elevation, 0.0);
        assert_eq!(v.plate, [0.0; 3]);
    }

    #[test]
    fn forest_patch_collapse_ceiling() {
        // HONEST FINDING (empirically measured, seed-42): a single tree's DAG floors
        // at ~653 tris / 8 clusters and STALLS — a thin branchy tube is almost all
        // silhouette geometry, so meshopt can't dissolve it further without breaking
        // the tube. DISJOINT welded trees share ZERO coincident verts, so the DAG's
        // bit-exact-position `merge_group` weld does NOTHING across trees: each tree
        // remains an independent simplification island and the coarsest LOD floors at
        // ≈ N × the single-tree floor (NOT "well below N×", as the original Tier-3
        // hypothesis guessed). The merge is a PACKAGING win (one asset / node / draw),
        // NOT a triangle-collapse win — real collapse requires CONNECTING geometry
        // (a ground/canopy skirt joining trunks into one manifold), which this POC
        // deliberately does not add. So the truthful assertion is a CEILING, not a
        // "collapses below" claim:  coarsest <= N × 653 + slack.
        const SINGLE_TREE_FLOOR: usize = 653; // measured seed-42 coarsest-LOD tris
        let bm = seed42_branch_mesh();
        let (instances, patch_origin) = synthetic_patch();
        let asset = bake_forest_patch(&bm, &instances, patch_origin);

        let coarse = coarsest_lod_tris(&asset);
        let ceiling = PATCH_TREES * SINGLE_TREE_FLOOR + PATCH_TREES * 64; // + per-tree slack
        assert!(
            coarse <= ceiling,
            "coarsest-LOD tris {coarse} exceeded ceiling {ceiling} (N={PATCH_TREES} × {SINGLE_TREE_FLOOR} floor + slack)"
        );
        // It must at least be a real DAG that reduced below LOD0 (N × 3240).
        let lod0 = PATCH_TREES * 3240;
        assert!(coarse < lod0, "coarsest LOD {coarse} did not reduce below LOD0 {lod0}");
    }
}

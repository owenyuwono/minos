//! Stage-2 cluster streaming — CPU-side residency selection.
//!
//! True Nanite resides only the clusters near the current LOD cut, so the DAG can
//! be far deeper than the GPU pool. This module is the brain: given the baked
//! per-face cluster DAGs and the camera, it picks **which clusters should be GPU-
//! resident** — a strict *superset* of the GPU cut (the clusters the cull will
//! actually draw) widened by a margin so small camera moves don't expose holes.
//! The renderer then uploads/evicts to match.
//!
//! Pure data + `glam`, no GPU — unit-testable headless. The selection metric
//! mirrors `shaders/nanite_cull.wgsl::project_error` exactly so the resident set
//! and the GPU cut agree.

use glam::DVec3;

use crate::cluster::ClusterAsset;

/// Near-plane clamp for error projection (matches the cull shader's `ZNEAR`).
const ZNEAR: f32 = 0.5;

/// Precomputed per-cluster selection data (world-absolute f64 centers so the
/// projection stays precise at planet scale).
#[derive(Clone, Copy, Debug)]
pub struct ClusterEntry {
    /// Index of the owning face/asset in the slice passed to [`ClusterResidency::new`].
    pub face: u32,
    /// Index of this cluster within its face asset.
    pub cluster: u32,
    /// World-absolute bounds-sphere center (`patch_origin + cluster.bounds.center`).
    pub center: DVec3,
    pub radius: f32,
    pub self_error: f32,
    pub parent_error: f32,
    /// World-absolute parent bounds-sphere center.
    pub pcenter: DVec3,
    pub pradius: f32,
}

/// A flat, camera-independent table over every cluster of every face, queried each
/// frame to produce the resident set.
pub struct ClusterResidency {
    entries: Vec<ClusterEntry>,
}

impl ClusterResidency {
    /// Build the table from the baked face DAGs (one entry per cluster).
    pub fn new(assets: &[ClusterAsset]) -> Self {
        let mut entries = Vec::new();
        for (fi, a) in assets.iter().enumerate() {
            for (ci, c) in a.clusters.iter().enumerate() {
                let center = a.patch_origin
                    + DVec3::new(
                        c.bounds.center.x as f64,
                        c.bounds.center.y as f64,
                        c.bounds.center.z as f64,
                    );
                let pcenter = a.patch_origin
                    + DVec3::new(
                        c.parent_bounds.center.x as f64,
                        c.parent_bounds.center.y as f64,
                        c.parent_bounds.center.z as f64,
                    );
                entries.push(ClusterEntry {
                    face: fi as u32,
                    cluster: ci as u32,
                    center,
                    radius: c.bounds.radius,
                    self_error: c.self_error,
                    parent_error: c.parent_error,
                    pcenter,
                    pradius: c.parent_bounds.radius,
                });
            }
        }
        Self { entries }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entries(&self) -> &[ClusterEntry] {
        &self.entries
    }

    /// Select the clusters that should be resident for `camera_world`. Returns
    /// indices into [`Self::entries`].
    ///
    /// `cot` = `1 / tan(fov_y / 2)`. `margin` (≥ 1) widens the cut band so the
    /// resident set is a strict superset of the exact GPU cut (`margin == 1`):
    /// a cluster is resident if `self_px ≤ tau·margin` AND `parent_px > tau/margin`.
    /// No frustum test here on purpose — residency stays direction-agnostic so
    /// rotating the camera never exposes a hole; the GPU cull does the frustum.
    pub fn select(
        &self,
        camera_world: DVec3,
        tau: f32,
        screen_h: f32,
        cot: f32,
        margin: f32,
    ) -> Vec<u32> {
        let margin = margin.max(1.0);
        let hi = tau * margin;
        let lo = tau / margin;
        let mut out = Vec::new();
        for (i, e) in self.entries.iter().enumerate() {
            let self_px = project_error(
                e.self_error,
                (e.center - camera_world).length() as f32,
                e.radius,
                screen_h,
                cot,
            );
            let parent_px = if e.parent_error >= 1e30 {
                1e30
            } else {
                project_error(
                    e.parent_error,
                    (e.pcenter - camera_world).length() as f32,
                    e.pradius,
                    screen_h,
                    cot,
                )
            };
            if self_px <= hi && parent_px > lo {
                out.push(i as u32);
            }
        }
        // Nearest-first, so if the GPU pool overflows the renderer drops the
        // FARTHEST (most likely off-screen / occluded) clusters, never near ones.
        out.sort_by(|&a, &b| {
            let da = (self.entries[a as usize].center - camera_world).length_squared();
            let db = (self.entries[b as usize].center - camera_world).length_squared();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }
}

/// Stage-2 streaming manager: wraps [`ClusterResidency`] with camera-delta
/// throttling so the (relatively expensive) reselection + GPU re-upload only runs
/// when the view has actually changed enough to alter the resident set.
pub struct ClusterStreamer {
    residency: ClusterResidency,
    last_cam: DVec3,
    selection: Vec<u32>,
    initialized: bool,
}

impl ClusterStreamer {
    pub fn new(assets: &[ClusterAsset]) -> Self {
        Self {
            residency: ClusterResidency::new(assets),
            last_cam: DVec3::ZERO,
            selection: Vec::new(),
            initialized: false,
        }
    }

    /// Per-cluster table (pass alongside the selection to
    /// `NaniteRenderer::update_geometry_selected`).
    pub fn entries(&self) -> &[ClusterEntry] {
        self.residency.entries()
    }

    /// Current resident selection (indices into [`Self::entries`]).
    pub fn selection(&self) -> &[u32] {
        &self.selection
    }

    pub fn resident_count(&self) -> usize {
        self.selection.len()
    }

    /// Advance streaming for `camera_world`. Returns `Some(selection)` when the
    /// resident set changed (caller re-uploads), else `None`. Skips reselection
    /// entirely while the camera has moved less than `move_threshold` world units
    /// since the last reselection — cheap to call every frame.
    pub fn update(
        &mut self,
        camera_world: DVec3,
        tau: f32,
        screen_h: f32,
        cot: f32,
        margin: f32,
        move_threshold: f64,
    ) -> Option<&[u32]> {
        if self.initialized && (camera_world - self.last_cam).length() < move_threshold {
            return None;
        }
        self.last_cam = camera_world;
        self.initialized = true;

        let sel = self.residency.select(camera_world, tau, screen_h, cot, margin);
        if sel == self.selection {
            return None;
        }
        self.selection = sel;
        Some(&self.selection)
    }
}

/// Project a world-space error of magnitude `err` (at bounds-sphere nearest point)
/// to pixels. Mirrors `nanite_cull.wgsl::project_error`.
fn project_error(err: f32, dist_to_center: f32, radius: f32, screen_h: f32, cot: f32) -> f32 {
    if err <= 0.0 {
        return 0.0;
    }
    let d = (dist_to_center - radius).max(ZNEAR);
    0.5 * screen_h * cot * err / d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::{bake_patch, PatchParams};
    use minos_planet::noise::Noise3D;
    use minos_planet::simple_height::SimpleHeightField;

    fn asset() -> ClusterAsset {
        bake_patch(
            &PatchParams {
                face: 2,
                level: 3,
                ix: 3,
                iy: 4,
                resolution: 64,
                radius: 6_371_000.0,
                height_scale: 8_848.0,
            },
            &SimpleHeightField { noise: Noise3D::new(7) },
        )
    }

    fn cot(fov_y: f32) -> f32 {
        1.0 / (fov_y * 0.5).tan()
    }

    #[test]
    fn table_has_one_entry_per_cluster() {
        let a = asset();
        let r = ClusterResidency::new(std::slice::from_ref(&a));
        assert_eq!(r.len(), a.cluster_count());
        assert!(!r.is_empty());
    }

    #[test]
    fn widened_band_is_superset_of_exact_cut() {
        let a = asset();
        let r = ClusterResidency::new(std::slice::from_ref(&a));
        // A camera a little above the patch surface.
        let cam = a.patch_origin * (1.0 + 2_000.0 / a.patch_origin.length());
        let (tau, sh, c) = (1.0f32, 1080.0f32, cot(1.0));

        let exact: std::collections::HashSet<u32> =
            r.select(cam, tau, sh, c, 1.0).into_iter().collect();
        let widened: std::collections::HashSet<u32> =
            r.select(cam, tau, sh, c, 2.5).into_iter().collect();

        assert!(!exact.is_empty(), "exact cut should select something");
        for k in &exact {
            assert!(widened.contains(k), "widened band must contain the exact cut");
        }
        assert!(widened.len() >= exact.len());
    }

    #[test]
    fn closer_camera_selects_more_clusters() {
        let a = asset();
        let r = ClusterResidency::new(std::slice::from_ref(&a));
        let dir = a.patch_origin.normalize();
        let surf = a.patch_origin.length();
        let near = dir * (surf + 3_000.0);
        let far = dir * (surf + 400_000.0);
        let (tau, sh, c) = (1.0f32, 1080.0f32, cot(1.0));

        let n = r.select(near, tau, sh, c, 1.5).len();
        let f = r.select(far, tau, sh, c, 1.5).len();
        assert!(
            n > f,
            "closer camera needs finer (more) clusters: near={n} far={f}"
        );
    }

    #[test]
    fn streamer_throttles_and_reselects() {
        let a = asset();
        let mut s = ClusterStreamer::new(std::slice::from_ref(&a));
        let dir = a.patch_origin.normalize();
        let surf = a.patch_origin.length();
        let near = dir * (surf + 3_000.0);
        let (tau, sh, c) = (1.0f32, 1080.0f32, cot(1.0));

        // First call always selects.
        assert!(s.update(near, tau, sh, c, 1.5, 100.0).is_some());
        assert!(s.resident_count() > 0);
        // Tiny move below threshold → no reselection.
        assert!(s.update(near + dir * 1.0, tau, sh, c, 1.5, 100.0).is_none());
        // Large move far away past threshold → reselects with a coarser (changed) set.
        let far = dir * (surf + 800_000.0);
        assert!(s.update(far, tau, sh, c, 1.5, 100.0).is_some());
    }

    #[test]
    fn selection_is_bounded_by_table() {
        let a = asset();
        let r = ClusterResidency::new(std::slice::from_ref(&a));
        let cam = a.patch_origin.normalize() * (a.patch_origin.length() + 1_000.0);
        let sel = r.select(cam, 1.0, 1080.0, cot(1.0), 4.0);
        assert!(sel.len() <= r.len());
    }
}

//! Cluster-DAG data types — the frozen, GPU-agnostic contract produced by the
//! baker ([`crate::bake`]) and consumed by the runtime pass (added in N1).
//!
//! Pure data + `glam`; no `ash`, no GPU. Positions and bounding-sphere centers
//! are stored **patch-origin-relative** (f32), matching enki's camera-relative
//! precision convention: the absolute planet-space origin lives in
//! [`ClusterAsset::patch_origin`] (f64) and the per-frame camera-relative
//! translation is applied at draw time.

use glam::{DVec3, Vec3};

/// World-space bounding sphere; `center` is patch-origin-relative (f32).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingSphere {
    pub center: Vec3,
    pub radius: f32,
}

/// One cluster-local vertex. Clusters are self-contained, so vertices shared
/// across cluster boundaries are duplicated — the standard meshlet layout.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClusterVertex {
    /// Patch-origin-relative position (f32).
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    /// Rock hardness 0..1 (soft→hard) — debug "material" view. Rides along by
    /// vertex identity through cluster build + DAG (meshopt simplifies on pos only).
    pub material: f32,
    /// Surface wetness 0..1 — debug "wetness" view. Same free-ride as `material`.
    pub wetness: f32,
    /// Volcano-cone influence 0..1 (arc + hotspot) — debug "volcano" view. Same free-ride.
    pub volcanism: f32,
    /// Signed normalized elevation h (~[-1,1]) — debug "height" view. Same free-ride.
    pub elevation: f32,
    /// Per-plate tint — debug "plate" view. Same free-ride as `material`.
    pub plate: [f32; 3],
    /// Terrain self-shadow: SINE of the max terrain elevation angle in 4 azimuths
    /// (`horizon_basis` tangents). Sun-direction-agnostic; same free-ride.
    pub horizon: [f32; 4],
}

/// Parent error for root clusters: `+inf` always passes the cut's upper test.
pub const ROOT_PARENT_ERROR: f32 = f32::INFINITY;

/// A single cluster: ≤64 verts, ≤128 triangles, plus the LOD-cut metadata.
///
/// LOD-cut test (runtime): draw iff `proj(self_error @ bounds) <= τ` **and**
/// `proj(parent_error @ parent_bounds) > τ`. `self_error`/`bounds` are shared by
/// every cluster in the same group; `parent_*` describe the group one DAG level
/// up. Monotonicity (`parent_error >= self_error`) and nested parent bounds are
/// guaranteed by the baker — together they make the per-cluster test a globally
/// consistent, crack-free DAG cut.
#[derive(Debug, Clone)]
pub struct Cluster {
    pub vertices: Vec<ClusterVertex>,
    /// Triangles as cluster-local vertex indices (each `< vertices.len() <= 64`).
    pub triangles: Vec<[u8; 3]>,
    /// Group bounding sphere (shared within the group); projects `self_error`.
    pub bounds: BoundingSphere,
    /// Group simplification error in world units (shared within group). `0` at LOD0.
    pub self_error: f32,
    /// Parent group's error (`>= self_error` by construction). `+inf` for roots.
    pub parent_error: f32,
    /// Parent group's bounding sphere (encloses this group's); projects `parent_error`.
    pub parent_bounds: BoundingSphere,
    /// Group id at this cluster's DAG level (clusters in a group share LOD state).
    pub group: u32,
    /// DAG level: 0 = finest (LOD0), increasing = coarser.
    pub lod: u8,
}

impl Cluster {
    /// A root cluster has no parent (its `parent_error` is `+inf`).
    #[inline]
    pub fn is_root(&self) -> bool {
        self.parent_error.is_infinite()
    }
}

/// The full baked cluster DAG for one terrain patch — the resident bundle for v1.
#[derive(Debug, Clone)]
pub struct ClusterAsset {
    pub clusters: Vec<Cluster>,
    /// Absolute planet-space origin (f64); cluster positions are relative to this.
    pub patch_origin: DVec3,
}

impl ClusterAsset {
    #[inline]
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }
    pub fn total_triangles(&self) -> usize {
        self.clusters.iter().map(|c| c.triangles.len()).sum()
    }
    /// Number of LOD levels present (max `lod` + 1), or 0 if empty.
    pub fn lod_levels(&self) -> usize {
        self.clusters
            .iter()
            .map(|c| c.lod as usize)
            .max()
            .map_or(0, |m| m + 1)
    }
}

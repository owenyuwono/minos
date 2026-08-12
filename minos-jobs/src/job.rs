//! Job descriptors passed between the renderer and the worker pool.

use minos_planet::quadtree::ChunkKey;
use minos_planet::ChunkMeshArrays;

/// A request to build the mesh for a specific chunk, with a scheduling priority.
pub struct MeshRequest {
    pub key: ChunkKey,
    /// Higher value = build sooner.
    pub priority: f32,
}

/// The completed output of a mesh-build job.
pub struct MeshResult {
    pub key: ChunkKey,
    /// Monotonically increasing generation counter; stale results can be discarded.
    pub generation: u64,
    pub arrays: ChunkMeshArrays,
}

/// Immutable template shared by all worker threads.
pub struct ChunkBuildTemplate {
    pub resolution: u32,
    pub radius: f64,
    pub height_scale: f64,
}

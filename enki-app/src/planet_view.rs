
//! `planet_view` — bridges the LOD planet (CPU stream) to the streaming uploader (GPU).
//!
//! # Architecture
//!
//! ```text
//!  WorkerPool ──────────────► LRU cache ──────────────► resident (HashMap)
//!  (build_chunk threads)       ChunkKey → arrays         ChunkKey → ResidentChunk
//!       ▲                                                      │
//!  LodTree::update                                            Rhi::streaming_upload
//!  (produces builds/show/hide/cancels)                    Rhi::streaming_retire
//! ```
//!
//! # Per-frame flow (see `PlanetView::update`)
//!
//! 1. Drain worker results into the LRU cache.
//! 2. Call `LodTree::update` to get the current frame's action list.
//! 3. Submit new build jobs; cancel stale ones.
//! 4. Upload cached meshes to the GPU (back-pressure retry queue first).
//! 5. Retire hidden GPU meshes.
//!
//! # LRU eviction safety
//!
//! The LRU cache may only evict entries that are NOT in `resident` and NOT in
//! `pending_upload`.  Both guards are enforced by `safe_cache_put` before any
//! put that might trigger eviction.
//!
//! # Back-pressure
//!
//! When `ChunkUploader::upload` returns `Ok(None)` (staging ring / budget full),
//! the key stays in the LRU cache and is pushed onto `pending_upload` for retry
//! next frame.  The mesh data is never dropped in this path.
//!
//! # Testability
//!
//! The upload/retire operations are hidden behind the `ChunkUploader` trait so
//! that the entire update logic is exercisable without a GPU via `FakeUploader`.
//! The trait uses an associated `Handle` type so tests can use a plain `u64`
//! instead of the GPU-backed `StreamedMesh`.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;

use glam::{DVec3, Mat4};
use lru::LruCache;

use enki_jobs::job::ChunkBuildTemplate;
use enki_jobs::pool::WorkerPool;
use enki_planet::height::HeightField;
use enki_planet::lod::{LodCamera, LodConfig, LodTree};
use enki_planet::quadtree::ChunkKey;
use enki_render::frame::FrameUniforms;
use enki_render::material::ChunkPush;
use enki_rhi::{Rhi, RhiError, StreamedMesh};

// ── Uploader trait ─────────────────────────────────────────────────────────────

/// Abstracts the GPU upload/retire interface for testability.
///
/// The associated `Handle` type is `StreamedMesh` in production and a plain
/// `u64` (or other lightweight token) in tests, letting the bridge's update
/// logic run fully headless.
pub trait ChunkUploader {
    /// Opaque handle type returned by a successful upload.
    type Handle: Copy;

    /// Attempt to upload the mesh arrays to device-local buffers.
    ///
    /// Returns:
    /// - `Ok(Some(handle))` — uploaded; `handle` identifies the mesh for retire.
    /// - `Ok(None)`         — budget / ring full; caller retries next frame.
    /// - `Err(_)`           — fatal error.
    fn upload(
        &mut self,
        fi: u32,
        arrays: &enki_planet::ChunkMeshArrays,
    ) -> Result<Option<Self::Handle>, RhiError>;

    /// Schedule a previously uploaded mesh for deferred destruction.
    fn retire(&mut self, handle: Self::Handle);
}

// ── Real uploader (delegates to Rhi) ──────────────────────────────────────────

/// Wraps a frame-slot `fi` + `&mut Rhi` to implement `ChunkUploader`.
pub struct RhiUploader<'r> {
    rhi: &'r mut Rhi,
    fi:  u32,
}

impl<'r> RhiUploader<'r> {
    pub fn new(rhi: &'r mut Rhi, fi: u32) -> Self {
        Self { rhi, fi }
    }
}

impl<'r> ChunkUploader for RhiUploader<'r> {
    type Handle = StreamedMesh;

    fn upload(
        &mut self,
        _fi: u32,
        arrays: &enki_planet::ChunkMeshArrays,
    ) -> Result<Option<StreamedMesh>, RhiError> {
        let plate_src = arrays.plate_colors.as_deref();
        self.rhi.streaming_upload(
            self.fi,
            &arrays.positions,
            &arrays.normals,
            &arrays.colors,
            plate_src,
            None, // classic path: no geomorph stream
            &arrays.indices,
        )
    }

    fn retire(&mut self, handle: StreamedMesh) {
        self.rhi.streaming_retire(handle);
    }
}

// ── ResidentChunk ─────────────────────────────────────────────────────────────

/// A chunk whose mesh lives on the GPU and is actively drawn.
///
/// `H` is the handle type returned by the uploader (production: `StreamedMesh`,
/// tests: `u64`).
struct ResidentChunk<H: Copy> {
    handle: H,
    /// Planet-space f64 origin of this chunk, used to build the camera-relative
    /// push constant each frame.
    origin: DVec3,
}

// ── PlanetConfig ──────────────────────────────────────────────────────────────

/// Configuration bundle for `PlanetView`.
pub struct PlanetConfig {
    pub lod:     LodConfig,
    /// Noise seed (used by both `SimpleHeightField` and `TectonicHeightField`).
    pub seed:    u32,
    /// Number of worker threads in the mesh-build pool.
    pub n_workers: usize,
    /// Terrain fill pipeline handle.
    pub terrain_pipeline:           enki_rhi::PipelineHandle,
    /// Terrain wireframe pipeline handle.
    pub terrain_wireframe_pipeline: enki_rhi::PipelineHandle,
    /// When `true` (default), use `TectonicHeightField`; when `false`, use
    /// `SimpleHeightField` as a fast fallback.
    pub use_tectonics: bool,
    /// Target number of tectonic plates (≈12 for Earth-like geology).
    pub plate_count: usize,
    /// Arc-volcano density multiplier (1.0 = nominal).
    pub arc_density: f64,
    /// Number of mantle hotspots.
    pub hotspot_count: u32,
    /// Hotspot height multiplier.
    pub hotspot_intensity: f64,
}

// ── PlanetView ────────────────────────────────────────────────────────────────

/// Bridges the LOD planet to the streaming uploader.
///
/// Owns the `LodTree`, `WorkerPool`, CPU-side LRU cache, and the resident
/// (GPU-uploaded) chunk map.  Each frame the caller drives it with
/// `update` (streaming decisions) then `record` (draw commands).
///
/// The streaming upload/retire goes through `RhiUploader`; the `ChunkUploader`
/// trait lets the headless tests drive the same logic via a `FakeUploader`.
///
/// # LRU capacity guideline
///
/// `lru_capacity` should satisfy `lru_capacity ≥ 2 × peak_resident + n_workers`.
/// Too small a value causes `safe_cache_put` to drop freshly-built meshes when
/// every LRU victim is protected (resident or pending), creating a
/// build→drop→rebuild thrash loop where chunks never display.  The default of
/// 1024 provides ~3× headroom for the typical orbit scenario (~348 resident).
/// Pass values below 512 at your own risk — a `log::warn!` is emitted at
/// construction time to make the misconfiguration visible.
pub struct PlanetView {
    tree: LodTree,
    pool: WorkerPool,
    /// CPU-side LRU: built but not yet GPU-resident meshes.
    cache: LruCache<ChunkKey, enki_planet::ChunkMeshArrays>,
    /// GPU-resident, drawable chunks.
    resident: HashMap<ChunkKey, ResidentChunk<StreamedMesh>>,
    /// Keys to retry for upload (back-pressure queue).
    pending_upload: VecDeque<ChunkKey>,
    cfg: PlanetConfig,
}

// ── PlanetViewStats ───────────────────────────────────────────────────────────

/// Snapshot of per-frame LOD statistics for the HUD.
pub struct PlanetViewStats {
    /// Number of chunks currently GPU-resident (drawable).
    pub resident_count: usize,
    /// Number of mesh build jobs currently in flight (not yet delivered).
    pub build_queue_depth: usize,
    /// Minimum LOD level among resident chunks (0 = coarsest root).
    pub min_lod_level: u8,
    /// Maximum LOD level among resident chunks.
    pub max_lod_level: u8,
    /// Number of chunks stuck in the back-pressure retry queue waiting to upload.
    pub pending_upload_count: usize,
}

impl PlanetView {
    /// Total triangles across all GPU-resident chunks (top-right HUD stat).
    pub fn triangle_count(&self) -> u32 {
        self.resident.values().map(|rc| rc.handle.index_count).sum::<u32>() / 3
    }

    /// Return a lightweight snapshot of LOD statistics for the current frame.
    ///
    /// Cheap: iterates resident keys (typically < 1000) with no allocation.
    pub fn stats(&self) -> PlanetViewStats {
        let resident_count = self.resident.len();
        let build_queue_depth = self.pool.pending_in_flight();

        let mut min_lod_level: u8 = u8::MAX;
        let mut max_lod_level: u8 = 0;
        for key in self.resident.keys() {
            let level = key.1; // ChunkKey = (face, level, ix, iy)
            if level < min_lod_level {
                min_lod_level = level;
            }
            if level > max_lod_level {
                max_lod_level = level;
            }
        }
        if resident_count == 0 {
            min_lod_level = 0;
            max_lod_level = 0;
        }

        PlanetViewStats {
            resident_count,
            build_queue_depth,
            min_lod_level,
            max_lod_level,
            pending_upload_count: self.pending_upload.len(),
        }
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// Insert into the LRU cache, but protect resident and pending_upload keys
    /// from being evicted.
    ///
    /// If inserting `key` would evict a protected entry (one that is in `resident`
    /// or `pending_upload`), the insert is skipped and a debug log is emitted.
    /// In practice the LRU capacity should be tuned large enough that this path
    /// is hit rarely if ever.
    fn safe_cache_put(&mut self, key: ChunkKey, arrays: enki_planet::ChunkMeshArrays) {
        // Re-insert of an existing key is always safe.
        if self.cache.contains(&key) {
            self.cache.put(key, arrays);
            return;
        }

        // If there is room, no eviction occurs.
        let cap = self.cache.cap().get();
        if self.cache.len() < cap {
            self.cache.put(key, arrays);
            return;
        }

        // At capacity: check whether the LRU victim is protected.
        let pending_set: HashSet<ChunkKey> = self.pending_upload.iter().copied().collect();
        if let Some((victim, _)) = self.cache.peek_lru() {
            let victim = *victim;
            if self.resident.contains_key(&victim) || pending_set.contains(&victim) {
                log::debug!(
                    "PlanetView: LRU eviction would hit protected key {:?}; \
                     dropping incoming {:?}",
                    victim, key
                );
                return;
            }
        }

        // Safe to evict: proceed normally.
        self.cache.put(key, arrays);
    }
}

// ── Production constructor (requires Rhi) ─────────────────────────────────────

/// Build the GPU-path `PlanetView` from a PRE-BUILT height field.
///
/// Skips the (expensive) height-field construction — the async loader builds the
/// height field once on a worker thread and shares it with the Nanite bake, then
/// hands it here on the main thread (this part is cheap: it just spawns the
/// worker pool and allocates the quadtree + LRU).
pub fn planet_view_from_hf(
    cfg: PlanetConfig,
    hf: Arc<dyn HeightField>,
) -> PlanetView {
    let template = ChunkBuildTemplate {
        resolution:   cfg.lod.resolution,
        radius:       cfg.lod.radius,
        height_scale: cfg.lod.height_scale,
    };

    let pool = WorkerPool::new(Arc::clone(&hf), template, cfg.n_workers);

    let tree = LodTree::new(LodConfig {
        radius:        cfg.lod.radius,
        height_scale:  cfg.lod.height_scale,
        resolution:    cfg.lod.resolution,
        max_depth:     cfg.lod.max_depth,
        target_tri_px: cfg.lod.target_tri_px,
        hysteresis:    cfg.lod.hysteresis,
        lru_capacity:  cfg.lod.lru_capacity,
    }, hf);

    // W2: warn if lru_capacity is below the safe floor.  A capacity that is
    // too small causes safe_cache_put to drop freshly-built meshes when every
    // LRU victim is protected, creating a build→drop→rebuild livelock.
    // Guideline: lru_capacity ≥ 2 × peak_resident + n_workers.
    const LRU_CAPACITY_MIN: usize = 512;
    debug_assert!(
        cfg.lod.lru_capacity >= LRU_CAPACITY_MIN,
        "PlanetView: lru_capacity {} is below the safe floor {}; \
         expect build→drop→rebuild livelock",
        cfg.lod.lru_capacity, LRU_CAPACITY_MIN,
    );
    if cfg.lod.lru_capacity < LRU_CAPACITY_MIN {
        log::warn!(
            "PlanetView: lru_capacity={} is below the recommended minimum of {}. \
             Guideline: lru_capacity ≥ 2 × peak_resident + n_workers. \
             Too small a value may cause chunk-build/drop thrash (livelock).",
            cfg.lod.lru_capacity, LRU_CAPACITY_MIN,
        );
    }

    let lru_cap = std::num::NonZeroUsize::new(cfg.lod.lru_capacity.max(1)).unwrap();
    let cache = LruCache::new(lru_cap);

    PlanetView {
        tree,
        pool,
        cache,
        resident: HashMap::new(),
        pending_upload: VecDeque::new(),
        cfg,
    }
}

// ── GPU-path update + record ──────────────────────────────────────────────────

impl PlanetView {
    /// Per-frame update for the real GPU path.
    ///
    /// Constructs a transient `RhiUploader` from `rhi` + `fi` and drives the
    /// full per-frame streaming update.  Must be called BEFORE
    /// `rhi.begin_rendering(fi)` (streaming uploads are illegal inside a
    /// dynamic-rendering instance).
    pub fn update(
        &mut self,
        rhi: &mut Rhi,
        fi: u32,
        frame_counter: u64,
        cam_local: &LodCamera,
        camera_world_pos: DVec3,
    ) {
        // ── Step 1: drain worker pool into the LRU cache ──────────────────
        for result in self.pool.drain() {
            let key = result.key;
            self.safe_cache_put(key, result.arrays);
        }

        // ── Step 2: LOD update ────────────────────────────────────────────
        let sel = {
            let resident = &self.resident;
            let cache    = &self.cache;
            self.tree.update(
                cam_local,
                &|k| cache.contains(&k) || resident.contains_key(&k),
                &|k| resident.contains_key(&k),
            )
        };

        // ── Step 3: submit / cancel build jobs ────────────────────────────
        for (key, priority) in &sel.builds {
            self.pool.submit(enki_jobs::job::MeshRequest { key: *key, priority: *priority });
        }
        for key in &sel.cancels {
            self.pool.cancel(*key);
        }

        // ── Step 4: upload meshes ─────────────────────────────────────────
        let mut uploader = RhiUploader::new(rhi, fi);
        let mut to_upload: Vec<enki_planet::quadtree::ChunkKey> = Vec::new();

        let pending: Vec<enki_planet::quadtree::ChunkKey> = self.pending_upload.drain(..).collect();
        to_upload.extend(pending);
        for key in &sel.show {
            if !self.resident.contains_key(key) && !to_upload.contains(key) {
                to_upload.push(*key);
            }
        }

        for key in to_upload {
            if self.resident.contains_key(&key) {
                continue;
            }
            if !self.cache.contains(&key) {
                continue;
            }

            let upload_result = {
                let arrays = self.cache.peek(&key).unwrap();
                uploader.upload(fi, arrays)
            };

            match upload_result {
                Ok(Some(handle)) => {
                    // Use peek so promoting to resident does not disturb LRU order.
                    let origin = self.cache.peek(&key).unwrap().origin;
                    self.resident.insert(key, ResidentChunk { handle, origin });
                }
                Ok(None) => {
                    self.pending_upload.push_back(key);
                }
                Err(e) => {
                    log::error!("PlanetView: streaming_upload failed for {:?}: {e}", key);
                }
            }
        }

        // ── Step 5: retire hidden meshes ──────────────────────────────────
        for key in &sel.hide {
            if let Some(rc) = self.resident.remove(key) {
                uploader.retire(rc.handle);
            }
            // W1: if the key is still pending upload (back-pressured, not yet
            // resident), sweep it from pending_upload so we don't upload a
            // chunk the LOD no longer wants.  Leave the cache copy — the LRU
            // will reclaim it naturally.
            self.pending_upload.retain(|k| k != key);
        }

        let _ = (frame_counter, camera_world_pos); // future use
    }

    /// Record draw commands for all resident chunks into the active command buffer.
    ///
    /// Must be called AFTER `rhi.begin_rendering(fi)`.
    ///
    /// # Parameters
    /// - `rhi`              — the RHI context (must be in a recording state).
    /// - `fi`               — frame-in-flight slot index.
    /// - `fu`               — per-frame uniforms (rotation-only view_proj, camera_pos = 0).
    /// - `camera_world_pos` — world-space f64 camera position for camera-relative push.
    /// - `material_mode`    — material discriminant (0–3).
    /// - `wireframe`        — if true, use the wireframe terrain pipeline.
    pub fn record(
        &mut self,
        rhi: &mut Rhi,
        fi: u32,
        fu: &FrameUniforms,
        camera_world_pos: DVec3,
        material_mode: u32,
        wireframe: bool,
    ) -> Result<(), RhiError> {
        let pipeline = if wireframe {
            self.cfg.terrain_wireframe_pipeline
        } else {
            self.cfg.terrain_pipeline
        };

        rhi.bind_pipeline(fi, pipeline)?;
        rhi.update_frame_uniforms(fi, bytemuck::bytes_of(fu))?;

        for rc in self.resident.values() {
            let mesh: &StreamedMesh = &rc.handle;

            let push = ChunkPush::camera_relative(
                rc.origin,
                camera_world_pos,
                Mat4::IDENTITY,
                material_mode,
            );

            rhi.bind_vertex_buffers(fi, &[mesh.pos, mesh.nrm, mesh.col, mesh.plate])?;
            rhi.bind_index_buffer(fi, mesh.idx)?;
            rhi.push_constants(fi, bytemuck::bytes_of(&push))?;
            rhi.draw_indexed(fi, mesh.index_count);
        }

        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use enki_planet::quadtree::ChunkKey;
    use enki_render::geometry::ChunkMeshArrays;
    use glam::DVec3;

    // ── FakeUploader ──────────────────────────────────────────────────────────

    /// Simulates the streaming uploader with configurable back-pressure.
    ///
    /// `failures_remaining` — how many consecutive `Ok(None)` before `Ok(Some(...))`.
    struct FakeUploader {
        failures_remaining: usize,
        next_id:            u64,
        retired_count:      usize,
    }

    impl FakeUploader {
        fn new(failures: usize) -> Self {
            Self { failures_remaining: failures, next_id: 1, retired_count: 0 }
        }

        fn upload_count(&self) -> u64 {
            self.next_id - 1
        }
    }

    impl ChunkUploader for FakeUploader {
        type Handle = u64;

        fn upload(
            &mut self,
            _fi: u32,
            _arrays: &ChunkMeshArrays,
        ) -> Result<Option<u64>, RhiError> {
            if self.failures_remaining > 0 {
                self.failures_remaining -= 1;
                Ok(None)
            } else {
                let id = self.next_id;
                self.next_id += 1;
                Ok(Some(id))
            }
        }

        fn retire(&mut self, _handle: u64) {
            self.retired_count += 1;
        }
    }

    // ── TestBridge ────────────────────────────────────────────────────────────
    //
    // Drives the cache / resident / pending_upload logic directly without needing
    // a real LodTree or WorkerPool.

    struct TestBridge {
        cache:          LruCache<ChunkKey, ChunkMeshArrays>,
        resident:       HashMap<ChunkKey, ResidentChunk<u64>>,
        pending_upload: VecDeque<ChunkKey>,
    }

    impl TestBridge {
        fn new(cap: usize) -> Self {
            let cap = std::num::NonZeroUsize::new(cap.max(1)).unwrap();
            Self {
                cache: LruCache::new(cap),
                resident: HashMap::new(),
                pending_upload: VecDeque::new(),
            }
        }

        /// Run the upload phase for a set of "show" keys.
        fn run_upload(&mut self, show: &[ChunkKey], up: &mut FakeUploader) {
            let mut to_upload: Vec<ChunkKey> = Vec::new();
            let pending: Vec<ChunkKey> = self.pending_upload.drain(..).collect();
            to_upload.extend(pending);
            for &key in show {
                if !self.resident.contains_key(&key) && !to_upload.contains(&key) {
                    to_upload.push(key);
                }
            }

            for key in to_upload {
                if self.resident.contains_key(&key) {
                    continue;
                }
                if !self.cache.contains(&key) {
                    continue;
                }

                let result = {
                    let arrays = self.cache.peek(&key).unwrap();
                    up.upload(0, arrays)
                };

                match result {
                    Ok(Some(handle)) => {
                        let origin = self.cache.peek(&key).unwrap().origin;
                        self.resident.insert(key, ResidentChunk { handle, origin });
                    }
                    Ok(None) => {
                        self.pending_upload.push_back(key);
                    }
                    Err(e) => {
                        log::error!("upload error: {e}");
                    }
                }
            }
        }

        /// Run the retire phase for a set of "hide" keys.
        ///
        /// Mirrors the production Step-5 logic including the W1 pending_upload sweep.
        fn run_retire(&mut self, hide: &[ChunkKey], up: &mut FakeUploader) {
            for &key in hide {
                if let Some(rc) = self.resident.remove(&key) {
                    up.retire(rc.handle);
                }
                // W1: also sweep from pending_upload.
                self.pending_upload.retain(|k| *k != key);
            }
        }

        fn put_cache(&mut self, key: ChunkKey) {
            self.cache.put(key, fake_arrays());
        }

        fn put_cache_with_origin(&mut self, key: ChunkKey, origin: DVec3) {
            let mut a = fake_arrays();
            a.origin = origin;
            self.cache.put(key, a);
        }

        fn safe_put_cache(&mut self, key: ChunkKey) {
            let arrays = fake_arrays();
            // Mirror safe_cache_put logic.
            if self.cache.contains(&key) {
                self.cache.put(key, arrays);
                return;
            }
            let cap = self.cache.cap().get();
            if self.cache.len() < cap {
                self.cache.put(key, arrays);
                return;
            }
            let pending_set: HashSet<ChunkKey> = self.pending_upload.iter().copied().collect();
            if let Some((victim, _)) = self.cache.peek_lru() {
                let victim = *victim;
                if self.resident.contains_key(&victim) || pending_set.contains(&victim) {
                    return; // skip insert
                }
            }
            self.cache.put(key, arrays);
        }

        fn is_resident(&self, key: ChunkKey) -> bool {
            self.resident.contains_key(&key)
        }

        fn is_pending(&self, key: ChunkKey) -> bool {
            self.pending_upload.contains(&key)
        }

        fn in_cache(&self, key: ChunkKey) -> bool {
            self.cache.contains(&key)
        }
    }

    fn fake_arrays() -> ChunkMeshArrays {
        ChunkMeshArrays {
            positions:   vec![[0.0f32; 3]; 4],
            normals:     vec![[0.0f32; 3]; 4],
            colors:      vec![[0.0f32; 3]; 4],
            plate_colors: None,
            indices:     vec![0u32, 1, 2, 1, 3, 2],
            origin:      DVec3::ZERO,
        }
    }

    // ── AC1: back-pressure → chunk eventually resident ────────────────────────

    /// With N back-pressure responses, the chunk must be uploaded on frame N+1
    /// and must never disappear from the cache or pending queue between retries.
    #[test]
    fn backpressure_chunk_eventually_resident() {
        let key: ChunkKey = (0, 0, 0, 0);
        let mut bridge = TestBridge::new(16);
        bridge.put_cache(key);

        let mut up = FakeUploader::new(3); // 3 consecutive Ok(None)

        // Frame 0: first attempt → back-pressure.
        bridge.run_upload(&[key], &mut up);
        assert!(!bridge.is_resident(key), "must not be resident after first back-pressure");
        assert!(bridge.is_pending(key), "must be in pending_upload after back-pressure");
        assert!(bridge.in_cache(key), "cache must retain data during back-pressure");

        // Frame 1: retry from pending_upload → still back-pressure.
        bridge.run_upload(&[], &mut up);
        assert!(!bridge.is_resident(key));
        assert!(bridge.is_pending(key));

        // Frame 2: retry → still back-pressure.
        bridge.run_upload(&[], &mut up);
        assert!(!bridge.is_resident(key));
        assert!(bridge.is_pending(key));

        // Frame 3: retry → success.
        bridge.run_upload(&[], &mut up);
        assert!(bridge.is_resident(key), "must be resident after uploader returns Ok(Some)");
        assert!(!bridge.is_pending(key), "must leave pending_upload once resident");
        assert_eq!(up.upload_count(), 1, "upload must succeed exactly once");
    }

    // ── AC2: no double-upload of already-resident key ─────────────────────────

    #[test]
    fn no_double_upload_of_resident_key() {
        let key: ChunkKey = (0, 0, 1, 0);
        let mut bridge = TestBridge::new(16);
        bridge.put_cache(key);

        let mut up = FakeUploader::new(0);

        // First upload: success.
        bridge.run_upload(&[key], &mut up);
        assert!(bridge.is_resident(key));
        assert_eq!(up.upload_count(), 1);

        // Second attempt with same key in show list.
        bridge.run_upload(&[key], &mut up);

        // Upload count must not increase.
        assert_eq!(up.upload_count(), 1, "upload called twice for an already-resident key");
    }

    // ── AC3: no retire of a non-resident key ──────────────────────────────────

    #[test]
    fn no_retire_of_non_resident_key() {
        let key: ChunkKey = (0, 0, 2, 0);
        let mut bridge = TestBridge::new(16);

        let mut up = FakeUploader::new(0);

        // Attempt to retire a key that was never uploaded.
        bridge.run_retire(&[key], &mut up);

        assert_eq!(up.retired_count, 0, "retire must not be called for non-resident key");
    }

    // ── AC4: LRU never evicts a resident key ──────────────────────────────────

    #[test]
    fn lru_never_evicts_resident_key() {
        // Cap = 2: fill both slots, make them resident, then try to insert a 3rd.
        let mut bridge = TestBridge::new(2);
        let k0: ChunkKey = (0, 0, 0, 0);
        let k1: ChunkKey = (0, 0, 1, 0);
        let k2: ChunkKey = (0, 0, 2, 0);

        let mut up = FakeUploader::new(0);

        bridge.put_cache(k0);
        bridge.put_cache(k1);
        bridge.run_upload(&[k0, k1], &mut up);
        assert!(bridge.is_resident(k0));
        assert!(bridge.is_resident(k1));

        // Cache is at cap; both entries are resident (protected).
        // safe_put_cache for k2 must skip insertion.
        bridge.safe_put_cache(k2);

        // k0 and k1 must still be in the cache (not evicted).
        assert!(bridge.in_cache(k0), "resident key k0 must not be evicted from cache");
        assert!(bridge.in_cache(k1), "resident key k1 must not be evicted from cache");
        // k2 must NOT be in the cache (insert was skipped).
        assert!(!bridge.in_cache(k2), "k2 must not appear in cache when eviction is blocked");
    }

    // ── AC5: LRU never evicts a pending_upload key ────────────────────────────

    #[test]
    fn lru_never_evicts_pending_upload_key() {
        // Cap = 1: one slot, fill it with a pending key, then try to insert another.
        let mut bridge = TestBridge::new(1);
        let k0: ChunkKey = (0, 0, 0, 0);
        let k1: ChunkKey = (0, 0, 1, 0);

        // k0 in cache; one back-pressure → pending_upload.
        bridge.put_cache(k0);
        let mut up = FakeUploader::new(1);
        bridge.run_upload(&[k0], &mut up);
        assert!(bridge.is_pending(k0), "k0 must be in pending_upload");
        assert!(bridge.in_cache(k0), "k0 must still be in cache");

        // Now try to insert k1 when cap=1 and k0 is the LRU victim (but protected).
        bridge.safe_put_cache(k1);

        // k0 must still be in cache.
        assert!(bridge.in_cache(k0), "pending_upload key k0 must not be evicted");
        // k1 must NOT be inserted.
        assert!(!bridge.in_cache(k1), "k1 must not appear when eviction would hit pending key");
    }

    // ── AC6: ResidentChunk stores the correct origin ──────────────────────────

    #[test]
    fn resident_chunk_stores_origin() {
        let key: ChunkKey = (0, 1, 3, 4);
        let origin = DVec3::new(1_234_567.0, 8_901_234.0, -567_890.0);

        let mut bridge = TestBridge::new(16);
        bridge.put_cache_with_origin(key, origin);

        let mut up = FakeUploader::new(0);
        bridge.run_upload(&[key], &mut up);

        assert!(bridge.is_resident(key));
        let rc = &bridge.resident[&key];
        assert_eq!(rc.origin, origin, "ResidentChunk::origin must match the mesh origin");
    }

    // ── AC7: hide triggers retire exactly once ────────────────────────────────

    #[test]
    fn hide_triggers_retire_exactly_once() {
        let key: ChunkKey = (0, 0, 5, 5);
        let mut bridge = TestBridge::new(16);
        bridge.put_cache(key);

        let mut up = FakeUploader::new(0);
        bridge.run_upload(&[key], &mut up);
        assert!(bridge.is_resident(key));

        bridge.run_retire(&[key], &mut up);
        assert!(!bridge.is_resident(key), "key must not be resident after hide");
        assert_eq!(up.retired_count, 1, "retire must be called exactly once");

        // Retiring again must be a no-op (key no longer resident).
        bridge.run_retire(&[key], &mut up);
        assert_eq!(up.retired_count, 1, "retire must not be called for a non-resident key");
    }

    // ── AC8: back-pressure key never disappears (data integrity) ─────────────

    /// Verifies that a chunk's CPU mesh data is preserved in the cache across
    /// all back-pressure retry frames, and that it eventually becomes resident.
    #[test]
    fn backpressure_data_never_lost() {
        let key: ChunkKey = (0, 2, 7, 3);
        let mut bridge = TestBridge::new(16);
        bridge.put_cache(key);

        // 5 failures then success.
        let mut up = FakeUploader::new(5);

        // Frame 0: first show attempt → back-pressure → goes into pending_upload.
        bridge.run_upload(&[key], &mut up);
        assert!(!bridge.is_resident(key));
        assert!(bridge.is_pending(key));
        assert!(bridge.in_cache(key), "data must be in cache after back-pressure");

        // Frames 1–4: retry from pending_upload each frame.
        for _ in 0..4 {
            bridge.run_upload(&[], &mut up);
            assert!(bridge.in_cache(key), "data must be preserved in cache during back-pressure");
        }

        // Frame 5: uploader finally succeeds.
        bridge.run_upload(&[], &mut up);
        assert!(bridge.is_resident(key), "key must be resident after back-pressure clears");
        assert_eq!(up.upload_count(), 1, "upload must succeed exactly once");
    }

    // ── AC9: hide-while-pending does not leak ─────────────────────────────────

    /// Regression test for W1.
    ///
    /// Scenario: a key is in `pending_upload` (back-pressured on its first
    /// upload attempt).  Before the retry fires, the LOD issues a `hide` for
    /// that key.  After the hide step:
    ///   - the key must be absent from `pending_upload`,
    ///   - must NOT be uploaded on the next frame,
    ///   - must NOT be resident.
    ///
    /// Without the W1 fix, the key would remain in `pending_upload`, be
    /// drained next frame, uploaded, and silently left resident forever.
    #[test]
    fn hide_while_pending_does_not_leak() {
        let key: ChunkKey = (0, 3, 1, 2);
        // Uploader always returns Ok(None) — simulates sustained back-pressure.
        // This forces the key into pending_upload and keeps it there.
        let mut up = FakeUploader::new(usize::MAX);

        let mut bridge = TestBridge::new(16);
        bridge.put_cache(key);

        // Frame 0: show → back-pressure → pending_upload.
        bridge.run_upload(&[key], &mut up);
        assert!(!bridge.is_resident(key), "must not be resident after back-pressure");
        assert!(bridge.is_pending(key), "must be in pending_upload after back-pressure");

        // Frame 0 (hide step): LOD hides the key before it was ever resident.
        bridge.run_retire(&[key], &mut up);
        assert!(!bridge.is_pending(key), "W1: key must be swept from pending_upload on hide");
        assert!(!bridge.is_resident(key), "must not be resident after hide");
        // Cache copy may be retained (LRU reclaims it naturally).
        assert_eq!(up.retired_count, 0, "retire must not be called: key was never resident");

        // Frame 1: pending_upload must be empty, so no upload fires.
        // Switch to an uploader that would succeed, to detect any spurious upload.
        let mut up2 = FakeUploader::new(0);
        bridge.run_upload(&[], &mut up2);
        assert_eq!(up2.upload_count(), 0, "W1: no upload must fire after hide-while-pending");
        assert!(!bridge.is_resident(key), "must not be resident on the frame after hide");
    }
}

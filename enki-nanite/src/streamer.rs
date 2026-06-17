//! Quadtree-driven residency + background cluster-DAG baking (N5c).
//!
//! This is the streaming brain: it reuses `enki-planet`'s [`LodTree`] (the exact
//! quadtree that drives the grid-mesh terrain) to decide which cube-sphere nodes
//! should be resident, then bakes a **Nanite cluster-DAG per node** on a small
//! background pool instead of a grid mesh. Whenever the visible set changes, the
//! resident assets are handed to [`crate::render::NaniteRenderer::update_geometry`].
//!
//! Two LOD mechanisms compose:
//!   * the **quadtree** streams finer/coarser *nodes* by distance (the virtual
//!     high-res base mesh), and
//!   * the **Nanite cull** picks the crack-free cluster cut *within* each resident
//!     node's DAG every frame.
//!
//! Together they give continuous, sub-pixel LOD with far less popping than the
//! quadtree's discrete whole-chunk swaps.
//!
//! This module is GPU-agnostic (pure CPU residency + baking); the GPU upload is
//! the renderer's job, keeping the crate's GPU/CPU split clean and the whole
//! thing deletable.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{unbounded, Receiver, Sender};
use glam::DVec3;

use enki_planet::height::HeightField;
use enki_planet::lod::{LodCamera, LodConfig, LodTree};
use enki_planet::quadtree::ChunkKey;

use crate::bake::{bake_patch, PatchParams};
use crate::cluster::ClusterAsset;

/// Tunables for the streaming Nanite terrain.
#[derive(Clone, Copy, Debug)]
pub struct StreamConfig {
    pub radius: f64,
    pub height_scale: f64,
    /// Grid resolution each node is tessellated at — its DAG's LOD0 source.
    pub node_resolution: u32,
    pub max_depth: u8,
    /// Target screen-space triangle size (px) driving quadtree split/merge.
    pub target_tri_px: f32,
    pub hysteresis: f32,
    /// Soft cap on cached (resident + ready) node DAGs.
    pub lru_capacity: usize,
    pub workers: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            radius: 50_000.0,
            height_scale: 1_200.0,
            node_resolution: 32,
            max_depth: 12,
            target_tri_px: 8.0,
            hysteresis: 0.25,
            lru_capacity: 1_024,
            workers: 4,
        }
    }
}

// ── background bake pool ──────────────────────────────────────────────────────

/// A tiny thread pool that bakes one node's cluster-DAG per job. Self-contained
/// (no shared `enki-jobs` coupling) so the crate stays deletable.
struct BakePool {
    /// `Option` so `Drop` can close the channel *before* joining workers.
    tx: Option<Sender<ChunkKey>>,
    rx: Receiver<(ChunkKey, ClusterAsset)>,
    workers: Vec<JoinHandle<()>>,
}

impl BakePool {
    fn new(
        hf: Arc<dyn HeightField>,
        radius: f64,
        height_scale: f64,
        resolution: u32,
        n_workers: usize,
    ) -> Self {
        let (tx_job, rx_job) = unbounded::<ChunkKey>();
        let (tx_done, rx_done) = unbounded::<(ChunkKey, ClusterAsset)>();
        let mut workers = Vec::with_capacity(n_workers.max(1));
        for _ in 0..n_workers.max(1) {
            let rx_job = rx_job.clone();
            let tx_done = tx_done.clone();
            let hf = Arc::clone(&hf);
            workers.push(std::thread::spawn(move || {
                // Exits when the job channel closes (all senders dropped).
                while let Ok((face, level, ix, iy)) = rx_job.recv() {
                    let params = PatchParams { face, level, ix, iy, resolution, radius, height_scale };
                    let asset = bake_patch(&params, hf.as_ref());
                    if tx_done.send(((face, level, ix, iy), asset)).is_err() {
                        break; // streamer gone
                    }
                }
            }));
        }
        Self { tx: Some(tx_job), rx: rx_done, workers }
    }

    fn submit(&self, key: ChunkKey) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(key);
        }
    }

    fn drain(&self) -> Vec<(ChunkKey, ClusterAsset)> {
        self.rx.try_iter().collect()
    }
}

impl Drop for BakePool {
    fn drop(&mut self) {
        // Close the job channel first so workers observe disconnect and exit,
        // then join (joining before this would deadlock on a live sender).
        self.tx = None;
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

// ── streamer ──────────────────────────────────────────────────────────────────

/// Owns the quadtree, the bake pool, and the CPU-side residency maps.
pub struct NaniteStreamer {
    tree: LodTree,
    pool: BakePool,
    cfg: StreamConfig,
    /// Keys submitted to the bake pool, not yet drained.
    submitted: HashSet<ChunkKey>,
    /// Baked DAGs not currently shown (a small re-show cache).
    ready: HashMap<ChunkKey, ClusterAsset>,
    /// Baked DAGs currently shown — the set handed to the GPU.
    resident: HashMap<ChunkKey, ClusterAsset>,
    /// Set when `resident` changes; the app re-uploads while true.
    dirty: bool,
    /// Last camera position (planet-local), for nearest-first upload ordering.
    cam_pos: DVec3,
}

impl NaniteStreamer {
    pub fn new(hf: Arc<dyn HeightField>, cfg: StreamConfig) -> Self {
        let lod = LodConfig {
            radius: cfg.radius,
            height_scale: cfg.height_scale,
            resolution: cfg.node_resolution,
            max_depth: cfg.max_depth,
            target_tri_px: cfg.target_tri_px,
            hysteresis: cfg.hysteresis,
            lru_capacity: cfg.lru_capacity,
        };
        let tree = LodTree::new(lod, Arc::clone(&hf));
        let pool = BakePool::new(
            hf,
            cfg.radius,
            cfg.height_scale,
            cfg.node_resolution,
            cfg.workers,
        );
        Self {
            tree,
            pool,
            cfg,
            submitted: HashSet::new(),
            ready: HashMap::new(),
            resident: HashMap::new(),
            // Start dirty so the (empty) initial set seeds all renderer slots.
            dirty: true,
            cam_pos: DVec3::ZERO,
        }
    }

    /// Advance residency for one frame: collect finished bakes, ask the quadtree
    /// what to build/show/hide, and submit new bakes. Cheap — no GPU work.
    pub fn update(&mut self, cam: &LodCamera) {
        self.cam_pos = cam.local_pos;

        // 1. Collect freshly-baked node DAGs into the ready cache.
        for (key, asset) in self.pool.drain() {
            self.submitted.remove(&key);
            self.ready.insert(key, asset);
        }

        // 2. Ask the quadtree for this frame's actions. Borrow the residency maps
        //    by field (disjoint from `self.tree`) so the closures don't conflict
        //    with the `&mut self.tree` the call needs.
        let ready = &self.ready;
        let resident = &self.resident;
        let resident_fn = |k: ChunkKey| ready.contains_key(&k) || resident.contains_key(&k);
        let visible_fn = |k: ChunkKey| resident.contains_key(&k);
        let sel = self.tree.update(cam, &resident_fn, &visible_fn);

        // 3. Apply actions.
        for k in sel.cancels {
            // Best-effort: forget we submitted it (a running bake can't be stopped;
            // its result lands in `ready` and is pruned if never shown).
            self.submitted.remove(&k);
        }
        for (k, _prio) in sel.builds {
            if !self.submitted.contains(&k)
                && !self.ready.contains_key(&k)
                && !self.resident.contains_key(&k)
            {
                self.submitted.insert(k);
                self.pool.submit(k);
            }
        }
        for k in sel.show {
            if let Some(a) = self.ready.remove(&k) {
                self.resident.insert(k, a);
                self.dirty = true;
            }
        }
        for k in sel.hide {
            if let Some(a) = self.resident.remove(&k) {
                self.ready.insert(k, a); // keep for fast re-show
                self.dirty = true;
            }
        }

        // 4. Bound the re-show cache (resident is never pruned — it's on screen).
        while self.ready.len() > self.cfg.lru_capacity {
            let Some(k) = self.ready.keys().next().copied() else { break };
            self.ready.remove(&k);
        }
    }

    /// Take and clear the dirty flag (true when the resident set changed).
    pub fn take_dirty(&mut self) -> bool {
        std::mem::replace(&mut self.dirty, false)
    }

    /// The currently-shown node DAGs, ordered **nearest-first** so that if the
    /// renderer's pool overflows it drops only the farthest nodes. Pass straight
    /// to `update_geometry`.
    pub fn visible_assets(&self) -> Vec<&ClusterAsset> {
        let cam = self.cam_pos;
        let mut v: Vec<&ClusterAsset> = self.resident.values().collect();
        v.sort_by(|a, b| {
            let da = (a.patch_origin - cam).length_squared();
            let db = (b.patch_origin - cam).length_squared();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        v
    }

    /// Number of resident (on-screen) nodes.
    pub fn resident_nodes(&self) -> usize {
        self.resident.len()
    }

    /// Total clusters across resident nodes (for the pool-capacity diagnostic).
    pub fn resident_clusters(&self) -> usize {
        self.resident.values().map(|a| a.cluster_count()).sum()
    }

    /// Nodes baking right now.
    pub fn pending(&self) -> usize {
        self.submitted.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use enki_planet::noise::Noise3D;
    use enki_planet::simple_height::SimpleHeightField;
    use glam::{DVec3, Mat4};

    fn hf() -> Arc<dyn HeightField> {
        Arc::new(SimpleHeightField { noise: Noise3D::new(7) })
    }

    fn cam(pos: DVec3) -> LodCamera {
        // A loose view-projection; exact framing doesn't matter for residency
        // convergence in this headless test (frustum just needs to be finite).
        let view = Mat4::look_at_rh(pos.as_vec3(), glam::Vec3::ZERO, glam::Vec3::Y);
        let proj = Mat4::perspective_rh(1.0, 1.0, 0.1, 1e9);
        LodCamera {
            local_pos: pos,
            v_fov_rad: 1.0,
            screen_h_px: 1080.0,
            view_proj_local: proj * view,
        }
    }

    #[test]
    fn converges_to_a_resident_set_near_camera() {
        let cfg = StreamConfig { workers: 2, ..Default::default() };
        let mut s = NaniteStreamer::new(hf(), cfg);
        let c = cam(DVec3::new(0.0, 0.0, cfg.radius * 1.5));

        // Pump frames + drain bakes until something becomes resident (bakes run
        // on background threads, so a few iterations are needed).
        let mut shown = 0;
        for _ in 0..2000 {
            s.update(&c);
            shown = s.resident_nodes();
            if shown > 0 && s.pending() == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(shown > 0, "expected at least one resident node near the camera");
        assert!(
            s.resident_clusters() > 0,
            "resident nodes must carry clusters"
        );
    }

    #[test]
    fn initial_state_is_dirty_then_clears() {
        let mut s = NaniteStreamer::new(hf(), StreamConfig::default());
        assert!(s.take_dirty(), "fresh streamer seeds the renderer (dirty)");
        assert!(!s.take_dirty(), "dirty clears after being taken");
    }
}

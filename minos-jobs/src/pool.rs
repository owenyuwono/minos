//! Thread pool that executes `ChunkBuildTemplate` jobs.
//!
//! # Architecture
//!
//! Workers share a `Mutex<SharedState>` that holds the priority queue and all
//! tracking state.  A `Condvar` wakes a sleeping worker when new work is
//! pushed.  Workers dequeue the highest-priority item under the lock, release
//! it, call `build_chunk`, then push the result into a crossbeam channel that
//! `drain()` reads non-blockingly.
//!
//! ```text
//!  submit()
//!    └─ lock(state) → push queue → condvar.notify_one()
//!
//!  worker loop
//!    ├─ lock(state) → pop queue → unlock
//!    ├─ build_chunk(&params, &*hf)     ← lock NOT held here
//!    └─ tx_result.send(MeshResult)
//!
//!  drain()
//!    └─ try_recv() × n → filter stale + cancelled → return Vec
//! ```
//!
//! ## Dedup
//!
//! `pending_keys`  — keys in the priority queue (not yet dispatched).
//! `inflight_keys` — keys currently being processed by a worker thread.
//! Submitting a key present in either set is a no-op.
//!
//! ## Cancel
//!
//! `cancel(key)` removes from the pending queue if present.  If the key is
//! in-flight it is added to `cancelled_keys`; `drain()` drops any result whose
//! key appears there and clears the marker.
//!
//! ## Generation
//!
//! `bump_generation()` increments a counter.  Each `WorkOrder` is stamped with
//! the generation current at dispatch time.  `drain()` drops results whose
//! generation differs from the current one.

use std::collections::HashSet;
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

use crossbeam_channel::{unbounded, Receiver};

use minos_planet::height::HeightField;
use minos_planet::mesher::{build_chunk, ChunkBuildParams};
use minos_planet::quadtree::ChunkKey;

use crate::job::{ChunkBuildTemplate, MeshRequest, MeshResult};
use crate::queue::PriorityQueue;

// ---------------------------------------------------------------------------
// Work order: pool → worker (via shared queue, not a separate channel)
// ---------------------------------------------------------------------------

struct WorkOrder {
    key: ChunkKey,
    generation: u64,
}

// ---------------------------------------------------------------------------
// Shared mutable state (behind a Mutex)
// ---------------------------------------------------------------------------

struct SharedState {
    /// Priority queue of not-yet-dispatched requests.
    queue: PriorityQueue,
    /// Keys currently in the queue (for O(1) dedup with inflight).
    pending_keys: HashSet<ChunkKey>,
    /// Keys currently being processed by a worker thread.
    inflight_keys: HashSet<ChunkKey>,
    /// Keys that were cancelled while in-flight; their result should be dropped.
    cancelled_keys: HashSet<ChunkKey>,
    /// Current generation; bumping invalidates all older in-flight results.
    generation: u64,
    /// Set to `true` when the pool is being dropped so workers exit.
    shutdown: bool,
}

impl SharedState {
    fn new() -> Self {
        SharedState {
            queue: PriorityQueue::new(),
            pending_keys: HashSet::new(),
            inflight_keys: HashSet::new(),
            cancelled_keys: HashSet::new(),
            generation: 0,
            shutdown: false,
        }
    }

    fn pending_in_flight(&self) -> usize {
        self.pending_keys.len() + self.inflight_keys.len()
    }

    /// Dequeue the highest-priority request and register it as in-flight.
    /// Returns `None` if the queue is empty.
    fn dequeue(&mut self) -> Option<WorkOrder> {
        let req = self.queue.pop()?;
        let key = req.key;
        self.pending_keys.remove(&key);
        self.inflight_keys.insert(key);
        Some(WorkOrder { key, generation: self.generation })
    }
}

// ---------------------------------------------------------------------------
// WorkerPool
// ---------------------------------------------------------------------------

/// A pool of worker threads that build chunk meshes in the background.
pub struct WorkerPool {
    /// Shared queue + tracking state.
    state: Arc<Mutex<SharedState>>,
    /// Condvar that wakes workers when work is pushed or shutdown begins.
    condvar: Arc<Condvar>,
    /// Channel from which `drain()` collects completed results.
    rx_result: Receiver<MeshResult>,
    /// Worker join handles — joined when `WorkerPool` is dropped.
    workers: Vec<thread::JoinHandle<()>>,
}

impl WorkerPool {
    /// Spawn `n_workers` threads, each sharing `hf` and `build`.
    pub fn new(
        hf: Arc<dyn HeightField>,
        build: ChunkBuildTemplate,
        n_workers: usize,
    ) -> Self {
        let state = Arc::new(Mutex::new(SharedState::new()));
        let condvar = Arc::new(Condvar::new());
        let (tx_result, rx_result) = unbounded::<MeshResult>();
        let build = Arc::new(build);

        let mut workers = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let hf = Arc::clone(&hf);
            let build = Arc::clone(&build);
            let state = Arc::clone(&state);
            let condvar = Arc::clone(&condvar);
            let tx_result = tx_result.clone();

            let handle = thread::spawn(move || {
                loop {
                    // Wait for work (or shutdown).
                    let order = {
                        let mut s = state.lock().unwrap();
                        loop {
                            if s.shutdown {
                                return;
                            }
                            if let Some(o) = s.dequeue() {
                                break o;
                            }
                            s = condvar.wait(s).unwrap();
                        }
                    };

                    let key = order.key;
                    let (face, level, ix, iy) = key;
                    let params = ChunkBuildParams {
                        face,
                        level,
                        ix,
                        iy,
                        resolution: build.resolution,
                        radius: build.radius,
                        height_scale: build.height_scale,
                    };

                    // Build the mesh — lock NOT held here.
                    let arrays = build_chunk(&params, &*hf);

                    let result = MeshResult { key, generation: order.generation, arrays };

                    // Send the result.  We do NOT remove the key from
                    // `inflight_keys` here.  Instead, `drain()` removes it
                    // when it processes each result from the channel.  This
                    // keeps `inflight_keys` as the canonical "result is
                    // somewhere between worker and drain()" set, so that
                    // `cancel()` can reliably set a discard marker for any key
                    // not yet observed by `drain()`.
                    let _ = tx_result.send(result);
                }
            });

            workers.push(handle);
        }

        WorkerPool { state, condvar, rx_result, workers }
    }

    /// Enqueue a mesh-build request.
    ///
    /// No-op if the key is already pending or in-flight.
    pub fn submit(&self, req: MeshRequest) {
        let key = req.key;
        {
            let mut s = self.state.lock().unwrap();
            if s.pending_keys.contains(&key) || s.inflight_keys.contains(&key) {
                return;
            }
            s.pending_keys.insert(key);
            s.queue.push(req);
        }
        // Wake one idle worker.
        self.condvar.notify_one();
    }

    /// Cancel a pending or in-flight job (best-effort).
    ///
    /// If the key is still pending it is removed from the queue and will never
    /// be dispatched.  If it is already in-flight the worker cannot be
    /// interrupted, but its result will be silently dropped by `drain()`.
    pub fn cancel(&self, key: ChunkKey) {
        let mut s = self.state.lock().unwrap();
        if s.pending_keys.remove(&key) {
            s.queue.remove(key);
        } else if s.inflight_keys.contains(&key) {
            s.cancelled_keys.insert(key);
        }
    }

    /// Drain all completed results since the last call (non-blocking).
    ///
    /// Drops results that are stale (wrong generation) or cancelled.
    ///
    /// Also removes each result's key from `inflight_keys` so that
    /// `pending_in_flight()` reaches 0 once all results are collected and
    /// so that `cancel()` can use `inflight_keys` as the canonical
    /// "not yet observed by drain" set.
    pub fn drain(&self) -> Vec<MeshResult> {
        let mut out = Vec::new();
        while let Ok(result) = self.rx_result.try_recv() {
            let discard = {
                let mut s = self.state.lock().unwrap();
                // Remove from inflight regardless — we're processing this result now.
                s.inflight_keys.remove(&result.key);
                let is_stale = result.generation != s.generation;
                let is_cancelled = s.cancelled_keys.remove(&result.key);
                is_stale || is_cancelled
            };
            if !discard {
                out.push(result);
            }
        }
        out
    }

    /// Monotonically increasing generation counter.
    pub fn generation(&self) -> u64 {
        self.state.lock().unwrap().generation
    }

    /// Increment the generation counter, causing all currently in-flight
    /// results to be treated as stale and dropped by `drain()`.
    pub fn bump_generation(&self) {
        let mut s = self.state.lock().unwrap();
        s.generation += 1;
    }

    /// Number of jobs currently queued or running.
    pub fn pending_in_flight(&self) -> usize {
        self.state.lock().unwrap().pending_in_flight()
    }
}

impl Drop for WorkerPool {
    fn drop(&mut self) {
        {
            let mut s = self.state.lock().unwrap();
            s.shutdown = true;
        }
        // Wake all workers so they observe shutdown=true and exit.
        self.condvar.notify_all();
        for handle in self.workers.drain(..) {
            let _ = handle.join();
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    use minos_planet::simple_height::SimpleHeightField;
    use minos_planet::noise::Noise3D;

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Tiny, trivially fast HeightField stub — returns a constant so each
    /// build_chunk invocation costs only tessellation math, no noise eval.
    struct FlatHf;
    impl HeightField for FlatHf {
        fn height(&self, _dir: glam::DVec3, _level: u8) -> f64 {
            0.0
        }
    }

    fn flat_hf() -> Arc<dyn HeightField> {
        Arc::new(FlatHf)
    }

    fn simple_hf() -> Arc<dyn HeightField> {
        Arc::new(SimpleHeightField { noise: Noise3D::new(42) })
    }

    fn template() -> ChunkBuildTemplate {
        ChunkBuildTemplate { resolution: 8, radius: 6_371_000.0, height_scale: 8848.0 }
    }

    /// Generate `n` distinct chunk keys on face 0.
    fn keys(n: u32) -> Vec<ChunkKey> {
        (0..n).map(|i| (0u8, 1u8, i, 0u32)).collect()
    }

    /// Drain in a polling loop until `n` results arrive or the deadline elapses.
    fn drain_until(pool: &WorkerPool, n: usize, timeout: Duration) -> Vec<MeshResult> {
        let deadline = Instant::now() + timeout;
        let mut collected: Vec<MeshResult> = Vec::new();
        while collected.len() < n && Instant::now() < deadline {
            collected.extend(pool.drain());
            if collected.len() < n {
                thread::sleep(Duration::from_millis(2));
            }
        }
        collected
    }

    // -----------------------------------------------------------------------
    // AC1: submit 100 keys; drain until all arrive; each exactly once.
    // -----------------------------------------------------------------------

    #[test]
    fn all_100_keys_delivered_exactly_once() {
        let pool = WorkerPool::new(simple_hf(), template(), 4);
        let ks = keys(100);

        for &k in &ks {
            pool.submit(MeshRequest { key: k, priority: 0.0 });
        }

        let results = drain_until(&pool, 100, Duration::from_secs(60));
        assert_eq!(results.len(), 100, "expected 100 results, got {}", results.len());

        let mut counts: HashMap<ChunkKey, usize> = HashMap::new();
        for r in &results {
            *counts.entry(r.key).or_insert(0) += 1;
        }
        for &k in &ks {
            let c = counts.get(&k).copied().unwrap_or(0);
            assert_eq!(c, 1, "key {:?} appeared {} times", k, c);
        }
    }

    // -----------------------------------------------------------------------
    // AC2: dedup — submitting the same key twice yields one result.
    // -----------------------------------------------------------------------

    #[test]
    fn dedup_same_key_yields_one_result() {
        let pool = WorkerPool::new(flat_hf(), template(), 2);
        let k: ChunkKey = (0, 1, 99, 99);

        pool.submit(MeshRequest { key: k, priority: 1.0 });
        pool.submit(MeshRequest { key: k, priority: 2.0 }); // duplicate

        let results = drain_until(&pool, 1, Duration::from_secs(10));
        // Give a short extra window to see if a second result sneaks through.
        thread::sleep(Duration::from_millis(100));
        let extra = pool.drain();

        assert_eq!(results.len() + extra.len(), 1,
            "expected exactly 1 result for deduped key, got {}",
            results.len() + extra.len());
    }

    // -----------------------------------------------------------------------
    // AC3: cancel — pending cancel never delivered;
    //               in-flight cancel dropped silently (not surfaced by drain).
    // -----------------------------------------------------------------------

    #[test]
    fn cancel_pending_never_delivered() {
        // Use 0 workers so nothing runs until we un-block.
        // But WorkerPool requires at least some capacity — use 1 worker and
        // fill it with a "decoy" job first, then cancel the rest while pending.
        //
        // Strategy: submit many keys; immediately cancel all but one before
        // any worker can pick them up (workers can't run without the lock).
        let pool = WorkerPool::new(flat_hf(), template(), 1);

        // Key 0 is the live one; keys 1-9 will be cancelled.
        let live: ChunkKey = (0, 1, 0, 0);
        let to_cancel: Vec<ChunkKey> = (1u32..10).map(|i| (0u8, 1u8, i, 0u32)).collect();

        pool.submit(MeshRequest { key: live, priority: 0.0 });
        for &k in &to_cancel {
            pool.submit(MeshRequest { key: k, priority: 0.0 });
        }
        // Cancel all the extras.
        for &k in &to_cancel {
            pool.cancel(k);
        }

        // Drain until the single live key arrives (or timeout).
        let results = drain_until(&pool, 1, Duration::from_secs(10));
        // Extra sleep to let any incorrectly delivered cancelled results arrive.
        thread::sleep(Duration::from_millis(100));
        let extra = pool.drain();

        // Only the live key should be delivered.
        let all: Vec<_> = results.iter().chain(extra.iter()).collect();
        assert!(all.iter().all(|r| r.key == live),
            "cancelled key appeared in drain: {:?}", all.iter().map(|r| r.key).collect::<Vec<_>>());
    }

    #[test]
    fn cancel_inflight_result_not_surfaced() {
        // Let a job start in-flight, then cancel it; drain should not surface it.
        // We can't stop the worker, but we can mark it cancelled and verify drain drops it.
        let pool = WorkerPool::new(flat_hf(), template(), 2);

        let k: ChunkKey = (0, 1, 77, 0);
        pool.submit(MeshRequest { key: k, priority: 0.0 });

        // Give the worker a moment to pick it up (move it to inflight).
        thread::sleep(Duration::from_millis(5));

        // Cancel while it may be in-flight.
        pool.cancel(k);

        // Drain until the pool finishes all work, then look for the key.
        let deadline = Instant::now() + Duration::from_secs(10);
        let mut found = false;
        while Instant::now() < deadline {
            let results = pool.drain();
            if results.iter().any(|r| r.key == k) {
                found = true;
                break;
            }
            if pool.pending_in_flight() == 0 {
                break;
            }
            thread::sleep(Duration::from_millis(5));
        }

        assert!(!found, "cancelled in-flight key was surfaced by drain");
    }

    // -----------------------------------------------------------------------
    // AC4: priority — higher-priority keys complete before lower-priority ones
    //      (with limited workers, ordering is observable in drain output).
    // -----------------------------------------------------------------------

    #[test]
    fn priority_ordering_observable() {
        // Use 1 worker so dispatch is strictly serial.
        // Submit 10 keys: first submit low priority, then high priority.
        // High-priority keys should be dispatched first and arrive first.
        let pool = WorkerPool::new(flat_hf(), template(), 1);

        // Submit low-priority keys first so they enter the queue first.
        let low_keys: Vec<ChunkKey> = (0u32..5).map(|i| (0u8, 1u8, i, 0u32)).collect();
        let high_keys: Vec<ChunkKey> = (5u32..10).map(|i| (0u8, 1u8, i, 0u32)).collect();

        for &k in &low_keys {
            pool.submit(MeshRequest { key: k, priority: 1.0 });
        }
        for &k in &high_keys {
            pool.submit(MeshRequest { key: k, priority: 100.0 });
        }

        // Drain all 10.
        let results = drain_until(&pool, 10, Duration::from_secs(30));
        assert_eq!(results.len(), 10);

        // Among the first 5 results the majority should be high-priority keys.
        // (With 1 worker it's possible the very first low-priority key was
        // already dispatched before high-priority ones were submitted, so we
        // allow 1 low-priority key in the first 5.)
        let high_set: HashSet<ChunkKey> = high_keys.iter().copied().collect();
        let first_5_high = results[..5].iter().filter(|r| high_set.contains(&r.key)).count();
        assert!(
            first_5_high >= 4,
            "expected ≥4 high-priority keys in first 5 results, got {first_5_high}. \
             Order: {:?}", results.iter().map(|r| (r.key, high_set.contains(&r.key))).collect::<Vec<_>>()
        );
    }

    // -----------------------------------------------------------------------
    // AC5: generation bump drops stale results.
    // -----------------------------------------------------------------------

    #[test]
    fn bump_generation_drops_stale_results() {
        let pool = WorkerPool::new(flat_hf(), template(), 2);

        // Submit some jobs, let them run, then bump the generation.
        let ks = keys(6);
        for &k in &ks {
            pool.submit(MeshRequest { key: k, priority: 0.0 });
        }

        // Wait for at least some results to be in-flight / done.
        thread::sleep(Duration::from_millis(20));

        // Bump generation: all in-flight results become stale.
        pool.bump_generation();
        let gen = pool.generation();
        assert_eq!(gen, 1);

        // Drain — all results from gen 0 should be dropped.
        let deadline = Instant::now() + Duration::from_secs(5);
        let mut non_stale_count = 0;
        while Instant::now() < deadline && pool.pending_in_flight() > 0 {
            let results = pool.drain();
            // Any results we receive here are from the new generation (gen=1),
            // but we haven't submitted new gen-1 jobs, so there should be none.
            non_stale_count += results.len();
            thread::sleep(Duration::from_millis(5));
        }
        // One final drain.
        non_stale_count += pool.drain().len();

        assert_eq!(non_stale_count, 0,
            "expected 0 non-stale results after generation bump, got {non_stale_count}");
    }

    // -----------------------------------------------------------------------
    // AC6: no deadlock; pool shuts down cleanly on drop.
    // -----------------------------------------------------------------------

    #[test]
    fn pool_drops_cleanly_with_pending_work() {
        let pool = WorkerPool::new(flat_hf(), template(), 2);
        // Submit work but don't drain — drop the pool while workers are busy.
        for &k in &keys(20) {
            pool.submit(MeshRequest { key: k, priority: 0.0 });
        }
        // Drop happens here; workers should exit cleanly via shutdown flag.
        drop(pool);
        // If we reach this line, no deadlock occurred.
    }

    // -----------------------------------------------------------------------
    // AC7: pending_in_flight counts correctly.
    // -----------------------------------------------------------------------

    #[test]
    fn pending_in_flight_accurate() {
        let pool = WorkerPool::new(flat_hf(), template(), 4);
        assert_eq!(pool.pending_in_flight(), 0);

        let ks = keys(10);
        for &k in &ks {
            pool.submit(MeshRequest { key: k, priority: 0.0 });
        }
        // Some combination of pending + inflight should equal total submitted
        // before any results are drained.
        let pif = pool.pending_in_flight();
        assert!(pif > 0, "pending_in_flight should be > 0 immediately after submit");

        // After draining all results, count should be 0.
        let _ = drain_until(&pool, 10, Duration::from_secs(30));
        thread::sleep(Duration::from_millis(10));
        let _ = pool.drain();
        assert_eq!(pool.pending_in_flight(), 0);
    }

    // -----------------------------------------------------------------------
    // AC8: real HeightField (SimpleHeightField) — verify build_chunk works.
    // -----------------------------------------------------------------------

    #[test]
    fn real_height_field_produces_valid_mesh() {
        let pool = WorkerPool::new(simple_hf(), template(), 2);
        let k: ChunkKey = (0, 1, 0, 0);
        pool.submit(MeshRequest { key: k, priority: 1.0 });

        let results = drain_until(&pool, 1, Duration::from_secs(30));
        assert_eq!(results.len(), 1);
        let mesh = &results[0].arrays;

        let res = template().resolution;
        let expected_verts = ((res + 1) * (res + 1) + 4 * (res + 1)) as usize;
        assert_eq!(mesh.positions.len(), expected_verts);
        // All positions finite.
        for p in &mesh.positions {
            assert!(p[0].is_finite() && p[1].is_finite() && p[2].is_finite());
        }
    }
}

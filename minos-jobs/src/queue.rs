//! Priority queue for pending mesh-build requests.
//!
//! Backed by a `BinaryHeap` ordered by `priority` descending, plus a
//! `HashMap<ChunkKey, usize>` that maps each pending key to its heap
//! position — enabling O(1) membership tests and lazy-deletion for `remove`.
//!
//! We use lazy deletion: `remove` marks a key as deleted in a `HashSet`;
//! `pop` skips entries that appear in that set.  This avoids a full heap
//! rebuild while keeping the data structure simple and allocation-free for
//! the common case where cancel is rare.

use std::cmp::Ordering;
use std::collections::HashSet;

use minos_planet::quadtree::ChunkKey;

use crate::job::MeshRequest;

// ---------------------------------------------------------------------------
// Internal heap entry
// ---------------------------------------------------------------------------

/// Wrapper around `MeshRequest` that implements `Ord` by priority descending.
/// We store priority as an ordered float (NaN-free in practice: priority is
/// a caller-controlled f32 used for screen-space pixels, never NaN).
struct HeapEntry {
    req: MeshRequest,
    /// `f32` bits reinterpreted as a comparable key.  Because we clamp NaN
    /// callers produce to 0.0 on push, this is safe to compare bit-wise.
    priority_bits: u32,
}

impl HeapEntry {
    fn new(req: MeshRequest) -> Self {
        // Treat NaN as lowest priority so it never blocks real work.
        let p = if req.priority.is_nan() { 0.0f32 } else { req.priority };
        HeapEntry {
            priority_bits: p.to_bits(),
            req,
        }
    }
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority_bits == other.priority_bits
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority_bits → higher in the max-heap → popped first.
        self.priority_bits.cmp(&other.priority_bits)
    }
}

// ---------------------------------------------------------------------------
// PriorityQueue
// ---------------------------------------------------------------------------

/// A max-priority queue of pending `MeshRequest`s keyed by `ChunkKey`.
///
/// Higher `priority` values are dequeued first.
///
/// * `push` is O(log n).
/// * `pop`  is O(log n) amortised (skips lazily-deleted entries).
/// * `remove` is O(1) (marks deleted; physical removal happens on next `pop`).
/// * `len` returns the number of *live* (not-deleted) entries.
pub struct PriorityQueue {
    heap: std::collections::BinaryHeap<HeapEntry>,
    /// Keys present in the heap (live, not deleted).
    present: HashSet<ChunkKey>,
    /// Keys that have been logically removed but not yet physically evicted.
    deleted: HashSet<ChunkKey>,
}

impl Default for PriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl PriorityQueue {
    pub fn new() -> Self {
        PriorityQueue {
            heap: std::collections::BinaryHeap::new(),
            present: HashSet::new(),
            deleted: HashSet::new(),
        }
    }

    /// Push `req`.  If the key is already present this is a no-op (the caller
    /// is responsible for dedup at the `WorkerPool` level before calling here).
    pub fn push(&mut self, req: MeshRequest) {
        let key = req.key;
        if self.present.contains(&key) {
            return;
        }
        // If it was previously deleted but still in the heap physically, we
        // un-delete it by removing the deletion marker and re-pushing.
        self.deleted.remove(&key);
        self.present.insert(key);
        self.heap.push(HeapEntry::new(req));
    }

    /// Pop the highest-priority live request, or `None` if empty.
    pub fn pop(&mut self) -> Option<MeshRequest> {
        loop {
            let entry = self.heap.pop()?;
            let key = entry.req.key;
            if self.deleted.remove(&key) {
                // Physically evict the deleted entry and keep looking.
                continue;
            }
            self.present.remove(&key);
            return Some(entry.req);
        }
    }

    /// Mark `key` as removed.  Returns `true` if the key was present.
    pub fn remove(&mut self, key: ChunkKey) -> bool {
        if self.present.remove(&key) {
            self.deleted.insert(key);
            true
        } else {
            false
        }
    }

    /// Number of live (not-deleted) entries.
    pub fn len(&self) -> usize {
        self.present.len()
    }
}

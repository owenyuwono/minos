//! Streaming page pool for resident node cluster-DAGs (N5).
//!
//! As the quadtree streams terrain nodes in/out, each resident node's cluster
//! data occupies one fixed-size **slot** in a large persistent device buffer.
//! This module is the slot bookkeeping — pure logic, no GPU — so it's unit-
//! testable headless (like `enki-rhi`'s streaming graveyard, which it mirrors).
//!
//! # Use-after-free safety
//! A slot freed on frame F is not reused until the GPU has completed F. We hold
//! evicted slots in a timeline-ordered graveyard and only return them to the
//! free list once `completed_frame >= retire_frame` — the same invariant the
//! `StreamingUploader` uses for whole-mesh buffers.

use std::collections::VecDeque;

/// Fixed-capacity slot allocator with deferred (timeline-gated) reclamation.
#[derive(Debug)]
pub struct PagePool {
    capacity: usize,
    /// Slots available for immediate allocation.
    free: Vec<u32>,
    /// Evicted slots awaiting GPU completion: `(slot, retire_frame)`, FIFO by
    /// retire_frame (which increases monotonically).
    graveyard: VecDeque<(u32, u64)>,
}

impl PagePool {
    /// Create a pool of `capacity` slots, all initially free.
    pub fn new(capacity: usize) -> Self {
        // Reversed so `pop` hands out low indices first (nicer for debugging).
        let free = (0..capacity as u32).rev().collect();
        Self { capacity, free, graveyard: VecDeque::new() }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Slots available right now (excludes those waiting in the graveyard).
    pub fn free_slots(&self) -> usize {
        self.free.len()
    }

    /// Slots currently allocated and in active use (neither free nor retiring).
    pub fn used_slots(&self) -> usize {
        self.capacity - self.free.len() - self.graveyard.len()
    }

    /// Slots evicted but not yet reclaimable (GPU may still be reading them).
    pub fn retiring_slots(&self) -> usize {
        self.graveyard.len()
    }

    /// Allocate a slot, or `None` if the pool is full (caller should evict or
    /// back off and retry next frame — never stall).
    pub fn alloc(&mut self) -> Option<u32> {
        self.free.pop()
    }

    /// Evict `slot`: it becomes reclaimable only once the GPU completes
    /// `retire_frame` (use `current_frame_counter + 1`, like the uploader).
    pub fn retire(&mut self, slot: u32, retire_frame: u64) {
        self.graveyard.push_back((slot, retire_frame));
    }

    /// Return graveyard slots whose `retire_frame` the GPU has now completed to
    /// the free list. Call once per frame with the timeline-semaphore value.
    pub fn collect(&mut self, completed_frame: u64) {
        while let Some(&(slot, retire_frame)) = self.graveyard.front() {
            if retire_frame <= completed_frame {
                self.free.push(slot);
                self.graveyard.pop_front();
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_pool_is_all_free() {
        let p = PagePool::new(8);
        assert_eq!(p.capacity(), 8);
        assert_eq!(p.free_slots(), 8);
        assert_eq!(p.used_slots(), 0);
        assert_eq!(p.retiring_slots(), 0);
    }

    #[test]
    fn alloc_until_full_then_none() {
        let mut p = PagePool::new(3);
        let a = p.alloc().unwrap();
        let b = p.alloc().unwrap();
        let c = p.alloc().unwrap();
        assert!(a != b && b != c && a != c, "slots must be distinct");
        assert_eq!(p.used_slots(), 3);
        assert_eq!(p.free_slots(), 0);
        assert!(p.alloc().is_none(), "full pool must return None");
    }

    #[test]
    fn retire_then_collect_reclaims_only_after_completion() {
        let mut p = PagePool::new(2);
        let s = p.alloc().unwrap();
        let _ = p.alloc().unwrap();
        assert_eq!(p.free_slots(), 0);

        // Evict s, to be reclaimed once the GPU completes frame 5.
        p.retire(s, 5);
        assert_eq!(p.retiring_slots(), 1);
        assert_eq!(p.used_slots(), 1); // the other alloc is still in use

        // GPU hasn't reached frame 5 yet — nothing reclaimed (use-after-free guard).
        p.collect(4);
        assert_eq!(p.free_slots(), 0);
        assert_eq!(p.retiring_slots(), 1);

        // GPU completes frame 5 — slot returns to the free list.
        p.collect(5);
        assert_eq!(p.free_slots(), 1);
        assert_eq!(p.retiring_slots(), 0);
        assert_eq!(p.alloc(), Some(s), "reclaimed slot is reused");
    }

    #[test]
    fn collect_respects_monotonic_retire_order() {
        let mut p = PagePool::new(3);
        let (s0, s1, s2) = (p.alloc().unwrap(), p.alloc().unwrap(), p.alloc().unwrap());
        p.retire(s0, 10);
        p.retire(s1, 12);
        p.retire(s2, 12);
        // Only frame-10 retirements are reclaimable at completed_frame=11.
        p.collect(11);
        assert_eq!(p.free_slots(), 1);
        assert_eq!(p.retiring_slots(), 2);
        // Both frame-12 retirements reclaim at 12.
        p.collect(12);
        assert_eq!(p.free_slots(), 3);
        assert_eq!(p.retiring_slots(), 0);
    }

    #[test]
    fn no_double_free_across_collects() {
        let mut p = PagePool::new(4);
        let s = p.alloc().unwrap();
        p.retire(s, 1);
        p.collect(100);
        p.collect(100); // second collect must not re-add the slot
        assert_eq!(p.free_slots(), 4);
    }
}

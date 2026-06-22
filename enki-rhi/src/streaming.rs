//! `streaming` — Dynamic geometry streaming: staging ring, per-frame budget,
//! copy→draw barrier, and deferred-destruction graveyard.
//!
//! # Architecture
//!
//! The module is split into two layers so the critical bookkeeping can be unit-
//! tested without a GPU:
//!
//!   - **Pure bookkeeping** (`StagingRing`, `Graveyard`): no Vulkan calls, no
//!     allocations, no unsafe code.  Fully exercised by headless `#[test]`s.
//!
//!   - **GPU layer** (`StreamingUploader`): holds the actual `VkBuffer` staging
//!     ring and drives `StagingRing` / `Graveyard` for real allocations.
//!     Exercised on Windows via the `--stress-streaming` harness in `enki-app`.
//!
//! # Single-queue seam
//!
//! All uploads run on the graphics queue (single-queue design).  The copy→draw
//! barrier below uses `VK_PIPELINE_STAGE_2_COPY_BIT` → `VERTEX_ATTRIBUTE_INPUT |
//! INDEX_INPUT` with no queue-family ownership transfer (QFOT) because src and
//! dst queue family are the same.
//!
//! TODO(future-transfer-queue): when a dedicated transfer queue is added, replace
//! the single `vkCmdPipelineBarrier2` below with a pair of barriers:
//!   1. On the transfer queue: TRANSFER_WRITE → release (srcQueueFamily=transfer,
//!      dstQueueFamily=graphics).
//!   2. On the graphics queue: acquire (srcQueueFamily=transfer,
//!      dstQueueFamily=graphics) → VERTEX_ATTRIBUTE_READ | INDEX_READ.
//!      The `StreamedMesh` handle, budget accounting, and graveyard are unchanged.

use std::collections::VecDeque;

use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
    MemoryLocation,
};

use crate::{BufferHandle, RhiError};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Default per-frame staging budget in bytes (4 MiB).
pub const DEFAULT_FRAME_BUDGET_BYTES: u64 = 4 * 1024 * 1024;

/// Tag bit set on all streamed `BufferHandle` ids.
///
/// Setting the top bit places streamed handles in a disjoint numeric range from
/// static `BufferStore` handles (which are small monotonic integers).  Without
/// this tag a streamed handle whose encoded value happened to equal a static
/// handle id would be mis-resolved by `Rhi::resolve_buffer`, which checks the
/// static store first.  With the tag the static-store lookup always fails for
/// streamed handles, so resolution falls through to the streamer correctly.
pub const STREAM_TAG: u64 = 1 << 63;

// The ring size is computed dynamically in StreamingUploader::new as
// per_frame_budget * frames_in_flight, so there is no fixed constant here.

// ── Staging ring (pure bookkeeping, GPU-free) ─────────────────────────────────

/// Bump-allocated ring of staging-memory offsets.
///
/// This struct tracks offsets into a pre-allocated staging buffer; it owns no
/// Vulkan resources.  `StagingUploader` drives it with the real buffer.
///
/// # Ring semantics
/// The ring is divided into equal per-frame segments.  Each frame only writes
/// into its own segment.  At `begin_frame(fi)` the segment for slot `fi` is
/// reset to its base offset — safe because by the time we reuse slot `fi` the
/// GPU has finished that frame (fence-waited in `begin_frame`).
pub struct StagingRing {
    /// Total ring capacity in bytes.
    pub capacity: u64,
    /// Size of each per-frame segment.
    segment_size: u64,
    /// Number of frame-in-flight slots.
    num_slots: usize,
    /// Per-slot write cursor (bytes used in the segment so far this frame).
    slot_used: Vec<u64>,
}

impl StagingRing {
    /// Create a new ring.  `capacity` is the total byte size; `num_slots` is
    /// the number of frames-in-flight (segments).
    pub fn new(capacity: u64, num_slots: usize) -> Self {
        assert!(num_slots > 0, "StagingRing: num_slots must be > 0");
        let segment_size = capacity / num_slots as u64;
        assert!(segment_size > 0, "StagingRing: segment_size would be 0");
        Self {
            capacity,
            segment_size,
            num_slots,
            slot_used: vec![0u64; num_slots],
        }
    }

    /// Reset the write cursor for frame slot `fi`.
    ///
    /// Call at the start of each frame after the fence for slot `fi` has been
    /// waited (guaranteeing the GPU has finished reading from that segment).
    pub fn reset_slot(&mut self, fi: usize) {
        debug_assert!(fi < self.num_slots);
        self.slot_used[fi] = 0;
    }

    /// Try to allocate `size` bytes in frame slot `fi`.
    ///
    /// Returns `Some(offset)` where `offset` is the byte offset into the ring
    /// buffer, or `None` if the segment is full.
    pub fn try_alloc(&mut self, fi: usize, size: u64) -> Option<u64> {
        debug_assert!(fi < self.num_slots);
        let used = self.slot_used[fi];
        if used + size > self.segment_size {
            return None;
        }
        let base = fi as u64 * self.segment_size;
        self.slot_used[fi] = used + size;
        Some(base + used)
    }

    /// Bytes used in the current slot (for budget accounting).
    pub fn slot_used(&self, fi: usize) -> u64 {
        self.slot_used[fi]
    }

    /// Per-slot segment size.
    pub fn segment_size(&self) -> u64 {
        self.segment_size
    }
}

// ── Per-frame budget (pure bookkeeping) ──────────────────────────────────────

/// Tracks total bytes staged in the current frame against a configurable cap.
///
/// The budget is independent of the ring because it counts user-visible mesh
/// data (positions + normals + colors + indices bytes) rather than the raw
/// staging-buffer allocation which may be padded for alignment.
pub struct FrameBudget {
    max_bytes: u64,
    used_bytes: u64,
}

impl FrameBudget {
    pub fn new(max_bytes: u64) -> Self {
        Self { max_bytes, used_bytes: 0 }
    }

    /// Try to reserve `bytes` from the remaining budget.
    ///
    /// Returns `true` on success (budget decremented), `false` if it would
    /// exceed the cap.
    pub fn try_reserve(&mut self, bytes: u64) -> bool {
        if self.used_bytes + bytes > self.max_bytes {
            return false;
        }
        self.used_bytes += bytes;
        true
    }

    /// Reset at the start of each new frame.
    pub fn reset(&mut self) {
        self.used_bytes = 0;
    }

    pub fn used(&self) -> u64 {
        self.used_bytes
    }

    pub fn remaining(&self) -> u64 {
        self.max_bytes.saturating_sub(self.used_bytes)
    }
}

// ── Graveyard (pure bookkeeping) ──────────────────────────────────────────────

/// One entry in the graveyard.
///
/// The `id` field is whatever the caller wants to use to key the entry (in the
/// GPU layer it's the `BufferHandle`'s raw u64 id so the GPU layer can look up
/// the real `VkBuffer`).  The pure bookkeeping layer only cares about the
/// retire-frame gating.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GraveyardEntry {
    /// Caller-supplied identifier (e.g. a `BufferHandle`'s raw id).
    pub id: u64,
    /// The timeline value the recording frame will signal (`frame_counter + 1`).
    ///
    /// The entry is freed only once `completed_frame >= retire_frame`, i.e.
    /// only after the GPU has finished the submission that drew this buffer.
    pub retire_frame: u64,
}

/// Deferred-destruction queue, keyed by GPU-completion frame index.
///
/// # Safety invariant
/// A buffer retired at frame N is freed only once `completed_frame >= N`.
/// This prevents use-after-free on the GPU.  `collect` enforces this by
/// draining only entries whose `retire_frame <= completed_frame`.
pub struct Graveyard {
    // VecDeque so we can drain from the front cheaply.
    entries: VecDeque<GraveyardEntry>,
}

impl Graveyard {
    pub fn new() -> Self {
        Self { entries: VecDeque::new() }
    }

    /// Record a buffer as retired at `retire_frame`.
    ///
    /// The caller must pass `frame_counter + 1` — the timeline value the
    /// current recording frame will signal when its submission completes.
    /// This ensures the buffer is freed only after the GPU finishes the frame
    /// that drew it, even when the mesh is drawn and retired in the same frame.
    pub fn retire(&mut self, id: u64, retire_frame: u64) {
        self.entries.push_back(GraveyardEntry { id, retire_frame });
    }

    /// Drain and return all entries whose `retire_frame <= completed_frame`.
    ///
    /// The caller is responsible for actually freeing the GPU resources.
    /// Returns an iterator of freed `GraveyardEntry`s.
    pub fn collect(&mut self, completed_frame: u64) -> Vec<GraveyardEntry> {
        let mut freed = Vec::new();
        while let Some(entry) = self.entries.front() {
            if entry.retire_frame <= completed_frame {
                freed.push(self.entries.pop_front().unwrap());
            } else {
                // Entries are in retire order; once we hit one that isn't ready,
                // nothing further is either.
                break;
            }
        }
        freed
    }

    /// Number of entries still waiting for GPU completion.
    pub fn pending_count(&self) -> usize {
        self.entries.len()
    }
}

impl Default for Graveyard {
    fn default() -> Self {
        Self::new()
    }
}

// ── StreamedMesh — handle bundle returned by upload() ─────────────────────────

/// Handle bundle for one mesh uploaded via `StreamingUploader::upload`.
///
/// Each field is a `BufferHandle` into the RHI's `BufferStore`.  The caller
/// holds onto this until the mesh is no longer needed, then calls
/// `Rhi::streaming_retire(mesh)` to schedule deferred destruction.
///
/// Layout matches the four vertex-stream bindings used by the terrain shader:
///   binding 0 = positions ([f32; 3] per vertex)
///   binding 1 = normals   ([f32; 3] per vertex)
///   binding 2 = colors    ([f32; 3] per vertex)
///   binding 3 = plate_colors ([f32; 3] per vertex, synthesized from colors if absent)
///   index buffer = u32 indices
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StreamedMesh {
    pub pos:         BufferHandle,
    pub nrm:         BufferHandle,
    pub col:         BufferHandle,
    pub plate:       BufferHandle,
    pub idx:         BufferHandle,
    pub index_count: u32,
}

// ── StreamingUploader — the GPU layer ─────────────────────────────────────────

/// Manages staged uploads to DEVICE_LOCAL buffers.
///
/// Lives inside `Rhi`.  The public API is exposed via `Rhi::streaming_upload`,
/// `Rhi::streaming_retire`, `Rhi::streaming_collect_garbage`, and
/// `Rhi::gpu_completed_frame`.
///
/// # Back-pressure
/// `upload` returns `Ok(None)` when the per-frame byte budget or the staging
/// ring segment is exhausted.  The caller retries next frame — never stalls.
///
/// # Copy→draw barrier (single-queue)
/// After each successful `vkCmdCopyBuffer`, a `vkCmdPipelineBarrier2` is
/// recorded:
///   srcStage = COPY  / srcAccess = TRANSFER_WRITE
///   dstStage = VERTEX_ATTRIBUTE_INPUT | INDEX_INPUT
///   dstAccess = VERTEX_ATTRIBUTE_READ | INDEX_READ
/// No queue-family ownership transfer is needed (same queue).
/// See module-level doc for the future transfer-queue seam.
pub struct StreamingUploader {
    // ── Staging ring ────────────────────────────────────────────────────────
    /// The single HOST_VISIBLE|HOST_COHERENT staging VkBuffer.
    staging_buffer: vk::Buffer,
    /// gpu-allocator allocation that backs `staging_buffer`.
    staging_alloc: Allocation,
    /// Pointer to the persistently mapped start of `staging_buffer`.
    staging_ptr: *mut u8,
    /// Pure-bookkeeping ring state (offsets, segment size).
    ring: StagingRing,

    // ── Budget ──────────────────────────────────────────────────────────────
    budget: FrameBudget,

    // ── Graveyard ───────────────────────────────────────────────────────────
    graveyard: Graveyard,

    // ── Device handle (for buffer creation) ─────────────────────────────────
    device: ash::Device,

    // ── Next handle id counter ───────────────────────────────────────────────
    /// Monotonically increasing id for streamed-mesh device-local buffers.
    /// Used as keys in the graveyard and the live-buffer map.
    next_id: u64,

    /// Live DEVICE_LOCAL buffers, keyed by their graveyard id.
    /// Each entry is (vk::Buffer, Allocation) for the 5 sub-buffers of one
    /// StreamedMesh.  We store them as a flat Vec of (id, [Option<...>; 5])
    /// where index 0..5 maps to pos/nrm/col/plate/idx.
    live: Vec<LiveEntry>,
}

struct LiveEntry {
    id: u64,
    buffers: [vk::Buffer; 5],
    allocs: Option<[Allocation; 5]>,
}

impl StreamingUploader {
    /// Allocate the staging ring buffer and initialise the uploader.
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        frames_in_flight: usize,
        per_frame_budget: u64,
    ) -> Result<Self, RhiError> {
        let ring_size = per_frame_budget * frames_in_flight as u64;

        let create_info = vk::BufferCreateInfo::default()
            .size(ring_size)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let staging_buffer = unsafe { device.create_buffer(&create_info, None) }
            .map_err(RhiError::Vulkan)?;

        let requirements =
            unsafe { device.get_buffer_memory_requirements(staging_buffer) };

        let staging_alloc = match allocator.allocate(&AllocationCreateDesc {
            name: "streaming_staging_ring",
            requirements,
            location: MemoryLocation::CpuToGpu,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(a) => a,
            Err(e) => {
                unsafe { device.destroy_buffer(staging_buffer, None) };
                return Err(RhiError::AllocationFailed(e.to_string()));
            }
        };

        if let Err(e) = unsafe {
            device.bind_buffer_memory(
                staging_buffer,
                staging_alloc.memory(),
                staging_alloc.offset(),
            )
        } {
            let _ = allocator.free(staging_alloc);
            unsafe { device.destroy_buffer(staging_buffer, None) };
            return Err(RhiError::Vulkan(e));
        }

        let staging_ptr = staging_alloc
            .mapped_ptr()
            .ok_or_else(|| {
                RhiError::AllocationFailed(
                    "staging ring allocation is not host-mapped".into(),
                )
            })?
            .as_ptr() as *mut u8;

        Ok(Self {
            staging_buffer,
            staging_alloc,
            staging_ptr,
            ring: StagingRing::new(ring_size, frames_in_flight),
            budget: FrameBudget::new(per_frame_budget),
            graveyard: Graveyard::new(),
            device: device.clone(),
            next_id: 1,
            live: Vec::new(),
        })
    }

    /// Reset the staging ring slot and per-frame budget for frame `fi`.
    ///
    /// Call at the start of each frame (after the per-frame fence has been
    /// waited, ensuring the GPU has finished reading from slot `fi`).
    pub fn begin_frame(&mut self, fi: usize) {
        self.ring.reset_slot(fi);
        self.budget.reset();
    }

    /// Upload mesh arrays to DEVICE_LOCAL vertex+index buffers.
    ///
    /// # Arguments
    /// - `fi`          — Frame-in-flight slot index (from `begin_frame`).
    /// - `frame_counter` — Current monotonic frame counter (from `Rhi::frame_counter`).
    /// - `cmd`         — The frame's graphics command buffer (must be recording).
    /// - `positions`   — Per-vertex positions `[f32; 3]`.
    /// - `normals`     — Per-vertex normals `[f32; 3]`.
    /// - `colors`      — Per-vertex colours `[f32; 3]`.
    /// - `plate_colors`— Optional per-vertex plate colours.  If `None`, `colors` is used.
    /// - `indices`     — Triangle indices (`u32`).
    ///
    /// # Returns
    /// - `Ok(Some(mesh))` — Upload recorded; barrier inserted; handles valid.
    /// - `Ok(None)`       — Budget or ring full; caller retries next frame.
    /// - `Err(_)`         — Fatal Vulkan / allocation error.
    ///
    /// # Safety
    /// The returned `StreamedMesh` handles are only valid as long as
    /// `streaming_retire` has not been called for them.  Accessing them after
    /// retirement (and GPU completion of the retire frame) is undefined.
    #[allow(clippy::too_many_arguments)]
    pub fn upload(
        &mut self,
        fi: usize,
        frame_counter: u64,
        cmd: vk::CommandBuffer,
        allocator: &mut Allocator,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        colors: &[[f32; 3]],
        plate_colors: Option<&[[f32; 3]]>,
        indices: &[u32],
    ) -> Result<Option<StreamedMesh>, RhiError> {
        let _ = frame_counter; // used in graveyard; not needed here but keep param for clarity

        let plate_src = plate_colors.unwrap_or(colors);

        // Build byte slices for each stream.
        let pos_bytes: &[u8] = bytemuck::cast_slice(positions);
        let nrm_bytes: &[u8] = bytemuck::cast_slice(normals);
        let col_bytes: &[u8] = bytemuck::cast_slice(colors);
        let plate_bytes: &[u8] = bytemuck::cast_slice(plate_src);
        let idx_bytes: &[u8] = bytemuck::cast_slice(indices);

        let total_bytes = (pos_bytes.len()
            + nrm_bytes.len()
            + col_bytes.len()
            + plate_bytes.len()
            + idx_bytes.len()) as u64;

        // ── Budget check ─────────────────────────────────────────────────────
        if !self.budget.try_reserve(total_bytes) {
            return Ok(None); // back-pressure: budget exhausted for this frame
        }

        // ── Staging ring allocation ───────────────────────────────────────────
        let staging_offset = match self.ring.try_alloc(fi, total_bytes) {
            Some(off) => off,
            None => {
                // Undo the budget reservation since we can't proceed.
                // (FrameBudget doesn't have a release, so we do the safe thing:
                //  the budget is already decremented; to avoid over-counting we
                //  just return None and the frame ends naturally.)
                return Ok(None);
            }
        };

        // ── Copy into staging ─────────────────────────────────────────────────
        let mut write_offset = staging_offset as usize;

        let offsets: [(u64, u64); 5] = {
            let mut o = staging_offset;
            let mut arr = [(0u64, 0u64); 5];
            for (i, bytes) in [pos_bytes, nrm_bytes, col_bytes, plate_bytes, idx_bytes]
                .iter()
                .enumerate()
            {
                arr[i] = (o, bytes.len() as u64);
                o += bytes.len() as u64;
            }
            arr
        };

        for bytes in [pos_bytes, nrm_bytes, col_bytes, plate_bytes, idx_bytes] {
            let dst = unsafe {
                std::slice::from_raw_parts_mut(
                    self.staging_ptr.add(write_offset),
                    bytes.len(),
                )
            };
            dst.copy_from_slice(bytes);
            write_offset += bytes.len();
        }

        // ── Allocate DEVICE_LOCAL vertex+index buffers ────────────────────────
        let usages: [vk::BufferUsageFlags; 5] = [
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
        ];
        let names = ["stream_pos", "stream_nrm", "stream_col", "stream_plate", "stream_idx"];
        let sizes = [
            pos_bytes.len() as u64,
            nrm_bytes.len() as u64,
            col_bytes.len() as u64,
            plate_bytes.len() as u64,
            idx_bytes.len() as u64,
        ];

        let mut dev_buffers = [vk::Buffer::null(); 5];
        let mut dev_allocs: [Option<Allocation>; 5] =
            [None, None, None, None, None];

        for i in 0..5 {
            let create_info = vk::BufferCreateInfo::default()
                .size(sizes[i])
                .usage(usages[i])
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let buf = match unsafe { self.device.create_buffer(&create_info, None) } {
                Ok(b) => b,
                Err(e) => {
                    // Roll back already-created buffers.
                    for j in 0..i {
                        if let Some(a) = dev_allocs[j].take() {
                            let _ = allocator.free(a);
                        }
                        unsafe { self.device.destroy_buffer(dev_buffers[j], None) };
                    }
                    return Err(RhiError::Vulkan(e));
                }
            };

            let reqs = unsafe { self.device.get_buffer_memory_requirements(buf) };
            let alloc = match allocator.allocate(&AllocationCreateDesc {
                name: names[i],
                requirements: reqs,
                location: MemoryLocation::GpuOnly,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            }) {
                Ok(a) => a,
                Err(e) => {
                    unsafe { self.device.destroy_buffer(buf, None) };
                    for j in 0..i {
                        if let Some(a) = dev_allocs[j].take() {
                            let _ = allocator.free(a);
                        }
                        unsafe { self.device.destroy_buffer(dev_buffers[j], None) };
                    }
                    return Err(RhiError::AllocationFailed(e.to_string()));
                }
            };

            if let Err(e) = unsafe {
                self.device
                    .bind_buffer_memory(buf, alloc.memory(), alloc.offset())
            } {
                let _ = allocator.free(alloc);
                unsafe { self.device.destroy_buffer(buf, None) };
                for j in 0..i {
                    if let Some(a) = dev_allocs[j].take() {
                        let _ = allocator.free(a);
                    }
                    unsafe { self.device.destroy_buffer(dev_buffers[j], None) };
                }
                return Err(RhiError::Vulkan(e));
            }

            dev_buffers[i] = buf;
            dev_allocs[i] = Some(alloc);
        }

        // ── Record vkCmdCopyBuffer (staging → device-local) ──────────────────
        for i in 0..5 {
            let (src_offset, size) = offsets[i];
            let region = vk::BufferCopy {
                src_offset,
                dst_offset: 0,
                size,
            };
            unsafe {
                self.device.cmd_copy_buffer(
                    cmd,
                    self.staging_buffer,
                    dev_buffers[i],
                    std::slice::from_ref(&region),
                );
            }
        }

        // ── Copy→draw barrier (single-queue, no QFOT needed) ─────────────────
        //
        // Ensures the TRANSFER_WRITE completes before VERTEX_ATTRIBUTE_READ /
        // INDEX_READ on the same queue.
        //
        // TODO(future-transfer-queue): replace with a release/acquire pair
        // spanning transfer→graphics queues.  src here becomes the release
        // barrier on the transfer queue; a matching acquire barrier goes at
        // the top of the graphics submit.
        let barriers: Vec<vk::BufferMemoryBarrier2> = dev_buffers
            .iter()
            .enumerate()
            .map(|(i, &buf)| {
                vk::BufferMemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::COPY)
                    .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .dst_stage_mask(
                        vk::PipelineStageFlags2::VERTEX_ATTRIBUTE_INPUT
                            | vk::PipelineStageFlags2::INDEX_INPUT,
                    )
                    .dst_access_mask(
                        vk::AccessFlags2::VERTEX_ATTRIBUTE_READ
                            | vk::AccessFlags2::INDEX_READ,
                    )
                    .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                    .buffer(buf)
                    .offset(0)
                    .size(sizes[i])
            })
            .collect();

        let dep_info = vk::DependencyInfo::default()
            .buffer_memory_barriers(&barriers);
        unsafe { self.device.cmd_pipeline_barrier2(cmd, &dep_info) };

        // ── Register live entry ───────────────────────────────────────────────
        let id = self.next_id;
        self.next_id += 1;

        // Pack allocations into a fixed array for storage.
        let allocs_array = [
            dev_allocs[0].take().unwrap(),
            dev_allocs[1].take().unwrap(),
            dev_allocs[2].take().unwrap(),
            dev_allocs[3].take().unwrap(),
            dev_allocs[4].take().unwrap(),
        ];

        self.live.push(LiveEntry {
            id,
            buffers: dev_buffers,
            allocs: Some(allocs_array),
        });

        // Manufacture BufferHandles from the live-entry id.
        // Encoding: STREAM_TAG | (id * 10 + sub_index) — one unique u64 per sub-buffer.
        //
        // STREAM_TAG (top bit set) puts streamed handles in a disjoint numeric range
        // from static BufferStore handles (small monotonic integers), preventing any
        // accidental collision in Rhi::resolve_buffer.  The sub-index lives in the low
        // bits; masking off STREAM_TAG recovers the raw id*10+sub value for lookup.
        let tag = STREAM_TAG;
        let pos_handle   = BufferHandle(tag | (id * 10));
        let nrm_handle   = BufferHandle(tag | (id * 10 + 1));
        let col_handle   = BufferHandle(tag | (id * 10 + 2));
        let plate_handle = BufferHandle(tag | (id * 10 + 3));
        let idx_handle   = BufferHandle(tag | (id * 10 + 4));

        Ok(Some(StreamedMesh {
            pos:         pos_handle,
            nrm:         nrm_handle,
            col:         col_handle,
            plate:       plate_handle,
            idx:         idx_handle,
            index_count: indices.len() as u32,
        }))
    }

    /// Schedule a streamed mesh for deferred destruction.
    ///
    /// The mesh's underlying VkBuffers are freed only once the GPU has
    /// completed `retire_frame` (i.e., once `gpu_completed_frame() >= retire_frame`).
    ///
    /// # Invariant
    /// `retire_frame` must be `frame_counter + 1` — the timeline value the current
    /// recording frame will signal (end_frame increments frame_counter and then
    /// signals that value on the timeline semaphore).  Passing `frame_counter + 1`
    /// guarantees the buffer is freed only after the GPU completes the frame that
    /// drew it, even when the mesh is drawn and retired in the same frame.
    pub fn retire(&mut self, mesh: StreamedMesh, retire_frame: u64) {
        // The graveyard key is the mesh's base id — mask off STREAM_TAG first.
        let raw = mesh.pos.0 & !STREAM_TAG;
        let id = raw / 10;
        self.graveyard.retire(id, retire_frame);
    }

    /// Free all graveyard entries whose retire_frame <= completed_frame.
    ///
    /// Call once per frame: `collect_garbage(rhi.gpu_completed_frame())`.
    ///
    /// INVARIANT: a buffer retired at frame N is freed only once
    /// `completed_frame >= N`.  This is the use-after-free guard.
    pub fn collect_garbage(&mut self, completed_frame: u64, allocator: &mut Allocator) {
        let freed = self.graveyard.collect(completed_frame);
        for entry in freed {
            if let Some(pos) = self.live.iter().position(|e| e.id == entry.id) {
                let mut live = self.live.remove(pos);
                if let Some(allocs) = live.allocs.take() {
                    for alloc in allocs {
                        let _ = allocator.free(alloc);
                    }
                }
                for &buf in &live.buffers {
                    if buf != vk::Buffer::null() {
                        unsafe { self.device.destroy_buffer(buf, None) };
                    }
                }
            }
        }
    }

    /// Number of retired meshes waiting for GPU completion in the graveyard.
    pub fn graveyard_size(&self) -> usize {
        self.graveyard.pending_count()
    }

    /// Look up the raw `VkBuffer` for a `StreamedMesh` sub-buffer handle.
    ///
    /// Used by `Rhi::bind_vertex_buffers` / `bind_index_buffer` to resolve
    /// streamed-mesh handles that are not in the main `BufferStore`.
    ///
    /// Masks off `STREAM_TAG` before decoding the id and sub-index.
    pub fn vk_buffer(&self, handle: BufferHandle) -> Option<vk::Buffer> {
        let raw = handle.0 & !STREAM_TAG;
        let id = raw / 10;
        let sub = (raw % 10) as usize;
        self.live
            .iter()
            .find(|e| e.id == id)
            .map(|e| e.buffers[sub])
    }

    /// Destroy the staging ring.  Call from `Rhi::drop` before destroying the device.
    pub fn destroy(mut self, allocator: &mut Allocator) {
        // Free remaining live buffers (device_wait_idle must have been called).
        for mut live in self.live.drain(..) {
            if let Some(allocs) = live.allocs.take() {
                for alloc in allocs {
                    let _ = allocator.free(alloc);
                }
            }
            for &buf in &live.buffers {
                if buf != vk::Buffer::null() {
                    unsafe { self.device.destroy_buffer(buf, None) };
                }
            }
        }

        let _ = allocator.free(self.staging_alloc);
        unsafe { self.device.destroy_buffer(self.staging_buffer, None) };
    }
}

// ── Headless bookkeeping tests ────────────────────────────────────────────────
//
// These tests exercise StagingRing, FrameBudget, and Graveyard WITHOUT any
// Vulkan device.  They are the primary correctness gate for the streaming
// subsystem because the GPU path only runs on Windows.

#[cfg(test)]
mod tests {
    use super::*;

    // ── StagingRing tests ─────────────────────────────────────────────────────

    /// A smaller upload that fits in the budget must succeed;
    /// an upload that would exceed the budget must return the back-pressure signal.
    #[test]
    fn budget_backpressure_smaller_succeeds_larger_fails() {
        let mut budget = FrameBudget::new(4 * 1024 * 1024); // 4 MiB cap

        // 3 MiB fits.
        assert!(budget.try_reserve(3 * 1024 * 1024));

        // Another 2 MiB would exceed the remaining 1 MiB — back-pressure.
        assert!(!budget.try_reserve(2 * 1024 * 1024));

        // Exactly 1 MiB remaining fits.
        assert!(budget.try_reserve(1 * 1024 * 1024));

        // Now nothing fits.
        assert!(!budget.try_reserve(1));
    }

    /// Budget resets across frames so the ring space can be reused.
    #[test]
    fn budget_resets_each_frame() {
        let mut budget = FrameBudget::new(1024);
        assert!(budget.try_reserve(1024));
        assert!(!budget.try_reserve(1));

        // Frame boundary.
        budget.reset();
        assert!(budget.try_reserve(1024));
    }

    /// Ring slot resets across frames, making space available again.
    #[test]
    fn ring_resets_and_reuses_space_across_frames() {
        let mut ring = StagingRing::new(6 * 1024, 3); // 3 slots, 2048 B each

        // Use up slot 0.
        assert!(ring.try_alloc(0, 2048).is_some());
        // Ring slot 0 is full.
        assert!(ring.try_alloc(0, 1).is_none());

        // Simulate frame advance: reset slot 0 (fence waited).
        ring.reset_slot(0);

        // Slot 0 is reusable.
        let off = ring.try_alloc(0, 1024);
        assert!(off.is_some());
        // Offset must be at slot 0's base (0 * 2048 = 0).
        assert_eq!(off.unwrap(), 0);
    }

    /// Ring allocations for different slots do not overlap.
    #[test]
    fn ring_slots_do_not_overlap() {
        let mut ring = StagingRing::new(4 * 1024, 2); // 2 slots, 2048 B each

        let off0 = ring.try_alloc(0, 1024).unwrap();
        let off1 = ring.try_alloc(1, 1024).unwrap();

        // Slot 1 must start after slot 0's segment.
        assert!(off1 >= ring.segment_size());
        // They must not overlap.
        assert!(off0 + 1024 <= off1 || off1 + 1024 <= off0);
    }

    /// An allocation larger than the segment is rejected.
    #[test]
    fn ring_rejects_oversized_alloc() {
        let mut ring = StagingRing::new(2048, 1); // 1 slot, 2048 B
        assert!(ring.try_alloc(0, 2049).is_none());
        assert!(ring.try_alloc(0, 2048).is_some()); // exactly fits
    }

    // ── Graveyard tests ───────────────────────────────────────────────────────

    /// A buffer retired at frame N must NOT be freed when completed_frame < N.
    #[test]
    fn graveyard_not_freed_before_retire_frame() {
        let mut g = Graveyard::new();
        g.retire(42, 10); // retire at frame 10

        // completed_frame 9 → not freed yet.
        let freed = g.collect(9);
        assert!(freed.is_empty(), "must not free before GPU completes retire_frame");
        assert_eq!(g.pending_count(), 1);
    }

    /// A buffer retired at frame N IS freed once completed_frame >= N.
    #[test]
    fn graveyard_freed_at_or_after_retire_frame() {
        let mut g = Graveyard::new();
        g.retire(42, 10);

        // completed_frame == retire_frame → freed.
        let freed = g.collect(10);
        assert_eq!(freed.len(), 1);
        assert_eq!(freed[0].id, 42);
        assert_eq!(freed[0].retire_frame, 10);
        assert_eq!(g.pending_count(), 0);
    }

    /// completed_frame > retire_frame also frees the entry.
    #[test]
    fn graveyard_freed_when_completed_exceeds_retire() {
        let mut g = Graveyard::new();
        g.retire(7, 3);
        let freed = g.collect(100);
        assert_eq!(freed.len(), 1);
        assert_eq!(freed[0].id, 7);
    }

    /// Retiring across several frames frees in the correct order with no double-free.
    #[test]
    fn graveyard_multiple_retires_correct_order_no_double_free() {
        let mut g = Graveyard::new();
        g.retire(1, 5);
        g.retire(2, 7);
        g.retire(3, 7);
        g.retire(4, 10);

        // Advance to frame 7 — frees ids 1, 2, 3.
        let freed = g.collect(7);
        let ids: Vec<u64> = freed.iter().map(|e| e.id).collect();
        assert_eq!(ids, vec![1, 2, 3], "must free in retire order");
        assert_eq!(g.pending_count(), 1); // id 4 still pending

        // Calling collect again at the same frame frees nothing (no double-free).
        let freed2 = g.collect(7);
        assert!(freed2.is_empty(), "must not double-free");

        // Advance to 10 — frees id 4.
        let freed3 = g.collect(10);
        assert_eq!(freed3.len(), 1);
        assert_eq!(freed3[0].id, 4);
        assert_eq!(g.pending_count(), 0);
    }

    /// Once freed, entries are gone — idempotent and safe.
    #[test]
    fn graveyard_idempotent_collect() {
        let mut g = Graveyard::new();
        g.retire(99, 1);
        let _ = g.collect(1);
        // Second collect at same or higher frame is a no-op.
        let freed = g.collect(1);
        assert!(freed.is_empty());
        let freed = g.collect(1000);
        assert!(freed.is_empty());
    }

    /// Frame-gated invariant: retire on frame N, collect at N-1 frees nothing,
    /// collect at N frees one, collect again frees nothing.
    #[test]
    fn graveyard_frame_gated_invariant() {
        let mut g = Graveyard::new();

        // Simulate: retire a buffer on frame 5 (the frame counter at submission).
        g.retire(55, 5);

        // GPU is still on frame 4 — must not free.
        assert!(g.collect(4).is_empty());

        // GPU completes frame 5 — must free exactly once.
        let freed = g.collect(5);
        assert_eq!(freed.len(), 1);
        assert_eq!(freed[0].id, 55);

        // No double-free on subsequent collects.
        assert!(g.collect(5).is_empty());
        assert!(g.collect(999).is_empty());
    }

    /// Fake-sink: record freed ids and assert graveyard's decisions match.
    #[test]
    fn graveyard_fake_sink_records_decisions() {
        let mut g = Graveyard::new();
        let mut freed_ids: Vec<u64> = Vec::new();

        // Retire five buffers across frames 2, 3, 5.
        for id in 0..5 {
            let frame = 2 + (id / 2) as u64; // ids 0,1→f2; 2,3→f3; 4→f5 (wait, let's be explicit)
            let _ = frame;
        }
        g.retire(10, 2);
        g.retire(11, 2);
        g.retire(12, 3);
        g.retire(13, 3);
        g.retire(14, 5);

        // Process frame by frame.
        for completed in [1u64, 2, 3, 4, 5] {
            let freed = g.collect(completed);
            freed_ids.extend(freed.iter().map(|e| e.id));
        }

        // All five must have been freed, in retire order.
        assert_eq!(freed_ids, vec![10, 11, 12, 13, 14]);
    }

    // ── Budget + ring integration ─────────────────────────────────────────────

    /// Uploading up to the budget limit works; the next one fails.
    /// After begin_frame (reset) the budget is available again.
    #[test]
    fn budget_and_ring_work_together_across_frames() {
        const CAP: u64 = 1024;
        let mut budget = FrameBudget::new(CAP);
        let mut ring = StagingRing::new(CAP * 2, 2); // 2 slots

        // Frame 0: fill to cap.
        assert!(budget.try_reserve(CAP));
        assert!(ring.try_alloc(0, CAP).is_some());

        // Frame 0: next alloc is rejected by budget.
        assert!(!budget.try_reserve(1));
        // And by ring too.
        assert!(ring.try_alloc(0, 1).is_none());

        // Frame boundary: reset slot 0 and budget.
        budget.reset();
        ring.reset_slot(0);

        // Now both accept another full-cap upload.
        assert!(budget.try_reserve(CAP));
        assert!(ring.try_alloc(0, CAP).is_some());
    }
}

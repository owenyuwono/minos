//! `buffer` — GPU buffer creation and host-visible upload.
//!
//! # Memory strategy (M1)
//! All buffers are `HOST_VISIBLE | HOST_COHERENT` (no staging copy). This is
//! intentional for M1 static geometry: it keeps the code simple and correct.
//! M2 will add a streaming uploader with DEVICE_LOCAL buffers and explicit
//! staging transfers.
//!
//! # Handle map
//! Buffers are tracked in an internal `BTreeMap<BufferHandle, AllocatedBuffer>`.
//! Each handle is a monotonically increasing `u64`. Destroying a buffer removes
//! it from the map and frees the allocation.

use std::collections::BTreeMap;

use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
    MemoryLocation,
};

use crate::{BufferHandle, RhiError};

// ── Internal buffer ─────────────────────────────────────────────────────────

pub(crate) struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    /// Byte size of the buffer.
    pub size: u64,
}

impl AllocatedBuffer {
    fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe { device.destroy_buffer(self.buffer, None) };
        let _ = allocator.free(self.allocation);
    }
}

// ── Buffer store ─────────────────────────────────────────────────────────────

/// Manages all GPU buffers allocated by the RHI.
///
/// Lives inside [`crate::Rhi`] and is torn down in `Drop` after the device is
/// idle.
pub(crate) struct BufferStore {
    map: BTreeMap<u64, AllocatedBuffer>,
    next_id: u64,
    device: ash::Device,
}

impl BufferStore {
    pub fn new(device: &ash::Device) -> Self {
        Self {
            map: BTreeMap::new(),
            next_id: 1,
            device: device.clone(),
        }
    }

    // ── Allocation helpers ──────────────────────────────────────────────────

    fn alloc_buffer(
        &mut self,
        allocator: &mut Allocator,
        size: u64,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<AllocatedBuffer, RhiError> {
        let create_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe { self.device.create_buffer(&create_info, None) }
            .map_err(RhiError::Vulkan)?;

        let requirements = unsafe { self.device.get_buffer_memory_requirements(buffer) };

        let allocation = match allocator.allocate(&AllocationCreateDesc {
            name,
            requirements,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        }) {
            Ok(a) => a,
            Err(e) => {
                unsafe { self.device.destroy_buffer(buffer, None) };
                return Err(RhiError::AllocationFailed(e.to_string()));
            }
        };

        if let Err(e) = unsafe {
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
        } {
            let _ = allocator.free(allocation);
            unsafe { self.device.destroy_buffer(buffer, None) };
            return Err(RhiError::Vulkan(e));
        }

        Ok(AllocatedBuffer { buffer, allocation, size })
    }

    fn insert(&mut self, buf: AllocatedBuffer) -> BufferHandle {
        let id = self.next_id;
        self.next_id += 1;
        self.map.insert(id, buf);
        BufferHandle(id)
    }

    // ── Public create API ───────────────────────────────────────────────────

    /// Create a vertex buffer and upload `data` into it immediately.
    ///
    /// The buffer is HOST_VISIBLE|HOST_COHERENT so no staging copy is needed.
    /// The returned handle is valid until [`Self::destroy_buffer`] is called.
    pub fn create_vertex_buffer(
        &mut self,
        allocator: &mut Allocator,
        data: &[u8],
    ) -> Result<BufferHandle, RhiError> {
        let size = data.len() as u64;
        let mut buf = self.alloc_buffer(
            allocator,
            size,
            vk::BufferUsageFlags::VERTEX_BUFFER,
            MemoryLocation::CpuToGpu,
            "vertex_buffer",
        )?;
        write_to_allocation(&mut buf, data)?;
        Ok(self.insert(buf))
    }

    /// Create an index buffer (u32 indices) and upload `indices` immediately.
    pub fn create_index_buffer(
        &mut self,
        allocator: &mut Allocator,
        indices: &[u32],
    ) -> Result<BufferHandle, RhiError> {
        let data = bytemuck::cast_slice(indices);
        let size = data.len() as u64;
        let mut buf = self.alloc_buffer(
            allocator,
            size,
            vk::BufferUsageFlags::INDEX_BUFFER,
            MemoryLocation::CpuToGpu,
            "index_buffer",
        )?;
        write_to_allocation(&mut buf, data)?;
        Ok(self.insert(buf))
    }

    /// Create a host-visible uniform buffer of `size` bytes (zero-initialised).
    ///
    /// Use [`Self::write_buffer`] to update its contents each frame.
    pub fn create_uniform_buffer(
        &mut self,
        allocator: &mut Allocator,
        size: u64,
    ) -> Result<BufferHandle, RhiError> {
        let buf = self.alloc_buffer(
            allocator,
            size,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            MemoryLocation::CpuToGpu,
            "uniform_buffer",
        )?;
        Ok(self.insert(buf))
    }

    /// Create a storage buffer (SSBO) of `data.len()` bytes.
    ///
    /// `host_visible = true` → `CpuToGpu` (mappable; readback via [`Self::read_buffer`])
    /// with `STORAGE | TRANSFER_SRC | TRANSFER_DST`; `data` is uploaded immediately.
    /// `host_visible = false` → `GpuOnly` with `STORAGE | TRANSFER_DST`; `data` is
    /// ignored (fill it via a staging copy — added with the N2 upload path).
    ///
    /// Generic (not Nanite-specific): any GPU-driven pass uses storage buffers.
    pub fn create_storage_buffer(
        &mut self,
        allocator: &mut Allocator,
        data: &[u8],
        host_visible: bool,
    ) -> Result<BufferHandle, RhiError> {
        let size = (data.len() as u64).max(4); // avoid zero-size allocations
        let (usage, location) = if host_visible {
            (
                vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC
                    | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::CpuToGpu,
            )
        } else {
            (
                vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                MemoryLocation::GpuOnly,
            )
        };
        let mut buf = self.alloc_buffer(allocator, size, usage, location, "storage_buffer")?;
        if host_visible && !data.is_empty() {
            write_to_allocation(&mut buf, data)?;
        }
        Ok(self.insert(buf))
    }

    /// Create a buffer of `size` bytes with caller-chosen usage flags, zero-init.
    ///
    /// Generic escape hatch for GPU-driven resources (indirect-args, visible-list,
    /// per-frame uniforms). `host_visible = true` → `CpuToGpu` (mappable, writable
    /// via [`Self::write_buffer`]); `false` → `GpuOnly`.
    pub fn create_gpu_buffer(
        &mut self,
        allocator: &mut Allocator,
        size: u64,
        host_visible: bool,
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferHandle, RhiError> {
        let location = if host_visible {
            MemoryLocation::CpuToGpu
        } else {
            MemoryLocation::GpuOnly
        };
        let buf = self.alloc_buffer(allocator, size.max(4), usage, location, "gpu_buffer")?;
        Ok(self.insert(buf))
    }

    /// Copy the contents of a host-visible buffer back to a `Vec<u8>`.
    ///
    /// Only valid for `CpuToGpu`/mappable buffers (e.g. a `host_visible` storage
    /// buffer). Returns the full allocated size.
    pub fn read_buffer(&self, handle: BufferHandle) -> Result<Vec<u8>, RhiError> {
        let buf = self.map.get(&handle.0).ok_or(RhiError::InvalidHandle)?;
        let slice = buf.allocation.mapped_slice().ok_or_else(|| {
            RhiError::AllocationFailed("buffer is not host-mapped (read_buffer)".into())
        })?;
        Ok(slice[..buf.size as usize].to_vec())
    }

    // ── Write / update ──────────────────────────────────────────────────────

    /// Write raw bytes into an existing host-visible buffer.
    ///
    /// `bytes.len()` must be ≤ the buffer's allocated size.
    pub fn write_buffer(
        &mut self,
        handle: BufferHandle,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        let buf = self.map.get_mut(&handle.0).ok_or(RhiError::InvalidHandle)?;
        write_to_allocation(buf, bytes)
    }

    /// Write raw bytes into an existing host-visible buffer at byte `offset`
    /// (partial update — e.g. a single cluster slot in the streaming page pool).
    ///
    /// `offset + bytes.len()` must be ≤ the buffer's allocated size.
    pub fn write_buffer_at(
        &mut self,
        handle: BufferHandle,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        let buf = self.map.get_mut(&handle.0).ok_or(RhiError::InvalidHandle)?;
        write_to_allocation_at(buf, offset, bytes)
    }

    // ── Destroy ─────────────────────────────────────────────────────────────

    /// Free a buffer and retire its handle. Calling with an unknown handle is a no-op.
    pub fn destroy_buffer(&mut self, allocator: &mut Allocator, handle: BufferHandle) {
        if let Some(buf) = self.map.remove(&handle.0) {
            buf.destroy(&self.device, allocator);
        }
    }

    /// Destroy all remaining buffers. Called from `Rhi::drop`.
    pub fn destroy_all(&mut self, allocator: &mut Allocator) {
        let keys: Vec<u64> = self.map.keys().copied().collect();
        for id in keys {
            if let Some(buf) = self.map.remove(&id) {
                buf.destroy(&self.device, allocator);
            }
        }
    }

    // ── Internal access ─────────────────────────────────────────────────────

    /// Return the raw `VkBuffer` for a handle. Used by command recording.
    pub fn vk_buffer(&self, handle: BufferHandle) -> Result<vk::Buffer, RhiError> {
        self.map
            .get(&handle.0)
            .map(|b| b.buffer)
            .ok_or(RhiError::InvalidHandle)
    }

    /// Return the raw `VkBuffer` for a handle plus its byte size.
    pub fn vk_buffer_with_size(
        &self,
        handle: BufferHandle,
    ) -> Result<(vk::Buffer, u64), RhiError> {
        self.map
            .get(&handle.0)
            .map(|b| (b.buffer, b.size))
            .ok_or(RhiError::InvalidHandle)
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn write_to_allocation(buf: &mut AllocatedBuffer, bytes: &[u8]) -> Result<(), RhiError> {
    let dst = buf.allocation.mapped_slice_mut().ok_or_else(|| {
        RhiError::AllocationFailed("buffer allocation is not host-mapped".into())
    })?;
    if bytes.len() > dst.len() {
        return Err(RhiError::AllocationFailed(format!(
            "write {} bytes into buffer of {} bytes",
            bytes.len(),
            dst.len()
        )));
    }
    dst[..bytes.len()].copy_from_slice(bytes);
    Ok(())
}

fn write_to_allocation_at(
    buf: &mut AllocatedBuffer,
    offset: u64,
    bytes: &[u8],
) -> Result<(), RhiError> {
    let dst = buf.allocation.mapped_slice_mut().ok_or_else(|| {
        RhiError::AllocationFailed("buffer allocation is not host-mapped".into())
    })?;
    let off = offset as usize;
    let end = off
        .checked_add(bytes.len())
        .ok_or_else(|| RhiError::AllocationFailed("write offset overflow".into()))?;
    if end > dst.len() {
        return Err(RhiError::AllocationFailed(format!(
            "write {} bytes at offset {} into buffer of {} bytes",
            bytes.len(),
            off,
            dst.len()
        )));
    }
    dst[off..end].copy_from_slice(bytes);
    Ok(())
}

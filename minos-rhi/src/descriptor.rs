//! `descriptor` — Per-frame uniform descriptor set layout, pool, and sets.
//!
//! # Layout contract
//! One descriptor set layout is created:
//!   - Set 0, Binding 0, UNIFORM_BUFFER, stages VERTEX | FRAGMENT
//!
//! One descriptor set per ring frame is allocated from a single pool. Each set
//! points at a per-frame `HOST_VISIBLE|HOST_COHERENT` uniform buffer whose size
//! is specified by the caller (size-oriented, not struct-typed; the concrete
//! `FrameUniforms` struct lives in `minos-render`).
//!
//! # Usage
//! ```text
//! let desc = DescriptorRing::new(&device, &mut buffers, frames_in_flight, uniform_size)?;
//! // each frame:
//! desc.update_frame_uniforms(&mut buffers, ring_index, &raw_bytes)?;
//! // then bind:
//! device.cmd_bind_descriptor_sets(cmd, ..., desc.set(ring_index), ...);
//! ```

use ash::vk;
use gpu_allocator::vulkan::Allocator;

use crate::{buffer::BufferStore, BufferHandle, RhiError};

// ── DescriptorRing ─────────────────────────────────────────────────────────

/// Holds the descriptor set layout, pool, per-frame sets, and per-frame UBOs.
///
/// One instance lives inside [`crate::Rhi`] and is destroyed before the device.
pub(crate) struct DescriptorRing {
    pub layout: vk::DescriptorSetLayout,
    pool: vk::DescriptorPool,
    /// One set per ring frame.
    pub sets: Vec<vk::DescriptorSet>,
    /// One UBO per ring frame. Allocated from `BufferStore`.
    pub ubo_handles: Vec<BufferHandle>,
    device: ash::Device,
}

impl DescriptorRing {
    /// Create layout, pool, per-frame sets, and per-frame UBOs.
    ///
    /// `ubo_size`: byte size of the per-frame uniform struct. The caller (minos-render)
    /// knows this via `std::mem::size_of::<FrameUniforms>()`.
    pub fn new(
        device: &ash::Device,
        buffers: &mut BufferStore,
        allocator: &mut Allocator,
        frames_in_flight: usize,
        ubo_size: u64,
    ) -> Result<Self, RhiError> {
        // ── Descriptor set layout ────────────────────────────────────────────
        let binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);

        let layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(std::slice::from_ref(&binding));

        let layout = unsafe { device.create_descriptor_set_layout(&layout_info, None) }
            .map_err(RhiError::Vulkan)?;

        // ── Descriptor pool ──────────────────────────────────────────────────
        let pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(frames_in_flight as u32);

        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(frames_in_flight as u32)
            .pool_sizes(std::slice::from_ref(&pool_size));

        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(RhiError::Vulkan)?;

        // ── Allocate one set per frame ────────────────────────────────────────
        let layouts: Vec<vk::DescriptorSetLayout> = vec![layout; frames_in_flight];
        let alloc_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(pool)
            .set_layouts(&layouts);

        let sets =
            unsafe { device.allocate_descriptor_sets(&alloc_info) }.map_err(RhiError::Vulkan)?;

        // ── Per-frame UBOs + initial descriptor writes ───────────────────────
        let mut ubo_handles = Vec::with_capacity(frames_in_flight);

        for (i, &set) in sets.iter().enumerate() {
            let handle = buffers.create_uniform_buffer(allocator, ubo_size)?;
            ubo_handles.push(handle);

            let (vk_buf, _) = buffers.vk_buffer_with_size(handle)?;

            let buf_info = vk::DescriptorBufferInfo::default()
                .buffer(vk_buf)
                .offset(0)
                .range(ubo_size);

            let write = vk::WriteDescriptorSet::default()
                .dst_set(set)
                .dst_binding(0)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .buffer_info(std::slice::from_ref(&buf_info));

            unsafe { device.update_descriptor_sets(std::slice::from_ref(&write), &[]) };

            log::debug!("DescriptorRing: frame {} UBO created ({} bytes)", i, ubo_size);
        }

        Ok(Self {
            layout,
            pool,
            sets,
            ubo_handles,
            device: device.clone(),
        })
    }

    /// Write raw bytes into the per-frame UBO identified by `ring_index`.
    ///
    /// `bytes.len()` must equal the `ubo_size` passed to [`Self::new`].
    pub fn update_frame_uniforms(
        &self,
        buffers: &mut BufferStore,
        ring_index: usize,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        let handle = *self
            .ubo_handles
            .get(ring_index)
            .ok_or(RhiError::InvalidHandle)?;
        buffers.write_buffer(handle, bytes)
    }

    /// Return the descriptor set for the given ring frame.
    pub fn set(&self, ring_index: usize) -> vk::DescriptorSet {
        self.sets[ring_index]
    }
}

impl Drop for DescriptorRing {
    fn drop(&mut self) {
        unsafe {
            // Pool destruction implicitly frees all allocated sets.
            self.device.destroy_descriptor_pool(self.pool, None);
            self.device.destroy_descriptor_set_layout(self.layout, None);
        }
    }
}

// ── Generic descriptor sets (storage/uniform; compute + vertex pulling) ───────

/// One binding in a generic descriptor set layout.
///
/// Generic GPU-driven support (not Nanite-specific): used to build the
/// storage-buffer sets a compute cull / indirect-draw pipeline needs.
#[derive(Clone, Copy, Debug)]
pub struct BindingDesc {
    pub binding: u32,
    pub ty: vk::DescriptorType,
    pub stages: vk::ShaderStageFlags,
}

/// A general-purpose descriptor pool + layout registry for storage/uniform sets,
/// separate from the per-frame UBO [`DescriptorRing`]. Owns every layout it
/// creates and destroys them (with the pool) on drop.
pub(crate) struct GeneralDescriptors {
    pool: vk::DescriptorPool,
    layouts: Vec<vk::DescriptorSetLayout>,
    device: ash::Device,
}

impl GeneralDescriptors {
    pub fn new(device: &ash::Device) -> Result<Self, RhiError> {
        let sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .descriptor_count(256),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(64),
            // TAA resolve: sampled HDR/depth/history images + a sampler.
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(64),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(16),
        ];
        let info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(64)
            .pool_sizes(&sizes);
        let pool = unsafe { device.create_descriptor_pool(&info, None) }
            .map_err(RhiError::Vulkan)?;
        Ok(Self {
            pool,
            layouts: Vec::new(),
            device: device.clone(),
        })
    }

    pub fn create_layout(
        &mut self,
        bindings: &[BindingDesc],
    ) -> Result<vk::DescriptorSetLayout, RhiError> {
        let vk_bindings: Vec<vk::DescriptorSetLayoutBinding> = bindings
            .iter()
            .map(|b| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(b.binding)
                    .descriptor_type(b.ty)
                    .descriptor_count(1)
                    .stage_flags(b.stages)
            })
            .collect();
        let info = vk::DescriptorSetLayoutCreateInfo::default().bindings(&vk_bindings);
        let layout = unsafe { self.device.create_descriptor_set_layout(&info, None) }
            .map_err(RhiError::Vulkan)?;
        self.layouts.push(layout);
        Ok(layout)
    }

    pub fn allocate(
        &self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, RhiError> {
        let layouts = [layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pool)
            .set_layouts(&layouts);
        let sets = unsafe { self.device.allocate_descriptor_sets(&info) }
            .map_err(RhiError::Vulkan)?;
        Ok(sets[0])
    }

    pub fn write_buffer(
        &self,
        set: vk::DescriptorSet,
        binding: u32,
        ty: vk::DescriptorType,
        buffer: vk::Buffer,
        offset: u64,
        range: u64,
    ) {
        let info = vk::DescriptorBufferInfo::default()
            .buffer(buffer)
            .offset(offset)
            .range(range);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(binding)
            .dst_array_element(0)
            .descriptor_type(ty)
            .buffer_info(std::slice::from_ref(&info));
        unsafe {
            self.device
                .update_descriptor_sets(std::slice::from_ref(&write), &[])
        };
    }

    /// Point a `SAMPLED_IMAGE` binding at `view` (in `layout`, normally
    /// `SHADER_READ_ONLY_OPTIMAL`). Used by the TAA resolve set.
    #[allow(dead_code)] // wired by the TAA pass.
    pub fn write_sampled_image(
        &self,
        set: vk::DescriptorSet,
        binding: u32,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) {
        let info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(layout);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(binding)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .image_info(std::slice::from_ref(&info));
        unsafe {
            self.device
                .update_descriptor_sets(std::slice::from_ref(&write), &[])
        };
    }

    /// Point a `SAMPLER` binding at `sampler`. Used by the TAA resolve set.
    #[allow(dead_code)] // wired by the TAA pass.
    pub fn write_sampler(&self, set: vk::DescriptorSet, binding: u32, sampler: vk::Sampler) {
        let info = vk::DescriptorImageInfo::default().sampler(sampler);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(set)
            .dst_binding(binding)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .image_info(std::slice::from_ref(&info));
        unsafe {
            self.device
                .update_descriptor_sets(std::slice::from_ref(&write), &[])
        };
    }
}

impl Drop for GeneralDescriptors {
    fn drop(&mut self) {
        unsafe {
            for &layout in &self.layouts {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
            self.device.destroy_descriptor_pool(self.pool, None);
        }
    }
}

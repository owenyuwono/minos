//! `enki-rhi` — the only `ash` consumer in the workspace.
//!
//! All Vulkan types are confined to this crate. Downstream crates depend on the
//! safe, typed abstractions exported here.

pub mod buffer;
pub mod command;
pub mod descriptor;
pub mod device;
pub mod image;
pub mod instance;
pub mod memory;
pub mod pipeline;
pub mod queue;
pub mod shader;
pub mod streaming;
pub mod surface;
pub mod swapchain;
pub mod sync;
mod taa_pass;

pub use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::mem::ManuallyDrop;
use thiserror::Error;

// ── Error ──────────────────────────────────────────────────────────────────

/// Top-level RHI error type. All fallible RHI operations return this.
#[derive(Debug, Error)]
pub enum RhiError {
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
    #[error("No suitable physical device found")]
    NoSuitableDevice,
    #[error("Required Vulkan extension not available: {0}")]
    MissingExtension(String),
    #[error("Required Vulkan layer not available: {0}")]
    MissingLayer(String),
    #[error("Allocation failed: {0}")]
    AllocationFailed(String),
    #[error("Buffer handle is invalid or already retired")]
    InvalidHandle,
    #[error("Surface error")]
    Surface,
    #[error(transparent)]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

// ── Config ─────────────────────────────────────────────────────────────────

/// Configuration passed to [`Rhi::new`].
#[derive(Debug, Clone)]
pub struct RhiConfig {
    /// Enable Vulkan validation layers. Should be `true` in debug, `false` in release.
    pub validation: bool,
    /// Number of frames in flight for CPU/GPU overlap. Typical: 2 or 3.
    pub frames_in_flight: u32,
    /// MSAA sample count (1 = off, 4 = 4×). Must be supported by the device.
    pub msaa_samples: u32,
    /// Application name passed to Vulkan instance creation.
    pub app_name: String,
    /// Byte size of the per-frame uniform buffer.
    /// Pass `std::mem::size_of::<YourFrameUniforms>() as u64`.
    /// Must be > 0.
    pub uniform_buffer_size: u64,
}

impl Default for RhiConfig {
    fn default() -> Self {
        Self {
            validation: cfg!(debug_assertions),
            frames_in_flight: 2,
            msaa_samples: 4,
            app_name: "enki".to_string(),
            uniform_buffer_size: 256, // generous default; caller should set explicitly
        }
    }
}

// ── Handle types ───────────────────────────────────────────────────────────

/// Opaque handle to a GPU buffer managed by the RHI.
/// Obtained from buffer creation methods; retired via [`Rhi::destroy_buffer`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BufferHandle(u64);

// Re-export pipeline and shader handles at the crate root for caller convenience.
pub use descriptor::BindingDesc;
pub use pipeline::ComputePipelineDesc;
pub use pipeline::GraphicsPipelineDesc;
pub use pipeline::PipelineHandle;
pub use shader::ShaderModule;

// Re-export streaming types at the crate root for caller convenience.
pub use streaming::{StreamedMesh, DEFAULT_FRAME_BUDGET_BYTES};

// ── Rhi ────────────────────────────────────────────────────────────────────

/// Top-level RHI handle. Owns all Vulkan resources.
///
/// # Lifetime
/// Create once at startup; drop at shutdown (after the device is idle).
/// There must be exactly one `Rhi` per window surface.
///
/// # Intended call sequence per frame
///
/// Two rendering instances are opened per frame: a 4× MSAA 3D pass and a
/// separate 1-sample UI pass on the resolved swapchain image.  The two-pass
/// split is required because `egui-ash-renderer` 0.11 hardcodes its pipeline
/// to `rasterizationSamples = TYPE_1` and cannot render into an MSAA image.
///
/// ```text
/// let fi = rhi.begin_frame()?;
/// if fi == u32::MAX { rhi.end_frame(fi)?; return Ok(()); } // window minimised
///
/// // Optional: record streaming uploads BEFORE begin_rendering.
/// // vkCmdCopyBuffer is illegal inside a dynamic-rendering instance.
/// // if let Some(mesh) = rhi.streaming_upload(fi, ...)? { ... }
///
/// // ── Pass 1: 4× MSAA 3D rendering instance ──────────────────────────
/// rhi.begin_rendering(fi);
///
/// rhi.set_viewport_scissor_full(fi);
/// rhi.bind_pipeline(fi, pipeline_handle)?;
/// rhi.update_frame_uniforms(fi, bytemuck::bytes_of(&my_frame_uniforms))?;
/// rhi.bind_vertex_buffers(fi, &[pos_buf, nrm_buf, col_buf, plate_buf])?;
/// rhi.bind_index_buffer(fi, idx_buf)?;
/// rhi.push_constants(fi, bytemuck::bytes_of(&my_push))?;
/// rhi.draw_indexed(fi, index_count);
///
/// // Optional: retire streamed meshes after drawing them.
/// // rhi.streaming_retire(mesh);
///
/// // ── Pass 2: 1-sample UI instance on the resolved swapchain image ────
/// // begin_ui_pass closes the MSAA instance, inserts an inter-pass barrier,
/// // and opens the 1-sample UI instance (loadOp=LOAD, no depth attachment).
/// rhi.begin_ui_pass(fi);
/// // egui records into the 1-sample instance here.
/// egui_state.render(&rhi, fi);
///
/// // end_frame closes the UI instance, then submits + presents.
/// rhi.end_frame(fi)?;
/// ```
pub struct Rhi {
    // ManuallyDrop lets us control destruction order in Drop::drop.
    // Required ordering: pipelines → descriptors/buffers → sync/commands →
    //   swapchain images → (msaa/depth) → swapchain →
    //   device (allocator first, then handle) → surface → instance.
    instance: ManuallyDrop<instance::Instance>,
    surface: ManuallyDrop<surface::Surface>,
    device: ManuallyDrop<device::Device>,
    queue: queue::Queue,
    swapchain: swapchain::Swapchain,
    depth_image: Option<image::AllocatedImage>,
    msaa_image: Option<image::AllocatedImage>,
    sync: ManuallyDrop<sync::SyncObjects>,
    commands: ManuallyDrop<command::CommandPools>,
    current_frame: usize,
    /// Tracks which swapchain image index was acquired for each ring-frame slot.
    /// Indexed by the frame-slot index (0..frames_in_flight).
    image_indices: Vec<u32>,
    frame_counter: u64,
    frames_in_flight: usize,
    msaa_sample_count: vk::SampleCountFlags,
    clear_color: [f32; 4],
    needs_recreate: bool,

    // ── M1 additions ───────────────────────────────────────────────────────
    /// All GPU buffers (vertex, index, uniform). Destroyed before the device.
    buffers: buffer::BufferStore,
    /// Per-frame descriptor sets + UBOs.
    /// ManuallyDrop so we control teardown order relative to `buffers` and `device`.
    descriptors: ManuallyDrop<descriptor::DescriptorRing>,
    /// General storage/uniform descriptor pool + layouts (compute + vertex pulling).
    /// Generic GPU-driven support; destroyed before the device.
    general_descriptors: ManuallyDrop<descriptor::GeneralDescriptors>,
    /// All graphics pipelines.
    /// ManuallyDrop so we destroy pipelines before sync/commands.
    pipelines: ManuallyDrop<pipeline::PipelineStore>,
    /// The current active pipeline layout — set by `bind_pipeline`, used by
    /// `push_constants` and descriptor binding. None until first `bind_pipeline`.
    current_pipeline_layout: Option<vk::PipelineLayout>,

    // ── M2 additions ───────────────────────────────────────────────────────
    /// Dynamic geometry streaming: staging ring, graveyard, DEVICE_LOCAL buffers.
    /// Destroyed after device_wait_idle, before the device itself.
    streamer: Option<streaming::StreamingUploader>,

    // ── TAA ─────────────────────────────────────────────────────────────────
    /// TAA resolve resources (offscreen targets + history + pipeline). `None`
    /// until [`Self::init_taa`]. Images recreated on resize, all freed in Drop.
    taa: Option<taa_pass::TaaResources>,
    /// When true (and `taa` is initialised), the frame bracket renders the 3D
    /// pass to the TAA targets and runs the resolve. Toggled by the app.
    taa_enabled: bool,
}

impl Rhi {
    /// Create the RHI: instance → debug messenger → surface → device → queues →
    /// allocator → swapchain → depth/MSAA images → sync primitives → command pools →
    /// descriptor ring → buffer/pipeline stores.
    pub fn new<W>(window: &W, config: RhiConfig) -> Result<Self, RhiError>
    where
        W: HasWindowHandle + HasDisplayHandle,
    {
        let display_handle = window
            .display_handle()
            .map_err(|e| RhiError::Other(e.to_string().into()))?
            .as_raw();
        let window_handle = window
            .window_handle()
            .map_err(|e| RhiError::Other(e.to_string().into()))?
            .as_raw();

        let instance = instance::Instance::new(display_handle, &config)?;
        let surface = surface::Surface::new(
            &instance.entry,
            &instance.handle,
            display_handle,
            window_handle,
        )?;
        let mut device = device::Device::new(&instance.handle, &surface, &config)?;
        let queue = queue::Queue::from_device(&device.handle, device.graphics_queue_family);

        let swapchain = swapchain::Swapchain::new(
            &instance.handle,
            &device.handle,
            device.physical,
            &surface,
            device.graphics_queue_family,
            None,
        )?;

        let msaa_sample_count = config_to_sample_count(config.msaa_samples);

        // Split borrow: device.handle (&ash::Device) and device.allocator (&mut Allocator)
        // are disjoint fields, but the borrow checker can't see through ManuallyDrop.
        // Using raw pointers is the standard pattern here.
        let (depth_image, msaa_image) =
            if swapchain.extent.width > 0 && swapchain.extent.height > 0 {
                let dev_h: *const ash::Device = &device.handle;
                let dev_a: *mut gpu_allocator::vulkan::Allocator = &mut *device.allocator;
                let depth = image::AllocatedImage::new_depth(
                    unsafe { &*dev_h },
                    unsafe { &mut *dev_a },
                    swapchain.extent,
                    msaa_sample_count,
                )?;
                let msaa = image::AllocatedImage::new_msaa_color(
                    unsafe { &*dev_h },
                    unsafe { &mut *dev_a },
                    swapchain.extent,
                    swapchain.format,
                    msaa_sample_count,
                )?;
                (Some(depth), Some(msaa))
            } else {
                (None, None)
            };

        let frames_in_flight = config.frames_in_flight;
        let sync = sync::SyncObjects::new(&device.handle, frames_in_flight)?;
        let commands = command::CommandPools::new(
            &device.handle,
            device.graphics_queue_family,
            frames_in_flight,
        )?;

        // ── M1: buffer store, descriptor ring, pipeline store ───────────────
        let mut buffers = buffer::BufferStore::new(&device.handle);
        let descriptors = {
            let dev_h: *const ash::Device = &device.handle;
            let dev_a: *mut gpu_allocator::vulkan::Allocator = &mut *device.allocator;
            descriptor::DescriptorRing::new(
                unsafe { &*dev_h },
                &mut buffers,
                unsafe { &mut *dev_a },
                frames_in_flight as usize,
                config.uniform_buffer_size,
            )?
        };
        let pipelines = pipeline::PipelineStore::new(&device.handle);
        let general_descriptors = descriptor::GeneralDescriptors::new(&device.handle)?;

        // ── M2: streaming uploader ─────────────────────────────────────────
        let streamer = {
            let dev_h: *const ash::Device = &device.handle;
            let dev_a: *mut gpu_allocator::vulkan::Allocator = &mut *device.allocator;
            streaming::StreamingUploader::new(
                unsafe { &*dev_h },
                unsafe { &mut *dev_a },
                frames_in_flight as usize,
                streaming::DEFAULT_FRAME_BUDGET_BYTES,
            )?
        };

        log::info!("RHI initialised ({} frames in flight)", frames_in_flight);

        Ok(Self {
            instance: ManuallyDrop::new(instance),
            surface: ManuallyDrop::new(surface),
            device: ManuallyDrop::new(device),
            queue,
            swapchain,
            depth_image,
            msaa_image,
            sync: ManuallyDrop::new(sync),
            commands: ManuallyDrop::new(commands),
            current_frame: 0,
            image_indices: vec![0u32; frames_in_flight as usize],
            frame_counter: 0,
            frames_in_flight: frames_in_flight as usize,
            msaa_sample_count,
            clear_color: [0.02, 0.03, 0.05, 1.0],
            needs_recreate: false,
            buffers,
            descriptors: ManuallyDrop::new(descriptors),
            general_descriptors: ManuallyDrop::new(general_descriptors),
            pipelines: ManuallyDrop::new(pipelines),
            current_pipeline_layout: None,
            streamer: Some(streamer),
            taa: None,
            taa_enabled: false,
        })
    }

    // ── Shader ─────────────────────────────────────────────────────────────

    /// Compile WGSL source to a `VkShaderModule`.
    ///
    /// The source may contain multiple entry points (e.g. `vs_main` and `fs_main`).
    /// All are included in the compiled module; which ones to use is chosen in
    /// [`GraphicsPipelineDesc`].
    ///
    /// # Errors
    /// Returns `RhiError::Other` on WGSL parse/validation failure, with the naga
    /// diagnostic in the message.
    pub fn create_shader_module(&self, wgsl_source: &str) -> Result<ShaderModule, RhiError> {
        shader::compile_wgsl(&self.device.handle, wgsl_source)
    }

    /// Destroy a shader module. Safe to call after pipeline creation because Vulkan
    /// does not need the module alive after the pipeline is compiled.
    pub fn destroy_shader_module(&self, module: ShaderModule) {
        unsafe { self.device.handle.destroy_shader_module(module.handle, None) };
    }

    // ── Buffers ────────────────────────────────────────────────────────────

    /// Create a vertex buffer and upload `data` into it (HOST_VISIBLE, no staging).
    ///
    /// The buffer is intended for static geometry (M1). M2 will add streaming
    /// DEVICE_LOCAL buffers via a staging uploader.
    pub fn create_vertex_buffer(&mut self, data: &[u8]) -> Result<BufferHandle, RhiError> {
        let dev_a: *mut gpu_allocator::vulkan::Allocator = &mut *self.device.allocator;
        self.buffers.create_vertex_buffer(unsafe { &mut *dev_a }, data)
    }

    /// Create an index buffer (u32 indices) and upload `indices` immediately.
    pub fn create_index_buffer(&mut self, indices: &[u32]) -> Result<BufferHandle, RhiError> {
        let dev_a: *mut gpu_allocator::vulkan::Allocator = &mut *self.device.allocator;
        self.buffers.create_index_buffer(unsafe { &mut *dev_a }, indices)
    }

    /// Destroy a buffer and retire its handle.
    pub fn destroy_buffer(&mut self, handle: BufferHandle) {
        let dev_a: *mut gpu_allocator::vulkan::Allocator = &mut *self.device.allocator;
        self.buffers.destroy_buffer(unsafe { &mut *dev_a }, handle);
    }

    /// Copy the contents of a host-visible buffer back into a `Vec<u8>` (the full
    /// allocated size). Only valid for `host_visible` buffers. Used by the flora
    /// screenshot path to read back a TRANSFER_DST readback buffer after a GPU
    /// image→buffer copy + `wait_idle`.
    pub fn read_buffer(&self, handle: BufferHandle) -> Result<Vec<u8>, RhiError> {
        self.buffers.read_buffer(handle)
    }

    // ── M2 Streaming API ───────────────────────────────────────────────────

    /// Upload mesh vertex/index data to DEVICE_LOCAL buffers via a staging ring.
    ///
    /// Records a `vkCmdCopyBuffer` and a copy→draw `vkCmdPipelineBarrier2` into
    /// the current frame's command buffer.  The barrier ensures the copy completes
    /// before any subsequent vertex/index reads on the same queue.
    ///
    /// # Back-pressure
    /// Returns `Ok(None)` when the per-frame byte budget or the staging ring
    /// segment is exhausted.  The caller retries next frame — never stalls.
    ///
    /// # Arguments
    /// - `fi`           — Frame-in-flight slot index (the value from `begin_frame`).
    /// - `positions`    — Per-vertex positions `[f32; 3]`.
    /// - `normals`      — Per-vertex normals `[f32; 3]`.
    /// - `colors`       — Per-vertex colours `[f32; 3]`.
    /// - `plate_colors` — Optional per-vertex plate colours; `colors` is reused if `None`.
    /// - `indices`      — Triangle indices (`u32`).
    ///
    /// # Returns
    /// `Ok(Some(StreamedMesh))` on success, `Ok(None)` on back-pressure,
    /// `Err(_)` on a fatal Vulkan/allocation error.
    pub fn streaming_upload(
        &mut self,
        fi: u32,
        positions: &[[f32; 3]],
        normals: &[[f32; 3]],
        colors: &[[f32; 3]],
        plate_colors: Option<&[[f32; 3]]>,
        indices: &[u32],
    ) -> Result<Option<StreamedMesh>, RhiError> {
        let fi = fi as usize;
        let cmd = self.commands.frames[fi].buffer;
        let frame_counter = self.frame_counter;
        let dev_a: *mut gpu_allocator::vulkan::Allocator = &mut *self.device.allocator;

        let streamer = self.streamer.as_mut().ok_or_else(|| {
            RhiError::Other("streaming uploader not initialised".into())
        })?;

        streamer.upload(
            fi,
            frame_counter,
            cmd,
            unsafe { &mut *dev_a },
            positions,
            normals,
            colors,
            plate_colors,
            indices,
        )
    }

    /// Schedule a `StreamedMesh` for deferred destruction.
    ///
    /// The underlying VkBuffers are freed only once `gpu_completed_frame() >= retire_frame`,
    /// where `retire_frame = frame_counter + 1` — the timeline value the current recording
    /// frame will signal when its submission completes.  Using `+ 1` here ensures the
    /// buffer is freed only after the GPU completes the frame that drew it, even when the
    /// mesh is drawn and retired in the same frame.
    ///
    /// INVARIANT: never free a buffer before the GPU has completed the frame that drew it.
    /// This is the use-after-free guard.
    pub fn streaming_retire(&mut self, mesh: StreamedMesh) {
        if let Some(streamer) = self.streamer.as_mut() {
            streamer.retire(mesh, self.frame_counter + 1);
        }
    }

    /// Free all graveyard entries whose retire_frame has been completed by the GPU.
    ///
    /// Call once per frame: `rhi.collect_streaming_garbage()`.
    pub fn collect_streaming_garbage(&mut self) {
        let completed = self.gpu_completed_frame();
        let dev_a: *mut gpu_allocator::vulkan::Allocator = &mut *self.device.allocator;
        if let Some(streamer) = self.streamer.as_mut() {
            streamer.collect_garbage(completed, unsafe { &mut *dev_a });
        }
    }

    /// Number of retired meshes still waiting for GPU completion in the graveyard.
    ///
    /// A non-zero value is normal: meshes are held until `gpu_completed_frame >= retire_frame`.
    /// A persistently growing value across LOD ascent cycles indicates a deferred-destruction leak.
    pub fn streaming_graveyard_size(&self) -> usize {
        self.streamer.as_ref().map_or(0, |s| s.graveyard_size())
    }

    /// Read the timeline semaphore counter, returning the highest frame index the
    /// GPU has fully completed.
    ///
    /// Safe to call at any time.  Returns 0 if the query fails (conservative — no
    /// buffers will be prematurely freed).
    pub fn gpu_completed_frame(&self) -> u64 {
        let result = unsafe {
            self.device.handle.get_semaphore_counter_value(self.sync.timeline)
        };
        result.unwrap_or(0)
    }

    // ── Descriptor ring helpers ─────────────────────────────────────────────

    /// Return the descriptor set layout for set 0 (UNIFORM_BUFFER at binding 0).
    ///
    /// Pass this to [`GraphicsPipelineDesc::set0_layout`] when creating pipelines.
    pub fn set0_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptors.layout
    }

    // ── Pipeline ───────────────────────────────────────────────────────────

    /// Create a graphics pipeline. See [`GraphicsPipelineDesc`] for all knobs.
    ///
    /// The `ShaderModule` in `desc` can be destroyed after this call returns.
    pub fn create_graphics_pipeline(
        &mut self,
        desc: &GraphicsPipelineDesc<'_>,
    ) -> Result<PipelineHandle, RhiError> {
        self.pipelines.create(desc)
    }

    /// Create a compute pipeline. See [`ComputePipelineDesc`].
    ///
    /// Generic GPU-compute support (not Nanite-specific). The `ShaderModule` can
    /// be destroyed after this returns.
    pub fn create_compute_pipeline(
        &mut self,
        desc: &ComputePipelineDesc<'_>,
    ) -> Result<PipelineHandle, RhiError> {
        self.pipelines.create_compute(desc)
    }

    /// Create a graphics pipeline with NO vertex input bindings (vertex pulling):
    /// the vertex shader reads geometry from storage buffers via `vertex_index`.
    pub fn create_graphics_pipeline_pulling(
        &mut self,
        desc: &GraphicsPipelineDesc<'_>,
    ) -> Result<PipelineHandle, RhiError> {
        self.pipelines.create_pulling(desc)
    }

    // ── Storage buffers (generic GPU-driven data) ───────────────────────────

    /// Create a storage buffer (SSBO). `host_visible = true` makes it mappable
    /// and readable via [`Self::read_storage_buffer`]; `false` is DEVICE_LOCAL.
    pub fn create_storage_buffer(
        &mut self,
        data: &[u8],
        host_visible: bool,
    ) -> Result<BufferHandle, RhiError> {
        let dev_a: *mut gpu_allocator::vulkan::Allocator = &mut *self.device.allocator;
        self.buffers
            .create_storage_buffer(unsafe { &mut *dev_a }, data, host_visible)
    }

    /// Read a host-visible buffer's contents back to CPU.
    pub fn read_storage_buffer(&self, handle: BufferHandle) -> Result<Vec<u8>, RhiError> {
        self.buffers.read_buffer(handle)
    }

    /// Create a buffer with caller-chosen usage flags (indirect-args, visible-list,
    /// per-frame uniforms, …). `host_visible` makes it CPU-mappable.
    pub fn create_gpu_buffer(
        &mut self,
        size: u64,
        host_visible: bool,
        usage: vk::BufferUsageFlags,
    ) -> Result<BufferHandle, RhiError> {
        let dev_a: *mut gpu_allocator::vulkan::Allocator = &mut *self.device.allocator;
        self.buffers
            .create_gpu_buffer(unsafe { &mut *dev_a }, size, host_visible, usage)
    }

    /// Write raw bytes into a host-visible buffer (per-frame uniforms / indirect args).
    pub fn write_storage_bytes(
        &mut self,
        handle: BufferHandle,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        self.buffers.write_buffer(handle, bytes)
    }

    /// Write raw bytes into a host-visible buffer at byte `offset` (partial update —
    /// e.g. one cluster slot in the Nanite streaming page pool, where the rest of
    /// the buffer must stay intact). `offset + bytes.len()` must be ≤ buffer size.
    pub fn write_storage_bytes_at(
        &mut self,
        handle: BufferHandle,
        offset: u64,
        bytes: &[u8],
    ) -> Result<(), RhiError> {
        self.buffers.write_buffer_at(handle, offset, bytes)
    }

    // ── Generic descriptor sets (storage/uniform) ───────────────────────────

    /// Create a descriptor set layout from `bindings`. The layout is owned by the
    /// RHI and destroyed at shutdown. Pass it to [`Self::create_compute_pipeline`]
    /// / [`GraphicsPipelineDesc`] and [`Self::allocate_descriptor_set`].
    pub fn create_descriptor_set_layout(
        &mut self,
        bindings: &[BindingDesc],
    ) -> Result<vk::DescriptorSetLayout, RhiError> {
        self.general_descriptors.create_layout(bindings)
    }

    /// Allocate a descriptor set with the given layout from the general pool.
    pub fn allocate_descriptor_set(
        &self,
        layout: vk::DescriptorSetLayout,
    ) -> Result<vk::DescriptorSet, RhiError> {
        self.general_descriptors.allocate(layout)
    }

    /// Point a `STORAGE_BUFFER` binding of `set` at the whole of `handle`.
    pub fn write_storage_binding(
        &self,
        set: vk::DescriptorSet,
        binding: u32,
        handle: BufferHandle,
    ) -> Result<(), RhiError> {
        let (buf, size) = self.buffers.vk_buffer_with_size(handle)?;
        self.general_descriptors
            .write_buffer(set, binding, vk::DescriptorType::STORAGE_BUFFER, buf, 0, size);
        Ok(())
    }

    /// Point a `UNIFORM_BUFFER` binding of `set` at the whole of `handle`.
    pub fn write_uniform_binding(
        &self,
        set: vk::DescriptorSet,
        binding: u32,
        handle: BufferHandle,
    ) -> Result<(), RhiError> {
        let (buf, size) = self.buffers.vk_buffer_with_size(handle)?;
        self.general_descriptors
            .write_buffer(set, binding, vk::DescriptorType::UNIFORM_BUFFER, buf, 0, size);
        Ok(())
    }

    /// Point a `SAMPLED_IMAGE` binding of `set` at `view` (layout normally
    /// `SHADER_READ_ONLY_OPTIMAL`). For the TAA resolve set's image bindings.
    pub fn write_sampled_image_binding(
        &self,
        set: vk::DescriptorSet,
        binding: u32,
        view: vk::ImageView,
        layout: vk::ImageLayout,
    ) {
        self.general_descriptors
            .write_sampled_image(set, binding, view, layout);
    }

    /// Point a `SAMPLER` binding of `set` at `sampler`. For the TAA resolve set.
    pub fn write_sampler_binding(
        &self,
        set: vk::DescriptorSet,
        binding: u32,
        sampler: vk::Sampler,
    ) {
        self.general_descriptors.write_sampler(set, binding, sampler);
    }

    // ── Samplers / images (post-process: TAA resolve) ───────────────────────

    /// Create a sampler (linear filter, clamp-to-edge). The caller owns it and
    /// must call [`Self::destroy_sampler`] before RHI teardown.
    pub fn create_sampler(&self) -> Result<vk::Sampler, RhiError> {
        let info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .min_lod(0.0)
            .max_lod(0.0);
        unsafe { self.device.handle.create_sampler(&info, None) }.map_err(RhiError::Vulkan)
    }

    /// Destroy a sampler created by [`Self::create_sampler`].
    pub fn destroy_sampler(&self, sampler: vk::Sampler) {
        unsafe { self.device.handle.destroy_sampler(sampler, None) };
    }

    /// Create a fullscreen post-process pipeline (no vertex input, no depth, one
    /// color attachment, custom set layout). For the TAA resolve pass.
    pub fn create_fullscreen_pipeline(
        &mut self,
        shader: &ShaderModule,
        vs_entry: &str,
        fs_entry: &str,
        set_layout: vk::DescriptorSetLayout,
        color_format: vk::Format,
    ) -> Result<PipelineHandle, RhiError> {
        self.pipelines
            .create_fullscreen(shader, vs_entry, fs_entry, set_layout, color_format)
    }

    // ── TAA (temporal anti-aliasing) ─────────────────────────────────────────

    /// Initialise the TAA resolve resources. Call once after the swapchain is
    /// sized. `shader` must hold the resolve fullscreen `vs`/`fs`; `ubo_size` is
    /// `size_of::<ResolveParams>()`. TAA stays OFF until `set_taa_enabled(true)`.
    pub fn init_taa(
        &mut self,
        shader: &ShaderModule,
        vs_entry: &str,
        fs_entry: &str,
        ubo_size: u64,
    ) -> Result<(), RhiError> {
        let frag = vk::ShaderStageFlags::FRAGMENT;
        let img = vk::DescriptorType::SAMPLED_IMAGE;
        let bindings = [
            BindingDesc { binding: 0, ty: img, stages: frag },
            BindingDesc { binding: 1, ty: img, stages: frag },
            BindingDesc { binding: 2, ty: img, stages: frag },
            BindingDesc { binding: 3, ty: vk::DescriptorType::SAMPLER, stages: frag },
            BindingDesc { binding: 4, ty: vk::DescriptorType::UNIFORM_BUFFER, stages: frag },
        ];
        let set_layout = self.create_descriptor_set_layout(&bindings)?;
        let pipeline = self.create_fullscreen_pipeline(
            shader, vs_entry, fs_entry, set_layout, self.swapchain.format,
        )?;
        let sampler = self.create_sampler()?;

        let fif = self.frames_in_flight;
        let mut sets = Vec::with_capacity(fif);
        let mut ubos = Vec::with_capacity(fif);
        for _ in 0..fif {
            let set = self.allocate_descriptor_set(set_layout)?;
            let ubo = self.create_gpu_buffer(
                ubo_size.max(16),
                true,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            // Static bindings (sampler + UBO); image bindings written per-resolve.
            self.write_sampler_binding(set, 3, sampler);
            self.write_uniform_binding(set, 4, ubo)?;
            sets.push(set);
            ubos.push(ubo);
        }

        let images = self.create_taa_images()?;

        self.taa = Some(taa_pass::TaaResources {
            images,
            history_layout: [vk::ImageLayout::UNDEFINED; 2],
            sampler,
            set_layout,
            pipeline,
            sets,
            ubos,
            write_index: 0,
        });
        log::info!("TAA initialised (resolve pipeline + targets ready; toggle currently OFF)");
        Ok(())
    }

    /// (Re)build the TAA offscreen images at the current extent (`None` if zero).
    fn create_taa_images(&mut self) -> Result<Option<taa_pass::TaaImages>, RhiError> {
        if self.swapchain.extent.width == 0 || self.swapchain.extent.height == 0 {
            return Ok(None);
        }
        let dev_h: *const ash::Device = &self.device.handle;
        let dev_a: *mut gpu_allocator::vulkan::Allocator = &mut *self.device.allocator;
        let images = taa_pass::TaaImages::new(
            unsafe { &*dev_h },
            unsafe { &mut *dev_a },
            self.swapchain.extent,
            self.swapchain.format,
            self.msaa_sample_count,
        )?;
        Ok(Some(images))
    }

    /// Enable/disable the TAA resolve path (no-op until `init_taa`).
    pub fn set_taa_enabled(&mut self, enabled: bool) {
        self.taa_enabled = enabled;
    }

    /// True once `init_taa` has run.
    pub fn taa_ready(&self) -> bool {
        self.taa.is_some()
    }

    /// Whether the TAA path is active this frame (enabled + initialised + images).
    pub fn taa_active(&self) -> bool {
        self.taa_enabled
            && self
                .taa
                .as_ref()
                .map(|t| t.images.is_some())
                .unwrap_or(false)
    }

    /// Run the TAA resolve for frame `fi`: reproject + 3×3 neighborhood-clamp +
    /// blend the current frame against history, then blit the result to the
    /// swapchain. MUST be called after all 3D draws and BEFORE `begin_ui_pass`.
    /// `params` is the `ResolveParams` byte image. No-op when TAA isn't active.
    pub fn taa_resolve(&mut self, fi: u32, params: &[u8]) -> Result<(), RhiError> {
        if fi == u32::MAX || !self.taa_active() {
            return Ok(());
        }
        let fi_idx = fi as usize;
        let image_index = self.image_indices[fi_idx] as usize;
        let cmd = self.commands.frames[fi_idx].buffer;
        let extent = self.swapchain.extent;

        // Pull the handles we need (immutable), then upload the resolve UBO.
        let (write_index, set, pipeline, ubo) = {
            let taa = self.taa.as_ref().unwrap();
            (taa.write_index, taa.sets[fi_idx], taa.pipeline, taa.ubos[fi_idx])
        };
        let prev_index = 1 - write_index;
        self.write_storage_bytes(ubo, params)?;

        let (cur_img, cur_view, depth_img, depth_view, prev_img, prev_view, cur_hist_img, cur_hist_view, prev_layout) = {
            let taa = self.taa.as_ref().unwrap();
            let im = taa.images.as_ref().unwrap();
            (
                im.current.image,
                im.current.view,
                im.resolved_depth.image,
                im.resolved_depth.view,
                im.history[prev_index].image,
                im.history[prev_index].view,
                im.history[write_index].image,
                im.history[write_index].view,
                taa.history_layout[prev_index],
            )
        };
        let swap_img = self.swapchain.images[image_index];
        let read = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

        // Point the per-frame set at this frame's images (sampler + UBO are static).
        self.write_sampled_image_binding(set, 0, cur_view, read);
        self.write_sampled_image_binding(set, 1, depth_view, read);
        self.write_sampled_image_binding(set, 2, prev_view, read);

        let layout = self.pipeline_layout(pipeline)?;
        let device = &self.device.handle;
        let color = color_subresource_range();
        let depth = depth_subresource_range();

        unsafe {
            // Close the 3D rendering instance opened by begin_rendering.
            device.cmd_end_rendering(cmd);

            // Transition: inputs → SHADER_READ, history[cur] → COLOR_ATTACHMENT.
            let b_cur = img_barrier(
                cur_img, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, read,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ, color,
            );
            let b_depth = img_barrier(
                depth_img, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL, read,
                vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ, depth,
            );
            let b_prev = img_barrier(
                prev_img,
                prev_layout, read,
                vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ, color,
            );
            let b_hist = img_barrier(
                cur_hist_img, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE, color,
            );
            let barriers = [b_cur, b_depth, b_prev, b_hist];
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&barriers));

            // Resolve pass → history[cur].
            let attach = vk::RenderingAttachmentInfo::default()
                .image_view(cur_hist_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::DONT_CARE)
                .store_op(vk::AttachmentStoreOp::STORE);
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&attach));
            device.cmd_begin_rendering(cmd, &info);
        }

        // Dynamic viewport/scissor for the fullscreen pipeline.
        self.set_viewport_scissor_full(fi);
        self.cmd_bind_pipeline(fi, vk::PipelineBindPoint::GRAPHICS, pipeline)?;
        self.cmd_bind_descriptor_set(fi, vk::PipelineBindPoint::GRAPHICS, layout, 0, set);

        let device = &self.device.handle;
        unsafe {
            device.cmd_draw(cmd, 3, 1, 0, 0); // fullscreen triangle
            device.cmd_end_rendering(cmd);

            // history[cur] → TRANSFER_SRC, swapchain → TRANSFER_DST, then blit.
            let to_src = img_barrier(
                cur_hist_img, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ, color,
            );
            let to_dst = img_barrier(
                swap_img, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE, color,
            );
            let pre_blit = [to_src, to_dst];
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&pre_blit));

            let sub = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1);
            let offsets = [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D { x: extent.width as i32, y: extent.height as i32, z: 1 },
            ];
            let region = vk::ImageBlit::default()
                .src_subresource(sub)
                .src_offsets(offsets)
                .dst_subresource(sub)
                .dst_offsets(offsets);
            device.cmd_blit_image(
                cmd,
                cur_hist_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                swap_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region),
                vk::Filter::NEAREST,
            );

            // swapchain → COLOR_ATTACHMENT for the UI pass.
            let to_color = img_barrier(
                swap_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                vk::AccessFlags2::COLOR_ATTACHMENT_WRITE | vk::AccessFlags2::COLOR_ATTACHMENT_READ, color,
            );
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&to_color)));
        }

        // Advance ping-pong + record the new layouts.
        let taa = self.taa.as_mut().unwrap();
        taa.history_layout[write_index] = vk::ImageLayout::TRANSFER_SRC_OPTIMAL;
        taa.history_layout[prev_index] = read;
        taa.write_index = prev_index;
        Ok(())
    }

    /// Split the 3D pass for refractive water (TAA path only). After opaque draws,
    /// this closes the MSAA opaque instance (resolving it to `current`), copies the
    /// resolved opaque color into a history scratch image (the refraction source),
    /// and reopens a 1× rendering instance on `current` (+ `resolved_depth`) so the
    /// water can sample the scene behind it. `taa_resolve` then closes this instance
    /// and runs the history blend as usual. No-op (returns false) when TAA is off.
    pub fn begin_water_pass(&mut self, fi: u32) -> Result<bool, RhiError> {
        if fi == u32::MAX || !self.taa_active() {
            return Ok(false);
        }
        let fi_idx = fi as usize;
        let cmd = self.commands.frames[fi_idx].buffer;
        let extent = self.swapchain.extent;
        let (cur_img, cur_view, depth_view, rdepth_img, refr_depth_img, scr_img, scr_old) = {
            let taa = self.taa.as_ref().unwrap();
            let im = taa.images.as_ref().unwrap();
            let w = taa.write_index;
            (im.current.image, im.current.view, im.resolved_depth.view,
             im.resolved_depth.image, im.refract_depth.image,
             im.history[w].image, taa.history_layout[w])
        };
        let color = color_subresource_range();
        let dr = depth_subresource_range();
        let read = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
        let device = &self.device.handle;
        unsafe {
            device.cmd_end_rendering(cmd); // close opaque instance → `current` = opaque color

            // current → TRANSFER_SRC, scratch → TRANSFER_DST.
            let b_cur = img_barrier(
                cur_img, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ, color,
            );
            let b_scr = img_barrier(
                scr_img, scr_old, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::ALL_COMMANDS, vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE, color,
            );
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&[b_cur, b_scr]));

            let sub = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .mip_level(0)
                .base_array_layer(0)
                .layer_count(1);
            let offs = [
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D { x: extent.width as i32, y: extent.height as i32, z: 1 },
            ];
            let region = vk::ImageBlit::default()
                .src_subresource(sub).src_offsets(offs)
                .dst_subresource(sub).dst_offsets(offs);
            device.cmd_blit_image(
                cmd, cur_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                scr_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                std::slice::from_ref(&region), vk::Filter::NEAREST,
            );

            // scratch → SHADER_READ (refraction source), current → COLOR_ATTACHMENT.
            let b_scr2 = img_barrier(
                scr_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, read,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ, color,
            );
            let b_cur2 = img_barrier(
                cur_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ,
                vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE, color,
            );
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&[b_scr2, b_cur2]));

            // Copy opaque depth → refract_depth (sampled by the water pass for
            // depth-based darkening). resolved_depth then returns to the depth
            // attachment layout for the reopened instance's depth test.
            let bd1 = img_barrier(
                rdepth_img, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ, dr,
            );
            let bd2 = img_barrier(
                refr_depth_img, vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE, dr,
            );
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&[bd1, bd2]));

            let depth_sub = vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::DEPTH).mip_level(0).base_array_layer(0).layer_count(1);
            let dcopy = vk::ImageCopy::default()
                .src_subresource(depth_sub).dst_subresource(depth_sub)
                .extent(vk::Extent3D { width: extent.width, height: extent.height, depth: 1 });
            device.cmd_copy_image(
                cmd, rdepth_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                refr_depth_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, std::slice::from_ref(&dcopy),
            );

            let bd3 = img_barrier(
                refr_depth_img, vk::ImageLayout::TRANSFER_DST_OPTIMAL, read,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_WRITE,
                vk::PipelineStageFlags2::FRAGMENT_SHADER, vk::AccessFlags2::SHADER_READ, dr,
            );
            let bd4 = img_barrier(
                rdepth_img, vk::ImageLayout::TRANSFER_SRC_OPTIMAL, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                vk::PipelineStageFlags2::TRANSFER, vk::AccessFlags2::TRANSFER_READ,
                vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS, vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ, dr,
            );
            device.cmd_pipeline_barrier2(cmd, &vk::DependencyInfo::default().image_memory_barriers(&[bd3, bd4]));

            // Reopen a 1× instance on current (+ depth), LOAD both (blend water over opaque).
            let color_att = vk::RenderingAttachmentInfo::default()
                .image_view(cur_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);
            let depth_att = vk::RenderingAttachmentInfo::default()
                .image_view(depth_view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .load_op(vk::AttachmentLoadOp::LOAD)
                .store_op(vk::AttachmentStoreOp::STORE);
            let info = vk::RenderingInfo::default()
                .render_area(vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent })
                .layer_count(1)
                .color_attachments(std::slice::from_ref(&color_att))
                .depth_attachment(&depth_att);
            device.cmd_begin_rendering(cmd, &info);
        }
        // `taa_resolve`'s b_hist transitions the scratch from UNDEFINED, so the
        // tracked layout doesn't need updating here.
        Ok(true)
    }

    /// The image view holding the opaque scene color this frame (refraction source
    /// for the water pass). Valid only between `begin_water_pass` and `taa_resolve`.
    pub fn refraction_src_view(&self) -> vk::ImageView {
        let taa = self.taa.as_ref().unwrap();
        let im = taa.images.as_ref().unwrap();
        im.history[taa.write_index].view
    }

    /// The opaque-scene depth this frame (for the water pass's depth-based
    /// darkening). Valid only between `begin_water_pass` and `taa_resolve`.
    pub fn refraction_depth_view(&self) -> vk::ImageView {
        let taa = self.taa.as_ref().unwrap();
        let im = taa.images.as_ref().unwrap();
        im.refract_depth.view
    }

    // ── GPU-driven command recording (compute + indirect; generic) ──────────

    /// Raw pipeline layout for a handle (for explicit push constants / descriptor
    /// binding in GPU-driven passes).
    pub fn pipeline_layout(&self, handle: PipelineHandle) -> Result<vk::PipelineLayout, RhiError> {
        Ok(self.pipelines.get(handle)?.layout)
    }

    /// Bind a pipeline at an explicit bind point WITHOUT touching the per-frame
    /// UBO ring (unlike [`Self::bind_pipeline`]).
    pub fn cmd_bind_pipeline(
        &self,
        fi: u32,
        bind_point: vk::PipelineBindPoint,
        handle: PipelineHandle,
    ) -> Result<(), RhiError> {
        let pipeline = self.pipelines.get(handle)?.pipeline;
        let cmd = self.commands.frames[fi as usize].buffer;
        unsafe { self.device.handle.cmd_bind_pipeline(cmd, bind_point, pipeline) };
        Ok(())
    }

    /// Bind one descriptor set at `set_index` for `bind_point` / `layout`.
    pub fn cmd_bind_descriptor_set(
        &self,
        fi: u32,
        bind_point: vk::PipelineBindPoint,
        layout: vk::PipelineLayout,
        set_index: u32,
        set: vk::DescriptorSet,
    ) {
        let cmd = self.commands.frames[fi as usize].buffer;
        unsafe {
            self.device
                .handle
                .cmd_bind_descriptor_sets(cmd, bind_point, layout, set_index, &[set], &[])
        };
    }

    /// Push constants at an explicit layout + stage mask.
    pub fn cmd_push_constants(
        &self,
        fi: u32,
        layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        bytes: &[u8],
    ) {
        let cmd = self.commands.frames[fi as usize].buffer;
        unsafe { self.device.handle.cmd_push_constants(cmd, layout, stages, 0, bytes) };
    }

    /// Record a compute dispatch.
    pub fn cmd_dispatch(&self, fi: u32, gx: u32, gy: u32, gz: u32) {
        let cmd = self.commands.frames[fi as usize].buffer;
        unsafe { self.device.handle.cmd_dispatch(cmd, gx, gy, gz) };
    }

    /// Record an indirect draw reading a `VkDrawIndirectCommand` from `handle`.
    /// The buffer must have `INDIRECT_BUFFER` usage.
    pub fn cmd_draw_indirect(
        &self,
        fi: u32,
        handle: BufferHandle,
        offset: u64,
        draw_count: u32,
        stride: u32,
    ) -> Result<(), RhiError> {
        let buf = self.resolve_buffer(handle)?;
        let cmd = self.commands.frames[fi as usize].buffer;
        unsafe {
            self.device
                .handle
                .cmd_draw_indirect(cmd, buf, offset, draw_count, stride)
        };
        Ok(())
    }

    /// Fill `size` bytes of a buffer with a repeated u32 (e.g. counter reset).
    pub fn cmd_fill_buffer(
        &self,
        fi: u32,
        handle: BufferHandle,
        offset: u64,
        size: u64,
        data: u32,
    ) -> Result<(), RhiError> {
        let buf = self.resolve_buffer(handle)?;
        let cmd = self.commands.frames[fi as usize].buffer;
        unsafe { self.device.handle.cmd_fill_buffer(cmd, buf, offset, size, data) };
        Ok(())
    }

    /// Buffer memory barrier (sync2): order a prior write against a later read —
    /// e.g. compute SSBO write → indirect-command read, or → vertex storage read.
    #[allow(clippy::too_many_arguments)]
    pub fn cmd_buffer_barrier(
        &self,
        fi: u32,
        handle: BufferHandle,
        src_stage: vk::PipelineStageFlags2,
        src_access: vk::AccessFlags2,
        dst_stage: vk::PipelineStageFlags2,
        dst_access: vk::AccessFlags2,
    ) -> Result<(), RhiError> {
        let buf = self.resolve_buffer(handle)?;
        let cmd = self.commands.frames[fi as usize].buffer;
        let barrier = vk::BufferMemoryBarrier2::default()
            .src_stage_mask(src_stage)
            .src_access_mask(src_access)
            .dst_stage_mask(dst_stage)
            .dst_access_mask(dst_access)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buf)
            .offset(0)
            .size(vk::WHOLE_SIZE);
        let dep = vk::DependencyInfo::default()
            .buffer_memory_barriers(std::slice::from_ref(&barrier));
        unsafe { self.device.handle.cmd_pipeline_barrier2(cmd, &dep) };
        Ok(())
    }

    // ── Frame recording — call between begin_frame and end_frame ────────────

    /// Write raw bytes into the per-frame UBO and bind the descriptor set (set 0)
    /// into the command buffer for the current frame.
    ///
    /// Call this once per frame after `begin_frame` **and** after `bind_pipeline`,
    /// before any draw calls.  The pipeline layout is required to bind set 0; if no
    /// pipeline is bound this returns `RhiError::Other` immediately so the missing
    /// bind cannot silently pass a frame with stale or unbound uniforms.
    ///
    /// # Errors
    /// Returns `RhiError::Other("no pipeline bound")` if called before
    /// `bind_pipeline`.
    ///
    /// `fi` is the ring index returned by `begin_frame`.
    /// `bytes` must match the `uniform_buffer_size` from [`RhiConfig`].
    pub fn update_frame_uniforms(&mut self, fi: u32, bytes: &[u8]) -> Result<(), RhiError> {
        let ring = fi as usize;
        self.descriptors.update_frame_uniforms(&mut self.buffers, ring, bytes)?;

        // Binding set 0 requires the current pipeline layout.  The R1 ordering
        // rule (bind_pipeline before update_frame_uniforms) is documented in
        // scene.rs; enforcing it here catches violations at the call site.
        let layout = self.current_pipeline_layout.ok_or_else(|| {
            RhiError::Other(
                "update_frame_uniforms called before bind_pipeline: set 0 not bound"
                    .to_string()
                    .into(),
            )
        })?;

        let cmd = self.commands.frames[ring].buffer;
        let set = self.descriptors.set(ring);
        unsafe {
            self.device.handle.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[set],
                &[],
            );
        }

        Ok(())
    }

    /// Set the viewport and scissor to cover the full swapchain extent.
    ///
    /// Must be called at least once per frame (dynamic state).
    pub fn set_viewport_scissor_full(&mut self, fi: u32) {
        let cmd = self.commands.frames[fi as usize].buffer;
        let extent = self.swapchain.extent;

        let viewport = vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: extent.width as f32,
            height: extent.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        };
        let scissor = vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
        };

        unsafe {
            self.device
                .handle
                .cmd_set_viewport(cmd, 0, std::slice::from_ref(&viewport));
            self.device
                .handle
                .cmd_set_scissor(cmd, 0, std::slice::from_ref(&scissor));
        }
    }

    /// Bind a graphics pipeline. Records `vkCmdBindPipeline` into the current frame's
    /// command buffer and remembers the pipeline layout for subsequent push-constant
    /// and descriptor-set commands.
    pub fn bind_pipeline(&mut self, fi: u32, handle: PipelineHandle) -> Result<(), RhiError> {
        let entry = self.pipelines.get(handle)?;
        let pipeline = entry.pipeline;
        let layout = entry.layout;
        self.current_pipeline_layout = Some(layout);

        let cmd = self.commands.frames[fi as usize].buffer;
        unsafe {
            self.device.handle.cmd_bind_pipeline(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
        }

        // Bind the per-frame descriptor set (set 0) now that we have a layout.
        let set = self.descriptors.set(fi as usize);
        unsafe {
            self.device.handle.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[set],
                &[],
            );
        }

        Ok(())
    }

    /// Bind up to 4 vertex buffers (bindings 0‥3) into the current command buffer.
    ///
    /// Provide exactly 4 handles — one per attribute stream:
    /// `[positions, normals, colors, plate_colors]`.
    ///
    /// Handles may come from either the static `BufferStore` (M1 path) or the
    /// `StreamingUploader` (M2 path).  Resolution falls back to the streamer
    /// when a handle is not found in the static store.
    pub fn bind_vertex_buffers(
        &mut self,
        fi: u32,
        handles: &[BufferHandle; 4],
    ) -> Result<(), RhiError> {
        let cmd = self.commands.frames[fi as usize].buffer;

        let mut vk_bufs = [vk::Buffer::null(); 4];
        let offsets = [0u64; 4];
        for (i, &h) in handles.iter().enumerate() {
            vk_bufs[i] = self.resolve_buffer(h)?;
        }

        unsafe {
            self.device
                .handle
                .cmd_bind_vertex_buffers(cmd, 0, &vk_bufs, &offsets);
        }

        Ok(())
    }

    /// Bind the index buffer (u32 indices).
    ///
    /// Handles may come from either the static `BufferStore` (M1) or the
    /// `StreamingUploader` (M2).
    pub fn bind_index_buffer(
        &mut self,
        fi: u32,
        handle: BufferHandle,
    ) -> Result<(), RhiError> {
        let cmd = self.commands.frames[fi as usize].buffer;
        let vk_buf = self.resolve_buffer(handle)?;
        unsafe {
            self.device.handle.cmd_bind_index_buffer(
                cmd,
                vk_buf,
                0,
                vk::IndexType::UINT32,
            );
        }
        Ok(())
    }

    /// Resolve a `BufferHandle` to a raw `VkBuffer`.
    ///
    /// Checks the static `BufferStore` first; falls back to the `StreamingUploader`'s
    /// live-buffer map for handles issued by `streaming_upload`.
    fn resolve_buffer(&self, handle: BufferHandle) -> Result<vk::Buffer, RhiError> {
        // Try the static store first (M1 handles).
        if let Ok(buf) = self.buffers.vk_buffer(handle) {
            return Ok(buf);
        }
        // Fall back to the streamer (M2 handles).
        if let Some(streamer) = &self.streamer {
            if let Some(buf) = streamer.vk_buffer(handle) {
                return Ok(buf);
            }
        }
        Err(RhiError::InvalidHandle)
    }

    /// Write push constants (stages VERTEX | FRAGMENT).
    ///
    /// `bytes.len()` must be ≤ the `push_constant_size` specified in
    /// [`GraphicsPipelineDesc`]. The layout used is the currently bound pipeline's
    /// layout (`bind_pipeline` must have been called first).
    pub fn push_constants(&mut self, fi: u32, bytes: &[u8]) -> Result<(), RhiError> {
        let layout = self
            .current_pipeline_layout
            .ok_or_else(|| RhiError::Other("push_constants called before bind_pipeline".into()))?;

        let cmd = self.commands.frames[fi as usize].buffer;
        unsafe {
            self.device.handle.cmd_push_constants(
                cmd,
                layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytes,
            );
        }
        Ok(())
    }

    /// Record a `vkCmdDrawIndexed` for `index_count` indices starting at index 0,
    /// vertex offset 0, instance 0.
    pub fn draw_indexed(&mut self, fi: u32, index_count: u32) {
        let cmd = self.commands.frames[fi as usize].buffer;
        unsafe {
            self.device
                .handle
                .cmd_draw_indexed(cmd, index_count, 1, 0, 0, 0);
        }
    }

    // ── Frame lifecycle ────────────────────────────────────────────────────

    /// Begin a frame: wait for the fence, acquire a swapchain image, reset + begin the
    /// command buffer, and record the image-layout-transition barriers.
    ///
    /// After this call, streaming uploads (`streaming_upload`) may be recorded into the
    /// command buffer — they land before the rendering instance begins, which is required
    /// by Vulkan (vkCmdCopyBuffer is illegal inside a dynamic-rendering instance).
    ///
    /// Call `begin_rendering(fi)` once all uploads are done to start the rendering
    /// instance, then issue draw calls, then call `end_frame(fi)`.
    ///
    /// Returns the ring frame index (0..frames_in_flight).
    /// Returns `Ok(u32::MAX)` when the surface is zero-extent (window minimized).
    pub fn begin_frame(&mut self) -> Result<u32, RhiError> {
        // Recreate swapchain if flagged from a previous present.
        if self.needs_recreate {
            self.recreate_swapchain()?;
            self.needs_recreate = false;
        }

        // Zero-extent surface (minimised window) — skip the frame entirely.
        if self.swapchain.extent.width == 0 || self.swapchain.extent.height == 0 {
            return Ok(u32::MAX);
        }

        let fi = self.current_frame;

        // Copy the per-frame sync handles out before any mutable borrows.
        let in_flight_fence  = self.sync.frames[fi].in_flight_fence;
        let image_available  = self.sync.frames[fi].image_available;

        // Wait for this frame slot's fence so we don't overwrite in-flight GPU work.
        unsafe {
            self.device.handle.wait_for_fences(
                std::slice::from_ref(&in_flight_fence),
                true,
                u64::MAX,
            )
        }
        .map_err(RhiError::Vulkan)?;

        // ── M2: per-frame streaming bookkeeping ────────────────────────────
        //
        // collect_garbage uses gpu_completed_frame() (timeline semaphore query)
        // to free graveyard entries whose retire_frame the GPU has passed.
        // Must run AFTER the fence wait so the timeline has advanced.
        //
        // begin_frame resets this slot's staging-ring cursor and per-frame budget.
        // Safe because the fence guarantees the GPU is no longer reading from
        // slot `fi`'s staging segment.
        self.collect_streaming_garbage();
        if let Some(streamer) = self.streamer.as_mut() {
            streamer.begin_frame(fi);
        }

        // Acquire the next swapchain image.
        let (image_index, suboptimal) = unsafe {
            self.swapchain.loader.acquire_next_image(
                self.swapchain.handle,
                u64::MAX,
                image_available,
                vk::Fence::null(),
            )
        }
        .or_else(|e| {
            if e == vk::Result::ERROR_OUT_OF_DATE_KHR {
                self.recreate_swapchain()?;
                // Return sentinel to skip this frame.
                Ok((u32::MAX, false))
            } else {
                Err(RhiError::Vulkan(e))
            }
        })?;

        if image_index == u32::MAX {
            return Ok(u32::MAX);
        }

        if suboptimal {
            self.needs_recreate = true;
        }

        // W2: thread image_index through the per-frame slot.
        self.image_indices[fi] = image_index;

        // Reset and begin the command buffer for this frame.
        let cmd = self.commands.frames[fi].buffer;
        let pool = self.commands.frames[fi].pool;
        unsafe {
            self.device
                .handle
                .reset_command_pool(pool, vk::CommandPoolResetFlags::empty())
        }
        .map_err(RhiError::Vulkan)?;

        unsafe {
            self.device.handle.begin_command_buffer(
                cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
        }
        .map_err(RhiError::Vulkan)?;

        // ── Image layout transitions ────────────────────────────────────────
        //
        // Three barriers in one vkCmdPipelineBarrier2 call:
        //   1. MSAA color image:  UNDEFINED → COLOR_ATTACHMENT_OPTIMAL
        //   2. Depth image:       UNDEFINED → DEPTH_ATTACHMENT_OPTIMAL
        //   3. Swapchain image:   UNDEFINED → COLOR_ATTACHMENT_OPTIMAL (resolve target)
        let msaa_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(self.msaa_image.as_ref().unwrap().image)
            .subresource_range(color_subresource_range());

        let depth_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS)
            .dst_access_mask(vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .image(self.depth_image.as_ref().unwrap().image)
            .subresource_range(depth_subresource_range());

        let resolve_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::NONE)
            .src_access_mask(vk::AccessFlags2::NONE)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .image(self.swapchain.images[image_index as usize])
            .subresource_range(color_subresource_range());

        if self.taa_active() {
            // TAA path: the 3D pass renders into the TAA MSAA target, resolving
            // colour → `current` and depth → `resolved_depth`. The swapchain is
            // NOT touched here — `taa_resolve` transitions it (TRANSFER_DST → blit
            // → COLOR_ATTACHMENT). Transition the four TAA attachments instead.
            let im = self.taa.as_ref().unwrap().images.as_ref().unwrap();
            let taa_barriers = [
                img_barrier(
                    im.hdr_msaa.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    color_subresource_range(),
                ),
                img_barrier(
                    im.current.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT, vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                    color_subresource_range(),
                ),
                img_barrier(
                    self.depth_image.as_ref().unwrap().image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE, depth_subresource_range(),
                ),
                img_barrier(
                    im.resolved_depth.image, vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
                    vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE,
                    vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
                    vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE, depth_subresource_range(),
                ),
            ];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&taa_barriers);
            unsafe { self.device.handle.cmd_pipeline_barrier2(cmd, &dep_info) };
        } else {
            let barriers = [msaa_barrier, resolve_barrier, depth_barrier];
            let dep_info = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            unsafe { self.device.handle.cmd_pipeline_barrier2(cmd, &dep_info) };
        }

        // Reset per-frame pipeline tracking (new frame, new bind state).
        self.current_pipeline_layout = None;

        // NOTE: do NOT call cmd_begin_rendering here.  The caller must record
        // any streaming uploads (vkCmdCopyBuffer + copy→draw barriers) into the
        // command buffer at this point, and then call begin_rendering(fi) to open
        // the dynamic-rendering instance before issuing draw commands.  Copies
        // recorded inside a dynamic-rendering instance are a Vulkan validation error.

        Ok(fi as u32)
    }

    /// Open the dynamic-rendering instance for frame `fi`.
    ///
    /// Must be called after `begin_frame` (and after any `streaming_upload` calls)
    /// and before any draw commands.  Closes with `end_frame`.
    ///
    /// Records `vkCmdBeginRendering` with:
    ///   - Color attachment: MSAA image, `LOAD_OP_CLEAR` (clears to the configured clear color).
    ///   - Resolve attachment: current swapchain image view.
    ///   - Depth attachment: depth image, `LOAD_OP_CLEAR` (clears to 0.0 for reversed-Z).
    ///
    /// Per-frame call order:
    /// ```text
    /// let fi = rhi.begin_frame()?;
    /// // ... rhi.streaming_upload(fi, ...) zero or more times ...
    /// rhi.begin_rendering(fi);
    /// // ... draw calls ...
    /// rhi.end_frame(fi)?;
    /// ```
    pub fn begin_rendering(&mut self, fi: u32) {
        let fi = fi as usize;
        let image_index = self.image_indices[fi] as usize;
        let cmd = self.commands.frames[fi].buffer;

        let clear_color_value = vk::ClearValue {
            color: vk::ClearColorValue { float32: self.clear_color },
        };
        let clear_depth_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth: 0.0, stencil: 0 },
        };

        // TAA path: render the 3D pass into the TAA MSAA target, resolving colour
        // → `current` and depth → `resolved_depth` (SAMPLE_ZERO), so the resolve
        // can sample them. Otherwise the original MSAA→swapchain path.
        let (color_view, color_resolve_view, depth_view, depth_resolve_view) = if self.taa_active() {
            let im = self.taa.as_ref().unwrap().images.as_ref().unwrap();
            (
                im.hdr_msaa.view,
                im.current.view,
                self.depth_image.as_ref().unwrap().view,
                Some(im.resolved_depth.view),
            )
        } else {
            (
                self.msaa_image.as_ref().unwrap().view,
                self.swapchain.image_views[image_index],
                self.depth_image.as_ref().unwrap().view,
                None,
            )
        };

        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(color_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_mode(vk::ResolveModeFlags::AVERAGE)
            .resolve_image_view(color_resolve_view)
            .resolve_image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(clear_color_value);

        let mut depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(depth_view)
            .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .clear_value(clear_depth_value);
        if let Some(rv) = depth_resolve_view {
            depth_attachment = depth_attachment
                .resolve_mode(vk::ResolveModeFlags::SAMPLE_ZERO)
                .resolve_image_view(rv)
                .resolve_image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL);
        }

        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&color_attachment))
            .depth_attachment(&depth_attachment);

        unsafe { self.device.handle.cmd_begin_rendering(cmd, &rendering_info) };
    }

    /// Close the MSAA 3D rendering instance and open a 1-sample UI pass on the
    /// resolved swapchain image.
    ///
    /// # What this does (in order)
    /// 1. `vkCmdEndRendering` — closes the 4× MSAA 3D instance opened by
    ///    `begin_rendering`. The MSAA→swapchain AVERAGE resolve has now written
    ///    the single-sample swapchain image; it remains in
    ///    `COLOR_ATTACHMENT_OPTIMAL`.
    /// 2. `vkCmdPipelineBarrier2` — orders the resolve write before the UI LOAD:
    ///    `srcStage = COLOR_ATTACHMENT_OUTPUT, srcAccess = COLOR_ATTACHMENT_WRITE`
    ///    → `dstStage = COLOR_ATTACHMENT_OUTPUT,
    ///       dstAccess = COLOR_ATTACHMENT_WRITE | COLOR_ATTACHMENT_READ`.
    ///    No layout change (both old and new layout =
    ///    `COLOR_ATTACHMENT_OPTIMAL`).
    /// 3. `vkCmdBeginRendering` — opens a new dynamic-rendering instance with
    ///    a **single** color attachment = `swapchain.image_views[image_index]`
    ///    (implicitly 1 sample, no `resolveMode`), `loadOp = LOAD` to preserve
    ///    the resolved 3D image, `storeOp = STORE`, and **no depth attachment**
    ///    (egui has depth test/write off).
    ///
    /// # Frame bracket invariant
    /// After this call the active rendering instance is the 1-sample UI pass.
    /// `end_frame` will close it with a single `vkCmdEndRendering`.  The full
    /// per-frame bracket is:
    /// ```text
    /// begin_rendering   → [3D draws]
    /// begin_ui_pass     → [egui draws]   (end_frame closes this one)
    /// end_frame
    /// ```
    /// Two `cmd_begin_rendering` calls, two `cmd_end_rendering` calls per frame.
    ///
    /// # Skip path
    /// When `fi == u32::MAX` (window minimised) this is a no-op, matching
    /// `begin_rendering` and `end_frame`.
    pub fn begin_ui_pass(&mut self, fi: u32) {
        if fi == u32::MAX {
            return;
        }

        let fi_idx = fi as usize;
        let image_index = self.image_indices[fi_idx] as usize;
        let cmd = self.commands.frames[fi_idx].buffer;

        if self.taa_active() {
            // TAA path: `taa_resolve` already closed the 3D instance and blitted the
            // resolved image to the swapchain, leaving it in COLOR_ATTACHMENT_OPTIMAL
            // with a TRANSFER→COLOR barrier. Nothing to close/barrier here.
        } else {
            // 1. Close the MSAA 3D instance.
            unsafe { self.device.handle.cmd_end_rendering(cmd) };

            // 2. Resolve→load barrier: order the AVERAGE resolve write before the
            //    UI loadOp=LOAD read+write.  No layout change needed — the swapchain
            //    image is already in COLOR_ATTACHMENT_OPTIMAL from the 3D pass.
            let inter_pass_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
                .dst_access_mask(
                    vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags2::COLOR_ATTACHMENT_READ,
                )
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image(self.swapchain.images[image_index])
                .subresource_range(color_subresource_range());

            let dep_info = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&inter_pass_barrier));
            unsafe { self.device.handle.cmd_pipeline_barrier2(cmd, &dep_info) };
        }

        // 3. Open the 1-sample UI rendering instance on the resolved swapchain image.
        //    - No MSAA image, no resolveMode — this is a plain 1-sample attachment.
        //    - loadOp = LOAD: preserves the resolved 3D colour.
        //    - storeOp = STORE: the UI's output must persist to present.
        //    - No depth attachment: egui has depth test and depth write disabled.
        let ui_color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(self.swapchain.image_views[image_index])
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::LOAD)
            .store_op(vk::AttachmentStoreOp::STORE);

        let ui_rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.swapchain.extent,
            })
            .layer_count(1)
            .color_attachments(std::slice::from_ref(&ui_color_attachment));

        unsafe { self.device.handle.cmd_begin_rendering(cmd, &ui_rendering_info) };
    }

    /// End a frame: end the **UI** command buffer instance, submit it, present the
    /// swapchain image.
    ///
    /// `frame_index` must match the value returned by the corresponding [`Self::begin_frame`].
    /// No-ops when `frame_index == u32::MAX` (zero-extent sentinel).
    ///
    /// `frame_index` is the ring-slot index returned by [`Self::begin_frame`] for this frame.
    /// It must not be confused with a monotonic frame counter.
    ///
    /// # Frame bracket
    /// `end_frame` closes the 1-sample UI rendering instance opened by
    /// `begin_ui_pass` (NOT the MSAA 3D instance — that was already closed by
    /// `begin_ui_pass`).  The invariant is:
    /// ```text
    /// begin_rendering(fi)  → [3D draws]
    /// begin_ui_pass(fi)    → [egui draws]
    /// end_frame(fi)        ← closes the UI instance here
    /// ```
    pub fn end_frame(&mut self, frame_index: u32) -> Result<(), RhiError> {
        if frame_index == u32::MAX {
            return Ok(());
        }

        let fi = frame_index as usize;
        let image_index = self.image_indices[fi] as usize;
        let cmd = self.commands.frames[fi].buffer;

        // ── End the UI rendering instance ───────────────────────────────────
        // begin_ui_pass already closed the 3D MSAA instance; this closes the
        // 1-sample UI instance it opened.
        unsafe { self.device.handle.cmd_end_rendering(cmd) };

        // ── Present barrier: COLOR_ATTACHMENT_OPTIMAL → PRESENT_SRC_KHR ───
        let present_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .dst_access_mask(vk::AccessFlags2::NONE)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .image(self.swapchain.images[image_index])
            .subresource_range(color_subresource_range());

        let dep_info =
            vk::DependencyInfo::default().image_memory_barriers(std::slice::from_ref(&present_barrier));
        unsafe { self.device.handle.cmd_pipeline_barrier2(cmd, &dep_info) };

        unsafe { self.device.handle.end_command_buffer(cmd) }
            .map_err(RhiError::Vulkan)?;

        // ── Submit (synchronization2 — vkQueueSubmit2) ─────────────────────
        self.frame_counter += 1;
        let frame_sync = &self.sync.frames[fi];

        let wait_info = [vk::SemaphoreSubmitInfo::default()
            .semaphore(frame_sync.image_available)
            .stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)];

        let signal_info = [
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.swapchain.render_finished[image_index])
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            vk::SemaphoreSubmitInfo::default()
                .semaphore(self.sync.timeline)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .value(self.frame_counter),
        ];

        let cmd_info = [vk::CommandBufferSubmitInfo::default().command_buffer(cmd)];

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_info)
            .command_buffer_infos(&cmd_info)
            .signal_semaphore_infos(&signal_info);

        // B2: Reset fence immediately before submit, not in begin_frame. This prevents
        // a stranded unsignalled fence if begin_frame exits early (OUT_OF_DATE / zero-extent).
        unsafe {
            self.device.handle.reset_fences(std::slice::from_ref(
                &self.sync.frames[fi].in_flight_fence,
            ))
        }
        .map_err(RhiError::Vulkan)?;

        unsafe {
            self.device.handle.queue_submit2(
                self.queue.handle,
                std::slice::from_ref(&submit_info),
                frame_sync.in_flight_fence,
            )
        }
        .map_err(RhiError::Vulkan)?;

        // ── Present ────────────────────────────────────────────────────────
        // Wait on render_finished (binary semaphore signalled by the submit above).
        // Using the per-image semaphore (B1) to avoid reuse while the presentation
        // engine is still waiting on a previous frame's semaphore.
        let render_finished_sem = self.swapchain.render_finished[image_index];
        let swapchain_handles = [self.swapchain.handle];
        let present_image_indices = [image_index as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(std::slice::from_ref(&render_finished_sem))
            .swapchains(&swapchain_handles)
            .image_indices(&present_image_indices);

        let present_result = unsafe {
            self.swapchain
                .loader
                .queue_present(self.queue.handle, &present_info)
        };

        match present_result {
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                self.needs_recreate = true;
            }
            Ok(false) => {}
            Err(e) => return Err(RhiError::Vulkan(e)),
        }

        self.current_frame = (fi + 1) % self.frames_in_flight;

        Ok(())
    }

    /// Block until the device is idle. Call before dropping the `Rhi`.
    pub fn wait_idle(&self) -> Result<(), RhiError> {
        unsafe { self.device.handle.device_wait_idle() }.map_err(RhiError::Vulkan)
    }

    // ── Swapchain accessors ────────────────────────────────────────────────

    /// Returns the swapchain colour format (e.g. `B8G8R8A8_SRGB`).
    ///
    /// Pass this as `color_format` when creating graphics pipelines so the
    /// pipeline matches the render target format exactly.
    pub fn swapchain_format(&self) -> vk::Format {
        self.swapchain.format
    }

    /// Returns the MSAA sample count the RHI was configured with.
    ///
    /// Pass this as `samples` when creating graphics pipelines so the
    /// pipeline matches the MSAA image.
    pub fn msaa_samples(&self) -> vk::SampleCountFlags {
        self.msaa_sample_count
    }

    /// Returns the current swapchain extent (width × height in pixels).
    ///
    /// Use this to compute the aspect ratio for `FrameUniforms::new`.
    pub fn extent(&self) -> vk::Extent2D {
        self.swapchain.extent
    }

    /// Number of frames-in-flight (ring depth). Use to size per-frame resources.
    pub fn frames_in_flight(&self) -> usize {
        self.frames_in_flight
    }

    // ── Raw Vulkan handle accessors (for egui integration) ─────────────────
    //
    // These expose the minimal raw handles needed for `egui-ash-renderer`.
    // They are intentionally narrow — callers must not use them to bypass
    // the RHI's resource-tracking invariants.

    /// Clone the logical `ash::Device` handle.
    ///
    /// `ash::Device` is a lightweight struct wrapping function pointers; cloning
    /// it does **not** create a new Vulkan device — it shares the same
    /// `vk::Device` handle.  The clone is valid as long as the `Rhi` is alive.
    pub fn device_handle(&self) -> ash::Device {
        self.device.handle.clone()
    }

    /// Reference to the `ash::Instance` handle.
    pub fn instance_handle(&self) -> &ash::Instance {
        &self.instance.handle
    }

    /// The selected Vulkan physical device handle.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.device.physical
    }

    /// The graphics/present queue handle.
    pub fn queue_handle(&self) -> vk::Queue {
        self.queue.handle
    }

    /// The graphics queue family index.
    pub fn graphics_queue_family(&self) -> u32 {
        self.device.graphics_queue_family
    }

    /// The command pool for frame slot `fi`.
    ///
    /// Valid between `begin_frame(fi)` and `end_frame(fi)`.
    /// The caller must not reset or destroy this pool.
    pub fn command_pool(&self, fi: u32) -> vk::CommandPool {
        self.commands.frames[fi as usize].pool
    }

    /// The primary command buffer for frame slot `fi`.
    ///
    /// Valid between `begin_frame(fi)` and `end_frame(fi)`.
    /// Use this to record additional commands (e.g. egui draw calls) into
    /// the frame's command buffer after `begin_rendering`.
    pub fn current_command_buffer(&self, fi: u32) -> vk::CommandBuffer {
        self.commands.frames[fi as usize].buffer
    }

    /// The raw `VkBuffer` backing a `BufferHandle` from the static buffer store.
    ///
    /// For self-contained sub-renderers (e.g. the flora sub-renderer) that build
    /// their OWN descriptor sets and must point a binding at an RHI-created
    /// buffer. Narrow + read-only: it does not bypass any resource-tracking
    /// invariant — the buffer is still owned and reclaimed by the RHI.
    pub fn vk_buffer(&self, handle: BufferHandle) -> Result<vk::Buffer, RhiError> {
        self.buffers.vk_buffer(handle)
    }

    // ── Private helpers ────────────────────────────────────────────────────

    fn recreate_swapchain(&mut self) -> Result<(), RhiError> {
        unsafe { self.device.handle.device_wait_idle() }.map_err(RhiError::Vulkan)?;

        // Split borrow: get raw ptrs to device.handle and device.allocator so the
        // borrow checker doesn't see two overlapping borrows of `self.device`.
        let dev_handle: *const ash::Device = &self.device.handle;
        let dev_alloc: *mut gpu_allocator::vulkan::Allocator = &mut *self.device.allocator;

        // Destroy old per-frame images (depth and MSAA).
        if let Some(img) = self.depth_image.take() {
            img.destroy(unsafe { &*dev_handle }, unsafe { &mut *dev_alloc });
        }
        if let Some(img) = self.msaa_image.take() {
            img.destroy(unsafe { &*dev_handle }, unsafe { &mut *dev_alloc });
        }
        // Destroy old TAA offscreen images (recreated at the new extent below).
        if let Some(taa) = self.taa.as_mut() {
            if let Some(images) = taa.images.take() {
                images.destroy(unsafe { &*dev_handle }, unsafe { &mut *dev_alloc });
            }
        }

        // B3: Create new swapchain FIRST using still-live old handle as oldSwapchain,
        // then destroy old resources. This satisfies VUID-VkSwapchainCreateInfoKHR-oldSwapchain.
        let old_handle = self.swapchain.handle;
        // Drain old image views and render_finished sems so swapchain.destroy() won't touch them.
        let old_image_views: Vec<_> = std::mem::take(&mut self.swapchain.image_views);
        let old_render_finished: Vec<_> = std::mem::take(&mut self.swapchain.render_finished);
        self.swapchain.images.clear();
        self.swapchain.handle = ash::vk::SwapchainKHR::null(); // prevent double destroy in Drop

        self.swapchain = swapchain::Swapchain::new(
            &self.instance.handle,
            unsafe { &*dev_handle },
            self.device.physical,
            &self.surface,
            self.device.graphics_queue_family,
            Some(old_handle),
        )?;

        // Now destroy old resources (after new swapchain is live).
        unsafe {
            for &sem in &old_render_finished {
                self.device.handle.destroy_semaphore(sem, None);
            }
            for &view in &old_image_views {
                self.device.handle.destroy_image_view(view, None);
            }
            self.swapchain.loader.destroy_swapchain(old_handle, None);
        }

        if self.swapchain.extent.width > 0 && self.swapchain.extent.height > 0 {
            self.depth_image = Some(image::AllocatedImage::new_depth(
                unsafe { &*dev_handle },
                unsafe { &mut *dev_alloc },
                self.swapchain.extent,
                self.msaa_sample_count,
            )?);
            self.msaa_image = Some(image::AllocatedImage::new_msaa_color(
                unsafe { &*dev_handle },
                unsafe { &mut *dev_alloc },
                self.swapchain.extent,
                self.swapchain.format,
                self.msaa_sample_count,
            )?);
            // Recreate TAA offscreen images at the new extent (history resets).
            if self.taa.is_some() {
                let images = taa_pass::TaaImages::new(
                    unsafe { &*dev_handle },
                    unsafe { &mut *dev_alloc },
                    self.swapchain.extent,
                    self.swapchain.format,
                    self.msaa_sample_count,
                )?;
                let taa = self.taa.as_mut().unwrap();
                taa.images = Some(images);
                taa.history_layout = [vk::ImageLayout::UNDEFINED; 2];
                taa.write_index = 0;
            }
        }

        log::info!(
            "Swapchain recreated: {}x{}",
            self.swapchain.extent.width,
            self.swapchain.extent.height
        );

        Ok(())
    }
}

impl Drop for Rhi {
    fn drop(&mut self) {
        // Best-effort idle wait. If this fails, continue teardown anyway.
        let _ = unsafe { self.device.handle.device_wait_idle() };

        unsafe {
            // 1. Destroy pipelines first — holds VkPipeline + VkPipelineLayout only;
            //    no memory allocations. ManuallyDrop::drop calls PipelineStore::drop
            //    which drains the map.
            ManuallyDrop::drop(&mut self.pipelines);

            // 2. Drop sync and command objects.
            ManuallyDrop::drop(&mut self.commands);
            ManuallyDrop::drop(&mut self.sync);

            // 3. Destroy descriptors (pool, layout) while device is alive.
            //    UBO buffer handles are in `buffers`; we free them in step 4.
            ManuallyDrop::drop(&mut self.descriptors);
            // 3b. Destroy general storage/uniform descriptor pool + layouts.
            ManuallyDrop::drop(&mut self.general_descriptors);

            // 4. Destroy streaming uploader (staging ring + all live DEVICE_LOCAL
            //    buffers), then all static buffers (UBOs etc).
            let dev_alloc: *mut gpu_allocator::vulkan::Allocator =
                &mut *self.device.allocator;
            if let Some(streamer) = self.streamer.take() {
                streamer.destroy(&mut *dev_alloc);
            }
            self.buffers.destroy_all(&mut *dev_alloc);

            // 4b. Destroy TAA images + sampler (its layout/pipeline/sets/UBOs were
            //     freed by the pipeline/descriptor/buffer stores above).
            let dev_handle: *const ash::Device = &self.device.handle;
            let dev_alloc: *mut gpu_allocator::vulkan::Allocator =
                &mut *self.device.allocator;
            if let Some(taa) = self.taa.take() {
                taa.destroy(&*dev_handle, &mut *dev_alloc);
            }

            // 5. Destroy depth and MSAA images.
            let dev_handle: *const ash::Device = &self.device.handle;
            let dev_alloc: *mut gpu_allocator::vulkan::Allocator =
                &mut *self.device.allocator;
            if let Some(img) = self.depth_image.take() {
                img.destroy(&*dev_handle, &mut *dev_alloc);
            }
            if let Some(img) = self.msaa_image.take() {
                img.destroy(&*dev_handle, &mut *dev_alloc);
            }

            // 6. Destroy swapchain image views and the swapchain handle.
            self.swapchain.destroy(&*dev_handle);

            // 7. Drop device: its Drop impl drops the allocator first, then calls
            //    destroy_device. This is the correct order.
            ManuallyDrop::drop(&mut self.device);

            // 8. Destroy surface.
            ManuallyDrop::drop(&mut self.surface);

            // 9. Destroy debug messenger and instance.
            ManuallyDrop::drop(&mut self.instance);
        }
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────

fn config_to_sample_count(msaa: u32) -> vk::SampleCountFlags {
    match msaa {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        _ => vk::SampleCountFlags::TYPE_4,
    }
}

fn color_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

/// Build a single `VkImageMemoryBarrier2` (sync2). Used by the TAA resolve pass.
#[allow(clippy::too_many_arguments)]
fn img_barrier(
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
    range: vk::ImageSubresourceRange,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(range)
}

fn depth_subresource_range() -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::DEPTH,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    }
}

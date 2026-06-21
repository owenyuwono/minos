//! TAA resolve resources — the GPU frame-bracket weaving's offscreen targets.
//!
//! Holds the MSAA color target the 3D pass renders into, the 1× resolved current
//! color + depth the resolve samples, the ping-pong history, and the resolve
//! pipeline / per-frame descriptor sets + UBOs.
//!
//! All color targets use the **swapchain color format** (not a wider HDR format)
//! on purpose: the existing 3D pipelines were built for the swapchain format +
//! MSAA sample count, so rendering them into these targets needs no pipeline
//! changes, and (for an sRGB swapchain) Vulkan's sRGB sample-decode / store-encode
//! gives correct *linear-space* temporal blending for free. The trade-off is 8-bit
//! history precision instead of HDR — fine for a first cut; can move to a wider
//! format later by also creating HDR-format 3D pipelines.

use ash::vk;
use gpu_allocator::vulkan::Allocator;

use crate::image::AllocatedImage;
use crate::{BufferHandle, PipelineHandle, RhiError};

/// The offscreen images TAA needs. Recreated on swapchain resize.
pub(crate) struct TaaImages {
    /// MSAA color target the 3D pass renders into (resolves to `current`).
    pub hdr_msaa: AllocatedImage,
    /// 1× resolved current-frame color (sampled by the resolve).
    pub current: AllocatedImage,
    /// 1× resolved depth (sampled by the resolve for reprojection).
    pub resolved_depth: AllocatedImage,
    /// 1× copy of the opaque depth, sampled by the water pass for depth-based
    /// darkening (the deeper the water column, the darker). Filled by a copy in
    /// `begin_water_pass` so `resolved_depth` stays the live depth attachment.
    pub refract_depth: AllocatedImage,
    /// Ping-pong history: the resolve writes one and reads the other.
    pub history: [AllocatedImage; 2],
}

impl TaaImages {
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
        color_format: vk::Format,
        samples: vk::SampleCountFlags,
    ) -> Result<Self, RhiError> {
        let one = vk::SampleCountFlags::TYPE_1;
        let color = vk::ImageAspectFlags::COLOR;
        let depth = vk::ImageAspectFlags::DEPTH;
        let sampled_color = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
        // current is also blitted from for the refraction capture (TRANSFER_SRC);
        // history is the refraction scratch (blit dest, TRANSFER_DST) + resolve blit src.
        let history_usage =
            sampled_color | vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;

        let hdr_msaa = AllocatedImage::new_render_target(
            device, allocator, extent, color_format, samples,
            vk::ImageUsageFlags::COLOR_ATTACHMENT, color, "taa_msaa",
        )?;
        let current = AllocatedImage::new_render_target(
            device, allocator, extent, color_format, one,
            sampled_color | vk::ImageUsageFlags::TRANSFER_SRC, color, "taa_current",
        )?;
        let resolved_depth = AllocatedImage::new_render_target(
            device, allocator, extent, vk::Format::D32_SFLOAT, one,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
            depth, "taa_depth",
        )?;
        let refract_depth = AllocatedImage::new_render_target(
            device, allocator, extent, vk::Format::D32_SFLOAT, one,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            depth, "taa_refract_depth",
        )?;
        let h0 = AllocatedImage::new_render_target(
            device, allocator, extent, color_format, one, history_usage, color, "taa_history0",
        )?;
        let h1 = AllocatedImage::new_render_target(
            device, allocator, extent, color_format, one, history_usage, color, "taa_history1",
        )?;

        Ok(Self { hdr_msaa, current, resolved_depth, refract_depth, history: [h0, h1] })
    }

    pub fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.hdr_msaa.destroy(device, allocator);
        self.current.destroy(device, allocator);
        self.resolved_depth.destroy(device, allocator);
        self.refract_depth.destroy(device, allocator);
        let [h0, h1] = self.history;
        h0.destroy(device, allocator);
        h1.destroy(device, allocator);
    }
}

/// All TAA GPU state, owned by [`crate::Rhi`].
pub(crate) struct TaaResources {
    /// `None` only while the surface is zero-extent (minimised); recreated on resize.
    pub images: Option<TaaImages>,
    /// Tracked layout of each history image (for correct `oldLayout`).
    pub history_layout: [vk::ImageLayout; 2],
    pub sampler: vk::Sampler,
    /// Resolve set layout (bindings 0–2 sampled image, 3 sampler, 4 uniform).
    /// Owned by `GeneralDescriptors` — NOT destroyed here. Retained for clarity.
    #[allow(dead_code)]
    pub set_layout: vk::DescriptorSetLayout,
    /// Resolve pipeline — owned by the pipeline store, NOT destroyed here.
    pub pipeline: PipelineHandle,
    /// One descriptor set + UBO per frame-in-flight.
    pub sets: Vec<vk::DescriptorSet>,
    pub ubos: Vec<BufferHandle>,
    /// Which history image the resolve writes this frame (the other is read).
    pub write_index: usize,
}

impl TaaResources {
    /// Free the GPU-owned images + sampler. The set layout, pipeline, descriptor
    /// sets, and UBO buffers are owned by the RHI's stores and freed there.
    pub fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(images) = self.images {
            images.destroy(device, allocator);
        }
        unsafe { device.destroy_sampler(self.sampler, None) };
    }
}

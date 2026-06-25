//! Scene-composite targets — the offscreen render targets for the 3D pass.
//!
//! Holds the MSAA color target the 3D pass renders into, the 1× resolved current
//! color + depth, a 1× copy of the opaque depth (sampled by the water pass for
//! depth-based darkening), and a 1× refraction scratch (the opaque scene colour the
//! refractive water samples). The composited `current` is blitted straight to the
//! swapchain by `present_composite` (no temporal resolve — there is no TAA).
//!
//! All color targets use the **swapchain color format** (not a wider HDR format)
//! on purpose: the existing 3D pipelines were built for the swapchain format +
//! MSAA sample count, so rendering them into these targets needs no pipeline
//! changes, and (for an sRGB swapchain) Vulkan's sRGB sample-decode / store-encode
//! keeps the composite in correct *linear* space for free. The trade-off is 8-bit
//! intermediate precision; can move to a wider format later by also creating
//! HDR-format 3D pipelines.

use ash::vk;
use gpu_allocator::vulkan::Allocator;

use crate::image::AllocatedImage;
use crate::{PipelineHandle, RhiError};

/// The offscreen images the 3D scene pass needs. Recreated on swapchain resize.
pub(crate) struct SceneImages {
    /// MSAA color target the 3D pass renders into (resolves to `current`).
    pub hdr_msaa: AllocatedImage,
    /// 1× resolved current-frame color (composited, then blitted to the swapchain).
    pub current: AllocatedImage,
    /// 1× resolved depth (depth-resolve target of the 3D pass; copied to
    /// `refract_depth` for the water pass).
    pub resolved_depth: AllocatedImage,
    /// 1× copy of the opaque depth, sampled by the water pass for depth-based
    /// darkening (the deeper the water column, the darker). Filled by a copy in
    /// `begin_water_pass` so `resolved_depth` stays the live depth attachment.
    pub refract_depth: AllocatedImage,
    /// 1× copy of the opaque scene color — the refraction source the water pass
    /// samples. Filled by a blit from `current` in `begin_water_pass`.
    pub refraction_scratch: AllocatedImage,
}

impl SceneImages {
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

        let hdr_msaa = AllocatedImage::new_render_target(
            device, allocator, extent, color_format, samples,
            vk::ImageUsageFlags::COLOR_ATTACHMENT, color, "scene_msaa",
        )?;
        // current is the composite target + the blit source to the swapchain + the
        // refraction-capture blit source (TRANSFER_SRC).
        let current = AllocatedImage::new_render_target(
            device, allocator, extent, color_format, one,
            sampled_color | vk::ImageUsageFlags::TRANSFER_SRC, color, "scene_current",
        )?;
        let resolved_depth = AllocatedImage::new_render_target(
            device, allocator, extent, vk::Format::D32_SFLOAT, one,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_SRC,
            depth, "scene_depth",
        )?;
        let refract_depth = AllocatedImage::new_render_target(
            device, allocator, extent, vk::Format::D32_SFLOAT, one,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            depth, "scene_refract_depth",
        )?;
        // Refraction source: blit destination (TRANSFER_DST) + sampled by the water
        // shader (SAMPLED). Never a render target, so no COLOR_ATTACHMENT.
        let refraction_scratch = AllocatedImage::new_render_target(
            device, allocator, extent, color_format, one,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            color, "scene_refraction_scratch",
        )?;

        Ok(Self { hdr_msaa, current, resolved_depth, refract_depth, refraction_scratch })
    }

    pub fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        self.hdr_msaa.destroy(device, allocator);
        self.current.destroy(device, allocator);
        self.resolved_depth.destroy(device, allocator);
        self.refract_depth.destroy(device, allocator);
        self.refraction_scratch.destroy(device, allocator);
    }
}

/// The scene-composite GPU targets, owned by [`crate::Rhi`].
pub(crate) struct SceneTargets {
    /// `None` only while the surface is zero-extent (minimised); recreated on resize.
    pub images: Option<SceneImages>,
    /// FXAA post-process: a fullscreen pipeline sampling `current` → swapchain.
    /// Built once in `init_scene_targets`. `images` swap on resize but these persist —
    /// only the binding-0 image view re-points (in `recreate_swapchain`). The pipeline
    /// is owned by the pipeline store; the sampler is freed in `destroy`.
    pub fxaa_pipeline: PipelineHandle,
    pub fxaa_sampler: vk::Sampler,
    pub fxaa_set: vk::DescriptorSet,
}

impl SceneTargets {
    /// Free the GPU-owned images + the FXAA sampler. (The FXAA pipeline + descriptor
    /// set + layout are owned by the RHI's pipeline/descriptor stores, freed there.)
    pub fn destroy(self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(images) = self.images {
            images.destroy(device, allocator);
        }
        unsafe { device.destroy_sampler(self.fxaa_sampler, None) };
    }
}

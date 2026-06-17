//! `image` — GPU image allocation (depth, MSAA color).

use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
    MemoryLocation,
};

use crate::RhiError;

pub(crate) struct AllocatedImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub allocation: Option<Allocation>,
    #[expect(dead_code)]
    pub format: vk::Format,
    #[expect(dead_code)]
    pub extent: vk::Extent2D,
}

impl AllocatedImage {
    pub fn new_depth(
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
        samples: vk::SampleCountFlags,
    ) -> Result<Self, RhiError> {
        let format = vk::Format::D32_SFLOAT;
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image =
            unsafe { device.create_image(&image_info, None) }.map_err(RhiError::Vulkan)?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "depth",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| RhiError::AllocationFailed(e.to_string()))?;

        unsafe {
            device.bind_image_memory(image, allocation.memory(), allocation.offset())
        }
        .map_err(RhiError::Vulkan)?;

        let view =
            Self::create_view(device, image, format, vk::ImageAspectFlags::DEPTH)?;

        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
            format,
            extent,
        })
    }

    pub fn new_msaa_color(
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
        format: vk::Format,
        samples: vk::SampleCountFlags,
    ) -> Result<Self, RhiError> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(
                vk::ImageUsageFlags::TRANSIENT_ATTACHMENT
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT,
            )
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image =
            unsafe { device.create_image(&image_info, None) }.map_err(RhiError::Vulkan)?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "msaa_color",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| RhiError::AllocationFailed(e.to_string()))?;

        unsafe {
            device.bind_image_memory(image, allocation.memory(), allocation.offset())
        }
        .map_err(RhiError::Vulkan)?;

        let view =
            Self::create_view(device, image, format, vk::ImageAspectFlags::COLOR)?;

        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
            format,
            extent,
        })
    }

    /// Generic render target / sampled image (TAA HDR color, history, resolved
    /// depth, …). Caller picks format, sample count, usage, and aspect.
    ///
    /// For a sampled color target use `usage = COLOR_ATTACHMENT | SAMPLED`,
    /// `aspect = COLOR`. For a sampled resolved-depth target use
    /// `usage = DEPTH_STENCIL_ATTACHMENT | SAMPLED`, `aspect = DEPTH`.
    #[allow(dead_code)] // wired by the TAA pass (see frame-bracket integration).
    #[allow(clippy::too_many_arguments)]
    pub fn new_render_target(
        device: &ash::Device,
        allocator: &mut Allocator,
        extent: vk::Extent2D,
        format: vk::Format,
        samples: vk::SampleCountFlags,
        usage: vk::ImageUsageFlags,
        aspect: vk::ImageAspectFlags,
        name: &str,
    ) -> Result<Self, RhiError> {
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(format)
            .extent(vk::Extent3D {
                width: extent.width,
                height: extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .samples(samples)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(usage)
            .initial_layout(vk::ImageLayout::UNDEFINED);

        let image =
            unsafe { device.create_image(&image_info, None) }.map_err(RhiError::Vulkan)?;
        let requirements = unsafe { device.get_image_memory_requirements(image) };

        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name,
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| RhiError::AllocationFailed(e.to_string()))?;

        unsafe { device.bind_image_memory(image, allocation.memory(), allocation.offset()) }
            .map_err(RhiError::Vulkan)?;

        let view = Self::create_view(device, image, format, aspect)?;

        Ok(Self {
            image,
            view,
            allocation: Some(allocation),
            format,
            extent,
        })
    }

    fn create_view(
        device: &ash::Device,
        image: vk::Image,
        format: vk::Format,
        aspect: vk::ImageAspectFlags,
    ) -> Result<vk::ImageView, RhiError> {
        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(format)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(aspect)
                    .base_mip_level(0)
                    .level_count(1)
                    .base_array_layer(0)
                    .layer_count(1),
            );
        unsafe { device.create_image_view(&view_info, None) }.map_err(RhiError::Vulkan)
    }

    /// Destroy the image view, image, and free the allocation.
    /// Must be called before the device and allocator are torn down.
    pub fn destroy(mut self, device: &ash::Device, allocator: &mut Allocator) {
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(alloc) = self.allocation.take() {
            let _ = allocator.free(alloc);
        }
    }
}

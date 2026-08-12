//! `swapchain` — VkSwapchainKHR and its image views.

use ash::{vk, Instance};

use crate::{surface::Surface, RhiError};

pub(crate) struct Swapchain {
    pub loader: ash::khr::swapchain::Device,
    pub handle: vk::SwapchainKHR,
    pub images: Vec<vk::Image>,
    pub image_views: Vec<vk::ImageView>,
    /// One binary semaphore per swapchain image. Signalled by queue_submit2,
    /// waited on by queue_present. Kept per-image (not per-ring-frame) to
    /// prevent reuse before the presentation engine is done with the semaphore.
    pub render_finished: Vec<vk::Semaphore>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(
        instance: &Instance,
        device: &ash::Device,
        physical: vk::PhysicalDevice,
        surface: &Surface,
        queue_family: u32,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Result<Self, RhiError> {
        let capabilities = unsafe {
            surface
                .loader
                .get_physical_device_surface_capabilities(physical, surface.handle)
        }
        .map_err(RhiError::Vulkan)?;

        let formats = unsafe {
            surface
                .loader
                .get_physical_device_surface_formats(physical, surface.handle)
        }
        .map_err(RhiError::Vulkan)?;

        // Prefer B8G8R8A8_SRGB / SRGB_NONLINEAR; fall back to first available.
        let format = formats
            .iter()
            .find(|f| {
                f.format == vk::Format::B8G8R8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .copied()
            .unwrap_or(formats[0]);

        let extent = choose_extent(&capabilities);

        let image_count = {
            let min = capabilities.min_image_count + 1;
            if capabilities.max_image_count > 0 {
                min.min(capabilities.max_image_count)
            } else {
                min
            }
        };

        let loader = ash::khr::swapchain::Device::new(instance, device);

        // COLOR_ATTACHMENT always; add TRANSFER_DST (when supported) so the TAA
        // resolve can blit its result onto the swapchain image. Universally
        // supported on desktop; the OFF path never uses it.
        let mut image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if capabilities
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            image_usage |= vk::ImageUsageFlags::TRANSFER_DST;
        }

        let create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(surface.handle)
            .min_image_count(image_count)
            .image_format(format.format)
            .image_color_space(format.color_space)
            .image_extent(extent)
            .image_array_layers(1)
            .image_usage(image_usage)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(std::slice::from_ref(&queue_family))
            .pre_transform(capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true)
            .old_swapchain(old_swapchain.unwrap_or(vk::SwapchainKHR::null()));

        let handle = unsafe { loader.create_swapchain(&create_info, None) }
            .map_err(RhiError::Vulkan)?;

        let images =
            unsafe { loader.get_swapchain_images(handle) }.map_err(RhiError::Vulkan)?;

        let image_views = images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(format.format)
                    .components(vk::ComponentMapping::default())
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(1)
                            .base_array_layer(0)
                            .layer_count(1),
                    );
                unsafe { device.create_image_view(&view_info, None) }
                    .map_err(RhiError::Vulkan)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let render_finished = images
            .iter()
            .map(|_| unsafe {
                device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(RhiError::Vulkan)?;

        log::info!(
            "Swapchain created: {}x{} {:?} ({} images)",
            extent.width,
            extent.height,
            format.format,
            images.len()
        );

        Ok(Self {
            loader,
            handle,
            images,
            image_views,
            render_finished,
            format: format.format,
            extent,
        })
    }

    /// Destroy image views, render_finished semaphores, and the swapchain handle.
    /// Does NOT destroy the VkImages — those are owned by the swapchain driver.
    pub fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for &sem in &self.render_finished {
                device.destroy_semaphore(sem, None);
            }
            for &view in &self.image_views {
                device.destroy_image_view(view, None);
            }
            self.loader.destroy_swapchain(self.handle, None);
        }
        self.render_finished.clear();
        self.image_views.clear();
        self.images.clear();
        self.handle = vk::SwapchainKHR::null();
    }
}

fn choose_extent(caps: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if caps.current_extent.width != u32::MAX {
        caps.current_extent
    } else {
        vk::Extent2D {
            width: 1280
                .max(caps.min_image_extent.width)
                .min(caps.max_image_extent.width),
            height: 720
                .max(caps.min_image_extent.height)
                .min(caps.max_image_extent.height),
        }
    }
}

// No Drop impl — Rhi::drop calls swapchain.destroy() before dropping the device.

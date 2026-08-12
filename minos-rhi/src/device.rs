//! `device` — physical device selection, logical device, gpu-allocator.

use ash::{vk, Instance};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use std::ffi::CStr;
use std::mem::ManuallyDrop;

use crate::{surface::Surface, RhiError};

pub(crate) struct Device {
    pub physical: vk::PhysicalDevice,
    pub graphics_queue_family: u32,
    /// Nanoseconds per GPU timestamp tick; `0.0` if the graphics queue can't
    /// timestamp (→ GPU timing disabled). Drives `gpu_timer::GpuTimer`.
    pub timestamp_period: f32,
    // allocator must be in ManuallyDrop so we can drop it before destroying the device
    pub allocator: ManuallyDrop<Allocator>,
    pub handle: ash::Device,
}

impl Device {
    pub fn new(
        instance: &Instance,
        surface: &Surface,
        _config: &crate::RhiConfig,
    ) -> Result<Self, RhiError> {
        let physical = select_physical_device(instance, surface)?;
        let queue_family = find_graphics_present_family(instance, surface, physical)?;

        let queue_priority = 1.0f32;
        let queue_create_info = vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family)
            .queue_priorities(std::slice::from_ref(&queue_priority));

        let swapchain_ext = c"VK_KHR_swapchain";
        let extension_names = [swapchain_ext.as_ptr()];

        // Query base device features so we can opt-in to fillModeNonSolid only when
        // the physical device actually supports it (lavapipe and all desktop GPUs do).
        let supported_features =
            unsafe { instance.get_physical_device_features(physical) };
        let base_features = if supported_features.fill_mode_non_solid == vk::TRUE {
            vk::PhysicalDeviceFeatures::default().fill_mode_non_solid(true)
        } else {
            log::warn!(
                "fillModeNonSolid not supported — wireframe view mode will be unavailable"
            );
            vk::PhysicalDeviceFeatures::default()
        };

        let mut vk13_features = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(true)
            .synchronization2(true);
        let mut vk12_features =
            vk::PhysicalDeviceVulkan12Features::default().timeline_semaphore(true);
        let mut features2 = vk::PhysicalDeviceFeatures2::default()
            .features(base_features)
            .push_next(&mut vk13_features)
            .push_next(&mut vk12_features);

        let create_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(std::slice::from_ref(&queue_create_info))
            .enabled_extension_names(&extension_names)
            .push_next(&mut features2);

        let handle = unsafe { instance.create_device(physical, &create_info, None) }
            .map_err(RhiError::Vulkan)?;

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: handle.clone(),
            physical_device: physical,
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        })
        .map_err(|e| RhiError::AllocationFailed(e.to_string()))?;

        let props = unsafe { instance.get_physical_device_properties(physical) };
        let name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_str()
            .unwrap_or("<unknown>");
        log::info!("Selected GPU: {}", name);

        // GPU timestamps need a non-zero `timestamp_period` AND a queue that reports
        // valid timestamp bits; otherwise leave it 0.0 so GpuTimer disables itself.
        let qf_props = unsafe { instance.get_physical_device_queue_family_properties(physical) };
        let valid_bits = qf_props
            .get(queue_family as usize)
            .map(|q| q.timestamp_valid_bits)
            .unwrap_or(0);
        let timestamp_period = if valid_bits > 0 { props.limits.timestamp_period } else { 0.0 };

        Ok(Self {
            physical,
            graphics_queue_family: queue_family,
            timestamp_period,
            allocator: ManuallyDrop::new(allocator),
            handle,
        })
    }
}

fn select_physical_device(
    instance: &Instance,
    surface: &Surface,
) -> Result<vk::PhysicalDevice, RhiError> {
    let devices =
        unsafe { instance.enumerate_physical_devices() }.map_err(RhiError::Vulkan)?;

    let mut best: Option<(vk::PhysicalDevice, bool)> = None;

    for &pdev in &devices {
        if !is_device_suitable(instance, surface, pdev) {
            continue;
        }
        let props = unsafe { instance.get_physical_device_properties(pdev) };
        let is_discrete = props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU;
        match best {
            None => best = Some((pdev, is_discrete)),
            Some((_, false)) if is_discrete => best = Some((pdev, is_discrete)),
            _ => {}
        }
    }

    best.map(|(d, _)| d).ok_or(RhiError::NoSuitableDevice)
}

fn is_device_suitable(
    instance: &Instance,
    surface: &Surface,
    pdev: vk::PhysicalDevice,
) -> bool {
    let props = unsafe { instance.get_physical_device_properties(pdev) };
    if props.api_version < vk::make_api_version(0, 1, 3, 0) {
        return false;
    }

    let exts = match unsafe { instance.enumerate_device_extension_properties(pdev) } {
        Ok(e) => e,
        Err(_) => return false,
    };
    let swapchain_name = c"VK_KHR_swapchain";
    let has_swapchain = exts.iter().any(|e| {
        unsafe { CStr::from_ptr(e.extension_name.as_ptr()) == swapchain_name }
    });
    if !has_swapchain {
        return false;
    }

    let mut vk13 = vk::PhysicalDeviceVulkan13Features::default();
    let mut vk12 = vk::PhysicalDeviceVulkan12Features::default();
    let mut features2 = vk::PhysicalDeviceFeatures2::default()
        .push_next(&mut vk13)
        .push_next(&mut vk12);
    unsafe { instance.get_physical_device_features2(pdev, &mut features2) };

    if vk13.dynamic_rendering == vk::FALSE || vk13.synchronization2 == vk::FALSE {
        return false;
    }
    if vk12.timeline_semaphore == vk::FALSE {
        return false;
    }

    find_graphics_present_family(instance, surface, pdev).is_ok()
}

pub(crate) fn find_graphics_present_family(
    instance: &Instance,
    surface: &Surface,
    pdev: vk::PhysicalDevice,
) -> Result<u32, RhiError> {
    let queue_families =
        unsafe { instance.get_physical_device_queue_family_properties(pdev) };

    for (idx, family) in queue_families.iter().enumerate() {
        if !family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
            continue;
        }
        let present_support = unsafe {
            surface.loader.get_physical_device_surface_support(
                pdev,
                idx as u32,
                surface.handle,
            )
        }
        .unwrap_or(false);
        if present_support {
            return Ok(idx as u32);
        }
    }
    Err(RhiError::NoSuitableDevice)
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            // Drop allocator first — it holds a clone of the device handle and must
            // release all allocations before the device is destroyed.
            ManuallyDrop::drop(&mut self.allocator);
            self.handle.destroy_device(None);
        }
    }
}

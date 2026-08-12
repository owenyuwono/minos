//! `sync` — Per-frame semaphores, fences, and a global timeline semaphore.

use ash::vk;

use crate::RhiError;

pub(crate) struct FrameSync {
    pub image_available: vk::Semaphore,
    pub in_flight_fence: vk::Fence,
}

pub(crate) struct SyncObjects {
    pub frames: Vec<FrameSync>,
    pub timeline: vk::Semaphore,
    device: ash::Device,
}

impl SyncObjects {
    pub fn new(device: &ash::Device, frames_in_flight: u32) -> Result<Self, RhiError> {
        let mut frames = Vec::with_capacity(frames_in_flight as usize);
        for _ in 0..frames_in_flight {
            let image_available = unsafe {
                device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
            }
            .map_err(RhiError::Vulkan)?;

            let in_flight_fence = unsafe {
                device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED),
                    None,
                )
            }
            .map_err(RhiError::Vulkan)?;

            frames.push(FrameSync {
                image_available,
                in_flight_fence,
            });
        }

        let mut timeline_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        let timeline = unsafe {
            device.create_semaphore(
                &vk::SemaphoreCreateInfo::default().push_next(&mut timeline_info),
                None,
            )
        }
        .map_err(RhiError::Vulkan)?;

        Ok(Self {
            frames,
            timeline,
            device: device.clone(),
        })
    }
}

impl Drop for SyncObjects {
    fn drop(&mut self) {
        unsafe {
            for frame in &self.frames {
                self.device.destroy_semaphore(frame.image_available, None);
                self.device.destroy_fence(frame.in_flight_fence, None);
            }
            self.device.destroy_semaphore(self.timeline, None);
        }
    }
}

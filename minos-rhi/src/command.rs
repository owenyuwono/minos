//! `command` — Per-frame command pools and primary command buffers.

use ash::vk;

use crate::RhiError;

pub(crate) struct FrameCommands {
    pub pool: vk::CommandPool,
    pub buffer: vk::CommandBuffer,
}

pub(crate) struct CommandPools {
    pub frames: Vec<FrameCommands>,
    device: ash::Device,
}

impl CommandPools {
    pub fn new(
        device: &ash::Device,
        queue_family: u32,
        frames_in_flight: u32,
    ) -> Result<Self, RhiError> {
        let mut frames = Vec::with_capacity(frames_in_flight as usize);
        for _ in 0..frames_in_flight {
            let pool = unsafe {
                device.create_command_pool(
                    &vk::CommandPoolCreateInfo::default()
                        .queue_family_index(queue_family)
                        .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                    None,
                )
            }
            .map_err(RhiError::Vulkan)?;

            let buffers = unsafe {
                device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(pool)
                        .level(vk::CommandBufferLevel::PRIMARY)
                        .command_buffer_count(1),
                )
            }
            .map_err(RhiError::Vulkan)?;

            frames.push(FrameCommands {
                pool,
                buffer: buffers[0],
            });
        }
        Ok(Self {
            frames,
            device: device.clone(),
        })
    }
}

impl Drop for CommandPools {
    fn drop(&mut self) {
        unsafe {
            for frame in &self.frames {
                // Command buffers are freed implicitly when the pool is destroyed.
                self.device.destroy_command_pool(frame.pool, None);
            }
        }
    }
}

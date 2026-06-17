//! `queue` — Graphics/present queue wrapper.

use ash::vk;

pub(crate) struct Queue {
    pub handle: vk::Queue,
    #[expect(dead_code)]
    pub family_index: u32,
}

impl Queue {
    pub fn from_device(device: &ash::Device, family_index: u32) -> Self {
        let handle = unsafe { device.get_device_queue(family_index, 0) };
        Self { handle, family_index }
    }
}

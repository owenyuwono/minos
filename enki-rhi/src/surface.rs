//! `surface` — VkSurfaceKHR creation from raw window handles.

use ash::{vk, Entry, Instance};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use crate::RhiError;

pub(crate) struct Surface {
    pub loader: ash::khr::surface::Instance,
    pub handle: vk::SurfaceKHR,
}

impl Surface {
    pub fn new(
        entry: &Entry,
        instance: &Instance,
        display_handle: RawDisplayHandle,
        window_handle: RawWindowHandle,
    ) -> Result<Self, RhiError> {
        let handle = unsafe {
            ash_window::create_surface(entry, instance, display_handle, window_handle, None)
        }
        .map_err(RhiError::Vulkan)?;

        let loader = ash::khr::surface::Instance::new(entry, instance);

        Ok(Self { loader, handle })
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_surface(self.handle, None);
        }
    }
}

//! `instance` — VkInstance creation with optional validation + debug messenger.

use ash::{vk, Entry, Instance as AshInstance};
use raw_window_handle::RawDisplayHandle;
use std::ffi::{c_void, CStr, CString};

use crate::RhiError;

pub(crate) struct Instance {
    pub entry: Entry,
    pub handle: AshInstance,
    pub debug_utils_loader: Option<ash::ext::debug_utils::Instance>,
    pub debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

unsafe extern "system" fn vulkan_debug_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let message = unsafe {
        CStr::from_ptr((*p_callback_data).p_message)
            .to_str()
            .unwrap_or("<invalid utf8>")
    };
    if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        log::error!("[Vulkan] {}", message);
    } else if message_severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        log::warn!("[Vulkan] {}", message);
    } else {
        log::info!("[Vulkan] {}", message);
    }
    vk::FALSE
}

impl Instance {
    pub fn new(display_handle: RawDisplayHandle, config: &crate::RhiConfig) -> Result<Self, RhiError> {
        let entry = unsafe { Entry::load() }.map_err(|e| RhiError::Other(e.to_string().into()))?;

        // Surface/swapchain extensions are hard requirements — fail if absent.
        let mut extension_names = ash_window::enumerate_required_extensions(display_handle)
            .map_err(|e| RhiError::Other(Box::new(e)))?
            .to_vec();

        let app_name = CString::new(config.app_name.as_str()).unwrap_or_default();
        let engine_name = CString::new("enki").unwrap();

        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .application_version(vk::make_api_version(0, 0, 1, 0))
            .engine_name(&engine_name)
            .engine_version(vk::make_api_version(0, 0, 1, 0))
            .api_version(vk::make_api_version(0, 1, 3, 0));

        // Enumerate available layers and extensions once.
        let available_layers = unsafe { entry.enumerate_instance_layer_properties() }
            .map_err(RhiError::Vulkan)?;
        let available_extensions = unsafe { entry.enumerate_instance_extension_properties(None) }
            .map_err(RhiError::Vulkan)?;

        let has_extension = |name: &CStr| {
            available_extensions.iter().any(|e| {
                let ext_name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) };
                ext_name == name
            })
        };

        let mut layer_names: Vec<*const i8> = Vec::new();
        let validation_layer = CString::new("VK_LAYER_KHRONOS_validation").unwrap();

        // Resolve whether validation is actually usable on this machine.
        let validation_layer_enabled = if config.validation {
            let layer_present = available_layers.iter().any(|l| {
                let name = unsafe { CStr::from_ptr(l.layer_name.as_ptr()) };
                name == validation_layer.as_c_str()
            });
            if layer_present {
                layer_names.push(validation_layer.as_ptr());
                true
            } else {
                log::warn!(
                    "Vulkan validation requested but VK_LAYER_KHRONOS_validation is not \
                     available — continuing without validation. Install the Vulkan validation \
                     layers (e.g. `vulkan-validationlayers`) to enable it."
                );
                false
            }
        } else {
            false
        };

        // VK_EXT_debug_utils is optional — enable only when available and requested.
        let debug_utils_enabled = validation_layer_enabled
            && has_extension(ash::ext::debug_utils::NAME);
        if validation_layer_enabled && !debug_utils_enabled {
            log::warn!(
                "VK_EXT_debug_utils not available — Vulkan debug messenger will not be installed."
            );
        }
        if debug_utils_enabled {
            extension_names.push(ash::ext::debug_utils::NAME.as_ptr());
        }

        let mut debug_create_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                    | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(vulkan_debug_callback));

        let mut create_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names);

        if debug_utils_enabled {
            create_info = create_info.push_next(&mut debug_create_info);
        }

        let handle = unsafe { entry.create_instance(&create_info, None) }
            .map_err(RhiError::Vulkan)?;

        let (debug_utils_loader, debug_messenger) = if debug_utils_enabled {
            let loader = ash::ext::debug_utils::Instance::new(&entry, &handle);
            let messenger = unsafe {
                loader.create_debug_utils_messenger(&debug_create_info, None)
            }
            .map_err(RhiError::Vulkan)?;
            (Some(loader), Some(messenger))
        } else {
            (None, None)
        };

        log::info!(
            "Vulkan instance created (API 1.3, validation={}, debug_utils={})",
            validation_layer_enabled,
            debug_utils_enabled,
        );

        Ok(Self { entry, handle, debug_utils_loader, debug_messenger })
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            if let (Some(loader), Some(messenger)) =
                (&self.debug_utils_loader, self.debug_messenger)
            {
                loader.destroy_debug_utils_messenger(messenger, None);
            }
            self.handle.destroy_instance(None);
        }
    }
}

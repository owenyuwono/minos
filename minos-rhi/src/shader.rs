//! `shader` — WGSL → SPIR-V compilation and VkShaderModule creation.
//!
//! # Design
//! A single [`ShaderModule`] wraps one `VkShaderModule` compiled from a WGSL
//! source file that may contain **multiple entry points** (e.g. `vs_main` and
//! `fs_main`). The caller specifies which entry-point names to use when
//! constructing a pipeline via [`crate::pipeline::GraphicsPipelineDesc`].
//!
//! # SPIR-V binding preservation
//! `@group(N) @binding(M)` annotations in WGSL are passed through to SPIR-V as
//! `DescriptorSet = N, Binding = M` using naga's identity binding map.

use ash::vk;
use naga::{
    back::spv,
    front::wgsl,
    valid::{Capabilities, ValidationFlags, Validator},
};
// Note: naga::ShaderStage used only when PipelineOptions is needed (per-entry compilation).
// For multi-entry modules we pass pipeline_options=None to write_vec.

use crate::RhiError;

// ── Handle ─────────────────────────────────────────────────────────────────

/// Opaque handle to a compiled [`VkShaderModule`] managed by the RHI.
///
/// Obtain via [`crate::Rhi::create_shader_module`].
/// The handle is only valid for the lifetime of the `Rhi` that created it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ShaderModule {
    pub(crate) handle: vk::ShaderModule,
}

// ── Compilation ─────────────────────────────────────────────────────────────

/// Compile WGSL source to a `VkShaderModule`.
///
/// The WGSL source may contain any number of entry points; the compiled module
/// includes all of them. Entry-point names are later referenced by name in
/// [`crate::pipeline::GraphicsPipelineDesc`].
///
/// Returns [`RhiError::Other`] on parse or validation failure, with the naga
/// diagnostic message included.
pub(crate) fn compile_wgsl(device: &ash::Device, wgsl_source: &str) -> Result<ShaderModule, RhiError> {
    // 1. Parse WGSL.
    let module = wgsl::parse_str(wgsl_source).map_err(|e| {
        let msg = format!("WGSL parse error: {}", e.emit_to_string(wgsl_source));
        log::error!("{}", msg);
        RhiError::Other(msg.into())
    })?;

    // 2. Validate (all capabilities, strict validation).
    let module_info = Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .map_err(|e| {
            let msg = format!("WGSL validation error: {:?}", e);
            log::error!("{}", msg);
            RhiError::Other(msg.into())
        })?;

    // 3. Emit SPIR-V with identity binding map (group→DescriptorSet, binding→Binding).
    //    `binding_map` left empty = identity: every @group/@binding passes through as-is.
    //    `pipeline_options = None` emits ALL entry points into the same module — the
    //    caller selects which entry point to use at pipeline-create time via the entry
    //    point name in `GraphicsPipelineDesc`.
    let spv_options = spv::Options {
        lang_version: (1, 3),
        flags: spv::WriterFlags::DEBUG,
        binding_map: Default::default(), // identity — preserves @group/@binding
        ..Default::default()
    };

    let spv_words = spv::write_vec(&module, &module_info, &spv_options, None)
        .map_err(|e| {
            let msg = format!("SPIR-V emission error: {:?}", e);
            log::error!("{}", msg);
            RhiError::Other(msg.into())
        })?;

    // 4. Create VkShaderModule.
    let create_info = vk::ShaderModuleCreateInfo::default().code(&spv_words);
    let handle = unsafe { device.create_shader_module(&create_info, None) }
        .map_err(RhiError::Vulkan)?;

    Ok(ShaderModule { handle })
}

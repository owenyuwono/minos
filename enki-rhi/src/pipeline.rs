//! `pipeline` — Graphics pipeline creation for the forward render pass.
//!
//! # Vertex input layout
//! Four separate vertex buffer bindings, one attribute each:
//!
//! | Binding | Location | Format              | Semantic    |
//! |---------|----------|---------------------|-------------|
//! | 0       | 0        | R32G32B32_SFLOAT    | position    |
//! | 1       | 1        | R32G32B32_SFLOAT    | normal      |
//! | 2       | 2        | R32G32B32_SFLOAT    | color       |
//! | 3       | 3        | R32G32B32_SFLOAT    | plate_color |
//!
//! # Depth
//! Reversed-Z: depth test GREATER, clear value 0.0 (set by `begin_frame`).
//! The pipeline does NOT define `minDepthBounds`/`maxDepthBounds`.
//!
//! # Dynamic state
//! `VIEWPORT` and `SCISSOR` are dynamic — `set_viewport_scissor_full` must be
//! called before each draw call.
//!
//! # Push constants
//! The concrete struct lives in `enki-render`. The RHI only needs the byte size
//! and stage mask, supplied via [`GraphicsPipelineDesc::push_constant_size`].

use std::collections::BTreeMap;
use std::ffi::CString;

use ash::vk;

use crate::{shader::ShaderModule, RhiError};

// ── Handle ──────────────────────────────────────────────────────────────────

/// Opaque handle to a compiled graphics pipeline.
///
/// Obtain via [`crate::Rhi::create_graphics_pipeline`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineHandle(pub(crate) u64);

// ── Desc ─────────────────────────────────────────────────────────────────────

/// Description of a graphics pipeline to create.
pub struct GraphicsPipelineDesc<'a> {
    /// Compiled shader module containing both vertex and fragment entry points.
    pub shader: ShaderModule,
    /// Entry-point name for the vertex shader stage (e.g. `"vs_main"`).
    pub vs_entry: &'a str,
    /// Entry-point name for the fragment shader stage (e.g. `"fs_main"`).
    pub fs_entry: &'a str,
    /// Byte size of the push-constant range (stages: VERTEX | FRAGMENT).
    /// Must be ≤ 128 bytes (guaranteed minimum across all Vulkan drivers).
    pub push_constant_size: u32,
    /// Descriptor set layout for set 0 (the per-frame UBO layout).
    pub set0_layout: vk::DescriptorSetLayout,
    /// Target swapchain/color format (used by dynamic rendering).
    pub color_format: vk::Format,
    /// Depth format — always `D32_SFLOAT` for enki.
    pub depth_format: vk::Format,
    /// MSAA sample count.
    pub samples: vk::SampleCountFlags,
    /// `true` → straight alpha-blend with depth write OFF (water / transparent).
    /// `false` → opaque, depth write ON.
    pub blend: bool,
    /// `true` → `FILL` polygon mode.
    /// `false` → `LINE` polygon mode (wireframe debug variant).
    pub fill: bool,
}

/// Description of a compute pipeline to create.
///
/// Generic (not Nanite-specific): any GPU-driven compute pass uses this.
pub struct ComputePipelineDesc<'a> {
    /// Compiled shader module containing the compute entry point.
    pub shader: ShaderModule,
    /// Entry-point name for the compute stage (e.g. `"cs_main"`).
    pub entry: &'a str,
    /// Byte size of the push-constant range (stage: COMPUTE). 0 for none.
    pub push_constant_size: u32,
    /// Descriptor set layouts bound to this pipeline (set 0, 1, …).
    pub set_layouts: &'a [vk::DescriptorSetLayout],
}

// ── Pipeline entry ──────────────────────────────────────────────────────────

pub(crate) struct PipelineEntry {
    pub pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
}

// ── Pipeline store ──────────────────────────────────────────────────────────

/// Manages all graphics pipelines allocated by the RHI.
pub(crate) struct PipelineStore {
    map: BTreeMap<u64, PipelineEntry>,
    next_id: u64,
    device: ash::Device,
}

impl PipelineStore {
    pub fn new(device: &ash::Device) -> Self {
        Self {
            map: BTreeMap::new(),
            next_id: 1,
            device: device.clone(),
        }
    }

    /// Create a graphics pipeline from `desc` and return its handle.
    pub fn create(
        &mut self,
        desc: &GraphicsPipelineDesc<'_>,
    ) -> Result<PipelineHandle, RhiError> {
        self.create_inner(desc, false, false, 4)
    }

    /// Like [`Self::create`] but with a 5th `vec3` vertex attribute at binding/location
    /// 4 (the CDLOD geomorph displacement stream).
    pub fn create_morph(
        &mut self,
        desc: &GraphicsPipelineDesc<'_>,
    ) -> Result<PipelineHandle, RhiError> {
        self.create_inner(desc, false, false, 5)
    }

    /// Like [`Self::create`] but with NO vertex input bindings — the vertex shader
    /// pulls geometry from storage buffers via `@builtin(vertex_index)`. Used by
    /// GPU-driven (e.g. cluster) draw paths.
    pub fn create_pulling(
        &mut self,
        desc: &GraphicsPipelineDesc<'_>,
    ) -> Result<PipelineHandle, RhiError> {
        self.create_inner(desc, true, false, 4)
    }

    /// Like [`Self::create_pulling`] but **depth-only**: no fragment stage, no color
    /// attachment (depth write ON). For a sun shadow-map caster pass that pulls the
    /// same geometry and projects with the light view-proj. `desc.color_format` /
    /// `fs_entry` are ignored; `depth_format` + `samples` are used.
    pub fn create_pulling_depth(
        &mut self,
        desc: &GraphicsPipelineDesc<'_>,
    ) -> Result<PipelineHandle, RhiError> {
        self.create_inner(desc, true, true, 4)
    }

    /// Depth-only pipeline WITH the standard 4×vec3 vertex input (no fragment, no
    /// color attachment). For a shadow-map caster of vertex-buffer geometry (e.g.
    /// the character). `desc.color_format`/`fs_entry` are ignored.
    pub fn create_depth(
        &mut self,
        desc: &GraphicsPipelineDesc<'_>,
    ) -> Result<PipelineHandle, RhiError> {
        self.create_inner(desc, false, true, 4)
    }

    fn create_inner(
        &mut self,
        desc: &GraphicsPipelineDesc<'_>,
        vertex_pulling: bool,
        depth_only: bool,
        n_vec3_attrs: usize,
    ) -> Result<PipelineHandle, RhiError> {
        // ── Pipeline layout ─────────────────────────────────────────────────
        let set_layouts = [desc.set0_layout];

        // A zero-size push-constant range is invalid; only declare one when used.
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .offset(0)
            .size(desc.push_constant_size);
        let mut layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        if desc.push_constant_size > 0 {
            layout_info = layout_info.push_constant_ranges(std::slice::from_ref(&push_range));
        }

        let layout = unsafe { self.device.create_pipeline_layout(&layout_info, None) }
            .map_err(RhiError::Vulkan)?;

        // ── Shader stages ───────────────────────────────────────────────────
        let vs_name = CString::new(desc.vs_entry).map_err(|_| {
            RhiError::Other("vs_entry contains null byte".to_string().into())
        })?;
        let fs_name = CString::new(desc.fs_entry).map_err(|_| {
            RhiError::Other("fs_entry contains null byte".to_string().into())
        })?;

        // Depth-only passes (shadow caster) have no fragment stage.
        let mut stages = vec![vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(desc.shader.handle)
            .name(&vs_name)];
        if !depth_only {
            stages.push(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(desc.shader.handle)
                    .name(&fs_name),
            );
        }

        // ── Vertex input ────────────────────────────────────────────────────
        // `n_vec3_attrs` separate bindings, each carries one vec3<f32> attribute
        // (4 = pos/nrm/col/plate; 5 adds the geomorph displacement at location 4).
        let bindings: Vec<_> = (0..n_vec3_attrs)
            .map(|i| {
                vk::VertexInputBindingDescription::default()
                    .binding(i as u32)
                    .stride(12) // sizeof(vec3) = 3 × 4
                    .input_rate(vk::VertexInputRate::VERTEX)
            })
            .collect();

        let attributes: Vec<_> = (0..n_vec3_attrs)
            .map(|i| {
                vk::VertexInputAttributeDescription::default()
                    .location(i as u32)
                    .binding(i as u32)
                    .format(vk::Format::R32G32B32_SFLOAT)
                    .offset(0)
            })
            .collect();

        let vertex_input = if vertex_pulling {
            // No vertex buffers — the shader pulls from storage via vertex_index.
            vk::PipelineVertexInputStateCreateInfo::default()
        } else {
            vk::PipelineVertexInputStateCreateInfo::default()
                .vertex_binding_descriptions(&bindings)
                .vertex_attribute_descriptions(&attributes)
        };

        // ── Input assembly ──────────────────────────────────────────────────
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        // ── Rasterisation ───────────────────────────────────────────────────
        let polygon_mode = if desc.fill {
            vk::PolygonMode::FILL
        } else {
            vk::PolygonMode::LINE
        };

        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(polygon_mode)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false)
            .line_width(1.0);

        // ── MSAA ────────────────────────────────────────────────────────────
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(desc.samples)
            .sample_shading_enable(false);

        // ── Depth / stencil — reversed-Z: compare GREATER ──────────────────
        let depth_write = !desc.blend; // blended pass skips depth write
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(depth_write)
            .depth_compare_op(vk::CompareOp::GREATER) // reversed-Z
            .depth_bounds_test_enable(false)
            .stencil_test_enable(false);

        // ── Colour blending ─────────────────────────────────────────────────
        let blend_attachment = if desc.blend {
            // Straight alpha-blend: srcColor*srcA + dstColor*(1-srcA).
            // Alpha channel: outA = srcA*1 + dstA*(1-srcA) — same contract
            // as water_pass.rs which documents dst_alpha_blend_factor = ONE_MINUS_SRC_ALPHA.
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .color_blend_op(vk::BlendOp::ADD)
                .src_alpha_blend_factor(vk::BlendFactor::ONE)
                .dst_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                .alpha_blend_op(vk::BlendOp::ADD)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
        } else {
            // Opaque — blending off.
            vk::PipelineColorBlendAttachmentState::default()
                .blend_enable(false)
                .color_write_mask(vk::ColorComponentFlags::RGBA)
        };

        // Depth-only: no color attachment to blend.
        let blend_attachments: &[vk::PipelineColorBlendAttachmentState] =
            if depth_only { &[] } else { std::slice::from_ref(&blend_attachment) };
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .attachments(blend_attachments);

        // ── Dynamic state ───────────────────────────────────────────────────
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);

        // Viewport / scissor count must be set even though they are dynamic.
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        // ── Dynamic rendering — no render pass object ───────────────────────
        // Depth-only: zero color attachments.
        let color_formats: &[vk::Format] =
            if depth_only { &[] } else { std::slice::from_ref(&desc.color_format) };
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(color_formats)
            .depth_attachment_format(desc.depth_format);

        // ── Assemble and create ──────────────────────────────────────────────
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipelines = unsafe {
            self.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
        }
        .map_err(|(_, e)| RhiError::Vulkan(e))?;

        let id = self.next_id;
        self.next_id += 1;
        self.map.insert(id, PipelineEntry { pipeline: pipelines[0], layout });

        Ok(PipelineHandle(id))
    }

    /// Create a fullscreen post-process pipeline: no vertex input (the VS emits a
    /// fullscreen triangle via `vertex_index`), no depth, no culling, a single
    /// color attachment, and a caller-supplied descriptor set layout. 1-sample.
    /// Currently unused (the TAA resolve was its only caller) but kept as a generic
    /// primitive for a future post-process pass (FXAA / bloom / tonemap).
    #[allow(dead_code)]
    pub(crate) fn create_fullscreen(
        &mut self,
        shader: &ShaderModule,
        vs_entry: &str,
        fs_entry: &str,
        set_layout: vk::DescriptorSetLayout,
        color_format: vk::Format,
    ) -> Result<PipelineHandle, RhiError> {
        let set_layouts = [set_layout];
        let layout_info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let layout = unsafe { self.device.create_pipeline_layout(&layout_info, None) }
            .map_err(RhiError::Vulkan)?;

        let vs_name = CString::new(vs_entry)
            .map_err(|_| RhiError::Other("vs_entry contains null byte".to_string().into()))?;
        let fs_name = CString::new(fs_entry)
            .map_err(|_| RhiError::Other("fs_entry contains null byte".to_string().into()))?;
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(shader.handle)
                .name(&vs_name),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(shader.handle)
                .name(&fs_name),
        ];

        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(false)
            .depth_write_enable(false);
        let blend_attachment = vk::PipelineColorBlendAttachmentState::default()
            .blend_enable(false)
            .color_write_mask(vk::ColorComponentFlags::RGBA);
        let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
            .attachments(std::slice::from_ref(&blend_attachment));
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);

        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(std::slice::from_ref(&color_format));

        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(layout)
            .push_next(&mut rendering_info);

        let pipelines = unsafe {
            self.device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&pipeline_info),
                None,
            )
        }
        .map_err(|(_, e)| {
            unsafe { self.device.destroy_pipeline_layout(layout, None) };
            RhiError::Vulkan(e)
        })?;

        let id = self.next_id;
        self.next_id += 1;
        self.map.insert(id, PipelineEntry { pipeline: pipelines[0], layout });
        Ok(PipelineHandle(id))
    }

    /// Create a compute pipeline from `desc` and return its handle.
    pub fn create_compute(
        &mut self,
        desc: &ComputePipelineDesc<'_>,
    ) -> Result<PipelineHandle, RhiError> {
        // ── Pipeline layout ─────────────────────────────────────────────────
        let push_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(desc.push_constant_size);

        let mut layout_info =
            vk::PipelineLayoutCreateInfo::default().set_layouts(desc.set_layouts);
        if desc.push_constant_size > 0 {
            layout_info = layout_info.push_constant_ranges(std::slice::from_ref(&push_range));
        }

        let layout = unsafe { self.device.create_pipeline_layout(&layout_info, None) }
            .map_err(RhiError::Vulkan)?;

        // ── Stage ───────────────────────────────────────────────────────────
        let entry = CString::new(desc.entry).map_err(|_| {
            RhiError::Other("compute entry contains null byte".to_string().into())
        })?;
        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(desc.shader.handle)
            .name(&entry);

        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(layout);

        let pipelines = unsafe {
            self.device.create_compute_pipelines(
                vk::PipelineCache::null(),
                std::slice::from_ref(&info),
                None,
            )
        }
        .map_err(|(_, e)| {
            // Layout would leak on failure — destroy it.
            unsafe { self.device.destroy_pipeline_layout(layout, None) };
            RhiError::Vulkan(e)
        })?;

        let id = self.next_id;
        self.next_id += 1;
        self.map.insert(id, PipelineEntry { pipeline: pipelines[0], layout });
        Ok(PipelineHandle(id))
    }

    /// Return the internal entry for a handle, or `InvalidHandle`.
    pub fn get(&self, handle: PipelineHandle) -> Result<&PipelineEntry, RhiError> {
        self.map.get(&handle.0).ok_or(RhiError::InvalidHandle)
    }

    /// Destroy all pipelines. Called from `Rhi::drop`.
    pub fn destroy_all(&mut self) {
        for (_, entry) in std::mem::take(&mut self.map) {
            unsafe {
                self.device.destroy_pipeline(entry.pipeline, None);
                self.device.destroy_pipeline_layout(entry.layout, None);
            }
        }
    }
}

impl Drop for PipelineStore {
    fn drop(&mut self) {
        self.destroy_all();
    }
}


//! `clouds` — volumetric clouds whose coverage is ADVECTED by the planet's wind.
//!
//! A fullscreen raymarch of a spherical cloud shell, drawn inside the ocean's
//! refraction split (the reopened 1× instance) so it composites over the resolved
//! opaque scene and can sample its depth for terrain occlusion.
//!
//! Motion comes from [`crate::clouds_advect`]: a background worker semi-Lagrangian-
//! advects a coverage field along the baked wind each tick (clouds translate, merge
//! at convergence, follow the flow). The render thread copies that evolving grid
//! into a per-FiF storage buffer; the shader samples it for placement. The shader
//! no longer warps noise by `wind×time` (that sheared/tore) — it just reads coverage.
//!
//! Modeled on [`crate::ocean::WaveSurface`] for the GPU plumbing + worker pattern.
//! Geometry is a fullscreen triangle (vertex-pulling) so the shell is intersected
//! analytically — one path for orbit / ground / fly-through. **TAA-on only.**
//!
//! ponytail: analytic in-shader noise (minos-rhi can't upload a 3D texture); coverage
//! is a coarse equirect grid in a storage buffer (no texture upload). See
//! `docs/clouds-advection-research.md`.

use std::sync::Arc;

use bytemuck::{cast_slice, Pod, Zeroable};
use minos_planet::height::HeightField;
use minos_render::{camera::Camera, frame::FrameUniforms};
use minos_rhi::{vk, BindingDesc, BufferHandle, GraphicsPipelineDesc, PipelineHandle, Rhi, RhiError};
use glam::Vec3;

use crate::clouds_advect::{AdvectParams, CloudAdvect, GRID_H, GRID_W};

const CLOUDS_WGSL: &str = include_str!("clouds.wgsl");

/// GUI-tunable cloud knobs. Frequencies/altitudes are in metres (planet scale).
#[derive(Debug, Clone, Copy)]
pub struct CloudParams {
    /// Overall coverage scale, 0..1 (multiplies the advected climate coverage).
    pub coverage: f32,
    /// Extinction per metre (optical thickness) — drives opacity + self-shadow.
    pub density: f32,
    /// Cloud base altitude above sea level (metres).
    pub base_alt_m: f32,
    /// Layer thickness (metres).
    pub thickness_m: f32,
    /// Wind advection speed (metres/second) — drives the coverage worker.
    pub wind_speed: f32,
    /// Base-shape feature size (metres) — larger = bigger, smoother clouds.
    pub noise_scale: f32,
    /// Primary raymarch step count.
    pub steps: f32,
    /// Henyey-Greenstein anisotropy (forward scatter / silver lining).
    pub hg_g: f32,
    /// Cloud type 0..1: stratus (low, flat) → cumulus (tall, billowy).
    pub cloud_type: f32,
    /// Powder (Beer-powder) dark-edge strength, 0..1.
    pub powder: f32,
    /// Pseudo-curl turbulence amount (wispy churning edges), 0..1.
    pub curl: f32,
    /// Climate influence: 0 (uniform coverage) .. 1 (full advected climate coverage).
    pub moisture_influence: f32,
    /// Advection form rate `k_form` (1/s) — how fast clouds re-form toward the climate.
    pub form_rate: f32,
    /// Advection decay rate `k_decay` (1/s) — exponential sink (anti-runaway).
    pub decay_rate: f32,
}

impl Default for CloudParams {
    fn default() -> Self {
        Self {
            coverage: 0.6,
            // Extinction/m. The OLD 0.12 was opaque + textured but read WHITE (no tonemap)
            // and as a SHEET (saturated coverage). Now that the body is ACES-tonemapped and
            // coverage is region-masked, density can sit high enough for dense, self-shadowed
            // BILLOW texture without going white/sheet — 0.015 gives opaque cores (strong
            // self-shadow detail) with a few-step soft edge. (0.12→1-step opaque slab.)
            density: 0.015,
            base_alt_m: 2000.0,
            thickness_m: 3000.0,
            wind_speed: 1500.0,
            noise_scale: 2500.0,
            steps: 48.0,
            hg_g: 0.4,
            cloud_type: 0.6,
            powder: 0.3,
            curl: 0.5,
            moisture_influence: 1.0,
            // form/decay set the worker's steady-state coverage c* = form·p/(form·p+decay).
            // The old 0.06/0.02 left even mean-moisture cells at c*~0.35 (above threshold),
            // so SL diffusion crept the field across the whole planet. Stronger decay keeps
            // c* patchy (only genuinely moist cells stay cloudy) and bounds the spread.
            form_rate: 0.12,
            decay_rate: 0.06,
        }
    }
}

/// GPU mirror of `clouds.wgsl`'s `CloudParams` (std140, all vec4 → 176 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CloudParamsGpu {
    right: [f32; 4],
    up: [f32; 4],
    fwd: [f32; 4],
    center_rel: [f32; 4], // xyz = planet centre − camera; w = planet radius
    shell: [f32; 4],      // base_alt, thickness, coverage, density
    wind: [f32; 4],       // unused (advection moved to the worker)
    march: [f32; 4],      // steps, hg_g, noise_scale, time
    screen: [f32; 4],     // width, height, tan_half_fov, aspect
    misc: [f32; 4],       // debug, proj_a, proj_b, _pad
    extra: [f32; 4],      // cloud_type, powder, curl, moisture_influence
    grid: [f32; 4],       // grid_w, grid_h, has_coverage, _pad
}

struct CloudFrame {
    frame_ubo: BufferHandle,
    cloud_ubo: BufferHandle,
    /// Per-FiF advected-coverage grid (written each frame from the worker).
    cov_buf: BufferHandle,
    set: vk::DescriptorSet,
}

/// Volumetric cloud renderer: one fullscreen raymarch pass + an advection worker.
pub struct Clouds {
    pipeline: PipelineHandle,
    layout: vk::PipelineLayout,
    sampler: vk::Sampler,
    /// 3-index buffer feeding `vertex_index` for the fullscreen triangle.
    idx: BufferHandle,
    frames: Vec<CloudFrame>,
    /// Background semi-Lagrangian advection — flows + evolves the cloud field. The
    /// shader samples this field directly, so all motion is advection (no warp).
    advect: CloudAdvect,
    /// CPU staging for the per-frame coverage upload.
    cov_scratch: Vec<f32>,
    base_radius: f64,
}

impl Clouds {
    pub fn new(
        rhi: &mut Rhi,
        color_format: vk::Format,
        base_radius: f64,
    ) -> Result<Self, RhiError> {
        let both = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let frag = vk::ShaderStageFlags::FRAGMENT;
        let layout_handle = rhi.create_descriptor_set_layout(&[
            BindingDesc { binding: 0, ty: vk::DescriptorType::UNIFORM_BUFFER, stages: both },
            BindingDesc { binding: 1, ty: vk::DescriptorType::UNIFORM_BUFFER, stages: both },
            BindingDesc { binding: 2, ty: vk::DescriptorType::SAMPLED_IMAGE, stages: frag },
            BindingDesc { binding: 3, ty: vk::DescriptorType::SAMPLER, stages: frag },
            BindingDesc { binding: 4, ty: vk::DescriptorType::STORAGE_BUFFER, stages: frag },
        ])?;
        let sampler = rhi.create_sampler()?;

        let shader = rhi.create_shader_module(CLOUDS_WGSL)?;
        // Vertex-pulling: the VS emits a fullscreen triangle from vertex_index, so no
        // vertex buffers. 1× target (the water split), blend → depth-write OFF.
        let pipeline = rhi.create_graphics_pipeline_pulling(&GraphicsPipelineDesc {
            shader,
            vs_entry: "vs_main",
            fs_entry: "fs_main",
            push_constant_size: 0,
            set0_layout: layout_handle,
            color_format,
            depth_format: vk::Format::D32_SFLOAT,
            samples: vk::SampleCountFlags::TYPE_1,
            blend: true,
            fill: true,
        })?;
        rhi.destroy_shader_module(shader);
        let layout = rhi.pipeline_layout(pipeline)?;

        // Index buffer values ARE the vertex indices the VS reads (no vertex buffer).
        let idx = rhi.create_index_buffer(&[0u32, 1, 2])?;

        let fif = rhi.frames_in_flight();
        let cov_bytes = (GRID_W * GRID_H * std::mem::size_of::<f32>()) as u64;
        let mut frames = Vec::with_capacity(fif);
        for _ in 0..fif {
            let frame_ubo = rhi.create_gpu_buffer(
                std::mem::size_of::<FrameUniforms>() as u64,
                true,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            let cloud_ubo = rhi.create_gpu_buffer(
                std::mem::size_of::<CloudParamsGpu>() as u64,
                true,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            let cov_buf = rhi.create_gpu_buffer(cov_bytes, true, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let set = rhi.allocate_descriptor_set(layout_handle)?;
            rhi.write_uniform_binding(set, 0, frame_ubo)?;
            rhi.write_uniform_binding(set, 1, cloud_ubo)?;
            rhi.write_sampler_binding(set, 3, sampler);
            rhi.write_storage_binding(set, 4, cov_buf)?;
            // Binding 2 (scene depth) is written per-frame in `record`.
            frames.push(CloudFrame { frame_ubo, cloud_ubo, cov_buf, set });
        }

        Ok(Self {
            pipeline,
            layout,
            sampler,
            idx,
            frames,
            advect: CloudAdvect::new(base_radius),
            cov_scratch: vec![0.0; GRID_W * GRID_H],
            base_radius,
        })
    }

    /// Record the cloud raymarch. Call inside the reopened 1× water instance,
    /// AFTER the waves (clouds are a shell above sea level), sampling the resolved
    /// opaque `scene_depth`. `time` is real elapsed seconds (detail dither only).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        rhi: &mut Rhi,
        fi: u32,
        fu: &FrameUniforms,
        camera: &Camera,
        scene_depth: vk::ImageView,
        extent: (u32, u32),
        time: f32,
        params: CloudParams,
        debug: bool,
    ) -> Result<(), RhiError> {
        // Drive the advection worker + pull its latest coverage grid for upload.
        // (active is driven each frame via `set_advect_active` from `clouds_enabled`;
        // record is only reached while clouds are drawn, so don't latch it here.)
        self.advect.set_params(AdvectParams {
            wind_speed: params.wind_speed,
            form_rate: params.form_rate,
            decay_rate: params.decay_rate,
        });
        self.advect.copy_latest(&mut self.cov_scratch);
        let has_coverage = self.advect.is_ready();

        let f = &self.frames[fi as usize];
        rhi.write_sampled_image_binding(
            f.set,
            2,
            scene_depth,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        rhi.write_storage_bytes(f.frame_ubo, bytemuck::bytes_of(fu))?;
        rhi.write_storage_bytes(f.cov_buf, cast_slice(&self.cov_scratch))?;

        // Camera basis for per-pixel ray reconstruction (no inverse matrix needed).
        let fwd = (camera.orientation * Vec3::NEG_Z).normalize();
        let right = (camera.orientation * Vec3::X).normalize();
        let up = (camera.orientation * Vec3::Y).normalize();
        let center_rel = (-camera.position).as_vec3(); // planet centre relative to camera
        let tan_half = (camera.fov_y_radians * 0.5).tan();
        let aspect = extent.0 as f32 / extent.1.max(1) as f32;

        // Reversed-Z projection constants (solve forward distance from sampled depth).
        let (near, far) = (camera.near, camera.far);
        let proj_a = near / (far - near);
        let proj_b = near * far / (far - near);

        let gpu = CloudParamsGpu {
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            fwd: [fwd.x, fwd.y, fwd.z, 0.0],
            center_rel: [center_rel.x, center_rel.y, center_rel.z, self.base_radius as f32],
            shell: [params.base_alt_m, params.thickness_m, params.coverage, params.density],
            wind: [0.0, 0.0, 0.0, 0.0], // unused (motion is in the advected field)
            march: [params.steps, params.hg_g, params.noise_scale, time],
            screen: [extent.0 as f32, extent.1 as f32, tan_half, aspect],
            misc: [if debug { 1.0 } else { 0.0 }, proj_a, proj_b, 0.0],
            extra: [params.cloud_type, params.powder, params.curl, params.moisture_influence],
            grid: [
                GRID_W as f32,
                GRID_H as f32,
                if has_coverage { 1.0 } else { 0.0 },
                0.0,
            ],
        };
        rhi.write_storage_bytes(f.cloud_ubo, bytemuck::bytes_of(&gpu))?;

        rhi.cmd_bind_descriptor_set(fi, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, f.set);
        rhi.cmd_bind_pipeline(fi, vk::PipelineBindPoint::GRAPHICS, self.pipeline)?;
        rhi.bind_index_buffer(fi, self.idx)?;
        rhi.draw_indexed(fi, 3);
        Ok(())
    }

    /// Supply the planet as the wind + moisture source once loaded; the worker
    /// bakes its grids and starts advecting. Until then clouds use uniform coverage.
    pub fn set_source(&mut self, hf: Arc<dyn HeightField>) {
        self.advect.set_source(hf);
    }

    /// Two-way idle gate: drive the advection worker's active state from the app's
    /// `clouds_enabled` each frame (mirrors the ocean worker's `active.store`), so the
    /// worker idles when clouds are toggled off instead of spinning at TICK_HZ forever.
    pub fn set_advect_active(&self, on: bool) {
        self.advect.set_active(on);
    }

    /// Free the caller-owned sampler before RHI teardown.
    pub fn destroy(&self, rhi: &Rhi) {
        rhi.destroy_sampler(self.sampler);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn cloud_params_gpu_is_std140_sized() {
        // 11 × vec4 = 176 bytes.
        assert_eq!(std::mem::size_of::<super::CloudParamsGpu>(), 176);
    }

    #[test]
    fn clouds_wgsl_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(super::CLOUDS_WGSL)
            .unwrap_or_else(|e| panic!("clouds.wgsl failed to parse: {e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("clouds.wgsl failed to validate: {e:?}"));
    }
}

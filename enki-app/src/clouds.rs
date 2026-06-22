//! `clouds` — wind-driven volumetric clouds.
//!
//! A fullscreen raymarch of a spherical cloud shell, drawn inside the ocean's
//! refraction split (the reopened 1× instance) so it composites over the resolved
//! opaque scene and can sample its depth for terrain occlusion. The headline
//! feature: clouds advect with the planet's baked wind field (one global vector
//! sampled at the sub-camera point each frame — see `record`).
//!
//! Modeled on [`crate::ocean::WaveSurface`] for the GPU plumbing (custom set +
//! per-FiF UBOs) and on [`crate::atmosphere::Atmosphere`] for the lifecycle.
//! Unlike those, the geometry is a fullscreen triangle (vertex-pulling), so the
//! shell is intersected analytically in the shader — one path for orbit / ground /
//! fly-through. **TAA-on only** (needs the water split); off → not drawn.
//!
//! ponytail: analytic in-shader noise (enki-rhi can't upload a 3D texture), full
//! res, no cloud reprojection. Upgrades live in `docs/clouds-research.md`.

use std::sync::Arc;

use bytemuck::{Pod, Zeroable};
use enki_planet::height::HeightField;
use enki_render::{camera::Camera, frame::FrameUniforms};
use enki_rhi::{vk, BindingDesc, BufferHandle, GraphicsPipelineDesc, PipelineHandle, Rhi, RhiError};
use glam::Vec3;

const CLOUDS_WGSL: &str = include_str!("clouds.wgsl");

/// GUI-tunable cloud knobs. Frequencies/altitudes are in metres (planet scale).
#[derive(Debug, Clone, Copy)]
pub struct CloudParams {
    /// Fraction of sky covered, 0..1.
    pub coverage: f32,
    /// Extinction per metre (optical thickness) — drives opacity + self-shadow.
    pub density: f32,
    /// Cloud base altitude above sea level (metres).
    pub base_alt_m: f32,
    /// Layer thickness (metres).
    pub thickness_m: f32,
    /// Wind advection speed (metres/second), scaled by the local wind strength.
    pub wind_speed: f32,
    /// Base-shape feature size (metres) — larger = bigger, smoother clouds.
    pub noise_scale: f32,
    /// Primary raymarch step count.
    pub steps: f32,
    /// Henyey-Greenstein anisotropy (forward scatter / silver lining).
    pub hg_g: f32,
}

impl Default for CloudParams {
    fn default() -> Self {
        Self {
            coverage: 0.5,
            density: 0.05,
            base_alt_m: 2000.0,
            thickness_m: 3000.0,
            wind_speed: 60.0,
            noise_scale: 2500.0,
            steps: 48.0,
            hg_g: 0.4,
        }
    }
}

/// GPU mirror of `clouds.wgsl`'s `CloudParams` (std140, all vec4 → 144 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct CloudParamsGpu {
    right: [f32; 4],
    up: [f32; 4],
    fwd: [f32; 4],
    center_rel: [f32; 4], // xyz = planet centre − camera; w = planet radius
    shell: [f32; 4],      // base_alt, thickness, coverage, density
    wind: [f32; 4],       // wind dir xyz (unit); w = advection speed (m/s)
    march: [f32; 4],      // steps, hg_g, noise_scale, time
    screen: [f32; 4],     // width, height, tan_half_fov, aspect
    misc: [f32; 4],       // debug, proj_a, proj_b, _pad
}

struct CloudFrame {
    frame_ubo: BufferHandle,
    cloud_ubo: BufferHandle,
    set: vk::DescriptorSet,
}

/// Volumetric cloud renderer: one fullscreen raymarch pass.
pub struct Clouds {
    pipeline: PipelineHandle,
    layout: vk::PipelineLayout,
    sampler: vk::Sampler,
    /// 3-index buffer feeding `vertex_index` for the fullscreen triangle.
    idx: BufferHandle,
    frames: Vec<CloudFrame>,
    /// Planet wind source (set once the heightfield loads; static until then).
    wind_src: Option<Arc<dyn HeightField>>,
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
            let set = rhi.allocate_descriptor_set(layout_handle)?;
            rhi.write_uniform_binding(set, 0, frame_ubo)?;
            rhi.write_uniform_binding(set, 1, cloud_ubo)?;
            rhi.write_sampler_binding(set, 3, sampler);
            // Binding 2 (scene depth) is written per-frame in `record`.
            frames.push(CloudFrame { frame_ubo, cloud_ubo, set });
        }

        Ok(Self { pipeline, layout, sampler, idx, frames, wind_src: None, base_radius })
    }

    /// Record the cloud raymarch. Call inside the reopened 1× water instance,
    /// BEFORE the waves (so the opaque sea occludes sky-clouds), sampling the
    /// resolved opaque `scene_depth`. `time` is real elapsed seconds (drift rate).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
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
        let f = &self.frames[fi as usize];
        rhi.write_sampled_image_binding(
            f.set,
            2,
            scene_depth,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        rhi.write_storage_bytes(f.frame_ubo, bytemuck::bytes_of(fu))?;

        // Camera basis for per-pixel ray reconstruction (no inverse matrix needed).
        let fwd = (camera.orientation * Vec3::NEG_Z).normalize();
        let right = (camera.orientation * Vec3::X).normalize();
        let up = (camera.orientation * Vec3::Y).normalize();
        let center_rel = (-camera.position).as_vec3(); // planet centre relative to camera
        let tan_half = (camera.fov_y_radians * 0.5).tan();
        let aspect = extent.0 as f32 / extent.1.max(1) as f32;

        // One global wind vector this frame, sampled below the camera. A spatially
        // varying advection would tear the field over time; this stays coherent and
        // changes smoothly as you fly to a new region (how Decima/UE5 do it).
        let sub = camera.position.normalize();
        let wind = self
            .wind_src
            .as_ref()
            .map(|h| h.wind_at(sub))
            .unwrap_or_default();
        let wdir = Vec3::new(wind.x, wind.y, wind.z);
        let wlen = wdir.length();
        let wnorm = if wlen > 1e-4 { wdir / wlen } else { Vec3::ZERO };

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
            wind: [wnorm.x, wnorm.y, wnorm.z, params.wind_speed * wind.speed],
            march: [params.steps, params.hg_g, params.noise_scale, time],
            screen: [extent.0 as f32, extent.1 as f32, tan_half, aspect],
            misc: [if debug { 1.0 } else { 0.0 }, proj_a, proj_b, 0.0],
        };
        rhi.write_storage_bytes(f.cloud_ubo, bytemuck::bytes_of(&gpu))?;

        rhi.cmd_bind_descriptor_set(fi, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, f.set);
        rhi.cmd_bind_pipeline(fi, vk::PipelineBindPoint::GRAPHICS, self.pipeline)?;
        rhi.bind_index_buffer(fi, self.idx)?;
        rhi.draw_indexed(fi, 3);
        Ok(())
    }

    /// Supply the planet as the wind source once loaded; until then clouds are static.
    pub fn set_wind_source(&mut self, hf: Arc<dyn HeightField>) {
        self.wind_src = Some(hf);
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
        // 9 × vec4 = 144 bytes.
        assert_eq!(std::mem::size_of::<super::CloudParamsGpu>(), 144);
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

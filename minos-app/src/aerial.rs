//! `aerial` — atmospheric scattering as a depth-aware aerial-perspective pass.
//!
//! A fullscreen pass (modeled on [`crate::clouds`]) drawn inside the ocean's
//! refraction split: it reads the resolved opaque scene DEPTH, integrates a
//! closed-form Rayleigh slab in-scatter along each camera→surface ray, and
//! composites it as `inscatter + background·transmittance`. Unlike the atmosphere
//! SHELL it grades by camera→surface distance, so distant sunlit terrain blues out
//! (aerial perspective) and sky pixels get an overhead dome — the two things a
//! shell geometrically cannot do. **TAA-on only** (rides `begin_water_pass`).
//!
//! No worker, no storage buffer, no 3D LUT — the thin atmosphere makes the slab
//! closed form exact (see `aerial.wgsl`). The shell (atmosphere.rs) stays as the
//! TAA-off fallback / optional orbital limb halo.

use bytemuck::{Pod, Zeroable};
use minos_render::{camera::Camera, frame::FrameUniforms};
use minos_rhi::{vk, BindingDesc, BufferHandle, GraphicsPipelineDesc, PipelineHandle, Rhi, RhiError};
use glam::Vec3;

const AERIAL_WGSL: &str = include_str!("aerial.wgsl");

/// GUI-tunable atmospheric-scattering knobs (planet scale, metres / per-metre).
#[derive(Debug, Clone, Copy)]
pub struct AerialParams {
    /// Atmosphere top above sea level (metres) — the outer scattering shell radius
    /// and the sky-chord length.
    pub height: f32,
    /// Rayleigh strength (1/m): the blue-onset distance. `beta_blue·s ≈ 1` at the
    /// distance where the haze reads clearly blue. Scale-dependent on a toy planet.
    pub beta: f32,
    /// Overall in-scatter brightness multiplier.
    pub intensity: f32,
    /// Extra multiplier for the sky-dome branch (sky pixels), so the overhead dome
    /// can be balanced against the terrain haze.
    pub sky_strength: f32,
    /// Vertical density scale height (metres): air thins as `exp(-altitude/falloff)`.
    /// Smaller = a thinner, sharper limb; large (→ `height`) approaches the old slab
    /// (and its white limb halo). Keeps grazing limb rays from saturating to white.
    pub falloff: f32,
}

impl Default for AerialParams {
    fn default() -> Self {
        // Tuned for R≈50 km: a ~10 km optical top with a ~2 km scale height (limb
        // thins to vacuum well before the top), blue onset over ~10 km of path, a
        // modest in-scatter so the LDR composite doesn't blow out. USER-tuned live.
        // sky_strength multiplies ONLY the overhead-dome (sky) pixels (no separate
        // SkyModel pass — this dome IS the sky over the near-black clear). It's a
        // narrow window: too low → dark blue, too high → the blue channel clips and
        // the sky desaturates to grey/white. ~2.0 sits between; β raises depth/hue.
        // All USER-tuned live via the Aerial GUI sliders.
        Self { height: 10_000.0, beta: 1.0e-4, intensity: 0.8, sky_strength: 2.0, falloff: 2_000.0 }
    }
}

/// GPU mirror of `aerial.wgsl`'s `Aerial` uniform (std140, all vec4 → 112 bytes).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct AerialParamsGpu {
    right:      [f32; 4],
    up:         [f32; 4],
    fwd:        [f32; 4],
    center_rel: [f32; 4], // xyz = planet centre − camera; w = planet radius R
    params:     [f32; 4], // height_m, beta, intensity, sky_strength
    screen:     [f32; 4], // width, height, tan_half_fov, aspect
    misc:       [f32; 4], // debug, proj_a, proj_b, _pad
}

struct AerialFrame {
    frame_ubo: BufferHandle,
    aerial_ubo: BufferHandle,
    set: vk::DescriptorSet,
}

/// Atmospheric-scattering renderer: one fullscreen aerial-perspective pass.
pub struct Aerial {
    pipeline: PipelineHandle,
    layout: vk::PipelineLayout,
    sampler: vk::Sampler,
    /// 3-index buffer feeding `vertex_index` for the fullscreen triangle.
    idx: BufferHandle,
    frames: Vec<AerialFrame>,
    base_radius: f64,
}

impl Aerial {
    pub fn new(rhi: &mut Rhi, color_format: vk::Format, base_radius: f64) -> Result<Self, RhiError> {
        let both = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let frag = vk::ShaderStageFlags::FRAGMENT;
        let layout_handle = rhi.create_descriptor_set_layout(&[
            BindingDesc { binding: 0, ty: vk::DescriptorType::UNIFORM_BUFFER, stages: both },
            BindingDesc { binding: 1, ty: vk::DescriptorType::UNIFORM_BUFFER, stages: both },
            BindingDesc { binding: 2, ty: vk::DescriptorType::SAMPLED_IMAGE, stages: frag },
            BindingDesc { binding: 3, ty: vk::DescriptorType::SAMPLER, stages: frag },
        ])?;
        let sampler = rhi.create_sampler()?;

        let shader = rhi.create_shader_module(AERIAL_WGSL)?;
        // Vertex-pulling fullscreen triangle (no vertex buffers). 1× target (the
        // water split), blend → depth-write OFF, depth-test GREATER.
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

        let idx = rhi.create_index_buffer(&[0u32, 1, 2])?;

        let fif = rhi.frames_in_flight();
        let mut frames = Vec::with_capacity(fif);
        for _ in 0..fif {
            let frame_ubo = rhi.create_gpu_buffer(
                std::mem::size_of::<FrameUniforms>() as u64,
                true,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            let aerial_ubo = rhi.create_gpu_buffer(
                std::mem::size_of::<AerialParamsGpu>() as u64,
                true,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            let set = rhi.allocate_descriptor_set(layout_handle)?;
            rhi.write_uniform_binding(set, 0, frame_ubo)?;
            rhi.write_uniform_binding(set, 1, aerial_ubo)?;
            rhi.write_sampler_binding(set, 3, sampler);
            // Binding 2 (scene depth) is written per-frame in `record`.
            frames.push(AerialFrame { frame_ubo, aerial_ubo, set });
        }

        Ok(Self { pipeline, layout, sampler, idx, frames, base_radius })
    }

    /// Record the aerial-perspective pass. Call inside the reopened 1× water
    /// instance, AFTER the ocean and BEFORE clouds, sampling the resolved opaque
    /// `scene_depth` (the pre-ocean depth — over water `s` is the seabed distance,
    /// so open ocean also blues from orbit, which is correct).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        rhi: &mut Rhi,
        fi: u32,
        fu: &FrameUniforms,
        camera: &Camera,
        scene_depth: vk::ImageView,
        extent: (u32, u32),
        params: AerialParams,
        debug: bool,
    ) -> Result<(), RhiError> {
        let f = &self.frames[fi as usize];
        rhi.write_sampled_image_binding(f.set, 2, scene_depth, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        rhi.write_storage_bytes(f.frame_ubo, bytemuck::bytes_of(fu))?;

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

        let gpu = AerialParamsGpu {
            right: [right.x, right.y, right.z, 0.0],
            up: [up.x, up.y, up.z, 0.0],
            fwd: [fwd.x, fwd.y, fwd.z, 0.0],
            center_rel: [center_rel.x, center_rel.y, center_rel.z, self.base_radius as f32],
            params: [params.height, params.beta, params.intensity, params.sky_strength],
            screen: [extent.0 as f32, extent.1 as f32, tan_half, aspect],
            misc: [if debug { 1.0 } else { 0.0 }, proj_a, proj_b, params.falloff],
        };
        rhi.write_storage_bytes(f.aerial_ubo, bytemuck::bytes_of(&gpu))?;

        rhi.cmd_bind_descriptor_set(fi, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, f.set);
        rhi.cmd_bind_pipeline(fi, vk::PipelineBindPoint::GRAPHICS, self.pipeline)?;
        rhi.bind_index_buffer(fi, self.idx)?;
        rhi.draw_indexed(fi, 3);
        Ok(())
    }

    /// Free the caller-owned sampler before RHI teardown.
    pub fn destroy(&self, rhi: &Rhi) {
        rhi.destroy_sampler(self.sampler);
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn aerial_params_gpu_is_std140_sized() {
        // 7 × vec4 = 112 bytes.
        assert_eq!(std::mem::size_of::<super::AerialParamsGpu>(), 112);
    }

    #[test]
    fn aerial_wgsl_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(super::AERIAL_WGSL)
            .unwrap_or_else(|e| panic!("aerial.wgsl failed to parse: {e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("aerial.wgsl failed to validate: {e:?}"));
    }
}

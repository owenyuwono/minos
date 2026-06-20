//! `ocean` — the planet's water surface.
//!
//! Two layers:
//!  - [`Ocean`] — the smooth translucent sea-level shell (full sphere), used for
//!    the orbit / far view. Alpha-blended, depth-test GREATER, depth-write OFF.
//!  - [`WaveSurface`] — a Tessendorf-style spectral FFT ocean (ported from
//!    `poseidon/`) that displaces a camera-anchored tangent-plane grid curved onto
//!    the sphere, for real waves, foam, and (later) refraction up close.
//!
//! The spectral sim runs on the CPU ([`fft`]/[`spectrum`]/[`sim`]) and uploads
//! displacement/normal/foam to a storage buffer the wave shader samples (tiled).
//! ponytail: CPU FFT first (verifiable, no storage-image support needed); GPU
//! compute is the perf upgrade path once this is proven.

pub mod fft;
pub mod sim;
pub mod spectrum;

use bytemuck::{cast_slice, Pod, Zeroable};
use enki_render::{
    frame::FrameUniforms, geometry::placeholder_sphere, material::ChunkPush,
    water_pass::{OCEAN_SURFACE_WGSL, WATER_WGSL},
};
use enki_rhi::{
    vk, BindingDesc, BufferHandle, GraphicsPipelineDesc, PipelineHandle, Rhi, RhiError,
};
use glam::{DVec3, Mat4, Vec3};

use sim::{OceanSim, OceanTexel, WaveParams};
use spectrum::{CascadeParams, Spectrum};

// ── Smooth sea-level shell (far / orbit view) ──────────────────────────────────

/// The ocean shell: one alpha-blended sphere, scaled to the sea level each frame.
pub struct Ocean {
    pipeline: PipelineHandle,
    pos:   BufferHandle,
    nrm:   BufferHandle,
    col:   BufferHandle,
    plate: BufferHandle,
    idx:   BufferHandle,
    count: u32,
    /// Radius the sphere mesh was built at (= planet sea-level datum, metres).
    base_radius: f64,
}

impl Ocean {
    /// Build the ocean pipeline + shell mesh. `base_radius` is the planet's
    /// sea-level datum (normalized height `e = 0`, i.e. `PLANET_RADIUS`).
    pub fn new(
        rhi:          &mut Rhi,
        color_format: vk::Format,
        samples:      vk::SampleCountFlags,
        base_radius:  f64,
    ) -> Result<Self, RhiError> {
        let shader = rhi.create_shader_module(WATER_WGSL)?;
        let pipeline = rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            shader,
            vs_entry:           "vs_main",
            fs_entry:           "fs_main",
            push_constant_size: std::mem::size_of::<ChunkPush>() as u32,
            set0_layout:        rhi.set0_layout(),
            color_format,
            depth_format:       vk::Format::D32_SFLOAT,
            samples,
            blend:              true, // alpha-blend, depth-write OFF
            fill:               true,
        })?;
        rhi.destroy_shader_module(shader);

        let mesh = placeholder_sphere(base_radius as f32, 128);
        let white = vec![[1.0f32, 1.0, 1.0]; mesh.positions.len()];
        // `placeholder_sphere` winds CW-from-outside; reverse so cull-BACK keeps
        // the near (camera-facing) hemisphere.
        let mut indices = mesh.indices.clone();
        for tri in indices.chunks_exact_mut(3) {
            tri.swap(0, 2);
        }
        let pos   = rhi.create_vertex_buffer(cast_slice(&mesh.positions))?;
        let nrm   = rhi.create_vertex_buffer(cast_slice(&mesh.normals))?;
        let col   = rhi.create_vertex_buffer(cast_slice(&white))?;
        let plate = rhi.create_vertex_buffer(cast_slice(&white))?;
        let idx   = rhi.create_index_buffer(&indices)?;

        Ok(Self { pipeline, pos, nrm, col, plate, idx, count: mesh.indices.len() as u32, base_radius })
    }

    /// Draw the shell with its surface at `base_radius + sea_level_m`.
    pub fn record(
        &self,
        rhi:         &mut Rhi,
        fi:          u32,
        fu:          &FrameUniforms,
        camera_pos:  DVec3,
        sea_level_m: f64,
    ) -> Result<(), RhiError> {
        rhi.bind_pipeline(fi, self.pipeline)?;
        rhi.update_frame_uniforms(fi, bytemuck::bytes_of(fu))?;

        let scale = shell_scale(self.base_radius, sea_level_m);
        let model = Mat4::from_scale(Vec3::splat(scale));
        let push  = ChunkPush::camera_relative(DVec3::ZERO, camera_pos, model, 0);

        rhi.bind_vertex_buffers(fi, &[self.pos, self.nrm, self.col, self.plate])?;
        rhi.bind_index_buffer(fi, self.idx)?;
        rhi.push_constants(fi, bytemuck::bytes_of(&push))?;
        rhi.draw_indexed(fi, self.count);
        Ok(())
    }
}

/// Uniform scale that places the shell surface at `base_radius + sea_level_m`.
fn shell_scale(base_radius: f64, sea_level_m: f64) -> f32 {
    ((base_radius + sea_level_m) / base_radius) as f32
}

// ── FFT wave surface (near / detailed view) ────────────────────────────────────

/// Grid tessellation (cells per side) of the camera-anchored wave patch.
const GRID_RES: u32 = 400;
/// Half-extent of the wave patch in metres (covers the near field + horizon at low altitude).
const PATCH_HALF: f32 = 4000.0;
/// FFT resolution (CPU cost ∝ N²·logN per frame).
const FFT_N: usize = 128;

/// GPU mirror of `ocean_surface.wgsl`'s `Ocean` uniform (std140, all vec4 → 192 B).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OceanParamsGpu {
    east:          [f32; 4],
    north:         [f32; 4],
    up:            [f32; 4],
    sub_point_rel: [f32; 4], // xyz + sea_radius
    cfg:           [f32; 4], // length_scale, patch_half, fft_n, fade_start
    deep_color:    [f32; 4],
    scatter_color: [f32; 4],
    foam_color:    [f32; 4],
    sky_horizon:   [f32; 4],
    sky_zenith:    [f32; 4],
    sun_color:     [f32; 4],
    shading:       [f32; 4], // sss, foam_threshold, foam_scale, alpha
    screen:        [f32; 4], // width, height, refract_strength, deep_tint
}

/// Per-frame-in-flight GPU resources for the wave surface.
struct WaveFrame {
    frame_ubo: BufferHandle,
    ocean_ubo: BufferHandle,
    field_buf: BufferHandle,
    set:       vk::DescriptorSet,
}

/// Camera-anchored FFT wave surface.
pub struct WaveSurface {
    pipeline:  PipelineHandle,
    layout:    vk::PipelineLayout,
    grid_pos:  BufferHandle,
    grid_idx:  BufferHandle,
    idx_count: u32,
    sim: OceanSim,
    /// Live-tunable wave params (GUI).
    pub params: WaveParams,
    frames: Vec<WaveFrame>,
    base_radius: f64,
}

impl WaveSurface {
    pub fn new(
        rhi:          &mut Rhi,
        color_format: vk::Format,
        _samples:     vk::SampleCountFlags,
        base_radius:  f64,
    ) -> Result<Self, RhiError> {
        // ── Grid mesh (positions in tangent metres; only .xz read by the shader) ──
        let (positions, indices) = grid_mesh(GRID_RES, PATCH_HALF);
        let grid_pos = rhi.create_vertex_buffer(cast_slice(&positions))?;
        let grid_idx = rhi.create_index_buffer(&indices)?;

        // ── Custom set 0: frame UBO + field storage + ocean UBO + scene refraction ──
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let frag = vk::ShaderStageFlags::FRAGMENT;
        let layout_handle = rhi.create_descriptor_set_layout(&[
            BindingDesc { binding: 0, ty: vk::DescriptorType::UNIFORM_BUFFER, stages },
            BindingDesc { binding: 1, ty: vk::DescriptorType::STORAGE_BUFFER, stages },
            BindingDesc { binding: 2, ty: vk::DescriptorType::UNIFORM_BUFFER, stages },
            BindingDesc { binding: 3, ty: vk::DescriptorType::SAMPLED_IMAGE, stages: frag },
            BindingDesc { binding: 4, ty: vk::DescriptorType::SAMPLER, stages: frag },
        ])?;
        let sampler = rhi.create_sampler()?;

        let shader = rhi.create_shader_module(OCEAN_SURFACE_WGSL)?;
        // The water draws into the TAA 1× `current` target (the refraction split), so
        // it is single-sample and OPAQUE (refraction carries the see-through).
        let pipeline = rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            shader,
            vs_entry:           "vs_main",
            fs_entry:           "fs_main",
            push_constant_size: 0,
            set0_layout:        layout_handle,
            color_format,
            depth_format:       vk::Format::D32_SFLOAT,
            samples:            vk::SampleCountFlags::TYPE_1,
            blend:              false,
            fill:               true,
        })?;
        rhi.destroy_shader_module(shader);
        let layout = rhi.pipeline_layout(pipeline)?;

        // ── Spectral sim (single cascade for now) ────────────────────────────────
        let cascade = CascadeParams {
            n: FFT_N,
            length_scale: 250.0,
            cutoff_low: 1e-4,
            cutoff_high: 9999.0,
            g: 9.81,
            depth: 500.0,
            local: Spectrum {
                scale: 1.0, wind_speed: 16.0, wind_dir_rad: 45f32.to_radians(),
                fetch: 100_000.0, spread_blend: 1.0, swell: 0.2, gamma: 3.3, short_waves_fade: 0.02,
            },
            swell: Spectrum {
                scale: 0.8, wind_speed: 2.0, wind_dir_rad: 70f32.to_radians(),
                fetch: 300_000.0, spread_blend: 1.0, swell: 1.0, gamma: 3.3, short_waves_fade: 0.01,
            },
        };
        let sim = OceanSim::new(&[cascade], 1337);

        // ── Per-frame-in-flight buffers + descriptor sets ────────────────────────
        let fif = rhi.frames_in_flight();
        let field_size = (FFT_N * FFT_N * std::mem::size_of::<OceanTexel>()) as u64;
        let mut frames = Vec::with_capacity(fif);
        for _ in 0..fif {
            let frame_ubo = rhi.create_gpu_buffer(
                std::mem::size_of::<FrameUniforms>() as u64, true, vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            let ocean_ubo = rhi.create_gpu_buffer(
                std::mem::size_of::<OceanParamsGpu>() as u64, true, vk::BufferUsageFlags::UNIFORM_BUFFER,
            )?;
            let field_buf = rhi.create_gpu_buffer(field_size, true, vk::BufferUsageFlags::STORAGE_BUFFER)?;
            let set = rhi.allocate_descriptor_set(layout_handle)?;
            rhi.write_uniform_binding(set, 0, frame_ubo)?;
            rhi.write_storage_binding(set, 1, field_buf)?;
            rhi.write_uniform_binding(set, 2, ocean_ubo)?;
            rhi.write_sampler_binding(set, 4, sampler);
            // Binding 3 (scene color) is written per-frame in `record` (it ping-pongs).
            frames.push(WaveFrame { frame_ubo, ocean_ubo, field_buf, set });
        }

        Ok(Self {
            pipeline, layout, grid_pos, grid_idx, idx_count: indices.len() as u32,
            sim, params: WaveParams::default(), frames, base_radius,
        })
    }

    /// Step the spectral sim + upload + draw the wave patch. Call after all opaque
    /// geometry, inside the open rendering instance.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        rhi:         &mut Rhi,
        fi:          u32,
        fu:          &FrameUniforms,
        camera_pos:  DVec3,
        sea_level_m: f64,
        time:        f32,
        dt:          f32,
        scene_view:  vk::ImageView,
        extent:      (u32, u32),
    ) -> Result<(), RhiError> {
        let f = &self.frames[fi as usize];

        // Point the refraction binding at this frame's resolved opaque scene.
        rhi.write_sampled_image_binding(f.set, 3, scene_view, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);

        // 1. CPU spectral step → upload field.
        let field = self.sim.step(time, dt, &self.params);
        rhi.write_storage_bytes(f.field_buf, cast_slice(field))?;

        // 2. Frame uniforms (view_proj + sun) — copy verbatim.
        rhi.write_storage_bytes(f.frame_ubo, bytemuck::bytes_of(fu))?;

        // 3. Ocean params: tangent frame at the sub-camera point (camera-relative).
        let sea_radius = self.base_radius + sea_level_m;
        let up = camera_pos.normalize();
        let sub_point = up * sea_radius;
        let sub_rel = (sub_point - camera_pos).as_vec3();
        let ref_axis = if up.y.abs() < 0.99 { DVec3::Y } else { DVec3::X };
        let east = ref_axis.cross(up).normalize().as_vec3();
        let north = up.cross(east.as_dvec3()).normalize().as_vec3();
        let up = up.as_vec3();

        let params = OceanParamsGpu {
            east:          [east.x, east.y, east.z, 0.0],
            north:         [north.x, north.y, north.z, 0.0],
            up:            [up.x, up.y, up.z, 0.0],
            sub_point_rel: [sub_rel.x, sub_rel.y, sub_rel.z, sea_radius as f32],
            cfg:           [self.sim.length_scale(), PATCH_HALF, FFT_N as f32, 0.75],
            deep_color:    srgb_lin(0x07, 0x1a, 0x26),
            scatter_color: srgb_lin(0x2e, 0x8f, 0x8f),
            foam_color:    srgb_lin(0xdc, 0xe7, 0xea),
            sky_horizon:   srgb_lin(0x9f, 0xb8, 0xcc),
            sky_zenith:    srgb_lin(0x2a, 0x5b, 0x9c),
            sun_color:     srgb_lin(0xff, 0xf1, 0xdc),
            shading:       [1.0, self.params.foam_threshold, 2.5, 0.92],
            screen:        [extent.0 as f32, extent.1 as f32, 0.05, 0.55],
        };
        rhi.write_storage_bytes(f.ocean_ubo, bytemuck::bytes_of(&params))?;

        // 4. Draw — custom set 0 (no rhi ring), reuse the 4×vec3 layout (grid in all).
        rhi.cmd_bind_pipeline(fi, vk::PipelineBindPoint::GRAPHICS, self.pipeline)?;
        rhi.cmd_bind_descriptor_set(fi, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, f.set);
        rhi.bind_vertex_buffers(fi, &[self.grid_pos, self.grid_pos, self.grid_pos, self.grid_pos])?;
        rhi.bind_index_buffer(fi, self.grid_idx)?;
        rhi.draw_indexed(fi, self.idx_count);
        Ok(())
    }
}

/// A triangulated `(res+1)²` grid in the XZ plane spanning `[-half, +half]`,
/// y = 0. Positions are `[gx, 0, gz]` (the shader reads only `.xz`).
fn grid_mesh(res: u32, half: f32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let n = res + 1;
    let mut positions = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            let gx = (i as f32 / res as f32 * 2.0 - 1.0) * half;
            let gz = (j as f32 / res as f32 * 2.0 - 1.0) * half;
            positions.push([gx, 0.0, gz]);
        }
    }
    let mut indices = Vec::with_capacity((res * res * 6) as usize);
    for j in 0..res {
        for i in 0..res {
            let a = j * n + i;
            let b = a + 1;
            let c = a + n;
            let d = c + 1;
            // Up-facing (toward a camera above) under cull-BACK / front-CCW:
            // east×north = up, so wind a→b→c and b→d→c.
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    (positions, indices)
}

/// sRGB byte → linear `vec4` (alpha 1).
fn srgb_lin(r: u8, g: u8, b: u8) -> [f32; 4] {
    let f = |c: u8| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 { c / 12.92 } else { ((c + 0.055) / 1.055).powf(2.4) }
    };
    [f(r), f(g), f(b), 1.0]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shell_scale_maps_sea_level_to_radius() {
        assert_eq!(shell_scale(50_000.0, 0.0), 1.0);
        assert!((50_000.0 * shell_scale(50_000.0, 500.0) as f64 - 50_500.0).abs() < 1e-3);
        assert!(shell_scale(50_000.0, -1_200.0) < 1.0);
    }

    #[test]
    fn grid_mesh_is_well_formed() {
        let (p, idx) = grid_mesh(4, 100.0);
        assert_eq!(p.len(), 25);
        assert_eq!(idx.len(), 4 * 4 * 6);
        assert_eq!(p[0], [-100.0, 0.0, -100.0]);
        assert_eq!(p[24], [100.0, 0.0, 100.0]);
        assert!(idx.iter().all(|&i| (i as usize) < p.len()));
    }

    #[test]
    fn ocean_params_is_std140_sized() {
        // 13 × vec4 = 208 bytes (std140-friendly).
        assert_eq!(std::mem::size_of::<OceanParamsGpu>(), 208);
    }
}

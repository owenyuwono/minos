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
mod fetch;
mod mesh;
pub mod sim;
pub mod spectrum;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use bytemuck::{cast_slice, Pod, Zeroable};
use enki_planet::height::HeightField;
use enki_render::{
    camera::Camera, frame::FrameUniforms, geometry::sphere_inward, material::ChunkPush,
    water_pass::{OCEAN_SURFACE_WGSL, WATER_WGSL},
};
use enki_rhi::{
    vk, BindingDesc, BufferHandle, GraphicsPipelineDesc, PipelineHandle, Rhi, RhiError,
};
use glam::{DVec3, Mat4, Vec3};

use crate::markers::overlay_pipeline;

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
        let pipeline = overlay_pipeline(
            rhi,
            WATER_WGSL,
            color_format,
            std::mem::size_of::<ChunkPush>() as u32,
            true, // alpha-blend, depth-write OFF
            samples,
        )?;

        // Inward-wound sphere so cull-BACK keeps the near (camera-facing) hemisphere.
        let mesh = sphere_inward(base_radius as f32, 128);
        let white = vec![[1.0f32, 1.0, 1.0]; mesh.positions.len()];
        let pos   = rhi.create_vertex_buffer(cast_slice(&mesh.positions))?;
        let nrm   = rhi.create_vertex_buffer(cast_slice(&mesh.normals))?;
        let col   = rhi.create_vertex_buffer(cast_slice(&white))?;
        let plate = rhi.create_vertex_buffer(cast_slice(&white))?;
        let idx   = rhi.create_index_buffer(&mesh.indices)?;

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

// ── FFT wave surface (projected-grid; near + horizon in one screen-space mesh) ──

/// FFT resolution per cascade (power of two) — matches poseidon's N=256.
/// Affordable at 3 cascades thanks to the rustfft backend.
const FFT_N: usize = 256;
/// Number of FFT cascades (disjoint wavenumber bands → no visible tiling).
const N_CASCADES: usize = 3;
/// Per-cascade world tile sizes (m) — poseidon's [250,17,5] for cm-scale ripple
/// detail; tiling is broken by the domain warp + world-anchored amplitude field.
const LENGTH_SCALES: [f32; 3] = [250.0, 17.0, 5.0];
/// Projected grid: screen-space lattice resolution (cells per side).
const PROJ_RES: u32 = 224;
/// Open-water swell FLOOR for wave intensity: even at zero wind, open ocean keeps
/// this fraction of full energy (so the low-wind equator isn't a dead blue band).
/// Wind adds the rest; `openness` then caps small/enclosed seas. ~0 = pure-wind (old).
const WAVE_BASE: f32 = 0.45;

/// GPU mirror of `ocean_surface.wgsl`'s `Ocean` uniform (std140, all vec4).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct OceanParamsGpu {
    east:          [f32; 4],
    north:         [f32; 4],
    up:            [f32; 4],
    sub_point_rel: [f32; 4], // xyz + sea_radius
    cfg:           [f32; 4], // patch_half, fft_n, fade_start, cascade_count
    scales:        [f32; 4], // per-cascade tile sizes (x,y,z = 3 cascades)
    amp:           [f32; 4], // noise_freq, amp_lo, amp_hi, debug_intensity
    depth_params:  [f32; 4], // near, far, extinction, deep_darkness
    deep_color:    [f32; 4],
    scatter_color: [f32; 4],
    foam_color:    [f32; 4],
    sky_horizon:   [f32; 4],
    sky_zenith:    [f32; 4],
    sun_color:     [f32; 4],
    shading:       [f32; 4], // sss, foam_threshold, foam_scale, alpha
    screen:        [f32; 4], // width, height, refract_strength, deep_tint
    center_rel:    [f32; 4], // xyz = planet centre − camera; w = cell factor
}

/// Per-frame-in-flight GPU resources for the wave surface.
struct WaveFrame {
    frame_ubo: BufferHandle,
    ocean_ubo: BufferHandle,
    field_buf: BufferHandle,
    set:       vk::DescriptorSet,
}

/// Refractive FFT ocean: a screen-space lattice ray-cast onto the sea sphere each
/// frame (projected grid) — fills the whole view, no patch, no shell.
pub struct WaveSurface {
    proj_pipeline:  PipelineHandle,   // projected grid (vs_projected)
    layout:    vk::PipelineLayout,
    /// Scene-texture sampler (caller-owned; freed in [`Self::destroy`]).
    sampler:   vk::Sampler,
    // Projected grid: static NDC lattice (CPU) + index buffer + a per-frame dynamic
    // vertex buffer of sphere-projected positions (rebuilt each frame).
    proj_ndc:  Vec<[f32; 2]>,
    proj_idx:  BufferHandle,
    proj_count: u32,
    proj_dyn:  Vec<BufferHandle>,
    proj_scratch: Vec<[f32; 3]>,
    // Per-vertex (intensity, wind angle), written each frame from the baked cubes →
    // binding 6, read by `vs_projected` via vertex_index.
    wind_dyn:  Vec<BufferHandle>,
    wind_scratch: Vec<[f32; 2]>,
    /// Baked + blurred planet wind (speed, world direction) — sampled per vertex
    /// for smooth intensity (softened shear bands) + wave direction. Empty until load.
    wind_speed_cube: Vec<f32>,
    wind_dir_cube:   Vec<[f32; 3]>,
    /// Baked openness cube (fetch proxy): wind is scaled by this per vertex so
    /// water enclosed by land stays calm. Empty until the heightfield loads.
    openness:  Vec<f32>,
    // ── Background FFT worker (keeps the heavy N=256×3 sim off the render thread) ──
    /// Latest completed field, written by the worker, copied to the GPU each frame.
    latest:        Arc<Mutex<Vec<OceanTexel>>>,
    /// Live params handed to the worker.
    shared_params: Arc<Mutex<WaveParams>>,
    /// Whether the worker should be stepping (false → it idles near the surface only).
    active:        Arc<AtomicBool>,
    /// Worker run flag (cleared on drop to join it).
    running:       Arc<AtomicBool>,
    worker:        Option<JoinHandle<()>>,
    length_scales: Vec<f32>,
    cascade_count: usize,
    /// Live-tunable wave params (GUI); pushed to the worker each frame.
    pub params: WaveParams,
    /// Debug view: heat-map the spatial wave intensity instead of shading water.
    pub debug_intensity: bool,
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
        // ── Projected grid: static NDC lattice + index buffer; the per-frame
        // sphere-projected positions live in a dynamic per-FiF vertex buffer. ──
        let (proj_ndc, proj_i) = mesh::ndc_grid(PROJ_RES);
        let proj_idx = rhi.create_index_buffer(&proj_i)?;
        let proj_count = proj_i.len() as u32;
        let proj_scratch = vec![[0.0f32; 3]; proj_ndc.len()];
        let wind_scratch = vec![[0.0f32; 2]; proj_ndc.len()];

        // ── Custom set 0: frame UBO + field storage + ocean UBO + scene refraction
        // + per-vertex wind (binding 6). ──
        let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
        let frag = vk::ShaderStageFlags::FRAGMENT;
        let vert = vk::ShaderStageFlags::VERTEX;
        let layout_handle = rhi.create_descriptor_set_layout(&[
            BindingDesc { binding: 0, ty: vk::DescriptorType::UNIFORM_BUFFER, stages },
            BindingDesc { binding: 1, ty: vk::DescriptorType::STORAGE_BUFFER, stages },
            BindingDesc { binding: 2, ty: vk::DescriptorType::UNIFORM_BUFFER, stages },
            BindingDesc { binding: 3, ty: vk::DescriptorType::SAMPLED_IMAGE, stages: frag },
            BindingDesc { binding: 4, ty: vk::DescriptorType::SAMPLER, stages: frag },
            BindingDesc { binding: 5, ty: vk::DescriptorType::SAMPLED_IMAGE, stages: frag },
            BindingDesc { binding: 6, ty: vk::DescriptorType::STORAGE_BUFFER, stages: vert },
        ])?;
        let sampler = rhi.create_sampler()?;

        let shader = rhi.create_shader_module(OCEAN_SURFACE_WGSL)?;
        // Draws into the TAA 1× `current` target (the refraction split): single-sample,
        // blend ON so depth-write is OFF; the shader outputs alpha 1 → opaque.
        let proj_pipeline = rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            shader,
            vs_entry:           "vs_projected",
            fs_entry:           "fs_main",
            push_constant_size: 0,
            set0_layout:        layout_handle,
            color_format,
            depth_format:       vk::Format::D32_SFLOAT,
            samples:            vk::SampleCountFlags::TYPE_1,
            blend:              true,
            fill:               true,
        })?;
        rhi.destroy_shader_module(shader);
        let layout = rhi.pipeline_layout(proj_pipeline)?;

        // ── Spectral sim: 3 cascades over disjoint wavenumber bands ──────────────
        // Each cascade carries one band (boundary = 2π/L · factor); overlaid at
        // different world tile sizes they hide the single-tile repetition.
        let boundary_factor = 6.0;
        let boundary = |i: usize| (2.0 * std::f32::consts::PI / LENGTH_SCALES[i]) * boundary_factor;
        let cascades: Vec<CascadeParams> = (0..N_CASCADES)
            .map(|i| {
                let cutoff_low = if i == 0 { 1e-4 } else { boundary(i) };
                let cutoff_high = if i + 1 < N_CASCADES { boundary(i + 1) } else { 9999.0 };
                CascadeParams {
                    n: FFT_N,
                    length_scale: LENGTH_SCALES[i],
                    cutoff_low,
                    cutoff_high,
                    g: 9.81,
                    depth: 500.0,
                    // Bake the dominant direction at +x (0 rad) so the per-vertex wind
                    // angle (applied as the tile-blend rotation) aligns waves to the
                    // local wind with no built-in offset; swell keeps a small cross.
                    local: Spectrum {
                        scale: 1.0, wind_speed: 16.0, wind_dir_rad: 0.0,
                        fetch: 100_000.0, spread_blend: 1.0, swell: 0.2, gamma: 3.3, short_waves_fade: 0.02,
                    },
                    swell: Spectrum {
                        scale: 0.8, wind_speed: 2.0, wind_dir_rad: 0.4,
                        fetch: 300_000.0, spread_blend: 1.0, swell: 1.0, gamma: 3.3, short_waves_fade: 0.01,
                    },
                }
            })
            .collect();
        let sim = OceanSim::new(&cascades, 1337);
        let length_scales = sim.length_scales();
        let cascade_count = sim.cascade_count();

        // ── Background sim worker ────────────────────────────────────────────────
        // N=256 × 3 cascades is too heavy for the render thread (esp. debug builds),
        // so the FFT runs here and double-buffers its result into `latest`.
        let latest = Arc::new(Mutex::new(vec![OceanTexel::zeroed(); FFT_N * FFT_N * N_CASCADES]));
        let shared_params = Arc::new(Mutex::new(WaveParams::default()));
        let active = Arc::new(AtomicBool::new(false));
        let running = Arc::new(AtomicBool::new(true));
        let worker = {
            let (latest, shared_params, active, running) =
                (Arc::clone(&latest), Arc::clone(&shared_params), Arc::clone(&active), Arc::clone(&running));
            let mut sim = sim;
            std::thread::Builder::new()
                .name("ocean-fft".into())
                .spawn(move || {
                    let start = Instant::now();
                    let mut last = start;
                    while running.load(Ordering::Relaxed) {
                        if !active.load(Ordering::Relaxed) {
                            std::thread::sleep(std::time::Duration::from_millis(40));
                            last = Instant::now();
                            continue;
                        }
                        let now = Instant::now();
                        let t = now.duration_since(start).as_secs_f32();
                        let dt = now.duration_since(last).as_secs_f32().min(0.1);
                        last = now;
                        let wp = *shared_params.lock().unwrap();
                        let field = sim.step(t, dt, &wp);
                        if let Ok(mut slot) = latest.lock() {
                            slot.copy_from_slice(field);
                        }
                    }
                })
                .expect("spawn ocean-fft worker")
        };

        // ── Per-frame-in-flight buffers + descriptor sets ────────────────────────
        let fif = rhi.frames_in_flight();
        let field_size = (FFT_N * FFT_N * N_CASCADES * std::mem::size_of::<OceanTexel>()) as u64;
        let proj_dyn_size = (proj_ndc.len() * std::mem::size_of::<[f32; 3]>()) as u64;
        let wind_dyn_size = (proj_ndc.len() * std::mem::size_of::<[f32; 2]>()) as u64;
        let mut frames = Vec::with_capacity(fif);
        let mut proj_dyn = Vec::with_capacity(fif);
        let mut wind_dyn = Vec::with_capacity(fif);
        for _ in 0..fif {
            // host-visible vertex/storage buffers rewritten per frame (projected
            // positions + per-vertex wind).
            proj_dyn.push(rhi.create_gpu_buffer(proj_dyn_size, true, vk::BufferUsageFlags::VERTEX_BUFFER)?);
            let wind_buf = rhi.create_gpu_buffer(wind_dyn_size, true, vk::BufferUsageFlags::STORAGE_BUFFER)?;
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
            rhi.write_storage_binding(set, 6, wind_buf)?;
            // Binding 3 (scene color) is written per-frame in `record` (it ping-pongs).
            frames.push(WaveFrame { frame_ubo, ocean_ubo, field_buf, set });
            wind_dyn.push(wind_buf);
        }

        Ok(Self {
            proj_pipeline, layout, sampler,
            proj_ndc, proj_idx, proj_count, proj_dyn, proj_scratch,
            wind_dyn, wind_scratch, openness: Vec::new(),
            wind_speed_cube: Vec::new(), wind_dir_cube: Vec::new(),
            latest, shared_params, active, running, worker: Some(worker),
            length_scales, cascade_count,
            params: WaveParams::default(), debug_intensity: false, frames, base_radius,
        })
    }

    /// Draw the refractive projected-grid ocean. Call after `begin_water_pass`,
    /// inside the reopened 1× instance, sampling the resolved opaque `scene_view` +
    /// `scene_depth`. `draw_waves` gates the FFT worker (waves only matter near the
    /// surface); the projected grid itself draws at every altitude.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        rhi:          &mut Rhi,
        fi:           u32,
        fu:           &FrameUniforms,
        camera:       &Camera,
        sea_level_m:  f64,
        scene_view:   vk::ImageView,
        scene_depth:  vk::ImageView,
        extent:       (u32, u32),
        draw_waves:   bool,
    ) -> Result<(), RhiError> {
        let camera_pos = camera.position;

        // Drive the background worker: hand it the live params + whether to step.
        *self.shared_params.lock().unwrap() = self.params;
        self.active.store(draw_waves, Ordering::Relaxed);

        let f = &self.frames[fi as usize];
        let read = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

        // Point the refraction bindings at this frame's resolved opaque scene.
        rhi.write_sampled_image_binding(f.set, 3, scene_view, read);
        rhi.write_sampled_image_binding(f.set, 5, scene_depth, read);

        // Upload the worker's latest field (only when the waves are drawn).
        if draw_waves {
            let slot = self.latest.lock().unwrap();
            rhi.write_storage_bytes(f.field_buf, cast_slice(&slot))?;
        }
        rhi.write_storage_bytes(f.frame_ubo, bytemuck::bytes_of(fu))?;

        // Tangent frame at the sub-camera point (FFT sampling) + projection extras:
        // planet centre relative to camera + a cell-size factor (cell ≈
        // distance·factor, since the projected grid is ~screen-uniform).
        let sea_radius = self.base_radius + sea_level_m;
        let up = camera_pos.normalize();
        let sub_rel = (up * sea_radius - camera_pos).as_vec3();
        let ref_axis = if up.y.abs() < 0.99 { DVec3::Y } else { DVec3::X };
        let east = ref_axis.cross(up).normalize().as_vec3();
        let north = up.cross(east.as_dvec3()).normalize().as_vec3();
        let up = up.as_vec3();
        let center_rel = (-camera_pos).as_vec3();
        let tan_half = (camera.fov_y_radians * 0.5).tan();
        let cell_factor = 2.0 * tan_half / PROJ_RES as f32;

        let s = &self.length_scales;
        let params = OceanParamsGpu {
            east:          [east.x, east.y, east.z, 0.0],
            north:         [north.x, north.y, north.z, 0.0],
            up:            [up.x, up.y, up.z, 0.0],
            sub_point_rel: [sub_rel.x, sub_rel.y, sub_rel.z, sea_radius as f32],
            cfg:           [0.0, FFT_N as f32, 0.0, self.cascade_count as f32],
            scales:        [s[0], s.get(1).copied().unwrap_or(0.0), s.get(2).copied().unwrap_or(0.0), 0.0],
            amp:           [9.0, 0.25, 1.6, if self.debug_intensity { 1.0 } else { 0.0 }],
            // near, far (globe camera), extinction (1/m), clarity (seabed visibility).
            depth_params:  [0.5, 750_000.0, 0.003, 0.6],
            deep_color:    srgb_lin(0x12, 0x44, 0x66), // deep ocean blue (not black)
            scatter_color: srgb_lin(0x36, 0xa6, 0x9c), // shallow teal
            foam_color:    srgb_lin(0xdc, 0xe7, 0xea),
            sky_horizon:   srgb_lin(0x9f, 0xb8, 0xcc),
            sky_zenith:    srgb_lin(0x2a, 0x5b, 0x9c),
            sun_color:     srgb_lin(0xff, 0xf1, 0xdc),
            shading:       [1.0, self.params.foam_threshold, 2.5, 0.92],
            screen:        [extent.0 as f32, extent.1 as f32, 0.05, 0.55],
            center_rel:    [center_rel.x, center_rel.y, center_rel.z, cell_factor],
        };
        rhi.write_storage_bytes(f.ocean_ubo, bytemuck::bytes_of(&params))?;

        // Project the NDC lattice onto the sea sphere (CPU f64) + sample planet wind
        // at each vertex (drives wave height/intensity), upload, draw.
        let fwd = (camera.orientation * Vec3::NEG_Z).as_dvec3();
        let rgt = (camera.orientation * Vec3::X).as_dvec3();
        let upc = (camera.orientation * Vec3::Y).as_dvec3();
        let tan_half = tan_half as f64;
        let aspect = extent.0 as f64 / extent.1.max(1) as f64;
        let center = -camera_pos;
        {
            let ndc = &self.proj_ndc;
            let scratch = &mut self.proj_scratch;
            let wscratch = &mut self.wind_scratch;
            let openness = &self.openness;
            let wspeed = &self.wind_speed_cube;
            let wvec = &self.wind_dir_cube;
            let east_d = east.as_dvec3();
            let north_d = north.as_dvec3();
            for (i, p) in ndc.iter().enumerate() {
                let dir = fwd
                    + rgt * (p[0] as f64 * tan_half * aspect)
                    + upc * (p[1] as f64 * tan_half);
                let hit = mesh::project_to_sphere(dir, center, sea_radius, 0.5); // cam-relative
                let wdir = (hit + camera_pos).normalize(); // world surface direction
                // Planet wind (baked + blurred → smooth, gradual bands): speed →
                // intensity, scaled by openness (calm when land-enclosed); direction
                // → wave-propagation angle in the (east, north) tangent frame so waves
                // travel WITH the wind. Open water keeps a swell FLOOR (`WAVE_BASE`) so
                // the low-wind equator stays energetic, not a hard blue band.
                let open = if openness.is_empty() { 1.0 } else { fetch::sample(openness, fetch::OPEN_RES, wdir) };
                let (intensity, angle) = if wspeed.is_empty() {
                    (0.5 * open, 0.0)
                } else {
                    let speed = fetch::sample(wspeed, fetch::OPEN_RES, wdir);
                    let v = fetch::sample_vec3(wvec, fetch::OPEN_RES, wdir);
                    let wv = DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64);
                    let ang = wv.dot(north_d).atan2(wv.dot(east_d)) as f32;
                    ((WAVE_BASE + (1.0 - WAVE_BASE) * speed) * open, ang)
                };
                wscratch[i] = [intensity, angle];
                let h = hit.as_vec3();
                scratch[i] = [h.x, h.y, h.z];
            }
        }
        rhi.write_storage_bytes(self.proj_dyn[fi as usize], cast_slice(&self.proj_scratch))?;
        rhi.write_storage_bytes(self.wind_dyn[fi as usize], cast_slice(&self.wind_scratch))?;
        let pd = self.proj_dyn[fi as usize];
        rhi.cmd_bind_descriptor_set(fi, vk::PipelineBindPoint::GRAPHICS, self.layout, 0, f.set);
        rhi.cmd_bind_pipeline(fi, vk::PipelineBindPoint::GRAPHICS, self.proj_pipeline)?;
        rhi.bind_vertex_buffers(fi, &[pd, pd, pd, pd])?;
        rhi.bind_index_buffer(fi, self.proj_idx)?;
        rhi.draw_indexed(fi, self.proj_count);
        Ok(())
    }

    /// Supply the planet as the wind source once it has finished loading; until
    /// then waves use a neutral wind. Drives per-vertex wave height + intensity,
    /// and bakes the openness (fetch) cube so land-enclosed water stays calm.
    pub fn set_wind_source(&mut self, hf: Arc<dyn HeightField>) {
        self.openness = fetch::bake_openness(hf.as_ref(), 0.0); // bake at the e=0 sea datum
        let (speed, dir) = fetch::bake_wind(hf.as_ref());
        self.wind_speed_cube = speed;
        self.wind_dir_cube = dir;
    }

    /// Free the caller-owned sampler. Call once before RHI teardown (buffers,
    /// pipelines, and descriptor layouts are RHI-tracked and freed on its drop).
    pub fn destroy(&self, rhi: &Rhi) {
        rhi.destroy_sampler(self.sampler);
    }
}

impl Drop for WaveSurface {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
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
    fn ocean_params_is_std140_sized() {
        // 17 × vec4 = 272 bytes (std140-friendly).
        assert_eq!(std::mem::size_of::<OceanParamsGpu>(), 272);
    }
}

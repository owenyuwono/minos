//! `wind` — the wind-streakline overlay.
//!
//! Particles ride the planet's baked wind field ([`HeightField::wind_at`]) on a
//! cap around the sub-camera point, leaving fading ribbon trails — a "living"
//! flow viz (Tier B) with a cheap time-varying gust, no fluid solver. The
//! particle math is pure CPU in [`sim`] (headless, unit-tested); this file owns
//! the GPU glue (pipeline + per-frame dynamic vertex buffers), modeled on the
//! ocean shell: standard `set0` (FrameUniforms) + `ChunkPush`, no custom set.

mod sim;

pub use sim::WindParams;

use std::sync::Arc;
use std::time::Instant;

use bytemuck::cast_slice;
use minos_planet::climate::WindSample;
use minos_planet::height::HeightField;
use minos_render::{frame::FrameUniforms, material::ChunkPush};
use minos_rhi::{vk, BufferHandle, PipelineHandle, Rhi, RhiError};
use glam::DVec3;

use crate::markers::{draw_ribbon, overlay_pipeline};
use sim::{build_indices, WindSim, VERT_COUNT};

const WIND_WGSL: &str = include_str!("wind.wgsl");

/// Wind streakline overlay: a fixed pool of trail ribbons drawn over the planet.
pub struct WindOverlay {
    /// Pipelines for the MSAA opaque pass and the 1× water-split instance. Wind is
    /// drawn AFTER the ocean (it's the top layer), so the sample count depends on
    /// whether the water pass split the frame into a 1× instance.
    pipeline_msaa: PipelineHandle,
    pipeline_1x: PipelineHandle,
    /// Per-frame-in-flight dynamic vertex buffers (positions, then colors).
    pos_dyn: Vec<BufferHandle>,
    col_dyn: Vec<BufferHandle>,
    idx: BufferHandle,
    idx_count: u32,
    sim: WindSim,
    pos_scratch: Vec<[f32; 3]>,
    col_scratch: Vec<[f32; 3]>,
    /// Wind source (the planet); `None` until the heightfield finishes loading.
    src: Option<Arc<dyn HeightField>>,
    /// Planet sea-level datum radius (metres).
    base_radius: f64,
    start: Instant,
    last: Instant,
    /// GUI-tunable params (speed / width / altitude / gust / intensity).
    pub params: WindParams,
}

impl WindOverlay {
    pub fn new(
        rhi: &mut Rhi,
        color_format: vk::Format,
        samples: vk::SampleCountFlags,
        base_radius: f64,
    ) -> Result<Self, RhiError> {
        // Two pipelines: the MSAA opaque pass + the 1× water-split instance (wind is
        // drawn after the ocean, so the sample count depends on the split). Both are
        // straight-alpha, depth-write OFF.
        let push = std::mem::size_of::<ChunkPush>() as u32;
        let pipeline_msaa = overlay_pipeline(rhi, WIND_WGSL, color_format, push, true, samples)?;
        let pipeline_1x =
            overlay_pipeline(rhi, WIND_WGSL, color_format, push, true, vk::SampleCountFlags::TYPE_1)?;

        let indices = build_indices();
        let idx_count = indices.len() as u32;
        let idx = rhi.create_index_buffer(&indices)?;

        let fif = rhi.frames_in_flight();
        let vbuf_size = (VERT_COUNT * std::mem::size_of::<[f32; 3]>()) as u64;
        let mut pos_dyn = Vec::with_capacity(fif);
        let mut col_dyn = Vec::with_capacity(fif);
        for _ in 0..fif {
            pos_dyn.push(rhi.create_gpu_buffer(vbuf_size, true, vk::BufferUsageFlags::VERTEX_BUFFER)?);
            col_dyn.push(rhi.create_gpu_buffer(vbuf_size, true, vk::BufferUsageFlags::VERTEX_BUFFER)?);
        }

        Ok(Self {
            pipeline_msaa,
            pipeline_1x,
            pos_dyn,
            col_dyn,
            idx,
            idx_count,
            sim: WindSim::new(),
            pos_scratch: Vec::with_capacity(VERT_COUNT),
            col_scratch: Vec::with_capacity(VERT_COUNT),
            src: None,
            base_radius,
            start: Instant::now(),
            last: Instant::now(),
            params: WindParams::default(),
        })
    }

    /// Supply the planet as the wind source once it has finished loading; until
    /// then the field is neutral (nothing flows).
    pub fn set_wind_source(&mut self, hf: Arc<dyn HeightField>) {
        self.src = Some(hf);
    }

    /// Advance the particles and draw their ribbons. Call AFTER the ocean (wind is
    /// the top atmosphere layer, so the opaque sea can't paint over it). `one_x`
    /// selects the pipeline: true inside the water-split 1× instance, false in the
    /// MSAA pass.
    pub fn record(
        &mut self,
        rhi: &mut Rhi,
        fi: u32,
        fu: &FrameUniforms,
        camera_pos: DVec3,
        sea_level_m: f64,
        one_x: bool,
    ) -> Result<(), RhiError> {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f32();
        self.last = now;
        let time = (now - self.start).as_secs_f32();

        let sea_radius = self.base_radius + sea_level_m;
        let radius = sea_radius + self.params.altitude as f64;
        let nadir = camera_pos.normalize_or(DVec3::Y);

        // Adaptive seed cap: the visible-cap half-angle from this altitude, so the
        // streaks spread to fill the disk far out and stay dense up close.
        let alt = (camera_pos.length() - sea_radius).max(1.0);
        let horizon = (sea_radius / (sea_radius + alt)).clamp(0.0, 1.0).acos();
        let mut params = self.params; // Copy — disjoint from &mut self.sim below
        params.cap = (horizon as f32 * 0.9).clamp(0.08, 1.4);

        // Step the sim (disjoint field borrows: &mut self.sim, &self.src).
        let src = &self.src;
        self.sim.step(dt, time, nadir, &params, |d| {
            src.as_ref().map_or(WindSample::default(), |h| h.wind_at(d))
        });
        self.sim
            .build_ribbons(camera_pos, radius, &params, &mut self.pos_scratch, &mut self.col_scratch);

        let fi_u = fi as usize;
        rhi.write_storage_bytes(self.pos_dyn[fi_u], cast_slice(&self.pos_scratch))?;
        rhi.write_storage_bytes(self.col_dyn[fi_u], cast_slice(&self.col_scratch))?;

        let pipeline = if one_x { self.pipeline_1x } else { self.pipeline_msaa };
        let p = self.pos_dyn[fi_u];
        let c = self.col_dyn[fi_u];
        // 4×vec3 layout: pos / (normal=pos, ignored) / color / (plate=pos, ignored).
        draw_ribbon(rhi, fi, fu, camera_pos, pipeline, p, p, c, self.idx, self.idx_count)
    }
}

#[cfg(test)]
mod shader_tests {
    use super::WIND_WGSL;

    #[test]
    fn wind_wgsl_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(WIND_WGSL)
            .unwrap_or_else(|e| panic!("wind.wgsl failed to parse: {e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("wind.wgsl failed to validate: {e:?}"));
    }
}

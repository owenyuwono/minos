//! `stress` — M2 streaming stress harness.
//!
//! Activated by `--stress-streaming` CLI argument.  Each frame:
//!   1. Generates hundreds of synthetic meshes of varying sizes and tries to
//!      upload them via `Rhi::streaming_upload` (driving the budget back-pressure).
//!   2. Maintains a rolling working set of `StreamedMesh` handles.
//!   3. Retires old meshes (calling `Rhi::streaming_retire`) so the graveyard
//!      logic is exercised every frame.
//!   4. Renders the currently-live meshes with the terrain pipeline.
//!
//! # How to run
//! ```
//! # Windows (PowerShell / Command Prompt):
//! cargo run --bin enki-app -- --stress-streaming
//!
//! # Or set the environment variable instead of the flag:
//! $env:ENKI_STRESS_STREAMING=1; cargo run --bin enki-app
//! ```
//!
//! Let it run for at least 60 seconds and watch for:
//!   - Validation-layer errors (device_lost / access violation → use-after-free).
//!   - GPU memory growing without bound (→ graveyard not draining).
//!   - Frame stalls / hitches (→ budget back-pressure should prevent this).
//!
//! A clean soak means: zero validation errors, stable RSS over time, smooth 60 Hz.

use std::collections::VecDeque;
use std::time::Instant;

use enki_render::{
    frame::FrameUniforms,
    geometry::ChunkMeshArrays,
    lights::Lights,
    material::ChunkPush,
    terrain_pass::TERRAIN_WGSL,
};
use enki_rhi::{vk, GraphicsPipelineDesc, PipelineHandle, Rhi, RhiError, StreamedMesh};
use glam::Mat4;

// ── Synthetic mesh generation ─────────────────────────────────────────────────

/// Generate a synthetic mesh of `n` vertices arranged in a flat grid.
///
/// Produces valid, non-degenerate geometry without pulling in the sphere
/// tessellator — keeps the stress harness independent of render geometry.
fn synthetic_mesh(vertex_count: usize, seed: u64) -> ChunkMeshArrays {
    assert!(vertex_count >= 3, "need at least 3 vertices for one triangle");

    // Simple LCG to produce varied per-vertex data without std::rand.
    let mut rng = seed;
    let mut lcg = move || -> f32 {
        rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (rng >> 33) as f32 / u32::MAX as f32
    };

    let positions: Vec<[f32; 3]> = (0..vertex_count)
        .map(|_| [lcg() * 2.0 - 1.0, lcg() * 2.0 - 1.0, lcg() * 2.0 - 1.0])
        .collect();

    // Flat normals pointing up — just valid data, not physically correct.
    let normals: Vec<[f32; 3]> = (0..vertex_count).map(|_| [0.0, 1.0, 0.0]).collect();

    let colors: Vec<[f32; 3]> = (0..vertex_count)
        .map(|_| [lcg(), lcg(), lcg()])
        .collect();

    // Triangle fan from vertex 0.  Produces vertex_count - 2 triangles.
    let indices: Vec<u32> = (1..vertex_count as u32 - 1)
        .flat_map(|i| [0u32, i, i + 1])
        .collect();

    ChunkMeshArrays {
        positions,
        normals,
        colors,
        plate_colors: None,
        indices,
        origin: glam::DVec3::ZERO,
    }
}

// ── StressHarness ─────────────────────────────────────────────────────────────

/// Maximum number of live streamed meshes in the working set at any time.
const WORKING_SET_SIZE: usize = 32;

/// Number of upload attempts per frame (most will be back-pressured after the
/// budget is exhausted, which is exactly what we want to stress).
const UPLOADS_PER_FRAME_ATTEMPT: usize = 256;

/// Vertex counts to cycle through — varying sizes stress both small and large uploads.
const VERTEX_COUNTS: &[usize] = &[3, 16, 64, 256, 512, 1024];

/// Every this many stress frames, draw a mesh and then retire it in the SAME frame.
///
/// This exercises the B1 fix: with the old `retire_frame = frame_counter` the GPU
/// would free the buffer one frame too early (→ VK_ERROR_DEVICE_LOST on the Windows
/// soak).  With `retire_frame = frame_counter + 1` the buffer lives until the frame's
/// submission completes.  Setting this to 7 (a prime) ensures coverage without being
/// too frequent (which would drain the working set quickly).
const SAME_FRAME_RETIRE_INTERVAL: u64 = 7;

/// Live entry: mesh handle + the frame it was uploaded on (for retire timing).
struct LiveMesh {
    mesh: StreamedMesh,
    /// Monotonic frame counter at upload time (for bookkeeping / logging).
    _uploaded_frame: u64,
}

/// The stress harness state.
pub struct StressHarness {
    pipeline: PipelineHandle,
    #[allow(dead_code)]
    lights: Lights,

    /// Rolling working set of live meshes.
    live: VecDeque<LiveMesh>,

    /// Monotonically increasing seed for `synthetic_mesh`.
    next_seed: u64,
    /// Frame counter (per-stress, not the RHI counter).
    frame: u64,

    start_time: Instant,
    last_log: Instant,

    /// Cumulative statistics.
    total_upload_attempts: u64,
    total_uploads_ok: u64,
    total_backpressure: u64,
    total_retired: u64,
}

impl StressHarness {
    /// Build the stress harness: compile one terrain pipeline and pre-fill the
    /// working set with a few initial meshes (they'll be retired next frame if
    /// the working set is full).
    pub fn new(
        rhi: &mut Rhi,
        color_format: vk::Format,
        samples: vk::SampleCountFlags,
    ) -> Result<Self, RhiError> {
        let shader = rhi.create_shader_module(TERRAIN_WGSL)?;
        let pipeline = rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            shader,
            vs_entry:           "vs_main",
            fs_entry:           "fs_main",
            push_constant_size: std::mem::size_of::<ChunkPush>() as u32,
            set0_layout:        rhi.set0_layout(),
            color_format,
            depth_format:       vk::Format::D32_SFLOAT,
            samples,
            blend:              false,
            fill:               true,
        })?;
        rhi.destroy_shader_module(shader);

        let now = Instant::now();
        Ok(Self {
            pipeline,
            lights: Lights::demiurge_default(),
            live: VecDeque::with_capacity(WORKING_SET_SIZE + 1),
            next_seed: 0,
            frame: 0,
            start_time: now,
            last_log: now,
            total_upload_attempts: 0,
            total_uploads_ok: 0,
            total_backpressure: 0,
            total_retired: 0,
        })
    }

    /// Upload phase: record staging copies for this frame.
    ///
    /// Must be called AFTER `begin_frame` and BEFORE `begin_rendering`.
    /// Records `vkCmdCopyBuffer` + copy→draw barriers into the command buffer
    /// while the dynamic-rendering instance is not yet open (required by Vulkan).
    ///
    /// Uploads as many synthetic meshes as the per-frame budget allows;
    /// evicts the oldest mesh from the working set when `WORKING_SET_SIZE` is exceeded.
    pub fn upload_phase(&mut self, rhi: &mut Rhi, fi: u32) -> Result<(), RhiError> {
        let rhi_frame = rhi.gpu_completed_frame(); // used for logging only

        for i in 0..UPLOADS_PER_FRAME_ATTEMPT {
            let vertex_count = VERTEX_COUNTS[(i + self.frame as usize) % VERTEX_COUNTS.len()];
            let mesh_data = synthetic_mesh(vertex_count, self.next_seed);
            self.next_seed += 1;
            self.total_upload_attempts += 1;

            match rhi.streaming_upload(
                fi,
                &mesh_data.positions,
                &mesh_data.normals,
                &mesh_data.colors,
                mesh_data.plate_colors.as_deref(),
                &mesh_data.indices,
            )? {
                Some(mesh) => {
                    self.total_uploads_ok += 1;
                    self.live.push_back(LiveMesh {
                        mesh,
                        _uploaded_frame: rhi_frame,
                    });

                    // Evict oldest mesh when working set is full.
                    if self.live.len() > WORKING_SET_SIZE {
                        if let Some(old) = self.live.pop_front() {
                            rhi.streaming_retire(old.mesh);
                            self.total_retired += 1;
                        }
                    }
                }
                None => {
                    // Budget or ring exhausted — back-pressure working as intended.
                    self.total_backpressure += 1;
                }
            }
        }

        Ok(())
    }

    /// Draw phase: issue draw commands for the live working set.
    ///
    /// Must be called AFTER `begin_rendering` and BEFORE `end_frame`.
    ///
    /// # Draw-then-retire same-frame path (exercises B1 fix)
    ///
    /// Every `SAME_FRAME_RETIRE_INTERVAL` frames this method draws the first
    /// mesh in the live set and then immediately retires it in the same frame.
    /// This is the exact pattern that triggered the B1 use-after-free: the mesh
    /// is drawn in frame N and retired (graveyard entry recorded) in the same
    /// recording session.  With `retire_frame = frame_counter + 1` the GPU will
    /// not free the buffer until after frame N's submission completes, which is
    /// correct.  With the old `frame_counter` the buffer could be freed one frame
    /// too early → `VK_ERROR_DEVICE_LOST`.
    pub fn draw_phase(
        &mut self,
        rhi: &mut Rhi,
        fi: u32,
        fu: &FrameUniforms,
    ) -> Result<(), RhiError> {
        rhi.set_viewport_scissor_full(fi);
        rhi.bind_pipeline(fi, self.pipeline)?;
        rhi.update_frame_uniforms(fi, bytemuck::bytes_of(fu))?;

        if self.live.is_empty() {
            // Nothing to draw this frame (all back-pressured or just started).
            self.frame += 1;
            return Ok(());
        }

        // Draw each live mesh.
        for live in &self.live {
            let mesh = &live.mesh;
            let push = ChunkPush::new(Mat4::IDENTITY, 0);
            rhi.bind_vertex_buffers(fi, &[mesh.pos, mesh.nrm, mesh.col, mesh.plate])?;
            rhi.bind_index_buffer(fi, mesh.idx)?;
            rhi.push_constants(fi, bytemuck::bytes_of(&push))?;
            rhi.draw_indexed(fi, mesh.index_count);
        }

        // ── Same-frame draw-then-retire (N3: exercises B1 fix) ────────────────
        //
        // Every SAME_FRAME_RETIRE_INTERVAL frames: draw the most-recently-uploaded
        // mesh and retire it in the same frame recording session.  The mesh was
        // drawn above; retiring it here exercises the case where retire_frame must
        // be frame_counter + 1 (not frame_counter) so the GPU cannot free the
        // buffer before the current submission completes.
        //
        // Pre-fix this would crash with VK_ERROR_DEVICE_LOST on Windows soak because
        // collect_garbage would free the buffer at timeline >= frame_counter while
        // the GPU was still reading it in the same frame.
        if self.frame.is_multiple_of(SAME_FRAME_RETIRE_INTERVAL) {
            if let Some(newest) = self.live.pop_back() {
                log::debug!(
                    "[stress] same-frame retire: frame={} live_remaining={}",
                    self.frame,
                    self.live.len()
                );
                rhi.streaming_retire(newest.mesh);
                self.total_retired += 1;
            }
        }

        // ── Periodic logging ──────────────────────────────────────────────────
        let now = Instant::now();
        if now.duration_since(self.last_log).as_secs_f32() >= 5.0 {
            let elapsed = now.duration_since(self.start_time).as_secs_f32();
            let bp_rate = if self.total_upload_attempts > 0 {
                self.total_backpressure as f32 / self.total_upload_attempts as f32 * 100.0
            } else {
                0.0
            };
            log::info!(
                "[stress] t={:.1}s  frame={}  live={}  uploads={}  backpressure={} ({:.1}%)  retired={}  gpu_done={}",
                elapsed,
                self.frame,
                self.live.len(),
                self.total_uploads_ok,
                self.total_backpressure,
                bp_rate,
                self.total_retired,
                rhi.gpu_completed_frame(),
            );
            self.last_log = now;
        }

        self.frame += 1;
        Ok(())
    }

    /// Returns how long the stress harness has been running.
    #[allow(dead_code)]
    pub fn elapsed_secs(&self) -> f32 {
        Instant::now().duration_since(self.start_time).as_secs_f32()
    }
}

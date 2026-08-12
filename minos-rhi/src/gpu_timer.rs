//! `gpu_timer` — whole-frame GPU time via Vulkan timestamp queries.
//!
//! Two timestamps (begin/end) per frame-in-flight slot. The previous use of a
//! slot's queries is read in `begin`, which runs right after `begin_frame`'s fence
//! wait — so that submission is GPU-complete and the read never stalls. The result
//! is therefore ~`frames_in_flight` frames stale, which is fine for a HUD readout.
//!
//! ponytail: one begin/end pair = whole-frame GPU time only. Add per-pass
//! timestamps (more queries) if you ever need a GPU breakdown.

use ash::vk;

pub(crate) struct GpuTimer {
    pool: vk::QueryPool,
    /// Nanoseconds per timestamp tick (device limit).
    period_ns: f32,
    /// Whether each slot's queries have been written at least once (guards the read).
    primed: Vec<bool>,
    /// Most recent computed GPU frame time (ms).
    last_ms: f32,
}

impl GpuTimer {
    /// `timestamp_period == 0.0` ⇒ timestamps unsupported on this queue ⇒ `None`.
    pub fn new(device: &ash::Device, fif: u32, timestamp_period: f32) -> Option<Self> {
        if timestamp_period <= 0.0 || fif == 0 {
            return None;
        }
        let info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::TIMESTAMP)
            .query_count(fif * 2);
        let pool = unsafe { device.create_query_pool(&info, None) }.ok()?;
        Some(Self {
            pool,
            period_ns: timestamp_period,
            primed: vec![false; fif as usize],
            last_ms: 0.0,
        })
    }

    /// Record at command-buffer begin (outside any render instance), AFTER the
    /// slot's fence wait. Reads the slot's previous result, then resets + stamps start.
    pub fn begin(&mut self, device: &ash::Device, cmd: vk::CommandBuffer, fi: u32) {
        let base = fi * 2;
        if self.primed[fi as usize] {
            let mut data = [0u64; 2];
            // WAIT is safe: begin_frame already waited this slot's fence, so the GPU
            // has finished writing both timestamps — the read returns immediately.
            let r = unsafe {
                device.get_query_pool_results(
                    self.pool,
                    base,
                    &mut data,
                    vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
                )
            };
            if r.is_ok() && data[1] >= data[0] {
                self.last_ms = (data[1] - data[0]) as f32 * self.period_ns * 1.0e-6;
            }
        }
        unsafe {
            device.cmd_reset_query_pool(cmd, self.pool, base, 2);
            device.cmd_write_timestamp(cmd, vk::PipelineStageFlags::TOP_OF_PIPE, self.pool, base);
        }
    }

    /// Record just before `end_command_buffer` (outside any render instance).
    pub fn end(&mut self, device: &ash::Device, cmd: vk::CommandBuffer, fi: u32) {
        unsafe {
            device.cmd_write_timestamp(
                cmd,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                self.pool,
                fi * 2 + 1,
            );
        }
        self.primed[fi as usize] = true;
    }

    /// Latest whole-frame GPU time (ms); `0.0` until the first slot completes.
    pub fn ms(&self) -> f32 {
        self.last_ms
    }

    pub fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_query_pool(self.pool, None) };
    }
}

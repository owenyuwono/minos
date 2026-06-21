//! `fps` — smoothed, throttled frame-time meter for the egui Performance readout.
//!
//! Raw `1/dt` per frame is pure noise. This averages frame time over a short
//! window and only refreshes the displayed value once per window (~2 Hz), so the
//! FPS/ms readout is stable enough to read. egui derives `FPS = 1/frame_time_s`
//! and `ms = frame_time_s * 1000` from the value here.

/// Window over which frame time is averaged before the displayed value refreshes.
const WINDOW_S: f32 = 0.5;

/// Smoothed, throttled frame-time meter.
pub struct FpsMeter {
    accum_time:   f32,
    accum_frames: u32,
    /// Last computed average frame time (seconds); refreshed once per window.
    frame_time_s: f32,
}

impl FpsMeter {
    pub fn new() -> Self {
        Self { accum_time: 0.0, accum_frames: 0, frame_time_s: 0.0 }
    }

    /// Feed one frame's delta time (seconds). Recomputes the displayed average
    /// every `WINDOW_S`, so the readout refreshes ~2×/sec instead of flickering.
    pub fn tick(&mut self, dt: f32) {
        self.accum_time   += dt;
        self.accum_frames += 1;
        if self.accum_time >= WINDOW_S && self.accum_frames > 0 {
            self.frame_time_s = self.accum_time / self.accum_frames as f32;
            self.accum_time   = 0.0;
            self.accum_frames = 0;
        }
    }

    /// Smoothed average frame time in seconds (throttled to ~2 Hz). `0.0` until the
    /// first window elapses.
    pub fn frame_time_s(&self) -> f32 {
        self.frame_time_s
    }
}

impl Default for FpsMeter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_over_window_and_throttles() {
        let mut m = FpsMeter::new();
        // No update before a full window has elapsed.
        m.tick(0.016);
        assert_eq!(m.frame_time_s(), 0.0, "display must not update mid-window");
        // Feed ~0.65 s of steady 16 ms frames → average lands on ~16 ms.
        for _ in 0..40 {
            m.tick(0.016);
        }
        assert!(
            (m.frame_time_s() - 0.016).abs() < 1e-4,
            "avg frame time {} should be ~0.016",
            m.frame_time_s()
        );
    }
}

//! `hud` — compact heads-up display for enki-app.
//!
//! Updates the window title each frame with a concise status line:
//!   `enki | Globe | 1234.5 km | M1 | FPS: 60`
//!
//! This serves as the always-visible status display complementing the
//! egui panel (which shows more detail but requires the cursor to be in
//! the window and may be dismissed).

use winit::window::Window;

use crate::controls::nav_mode::NavMode;
use crate::planet_view::PlanetViewStats;

// ── Hud ───────────────────────────────────────────────────────────────────

/// Lightweight HUD — updates the window title with key telemetry.
pub struct Hud {
    /// Smoothed FPS accumulator.
    fps_accum:  f32,
    fps_frames: u32,
    /// Running smoothed FPS display value.
    fps_display: f32,
}

impl Hud {
    pub fn new() -> Self {
        Self {
            fps_accum:   0.0,
            fps_frames:  0,
            fps_display: 0.0,
        }
    }

    /// Update the HUD and push a new window title.
    ///
    /// Call once per frame after computing `dt`.
    /// `planet_stats` is `Some` when in planet mode.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        window:        &Window,
        dt:            f32,
        nav_mode:      NavMode,
        altitude_m:    f64,
        material_mode: u32,
        wireframe:     bool,
        planet_stats:  Option<&PlanetViewStats>,
    ) {
        // Smooth FPS over ~16 frames to reduce flicker.
        self.fps_accum  += if dt > 0.0 { 1.0 / dt } else { 0.0 };
        self.fps_frames += 1;
        if self.fps_frames >= 16 {
            self.fps_display = self.fps_accum / self.fps_frames as f32;
            self.fps_accum  = 0.0;
            self.fps_frames = 0;
        }

        let mode_str = match nav_mode {
            NavMode::Globe       => "Globe",
            NavMode::Placement   => "Placement",
            NavMode::Surface     => "Surface",
        };

        let alt_str = if altitude_m >= 1_000.0 {
            format!("{:.1} km", altitude_m / 1_000.0)
        } else {
            format!("{:.0} m", altitude_m)
        };

        let wf_str = if wireframe { "+WF" } else { "" };

        let title = if let Some(ps) = planet_stats {
            format!(
                "enki | {} | {} | M{}{} | {:.0} FPS | LOD chunks={} q={} lv={}-{}",
                mode_str, alt_str, material_mode, wf_str, self.fps_display,
                ps.resident_count, ps.build_queue_depth,
                ps.min_lod_level, ps.max_lod_level,
            )
        } else {
            format!(
                "enki | {} | {} | M{}{} | {:.0} FPS",
                mode_str, alt_str, material_mode, wf_str, self.fps_display
            )
        };
        window.set_title(&title);
    }
}

impl Default for Hud {
    fn default() -> Self {
        Self::new()
    }
}

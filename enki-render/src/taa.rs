//! Temporal anti-aliasing (TAA) — CPU-side jitter + history bookkeeping.
//!
//! TAA hides the residual shading shimmer of discrete LOD (e.g. Nanite cluster
//! swaps) and provides cheap anti-aliasing by jittering the camera sub-pixel each
//! frame and accumulating into a temporally-stable history buffer. This module is
//! the **CPU half**: the Halton jitter sequence, the jittered projection used for
//! rasterization, and the previous/current *un-jittered* view-projection matrices
//! the GPU resolve pass needs for motion-vector reprojection.
//!
//! The GPU half (offscreen HDR target, history ping-pong, the resolve pass that
//! reprojects + neighborhood-clamps + blends) lives in the resolve shader
//! [`TAA_RESOLVE_WGSL`] and the RHI's TAA pass. See the resolve shader for the
//! exact reconstruction math.
//!
//! # Usage (per frame)
//! ```ignore
//! // 1. Build the un-jittered view-proj as usual.
//! let view_proj = proj * view;
//! // 2. Jitter the PROJECTION for rasterization (terrain + Nanite both use this).
//! let jittered = taa.jitter_projection(proj, w, h) ; let raster_vp = jittered * view;
//! // 3. Feed `raster_vp` to FrameUniforms.view_proj.
//! // 4. Roll history with the UN-JITTERED view-proj (used by the resolve shader).
//! taa.advance(view_proj);
//! // 5. Upload taa.resolve_params(w, h, alpha) to the resolve UBO.
//! ```

use glam::{DVec3, Mat4, Vec2};

/// The TAA resolve fragment + fullscreen-vertex shader (WGSL).
///
/// Entry points: `vs_fullscreen`, `fs_resolve`. Set-0 layout expected by the
/// pass (all in the fragment stage unless noted):
/// - `@binding(0)` `texture_2d<f32>` — current frame HDR color
/// - `@binding(1)` `texture_2d<f32>` — current frame depth (single-sample)
/// - `@binding(2)` `texture_2d<f32>` — previous resolved history
/// - `@binding(3)` `sampler`         — linear, clamp-to-edge
/// - `@binding(4)` `var<uniform>`    — [`ResolveParams`]
pub const TAA_RESOLVE_WGSL: &str = include_str!("shaders/taa_resolve.wgsl");

/// Number of jitter samples before the Halton sequence repeats. 8 is a common
/// choice (good coverage, short enough that history stays fresh).
pub const JITTER_SEQUENCE_LEN: u32 = 8;

/// Default history blend weight (fraction of the accumulated history kept each
/// frame). Higher = smoother/more stable but more ghosting. ~0.9 is typical.
pub const DEFAULT_HISTORY_BLEND: f32 = 0.9;

/// Radical-inverse (van der Corput) of `index` in `base` — one Halton coordinate
/// in `[0, 1)`. Base 2 and 3 give the standard low-discrepancy Halton(2,3) jitter.
fn halton(mut index: u32, base: u32) -> f32 {
    let mut result = 0.0f32;
    let mut f = 1.0f32 / base as f32;
    while index > 0 {
        result += f * (index % base) as f32;
        index /= base;
        f /= base as f32;
    }
    result
}

/// Std140 uniform block consumed by the resolve shader (matches the WGSL
/// `ResolveParams` struct). 3 × mat4 + 3 × vec4 = 240 bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ResolveParams {
    /// Inverse of the **current** un-jittered view-proj (clip → world).
    pub cur_inv_view_proj: [[f32; 4]; 4],
    /// **Previous** frame's un-jittered view-proj (world → prev clip).
    pub prev_view_proj: [[f32; 4]; 4],
    /// **Current** un-jittered view-proj (world → clip). Used by the screen-space
    /// sun-shadow march to project marched points back to screen.
    pub cur_view_proj: [[f32; 4]; 4],
    /// `(1/width, 1/height, width, height)`.
    pub texel: [f32; 4],
    /// `(history_blend_alpha, history_valid 0|1, _, _)`.
    pub misc: [f32; 4],
    /// Screen-space sun shadow: `xyz` = sun direction (camera-relative/body frame,
    /// toward the sun), `w` = shadow strength (0 = disabled).
    pub sun: [f32; 4],
}

// SAFETY: plain old data, no padding (all fields 16-byte aligned).
unsafe impl bytemuck::Zeroable for ResolveParams {}
unsafe impl bytemuck::Pod for ResolveParams {}

impl ResolveParams {
    /// View as raw bytes for upload to the resolve UBO.
    pub fn as_bytes(&self) -> &[u8] {
        bytemuck::bytes_of(self)
    }
}

/// Sub-pixel jitter generator + view-projection history for the TAA resolve.
#[derive(Clone, Copy, Debug)]
pub struct TaaJitter {
    frame: u32,
    cur_view_proj: Mat4,
    prev_view_proj: Mat4,
    /// Camera world position (f64) for the current / previous frame. Needed
    /// because enki renders **camera-relative** (rotation-only view-proj, camera
    /// at the origin): the per-frame world origin moves with the camera, so the
    /// reprojection must add the camera translation delta (see `resolve_params`).
    cur_cam: DVec3,
    prev_cam: DVec3,
    has_prev: bool,
}

impl Default for TaaJitter {
    fn default() -> Self {
        Self::new()
    }
}

impl TaaJitter {
    pub fn new() -> Self {
        Self {
            frame: 0,
            cur_view_proj: Mat4::IDENTITY,
            prev_view_proj: Mat4::IDENTITY,
            cur_cam: DVec3::ZERO,
            prev_cam: DVec3::ZERO,
            has_prev: false,
        }
    }

    /// This frame's jitter offset in **pixels**, each component in `[-0.5, 0.5)`.
    pub fn jitter_px(&self) -> Vec2 {
        let i = self.frame % JITTER_SEQUENCE_LEN + 1; // +1: skip the zero sample
        Vec2::new(halton(i, 2) - 0.5, halton(i, 3) - 0.5)
    }

    /// Apply this frame's jitter to a projection matrix for a `width`×`height`
    /// viewport. The offset is a sub-pixel NDC shift (`2·px/dim`) added to the
    /// projection's third (clip-`w`-producing) column so it survives perspective
    /// divide. Use the result for rasterization ONLY; reproject with the
    /// un-jittered matrices.
    ///
    /// NOTE (GPU verification): the SIGN of the offset and that column 2 is the
    /// correct insertion point for this reversed-Z Vulkan projection should be
    /// confirmed visually (jitter should shimmer sub-pixel, not drift).
    pub fn jitter_projection(&self, proj: Mat4, width: f32, height: f32) -> Mat4 {
        let j = self.jitter_px();
        let mut p = proj;
        p.z_axis.x += 2.0 * j.x / width.max(1.0);
        p.z_axis.y += 2.0 * j.y / height.max(1.0);
        p
    }

    /// Roll the view-projection history forward. Pass the **un-jittered**
    /// rotation-only view-projection AND the camera's world position for this
    /// frame. Call once per frame after building them.
    pub fn advance(&mut self, unjittered_view_proj: Mat4, camera_world: DVec3) {
        if self.has_prev {
            self.prev_view_proj = self.cur_view_proj;
            self.prev_cam = self.cur_cam;
        } else {
            self.prev_view_proj = unjittered_view_proj;
            self.prev_cam = camera_world;
        }
        self.cur_view_proj = unjittered_view_proj;
        self.cur_cam = camera_world;
        self.has_prev = true;
        self.frame = self.frame.wrapping_add(1);
    }

    /// `true` once at least one frame of history exists (resolve must fall back to
    /// the current frame until then).
    pub fn has_history(&self) -> bool {
        self.has_prev
    }

    /// Build the resolve UBO contents for this frame.
    ///
    /// `cur_inv_view_proj` reconstructs a point **relative to the current camera**
    /// (the rotation-only view-proj puts the camera at the origin). To reproject
    /// it into the previous frame we must express it relative to the *previous*
    /// camera, so the previous view-proj is pre-multiplied by a translation of the
    /// camera delta `(cur_cam - prev_cam)`: `prev' = prev_vp · T(cur_cam-prev_cam)`.
    /// Without this the motion vectors would be wrong whenever the camera moves.
    /// `sun_dir` is the (camera-relative/body-frame) direction toward the sun and
    /// `shadow_strength` ∈ [0,1] the screen-space sun-shadow darkening (0 = off).
    pub fn resolve_params(
        &self,
        width: f32,
        height: f32,
        alpha: f32,
        sun_dir: glam::Vec3,
        shadow_strength: f32,
    ) -> ResolveParams {
        let delta = (self.cur_cam - self.prev_cam).as_vec3();
        let prev_vp = self.prev_view_proj * Mat4::from_translation(delta);
        ResolveParams {
            cur_inv_view_proj: self.cur_view_proj.inverse().to_cols_array_2d(),
            prev_view_proj: prev_vp.to_cols_array_2d(),
            cur_view_proj: self.cur_view_proj.to_cols_array_2d(),
            texel: [1.0 / width.max(1.0), 1.0 / height.max(1.0), width, height],
            misc: [
                alpha,
                if self.has_prev { 1.0 } else { 0.0 },
                0.0,
                0.0,
            ],
            sun: [sun_dir.x, sun_dir.y, sun_dir.z, shadow_strength],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn halton_known_values() {
        assert!((halton(1, 2) - 0.5).abs() < 1e-6);
        assert!((halton(2, 2) - 0.25).abs() < 1e-6);
        assert!((halton(3, 2) - 0.75).abs() < 1e-6);
        assert!((halton(1, 3) - 1.0 / 3.0).abs() < 1e-6);
        assert!((halton(2, 3) - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn jitter_in_half_pixel_range() {
        let mut t = TaaJitter::new();
        for _ in 0..64 {
            let j = t.jitter_px();
            assert!(j.x >= -0.5 && j.x < 0.5, "jitter.x out of range: {}", j.x);
            assert!(j.y >= -0.5 && j.y < 0.5, "jitter.y out of range: {}", j.y);
            t.advance(Mat4::IDENTITY, DVec3::ZERO);
        }
    }

    #[test]
    fn jitter_projection_offsets_third_column() {
        let mut t = TaaJitter::new();
        t.advance(Mat4::IDENTITY, DVec3::ZERO); // frame 1 has non-zero jitter in both axes
        let proj = crate::projection::reversed_z_perspective(1.0, 1.6, 0.5, 1e6);
        let jp = t.jitter_projection(proj, 1920.0, 1080.0);
        // Only the z column's x/y should change; other columns are untouched.
        assert_ne!(jp.z_axis.x, proj.z_axis.x);
        assert_ne!(jp.z_axis.y, proj.z_axis.y);
        assert_eq!(jp.x_axis, proj.x_axis);
        assert_eq!(jp.y_axis, proj.y_axis);
        assert_eq!(jp.w_axis, proj.w_axis);
    }

    #[test]
    fn history_starts_invalid_then_valid() {
        let mut t = TaaJitter::new();
        assert!(!t.has_history());
        let p = t.resolve_params(1920.0, 1080.0, 0.9, glam::Vec3::Y, 0.0);
        assert_eq!(p.misc[1], 0.0, "history must be invalid on the first frame");
        t.advance(Mat4::IDENTITY, DVec3::ZERO);
        assert!(t.has_history());
        let p2 = t.resolve_params(1920.0, 1080.0, 0.9, glam::Vec3::Y, 0.0);
        assert_eq!(p2.misc[1], 1.0);
    }

    #[test]
    fn resolve_shader_is_valid_wgsl() {
        let module = naga::front::wgsl::parse_str(TAA_RESOLVE_WGSL)
            .expect("taa_resolve.wgsl must parse");
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .expect("taa_resolve.wgsl must validate");
    }
}

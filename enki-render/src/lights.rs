//! Light sources used to fill `FrameUniforms`.
//!
//! # Design
//! `Lights` holds scene lighting parameters in a CPU-friendly form.
//! `FrameUniforms::new` converts this into the GPU-ready struct.
//!
//! Directional intensities are stored separately and multiplied into the color
//! when building `FrameUniforms`, so the color fields in `Lights` carry
//! normalised linear-sRGB hues and intensities stay human-editable numbers.

/// A directional (sun) light.
#[derive(Debug, Clone, Copy)]
pub struct DirectionalLight {
    /// World-space direction the light travels *toward* (unit vector, pointing
    /// from the scene toward the light source — i.e. "sun direction").
    /// Must be non-zero; [`Lights::demiurge_default`] always normalises these.
    pub direction: glam::Vec3,
    /// Linear-sRGB color of the light, pre-multiplied by intensity.
    /// Stored pre-multiplied so `FrameUniforms` can copy directly.
    pub color_intensity: glam::Vec3,
}

/// Hemisphere ambient light: smoothly varies albedo × `sky_color` for
/// fragments facing up and `ground_color` for fragments facing down.
#[derive(Debug, Clone, Copy)]
pub struct HemisphereLight {
    /// Color for fully sky-facing surfaces (linear sRGB, pre-multiplied by intensity).
    pub sky: glam::Vec3,
    /// Color for fully ground-facing surfaces (linear sRGB, pre-multiplied by intensity).
    pub ground: glam::Vec3,
}

/// All scene lights consumed by the render layer.
///
/// Two directional lights (sun + fill), one hemisphere, and one ambient white term.
#[derive(Debug, Clone)]
pub struct Lights {
    /// Isotropic ambient white, multiplied into albedo uniformly.
    /// Scalar; the GPU receives `vec4(ambient, ambient, ambient, 1.0)`.
    pub ambient: f32,
    /// Hemisphere ambient (sky/ground gradient).
    pub hemisphere: HemisphereLight,
    /// Primary directional light (sun).
    pub directional_0: DirectionalLight,
    /// Secondary directional light (fill/rim).
    pub directional_1: DirectionalLight,
}

impl Lights {
    /// Prototype "Demiurge" lighting setup:
    ///
    /// | Term          | Value                                         |
    /// |---------------|-----------------------------------------------|
    /// | Ambient white | 0.35                                          |
    /// | Hemi sky      | `#9ab4ff` × 0.6                              |
    /// | Hemi ground   | `#20180e` × 0.6                              |
    /// | Sun dir       | `(1, 0.8, 0.6)` normalised, white × 1.1     |
    /// | Fill dir      | `(-0.9, -0.3, -1)` normalised, white × 0.7  |
    ///
    /// All directions point *from scene toward light source* (normalised).
    /// Intensity is premultiplied into the color stored in `color_intensity`.
    pub fn demiurge_default() -> Self {
        // --- Hemisphere colors: hex → linear sRGB, scaled by intensity 0.6
        // #9ab4ff = sRGB(154, 180, 255) → linear (0.605, 0.706, 1.0)
        // #9ab4ff = sRGB(154, 180, 255)
        let sky_srgb = glam::Vec3::new(154.0 / 255.0, 180.0 / 255.0, 1.0);
        let sky_linear = srgb_to_linear(sky_srgb) * 0.6;

        // #20180e = sRGB(32, 24, 14) → linear (0.125, 0.094, 0.055)
        let ground_srgb = glam::Vec3::new(32.0 / 255.0, 24.0 / 255.0, 14.0 / 255.0);
        let ground_linear = srgb_to_linear(ground_srgb) * 0.6;

        // --- Directional lights (intensity premultiplied)
        let sun_dir = glam::Vec3::new(1.0, 0.8, 0.6).normalize();
        let fill_dir = glam::Vec3::new(-0.9, -0.3, -1.0).normalize();

        Self {
            ambient: 0.35,
            hemisphere: HemisphereLight {
                sky: sky_linear,
                ground: ground_linear,
            },
            directional_0: DirectionalLight {
                direction: sun_dir,
                color_intensity: glam::Vec3::splat(1.0) * 1.1,
            },
            directional_1: DirectionalLight {
                direction: fill_dir,
                color_intensity: glam::Vec3::splat(1.0) * 0.7,
            },
        }
    }

    /// dryad's EXACT viewer lighting (flora viewer / "a better No Man's Sky"
    /// reference). Mirrors dryad/src/viewer.js: a single key `DirectionalLight`
    /// `#fff4e0 × 3.0` along the sun direction, a `HemisphereLight`
    /// `sky #88aacc / ground #443322 × 0.3`, NO second directional, and NO
    /// isotropic white ambient (dryad's ambient is the HemisphereLight + IBL).
    ///
    /// | Term        | Value                                       |
    /// |-------------|---------------------------------------------|
    /// | Ambient     | 0.0 (dryad has no white ambient term)       |
    /// | Hemi sky    | `#88aacc` × 0.3                             |
    /// | Hemi ground | `#443322` × 0.3                             |
    /// | Sun dir     | `(1, 0.8, 0.6)` normalised, `#fff4e0` × 3.0 |
    /// | Fill dir    | none (color_intensity = 0)                  |
    ///
    /// `environmentIntensity 0.6` and `toneMappingExposure 1.0` are NOT applied
    /// here — they are already baked into the IBL SH/spec and the ACES call.
    /// Additive sibling of [`Self::demiurge_default`]; the shared planet/terrain
    /// path keeps using `demiurge_default`.
    pub fn dryad_default() -> Self {
        // sun key #fff4e0 = sRGB(255, 244, 224), premultiplied by intensity 3.0.
        let sun_srgb = glam::Vec3::new(255.0 / 255.0, 244.0 / 255.0, 224.0 / 255.0);
        let sun_lin = srgb_to_linear(sun_srgb) * 3.0;

        // hemi sky #88aacc / ground #443322, premultiplied by intensity 0.3.
        let sky_lin =
            srgb_to_linear(glam::Vec3::new(136.0 / 255.0, 170.0 / 255.0, 204.0 / 255.0)) * 0.3;
        let ground_lin =
            srgb_to_linear(glam::Vec3::new(68.0 / 255.0, 51.0 / 255.0, 34.0 / 255.0)) * 0.3;

        // dryad sunPosition = normalized lightDir (flora's existing sun dir).
        let sun_dir = glam::Vec3::new(1.0, 0.8, 0.6).normalize();

        Self {
            ambient: 0.0,
            hemisphere: HemisphereLight {
                sky: sky_lin,
                ground: ground_lin,
            },
            directional_0: DirectionalLight {
                direction: sun_dir,
                color_intensity: sun_lin,
            },
            // No second directional in dryad: zero color → contributes nothing
            // (direction irrelevant). Keep a unit dir to avoid NaNs downstream.
            directional_1: DirectionalLight {
                direction: glam::Vec3::Y,
                color_intensity: glam::Vec3::ZERO,
            },
        }
    }
}

/// Approximate gamma-expand from sRGB to linear (component-wise).
/// Uses the piecewise IEC 61966-2-1 curve.
fn srgb_to_linear(c: glam::Vec3) -> glam::Vec3 {
    glam::Vec3::new(
        srgb_channel_to_linear(c.x),
        srgb_channel_to_linear(c.y),
        srgb_channel_to_linear(c.z),
    )
}

fn srgb_channel_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

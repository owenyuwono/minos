//! Sun render pass — shader source + naga validation.
//!
//! A camera-facing billboard (limb-darkened disk + corona) drawn at the sky-shell
//! distance along the sun direction. Pipeline contract (created by the consumer,
//! `enki-app/src/solar`): standard set 0 (`FrameUniforms` at binding 0), `ChunkPush`
//! push constant (80 B), the shared 4×vec3 vertex layout, alpha-blend + depth-write
//! OFF + depth-test GREATER. Draw **after** opaque terrain so reversed-Z lets the
//! planet limb eclipse it.

/// WGSL for the sun billboard. Entry points `vs_main` / `fs_main`.
pub const SUN_WGSL: &str = include_str!("shaders/sun.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_wgsl_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(SUN_WGSL)
            .unwrap_or_else(|e| panic!("sun.wgsl failed to parse: {e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("sun.wgsl failed to validate: {e:?}"));
    }
}

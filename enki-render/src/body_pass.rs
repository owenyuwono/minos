//! Solar-system body render pass — shader source + naga validation.
//!
//! A lit/emissive UV-sphere drawn camera-relative on the sky shell (the sun + distant
//! planets). Pipeline contract (created by `enki-app/src/solar`): standard set 0
//! (`FrameUniforms` at binding 0), a 96-byte `BodyPush` push constant, the shared
//! 4×vec3 vertex layout (pos + normal used), alpha-blend + depth-write OFF + depth-test
//! GREATER. Draw after opaque terrain so reversed-Z lets the focused planet limb
//! eclipse far bodies.

/// WGSL for the body sphere. Entry points `vs_main` / `fs_main`.
pub const BODY_WGSL: &str = include_str!("shaders/body.wgsl");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn body_wgsl_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(BODY_WGSL)
            .unwrap_or_else(|e| panic!("body.wgsl failed to parse: {e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("body.wgsl failed to validate: {e:?}"));
    }
}

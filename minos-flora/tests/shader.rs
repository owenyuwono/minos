//! Headless WGSL validation gate for flora.wgsl.
//!
//! Mirrors the RHI's runtime path: parse + validate with naga (the same
//! frontend minos uses to compile shaders), so any WGSL error in flora.wgsl
//! fails `cargo test` instead of only surfacing when the app launches.
//!
//! ponytail: validation only — does not render. It catches parse/type/binding
//! errors, which is the cheap check that protects the deletable-crate contract.

const FLORA_WGSL: &str = include_str!("../shaders/flora.wgsl");
// SCENE staging (sky/ground/shadow) for the standalone flora viewer. Same naga
// gate as the tree shader so a broken staging.wgsl fails `cargo test` too.
const STAGING_WGSL: &str = include_str!("../shaders/staging.wgsl");

#[test]
fn shaders_parse_and_validate() {
    for (name, src) in [("flora", FLORA_WGSL), ("staging", STAGING_WGSL)] {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name} WGSL parse failed:\n{e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        let info = validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} WGSL validation failed:\n{e:?}"));

        // spv-out gate: the RHI compiles WGSL → SPIR-V via naga's spv backend at
        // runtime (minos-rhi::shader::compile_wgsl). Run that same backend here so
        // backend-only failures (e.g. an unsupported construct the validator
        // accepts) fail `cargo test` instead of only surfacing at launch.
        let opts = naga::back::spv::Options::default();
        let words = naga::back::spv::write_vec(&module, &info, &opts, None)
            .unwrap_or_else(|e| panic!("{name} WGSL → SPIR-V (spv-out) failed:\n{e:?}"));
        assert!(!words.is_empty(), "{name} produced empty SPIR-V");
    }
}

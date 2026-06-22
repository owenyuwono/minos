// body.wgsl — a solar-system body as a real lit/emissive sphere (Tier 1).
//
// A UV-sphere drawn camera-relative on the sky shell. A STAR is emissive
// (limb-darkened, HDR-overdriven before ACES so the core saturates to white); a
// PLANET is lambert-lit by its OWN sun direction → a real day/night terminator.
// Pipeline: alpha-blend, depth-test GREATER, depth-write OFF — the focused planet
// limb eclipses far bodies for free under reversed-Z.

struct Frame { view_proj: mat4x4<f32> };
@group(0) @binding(0) var<uniform> frame: Frame;

struct Push {
    model:   mat4x4<f32>,
    sun_dir: vec4<f32>,   // xyz: surface→sun (rendered frame); w: 1 = star, 0 = planet
    color:   vec4<f32>,   // albedo (planet) / emissive tint (star)
};
var<immediate> pc: Push;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) world_rel: vec3<f32>,
    @location(1) normal:    vec3<f32>,
};

@vertex
fn vs_main(@location(0) position: vec3<f32>, @location(1) nrm: vec3<f32>) -> VsOut {
    var out: VsOut;
    let wr = pc.model * vec4<f32>(position, 1.0);
    out.clip = frame.view_proj * wr;
    out.world_rel = wr.xyz;
    let m3 = mat3x3<f32>(pc.model[0].xyz, pc.model[1].xyz, pc.model[2].xyz);
    out.normal = normalize(m3 * nrm);
    return out;
}

const SUN_HDR: f32 = 9.0;

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    if (pc.sun_dir.w > 0.5) {
        // Star: emissive, limb-darkened (brightest facing the camera).
        let view = normalize(-in.world_rel);
        let ndv = max(dot(n, view), 0.0);
        let limb = 0.55 + 0.45 * sqrt(ndv);
        return vec4<f32>(aces(pc.color.xyz * (SUN_HDR * limb)), 1.0);
    }
    // Planet: lambert day side + dim ambient night side.
    let ndl = max(dot(n, pc.sun_dir.xyz), 0.0);
    let lit = pc.color.xyz * (0.06 + 0.94 * ndl);
    return vec4<f32>(aces(lit), 1.0);
}

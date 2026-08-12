// sun.wgsl — billboard sun: limb-darkened photosphere disk + analytic corona glow.
//
// A camera-facing quad placed at the sky-shell distance along the sun direction
// (the billboard basis + position are baked into the model matrix CPU-side). The
// fragment draws a bright disk in the centre (inner R_DISK of the quad) and a soft
// corona out to the rim. HDR-overdriven before ACES (no bloom target yet) so the
// core saturates to hot white. Pipeline: alpha-blend, depth-test GREATER, depth-write
// OFF — the planet limb eclipses the sun for free under reversed-Z.

// We only read view_proj (offset 0 of FrameUniforms); the rest is unused here.
struct Frame {
    view_proj: mat4x4<f32>,
};
@group(0) @binding(0) var<uniform> frame: Frame;

// Matches ChunkPush's layout (model + material_mode + 3 pad u32 = 80 bytes).
struct Push {
    model: mat4x4<f32>,
    _pad:  vec4<u32>,
};
var<immediate> pc: Push;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>, // quad-local coord, corners at (±1, ±1)
};

@vertex
fn vs_main(@location(0) position: vec3<f32>) -> VsOut {
    var out: VsOut;
    out.clip = frame.view_proj * (pc.model * vec4<f32>(position, 1.0));
    out.uv = position.xy;
    return out;
}

// Inner fraction of the quad radius occupied by the photosphere disk (rest = corona).
const R_DISK: f32 = 0.25;
const SUN_RGB:    vec3<f32> = vec3<f32>(1.0, 0.93, 0.82); // warm white photosphere
const CORONA_RGB: vec3<f32> = vec3<f32>(1.0, 0.85, 0.60); // warmer corona
const SUN_HDR: f32 = 9.0;                                  // over-drive → ACES clips core to white

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let r = length(in.uv);
    if (r > 1.0) { discard; } // inscribe a circle in the quad — no square corners

    if (r < R_DISK) {
        // Photosphere with limb darkening toward the edge.
        let x = r / R_DISK;
        let limb = 0.45 + 0.55 * sqrt(max(0.0, 1.0 - x * x));
        return vec4<f32>(aces(SUN_RGB * (SUN_HDR * limb)), 1.0);
    }
    // Corona: soft falloff from the disk edge (t=0) to the quad rim (t=1).
    let t = (r - R_DISK) / (1.0 - R_DISK);
    let glow = pow(1.0 - t, 3.0);
    return vec4<f32>(aces(CORONA_RGB * (SUN_HDR * 0.5 * glow)), glow * 0.85);
}

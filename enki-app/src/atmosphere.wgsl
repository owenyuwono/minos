// atmosphere.wgsl — translucent atmosphere shell.
//
// A sky-tinted shell sphere at R + height, alpha-blended over the planet. Alpha
// is densest at the limb (grazing view = longest air column) and on the day side;
// a forward sun-glow brightens the sunward limb. Analytic — no scattering
// integral (ponytail: a single shell; port ki Atmosphere.ts for true Rayleigh/Mie).
//
// Pipeline contract: set0 binding 0 = FrameUniforms; ChunkPush push constants
// (model = uniform scale; pad0 reused as `density`). 4×vec3 vertex slots — only
// position (0) + normal (1) are read. Depth GREATER, depth-write OFF, alpha-blend.

struct FrameUniforms {
    view_proj  : mat4x4<f32>,
    camera_pos : vec4<f32>,
    sun0_dir   : vec4<f32>,
    sun0_color : vec4<f32>,
    sun1_dir   : vec4<f32>,
    sun1_color : vec4<f32>,
    hemi_sky   : vec4<f32>,
    hemi_ground: vec4<f32>,
    ambient    : vec4<f32>,
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;

struct ChunkPush {
    model         : mat4x4<f32>,
    material_mode : u32,
    density       : f32, // reused pad slot
    _p1           : f32,
    _p2           : f32,
}

var<immediate> pc: ChunkPush;

struct VertexIn {
    @location(0) position   : vec3<f32>,
    @location(1) normal     : vec3<f32>,
    @location(2) color      : vec3<f32>,
    @location(3) plate_color: vec3<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0)       world_pos: vec3<f32>,
    @location(1)       normal   : vec3<f32>,
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    let world = pc.model * vec4<f32>(v.position, 1.0);
    out.clip_pos = frame.view_proj * world;
    out.world_pos = world.xyz;            // camera-relative
    out.normal = normalize(v.normal);     // uniform scale → radial normal preserved
    return out;
}

// Analytic single-scatter constants (ported from ki Atmosphere.ts).
const ATMO_TINT:    vec3<f32> = vec3<f32>(0.027, 0.147, 1.0); // #2e6bff (linear) — sky blue
const SUN_INTENSITY: f32 = 1.3;
const SCALE_HEIGHT:  f32 = 3.0;  // limb-glow pow exponent (higher = thinner band)
const SKY_FLOOR:     f32 = 0.35; // zenith in-scatter floor → blue dome overhead, not black
const BETA_R: vec3<f32> = vec3<f32>(0.175, 0.408, 1.0); // Rayleigh β ∝ 1/λ⁴ (blue=1)

// Analytic single-scattering shell (ki Atmosphere.ts node graph, ported to WGSL):
//   glow   = max(limb pow-ramp, air-mass dome curve)  → bright limb + lit dome
//   density= mix(SKY_FLOOR, 1, glow)                   → blue sky overhead (not black)
//   phase  = Rayleigh 0.75(1+mu²)                      → brighter toward the sun
//   sunUp  = day/night terminator                      → night-side limb stays dark
//   redden = per-channel slant-air-mass extinction     → sunrise/sunset reddening
// All camera-relative: viewDir = cam→fragment, nFrag = shell radial normal.
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let nFrag   = normalize(in.normal);
    let viewDir = normalize(in.world_pos);          // camera at origin → cam→fragment
    let sunDir  = frame.sun0_dir.xyz;

    // Limb glow + air-mass broadening. BOTH use abs(viewDir·nFrag) so they read as
    // a limb-concentrated ring from orbit (the shell only ever renders the near,
    // camera-facing hemisphere, where viewDir·nFrag ≤ 0). Without abs the air-mass
    // term floored to 0 across the whole disc → flat blue fill over the planet.
    let cosView     = abs(dot(viewDir, nFrag));
    let grazing     = clamp(1.0 - cosView, 0.0, 1.0);
    let horizonGlow = pow(grazing, max(SCALE_HEIGHT, 0.1));
    let relAirMass  = 1.0 / max(cosView, 0.05) - 1.0;
    let airMassGlow = 1.0 - exp(-relAirMass * 0.6);
    let glow        = max(horizonGlow, airMassGlow);
    let density     = mix(SKY_FLOOR, 1.0, glow);

    // Rayleigh phase (toward sun) + day/night terminator.
    let mu     = dot(viewDir, sunDir);
    let phase  = 0.75 * (1.0 + mu * mu);
    let muSun  = dot(nFrag, sunDir);
    let sunUp  = clamp(muSun * 2.0 + 0.3, 0.0, 1.0);
    let amount = density * phase * sunUp * pc.density;

    // Wavelength reddening from the combined view+sun slant air mass (low sun → red).
    let amView = 1.0 / max(abs(dot(viewDir, nFrag)), 0.05);
    let amSun  = 1.0 / max(muSun, 0.05);
    let slant  = clamp(amView + amSun - 2.0, 0.0, 40.0);
    let redden = exp(-BETA_R * pc.density * 0.06 * slant);

    let scatter  = ATMO_TINT * SUN_INTENSITY * amount * redden;
    // Gate alpha by the limb glow so the disc centre stays see-through from orbit
    // (the SKY_FLOOR baseline otherwise veils the planet); the limb halo is unchanged.
    let envelope = clamp(amount * glow, 0.0, 1.0);
    return vec4<f32>(scatter, envelope);
}

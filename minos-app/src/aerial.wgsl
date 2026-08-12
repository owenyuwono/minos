// aerial.wgsl — atmospheric scattering as a fullscreen aerial-perspective pass.
//
// Where the atmosphere SHELL (atmosphere.wgsl) is a sphere that can only glow at
// the limb, this pass reads the opaque scene DEPTH and integrates Rayleigh
// in-scatter along the camera→surface ray, so it satisfies what a shell can't:
//   (1) standing on the surface, sky pixels (no terrain) integrate the full
//       atmosphere chord → a blue overhead dome + bright horizon band;
//   (2) from orbit, terrain pixels integrate the camera→terrain segment → distant
//       SUNLIT land/ocean blue out with distance (true aerial perspective).
//
// Optically thin (~atmo_height over R~50 km), so a constant-density slab closed
// form is exact for uniform beta — NO raymarch, NO 3D LUT. The day-side gate
// (`sun_vis` from the surface radial-up · sun) makes only the directly-lit
// hemisphere tint, matching "the area that gets direct sunlight".
//
// Pipeline contract: custom set0 — 0 FrameUniforms UBO, 1 Aerial UBO, 2 scene
// depth (SAMPLED), 3 sampler. Vertex-pulling fullscreen triangle (depth GREATER,
// depth-write OFF, alpha-blend). Drawn in the water-split 1× instance, AFTER the
// ocean and BEFORE clouds. The SRC_ALPHA blend + un-premultiplied output composite
// as `inscatter + background·transmittance` (the aerial-perspective form).

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

struct Aerial {
    right     : vec4<f32>, // camera right (xyz)
    up        : vec4<f32>, // camera up (xyz)
    fwd       : vec4<f32>, // camera forward (xyz)
    center_rel: vec4<f32>, // xyz = planet centre − camera (camera-relative); w = planet radius R
    params    : vec4<f32>, // height_m, beta(1/m), intensity, sky_strength
    screen    : vec4<f32>, // width, height, tan_half_fov, aspect
    misc      : vec4<f32>, // debug, proj_a, proj_b, _pad
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<uniform> aerial: Aerial;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_samp: sampler;

// Rayleigh β ∝ 1/λ⁴, normalised to blue=1 — the per-channel scatter/extinction
// ratio that blues out distant surfaces (blue saturates first as the path grows).
const BETA_R: vec3<f32> = vec3<f32>(0.175, 0.408, 1.0);
// Scalar mean of BETA_R for the single blend-alpha (background extinction).
const BETA_MEAN: f32 = 0.528;

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       ndc:      vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    // Oversized triangle: vid 0 → (-1,-1), 1 → (-1,3), 2 → (3,-1). z=1 (reversed-Z
    // near) so depth-test GREATER passes over every opaque pixel (mirrors clouds).
    let p = vec2<f32>(
        select(-1.0, 3.0, vid == 2u),
        select(-1.0, 3.0, vid == 1u),
    );
    out.clip_pos = vec4<f32>(p, 1.0, 1.0);
    out.ndc = p;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let ndc = in.ndc;
    let tan_half = aerial.screen.z;
    let aspect = aerial.screen.w;

    // Camera-relative ray (camera at origin). ndc.y=-1 is screen-top → +up.
    let dir = normalize(
        aerial.fwd.xyz
        + aerial.right.xyz * (ndc.x * tan_half * aspect)
        + aerial.up.xyz * (-ndc.y * tan_half)
    );

    let c = aerial.center_rel.xyz;       // planet centre relative to camera
    let big_r = aerial.center_rel.w;
    let r_top = big_r + aerial.params.x; // atmosphere outer radius

    // Ray vs the outer atmosphere sphere (origin = camera = 0). oc = −c.
    let oc = -c;
    let b = dot(dir, oc);
    let cc = dot(oc, oc) - r_top * r_top;
    let disc = b * b - cc;
    if (disc < 0.0) { return vec4<f32>(0.0); }   // ray misses the atmosphere → space
    let sq = sqrt(disc);
    let t0 = max(-b - sq, 0.0);                   // entry (0 if camera already inside)
    var t1 = -b + sq;                             // exit
    if (t1 <= 0.0) { return vec4<f32>(0.0); }     // atmosphere entirely behind camera

    // Terrain/ocean clip: solve forward distance from reversed-Z depth (no inverse
    // matrix), divide by the ray's forward cosine. scene_d==0 → sky (full chord).
    let uv = ndc * 0.5 + 0.5;
    let scene_d = textureSampleLevel(depth_tex, depth_samp, uv, 0.0).r;
    var hit_surface = false;
    if (scene_d > 0.0) {
        let fwd_dist = aerial.misc.z / (scene_d + aerial.misc.y); // proj_b/(d+proj_a)
        let t_scene = fwd_dist / max(dot(dir, aerial.fwd.xyz), 1e-3);
        t1 = min(t1, t_scene);
        hit_surface = true;
    }
    let s = max(t1 - t0, 0.0);                     // in-atmosphere path length (m)
    if (s <= 0.0) { return vec4<f32>(0.0); }

    // Day-side gate from the FAR point's radial up (terrain hit, or shell exit for
    // sky). Soft terminator: 1 well into the day side, 0 just past the night edge.
    let hit = dir * t1 - c;
    let up_m = normalize(hit);
    let sun_dir = normalize(frame.sun0_dir.xyz);
    let sun_vis = clamp(dot(up_m, sun_dir) * 2.0 + 0.3, 0.0, 1.0);
    if (sun_vis <= 0.0) { return vec4<f32>(0.0); } // night side: atmosphere is invisible

    // Density-weighted optical path (Chapman airmass via a short march). The
    // atmosphere thins with altitude as exp(−alt/H), so a grazing limb ray — most of
    // which is high, near-vacuum air — accumulates a BOUNDED `od` instead of the raw
    // chord `s`. Without this the constant-density slab blows the limb out to a white
    // ring (all channels saturate); with it the limb fades softly to space. Near the
    // surface density ≈ 1 so `od ≈ s` and the ground aerial perspective is unchanged.
    let h_scale = max(aerial.misc.w, 1.0);
    let steps = 16;
    let ds = s / f32(steps);
    var od = 0.0;
    var tt = t0 + ds * 0.5;
    for (var i = 0; i < steps; i = i + 1) {
        let alt = max(length(dir * tt - c) - big_r, 0.0);
        od = od + exp(-alt / h_scale) * ds;
        tt = tt + ds;
    }
    if (od <= 0.0) { return vec4<f32>(0.0); }

    // Rayleigh single-scatter (exact for the marched optical depth): in-scatter
    // saturates as 1−exp(−β·od) per channel → blue leads, so the far field blues out.
    let beta = BETA_R * aerial.params.y;
    let mu = dot(dir, sun_dir);
    let phase = 0.75 * (1.0 + mu * mu);
    let inscat_frac = vec3<f32>(1.0) - exp(-beta * od);

    // Sky pixels carry the sky_strength multiplier so the overhead dome can be
    // balanced against the terrain haze independently.
    var gain = aerial.params.z;
    if (!hit_surface) { gain = gain * aerial.params.w; }

    let inscatter = frame.sun0_color.xyz * phase * sun_vis * gain * inscat_frac;

    // Scalar background transmittance for the SRC_ALPHA blend, gated by the same
    // day-side term so the night side is a true no-op (background untouched).
    let alpha = clamp((1.0 - exp(-aerial.params.y * BETA_MEAN * od)) * sun_vis, 0.0, 1.0);

    if (aerial.misc.x > 0.5) {
        // Debug: show the raw in-scatter (opaque), no background composite.
        return vec4<f32>(inscatter, 1.0);
    }
    if (alpha <= 0.001) { return vec4<f32>(0.0); }
    // Un-premultiply for SRC_ALPHA blend → inscatter + background·(1−alpha).
    return vec4<f32>(inscatter / alpha, alpha);
}

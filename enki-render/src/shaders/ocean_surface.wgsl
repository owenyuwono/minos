// ocean_surface.wgsl — FFT wave surface for the planet ocean.
//
// A camera-anchored tangent-plane grid is curved onto the sea-level sphere and
// displaced by the CPU spectral sim (uploaded as a storage buffer of OceanTexel).
// Shading ported from poseidon's oceanSurfaceMaterial (Fresnel sky reflection,
// reflected-sun glint, subsurface scatter, Jacobian foam), adapted to the planet's
// LOCAL up (not world +Y) and enki's camera-relative / reversed-Z conventions.
//
// Camera-relative: positions are built relative to the camera (which sits at the
// origin of frame.view_proj, a rotation-only view), so all vertex math stays in
// small near-origin floats. Reversed-Z is automatic via frame.view_proj.
//
// set 0 (custom, bound via cmd_bind_descriptor_set — NOT the rhi set0 ring):
//   binding 0: Frame   (uniform)  — mirrors enki_render::FrameUniforms (192 B)
//   binding 1: field   (storage)  — array<OceanTexel>, the sim output
//   binding 2: Ocean   (uniform)  — tangent frame + tiling + colors

struct Frame {
    view_proj  : mat4x4<f32>,
    camera_pos : vec4<f32>,   // always (0,0,0,1) in camera-relative space
    sun0_dir   : vec4<f32>,
    sun0_color : vec4<f32>,
    sun1_dir   : vec4<f32>,
    sun1_color : vec4<f32>,
    hemi_sky   : vec4<f32>,
    hemi_ground: vec4<f32>,
    ambient    : vec4<f32>,
}

struct OceanTexel {
    disp   : vec4<f32>,   // xyz displacement (m), w foam
    normal : vec4<f32>,   // xyz tangent-space normal (x=east, y=up, z=north)
}

struct Ocean {
    east          : vec4<f32>,   // tangent basis at the sub-camera point (unit dirs)
    north         : vec4<f32>,
    up            : vec4<f32>,   // surface normal at sub-point
    sub_point_rel : vec4<f32>,   // xyz = (sub_point - camera); w = sea_radius
    cfg           : vec4<f32>,   // x length_scale, y patch_half, z grid_dim(N), w fade_start(0..1)
    deep_color    : vec4<f32>,
    scatter_color : vec4<f32>,
    foam_color    : vec4<f32>,
    sky_horizon   : vec4<f32>,
    sky_zenith    : vec4<f32>,
    sun_color     : vec4<f32>,
    shading       : vec4<f32>,   // x sss, y foam_threshold, z foam_scale, w alpha
    screen        : vec4<f32>,   // x width, y height, z refract_strength, w tint
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> field: array<OceanTexel>;
@group(0) @binding(2) var<uniform> ocean: Ocean;
// Refraction source: the resolved opaque scene color behind the water.
@group(0) @binding(3) var scene_tex: texture_2d<f32>;
@group(0) @binding(4) var scene_samp: sampler;

// ── Bilinear sample of the tiled (wrapping) FFT field ──────────────────────────

fn wrap_index(i: i32, n: i32) -> i32 {
    return ((i % n) + n) % n;
}

fn sample_field(uv: vec2<f32>) -> OceanTexel {
    let n = i32(ocean.cfg.z);
    let fn_ = ocean.cfg.z;
    // tile uv into [0,1) then to texel space.
    let t = (uv - floor(uv)) * fn_;
    let x0 = i32(floor(t.x));
    let y0 = i32(floor(t.y));
    let fx = t.x - floor(t.x);
    let fy = t.y - floor(t.y);
    let x1 = wrap_index(x0 + 1, n);
    let y1 = wrap_index(y0 + 1, n);
    let xa = wrap_index(x0, n);
    let ya = wrap_index(y0, n);

    let a = field[ya * n + xa];
    let b = field[ya * n + x1];
    let c = field[y1 * n + xa];
    let d = field[y1 * n + x1];

    var o: OceanTexel;
    let dab = mix(a.disp, b.disp, fx);
    let dcd = mix(c.disp, d.disp, fx);
    o.disp = mix(dab, dcd, fy);
    let nab = mix(a.normal, b.normal, fx);
    let ncd = mix(c.normal, d.normal, fx);
    o.normal = mix(nab, ncd, fy);
    return o;
}

// ── Vertex ─────────────────────────────────────────────────────────────────────

struct VsIn {
    // Reuses the standard 4×vec3 vertex layout; only position is read, and only
    // its .xz (the tangent-plane offset in metres, [-patch_half, +patch_half]).
    @location(0) position: vec3<f32>,
}

struct VsOut {
    @builtin(position) clip       : vec4<f32>,
    @location(0)       world_pos  : vec3<f32>,   // camera-relative
    @location(1)       uv         : vec2<f32>,   // tile uv for fragment field fetch
    @location(2)       fade       : f32,
}

fn edge_fade(grid: vec2<f32>) -> f32 {
    let r = max(abs(grid.x), abs(grid.y));
    let start = ocean.cfg.w * ocean.cfg.y;
    return 1.0 - smoothstep(start, ocean.cfg.y, r);
}

@vertex
fn vs_main(v: VsIn) -> VsOut {
    var out: VsOut;
    let grid = v.position.xz;
    let gx = grid.x;
    let gz = grid.y;
    let length_scale = ocean.cfg.x;
    let uv = grid / length_scale;
    let fade = edge_fade(grid);

    let disp = sample_field(uv).disp.xyz * fade;

    let east = ocean.east.xyz;
    let north = ocean.north.xyz;
    let up = ocean.up.xyz;
    let sea_radius = ocean.sub_point_rel.w;
    // Parabolic sphere curvature (sagitta) — exact to 2nd order over the patch.
    let sag = (gx * gx + gz * gz) / (2.0 * sea_radius);

    var p = ocean.sub_point_rel.xyz + east * gx + north * gz - up * sag;
    p = p + east * disp.x + north * disp.z + up * disp.y;

    out.clip = frame.view_proj * vec4<f32>(p, 1.0);
    out.world_pos = p;
    out.uv = uv;
    out.fade = fade;
    return out;
}

// ── Sky (shared by dome look + reflection), poseidon sky.js adapted to local up ─

fn sky_color(dir: vec3<f32>) -> vec3<f32> {
    let up = ocean.up.xyz;
    let h = dot(dir, up);
    let t = smoothstep(-0.05, 0.4, h);
    let grad = mix(ocean.sky_horizon.xyz, ocean.sky_zenith.xyz, t);
    let sd = max(dot(dir, frame.sun0_dir.xyz), 0.0);
    let disc = pow(sd, 1200.0) * ocean.sun_color.xyz * 8.0;
    let halo = pow(sd, 7.0) * ocean.sun_color.xyz * 0.35;
    return grad + disc + halo;
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ── Fragment ─────────────────────────────────────────────────────────────────

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let t = sample_field(in.uv);
    let east = ocean.east.xyz;
    let north = ocean.north.xyz;
    let up = ocean.up.xyz;

    // Tangent-space normal → world; flatten toward `up` at the faded edge so the
    // patch meets the smooth shell seamlessly.
    var n_t = t.normal.xyz;
    n_t = mix(vec3<f32>(0.0, 1.0, 0.0), n_t, in.fade);
    let N = normalize(east * n_t.x + up * n_t.y + north * n_t.z);

    let V = normalize(-in.world_pos);   // camera at origin (camera-relative)

    // Fresnel (Schlick, F0 = 0.02).
    let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);

    // Reflected sky (clamp ray into the upper hemisphere about local up).
    var R = reflect(-V, N);
    let ru = dot(R, up);
    R = R - up * min(ru, 0.0) + up * 0.02;   // lift below-horizon rays just above it
    let reflection = sky_color(normalize(R));

    // Screen-space refraction: sample the resolved opaque scene behind the water,
    // distorted by the surface normal. (scene_tex is the pre-water scene color.)
    let res = ocean.screen.xy;
    let suv = in.clip.xy / res;
    let distort = vec2<f32>(N.x, N.z) * ocean.screen.z;
    let refr_uv = clamp(suv + distort, vec2<f32>(0.002), vec2<f32>(0.998));
    let bottom = textureSampleLevel(scene_tex, scene_samp, refr_uv, 0.0).rgb;
    // Tint the refracted bottom toward deep-water colour; fade to clean pass-through
    // at the patch edge so it meets the smooth shell seamlessly.
    let tinted = mix(bottom, ocean.deep_color.xyz, ocean.screen.w);
    var body = mix(bottom, tinted, in.fade);

    // Subsurface scatter on back-lit crests (poseidon) adds a teal glow.
    let H = normalize(-N + frame.sun0_dir.xyz);
    let sss = pow(clamp(dot(V, -H), 0.0, 1.0), 4.0) * ocean.shading.x;
    body = mix(body, ocean.scatter_color.xyz, clamp(sss * in.fade, 0.0, 0.5));

    var water = mix(body, reflection, fresnel);

    // Foam from accumulated Jacobian turbulence (already thresholded in the sim → w).
    let coverage = smoothstep(0.2, 0.9, t.disp.w * in.fade);
    let foam_light = 0.55 + 0.6 * clamp(dot(N, frame.sun0_dir.xyz), 0.0, 1.0);
    let foam_shaded = ocean.foam_color.xyz * foam_light;
    water = mix(water, foam_shaded, coverage);

    // Opaque pass (blend off): refraction already carries the see-through, so the
    // wave fully determines the pixel.
    return vec4<f32>(aces(water), 1.0);
}

// ocean_surface.wgsl — unified planet ocean (smooth shell + multi-cascade FFT waves).
//
// One shader, two vertex entries sharing the fragment shading:
//   vs_main  — the camera-anchored FFT wave patch (tangent grid curved onto the
//              sphere, displaced by the summed cascades + a world-anchored amplitude
//              field; the cascade UVs are domain-warped to break tile repetition).
//   vs_shell — the smooth sea-level sphere shell (far / orbit), flat.
// Both share the SAME water shading (refraction + depth-based absorption + Fresnel
// sky reflection), so the shell and the waves match in colour and both darken with
// ocean depth and both refract the scene behind them.
//
// Drawn in the TAA 1× water pass (after begin_water_pass): set 0 carries the
// resolved opaque scene COLOR (binding 3) and DEPTH (binding 5) to sample.

struct Frame {
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

struct OceanTexel {
    disp  : vec4<f32>,   // xyz displacement (m), w raw Jacobian turbulence
    deriv : vec4<f32>,   // dDy/dx, dDy/dz, λ·dDx/dx, λ·dDz/dz
}

struct Ocean {
    east          : vec4<f32>,
    north         : vec4<f32>,
    up            : vec4<f32>,
    sub_point_rel : vec4<f32>,   // xyz = sub_point - camera; w = sea_radius
    cfg           : vec4<f32>,   // patch_half, fft_n, fade_start, cascade_count
    scales        : vec4<f32>,   // per-cascade tile sizes (x,y,z)
    amp           : vec4<f32>,   // noise_freq, amp_lo, amp_hi, debug
    depth_params  : vec4<f32>,   // near, far, extinction, deep_darkness
    deep_color    : vec4<f32>,
    scatter_color : vec4<f32>,
    foam_color    : vec4<f32>,
    sky_horizon   : vec4<f32>,
    sky_zenith    : vec4<f32>,
    sun_color     : vec4<f32>,
    shading       : vec4<f32>,   // sss, foam_threshold, foam_scale, alpha
    screen        : vec4<f32>,   // width, height, refract_strength, deep_tint
    center_rel    : vec4<f32>,   // xyz = planet centre − camera; w = cell factor
}

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> field: array<OceanTexel>;
@group(0) @binding(2) var<uniform> ocean: Ocean;
@group(0) @binding(3) var scene_tex: texture_2d<f32>;
@group(0) @binding(4) var scene_samp: sampler;
@group(0) @binding(5) var scene_depth: texture_2d<f32>;
@group(0) @binding(6) var<storage, read> wind: array<f32>;  // per-vertex wind 0..1

// ── Cascade field sampling (bilinear, wrapping) ────────────────────────────────

fn wrap_index(i: i32, n: i32) -> i32 { return ((i % n) + n) % n; }

fn cascade_scale(c: i32) -> f32 {
    if (c == 0) { return ocean.scales.x; }
    if (c == 1) { return ocean.scales.y; }
    return ocean.scales.z;
}

fn sample_cascade(c: i32, uv: vec2<f32>) -> OceanTexel {
    let n = i32(ocean.cfg.y);
    let base = c * n * n;
    let t = (uv - floor(uv)) * ocean.cfg.y;
    let x0 = i32(floor(t.x)); let y0 = i32(floor(t.y));
    let fx = t.x - floor(t.x); let fy = t.y - floor(t.y);
    let xa = wrap_index(x0, n); let ya = wrap_index(y0, n);
    let xb = wrap_index(x0 + 1, n); let yb = wrap_index(y0 + 1, n);
    let a = field[base + ya * n + xa];
    let b = field[base + ya * n + xb];
    let cc = field[base + yb * n + xa];
    let d = field[base + yb * n + xb];
    var o: OceanTexel;
    o.disp  = mix(mix(a.disp, b.disp, fx),  mix(cc.disp, d.disp, fx),  fy);
    o.deriv = mix(mix(a.deriv, b.deriv, fx), mix(cc.deriv, d.deriv, fx), fy);
    return o;
}

// ── Tile blending — breaks the FFT tile repeat without distorting waves ─────────
// Each blend cell rotates + offsets the sample coordinate; 4 cells are bilinearly
// blended (and the returned horizontal vectors rotated back into the grid frame),
// so no region reads as a single repeating tile, and the rotations read as
// crossing / multi-directional seas.

const BLEND_M: f32 = 240.0; // blend-cell size (m); randomised rotation/offset per cell

fn hash2(p: vec2<f32>) -> vec2<f32> {
    let q = vec2<f32>(dot(p, vec2<f32>(127.1, 311.7)), dot(p, vec2<f32>(269.5, 183.3)));
    return fract(sin(q) * 43758.5453);
}

fn cell_rot(cell: vec2<f32>) -> mat2x2<f32> {
    let a = hash2(cell).x * 6.2831853;
    let c = cos(a); let s = sin(a);
    return mat2x2<f32>(c, s, -s, c);
}

struct Blend { c0: vec2<f32>, c1: vec2<f32>, c2: vec2<f32>, c3: vec2<f32>, w: vec4<f32> }

fn blend_cells(grid: vec2<f32>) -> Blend {
    let g = grid / BLEND_M;
    let gi = floor(g);
    let gf = g - gi;
    var b: Blend;
    b.c0 = gi;
    b.c1 = gi + vec2<f32>(1.0, 0.0);
    b.c2 = gi + vec2<f32>(0.0, 1.0);
    b.c3 = gi + vec2<f32>(1.0, 1.0);
    b.w = vec4<f32>((1.0 - gf.x) * (1.0 - gf.y), gf.x * (1.0 - gf.y),
                    (1.0 - gf.x) * gf.y, gf.x * gf.y);
    return b;
}

fn sample_blend(c: i32, grid: vec2<f32>, b: Blend) -> OceanTexel {
    let tile = cascade_scale(c);
    var cells = array<vec2<f32>, 4>(b.c0, b.c1, b.c2, b.c3);
    var disp = vec4<f32>(0.0);
    var deriv = vec4<f32>(0.0);
    for (var k = 0; k < 4; k = k + 1) {
        let cell = cells[k];
        let r = cell_rot(cell);
        let off = hash2(cell + vec2<f32>(19.7, 4.3)) * 2048.0;
        let tx = sample_cascade(c, (r * grid + off) / tile);
        let rt = transpose(r); // rotate the sampled horizontal vectors back to grid frame
        let dh = rt * vec2<f32>(tx.disp.x, tx.disp.z);
        let sl = rt * vec2<f32>(tx.deriv.x, tx.deriv.y);
        disp = disp + b.w[k] * vec4<f32>(dh.x, tx.disp.y, dh.y, tx.disp.w);
        deriv = deriv + b.w[k] * vec4<f32>(sl.x, sl.y, tx.deriv.z, tx.deriv.w);
    }
    var o: OceanTexel;
    o.disp = disp;
    o.deriv = deriv;
    return o;
}

// ── Vertex ─────────────────────────────────────────────────────────────────────

struct VsIn { @location(0) position: vec3<f32>, @location(1) normal: vec3<f32> }

struct VsOut {
    @builtin(position) clip      : vec4<f32>,
    @location(0)       world_pos : vec3<f32>,
    @location(1)       grid      : vec2<f32>,
    @location(2)       fade      : f32,
    @location(3)       wdir      : vec3<f32>,
    @location(4)       wind      : f32,   // wind strength 0..1 (drives amp + intensity)
}

// Projected grid: each vertex is a screen-space lattice point already ray-cast
// onto the sea sphere on the CPU (camera-relative). Derive its local frame from
// the sphere normal + the sub-point tangent, displace by the FFT (faded by the
// ~screen-uniform cell size), and shade as a wave.
@vertex
fn vs_projected(v: VsIn, @builtin(vertex_index) vi: u32) -> VsOut {
    var out: VsOut;
    let base = v.position;                                   // on the sea sphere, cam-relative
    let up = normalize(base - ocean.center_rel.xyz);         // sphere normal at this point
    let east = ocean.east.xyz; let north = ocean.north.xyz;
    let rel = base - ocean.sub_point_rel.xyz;
    let grid = vec2<f32>(dot(rel, east), dot(rel, north));   // tangent coords for FFT
    let cell = length(base) * ocean.center_rel.w;            // ~screen-uniform world cell
    let ws = wind[vi];                                       // wind strength 0..1 at this vertex
    let amp = mix(ocean.amp.y, ocean.amp.z, ws);            // wind drives wave height
    let b = blend_cells(grid);
    let cc = i32(ocean.cfg.w);

    var disp = vec3<f32>(0.0);
    for (var c = 0; c < 3; c = c + 1) {
        if (c >= cc) { break; }
        let tile = cascade_scale(c);
        let w = 1.0 - smoothstep(tile * 0.4, tile * 0.8, cell);
        disp = disp + sample_blend(c, grid, b).disp.xyz * w;
    }
    disp = disp * amp;

    let p = base + east * disp.x + north * disp.z + up * disp.y;
    out.clip = frame.view_proj * vec4<f32>(p, 1.0);
    out.world_pos = p;
    out.grid = grid;
    out.fade = 1.0;
    out.wdir = up;
    out.wind = ws;
    return out;
}

// ── Shading ────────────────────────────────────────────────────────────────────

// `gloss` (1 near → 0 far) gates the sharp mirror sun disc: a near-mirror surface
// turns the sun into sparkly dots wherever a per-pixel normal lines up, so the
// sharp disc is kept only where the surface is actually resolved; the broad sky
// gradient + soft sun glow always remain.
fn sky_color(dir: vec3<f32>, gloss: f32) -> vec3<f32> {
    let h = dot(dir, ocean.up.xyz);
    let grad = mix(ocean.sky_horizon.xyz, ocean.sky_zenith.xyz, smoothstep(-0.05, 0.4, h));
    let sd = max(dot(dir, frame.sun0_dir.xyz), 0.0);
    let disc = pow(sd, 1200.0) * 8.0 * gloss;
    let glow = pow(sd, 7.0) * 0.35;
    return grad + (disc + glow) * ocean.sun_color.xyz;
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn heat(t: f32) -> vec3<f32> {
    let x = clamp(t, 0.0, 1.0);
    return clamp(vec3<f32>(1.5 - abs(4.0 * x - 3.0), 1.5 - abs(4.0 * x - 2.0), 1.5 - abs(4.0 * x - 1.0)),
                 vec3<f32>(0.0), vec3<f32>(1.0));
}

fn view_dist(d: f32) -> f32 {
    let n = ocean.depth_params.x; let f = ocean.depth_params.y;
    return n * f / (n + d * (f - n)); // reversed-Z (d=1 near, d=0 far)
}

// Shared water body: screen-space refraction + depth-based absorption + Fresnel sky.
// `fp` = per-pixel world footprint (m); it roughens the specular with distance.
fn shade_water(world_pos: vec3<f32>, N: vec3<f32>, fade: f32, frag_xy: vec2<f32>, frag_depth: f32, fp: f32) -> vec3<f32> {
    let up = ocean.up.xyz;
    let V = normalize(-world_pos);
    let fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(N, V), 0.0), 5.0);

    // Specular gloss: sharp only where the surface is resolved (small footprint),
    // so distant / zoomed-out water reflects the smooth sky, not a sparkly sun.
    let gloss = 1.0 - smoothstep(4.0, 30.0, fp);
    var R = reflect(-V, N);
    R = R - up * min(dot(R, up), 0.0) + up * 0.02;
    let reflection = sky_color(normalize(R), gloss);

    let suv = frag_xy / ocean.screen.xy;
    let refr_uv = clamp(suv + vec2<f32>(N.x, N.z) * ocean.screen.z, vec2<f32>(0.002), vec2<f32>(0.998));
    let bottom = textureSampleLevel(scene_tex, scene_samp, refr_uv, 0.0).rgb;

    // Depth-based absorption. `column` is the water thickness ALONG the view ray;
    // project it toward vertical so the darkening tracks how DEEP the ocean is, not
    // the view angle (grazing is masked by Fresnel reflection anyway).
    let scene_d = textureSampleLevel(scene_depth, scene_samp, refr_uv, 0.0).r;
    let column = max(view_dist(scene_d) - view_dist(frag_depth), 0.0);
    let vdepth = column * max(abs(dot(V, up)), 0.25);
    let trans = exp(-vdepth * ocean.depth_params.z); // 1 shallow → 0 deep

    // Water's own colour by depth (deep blue → teal shallow); the refracted seabed
    // shows through where the water is shallow/clear (depth_params.w = clarity).
    let watercol = mix(ocean.deep_color.xyz, ocean.scatter_color.xyz, trans);
    var body = mix(watercol, bottom, trans * ocean.depth_params.w);

    // Subsurface scatter on back-lit crests.
    let H = normalize(-N + frame.sun0_dir.xyz);
    let sss = pow(clamp(dot(V, -H), 0.0, 1.0), 4.0) * ocean.shading.x;
    body = mix(body, ocean.scatter_color.xyz, clamp(sss * fade, 0.0, 0.3));

    return mix(body, reflection, fresnel);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Per-pixel world footprint (m), computed in uniform control flow (drives the
    // detail + specular roughening with distance).
    let fp = max(length(dpdx(in.world_pos)), length(dpdy(in.world_pos)));

    // Debug: heat-map the wind field that drives wave height + intensity.
    if (ocean.amp.w > 0.5) {
        return vec4<f32>(heat(in.wind) * (0.4 + 0.6 * in.fade), 1.0);
    }

    // Sum cascade derivatives (→ normal) + foam. Tile-blended (breaks the repeat),
    // wind-driven amplitude, and faded by the per-pixel screen FOOTPRINT (world
    // metres per pixel) so far / zoomed-out water doesn't alias into white speckle.
    let cc = i32(ocean.cfg.w);
    let amp = mix(ocean.amp.y, ocean.amp.z, in.wind);
    let b = blend_cells(in.grid);
    var deriv = vec4<f32>(0.0);
    var foam = 0.0;
    for (var c = 0; c < 3; c = c + 1) {
        if (c >= cc) { break; }
        let tile = cascade_scale(c);
        let w = 1.0 - smoothstep(tile * 0.4, tile * 0.9, fp);
        let tx = sample_blend(c, in.grid, b);
        deriv = deriv + tx.deriv * w;
        if (c < cc - 1) { foam = foam + clamp((ocean.shading.y - tx.disp.w) * ocean.shading.z, 0.0, 1.0) * w; }
    }
    let east = ocean.east.xyz; let north = ocean.north.xyz; let up = ocean.up.xyz;
    let slope_x = amp * deriv.x / (1.0 + deriv.z);
    let slope_z = amp * deriv.y / (1.0 + deriv.w);
    var n_t = normalize(vec3<f32>(-slope_x, 1.0, -slope_z));
    n_t = mix(vec3<f32>(0.0, 1.0, 0.0), n_t, in.fade);
    let N = normalize(east * n_t.x + up * n_t.y + north * n_t.z);

    var water = shade_water(in.world_pos, N, in.fade, in.clip.xy, in.clip.z, fp);

    // Foam is a near-field detail only — fade it out entirely as pixels cover
    // more ocean, so zoomed-out water shows no whitecaps. (Tune the 8..35 m
    // footprint band for how close you must be to see foam.)
    let foam_fade = 1.0 - smoothstep(8.0, 35.0, fp);
    let coverage = smoothstep(0.2, 0.9, foam * clamp(amp, 0.0, 1.3) * in.fade) * foam_fade;
    let foam_light = 0.55 + 0.6 * clamp(dot(N, frame.sun0_dir.xyz), 0.0, 1.0);
    water = mix(water, ocean.foam_color.xyz * foam_light, coverage);

    return vec4<f32>(aces(water), 1.0);
}

// clouds.wgsl — wind-driven volumetric clouds (fullscreen raymarch).
//
// A spherical cloud shell [R+base, R+base+thickness] ray-marched in a fullscreen
// pass: the VS emits an oversized triangle via vertex_index (no vertex buffers),
// the FS reconstructs a camera-relative ray, intersects the shell, and marches
// analytic Perlin-Worley-ish noise advected by the planet's wind. Lighting is
// Beer + Henyey-Greenstein + a short sun light-march + Hillaire multiscatter
// octaves (energy-conserving integration), sun/ambient driven by FrameUniforms.
// Terrain occlusion samples the resolved opaque depth (reversed-Z), no inverse
// matrix needed. Camera at origin (camera-relative); planet centre = center_rel.
//
// Pipeline contract: custom set0 — 0 FrameUniforms UBO, 1 CloudParams UBO,
// 2 scene depth (SAMPLED), 3 sampler. Vertex-pulling pipeline (depth GREATER,
// depth-write OFF, alpha-blend). Drawn inside the water-split 1× instance.
//
// ponytail: analytic noise (no 3D texture — enki-rhi can't upload one), single
// Worley octave, transmittance-only empty-space skip. Upgrades: detail octaves,
// curl turbulence, powder, half-res + cloud reprojection. See docs/clouds-research.md.

const PI: f32 = 3.14159265359;

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

struct CloudParams {
    right     : vec4<f32>, // camera right (xyz)
    up        : vec4<f32>, // camera up (xyz)
    fwd       : vec4<f32>, // camera forward (xyz)
    center_rel: vec4<f32>, // xyz = planet centre − camera (camera-relative); w = planet radius R
    shell     : vec4<f32>, // base_alt_m, thickness_m, coverage, density(extinction/m)
    wind      : vec4<f32>, // wind dir xyz (world tangent, unit); w = advection speed (m/s)
    march     : vec4<f32>, // steps, hg_g, noise_scale(m), time(s)
    screen    : vec4<f32>, // width, height, tan_half_fov, aspect
    misc      : vec4<f32>, // debug, proj_a, proj_b, _pad
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<uniform> cloud: CloudParams;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_samp: sampler;

// ── Hashes (Dave Hoskins) ───────────────────────────────────────────────────
fn hash13(p3in: vec3<f32>) -> f32 {
    var p3 = fract(p3in * 0.1031);
    p3 += dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}

fn hash33(p3in: vec3<f32>) -> vec3<f32> {
    var p3 = fract(p3in * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 += dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}

// ── Value noise + fBm (cloud base shape) ──────────────────────────────────────
fn vnoise(x: vec3<f32>) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let c000 = hash13(i + vec3<f32>(0.0, 0.0, 0.0));
    let c100 = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = hash13(i + vec3<f32>(0.0, 1.0, 0.0));
    let c110 = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = hash13(i + vec3<f32>(0.0, 0.0, 1.0));
    let c101 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = hash13(i + vec3<f32>(0.0, 1.0, 1.0));
    let c111 = hash13(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, u.x);
    let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x);
    let x11 = mix(c011, c111, u.x);
    let y0 = mix(x00, x10, u.y);
    let y1 = mix(x01, x11, u.y);
    return mix(y0, y1, u.z);
}

fn fbm(p0: vec3<f32>) -> f32 {
    var p = p0;
    var amp = 0.5;
    var sum = 0.0;
    for (var i = 0; i < 3; i = i + 1) {
        sum += amp * vnoise(p);
        p *= 2.02;
        amp *= 0.5;
    }
    return sum / 0.875; // normalize (0.5+0.25+0.125) → ~[0,1]
}

// ── Worley F1 (detail erosion) ────────────────────────────────────────────────
fn worley(x: vec3<f32>) -> f32 {
    let ip = floor(x);
    let fp = fract(x);
    var min_d = 1.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let g = vec3<f32>(f32(dx), f32(dy), f32(dz));
                let o = hash33(ip + g);
                let r = g + o - fp;
                min_d = min(min_d, dot(r, r));
            }
        }
    }
    return sqrt(min_d); // ~[0,1]
}

fn remap(v: f32, a: f32, b: f32, c: f32, d: f32) -> f32 {
    return c + (v - a) * (d - c) / (b - a);
}

fn hg(cos_t: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(max(1.0 + g2 - 2.0 * g * cos_t, 1e-4), 1.5));
}

// Interleaved gradient noise, animated per frame → breaks march banding.
fn ign(px: vec2<f32>, frame_n: f32) -> f32 {
    let p = px + 5.588238 * fract(frame_n * 0.6180339887);
    return fract(52.9829189 * fract(0.06711056 * p.x + 0.00583715 * p.y));
}

// Cloud density at a planet-centric position; `hfrac` is the 0..1 height in the shell.
fn density_at(pos: vec3<f32>, hfrac: f32) -> f32 {
    let inv = 1.0 / cloud.march.z; // 1 / noise_scale (m)
    let t = cloud.march.w;
    let ws = cloud.wind.w;
    let wdir = cloud.wind.xyz;

    // Wind: base scrolls slowly; height-skew makes tops lead bases (wind shear).
    let base_p = (pos + wdir * (t * ws) + wdir * (hfrac * cloud.march.z * 0.6)) * inv;
    var shape = fbm(base_p);

    // Height gradient window → flat bottoms, defined tops (density→0 at floor/ceiling).
    let grad = clamp(hfrac / 0.15, 0.0, 1.0) * clamp((1.0 - hfrac) / 0.4, 0.0, 1.0);
    shape *= grad;

    // Coverage gate, then re-multiply by coverage (the bit ki omits → rounded edges).
    let cov = cloud.shell.z;
    var d = clamp(remap(shape, 1.0 - cov, 1.0, 0.0, 1.0), 0.0, 1.0) * cov;
    if (d <= 0.0) { return 0.0; }

    // Detail erosion (edge-only): faster-scrolling Worley + a small upward boil so
    // the edges churn instead of sliding rigidly (ki's "decal" failure).
    let det_p = (pos + wdir * (t * ws * 1.8) + vec3<f32>(0.0, 1.0, 0.0) * (t * ws * 0.5)) * (inv * 4.0);
    let er = worley(det_p);
    d = clamp(remap(d, er * 0.3, 1.0, 0.0, 1.0), 0.0, 1.0);
    return d;
}

// Optical depth toward the sun (short light-march).
fn light_optical_depth(pos: vec3<f32>, sun_dir: vec3<f32>, ri: f32, ro: f32) -> f32 {
    let ls = (ro - ri) * 0.12;
    var lp = pos;
    var acc = 0.0;
    for (var j = 0; j < 5; j = j + 1) {
        lp += sun_dir * ls;
        let r = length(lp);
        let hf = (r - ri) / (ro - ri);
        if (hf >= 0.0 && hf <= 1.0) {
            acc += density_at(lp, hf) * cloud.shell.w * ls;
        }
    }
    return acc;
}

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0)       ndc:      vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
    // Oversized fullscreen triangle: (-1,-1), (3,-1), (-1,3). CCW (front-facing)
    // in the positive-height Vulkan viewport, so cull-BACK keeps it. z=1 (reversed-Z
    // near) → passes depth GREATER everywhere; depth-write is off.
    var out: VsOut;
    // vid 0 → (-1,-1), 1 → (-1,3), 2 → (3,-1). This winding is front-facing under
    // cull-BACK: we bypass the projection's -f Y-flip, so only the framebuffer's
    // Y-down flip applies (terrain gets both → they cancel). Math-CW here = CCW in
    // the Y-down framebuffer = front. (Flip vid1/vid2 if it ever culls to nothing.)
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
    let tan_half = cloud.screen.z;
    let aspect = cloud.screen.w;

    // Camera-relative ray (camera at origin). ndc.y=-1 is screen-top → +up.
    let dir = normalize(
        cloud.fwd.xyz
        + cloud.right.xyz * (ndc.x * tan_half * aspect)
        + cloud.up.xyz * (-ndc.y * tan_half)
    );

    let c = cloud.center_rel.xyz;       // planet centre relative to camera
    let big_r = cloud.center_rel.w;
    let ri = big_r + cloud.shell.x;     // inner cloud radius
    let ro = ri + cloud.shell.y;        // outer cloud radius

    // Ray/sphere against inner & outer shells (origin = 0). oc = -c.
    let oc = -c;
    let b = dot(dir, oc);
    let c_out = dot(oc, oc) - ro * ro;
    let c_in = dot(oc, oc) - ri * ri;
    let disc_out = b * b - c_out;
    if (disc_out < 0.0) { return vec4<f32>(0.0); } // ray misses the cloud layer
    let sq_out = sqrt(disc_out);
    let to0 = -b - sq_out;
    let to1 = -b + sq_out;
    if (to1 <= 0.0) { return vec4<f32>(0.0); }     // layer entirely behind camera

    var t_enter = max(to0, 0.0);
    var t_exit = to1;
    let disc_in = b * b - c_in;
    if (disc_in > 0.0) {
        let sq_in = sqrt(disc_in);
        let ti0 = -b - sq_in;
        let ti1 = -b + sq_in;
        if (ti0 > t_enter) {
            t_exit = min(t_exit, ti0);          // near band: stop at inner shell (planet ahead)
        } else if (ti1 > t_enter) {
            t_enter = max(t_enter, ti1);        // we're in the hollow → cloud starts past it
        }
    }

    // Terrain occlusion: clamp the march to the opaque surface (reversed-Z, no inverse
    // matrix — solve forward distance from depth, divide by the ray's forward cosine).
    let uv = ndc * 0.5 + 0.5;
    let scene_d = textureSampleLevel(depth_tex, depth_samp, uv, 0.0).r;
    if (scene_d > 0.0) {
        let fwd_dist = cloud.misc.z / (scene_d + cloud.misc.y); // proj_b / (d + proj_a) = -z_view
        let t_scene = fwd_dist / max(dot(dir, cloud.fwd.xyz), 1e-3);
        t_exit = min(t_exit, t_scene);
    }

    if (t_exit <= t_enter) { return vec4<f32>(0.0); }

    let sun_dir = normalize(frame.sun0_dir.xyz);
    let cos_t = dot(dir, sun_dir);
    let g = cloud.march.y;
    let steps = max(i32(cloud.march.x), 1);
    let seg = t_exit - t_enter;
    let dt = seg / f32(steps);

    let px = uv * cloud.screen.xy;
    let frame_n = floor(cloud.march.w * 60.0);
    let jitter = ign(px, frame_n);

    let sun_col = frame.sun0_color.xyz;
    let amb = frame.hemi_sky.xyz * 0.5 + frame.hemi_ground.xyz * 0.1;

    var transmit = 1.0;
    var scattered = vec3<f32>(0.0);
    var t = t_enter + dt * jitter;
    for (var i = 0; i < steps; i = i + 1) {
        if (transmit < 0.01) { break; }
        let pos = dir * t - c;          // planet-centric sample position
        let r = length(pos);
        let hfrac = (r - ri) / (ro - ri);
        if (hfrac >= 0.0 && hfrac <= 1.0) {
            let dens = density_at(pos, hfrac);
            if (dens > 0.0) {
                let sigma_t = dens * cloud.shell.w;
                let tau = light_optical_depth(pos, sun_dir, ri, ro);
                // Hillaire multiscatter octaves (a=b=c=0.5): cheap soft interior glow.
                var msum = 0.0;
                var oa = 1.0; var ob = 1.0; var oc2 = 1.0;
                for (var n = 0; n < 3; n = n + 1) {
                    msum += ob * exp(-tau * oa) * hg(cos_t, g * oc2);
                    oa *= 0.5; ob *= 0.5; oc2 *= 0.5;
                }
                let lum = sun_col * msum + amb;
                let step_t = exp(-sigma_t * dt);
                // energy-conserving integration (step-count independent)
                scattered += transmit * (1.0 - step_t) * lum;
                transmit *= step_t;
            }
        }
        t += dt;
    }

    let alpha = clamp(1.0 - transmit, 0.0, 1.0);

    if (cloud.misc.x > 0.5) {
        // Debug "Density" view: opaque coverage heatmap.
        return vec4<f32>(alpha, alpha * 0.6, 1.0 - alpha, 1.0);
    }

    if (alpha <= 0.001) { return vec4<f32>(0.0); }
    // Un-premultiply for SRC_ALPHA blend (hardware re-multiplies by alpha).
    return vec4<f32>(scattered / alpha, alpha);
}

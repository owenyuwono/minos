// clouds.wgsl — wind-driven volumetric clouds (fullscreen raymarch).
//
// A spherical cloud shell [R+base, R+base+thickness] ray-marched in a fullscreen
// pass: the VS emits an oversized triangle via vertex_index (no vertex buffers),
// the FS reconstructs a camera-relative ray, intersects the shell, and marches
// analytic Perlin-Worley-ish noise advected by the planet's wind. Lighting is
// Beer + Henyey-Greenstein + a short sun light-march + Hillaire multiscatter
// octaves (energy-conserving integration), powder dark-edge, sun/ambient from
// FrameUniforms. Coverage is gated by the baked climate **moisture** field
// (binding 4, equirect). Terrain occlusion samples the resolved opaque depth
// (reversed-Z), no inverse matrix needed. Camera at origin (camera-relative);
// planet centre = center_rel.
//
// Pipeline contract: custom set0 — 0 FrameUniforms UBO, 1 CloudParams UBO,
// 2 scene depth (SAMPLED), 3 sampler, 4 moisture grid (STORAGE). Vertex-pulling
// pipeline (depth GREATER, depth-write OFF, alpha-blend). Drawn in the water-split
// 1× instance, AFTER the ocean (clouds are a shell above sea level).
//
// ponytail: analytic noise (enki-rhi can't upload a 3D texture), single Worley
// octave, pseudo-curl turbulence (not Bridson divergence-free), empty-space skip
// via a cheap-density gate (no variable stepping), full-res. Upgrades: true curl,
// half-res + cloud reprojection. See docs/clouds-research.md.

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
    wind      : vec4<f32>, // w = advection speed scale (m/s); xyz unused (wind is per-sample now)
    march     : vec4<f32>, // steps, hg_g, noise_scale(m), time(s)
    screen    : vec4<f32>, // width, height, tan_half_fov, aspect
    misc      : vec4<f32>, // debug, proj_a, proj_b, _pad
    extra     : vec4<f32>, // cloud_type, powder, curl, moisture_influence
    grid      : vec4<f32>, // grid_w, grid_h, has_coverage(0/1), _pad
}

@group(0) @binding(0) var<uniform> frame: FrameUniforms;
@group(0) @binding(1) var<uniform> cloud: CloudParams;
@group(0) @binding(2) var depth_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_samp: sampler;
// Advected cloud COVERAGE 0..1, per equirect cell — evolved by the semi-Lagrangian
// worker (clouds_advect.rs). Clouds translate + merge by THIS field moving, not by
// warping noise. 1 f32 per cell, bilinearly sampled (matches the worker's bilinear).
@group(0) @binding(4) var<storage, read> coverage_grid: array<f32>;
// Per-location wind VELOCITY (world tangent × speed), 3 f32/cell — the SAME local
// wind the streaks follow. The cloud texture advects along this so it curves with
// the flow (not a global vector). Shear bounded by the reset-blend in density_at.
@group(0) @binding(5) var<storage, read> wind_grid: array<f32>;

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

// Pseudo-curl turbulence (cheap noise vector, NOT divergence-free Bridson curl).
fn noise3(p: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(vnoise(p), vnoise(p + 19.19), vnoise(p + 47.77)) * 2.0 - vec3<f32>(1.0);
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

// Equirect texel coords for a direction (CPU bake uses the matching
// dir = (cosLat·sinLon, sinLat, cosLat·cosLon)). Returns the 4 bilinear corner
// indices (longitude wraps, latitude clamps) + the lerp weights, shared by both
// grid samplers so the fields are CONTINUOUS — nearest sampling makes the wind
// jump per cell, which shows as a grid once advection (wind×time) is large.
struct GridLerp { ix0: i32, ix1: i32, iy0: i32, iy1: i32, tx: f32, ty: f32, gw: i32 }

fn grid_lerp(d: vec3<f32>) -> GridLerp {
    let lon = atan2(d.x, d.z);
    let lat = asin(clamp(d.y, -1.0, 1.0));
    let gw = i32(cloud.grid.x);
    let gh = i32(cloud.grid.y);
    let fx = (lon / (2.0 * PI) + 0.5) * f32(gw) - 0.5;
    let fy = (lat / PI + 0.5) * f32(gh) - 0.5;
    let x0 = i32(floor(fx));
    let y0 = i32(floor(fy));
    var o: GridLerp;
    o.ix0 = ((x0 % gw) + gw) % gw;     // longitude wraps
    o.ix1 = (o.ix0 + 1) % gw;
    o.iy0 = clamp(y0, 0, gh - 1);      // latitude clamps at the poles
    o.iy1 = clamp(y0 + 1, 0, gh - 1);
    o.tx = fx - floor(fx);
    o.ty = fy - floor(fy);
    o.gw = gw;
    return o;
}

// Bilinear advected COVERAGE at a direction (the worker's evolving field).
fn sample_coverage(d: vec3<f32>) -> f32 {
    let g = grid_lerp(d);
    let c00 = coverage_grid[g.iy0 * g.gw + g.ix0];
    let c10 = coverage_grid[g.iy0 * g.gw + g.ix1];
    let c01 = coverage_grid[g.iy1 * g.gw + g.ix0];
    let c11 = coverage_grid[g.iy1 * g.gw + g.ix1];
    return mix(mix(c00, c10, g.tx), mix(c01, c11, g.tx), g.ty);
}

// Bilinear local wind VELOCITY (world tangent × speed) at a direction.
fn sample_wind(d: vec3<f32>) -> vec3<f32> {
    if (cloud.grid.w < 0.5) { return vec3<f32>(0.0); } // grid not baked yet
    let g = grid_lerp(d);
    let b00 = (g.iy0 * g.gw + g.ix0) * 3;
    let b10 = (g.iy0 * g.gw + g.ix1) * 3;
    let b01 = (g.iy1 * g.gw + g.ix0) * 3;
    let b11 = (g.iy1 * g.gw + g.ix1) * 3;
    let w00 = vec3<f32>(wind_grid[b00], wind_grid[b00 + 1], wind_grid[b00 + 2]);
    let w10 = vec3<f32>(wind_grid[b10], wind_grid[b10 + 1], wind_grid[b10 + 2]);
    let w01 = vec3<f32>(wind_grid[b01], wind_grid[b01 + 1], wind_grid[b01 + 2]);
    let w11 = vec3<f32>(wind_grid[b11], wind_grid[b11 + 1], wind_grid[b11 + 2]);
    return mix(mix(w00, w10, g.tx), mix(w01, w11, g.tx), g.ty);
}

// Effective coverage at a direction: base scale × the advected climate coverage.
// `moisture_influence` blends uniform (0) ↔ full advected climate coverage (1).
fn coverage_at(dir: vec3<f32>) -> f32 {
    var cov = cloud.shell.z;
    if (cloud.grid.z > 0.5) {
        cov = cloud.shell.z * mix(1.0, sample_coverage(dir), cloud.extra.w);
    }
    return cov;
}

// Vertical density window, blended stratus(low,thin) ↔ cumulus(tall,billowy).
fn height_gradient(h: f32, cloud_type: f32) -> f32 {
    let strat = clamp(h / 0.08, 0.0, 1.0) * clamp((0.45 - h) / 0.35, 0.0, 1.0);
    let cumu = clamp(h / 0.20, 0.0, 1.0) * clamp((1.0 - h) / 0.45, 0.0, 1.0);
    return mix(strat, cumu, cloud_type);
}

// Cloud density at a planet-centric position. `warp` (metres) advects the SHAPE noise
// — pass the per-location wind drift so the texture flows along the wind like the
// streaks. `cheap` skips Worley erosion + curl (empty-space scout + sun light-march).
fn density_at(pos: vec3<f32>, hfrac: f32, cheap: bool, warp: vec3<f32>) -> f32 {
    let dir = normalize(pos);
    let cov = coverage_at(dir);
    if (cov <= 0.001) { return 0.0; }

    let inv = 1.0 / cloud.march.z; // 1 / noise_scale (m)
    let base_p = (pos + warp) * inv;
    var shape = fbm(base_p);
    shape *= height_gradient(hfrac, cloud.extra.x);

    // Coverage gate, then re-multiply by coverage (the bit ki omits → rounded edges).
    var d = clamp(remap(shape, 1.0 - cov, 1.0, 0.0, 1.0), 0.0, 1.0) * cov;
    if (d <= 0.0 || cheap) { return d; }

    // Detail erosion (edge-only): drifting Worley + pseudo-curl turbulence.
    var det = (pos + warp) * (inv * 4.0);
    if (cloud.extra.z > 0.0) {
        det += noise3(det * 0.5) * cloud.extra.z;
    }
    let er = worley(det);
    d = clamp(remap(d, er * 0.3, 1.0, 0.0, 1.0), 0.0, 1.0);
    return d;
}

// Optical depth toward the sun (short, cheap light-march; warped consistently).
fn light_optical_depth(pos: vec3<f32>, sun_dir: vec3<f32>, ri: f32, ro: f32, warp: vec3<f32>) -> f32 {
    let ls = (ro - ri) * 0.12;
    var lp = pos;
    var acc = 0.0;
    for (var j = 0; j < 5; j = j + 1) {
        lp += sun_dir * ls;
        let r = length(lp);
        let hf = (r - ri) / (ro - ri);
        if (hf >= 0.0 && hf <= 1.0) {
            acc += density_at(lp, hf, true, warp) * cloud.shell.w * ls;
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
    let powder_amt = cloud.extra.y;
    let steps = max(i32(cloud.march.x), 1);
    let seg = t_exit - t_enter;
    let dt = seg / f32(steps);

    // STATIC ray-start dither (not animated): TAA doesn't denoise this depth-off
    // overlay, so a per-frame-animated dither would flicker. A fixed spatial pattern
    // breaks banding without flicker. (Animate it again only once cloud reprojection
    // lands — see docs/clouds-advection-research.md.)
    let px = uv * cloud.screen.xy;
    let jitter = ign(px, 0.0);

    let sun_col = frame.sun0_color.xyz;
    // Brighter, whiter fill so clouds read as light cloud, not dark grey: stronger
    // sky ambient + a neutral white floor that keeps shadowed/interior cloud pale.
    let amb = frame.hemi_sky.xyz * 1.1 + frame.hemi_ground.xyz * 0.3 + vec3<f32>(0.25);

    var transmit = 1.0;
    var scattered = vec3<f32>(0.0);
    var t = t_enter + dt * jitter;
    for (var i = 0; i < steps; i = i + 1) {
        if (transmit < 0.01) { break; }
        let pos = dir * t - c;          // planet-centric sample position
        let r = length(pos);
        let hfrac = (r - ri) / (ro - ri);
        if (hfrac >= 0.0 && hfrac <= 1.0) {
            // Per-location wind advection (texture flows along the local wind like the
            // streaks), 2-phase reset-blend. The reset period scales INVERSELY with wind
            // speed so the warp DISTANCE is capped (≤0.6 feature) regardless of speed —
            // high wind just resets faster, it never stretches/tears.
            let sdir = normalize(pos);
            let wv = sample_wind(sdir) * cloud.wind.w;        // local velocity (m/s)
            let wlen = max(length(wv), 1e-3);
            let tau = clamp(cloud.march.z * 0.8 / wlen, 2.0, 30.0);
            let tt = cloud.march.w / tau;
            let cyc0 = floor(tt);
            let cyc1 = floor(tt + 0.5);
            let ph0 = tt - cyc0;
            let ph1 = (tt + 0.5) - cyc1;
            // Per-cycle random offset (metres) so each reset shows DIFFERENT noise —
            // the cross-fade reads as clouds EVOLVING, not the same pattern looping.
            // Changes only while the resetting phase has zero blend weight → seamless.
            let dec = cloud.march.z * 120.0;
            let off0 = (hash33(vec3<f32>(cyc0, 7.0, 3.0)) - vec3<f32>(0.5)) * dec;
            let off1 = (hash33(vec3<f32>(cyc1, 1.0, 9.0)) - vec3<f32>(0.5)) * dec;
            let warp0 = -wv * (ph0 * tau) + off0;             // −: move WITH the wind
            let warp1 = -wv * (ph1 * tau) + off1;
            let bl = abs(ph0 - 0.5) * 2.0;
            // Empty-space scout: cheap density first; only do erosion + the sun
            // light-march where there's actually cloud.
            let cheap = density_at(pos, hfrac, true, warp0);
            if (cheap > 0.01) {
                let dens = mix(density_at(pos, hfrac, false, warp0), density_at(pos, hfrac, false, warp1), bl);
                if (dens > 0.0) {
                    let sigma_t = dens * cloud.shell.w;
                    let tau = light_optical_depth(pos, sun_dir, ri, ro, warp0);
                    // Hillaire multiscatter octaves (a=b=c=0.5): soft interior glow.
                    var msum = 0.0;
                    var oa = 1.0; var ob = 1.0; var oc2 = 1.0;
                    for (var n = 0; n < 3; n = n + 1) {
                        msum += ob * exp(-tau * oa) * hg(cos_t, g * oc2);
                        oa *= 0.5; ob *= 0.5; oc2 *= 0.5;
                    }
                    // Powder (Beer-powder dark edge), gated by the slider.
                    let powd = 1.0 - exp(-dens * 2.0);
                    let sun_term = sun_col * msum * mix(1.0, powd, powder_amt);
                    let lum = sun_term + amb;
                    let step_t = exp(-sigma_t * dt);
                    // energy-conserving integration (step-count independent)
                    scattered += transmit * (1.0 - step_t) * lum;
                    transmit *= step_t;
                }
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

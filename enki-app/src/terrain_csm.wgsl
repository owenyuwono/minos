// terrain_csm.wgsl — opaque voxel-terrain pass that RECEIVES the 3-cascade sun CSM.
//
// Variant of enki-render `terrain.wgsl`: identical vertex contract (4×vec3 streams +
// `ChunkPush`, camera-relative, reversed-Z), but set0 is the character-style CSM set
// — a frame UBO carrying the 3 cascade matrices + shadow params, 3 cascade depth
// maps, and a comparison sampler. Mode 0 multiplies the sun term by a 3×3-PCF shadow.
// The `select_cascade`/`sample_shadow` snippet is copied verbatim from `character.wgsl`
// (keep them in lockstep). Bound only when `rhi.has_shadow_map()`; otherwise the voxel
// terrain falls back to the shared set0-only terrain pipeline.

const SHADOW_CASCADES: u32 = 3u; // MUST match SHADOW_CASCADES in voxel_view.rs / main.rs
const DETAIL_FADE_NEAR: f32 = 18.0; // m — ground detail at full strength closer than this
const DETAIL_FADE_FAR:  f32 = 55.0; // m — faded to nothing beyond (kills far-field shimmer)
const POM_STEPS: i32 = 10;          // parallax-occlusion march steps (near-field only)
const POM_INV_STEPS: f32 = 0.1;     // = 1.0 / POM_STEPS

struct Frame {
    view_proj    : mat4x4<f32>,
    camera_pos   : vec4<f32>,             // world camera position (w unused) — day/night gate
    sun0_dir     : vec4<f32>,
    sun0_color   : vec4<f32>,
    sun1_dir     : vec4<f32>,
    sun1_color   : vec4<f32>,
    hemi_sky     : vec4<f32>,
    hemi_ground  : vec4<f32>,
    ambient      : vec4<f32>,
    cascade_vp   : array<mat4x4<f32>, 3>, // camera-relative world → light clip
    shadow_params: vec4<f32>,             // [depth_bias, normal_bias, strength, enabled]
    morph_a      : vec4<f32>,             // CDLOD: [radius, screen_k, split_px, merge_px]
    morph_b      : vec4<f32>,             // [morph_region, _, _, _]
}

@group(0) @binding(0) var<uniform> frame : Frame;
@group(0) @binding(1) var shadow_map0 : texture_depth_2d;
@group(0) @binding(2) var shadow_map1 : texture_depth_2d;
@group(0) @binding(3) var shadow_map2 : texture_depth_2d;
@group(0) @binding(4) var shadow_samp : sampler_comparison;

struct ChunkPush {
    model         : mat4x4<f32>,
    material_mode : u32,
    dbg_id        : u32,   // per-leaf id (modes 3/4)
    dbg_level     : u32,   // LOD level (mode 5)
    _pad2         : u32,
}
var<immediate> pc: ChunkPush;

struct VertexIn {
    @builtin(vertex_index) vid: u32,
    @location(0) position   : vec3<f32>,
    @location(1) normal     : vec3<f32>,
    @location(2) color      : vec3<f32>,
    @location(3) plate_color: vec3<f32>,
    @location(4) morph_disp : vec3<f32>, // CDLOD displacement toward the parent LOD surface
}
struct VertexOut {
    @builtin(position) clip_pos    : vec4<f32>,
    @location(0)       world_normal: vec3<f32>,
    @location(1)       albedo      : vec3<f32>,
    @location(2)       world_pos   : vec3<f32>, // camera-relative (light cascades expect this)
    @location(3)       plate_color : vec3<f32>,
    @location(4) @interpolate(flat) vid : u32,  // provoking-vertex seed (Triangle view)
}

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    // CDLOD geomorph: lerp the vertex toward the parent-LOD surface (carried per-vertex
    // in morph_disp) by camera distance, so coarse↔fine LOD swaps are continuous (no pop).
    // The band maps a leaf's [split, merge] screen-threshold distances (from its level +
    // the screen factors in morph_a) to morph 0→1. length(world_un) is camera distance
    // because rendering is camera-relative (camera at the origin).
    let world_un = pc.model * vec4<f32>(v.position, 1.0);
    let radius   = frame.morph_a.x;
    let k        = frame.morph_a.y;   // screen_h / (2·tan(fov/2))
    let split_px = frame.morph_a.z;
    let region   = frame.morph_b.x;
    let ns   = 1.5707963267 * radius / exp2(f32(pc.dbg_level)); // node size (m)
    // Both bounds key off split_px so the morph reaches 1 exactly at the distance this
    // leaf APPEARED (its parent split = 2·ns_self distance): then finer children appear
    // fully morphed-to-parent → the split is continuous. `near` is where this leaf in
    // turn gives way to its own children (full detail, morph 0). far = 2·near.
    let near = ns * k / split_px;
    let far  = 2.0 * near;
    let lo   = far - region * (far - near);
    let morph = smoothstep(lo, far, length(world_un.xyz));
    let world_pos = pc.model * vec4<f32>(v.position + morph * v.morph_disp, 1.0);
    out.clip_pos = frame.view_proj * world_pos;
    let m3 = mat3x3<f32>(pc.model[0].xyz, pc.model[1].xyz, pc.model[2].xyz);
    out.world_normal = m3 * v.normal;
    out.albedo = v.color;
    out.world_pos = world_pos.xyz; // camera-relative — same space as cascade_vp
    out.plate_color = v.plate_color;
    out.vid = v.vid;
    return out;
}

// Geometry-debug colouring (modes 3/4/5) — identical to enki-render terrain.wgsl.
fn hash_u32(x: u32) -> u32 {
    var h = x * 0x9E3779B1u; h = h ^ (h >> 16u); h = h * 0x85EBCA77u; h = h ^ (h >> 13u); return h;
}
fn hash2(a: u32, b: u32) -> u32 {
    var h = a * 0x9E3779B1u ^ b * 0x85EBCA77u; h = h ^ (h >> 15u); return h;
}
fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s; let hp = h / 60.0; let x = c * (1.0 - abs(hp % 2.0 - 1.0));
    var rgb: vec3<f32>; let hi = i32(hp);
    if (hi == 0) { rgb = vec3<f32>(c, x, 0.0); }
    else if (hi == 1) { rgb = vec3<f32>(x, c, 0.0); }
    else if (hi == 2) { rgb = vec3<f32>(0.0, c, x); }
    else if (hi == 3) { rgb = vec3<f32>(0.0, x, c); }
    else if (hi == 4) { rgb = vec3<f32>(x, 0.0, c); }
    else { rgb = vec3<f32>(c, 0.0, x); }
    return rgb + vec3<f32>(v - c);
}
fn id_color(id: u32) -> vec3<f32> {
    let h = hash_u32(id);
    let hue = f32(h % 3600u) / 10.0;
    let sat = 0.55 + f32((h >> 8u) & 0xFFu) / 255.0 * 0.30;
    let val = 0.80 + f32((h >> 16u) & 0xFFu) / 255.0 * 0.20;
    return hsv2rgb(hue, sat, val);
}

fn hemisphere_ambient(n: vec3<f32>, sky: vec3<f32>, ground: vec3<f32>) -> vec3<f32> {
    let t = n.y * 0.5 + 0.5;
    return mix(ground, sky, t);
}
fn directional_diffuse(n: vec3<f32>, light_dir: vec3<f32>, color: vec3<f32>) -> vec3<f32> {
    let n_dot_l = max(dot(n, light_dir), 0.0);
    return color * n_dot_l;
}
fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51; let b = 0.03; let c = 2.43; let d = 0.59; let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// --- CSM receiver (verbatim from character.wgsl; keep in lockstep) ---------------
fn select_cascade(world_rel: vec3<f32>, n: vec3<f32>, uv_out: ptr<function, vec2<f32>>, ref_out: ptr<function, f32>) -> u32 {
    for (var i = 0u; i < SHADOW_CASCADES; i = i + 1u) {
        let scale = f32(1u << i);
        let p = world_rel + n * (frame.shadow_params.y * scale);
        let clip = frame.cascade_vp[i] * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) { continue; }
        let ndc = clip.xyz / clip.w;
        let u = ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);
        let m = 0.02;
        if (u.x > m && u.x < 1.0 - m && u.y > m && u.y < 1.0 - m && ndc.z > 0.0 && ndc.z < 1.0) {
            *uv_out = u;
            *ref_out = ndc.z + frame.shadow_params.x * scale;
            return i;
        }
    }
    return SHADOW_CASCADES;
}
fn sample_shadow(world_rel: vec3<f32>, n: vec3<f32>) -> f32 {
    if (frame.shadow_params.w < 0.5) { return 1.0; }
    var uv = vec2<f32>(0.0, 0.0);
    var ref_depth = 0.0;
    let c = select_cascade(world_rel, n, &uv, &ref_depth);
    if (c >= SHADOW_CASCADES) { return 1.0; }
    let texel = 1.0 / f32(textureDimensions(shadow_map0).x);
    var sum = 0.0;
    for (var dy = -1; dy <= 1; dy = dy + 1) {
        for (var dx = -1; dx <= 1; dx = dx + 1) {
            let o = uv + vec2<f32>(f32(dx), f32(dy)) * texel;
            switch (c) {
                case 0u: { sum = sum + textureSampleCompareLevel(shadow_map0, shadow_samp, o, ref_depth); }
                case 1u: { sum = sum + textureSampleCompareLevel(shadow_map1, shadow_samp, o, ref_depth); }
                default: { sum = sum + textureSampleCompareLevel(shadow_map2, shadow_samp, o, ref_depth); }
            }
        }
    }
    return sum / 9.0;
}

// ── Procedural ground detail: 3D value noise + Worley (lifted from clouds.wgsl) ───
// Pure ALU (no bindings), so the per-pixel detail normal needs no texture upload.
fn hash13(p3in: vec3<f32>) -> f32 {
    var p3 = fract(p3in * 0.1031);
    p3 = p3 + dot(p3, p3.zyx + 31.32);
    return fract((p3.x + p3.y) * p3.z);
}
fn hash33(p3in: vec3<f32>) -> vec3<f32> {
    var p3 = fract(p3in * vec3<f32>(0.1031, 0.1030, 0.0973));
    p3 = p3 + dot(p3, p3.yxz + 33.33);
    return fract((p3.xxy + p3.yxx) * p3.zyx);
}
fn vnoise(x: vec3<f32>) -> f32 {
    let i = floor(x);
    let f = fract(x);
    let u = f * f * (3.0 - 2.0 * f);
    let c000 = hash13(i + vec3<f32>(0.0, 0.0, 0.0)); let c100 = hash13(i + vec3<f32>(1.0, 0.0, 0.0));
    let c010 = hash13(i + vec3<f32>(0.0, 1.0, 0.0)); let c110 = hash13(i + vec3<f32>(1.0, 1.0, 0.0));
    let c001 = hash13(i + vec3<f32>(0.0, 0.0, 1.0)); let c101 = hash13(i + vec3<f32>(1.0, 0.0, 1.0));
    let c011 = hash13(i + vec3<f32>(0.0, 1.0, 1.0)); let c111 = hash13(i + vec3<f32>(1.0, 1.0, 1.0));
    let x00 = mix(c000, c100, u.x); let x10 = mix(c010, c110, u.x);
    let x01 = mix(c001, c101, u.x); let x11 = mix(c011, c111, u.x);
    return mix(mix(x00, x10, u.y), mix(x01, x11, u.y), u.z);
}
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
    return sqrt(min_d);
}
// Material-aware detail height at world point `p` (m), feature wavelength `scale` (m).
// slope∈[0,1] (0 flat → 1 cliff) blends sediment↔rock character:
//   flats  = gentle 2-oct FBM + Worley pebble/cracked-soil cells
//   cliffs = ridged noise → rock fractures/strata
// ponytail: ~5 noise taps, called 5× (4 grad + 1 albedo) ≈ 25 vnoise/fragment near-field
// (faded out past DETAIL_FADE_FAR, early-out below). Drop an octave if fragment-bound.
fn detail_h(p: vec3<f32>, scale: f32, slope: f32) -> f32 {
    let q = p / max(scale, 0.05);
    let fbm = vnoise(q) * 0.6 + vnoise(q * 2.03) * 0.3;
    let cells = 1.0 - worley(q * 1.7);
    let r0 = 1.0 - abs(2.0 * vnoise(q * 1.3) - 1.0);
    let ridged = r0 * r0;
    let flat_d = fbm * 0.7 + cells * 0.3;
    return mix(flat_d, ridged, smoothstep(0.35, 0.7, slope));
}
// Cheaper height (no Worley cells) for the POM march inner loop — the gross profile is
// enough for parallax; full detail_h (with cells) is read only at the final hit point.
fn detail_h_fast(p: vec3<f32>, scale: f32, slope: f32) -> f32 {
    let q = p / max(scale, 0.05);
    let fbm = vnoise(q) * 0.6 + vnoise(q * 2.03) * 0.3;
    let r0 = 1.0 - abs(2.0 * vnoise(q * 1.3) - 1.0);
    return mix(fbm * 0.9, r0 * r0, smoothstep(0.35, 0.7, slope));
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let n = normalize(in.world_normal);

    if pc.material_mode == 1u || (pc.material_mode >= 7u && pc.material_mode <= 10u) { return vec4<f32>(in.albedo, 1.0); }
    if pc.material_mode == 2u { return vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0); }
    if pc.material_mode == 6u { return vec4<f32>(in.plate_color, 1.0); }

    // Geometry debug: 3 Triangle (per-tri), 4 Cluster (per-leaf), 5 LOD (per-level).
    if pc.material_mode == 3u { return vec4<f32>(id_color(hash2(pc.dbg_id, in.vid)), 1.0); }
    if pc.material_mode == 4u { return vec4<f32>(id_color(pc.dbg_id), 1.0); }
    if pc.material_mode == 5u { return vec4<f32>(id_color(pc.dbg_level + 1u), 1.0); }

    // Mode 0 Lit — sun0 is shadowed by the CSM (terrain + trees + character casters).
    let albedo = in.albedo;
    let sh = sample_shadow(in.world_pos, n);
    // Day/night gate: the ambient + hemisphere (sky/ground bounce) light only exists on
    // the SUNLIT hemisphere — without this the night-side land stays flat-lit while the
    // ocean (sun-driven) goes dark. radial_up = absolute world dir (camera-relative
    // world_pos + camera_pos); soft terminator on radial_up·sun, low night floor so it's
    // dark-but-not-pure-black. The directional sun terms are already self-gating (n·sun).
    let wp = in.world_pos + frame.camera_pos.xyz; // absolute world pos (radial up + detail anchor)
    let radial_up = normalize(wp);
    let day = smoothstep(-0.15, 0.25, dot(radial_up, frame.sun0_dir.xyz));
    let amb_gate = mix(0.05, 1.0, day);

    // ── Procedural ground detail (normal + parallax) ─────────────────────────────
    // Perturb the shading normal (+ a small albedo tint) from the analytic 3D-noise height
    // field, and PARALLAX-shift the sample point by marching the view ray against that
    // height (POM) so the bumps read as real depth. Pure shading — no texture (enki-rhi
    // can't upload one), no displacement (mesh IS the collider, must not move). Faded out
    // past DETAIL_FADE_FAR. Knobs: morph_b.y strength, .z scale (m), .w POM depth (m).
    var nd = n;
    var alb = albedo;
    let fade = 1.0 - smoothstep(DETAIL_FADE_NEAR, DETAIL_FADE_FAR, length(in.world_pos));
    let dstr = frame.morph_b.y * fade;
    if (dstr > 0.001) {
        let dscale = frame.morph_b.z;
        let slope = clamp(1.0 - dot(n, radial_up), 0.0, 1.0); // 0 flat → 1 cliff

        // Parallax occlusion: march the view ray against the height field; the hit point
        // `pp` (parallax-shifted from `wp`) is where we then read the normal + albedo, so
        // bumps gain motion parallax + self-occlusion. Marches the cheap (no-Worley) height;
        // full detail_h is sampled only at the hit. ponytail: if the parallax pushes the
        // WRONG way, negate `pmax` (the usual POM sign gotcha).
        var pp = wp;
        let depth = frame.morph_b.w * fade; // POM amplitude (m), distance-faded (NOT coupled to normal strength)
        if (depth > 0.0005) {
            let D = normalize(in.world_pos);                  // camera→fragment (camera at origin)
            let c = dot(D, n);                                // view·normal (<0 = into surface)
            let pmax = (D - c * n) * (depth / max(-c, 0.30)); // tangent shift over full depth
            var rh = 1.0;                                     // normalised ray height (1=top → 0)
            var off = 0.0;                                    // fraction along pmax
            var hc = detail_h_fast(wp, dscale, slope);
            for (var i = 0; i < POM_STEPS; i = i + 1) {
                if (rh <= hc) { break; }
                rh = rh - POM_INV_STEPS;
                off = off + POM_INV_STEPS;
                hc = detail_h_fast(wp + pmax * off, dscale, slope);
            }
            // relief refine: interpolate the crossing between the last two samples.
            let po = off - POM_INV_STEPS;
            let after  = hc - rh;
            let before = detail_h_fast(wp + pmax * po, dscale, slope) - (rh + POM_INV_STEPS);
            let t = clamp(after / max(after - before, 1e-4), 0.0, 1.0);
            pp = wp + pmax * mix(off, po, t);
        }

        let e = max(dscale * 0.15, 0.02);                     // finite-diff step (m)
        // 4-tap tetrahedron gradient of detail_h (same kernel as the mesher's gradient_normal).
        let k0 = vec3<f32>( 1.0, -1.0, -1.0); let k1 = vec3<f32>(-1.0, -1.0, 1.0);
        let k2 = vec3<f32>(-1.0, 1.0, -1.0); let k3 = vec3<f32>( 1.0, 1.0, 1.0);
        let g = k0 * detail_h(pp + k0 * e, dscale, slope)
              + k1 * detail_h(pp + k1 * e, dscale, slope)
              + k2 * detail_h(pp + k2 * e, dscale, slope)
              + k3 * detail_h(pp + k3 * e, dscale, slope);
        let g_tan = g - dot(g, n) * n;                        // tangential gradient only
        // ponytail: flip to (n + dstr*g_tan) if lighting looks INVERTED (bump sign).
        nd = normalize(n - dstr * g_tan);
        alb = albedo * (0.88 + 0.24 * detail_h(pp, dscale, slope)); // break the flat colour wash
    }

    let ambient_term = frame.ambient.xyz * alb * amb_gate;
    let hemi_term    = hemisphere_ambient(nd, frame.hemi_sky.xyz, frame.hemi_ground.xyz) * alb * amb_gate;
    let diff0        = directional_diffuse(nd, frame.sun0_dir.xyz, frame.sun0_color.xyz) * alb * sh;
    let diff1        = directional_diffuse(nd, frame.sun1_dir.xyz, frame.sun1_color.xyz) * alb;
    var lit = ambient_term + hemi_term + diff0 + diff1;
    lit = aces_filmic(lit);
    return vec4<f32>(lit, 1.0);
}

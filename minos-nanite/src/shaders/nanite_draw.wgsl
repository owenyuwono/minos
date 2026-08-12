// Nanite vertex-pulling draw + debug-color fragment.
//
// One indirect, non-indexed draw of (visible_clusters * MAX_TRIS * 3) vertices.
// The vertex shader maps vertex_index -> (visible slot, triangle, corner), pulls
// the vertex from the cluster storage buffers, and transforms it camera-relative.
// Triangles past a cluster's real tri_count are emitted off-screen (degenerate).
//
// The fragment view mode (unified with the classic terrain.wgsl scheme):
//   0 = lit, 1 = unlit (flat albedo), 2 = normal,
//   3 = per-triangle, 4 = per-cluster, 5 = per-LOD,
//   6 = plate (per-plate tint), 7 = height (elevation grayscale),
//   8 = material (rock hardness ramp), 9 = wetness (dry→wet ramp),
//   10 = volcano (arc/hotspot cone influence, dark→hot ramp).

struct ClusterMeta {
    bounds: vec4<f32>,
    parent_bounds: vec4<f32>,
    err: vec4<f32>,    // self_error, parent_error, lod, _
    range: vec4<u32>,  // tri_offset, tri_count, node, stable_id (debug color)
    cone: vec4<f32>,   // backface normal cone (cull-only; unused here, layout match)
};

struct FrameData {
    view_proj: mat4x4<f32>,
    lod: vec4<f32>,
    sun0_dir: vec4<f32>,
    sun0_color: vec4<f32>,
    sun1_dir: vec4<f32>,
    sun1_color: vec4<f32>,
    hemi_sky: vec4<f32>,
    hemi_ground: vec4<f32>,
    ambient: vec4<f32>,
    planes: array<vec4<f32>, 6>,
    debug: vec4<u32>,
    cam_world: vec4<f32>, // true camera world pos (xyz) — for radial-up reconstruction
    light_view_proj_cascades: array<mat4x4<f32>, 3>, // per-cascade world(cam-rel) → light clip
    shadow_params: vec4<f32>,     // [depth_bias, normal_bias, strength, enabled]
};

const SHADOW_CASCADES: u32 = 3u; // MUST match render.rs SHADOW_CASCADES

struct Push {
    mode: u32,
};

@group(0) @binding(0) var<storage, read> verts:     array<f32>;
@group(0) @binding(1) var<storage, read> tris:      array<u32>;
@group(0) @binding(2) var<storage, read> clusters:  array<ClusterMeta>;
@group(0) @binding(3) var<storage, read> visible:   array<u32>;
@group(0) @binding(5) var<storage, read> frame:     FrameData;
@group(0) @binding(6) var<storage, read> node_xlat: array<vec4<f32>>;
// Per-cascade sun shadow maps (depth) + shared comparison sampler — receiver side.
@group(0) @binding(8)  var shadow_map0: texture_depth_2d;
@group(0) @binding(9)  var shadow_map1: texture_depth_2d;
@group(0) @binding(10) var shadow_map2: texture_depth_2d;
@group(0) @binding(11) var shadow_samp: sampler_comparison;

var<immediate> pc: Push;

const MAX_TRIS: u32 = 128u;
const ZNEAR: f32 = 0.5;
// LOD cross-fade band upper bound — MUST match nanite_cull.wgsl::FADE_HI.
const FADE_HI: f32 = 1.5;

// Project a world-space error to pixels (matches nanite_cull.wgsl::project_error).
fn project_error(err: f32, center: vec3<f32>, radius: f32, screen_h: f32, cot: f32) -> f32 {
    if (err <= 0.0) { return 0.0; }
    let d = max(length(center) - radius, ZNEAR);
    return 0.5 * screen_h * cot * err / d;
}

// STABLE per-pixel dither in [0,1) (no frame term → no flicker without TAA). Used
// to partition pixels between overlapping LOD levels in a transition band.
fn dither_value(px: vec2<f32>) -> f32 {
    let p = vec2<u32>(u32(px.x), u32(px.y));
    var h = p.x * 73856093u ^ p.y * 19349663u;
    h = h ^ (h >> 16u);
    h = h * 0x85EBCA77u;
    h = h ^ (h >> 13u);
    return f32(h & 0x00FFFFFFu) / f32(0x01000000u);
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) @interpolate(flat) cluster_id: u32,
    @location(3) @interpolate(flat) tri_id: u32,
    @location(4) @interpolate(flat) lod: u32,
    // LOD cross-fade interval [t_self, t_par): this cluster owns dither values in
    // this sub-range of [0,1). Adjacent LODs share boundaries → exact partition.
    @location(5) @interpolate(flat) t_self: f32,
    @location(6) @interpolate(flat) t_par: f32,
    @location(7) material: f32,
    @location(8) wetness: f32,
    @location(9) volcanism: f32,
    @location(10) elevation: f32,
    @location(11) plate: vec3<f32>,
    @location(12) horizon: vec4<f32>,   // per-azimuth horizon-elevation sines (self-shadow)
    @location(13) world_rel: vec3<f32>, // camera-relative world pos (for radial up)
};

@vertex
fn vs_pull(@builtin(vertex_index) vid: u32) -> VsOut {
    var out: VsOut;
    let per = MAX_TRIS * 3u;
    let vis = vid / per;
    let local = vid % per;
    let tri = local / 3u;
    let corner = local % 3u;

    let ci = visible[vis]; // page-pool slot index (also the meta/geometry index)
    let m = clusters[ci];

    let t = node_xlat[m.range.z].xyz;

    // LOD cross-fade partition. This cluster owns dither values in [t_self, t_par):
    //   t_self = how far self_px is into the band (0 at tau → 1 at tau*FADE_HI):
    //            this cluster's finer children own [0, t_self), it owns [t_self, ..).
    //   t_par  = how far parent_px is into the band: this cluster owns [.., t_par),
    //            its coarser parent owns [t_par, 1]. A child's t_par == this self's
    //            t_self, so the intervals tile [0,1] exactly across the hierarchy.
    let tau = frame.lod.x;
    let band = max(tau * (FADE_HI - 1.0), 1e-6);
    let center = m.bounds.xyz + t;
    let self_px = project_error(m.err.x, center, m.bounds.w, frame.lod.y, frame.lod.z);
    var parent_px: f32 = 1e30;
    if (m.err.y < 1e30) {
        let pcenter = m.parent_bounds.xyz + t;
        parent_px = project_error(m.err.y, pcenter, m.parent_bounds.w, frame.lod.y, frame.lod.z);
    }
    out.t_self = clamp((self_px - tau) / band, 0.0, 1.0);
    out.t_par = clamp((parent_px - tau) / band, 0.0, 1.0);

    if (tri >= m.range.y) {
        // Past this cluster's real triangle count -> degenerate / off-screen.
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0);
        out.normal = vec3<f32>(0.0, 1.0, 0.0);
        out.color = vec3<f32>(0.0, 0.0, 0.0);
        out.cluster_id = m.range.w; // stable id (not buffer slot) → flicker-free debug
        out.tri_id = 0u;
        out.lod = 0u;
        out.material = 0.0;
        out.wetness = 0.0;
        out.volcanism = 0.0;
        out.elevation = 0.0;
        out.plate = vec3<f32>(0.0);
        out.horizon = vec4<f32>(0.0);
        out.world_rel = vec3<f32>(0.0);
        return out;
    }

    // Fixed-stride page pool: this cluster's tris start at slot ci * MAX_TRIS, and
    // tris store GLOBAL vertex indices (ci*MAX_VERTS + local), so no per-cluster
    // vertex base is needed.
    let gt = ci * MAX_TRIS + tri;
    let vidx = tris[gt * 3u + corner];
    let base = vidx * 20u; // stride: pos3 + normal3 + color3 + material1 + wetness1 + volcanism1 + elevation1 + plate3 + horizon4
    let pos = vec3<f32>(verts[base + 0u], verts[base + 1u], verts[base + 2u]);
    let nrm = vec3<f32>(verts[base + 3u], verts[base + 4u], verts[base + 5u]);
    let col = vec3<f32>(verts[base + 6u], verts[base + 7u], verts[base + 8u]);

    let world_rel = pos + t;
    out.clip = frame.view_proj * vec4<f32>(world_rel, 1.0);
    out.normal = nrm;
    out.color = col;
    out.cluster_id = m.range.w; // stable id (not buffer slot) → flicker-free debug
    out.tri_id = tri;
    out.lod = u32(m.err.z);
    out.material = verts[base + 9u];
    out.wetness = verts[base + 10u];
    out.volcanism = verts[base + 11u];
    out.elevation = verts[base + 12u];
    out.plate = vec3<f32>(verts[base + 13u], verts[base + 14u], verts[base + 15u]);
    out.horizon = vec4<f32>(verts[base + 16u], verts[base + 17u], verts[base + 18u], verts[base + 19u]);
    out.world_rel = world_rel;
    return out;
}

struct DepthOut {
    @builtin(position) clip: vec4<f32>,
};

// Sun shadow-map caster: pull the same geometry as `vs_pull`, project with the
// light view-proj (depth-only, no attributes). Degenerate tris go off-screen.
@vertex
fn vs_depth(@builtin(vertex_index) vid: u32) -> DepthOut {
    var out: DepthOut;
    let per = MAX_TRIS * 3u;
    let vis = vid / per;
    let local = vid % per;
    let tri = local / 3u;
    let corner = local % 3u;

    let ci = visible[vis];
    let m = clusters[ci];
    let t = node_xlat[m.range.z].xyz;

    if (tri >= m.range.y) {
        out.clip = vec4<f32>(2.0, 2.0, 2.0, 1.0); // degenerate → clipped
        return out;
    }
    let gt = ci * MAX_TRIS + tri;
    let vidx = tris[gt * 3u + corner];
    let base = vidx * 20u;
    let world_rel = vec3<f32>(verts[base + 0u], verts[base + 1u], verts[base + 2u]) + t;
    // pc.mode carries the cascade index for the shadow pipeline (per-draw push).
    out.clip = frame.light_view_proj_cascades[pc.mode] * vec4<f32>(world_rel, 1.0);
    return out;
}

fn hash_u32(x: u32) -> u32 {
    var h = x * 0x9E3779B1u;
    h = h ^ (h >> 16u);
    h = h * 0x85EBCA77u;
    h = h ^ (h >> 13u);
    return h;
}

fn hash2(a: u32, b: u32) -> u32 {
    var h = a * 0x9E3779B1u ^ b * 0x85EBCA77u;
    h = h ^ (h >> 15u);
    return h;
}

fn hsv2rgb(h: f32, s: f32, v: f32) -> vec3<f32> {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - abs(hp % 2.0 - 1.0));
    var rgb: vec3<f32>;
    let hi = i32(hp);
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

fn aces_filmic(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Material (rock hardness) ramp, soft→hard: 5-stop indigo→cyan→lime→orange→crimson
// at [0, 0.25, 0.5, 0.75, 1] (mirrors ki's materials view).
fn material_ramp(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.29, 0.13, 0.65); // indigo
    let c1 = vec3<f32>(0.10, 0.75, 0.85); // cyan
    let c2 = vec3<f32>(0.55, 0.85, 0.20); // lime
    let c3 = vec3<f32>(0.95, 0.55, 0.10); // orange
    let c4 = vec3<f32>(0.85, 0.10, 0.20); // crimson
    if (t < 0.25) { return mix(c0, c1, t / 0.25); }
    else if (t < 0.5) { return mix(c1, c2, (t - 0.25) / 0.25); }
    else if (t < 0.75) { return mix(c2, c3, (t - 0.5) / 0.25); }
    return mix(c3, c4, (t - 0.75) / 0.25);
}

// Hypsometric elevation ramp (debug Height view): ocean blues → green lowland →
// yellow → brown highland → white peaks. The land range (most terrain sits in
// ~0–0.4) gets closely-spaced, distinct bands so small height differences —
// valleys vs ridges, and whether rivers sit in them — read with strong contrast,
// unlike the old flat grayscale that crushed everything into mid-grays.
fn height_ramp(e: f32) -> vec3<f32> {
    if (e < 0.0) {
        // Ocean: shallow cyan at the coast → deep navy in the abyss.
        let d = clamp(-e / 0.5, 0.0, 1.0);
        return mix(vec3<f32>(0.20, 0.55, 0.70), vec3<f32>(0.02, 0.06, 0.20), d);
    }
    let c0 = vec3<f32>(0.18, 0.52, 0.24); // coast green
    let c1 = vec3<f32>(0.45, 0.68, 0.28); // lowland green
    let c2 = vec3<f32>(0.82, 0.80, 0.38); // yellow
    let c3 = vec3<f32>(0.72, 0.50, 0.28); // tan
    let c4 = vec3<f32>(0.48, 0.35, 0.26); // dark brown
    let c5 = vec3<f32>(0.97, 0.97, 0.98); // snow / peak
    if (e < 0.10) { return mix(c0, c1, e / 0.10); }
    else if (e < 0.22) { return mix(c1, c2, (e - 0.10) / 0.12); }
    else if (e < 0.36) { return mix(c2, c3, (e - 0.22) / 0.14); }
    else if (e < 0.52) { return mix(c3, c4, (e - 0.36) / 0.16); }
    return mix(c4, c5, clamp((e - 0.52) / 0.20, 0.0, 1.0));
}

// Wetness ramp, dry→wet: dry tan (#b8a07a, neutral) → wet blue (#2a6fb0), so
// rivers/lakes read as blue threads over a neutral surface.
fn wetness_ramp(t: f32) -> vec3<f32> {
    let dry = vec3<f32>(0.722, 0.627, 0.478); // #b8a07a
    let wet = vec3<f32>(0.165, 0.435, 0.690); // #2a6fb0
    return mix(dry, wet, t);
}

// Volcano ramp, none→peak: near-black basalt → deep red → orange → hot yellow,
// so arc/hotspot cones read as glowing hot spots over a dark surface.
fn volcano_ramp(t: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.06, 0.06, 0.08); // near-black basalt (no cone influence)
    let c1 = vec3<f32>(0.55, 0.08, 0.04); // deep red
    let c2 = vec3<f32>(0.95, 0.45, 0.10); // orange
    let c3 = vec3<f32>(1.00, 0.92, 0.55); // hot yellow-white
    if (t < 0.5) { return mix(c0, c1, t / 0.5); }
    else if (t < 0.8) { return mix(c1, c2, (t - 0.5) / 0.3); }
    return mix(c2, c3, (t - 0.8) / 0.2);
}

// Soft terrain self-shadow falloff. ponytail: a shader const (tuning knob —
// rebuild to change); promote to a push constant + GUI slider when iterating.
const HORIZON_STRENGTH: f32 = 12.0;

// Fraction of ambient + hemisphere sky-fill that survives on the deep night side.
// 0 = pitch-black night; small floor keeps the night disc faintly visible vs space.
// MUST match the floor + smoothstep band in minos-render ocean_surface.wgsl (shade_water)
// so land and sea darken together across the terminator — no shared cross-crate const.
const NIGHT_FILL: f32 = 0.05;

// Mirror of minos_planet::face_bases::horizon_basis — MUST stay byte-identical so
// the bake's azimuth-0 matches the runtime. Columns = (tan_a, tan_b, up).
fn horizon_basis(up: vec3<f32>) -> mat3x3<f32> {
    let r = select(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0), abs(up.y) < 0.999);
    let tan_a = normalize(r - up * dot(r, up));
    let tan_b = cross(up, tan_a);
    return mat3x3<f32>(tan_a, tan_b, up);
}

// Terrain self-shadow for the primary sun: 1 = lit, <1 = shadowed. `horizon` holds
// per-azimuth horizon-elevation SINES (baked); compare to the sun's elevation sine
// in the local tangent frame, bilinear in azimuth across the 4 slots.
fn terrain_shadow(world_rel: vec3<f32>, horizon: vec4<f32>, sun: vec3<f32>) -> f32 {
    let up = normalize(world_rel + frame.cam_world.xyz);
    let sun_elev = dot(sun, up);
    if (sun_elev <= 0.0) { return 1.0; } // night side: d0 already ~0
    let b = horizon_basis(up);
    let sun_h = sun - up * sun_elev; // horizontal sun direction
    let az = atan2(dot(sun_h, b[1]), dot(sun_h, b[0])); // [-pi,pi], 0 = tan_a
    var f = az / 1.5707963; // quadrant index in [-2,2]
    if (f < 0.0) { f = f + 4.0; }
    let i0 = u32(floor(f)) % 4u;
    let i1 = (i0 + 1u) % 4u;
    let h = mix(horizon[i0], horizon[i1], fract(f));
    return 1.0 - clamp((h - sun_elev) * HORIZON_STRENGTH, 0.0, 1.0);
}

// Pick the tightest cascade whose light-space uv contains `world_rel` (offset along
// `n` by the cascade-scaled normal bias). Returns the cascade index (or
// SHADOW_CASCADES = "outside all"), and writes uv + reversed-Z reference depth.
fn select_cascade(world_rel: vec3<f32>, n: vec3<f32>, uv_out: ptr<function, vec2<f32>>, ref_out: ptr<function, f32>) -> u32 {
    for (var i = 0u; i < SHADOW_CASCADES; i = i + 1u) {
        let scale = f32(1u << i); // coarser cascade → bigger texels → more bias
        let p = world_rel + n * (frame.shadow_params.y * scale);
        let clip = frame.light_view_proj_cascades[i] * vec4<f32>(p, 1.0);
        if (clip.w <= 0.0) { continue; }
        let ndc = clip.xyz / clip.w;
        let u = ndc.xy * 0.5 + vec2<f32>(0.5, 0.5);
        let m = 0.02; // margin so the 3×3 PCF kernel never reads past the cascade edge
        if (u.x > m && u.x < 1.0 - m && u.y > m && u.y < 1.0 - m && ndc.z > 0.0 && ndc.z < 1.0) {
            *uv_out = u;
            *ref_out = ndc.z + frame.shadow_params.x * scale;
            return i;
        }
    }
    return SHADOW_CASCADES;
}

// Cascaded sun shadow (PCF). 1 = lit, 0 = fully shadowed. Reversed-Z: the caster
// (closest to light) stored the GREATER depth → a receiver behind it has a SMALLER
// ndc.z → GREATER_OR_EQUAL fails → shadowed. Picks the tightest covering cascade
// (sharp near, coarse far); outside all → lit. ponytail: hard cascade boundary — a
// cross-cascade blend band is the seam-hiding upgrade.
fn sample_shadow(world_rel: vec3<f32>, n: vec3<f32>) -> f32 {
    if (frame.shadow_params.w < 0.5) { return 1.0; } // disabled
    var uv = vec2<f32>(0.0, 0.0);
    var ref_depth = 0.0;
    let c = select_cascade(world_rel, n, &uv, &ref_depth);
    if (c >= SHADOW_CASCADES) { return 1.0; } // outside every cascade → lit
    // 3×3 PCF (each tap is 2×2 hardware PCF → ~6×6 effective). Texture picked by a
    // switch — WGSL can't dynamically index separate texture bindings. All cascades
    // share the same pixel size, so shadow_map0's dimensions set the texel step.
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

@fragment
fn fs_color(in: VsOut) -> @location(0) vec4<f32> {
    // Dithered LOD cross-fade (frame.debug.y = enabled). This cluster keeps only the
    // pixels whose stable dither value lands in its [t_self, t_par) interval; the
    // coarser parent / finer child own the rest → exact partition, no holes/overlap.
    // (Guard t_par > t_self so a degenerate interval keeps the cluster fully.)
    if (frame.debug.y == 1u && in.t_par > in.t_self) {
        let d = dither_value(in.clip.xy);
        if (d < in.t_self || d >= in.t_par) {
            discard;
        }
    }

    var rgb: vec3<f32>;
    if (pc.mode == 0u) {
        // Lit — matches terrain.wgsl mode 0 (ambient + hemisphere + 2 suns + ACES).
        let n = normalize(in.normal);
        let albedo = in.color;
        let sun_dir = normalize(frame.sun0_dir.xyz);
        // Planet-scale day/night: ambient + hemisphere are sky/bounce light that only
        // exists where the sun is up. Fade them by the LOCAL sun elevation (radial up
        // · sun) so the night hemisphere goes dark and the day/night terminator reads
        // from orbit (otherwise the night side is as bright as the day side → a flat
        // globe). Transition sits right at the terminator: full fill on the lit side
        // (so shadowed valleys keep ambient), dropping to NIGHT_FILL just into the dark.
        // ponytail: NIGHT_FILL + band are shader-const knobs; promote to GUI if tuned.
        let up = normalize(in.world_rel + frame.cam_world.xyz);
        let day = mix(NIGHT_FILL, 1.0, smoothstep(-0.2, 0.0, dot(up, sun_dir)));
        let ambient_term = frame.ambient.xyz * albedo * day;
        let hemi = hemisphere_ambient(n, frame.hemi_sky.xyz, frame.hemi_ground.xyz) * albedo * day;
        let d0 = max(dot(n, frame.sun0_dir.xyz), 0.0) * frame.sun0_color.xyz * albedo;
        let d1 = max(dot(n, frame.sun1_dir.xyz), 0.0) * frame.sun1_color.xyz * albedo;
        // Sun occlusion: terrain self-shadow × the character's cast shadow. Darken
        // the primary-sun direct term, leaving ambient/hemisphere/fill so shadowed
        // ground isn't black.
        var sh = terrain_shadow(in.world_rel, in.horizon, sun_dir);
        sh = sh * sample_shadow(in.world_rel, n); // sun shadow map (objects + terrain)
        rgb = aces_filmic(ambient_term + hemi + d0 * sh + d1);
    } else if (pc.mode == 1u) {
        // Unlit — flat biome albedo (no lighting, no ACES).
        rgb = in.color;
    } else if (pc.mode == 2u) {
        // Normal — world-space normal → RGB.
        rgb = normalize(in.normal) * 0.5 + vec3<f32>(0.5);
    } else if (pc.mode == 3u) {
        rgb = id_color(hash2(in.cluster_id, in.tri_id));
    } else if (pc.mode == 4u) {
        rgb = id_color(in.cluster_id);
    } else if (pc.mode == 5u) {
        rgb = id_color(in.lod + 1u);
    } else if (pc.mode == 6u) {
        // Plate — per-plate tint.
        rgb = in.plate;
    } else if (pc.mode == 7u) {
        // Height — hypsometric tint (ocean blues → green → brown → white peaks),
        // high-contrast in the low-land range so valleys/ridges read clearly.
        rgb = height_ramp(in.elevation);
    } else if (pc.mode == 8u) {
        rgb = material_ramp(clamp(in.material, 0.0, 1.0));
    } else if (pc.mode == 9u) {
        rgb = wetness_ramp(clamp(in.wetness, 0.0, 1.0));
    } else if (pc.mode == 14u) {
        // DEBUG: tint by the selected shadow cascade — c0 = RED, c1 = GREEN,
        // c2 = BLUE, BLACK = outside all cascades. Visualizes cascade coverage/rings.
        let n14 = normalize(in.normal);
        var uv14 = vec2<f32>(0.0, 0.0);
        var ref14 = 0.0;
        let c14 = select_cascade(in.world_rel, n14, &uv14, &ref14);
        if (c14 == 0u) { rgb = vec3<f32>(0.8, 0.2, 0.2); }
        else if (c14 == 1u) { rgb = vec3<f32>(0.2, 0.8, 0.2); }
        else if (c14 == 2u) { rgb = vec3<f32>(0.2, 0.4, 0.9); }
        else { rgb = vec3<f32>(0.05, 0.05, 0.05); }
    } else {
        rgb = volcano_ramp(clamp(in.volcanism, 0.0, 1.0));
    }
    return vec4<f32>(rgb, 1.0);
}

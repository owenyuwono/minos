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
};

struct Push {
    mode: u32,
};

@group(0) @binding(0) var<storage, read> verts:     array<f32>;
@group(0) @binding(1) var<storage, read> tris:      array<u32>;
@group(0) @binding(2) var<storage, read> clusters:  array<ClusterMeta>;
@group(0) @binding(3) var<storage, read> visible:   array<u32>;
@group(0) @binding(5) var<storage, read> frame:     FrameData;
@group(0) @binding(6) var<storage, read> node_xlat: array<vec4<f32>>;

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
        return out;
    }

    // Fixed-stride page pool: this cluster's tris start at slot ci * MAX_TRIS, and
    // tris store GLOBAL vertex indices (ci*MAX_VERTS + local), so no per-cluster
    // vertex base is needed.
    let gt = ci * MAX_TRIS + tri;
    let vidx = tris[gt * 3u + corner];
    let base = vidx * 16u; // stride: pos3 + normal3 + color3 + material1 + wetness1 + volcanism1 + elevation1 + plate3
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
        let ambient_term = frame.ambient.xyz * albedo;
        let hemi = hemisphere_ambient(n, frame.hemi_sky.xyz, frame.hemi_ground.xyz) * albedo;
        let d0 = max(dot(n, frame.sun0_dir.xyz), 0.0) * frame.sun0_color.xyz * albedo;
        let d1 = max(dot(n, frame.sun1_dir.xyz), 0.0) * frame.sun1_color.xyz * albedo;
        rgb = aces_filmic(ambient_term + hemi + d0 + d1);
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
        // Height — signed normalized elevation → grayscale (ocean dark, peaks bright).
        let g = clamp(in.elevation * 0.5 + 0.5, 0.0, 1.0);
        rgb = vec3<f32>(g);
    } else if (pc.mode == 8u) {
        rgb = material_ramp(clamp(in.material, 0.0, 1.0));
    } else if (pc.mode == 9u) {
        rgb = wetness_ramp(clamp(in.wetness, 0.0, 1.0));
    } else {
        rgb = volcano_ramp(clamp(in.volcanism, 0.0, 1.0));
    }
    return vec4<f32>(rgb, 1.0);
}

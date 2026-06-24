// Nanite cull / LOD-cut compute pass.
//
// One thread per cluster: frustum-cull against the cluster bounds, then apply the
// crack-free two-sided LOD-cut test (self_error <= tau && parent_error > tau,
// projected to pixels). Survivors are appended to `visible` and the indirect
// draw's vertex count is advanced atomically (MAX_TRIS*3 verts per visible cluster).

struct ClusterMeta {
    bounds: vec4<f32>,        // center.xyz (patch-relative), radius
    parent_bounds: vec4<f32>, // encloses bounds
    err: vec4<f32>,           // self_error, parent_error, lod, _
    range: vec4<u32>,         // tri_offset, tri_count, _, _
    cone: vec4<f32>,          // backface normal cone: axis.xyz, sin(half_angle)
};

struct FrameData {
    view_proj: mat4x4<f32>,
    lod: vec4<f32>,               // tau_px, screen_h, cot_half_fov, cluster_count
    // Lights (unused by cull, but the layout must match the draw shader / CPU).
    sun0_dir: vec4<f32>,
    sun0_color: vec4<f32>,
    sun1_dir: vec4<f32>,
    sun1_color: vec4<f32>,
    hemi_sky: vec4<f32>,
    hemi_ground: vec4<f32>,
    ambient: vec4<f32>,
    planes: array<vec4<f32>, 6>,  // frustum planes in camera-relative space
    debug: vec4<u32>,             // debug_mode, _, _, _
};

struct DrawArgs {
    vertex_count: atomic<u32>,
    instance_count: u32,
    first_vertex: u32,
    first_instance: u32,
};

@group(0) @binding(2) var<storage, read>       clusters:  array<ClusterMeta>;
@group(0) @binding(3) var<storage, read_write> visible:   array<u32>;
@group(0) @binding(4) var<storage, read_write> args:      DrawArgs;
@group(0) @binding(5) var<storage, read>       frame:     FrameData;
// Per-node camera-relative translation (node_origin - camera), indexed by node.
@group(0) @binding(6) var<storage, read>       node_xlat: array<vec4<f32>>;
// Active cluster-slot list: the page-pool slots resident this frame. The cull
// dispatches one thread per active slot (count = frame.lod.w).
@group(0) @binding(7) var<storage, read>       active_slots: array<u32>;

const MAX_TRIS: u32 = 128u;
const ZNEAR: f32 = 0.5;
// LOD cross-fade band: with dithering on, a cluster stays selected until its error
// reaches tau*FADE_HI (instead of tau), so it overlaps its finer replacement across
// the band and the draw can dither-blend the transition. Must match the draw shader.
const FADE_HI: f32 = 1.5;

// Project a cluster's world-space error of magnitude `err` to pixels.
//
// `center` is the cluster bounds center (camera-relative; camera at origin) and
// `radius` is the cluster's SPATIAL bounding radius. We measure from the NEAREST
// point of the bounding sphere (`dist(center) - radius`), not the center — a
// large/near cluster's near edge is much closer than its center, and projecting
// from the center under-estimates the on-screen error, causing late, poppy LOD
// switches. (Matches the quadtree metric and the canonical Nanite formula.)
fn project_error(err: f32, center: vec3<f32>, radius: f32, screen_h: f32, cot: f32) -> f32 {
    if (err <= 0.0) {
        return 0.0;
    }
    let d = max(length(center) - radius, ZNEAR);
    return 0.5 * screen_h * cot * err / d;
}

@compute @workgroup_size(64)
fn cs_cull(@builtin(global_invocation_id) gid: vec3<u32>) {
    let gidx = gid.x;
    let count = u32(frame.lod.w);
    if (gidx >= count) {
        return;
    }
    // Resolve this thread's resident page-pool slot; `ci` is the slot index, which
    // is also the meta index and (via fixed stride) the geometry location.
    let ci = active_slots[gidx];
    let m = clusters[ci];

    let t = node_xlat[m.range.z].xyz;
    let center = m.bounds.xyz + t;

    // Frustum cull against the cluster bounds sphere.
    let r = m.bounds.w;
    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        let pl = frame.planes[i];
        if (dot(pl.xyz, center) + pl.w < -r) {
            return;
        }
    }

    // Backface normal-cone cull: skip a cluster whose every triangle faces away.
    // `view` is the camera→cluster direction (camera at the origin), `cone.xyz` the
    // outward axis, `cone.w` = sin(half-angle). When all faces point away,
    // `dot(axis, view) > sin(half-angle)` holds (derivation: the worst-case face,
    // tilted toward the view, still faces away iff `dot(axis,view) > sin α`). The
    // sentinel `cone.w >= 2` (degenerate / >hemisphere cone) never fires. Skip when
    // the camera is inside the bounds (view direction ill-defined, apex matters).
    let dlen = length(center);
    if (dlen > r && dot(m.cone.xyz, center / dlen) > m.cone.w) {
        return;
    }

    let tau = frame.lod.x;
    let screen_h = frame.lod.y;
    let cot = frame.lod.z;

    let self_px = project_error(m.err.x, center, m.bounds.w, screen_h, cot);
    var parent_px: f32;
    if (m.err.y >= 1e30) {
        parent_px = 1e30; // root: parent error is +inf -> always > tau
    } else {
        let pcenter = m.parent_bounds.xyz + t;
        parent_px = project_error(m.err.y, pcenter, m.parent_bounds.w, screen_h, cot);
    }

    // With dithering on, widen the upper bound to tau*FADE_HI so a fading-out
    // cluster overlaps its finer replacement across the band (the draw partitions
    // pixels between them). Off → strict crack-free cut (hi = tau).
    let dither_on = frame.debug.y == 1u;
    let hi = select(tau, tau * FADE_HI, dither_on);
    if (self_px <= hi && parent_px > tau) {
        let old = atomicAdd(&args.vertex_count, MAX_TRIS * 3u);
        let slot = old / (MAX_TRIS * 3u);
        visible[slot] = ci;
    }
}

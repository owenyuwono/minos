//! `rivers` — drainage-network water surface computed from terrain HEIGHT +
//! PRECIPITATION (NO erosion data), drawn as translucent blue ribbons. Reads only
//! `HeightField::{height, moisture}` + the cube-sphere `cubemap` helpers.
//!
//! Why a MESH and not per-vertex paint: rivers are far thinner than the terrain
//! mesh's vertex spacing, so painting `wetness` into the terrain albedo aliases away
//! at every LOD. A swept ribbon is its own geometry, crisp at any zoom.
//!
//! Why height+precip and not the erosion flow field: the erosion sim is MFD +
//! 4-pass blurred — diffuse, so collapsing it to single-downstream broke into
//! disconnected stubs. Instead we run the textbook hydrology pipeline
//! (see docs/rivers-research.md): **Priority-Flood (Barnes 2014)** fills depressions
//! and builds a drainage TREE where every land cell drains downhill to the sea
//! (connected by construction), then **precipitation accumulates** down the tree
//! (wetter basins → bigger rivers), threshold → channel network → junction graph
//! (source / confluence nodes) → Catmull-Rom + Laplacian smooth → ribbon (half-width
//! ∝ √discharge). Built ONCE when the heightfield arrives; only the camera-relative
//! position buffer is rewritten per frame (mirrors `markers.rs`). ponytail: the
//! erosion-based `flow_*`/`lake_mask` trait methods are no longer used here.

use bytemuck::cast_slice;
use enki_planet::cubemap::{neighbor_texel, tex_ang, texel_index, texel_to_dir};
use enki_planet::height::HeightField;
use enki_render::{frame::FrameUniforms, material::ChunkPush};
use enki_rhi::{vk, BufferHandle, PipelineHandle, Rhi, RhiError};
use glam::DVec3;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::Arc;

use crate::markers::{draw_ribbon, overlay_pipeline};

const RIVERS_WGSL: &str = include_str!("rivers.wgsl");

// --- Extraction tuning (ponytail: tunables, not derived — all USER-verified visually) ---

/// Drainage-extraction grid resolution per cube face. Higher = finer channels but
/// the bake is `6·RES²` `height()` calls — the dominant cost. 160 ≈ ~3 s one-time.
const GRID_RES: usize = 160;
/// `HeightField` LOD for the ROUTING height sample (the broad/medium terrain drives
/// where water collects; fine FBM bumps don't matter for routing → cheaper level).
const ROUTE_LEVEL: u8 = 5;
/// Priority-Flood fill increment — a tiny monotonic raise so filled depressions /
/// flats still have a downstream gradient toward their spill point.
const FILL_EPS: f32 = 1e-7;
/// Channel-initiation threshold on the precipitation-weighted, log-normalized
/// drainage accumulation (0..1, peaks at 1.0 at the largest river mouth). The
/// SINGLE dominant density knob — lower = denser network (more tributaries),
/// higher = only trunks. See docs/rivers-research.md.
const CHANNEL_THRESHOLD: f64 = 0.22;
/// `HeightField` LOD used to drape the water line (the FINE terrain, so the ribbon
/// sits in the real valley even though routing uses the coarser ROUTE_LEVEL).
const DRAPE_LEVEL: u8 = 8;
/// Per-edge downstream-walk guard (against a noisy flow-direction micro-cycle).
const MAX_STEPS: usize = 8192;
/// Catmull-Rom substeps per edge segment (kills the ~grid-pitch angularity).
const SMOOTH_SUB: usize = 4;
/// Laplacian centerline-smoothing passes — relaxes the D8/channel-preference
/// staircase (axis-cell hops on diagonal flow) into a natural meander.
const LAPLACE_ITERS: usize = 12;
/// Skip graph EDGES shorter than this many cells (a confluence sitting one or two
/// cells below another is real but invisibly short — drop the sub-pixel reach).
const MIN_REACH_POINTS: usize = 3;

/// Channel half-width (world m) at zero discharge + the extra at full discharge
/// (×√acc). Thin: on a ~50 km-radius planet a trunk reads as a ~120 m thread, not
/// a km-wide quad. Bump if rivers are too faint; shrink if they read as fat blobs.
const WIDTH_BASE: f64 = 18.0;
const WIDTH_SCALE: f64 = 150.0;
/// Lift above the draped terrain height (m) so the ribbon clears the channel floor
/// (and its reversed-Z depth beats the terrain → no z-fight), staying below banks.
const LIFT_M: f64 = 6.0;

/// Translucent water albedo (sun-lit + glinted in the fragment).
const WATER_COL: [f32; 3] = [0.05, 0.18, 0.30];

// 8-neighbour offsets (excluding 0,0) for the discrete downstream step.
const NB_DX: [i32; 8] = [-1, 0, 1, -1, 1, -1, 0, 1];
const NB_DY: [i32; 8] = [-1, -1, -1, 0, 0, 1, 1, 1];

/// The traced ribbon mesh in planet-centred world space (f64 for camera-relative
/// precision), plus a uniform water color and double-sided indices.
#[derive(Default)]
pub struct RiverMesh {
    /// World positions (planet centre = origin), f64 for the per-frame subtract.
    pub world: Vec<DVec3>,
    /// Per-vertex radial-up normals (for the water shading).
    pub normals: Vec<[f32; 3]>,
    /// Triangle indices (double-sided — the shared pipeline culls BACK).
    pub indices: Vec<u32>,
}

pub struct Rivers {
    pipeline: PipelineHandle,
    base_radius: f64,
    /// Heightfield + scale, stashed by `set_source`; the trace runs lazily.
    source: Option<(Arc<dyn HeightField>, f64)>,
    /// Traced geometry (built once `source` is set and `build` runs).
    mesh: RiverMesh,
    /// Per-FiF camera-relative position buffers + the static color buffer + index.
    pos_dyn: Vec<BufferHandle>,
    col_buf: Option<BufferHandle>,
    nrm_buf: Option<BufferHandle>,
    idx: Option<BufferHandle>,
    idx_count: u32,
    /// Camera-relative scratch, rebuilt each frame.
    pos_scratch: Vec<[f32; 3]>,
    built: bool,
}

impl Rivers {
    pub fn new(
        rhi: &mut Rhi,
        color_format: vk::Format,
        samples: vk::SampleCountFlags,
        base_radius: f64,
    ) -> Result<Self, RhiError> {
        // Translucent water: alpha blend + depth-write OFF (depth-test GREATER
        // stays on, so terrain in front occludes the ribbon; the small LIFT_M
        // keeps the river just in front of its own channel floor → no z-fight).
        let pipeline = overlay_pipeline(
            rhi,
            RIVERS_WGSL,
            color_format,
            std::mem::size_of::<ChunkPush>() as u32,
            true,
            samples,
        )?;

        Ok(Self {
            pipeline,
            base_radius,
            source: None,
            mesh: RiverMesh::default(),
            pos_dyn: Vec::new(),
            col_buf: None,
            nrm_buf: None,
            idx: None,
            idx_count: 0,
            pos_scratch: Vec::new(),
            built: false,
        })
    }

    /// Stash the heightfield; the (expensive) trace + GPU upload happen on the
    /// first `record` after this (so the call site needs no `&mut Rhi`).
    pub fn set_source(&mut self, hf: Arc<dyn HeightField>, height_scale: f64) {
        self.source = Some((hf, height_scale));
        self.built = false;
    }

    /// Draw the river network. Builds the mesh + GPU buffers on the first call
    /// after `set_source`. Call inside the opaque MSAA / translucent overlay slot.
    pub fn record(
        &mut self,
        rhi: &mut Rhi,
        fi: u32,
        fu: &FrameUniforms,
        camera_pos: DVec3,
    ) -> Result<(), RhiError> {
        if !self.built {
            self.build(rhi)?;
        }
        if self.idx_count == 0 {
            return Ok(());
        }

        // Per-frame camera-relative positions (f64 subtract → f32; mirrors markers).
        self.pos_scratch.clear();
        for &w in &self.mesh.world {
            self.pos_scratch.push((w - camera_pos).as_vec3().to_array());
        }
        let fi_u = fi as usize;
        rhi.write_storage_bytes(self.pos_dyn[fi_u], cast_slice(&self.pos_scratch))?;

        // Positions are already camera-relative → identity model (origin == camera).
        let p = self.pos_dyn[fi_u];
        let n = self.nrm_buf.unwrap();
        let c = self.col_buf.unwrap();
        draw_ribbon(rhi, fi, fu, camera_pos, self.pipeline, p, n, c, self.idx.unwrap(), self.idx_count)
    }

    /// Trace the network (CPU) and create + upload the GPU buffers (once).
    fn build(&mut self, rhi: &mut Rhi) -> Result<(), RhiError> {
        self.built = true;
        let Some((hf, height_scale)) = self.source.clone() else { return Ok(()); };

        self.mesh = trace_rivers(hf.as_ref(), self.base_radius, height_scale);
        let vcount = self.mesh.world.len();
        self.idx_count = self.mesh.indices.len() as u32;
        log::info!(
            "rivers: traced {} verts, {} tris",
            vcount,
            self.idx_count / 3
        );
        if vcount == 0 || self.idx_count == 0 {
            return Ok(());
        }

        // Static colors (uniform water) + radial-up normals, uploaded once.
        let colors: Vec<[f32; 3]> = vec![WATER_COL; vcount];
        let vbuf_size = (vcount * std::mem::size_of::<[f32; 3]>()) as u64;
        let fif = rhi.frames_in_flight();
        self.pos_dyn.clear();
        for _ in 0..fif {
            self.pos_dyn.push(rhi.create_gpu_buffer(
                vbuf_size,
                true,
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?);
        }
        let nrm = rhi.create_gpu_buffer(vbuf_size, true, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        let col = rhi.create_gpu_buffer(vbuf_size, true, vk::BufferUsageFlags::VERTEX_BUFFER)?;
        rhi.write_storage_bytes(nrm, cast_slice(&self.mesh.normals))?;
        rhi.write_storage_bytes(col, cast_slice(&colors))?;
        self.nrm_buf = Some(nrm);
        self.col_buf = Some(col);
        self.idx = Some(rhi.create_index_buffer(&self.mesh.indices)?);
        self.pos_scratch = Vec::with_capacity(vcount);
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Pure tracer (headless, unit-tested) — port of RiverNetwork.build.
// ---------------------------------------------------------------------------

/// Uniform Catmull-Rom interpolation of one scalar component.
#[inline]
fn catmull(p0: f64, p1: f64, p2: f64, p3: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;
    0.5 * ((2.0 * p1)
        + (-p0 + p2) * t
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * t3)
}

#[inline]
fn half_width_for(acc: f64) -> f64 {
    0.5 * (WIDTH_BASE + WIDTH_SCALE * acc.max(0.0).sqrt())
}

/// Laplacian smoothing of a polyline (neighbour-average, endpoints fixed). Used to
/// relax the discrete-routing staircase before splining. Slight chord-sag at this
/// step size (~m on a 50 km planet) is negligible vs the visual win.
fn laplacian_smooth(pts: &[DVec3], iters: usize) -> Vec<DVec3> {
    let n = pts.len();
    let mut cur = pts.to_vec();
    if n < 3 {
        return cur;
    }
    let mut tmp = cur.clone();
    for _ in 0..iters {
        for i in 1..n - 1 {
            tmp[i] = cur[i] * 0.5 + (cur[i - 1] + cur[i + 1]) * 0.25;
        }
        std::mem::swap(&mut cur, &mut tmp);
    }
    cur
}

/// Decode a flat cube-sphere cell index → (face, x, y). Mirrors erosion's
/// `cell_to_face_xy`.
#[inline]
fn cell_to_fxy(c: u32, res: usize) -> (usize, usize, usize) {
    let c = c as usize;
    let rr = res * res;
    (c / rr, (c % rr) % res, (c % rr) / res)
}

const SINK: u32 = u32::MAX;

/// Priority-Flood min-heap item (pop the LOWEST filled height first; idx breaks ties
/// deterministically). `BinaryHeap` is a max-heap, so `Ord` is reversed.
#[derive(Clone, Copy, PartialEq)]
struct HeapItem {
    filled: f32,
    idx: u32,
}
impl Eq for HeapItem {}
impl Ord for HeapItem {
    fn cmp(&self, o: &Self) -> Ordering {
        o.filled
            .partial_cmp(&self.filled)
            .unwrap_or(Ordering::Equal)
            .then_with(|| o.idx.cmp(&self.idx))
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

/// Extract the river network from terrain HEIGHT + PRECIPITATION (no erosion data):
///   1. Sample broad `height()` + `moisture()` (precip) over the grid.
///   2. Priority-Flood (Barnes 2014) with flow-direction assignment: seed every sea
///      cell, pop lowest, fill each neighbour to `max(h, parent+ε)` and point it
///      downstream at the cell that reached it → a depression-filled drainage TREE
///      where every land cell drains downhill to the sea (connected by construction).
///   3. Accumulate precip down the tree (wetter basins → bigger rivers).
///   4. Threshold the accumulation → channel cells; collapse to a junction graph
///      (source = in-degree 0, confluence = ≥2) and sweep ribbons (width ∝ √acc).
/// See docs/rivers-research.md. `radius`/`height_scale` drape the water line.
pub fn trace_rivers(hf: &dyn HeightField, radius: f64, height_scale: f64) -> RiverMesh {
    let mut mesh = RiverMesh::default();
    let res = GRID_RES;
    let n = 6 * res * res;
    // A reach segment is ~one grid step / SMOOTH_SUB; a jump beyond a few grid steps
    // is a cube-face seam crossing → don't bridge it with a giant quad.
    let max_seg = 4.0 * tex_ang(res) * radius;

    // --- 1. Sample routing height + precipitation per cell. ---
    let mut height = vec![0.0f32; n];
    let mut precip = vec![0.0f32; n];
    for face in 0..6usize {
        for y in 0..res {
            for x in 0..res {
                let c = texel_index(face, x, y, res);
                let dir = texel_to_dir(face, x, y, res);
                height[c] = hf.height(dir, ROUTE_LEVEL) as f32;
                precip[c] = hf.moisture(dir).max(0.0);
            }
        }
    }

    // --- 2. Priority-Flood with flow-direction (Barnes 2014). Seed the sea (h<0);
    //        each cell drains to whoever reached it → a tree rooted at the ocean.
    //        `order` = ascending-fill pop order (reverse = upstream→downstream). ---
    let mut filled = height.clone();
    let mut downstream = vec![SINK; n];
    let mut processed = vec![false; n];
    let mut order: Vec<u32> = Vec::with_capacity(n);
    let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
    for c in 0..n {
        if height[c] < 0.0 {
            processed[c] = true;
            heap.push(HeapItem { filled: filled[c], idx: c as u32 });
        }
    }
    if heap.is_empty() {
        return mesh; // no ocean outlet → no rivers
    }
    while let Some(item) = heap.pop() {
        let c = item.idx as usize;
        order.push(c as u32);
        let (f, x, y) = cell_to_fxy(c as u32, res);
        for k in 0..8 {
            let nt = neighbor_texel(f, x, y, NB_DX[k], NB_DY[k], res);
            let nc = texel_index(nt.face, nt.x, nt.y, res);
            if processed[nc] {
                continue;
            }
            processed[nc] = true;
            let nf = height[nc].max(item.filled + FILL_EPS);
            filled[nc] = nf;
            downstream[nc] = c as u32;
            heap.push(HeapItem { filled: nf, idx: nc as u32 });
        }
    }

    // --- 3. Accumulate precipitation downstream, then log-normalize to 0..1. ---
    let mut flow: Vec<f64> = precip.iter().map(|&p| p as f64).collect();
    for &c in order.iter().rev() {
        let d = downstream[c as usize];
        if d != SINK {
            flow[d as usize] += flow[c as usize];
        }
    }
    let fmax = flow.iter().cloned().fold(0.0f64, f64::max);
    let lmax = (1.0 + fmax).ln().max(1e-9);
    let acc: Vec<f32> = flow.iter().map(|&q| ((1.0 + q).ln() / lmax) as f32).collect();

    // --- 4. Channel mask (land + enough discharge) + in-degree on the tree. ---
    let mut is_channel = vec![false; n];
    for c in 0..n {
        if height[c] >= 0.0 && acc[c] as f64 >= CHANNEL_THRESHOLD {
            is_channel[c] = true;
        }
    }
    let mut indeg = vec![0u16; n];
    for c in 0..n {
        if is_channel[c] {
            let d = downstream[c];
            if d != SINK && is_channel[d as usize] {
                indeg[d as usize] += 1;
            }
        }
    }

    // --- 5. Walk one reach from every node (source = in-degree 0; confluence = ≥2)
    //        down to the next node / outlet. In-degree-1 cells are interior. ---
    let mut cw: Vec<DVec3> = Vec::new();
    let mut chw: Vec<f64> = Vec::new();
    for c in 0..n {
        if !is_channel[c] || indeg[c] == 1 {
            continue;
        }
        walk_reach(
            hf, radius, height_scale, res, c as u32,
            &downstream, &indeg, &is_channel, &acc, max_seg,
            &mut cw, &mut chw, &mut mesh,
        );
    }
    mesh
}

/// Collect one graph edge (a chain of channel cells) from `start` downstream to the
/// next confluence or outlet, drape it on the fine terrain, and emit the ribbon.
#[allow(clippy::too_many_arguments)]
fn walk_reach(
    hf: &dyn HeightField,
    radius: f64,
    height_scale: f64,
    res: usize,
    start: u32,
    downstream: &[u32],
    indeg: &[u16],
    is_channel: &[bool],
    acc: &[f32],
    max_seg: f64,
    cw: &mut Vec<DVec3>,
    chw: &mut Vec<f64>,
    mesh: &mut RiverMesh,
) {
    cw.clear();
    chw.clear();
    let mut cur = start;
    for step in 0..=MAX_STEPS {
        let (f, x, y) = cell_to_fxy(cur, res);
        let dir = texel_to_dir(f, x, y, res);
        // Drape on the FINE terrain so the ribbon sits in the real valley.
        let h = hf.height(dir, DRAPE_LEVEL);
        cw.push(dir * (radius + h * height_scale + LIFT_M));
        chw.push(half_width_for(acc[cur as usize] as f64));

        // Reached a downstream confluence node (after moving) → this reach ends here.
        if step > 0 && indeg[cur as usize] >= 2 {
            break;
        }
        let d = downstream[cur as usize];
        if d == SINK || !is_channel[d as usize] {
            break; // outlet (sea / sub-threshold)
        }
        cur = d;
    }
    emit_ribbon(cw, chw, max_seg, mesh);
}

/// Catmull-Rom-subdivide the centerline + sweep a double-sided water ribbon.
/// Reaches shorter than `MIN_REACH_POINTS` are dropped (merge-stubs); segments
/// longer than `max_seg` (face-seam jumps) are left un-triangulated (no quad
/// spanning the planet).
fn emit_ribbon(cw: &[DVec3], chw: &[f64], max_seg: f64, mesh: &mut RiverMesh) {
    let c = cw.len();
    if c < MIN_REACH_POINTS {
        return;
    }

    // Laplacian pre-smooth (endpoints fixed): relax the D8 / channel-preference
    // staircase — the router hops between axis-aligned channel cells when the true
    // flow is diagonal, so the raw centerline sawtooths. A few neighbour-averaging
    // passes straighten it into a smooth meander before the Catmull-Rom subdivision.
    let cw = laplacian_smooth(cw, LAPLACE_ITERS);
    let cw = cw.as_slice();

    // Subdivide cw/chw → smoothed s* (skip if too short).
    let mut sw: Vec<DVec3> = Vec::new();
    let mut shw: Vec<f64> = Vec::new();
    if SMOOTH_SUB <= 1 || c < 3 {
        sw.extend_from_slice(cw);
        shw.extend_from_slice(chw);
    } else {
        for i in 0..c - 1 {
            let i0 = i.saturating_sub(1);
            let i1 = i;
            let i2 = i + 1;
            let i3 = if i < c - 2 { i + 2 } else { c - 1 };
            for j in 0..SMOOTH_SUB {
                let t = j as f64 / SMOOTH_SUB as f64;
                sw.push(DVec3::new(
                    catmull(cw[i0].x, cw[i1].x, cw[i2].x, cw[i3].x, t),
                    catmull(cw[i0].y, cw[i1].y, cw[i2].y, cw[i3].y, t),
                    catmull(cw[i0].z, cw[i1].z, cw[i2].z, cw[i3].z, t),
                ));
                shw.push(chw[i1] + (chw[i2] - chw[i1]) * t);
            }
        }
        sw.push(cw[c - 1]);
        shw.push(chw[c - 1]);
    }

    let m = sw.len();
    if m < 2 {
        return;
    }

    let base = mesh.world.len() as u32;
    for i in 0..m {
        let w = sw[i];
        let p = w.normalize_or_zero(); // radial up
        // Path tangent from neighbours, projected into the tangent plane.
        let ia = if i > 0 { i - 1 } else { 0 };
        let ib = if i < m - 1 { i + 1 } else { m - 1 };
        let mut t = sw[ib] - sw[ia];
        t -= p * t.dot(p);
        let t = if t.length() > 1e-9 {
            t.normalize()
        } else {
            // Degenerate: any tangent in the plane.
            let alt = DVec3::new(p.y, -p.x, 0.0);
            alt.normalize_or_zero()
        };
        // Cross-channel = p × tangent (unit, tangent to the sphere).
        let e = p.cross(t).normalize_or_zero();
        let hw = shw[i];
        let n = [p.x as f32, p.y as f32, p.z as f32];
        // Left then right vertex for this centerline point.
        mesh.world.push(w + e * hw);
        mesh.normals.push(n);
        mesh.world.push(w - e * hw);
        mesh.normals.push(n);
    }

    // Double-sided triangles per segment (the shared pipeline culls BACK). Skip
    // any segment whose centerline jumps farther than `max_seg` (a cube-face seam
    // crossing) so we never bridge it with a huge quad.
    let max_seg2 = max_seg * max_seg;
    for i in 0..(m - 1) {
        if (sw[i + 1] - sw[i]).length_squared() > max_seg2 {
            continue;
        }
        let iu = i as u32;
        let la = base + 2 * iu; // left[i]
        let ra = la + 1; // right[i]
        let lb = base + 2 * (iu + 1); // left[i+1]
        let rb = lb + 1; // right[i+1]
        mesh.indices.extend_from_slice(&[
            la, ra, lb, ra, rb, lb, // front
            la, lb, ra, ra, lb, rb, // back
        ]);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const RES_TEST: usize = 32;

    /// A synthetic planet: north hemisphere is land sloping down to an equatorial
    /// sea, plus medium-scale bumps so the drainage branches. Uniform precipitation.
    struct ChannelHf;
    impl HeightField for ChannelHf {
        fn height(&self, dir: DVec3, _l: u8) -> f64 {
            // h>0 north (land, high pole), <0 south (sea); bumps dissect it.
            0.3 * dir.y + 0.05 * (dir.x * 8.0).sin() * (dir.z * 8.0).sin()
        }
        fn moisture(&self, _dir: DVec3) -> f32 {
            0.6
        }
    }

    #[test]
    fn rivers_wgsl_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(RIVERS_WGSL)
            .unwrap_or_else(|e| panic!("rivers.wgsl failed to parse: {e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("rivers.wgsl failed to validate: {e:?}"));
    }

    #[test]
    fn catmull_passes_through_control_points() {
        // Catmull-Rom passes through p1 at t=0 and p2 at t=1.
        assert!((catmull(0.0, 1.0, 2.0, 3.0, 0.0) - 1.0).abs() < 1e-12);
        assert!((catmull(0.0, 1.0, 2.0, 3.0, 1.0) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn half_width_monotonic_in_discharge() {
        assert!(half_width_for(0.0) < half_width_for(0.25));
        assert!(half_width_for(0.25) < half_width_for(0.5));
        // Zero-discharge channel still has the base width.
        assert!((half_width_for(0.0) - 0.5 * WIDTH_BASE).abs() < 1e-9);
    }

    #[test]
    fn traces_a_coherent_channel() {
        let mesh = trace_rivers(&ChannelHf, 50_000.0, 1_200.0);
        // The synthetic channel must produce SOME ribbon geometry, well-formed.
        assert!(!mesh.world.is_empty(), "no river geometry traced");
        assert_eq!(mesh.world.len(), mesh.normals.len());
        assert!(!mesh.indices.is_empty());
        assert!(mesh.indices.len() % 3 == 0, "indices not whole triangles");
        let vmax = *mesh.indices.iter().max().unwrap();
        assert!((vmax as usize) < mesh.world.len(), "index out of range");
        // All positions are finite and near the planet surface (~radius).
        for w in &mesh.world {
            assert!(w.is_finite());
            let r = w.length();
            assert!((r - 50_000.0).abs() < 5_000.0, "river vert radius {r} off-surface");
        }
        let _ = RES_TEST;
    }

    #[test]
    fn no_ocean_traces_nothing() {
        // All-land (no h<0 cell) → Priority-Flood has no sea seed → no outlet, no
        // drainage tree, no rivers.
        struct FlatHf;
        impl HeightField for FlatHf {
            fn height(&self, _d: DVec3, _l: u8) -> f64 {
                0.1
            }
        }
        let mesh = trace_rivers(&FlatHf, 50_000.0, 1_200.0);
        assert!(mesh.world.is_empty() && mesh.indices.is_empty());
    }
}

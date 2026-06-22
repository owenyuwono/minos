//! `rivers` — drainage-network water surface traced from the baked erosion flow
//! field, drawn as translucent blue ribbons. A 1:1-in-spirit port of ki's
//! `RiverNetwork.ts`, adapted to enki's `HeightField` trait (`flow_accum_at` /
//! `flow_dir_at` / `lake_mask_at` / `height`) + the cube-sphere `cubemap` helpers.
//!
//! Why a MESH and not per-vertex paint: rivers are 1–2 erosion-texels wide, far
//! thinner than the terrain mesh's vertex spacing, so painting `wetness` into the
//! terrain albedo aliases away at every LOD. A traced ribbon is its own geometry,
//! independent of terrain tessellation, so it stays crisp at any zoom.
//!
//! Tracing: seed from channel HEADS (a cell where acc ≥ threshold but one step
//! upstream is below it), then walk downhill on a cube-sphere `visited` grid —
//! reaches that enter an already-traced cell merge into it (a converging tree).
//! Each reach's coarse centerline is Catmull-Rom subdivided and swept into a
//! ribbon whose half-width ∝ √discharge (hydraulic geometry). Built ONCE on the
//! main thread when the heightfield arrives; only the camera-relative position
//! buffer is rewritten per frame (mirrors `markers.rs`).

use bytemuck::cast_slice;
use enki_planet::cubemap::{neighbor_texel, tex_ang, texel_index, texel_to_dir};
use enki_planet::height::HeightField;
use enki_render::{frame::FrameUniforms, material::ChunkPush};
use enki_rhi::{vk, BufferHandle, PipelineHandle, Rhi, RhiError};
use glam::DVec3;
use std::sync::Arc;

use crate::markers::{draw_ribbon, overlay_pipeline};

const RIVERS_WGSL: &str = include_str!("rivers.wgsl");

// --- Tracer tuning (ponytail: tunables, not derived — all USER-verified visually) ---

/// Seed / visited grid resolution per cube face.
const GRID_RES: usize = 96;
/// Channel-initiation threshold on the log-normalized acc 0..1 field. The 4-pass
/// blur on `flow_accum` caps the field at ~0.48 (CDF over land: ≥0.20 ~23%,
/// ≥0.25 ~10%, ≥0.30 ~3.6%). High here → only SIGNIFICANT channels seed heads,
/// so the network is a sparse trunk+tributary tree, not the whole drainage basin.
const THRESHOLD: f64 = 0.26;
/// Discrete downstream step length as a multiple of one grid texel's pitch.
const STEP_MUL: f64 = 1.0;
/// `HeightField` LOD used to drape the water line (matches the fine terrain).
const DRAPE_LEVEL: u8 = 8;
/// Per-reach loop guard.
const MAX_STEPS: usize = 4000;
/// Catmull-Rom substeps per traced segment (kills the ~grid-pitch angularity).
const SMOOTH_SUB: usize = 4;
/// Floor on acc before a cell is even considered as a seed (skip flat interiors).
const SEED_FLOOR: f64 = 0.04;
/// Skip reaches shorter than this many TRACED points. Adjacent seeds in one basin
/// hit `visited` after a step or two and would emit tiny fragments that stipple
/// the whole continent — this drops those merge-stubs, keeping real trunks/tribs.
const MIN_REACH_POINTS: usize = 5;

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

/// Trace the whole network. `radius`/`height_scale` drape the water line; the
/// drainage comes from `hf.flow_accum_at` / `flow_dir_at` / `lake_mask_at`.
pub fn trace_rivers(hf: &dyn HeightField, radius: f64, height_scale: f64) -> RiverMesh {
    let mut mesh = RiverMesh::default();
    let res = GRID_RES;
    let mut visited = vec![0u8; 6 * res * res];
    let step_arc = tex_ang(res) * STEP_MUL;
    let (cos_arc, sin_arc) = (step_arc.cos(), step_arc.sin());
    // A normal smoothed segment is ~`step_arc*radius/SMOOTH_SUB`; anything far
    // beyond one whole grid step is a face-seam jump → don't bridge it (no quad
    // spanning the planet). Guard at ~4 grid steps of world distance.
    let max_seg = 4.0 * tex_ang(res) * radius;

    // Per-reach centerline (world position) + half-width, reused per trace.
    let mut cw: Vec<DVec3> = Vec::new();
    let mut chw: Vec<f64> = Vec::new();

    // Seed from channel heads (sea/lake/low-acc cells are skipped).
    for face in 0..6usize {
        for y in 0..res {
            for x in 0..res {
                let dir = texel_to_dir(face, x, y, res);
                let a = hf.flow_accum_at(dir) as f64;
                if visited[texel_index(face, x, y, res)] == 1 || a < SEED_FLOOR {
                    continue;
                }
                let h = hf.height(dir, DRAPE_LEVEL);
                if h < 0.0 || hf.lake_mask_at(dir) as f64 > 0.5 || a < THRESHOLD {
                    continue;
                }
                // Head test: one step UPWIND (rotate dir against the flow) is below
                // threshold. Incoherent flow → treat as a head.
                let f = project_flow(hf.flow_dir_at(dir), dir);
                let is_head = if f.length() > 1e-6 {
                    let fhat = f.normalize();
                    let up = (dir * cos_arc - fhat * sin_arc).normalize();
                    (hf.flow_accum_at(up) as f64) < THRESHOLD
                } else {
                    true
                };
                if !is_head {
                    continue;
                }
                trace_reach(
                    hf, radius, height_scale, res, face, x, y, max_seg,
                    &mut visited, &mut cw, &mut chw, &mut mesh,
                );
            }
        }
    }
    mesh
}

/// Project a flow vector into `dir`'s tangent plane (drop the radial component).
#[inline]
fn project_flow(f: DVec3, dir: DVec3) -> DVec3 {
    f - dir * f.dot(dir)
}

#[allow(clippy::too_many_arguments)]
fn trace_reach(
    hf: &dyn HeightField,
    radius: f64,
    height_scale: f64,
    res: usize,
    sf: usize,
    sx: usize,
    sy: usize,
    max_seg: f64,
    visited: &mut [u8],
    cw: &mut Vec<DVec3>,
    chw: &mut Vec<f64>,
    mesh: &mut RiverMesh,
) {
    let (mut cf, mut cx, mut cy) = (sf, sx, sy);
    cw.clear();
    chw.clear();
    let mut last_dir: Option<DVec3> = None;

    for _ in 0..MAX_STEPS {
        let idx = texel_index(cf, cx, cy, res);
        let already = visited[idx] == 1;

        let dir = texel_to_dir(cf, cx, cy, res);
        let acc = hf.flow_accum_at(dir) as f64;
        let h = hf.height(dir, DRAPE_LEVEL);
        // height() already carries the V-incision, so drape at the surface + lift.
        let r = radius + h * height_scale + LIFT_M;
        cw.push(dir * r);
        chw.push(half_width_for(acc));

        if already {
            break; // merged into an already-traced reach
        }
        visited[idx] = 1;
        if h < 0.0 || hf.lake_mask_at(dir) as f64 > 0.5 {
            break;
        }

        // Discrete downstream step: the 8-neighbour the flow points toward. Two
        // reaches entering one cell pick the SAME neighbour → they merge.
        let f = project_flow(hf.flow_dir_at(dir), dir);
        let g = if f.length() > 1e-3 {
            let g = f.normalize();
            last_dir = Some(g);
            g
        } else if let Some(g) = last_dir {
            g
        } else {
            break;
        };

        let mut best_dot = f64::NEG_INFINITY;
        let (mut bf, mut bx, mut by) = (cf, cx, cy);
        for k in 0..8 {
            let nt = neighbor_texel(cf, cx, cy, NB_DX[k], NB_DY[k], res);
            let ndir = texel_to_dir(nt.face, nt.x, nt.y, res);
            let d = (ndir - dir).dot(g);
            if d > best_dot {
                best_dot = d;
                bf = nt.face;
                bx = nt.x;
                by = nt.y;
            }
        }
        if best_dot <= 0.0 || (bf == cf && bx == cx && by == cy) {
            break;
        }
        cf = bf;
        cx = bx;
        cy = by;
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

    /// A synthetic field with a single coherent downhill channel: flow points
    /// toward +Y (north), discharge ramps up toward the equator, no lakes.
    struct ChannelHf;
    impl HeightField for ChannelHf {
        fn height(&self, dir: DVec3, _l: u8) -> f64 {
            // Land everywhere except a polar sink (h<0 near +Y) so traces terminate.
            0.2 - 0.5 * dir.y.max(0.0)
        }
        fn flow_accum_at(&self, dir: DVec3) -> f32 {
            // High discharge in a band near the equator, fading poleward.
            (0.45 * (1.0 - dir.y.abs())) as f32
        }
        fn flow_dir_at(&self, dir: DVec3) -> DVec3 {
            // Downhill = toward +Y, projected to the tangent plane.
            project_flow(DVec3::Y, dir)
        }
        fn lake_mask_at(&self, _dir: DVec3) -> f32 {
            0.0
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
    fn empty_field_traces_nothing() {
        // Default HeightField (flow_accum 0 everywhere) → no seeds, no geometry.
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

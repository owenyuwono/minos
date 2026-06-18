//! Runtime GPU rendering of the baked cluster DAG (N2).
//!
//! This module turns a [`ClusterAsset`] into GPU buffers and drives the
//! GPU-driven pipeline: a compute pass selects the crack-free LOD cut + frustum-
//! culls clusters into a visible list, then a single indirect, vertex-pulling
//! draw rasterizes them. A push-constant debug mode flat-colors by
//! triangle / cluster / LOD (Unreal-style), mirroring [`crate::debug::id_color`].
//!
//! v1 layout notes:
//! - All geometry is flattened into flat storage buffers (no vertex buffers); the
//!   vertex shader pulls via `@builtin(vertex_index)`.
//! - Each visible cluster contributes a fixed `MAX_CLUSTER_TRIS * 3` vertices;
//!   triangles past a cluster's real `tri_count` are emitted degenerate. Simple,
//!   ~30% vertex waste — fine for v1; a tight triangle-append comes later.
//!
//! The GPU object [`NaniteRenderer`] (buffers, pipelines, per-frame ring) is built
//! on top of this flattening in the following steps.

use std::collections::{HashMap, HashSet};

use crate::bake::cluster_build::{MAX_CLUSTER_TRIS, MAX_CLUSTER_VERTS};
use crate::cluster::ClusterAsset;
use crate::residency::ClusterEntry;
use crate::stream::PagePool;

use enki_render::frame::FrameUniforms;
use enki_rhi::{
    vk, BindingDesc, BufferHandle, ComputePipelineDesc, GraphicsPipelineDesc, PipelineHandle, Rhi,
    RhiError,
};
use glam::{DVec3, Mat4};

/// Cull / LOD-cut compute shader (WGSL). Entry point: `cs_cull`.
pub const CULL_WGSL: &str = include_str!("shaders/nanite_cull.wgsl");
/// Vertex-pulling draw shader (WGSL). Entry points: `vs_pull`, `fs_color`.
pub const DRAW_WGSL: &str = include_str!("shaders/nanite_draw.wgsl");

/// Per-cluster metadata, std430-compatible (matches the WGSL `ClusterMeta`).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuClusterMeta {
    /// Group bounding sphere: center.xyz (patch-origin-relative), radius.
    pub bounds: [f32; 4],
    /// Parent group's bounding sphere (encloses `bounds`).
    pub parent_bounds: [f32; 4],
    /// `[self_error, parent_error, lod, _pad]`. `parent_error` is `+inf` for roots.
    pub err: [f32; 4],
    /// `[tri_offset, tri_count, node_index, _pad]` — index range into `tris`, plus
    /// the node this cluster belongs to (selects its per-node origin/translation).
    pub range: [u32; 4],
}

/// Stable per-cluster id for the debug view (face/node in the high bits, cluster
/// index in the low bits). Independent of the cluster's transient GPU buffer slot,
/// so debug colors stay put when the streaming set is re-packed.
fn stable_id(node: u32, cluster: u32) -> u32 {
    (node << 20) | (cluster & 0x000F_FFFF)
}

// ── Per-frame uniform (std430, matches WGSL `FrameData`) ─────────────────────

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuFrameData {
    view_proj: [[f32; 4]; 4],
    /// tau_px, screen_h, cot_half_fov, cluster_count
    lod: [f32; 4],
    // Lights — mirror FrameUniforms so the lit debug mode matches the terrain.
    sun0_dir: [f32; 4],
    sun0_color: [f32; 4],
    sun1_dir: [f32; 4],
    sun1_color: [f32; 4],
    hemi_sky: [f32; 4],
    hemi_ground: [f32; 4],
    ambient: [f32; 4],
    planes: [[f32; 4]; 6],
    debug: [u32; 4],
}

/// Gribb–Hartmann frustum planes from a view-projection matrix, normalized.
/// Plane test: a point `p` is inside iff `dot(plane.xyz, p) + plane.w >= 0`.
fn extract_planes(vp: Mat4) -> [[f32; 4]; 6] {
    let r0 = vp.row(0);
    let r1 = vp.row(1);
    let r2 = vp.row(2);
    let r3 = vp.row(3);
    let raw = [
        r3 + r0, // left
        r3 - r0, // right
        r3 + r1, // bottom
        r3 - r1, // top
        r3 + r2, // near
        r3 - r2, // far
    ];
    let mut out = [[0.0f32; 4]; 6];
    for (i, p) in raw.iter().enumerate() {
        let len = (p.x * p.x + p.y * p.y + p.z * p.z).sqrt();
        let inv = if len > 1e-8 { 1.0 / len } else { 1.0 };
        out[i] = [p.x * inv, p.y * inv, p.z * inv, p.w * inv];
    }
    out
}

// ── NaniteRenderer ───────────────────────────────────────────────────────────

/// Resident-cluster pool capacity (number of GPU cluster slots). The DAG may be
/// far larger; only the near-cut working set is resident at any time.
const POOL_MAX_CLUSTERS: usize = 49_152;
/// Floats per cluster slot (fixed stride): `MAX_CLUSTER_VERTS` verts × 9 floats.
const SLOT_FLOATS: usize = MAX_CLUSTER_VERTS * 9;
/// Indices per cluster slot (fixed stride): `MAX_CLUSTER_TRIS` tris × 3.
const SLOT_INDICES: usize = MAX_CLUSTER_TRIS * 3;
const POOL_MAX_FLOATS: usize = POOL_MAX_CLUSTERS * SLOT_FLOATS;
const POOL_MAX_INDICES: usize = POOL_MAX_CLUSTERS * SLOT_INDICES;
/// `node_xlat` capacity (one per cube face; padded for headroom).
const POOL_MAX_NODES: usize = 64;

/// Slots to retire this update: resident keys that are no longer desired. Pure so
/// it can be unit-tested without a GPU.
fn residency_evictions(
    resident: &HashMap<u32, u32>,
    desired: &HashSet<u32>,
) -> Vec<(u32, u32)> {
    resident
        .iter()
        .filter(|(k, _)| !desired.contains(*k))
        .map(|(&k, &s)| (k, s))
        .collect()
}

/// Per-frame-in-flight working set (the geometry pool itself is shared).
struct NaniteFrame {
    visible_buf: BufferHandle,
    args_buf: BufferHandle,
    frame_buf: BufferHandle,
    node_xlat_buf: BufferHandle,
    /// The active cluster-slot list for this frame (cull iterates it).
    active_slots_buf: BufferHandle,
    set: vk::DescriptorSet,
}

/// GPU-driven renderer with an **incremental cluster page pool**: each resident
/// cluster keeps a stable GPU slot (fixed-stride geometry), so a streaming
/// reselection only uploads the *added* clusters and frees the *removed* ones —
/// no whole-set re-pack, no per-frame hitch. The compute pass culls the active
/// slots into a visible list; a single indirect, vertex-pulling draw rasterizes
/// them, with a debug color mode.
///
/// **Use-after-free safety:** a cluster's geometry is only WRITTEN to a *free*
/// slot (never one an in-flight frame references), and an evicted slot is only
/// returned to the free list by the [`PagePool`] timeline graveyard once the GPU
/// has finished every frame that referenced it. The shared geometry buffers are
/// thus never mutated in a region a live frame is reading.
///
/// All GPU resources are owned by the RHI's stores and freed at RHI teardown, so
/// this holds only handles — drop it before the `Rhi`.
#[allow(dead_code)] // set_layout is a resident-ownership handle
pub struct NaniteRenderer {
    set_layout: vk::DescriptorSetLayout,
    cull_pipeline: PipelineHandle,
    draw_pipeline: PipelineHandle,
    // Shared, slot-indexed geometry pool (NOT per-frame).
    verts_buf: BufferHandle,
    tris_buf: BufferHandle,
    meta_buf: BufferHandle,
    pool: PagePool,
    /// stable cluster key (`stable_id(face, cluster)`) → GPU slot.
    resident: HashMap<u32, u32>,
    /// Per-face planet-space origins (node index in `meta.range[2]` is the face).
    face_origins: Vec<DVec3>,
    /// Current active slot list (uploaded to each frame's `active_slots_buf`).
    active_slots: Vec<u32>,
    frames: Vec<NaniteFrame>,
    debug_mode: u32,
    /// Visible cluster count read back from the last completed cull (diagnostic).
    last_visible: u32,
}

impl NaniteRenderer {
    /// Build the cull + draw pipelines and the shared cluster page pool, then seed
    /// `initial` as permanently-resident clusters (usually empty — the streamer
    /// fills the pool via [`Self::update_residency`]).
    pub fn new(rhi: &mut Rhi, initial: &[ClusterAsset]) -> Result<Self, RhiError> {
        // One descriptor set layout (8 storage bindings) for both pipelines.
        // 0..2 = shared geometry (verts/tris/meta); 3..7 = per-frame working set.
        // FRAGMENT is required because the lit draw mode reads `frame` (binding 5).
        let stages = vk::ShaderStageFlags::COMPUTE
            | vk::ShaderStageFlags::VERTEX
            | vk::ShaderStageFlags::FRAGMENT;
        let bindings: Vec<BindingDesc> = (0u32..8)
            .map(|binding| BindingDesc {
                binding,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stages,
            })
            .collect();
        let set_layout = rhi.create_descriptor_set_layout(&bindings)?;

        // Cull (compute) pipeline.
        let cull_mod = rhi.create_shader_module(CULL_WGSL)?;
        let cull_pipeline = rhi.create_compute_pipeline(&ComputePipelineDesc {
            shader: cull_mod,
            entry: "cs_cull",
            push_constant_size: 0,
            set_layouts: &[set_layout],
        })?;
        rhi.destroy_shader_module(cull_mod);

        // Draw (vertex-pulling) pipeline.
        let draw_mod = rhi.create_shader_module(DRAW_WGSL)?;
        let draw_pipeline = rhi.create_graphics_pipeline_pulling(&GraphicsPipelineDesc {
            shader: draw_mod,
            vs_entry: "vs_pull",
            fs_entry: "fs_color",
            push_constant_size: 4, // u32 debug mode
            set0_layout: set_layout,
            color_format: rhi.swapchain_format(),
            depth_format: vk::Format::D32_SFLOAT,
            samples: rhi.msaa_samples(),
            blend: false,
            fill: true,
        })?;
        rhi.destroy_shader_module(draw_mod);

        let storage = vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST;

        // Shared geometry pool (one set, fixed-stride per slot).
        let verts_buf = rhi.create_gpu_buffer((POOL_MAX_FLOATS * 4) as u64, true, storage)?;
        let tris_buf = rhi.create_gpu_buffer((POOL_MAX_INDICES * 4) as u64, true, storage)?;
        let meta_buf = rhi.create_gpu_buffer(
            (POOL_MAX_CLUSTERS * std::mem::size_of::<GpuClusterMeta>()) as u64,
            true,
            storage,
        )?;

        // Per-frame working set.
        let fif = rhi.frames_in_flight();
        let mut frames = Vec::with_capacity(fif);
        for _ in 0..fif {
            let visible_buf = rhi.create_gpu_buffer((POOL_MAX_CLUSTERS * 4) as u64, true, storage)?;
            let args_buf = rhi.create_gpu_buffer(
                16,
                true,
                storage | vk::BufferUsageFlags::INDIRECT_BUFFER,
            )?;
            let frame_buf =
                rhi.create_gpu_buffer(std::mem::size_of::<GpuFrameData>() as u64, true, storage)?;
            let node_xlat_buf = rhi.create_gpu_buffer((POOL_MAX_NODES * 16) as u64, true, storage)?;
            let active_slots_buf =
                rhi.create_gpu_buffer((POOL_MAX_CLUSTERS * 4) as u64, true, storage)?;
            let set = rhi.allocate_descriptor_set(set_layout)?;
            rhi.write_storage_binding(set, 0, verts_buf)?;
            rhi.write_storage_binding(set, 1, tris_buf)?;
            rhi.write_storage_binding(set, 2, meta_buf)?;
            rhi.write_storage_binding(set, 3, visible_buf)?;
            rhi.write_storage_binding(set, 4, args_buf)?;
            rhi.write_storage_binding(set, 5, frame_buf)?;
            rhi.write_storage_binding(set, 6, node_xlat_buf)?;
            rhi.write_storage_binding(set, 7, active_slots_buf)?;
            frames.push(NaniteFrame {
                visible_buf,
                args_buf,
                frame_buf,
                node_xlat_buf,
                active_slots_buf,
                set,
            });
        }

        log::info!("NaniteRenderer: page pool {POOL_MAX_CLUSTERS} slots, {fif} frames-in-flight");

        let mut r = Self {
            set_layout,
            cull_pipeline,
            draw_pipeline,
            verts_buf,
            tris_buf,
            meta_buf,
            pool: PagePool::new(POOL_MAX_CLUSTERS),
            resident: HashMap::new(),
            face_origins: Vec::new(),
            active_slots: Vec::new(),
            frames,
            debug_mode: 0,
            last_visible: 0,
        };

        // Seed `initial` as permanently-resident clusters (usually empty).
        if !initial.is_empty() {
            r.face_origins = initial.iter().map(|a| a.patch_origin).collect();
            for (face, a) in initial.iter().enumerate() {
                for ci in 0..a.clusters.len() {
                    let Some(slot) = r.pool.alloc() else { break };
                    r.write_cluster(rhi, slot, a, ci, face as u32)?;
                    r.resident.insert(stable_id(face as u32, ci as u32), slot);
                }
            }
            r.active_slots = r.resident.values().copied().collect();
        }
        Ok(r)
    }

    /// Visible cluster count from the most recently completed cull pass.
    /// Diagnostic only — ~2 frames stale (read at the start of the slot's reuse).
    pub fn last_visible_clusters(&self) -> u32 {
        self.last_visible
    }

    /// Number of clusters currently resident in the GPU pool.
    pub fn resident_count(&self) -> usize {
        self.resident.len()
    }

    /// Write one cluster's geometry + meta into GPU `slot` (fixed-stride). Only
    /// ever called on a FREE slot, so it never races an in-flight frame.
    fn write_cluster(
        &mut self,
        rhi: &mut Rhi,
        slot: u32,
        asset: &ClusterAsset,
        cluster_idx: usize,
        face: u32,
    ) -> Result<(), RhiError> {
        let c = &asset.clusters[cluster_idx];

        let mut vbuf: Vec<f32> = Vec::with_capacity(c.vertices.len() * 9);
        for v in &c.vertices {
            vbuf.extend_from_slice(&v.position);
            vbuf.extend_from_slice(&v.normal);
            vbuf.extend_from_slice(&v.color);
        }
        let verts_off = slot as u64 * SLOT_FLOATS as u64 * 4;
        rhi.write_storage_bytes_at(self.verts_buf, verts_off, bytemuck::cast_slice(&vbuf))?;

        // Triangles store GLOBAL vertex indices (slot base + local) so the draw can
        // address by fixed stride without a per-cluster vertex offset.
        let vbase = slot * MAX_CLUSTER_VERTS as u32;
        let mut tbuf: Vec<u32> = Vec::with_capacity(c.triangles.len() * 3);
        for t in &c.triangles {
            tbuf.push(vbase + t[0] as u32);
            tbuf.push(vbase + t[1] as u32);
            tbuf.push(vbase + t[2] as u32);
        }
        let tris_off = slot as u64 * SLOT_INDICES as u64 * 4;
        rhi.write_storage_bytes_at(self.tris_buf, tris_off, bytemuck::cast_slice(&tbuf))?;

        let meta = GpuClusterMeta {
            bounds: [c.bounds.center.x, c.bounds.center.y, c.bounds.center.z, c.bounds.radius],
            parent_bounds: [
                c.parent_bounds.center.x,
                c.parent_bounds.center.y,
                c.parent_bounds.center.z,
                c.parent_bounds.radius,
            ],
            err: [c.self_error, c.parent_error, c.lod as f32, 0.0],
            // range[0] (tri_offset) is implicit (slot * MAX_TRIS) with fixed stride.
            range: [0, c.triangles.len() as u32, face, stable_id(face, cluster_idx as u32)],
        };
        let meta_off = slot as u64 * std::mem::size_of::<GpuClusterMeta>() as u64;
        rhi.write_storage_bytes_at(self.meta_buf, meta_off, bytemuck::bytes_of(&meta))?;
        Ok(())
    }

    /// Reconcile the GPU-resident cluster set to `selected` (indices into
    /// `entries`, nearest-first). Uploads only the newly-needed clusters and frees
    /// the no-longer-needed ones — O(delta), not O(working set), so there's no
    /// per-frame re-pack hitch.
    ///
    /// `completed_frame` is the timeline value the GPU has finished (reclaims
    /// graveyard slots); `retire_at` is when evicted slots become reclaimable (use
    /// `current_frame + frames_in_flight`).
    pub fn update_residency(
        &mut self,
        rhi: &mut Rhi,
        completed_frame: u64,
        retire_at: u64,
        assets: &[ClusterAsset],
        entries: &[ClusterEntry],
        selected: &[u32],
    ) -> Result<(), RhiError> {
        self.face_origins = assets.iter().map(|a| a.patch_origin).collect();

        // Reclaim slots the GPU has finished with (graveyard → free list).
        self.pool.collect(completed_frame);

        // Desired keys for this camera.
        let mut desired: HashSet<u32> = HashSet::with_capacity(selected.len());
        for &si in selected {
            let e = &entries[si as usize];
            desired.insert(stable_id(e.face, e.cluster));
        }

        // Evict: resident keys no longer desired → retire their slots (geometry
        // left intact until the slot is reallocated, so in-flight frames are safe).
        for (key, slot) in residency_evictions(&self.resident, &desired) {
            self.pool.retire(slot, retire_at);
            self.resident.remove(&key);
        }

        // Add: desired clusters not yet resident, in nearest-first order so a full
        // pool drops only the farthest.
        let mut full = false;
        for &si in selected {
            let e = &entries[si as usize];
            let key = stable_id(e.face, e.cluster);
            if self.resident.contains_key(&key) {
                continue;
            }
            match self.pool.alloc() {
                Some(slot) => {
                    self.write_cluster(rhi, slot, &assets[e.face as usize], e.cluster as usize, e.face)?;
                    self.resident.insert(key, slot);
                }
                None => {
                    full = true;
                    break;
                }
            }
        }
        if full {
            log::warn!(
                "Nanite page pool full: {} clusters resident, far clusters skipped",
                self.resident.len()
            );
        }

        // Rebuild the active slot list (cull iterates it).
        self.active_slots = self.resident.values().copied().collect();
        Ok(())
    }

    /// Update per-frame uniforms + the active-slot list for frame slot `fi`. Call
    /// each frame after `begin_frame`, before `record_cull`. `fu` supplies the
    /// rotation-only camera-relative view-proj AND the scene lights.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        rhi: &mut Rhi,
        fi: u32,
        camera_world: DVec3,
        fu: &FrameUniforms,
        screen_h: f32,
        fov_y: f32,
        tau_px: f32,
        debug_mode: u32,
        // `dither`: enable dithered LOD cross-fade (only useful with TAA on, which
        // resolves the per-frame stipple into a smooth blend; false = hard swap).
        // `frame_index`: varies the dither pattern each frame for TAA to average.
        dither: bool,
        frame_index: u32,
    ) -> Result<(), RhiError> {
        self.debug_mode = debug_mode;
        let args_buf = self.frames[fi as usize].args_buf;
        let frame_buf = self.frames[fi as usize].frame_buf;
        let node_xlat_buf = self.frames[fi as usize].node_xlat_buf;
        let active_slots_buf = self.frames[fi as usize].active_slots_buf;
        let active_count = self.active_slots.len() as u32;

        // Read back this slot's previous result (GPU-complete after begin_frame's
        // fence wait) before we overwrite the args — a diagnostic visible count.
        if let Ok(bytes) = rhi.read_storage_buffer(args_buf) {
            if bytes.len() >= 4 {
                let vc = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                self.last_visible = vc / (MAX_CLUSTER_TRIS as u32 * 3);
            }
        }

        // Upload this frame's active slot list (per-FiF buffer → no in-flight race).
        if active_count > 0 {
            rhi.write_storage_bytes(active_slots_buf, bytemuck::cast_slice(&self.active_slots))?;
        }

        // Per-face camera-relative translations (≤6; f64 subtract → f32, so each
        // face's small near-origin coords stay precise at planet scale).
        let node_xlat: Vec<[f32; 4]> = self
            .face_origins
            .iter()
            .map(|&origin| {
                let t = (origin - camera_world).as_vec3();
                [t.x, t.y, t.z, 0.0]
            })
            .collect();
        if !node_xlat.is_empty() {
            rhi.write_storage_bytes(node_xlat_buf, bytemuck::cast_slice(&node_xlat))?;
        }

        let cot = 1.0 / (fov_y * 0.5).tan();
        let view_proj = Mat4::from_cols_array_2d(&fu.view_proj);
        let data = GpuFrameData {
            view_proj: fu.view_proj,
            lod: [tau_px, screen_h, cot, active_count as f32],
            sun0_dir: fu.sun0_dir,
            sun0_color: fu.sun0_color,
            sun1_dir: fu.sun1_dir,
            sun1_color: fu.sun1_color,
            hemi_sky: fu.hemi_sky,
            hemi_ground: fu.hemi_ground,
            ambient: fu.ambient,
            planes: extract_planes(view_proj),
            // debug: [debug_mode, dither_enabled, frame_index, _]
            debug: [debug_mode, dither as u32, frame_index, 0],
        };
        rhi.write_storage_bytes(frame_buf, bytemuck::bytes_of(&data))?;
        // Reset indirect args: vertex_count = 0, instance_count = 1.
        let args0: [u32; 4] = [0, 1, 0, 0];
        rhi.write_storage_bytes(args_buf, bytemuck::cast_slice(&args0))?;
        Ok(())
    }

    /// Record the cull/LOD-cut compute dispatch + barriers. MUST be called BEFORE
    /// `begin_rendering` (compute dispatch is illegal inside a rendering instance).
    pub fn record_cull(&self, rhi: &Rhi, fi: u32) -> Result<(), RhiError> {
        let f = &self.frames[fi as usize];
        let layout = rhi.pipeline_layout(self.cull_pipeline)?;
        rhi.cmd_bind_pipeline(fi, vk::PipelineBindPoint::COMPUTE, self.cull_pipeline)?;
        rhi.cmd_bind_descriptor_set(fi, vk::PipelineBindPoint::COMPUTE, layout, 0, f.set);
        let groups = (self.active_slots.len() as u32).max(1).div_ceil(64);
        rhi.cmd_dispatch(fi, groups, 1, 1);

        // Order compute writes before the indirect-command read and the vertex read.
        rhi.cmd_buffer_barrier(
            fi,
            f.args_buf,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::DRAW_INDIRECT,
            vk::AccessFlags2::INDIRECT_COMMAND_READ,
        )?;
        rhi.cmd_buffer_barrier(
            fi,
            f.visible_buf,
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            vk::AccessFlags2::SHADER_WRITE,
            vk::PipelineStageFlags2::VERTEX_SHADER,
            vk::AccessFlags2::SHADER_READ,
        )?;
        Ok(())
    }

    /// Record the indirect, vertex-pulling draw. MUST be called INSIDE the
    /// rendering instance (between `begin_rendering` and `end_frame`).
    pub fn record_draw(&self, rhi: &Rhi, fi: u32) -> Result<(), RhiError> {
        let f = &self.frames[fi as usize];
        let layout = rhi.pipeline_layout(self.draw_pipeline)?;
        rhi.cmd_bind_pipeline(fi, vk::PipelineBindPoint::GRAPHICS, self.draw_pipeline)?;
        rhi.cmd_bind_descriptor_set(fi, vk::PipelineBindPoint::GRAPHICS, layout, 0, f.set);
        rhi.cmd_push_constants(
            fi,
            layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            bytemuck::bytes_of(&self.debug_mode),
        );
        // One indirect draw; vertex_count was written by the cull pass.
        rhi.cmd_draw_indirect(fi, f.args_buf, 0, 1, 16)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::{bake_patch, PatchParams};
    use enki_planet::noise::Noise3D;
    use enki_planet::simple_height::SimpleHeightField;

    fn asset() -> ClusterAsset {
        bake_patch(
            &PatchParams {
                face: 2,
                level: 4,
                ix: 5,
                iy: 6,
                resolution: 48,
                radius: 6_371_000.0,
                height_scale: 8_848.0,
            },
            &SimpleHeightField { noise: Noise3D::new(99) },
        )
    }

    #[test]
    fn residency_evictions_picks_undesired_keys() {
        let mut resident: HashMap<u32, u32> = HashMap::new();
        resident.insert(10, 0); // key 10 -> slot 0
        resident.insert(11, 1);
        resident.insert(12, 2);
        let mut desired: HashSet<u32> = HashSet::new();
        desired.insert(11); // keep 11; 10 and 12 should be evicted
        desired.insert(99); // not resident — ignored

        let mut ev = residency_evictions(&resident, &desired);
        ev.sort();
        assert_eq!(ev, vec![(10, 0), (12, 2)]);
    }

    #[test]
    fn page_pool_slot_is_reused_only_after_completion() {
        // Mirrors the renderer's slot lifecycle: alloc → retire → (gpu finishes) →
        // collect → realloc. The retired slot must NOT come back before completion.
        let mut pool = PagePool::new(2);
        let s0 = pool.alloc().unwrap();
        let s1 = pool.alloc().unwrap();
        assert!(pool.alloc().is_none(), "pool full");

        // Evict s0 this frame; reclaimable once the GPU completes frame 5.
        pool.retire(s0, 5);
        pool.collect(4); // GPU not there yet
        assert!(pool.alloc().is_none(), "evicted slot must wait for completion");
        pool.collect(5); // GPU done with frame 5
        assert_eq!(pool.alloc(), Some(s0), "slot reused after completion");
        let _ = s1;
    }

    #[test]
    fn shaders_parse_and_validate() {
        // Compile the WGSL with naga (the same frontend the RHI uses at runtime),
        // so shader errors surface in `cargo test` rather than only when the app runs.
        for (name, src) in [("cull", super::CULL_WGSL), ("draw", super::DRAW_WGSL)] {
            let module = naga::front::wgsl::parse_str(src)
                .unwrap_or_else(|e| panic!("{name} WGSL parse failed:\n{e:?}"));
            let mut validator = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            );
            validator
                .validate(&module)
                .unwrap_or_else(|e| panic!("{name} WGSL validation failed:\n{e:?}"));
        }
    }

    #[test]
    fn cluster_meta_is_monotonic_and_finite_bounds() {
        let a = asset();
        for c in &a.clusters {
            assert!(c.parent_error >= c.self_error, "non-monotonic error");
            assert!(c.bounds.radius > 0.0, "degenerate bounds radius");
            // Sphere centers must be finite (parent_error may be +inf for roots).
            assert!(c.bounds.center.is_finite());
            assert!(c.parent_bounds.center.is_finite());
        }
    }
}

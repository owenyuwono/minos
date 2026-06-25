//! VoxelView — the on-demand voxel terrain renderer.
//!
//! Sibling to [`crate::planet_view::PlanetView`], but each visible cube-sphere
//! quadtree leaf is meshed as a transvoxel block (`enki_voxel::mesh_leaf`). Drives
//! the SAME `LodTree` LOD selection; transvoxel transition-cell flags come from
//! neighbour LOD levels (re-meshed when a neighbour splits/merges), so coarse↔fine
//! seams stay crack-free.
//!
//! **Phase 3 — async:** meshing runs on a background worker thread. The render
//! thread only queues requests and uploads finished meshes; the previous mesh stays
//! drawn until its replacement arrives → no hitch on LOD change, edits, or caves.
//! A per-key request generation discards stale results (e.g. a mesh requested before
//! a dig edit). Cross-cube-face neighbours are treated as same-level (the inherent
//! cube-edge seam, an accepted artifact).

use std::collections::HashMap;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread::JoinHandle;

use glam::{DVec3, Mat4};
use lru::LruCache;
use std::num::NonZeroUsize;

use enki_planet::height::HeightField;
use enki_planet::lod::{LodCamera, LodConfig, LodTree};
use enki_planet::quadtree::ChunkKey;
use enki_planet::ChunkMeshArrays;
use enki_render::frame::FrameUniforms;
use enki_render::material::ChunkPush;
use enki_rhi::{
    vk, BindingDesc, BufferHandle, GraphicsPipelineDesc, PipelineHandle, Rhi, RhiError, StreamedMesh,
};
use bytemuck::{Pod, Zeroable};
use enki_voxel::{CaveField, CaveParams, Edit, TransitionSide, TransitionSides};
use crate::controls::terrain_grid::{dir_to_face_uv, SurfaceCollider};

/// Depth-only sun-shadow caster: project the leaf vertex by the light's
/// (view-proj × model). Reads only location 0 (position); the pipeline's other 3
/// vertex bindings are bound (terrain layout) but unused. Mirrors `character.rs`.
const VOXEL_DEPTH_WGSL: &str = "\
var<immediate> mvp: mat4x4<f32>;\n\
@vertex\n\
fn vs_depth(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {\n\
    return mvp * vec4<f32>(pos, 1.0);\n\
}\n";

/// Lit voxel-terrain shader that RECEIVES the 3-cascade sun CSM (own descriptor set:
/// frame UBO + 3 cascade depth maps + comparison sampler). See `terrain_csm.wgsl`.
const TERRAIN_CSM_WGSL: &str = include_str!("terrain_csm.wgsl");

/// Sun shadow cascades — MUST match `terrain_csm.wgsl`, `character.rs`, and main.rs.
const SHADOW_CASCADES: u32 = 3;

/// GPU mirror of `terrain_csm.wgsl`'s `Frame` UBO (std140-friendly). Identical layout
/// to `character.rs::CharFrame` — the voxel terrain is just another CSM receiver.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct TerrainCsmFrame {
    view_proj:     [[f32; 4]; 4],
    camera_pos:    [f32; 4], // world camera pos (w unused) — day/night ambient gate
    sun0_dir:      [f32; 4],
    sun0_color:    [f32; 4],
    sun1_dir:      [f32; 4],
    sun1_color:    [f32; 4],
    hemi_sky:      [f32; 4],
    hemi_ground:   [f32; 4],
    ambient:       [f32; 4],
    cascade_vp:    [[[f32; 4]; 4]; SHADOW_CASCADES as usize], // camera-relative world → light clip
    shadow_params: [f32; 4],                                 // [depth_bias, normal_bias, strength, enabled]
}

/// Immutable snapshot the worker meshes against (cheap to Arc-clone per request).
/// Carries `CaveParams` (Copy) not a `CaveField`, so the worker rebuilds the noise
/// locally — keeps `MeshCtx` trivially `Send` regardless of `Noise3D`'s bounds.
struct MeshCtx {
    hf: Arc<dyn HeightField>,
    cave_params: Option<CaveParams>,
    edits: Vec<Edit>,
    radius: f64,
    height_scale: f64,
    subdiv: usize,
    /// Data debug view baked into vertex color (0 none; 1 Height/2 Material/3 Wetness/
    /// 4 Volcano). See `mesh_leaf`'s `dbg_field`.
    dbg_field: u8,
}

struct MeshReq {
    key: ChunkKey,
    sides: TransitionSides,
    gen: u64,
    ctx: Arc<MeshCtx>,
}

struct MeshDone {
    key: ChunkKey,
    sides: TransitionSides,
    gen: u64,
    arrays: ChunkMeshArrays,
}

struct ResidentVoxel {
    mesh: StreamedMesh,
    origin: DVec3,
    sides: TransitionSides,
    /// CPU geometry kept for collision — the rendered triangles ARE the collider.
    /// `Arc` so the per-frame collider snapshot clones a handle, not the data.
    positions: Arc<[[f32; 3]]>,
    indices: Arc<[u32]>,
}

/// An immutable snapshot of the resident voxel leaves' CPU geometry — the SINGLE source
/// of truth for ground collision. The character feet + scattered flora raycast THESE
/// triangles (the exact ones the GPU draws), so they sit on the rendered surface. That
/// matters because while the LOD streams in, the drawn mesh is COARSER than the
/// full-detail analytic curve, so the analytic sinks BELOW it (= "everything submerged").
/// Rebuilt each frame (cheap: per-leaf geometry is `Arc`-shared) and handed out as an
/// `Arc`, so consumers query it without borrowing `VoxelView`.
pub struct VoxelCollider {
    leaves: Vec<ColliderLeaf>,
}

struct ColliderLeaf {
    key: ChunkKey, // (face, level, ix, iy) → the cube-face patch bounds for the find
    origin: DVec3,
    positions: Arc<[[f32; 3]]>, // leaf-local (relative to `origin`)
    indices: Arc<[u32]>,
}

/// Möller–Trumbore ray/triangle in f64. Ray = `ro` + t·`rd`; returns the hit distance
/// `t` (> 0) along `rd`, or `None` (miss / parallel / behind).
fn ray_tri(ro: DVec3, rd: DVec3, v0: DVec3, v1: DVec3, v2: DVec3) -> Option<f64> {
    let (e1, e2) = (v1 - v0, v2 - v0);
    let p = rd.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-9 {
        return None; // ray parallel to the triangle plane
    }
    let inv = 1.0 / det;
    let tv = ro - v0;
    let u = tv.dot(p) * inv;
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = tv.cross(e1);
    let v = rd.dot(q) * inv;
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let t = e2.dot(q) * inv;
    (t > 0.0).then_some(t)
}

impl SurfaceCollider for VoxelCollider {
    fn ground_radius(&self, dir: DVec3) -> Option<f64> {
        // Find the FINEST resident leaf whose cube-face patch contains `dir`
        // (bounds derived exactly as `enki_voxel::mesh_leaf`).
        let (face, cu, cv) = dir_to_face_uv(dir);
        let mut best: Option<&ColliderLeaf> = None;
        for leaf in &self.leaves {
            let (lf, level, ix, iy) = leaf.key;
            if lf as usize != face {
                continue;
            }
            let scale = 1.0 / (1u64 << level) as f64;
            let cu0 = ix as f64 * scale * 2.0 - 1.0;
            let cu1 = (ix as f64 + 1.0) * scale * 2.0 - 1.0;
            let cv0 = iy as f64 * scale * 2.0 - 1.0;
            let cv1 = (iy as f64 + 1.0) * scale * 2.0 - 1.0;
            let contains = cu >= cu0 && cu <= cu1 && cv >= cv0 && cv <= cv1;
            if contains && best.map_or(true, |b| level > b.key.1) {
                best = Some(leaf);
            }
        }
        let leaf = best?;

        // Raycast the planet-centre→dir ray against the leaf's ACTUAL triangles. The
        // top surface is the FARTHEST hit (caves carve solid below it, but you stand
        // on top). Verts are leaf-local → world by adding the leaf origin (f64).
        let world = |i: u32| -> DVec3 {
            let p = leaf.positions[i as usize];
            leaf.origin + DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64)
        };
        let mut hit: Option<f64> = None;
        for tri in leaf.indices.chunks_exact(3) {
            if let Some(t) = ray_tri(DVec3::ZERO, dir, world(tri[0]), world(tri[1]), world(tri[2])) {
                if hit.map_or(true, |h| t > h) {
                    hit = Some(t);
                }
            }
        }
        hit
    }
}

pub struct VoxelView {
    tree: LodTree,
    resident: HashMap<ChunkKey, ResidentVoxel>,
    /// Keys with an outstanding mesh request → its generation. Present means
    /// "wanted/in-flight"; a result is accepted only if its gen still matches.
    requested: HashMap<ChunkKey, u64>,
    gen: u64,
    /// Throttle counter for the diagnostic log.
    dbg: u64,
    /// CPU cache of meshed leaves (built, maybe not GPU-resident). Decouples
    /// "meshed" from "drawn" so LOD thrash + async latency never drop results — the
    /// fix for the resident=0 deadlock.
    cache: LruCache<ChunkKey, (ChunkMeshArrays, TransitionSides)>,
    /// Caves toggled/retuned → invalidate cache + resident, re-mesh next update.
    dirty_caves: bool,
    /// Data debug view selector (0 none; 1 Height/2 Material/3 Wetness/4 Volcano),
    /// baked into vertex color by the mesher.
    dbg_field: u8,
    /// Data-view field changed → re-mesh resident leaves IN PLACE with the new color
    /// (old mesh stays drawn until the new one arrives — never blanks, unlike caves).
    dirty_recolor: bool,
    req_txs: Vec<Sender<MeshReq>>,
    rr: usize,
    done_rx: Receiver<MeshDone>,
    _workers: Vec<JoinHandle<()>>,
    /// Count of leaves that panicked in the mesher. If this climbs while the LOD won't
    /// subdivide, fine chunks are crashing (a bug), not just slow.
    mesh_panics: Arc<std::sync::atomic::AtomicU64>,
    ctx: Arc<MeshCtx>,
    cave_params: CaveParams,
    caves_on: bool,
    edits: Vec<Edit>,
    pending_edits: Vec<Edit>,
    pipeline: PipelineHandle,
    /// Depth-only pipeline for casting resident leaves into the sun shadow map.
    shadow_pipeline: PipelineHandle,
    /// CSM-receiving lit pipeline + its own set0 (frame UBO + 3 cascade depth maps +
    /// comparison sampler). Used in `record` when a shadow map exists so the terrain
    /// RECEIVES sun shadows — the shared `pipeline` is set0-only and can't sample it.
    csm_pipeline: PipelineHandle,
    #[allow(dead_code)] // kept for ownership/clarity; the sets + pipeline hold the live ref
    csm_layout: vk::DescriptorSetLayout,
    csm_frame_ubo: Vec<BufferHandle>,
    csm_set: Vec<vk::DescriptorSet>,
    /// Collision snapshot of the resident leaves (the rendered triangles), rebuilt each
    /// `update`. Handed to the character + flora as the single grounding source.
    collider: Arc<VoxelCollider>,
}

impl VoxelView {
    pub fn new(
        rhi: &mut Rhi,
        hf: Arc<dyn HeightField>,
        lod: LodConfig,
        subdiv: usize,
        pipeline: PipelineHandle,
    ) -> Self {
        let radius = lod.radius;
        let height_scale = lod.height_scale;
        let tree = LodTree::new(lod, Arc::clone(&hf));
        let cave_params = CaveParams::default();

        // Depth-only caster pipeline (terrain vertex layout; reads only position).
        // Falls back to the lit pipeline handle if the depth pipeline fails to build.
        let shadow_pipeline = match rhi.create_shader_module(VOXEL_DEPTH_WGSL) {
            Ok(sm) => {
                let built = rhi.create_graphics_pipeline_depth(&GraphicsPipelineDesc {
                    shader: sm,
                    vs_entry: "vs_depth",
                    fs_entry: "vs_depth", // ignored (depth-only)
                    push_constant_size: 64, // mat4: light_view_proj * model
                    set0_layout: rhi.set0_layout(),
                    color_format: rhi.swapchain_format(), // ignored by depth-only
                    depth_format: vk::Format::D32_SFLOAT,
                    samples: vk::SampleCountFlags::TYPE_1,
                    blend: false,
                    fill: true,
                });
                rhi.destroy_shader_module(sm);
                built.unwrap_or(pipeline)
            }
            Err(e) => {
                log::error!("voxel depth-caster shader failed: {e}");
                pipeline
            }
        };

        // CSM-receiving lit pipeline: own set0 (frame UBO + 3 cascade depth textures +
        // comparison sampler), same 4×vec3 vertex layout + ChunkPush as the shared
        // terrain pipeline. Verbatim from `character.rs` — the terrain is just another
        // CSM receiver. Built unconditionally (cheap); record() uses it only when a
        // shadow map exists. expect(): a startup pipeline failure is a hard bug.
        let (csm_pipeline, csm_layout, csm_frame_ubo, csm_set) = {
            let vf = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
            let frag = vk::ShaderStageFlags::FRAGMENT;
            let mut csm_bindings = vec![BindingDesc {
                binding: 0,
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stages: vf,
            }];
            for c in 0..SHADOW_CASCADES {
                csm_bindings.push(BindingDesc {
                    binding: 1 + c,
                    ty: vk::DescriptorType::SAMPLED_IMAGE,
                    stages: frag,
                });
            }
            csm_bindings.push(BindingDesc {
                binding: 1 + SHADOW_CASCADES,
                ty: vk::DescriptorType::SAMPLER,
                stages: frag,
            });
            let layout = rhi
                .create_descriptor_set_layout(&csm_bindings)
                .expect("voxel CSM set layout");
            let shader = rhi
                .create_shader_module(TERRAIN_CSM_WGSL)
                .expect("terrain_csm.wgsl");
            let pl = rhi
                .create_graphics_pipeline(&GraphicsPipelineDesc {
                    shader,
                    vs_entry: "vs_main",
                    fs_entry: "fs_main",
                    push_constant_size: std::mem::size_of::<ChunkPush>() as u32,
                    set0_layout: layout,
                    color_format: rhi.swapchain_format(),
                    depth_format: vk::Format::D32_SFLOAT,
                    samples: rhi.msaa_samples(),
                    blend: false,
                    fill: true,
                })
                .expect("voxel CSM pipeline");
            rhi.destroy_shader_module(shader);
            let frames = rhi.frames_in_flight();
            let mut ubos = Vec::with_capacity(frames);
            let mut sets = Vec::with_capacity(frames);
            for _ in 0..frames {
                let ubo = rhi
                    .create_gpu_buffer(
                        std::mem::size_of::<TerrainCsmFrame>() as u64,
                        true,
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                    )
                    .expect("voxel CSM ubo");
                let set = rhi.allocate_descriptor_set(layout).expect("voxel CSM set");
                rhi.write_uniform_binding(set, 0, ubo).expect("voxel CSM ubo bind");
                ubos.push(ubo);
                sets.push(set);
            }
            (pl, layout, ubos, sets)
        };

        let ctx = Arc::new(MeshCtx {
            hf: Arc::clone(&hf),
            cave_params: Some(cave_params),
            edits: Vec::new(),
            radius,
            height_scale,
            subdiv,
            dbg_field: 0,
        });

        // A small pool of mesher threads (the real heightfield is expensive, so one
        // thread can't keep up with the visible cut). Round-robin requests across
        // per-worker channels — no shared lock contention on the hot path. (More cores
        // don't unlock finer LOD: the churn at finer target_tri_px is LOD instability,
        // not mesher throughput — see main.rs LodConfig.)
        let n_workers = 4;
        let (done_tx, done_rx) = mpsc::channel::<MeshDone>();
        let mesh_panics = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut req_txs = Vec::with_capacity(n_workers);
        let mut workers = Vec::with_capacity(n_workers);
        for w in 0..n_workers {
            let (req_tx, req_rx) = mpsc::channel::<MeshReq>();
            let done_tx = done_tx.clone();
            let panics = Arc::clone(&mesh_panics);
            let worker = std::thread::Builder::new()
                .name(format!("voxel-mesher-{w}"))
                .spawn(move || {
                    while let Ok(req) = req_rx.recv() {
                        // catch_unwind so a single bad leaf logs + is skipped instead of
                        // silently killing the worker (→ a blank planet forever).
                        let meshed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let c = &req.ctx;
                            let caves = c.cave_params.map(CaveField::new);
                            let (face, level, ix, iy) = req.key;
                            enki_voxel::mesh_leaf(
                                c.hf.as_ref(),
                                face,
                                level,
                                ix,
                                iy,
                                c.radius,
                                c.height_scale,
                                c.subdiv,
                                req.sides,
                                caves.as_ref(),
                                &c.edits,
                                c.dbg_field,
                            )
                        }));
                        let arrays = match meshed {
                            Ok(a) => a,
                            Err(_) => {
                                panics.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                log::error!("voxel mesh_leaf panicked for key {:?} — skipped", req.key);
                                continue;
                            }
                        };
                        if done_tx
                            .send(MeshDone { key: req.key, sides: req.sides, gen: req.gen, arrays })
                            .is_err()
                        {
                            break; // main side dropped
                        }
                    }
                })
                .expect("spawn voxel-mesher thread");
            req_txs.push(req_tx);
            workers.push(worker);
        }

        Self {
            tree,
            resident: HashMap::new(),
            requested: HashMap::new(),
            gen: 0,
            dbg: 0,
            cache: LruCache::new(NonZeroUsize::new(4096).unwrap()),
            dirty_caves: false,
            dbg_field: 0,
            dirty_recolor: false,
            req_txs,
            rr: 0,
            done_rx,
            _workers: workers,
            mesh_panics,
            ctx,
            cave_params,
            caves_on: true,
            edits: Vec::new(),
            pending_edits: Vec::new(),
            pipeline,
            shadow_pipeline,
            csm_pipeline,
            csm_layout,
            csm_frame_ubo,
            csm_set,
            collider: Arc::new(VoxelCollider { leaves: Vec::new() }),
        }
    }

    /// Record all resident leaves as depth-only casters into the (already-open) sun
    /// shadow pass for one cascade. `lvp` is that cascade's camera-relative light
    /// view-proj; per leaf we push `lvp · model` (model = origin−camera translation).
    pub fn record_shadow(
        &self,
        rhi: &mut Rhi,
        fi: u32,
        lvp: Mat4,
        camera_world_pos: DVec3,
    ) -> Result<(), RhiError> {
        rhi.bind_pipeline(fi, self.shadow_pipeline)?;
        for rv in self.resident.values() {
            let model = Mat4::from_translation((rv.origin - camera_world_pos).as_vec3());
            let mvp = (lvp * model).to_cols_array_2d();
            rhi.bind_vertex_buffers(fi, &[rv.mesh.pos, rv.mesh.nrm, rv.mesh.col, rv.mesh.plate])?;
            rhi.bind_index_buffer(fi, rv.mesh.idx)?;
            rhi.push_constants(fi, bytemuck::bytes_of(&mvp))?;
            rhi.draw_indexed(fi, rv.mesh.index_count);
        }
        Ok(())
    }

    fn rebuild_ctx(&mut self) {
        self.ctx = Arc::new(MeshCtx {
            hf: Arc::clone(&self.ctx.hf),
            cave_params: self.caves_on.then_some(self.cave_params),
            edits: self.edits.clone(),
            radius: self.ctx.radius,
            height_scale: self.ctx.height_scale,
            subdiv: self.ctx.subdiv,
            dbg_field: self.dbg_field,
        });
    }

    /// Data debug views (View → Planet: 7 Height/8 Material/9 Wetness/10 Volcano) bake
    /// the selected heightfield channel into vertex color; terrain*.wgsl shows it raw
    /// for those modes. Switching field remeshes (reuses the caves dirty path) — a brief
    /// hitch, vs. a 5th vertex stream through the shared 4×vec3 layout. Non-data modes
    /// → 0 = normal biome color (no remesh when toggling Lit/Normal/Plate/geometry).
    pub fn set_view_mode(&mut self, view_mode: u32) {
        let field = match view_mode {
            7 => 1u8,
            8 => 2,
            9 => 3,
            10 => 4,
            _ => 0,
        };
        if field != self.dbg_field {
            self.dbg_field = field;
            self.rebuild_ctx();
            self.dirty_recolor = true;
        }
    }

    pub fn caves_enabled(&self) -> bool {
        self.caves_on
    }

    pub fn set_caves_enabled(&mut self, on: bool) {
        if self.caves_on != on {
            self.caves_on = on;
            self.rebuild_ctx();
            self.dirty_caves = true;
        }
    }

    pub fn cave_strength(&self) -> f64 {
        self.cave_params.strength
    }

    pub fn set_cave_strength(&mut self, strength: f64) {
        if (self.cave_params.strength - strength).abs() > 1e-6 {
            self.cave_params.strength = strength;
            if self.caves_on {
                self.rebuild_ctx();
                self.dirty_caves = true;
            }
        }
    }

    /// Queue a sphere edit (applied next `update`). `dig` carves air, else fills solid.
    pub fn queue_edit(&mut self, center: DVec3, radius: f64, dig: bool) {
        self.pending_edits.push(Edit { center, radius, dig });
    }

    /// Send an async mesh request for `key` (bumps its generation → any older
    /// in-flight result for the key is discarded on arrival).
    fn request(&mut self, key: ChunkKey, sides: TransitionSides) {
        let gen = self.gen;
        self.gen = self.gen.wrapping_add(1);
        self.requested.insert(key, gen);
        let i = self.rr % self.req_txs.len();
        self.rr = self.rr.wrapping_add(1);
        let _ = self.req_txs[i].send(MeshReq { key, sides, gen, ctx: Arc::clone(&self.ctx) });
    }

    /// Transvoxel transition sides for `key`: flag each edge whose neighbour is
    /// COARSER (one level up) — where transvoxel stitches the 2× jump.
    fn compute_sides(&self, key: ChunkKey) -> TransitionSides {
        let (face, level, ix, iy) = key;
        let coarser = |dix: i64, diy: i64| -> bool {
            let max = 1i64 << level;
            let nix = ix as i64 + dix;
            let niy = iy as i64 + diy;
            if nix < 0 || niy < 0 || nix >= max || niy >= max {
                return false; // cube-face boundary — accepted seam
            }
            if self.resident.contains_key(&(face, level, nix as u32, niy as u32)) {
                return false; // same-level neighbour — crack-free by shared param
            }
            level > 0
                && self
                    .resident
                    .contains_key(&(face, level - 1, (nix as u32) / 2, (niy as u32) / 2))
        };
        let mut s = TransitionSide::none();
        if coarser(-1, 0) {
            s |= TransitionSide::LowX;
        }
        if coarser(1, 0) {
            s |= TransitionSide::HighX;
        }
        if coarser(0, -1) {
            s |= TransitionSide::LowY;
        }
        if coarser(0, 1) {
            s |= TransitionSide::HighY;
        }
        s
    }

    /// Upload OWNED mesh arrays to GPU resident (replacing + retiring any prior). On
    /// ring/budget exhaustion, fall back to caching so a later show retries.
    fn upload_arrays(
        &mut self,
        rhi: &mut Rhi,
        fi: u32,
        key: ChunkKey,
        arrays: ChunkMeshArrays,
        sides: TransitionSides,
    ) {
        let plate = arrays.plate_colors.as_deref();
        match rhi.streaming_upload(
            fi,
            &arrays.positions,
            &arrays.normals,
            &arrays.colors,
            plate,
            &arrays.indices,
        ) {
            Ok(Some(mesh)) => {
                let rv = ResidentVoxel {
                    mesh,
                    origin: arrays.origin,
                    sides,
                    positions: Arc::from(arrays.positions),
                    indices: Arc::from(arrays.indices),
                };
                if let Some(old) = self.resident.insert(key, rv) {
                    rhi.streaming_retire(old.mesh);
                }
            }
            Ok(None) => {
                self.cache.put(key, (arrays, sides)); // budget full → cache; retry on show
            }
            Err(e) => log::error!("voxel leaf upload failed: {e}"),
        }
    }

    /// Upload a cached leaf to GPU resident, keeping the cache entry (instant re-show).
    fn upload_from_cache(&mut self, rhi: &mut Rhi, fi: u32, key: ChunkKey) {
        let res = if let Some((arrays, sides)) = self.cache.peek(&key) {
            let plate = arrays.plate_colors.as_deref();
            let r = rhi.streaming_upload(
                fi,
                &arrays.positions,
                &arrays.normals,
                &arrays.colors,
                plate,
                &arrays.indices,
            );
            let positions: Arc<[[f32; 3]]> = Arc::from(arrays.positions.as_slice());
            let indices: Arc<[u32]> = Arc::from(arrays.indices.as_slice());
            Some((r, arrays.origin, *sides, positions, indices))
        } else {
            None
        };
        match res {
            Some((Ok(Some(mesh)), origin, sides, positions, indices)) => {
                let rv = ResidentVoxel { mesh, origin, sides, positions, indices };
                if let Some(old) = self.resident.insert(key, rv) {
                    rhi.streaming_retire(old.mesh);
                }
            }
            Some((Err(e), ..)) => log::error!("voxel leaf upload failed: {e}"),
            _ => {} // not cached, or Ok(None) budget full → retry next frame
        }
    }

    /// Per-frame: drain meshes → cache; LOD-select; upload the visible cut from cache
    /// (request misses); retire hidden; re-mesh on caves/edits/transition-side changes.
    pub fn update(&mut self, rhi: &mut Rhi, fi: u32, cam: &LodCamera) {
        // 1. Drain finished meshes. A re-mesh of a DRAWN leaf replaces it now; others
        //    go to the CPU cache (kept regardless of LOD visibility → never dropped).
        while let Ok(done) = self.done_rx.try_recv() {
            if self.requested.get(&done.key) == Some(&done.gen) {
                self.requested.remove(&done.key);
                if self.resident.contains_key(&done.key) {
                    self.upload_arrays(rhi, fi, done.key, done.arrays, done.sides);
                } else {
                    self.cache.put(done.key, (done.arrays, done.sides));
                }
            }
        }

        // 2. Caves toggled/retuned → drop the whole cache + resident; the LOD below
        //    re-requests everything fresh with the new density.
        if self.dirty_caves {
            self.dirty_caves = false;
            self.cache.clear();
            for (_, rv) in self.resident.drain() {
                rhi.streaming_retire(rv.mesh);
            }
            self.requested.clear();
        }

        // 2b. Data-view field changed → re-mesh resident leaves IN PLACE with the new
        //     color. Old meshes stay drawn until replaced (never blanks), the stale-
        //     colored cache is dropped, and we supersede any in-flight (older-field)
        //     request so the latest field wins. Bounded by the resident (drawn) set.
        if self.dirty_recolor {
            self.dirty_recolor = false;
            self.cache.clear();
            let keys: Vec<ChunkKey> = self.resident.keys().copied().collect();
            for key in keys {
                if let Some(sides) = self.resident.get(&key).map(|r| r.sides) {
                    self.requested.remove(&key);
                    self.request(key, sides);
                }
            }
        }

        // 3. Apply queued edits → rebuild ctx + invalidate overlapping leaves (cache +
        //    resident + in-flight) so they re-mesh with the edit.
        if !self.pending_edits.is_empty() {
            let new_edits: Vec<Edit> = self.pending_edits.drain(..).collect();
            self.edits.extend_from_slice(&new_edits);
            self.rebuild_ctx();
            let radius = self.ctx.radius;
            let near = |origin: DVec3, level: u8| {
                let bound = std::f64::consts::FRAC_PI_2 * radius / (1u64 << level) as f64;
                new_edits.iter().any(|e| origin.distance(e.center) < e.radius + bound)
            };
            let hit: Vec<ChunkKey> = self
                .resident
                .iter()
                .filter(|(k, rv)| near(rv.origin, k.1))
                .map(|(k, _)| *k)
                .collect();
            for key in hit {
                if let Some(rv) = self.resident.remove(&key) {
                    rhi.streaming_retire(rv.mesh);
                }
                self.cache.pop(&key);
                self.requested.remove(&key);
            }
        }

        // 4. LOD selection. "Available" = cached OR resident OR in-flight, so the tree
        //    settles instead of re-requesting forever; "visible" = resident (drawn).
        let sel = {
            let cache = &self.cache;
            let resident = &self.resident;
            // "Available" for the LOD = MESHED (cached) or drawn (resident) — NOT merely
            // requested/in-flight. Otherwise the tree hides a coarse parent the instant
            // its children are *requested* (before the slow worker meshes them), blanking
            // the area on zoom-in. Dedup against re-requesting in-flight leaves is still
            // handled explicitly below via `requested`.
            self.tree.update(
                cam,
                &|k| cache.contains(&k) || resident.contains_key(&k),
                &|k| resident.contains_key(&k),
            )
        };

        // 5. Hide → retire the GPU mesh but KEEP the cache (instant re-show) and any
        //    in-flight request (it still completes → cache; nothing dropped).
        for key in &sel.hide {
            if let Some(rv) = self.resident.remove(key) {
                rhi.streaming_retire(rv.mesh);
            }
        }
        // 6. Cancel → drop the in-flight request (genuinely no longer wanted).
        for key in &sel.cancels {
            self.requested.remove(key);
        }
        // 7. Builds → request a mesh if we have it nowhere yet.
        for (key, _prio) in &sel.builds {
            if !self.cache.contains(key)
                && !self.resident.contains_key(key)
                && !self.requested.contains_key(key)
            {
                let sides = self.compute_sides(*key);
                self.request(*key, sides);
            }
        }
        // 8. Show → upload from cache (or request the miss).
        for key in &sel.show {
            if self.resident.contains_key(key) {
                continue;
            }
            if self.cache.contains(key) {
                self.upload_from_cache(rhi, fi, *key);
            } else if !self.requested.contains_key(key) {
                let sides = self.compute_sides(*key);
                self.request(*key, sides);
            }
        }

        // 9. Re-mesh resident leaves whose transition sides changed (neighbour LOD) so
        //    coarse↔fine seams stay crack-free; the new mesh replaces it on arrival.
        let keys: Vec<ChunkKey> = self.resident.keys().copied().collect();
        for key in keys {
            if self.requested.contains_key(&key) {
                continue;
            }
            let new_sides = self.compute_sides(key);
            if self.resident.get(&key).map(|r| r.sides) != Some(new_sides) {
                self.cache.pop(&key);
                self.request(key, new_sides);
            }
        }

        // Rebuild the collision snapshot from the now-final resident set — the SINGLE
        // source of truth for grounding. Cheap: per-leaf geometry is Arc-shared, so this
        // only clones handles + the small per-leaf metadata.
        self.collider = Arc::new(VoxelCollider {
            leaves: self
                .resident
                .iter()
                .map(|(k, rv)| ColliderLeaf {
                    key: *k,
                    origin: rv.origin,
                    positions: Arc::clone(&rv.positions),
                    indices: Arc::clone(&rv.indices),
                })
                .collect(),
        });

        self.dbg = self.dbg.wrapping_add(1);
        // LOD telemetry: altitude + resident level histogram make a zoom test a glance —
        // watch `levels` climb (orbit ~{2..4} → surface ~{11,12}) as you zoom in. Always
        // on, every ~1.5 s. If `levels` never climbs on zoom-in, the LOD is stalled, not
        // the mesher.
        if self.dbg % 90 == 1 {
            use std::collections::BTreeMap;
            let mut levels: BTreeMap<u8, u32> = BTreeMap::new();
            for k in self.resident.keys() {
                *levels.entry(k.1).or_default() += 1;
            }
            let alt = cam.local_pos.length() - self.ctx.radius;
            let panics = self.mesh_panics.load(std::sync::atomic::Ordering::Relaxed);
            log::info!(
                "[voxel] alt={:.0}m resident={} cache={} req={} panics={} | builds={} show={} hide={} levels={:?}",
                alt, self.resident.len(), self.cache.len(), self.requested.len(), panics,
                sel.builds.len(), sel.show.len(), sel.hide.len(), levels,
            );
        }
    }

    /// The collision snapshot as the generic `SurfaceCollider` trait object — for the
    /// character controller, which is feature-agnostic (doesn't know `VoxelCollider`).
    pub fn collider_dyn(&self) -> Arc<dyn SurfaceCollider> {
        self.collider.clone()
    }

    /// Raycast the rendered surface along `dir` (`None` if no resident leaf covers it →
    /// caller falls back to the analytic grounding). For per-frame flora re-grounding.
    pub fn ground_radius(&self, dir: DVec3) -> Option<f64> {
        self.collider.ground_radius(dir)
    }

    pub fn record(
        &self,
        rhi: &mut Rhi,
        fi: u32,
        fu: &FrameUniforms,
        camera_world_pos: DVec3,
        material_mode: u32,
        cascade_mvps: [Mat4; 3],
        shadow_params: [f32; 4],
    ) -> Result<(), RhiError> {
        // TEMP diagnostic: log once whenever the view mode changes, so we can see
        // exactly what reaches the draw (mode value, which pipeline, leaves drawn).
        {
            use std::sync::atomic::{AtomicU32, Ordering};
            static LAST: AtomicU32 = AtomicU32::new(u32::MAX);
            if LAST.swap(material_mode, Ordering::Relaxed) != material_mode {
                let lvls: std::collections::BTreeSet<u8> =
                    self.resident.keys().map(|k| k.1).collect();
                log::info!(
                    "[voxel-dbg] record mode={} csm_path={} resident={} lod_levels={:?}",
                    material_mode, rhi.has_shadow_map(), self.resident.len(), lvls
                );
            }
        }
        // CSM-receiving path (the planet default): bind our own set0 (frame UBO with
        // the cascade matrices + 3 cascade depth maps + comparison sampler) so the
        // terrain RECEIVES sun shadows. Falls back to the shared, shadow-blind
        // pipeline only when no shadow map exists.
        if rhi.has_shadow_map() {
            let fidx = fi as usize;
            // NOT fu.camera_pos — that is (0,0,0) in camera-relative space. The shader
            // reconstructs the absolute radial direction (planet centre at the world
            // origin) as world_pos + camera_pos, so it needs the camera's WORLD position.
            let cam = camera_world_pos.as_vec3();
            let cf = TerrainCsmFrame {
                view_proj: fu.view_proj,
                camera_pos: [cam.x, cam.y, cam.z, 1.0],
                sun0_dir: fu.sun0_dir,
                sun0_color: fu.sun0_color,
                sun1_dir: fu.sun1_dir,
                sun1_color: fu.sun1_color,
                hemi_sky: fu.hemi_sky,
                hemi_ground: fu.hemi_ground,
                ambient: fu.ambient,
                cascade_vp: cascade_mvps.map(|m| m.to_cols_array_2d()),
                shadow_params,
            };
            rhi.write_storage_bytes(self.csm_frame_ubo[fidx], bytemuck::bytes_of(&cf))?;
            let set = self.csm_set[fidx];
            let read = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            for c in 0..SHADOW_CASCADES {
                rhi.write_sampled_image_binding(set, 1 + c, rhi.shadow_map_view(fi, c), read);
            }
            rhi.write_sampler_binding(set, 1 + SHADOW_CASCADES, rhi.shadow_map_sampler());
            let layout = rhi.pipeline_layout(self.csm_pipeline)?;
            rhi.set_viewport_scissor_full(fi);
            rhi.cmd_bind_pipeline(fi, vk::PipelineBindPoint::GRAPHICS, self.csm_pipeline)?;
            rhi.cmd_bind_descriptor_set(fi, vk::PipelineBindPoint::GRAPHICS, layout, 0, set);
            let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
            for (key, rv) in self.resident.iter() {
                let (face, level, ix, iy) = *key;
                // Stable per-leaf colour seed (modes 3/4) + LOD level (mode 5). Packed
                // distinct per leaf; the shader's id_color hashes it into a hue.
                let dbg_id = (face as u32) | ((level as u32) << 3) | ((ix as u32) << 8) | ((iy as u32) << 20);
                let push = ChunkPush::camera_relative(
                    rv.origin, camera_world_pos, Mat4::IDENTITY, material_mode,
                ).with_dbg(dbg_id, level as u32);
                rhi.bind_vertex_buffers(fi, &[rv.mesh.pos, rv.mesh.nrm, rv.mesh.col, rv.mesh.plate])?;
                rhi.bind_index_buffer(fi, rv.mesh.idx)?;
                rhi.cmd_push_constants(fi, layout, stages, bytemuck::bytes_of(&push));
                rhi.draw_indexed(fi, rv.mesh.index_count);
            }
        } else {
            rhi.bind_pipeline(fi, self.pipeline)?;
            rhi.update_frame_uniforms(fi, bytemuck::bytes_of(fu))?;
            for (key, rv) in self.resident.iter() {
                let (face, level, ix, iy) = *key;
                let dbg_id = (face as u32) | ((level as u32) << 3) | ((ix as u32) << 8) | ((iy as u32) << 20);
                let push = ChunkPush::camera_relative(
                    rv.origin, camera_world_pos, Mat4::IDENTITY, material_mode,
                ).with_dbg(dbg_id, level as u32);
                rhi.bind_vertex_buffers(fi, &[rv.mesh.pos, rv.mesh.nrm, rv.mesh.col, rv.mesh.plate])?;
                rhi.bind_index_buffer(fi, rv.mesh.idx)?;
                rhi.push_constants(fi, bytemuck::bytes_of(&push))?;
                rhi.draw_indexed(fi, rv.mesh.index_count);
            }
        }
        Ok(())
    }

    /// Total triangles across all resident leaves (HUD stat).
    pub fn triangle_count(&self) -> u32 {
        self.resident.values().map(|r| r.mesh.index_count).sum::<u32>() / 3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // terrain_csm.wgsl is only validated at runtime (create_shader_module) — naga
    // errors otherwise surface as a black terrain, not a compile error. Gate it here
    // like every other app shader (a black screen still passes `cargo check`).
    #[test]
    fn terrain_csm_wgsl_validates() {
        let module = naga::front::wgsl::parse_str(TERRAIN_CSM_WGSL)
            .unwrap_or_else(|e| panic!("terrain_csm WGSL failed to parse: {e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("terrain_csm WGSL failed to validate: {e:?}"));
    }
}

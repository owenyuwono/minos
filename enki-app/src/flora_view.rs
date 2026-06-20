// flora_view.rs — enki-app's GPU glue for enki-flora (behind `--features flora`).
//
// ponytail: smallest thing that renders ONE lit tree. This module is the whole
// render path: resolve a single genome, build the branch tube mesh + expand the
// foliage cards into a leaf quad mesh on the CPU, upload both once, and draw
// them each frame. It reuses enki's EXISTING non-pulling 4×vec3 vertex layout,
// reversed-Z depth, FrameUniforms lights, and the camera-relative f64→f32 push
// scheme — NO dryad IBL/HDRI/EffectComposer/post, NO wind, NO instancing, NO
// storage buffers (see flora.wgsl for the matching shader-side simplifications).
//
// The whole thing lives in enki-app (not enki-flora) so enki-flora stays
// dep-light (glam only) and the deletable-crate contract holds: deleting flora
// is dropping this file + the `#[cfg(feature="flora")]` wiring in main.rs.

use enki_flora::color::pigment_to_color;
use enki_flora::genome::{random_genome, resolve, Env, Genome, Resolved};
use enki_flora::mesh::{build_branch_mesh, WindBone};
use enki_flora::wind_solver::WindSolver;
use enki_rhi::{BufferHandle, Rhi, RhiError};
use glam::{DVec3, Mat3, Mat4, Quat, Vec3};

use crate::flora_render::{BranchDepthPush, FloraPipeline, FloraRenderer, LeafDepthPush};

/// Debug render mode for the viewer. `Lit/Unlit/Normals/Ao` select an FS arm via
/// the `debug_mode` push lane (0/1/2/3); `Wireframe` swaps to the `fill:false`
/// line pipelines instead (its FS output is irrelevant, so it carries mode 0).
///
/// `Triangle/Cluster/Lod` are the enki-nanite virtualized-geometry debug views:
/// they are only meaningful when the branch mesh is drawn through the Nanite
/// cull+draw path (`--features flora,nanite`). They select nanite_draw.wgsl's
/// per-triangle / per-cluster / per-LOD id-color modes (3/4/5). Without nanite
/// they fall through `is_nanite()==false` and the viewer renders the plain branch
/// (treated like Lit) — see `nanite_debug_mode`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RenderMode {
    #[default]
    Lit,
    Unlit,
    Wireframe,
    Normals,
    Ao,
    /// Nanite per-triangle id color (nanite_draw mode 3).
    Triangle,
    /// Nanite per-cluster id color (nanite_draw mode 4).
    Cluster,
    /// Nanite per-LOD id color (nanite_draw mode 5).
    Lod,
}

impl RenderMode {
    /// The FS `debug_mode` lane value (0=lit, 1=unlit, 2=normals, 3=ao).
    /// Wireframe falls through the lit FS (drawn by the line pipeline), so 0.
    /// The Nanite-only modes have no plain-FS arm; they render Lit (0) when the
    /// plain branch path is used (nanite off), so the tree still shows.
    fn debug_mode(self) -> f32 {
        match self {
            RenderMode::Lit | RenderMode::Wireframe => 0.0,
            RenderMode::Unlit => 1.0,
            RenderMode::Normals => 2.0,
            RenderMode::Ao => 3.0,
            // No plain-branch FS arm — fall back to Lit's lane.
            RenderMode::Triangle | RenderMode::Cluster | RenderMode::Lod => 0.0,
        }
    }
    fn is_wireframe(self) -> bool {
        matches!(self, RenderMode::Wireframe)
    }

    /// True ONLY for the enki-nanite virtualized-geometry debug views
    /// (Triangle/Cluster/Lod) — the modes whose whole purpose is to show the
    /// branch meshlets in per-triangle/cluster/LOD debug colors via Nanite's
    /// cull+draw. Lit/Unlit/Wireframe/Normals/Ao all use flora's OWN full render
    /// path (textured PBR bark + IBL + PCF shadows + green leaf cards + bloom);
    /// routing Lit through Nanite's generic draw flattens the bark and miscolors
    /// the leaves, so Lit deliberately stays OFF the Nanite path.
    pub fn uses_nanite(self) -> bool {
        matches!(
            self,
            RenderMode::Triangle | RenderMode::Cluster | RenderMode::Lod
        )
    }

    /// The nanite_draw.wgsl push-constant color mode for this view (3=triangle,
    /// 4=cluster, 5=LOD). Only meaningful when `uses_nanite()`; other modes
    /// return 0.
    pub fn nanite_debug_mode(self) -> u32 {
        match self {
            RenderMode::Triangle => 3,
            RenderMode::Cluster => 4,
            RenderMode::Lod => 5,
            // Non-Nanite modes → unused (caller gates on `uses_nanite()`).
            _ => 0,
        }
    }
}

/// Leaf rendering mode: how the baked leaf sprite maps onto the foliage SoA.
///
/// * `Cluster` (default) — the existing path: ONE quad per foliage cluster
///   anchor, textured with the baked 5-8 leaf CLUSTER sprite
///   (`bake_leaf_cluster`).
/// * `Single` — each cluster anchor is fanned into `LEAVES_PER_CLUMP` individual
///   single-leaf cards (a port of dryad's `expandClumpsToLeaves`), each textured
///   with the SINGLE-leaf sprite (`bake_leaf_single`). 1 card = 1 leaf.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum LeafMode {
    #[default]
    Cluster,
    Single,
}

impl LeafMode {
    /// Parse `FLORA_LEAFMODE` (screenshot env). `single` ⇒ Single, anything else
    /// (incl. unset/`cluster`) ⇒ Cluster.
    pub fn from_env() -> Self {
        match std::env::var("FLORA_LEAFMODE").ok().as_deref() {
            Some(v) if v.eq_ignore_ascii_case("single") => LeafMode::Single,
            _ => LeafMode::Cluster,
        }
    }
}

/// `BranchPush` mirror (flora.wgsl) — 128 bytes: model + wind + 3 bark vec4s.
///
/// Layout (16-byte aligned, == 128-byte push limit exactly):
///   model     64  mat4x4  camera-relative model
///   wind      16  vec4    (time, strength, dir_x, dir_z) — global-field wind.
///                         Repurposes the slot the never-sampled `wood_tint`
///                         color occupied, so the push stays exactly 128 bytes.
///   bark0     16  vec4    (bark_hue, bark_lightness, bark_relief, bark_lenticels)
///   bark1     16  vec4    (bark_scale, bark_orient, bark_plates, bark_shed)
///   bark2     16  vec4    (bark_under_hue, woodiness, 0, 0)
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BranchPush {
    model: [[f32; 4]; 4],
    wind: [f32; 4],
    bark0: [f32; 4],
    bark1: [f32; 4],
    bark2: [f32; 4],
}

/// `LeafPush` mirror (flora.wgsl) — 128 bytes: model + pigment + leaf genes + wind.
///
/// Layout (16-byte aligned, == 128-byte push limit exactly):
///   model       64  mat4x4
///   pigment     16  vec4   xyz = per-tree base leaf color (pigment ramp); w unused
///   leaf_params 16  vec4   (leaf_tip, leaf_width, leaf_serration, leaf_lobing)
///   leaf_params2 16 vec4   (leaf_skew, leaf_length, 0, 0)
///   wind        16  vec4   (time, strength, dir_x, dir_z) — 112→128B, exact fit
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct LeafPush {
    model: [[f32; 4]; 4],
    pigment: [f32; 4],
    leaf_params: [f32; 4],
    leaf_params2: [f32; 4],
    wind: [f32; 4],
}

/// One uploaded indexed mesh in the 4×vec3 layout (positions/normals/uv3/attr3).
struct GpuMesh {
    pos: BufferHandle,
    nrm: BufferHandle,
    uv: BufferHandle,    // vec3: branch=(angle, arc_len, 0); leaf=(u, v, 0)
    attr: BufferHandle,  // vec3: branch=(ao, 0, 0);          leaf=(age, exposure, 0)
    idx: BufferHandle,
    index_count: u32,
}

// ponytail: no explicit teardown. Like PlanetView, FloraView's BufferHandles are
// reclaimed by the RHI's BufferStore::destroy_all at device shutdown, so no Drop
// impl / per-handle destroy is needed for the one-tree lifetime.

/// The flora renderer: the branch/leaf meshes + per-tree appearance genes for
/// one tree, plus that tree's f64 world origin (for the camera-relative push each
/// frame). The PIPELINES are NOT owned here — they live in `FloraRenderer` (the
/// flora-owned sub-renderer); `record` takes `&FloraRenderer` and selects which
/// of its pipelines to bind. This keeps FloraView a pure mesh+gene holder.
pub struct FloraView {
    mode: RenderMode,
    leaf_mode: LeafMode,
    branch: GpuMesh,
    leaf: Option<GpuMesh>, // None when the tree has zero leaf cards
    origin: DVec3,
    bark0: [f32; 4],          // (bark_hue, bark_lightness, bark_relief, bark_lenticels)
    bark1: [f32; 4],          // (bark_scale, bark_orient, bark_plates, bark_shed)
    bark2: [f32; 4],          // (bark_under_hue, woodiness, 0, 0)
    leaf_pigment: [f32; 4],   // per-tree base leaf color (pigment ramp)
    leaf_params: [f32; 4],    // (leaf_tip, leaf_width, leaf_serration, leaf_lobing)
    leaf_params2: [f32; 4],   // (leaf_skew, leaf_length, 0, 0)
    leaf_genes: enki_flora::leaf_texture::LeafGenes, // CPU leaf-texture bake inputs
    bounds: ([f32; 3], [f32; 3]), // branch-mesh AABB (min, max), local space
    leaf_count: u32,          // resolved foliage card count (stats)
    bone_count: u32,          // skeleton bone count (stats)
    // Hierarchical wind solver (one wind bone per branch chain). Owns the
    // bones_wind hierarchy + per-bone phases; solved + uploaded each frame.
    wind_solver: WindSolver,
    // Reusable solver output scratch (count*16 f32, column-major mat4s) to avoid
    // a per-frame allocation in the hot path.
    wind_scratch: Vec<f32>,
}

impl FloraView {
    /// Build the one tree: resolve `seed`, mesh it, upload buffers. `origin` is
    /// the tree's f64 world position (surface point). Pipelines live in the
    /// flora-owned `FloraRenderer`, not here.
    pub fn new(rhi: &mut Rhi, seed: u32, origin: DVec3) -> Result<Self, RhiError> {
        Self::new_with_mode(rhi, seed, origin, LeafMode::default())
    }

    /// As [`new`], but with an explicit leaf mode (Cluster/Single).
    pub fn new_with_mode(
        rhi: &mut Rhi,
        seed: u32,
        origin: DVec3,
        leaf_mode: LeafMode,
    ) -> Result<Self, RhiError> {
        let env = Env::default();
        let genome = random_genome(&env, seed);
        Self::from_genome_with_mode(rhi, &genome, &env, origin, leaf_mode)
    }

    /// Build the tree from an EDITED genome + env (the egui-driven CAS path).
    ///
    /// Resolves `genome` against `env` then meshes/uploads it. Determinism is
    /// preserved: `resolve` is the same pure function `new` calls, so an edited
    /// genome that equals a `random_genome(env, seed)` output produces a
    /// bit-identical tree. `new` now delegates here after its seed→genome draw.
    pub fn from_genome(
        rhi: &mut Rhi,
        genome: &Genome,
        env: &Env,
        origin: DVec3,
    ) -> Result<Self, RhiError> {
        Self::from_genome_with_mode(rhi, genome, env, origin, LeafMode::default())
    }

    /// As [`from_genome`], with an explicit leaf mode (Cluster/Single).
    pub fn from_genome_with_mode(
        rhi: &mut Rhi,
        genome: &Genome,
        env: &Env,
        origin: DVec3,
        leaf_mode: LeafMode,
    ) -> Result<Self, RhiError> {
        let resolved = resolve(genome, env);
        // ── Single-leaf fan GATE (dryad expandClumpsToLeaves, foliage.js:925-926).
        //    Only broadleaf canopies fan; compound fronds / needles / spiny / very
        //    narrow leaves pass through as one whole-unit card per anchor (K=1).
        //    Computed HERE because `rosette`/`spininess` live on the genome, not on
        //    `Resolved` — they're folded into a `(frondyness, spininess)` pair.
        let narrow_factor = ((0.30 - genome.leaf_width) / 0.30).clamp(0.0, 1.0);
        let frondyness = genome
            .leaf_division
            .max(genome.rosette)
            .max(narrow_factor);
        let spininess = genome.spininess;
        // dryad keys the leaf-cluster RNG on the tree's leaf seed; we use the
        // genome's structural seed (the lone Seed gene) so reseeding rebuilds a
        // bit-stable cluster texture in lockstep with the rest of the tree.
        Self::from_resolved(
            rhi,
            resolved,
            genome.structural_seed,
            origin,
            leaf_mode,
            frondyness,
            spininess,
        )
    }

    /// Mesh + upload from an already-resolved tree. Genome-source-agnostic: both
    /// `new` (seed) and `from_genome` (edited) funnel here. `tex_seed` keys the
    /// CPU leaf-texture bake (genome.structural_seed). `leaf_mode` selects the
    /// Cluster (one card/cluster) vs Single (fan into individual leaves) path;
    /// `frondyness`/`spininess` are the single-fan gate (computed in `from_genome`).
    #[allow(clippy::too_many_arguments)]
    fn from_resolved(
        rhi: &mut Rhi,
        resolved: Resolved,
        tex_seed: u32,
        origin: DVec3,
        leaf_mode: LeafMode,
        frondyness: f64,
        spininess: f64,
    ) -> Result<Self, RhiError> {
        // The bark albedo is FULLY procedural in-shader (ridged-FBM furrows +
        // voronoi plates from the bark genes below). `pig` (pigment-derived
        // color) still feeds the leaf base; `w` (woodiness) feeds bark2.
        let pig = pigment_to_color(resolved.pigment);
        let w = resolved.woodiness as f32;

        // ── Per-tree bark genes → push-constant vec4s (dryad bark uniforms). ──
        // Reseeding the genome re-rolls these, so the procedural bark changes.
        let bark0 = [
            resolved.bark_hue as f32,
            resolved.bark_lightness as f32,
            resolved.bark_relief as f32,
            resolved.bark_lenticels as f32,
        ];
        let bark1 = [
            resolved.bark_scale as f32,
            resolved.bark_orient as f32,
            resolved.bark_plates as f32,
            resolved.bark_shed as f32,
        ];
        let bark2 = [resolved.bark_under_hue as f32, w, 0.0, 0.0];

        // ── Branch mesh → 4×vec3 streams. ──
        let bm = build_branch_mesh(&resolved.graph);
        // uv (2V) widened to vec3 with the per-vertex tube RADIUS packed into .z
        // (the bark shader's `barkFeatureScale`); ao (1V) widened to vec3 (.yz=0).
        let mut uv3 = Vec::with_capacity(bm.vertex_count as usize * 3);
        for (c, &r) in bm.uvs.chunks_exact(2).zip(bm.radii.iter()) {
            uv3.extend_from_slice(&[c[0], c[1], r]);
        }
        // attr = (ao, wind boneIndex, boneFraction) — the branch VS skins by its
        // bone matrix using .y/.z (was (ao,0,0); the wind bake fills the pad lanes).
        let mut attr3 = Vec::with_capacity(bm.vertex_count as usize * 3);
        for ((&ao, &bidx), &bfrac) in bm.ao.iter().zip(&bm.bone_index).zip(&bm.bone_fraction) {
            attr3.extend_from_slice(&[ao, bidx, bfrac]);
        }
        let branch = GpuMesh {
            pos: rhi.create_vertex_buffer(bytemuck::cast_slice(&bm.positions))?,
            nrm: rhi.create_vertex_buffer(bytemuck::cast_slice(&bm.normals))?,
            uv: rhi.create_vertex_buffer(bytemuck::cast_slice(&uv3))?,
            attr: rhi.create_vertex_buffer(bytemuck::cast_slice(&attr3))?,
            idx: rhi.create_index_buffer(&bm.indices)?,
            index_count: bm.indices.len() as u32,
        };
        let bounds = (bm.bounds.min, bm.bounds.max);
        let bone_count = resolved.graph.bones.len() as u32;

        // ── Hierarchical wind solver (one wind bone per branch chain). Built once
        //    per tree from the mesher's bones_wind hierarchy. Solved + uploaded
        //    each frame; strength==0 → identities → exact static rest pose. ──
        let wind_solver = WindSolver::new(&bm.bones_wind);
        let wind_scratch = vec![0.0_f32; wind_solver.bone_count() * 16];

        // ── Leaf cards → CPU-expanded quad mesh in the same 4×vec3 layout. Each
        //    leaf's nearest wind bone (by pivot distance) is baked into uv.z so
        //    the leaf bone-follows its twig's hierarchical sway. In `Single` mode
        //    each cluster anchor is first fanned into individual single-leaf
        //    cards (expand_clumps_to_leaves); in `Cluster` mode the SoA passes
        //    through unchanged (one card per cluster). `leaf_count` reports the
        //    POST-EXPANSION card count (so the GUI stat reflects Single's 6×). ──
        let foliage_owned;
        let foliage_ref = match leaf_mode {
            LeafMode::Cluster => &resolved.foliage,
            LeafMode::Single => {
                foliage_owned = expand_clumps_to_leaves(
                    &resolved.foliage,
                    tex_seed,
                    frondyness,
                    spininess,
                );
                &foliage_owned
            }
        };
        let leaf_count = foliage_ref.count as u32;
        let leaf = build_leaf_mesh(rhi, foliage_ref, &bm.bones_wind)?;

        // ── Per-tree leaf appearance: pigment base color + shape genes. ──
        // Leaf pigment uses the same hue-wheel ramp as wood, but undiluted by
        // woodiness — leaves keep the full pigment hue as their base albedo.
        let leaf_pigment = [pig[0] as f32, pig[1] as f32, pig[2] as f32, 1.0];
        let leaf_params = [
            resolved.leaf_tip as f32,
            resolved.leaf_width as f32,
            resolved.leaf_serration as f32,
            resolved.leaf_lobing as f32,
        ];
        let leaf_params2 = [
            resolved.leaf_skew as f32,
            resolved.leaf_length as f32,
            0.0,
            0.0,
        ];

        // ── CPU leaf-cluster texture genes (dryad makeLeafClusterTexture inputs).
        //    The FloraRenderer bakes + uploads the color/normal sprites from this
        //    on tree build (see FloraRenderer::update_leaf_texture). Pigment is the
        //    raw [0,1] scalar (the bake derives the base HSL via pigment_to_color),
        //    NOT the resolved RGB. ──
        let leaf_genes = enki_flora::leaf_texture::LeafGenes {
            pigment: resolved.pigment,
            leaf_width: resolved.leaf_width,
            leaf_length: resolved.leaf_length,
            leaf_tip: resolved.leaf_tip,
            leaf_serration: resolved.leaf_serration,
            leaf_lobing: resolved.leaf_lobing,
            leaf_skew: resolved.leaf_skew,
            seed: tex_seed,
        };

        // ── Pipelines are NOT created here. They live in the flora-OWNED
        //    sub-renderer (`FloraRenderer`), built once at viewer startup. This
        //    module only meshes + uploads + holds the per-tree appearance genes;
        //    `record` borrows the renderer's pipelines. ──

        Ok(Self {
            mode: RenderMode::default(),
            leaf_mode,
            branch,
            leaf,
            origin,
            bark0,
            bark1,
            bark2,
            leaf_pigment,
            leaf_params,
            leaf_params2,
            leaf_genes,
            bounds,
            leaf_count,
            bone_count,
            wind_solver,
            wind_scratch,
        })
    }

    /// The CPU leaf-texture genes for this tree (dryad makeLeafClusterTexture
    /// inputs). The viewer feeds this to `FloraRenderer::update_leaf_texture`
    /// after a (re)build so the per-genome leaf color/normal sprites are re-baked
    /// + re-uploaded.
    pub fn leaf_genes(&self) -> enki_flora::leaf_texture::LeafGenes {
        self.leaf_genes
    }

    /// This tree's leaf mode (Cluster vs Single) — the viewer reads it to select
    /// the cluster vs single sprite bake in `update_leaf_texture`.
    pub fn leaf_mode(&self) -> LeafMode {
        self.leaf_mode
    }

    /// Solve this frame's hierarchical-wind bone matrices into the internal
    /// scratch and return them as a flat column-major `[f32]` (`len ==
    /// bone_count*16`) ready for upload to the set1/binding 6 storage buffer.
    /// `wind = [time, strength, dir_x, dir_z]`; `strength == 0` ⇒ all identities
    /// ⇒ the shader reproduces the exact static rest pose. Call once per frame
    /// (before the shadow + scene passes) and pass the result to
    /// `FloraRenderer::set_bone_matrices`.
    pub fn solve_wind(&mut self, wind: [f32; 4]) -> &[f32] {
        self.wind_solver.solve_into(
            &mut self.wind_scratch,
            wind[0] as f64,
            wind[1] as f64,
            wind[2] as f64,
            wind[3] as f64,
        );
        &self.wind_scratch
    }

    /// Number of wind bones (== solver output mat4 count).
    pub fn wind_bone_count(&self) -> usize {
        self.wind_solver.bone_count()
    }

    /// Select the debug render mode. Picks the wireframe pipelines for
    /// `Wireframe`; otherwise writes the `debug_mode` push lane so the FS selects
    /// lit/unlit/normals/ao. Persists across `record` calls (the tree survives a
    /// mode switch — no rebuild needed). Survives a tree rebuild only if re-set,
    /// so the viewer re-applies it after `from_genome` (cheap, idempotent).
    pub fn set_render_mode(&mut self, mode: RenderMode) {
        self.mode = mode;
    }

    /// Triangle count across branch + leaf meshes (sum of index counts / 3).
    pub fn triangle_count(&self) -> u32 {
        let leaf = self.leaf.as_ref().map(|l| l.index_count).unwrap_or(0);
        (self.branch.index_count + leaf) / 3
    }

    /// Resolved foliage card count (one quad per leaf).
    pub fn leaf_count(&self) -> u32 {
        self.leaf_count
    }

    /// Skeleton bone count.
    pub fn bone_count(&self) -> u32 {
        self.bone_count
    }

    /// Draw calls this tree contributes to the main scene pass: one for the
    /// branch mesh, plus one for the leaf mesh when the tree has leaf cards.
    /// (Sky + ground are drawn by `Staging`, counted separately by the viewer.)
    pub fn draw_call_count(&self) -> u32 {
        1 + if self.leaf.is_some() { 1 } else { 0 }
    }

    /// Branch-mesh AABB in the tree's local space (min, max). Used by a viewer
    /// to frame the tree (auto-fit the orbit distance to the mesh extent).
    pub fn local_bounds(&self) -> ([f32; 3], [f32; 3]) {
        self.bounds
    }

    /// The branch-mesh AABB in the tree-LOCAL WORLD frame (after the model
    /// rotation, before the camera-relative translation) — the frame the shadow
    /// caster + receiver both project from. Used to FIT the sun shadow ortho
    /// frustum. Transforms the 8 corners of the raw mesh AABB by the rotation and
    /// re-bounds. (min, max).
    pub fn world_bounds(&self) -> (Vec3, Vec3) {
        let rot = self.rotation_quat();
        let (mn, mx) = self.bounds;
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for i in 0..8 {
            let c = Vec3::new(
                if i & 1 == 0 { mn[0] } else { mx[0] },
                if i & 2 == 0 { mn[1] } else { mx[1] },
                if i & 4 == 0 { mn[2] } else { mx[2] },
            );
            let w = rot * c;
            lo = lo.min(w);
            hi = hi.max(w);
        }
        (lo, hi)
    }

    /// Draw the tree this frame. Must be called INSIDE the 3D rendering instance
    /// (shares terrain's reversed-Z depth). `camera_world_pos` is the f64 camera
    /// position used for the camera-relative model translation. The tree is drawn
    /// at its real `self.origin` (set at `new()`).
    ///
    /// `wind = [time_secs, strength, dir_x, dir_z]` drives the procedural
    /// global-field sway in the vertex shaders. `strength == 0.0` is an EXACT
    /// static rest pose (the shader's `windOffset` returns zero), so passing
    /// `[_, 0.0, _, _]` freezes the tree with no animation cost or jitter.
    ///
    /// `renderer` is the flora-OWNED sub-renderer: this method binds ITS pipelines
    /// + set0/set1 (not enki's `bind_pipeline`) and pushes against ITS layout, so
    /// the tree renders through flora-owned Vulkan objects. The caller must have
    /// uploaded `fu` via `renderer.set_frame_uniforms(rhi, fi, fu)` first.
    pub fn record(
        &mut self,
        rhi: &mut Rhi,
        renderer: &FloraRenderer,
        fi: u32,
        camera_world_pos: DVec3,
        wind: [f32; 4],
    ) -> Result<(), RhiError> {
        self.record_branch(rhi, renderer, fi, camera_world_pos, wind)?;
        self.record_leaves(rhi, renderer, fi, camera_world_pos, wind)
    }

    /// Record ONLY the branch mesh through the plain flora branch pipeline. Split
    /// out of `record` so the Nanite branch path can draw the branches itself (its
    /// own cull+draw) while the leaves still render as cards via `record_leaves`.
    pub fn record_branch(
        &mut self,
        rhi: &mut Rhi,
        renderer: &FloraRenderer,
        fi: u32,
        camera_world_pos: DVec3,
        wind: [f32; 4],
    ) -> Result<(), RhiError> {
        // Camera-relative model: orient local +Y to the surface normal at origin,
        // translate by (origin - camera) in f64 then cast f32 (no big-coord f32
        // cancellation). ponytail: surface normal = radial (planet-centred) dir.
        let model = self.model(camera_world_pos);
        let wire = self.mode.is_wireframe();
        let dbg = self.mode.debug_mode();

        let branch_pipe = if wire { FloraPipeline::BranchWire } else { FloraPipeline::Branch };
        renderer.bind(rhi, fi, branch_pipe);
        rhi.set_viewport_scissor_full(fi);
        // Write the debug mode into the spare bark2.z lane (see flora.wgsl).
        let mut bark2 = self.bark2;
        bark2[2] = dbg;
        let bpush = BranchPush {
            model: model.to_cols_array_2d(),
            wind,
            bark0: self.bark0,
            bark1: self.bark1,
            bark2,
        };
        rhi.bind_vertex_buffers(
            fi,
            &[self.branch.pos, self.branch.nrm, self.branch.uv, self.branch.attr],
        )?;
        rhi.bind_index_buffer(fi, self.branch.idx)?;
        renderer.push(rhi, fi, bytemuck::bytes_of(&bpush));
        rhi.draw_indexed(fi, self.branch.index_count);
        Ok(())
    }

    /// Record ONLY the leaf cards. Used both by `record` (full tree) and by the
    /// Nanite branch path (which draws the branches via Nanite, then the leaves
    /// here unchanged — Nanite is for the branch geometry only).
    pub fn record_leaves(
        &mut self,
        rhi: &mut Rhi,
        renderer: &FloraRenderer,
        fi: u32,
        camera_world_pos: DVec3,
        wind: [f32; 4],
    ) -> Result<(), RhiError> {
        let model = self.model(camera_world_pos);
        let wire = self.mode.is_wireframe();
        let dbg = self.mode.debug_mode();

        if let Some(leaf) = &self.leaf {
            let leaf_pipe = if wire { FloraPipeline::LeafWire } else { FloraPipeline::Leaf };
            renderer.bind(rhi, fi, leaf_pipe);
            rhi.set_viewport_scissor_full(fi);
            // Write the debug mode into the spare leaf_params2.z lane.
            let mut leaf_params2 = self.leaf_params2;
            leaf_params2[2] = dbg;
            let lpush = LeafPush {
                model: model.to_cols_array_2d(),
                pigment: self.leaf_pigment,
                leaf_params: self.leaf_params,
                leaf_params2,
                wind,
            };
            rhi.bind_vertex_buffers(fi, &[leaf.pos, leaf.nrm, leaf.uv, leaf.attr])?;
            rhi.bind_index_buffer(fi, leaf.idx)?;
            renderer.push(rhi, fi, bytemuck::bytes_of(&lpush));
            rhi.draw_indexed(fi, leaf.index_count);
        }
        Ok(())
    }

    /// The model ROTATION quaternion (the orthonormal 3×3 of `model()` that
    /// orients tree-local +Y to the surface normal). The shadow depth pass needs
    /// it to rotate vertex positions into the same tree-local frame the receiver
    /// projects from. At the world origin this is a fixed basis (right=Z,up=Y,
    /// forward=-X) — see `model()`.
    fn rotation_quat(&self) -> Quat {
        let up = self.origin.normalize_or(DVec3::Y).as_vec3();
        let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
        let right = reference.cross(up).normalize();
        let forward = right.cross(up).normalize();
        Quat::from_mat3(&Mat3::from_cols(right, up, forward))
    }

    /// Record the tree's DEPTH-ONLY shadow casters (branch + leaf) into the shadow
    /// pass. Call between `renderer.begin_shadow_pass` and `end_shadow_pass`. Uses
    /// the SAME branch/leaf meshes as the lit pass, drawn with the light's
    /// view-proj + the model rotation (so the caster matches the receiver). `wind`
    /// is the same vec4 the lit pass uses, so the shadow silhouette sways in sync.
    pub fn record_shadow(
        &self,
        rhi: &mut Rhi,
        renderer: &FloraRenderer,
        fi: u32,
        light_view_proj: Mat4,
        wind: [f32; 4],
    ) -> Result<(), RhiError> {
        let rot = self.rotation_quat();
        let lvp = light_view_proj.to_cols_array_2d();
        let rot4 = [rot.x, rot.y, rot.z, rot.w];

        // ── Branch depth caster ──
        renderer.bind_depth(rhi, fi, FloraPipeline::BranchDepth);
        let bpush = BranchDepthPush {
            light_view_proj: lvp,
            rot: rot4,
            wind,
        };
        rhi.bind_vertex_buffers(
            fi,
            &[self.branch.pos, self.branch.nrm, self.branch.uv, self.branch.attr],
        )?;
        rhi.bind_index_buffer(fi, self.branch.idx)?;
        renderer.push(rhi, fi, bytemuck::bytes_of(&bpush));
        rhi.draw_indexed(fi, self.branch.index_count);

        // ── Leaf depth caster (alpha-tested cutout silhouette) ──
        if let Some(leaf) = &self.leaf {
            renderer.bind_depth(rhi, fi, FloraPipeline::LeafDepth);
            let lpush = LeafDepthPush {
                light_view_proj: lvp,
                rot: rot4,
                wind,
                leaf_params: self.leaf_params,
                leaf_params2: self.leaf_params2,
            };
            rhi.bind_vertex_buffers(fi, &[leaf.pos, leaf.nrm, leaf.uv, leaf.attr])?;
            rhi.bind_index_buffer(fi, leaf.idx)?;
            renderer.push(rhi, fi, bytemuck::bytes_of(&lpush));
            rhi.draw_indexed(fi, leaf.index_count);
        }

        Ok(())
    }

    /// Camera-relative model matrix: rotation aligning local +Y to the surface
    /// normal at `origin`, with the camera-relative translation in the w column.
    fn model(&self, camera_world_pos: DVec3) -> Mat4 {
        // Radial surface normal. ponytail: at the world origin (the standalone
        // viewer's placement) the radial is undefined, so fall back to +Y.
        let up = self.origin.normalize_or(DVec3::Y).as_vec3();
        // Any tangent basis; pick a stable reference not parallel to up.
        let reference = if up.x.abs() < 0.9 { Vec3::X } else { Vec3::Z };
        let right = reference.cross(up).normalize();
        // ponytail fix: right-handed basis (forward = right×up). The old up×right
        // gave determinant −1 (a reflection) → flipped winding → back-face culled.
        let forward = right.cross(up).normalize();
        // Columns: x=right, y=up, z=forward (orient tree +Y to the surface).
        let mut m = Mat4::from_cols(
            right.extend(0.0),
            up.extend(0.0),
            forward.extend(0.0),
            glam::Vec4::W,
        );
        let rel = (self.origin - camera_world_pos).as_vec3();
        m.w_axis.x = rel.x;
        m.w_axis.y = rel.y;
        m.w_axis.z = rel.z;
        m.w_axis.w = 1.0;
        m
    }
}

/// Expand the foliage SoA into a single indexed quad mesh (4 verts + 6 indices
/// per leaf) in the 4×vec3 layout. Returns `None` if there are no leaves.
///
/// ponytail: leaves are CPU-expanded oriented quads (one mesh, one draw) rather
/// than GPU-instanced — see flora.wgsl for why (avoids net-new RHI). Each card
/// is sized scale×(0.5 wide, 0.65 tall) on the (tangent, bitangent) basis with
/// the genome `rotation` roll, matching the dropped vs_leaf math.
///
/// The `normal` stream carries the dryad CANOPY SPHERE-NORMAL (outward from the
/// AABB center of all leaf positions), softly blended (80%) with the card face
/// normal, so the crown shades as a soft volume instead of flat cards. `attr.z`
/// carries a per-leaf variation seed (a cheap stable position hash) used in the
/// shader for per-leaf lightness/hue jitter.
// ── Single-leaf expansion (port of dryad expandClumpsToLeaves, foliage.js:918) ──
// Each cluster anchor is fanned into K individual single-leaf cards. The whole
// transform is ORIENTATION-only (every leaf shares the anchor base `P` —
// "embedded pivot"), with a per-leaf scale shrink + roll jitter.
//
// ponytail / approximations vs dryad:
//   * RNG draw order is matched EXACTLY (per clump: az0; per leaf: sizeJ then
//     rollJitter) so the spray is bit-stable on reseed, keyed on the structural
//     seed ^ LEAF_FAN_SALT (its own isolated stream — same as dryad).
//   * The frondyness/spininess GATE (K=1 passthrough for compound fronds /
//     needles / spiny / very-narrow leaves) is ported so palm/cactus/conifer
//     keep one whole-unit card per anchor.
//   * Vector math is inlined here (norm/cross/rotate/dot on [f32;3]) rather than
//     re-exposing foliage.rs's private f64 helpers — a tiny duplication, but it
//     keeps the SoA in its native f32 and avoids widening foliage's API.

const LEAVES_PER_CLUMP: usize = 6; // dryad foliage.js:57
const LEAF_FAN_SALT: u32 = 0x1EAF_0FA2; // dryad foliage.js:60
const LEAF_FAN_SCALE: f32 = 0.58; // dryad LEAF_SCALE (foliage.js:950)
const LEAF_FAN_SPLAY: f32 = 0.55; // dryad SPLAY (foliage.js:951)

#[inline]
fn v_norm(v: [f32; 3]) -> [f32; 3] {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    if l < 1e-10 {
        [1.0, 0.0, 0.0]
    } else {
        [v[0] / l, v[1] / l, v[2] / l]
    }
}
#[inline]
fn v_cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
fn v_dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
/// Rodrigues rotation of `v` around unit axis `ax` by `angle` (radians).
#[inline]
fn v_rotate(v: [f32; 3], ax: [f32; 3], angle: f32) -> [f32; 3] {
    let (s, c) = angle.sin_cos();
    let d = v_dot(v, ax);
    let cr = v_cross(ax, v);
    [
        v[0] * c + cr[0] * s + ax[0] * d * (1.0 - c),
        v[1] * c + cr[1] * s + ax[1] * d * (1.0 - c),
        v[2] * c + cr[2] * s + ax[2] * d * (1.0 - c),
    ]
}

/// Expand each foliage cluster into `K = LEAVES_PER_CLUMP` single-leaf cards.
/// Returns a fresh `FoliageSoA` so `build_leaf_mesh` runs UNCHANGED on it. The
/// gate (`frondyness`/`spininess`) yields K=1 (a clone) for non-broadleaf forms.
/// `seed` is the genome structural seed (keys the isolated fan RNG).
fn expand_clumps_to_leaves(
    f: &enki_flora::foliage::FoliageSoA,
    seed: u32,
    frondyness: f64,
    spininess: f64,
) -> enki_flora::foliage::FoliageSoA {
    use enki_flora::rng::Mulberry32;
    let k = if frondyness > 0.5 || spininess > 0.05 {
        1
    } else {
        LEAVES_PER_CLUMP
    };
    if k <= 1 || f.count == 0 {
        return f.clone();
    }
    let golden = enki_flora::foliage::GOLDEN_ANGLE as f32;
    let mut rng = Mulberry32::new(seed ^ LEAF_FAN_SALT);

    let out_n = f.count * k;
    let mut position = Vec::with_capacity(3 * out_n);
    let mut normal = Vec::with_capacity(3 * out_n);
    let mut tangent = Vec::with_capacity(3 * out_n);
    let mut scale = Vec::with_capacity(out_n);
    let mut rotation = Vec::with_capacity(out_n);
    let mut age_color = Vec::with_capacity(out_n);
    let mut exposure = Vec::with_capacity(out_n);
    let mut bone_index = Vec::with_capacity(out_n);

    for c in 0..f.count {
        let c3 = c * 3;
        let p = [f.position[c3], f.position[c3 + 1], f.position[c3 + 2]];
        let t = v_norm([f.tangent[c3], f.tangent[c3 + 1], f.tangent[c3 + 2]]);
        let nf = v_norm([f.normal[c3], f.normal[c3 + 1], f.normal[c3 + 2]]);
        let s = f.scale[c];
        let age = f.age_color[c];
        let exp = f.exposure[c];
        let bone = f.bone_index[c];
        let roll0 = f.rotation[c];
        let x = v_norm(v_cross(t, nf)); // clump "right" axis

        let az0 = (rng.next() as f32) * std::f32::consts::TAU; // one draw / clump
        for j in 0..k {
            let az = az0 + j as f32 * golden;
            let radial = v_rotate(x, t, az);
            // Embedded pivot: every leaf shares the anchor base `p`.
            // Midrib fans from clump tangent toward radial by SPLAY.
            let lt = v_norm([
                t[0] * (1.0 - LEAF_FAN_SPLAY) + radial[0] * LEAF_FAN_SPLAY,
                t[1] * (1.0 - LEAF_FAN_SPLAY) + radial[1] * LEAF_FAN_SPLAY,
                t[2] * (1.0 - LEAF_FAN_SPLAY) + radial[2] * LEAF_FAN_SPLAY,
            ]);
            // Face normal: clump normal re-orthogonalized against the midrib.
            let d = v_dot(nf, lt);
            let mut ln = v_norm([nf[0] - lt[0] * d, nf[1] - lt[1] * d, nf[2] - lt[2] * d]);
            if !ln[0].is_finite() {
                ln = nf; // NaN guard (radial ∥ tangent)
            }
            let size_j = 0.85 + (rng.next() as f32) * 0.30; // 1 draw / leaf
            position.extend_from_slice(&p);
            normal.extend_from_slice(&ln);
            tangent.extend_from_slice(&lt);
            scale.push(s * LEAF_FAN_SCALE * size_j);
            rotation.push(roll0 + ((rng.next() as f32) - 0.5) * 0.4); // 1 draw / leaf
            age_color.push(age);
            exposure.push(exp);
            bone_index.push(bone);
        }
    }

    enki_flora::foliage::FoliageSoA {
        count: out_n,
        position,
        normal,
        tangent,
        scale,
        rotation,
        age_color,
        exposure,
        bone_index,
        shape: f.shape,
    }
}

fn build_leaf_mesh(
    rhi: &mut Rhi,
    foliage: &enki_flora::foliage::FoliageSoA,
    bones_wind: &[WindBone],
) -> Result<Option<GpuMesh>, RhiError> {
    let n = foliage.count;
    if n == 0 {
        return Ok(None);
    }

    // Per-leaf nearest wind bone: the foliage `bone_index` is always 0 (resolve
    // has no nodeToBone), so we map each leaf to its NEAREST branch-bone PIVOT
    // here instead — a faithful-enough leaf bone-follow without threading
    // nodeToBone through the deterministic foliage path. // ponytail: O(leaves ×
    // bones) nearest-pivot once at build time (≤ a few × 10⁴ × 10³ for one tree);
    // skip the trunk anchor (bone 0) so leaves follow a swaying twig, not the
    // pinned base. Prefer non-rigid bones; fall back to 0 if none.
    let nearest_bone = |p: Vec3| -> f32 {
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for (bi, b) in bones_wind.iter().enumerate() {
            if b.is_rigid == 1 {
                continue;
            }
            let pv = Vec3::from_array(b.pivot);
            let d = (p - pv).length_squared();
            if d < best_d {
                best_d = d;
                best = bi;
            }
        }
        best as f32
    };

    // ── Pass 1: canopy AABB center (dryad leafMesh.js:766-777). NOT the mean ──
    // and NOT the tree origin: the midpoint of the per-axis min/max of every
    // leaf anchor position. Per-leaf canopy normals radiate outward from it.
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for i in 0..n {
        let p = Vec3::new(
            foliage.position[3 * i],
            foliage.position[3 * i + 1],
            foliage.position[3 * i + 2],
        );
        lo = lo.min(p);
        hi = hi.max(p);
    }
    let center = (lo + hi) * 0.5;

    let mut pos = Vec::with_capacity(n * 4 * 3);
    let mut nrm = Vec::with_capacity(n * 4 * 3);
    let mut uv = Vec::with_capacity(n * 4 * 3);
    let mut attr = Vec::with_capacity(n * 4 * 3);
    let mut idx = Vec::with_capacity(n * 6);

    // Quad corners in (tangent, bitangent) space: 0:(-1,-1) 1:(1,-1) 2:(-1,1) 3:(1,1).
    const CORNERS: [(f32, f32); 4] = [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)];
    // dryad canopyBlend = 0.8 (uWeep=0): shading normal is 80% canopy-sphere
    // normal, 20% card face normal — a soft volume, but the card keeps a little
    // of its own facing so flat-on leaves don't read perfectly spherical.
    const CANOPY_BLEND: f32 = 0.8;

    for i in 0..n {
        let p = Vec3::new(
            foliage.position[3 * i],
            foliage.position[3 * i + 1],
            foliage.position[3 * i + 2],
        );
        let normal = Vec3::new(
            foliage.normal[3 * i],
            foliage.normal[3 * i + 1],
            foliage.normal[3 * i + 2],
        );
        let tangent = Vec3::new(
            foliage.tangent[3 * i],
            foliage.tangent[3 * i + 1],
            foliage.tangent[3 * i + 2],
        );
        let scale = foliage.scale[i];
        let roll = foliage.rotation[i];
        let age = foliage.age_color[i];
        let exposure = foliage.exposure[i];

        // Orthonormal leaf basis (Gram-Schmidt the tangent against the normal).
        let nn = normal.normalize_or(Vec3::Y);
        let mut t = tangent.normalize_or(Vec3::X);
        t = (t - nn * nn.dot(t)).normalize_or(Vec3::X);
        let b = nn.cross(t);
        // Roll about the leaf normal.
        let (sn, cs) = roll.sin_cos();
        let t_r = t * cs + b * sn;
        let b_r = b * cs - t * sn;

        // Canopy sphere-normal: outward from the AABB center (dryad:837-849).
        // Fallback to +Y exactly at the center. Blend 80% canopy / 20% card.
        let cn = (p - center).normalize_or(Vec3::Y);
        let lit_n = (cn * CANOPY_BLEND + nn * (1.0 - CANOPY_BLEND)).normalize_or(cn);

        // Per-leaf variation seed in [0,1): a stable hash of the leaf position.
        // (No Math.random; deterministic on reseed, like dryad's index LCGs.)
        let seed = leaf_hash(p);
        // Nearest non-rigid wind bone (anchor = the leaf cluster base `p`), baked
        // into uv.z so the leaf VS bone-follows its twig's hierarchical sway.
        let bone_idx = nearest_bone(p);

        // Leaf card size on the (tangent, bitangent) basis. Kept at dryad's
        // 0.5 wide × 0.65 tall: with the now-binary alpha cutout (flora.wgsl
        // fs_leaf, alphaTest 0.5) overlapping leaves occlude opaquely, so this
        // size reads as many distinct leaf SILHOUETTES rather than a few large
        // overlapping pale blobs (bigger cards merged the sunlit crown into one
        // glassy mass — the opposite of "visible leaf silhouettes"). // ponytail.
        let half_w = scale * 0.5;
        let half_h = scale * 0.65;
        let base = (pos.len() / 3) as u32;

        for &(cx, cy) in &CORNERS {
            let offset = t_r * (cx * half_w) + b_r * (cy * half_h);
            let corner = p + offset;
            pos.extend_from_slice(&[corner.x, corner.y, corner.z]);
            // slot1 = canopy-blended lighting normal (the dryad volume trick).
            nrm.extend_from_slice(&[lit_n.x, lit_n.y, lit_n.z]);
            // UV in [0,1]² from the [-1,1] corner; .z = wind boneIndex.
            uv.extend_from_slice(&[cx * 0.5 + 0.5, cy * 0.5 + 0.5, bone_idx]);
            // attr = (age, exposure, per-leaf variation seed).
            attr.extend_from_slice(&[age, exposure, seed]);
        }
        // Two CCW triangles: [0,1,2, 2,1,3].
        idx.extend_from_slice(&[base, base + 1, base + 2, base + 2, base + 1, base + 3]);
    }

    Ok(Some(GpuMesh {
        pos: rhi.create_vertex_buffer(bytemuck::cast_slice(&pos))?,
        nrm: rhi.create_vertex_buffer(bytemuck::cast_slice(&nrm))?,
        uv: rhi.create_vertex_buffer(bytemuck::cast_slice(&uv))?,
        attr: rhi.create_vertex_buffer(bytemuck::cast_slice(&attr))?,
        idx: rhi.create_index_buffer(&idx)?,
        index_count: idx.len() as u32,
    }))
}

/// Cheap stable per-leaf hash in `[0,1)` from the leaf anchor position.
/// Deterministic (no rng), so reseeding the genome reshuffles the per-leaf
/// jitter coherently with the new layout — dryad uses index-seeded LCGs, but a
/// position hash is just as stable and needs no leaf index plumbed downstream.
fn leaf_hash(p: Vec3) -> f32 {
    // Mix the three coords through a fract-of-sine hash (classic GPU-style),
    // folded so nearby leaves still differ. Result in [0,1).
    let h = (p.x * 127.1 + p.y * 311.7 + p.z * 74.7).sin() * 43758.547;
    h - h.floor()
}

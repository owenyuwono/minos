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
use enki_flora::mesh::{build_branch_mesh_with, BranchMeshOpts, WindBone};
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

/// Render-side leaf-placement tuning knobs (phyllotaxis adjustment), applied in
/// [`build_leaf_mesh`] on top of the gen-core SoA orientation/droop/size. These
/// are RENDER-ONLY — NOT genome genes — so editing them never touches the golden
/// gen-core path (`resolve`/`generate_foliage`); golden stays bit-stable.
///
/// The SoA already bakes the dryad phyllotaxis (azimuth = golden-blend, outward
/// `out_angle` insertion lean, PHOTO_STRENGTH 0.25 up-bias, weep). These scalars
/// are an EXTRA render-side adjustment over that baked tangent/droop/size:
///
/// * `lift` — insertion/elevation adjustment. Lerps the base→tip leaf tangent
///   between MORE-OUTWARD-HORIZONTAL (`< 0`) and MORE-UP toward +Y (`> 0`). 0 ⇒
///   use the baked SoA tangent unchanged.
/// * `up_bias` — extra phototropic mix of the leaf tangent toward +Y (on top of
///   the gen-core PHOTO_STRENGTH 0.25). 0 ⇒ no extra up-bias.
/// * `droop` — multiplier on the gravity bend/droop (`LEAF_BEND` 0.45). 1.0 ⇒
///   the default gentle droop; > 1 droops more, 0 removes the droop.
/// * `size` — uniform scale on the leaf card (on top of the width/length genes
///   and `CANOPY_FILL`). 1.0 ⇒ unchanged.
/// * `density` — render-side leaf-cluster subsample fraction in [0,1]. The
///   gen-core `appendageDensity` gene → cluster COUNT is cliffy (dense or
///   nothing); this deterministically keeps each anchor iff a stable per-anchor
///   hash < `density`, giving a SMOOTH sparse→full range. 1.0 ⇒ keep all.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeafTuning {
    pub lift: f32,
    pub up_bias: f32,
    pub droop: f32,
    pub size: f32,
    pub density: f32,
}

impl LeafTuning {
    /// Parse the optional `FLORA_LEAFDENSITY=<0..1>` env var (headless capture
    /// of the sparse→full range). Out of range or unset ⇒ the 0.7 default.
    // ponytail: one parse, falls through to the struct default.
    pub fn density_from_env(default: f32) -> f32 {
        std::env::var("FLORA_LEAFDENSITY")
            .ok()
            .and_then(|s| s.trim().parse::<f32>().ok())
            .map(|d| d.clamp(0.0, 1.0))
            .unwrap_or(default)
    }
}

impl Default for LeafTuning {
    /// Identity defaults that reproduce the CURRENT orientation exactly: no extra
    /// lift, no extra up-bias, ×1.0 droop. `size`/`density` carry the new
    /// render-side defaults (smaller, proportional leaf; a 0.7 cluster subsample
    /// for a leafy-but-not-solid canopy). The dryad-inspired up_bias/insertion
    /// 0.25 is ALREADY baked into the SoA by gen-core, so the render-side EXTRA
    /// bias starts at 0.
    fn default() -> Self {
        Self {
            lift: 0.0,
            up_bias: 0.0,
            droop: 1.0,
            // ponytail: smaller default leaf reads proportional to the tree.
            size: 1.0,
            // ponytail: render-side subsample default (smooth, not the gene cliff).
            density: 0.7,
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
        Self::from_genome_with_mode_tuned(rhi, genome, env, origin, leaf_mode, LeafTuning::default())
    }

    /// As [`from_genome_with_mode`], with explicit render-side leaf-placement
    /// tuning. The tuning is RENDER-ONLY (it only reaches `build_leaf_mesh`), so
    /// it never affects the golden gen-core path.
    pub fn from_genome_with_mode_tuned(
        rhi: &mut Rhi,
        genome: &Genome,
        env: &Env,
        origin: DVec3,
        leaf_mode: LeafMode,
        tuning: LeafTuning,
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
            tuning,
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
        tuning: LeafTuning,
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
        // Twig-tip wood cull: drop the bark TUBE on leaf-bearing terminal twigs
        // (branch_level ≥ TWIG_MIN_LEVEL — the SAME level the leaf gate clothes),
        // so the canopy reads as a leafy mass, not bare reddish sticks poking
        // through. Trunk + thick branches (level < 3) keep their bark. Bones are
        // unaffected, so wind/nanite/leaf-skinning are untouched.
        let bm = build_branch_mesh_with(
            &resolved.graph,
            BranchMeshOpts { twig_cull_level: Some(TWIG_MIN_LEVEL as i32) },
        );
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
        // ponytail: thread the leaf mode + width/length genes so the card aspect
        // matches dryad (square at default genes) instead of a fixed blade shape.
        let leaf = build_leaf_mesh(
            rhi,
            foliage_ref,
            &bm.bones_wind,
            leaf_mode,
            resolved.leaf_width,
            resolved.leaf_length,
            resolved.weep,
            tuning,
        )?;

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

    /// This tree's bark push-constant vec4s (the SAME genes `fs_branch` reads):
    /// `bark0 = (hue, lightness, relief, lenticels)`, `bark1 = (scale, orient,
    /// plates, shed)`, `bark2 = (under_hue, woodiness, 0, 0)`. The Inspector's
    /// CPU bark swatch feeds these to `enki_flora::bark_swatch::bake_bark_swatch`.
    pub fn bark_genes(&self) -> ([f32; 4], [f32; 4], [f32; 4]) {
        (self.bark0, self.bark1, self.bark2)
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
        // In-scene: tell fs_branch to ACES-tonemap (matches terrain). Viewer: 0.
        bark2[3] = if renderer.in_scene() { 1.0 } else { 0.0 };
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
            // In-scene: ACES-tonemap in fs_leaf (matches terrain). Viewer: 0.
            leaf_params2[3] = if renderer.in_scene() { 1.0 } else { 0.0 };
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

// ── Twigs-only leaf gate (render-side; does NOT touch resolve()/golden) ──────
// "Twigs = fine terminal branches." dryad/generate_foliage already excludes the
// trunk (branch_level 0) and roots, but still places sparse clusters on thick
// structural branches (branch_level 1–2). We render leaves ONLY where the
// source node is a real twig:
//   keep iff  branch_level >= TWIG_MIN_LEVEL                              (always)
//          OR (is_terminal AND source_radius <= TWIG_RADIUS_FRAC*max)    (thin tips)
//
// CRITICAL — the leaf gate must COVER the branch-mesh twig cull. The mesh now
// culls the bark tube on every TIP chain whose branch_level >= TWIG_MIN_LEVEL
// (see build_branch_mesh_with above). If the leaf gate dropped any of those same
// anchors we'd get a BALD GAP (no bark, no leaf). So the `branch_level >=
// TWIG_MIN_LEVEL` clause is now UNCONDITIONAL (no radius cap) — every twig the
// mesh stripped of bark is guaranteed leaves. The relative radius cap only
// belt-and-suspenders the SECONDARY `is_terminal` clause, which catches twigs
// tagged terminal at a lower level (level 1–2) on stubby specimens; those still
// have their bark (the cull only touches level >= 3), so it is safe to be picky
// there. The cap was raised 0.40 → 0.55 so fewer terminal twigs on shallow-taper
// trees are stranded.
const TWIG_MIN_LEVEL: f32 = 3.0; // OUTER_LEVEL_THRESHOLD (foliage.js:139)
const TWIG_RADIUS_FRAC: f32 = 0.55; // keep terminal-clause anchors on wood ≤ 55% of the thickest anchor radius

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
    // Carry source-node metadata through the fan 1:1 so the twig filter still
    // works post-expansion.
    let mut source_radius = Vec::with_capacity(out_n);
    let mut source_branch_level = Vec::with_capacity(out_n);
    let mut source_is_terminal = Vec::with_capacity(out_n);

    for c in 0..f.count {
        let c3 = c * 3;
        let p = [f.position[c3], f.position[c3 + 1], f.position[c3 + 2]];
        let t = v_norm([f.tangent[c3], f.tangent[c3 + 1], f.tangent[c3 + 2]]);
        let nf = v_norm([f.normal[c3], f.normal[c3 + 1], f.normal[c3 + 2]]);
        let s = f.scale[c];
        let age = f.age_color[c];
        let exp = f.exposure[c];
        let bone = f.bone_index[c];
        let src_r = f.source_radius[c];
        let src_bl = f.source_branch_level[c];
        let src_term = f.source_is_terminal[c];
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
            source_radius.push(src_r);
            source_branch_level.push(src_bl);
            source_is_terminal.push(src_term);
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
        source_radius,
        source_branch_level,
        source_is_terminal,
    }
}

fn build_leaf_mesh(
    rhi: &mut Rhi,
    foliage: &enki_flora::foliage::FoliageSoA,
    bones_wind: &[WindBone],
    // ponytail: leaf mode + width/length genes drive the card aspect (dryad
    // leafMesh.js buildInstanceMatrix scales the card by gene-derived factors).
    leaf_mode: LeafMode,
    leaf_width: f64,
    leaf_length: f64,
    // ponytail: willow weep gene scales the per-leaf gravity DROOP (dryad uWeep
    // drives the willow hang). 0 → just the default gentle bend/droop.
    weep: f64,
    // Render-side leaf-placement tuning (lift/up_bias/droop/size). Identity
    // default reproduces the current look; golden never runs this fn.
    tuning: LeafTuning,
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

    // ── Twigs-only gate: per-tree thickest anchor radius for the RELATIVE cap. ──
    let mut max_src_radius = 0.0f32;
    for i in 0..n {
        let r = foliage.source_radius[i];
        if r > max_src_radius {
            max_src_radius = r;
        }
    }
    let twig_radius_max = if max_src_radius > 0.0 {
        max_src_radius * TWIG_RADIUS_FRAC
    } else {
        f32::INFINITY // no metadata → keep all (graceful fallback)
    };
    // A leaf anchor is a "twig" iff it is an outer-level branch (unconditional,
    // so it always covers the mesh's bark cull) OR a thin terminal tip.
    let is_twig = |i: usize| -> bool {
        let terminal = foliage.source_is_terminal[i] >= 0.5;
        let level = foliage.source_branch_level[i];
        let radius = foliage.source_radius[i];
        level >= TWIG_MIN_LEVEL || (terminal && radius <= twig_radius_max)
    };

    // ── Base-anchored CURVED STRIP (dryad leafMesh.js). Each leaf is a 2-wide ×
    // (SEG+1)-row grid: base row sits AT the foliage anchor (on the twig), the
    // strip extends up the leaf tangent to the tip, BENDS along the face normal
    // (LEAF_BEND) and DROOPS toward gravity (×t², scaled by the weep gene). Per
    // vertex t = row/SEG ∈ [0,1] (base→tip) is carried in uv.y, so the FS samples
    // the sprite base→tip AND the VS graduates the wind gust by t (tip sways, base
    // pinned). dryad subdivides a PlaneGeometry in the GPU; enki CPU-expands.
    const SEG: usize = 6; // LEAF_LENGTH_SEGMENTS (leafMesh.js:77) → 7 rows, 14 verts, 12 tris
    const ROWS: usize = SEG + 1;
    let mut pos = Vec::with_capacity(n * ROWS * 2 * 3);
    let mut nrm = Vec::with_capacity(n * ROWS * 2 * 3);
    let mut uv = Vec::with_capacity(n * ROWS * 2 * 3);
    let mut attr = Vec::with_capacity(n * ROWS * 2 * 3);
    let mut idx = Vec::with_capacity(n * SEG * 6);

    // dryad canopyBlend = 0.8 (uWeep=0): shading normal is 80% canopy-sphere
    // normal, 20% card face normal — a soft volume, but the card keeps a little
    // of its own facing so flat-on leaves don't read perfectly spherical. The weep
    // gene shifts the blend toward the geometric (sky-facing) normal so willow
    // leaves under a hanging canopy catch overhead light (dryad leafMesh.js:457:
    // canopyBlend = 0.8 * (1 - uWeep)).
    let weep_f = (weep as f32).clamp(0.0, 1.0);
    let canopy_blend: f32 = 0.8 * (1.0 - weep_f);
    // Gravity bend/droop: tip droops by this fraction of leaf length, quadratic in
    // t (dryad LEAF_BEND_DEFAULT = 0.45, leafMesh.js:67,379). The weep gene
    // increases the droop for willows; default (weep≈0) keeps the gentle 0.45 bend.
    const LEAF_BEND_DEFAULT: f32 = 0.45;
    // Render-side DROOP multiplier on the gravity bend/droop (tuning.droop, 1.0
    // = unchanged). Non-negative so a slider can't invert the curl.
    let leaf_bend = LEAF_BEND_DEFAULT * (1.0 + weep_f * 1.5) * tuning.droop.max(0.0);

    for i in 0..n {
        // ── Twigs only: skip anchors that sit on the trunk / thick structural
        // branches. Leaves render ONLY on fine terminal twigs. ──
        if !is_twig(i) {
            continue;
        }
        // ── Render-side DENSITY subsample. The gen-core appendageDensity gene →
        // cluster COUNT is cliffy (dense or nothing); subsample the kept anchors
        // by a stable per-anchor hash so `tuning.density` gives a SMOOTH
        // sparse→full range that composes with is_twig (keep iff twig AND passes
        // density). Deterministic in `i` (NOT rng-per-frame) so it never
        // flickers; density ≥ 1 keeps everything. // ponytail: hash fract, no rng.
        let density = tuning.density.clamp(0.0, 1.0);
        if density < 1.0 {
            let h = {
                let s = (i as f32 * 12.9898).sin() * 43758.5453;
                s - s.floor() // fract → [0,1)
            };
            if h >= density {
                continue;
            }
        }
        let anchor = Vec3::new(
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
        let mut t_r = t * cs + b * sn;
        let mut b_r = b * cs - t * sn;

        // ── Seat the leaf BASE on the twig WOOD SURFACE. ──
        // The generation anchor for body/apical clusters is already pushed out
        // from the branch axis by mid_radius (foliage.rs cluster_base), so it
        // sits on the bark. To GUARANTEE the base touches — and never sinks
        // into — the twig wood regardless of the generation path (the spine
        // pass anchors on the centerline), explicitly seat the base a
        // source_radius (the wood radius at this anchor) outward along the
        // leaf's outward direction. The outward direction is the radial
        // component of the leaf tangent (the blade leans out from the twig);
        // fall back to canopy-outward if the tangent is ~vertical.
        let radius = foliage.source_radius[i];
        let canopy_out = (anchor - center).normalize_or(Vec3::Y);
        // Radial = leaf tangent with its along-canopy-up component removed, i.e.
        // the horizontal lean away from the twig; blended toward canopy-outward
        // for stability.
        let tan_out = t_r - canopy_out * t_r.dot(canopy_out);
        let outward = (canopy_out + tan_out)
            .normalize_or(canopy_out)
            .normalize_or(Vec3::Y);
        // The anchor sits on the ORIGINAL (untapered) wood surface
        // (cluster_base = centerline + radial*mid_radius). The branch mesher now
        // TAPERS the twig tube toward the tip (mesh.rs twig taper), so the
        // rendered wood surface is pulled INWARD of the anchor by the taper gap.
        // Seating the base further OUT would float it off the thinned wood, so we
        // instead seat it slightly INWARD (toward the canopy center) — a small
        // insertion that lands the base ON / just inside the rendered twig wood
        // for every leaf, never hanging in air. `radius` (mid_radius) is tiny on
        // twigs by construction, so the insertion is small and reads as the leaf
        // emerging from the bark. // ponytail: one mul-add, no extra alloc.
        const SEAT_FRAC: f32 = -0.6; // negative = seat inward (insertion)
        let p = anchor + outward * (radius * SEAT_FRAC);

        // ── Render-side LIFT (insertion/elevation) + UP_BIAS adjustment. ──
        // The base→tip direction is `t_r` (the SoA-baked outward+phototropic
        // lean). `lift` re-aims it between MORE-OUTWARD-HORIZONTAL (lift < 0,
        // toward `outward` flattened to horizontal) and MORE-UP (lift > 0,
        // toward +Y); `up_bias` adds an extra phototropic mix toward +Y on top
        // of the gen-core PHOTO_STRENGTH. Both default to 0 → `t_r` unchanged.
        // After re-aiming the length axis we re-orthonormalize the width axis
        // `b_r` against it (about the leaf face normal) so the card stays planar.
        if tuning.lift != 0.0 || tuning.up_bias != 0.0 {
            // The card face normal (rotation axis): preserved so width stays in
            // the original leaf plane as we tilt the length axis.
            let axis = t_r.cross(b_r).normalize_or(nn);
            // LIFT target: a horizontal outward direction (for lift<0) vs +Y
            // (for lift>0). `outward` already points away from the twig; flatten
            // its +Y component for the "more horizontal" target.
            let horiz = (outward - Vec3::Y * outward.dot(Vec3::Y)).normalize_or(outward);
            let lift = tuning.lift.clamp(-1.0, 1.0);
            let target = if lift >= 0.0 {
                Vec3::Y
            } else {
                horiz
            };
            let mut t_adj = t_r.lerp(target, lift.abs()).normalize_or(t_r);
            // Extra phototropic up-bias toward +Y.
            let ub = tuning.up_bias.clamp(0.0, 1.0);
            if ub > 0.0 {
                t_adj = t_adj.lerp(Vec3::Y, ub).normalize_or(t_adj);
            }
            t_r = t_adj;
            // Re-derive the width axis orthogonal to the new length axis, in the
            // original leaf plane (about `axis`), so the card stays a planar quad.
            b_r = axis.cross(t_r).normalize_or(b_r);
        }

        // Canopy sphere-normal: outward from the AABB center (dryad:837-849).
        // Fallback to +Y exactly at the center. Blend (1-weep)*0.8 canopy / rest
        // card — weep leans willow leaves on their geometric normal.
        let cn = (p - center).normalize_or(Vec3::Y);

        // Per-leaf variation seed in [0,1): a stable hash of the leaf position.
        // (No Math.random; deterministic on reseed, like dryad's index LCGs.)
        let seed = leaf_hash(p);
        // Nearest non-rigid wind bone (anchor = the leaf cluster base `p`), baked
        // into uv.z so the leaf VS bone-follows its twig's hierarchical sway.
        let bone_idx = nearest_bone(p);

        // Leaf card size on the (tangent, bitangent) basis. dryad scales the card
        // by gene-derived width/length factors on an EQUAL 0.5 base, so default
        // genes give a SQUARE card (a broad ovate), not a fixed 0.5×0.65 blade.
        // (Old fixed 0.65-tall card read as a thin spike for the single sprite.)
        // With the binary alpha cutout (flora.wgsl fs_leaf, alphaTest 0.5)
        // overlapping leaves still occlude opaquely → distinct silhouettes.
        // ponytail: dryad leafWidthFactor/leafLengthFactor (leafTexture.js:478-484).
        let (wf, lf) = match leaf_mode {
            LeafMode::Single => (
                0.4 + leaf_width as f32 * 1.2,  // [0.4,1.6], 1.0 at width 0.5
                0.55 + leaf_length as f32 * 1.0, // [0.55,1.55], ~1.0 at length 0.45
            ),
            LeafMode::Cluster => (1.0, 1.0), // xScale baked into the cluster sprite; card stays square
        };
        // Canopy-fill gain: enlarge every card so adjacent leaves OVERLAP and fuse
        // into a continuous leafy mass (dryad's LEAF_BASE 0.80 "cards fuse rather
        // than isolated sprigs on visible twigs" intent). With the twig-tip bark
        // cull, a sparse card layout would let the background show through where
        // the wood used to be; the overlap closes the canopy so no twig reads bare.
        // Applied to BOTH axes so the silhouette stays leaf-shaped, not stretched.
        // ponytail: dropped from 1.25 → 0.6 — even the min size read oversized;
        // the smaller base makes the DEFAULT canopy proportional to the tree
        // (the "Leaf size" slider still spans 0.2..2.0 for finer / fuller tuning).
        const CANOPY_FILL: f32 = 0.6;
        // Render-side uniform LEAF_SIZE scale (tuning.size, 1.0 = unchanged),
        // applied to BOTH axes on top of the width/length genes + CANOPY_FILL so
        // the silhouette stays leaf-shaped. Non-negative.
        let size = tuning.size.max(0.0);
        let half_w = scale * 0.5 * wf * CANOPY_FILL * size;
        // Base-anchored strip: full leaf LENGTH up the tangent (dryad leafLen =
        // s*lengthFactor). Was half_h=scale*0.5*lf for the CENTERED quad; the
        // base-anchored strip extends the full `length` from the anchor to the tip.
        let length = scale * lf * CANOPY_FILL * size;
        // Card face normal (the geometric leaf facing) = t_r × b_r ≈ nn.
        let face_n = t_r.cross(b_r).normalize_or(nn);
        // Gravity DOWN in tree-local space. The tree stands with local +Y = surface
        // normal (model() aligns local +Y to the up at the origin), so local -Y is
        // gravity. The model rotation is applied in the VS; for the upright single
        // specimen this is ≈ world-down (matches dryad's view-space −Y droop).
        let down = Vec3::NEG_Y;
        let base = (pos.len() / 3) as u32;

        // Emit ROWS rows (base→tip), 2 cross columns each. Per row r: t = r/SEG.
        for r in 0..ROWS {
            let t = r as f32 / SEG as f32;
            // BEND: curl the blade OUT of its plane along the face normal, quadratic
            // in t (so the tip bows). Amount ∝ leaf length × leaf_bend.
            let bend = face_n * (t * t * length * leaf_bend * 0.5);
            // DROOP: pull the blade toward gravity, quadratic in t, scaled by the
            // weep gene (dryad's gravity term). Present at wind strength 0 (it's
            // SHAPE, not wind). Default weep≈0 → a gentle droop from leaf_bend.
            let droop = down * (t * t * length * leaf_bend);
            // base (t=0) at anchor, tip (t=1) OUTWARD along the tangent. dryad bakes
            // the outward-radial + up-bias lean into clusterTangent (= our SoA
            // tangent → `t_r`), so the blade LENGTH must run along `t_r` to stick OUT
            // from the twig (leafMesh.js col1 = tangent = length). The bitangent
            // `b_r` is the in-plane WIDTH. (Was: length=b_r/width=t_r — that grew the
            // blade UP and wasted the outward lean on width, so gravity made it
            // rise-then-curl. Now the outward tip hangs.)
            let up_off = t_r * (t * length);
            for &cx in &[-1.0f32, 1.0f32] {
                let corner = p + b_r * (cx * half_w) + up_off + bend + droop;
                pos.extend_from_slice(&[corner.x, corner.y, corner.z]);
                // ── Recompute the bent-strip face normal for lighting (dryad keeps
                // the card normal flat but the curl SHOULD catch light). The up-
                // tangent tilts as the blade curls/droops: d(up_off+bend+droop)/dt.
                // The recomputed card normal then blends with the canopy sphere
                // normal (weep-scaled) into slot1 — sphere-dominant like dryad.
                let d_up = t_r * length
                    + face_n * (t * length * leaf_bend) // d(bend)/dt = 2t·…·0.5
                    + down * (2.0 * t * length * leaf_bend); // d(droop)/dt
                let up_tan = d_up.normalize_or(t_r);
                // card_n = length × width = up_tan × b_r ≈ nn (the lit face normal).
                let card_n = up_tan.cross(b_r).normalize_or(face_n);
                let lit_n = (cn * canopy_blend + card_n * (1.0 - canopy_blend))
                    .normalize_or(cn);
                nrm.extend_from_slice(&[lit_n.x, lit_n.y, lit_n.z]);
                // UV: .x = cross [0,1], .y = t (base→tip, sprite mapped once up the
                // strip — dryad's continuous V across the subdivided plane); .z =
                // wind boneIndex. uv.y IS the per-vertex t the VS reads for the
                // base-anchored wind graduation (tip sways, base pinned).
                uv.extend_from_slice(&[cx * 0.5 + 0.5, t, bone_idx]);
                // attr = (age, exposure, per-leaf variation seed).
                attr.extend_from_slice(&[age, exposure, seed]);
            }
        }
        // SEG segments, 2 CCW tris each. Row r: a=2r,b=2r+1,c=2r+2,d=2r+3.
        for r in 0..SEG as u32 {
            let a = base + 2 * r;
            idx.extend_from_slice(&[a, a + 1, a + 2, a + 2, a + 1, a + 3]);
        }
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

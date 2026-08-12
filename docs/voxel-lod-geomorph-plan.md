# Voxel LOD geomorph + analytic grounding — implementation plan

**Goal:** kill terrain LOD **popping** and trees **submerging** on the default voxel path,
TAA-free (TAA was stripped in `bab12ea`) and bake-free (no Nanite DAG).

## Status — IMPLEMENTED 2026-06-25 (needs USER visual-verify)
- **Piece 1 (geomorph) DONE** — Steps 0–3 + 5. 6th `morph` vertex stream through the rhi
  streaming pool + `create_graphics_pipeline_morph`; `mesh_leaf` bakes radial morph_disp;
  `terrain_csm.wgsl` lerps by camera distance with both band bounds keyed off `split_px`
  (children appear fully-morphed-to-parent at a split → continuous). GUI **LOD morph region**
  slider. Tests: `morph_targets_parent_surface`, `terrain_csm_wgsl_validates`; all configs build.
- **Step 4 (depth-caster morph) DEFERRED** — caster casts un-morphed → small transient
  self-shadow offset *during* a transition only. Add if shadow shimmer is visible.
- **Non-shadow fallback + normal morph DEFERRED** — shadows-off path doesn't morph (non-default);
  normals aren't morphed (shading may shift slightly at transitions). Add if visible.
- **Piece 2 (grounding) SKIPPED** — Piece 1 + the existing 65 m collider re-ground keep trees on
  the now-smooth surface (no submersion). Residual: a small hop when a *near* leaf refines (the
  collider is un-morphed). If visible, ground on the LOD-stable analytic surface — but the
  collider exists *because* the analytic floated, so validate that match first.

## Root cause (one fact, two symptoms)

The transvoxel isosurface position is **LOD-dependent**: each quadtree level samples a
different lattice, so coarse vs fine cross `density = surface_radius(p̂) − |p| = 0` at
**different world radii** (~meters near the player, tens of m at coarse levels). Transition
cells (`mesh_leaf(..., transitions)`) heal *cracks* (topology), never *position* (geometry).

- **Popping** = that surface jumping on a hard LOD swap (parent retired same frame the split fires).
- **Submersion** = near trees re-ground to the drawn mesh every frame (`main.rs:1627-1643`,
  `TREE_REGROUND_RADIUS_M = 65 m`) and therefore **pop with it**; far trees keep a stale Y.

## Approach: CDLOD radial geomorph (exploit that the terrain is analytic)

Geomorph is "hard for transvoxel" in the literature only because general marching-cubes has
no vertex correspondence across LODs, so you must **bake** a coarse mesh to morph toward. minos
doesn't: the surface is `surface_radius(hf, p̂, level)` (`minos-voxel/src/lib.rs:117`), an
analytic function. The morph target is a **function call at the parent level**, not a bake —
which is exactly why the DAG dealbreaker doesn't apply here.

Each fine vertex gets a precomputed **morph displacement** toward where the *parent* LOD puts
the surface. The vertex shader lerps it in by a factor driven by **per-vertex camera distance**
mapped onto the existing split/merge hysteresis band. At the instant a leaf merges, it is
already `==` its parent → the swap is invisible. **No dither, no TAA, no 2× draw, no dual
residency, no shadow self-intersection** (single mesh, morphed in place). And because the near
re-ground now rides a *continuous* surface, near-tree submersion is fixed by the same change.

This is CDLOD (Strugar 2010) applied radially.

## The one piece of real plumbing

The morph needs one per-vertex scalar/vector (`Δr` toward the parent surface) that **cannot be
computed on the GPU** (the `HeightField` is CPU-baked; minos-rhi can't upload it — see
`minos-rhi-no-texture-upload`). So it must ride as a **5th vertex stream**. `GraphicsPipelineDesc`
currently has no vertex-layout field (`voxel_view.rs:262,315`) — minos-rhi hardcodes 4×vec3 in
`minos-rhi/src/pipeline.rs` (shared by 6 pipelines). Extending it is the CLAUDE-flagged "5th
channel" chore; doing it **also unblocks data modes 7–10** on the voxel/classic paths.

Per-vertex data ⇒ a vertex stream is the right mechanism (clean per-leaf `bind_vertex_buffers`).
A storage-buffer-by-`vertex_index` (ocean's wind pattern) would dodge the layout change but force
a per-leaf descriptor rewrite mid-frame — worse. Take the vertex stream.

---

## Piece 1 — Geomorph (the main fix)

### Step 0 — minos-rhi: optional 5th vertex attribute (enabler)
- `minos-rhi/src/pipeline.rs`: add an optional field to `GraphicsPipelineDesc`
  (e.g. `extra_vec3_attr: bool` or `vertex_attrs: u8`, default = current 4×vec3). When set,
  append a `vec3<f32> @location(4)` binding (slot 4). Leaves the other 5 pipelines untouched.
- Confirm `bind_vertex_buffers` accepts a 5-handle slice (flora already uses
  `bind_vertex_buffers_slice` with 6 streams — the path exists).
- **Test:** existing `minos-rhi` pipeline tests compile; add a tiny assert that a 5-attr desc builds.

### Step 1 — minos-voxel: bake the morph displacement
- `minos-voxel/src/lib.rs` `mesh_leaf` (`:262`, already has `hf: &dyn HeightField`): for each
  output vertex at world `v` (local frame, origin = leaf origin), with `dir = (v+origin).normalize()`:
  - `r_parent = surface_radius(hf, dir, radius, height_scale, level.saturating_sub(1))` (`:117`).
  - `morph_disp = dir * r_parent − (v_world)` expressed in the **same local frame as `positions`**
    (i.e. `dir*r_parent − origin − v_local`), stored f32. Radial by construction; captures the
    dominant pop. (Lateral tessellation diff is second-order — start radial-only.)
  - At `level == 0` (roots) `morph_disp = 0`.
- Add `morph_disp: Vec<[f32;3]>` to `ChunkMeshArrays` (parallel to `positions`).
- **Test (golden-style):** for a `SimpleHeightField`, assert a fully-morphed vertex
  (`v_local + morph_disp`) lies within ε of `surface_radius(..., level-1)` along its `dir`
  (mirrors the existing `varying terrain` accuracy tests at `:475`). Existing crack-free /
  determinism tests must still pass (morph_disp is additive, doesn't touch positions/indices).

### Step 2 — voxel_view: upload + bind the stream, pass the morph band
- Add a `morph` buffer to the streamed mesh (alongside `pos/nrm/col/plate`); upload it in
  `upload_arrays` (`:581`).
- Build `csm_pipeline` (`:288`) and `shadow_pipeline` (`:260`) with the Step-0 5-attr flag.
- `record` (`:830`): bind `[pos,nrm,col,plate,morph]` (`:897/911`).
- Morph band per level → the CSM UBO. Add to `TerrainCsmFrame` (`:59`) the few scalars the
  shader needs to turn `dbg_level` + camera distance into a `[start,end]` morph range:
  `screen_h`, `tan_half_fov`, `split_thresh`, `merge_thresh`, `radius` (or precompute a
  `morph_band: [vec2; MAX_LEVELS]` array on the CPU each frame and index by `dbg_level` —
  simpler, no shader math). `dbg_level` is already in `ChunkPush` → **no ChunkPush change**.

### Step 3 — shaders: morph the position
- `terrain_csm.wgsl` `vs_main` (`:59`): add `@location(4) morph_disp: vec3<f32>`.
  ```
  let world_un = pc.model * vec4(v.position, 1.0);
  let dist     = length(world_un.xyz);              // camera-relative ⇒ |.| = camera distance
  let band     = frame.morph_band[pc.dbg_level];    // [start(near,0) , end(far,1)]
  let morph    = smoothstep(band.x, band.y, dist);
  let m3       = mat3x3(pc.model[0].xyz, pc.model[1].xyz, pc.model[2].xyz);
  let world    = vec4(world_un.xyz + morph * (m3 * morph_disp), 1.0);
  out.clip_pos = frame.view_proj * world;
  out.world_pos = world.xyz;                         // keep CSM/shading in sync
  ```
  Band endpoints (CPU): `morph = 1` at the distance where the **parent** `proj_px = merge_thresh`
  (leaf about to merge); `morph = 0` where the leaf's own `proj_px = split_thresh`. Convert with
  `dist = node_size·screen_h / (2·proj_px·tan(fov/2))` (`lod.rs::compute_proj_px`, `:408`).
- `minos-render/src/shaders/terrain.wgsl` (no-shadow fallback): same edit, OR skip — it's only
  used when `!has_shadow_map()` (non-default). Lazy: skip first, port if anyone runs shadows-off.
- **Crack-freeness:** morph is a pure function of per-vertex distance + per-vertex disp, so shared
  edge vertices (identical world pos) morph identically → no new cracks. Verify L↔L-1 transition
  edges by eye in mode 4 (Cluster) during a zoom.

### Step 4 — depth caster parity (follow-up, not blocking)
- `VOXEL_DEPTH_WGSL` (`:41`) currently reads only `@location(0)`. To keep self-shadows aligned
  during a transition, morph it identically (needs `morph_disp` @loc4 + the band + camera-relative
  model to get `dist`). **Defer:** the offset is small and transient; ship Steps 1–3 first, add
  this if shadow shimmer is visible during zoom.

### Step 5 — GUI knob (per `tunable-params-prefer-ui`)
- Expose the morph **start fraction** within the hysteresis band (how early morph begins) as a
  Voxel-section slider. Default ~0.5; lets you trade morph smoothness vs detail-retention live.

### Optional polish — normal morph
- If shading (not silhouette) still visibly pops, bake a `morph_nrm` too and
  `n = normalize(mix(n_fine, n_coarse, morph))`. Another vec3 stream; only if needed.

---

## Piece 2 — Grounding consistency (secondary; mostly fixed by Piece 1)

Piece 1 already fixes the dominant submersion (near re-ground rides a continuous surface). The
rest is making the *reference* stable:

- **Lean on analytic `ground_radius`** (`terrain_grid.rs:44`, level 12) — it's LOD-independent
  and, by design (`GROUND_LEVEL = 12`), tracks the finest drawn surface. Scatter already uses it.
- **Narrow the per-frame mesh re-ground** (`main.rs:1627-1643`): its job is no longer chasing LOD
  pops (gone) — only matching **edits/digs** (which change density, not `surface_radius`) and exact
  transvoxel interpolation. Keep it for edited regions; elsewhere analytic is enough. Optionally
  drop the raycast where no edit overlaps (less code, no residency dependence).
- **Async gap:** when no leaf is resident, hold last Y (don't snap to a mismatched fallback).

---

## Caveats (honest)
- **Caves/overhangs** (non-radial density) have no single radius → no cheap radial morph. They're
  WIP/interior: accept their pop, or dither *just* cave leaves later (dither stays in the back pocket).
- **Edits/digs** aren't in `surface_radius` → analytic grounding can't see a hole; keep the mesh
  raycast there (Piece 2).
- **Morph band tuning** within the hysteresis range is the calibration knob — too late → a flash of
  un-morphed geometry at the swap; the GUI slider (Step 5) is for dialing it.

## Sequencing (smallest shippable first)
1. Step 0 (minos-rhi 5th attr) — enabler, isolated, also unblocks modes 7–10.
2. Steps 1–3 (bake → upload/bind → shader morph) on the **CSM path only** — this is the visible win.
3. Step 5 GUI knob; eyeball with mode 4/5 during a fly-in.
4. Piece 2 grounding narrowing.
5. Step 4 caster parity + optional normal morph — only if artifacts remain.

## Tests
- `cargo test -p minos-voxel` — morph-target accuracy (new) + existing crack-free/determinism.
- `cargo test -p minos-app` / `-p minos-rhi` — naga validates the edited WGSL; pipeline builds.
- A CPU unit test for the morph-band math: `morph→1` at merge distance, `→0` at split distance.
- **Visual is USER-verified** (headless can't see pixels): fly-in screenshot, watch a tree at the
  geometry/impostor edge and a ridgeline during LOD changes.

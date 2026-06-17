# enki — Nanite-Style Virtualized Terrain Plan

> Status: **proposed, decisions locked** (planning only — no code yet). Sibling to
> `docs/foundation-plan.md`.

## Locked decisions (2026-06-15)

| Decision | Choice | Notes |
|---|---|---|
| **Target** | Virtualize the **terrain** (not static props) | The harder, more novel fork; chosen deliberately over the static-mesh recommendation. |
| **v1 scope** | **N0–N2** (cluster-LOD only) | Offline cluster DAG + GPU compute LOD-cut + one GPU-driven indirect draw. Occlusion (N3), vis-buffer/software-raster (N4), streaming (N5) are roadmapped, not built. |
| **Bake target** | One **mid-level cube-face patch**, ~**1–2M** base triangles | Region `(face, level, ix, iy)` + finest resolution are **config params**; this is the default. Bakes in seconds; ~10–15 LOD levels. |
| **Grouping / partition** | **METIS from the start** (`metis` crate, FFI) | What Nanite itself uses; Apache-2.0 (commercial-friendly). Isolated behind a thin module so it stays testable. |
| **Simplify / cluster lib** | **`meshopt` 0.6.2** | Verified to expose every needed API (below). |
| **Bake timing** | **At startup, in-memory** | No on-disk asset format in v1. N5 streaming will define the page-structured on-disk format. |
| **Rasterization** | **Hardware only** | Software raster + 64-bit vis-buffer deferred to N4 (blocked on naga u64 atomics). |
| **Mesh shaders** | **No** | naga/WGSL can't target task/mesh stages; compute-cull + indirect-draw is the portable path. |
| **Modularity** | Self-contained, **feature-gated `enki-nanite` crate** | Plug-and-play: a deletable folder; per-mesh opt-in; lower crates (`enki-rhi`/`-render`/`-planet`) never reference it. See **Modularity** below. |

## Goal & honest scope of v1

Replace, *for a bounded region of terrain*, the CPU cube-sphere quadtree mesh path with a
GPU-driven cluster-LOD representation in the spirit of Nanite: a pre-built directed-acyclic
graph (DAG) of ~128-triangle clusters across many LOD levels, from which the GPU selects a
crack-free, view-dependent **cut** each frame and rasterizes only the surviving clusters through
a single indirect draw.

**What v1 is NOT:** with no streaming, the cluster DAG must be fully resident, so v1 targets
**one configurable cube-face patch** baked at a chosen finest resolution — high enough to show
real micro-LOD working. A whole planet at meter-scale detail is a streaming problem (N5). The
existing quadtree keeps serving the rest of the planet; v1 renders the Nanite patch as a new
`RenderMode::NaniteTerrain` we can A/B against it.

**Procedural/static tension (resolved):** Nanite assumes static, finite, watertight meshes; our
terrain is procedural/continuous. v1 **bakes** the heightfield region to a fixed finest mesh and
virtualizes *that*. No runtime procedural cluster generation in scope.

## Verified facts (capability audit + dependency check)

- **RHI gap:** `enki-rhi` is graphics-only — Vulkan 1.3 with `dynamicRendering`,
  `synchronization2`, `timelineSemaphore`, `fillModeNonSolid`; one graphics+present queue; one
  `UNIFORM_BUFFER` descriptor; 128-byte push constants; fixed 4×`vec3` vertex bindings.
  No compute, storage buffers, indirect, atomics, descriptor indexing, or BDA.
- **v1 needs NO new Vulkan device features.** `STORAGE_BUFFER` is core; vertex-stage SSBO reads
  need no feature; compute atomics need no feature; a single `vkCmdDrawIndirect` (`drawCount=1`,
  GPU-written `vertexCount`) needs no feature. Compute runs on the existing graphics queue (a
  graphics queue is always compute-capable). **N1 is plumbing, not feature-hunting.**
- **`meshopt` 0.6.2 exposes everything N0 needs** — `clusterize::build_meshlets` (max 64 vert /
  128 tri), `simplify::simplify_with_attributes_and_locks(... vertex_lock: &[bool] ...)` (the
  crack-free primitive — set `true` on shared group-seam verts), `simplify::simplify_scale`
  (+ `SimplifyOptions::ErrorAbsolute`), `clusterize::compute_cluster_bounds` (sphere + normal
  cone). `partition_clusters` also exists but we use **METIS** instead.
- **naga 29.0.3** (the version in `Cargo.lock`) supports the full v1 WGSL surface: `@compute`,
  `var<storage, read_write>`, `atomic<u32>` add/max, storage reads from `@vertex`,
  `@builtin(vertex_index)`. (Only u64 atomics are missing — deferred to N4.)
- **Crate dependency graph:** `enki-rhi → enki-render → enki-planet → enki-jobs → enki-app`
  (low→high). The runtime pass needs `enki-rhi` (lowest) and the baker needs `HeightField`
  (`enki-planet`, high). A cycle only arises *if a lower crate consumes cluster types* — nothing
  forces that, so **all Nanite code lives in one self-contained crate above enki-planet**, with
  `enki-app` its only consumer (see **Crates** and **Modularity**).

## Crates & placement (single self-contained, deletable crate)

**All Nanite code lives in one new `enki-nanite` crate** — baker, data types, runtime GPU pass,
and WGSL shaders — sitting above `enki-planet`, depending only **downward**, consumed only by
`enki-app`. This is what makes it a deletable, plug-and-play folder (see **Modularity**).

- **`enki-nanite`** *(new)* — three internal modules, one crate:
  - `cluster` — data-only types (frozen contract): `Cluster`, `ClusterBounds` (sphere + normal
    cone), `ClusterDag` (parent links, group ids, self/parent error, lod), `ClusterAsset` (the
    resident bundle + patch origin). No GPU, no `ash`.
  - `bake` — tessellate the patch from `HeightField` + build the DAG (meshopt + METIS), emitting
    `cluster` types. CPU-only, no GPU — fully unit-testable headless (à la
    `enki-planet/tests/determinism.rs`).
  - `render` — the runtime GPU pass (`NaniteRenderer` facade + cull/draw + WGSL), using the RHI.
  - Deps: `enki-rhi` (GPU), `enki-render` (reuse `FrameUniforms`, lighting/ACES), `enki-planet`
    (`HeightField`, `face_bases`/`cube_to_sphere`), `meshopt`, `metis`, `glam`. **Nothing in
    `enki-rhi`/`enki-render`/`enki-planet` ever names `enki-nanite`.**
- **`enki-rhi`** — gains *generic, un-branded* primitives usable with or without Nanite: compute
  pipelines, storage buffers, one-shot device-local upload, a storage descriptor set, indirect
  draw + barriers, a zero-vertex-binding pipeline variant. These stay if Nanite is deleted.
- **`enki-app`** — under `#[cfg(feature = "nanite")]`: picks the bake region, calls the baker at
  startup, registers the patch with `NaniteRenderer`, calls `record()` in the frame loop, adds
  `RenderMode::NaniteTerrain` + HUD counters. This is the *only* crate that references
  `enki-nanite`, and only behind the feature.

## Modularity & plug-and-play

Two requirements drive this: (a) deleting the whole Nanite folder must leave a working engine;
(b) Nanite must apply to *some* meshes while the rest use the classic path, in the same frame.

**Containment rules (enforced, not aspirational):**
- One crate (`enki-nanite`) holds 100% of Nanite logic, types, and shaders. Lower crates are
  forbidden from referencing it (a grep for `nanite` in `enki-rhi`/`enki-render`/`enki-planet`
  must return nothing — an N2 acceptance check).
- RHI additions are **generic and un-prefixed** (`create_compute_pipeline`, not
  `create_nanite_pipeline`), so they have independent value and never become orphaned on deletion.
- The baker depends on the existing `HeightField` **trait**, so it plugs onto any height source
  without coupling to terrain internals.

**The app seam (single public surface):** `enki-nanite` exposes one facade —
`NaniteRenderer::new(rhi)`, `register(asset, transform) -> NaniteInstanceId`,
`update(camera, &FrameUniforms)`, `record(rhi, frame_index)`. Everything else is crate-internal.
`enki-app` holds `#[cfg(feature = "nanite")] Option<NaniteRenderer>` and calls `record()` inside
the existing MSAA instance alongside the classic terrain/water passes. Because both paths write
the shared **reversed-Z depth + MSAA color**, they composite by depth automatically — record
order doesn't affect correctness. *(Optional later polish: a generic `ScenePass` trait in
`enki-render` so the frame loop iterates passes uniformly; not required for v1.)*

**Per-mesh opt-in:** `NaniteRenderer` renders only the `(ClusterAsset, transform)` instances the
app registers; everything unregistered flows through the classic path. The two are independent
lists — mixing is the default, not a special case.

**Feature flag:** a `nanite` Cargo feature gates the optional `enki-app → enki-nanite` dependency
and all wiring. Off ⇒ the crate isn't linked and the classic engine builds/runs unchanged.

**Deletion checklist (what removing Nanite actually touches — nothing else):**
1. delete the `enki-nanite/` folder;
2. remove `"enki-nanite"` from `[workspace].members` in the root `Cargo.toml`;
3. remove the optional dep + `nanite` feature from `enki-app/Cargo.toml`;
4. delete the `#[cfg(feature = "nanite")]` blocks in `enki-app` (renderer wiring + the
   `RenderMode::NaniteTerrain` arm).
The generic RHI primitives from N1 stay (they're not Nanite-specific). Workspace compiles clean.

**Coexistence caveat (honest):** disjoint Nanite regions / separate props composite cleanly via
depth. Mixing a Nanite-virtualized terrain patch **adjacent** to a classic quadtree patch can
crack at the shared seam (two different LOD schemes meeting) — out of scope for v1's single
patch; if mixed adjacent terrain is wanted later, keep the boundary on a locked, matched edge or
let one scheme own each contiguous region.

## Key technical design

- **Cluster size:** 128 triangles / 64 vertices.
- **Source mesh:** reuse cube-to-sphere + `HeightField` to tessellate the bake patch at the
  finest resolution into ONE watertight indexed mesh. **No skirts** — locked boundaries replace
  them. The interior grid already shares vertices (watertight); the **4 outer patch-boundary
  edges** are flagged and passed as `vertex_lock = true` at *every* simplify level so the patch
  can later tile seamlessly across neighbors and cube-face edges. (Trade-off: a fully-locked
  border means the coarsest root keeps full border-edge resolution — correct for tiling, fine
  for a single-patch demo.)
- **Baked attributes:** position (patch-origin-relative f32), normal, color (drop `plate_colors`).
  Use `simplify_with_attributes_and_locks` with normal+color weights so attributes don't swim.
- **DAG build invariants (the make-or-break):**
  1. group clusters (~8–32) via **METIS** k-way on the cluster adjacency graph (edge weight =
     shared boundary edges); **re-partition each level** so old boundaries become decimatable;
  2. lock each group's boundary verts (`vertex_lock`) during simplify to 0.5×;
  3. **monotonic error:** `parentError = simplifyError·simplifyScale + max(childError)`;
  4. **nested bounds:** compute each group's bounding sphere *manually to enclose all child
     spheres* (do **not** trust `compute_cluster_bounds` across a merged group);
  5. **group-shared bounds/error:** every cluster in a group shares one sphere + one error, so a
     cut can only transition at group boundaries — never inside one.
- **LOD-cut test (runtime, per cluster):** draw iff `proj(selfError) ≤ τ && proj(parentError) > τ`,
  τ ≈ 1 px. `px = screenH · cot(fovY/2) · r / sqrt(d² − r²)`; clamp to "force finest" when the
  camera is inside the sphere (`r ≥ d`) to avoid NaNs.
- **Precision:** cluster vertex positions and bounding-sphere centers are patch-origin-relative
  f32; the per-patch **camera-relative translation** is recomputed in f64 each frame on the CPU
  and uploaded (mirrors `ChunkPush::camera_relative`). Consistent with the reversed-Z f32 clip
  convention.
- **Draw packing (Bevy-style):** the cull pass appends survivors and writes one
  `VkDrawIndirectCommand { vertexCount = visibleClusters·128·3, instanceCount = 1 }`; the draw is
  one **non-indexed** `cmd_draw_indirect`. The WGSL vertex shader pulls geometry from the cluster
  SSBO via `gl_VertexIndex` → (visible-slot, triangle, corner) → cluster id → vertex.

## Milestones (risk-first; Stream A offline + Stream B RHI → converge at N2)

Per milestone: freeze interface contracts → spawn ≤5 file-disjoint engineer agents in parallel →
reviewer checks acceptance → fix-loop → advance.

### N0 — Cluster DAG bake  *(Stream A · CPU/Rust · no GPU)*
The hardest correctness problem, needs no device — goes first.
- **N0.0 (gating spike):** get the **`metis` crate building/linking on Windows** (vendored
  cmake build or prebuilt METIS) — this is the one friction point of choosing METIS-from-start;
  prove it compiles and round-trips a trivial `PartGraphKway` before the algorithm work.
- Add `meshopt`, `metis` deps. New `enki-nanite` crate; new `enki-render::cluster` types.
- High-res patch tessellator: `(face, level, ix, iy) + finest_res` → watertight indexed mesh from
  `HeightField`, with the 4 outer-edge vertices flagged for locking.
- Build pipeline: `build_meshlets` (128/64) → cluster adjacency graph → **METIS** group partition
  → lock group + patch boundary verts → `simplify_with_attributes_and_locks` to 0.5× →
  `compute_cluster_bounds` + manual nested group sphere → monotonic error via `simplify_scale` →
  re-`build_meshlets` → record DAG links → recurse to ≤ few roots / cap ~25 levels.
- **Acceptance (headless tests):** error monotonic across every DAG edge; parent bounds enclose
  children; group-boundary verts bit-identical across levels (crack-free proof); cluster count
  converges; bake is deterministic for a fixed `(region, res)`.

### N1 — RHI: compute + storage + indirect  *(Stream B)*  — pure plumbing, no new device features
- `pipeline.rs`: `ComputePipelineDesc` + `create_compute_pipeline`; allow a **zero-vertex-binding**
  graphics pipeline (vertex pulling has no vertex buffers).
- `shader.rs`: confirm naga compiles a WGSL `@compute` entry point.
- `buffer.rs`: a **one-shot device-local upload** helper (`STORAGE_BUFFER | TRANSFER_DST`; the
  indirect-arg buffer adds `INDIRECT_BUFFER`) — resident data, not the per-frame streamer.
- `descriptor.rs`: a storage-buffer set/layout visible to COMPUTE + VERTEX (cluster data,
  per-patch data, visible-list, indirect args). Fixed multi-binding — **no bindless in v1**.
- `lib.rs`/`command.rs`: `cmd_dispatch`; `cmd_draw_indirect`; the compute→indirect/vertex barrier
  (`SHADER_WRITE → INDIRECT_COMMAND_READ | SHADER_READ`); per-frame `cmd_fill_buffer` to zero the
  visible counter / indirect-arg before the cull dispatch.
- **Acceptance:** trivial compute fills an SSBO (verified by readback); a GPU-written single
  `cmd_draw_indirect` renders a triangle; validation-clean; zero leaks.

### N2 — Runtime cluster renderer  *(A + B converge)*
- Bake at startup; upload `ClusterAsset` to resident SSBOs once. Per-patch uniform: camera-relative
  f64→f32 translation + LOD params (τ, screen height, `cot(fovY/2)`) + the 6 frustum planes
  (CPU-extracted via the existing `Frustum::from_view_proj`).
- **Cull/select compute pass:** one thread per cluster → project self/parent error spheres, apply
  the two-sided cut test + frustum cull → `atomicAdd` survivor into the visible-list SSBO and
  update `VkDrawIndirectCommand.vertexCount`.
- **Draw pass:** one non-indexed `cmd_draw_indirect`; WGSL vertex shader pulls pos/nrm/col from
  the cluster SSBO and transforms by `view_proj · camera_relative_patch`; fragment reuses the
  existing ACES/lighting; renders into the existing 4× MSAA forward instance.
- **App integration (live, in-engine — the primary debug view):** bake the patch at startup;
  add a feature-gated Nanite render path in `enki-app` (`#[cfg(feature = "nanite")]`) selectable
  at runtime, and a **"Nanite" section in the existing egui panel** (`gui.rs`) alongside the
  material/wireframe controls: an **enable** checkbox + a **debug-mode selector**
  (Off / Triangle / Cluster / LOD) + live counters (visible clusters / triangles / draw count).
  The selected mode flows into the pass as a push constant.
- **Debug viz** — flat-color modes like Unreal's Nanite views: **Triangle** (every triangle a
  distinct color → micro-poly density), **Cluster** (every cluster distinct → meshlet structure),
  **LOD** (color by DAG level → see the cut). The fragment color is `id_color(hash)` keyed by the
  `(cluster_id, triangle_id)` the vertex-pulling shader already computes, so it's nearly free. The
  WGSL `id_color` mirrors `enki_nanite::debug::id_color`, so the live view matches the offline PLY
  exporter (`debug::export_ply`, shipped in N0 as a dev tool:
  `cargo run -p enki-nanite --example export_debug`).
- **Acceptance:** crack-free LOD as the camera approaches/recedes; visible-cluster and triangle
  counts adapt with distance; draws collapse to one indirect; visual parity with the quadtree at
  matched detail; validation-clean; stable frame time.
- **Modularity acceptance:** building with `--no-default-features` (feature `nanite` off) compiles
  and runs the classic engine unchanged; `enki-rhi`/`enki-render`/`enki-planet` contain zero
  references to `nanite` (grep check); the deletion checklist leaves a clean-compiling workspace.

## OUT of v1 (deferred roadmap)

- **N3 — Occlusion culling (two-pass HZB):** depth-pyramid min-reduction; main pass vs last-frame
  HZB → rebuild → post pass re-tests the occluded set. Frame of feedback latency; biggest
  dense-scene win.
- **N4 — Visibility buffer + software rasterizer + deferred materials:** u64 SSBO vis-buffer with
  `atomicMax(depth|clusterId|triId)`, compute scanline raster for micro-triangles, deferred
  attribute resolve. **Blocked by naga** (no u64 atomics → hand-written SPIR-V or a naga patch).
- **N5 — Geometry streaming (whole planet):** fixed-size GPU page pool (one persistent
  DEVICE_LOCAL buffer, 128 KB slots) fed by the existing staging ring; GPU feedback request
  buffer + CPU readback; eviction guarded by the **existing timeline-semaphore graveyard**
  (`retire_frame = frame_counter + 1`, free at `gpu_completed_frame`). Defines the on-disk
  page-structured asset format. The quadtree becomes the coarse residency driver; per-patch DAGs
  drive fine LOD. This is what turns v1's single resident patch into a continuous planet.

## Top risks & mitigations

1. **DAG crack-free correctness** — dominant risk; pure CPU, so N0 retires it first behind
   headless invariant tests before any GPU work.
2. **METIS on Windows** — the one build-friction cost of METIS-from-start; de-risked by the
   N0.0 gating spike before algorithm work.
3. **GPU-written indirect args** need exact compute→`INDIRECT_COMMAND_READ` barriers; a missing
   barrier is intermittent corruption. Enable GPU-assisted validation throughout.
4. **Brute-force cull cost** — v1 scans all clusters (fine for one patch); add a cluster BVH +
   persistent-threads queue only if profiling shows the scan dominates.
5. **naga coverage** — all v1 WGSL features verified on 29.0.3; only u64 atomics (N4) are a gap.

## Primary references

- Karis et al., "A Deep Dive into Nanite Virtualized Geometry," SIGGRAPH 2021.
- jglrxavpok, "Recreating Nanite" series.
- jms55 (Bevy), "Virtual Geometry in Bevy 0.14 / 0.16" (METIS switch, hardware-raster subset).
- zeux/meshoptimizer + `meshopt` 0.6.2 Rust crate; METIS (Karypis lab) + `metis` Rust crate.
- elopezr, "A Macro View of Nanite"; thecandidstartup, "Nanite Graphics Pipeline."

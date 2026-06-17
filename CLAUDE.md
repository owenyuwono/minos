# enki — agent guide

enki is a Rust + Vulkan (via `ash`) renderer for a **procedural cube-sphere planet**, with a
custom RHI and a Nanite-style virtualized-geometry path for the terrain.

## Build / run

`cargo` is **not on PATH**. In PowerShell prefix with:
```
$env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
```
- Run: `cargo run -p enki-app`
- Check/build: `cargo check -p enki-app` (use `check`, not `build`, while the app is running — the `.exe` is locked).
- Classic engine (no Nanite): `cargo build -p enki-app --no-default-features`.
- Vendored C deps: `metis` builds clean on MSVC (no cmake/libclang needed).

Both feature configs (default = `nanite` on, and `--no-default-features`) must always compile.

## Workspace crates

- **enki-rhi** — Vulkan RHI: frame bracket (`begin_frame`/`begin_rendering`/`begin_ui_pass`/`end_frame`),
  pipelines, buffers, descriptors, images, swapchain, and the TAA pass (`taa_pass.rs`).
- **enki-render** — camera, `FrameUniforms`, lights, projection, TAA jitter + resolve (`taa.rs`).
- **enki-planet** — heightfield (tectonic + simple), cube-sphere bases, quadtree LOD (`lod.rs`), mesher, climate/biome coloring.
- **enki-jobs** — background worker pool for chunk meshing.
- **enki-app** — winit 0.30 app shell, egui panel (`gui.rs`), HUD, `PlanetView` (quadtree grid terrain), async startup loader (`loading.rs`), and the Nanite integration.
- **enki-nanite** — Nanite-style virtualized geometry. **Self-contained, feature-gated, deletable** (see below).

## Core conventions

- **Camera-relative precision:** CPU is f64 (`glam::DVec3`); GPU is f32. The view is rotation-only with the camera at the origin each frame; per-object/per-node translations are f64-subtracted (`origin - camera`) then cast to f32. This keeps planet-scale (radius ~50 km, but coords up to ~6.37e6 in tests) precise.
- **Reversed-Z depth:** `D32_SFLOAT`, compare `GREATER`, clear `0.0`, far plane = depth 0. Sky/cleared depth is 0.
- **WGSL via naga 29:** push constants use `var<immediate>` (NOT `var<push_constant>`); `meta` is a reserved keyword (don't name a binding `meta`). Shaders are **validated by naga inside unit tests** (`cargo test` catches shader errors headlessly — e.g. `shaders_parse_and_validate`).

## enki-nanite (virtualized terrain)

**Modularity contract:** ALL Nanite code/types/shaders live in this crate. Lower crates (`enki-rhi`/`-render`/`-planet`) never reference it; `enki-app` is the only consumer, behind `#[cfg(feature = "nanite")]`. Removing Nanite = delete the folder + drop the optional dep/feature + the cfg blocks in `enki-app`.

**Architecture (the "true Nanite" path):**
1. **Bake (offline, at startup):** `bake::tessellate` a heightfield patch (no skirts; analytic face-independent normals) → `cluster_build` meshlets (≤64 verts / ≤128 tris) → `bake::dag` builds a DAG of LOD levels (meshopt simplify + METIS grouping; group **boundaries locked** so the cut is crack-free; monotonic error). `bake_planet` bakes **6 deep per-face DAGs** (res 1024), each with its own f64 `patch_origin`. Held in RAM.
2. **Residency (CPU, `residency.rs`):** `ClusterResidency::select` picks the clusters that should be GPU-resident for the camera — a *superset* of the GPU cut, widened by a margin, **nearest-first** sorted. `ClusterStreamer` wraps it with camera-delta throttling.
3. **Page pool (`render.rs`):** the resident set streams into a **shared, fixed-stride, slot-indexed** GPU pool (`PagePool` from `stream.rs`, timeline-graveyard reuse). `update_residency` uploads only *added* clusters and frees *removed* ones (O(delta), no whole-set re-pack) — this is what makes zoom smooth. Each cluster owns a stable slot; geometry is only ever written to FREE slots (never mid-flight). A per-frame `active_slots` buffer lists the live slots.
4. **GPU cull (`nanite_cull.wgsl`):** one thread per active slot; frustum-cull + the two-sided screen-error cut (`self_px ≤ tau && parent_px > tau`). Appends slots to a `visible` list + advances an indirect draw arg.
5. **Draw (`nanite_draw.wgsl`):** one indirect, non-indexed, vertex-pulling draw; fixed-stride addressing (`slot*MAX_TRIS + tri`, tris store global vertex indices). Debug color modes: 0=lit (matches terrain lighting), 1=triangle, 2=cluster, 3=LOD — keyed on a **stable id** (`meta.range[3]`), not the buffer slot, so colors don't flicker on re-pack.
6. **Popping → dithered LOD cross-fade + TAA:** with TAA on, the cull widens to a transition band (`tau*FADE_HI`); the draw partitions pixels between overlapping LODs by an exact interval `[t_self, t_par)` (no holes/overlap) using a stable per-pixel dither, and TAA resolves the stipple into a smooth blend. Mirrors Unreal's dither+TAA. Gated on `taa_enabled` (off → strict hard cut).

**Gotchas:**
- Descriptor binding stages: the lit fragment reads the `frame` buffer (lights) → its binding needs the `FRAGMENT` stage flag, not just VERTEX/COMPUTE.
- Per-face origins (≤6) drive `node_xlat` (camera-relative translation per face), indexed by `meta.range[2]`.
- Nanite is **mutually exclusive** with the quadtree terrain (`PlanetView`) — when Nanite is active the quadtree isn't rendered.

## TAA

`enki-render/taa.rs` (Halton jitter, `ResolveParams`, camera-relative reprojection — the engine renders camera-at-origin, so the resolve bakes the camera-translation delta into the previous view-proj) + `shaders/taa_resolve.wgsl` (depth→world reproject, 3×3 neighborhood clamp, history blend) + `enki-rhi/taa_pass.rs` (MSAA target → resolved current+depth, 2 ping-pong history, resolve pipeline). Toggle in the egui View section, **default OFF**; the frame bracket branches on `taa_active()` so OFF is byte-identical to the non-TAA path. History is 8-bit (swapchain format, for sRGB linear-space blend without duplicating pipelines).

## Testing

- `cargo test -p enki-nanite` — bake invariants (crack-free, monotonic error), residency/page-pool logic, flatten, and **naga shader validation**. Headless.
- `cargo test -p enki-rhi` / `-p enki-render` — RHI buffer/descriptor + TAA jitter/resolve.

## Known open items / WIP

- TAA temporal accumulation should be confirmed on-GPU (jitter sign, reprojection under motion, layout/sync). The dither is currently *stable* (degrades to a gradual erosion without TAA); switch it to temporal once TAA accumulation is verified.
- Backface cluster culling (normal cones are baked into `ClusterAsset` but not yet plumbed to `GpuClusterMeta`/cull) + the ~30% degenerate-vertex draw waste.
- On-demand procedural baking for sub-meter close-up detail (current detail is capped by the startup bake resolution).
- Cube-edge seams are an inherent inflated-cube artifact (deprioritized).
</content>

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
- **enki-app** — winit 0.30 app shell, egui panel (`gui.rs`), HUD, `PlanetView` (quadtree grid terrain), async startup loader (`loading.rs`), the Nanite integration, and the third-person controller + animated `Character` (see below).
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
5. **Draw (`nanite_draw.wgsl`):** one indirect, non-indexed, vertex-pulling draw; fixed-stride addressing (`slot*MAX_TRIS + tri`, tris store global vertex indices). Per-vertex stride is **16 floats** (pos3+normal3+color3+material1+wetness1+volcanism1+elevation1+plate3) — must match `render.rs` `SLOT_FLOATS`. Surface color comes from the unified `view_mode` (see **View modes**); id-based modes key on a **stable id** (`meta.range[3]`), not the buffer slot, so they don't flicker on re-pack.
6. **Popping → dithered LOD cross-fade + TAA:** with TAA on, the cull widens to a transition band (`tau*FADE_HI`); the draw partitions pixels between overlapping LODs by an exact interval `[t_self, t_par)` (no holes/overlap) using a stable per-pixel dither, and TAA resolves the stipple into a smooth blend. Mirrors Unreal's dither+TAA. Gated on `taa_enabled` (off → strict hard cut).

**Gotchas:**
- Descriptor binding stages: the lit fragment reads the `frame` buffer (lights) → its binding needs the `FRAGMENT` stage flag, not just VERTEX/COMPUTE.
- Per-face origins (≤6) drive `node_xlat` (camera-relative translation per face), indexed by `meta.range[2]`.
- Nanite is **mutually exclusive** with the quadtree terrain (`PlanetView`) — when Nanite is active the quadtree isn't rendered.

## TAA

`enki-render/taa.rs` (Halton jitter, `ResolveParams`, camera-relative reprojection — the engine renders camera-at-origin, so the resolve bakes the camera-translation delta into the previous view-proj) + `shaders/taa_resolve.wgsl` (depth→world reproject, 3×3 neighborhood clamp, history blend) + `enki-rhi/taa_pass.rs` (MSAA target → resolved current+depth, 2 ping-pong history, resolve pipeline). Toggle in the egui View section — **now default ON** alongside Nanite (see View modes); the frame bracket branches on `taa_active()` so OFF is byte-identical to the non-TAA path. History is 8-bit (swapchain format, for sRGB linear-space blend without duplicating pipelines).

## View modes

A single **`view_mode` (0–10)** drives surface shading in BOTH render paths (push value: `pc.material_mode` in `terrain.wgsl`, `pc.mode` in `nanite_draw.wgsl`), shown in the egui View section as two grouped selectors:
- **View** (shading): 0 Lit · 1 Unlit · 2 Normal · 3 Triangle · 4 Cluster · 5 LOD (3–5 Nanite-only, greyed otherwise).
- **Planet** (data): 6 Plate · 7 Height · 8 Material (rock hardness) · 9 Wetness · 10 Volcano (arc/hotspot cones).

Data modes read **per-vertex channels** baked at tessellation (`material`/`wetness`/`volcanism`/`elevation`/`plate`), riding alongside `color` through the whole Nanite DAG (cluster→tessellate→dag→cluster_build→`render.rs` vbuf→WGSL pull). Adding a channel = bump the Nanite vertex stride (`render.rs SLOT_FLOATS` + the `base + k` indices in `nanite_draw.wgsl`) in lockstep.

- **Nanite (default renderer) = full parity** (all 11 modes). **Nanite + TAA are default-ON** (`main.rs` AppState).
- **Classic/quadtree = partial**: Lit/Unlit/Normal/Plate only; Height/Material/Wetness/Volcano fall back to Lit — **blocked** on the classic vertex layout being hardcoded `4×vec3` in `enki-rhi/src/pipeline.rs` (shared by 6 pipelines; a 5th data channel needs an optional `vec4` binding gated by a `GraphicsPipelineDesc` flag). Low priority since Nanite is the default.

## Ocean (FFT waves + foam + refraction)

A **unified refractive ocean** in `enki-app/src/ocean/`. With TAA on, the shell **and** waves render together in one refractive 1× pass (`WaveSurface`) sharing the exact shading, so they match in colour, depth-darkening, and refraction. With TAA off, the old `Ocean` (alpha-blended MSAA shell) is the fallback.
- **`WaveSurface`** — owns the smooth sea-level **shell sphere** (always) + a Tessendorf-style **multi-cascade spectral FFT** wave patch (drawn only below ~20 km altitude). One shader, two vertex entries (`vs_shell` / `vs_main`) feeding a shared fragment.
- **`Ocean`** — the smooth translucent shell, **TAA-off fallback only**.

**Sim (CPU, `ocean/{fft,spectrum,sim}.rs`):** JONSWAP×TMA×Donelan spectrum → `h0(k)` → time-evolve → inverse FFT → displacement (xyz, choppy) + slope normals + **Jacobian foam**. **3 cascades at poseidon's N=256, tiles [250,17,5] m** (→ cm-scale ripples). FFT is **`rustfft`** (the hand-rolled radix-2 only did N≤64). Too heavy for the render thread, so the sim runs on a **background worker** (`std::thread`) that double-buffers its latest field into a `Mutex<Vec<OceanTexel>>`; `record` uploads that to the GPU each frame and the worker idles when no waves are drawn. ponytail: GPU compute is still the eventual upgrade (enki-rhi has compute + storage buffers), but the worker keeps it 60 fps now.

**Wave mesh modes (`ocean/mesh.rs`, GUI "Ocean → Mesh", `enum OceanMesh`):** the wave geometry is selectable — **Warped** (A, default): one camera-anchored tangent patch radially warped so cells bunch under the camera; **Clipmap** (D): concentric power-of-two rings (dense center → coarse rim, crack-free via transition fans — unit-tested watertight), reuses `vs_main`; **Projected** (C): a screen-space lattice ray-cast onto the sea sphere each frame on the CPU (f64) into a per-FiF dynamic vertex buffer, drawn by `vs_projected` — fills the whole view (no patch, no shell). A/D meshes are tangent-space `[gx, cell, gz]` (`.y` = local cell size); all modes fade each cascade where the cell can't resolve it (Nyquist). `mesh.rs` holds the generators + tests (clipmap watertightness, ray/sphere projection).

**Surface (`shaders/ocean_surface.wgsl`):** a camera-anchored tangent-plane grid curved onto the sphere (sagitta), FFT-displaced; camera-relative + reversed-Z. Cascade UVs are **domain-warped** by a world-anchored noise to break tile repetition; a world-anchored **amplitude field** makes some regions rough/calm (also the "Ocean → Intensity" view-mode heat-map). Custom set 0 (frame UBO + field storage + ocean UBO + scene **color** + scene **depth**) bound via `cmd_bind_pipeline`/`cmd_bind_descriptor_set`. Shading = Fresnel sky reflection + sun glint + SSS + foam + **depth-based absorption** (deeper water column → darker, via the scene depth; the column is projected toward vertical so it tracks ocean *depth*, not view angle).

**Refraction + depth (TAA path only):** `rhi.begin_water_pass()` splits the 3D pass — closes the opaque MSAA instance (resolving to the TAA `current` image + `resolved_depth`), blits opaque color into a history scratch (refraction source) **and copies `resolved_depth`→`refract_depth`** (the depth source for darkening), then reopens a **1× instance on `current`** the water draws into (blend on, alpha 1 → opaque, depth-write off so the shell can't occlude wave troughs). Requires TAA on; with TAA off only the shell shows. Reuses TAA's `current`/history (no new color image) — `taa_resolve` runs unchanged.

**Gotchas:** the grid mesh and `placeholder_sphere` must be wound so cull-BACK keeps the **camera-facing** face. Graphics pipelines with no push constants pass `push_constant_size: 0`. Shell + wave share **one** ocean UBO (only `shell_model` differs, read solely by `vs_shell`) so it's written once. GUI: Ocean section (toggles + sea-level / choppiness / foam) + view-mode rows **View / Planet / Ocean** (Ocean 11 Surface, 12 Intensity).

## Third-person controller + character

A surface controller with an animated character. Lives in `enki-app` (`controls/third_person.rs` + `character.rs`); nothing below `enki-app` references it.

**Nav modes (`controls/nav_mode.rs`):** `Globe → Placement → Surface → Globe`. `Surface` (renamed from the old `FirstPerson`) hosts the `ThirdPersonController`; it has a **1st/3rd-person view toggle** (V), so a single mode/controller serves both. The standalone `FirstPersonController` is **no longer wired** but kept for its tests + the shared grounding helpers — don't delete it without moving those. Spawn = the existing `Tab → Placement → click` ray-cast.

**`ThirdPersonController`:** headless (no winit/GPU, fully unit-tested). Feet ride the terrain via `first_person::surface_radius` (the single source of the feet-on-ground invariant — both walkers call it so they can't drift from the bake). Movement is **camera-relative WASD**; the body yaws toward its travel direction. Mouse orbits the chase cam (`cam_dir`/`cam_pitch`), scroll sets the boom (`cam_dist`), and an **8-step occlusion march** pulls the boom in so the camera never clips terrain. `camera()` branches on the view: third = boom behind a shoulder anchor; first = at the eye (caller hides the mesh).

**`Character` (CPU skinning):** an ~11-bone box humanoid built procedurally (`build_rig`); a hierarchical gait solver (`solve_pose`, mirrors the flora wind-solver shape) swings legs/arms in anti-phase scaled by speed, with idle breathing. **Skinned on the CPU** (one bone/vertex) into a **per-frame-in-flight pos/nrm vertex-buffer ring** (`write_storage_bytes` each frame; mirrors flora's `bone_ubos` ring so writing this frame never races the GPU reading last frame's), then drawn with the **existing terrain pipeline** (`material_mode 0`) via `ChunkPush::camera_relative`. Rest pose at `speed==0 && phase==0` is an exact standing figure.
- ponytail: **CPU skin + terrain pipeline** — no skinning shader, no custom descriptor set, no extra vertex channel. One low-poly character is trivial to re-skin per frame; move to the GPU/flora skinning path only when there's a crowd. glTF import + 4-weight skinning is the other upgrade path (the rig generalizes to it).

**Gotchas:**
- Character verts are in **local character space** (feet at origin, +Y up, +Z facing); planet placement is the draw's model matrix `from_cols(right, up, forward, _)` with `right = up × forward` (det +1 → CCW winding preserved for cull-BACK). Boxes are wound CCW outward in `push_box`.
- Drawn **after terrain/Nanite, before the ocean**, inside the opaque MSAA instance; only when `nav == Surface` and the view is third-person. `Character::update` (skin + upload) runs *before* `begin_rendering` (it's a host-visible memcpy, fine either side); `draw` runs inside.
- `Character::new` (built at startup in Planet mode) clones the terrain pipeline desc byte-for-byte, so it inherits reversed-Z / MSAA / swapchain format with no special-casing.
- Controls: Tab → click to drop · WASD (camera-relative) · Shift sprint · mouse orbit · scroll zoom · **V** 1st/3rd · Esc → orbit.

## Terrain ↔ ki parity (determinism gate)

The landmass (tectonics + heightFn + rockHardness + cube-sphere) is a faithful port of **ki** ("Demiurge", a TS/WebGPU sibling at `../ki`). `enki-planet/tests/determinism.rs` gates it against `tests/goldens/golden.json`.
- **`golden.json` is dumped from ki itself** — `cd ../ki && npx -y tsx tools/dump-golden.mjs` (imports ki's modules at their native config; writes to enki's golden path). `noise`/`fbm`/`ridged`/`tectonics` are ki-sourced byte gates (1e-9); `height` is an enki self-snapshot (enki ships a B-path erosion the dump tool doesn't run) — regen after a *reviewed* height change via `cargo test -p enki-planet --test determinism -- --ignored regen_golden_height_samples`.
- **Tectonics bakes at `RES=512`** (`tectonics.rs`) to match ki seed-for-seed — the crack/arc bands are absolute radians, so a smaller RES bakes a *different* set of continents. Was 256; **do not revert.** enki matches ki on **28/30 tectonic samples byte-exact**; the gate carves out conv/shear at the exact poles (`degenerate_pole`) — gradient-derived (`t_hat = −∇dist/|∇dist|`) and irreducibly amplified across JS↔Rust float at the near-zero-gradient singularity, NOT an algorithm divergence.
- **Cost:** RES=512 makes the tectonics bake ~4× heavier (~18s debug / ~1–3s release, async on the loader thread) — a slow first-load is expected.

## Testing

- `cargo test -p enki-nanite` — bake invariants (crack-free, monotonic error), residency/page-pool logic, flatten, and **naga shader validation**. Headless.
- `cargo test -p enki-rhi` / `-p enki-render` — RHI buffer/descriptor + TAA jitter/resolve + `ocean_surface.wgsl` naga validation.
- `cargo test -p enki-app` — ocean FFT/spectrum/sim math (FFT vs DFT, spectrum energy, wave animation, unit normals); nav-mode transitions; `ThirdPersonController` (grounding, camera finiteness, turn-toward-velocity, occlusion stays above terrain, zoom clamp); `Character` rig/gait (rest pose grounded, limbs animate, anti-phase legs).

## Known open items / WIP

- TAA temporal accumulation should be confirmed on-GPU (jitter sign, reprojection under motion, layout/sync). The dither is currently *stable* (degrades to a gradual erosion without TAA); switch it to temporal once TAA accumulation is verified.
- Backface cluster culling (normal cones are baked into `ClusterAsset` but not yet plumbed to `GpuClusterMeta`/cull) + the ~30% degenerate-vertex draw waste.
- On-demand procedural baking for sub-meter close-up detail (current detail is capped by the startup bake resolution).
- Cube-edge seams are an inherent inflated-cube artifact (deprioritized).
- Character is procedural + CPU-skinned: no glTF import, no foot-IK/slope-align (feet ride terrain *height*, not tilt), no GPU skinning. Upgrade paths in order of likely need: glTF rigged import → GPU skinning (flora path) for crowds → foot IK.
- **ki feature-parity roadmap** (the planet still lacks ki's "living world" layers) — top wins in order: (1) **paint rivers + lakes into the lit terrain color** (geometry already carves valleys + flattens lakes and `hf.wetness()` is sampled at `tessellate.rs:163` — it's just not shown; ~S), (2) **atmosphere scattering shell** (port ki `Atmosphere.ts`; kills the flat near-black void + hard silhouette; ~M), (3) **volumetric clouds** over it (ki `VolumetricClouds.ts`; baked `wind_at` + moisture cubemap already exist; ~L), then wind-flow streaklines + climate/wind/tectonics **data debug views** (data exists, slots into the per-vertex-channel pattern). **Skip:** ki's fan/floodplain/delta deposit classifier (dead on ki's own B-path), caves/interior (XL, incompatible with the heightfield path), subsurface.
</content>

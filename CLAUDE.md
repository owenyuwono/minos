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
- Flora (procedural tree) viewer: `cargo run -p enki-app --bin flora_viewer --features flora` — the showcase bin. The planet `enki` bin scatters flora on the surface **only when built `--features flora`** (default build never references flora); see the Flora section.
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
- **enki-flora** — procedural tree generation: a 1:1 Rust port of the `../dryad` JS flora generator (pure gen-core, no Vulkan, golden-gated) + the flora WGSL shaders. Its GPU render path lives in `enki-app` (`flora_*.rs`). **Self-contained, feature-gated (`flora`), deletable** (see the Flora section).

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

A **refractive ocean** in `enki-app/src/ocean/`. With TAA on, `WaveSurface` draws the whole ocean as one refractive 1× pass — a Tessendorf-style **multi-cascade spectral FFT** surface on a projected grid (see below). With TAA off, the old `Ocean` (alpha-blended MSAA shell sphere) is the fallback.
- **`WaveSurface`** — the projected-grid FFT ocean (covers near field + horizon, no patch/shell).
- **`Ocean`** — a smooth translucent sea-level shell sphere, **TAA-off fallback only**.

**Sim (CPU, `ocean/{fft,spectrum,sim}.rs`):** JONSWAP×TMA×Donelan spectrum → `h0(k)` → time-evolve → inverse FFT → displacement (xyz, choppy) + slope normals + **Jacobian foam**. **3 cascades at poseidon's N=256, tiles [250,17,5] m** (→ cm-scale ripples). FFT is **`rustfft`** (the hand-rolled radix-2 only did N≤64). Too heavy for the render thread, so the sim runs on a **background worker** (`std::thread`) that double-buffers its latest field into a `Mutex<Vec<OceanTexel>>`; `record` uploads that to the GPU each frame and the worker idles when no waves are drawn. ponytail: GPU compute is still the eventual upgrade (enki-rhi has compute + storage buffers), but the worker keeps it 60 fps now.

**Wave mesh — projected grid (`ocean/mesh.rs`):** the wave geometry is a **screen-space lattice** (`ndc_grid`, `PROJ_RES`) whose vertices are ray-cast onto the sea sphere every frame on the CPU in f64 (`project_to_sphere`; misses snap to the limb silhouette) into a per-FiF dynamic vertex buffer, drawn by `vs_projected`. It fills the whole view — near field **and** horizon — with **no patch and no shell** (the projected grid IS the entire ocean), and the cells are ~screen-uniform so density follows perspective for free. `mesh.rs` has the two generators + ray/sphere projection unit tests. (Earlier warped-patch + clipmap-ring modes were removed — mesh density only moves vertex *displacement*, which is invisible vs the per-pixel normals; the projected grid is the only one that changed what you see by fixing the patch boundary.)

**Surface (`shaders/ocean_surface.wgsl`):** `vs_projected` builds each vertex's frame from the sphere normal (`center_rel` = planet centre − camera) + the sub-point tangent (for FFT sampling), FFT-displaces, and `fs_main` shades; camera-relative + reversed-Z. **Tile blending (`sample_blend`)** breaks the FFT tile repeat: each ~`BLEND_M`-sized cell rotates+offsets the sample coord and 4 cells are bilinearly blended (horizontal vectors rotated back into the grid frame), so no region reads as a single repeating tile and the rotations look like **crossing / multi-directional seas** — this replaced the old (too-low-freq) domain warp. **Wind-driven** height + intensity: the planet's baked wind (`HeightField::wind_speed_at` → `Climate::wind_speed_at`) is sampled per projected vertex on the CPU each frame into a storage buffer (binding 6), read by `vs_projected` via `vertex_index`, passed to the fragment — so wave amplitude/roughness/foam follow real wind (and the "Ocean → Intensity" heat-map shows it). The wind source is set on the ocean once the heightfield finishes loading (neutral 0.5 until then). **Distance LOD**: a per-pixel screen footprint (`fwidth(world_pos)`) fades each cascade's foam + normals where the pixel can't resolve them, and roughens the specular sun disc with distance — so far / zoomed-out water doesn't alias into white speckle (foam near-field only). Custom set 0 (frame UBO + field storage + ocean UBO + scene **color** + scene **depth** + per-vertex **wind**). Shading = Fresnel sky reflection + sun glint + SSS + foam + **depth-based absorption** (deeper column → darker, projected toward vertical so it tracks ocean *depth*, not view angle).

**Refraction + depth (TAA path only):** `rhi.begin_water_pass()` splits the 3D pass — closes the opaque MSAA instance (resolving to the TAA `current` image + `resolved_depth`), blits opaque color into a history scratch (refraction source) **and copies `resolved_depth`→`refract_depth`** (the depth source for darkening), then reopens a **1× instance on `current`** the water draws into (blend on, alpha 1 → opaque, depth-write off). Requires TAA on; with TAA off only the fallback shell shows. Reuses TAA's `current`/history (no new color image) — `taa_resolve` runs unchanged.

**Gotchas:** the projected `ndc_grid` is wound so cull-BACK keeps the camera-facing face (its triangle normal `rgt×up = −fwd` points at the camera, like the terrain's `east×north = up`); the projection's `-f` Y-flip makes world-CCW front-facing. Graphics pipelines with no push constants pass `push_constant_size: 0`. GUI: Ocean section (toggles + sea-level / choppiness / foam) + view-mode rows **View / Planet / Ocean** (Ocean 11 Surface, 12 Intensity).

## Wind streaks + atmosphere + reference markers

Three thin overlay layers in `enki-app`, drawn over the planet (Planet mode). All reuse the standard `set0` (FrameUniforms) + `ChunkPush` and the hardcoded 4×vec3 vertex layout — **none touch the shared terrain path**; each naga-validates its WGSL in `cargo test -p enki-app`. GUI: **Wind / Atmosphere / Reference** sections.

- **Wind streaklines (`src/wind/`)** — a "living" flow viz of the planet's baked wind field (NOT clouds; debug-grade). `sim.rs` (pure, headless, unit-tested) advects a fixed particle pool through `HeightField::wind_at` — the baked **unit-tangent velocity + speed** cubemap (`Climate::wind_at`, newly exposed on the trait; `WindSample` is world-space xyz, zero at poles) — on a cap around the sub-camera point, plus a cheap time-varying **gust** (rotate the wind about the local normal — no fluid solver; this is Tier B, not Navier–Stokes). Each particle keeps a short trail → a **camera-facing ribbon** (constant screen width), colored blue→white by speed, faded head→tail + birth/death. `mod.rs` owns the GPU glue (per-FiF dynamic vertex buffers; pos in slot 0, packed `[fade, speed, 0]` in the color slot; fragment ramps + alpha). Ribbons are **double-sided** (both index windings — the shared pipeline culls BACK and these are billboards). Drawn **after the ocean** (streaks ride ~1.5 km up = top layer, so the opaque sea can't paint over them); `WindOverlay` holds **two pipelines (MSAA + 1×)** and picks by whether `begin_water_pass` split the frame.
- **Atmosphere shell (`atmosphere.rs` + `.wgsl`)** — one translucent sphere at `R + height` (ocean-shell pattern), shaded analytically: limb-dense alpha (grazing angle), day/night from `frame.sun0_dir`, sunward forward-glow. Gives a visible halo + soft silhouette + the wind's home altitude. Density is smuggled through a `ChunkPush` pad slot. **ponytail: a single analytic shell, NOT real Rayleigh/Mie — and it's an *orbital* halo (back-face-culls away from inside the shell; the surface sky stays `SkyModel`).** Upgrade = port ki `Atmosphere.ts`.
- **Reference markers (`markers.rs` + `.wgsl`)** — toggleable equator ring (green) + N/S pole spikes (red/blue), camera-facing ribbons. **Opaque (`blend:false` → depth-write ON)** so the water pass depth-tests them away and they show through the sea. Default off.

**Draw order (Planet mode):** terrain/Nanite → bodies → atmosphere → markers → character/trees → ocean → **wind (top)**.

**Gotcha — why overlays vanished over water at first:** a translucent overlay drawn *before* the opaque ocean with depth-write OFF gets painted over wherever there's sea (fine over land). Fixes: opaque markers **write depth** (water depth-tests them away); translucent wind **draws after** the water (top layer). The reopened water instance LOADs the resolved opaque depth (`begin_water_pass`), so depth-writing in the opaque pass is what occludes the sea.

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

## Flora (procedural trees — `enki-flora` + `flora_viewer`)

A **1:1 Rust/Vulkan port of the `../dryad` JS flora generator** (procedural trees), shown in a standalone CAS-style viewer. **Self-contained, feature-gated (`flora`), deletable**: the pure generator is `enki-flora`; ALL GPU glue is in `enki-app` behind `#[cfg(feature="flora")]` (`flora_render.rs`/`flora_view.rs`/`flora_ibl.rs`/`flora_nanite.rs`/`bark_swatch.rs` + `src/bin/flora_viewer.rs`); the default build never references it. `enki-app` is `lib + bin` so the `flora_viewer` bin can share `flora_view` via the lib.

**Two halves, meeting at `resolve(&Genome, &Env)`:**
- **Gen-core (`enki-flora/src`, pure, NO Vulkan, deterministic):** `rng` (mulberry32) → `genome` (`random_genome`/`resolve`) → `skeleton`→`roots`→`proportions`→`foliage`, plus `leaf_texture`/`wind_solver`/`mesh`/`bark_swatch`/`color`/`allometry`. A faithful port of dryad's; **determinism is load-bearing** (see the golden contract below).
- **Render (flora-OWNED raw-`ash` sub-renderer, `flora_render.rs` + `enki-flora/shaders/{flora,staging,post}.wgsl`):** modeled on the egui renderer — flora owns its own VkPipelines + descriptor **set0**(FrameUniforms)/**set1**(shadow map + IBL + wind bone buffer + leaf textures) + its own `gpu-allocator` + offscreen targets, recorded into the viewer's frame bracket. enki-rhi's constrained `create_graphics_pipeline` (one hardcoded set/layout) is bypassed for flora's multi-set needs; the **shared rhi/terrain path is untouched**.

**Render pipeline is 1:1 with dryad:** Cook-Torrance PBR (`pbr_shade`) bark/leaf/ground; HDRI IBL (SH9 irradiance + roughness-mip equirect specular baked from `enki-app/assets/flora/kloofendal_43d_clear_1k.hdr`, embedded via `include_bytes!`, in `flora_ibl.rs`); **PCF sun shadow map** (the tree self-shadows — bark+leaf sample it); dryad light rig (sun `#fff4e0`×3.0, hemi×0.3, env 0.6, exposure 1.0) + Preetham sky (`staging.wgsl`); UnrealBloom → **single ACES** in the composite (linear-HDR offscreen intermediate, no double-tonemap, `post.wgsl`).

**Leaves:** baked superformula+venation textures (`leaf_texture.rs`, color + Sobel-normal, per-genome) on **base-anchored 6-segment curved strips** that grow OUTWARD from the twig (base→tip length axis = the cluster tangent, NOT up) and **droop** (t² world-down × `LEAF_BEND` 0.45). Modes: `Cluster` (1 card = the 5–8-leaf sprite, default) / `Single` (1 card = 1 leaf, `bake_leaf_single`). Single-plane, **double-sided** (back-face normal flip → thin edge-on; dryad's crossed-card multi-plane mode is NOT ported). Render-side **twigs-only** gate; the fine twig wood is **tapered** toward the tip (rendered thin, not deleted) so leaves attach without bare sticks poking. Render-side `LeafTuning` sliders (insertion / up-bias / droop / size / density) live in the View panel and are **golden-free**.

**Wind:** hierarchical bone-skinned (ported `windSolver`+`windSkinGlsl`) — the mesher bakes per-vertex bone weights + a `bones_wind` hierarchy, a per-frame Rust solver composes per-bone matrices → set1 storage buffer → vertex skinning. The leaf gust is **coherent along `windDir`** (not random per-leaf tumble), with an **editable arrow gizmo** at the base (drag on the ground plane to set wind direction). `strength=0` = byte-identical static.

**UI:** egui "flora · CAS" panel — left: preset header + tab strip **Climate/Trunk/Branches/Leaves/Root** (dryad's gene grouping); top-left View window: render modes + Wind + Bloom + Leaf-placement + **Dappled shadows** toggles; top-right Stats. Plus a debug **Inspector** dock (leaf texture/normal/shape, pigment swatch+ramp, bark swatch (CPU-approx of `barkAlbedo`), cross-section). Render modes include **Nanite Triangle/Cluster/LOD**: the branch mesh is baked into `enki-nanite` clusters (`flora_nanite.rs`, via `build_base_clusters`+`build_dag` — the arbitrary-mesh path) and drawn through the Nanite cull+draw, leaves hidden in those debug views.

**Dappled shadows** (View toggle / `FLORA_DAPPLED`): bumps the shadow map to 4096² and a `dappled` flag (`ShadowUniforms.params.w`) lets the real self-shadow dominate (softens the canopy-sphere-normal + exposure fake) → sun-through-canopy. Off = byte-identical soft look. Pure per-frame uniform, no rebuild.

**Determinism / golden contract:** `enki-flora/tests/golden.rs` (`golden_vector_matches_dryad`) byte-compares `resolve()`'s foliage (count/position/scale/rotation/shape) against a dryad-JS dump (`tools/dump_golden.mjs` imports the real `../dryad` JS → `tests/golden.json`). Bar: rng draws + genes EXACT (1e-12), geometry 1e-6 (cross-language trig ~1 ULP). **NEVER reorder genome rng draws (incl. the RESERVED no-ops) or edit `foliage.rs`/`skeleton.rs` placement math / `GOLDEN_ANGLE` without a flag-gated path** — it breaks golden. Mesh changes go through `seed_42_mesh*` determinism tests (NOT loosened). ALL render-side work (`flora_view::build_leaf_mesh`, the shaders, `LeafTuning`, the twig taper) is golden-free. naga + a spv-out emit test validate every WGSL in `cargo test`. NOTE: dryad `resolve(genome, {})` yields NaN (no `gravity` default) — always pass an explicit neutral `Env::default()`.

**Phyllotaxis:** the leaf-cluster azimuth is a 1:1 port — `az_step = (1-radialOrder)·π + radialOrder·GOLDEN_ANGLE` (`GOLDEN_ANGLE = π(3−√5) ≈ 137.5°`). Gene-gated by `radialOrder`; the default broadleaf (`≈0.25`) blends toward alternating (180°), so set `radialOrder=1.0` for the visible golden-angle spiral.

**On the planet (`--features flora`):** `flora_scatter` places instances on the surface; each tree draws at its **own distance-driven leaf LOD** (`leaf_lod_at` → `record_at(..., lod)` thins cards / fades with distance), and far trees fall back to **alpha-blended impostor billboards** (`record_impostor_at`, `impostor_blend` > 0) that face the camera — so a grove scales without per-tree geometry. Drawn in the opaque pass after the character, before the ocean.

**Visual correctness is USER-verified** (headless agents can't see pixels). For render work: capture a PNG via `FLORA_SCREENSHOT` and LOOK at it — compile/naga/Vulkan-validation green ≠ visually correct (the first 1:1 build rendered a black void with a bloom-blown canopy and still passed every gate).

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
- **ki feature-parity roadmap** (the planet still lacks ki's "living world" layers) — top wins in order: (1) **volumetric clouds** (ki `VolumetricClouds.ts`; baked `wind_at` + moisture cubemap already exist; ~L) — **the current next goal** (the wind streaks are a debug stand-in, not clouds), (2) **paint rivers + lakes into the lit terrain color** (geometry already carves valleys + flattens lakes and `hf.wetness()` is sampled at `tessellate.rs:163` — it's just not shown; ~S), then climate/wind/tectonics **data debug views** (data exists, slots into the per-vertex-channel pattern). **Done (stand-in):** wind-flow **streaklines** + an **analytic atmosphere halo** (see the Wind/atmosphere section) — a true Rayleigh/Mie scattering shell (port ki `Atmosphere.ts`) is the upgrade past the current single analytic shell. **Skip:** ki's fan/floodplain/delta deposit classifier (dead on ki's own B-path), caves/interior (XL, incompatible with the heightfield path), subsurface.
</content>

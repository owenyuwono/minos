# Wind-Driven Volumetric Clouds for minos — Implementation Research Report

*Deep research synthesis (SOTA-led; ki treated as a cautionary reference, not a template).
Produced from a multi-agent sweep: ki source read + critique, minos infra map, 6 external SOTA
facets, 3 adversarial verifications. Date: 2026-06-22.*

## 1. Goal & scope

Add a **wind-driven volumetric cloud layer** to minos's cube-sphere planet that:

- Reads as **genuinely good** (not a sliding decal, not grainy mush, not a flat cotton-ball blob).
- **Moves with the planet's baked wind field** — the headline feature.
- Works as **one code path from orbit to ground** (camera flies between, may eventually fly *through*
  the layer), on minos's ~50 km radius cube-sphere with floating-origin f64→f32 camera-relative
  precision and reversed-Z.
- Fits a solo dev following "laziest solution that works": shader-only where possible, no new asset
  pipeline, no new RHI subsystem, reusing the atmosphere-shell / ocean-water-split / wind-overlay
  patterns already in `minos-app`.

**Non-negotiable engineering reality (feasibility verification):** minos-rhi has **no
CPU-data→sampled-image path** — `image.rs` only builds `TYPE_2D`, `array_layers(1)`, and
`cmd_copy_buffer_to_image` does not exist outside flora's self-contained raw-ash renderer. So
**anything needing a 3D noise texture or an uploaded climate cubemap is greenfield RHI work**, not
"plumbing onto an existing path." This single fact pushes the design toward **analytic in-shader
noise + CPU-baked coarse coverage in a storage buffer** — the route minos's wind/ocean code already
proves.

## 2. What ki does & why it falls short

ki's `VolumetricClouds.ts` (1224 lines) is a **single fragment shader** on a `DoubleSide` sphere
shell, ray-marching the annulus between `innerR = R + heightScale·1.2` and `outerR = R +
heightScale·1.8`. Its **lighting backbone is production-shaped and good**:

- Dual-lobe phase (Cornette-Shanks forward `g=0.35` + HG back `g=−0.3`, blended 0.5).
- View-gated powder (`1−exp(−2d)`, faded by `cosθ` so it only darkens backlit edges).
- 3-octave Wrenninge/Hillaire multiscatter (`kₙ=0.5ⁿ` scaling extinction/contribution/anisotropy).
- True nested light march (5 taps toward the sun), early-out at `T<0.01`.

**ki looks mediocre because of the sampling/animation layer, not the lighting.** Shortcomings to *avoid*:

1. **No temporal accumulation → persistent crawling grain.** 8–48 steps with only a *static*
   screen-locked IGN dither (redistributes error, doesn't reduce it). The half-res `CloudCompositor`
   was built as the *foundation* for a reproject pass **never written**. ki's self-diagnosed #1 failure.
2. **Wind = rigid great-circle rotation → "sliding decal."** Whole field rotated as one rigid body
   (`dAdv = normalize(dLocal·cos(arc) + windDir·sin(arc))`); no curl, no detail-octave scroll, no
   in-place boil. Clouds translate, never churn. **The weakest part of the headline feature — what
   minos must beat.**
3. **Worley capped at 2 octaves → soft blobby silhouettes** ("EXPENSIVE 27-cell loop/sample"; the 3D
   density cache that would have unlocked more never landed).
4. **Empty-space scout calls the *full* density** (2-oct Worley + 4-oct FBM + weather warp + Voronoi),
   wasting ~2× the budget → too-few steps → feeds the grain in (1).
5. **Sun/ambient color hardcoded, decoupled from sky** → stays warm-white at sunset, reads pasted-on.
6. **Missing post-gate `*= coverage` multiply** → margins look cut-out/hard.
7. **Single height profile** (cloud-type channel unused) → no stratus/cumulus variety.
8. **No terrain depth occlusion** (three.js log-depth vs linear-depth don't compose) → peaks never
   poke through. **minos does NOT have this limitation** (see §6).

**Worth learning from (ki's good half):** wind+moisture baked once into equirect textures;
convergence derived as `−(∂u/∂x + ∂v/∂y)`; wind drives clouds three ways — bulk advection,
wind-aligned domain warp, weather-placement fold. **Copy the coupling structure, not the visuals.**

## 3. minos infrastructure: what exists vs gaps

CLAUDE.md's "baked `wind_at` + moisture cubemap already exist" is **half true and misleadingly framed**.

**Wind — TRUE, runtime-ready (CPU side):**
- `HeightField::wind_at(dir: DVec3) -> WindSample` (`height.rs:48`), `wind_speed_at(dir) -> f32`
  (`height.rs:41`) → `Climate::wind_at` (`climate.rs:325`).
- `WindSample { x, y, z, speed }` (`climate.rs:116`): **world-space planet-local tangent-plane unit
  vector** (NOT face-tangent UV), `speed ∈ [0,1]` dimensionless, **near-zero at poles**.
- Baked `MOIST_RES=128`, 6 faces × 128²; zonal 3-cell + divergence-free Gaussian-streamfunction
  vortices + Coriolis tilt + box blur (1:1 port of ki `bakeWind`).
- **Already consumed correctly at runtime:** wind overlay (`wind/mod.rs:146-154`) and ocean
  (`ocean/mod.rs:458`) CPU-sample per vertex/particle → `write_storage_bytes` into a per-FiF storage
  buffer. **This is the pattern to copy.**

**Moisture — EXISTS in RAM but NOT cleanly queryable:**
- Private `Climate.moisture_field: Vec<f32>` (`climate.rs:209`), 128²×6, [0,1], from `bake_moisture`
  (`climate.rs:442`).
- **No direction-only `moisture(dir)` accessor.** Only escape: `HeightField::climate(dir, height) ->
  (temp, moisture)` (`height.rs:20`). NOTE `wetness(dir)` (`height.rs:30`) is **surface water
  (rivers/lakes), NOT atmospheric humidity** — wrong field.
- **Gap:** add a `moisture(dir) -> f32` trait method (trivial — `Climate::sample` at fixed height).

**Cloud coverage — DOES NOT EXIST.** No coverage/cloud-density field is baked/stored/queryable.
**Clouds must synthesize coverage** from moisture + noise weather-map + (optional) wind convergence.
Convergence is **not computed in minos** — deriving it CPU-side is more work than ki's port implies.

**GPU upload — the real blocker:** no climate cubemap is uploaded as a texture today; everything is
CPU-side `Vec<f32>` sampled via `cubemap::sample_smooth`. `ClimateBaked` (`climate.rs:128`,
`to_baked()` `:345`) is upload-*shaped* but built for worker sharing. **minos-rhi cannot upload a
sampled image from CPU data** (no `cmd_copy_buffer_to_image`, no cube/array view, no
R32_SFLOAT-from-data ctor; the only buffer→image copy is in flora's deletable renderer).

**Net:** lazy-correct route = **CPU-sample wind + new `moisture(dir)`/coverage into a low-res lat-lon
storage buffer**, indexed analytically in the cloud shader. Sidesteps the missing texture path
entirely (same shape as ki's 512×256 bake, but a storage buffer not a texture).

## 4. SOTA landscape

Canonical reference: **Schneider & Vos, "The Real-Time Volumetric Cloudscapes of Horizon Zero Dawn"
(SIGGRAPH 2015)** — Nubis1, ~2 ms on 2015 PS4. The "good clouds" look is almost entirely: two noise
fields (Perlin-Worley base + Worley detail erosion) + height/coverage gradient + Beer + HG + ~6-tap
light march + blue-noise dither at half-res. **Stop there for a solo dev.** Nubis2 (fly-through,
superstorms) and Nubis3 (voxel modeling) are AAA/research-tier and *break the procedural contract*
(Nubis3 needs Houdini-authored voxels + TYPE_3D textures minos cannot even upload).

### 4a. Modeling (density field f(p) → [0,1])

**Base shape — Perlin-Worley** (Worley rounds Perlin so it reads cloud-like):
```
lowFreqFBM = nG*0.625 + nB*0.25 + nA*0.125          // 4:2:1 octaves
baseCloud  = remapClamped(nR, lowFreqFBM - 0.9, 1.0, 0.0, 1.0)   // FBM dilates → wispy edges
```
`remap(v,a,b,c,d) = c + (v-a)/(b-a)*(d-c)`.

**Height gradient (multiply, don't add)** — a vertical density *window*, blended across
stratus/stratocumulus/cumulus by a `cloudType` scalar:
```
gradient = smoothstep(g.x, g.y, h) - smoothstep(g.z, g.w, h)   // h = height fraction [0,1]
base    *= gradient   // density→0 at slab floor/ceiling → flat bottoms, defined tops
```
*MVP:* one gradient + one coverage scalar to start.

**Coverage (apply, THEN re-multiply — the fix ki missed):**
```
cov = remapClamped(base, 1.0 - coverage, 1.0, 0.0, 1.0)
cov *= coverage     // ki omits this → edges look cut-out
```

**Detail erosion (edge-only, never adds density inward):**
```
hfFBM    = nR*0.625 + nG*0.25 + nB*0.125
modifier = mix(hfFBM, 1.0 - hfFBM, clamp(h*2,0,1))   // wispy bottoms, billowy tops
final    = remap(cov, modifier * 0.2, 1.0, 0.0, 1.0) // 0.2-0.35 erode strength
```
**This erosion is the most-skipped step and the #1 reason hobby clouds look like mashed potato.
Non-negotiable for "looks good."**

**Weather map** (R=coverage, G=wetness, B=type), classically a 512² 5-octave Perlin texture sampled
in world XZ → on a cube-sphere maps naturally to minos's moisture/coverage field.

**Nubis3 verdict:** skip (fluid-sim-baked voxels — antithetical to a procedural planet + blocked by
RHI). Borrow only renderer-side ideas (compressed-SDF empty-space skip, near/far temporal split) if
profiling demands.

### 4b. Lighting

- **Beer–Lambert:** `T *= exp(-σ_t·ρ·Δx)`, `C += T·S·Δx`, early-out at `T < 0.01`.
- **Henyey-Greenstein** (verified): `p(θ,g) = (1/4π)(1−g²)/(1+g²−2g·cosθ)^1.5`, `cosθ =
  dot(rayDir, sunDir)`. **Caveat:** θ uses the camera-*ray* direction (along the march), not
  vector-to-camera — state it explicitly or the silver lining lands on the wrong side. Dual-lobe
  `α·p(g_fwd≈0.8) + (1−α)·p(g_back≈−0.3)`, or a single Cornette-Shanks lobe (arguably more physical).
- **Powder** (`2·exp(−d)·(1−exp(−2d))`, view-gated): an **empirical hack** with no radiative-transfer
  derivation; Schneider abandoned it in Nubis2/3. **Optional polish — ship without it, add if backlit
  edges blow out.**
- **Multiscatter octaves (Wrenninge/Hillaire):** `Σₙ σ_s·bⁿ · phase(θ, g·cⁿ) · exp(−σ_e·aⁿ·d_sun)`,
  `a=b=c=0.5`, N=2–4. Cheap (no extra density samples), gives the soft glowing interior. *Caveat:*
  the three-coefficient form is Wrenninge/Frostbite; Ford 2024 (N=8) only scales extinction — don't
  conflate.
- **Energy-conserving integration (Hillaire):** replace `·Δx` with `(1−exp(−σ_t·Δx))/σ_t` so
  brightness is step-count-independent — what lets clouds render correctly at ~8–16 steps.
- **Light march:** ~6 cone samples to the sun (5 near + 1 long); jitter the cone; drop to cheap base
  noise once accumulated alpha > 0.3 (HZD "2× faster").
- **Sun/ambient from the sky** (ki's failure #5): drive tint + ambient from `frame.sun0_dir`/
  `sun0_color` — a couple shader lines, strictly better than hardcoded constants.

### 4c. Performance / temporal — and the critical minos TAA caveat

- **Half-res raymarch + bicubic upscale** ("not noticeably different," ~4× cheaper); ki's
  `CloudCompositor` proves the pattern. *But* a dedicated half-res target hits the missing-image path;
  **march at full res first, add half-res only if framerate demands.**
- **Blue-noise / 4×4 Bayer ray-start offset** — biggest banding fix; **animate per frame** (index by
  TAA frame index) or TAA bakes the banding in.
- **Two-tier empty-space skipping:** cheap low-octave scout in big steps → on hit, step back, switch
  to full density in fine steps; revert after ~6 empty samples. (ki scouted with full density.)
- **Distance-adaptive step size** + **transmittance early-out** + **ray/shell bounds clamp** for the
  horizon-grazing case.

**⚠ The TAA interaction — biggest oversold claim (verification flag).** minos's `taa_resolve`
reprojects **by reconstructing world-pos from the depth buffer** (`if (depth > 0.0)`). Clouds draw
**depth-write OFF**, so:
- **Cloud-over-sky pixels have depth==0 → the resolve SKIPS reprojection/history entirely**, applying
  only the 3×3 neighborhood clamp. Most cloud pixels (sky) get **no temporal accumulation** → a
  half-res/stippled raymarch **shimmers** there, not denoises.
- **Cloud-over-terrain pixels reproject as if glued to the terrain behind them** → wrong parallax
  under camera motion → smearing/ghosting on moving clouds.

So **"TAA denoises the cloud stipple for free, like Nanite" is FALSE** — Nanite writes real geometry
depth; a depth-off overlay does not. Compounding: CLAUDE.md's own open items say TAA accumulation is
**not yet verified on-GPU**, and the Nanite dither is kept *stable* (non-temporal) for that reason.
Clouds move (wind) and the camera moves — exactly the ghosting case.

**Two honest options:**
- **(A) Frostbite-style:** cheap jittered march + engine TAA — *requires* writing a plausible cloud
  **motion vector** (HZD's transmittance-weighted absorption-depth reprojected by previous view-proj).
  Real new work, not free.
- **(B) Sample-count high enough not to rely on TAA**, with animated dither, accepting residual grain
  (defeats half-res but lowest-risk first cut).

**Recommendation:** start with **(B)** at full res + animated IGN dither, then bump samples or invest
in self-contained cloud reprojection. **Do not assume engine TAA cleans it up.**

### 4d. Planet-scale spherical shell + flying through

- **Spherical shell:** `r_inner = R + h_min`, `r_outer = R + h_max`, march the annulus. Ray/sphere
  intersect both radii → `[t_enter, t_exit]`, with four camera-region cases (below/inside/above/limb)
  feeding the *same* marcher. **At 50 km radius the planet curves ~127× harder than Earth — a flat
  slab breaks immediately; the shell formulation is non-optional** and is what makes
  orbit/ground/fly-through one code path.
- **Start the march at shell entry, never the camera** (skip empty vacuum from orbit).
- **Clamp the edge-on chord** + distance-adaptive steps for horizon-grazing rays.
- **Fade density to zero at both shell boundaries** so crossing a sphere doesn't pop.
- **Rescale all Earth-tuned constants** (HZD's 1.5–4 km band → minos's 50 km radius).

### 4e. Wind animation

Dominant technique: **domain scrolling** `sample_pos = world_pos + wind_dir·time·cloud_speed`.
Refinements (shader-only, priority order):

1. **Height skew (cheap, high-value):** `sample_pos += h·wind_dir·cloud_top_offset` — tops lead bases
   (wind shear). One MAD/step.
2. **Two-scale motion:** advect base slow, detail faster + upward-biased (`+vec3(0,
   time·detail_rise, 0)`) so edges **boil** instead of sliding. *Cheapest fix for ki's "sliding decal"
   and the single most important thing to do differently from ki.*
3. **Curl noise** (Bridson 2007, divergence-free `v = ∇×Ψ`) offsets the detail sample → wispy swirls.
   Defer for v1 (exposing curl to the GPU hits the texture wall).
4. **Animate the weather map** for arriving fronts.

**Mapping minos's wind cubemap to advection — the lazy approximation:** domain scroll `+wind·time` is
valid **only if wind is uniform** across the sampled domain; a spatially-varying `+wind(p)·time` is a
non-rigid warp whose distortion **grows linearly with time**, tearing the field apart over minutes. So:
- **Lazy-correct default (recommended):** sample `wind_at(sub_camera_point)` **once per frame** → one
  global wind vector for all cloud samples that frame. One cubemap fetch, no artifacts, smoothly
  changes as you fly to a new region. Exactly how Decima/UE5 work (global wind only). Reuses the
  `wind_at` you already expose.
- **Middle ground (if uniform looks flat):** per-column wind (one lookup per ray, constant down the
  column) for the *base* scroll, global for detail/curl.
- **Never:** true semi-Lagrangian advection (full-volume back-trace per frame).

Copy ki's wind→coverage *structure*, but **add the detail scroll (boil) ki lacks** so motion isn't a
rigid rotation.

## 5. Recommended approach (lazy-first, looks-good)

**Port the Nubis1 (2015) procedural density model with analytic in-shader noise, driven by a
CPU-baked coarse coverage/wind storage buffer, drawn inside the existing water split, with
self-managed sampling quality (do NOT rely on engine TAA for denoise).** Not a 1:1 ki port (that
reproduces ki's mediocrity); not Nubis3 voxels (blocked by RHI + breaks the procedural contract).

### Minimum viable feature set (the "looks good" threshold)

All shader-only / no new assets / no new RHI subsystem:

1. **Spherical shell** `R+h_min`..`R+h_max`, ray/sphere-intersected, marched in **camera-relative f32**
   (ray origin 0, planet center = `(-camera_pos).as_vec3()`), reversed-Z aware. ~48–64 primary steps,
   transmittance early-out.
2. **Analytic Perlin-Worley base + Worley detail erosion** (erosion mandatory — fixes soft blobs and
   cotton-ball). WGSL functions, no 3D texture.
3. The **four cheap Nubis fixes ki omitted:** (a) post-gate `*= coverage`, (b) ×height-gradient
   window, (c) edge-erosion remap, (d) cheap low-octave scout for empty-space skip.
4. **Beer + single-lobe HG (`g≈0.3`) + ~6-tap cone light march + 3-octave multiscatter + Hillaire
   energy-conserving integration.** Reuse ki's lighting backbone but **drive sun/ambient from
   `frame.sun0_dir`/`sun0_color`**.
5. **Wind = sub-camera global vector, two-scale scroll (base slow + detail fast/upward boil) + height
   skew.** Beats ki's rigid-rotation decal in three shader lines.
6. **Coverage** from a CPU-baked low-res lat-lon storage buffer (moisture + noise weather-map; add
   `moisture(dir)` or bake a coverage grid).
7. **Terrain depth occlusion** via `refraction_depth_view()` (reversed-Z) — a genuine advantage over
   ki; peaks poke through.
8. **Animated IGN/blue-noise ray-start dither** (frame-indexed) to break banding — but **size sample
   count so the image is acceptable WITHOUT relying on TAA** (§4c).

### Prioritized nice-to-have ladder

- **Tier 2 (cheap wins):** powder term; curl-noise detail offset; adaptive stepping (HZD 2× win);
  3-preset cloud-type blend.
- **Tier 3 (only if wanted):** self-contained cloud reprojection w/ motion vector (the *correct* way
  to get half-res + temporal denoise — defer until shimmer demands it); half-res RT (needs the missing
  image path); animated/painted weather map; high-altitude cirrus.
- **Never:** Nubis3 voxels, baked 3D Perlin-Worley textures (RHI-blocked), path tracing, full
  multiscatter, ML denoise, lightning/superstorm sim, semi-Lagrangian advection.

## 6. Concrete implementation plan

**New files** (mirror `atmosphere.rs`/`.wgsl` + `wind/`):
- `minos-app/src/clouds.rs` — `Clouds` struct (pipeline(s), shell mesh, params, CPU coverage/wind
  storage buffer, `new`/`record`), `CloudParams` (all sliders), naga-validation unit test.
- `minos-app/src/clouds.wgsl` — `vs_main` (camera-relative shell vertex, like atmosphere) + `fs_main`
  (the raymarch).
- `minos-planet/src/height.rs` — add `fn moisture(&self, dir: DVec3) -> f32` to the trait (default +
  `sampler.rs` impl delegating to `Climate::sample`). Optionally a CPU coverage bake.

**Data flow (wind/moisture → shader), reusing the proven pattern:**
1. On heightfield load, CPU-bake a **low-res lat-lon coverage+wind grid** (e.g. 256×128) by sampling
   `wind_at`/`moisture(dir)` (+ CPU convergence if wanted) → `write_storage_bytes` into a per-FiF
   storage buffer (mirrors `wind/mod.rs:153`, `ocean/mod.rs:458`). **Sidesteps the missing
   texture-upload path.**
2. Per frame: sample `wind_at(sub_camera_point)` → one global `wind_uvw`; push via the cloud UBO.
3. Shader indexes the grid analytically (equirect `u=atan2(d.x,d.z)/2π+0.5, v=asin(d.y)/π+0.5`,
   matching ki's convention) and advects per §4e.

**Draw-order slot:** insert **inside the existing `begin_water_pass` split, before
`WaveSurface::record`** (after `main.rs:1760`). The reopened 1× instance already exposes
`refraction_src_view()` (opaque scene color) and `refraction_depth_view()` (opaque depth copied to
the SAMPLED `refract_depth` image, `lib.rs:1004-1039`) — bind both as SAMPLED_IMAGE+SAMPLER like
`ocean/mod.rs:392-393`. **No new RHI entry point.**

- **Ordering hazard (verification flag):** the ocean writes **opaque alpha=1** into `current`.
  Clouds-*before*-waves → sea composites over sky-clouds (correct sea occlusion) but clouds can't show
  through clear shallow water; clouds-*after*-waves → clouds over sea. **Mutually exclusive in one 1×
  instance.** For sky clouds occluded by sea, draw **before** the waves.

**Scene depth / occlusion vs the water pass:** sample `refraction_depth_view()`, reconstruct
view-space distance from NDC + the **rotation-only** `view_proj` (no `inv_view_proj` is exposed —
reconstruct manually) in camera-relative f32. **Reversed-Z: larger depth = nearer; depth==0 = sky →
march to the far shell.** Early-terminate when accumulated `t` exceeds scene-depth distance. The
fiddliest correctness bit — get the reversed-Z sign and camera-relative reconstruction right.

**TAA interaction:** `taa_resolve` runs unchanged but **does not denoise the cloud stipple** (§4c).
Do **not** design around free TAA denoise. First cut: full-res march + animated dither sized for
standalone quality. Half-res/low-step later requires a **self-contained cloud reprojection** (cloud
history + motion vector from transmittance-weighted absorption depth) — a second small TAA, decoupled
from the engine's depth-based one. **TAA-on-only** (matches the refractive ocean; `begin_water_pass`
returns false with TAA off). A TAA-off path needs an analytic atmosphere-style shell fallback.

**Pipelines:** drawn inside the water split, the instance is single-sample `TYPE_1` → **only the 1×
pipeline is needed.** Add the MSAA pipeline (WindOverlay pattern, `wind/mod.rs:62-75`) only for a
no-split fallback.

**Params — reuse vs add:** a raymarch needs >3 scalars (coverage, σ_t, base alt, thickness, steps,
scroll/detail speed, skew, HG g, powder, wind_uvw…) **and** a custom descriptor set for the
depth/color SAMPLED images + coverage storage buffer. So **use a small custom UBO like the ocean's
`OceanParamsGpu` (`ocean/mod.rs:417-436`), NOT the `ChunkPush._pad` smuggle** (3 f32 slots — fine
only for a trivial analytic shell).

**GUI:** add a `Clouds` `CollapsingHeader` cloned from the Atmosphere section (`gui.rs:~349`); wire 3
edits traced from `atmo_enabled`/`atmo` (GuiOutput, `EguiState::build`, AppState + `record` gate). Add
a debug view-mode slot (cloud-density heatmap) like Ocean's 11/12. Per "tunable params prefer UI,"
expose every constant as a slider.

**Reuse:** `begin_water_pass` split, `refraction_src_view`/`refraction_depth_view`, the ocean's custom
descriptor set + `write_sampled_image_binding`/`write_storage_binding`, the wind/ocean
CPU-sample→storage-buffer pattern, `frame.sun0_dir`/`sun0_color`, `ChunkPush::camera_relative`,
reversed-Z pipeline desc, the naga-validation test. **Add:** the raymarch shader, analytic noise
functions, the coverage-grid bake + `moisture(dir)`, the custom cloud UBO, and (Tier 3) cloud
reprojection.

## 7. Risks & open questions

1. **TAA does not denoise the clouds (highest risk).** Free-temporal-denoise assumption is false for
   a depth-off overlay (§4c, verified). Mitigation: full-res + animated dither standalone first; budget
   time for self-contained reprojection if shimmer is unacceptable under wind+camera motion. Engine TAA
   accumulation is itself unverified on-GPU.
2. **Analytic Worley perf wall.** 27-cell loop/sample × ~48 primary × ~6 light steps full-screen. ki
   capped at 2 octaves for this. Mitigate (no new infra): cheap low-octave scout, distance-LOD step
   reduction, transmittance early-out. Half-res RT and baked 3D textures hit the missing image path —
   defer.
3. **Coverage field is greenfield.** No coverage exists; convergence isn't computed. CPU coverage bake
   is more work than ki's port implies. Open: is moisture+noise enough, or is convergence needed for
   believable fronts?
4. **Sliding-decal motion** if you copy ki's rigid rotation. Must add two-scale detail scroll/boil +
   height skew. Open: does global sub-camera wind read as too uniform, needing per-column wind?
5. **Reversed-Z + camera-relative depth reconstruction** is fiddliest; no `inv_view_proj` exposed.
   Risk of clouds clipping terrain wrong.
6. **Ordering trade-off** (clouds before vs after waves) can't satisfy both sea-occludes-cloud and
   cloud-through-water in one 1× instance.
7. **No TAA-off fallback** without a second (analytic-shell) renderer — acceptable only if TAA stays
   default-ON.
8. **Scale tuning.** All Earth-tuned constants need rescaling to 50 km radius.

## 8. Cited sources

**Modeling / canonical**
- HZD/Nubis1, Schneider & Vos, SIGGRAPH 2015 — https://www.advances.realtimerendering.com/s2015/The%20Real-time%20Volumetric%20Cloudscapes%20of%20Horizon%20-%20Zero%20Dawn%20-%20ARTR.pdf
- ARTR 2015 index — https://advances.realtimerendering.com/s2015/
- Nubis (Decima), SIGGRAPH 2017 — https://advances.realtimerendering.com/s2017/Nubis%20-%20Authoring%20Realtime%20Volumetric%20Cloudscapes%20with%20the%20Decima%20Engine%20-%20Final%20.pdf
- Nubis Evolved (2022) — https://www.guerrilla-games.com/read/nubis-evolved
- HFW Superstorms (GDC 2022) — https://gdcvault.com/play/1027688/The-Real-Time-Volumetric-Superstorms
- Nubis Cubed (Nubis3), SIGGRAPH 2023 — https://advances.realtimerendering.com/s2023/Nubis%20Cubed%20(Advances%202023).pdf
- GPU Pro 7 ch. (Schneider) — https://www.taylorfrancis.com/chapters/edit/10.1201/b21261-11/real-time-volumetric-cloudscapes-andrew-schneider

**Reference implementations / shaders**
- Meteoros (Vulkan) — https://github.com/AmanSachan1/Meteoros
- OfenPower RealtimeVolumetricCloudRenderer (wind/skew GLSL) — https://github.com/OfenPower/RealtimeVolumetricCloudRenderer/blob/master/Shader/volumetricCloudAtmosphere.frag
- TylerDodds VolumetricCloudsTutorial — https://github.com/TylerDodds/VolumetricCloudsTutorial
- Sebastian Lague Clouds — https://github.com/SebLague/Clouds · video https://www.youtube.com/watch?v=4QOcCGI6xOU
- mccannd Project-Marshmallow — https://github.com/mccannd/Project-Marshmallow
- pixelsnafu cloud resources index — https://gist.github.com/pixelsnafu/e3904c49cbd8ff52cb53d95ceda3980e

**Lighting / scattering**
- Hillaire, Frostbite Sky/Atmosphere/Cloud, SIGGRAPH 2016 — https://www.ea.com/frostbite/news/physically-based-sky-atmosphere-and-cloud-rendering
- SIGGRAPH 2016 PBS course — https://blog.selfshadow.com/publications/s2016-shading-course/
- Toft/Bowles/Zimmermann, arXiv:1609.05344 — https://arxiv.org/abs/1609.05344
- Parker Ford thesis 2024 — https://digital.lib.washington.edu/server/api/core/bitstreams/fe31500e-8547-4688-95ee-f08e7e096d19/content
- Henyey–Greenstein — https://en.wikipedia.org/wiki/Henyey%E2%80%93Greenstein_phase_function
- Scratchapixel volume rendering — https://www.scratchapixel.com/lessons/3d-basic-rendering/volume-rendering-for-developers/ray-marching-get-it-right.html

**Performance / temporal**
- bitsquid Volumetric Clouds — http://bitsquid.blogspot.com/2016/07/volumetric-clouds.html
- jpgrenier Volumetric Clouds — https://www.jpgrenier.org/clouds.html
- Vertex Fragment — upsampling https://www.vertexfragment.com/ramblings/volumetric-cloud-upsampling/ · banding https://www.vertexfragment.com/ramblings/volumetric-cloud-banding/
- Maxime Heckel cloudscapes — https://blog.maximeheckel.com/posts/real-time-cloudscapes-with-volumetric-raymarching/

**Planet-scale shell + atmosphere coupling**
- Skybolt planetwide clouds — https://prograda.com/2021/07/28/rendering-planetwide-volumetric-clouds-in-skybolt/
- Hillaire 2020 Scalable Sky/Atmosphere — https://diglib.eg.org/items/8a3e5350-18b3-46bd-9274-3add5af88c75
- Bruneton & Neyret 2008 — https://inria.hal.science/inria-00288758/document · impl https://github.com/ebruneton/precomputed_atmospheric_scattering
- Unreal Volumetric Cloud Component — https://dev.epicgames.com/documentation/unreal-engine/volumetric-cloud-component-in-unreal-engine
- SOTA survey (raymarching clouds) — https://www.researchgate.net/publication/343404421_The_Current_State_of_the_Art_in_Real-Time_Cloud_Rendering_With_Raymarching
- Unity HDRP volumetric clouds — https://docs.unity3d.com/Packages/com.unity.render-pipelines.high-definition@17.0/manual/create-realistic-clouds-volumetric-clouds.html

**Wind / animation**
- Bridson Curl-Noise, SIGGRAPH 2007 — https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf
- GameDev.net HZD cloud thread — https://www.gamedev.net/forums/topic/680832-horizonzero-dawn-cloud-system/

**Floating-origin / precision**
- VIRUP, arXiv:2110.04308 — https://arxiv.org/pdf/2110.04308
- Frozen Fractal floating origin — https://frozenfractal.com/blog/2024/4/11/around-the-world-14-floating-the-origin/
- PBRT Managing Rounding Error — https://www.pbr-book.org/3ed-2018/Shapes/Managing_Rounding_Error

**minos source (integration map):** `minos-planet/src/climate.rs` (`:116` WindSample, `:128`
ClimateBaked, `:209` moisture_field, `:313` sample, `:325` wind_at, `:442` bake_moisture, `:562`
bake_wind); `height.rs` (`:20` climate, `:30` wetness, `:41`/`:48` wind); `sampler.rs:472-496`;
`cubemap.rs`; `minos-app/src/atmosphere.rs`(+`.wgsl`); `wind/mod.rs:33-156`; `ocean/mod.rs:223-472`;
`minos-rhi/src/lib.rs:938-1078` (begin_water_pass + refraction views); `minos-rhi/src/image.rs`
(TYPE_2D-only — the upload gap); `flora_render.rs` (the only buffer→image copy);
`main.rs:1638-1793` (draw order) + `:1886-1917` (TAA resolve); `minos-render/src/material.rs:38-122`
(ChunkPush + _pad + camera_relative); `taa_resolve.wgsl` (depth>0 reprojection gate);
`gui.rs:~349` (Atmosphere section).

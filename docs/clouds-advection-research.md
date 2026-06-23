# How to Make enki's Clouds Move, Merge, and Follow the Wind (True Advection)

*Deep-research synthesis (semi-Lagrangian vs noise-trick), for a solo dev who wants the laziest thing
that genuinely looks like weather flowing. Multi-agent sweep: enki/ki infra read + 4 external SOTA
facets + 3 adversarial verifications. Date: 2026-06-23. Companion to `clouds-research.md`.*

## 0. Verdict (lead with the decision)

**Yes — "merging" REQUIRES a transported field. No noise technique can fake it.**

Merging means *mass piling up where the flow converges* (∇·u < 0, more arrives than leaves). That can
only happen if you store an actual quantity (cloud coverage) and transport it each step so it can
*accumulate*. Every cheaper option fails by construction:

- **Domain warp (today)** — `sample(p − wind(p)·t)`: shears a static field. Mass never moves between
  cells; drop the warp and the field is byte-identical. **Can't merge.** Offset `wind·t` grows
  unbounded → tearing (the current bug).
- **Flow noise** (Perlin/Neyret 2001) — rotates gradients: boils *in place*. No translation, no
  accumulation. The co-inventor classes it as detail-amplification, not transport.
- **Curl noise** (Bridson 2007) — `v = ∇×ψ` is divergence-free *by identity ∇·(∇×ψ)=0*. It is
  **mathematically forbidden from converging** — built precisely to *delete* accumulation gutters.
- **3-layer time-slice blend** — K warp copies on staggered clocks, cross-faded. **Bounds the shear**
  (fixes today's tearing) and translates, but still warps a static field: convergence produces
  *overlap*, not accumulation. **No real merge.**

Only **semi-Lagrangian advection (Stam)** and its crisp-staying upgrade **IBFV feedback (van Wijk)**
genuinely merge — and both require storing + stepping a field every frame. A **storage-buffer (SSBO)
grid is all they need; no texture upload** — exactly enki's constraint.

**Recommendation:** semi-Lagrangian advection of a coverage scalar on the existing 256×128 equirect
storage buffer, on a CPU background worker (ocean-FFT / wind-streak pattern), backtrace in 3D
Cartesian on the unit sphere (dodges the pole singularity), sourced by the moisture grid + decay to
stay bounded and alive. **Medium** effort, not free, but the only path satisfying all three goals.

**Two honest caveats (from verification):**
1. **Merging will be WEAK with enki's current wind field.** The swirl vortices are divergence-free by
   construction (curl of a Gaussian streamfunction, `climate.rs:596-628`) → ~zero convergence. The
   only convergence is the smooth, box-blurred zonal/Coriolis tilt (`climate.rs:639-651`) — gentle
   broad bands, not dramatic fronts. You get **translation + flow-following strongly; merging weakly.**
   Strong merging needs adding divergence to the wind model (out of scope).
2. **Visual correctness is USER-verified.** Tuning source/decay/diffusion to *look* like weather (vs
   merely run) is real iteration a headless agent can't shortcut. Budget it as a feature.

## 1. The problem & why domain-warp fails

`density_at` (`clouds.wgsl:230-236`) offsets the sample coord of a *static* noise field:
```wgsl
let wvel  = sample_wind(dir) * cloud.wind.w;   // world tangent velocity (m/s)
let drift = wvel * cloud.march.w;              // metres carried by elapsed time
let base_p = (pos + drift + ...) * inv;
var shape = fbm(base_p);
```
- **Stretch, not transport.** A per-pixel distortion of a frozen field — density never moves to a new
  cell, so systems never translate coherently and **never merge**.
- **Unbounded shear.** `drift = wind(p)·t`; with varying wind the displacement gradient grows without
  bound → tears into spaghetti over minutes.

Coverage is static too: `coverage_at` (`clouds.wgsl:200-209`) = `shell.z · smoothstep(0.2,0.7,
moisture(dir))` — a climate-fixed mask. **The fix is to make coverage an advected field.**

## 2. The architectural fork

| Technique | Stores field/frame? | Translate | **Merge (∇·u<0)** | Follow path | effort |
|---|---|---|---|---|---|
| Current (domain warp) | No | ✓ but shears unbounded | ✗ | partial | broken |
| Flow noise (2001) | No | ✗ (in-place swirl) | ✗ | ✗ | tiny |
| Curl noise (2007) | No | ~ (detail) | **✗ forbidden ∇·v=0** | ✓ | small |
| 3-layer time-blend | No | ✓ shear *bounded* | ✗ (overlap illusion) | ✓ | **small (shader)** |
| **Semi-Lagrangian** (Stam) | **Yes (SSBO)** | ✓ | **✓ accumulation** | ✓✓ | medium (worker) |
| **IBFV feedback** (van Wijk) | **Yes (ping-pong)** | ✓ | **✓ stays crisp** | ✓✓ | medium (worker) |

**What AAA does (and why it doesn't transfer):** Nubis/HZD/Decima/UE5 use runtime wind-offset scroll +
curl distortion + weather-state cross-fade; their real fluid sims are **offline-baked to voxels**.
Schneider: *"the coverage signal is not animated"* (matrix label: *"Pseudomotion Only"*). They skip
merging because it was never their requirement and they cross-fade *authored* weather states — an
asset pipeline enki lacks. So "AAA skips advection" answers a different question. If merge is
negotiable, the time-blend+curl recipe is the AAA-correct cheap move; if not, you need a transported
field.

**Winner: semi-Lagrangian advection of a coverage grid on a CPU worker** (option to add IBFV
re-injection / MacCormack later). Only path delivering translate + merge + flow-following; trivial
compute at 32K cells; reuses enki's storage-buffer + worker + bilinear-sampler patterns; no texture
upload.

## 3. Recommended algorithm

### 3.1 Field & update
Store coverage `c ∈ [0,1]` on the 256×128 equirect grid, double-buffered (`c_prev`, `c_next`). Each
worker step solves transport-with-source:
```
∂c/∂t + (u·∇)c = S(c)        S(c) = k_form·(target − c) − k_decay·c
```
via the **semi-Lagrangian backtrace** (Stam): trace each cell *backward* along velocity, bilinearly
sample the *previous* field, apply source.

### 3.2 Backtrace in 3D Cartesian on the unit sphere (pole-safe)
Do **not** backtrace in (lon,lat) — the basis is singular at the poles. ECMWF-style 3D fix — the pole
is just another unit vector:
```
1. p     = lonlat_to_unit(lon_i, lat_j)
2. u     = sample_wind(p)                       // world tangent×speed (m/s), what bake stores
3. p_dep = normalize(p − u·dt / R_planet)       // backward tangent step, snap to sphere
                                                //   (great-circle rotate for large dt)
4. (lon,lat) = unit_to_lonlat(p_dep)
5. c_adv = bilinear(c_prev, lon, lat)           // wrap longitude, clamp/fold latitude
6. c_next[i,j] = c_adv + dt·( k_form·(target − c_adv) − k_decay·c_adv )
```
Swap buffers, upload `c_next`.

### 3.3 Moisture source / relax (mandatory — keeps it alive)
Pure advection of a bounded field smears toward constant. `target` = baked moisture coverage
(`smoothstep(0.2,0.7,moisture)`). `k_form`: too strong → tracks the static mask (advection becomes
cosmetic, ki's failure); too weak → diffusion wins (mush). **Narrow sweet spot, USER-tuned.** `k_decay`
= clean exponential sink so convergence can't ratchet into a permanent solid sheet.

### 3.4 Diffusion control
Bilinear backtrace averages 4 neighbors/step → cumulative blur. But at 256×128 the grid is a
low-freq **placement mask** (~1200 m/cell); crisp detail comes from the shader's fbm/Worley. Plan:
- **v1:** plain bilinear SL + source/decay. Ship it.
- **If too soft:** unconditionally-stable **MacCormack** (Selle/Fedkiw) — fwd then reverse advect,
  correct by ½ round-trip error, **mandatorily clamp** to the [min,max] of the 4 forward bilinear
  neighbors (revert to plain SL if departure leaves domain). ~2–3× cost = still sub-ms. The limiter
  is what restores stability — don't drop it.
- Skip explicit κ∇²c (SL already over-diffuses) and vorticity confinement (wind is baked).
- **IBFV** (optional): re-inject jittered band-limited noise each step to restore high freqs.

### 3.5 Sphere/equirect
Longitude wraps (`grid_lerp` already does, `clouds.wgsl:163-164`); latitude ideally **folds** across
the pole (lon→lon+180°) not clamps. Pole risk: equirect cells collapse as `cos(lat)` → anisotropic
smearing, **partially self-limiting** (wind→0 at poles, `polar_fade`). 3D backtrace removes the
*singularity* not the storage non-uniformity; add a `cos(lat)`-weighted polar smooth if noisy.

### 3.6 Timestep / CFL / resolution
SL is **CFL-free / unconditionally stable** (new value = convex combination of 4 old neighbors). But
stability ≠ accuracy: keep departure within a few cells/step (`Δx ≈ 1.2 km` at equator → pick `dt`
so `C = |u|·dt/Δx ≈ 1–5`). Run the worker on its own fixed tick (5–20 Hz), decoupled from render.
Large-dt swirl → iterated-midpoint departure (2 fixed-point iters) if vortex motion looks off. Keep
256×128 (already baked) — the 3D backtrace gets ~90% of the pole benefit with zero re-bake.

### 3.7 Convergence (optional merge bias)
Derive `−(∂u/∂x + ∂v/∂y)` on the CPU by central-differencing the wind grid (ki does this); add
`+ k_conv·conv·dt` to the source. **Weak with enki's divergence-free swirl** — modest bias, not
dramatic merging. Derive *in the worker* (it holds the wind grid) — no `enki-planet` change. Stencil
template: `climate.rs:680-698`.

## 4. enki integration plan

The GPU/threading/upload/sampling scaffold already exists. New work is **purely CPU + additive.**

**Reuse (file refs):**
- **Worker** — clone `WaveSurface`'s thread (`ocean/mod.rs:280-313`): `Arc<Mutex<Vec<f32>>>` double
  buffer, `active`/`running` AtomicBool, idle when not drawn, `Instant`-derived `dt`, Drop-join
  (`:478-485`). Holds two `Vec<f32>` ping-pong grids (32768 f32 = 128 KB each).
- **Upload** — `rhi.write_storage_bytes(coverage_buf, cast_slice(&latest.lock()))` like
  `ocean/mod.rs:387-390`. 128 KB/frame is ~25× *smaller* than the ocean's shipping field upload.
- **Sphere-step math** — already written + unit-tested in `wind/sim.rs:215-231` (`advect`) and
  `:299-349` (tests). Backtrace is the same with `−u`.
- **Equirect bake** — `clouds.rs:310-331` `bake_grids` is the loop to clone for seeding `target[]`.
- **GPU bilinear sampler** — `grid_lerp` + `sample_moisture` (`clouds.wgsl:153-181`). Add a 4-line
  `sample_coverage(dir)` against a new binding.

**Coverage ping-pong & binding (only new GPU plumbing):** add **binding 6** to the cloud layout
(`clouds.rs:132-139`), **per-FiF** (like the ocean's `field_buf` at binding 1, not the shared static
`moisture_buf`), `create_gpu_buffer(size, host_visible=true, STORAGE_BUFFER)`.

**Feed the raymarch:** the single seam is `coverage_at(dir)` (`clouds.wgsl:200-209`) — replace its
`sample_moisture` with `sample_coverage`. Everything downstream untouched. **Trim/delete the
`drift = wvel*time` warp** (`:230-236`) — translation now lives in the advected coverage; keep at most
a tiny detail boil. *This also deletes the unbounded-shear bug.*

**Files:** `clouds.rs` (binding 6, per-FiF buffers, spawn worker, upload), `clouds_advect.rs` (new SL
worker), `clouds.wgsl` (add `sample_coverage`, repoint `coverage_at`, trim warp). **No RHI change, no
`enki-planet` change, no texture-upload path.**

## 5. Cost & quality
- **CPU:** 32,768 cells × ~few dozen flops = ~1–2 Mflop/step → **sub-ms even in debug** (orders below
  the ocean worker's 3×N=256 FFT). MacCormack ~2–3× = still <1 ms. Upload 128 KB/frame = negligible.
- **Diffusion/blur:** real — mitigated by (1) source/relax (necessary), (2) coarse mask + render-time
  fbm/Worley detail, (3) MacCormack, (4) IBFV. Tuning source/decay is the main USER iteration.
- **Pole:** smearing/pinching, bounded by zero polar wind; fold/smooth or accept.
- **Look:** clouds **drift coherently along the bands and curve through vortices** (unambiguous win
  over ki's rigid rotation / the global vector). They **gather gently** in convergence bands but
  **won't show dramatic frontal merging** (divergence-free swirl). Shear tearing gone. Net: convincing
  "weather flowing," modest "weather building."

## 6. Lazy-first staging
- **Stage 0 (shader-only, ship today, IF merge is negotiable):** 3-layer time-slice blend in
  `clouds.wgsl` (K=2–3 warp copies, `sample(p − wind(p)·(t mod τ))`, phase-offset, cross-faded). Kills
  the shear, no worker. **Move + bounded shear, no merge.** AAA-correct cheap path.
- **Stage 1 (the real deliverable — move + merge):** plain bilinear SL advection of a coverage grid on
  a worker (§3.1–3.6), moisture source + decay, 3D backtrace, longitude-wrap. Repoint `coverage_at`,
  trim the warp.
- **Stage 2 (only if it looks wrong):** MacCormack+limiter (mushy); convergence source (stronger
  gather); IBFV (crisper); iterated-midpoint (vortex motion); `cos(lat)` polar smooth; gentle gust to
  keep it lively.
- **Skip:** cube-sphere store; vorticity confinement; explicit κ∇²; full Navier-Stokes/pressure;
  offline voxel bake; bicubic (MacCormack is the better trade).

## 7. Risks / open questions
1. **Merging weak by design** (divergence-free swirl). Is "gentle gather in bands" enough, or does the
   wind model need divergence/fronts (out of scope)?
2. **Source/decay tuning** is a narrow USER-verified sweet spot (cosmetic ↔ mush).
3. **Diffusion:** is plain SL + render detail crisp enough for v1, or is MacCormack/IBFV needed?
4. **Poles:** clamp pinch + cell collapse; bounded but not eliminated.
5. **Steady-state staleness:** fixed wind + source settles; needs a gust perturbation.
6. **Worker tick (5–20 Hz) vs render (60):** confirm no stepping; lerp two worker frames if visible.

## 8. Sources
- Stam, "Stable Fluids," SIGGRAPH 1999 — https://pages.cs.wisc.edu/~chaol/data/cs777/stam-stable_fluids.pdf
- Stam, "Real-Time Fluid Dynamics for Games," GDC 2003 — http://graphics.cs.cmu.edu/nsp/course/15-464/Fall09/papers/StamFluidforGames.pdf
- Harris, "Fast Fluid Dynamics Simulation on the GPU," GPU Gems 38 — https://developer.nvidia.com/gpugems/gpugems/part-vi-beyond-triangles/chapter-38-fast-fluid-dynamics-simulation-gpu
- GPU Gems 3 Ch.30, Real-Time 3D Fluids — https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-30-real-time-simulation-and-rendering-3d-fluids
- Selle/Fedkiw, "An Unconditionally Stable MacCormack Method," 2008 — https://faculty.cc.gatech.edu/~jarek/papers/maccormack.pdf
- Kim et al., "FlowFixer: BFECC for fluid simulation" — https://www.researchgate.net/publication/221314833_FlowFixer_using_BFECC_for_fluid_simulation
- Fedkiw/Stam/Jensen, "Visual Simulation of Smoke" — https://graphics.stanford.edu/papers/smoke/smoke.pdf
- van Wijk, "Image Based Flow Visualization," SIGGRAPH 2002 — https://vanwijk.win.tue.nl/ibfv/ibfv.pdf
- Bridson et al., "Curl-Noise for Procedural Fluid Flow," SIGGRAPH 2007 — https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph2007-curlnoise.pdf
- Neyret, "Advected Textures," SCA 2003 — http://www-evasion.imag.fr/Publications/2003/Ney03/neyret161.pdf
- Gustavson & McEwan, "Tiling Simplex Noise and Flow Noise," JCGT 2022 — https://www.jcgt.org/published/0011/01/02/paper-lowres.pdf
- Perlin & Neyret, "Flow Noise," SIGGRAPH 2001 — https://inria.hal.science/inria-00537499v1
- Max & Becker, "Flow Visualization Using Moving Textures," 1995 — https://escholarship.org/uc/item/6472w2f4
- ECMWF, semi-Lagrangian advection in the IFS — https://www.ecmwf.int/en/newsletter/173/earth-system-science/new-way-computing-semi-lagrangian-advection-ifs
- ECMWF, semi-Lagrangian in atmospheric modelling — https://www.ecmwf.int/sites/default/files/elibrary/2014/9054-semi-lagrangian-technique-atmospheric-modelling-current-status-and-future-challenges.pdf
- "Semi-Lagrangian treatment of advection on the sphere" — https://www.sciencedirect.com/science/article/pii/S0895717700001849
- Great-circle navigation using vectors — https://www.av8n.com/physics/great-circle-vec.htm
- Schneider & Vos, HZD Cloudscapes, SIGGRAPH 2015 — https://www.advances.realtimerendering.com/s2015/The%20Real-time%20Volumetric%20Cloudscapes%20of%20Horizon%20-%20Zero%20Dawn%20-%20ARTR.pdf
- Schneider, Nubis (Decima), SIGGRAPH 2017 — https://advances.realtimerendering.com/s2017/Nubis%20-%20Authoring%20Realtime%20Volumetric%20Cloudscapes%20with%20the%20Decima%20Engine%20-%20Final%20.pdf
- Schneider, Nubis Cubed, SIGGRAPH 2023 — https://advances.realtimerendering.com/s2023/Nubis%20Cubed%20(Advances%202023).pdf
- UE5 Volumetric Cloud Material — https://dev.epicgames.com/documentation/en-us/unreal-engine/volumetric-cloud-material-in-unreal-engine
- Harris, "Real-Time Cloud Simulation and Rendering" (PhD) — http://www.cs.unc.edu/xcms/wpfiles/dissertations/harris.pdf
- niels747, "2D Weather Sandbox" (GLSL grid advection) — https://github.com/niels747/2D-Weather-Sandbox

**enki/ki refs:** `clouds.wgsl:153-209,230-236` · `clouds.rs:132-139,178-197,310-331` ·
`ocean/mod.rs:280-313,334-338,387-390,478-485` · `wind/sim.rs:215-231,299-349` ·
`climate.rs:307,588,596-628,639-661,680-698` · `enki-rhi/src/lib.rs:537,549,560` ·
`ki/src/planet/VolumetricClouds.ts` (static-coverage rotation + baked convergence — what NOT to copy).

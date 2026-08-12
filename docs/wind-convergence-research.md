# Convergence-Driven Cloud Aggregation for minos's Baked Wind

*Deep-research synthesis. Goal: make clouds form and MERGE at convergence zones instead of merely
drifting. Multi-agent sweep (minos/ki read + 4 external + 3 adversarial verifications). Date: 2026-06-23.
Companion to `clouds-advection-research.md`.*

## 0. TL;DR — recommended approach

minos's baked wind is the **curl of a Gaussian streamfunction** (`vel = dir × ∇ψ`, `climate.rs:626-628`),
**divergence-free by the identity `div(curl) ≡ 0`**. A div-free wind can *transport* a coverage field
but never *concentrate* it — the precise, verified reason clouds drift but never merge. The fix is two
parts, and **the load-bearing one is in the cloud worker, not the wind bake**:

1. **`bake_wind` (small):** add a divergent inflow toward existing LOW vortices (outflow from HIGHs),
   reusing the vortex table — **zero new RNG draws, gated behind `convergence_strength: Option<f64>`
   default off → byte-identical / ki-parity preserved.**
2. **`clouds_advect.rs` (makes merging visible):** compute divergence by **central-differencing the
   speed-weighted velocity grid the worker already holds** (`wind_vel = dir·speed`,
   `clouds_advect.rs:231`) — NOT the stored unit-direction wind — and add an **explicit moisture-gated
   convergence source term**. The SL advection is **advective (non-conservative)** (proven by
   `uniform_field_stays_uniform`), so it will *not* pile coverage on its own even with divergent wind.
   The source term is **required, not optional.**

**The biggest trap (verified):** wind is stored as a **unit direction** after `normalize`
(`climate.rs:653-658`). Divergence of a normalized field ≠ divergence of velocity — so the §1 wind-bake
change is *necessary but insufficient* and **largely invisible by itself**. You recover a real,
sign-correct convergence signal only by differencing the **speed-weighted** `wind_vel` in the worker.
Realistic expectation: **gentle gather at lows/ITCZ, not dramatic frontal piling.**

Laziest viable build = §1 (~8 lines) + §4 worker-side `div` + source term (~30 lines) + one slider. No
fluid sim, no Poisson solve, no per-frame cost beyond a finite-difference pass on the 256×128 grid.

## 1. Why they don't merge

Merging = mass piling up where horizontal flow converges (`∇·u < 0`). minos's swirl vortices are
`dir × ∇ψ` (curl of a Gaussian streamfunction) → vorticity but **identically zero divergence**. Only
the broad, box-blurred zonal+Coriolis tilt has any convergence. Strong flow-following, ~zero piling —
the documented conclusion in `clouds-advection-research.md`. Convergence buys: **ITCZ/cyclone
aggregation** (converge → ascend → condense → densest cloud) and **divergence clearing** (subtropical
highs → subside → clear). Both are sign-of-divergence effects a div-free wind can't express.

## 2. The physics, minimally

**Helmholtz decomposition.** `u = k̂×∇ψ (rotational, div=0) + ∇χ (divergent, curl=0, div(u)=∇²χ)`. minos
has the first term; convergence lives entirely in the second. They **add** — append `+∇χ` without
touching the swirl.

**Cheap-correct mechanism — ageostrophic/frictional inflow.** Aloft wind is ~geostrophic (parallel to
isobars → div-free). At the surface, friction turns it *across* isobars toward low pressure (~30° land,
~10° ocean). That cross-isobar inflow IS the divergent surface wind — converges into lows (→ ascent →
cloud), diverges out of highs (→ subsidence → clear). minos's `cross_isobar` tilt (`climate.rs:639-651`)
is a **pure rotation about the normal** — adds ~no usable divergence and gets normalized away.

**ITCZ/Hadley.** Mean convergence at 0° (ITCZ, cloudiest band) and ~60° (polar front); divergence at
~30° (subtropical highs/deserts) and poles.

**Verification caveats (reshape the plan):**
- **Gaussian-amplitude sign is convention-dependent — easy to get backwards.** Don't trust it
  analytically; **pick the slider sign empirically in the viewer** (whichever gathers clouds at lows).
- Geostrophic-is-div-free is exact only on the f-plane; β-effect divergence is immaterial here.
- **The unit-direction normalize is the real bottleneck, not the meteorology.**

## 3. Recommended construction (pick the laziest)

### 3a. Ageostrophic radial inflow toward existing LOW vortices (RECOMMENDED)
The vortex loop (`climate.rs:600-629`) already computes `(gx,gy,gz) = g_scale·d_h` = **`∇ψ` itself**
(today crossed with `dir` to make the div-free swirl). The **divergent field is `(gx,gy,gz)` used
directly**. `v.a` carries the LOW/HIGH sign (`a_sign = +1` LOW / `−1` HIGH, `climate.rs:534`), so
`+g_scale·d_h` flows **toward LOWs (convergence), away from HIGHs (divergence)** — exactly right.

```rust
// inside the existing loop, alongside vort_*:
div_x += gx;  div_y += gy;  div_z += gz;   // ∇ψ = radial in/outflow
// at the combine (climate.rs:635-637):
let cs = params.convergence_strength.unwrap_or(0.0);   // Option, default 0.0
let mut wx = zon_x + swirl_strength*eq_mask*vort_x + cs*eq_mask*div_x;  // wy, wz alike
```
~6 adds/vortex/texel in a loop that already runs. **Zero new RNG draws** → stream-201 walk +
`wind_deterministic` untouched.

> **Analytic gradient-of-Gaussian-on-sphere** (singularity-free) if you build a separate blob list:
> for center `c`, `χ = A·exp(κ(dir·c − 1))`, `κ = 1/σ²` → `∇_S χ = κ·χ·(c − (dir·c)·dir)`. No arccos,
> finite at poles/center. Same shape as `d_h`, so (a) already *is* this gradient.

### 3b. Velocity-potential χ = sum of Gaussians (only if 3a too weak)
A *separate* blob list decoupled from swirl: sinks at lows + an **ITCZ band sink**
`χ_band = B_eq·exp(-lat²/2σ_eq²)` + source bands at ±30°. Emergent ITCZ that *advects* coverage into
the belt. More tuning surface; only if (a) doesn't bite.

### Bound + gate
- Clamp the divergent add-on to a fraction of the rotational speed (`v_div.clamp_length_max(div_frac*
  swirl_speed)`); keep `Σ amp ≈ 0` (no global mass drift).
- `convergence_strength` slider in the Wind GUI, `ClimateParams.convergence_strength: Option<f64>` like
  `swirl_strength`, default `None`.
- **Do NOT feed the divergent magnitude into the speed channel `raw_s` (`climate.rs:660-663`).** `raw_s`
  takes `|magnitude|` and is **sign-blind** — a HIGH (clear) and a LOW (form) both raise `|∇χ|`, giving
  the cloud step no usable sign while needlessly disturbing the ocean. **Leave speed alone.**

## 4. Cloud coupling — where merging becomes visible

**Fact 1 — advection alone won't merge.** The SL step `c_next = bilinear_backtrace(c) + dt·source`
(`clouds_advect.rs:262-272`) is **advective/non-conservative** — conserves `c` along trajectories, not
mass, no `−c(∇·u)` pile-up. Proven by `uniform_field_stays_uniform`. **An explicit source term is
required.**

**Fact 2 — divergence must come from the speed-weighted velocity.** The stored field was normalized
(`climate.rs:653-658`), destroying the magnitude that encodes divergence. The worker holds the right
thing: `wind_vel[idx] = dir·speed` (`clouds_advect.rs:231`). **Central-difference THAT** on the 256×128
grid:
```rust
let du_dx = (u_east(i+1,j) - u_east(i-1,j)) / (2*dx);
let dv_dy = (v_north(i,j+1) - v_north(i,j-1)) / (2*dy);
let div   = du_dx + dv_dy;          // <0 = convergence
divergence[idx] = div;              // worker-internal, no GPU upload
```
Exactly what ki's `_bakeFavTexture` does (`VolumetricClouds.ts:557-590`); sidesteps the unit-dir trap.
~30 lines, no `minos-planet` change. Guard the pole basis singularity (`|east| ≥ 1e-5` → div = 0).

**The source term** (replace `clouds_advect.rs:271`):
```rust
let conv = (-divergence[idx]).max(0.0);                      // convergence magnitude ≥ 0
let s = p.form_rate  * target[idx] * (1.0 - c_adv)          // existing humid formation
      + p.conv_gain  * conv * target[idx] * (1.0 - c_adv)   // NEW: grow at converging, moist cells
      - p.decay_rate * c_adv;                                // existing decay (already clears divergent)
```
- **Gate by `target` (moisture)** — convergence over dry = no cloud; lows over moist ocean = cloud
  factories. This turns "brighter bands" into actual aggregation. Without the gate → the brighter-bands
  failure mode.
- **`(1−c_adv)` saturation** (Tiedtke clear-fraction) — converts clear sky to cloud, self-limits at 1.
- **Divergence-clearing is REDUNDANT** — the existing `−decay_rate·c` already relaxes divergent/quiet
  zones to clear; a div-proportional sink only sharpens. Skip for the lazy build.
- Add `conv_gain: f32` to `AdvectParams` (default 0.0), plumbed from the same slider family as
  `form_rate`.

Chain: **§3 makes the velocity convergent → §4 differences the speed-weighted velocity → the source
grows coverage at moist converging cells → advection curls it downwind = visible aggregation at lows /
the ITCZ.**

## 5. Integration plan

| Step | File / function | Change |
|---|---|---|
| Wind divergent term | `climate.rs:600-629` (vortex loop), `:635-637` (combine) | `div_x/y/z += gx/gy/gz`; mix `+ cs*eq_mask*div_*`. **No new RNG.** Don't touch `raw_s`. |
| Param | `climate.rs:87-104`, `:234-243`, `:466-475` | Add `convergence_strength: Option<f64>` plumbed like `swirl_strength`. Default `None`. |
| GUI | `minos-app/src/gui.rs` Wind section | One `convergence_strength` slider. |
| Cloud divergence | `clouds_advect.rs:227-234` | Central-difference speed-weighted `wind_vel` → worker `divergence[]`. Pole guard. |
| Cloud source | `clouds_advect.rs:271`; params `:35-54` | Add `conv_gain` + moisture-gated convergence source. |
| Determinism | `determinism.rs:355-372` | **Nothing** — all wind params `None` → bake byte-identical. Wind not golden-gated; golden `convergence` is **tectonic** (`tectonics.rs:131-132`). |

**Determinism (verified safe):** wind feeds `bake_moisture`'s rain-shadow march, and `height()` reads
moisture (height golden gated 1e-9) — but `convergence_strength = None` → bake unchanged → golden
unmoved. Only ship a non-zero **default** if you regen height goldens; don't — keep default = ki parity.

**Ocean side-effect (verified):** the ocean reads `wind_at().speed` + dir via a separate coarse re-bake
(`ocean/fetch.rs:186-205`). At `convergence_strength = 0` → wind byte-identical → **ocean byte-identical**
(your active wave tuning undisturbed). On → the inflow is added *before* normalize, so it rotates the
stored *direction* near lows/highs (slight wave-angle shift) but, because speed is untouched, wave
*intensity* is unchanged. Keeping speed out of `raw_s` is exactly the mitigation.

## 6. Cost, risks, open questions
- **Bake cost:** negligible (~6 adds/vortex; one finite-diff pass on 256×128). No per-frame sim.
- **Ocean:** none at default; wave-*angle* only when on (not intensity).
- **Will merging be visible? (headline risk):** §3 alone — weakly, maybe not. The inflow must survive
  normalize (the killer) → Coriolis rotate → box-blur → diffusive SL. The §4 speed-weighted difference +
  moisture-gated source is what recovers + grows it. **Expect "gentle gather at lows/ITCZ," not frontal
  piling.** If too subtle: stronger/asymmetric inflow, less blur on the divergent part, or a separate
  un-normalized divergence cube. **Look at the divergence debug field first** (USER-verified).
- **Symmetric-pair cancellation:** sink/source pairs can average toward zero div after blur; `n_high=4`
  vs `n_low=3` gives slight asymmetry. Check the recovered worker `div` is non-trivial.
- **GUI non-orthogonality:** (a) ties convergence to the swirl vortices, so the two sliders interact.
  (b) decouples them if it matters.
- **Parity:** preserved exactly while default off. ki never solves dynamic merging (its convergence is a
  static-coverage diagnostic, not transport) — this is genuinely new vs ki.

## 7. Lazy-first staging
**MVP (do first):** §3a (`div_* += g*` + `convergence_strength`) + §4 (worker central-diff →
`divergence[]`, moisture-gated `conv_gain` source) + one slider. ~8 lines `climate.rs`, ~30
`clouds_advect.rs`. Turn the slider up, pick the sign that gathers clouds at lows, **look at it**.

**Nice-to-haves (if MVP too subtle):** §3b ITCZ band sink + ±30° source bands; divergence-proportional
clearing sink; separate un-normalized divergence cube; convergence→cloud-type (cumulonimbus over strong
convergence); true non-norm-preserving Ekman `cross_isobar`.

**Skip:** feeding divergence into `raw_s` (sign-blind, disturbs ocean); Poisson/Helmholtz inversion or
per-frame fluid solver (you *construct* χ, no solve); conservative/flux-form SL rewrite; cube-sphere
divergence damping / polar filters (dodged by the analytic Cartesian gradient + backtrace).

## 8. Sources
**Helmholtz / streamfunction–velocity-potential**
- https://en.wikipedia.org/wiki/Helmholtz_decomposition
- https://glossary.ametsoc.org/wiki/Velocity_potential · https://glossary.ametsoc.org/wiki/Streamfunction
- https://en.wikipedia.org/wiki/Stream_function
- https://journals.ametsoc.org/view/journals/mwre/114/8/1520-0493_1986_114_1547_cotsav_2_0_co_2.xml
- https://arxiv.org/pdf/2102.11760 · https://www.chebfun.org/examples/sphere/HelmholtzDecomposition.html
- https://github.com/Xunius/py_helmholtz

**Sources/sinks & Gaussian-on-sphere gradient**
- https://web.mit.edu/fluids-modules/www/potential_flows/LecturesHTML/lec1011/node17.html · https://en.wikipedia.org/wiki/Potential_flow
- https://www.sciencedirect.com/science/article/abs/pii/S0021999109001338 · https://arxiv.org/pdf/1911.10944
- https://en.wikipedia.org/wiki/Great-circle_distance

**Pressure → wind → convergence physics**
- https://courses.ems.psu.edu/meteo3/l6_p6.html · https://courses.ems.psu.edu/meteo3/l11_p3.html
- http://ww2010.atmos.uiuc.edu/(Gh)/guides/mtr/fw/geos.rxml
- https://en.wikipedia.org/wiki/Ekman_layer · https://uw.pressbooks.pub/ocean285/chapter/ekman-pumping/
- https://www.inscc.utah.edu/~krueger/5270/holton_pbl.pdf · https://courses.ems.psu.edu/meteo300/node/726

**Moisture-flux convergence & cloud schemes**
- https://journals.ametsoc.org/view/journals/wefo/20/3/waf858_1.xml · https://www.spc.noaa.gov/publications/banacos/mfc-sls.pdf
- https://journals.ametsoc.org/doi/10.1175/1520-0493(1993)121%3C3040:ROCILS%3E2.0.CO;2
- https://www.ecmwf.int/sites/default/files/elibrary/2009/12778-cloud-parametrization.pdf

**ITCZ / Hadley**
- https://en.wikipedia.org/wiki/Intertropical_Convergence_Zone · https://en.wikipedia.org/wiki/Hadley_cell
- https://science.nasa.gov/earth/earth-observatory/the-intertropical-convergence-zone-703/

**Procedural / games**
- https://dl.acm.org/doi/10.1145/344779.344795 · http://cs.bilkent.edu.tr/~cansin/projects/cs567-animation/clouds/clouds-paper.pdf
- https://niels747.github.io/2D-Weather-Sandbox/ · https://nickmcd.me/2018/07/10/procedural-weather-patterns/
- https://freezedriedmangos.github.io/realistic-planet-generation-and-simulation/

**minos/ki refs:** `climate.rs:500-676` (`bake_wind`; swirl `:626-628`; combine `:635-637`; normalize
`:653-658`; speed `:660-663`); `:87-104`/`:234-243`/`:466-475` (params); `:855` (`wind_deterministic`);
`:307-313` (`wind_at`). `clouds_advect.rs:227-234`,`:262-272`,`:271`,`:332`,`:35-54`.
`ocean/mod.rs:472-505`, `ocean/fetch.rs:186-205`. `determinism.rs:355-372`, `tectonics.rs:131-132`.
ki `VolumetricClouds.ts:521-612` (`_bakeFavTexture` — convergence as static bias, not transport).

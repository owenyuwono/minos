# Rivers research — realistic dendritic river networks (real-time, planet-scale)

Deep-research synthesis (2026-06-22). Sources are primary peer-reviewed papers +
authoritative tool docs; every claim below survived 3-vote adversarial verification.

## The key realization

The earlier ribbon attempt failed for a reason the literature names exactly. The
**width law was correct** (half-width ∝ √discharge — this is the textbook
Leopold–Maddock downstream hydraulic-geometry exponent). The failures were:

1. **Tracing a heavily-blurred accumulation field.** enki blurs `flow_accum` 4
   passes (for seam-smooth terrain). Box-blur smears the sharp drainage lines into
   fat gradients, so streamlines wander over a smeared blob instead of a crisp
   channel. → use the **UNblurred** discharge for the *network* (keep a blurred
   copy for terrain incision if the carve gates need it).
2. **Per-cell streamline tracing + channel-head over-seeding.** "Too small a
   flow-accumulation threshold extracts pseudo rivers" and a constant area-only
   threshold over-predicts channels (Montgomery & Dietrich 1989; MDPI 2021). This
   IS the blobby/stippled result.

The fix is not more/less blur — it's a **junction-node graph** built from a
**principled threshold** on the **unblurred** discharge.

## Recommended pipeline (Peytavie et al. 2019 "Procedural Riverscapes" — 1:1 match)

This maps almost exactly onto enki's existing MFD accumulation + priority-flood fill.

**Stage 1 — EXTRACT a channel raster.** Threshold per-cell drainage area (Freeman
MFD, which enki already has) into a discrete set of channel cells. Pick the
threshold *principally*, not by eye:
- **Slope–area law** `A·Sᵏ ≥ C` (steeper slopes channelize with less upslope area)
  — discriminates true channel heads, recovers finer headwater branches. Combine
  slope WITH area; don't use area alone.
- or **Tarboton's constant-drop test** (TauDEM/QGIS "Drop Analysis"): smallest
  support area where the mean Strahler first-order stream drop stays statistically
  indistinguishable (|t|<2) from higher orders — the channel↔hillslope break.
- Run on **unblurred** discharge. Expose as a UI slider (project convention).

**Stage 2 — COLLAPSE to a junction graph (this is what fixes the blobs).** Don't
ribbon every channel cell. Walk the thresholded raster and place a NODE only where
**≥2 contributors meet** (a confluence); connect nodes with **piecewise-cubic
splines**; compute **Strahler order bottom-up** (leaf = order 1; a confluence
becomes i+1 only when ≥2 tributaries share the max order i). Prune low-order (1–2)
twigs to control density without re-thresholding. Guarantee connectivity with
**priority-flood depression filling** (Barnes/Lindsay-Martin 2014 — enki already
uses this family for lakes), leaving the surface "slightly proud with an available
flow channel."

**Stage 3 — RENDER each edge as a swept ribbon.**
- Width via hydraulic geometry: mean flow `Φ = 0.42·A^0.69` (Dunne & Leopold;
  A = upstream area), cross-section area = Φ/velocity ⇒ width ∝ Q^~0.5. enki's
  log-normalized discharge gives A directly — **the √discharge width was right.**
- **Flow-weighted confluence angles** (Génevaux 2013): near-perpendicular when
  tributary flows differ greatly, small-angle when comparable — avoids messy merges.
- Sweep the spline with a 1-D cross-section profile `h(p) = uz(p) + profile(d(p))`.
- Animated water surface WITHOUT fluid sim: a **blend-flow tree** of small
  procedural flow primitives (calm/turbulent/wave/cascade/vortex/ripple, ~50–90 B
  each) warped by the per-edge velocity field — flow-aligned scrolling, reuses
  enki's reversed-Z / refraction water pass. (Simpler interim: scrolling normals /
  flow-map along the ribbon tangent.)

## Alternative (if AUTHORED branching control is wanted): Génevaux 2013 grammar tree

Grow the network as an **L-system Horton–Strahler tree** instead of extracting it:
each new node constrained strictly higher than its ancestors + a Lipschitz slope
cap (no cliffs) + collision rejection ⇒ a guaranteed connected, monotonically-
downhill, non-self-colliding branching tree. Branching tuned by three probabilities
Pc/Ps/Pa (continuation / symmetric / asymmetric, sum 1). Downside: the terrain must
then be warped to match the synthesized network. Use only if erosion-derived
branching isn't controllable enough.

## enki-specific mapping + pitfalls

- **Unblurred discharge:** bake a second `flow_accum_sharp` field (store the
  pre-blur `q` snapshot in `erosion.rs`) for the network; the blurred `acc_at` stays
  for the golden-gated height path → **no height-golden regen.**
- **Cube-sphere seams (open):** extract per-face (6 graphs stitched at edges, like
  the Nanite per-face DAGs) or on a unified sphere param. No source covers spherical
  network topology — treat like the existing per-face DAG seam handling.
- **Bake at startup, not per-frame.** Derzapf 2011's "instant full-planet rivers, no
  preprocessing" claim was **REFUTED (0-3)**. Bake the graph + ribbon like the DAGs.
- **LOD:** Derzapf confirms GPU view-dependent refinement is the planet-scale
  approach; enki already has the Nanite virtualized path — bake the ribbon into it,
  or keep a screen-stable spline-width overlay, so rivers don't alias from orbit.
- **Discharge constant `0.42·A^0.69`** is region/unit-specific (Dunne & Leopold) —
  on a 50 km toy planet it's purely a relative-width scaling knob.
- **Carve vs overlay:** enki already V-incises valleys keyed to discharge; rivers
  can stay an overlay ribbon (current approach) — fine.
- Stream-power erosion itself yields emergent dendritic networks at the
  uplift↔incision equilibrium (Schott 2023) — the structure is real in enki's field;
  the visible network just needs the graph + correct threshold, not a blurred trace.

## What was REFUTED (don't rely on)

- Anisotropic-diffusion (Perona–Malik) pre-smoothing is *necessary* for clean
  networks — **0-3.** Clean branches come from the junction graph + correct
  threshold, not from more/less filtering.
- Derzapf "instant full-planet rivers, no preprocessing" — **0-3.** Bake it.
- `Dd ∝ 1/√(S_a)` specifically — **1-2** (the inverse relationship holds; that exact
  form doesn't).

## Sources (primary unless noted)

- Tarboton 1991, *On the extraction of channel networks from DEMs* (constant-drop) — https://hydrology.usu.edu/dtarb/hp91.pdf
- Montgomery & Dietrich 1989, *Source areas, drainage density, channel initiation* (slope–area) — http://geomorphology.sese.asu.edu/Papers/Montgomery-Dietrich_DrainageDensity_WRR1989.pdf
- Pelletier 2013, *A robust two-parameter method for the extraction of drainage networks* — https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2012WR012452
- MDPI ISPRS-IJGI 2021, flow-accumulation-threshold sensitivity — https://www.mdpi.com/2220-9964/10/3/186
- Avcioglu et al. 2017 (slope–area channel heads) — https://onlinelibrary.wiley.com/doi/abs/10.1111/1752-1688.12512
- Génevaux et al. 2013, *Terrain Generation Using Procedural Models Based on Hydrology* (SIGGRAPH) — https://www.cs.purdue.edu/cgvlab/www/resources/papers/Genevaux-ACM_Trans_Graph-2013-Terrain_Generation_Using_Procedural_Models_Based_on_Hydrology.pdf
- **Peytavie et al. 2019, *Procedural Riverscapes*** (closest match) — https://people.cs.uct.ac.za/~jgain/wp-content/papercite-data/pdf/peytavie2019.pdf
- Schott et al. 2023, *Large-scale Terrain Authoring through Interactive Erosion Simulation* — https://dl.acm.org/doi/10.1145/3592787
- Derzapf et al. 2011, *River Networks for Instant Procedural Planets* (LOD) — https://cg.cs.uni-bonn.de/backend/v1/files/publications/derzapfPlanets.pdf
- GRASS `r.stream.order` / `r.stream.extract` (Strahler ordering, extract-then-order) — https://grass.osgeo.org/grass-stable/manuals/addons/r.stream.order.html
- Vlachos 2010 *Water Flow in Portal 2* (flow maps) — https://advances.realtimerendering.com/s2010/Vlachos-Waterflow(SIGGRAPH%202010%20Advanced%20RealTime%20Rendering%20Course).pdf

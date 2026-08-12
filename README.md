# minos

A procedural cube-sphere planet renderer in Rust and Vulkan, built on a custom RHI over
[`ash`](https://github.com/ash-rs/ash). You orbit a ~50 km world, drop onto it, and walk
around on terrain that is meshed on demand as you approach it.

It is a prototype, not a game — the long-term target is a floating-origin solar system you
fly between and land on, and terrain that holds up from orbit down to your feet is the wall
that has to fall first.

## What's in it

- **Voxel terrain (default).** An on-demand transvoxel isosurface over an SDF of the baked
  heightfield, quadtree-LOD'd and meshed on background threads, with CDLOD geomorphing so
  levels blend instead of popping. Diggable (sphere CSG dig/fill) and optionally cave-carved.
- **Planet generation ported from `ki`,** its TypeScript/WebGPU predecessor. Tectonic plates,
  hydraulic erosion, wind, moisture and climate bands drive elevation, biome colour and where
  forests grow. Gated by byte-exact golden vectors dumped from the original TypeScript.
- **Shadows.** A 3-cascade CSM, player-centred and texel-snapped, cast and received by
  terrain, character and nearby trees.
- **Ocean.** A Tessendorf-style multi-cascade spectral FFT surface on a projected screen-space
  grid, with wind-driven amplitude, Jacobian foam, refraction and depth absorption.
- **Sky.** Depth-aware aerial perspective, plus volumetric clouds whose coverage field is
  advected by the planet's own baked wind.
- **Flora.** A 1:1 Rust port of `dryad`, a JavaScript procedural tree generator, scattered
  deterministically across the surface with geometry/impostor LOD.
- **A character to walk with.** CPU-skinned procedural humanoid with a gait solver, whose feet
  ray-cast against the actual drawn triangles rather than the analytic height function.

Rendering is camera-relative f64→f32 with reversed-Z depth throughout, so planet-scale
coordinates stay precise.

## Requirements

- A recent stable Rust (edition 2021; developed on 1.96)
- A Vulkan 1.3 GPU — the RHI refuses to start below 1.3, and requires dynamic rendering,
  synchronization2 and timeline semaphores
- Developed and tested on Windows 11 with the MSVC toolchain

Vendored C dependencies (`metis`) build clean on MSVC — no cmake or libclang needed.

## Running

```
make run        # the planet, debug build
make release    # the planet, optimized — do this one
make viewer     # standalone procedural-tree viewer
make classic    # quadtree engine, no voxel terrain or flora
make check      # compile check; safe while the app is running
make test       # full workspace test suite
```

Or without make: `cargo run -p minos-app --release`.

Prefer `release` for anything interactive. The startup tectonics bake runs at RES=512 to match
ki seed-for-seed, which costs ~18s in a debug build versus ~1–3s optimized. It runs
asynchronously on a loader thread either way, so the window appears immediately.

## Controls

| Key | Action |
| --- | --- |
| `Tab` | Cycle nav mode: Globe → Placement → Surface |
| click | In Placement mode, drop the character where you clicked |
| `WASD` | Move (camera-relative on the surface, free-flight in space) |
| `Shift` | Sprint |
| mouse / scroll | Orbit the camera / zoom the chase boom |
| `V` | First ↔ third person, on the surface |
| `F` | Toggle free-flight space camera |
| `M` | Cycle view mode (lit, normals, LOD, plates, wetness, ocean, …) |
| `G` / `H` | Dig / fill the terrain at your feet |
| `W` | Wireframe, while in Globe or Placement mode |
| `Esc` | Leave the surface, back to orbit |

Everything else — sun angle, shadow tuning, ocean, clouds, tree density, view modes — is in
the egui panel.

## Layout

| Crate | What it owns |
| --- | --- |
| `minos-app` | winit shell, egui panel, the renderers, controller and character |
| `minos-rhi` | Vulkan RHI: frame bracket, pipelines, descriptors, swapchain, shadow maps, TAA |
| `minos-render` | Camera, frame uniforms, lights, projection, TAA jitter and resolve |
| `minos-planet` | Heightfield, tectonics, erosion, climate, cube-sphere bases, quadtree LOD |
| `minos-voxel` | Transvoxel mesher — pure mesh generation, no Vulkan |
| `minos-flora` | Procedural tree generator — pure, deterministic, golden-gated |
| `minos-nanite` | Nanite-style virtualized geometry (feature-gated, currently unplugged from the app) |
| `minos-jobs` | Background worker pool for chunk meshing |

Default features are `flora,voxel`. `minos-voxel`, `minos-flora` and `minos-nanite` are each
self-contained and deletable: turn the feature off and none of it compiles.

## Tests

```
make test
```

Over 300 tests, all headless. They cover the terrain and flora goldens (byte-compared against the
original TypeScript and JavaScript implementations), the ocean FFT and spectrum maths, voxel
seam and isosurface invariants, the character rig and controller — and they validate every
WGSL shader through naga, so shader errors are caught without a GPU.

## Further reading

`CLAUDE.md` is the deep architecture guide: every subsystem, its contracts, and the reasoning
behind the parts that look strange. `docs/` holds the research write-ups behind the bigger
features (clouds, rivers, LOD geomorphing, wind).

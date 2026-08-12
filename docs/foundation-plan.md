# minos — Foundation Build Plan

> Authoritative, approved plan: `~/.claude/plans/floofy-crunching-swing.md`. This doc mirrors it for in-repo reference.

**Scope: engine foundations ONLY — Navigation · Rendering · Shaders · Streaming.** Planet/content generation is deferred to a later phase. The engine is exercised with **placeholder geometry** (an analytic sphere at planet radius + synthetic streamed patches); the `ChunkMeshArrays` contract is frozen now so real terrain plugs in later without reshaping anything.

Target: native **Rust + Vulkan (`ash`)**, Windows-only, own RHI (not wgpu). Ultimately hosts the **demiurge** procedural-planet renderer (`../demiurge`).

## Crates
- **minos-rhi** — the only `ash` consumer (device, swapchain, sync, memory, buffers, pipelines, streaming)
- **minos-render** — forward pipeline, camera/projection (reversed-Z), materials/view-variants, lights, ACES, placeholder-geometry provider
- **minos-app** — window/input, navigation (controllers + nav state machine), egui/HUD, wiring
- *(deferred)* **minos-planet**, **minos-jobs** — the terrain phase

## Key decisions
- **Reversed-Z float depth** (`D32_SFLOAT`, clear 0.0, `GREATER`); near 0.5 → far ~750 000
- Single graphics+present queue first; dedicated **transfer queue + QFOT** as a follow-up once streaming is proven
- **Vulkan 1.3** (dynamic rendering + timeline semaphores in core)
- `gpu-allocator`, `naga` (WGSL→SPIR-V), `winit 0.30`, `egui` (M3), `glam`
- **Streaming safety:** a buffer retired on frame F is freed only when GPU `completed_frame ≥ F` (deferred-destruction graveyard)
- 2 terrain pipelines (view modes 0–3 via push-constant branch; wireframe = `LINE`) + 1 water; ACES in-shader + `_SRGB` swapchain; **no textures in the GPU render path**
- **OUT:** compute, storage/3D textures, atomics, readback, instancing, shadows, IBL

## Milestones (risk-first)
- **M0 Bootstrap** — window + device + swapchain + reversed-Z depth + MSAA + clear; validation-clean; resize
- **M1 Rendering + Shaders + depth proof** — forward pipeline, WGSL terrain/water/wireframe, 4 lights, ACES; placeholder sphere + patches under an orbit camera; prove no z-fighting 0.5 → 750 000
- **M2 Streaming** — staging ring + frame-paced uploads + deferred-destroy graveyard; stress harness; zero leaks / use-after-free / stalls
- **M3 Navigation + materials + GUI** — globe + first-person + surface picker (ray-vs-sphere) + nav state machine; 5 view variants; egui + HUD
- **M4 Foundation-complete** — integrated soak ≥60 s; validation-clean, no leaks, stable frame time

## Execution model
Two parallel streams: **A** (RHI/GPU: M0 → M1 → M2) and **B** (App/Navigation, overlaps M2), converging at M4. Per-milestone loop: freeze interface contracts → spawn ≤5 file-disjoint engineer agents in parallel → reviewer checks acceptance criteria → fix-loop → advance.

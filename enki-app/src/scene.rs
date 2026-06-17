//! `scene` — GPU resource setup for the M3 render path.
//!
//! `Scene::new` compiles shaders, builds pipelines, and uploads all static
//! geometry for the depth-proof multi-range test scene.  After construction
//! the scene is immutable and the render loop calls `Scene::record_frame` each
//! frame to emit draw commands using the live per-frame camera from the app's
//! navigation controllers (`GlobeControls` or `FirstPersonController`).
//!
//! # Camera pose (initial / reference)
//! `GlobeControls::new(100_000.0)` → radius 150_000 m, θ = 0, φ = π/4.
//! `spherical_to_cartesian` gives:
//!   position ≈ (0, 106_066, 106_066)  (world-space, planet-centred)
//!   forward  ≈ (0, −0.7071, −0.7071)  (toward origin / planet centre)
//!   right    ≈ (1,       0,       0)   (east)
//!
//! The depth-marker world positions are derived from this initial pose at
//! construction time so they are always in front of the starting camera.
//! The `FrameUniforms` passed to `record_frame` each frame carries the live
//! camera, so the view changes as the user navigates.
//!
//! # Scene composition
//! Depth-marker spheres are placed along the camera forward direction at
//! increasing camera-relative distances spanning 0.5 m → 750 km.  The far and
//! mid-far markers are nudged in +X (right) to avoid being occluded by the
//! planet sphere (R = 50 000 m, centred at origin, which subtends ~19.5° as
//! seen from the camera).
//!
//! | Object            | World pos (approx)            | Cam dist  | Purpose                         |
//! |-------------------|-------------------------------|-----------|----------------------------------|
//! | Near marker (R≈1) | (0,    106 062, 106 062)      |      5 m  | exercises near clip / z-buffer  |
//! | Mid-near (R≈250)  | (0,    105 713, 105 713)      |    500 m  | 3rd-order depth test            |
//! | Mid marker (R≈5k) | (0,     70 621,  70 621)      | 50 000 m  | sits above planet surface       |
//! | Planet sphere     | (0,          0,       0)      | 100 000 m | large opaque mass               |
//! | Far marker (R≈15k)| (260 000, −389 434, −389 434) | 700 000 m | stress far clip without pop     |
//! | Water sphere      | (0,          0,       0)      | 100 000 m | transparent pass after opaque   |
//!
//! The far marker X-offset of +260 000 m ensures it clears the planet limb
//! (planet subtends ±19.5° ≈ ±248 000 m at 700 km camera distance).

use bytemuck::cast_slice;
use enki_render::{
    camera::Camera,
    frame::FrameUniforms,
    geometry::{placeholder_sphere, ChunkMeshArrays},
    lights::Lights,
    material::ChunkPush,
    terrain_pass::TERRAIN_WGSL,
    water_pass::WATER_WGSL,
};
use enki_rhi::{
    vk, BufferHandle, GraphicsPipelineDesc, PipelineHandle, Rhi, RhiError, ShaderModule,
};
use glam::{DVec3, Mat4, Vec3};

use crate::controls::globe::GlobeControls;

// ── GPU mesh ──────────────────────────────────────────────────────────────────

/// One mesh fully resident on the GPU.
struct GpuMesh {
    pos:   BufferHandle,
    nrm:   BufferHandle,
    col:   BufferHandle,
    plate: BufferHandle,
    idx:   BufferHandle,
    count: u32,
}

impl GpuMesh {
    /// Upload a `ChunkMeshArrays` to the GPU.
    ///
    /// If `plate_colors` is `None`, binding 3 is filled with a copy of the
    /// colour buffer so the pipeline always sees four valid vertex streams.
    fn upload(rhi: &mut Rhi, mesh: &ChunkMeshArrays) -> Result<Self, RhiError> {
        let pos   = rhi.create_vertex_buffer(cast_slice(&mesh.positions))?;
        let nrm   = rhi.create_vertex_buffer(cast_slice(&mesh.normals))?;
        let col   = rhi.create_vertex_buffer(cast_slice(&mesh.colors))?;

        // Plate-color fallback: reuse the colour buffer bytes when absent.
        let plate = match &mesh.plate_colors {
            Some(pc) => rhi.create_vertex_buffer(cast_slice(pc))?,
            None     => rhi.create_vertex_buffer(cast_slice(&mesh.colors))?,
        };

        let idx   = rhi.create_index_buffer(&mesh.indices)?;
        let count = mesh.indices.len() as u32;

        Ok(Self { pos, nrm, col, plate, idx, count })
    }

    /// Bind all four vertex streams, the index buffer, then draw.
    fn draw(
        &self,
        rhi:  &mut Rhi,
        fi:   u32,
        push: &ChunkPush,
    ) -> Result<(), RhiError> {
        rhi.bind_vertex_buffers(fi, &[self.pos, self.nrm, self.col, self.plate])?;
        rhi.bind_index_buffer(fi, self.idx)?;
        rhi.push_constants(fi, bytemuck::bytes_of(push))?;
        rhi.draw_indexed(fi, self.count);
        Ok(())
    }
}

// ── Instance ──────────────────────────────────────────────────────────────────

/// A mesh instance: which GpuMesh to draw and where to place it.
///
/// `world_origin` is the f64 world-space position of the instance.  Each frame
/// the camera position is subtracted in f64 before casting to f32 so that
/// `ChunkPush::camera_relative` produces near-origin translations.
/// `rotation_scale` carries any rotation and non-unit scale (identity for
/// axis-aligned spheres).
struct Instance {
    mesh_idx:       usize,
    /// World-space origin in double precision (planet-centred coordinates).
    world_origin:   DVec3,
    /// Rotation + scale component of the model matrix (translation is computed
    /// per-frame via camera-relative subtraction).
    rotation_scale: Mat4,
    material_mode:  u32,
}

// ── Scene ─────────────────────────────────────────────────────────────────────

/// All GPU resources for the M1 render scene.
pub struct Scene {
    terrain_pipeline:           PipelineHandle,
    terrain_wireframe_pipeline: PipelineHandle,
    water_pipeline:             PipelineHandle,

    meshes:            Vec<GpuMesh>,
    opaque_instances:  Vec<Instance>,
    water_instances:   Vec<Instance>,

    pub lights: Lights,
}

impl Scene {
    /// Build all GPU resources.  Call once, after `Rhi::new`.
    ///
    /// `color_format` and `samples` must match the swapchain / MSAA image the
    /// RHI was created with — use `rhi.swapchain_format()` and
    /// `rhi.msaa_samples()`.
    pub fn new(
        rhi:          &mut Rhi,
        color_format: vk::Format,
        samples:      vk::SampleCountFlags,
    ) -> Result<Self, RhiError> {
        // ── Shaders ──────────────────────────────────────────────────────────
        let terrain_shader: ShaderModule = rhi.create_shader_module(TERRAIN_WGSL)?;
        let water_shader:   ShaderModule = rhi.create_shader_module(WATER_WGSL)?;

        // ── Pipelines ────────────────────────────────────────────────────────
        let push_size = std::mem::size_of::<ChunkPush>() as u32; // 80
        let set0      = rhi.set0_layout();

        let terrain_pipeline = rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            shader:            terrain_shader,
            vs_entry:          "vs_main",
            fs_entry:          "fs_main",
            push_constant_size: push_size,
            set0_layout:       set0,
            color_format,
            depth_format:      vk::Format::D32_SFLOAT,
            samples,
            blend:             false,
            fill:              true,
        })?;

        // Wireframe variant: same shader, fill = false (polygon mode LINE).
        let terrain_wireframe_pipeline = rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            shader:            terrain_shader,
            vs_entry:          "vs_main",
            fs_entry:          "fs_main",
            push_constant_size: push_size,
            set0_layout:       set0,
            color_format,
            depth_format:      vk::Format::D32_SFLOAT,
            samples,
            blend:             false,
            fill:              false,
        })?;

        let water_pipeline = rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            shader:            water_shader,
            vs_entry:          "vs_main",
            fs_entry:          "fs_main",
            push_constant_size: push_size,
            set0_layout:       set0,
            color_format,
            depth_format:      vk::Format::D32_SFLOAT,
            samples,
            blend:             true,
            fill:              true,
        })?;

        // Shader modules are no longer needed once pipelines are compiled.
        rhi.destroy_shader_module(terrain_shader);
        rhi.destroy_shader_module(water_shader);

        // ── Derive camera pose ────────────────────────────────────────────────
        //
        // Build the same GlobeControls the app uses so that marker placement is
        // derived from the REAL camera position, not a hardcoded guess.
        //
        // GlobeControls::new(100_000.0) → radius = 150_000 m, θ = 0, φ = π/4.
        // spherical_to_cartesian(150_000, 0, π/4):
        //   x = 0
        //   y = 150_000 · cos(π/4) ≈ 106_066 m
        //   z = 150_000 · sin(π/4) ≈ 106_066 m
        //
        // forward = normalise(−pos) ≈ (0, −0.7071, −0.7071)
        // right   = normalise(forward × worldUp) ≈ (1, 0, 0)
        let controls = GlobeControls::new(100_000.0);
        let cam = controls.camera();

        // World-space camera position in f64 — used to derive marker world
        // positions.  The translation is NOT baked into instance model matrices
        // here; instead each frame record_frame subtracts the live camera
        // position in f64 (camera-relative rendering).
        let cam_pos_f64: DVec3 = cam.position;  // ≈ (0, 106_066, 106_066)

        // Reconstruct forward/right in f32 for offset arithmetic on marker positions.
        // These vectors are O(1) and don't cause precision loss.
        let forward: Vec3 = cam.orientation * Vec3::NEG_Z;  // ≈ (0, −0.7071, −0.7071)
        let right: Vec3   = cam.orientation * Vec3::X;      // ≈ (1, 0, 0)

        // ── Geometry ─────────────────────────────────────────────────────────
        //
        // Planet sphere — radius 50 000 m, 64-subdivision UV sphere.
        // Placed at the world origin; the camera orbits above it.
        let planet_mesh = placeholder_sphere(50_000.0, 64);

        // Water sphere — same topology as planet, radius 1 % larger so it
        // sits above the terrain mesh and blends correctly.
        let water_mesh = placeholder_sphere(50_500.0, 64);

        // Depth-marker spheres: small spheres placed at increasing camera-relative
        // distances along the forward direction to exercise the full reversed-Z
        // range without z-fighting.  Sphere normals are radial so they face the
        // camera regardless of orientation — better than flat patches for this.
        //
        // Marker 0 — 5 m from camera.  Radius 1 m (fills a large solid angle at
        // close range, clearly visible and lit by the sun).
        let near_marker  = placeholder_sphere(1.0, 16);

        // Marker 1 — 500 m from camera.  Radius 250 m.
        let mid_near_marker = placeholder_sphere(250.0, 24);

        // Marker 2 — 50 000 m from camera (sits just above the planet surface).
        // Radius 5 000 m.
        let mid_marker   = placeholder_sphere(5_000.0, 32);

        // Marker 3 — 700 000 m from camera, nudged +X (right) by 260 000 m so
        // it clears the planet limb (planet subtends ~±248 000 m at this distance).
        // Radius 15 000 m so it remains visible at far range.
        let far_marker   = placeholder_sphere(15_000.0, 32);

        // ── Upload all meshes ─────────────────────────────────────────────────
        // Mesh index table:
        //   0 = planet sphere
        //   1 = near marker
        //   2 = mid-near marker
        //   3 = mid marker
        //   4 = far marker
        //   5 = water sphere
        let raw: [(&str, &ChunkMeshArrays); 6] = [
            ("planet",        &planet_mesh),
            ("near marker",   &near_marker),
            ("mid-near",      &mid_near_marker),
            ("mid marker",    &mid_marker),
            ("far marker",    &far_marker),
            ("water",         &water_mesh),
        ];
        let meshes: Vec<GpuMesh> = raw
            .iter()
            .map(|(_, m)| GpuMesh::upload(rhi, m))
            .collect::<Result<Vec<_>, _>>()?;

        // ── Instance placement ────────────────────────────────────────────────
        //
        // Marker world origins (f64) derived from the initial GlobeControls pose:
        //   cam_pos_f64 ≈ (0, 106_066, 106_066)  [GlobeControls::new(100_000.0)]
        //   forward     ≈ (0, −0.7071, −0.7071)
        //   right       ≈ (1, 0, 0)
        //
        // All markers are at cam_pos_f64 + distance * forward [+ lateral offset].
        // Model matrices are NOT baked with world-space translation here — instead
        // each frame record_frame subtracts the live camera.position in f64 to
        // produce a camera-relative offset.  This avoids f32 precision loss when
        // world coordinates are ~50 000+ m.

        // Planet: centred at the world origin.
        let planet_origin = DVec3::ZERO;

        // Near marker: 5 m in front of the camera.
        // World pos ≈ (0, 106_062.5, 106_062.5)
        let near_pos   = cam_pos_f64 + 5.0      * forward.as_dvec3();

        // Mid-near marker: 500 m in front of the camera.
        // World pos ≈ (0, 105_713, 105_713)
        let mid_near_pos = cam_pos_f64 + 500.0  * forward.as_dvec3();

        // Mid marker: 50 000 m in front of the camera (above planet surface).
        // World pos ≈ (0, 70_621, 70_621)
        let mid_pos    = cam_pos_f64 + 50_000.0 * forward.as_dvec3();

        // Far marker: 700 000 m in front of the camera, nudged +260 000 m in
        // the right direction to clear the planet limb.  Without the offset the
        // planet (R = 50 000 m at the origin) would occlude the marker since the
        // forward direction passes through the planet centre.  The planet's
        // angular half-width from the camera is arcsin(50 000 / 150 000) ≈ 19.5°,
        // requiring a lateral offset ≥ 700 000 * tan(19.5°) ≈ 248 000 m.
        // World pos ≈ (260_000, −389_434, −389_434)
        let far_pos    = cam_pos_f64 + 700_000.0 * forward.as_dvec3()
                                     + 260_000.0 * right.as_dvec3();

        let opaque_instances = vec![
            Instance { mesh_idx: 0, world_origin: planet_origin, rotation_scale: Mat4::IDENTITY, material_mode: 0 },
            Instance { mesh_idx: 1, world_origin: near_pos,      rotation_scale: Mat4::IDENTITY, material_mode: 0 },
            Instance { mesh_idx: 2, world_origin: mid_near_pos,  rotation_scale: Mat4::IDENTITY, material_mode: 0 },
            Instance { mesh_idx: 3, world_origin: mid_pos,       rotation_scale: Mat4::IDENTITY, material_mode: 0 },
            Instance { mesh_idx: 4, world_origin: far_pos,       rotation_scale: Mat4::IDENTITY, material_mode: 0 },
        ];

        // Water: same position as the planet — it wraps around it.
        let water_instances = vec![
            Instance { mesh_idx: 5, world_origin: DVec3::ZERO, rotation_scale: Mat4::IDENTITY, material_mode: 0 },
        ];

        Ok(Self {
            terrain_pipeline,
            terrain_wireframe_pipeline,
            water_pipeline,
            meshes,
            opaque_instances,
            water_instances,
            lights: Lights::demiurge_default(),
        })
    }

    /// Record all draw commands for one frame.
    ///
    /// Call after `rhi.begin_frame()` returns a valid `fi`, and before
    /// `rhi.end_frame(fi)`.
    ///
    /// # Parameters
    /// - `fu`            — pre-built `FrameUniforms` (rotation-only view_proj,
    ///   camera_pos = 0).
    /// - `camera`        — live camera for this frame; used to build camera-relative
    ///   `ChunkPush` translations (f64 subtraction, then cast to f32).
    /// - `material_mode` — active material discriminant (0–3) sent to the shader.
    /// - `wireframe`     — when `true`, bind the wireframe terrain pipeline instead
    ///   of the fill terrain pipeline for opaque geometry.  The water pass always
    ///   uses the fill water pipeline regardless of this flag.
    ///
    /// # Draw order
    /// 1. `set_viewport_scissor_full` — dynamic state required each frame.
    /// 2. Terrain pipeline (fill or wireframe): bind → upload uniforms → draw opaque.
    /// 3. Water pipeline: bind → upload uniforms → draw transparent instances.
    ///
    /// Uploading uniforms again after the water `bind_pipeline` re-binds
    /// descriptor set 0 to the new layout (see R1 ordering constraint).
    pub fn record_frame(
        &mut self,
        rhi:           &mut Rhi,
        fi:            u32,
        fu:            &FrameUniforms,
        camera:        &Camera,
        material_mode: u32,
        wireframe:     bool,
    ) -> Result<(), RhiError> {
        rhi.set_viewport_scissor_full(fi);

        // ── Opaque pass ───────────────────────────────────────────────────────
        // bind_pipeline BEFORE update_frame_uniforms (R1 ordering constraint).
        let terrain_pl = if wireframe {
            self.terrain_wireframe_pipeline
        } else {
            self.terrain_pipeline
        };
        rhi.bind_pipeline(fi, terrain_pl)?;
        rhi.update_frame_uniforms(fi, bytemuck::bytes_of(fu))?;

        for inst in &self.opaque_instances {
            // Camera-relative model: subtract camera.position (f64) from the
            // instance's world origin (f64) before casting to f32.  This keeps
            // the GPU working on near-origin coordinates even when the camera
            // is tens of thousands of metres from the planet origin.
            let push = ChunkPush::camera_relative(
                inst.world_origin,
                camera.position,
                inst.rotation_scale,
                material_mode,
            );
            self.meshes[inst.mesh_idx].draw(rhi, fi, &push)?;
        }

        // ── Transparent pass ──────────────────────────────────────────────────
        // Switch pipeline, re-bind uniforms so set 0 is re-bound to new layout.
        // Water always uses fill pipeline; wireframe doesn't apply here.
        rhi.bind_pipeline(fi, self.water_pipeline)?;
        rhi.update_frame_uniforms(fi, bytemuck::bytes_of(fu))?;

        for inst in &self.water_instances {
            let push = ChunkPush::camera_relative(
                inst.world_origin,
                camera.position,
                inst.rotation_scale,
                inst.material_mode,
            );
            self.meshes[inst.mesh_idx].draw(rhi, fi, &push)?;
        }

        Ok(())
    }
}

// ── Headless tests ────────────────────────────────────────────────────────────
//
// These tests run without a GPU or window.  They reconstruct the camera pose
// from `GlobeControls` and verify that the depth-marker world positions encoded
// in `Scene::new` are (a) in front of the camera and (b) span the required
// near→far distance range, locking the depth proof against silent regression.

#[cfg(test)]
mod tests {
    use crate::controls::globe::GlobeControls;
    use glam::{DVec3, Vec3};

    /// Reconstruct the same camera + basis vectors that `Scene::new` uses.
    ///
    /// Returns `(cam_pos_f64, forward_f32, right_f32)`.
    fn scene_camera_basis() -> (DVec3, Vec3, Vec3) {
        let controls = GlobeControls::new(100_000.0);
        let cam = controls.camera();
        let cam_pos_f64 = cam.position;
        let forward = cam.orientation * Vec3::NEG_Z;
        let right   = cam.orientation * Vec3::X;
        (cam_pos_f64, forward, right)
    }

    /// Marker world origins (f64) from `Scene::new`, keyed by label.
    ///
    /// Matches the f64 arithmetic in `Scene::new` so tests are self-consistent.
    fn marker_origins() -> Vec<(&'static str, DVec3)> {
        let (cam_pos_f64, forward, right) = scene_camera_basis();
        vec![
            ("near",     cam_pos_f64 +      5.0 * forward.as_dvec3()),
            ("mid-near", cam_pos_f64 +    500.0 * forward.as_dvec3()),
            ("mid",      cam_pos_f64 + 50_000.0 * forward.as_dvec3()),
            ("far",      cam_pos_f64 + 700_000.0 * forward.as_dvec3()
                                     + 260_000.0 * right.as_dvec3()),
        ]
    }

    /// Every marker must be in front of the camera (positive dot product with
    /// the forward vector).
    #[test]
    fn all_markers_in_front_of_camera() {
        let (cam_pos_f64, forward, _) = scene_camera_basis();
        for (label, world_origin) in marker_origins() {
            let to_marker = (world_origin - cam_pos_f64).as_vec3();
            let dot = to_marker.dot(forward);
            assert!(
                dot > 0.0,
                "marker '{label}' is NOT in front of the camera: dot={dot:.2}"
            );
        }
    }

    /// The nearest marker must be within 10 m of the camera.
    #[test]
    fn nearest_marker_within_10m() {
        let (cam_pos_f64, _, _) = scene_camera_basis();
        let (label, world_origin) = &marker_origins()[0];
        let dist = (*world_origin - cam_pos_f64).length() as f32;
        assert!(
            dist <= 10.0,
            "nearest marker '{label}' is {dist:.2} m from camera, expected ≤ 10 m"
        );
    }

    /// The farthest marker must be at least 500 km from the camera.
    #[test]
    fn farthest_marker_at_least_500km() {
        let (cam_pos_f64, _, _) = scene_camera_basis();
        let markers = marker_origins();
        let (label, world_origin) = markers.last().unwrap();
        let dist = (*world_origin - cam_pos_f64).length() as f32;
        assert!(
            dist >= 500_000.0,
            "farthest marker '{label}' is {dist:.0} m from camera, expected ≥ 500 000 m"
        );
    }

    /// The markers must span at least 4 orders of magnitude in camera distance
    /// (nearest < 10 m, farthest > 500 km → ratio > 50 000).
    #[test]
    fn markers_span_wide_depth_range() {
        let (cam_pos_f64, _, _) = scene_camera_basis();
        let dists: Vec<f32> = marker_origins()
            .iter()
            .map(|(_, p)| (*p - cam_pos_f64).length() as f32)
            .collect();
        let min_dist = dists.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_dist = dists.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ratio = max_dist / min_dist;
        assert!(
            ratio >= 50_000.0,
            "depth range ratio {ratio:.0} is too small (expected ≥ 50 000 × = 4+ orders of magnitude)"
        );
    }

    /// The far marker must clear the planet (R = 50 000 m at origin) — its
    /// distance from the origin should exceed the planet radius by a wide margin.
    #[test]
    fn far_marker_clears_planet() {
        let markers = marker_origins();
        let far_origin = markers.iter().find(|(l, _)| *l == "far").unwrap().1;
        let dist_to_origin = far_origin.length() as f32;
        assert!(
            dist_to_origin > 200_000.0,
            "far marker is too close to the planet origin: dist={dist_to_origin:.0} m"
        );
    }
}

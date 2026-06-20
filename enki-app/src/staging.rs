// staging.rs — SCENE staging for the standalone flora viewer (sky/ground/shadow).
//
// ponytail: the cheapest believable lit "studio" around ONE tree, owned by the
// VIEWER (not FloraView — the tree stays tree-only). It reuses enki's EXISTING
// non-pulling 4×vec3 vertex layout, reversed-Z depth, FrameUniforms lights, and
// the camera-relative f64->f32 push scheme — NO net-new RHI (no second color
// target, no HDR target, no depth-sampler, no shadow-map descriptor sets, no
// post pass). Three tiny meshes + three pipelines from one staging.wgsl:
//
//   SKY    — fullscreen horizon→zenith gradient at clip z=0 (reversed-Z far),
//            drawn FIRST so it sits behind everything; replaces the flat clear.
//   GROUND — large lit ±HALF m plane at y=0 (the tree base sits at y=0).
//   SHADOW — soft radial contact disc lifted to y+epsilon under the trunk,
//            alpha-blended, drawn after the ground and before the tree.
//
// // ponytail: a real PCF sun-shadow-map is the upgrade over the contact disc —
// // it needs a depth render target + a depth-sampler descriptor set that
// // bind_pipeline doesn't expose today, i.e. net-new RHI this viewer avoids.
// // ponytail: BLOOM deferred — there is no HDR/extra color target, blur, or
// // generic post pass on the non-TAA path (only TAA has offscreen+resolve), so
// // a bright-pass+blur+composite would be a net-new multi-pass subsystem in
// // enki-rhi. Bark/leaf already shade to a pleasant LDR; skip it.
// // SMAA is intentionally NOT added: the 3D pass already runs MSAA 4× (samples
// // come from rhi.msaa_samples()), so edges are already resolved.

use enki_rhi::{BufferHandle, Rhi, RhiError};
use glam::{DVec3, Mat4};

use crate::flora_render::{FloraPipeline, FloraRenderer};

/// Half-extent of the ground plane (metres). Big enough to fill the frame at the
/// orbit distance and read as "ground", small enough that f32 stays precise.
const GROUND_HALF: f32 = 200.0;
/// Lift the contact disc this far above y=0 so the alpha-blended (depth-write
/// OFF) disc wins the reversed-Z GREATER test against the opaque ground without
/// needing a depth-bias knob (the RHI has none).
const SHADOW_LIFT: f32 = 0.02;

/// 64-byte push: a single camera-relative model matrix (ground/shadow use it;
/// the sky ignores it). Mirrors `StagePush` in staging.wgsl.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct StagePush {
    model: [[f32; 4]; 4],
}

/// One uploaded indexed mesh in the 4×vec3 layout (pos/nrm/uv/attr).
struct GpuMesh {
    pos: BufferHandle,
    nrm: BufferHandle,
    uv: BufferHandle,
    attr: BufferHandle,
    idx: BufferHandle,
    index_count: u32,
}

// ponytail: no Drop — like FloraView, the BufferHandles/pipelines are reclaimed
// by the RHI at device shutdown for the viewer's whole-run lifetime.

/// Scene staging: sky gradient + ground plane + grounding contact shadow. Holds
/// only the three meshes — the PIPELINES live in the flora-owned `FloraRenderer`
/// (sky/ground/shadow), which `record` borrows. Mirrors FloraView's split.
pub struct Staging {
    sky: GpuMesh,
    ground: GpuMesh,
    shadow: GpuMesh,
}

impl Staging {
    /// Build the three staging meshes. `shadow_radius` sizes the contact disc to
    /// the tree's canopy/trunk footprint (caller passes the branch-mesh AABB
    /// horizontal extent). Pipelines live in the flora-owned `FloraRenderer`.
    pub fn new(rhi: &mut Rhi, shadow_radius: f32) -> Result<Self, RhiError> {
        // ── Meshes ──
        // SKY: a fullscreen quad whose POSITIONS are NDC corners (±1). The sky VS
        // reads position.xy as clip-space directly (z forced to 0 = reversed-Z
        // far), so no view_proj / camera math. normal/uv/attr are dummy streams
        // present only because the fixed layout binds 4 buffers.
        let sky = quad_mesh(
            rhi,
            &[
                [-1.0, -1.0, 0.0],
                [1.0, -1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
        )?;

        // GROUND: a big quad in the XZ plane at y=0, normal +Y, uv 0..1 for the
        // radial edge fade. Drawn camera-relative via the model push.
        let h = GROUND_HALF;
        let ground = plane_mesh(rhi, h, 0.0)?;

        // SHADOW: a small quad sized to the tree footprint, lifted to y+epsilon.
        let r = shadow_radius.max(1.0);
        let shadow = plane_mesh(rhi, r, SHADOW_LIFT)?;

        // ── Pipelines live in the flora-owned FloraRenderer (sky/ground/shadow),
        //    built once at viewer startup. Staging only holds the 3 meshes. ──

        Ok(Self { sky, ground, shadow })
    }

    /// Record sky → ground → contact shadow through the flora-OWNED `renderer`'s
    /// pipelines. Call INSIDE the 3D rendering instance, BEFORE FloraView::record
    /// (so the tree depth-tests against the ground and overdraws the sky).
    /// `camera_world_pos` builds the camera-relative model; the tree sits at world
    /// origin so the ground is at -camera. The caller must have uploaded the frame
    /// uniforms via `renderer.set_frame_uniforms(rhi, fi, fu)` first.
    pub fn record(
        &mut self,
        rhi: &mut Rhi,
        renderer: &FloraRenderer,
        fi: u32,
        camera_world_pos: DVec3,
    ) -> Result<(), RhiError> {
        // Camera-relative model for the y=0 world plane: translation = -camera.
        let model = Self::ground_model(camera_world_pos, 0.0);
        let shadow_model = Self::ground_model(camera_world_pos, SHADOW_LIFT);

        // ── SKY (first; fullscreen at reversed-Z far) ──
        renderer.bind(rhi, fi, FloraPipeline::Sky);
        rhi.set_viewport_scissor_full(fi);
        let identity = StagePush { model: Mat4::IDENTITY.to_cols_array_2d() };
        rhi.bind_vertex_buffers(fi, &[self.sky.pos, self.sky.nrm, self.sky.uv, self.sky.attr])?;
        rhi.bind_index_buffer(fi, self.sky.idx)?;
        renderer.push(rhi, fi, bytemuck::bytes_of(&identity));
        rhi.draw_indexed(fi, self.sky.index_count);

        // ── GROUND (opaque; writes reversed-Z depth) ──
        renderer.bind(rhi, fi, FloraPipeline::Ground);
        let gpush = StagePush { model: model.to_cols_array_2d() };
        rhi.bind_vertex_buffers(fi, &[self.ground.pos, self.ground.nrm, self.ground.uv, self.ground.attr])?;
        rhi.bind_index_buffer(fi, self.ground.idx)?;
        renderer.push(rhi, fi, bytemuck::bytes_of(&gpush));
        rhi.draw_indexed(fi, self.ground.index_count);

        // ── SHADOW (alpha-blended contact disc; after ground, before tree) ──
        renderer.bind(rhi, fi, FloraPipeline::Shadow);
        let spush = StagePush { model: shadow_model.to_cols_array_2d() };
        rhi.bind_vertex_buffers(fi, &[self.shadow.pos, self.shadow.nrm, self.shadow.uv, self.shadow.attr])?;
        rhi.bind_index_buffer(fi, self.shadow.idx)?;
        renderer.push(rhi, fi, bytemuck::bytes_of(&spush));
        rhi.draw_indexed(fi, self.shadow.index_count);

        Ok(())
    }

    /// Camera-relative model for a world-space y-plane: pure translation of
    /// (world_origin - camera) = (-camera) since the plane is centred at world
    /// origin, with the plane lifted by `lift` in world +Y.
    fn ground_model(camera_world_pos: DVec3, lift: f32) -> Mat4 {
        let rel = (DVec3::ZERO - camera_world_pos).as_vec3();
        let mut m = Mat4::IDENTITY;
        m.w_axis.x = rel.x;
        m.w_axis.y = rel.y + lift;
        m.w_axis.z = rel.z;
        m.w_axis.w = 1.0;
        m
    }
}

/// Build a 4-vertex / 6-index quad from explicit corner positions in the order
/// [bottom-left, bottom-right, top-left, top-right] (CCW tris [0,1,2, 2,1,3]).
/// normal = +Y, uv = corner in [0,1]², attr = 0 (padding stream).
fn quad_mesh(rhi: &mut Rhi, corners: &[[f32; 3]; 4]) -> Result<GpuMesh, RhiError> {
    let pos: Vec<f32> = corners.iter().flatten().copied().collect();
    let nrm: Vec<f32> = [0.0_f32, 1.0, 0.0].repeat(4);
    // uv matches CORNERS order: 0:(0,0) 1:(1,0) 2:(0,1) 3:(1,1).
    let uv: Vec<f32> = vec![
        0.0, 0.0, 0.0, //
        1.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, //
        1.0, 1.0, 0.0,
    ];
    let attr: Vec<f32> = vec![0.0; 12];
    let idx: [u32; 6] = [0, 1, 2, 2, 1, 3];
    Ok(GpuMesh {
        pos: rhi.create_vertex_buffer(bytemuck::cast_slice(&pos))?,
        nrm: rhi.create_vertex_buffer(bytemuck::cast_slice(&nrm))?,
        uv: rhi.create_vertex_buffer(bytemuck::cast_slice(&uv))?,
        attr: rhi.create_vertex_buffer(bytemuck::cast_slice(&attr))?,
        idx: rhi.create_index_buffer(&idx)?,
        index_count: idx.len() as u32,
    })
}

/// Build a horizontal quad of half-extent `half` in the XZ plane at height `y`,
/// wound CCW when viewed from above (+Y normal), uv 0..1 across it.
fn plane_mesh(rhi: &mut Rhi, half: f32, y: f32) -> Result<GpuMesh, RhiError> {
    // Corner order [BL, BR, TL, TR] in (x,z); CCW from above with [0,1,2,2,1,3].
    let c = |x: f32, z: f32| [x, y, z];
    let corners = [
        c(-half, -half), // 0 BL
        c(half, -half),  // 1 BR
        c(-half, half),  // 2 TL
        c(half, half),   // 3 TR
    ];
    quad_mesh(rhi, &corners)
}

//! `character` — a procedurally-rigged, CPU-skinned humanoid for the
//! third-person controller.
//!
//! The character is a small box skeleton (~11 bones). A hierarchical gait solver
//! produces one world matrix per bone each frame; the mesh is **skinned on the
//! CPU** (one bone per vertex) and the transformed positions/normals are written
//! into a per-frame-in-flight vertex-buffer ring, then drawn with the **existing
//! terrain pipeline** (lit, `ChunkPush::camera_relative`).
//!
//! ponytail: CPU skinning + the terrain pipeline — no skinning shader, no custom
//! descriptor set, no extra vertex channel. One low-poly character is trivial to
//! re-skin per frame. Move to GPU skinning (the flora path) only when there's a
//! crowd. The whole character placement onto the planet is the same
//! camera-relative model matrix the terrain uses.

#![allow(dead_code)]

use bytemuck::cast_slice;
use enki_render::{
    camera::Camera, frame::FrameUniforms, material::ChunkPush, terrain_pass::TERRAIN_WGSL,
};
use enki_rhi::{
    vk, BufferHandle, GraphicsPipelineDesc, PipelineHandle, Rhi, RhiError, ShaderModule,
};
use glam::{DVec3, Mat4, Vec3, Vec4};

// ── Body dimensions (metres) — shared by the skeleton offsets and the boxes ──

const HIP_HEIGHT: f32 = 0.95;
const TORSO_LEN: f32 = 0.55;
const UPPER_ARM_LEN: f32 = 0.30;
const LOWER_ARM_LEN: f32 = 0.27;
const UPPER_LEG_LEN: f32 = 0.45;
const LOWER_LEG_LEN: f32 = 0.42;
const SHOULDER_X: f32 = 0.22;
const HIP_X: f32 = 0.11;
/// Walk reference speed (m/s) — gait amplitude/cadence are scaled against this.
const WALK_SPEED: f32 = 8.0;

// ── Skeleton ────────────────────────────────────────────────────────────────
//
// Bone indices (parent must be < child for the top-down world composition).
const PELVIS: usize = 0;
const TORSO: usize = 1;
const HEAD: usize = 2;
const UPPER_ARM_L: usize = 3;
const LOWER_ARM_L: usize = 4;
const UPPER_ARM_R: usize = 5;
const LOWER_ARM_R: usize = 6;
const UPPER_LEG_L: usize = 7;
const LOWER_LEG_L: usize = 8;
const UPPER_LEG_R: usize = 9;
const LOWER_LEG_R: usize = 10;
const NUM_BONES: usize = 11;

/// `(parent, joint offset in the parent's local frame)`. The root's "parent" is
/// −1; its offset places the pelvis at the rest hip height above the feet.
const BONES: [(i32, [f32; 3]); NUM_BONES] = [
    (-1,                 [0.0, HIP_HEIGHT, 0.0]),       // PELVIS (root)
    (PELVIS as i32,      [0.0, 0.10, 0.0]),             // TORSO
    (TORSO as i32,       [0.0, TORSO_LEN, 0.0]),        // HEAD
    (TORSO as i32,       [SHOULDER_X, TORSO_LEN - 0.05, 0.0]),  // UPPER_ARM_L
    (UPPER_ARM_L as i32, [0.0, -UPPER_ARM_LEN, 0.0]),   // LOWER_ARM_L
    (TORSO as i32,       [-SHOULDER_X, TORSO_LEN - 0.05, 0.0]), // UPPER_ARM_R
    (UPPER_ARM_R as i32, [0.0, -UPPER_ARM_LEN, 0.0]),   // LOWER_ARM_R
    (PELVIS as i32,      [HIP_X, -0.10, 0.0]),          // UPPER_LEG_L
    (UPPER_LEG_L as i32, [0.0, -UPPER_LEG_LEN, 0.0]),   // LOWER_LEG_L
    (PELVIS as i32,      [-HIP_X, -0.10, 0.0]),         // UPPER_LEG_R
    (UPPER_LEG_R as i32, [0.0, -UPPER_LEG_LEN, 0.0]),   // LOWER_LEG_R
];

// ── Rest mesh ─────────────────────────────────────────────────────────────

/// The static rest mesh: per-vertex local position (in its bone's frame),
/// normal, colour, and the bone that drives it, plus the index buffer.
struct RigMesh {
    pos: Vec<[f32; 3]>,
    nrm: Vec<[f32; 3]>,
    col: Vec<[f32; 3]>,
    bone: Vec<usize>,
    idx: Vec<u32>,
}

/// Append a box (centre `c`, half-extents `h`, all in the bone's local frame)
/// driven by `bone`, wound CCW outward so cull-BACK keeps the outer faces.
#[allow(clippy::too_many_arguments)]
fn push_box(
    m: &mut RigMesh,
    bone: usize,
    c: Vec3,
    h: Vec3,
    color: [f32; 3],
) {
    // (normal, u, v) for each face with u × v = normal (right-handed),
    // so corners (−u−v, +u−v, +u+v, −u+v) wind CCW seen from outside.
    const FACES: [(Vec3, Vec3, Vec3); 6] = [
        (Vec3::X, Vec3::Y, Vec3::Z),
        (Vec3::NEG_X, Vec3::Z, Vec3::Y),
        (Vec3::Y, Vec3::Z, Vec3::X),
        (Vec3::NEG_Y, Vec3::X, Vec3::Z),
        (Vec3::Z, Vec3::X, Vec3::Y),
        (Vec3::NEG_Z, Vec3::Y, Vec3::X),
    ];
    for (n, u, v) in FACES {
        let base = m.pos.len() as u32;
        let hn = (h * n).abs().element_sum(); // half-extent along the normal axis
        let hu = (h * u).abs().element_sum();
        let hv = (h * v).abs().element_sum();
        let center = c + n * hn;
        let corners = [
            center - u * hu - v * hv,
            center + u * hu - v * hv,
            center + u * hu + v * hv,
            center - u * hu + v * hv,
        ];
        for corner in corners {
            m.pos.push(corner.to_array());
            m.nrm.push(n.to_array());
            m.col.push(color);
            m.bone.push(bone);
        }
        m.idx.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }
}

/// Build the rest-pose box humanoid. Boxes are expressed in their bone's local
/// frame (origin at the joint), so skinning by the bone's world matrix places
/// them correctly.
fn build_rig() -> RigMesh {
    let mut m = RigMesh {
        pos: Vec::new(), nrm: Vec::new(), col: Vec::new(), bone: Vec::new(), idx: Vec::new(),
    };

    let jacket = [0.20, 0.35, 0.65];
    let pants = [0.12, 0.14, 0.22];
    let skin = [0.80, 0.62, 0.50];
    let boots = [0.10, 0.10, 0.12];

    // Pelvis straddles its joint; torso rises from it to the neck.
    push_box(&mut m, PELVIS, Vec3::ZERO, Vec3::new(0.16, 0.12, 0.11), pants);
    push_box(&mut m, TORSO, Vec3::new(0.0, TORSO_LEN * 0.5, 0.0), Vec3::new(0.18, TORSO_LEN * 0.5, 0.11), jacket);
    // Head sits above a short neck gap.
    push_box(&mut m, HEAD, Vec3::new(0.0, 0.06 + 0.13, 0.0), Vec3::new(0.13, 0.13, 0.12), skin);
    // Arms hang down from the shoulders (−Y), sleeves then hands.
    for (upper, lower) in [(UPPER_ARM_L, LOWER_ARM_L), (UPPER_ARM_R, LOWER_ARM_R)] {
        push_box(&mut m, upper, Vec3::new(0.0, -UPPER_ARM_LEN * 0.5, 0.0), Vec3::new(0.06, UPPER_ARM_LEN * 0.5, 0.06), jacket);
        push_box(&mut m, lower, Vec3::new(0.0, -LOWER_ARM_LEN * 0.5, 0.0), Vec3::new(0.05, LOWER_ARM_LEN * 0.5, 0.05), skin);
    }
    // Legs hang down from the hips, thighs then shins/boots.
    for (upper, lower) in [(UPPER_LEG_L, LOWER_LEG_L), (UPPER_LEG_R, LOWER_LEG_R)] {
        push_box(&mut m, upper, Vec3::new(0.0, -UPPER_LEG_LEN * 0.5, 0.0), Vec3::new(0.075, UPPER_LEG_LEN * 0.5, 0.075), pants);
        push_box(&mut m, lower, Vec3::new(0.0, -LOWER_LEG_LEN * 0.5, 0.0), Vec3::new(0.065, LOWER_LEG_LEN * 0.5, 0.065), boots);
    }
    m
}

// ── Gait solver ─────────────────────────────────────────────────────────────

/// Solve the per-bone **world** matrices (in character-local space, feet at the
/// origin) for the given walk `phase` (radians) and ground `speed` (m/s).
///
/// At `speed == 0` and `phase == 0` every animated rotation is zero → the exact
/// rest pose (a standing figure). Legs/arms swing in anti-phase; speed scales
/// both the swing amplitude and (via the caller's phase advance) the cadence.
fn solve_pose(phase: f32, speed: f32) -> [Mat4; NUM_BONES] {
    let amp = (speed / WALK_SPEED).clamp(0.0, 1.2); // 0 idle … 1 walk … >1 sprint
    let s = phase.sin();
    let s_opp = (phase + std::f32::consts::PI).sin();
    let leg = 0.5 * amp;   // hip swing
    let knee = 0.7 * amp;  // calf bend (back only)
    let arm = 0.4 * amp;   // shoulder swing (opposite the legs)
    let elbow = 0.3 * amp;
    let breath = 0.025 * phase.sin(); // tiny always-on idle motion (0 at phase 0)

    // Per-bone local animation rotation (applied at the joint, after the offset).
    let anim = |b: usize| -> Mat4 {
        match b {
            UPPER_LEG_L => Mat4::from_rotation_x(leg * s),
            UPPER_LEG_R => Mat4::from_rotation_x(leg * s_opp),
            LOWER_LEG_L => Mat4::from_rotation_x(knee * s.max(0.0)),
            LOWER_LEG_R => Mat4::from_rotation_x(knee * s_opp.max(0.0)),
            UPPER_ARM_L => Mat4::from_rotation_x(-arm * s + breath),
            UPPER_ARM_R => Mat4::from_rotation_x(-arm * s_opp + breath),
            LOWER_ARM_L => Mat4::from_rotation_x(elbow * (-s).max(0.0) + 0.15),
            LOWER_ARM_R => Mat4::from_rotation_x(elbow * (-s_opp).max(0.0) + 0.15),
            TORSO => Mat4::from_rotation_y(0.05 * amp * s),
            HEAD => Mat4::from_rotation_y(-0.03 * amp * s),
            _ => Mat4::IDENTITY,
        }
    };

    // Vertical bob on the pelvis (twice per stride), tiny breathing rise at idle.
    let bob = 0.02 * amp * (1.0 - (2.0 * phase).cos()) + 0.01 * breath;

    let mut world = [Mat4::IDENTITY; NUM_BONES];
    for b in 0..NUM_BONES {
        let (parent, offset) = BONES[b];
        let mut off = Vec3::from(offset);
        if b == PELVIS {
            off.y += bob;
        }
        let local = Mat4::from_translation(off) * anim(b);
        world[b] = if parent < 0 {
            local
        } else {
            world[parent as usize] * local
        };
    }
    world
}

// ── Character ───────────────────────────────────────────────────────────────

/// A CPU-skinned humanoid the third-person controller drives. Owns its draw
/// pipeline + the static colour/index buffers + a per-frame position/normal ring.
pub struct Character {
    pipeline: PipelineHandle,

    // Static streams (never change): colour (also reused as the plate slot) + index.
    col: BufferHandle,
    idx: BufferHandle,
    index_count: u32,

    // Per-frame-in-flight skinned streams, rewritten each frame so writing this
    // frame's verts never races the GPU reading last frame's.
    pos_ring: Vec<BufferHandle>,
    nrm_ring: Vec<BufferHandle>,

    // Rest mesh (bone-local) + scratch the CPU skinning writes into.
    rest: RigMesh,
    skinned_pos: Vec<[f32; 3]>,
    skinned_nrm: Vec<[f32; 3]>,

    /// Walk-cycle phase (radians, wrapped to 0..2π).
    phase: f32,
}

impl Character {
    /// Build the character's GPU resources. `color_format`/`samples` must match
    /// the 3D MSAA pass (use `rhi.swapchain_format()` / `rhi.msaa_samples()`).
    pub fn new(
        rhi: &mut Rhi,
        color_format: vk::Format,
        samples: vk::SampleCountFlags,
    ) -> Result<Self, RhiError> {
        let rest = build_rig();
        let vert_count = rest.pos.len();

        // Lit terrain pipeline (same set0/push/vertex contract as the planet).
        let shader: ShaderModule = rhi.create_shader_module(TERRAIN_WGSL)?;
        let pipeline = rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            shader,
            vs_entry: "vs_main",
            fs_entry: "fs_main",
            push_constant_size: std::mem::size_of::<ChunkPush>() as u32,
            set0_layout: rhi.set0_layout(),
            color_format,
            depth_format: vk::Format::D32_SFLOAT,
            samples,
            blend: false,
            fill: true,
        })?;
        rhi.destroy_shader_module(shader);

        // Static colour stream; the plate slot reuses it (mode 0 ignores plate).
        let col = rhi.create_vertex_buffer(cast_slice(&rest.col))?;
        let idx = rhi.create_index_buffer(&rest.idx)?;

        // Per-frame position/normal ring (host-visible, rewritten each frame).
        let stream_bytes = (vert_count * std::mem::size_of::<[f32; 3]>()) as u64;
        let frames = rhi.frames_in_flight();
        let mut pos_ring = Vec::with_capacity(frames);
        let mut nrm_ring = Vec::with_capacity(frames);
        for _ in 0..frames {
            pos_ring.push(rhi.create_gpu_buffer(stream_bytes, true, vk::BufferUsageFlags::VERTEX_BUFFER)?);
            nrm_ring.push(rhi.create_gpu_buffer(stream_bytes, true, vk::BufferUsageFlags::VERTEX_BUFFER)?);
        }

        Ok(Self {
            pipeline,
            col,
            idx,
            index_count: rest.idx.len() as u32,
            pos_ring,
            nrm_ring,
            skinned_pos: vec![[0.0; 3]; vert_count],
            skinned_nrm: vec![[0.0; 3]; vert_count],
            rest,
            phase: 0.0,
        })
    }

    /// Advance the gait, skin the mesh on the CPU, and upload this frame's
    /// positions/normals. `speed_mps` is the controller's reported ground speed
    /// (0 = idle).
    pub fn update(
        &mut self,
        rhi: &mut Rhi,
        fi: u32,
        speed_mps: f32,
        dt: f32,
    ) -> Result<(), RhiError> {
        // Cadence: faster steps when moving; a slow breathing rate when idle.
        let cycles_hz = if speed_mps > 0.05 { 0.8 + 0.10 * speed_mps } else { 0.25 };
        self.phase = (self.phase + std::f32::consts::TAU * cycles_hz * dt) % std::f32::consts::TAU;

        let bones = solve_pose(self.phase, speed_mps);
        for i in 0..self.rest.pos.len() {
            let b = self.rest.bone[i];
            let p = bones[b].transform_point3(Vec3::from(self.rest.pos[i]));
            let n = bones[b].transform_vector3(Vec3::from(self.rest.nrm[i])).normalize_or_zero();
            self.skinned_pos[i] = p.to_array();
            self.skinned_nrm[i] = n.to_array();
        }

        let fi = fi as usize;
        rhi.write_storage_bytes(self.pos_ring[fi], cast_slice(&self.skinned_pos))?;
        rhi.write_storage_bytes(self.nrm_ring[fi], cast_slice(&self.skinned_nrm))?;
        Ok(())
    }

    /// Draw the character at `feet` (world f64) facing `facing` (unit tangent).
    /// Call inside the opaque 3D pass, after terrain/Nanite and before the ocean.
    pub fn draw(
        &mut self,
        rhi: &mut Rhi,
        fi: u32,
        fu: &FrameUniforms,
        camera: &Camera,
        feet: DVec3,
        facing: Vec3,
    ) -> Result<(), RhiError> {
        // Local→world basis: +Y = surface up, +Z = facing, +X = up × forward
        // (right-handed, det +1 → winding preserved for cull-BACK).
        let up = feet.normalize().as_vec3();
        let forward = facing.normalize();
        let right = up.cross(forward).normalize();
        let basis = Mat4::from_cols(
            right.extend(0.0),
            up.extend(0.0),
            forward.extend(0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
        );
        let push = ChunkPush::camera_relative(feet, camera.position, basis, 0);

        let fidx = fi as usize;
        rhi.set_viewport_scissor_full(fi);
        rhi.bind_pipeline(fi, self.pipeline)?;
        rhi.update_frame_uniforms(fi, bytemuck::bytes_of(fu))?;
        rhi.bind_vertex_buffers(fi, &[self.pos_ring[fidx], self.nrm_ring[fidx], self.col, self.col])?;
        rhi.bind_index_buffer(fi, self.idx)?;
        rhi.push_constants(fi, bytemuck::bytes_of(&push))?;
        rhi.draw_indexed(fi, self.index_count);
        Ok(())
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Skin the rest mesh on the CPU (no GPU) for a given phase/speed.
    fn skin(phase: f32, speed: f32) -> Vec<Vec3> {
        let rig = build_rig();
        let bones = solve_pose(phase, speed);
        rig.pos
            .iter()
            .zip(&rig.bone)
            .map(|(p, &b)| bones[b].transform_point3(Vec3::from(*p)))
            .collect()
    }

    #[test]
    fn rig_is_well_formed() {
        let m = build_rig();
        assert_eq!(m.pos.len(), NUM_BONES * 24, "11 boxes × 24 verts");
        assert_eq!(m.idx.len(), NUM_BONES * 36, "11 boxes × 36 indices");
        let max = m.pos.len() as u32 - 1;
        assert!(m.idx.iter().all(|&i| i <= max), "index out of range");
        // parent < child everywhere (precondition for the top-down solve).
        for (b, (parent, _)) in BONES.iter().enumerate() {
            assert!(*parent < b as i32, "bone {b} parent {parent} not < child");
        }
    }

    #[test]
    fn rest_pose_stands_with_feet_near_origin() {
        let world = skin(0.0, 0.0);
        assert!(world.iter().all(|v| v.is_finite()), "rest pose not finite");
        let min_y = world.iter().map(|v| v.y).fold(f32::INFINITY, f32::min);
        let max_y = world.iter().map(|v| v.y).fold(f32::NEG_INFINITY, f32::max);
        // Feet at ~0, head near ~1.8 m — a standing figure rooted at the origin.
        assert!(min_y.abs() < 0.15, "feet should rest near y=0, got {min_y}");
        assert!((1.5..2.1).contains(&max_y), "head height {max_y} out of range");
    }

    #[test]
    fn walking_moves_the_limbs() {
        // At a mid-stride phase with speed, the pose must differ from rest.
        let rest = skin(0.0, 0.0);
        let mid = skin(std::f32::consts::FRAC_PI_2, WALK_SPEED);
        let moved = rest.iter().zip(&mid).any(|(a, b)| (*a - *b).length() > 0.05);
        assert!(moved, "limbs did not move while walking");
        assert!(mid.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn legs_swing_in_antiphase() {
        // The two thighs should swing oppositely (one forward, one back) mid-stride.
        let bones = solve_pose(std::f32::consts::FRAC_PI_2, WALK_SPEED);
        // A point at the knee in each thigh's local frame.
        let knee = Vec3::new(0.0, -UPPER_LEG_LEN, 0.0);
        let l = bones[UPPER_LEG_L].transform_point3(knee);
        let r = bones[UPPER_LEG_R].transform_point3(knee);
        // Forward is +Z; the knees should be on opposite sides in Z.
        assert!(l.z * r.z < 0.0, "thighs not in anti-phase: l.z={} r.z={}", l.z, r.z);
    }
}

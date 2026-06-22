//! `solar` — the sun + other solar-system bodies, drawn as real lit/emissive spheres.
//!
//! [`BodyRenderer`] draws every NON-focused body (the focused one is the detailed
//! Nanite planet) as a UV-sphere on the sky shell: the star emissive, planets
//! lambert-lit by their own sun direction. Mirrors the `Ocean` pipeline pattern (own
//! pipeline, standard set 0, alpha-blend / depth-write OFF) so the focused planet limb
//! eclipses far bodies for free under reversed-Z. All directions are rotated into the
//! focused planet's spun frame so the sky + lighting move together (day/night).

use bytemuck::{cast_slice, Pod, Zeroable};
use enki_render::{
    body_pass::BODY_WGSL,
    camera::Camera,
    frame::FrameUniforms,
    geometry::sphere_inward,
    sky::SKY_SHELL_M,
    system::{BodyKind, SolarSystem},
};
use enki_rhi::{vk, BufferHandle, PipelineHandle, Rhi, RhiError};
use glam::{Mat4, Quat, Vec3};

use crate::markers::overlay_pipeline;

/// Per-body push constant (96 B ≤ 128 B limit). Mirrors `Push` in body.wgsl.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BodyPush {
    model:   [[f32; 4]; 4],
    sun_dir: [f32; 4], // xyz: surface→sun (rendered frame); w: 1 = star, 0 = planet
    color:   [f32; 4],
}

/// Draws the sun + distant planets as lit spheres. Own pipeline + unit-sphere mesh;
/// consumes the engine set-0 frame UBO (same ring as terrain/ocean).
pub struct BodyRenderer {
    pipeline: PipelineHandle,
    pos: BufferHandle,
    nrm: BufferHandle,
    idx: BufferHandle,
    count: u32,
}

impl BodyRenderer {
    pub fn new(
        rhi: &mut Rhi,
        color_format: vk::Format,
        samples: vk::SampleCountFlags,
    ) -> Result<Self, RhiError> {
        let pipeline = overlay_pipeline(
            rhi,
            BODY_WGSL,
            color_format,
            std::mem::size_of::<BodyPush>() as u32,
            true, // alpha-blend, depth-write OFF
            samples,
        )?;

        // Unit sphere, inward-wound so cull-BACK keeps the camera-facing hemisphere.
        let mesh = sphere_inward(1.0, 48);
        let pos = rhi.create_vertex_buffer(cast_slice(&mesh.positions))?;
        let nrm = rhi.create_vertex_buffer(cast_slice(&mesh.normals))?;
        let idx = rhi.create_index_buffer(&mesh.indices)?;
        Ok(Self { pipeline, pos, nrm, idx, count: mesh.indices.len() as u32 })
    }

    /// Draw every non-focused body as a sphere on the sky shell, sized by its true
    /// angular radius. `spin` rotates heliocentric directions into the focused planet's
    /// spun frame for day/night (on/orbiting the planet); pass `Quat::IDENTITY` in free
    /// space so the sun stays fixed (heliocentric).
    pub fn record(
        &self,
        rhi: &mut Rhi,
        fi: u32,
        fu: &FrameUniforms,
        camera: &Camera,
        system: &SolarSystem,
        spin: Quat,
    ) -> Result<(), RhiError> {
        rhi.bind_pipeline(fi, self.pipeline)?;
        rhi.update_frame_uniforms(fi, bytemuck::bytes_of(fu))?;
        rhi.bind_vertex_buffers(fi, &[self.pos, self.nrm, self.nrm, self.nrm])?;
        rhi.bind_index_buffer(fi, self.idx)?;

        let q = spin;
        let focused_pos = system.focused().world_pos;
        let camera_world = focused_pos + camera.position; // camera in heliocentric f64
        let star_pos = system.star().world_pos;

        for (i, body) in system.bodies.iter().enumerate() {
            if i == system.focused {
                continue; // the focused body is the detailed Nanite planet at the origin
            }
            let to_body = body.world_pos - camera_world;
            let dist = to_body.length();
            if dist < 1.0 {
                continue;
            }
            // Rendered-frame direction + true angular radius; placed on the shell.
            let dir = (q * (to_body / dist).as_vec3()).normalize();
            let ar = (body.radius_m / dist).clamp(-1.0, 1.0).asin();
            // Sphere of true angular radius `ar` rendered at the shell: R = D·sin(ar).
            let r_render = (SKY_SHELL_M * ar.sin()) as f32;
            let center_rel = dir * (SKY_SHELL_M as f32);
            let model = Mat4::from_translation(center_rel) * Mat4::from_scale(Vec3::splat(r_render));

            let is_star = body.kind == BodyKind::Star;
            // The body's OWN sun direction (rotated into the rendered frame); star = unused.
            let sun = if is_star {
                Vec3::ZERO
            } else {
                (q * (star_pos - body.world_pos).as_vec3().normalize_or_zero()).normalize_or_zero()
            };
            let push = BodyPush {
                model: model.to_cols_array_2d(),
                sun_dir: [sun.x, sun.y, sun.z, if is_star { 1.0 } else { 0.0 }],
                color: [body.color[0], body.color[1], body.color[2], 1.0],
            };
            rhi.push_constants(fi, bytemuck::bytes_of(&push))?;
            rhi.draw_indexed(fi, self.count);
        }
        Ok(())
    }
}

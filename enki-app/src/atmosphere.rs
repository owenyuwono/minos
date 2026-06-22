//! `atmosphere` — the planet's translucent atmosphere shell.
//!
//! One alpha-blended sphere at `R + height`, shaded analytically (limb-dense
//! alpha + day-side + sun forward-glow) so the planet gets a visible halo and a
//! soft silhouette, and the wind streaks have an atmosphere to live in. Modeled
//! on the ocean shell ([`crate::ocean::Ocean`]): standard `set0` (FrameUniforms)
//! + `ChunkPush`, no custom descriptor set.
//!
//! ponytail: a single analytic shell, not a true scattering integral — port ki
//! `Atmosphere.ts` (Rayleigh/Mie) if the look needs physical accuracy.

use bytemuck::cast_slice;
use enki_render::{frame::FrameUniforms, geometry::placeholder_sphere, material::ChunkPush};
use enki_rhi::{vk, BufferHandle, GraphicsPipelineDesc, PipelineHandle, Rhi, RhiError};
use glam::{DVec3, Mat4, Vec3};

const ATMOSPHERE_WGSL: &str = include_str!("atmosphere.wgsl");

/// GUI-tunable atmosphere knobs.
#[derive(Debug, Clone, Copy)]
pub struct AtmoParams {
    /// Shell height above sea level (metres).
    pub height: f32,
    /// Overall opacity / haze density.
    pub density: f32,
}

impl Default for AtmoParams {
    fn default() -> Self {
        // ~5% of the 50 km radius — thin Earth-like air reads as ~1.6%, exaggerated
        // here so the halo is visible. Tunable.
        Self { height: 2500.0, density: 0.9 }
    }
}

/// Translucent atmosphere shell: one sphere, scaled to `R + height` each frame.
pub struct Atmosphere {
    pipeline: PipelineHandle,
    pos: BufferHandle,
    nrm: BufferHandle,
    idx: BufferHandle,
    count: u32,
    base_radius: f64,
}

impl Atmosphere {
    pub fn new(
        rhi: &mut Rhi,
        color_format: vk::Format,
        samples: vk::SampleCountFlags,
        base_radius: f64,
    ) -> Result<Self, RhiError> {
        let shader = rhi.create_shader_module(ATMOSPHERE_WGSL)?;
        let pipeline = rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            shader,
            vs_entry: "vs_main",
            fs_entry: "fs_main",
            push_constant_size: std::mem::size_of::<ChunkPush>() as u32,
            set0_layout: rhi.set0_layout(),
            color_format,
            depth_format: vk::Format::D32_SFLOAT,
            samples,
            blend: true, // alpha-blend, depth-write OFF
            fill: true,
        })?;
        rhi.destroy_shader_module(shader);

        let mesh = placeholder_sphere(base_radius as f32, 96);
        // `placeholder_sphere` winds CW-from-outside; reverse so cull-BACK keeps the
        // near (camera-facing) hemisphere — the halo + haze layer in front.
        let mut indices = mesh.indices.clone();
        for tri in indices.chunks_exact_mut(3) {
            tri.swap(0, 2);
        }
        let pos = rhi.create_vertex_buffer(cast_slice(&mesh.positions))?;
        let nrm = rhi.create_vertex_buffer(cast_slice(&mesh.normals))?;
        let idx = rhi.create_index_buffer(&indices)?;

        Ok(Self { pipeline, pos, nrm, idx, count: mesh.indices.len() as u32, base_radius })
    }

    /// Draw the shell at `base_radius + params.height`.
    pub fn record(
        &self,
        rhi: &mut Rhi,
        fi: u32,
        fu: &FrameUniforms,
        camera_pos: DVec3,
        params: AtmoParams,
    ) -> Result<(), RhiError> {
        rhi.bind_pipeline(fi, self.pipeline)?;
        rhi.update_frame_uniforms(fi, bytemuck::bytes_of(fu))?;

        let scale = ((self.base_radius + params.height as f64) / self.base_radius) as f32;
        let model = Mat4::from_scale(Vec3::splat(scale));
        let mut push = ChunkPush::camera_relative(DVec3::ZERO, camera_pos, model, 0);
        // Smuggle density into the first pad slot (read as f32 by atmosphere.wgsl).
        push._pad[0] = params.density.to_bits();

        // Only position (0) + normal (1) are read; slots 2/3 alias the normal buffer.
        rhi.bind_vertex_buffers(fi, &[self.pos, self.nrm, self.nrm, self.nrm])?;
        rhi.bind_index_buffer(fi, self.idx)?;
        rhi.push_constants(fi, bytemuck::bytes_of(&push))?;
        rhi.draw_indexed(fi, self.count);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn atmosphere_wgsl_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(super::ATMOSPHERE_WGSL)
            .unwrap_or_else(|e| panic!("atmosphere.wgsl failed to parse: {e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("atmosphere.wgsl failed to validate: {e:?}"));
    }
}

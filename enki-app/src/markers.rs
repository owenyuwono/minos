//! `markers` — reference markers on the globe: the equator ring + north/south
//! pole spikes, drawn as camera-facing ribbons (constant screen width) so they
//! read as crisp lines at any distance. Depth-tested, so the globe hides the far
//! side. Each group toggles independently. Geometry is a fixed template; only the
//! camera-facing offset is recomputed per frame (cheap — a few hundred verts).

use bytemuck::cast_slice;
use enki_render::{frame::FrameUniforms, material::ChunkPush};
use enki_rhi::{vk, BufferHandle, GraphicsPipelineDesc, PipelineHandle, Rhi, RhiError};
use glam::{DVec3, Mat4, Vec3};

const MARKERS_WGSL: &str = include_str!("markers.wgsl");

/// Equator ring resolution (segments).
const EQUATOR_SEGS: usize = 128;
/// Pole spike length above the surface (metres) — pokes clear of the atmosphere.
const SPIKE_LEN: f32 = 4000.0;
/// Equator ring height above sea level (metres).
const EQUATOR_OFFSET: f32 = 300.0;
/// Ribbon half-width as a fraction of camera distance (≈ screen-constant px).
const WIDTH: f32 = 0.003;

const COL_EQUATOR: [f32; 3] = [0.25, 0.95, 0.40]; // green
const COL_NORTH: [f32; 3] = [0.95, 0.28, 0.22]; // red
const COL_SOUTH: [f32; 3] = [0.32, 0.55, 0.98]; // blue

/// One template vertex of a marker polyline.
struct MVert {
    /// Unit surface direction.
    dir: DVec3,
    /// Radial offset above sea level (metres).
    offset: f32,
    /// World-space polyline tangent (static, unit) for the ribbon cross-section.
    tangent: DVec3,
    color: [f32; 3],
    /// In the poles group (vs. the equator group) — selects which toggle gates it.
    poles: bool,
}

pub struct Markers {
    pipeline: PipelineHandle,
    pos_dyn: Vec<BufferHandle>,
    col_dyn: Vec<BufferHandle>,
    idx: BufferHandle,
    idx_count: u32,
    template: Vec<MVert>,
    pos_scratch: Vec<[f32; 3]>,
    col_scratch: Vec<[f32; 3]>,
    base_radius: f64,
}

impl Markers {
    pub fn new(
        rhi: &mut Rhi,
        color_format: vk::Format,
        samples: vk::SampleCountFlags,
        base_radius: f64,
    ) -> Result<Self, RhiError> {
        let shader = rhi.create_shader_module(MARKERS_WGSL)?;
        let pipeline = rhi.create_graphics_pipeline(&GraphicsPipelineDesc {
            shader,
            vs_entry: "vs_main",
            fs_entry: "fs_main",
            push_constant_size: std::mem::size_of::<ChunkPush>() as u32,
            set0_layout: rhi.set0_layout(),
            color_format,
            depth_format: vk::Format::D32_SFLOAT,
            samples,
            // Opaque (alpha 1) so depth-write is ON: the markers write depth in the
            // opaque pass, so the later water pass fails its depth test over them and
            // they show through the sea (instead of being painted over).
            blend: false,
            fill: true,
        })?;
        rhi.destroy_shader_module(shader);

        let (template, segments) = build_template();
        let indices = build_indices(&segments);
        let idx_count = indices.len() as u32;
        let idx = rhi.create_index_buffer(&indices)?;

        let vert_count = template.len() * 2;
        let fif = rhi.frames_in_flight();
        let vbuf_size = (vert_count * std::mem::size_of::<[f32; 3]>()) as u64;
        let mut pos_dyn = Vec::with_capacity(fif);
        let mut col_dyn = Vec::with_capacity(fif);
        for _ in 0..fif {
            pos_dyn.push(rhi.create_gpu_buffer(vbuf_size, true, vk::BufferUsageFlags::VERTEX_BUFFER)?);
            col_dyn.push(rhi.create_gpu_buffer(vbuf_size, true, vk::BufferUsageFlags::VERTEX_BUFFER)?);
        }

        Ok(Self {
            pipeline,
            pos_dyn,
            col_dyn,
            idx,
            idx_count,
            pos_scratch: Vec::with_capacity(vert_count),
            col_scratch: Vec::with_capacity(vert_count),
            template,
            base_radius,
        })
    }

    /// Draw the enabled marker groups. Call inside the opaque MSAA instance.
    pub fn record(
        &mut self,
        rhi: &mut Rhi,
        fi: u32,
        fu: &FrameUniforms,
        camera_pos: DVec3,
        sea_level_m: f64,
        poles: bool,
        equator: bool,
    ) -> Result<(), RhiError> {
        if !poles && !equator {
            return Ok(());
        }
        let sea_radius = self.base_radius + sea_level_m;
        self.build(camera_pos, sea_radius, poles, equator);

        let fi_u = fi as usize;
        rhi.write_storage_bytes(self.pos_dyn[fi_u], cast_slice(&self.pos_scratch))?;
        rhi.write_storage_bytes(self.col_dyn[fi_u], cast_slice(&self.col_scratch))?;

        rhi.bind_pipeline(fi, self.pipeline)?;
        rhi.update_frame_uniforms(fi, bytemuck::bytes_of(fu))?;
        let push = ChunkPush::camera_relative(camera_pos, camera_pos, Mat4::IDENTITY, 0);
        let p = self.pos_dyn[fi_u];
        let c = self.col_dyn[fi_u];
        rhi.bind_vertex_buffers(fi, &[p, p, c, p])?;
        rhi.bind_index_buffer(fi, self.idx)?;
        rhi.push_constants(fi, bytemuck::bytes_of(&push))?;
        rhi.draw_indexed(fi, self.idx_count);
        Ok(())
    }

    fn build(&mut self, camera_pos: DVec3, sea_radius: f64, poles: bool, equator: bool) {
        self.pos_scratch.clear();
        self.col_scratch.clear();
        for v in &self.template {
            let enabled = if v.poles { poles } else { equator };
            let world = v.dir * (sea_radius + v.offset as f64);
            let rel = (world - camera_pos).as_vec3();
            let dist = rel.length();
            let view = (-rel).normalize_or_zero();
            let t = v.tangent.as_vec3();
            let mut off = Vec3::ZERO;
            if enabled && dist > 1e-4 && t.length() > 1e-7 {
                let o = t.normalize().cross(view);
                if o.length() > 1e-7 {
                    off = o.normalize() * (WIDTH * dist);
                }
            }
            let a = rel - off;
            let b = rel + off;
            self.pos_scratch.push([a.x, a.y, a.z]);
            self.col_scratch.push(v.color);
            self.pos_scratch.push([b.x, b.y, b.z]);
            self.col_scratch.push(v.color);
        }
    }
}

/// Build the marker template + the segment list (point-index pairs, no segment
/// crossing between polylines).
fn build_template() -> (Vec<MVert>, Vec<(u32, u32)>) {
    let mut t = Vec::new();
    let mut segs = Vec::new();

    // Equator ring in the XZ plane (Y is the polar axis). Closed (last == first).
    let start = t.len() as u32;
    for k in 0..=EQUATOR_SEGS {
        let th = (k as f64 / EQUATOR_SEGS as f64) * std::f64::consts::TAU;
        t.push(MVert {
            dir: DVec3::new(th.cos(), 0.0, th.sin()),
            offset: EQUATOR_OFFSET,
            tangent: DVec3::new(-th.sin(), 0.0, th.cos()),
            color: COL_EQUATOR,
            poles: false,
        });
    }
    for k in 0..EQUATOR_SEGS as u32 {
        segs.push((start + k, start + k + 1));
    }

    // North + south pole spikes (radial, sticking out past the atmosphere).
    for (axis, color) in [(DVec3::Y, COL_NORTH), (DVec3::NEG_Y, COL_SOUTH)] {
        let base = t.len() as u32;
        t.push(MVert { dir: axis, offset: 0.0, tangent: axis, color, poles: true });
        t.push(MVert { dir: axis, offset: SPIKE_LEN, tangent: axis, color, poles: true });
        segs.push((base, base + 1));
    }

    (t, segs)
}

/// Double-sided triangle-list indices for each ribbon segment (point pair).
fn build_indices(segments: &[(u32, u32)]) -> Vec<u32> {
    let mut idx = Vec::with_capacity(segments.len() * 12);
    for &(a, b) in segments {
        let (va, vb) = (a * 2, b * 2);
        idx.extend_from_slice(&[
            va, va + 1, vb, vb, va + 1, vb + 1, // front
            va, vb, va + 1, vb, vb + 1, va + 1, // back
        ]);
    }
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn template_and_indices_consistent() {
        let (t, segs) = build_template();
        // 129 equator points + 2×2 spike points.
        assert_eq!(t.len(), EQUATOR_SEGS + 1 + 4);
        let idx = build_indices(&segs);
        assert_eq!(idx.len(), segs.len() * 12);
        let vert_count = (t.len() * 2) as u32;
        assert!(*idx.iter().max().unwrap() < vert_count);
        // Unit directions.
        for v in &t {
            assert!((v.dir.length() - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn markers_wgsl_validates_with_naga() {
        let module = naga::front::wgsl::parse_str(MARKERS_WGSL)
            .unwrap_or_else(|e| panic!("markers.wgsl failed to parse: {e:?}"));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("markers.wgsl failed to validate: {e:?}"));
    }
}

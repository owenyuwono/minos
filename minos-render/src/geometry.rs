//! Geometry contract — the single bridge between the streaming uploader and the
//! geometry provider (placeholder or future terrain). Both sides depend only on
//! this file; neither knows the other's internals.

/// All vertex and index data needed to upload one terrain chunk to the GPU.
///
/// Layout rules (enforced by the uploader, not here):
/// - `positions`, `normals`, `colors` are parallel arrays — index N in each refers
///   to the same vertex.
/// - `plate_colors` is optional; absent means the material variant doesn't use it.
/// - `indices` are triangle-list, CCW winding, 32-bit.
/// - `origin` is in planet-space world coordinates (double precision so chunks far
///   from the origin don't lose precision before the camera-relative transform).
#[derive(Debug, Clone)]
pub struct ChunkMeshArrays {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 3]>,
    /// Per-vertex tectonic plate color. `None` when the plate-color view variant
    /// is not active, to avoid allocating the buffer unnecessarily.
    pub plate_colors: Option<Vec<[f32; 3]>>,
    pub indices: Vec<u32>,
    /// Planet-space origin of this chunk (double precision). The GPU shader
    /// receives a camera-relative float offset computed on the CPU.
    pub origin: glam::DVec3,
}

// ── Placeholder geometry ─────────────────────────────────────────────────────

/// UV sphere with outward normals and an elevation/latitude color gradient.
///
/// `radius` controls the sphere size (use ~50_000 for planet-scale tests).
/// `subdivisions` is the number of latitude and longitude rings; minimum clamped
/// to 4 to avoid degenerate meshes. Reasonable values: 32–64.
///
/// Winding: CCW viewed from outside (right-hand outward normal).
///
/// Color: gradient from deep blue at poles to warm green/brown at equator,
/// giving a clear indication of shading direction when lit.
///
/// `origin` is set to `DVec3::ZERO` — the app positions the sphere by setting
/// `ChunkPush.model` to a translation matrix.
pub fn placeholder_sphere(radius: f32, subdivisions: u32) -> ChunkMeshArrays {
    let stacks = subdivisions.max(4);   // latitude rings
    let slices = (subdivisions * 2).max(8); // longitude segments

    let vert_count = ((stacks + 1) * (slices + 1)) as usize;
    let mut positions = Vec::with_capacity(vert_count);
    let mut normals = Vec::with_capacity(vert_count);
    let mut colors = Vec::with_capacity(vert_count);

    // Build vertex grid (stacks + 1 rows × slices + 1 columns)
    for stack in 0..=stacks {
        let phi = std::f32::consts::PI * stack as f32 / stacks as f32; // 0..PI (north to south)
        let sin_phi = phi.sin();
        let cos_phi = phi.cos();

        // t ∈ [0,1]: 0 = north pole, 0.5 = equator, 1 = south pole
        let t = stack as f32 / stacks as f32;

        for slice in 0..=slices {
            let theta = 2.0 * std::f32::consts::PI * slice as f32 / slices as f32;
            let x = sin_phi * theta.cos();
            let y = cos_phi;
            let z = sin_phi * theta.sin();

            // Normal = unit sphere point (outward)
            normals.push([x, y, z]);
            positions.push([x * radius, y * radius, z * radius]);

            // Latitude gradient: poles (t≈0 or t≈1) deep blue,
            // equator (t≈0.5) warm earthy green/brown.
            let equator_blend = 1.0 - (2.0 * t - 1.0).abs(); // 0 at poles, 1 at equator
            let pole_col = [0.12f32, 0.25, 0.70]; // deep blue
            let equator_col = [0.28f32, 0.55, 0.22]; // green
            let r = pole_col[0] + equator_blend * (equator_col[0] - pole_col[0]);
            let g = pole_col[1] + equator_blend * (equator_col[1] - pole_col[1]);
            let b = pole_col[2] + equator_blend * (equator_col[2] - pole_col[2]);
            colors.push([r, g, b]);
        }
    }

    // Build triangle list from the grid
    let tri_count = (stacks * slices * 2) as usize;
    let mut indices = Vec::with_capacity(tri_count * 3);
    let row_len = slices + 1;

    for stack in 0..stacks {
        for slice in 0..slices {
            let tl = stack * row_len + slice;
            let tr = tl + 1;
            let bl = tl + row_len;
            let br = bl + 1;

            // CCW winding viewed from outside:
            // Upper triangle
            indices.push(tl);
            indices.push(br);
            indices.push(tr);
            // Lower triangle
            indices.push(tl);
            indices.push(bl);
            indices.push(br);
        }
    }

    ChunkMeshArrays {
        positions,
        normals,
        colors,
        plate_colors: None,
        indices,
        origin: glam::DVec3::ZERO,
    }
}

/// UV sphere with INWARD-facing normals — the same geometry as
/// [`placeholder_sphere`] but with reversed triangle winding (so cull-BACK keeps
/// the inside-facing surface). For shell volumes seen from inside (ocean/atmosphere).
pub fn sphere_inward(radius: f32, subdivisions: u32) -> ChunkMeshArrays {
    let mut mesh = placeholder_sphere(radius, subdivisions);
    for tri in mesh.indices.chunks_exact_mut(3) {
        tri.swap(0, 2);
    }
    mesh
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn validate_mesh(mesh: &ChunkMeshArrays, label: &str) {
        let n = mesh.positions.len();
        assert!(n > 0, "{label}: no vertices");
        assert_eq!(mesh.normals.len(), n, "{label}: normals/positions mismatch");
        assert_eq!(mesh.colors.len(), n, "{label}: colors/positions mismatch");
        assert!(!mesh.indices.is_empty(), "{label}: no indices");
        assert!(
            mesh.indices.len().is_multiple_of(3),
            "{label}: index count not a multiple of 3"
        );

        // All indices must be in range.
        let max_idx = n as u32 - 1;
        for &idx in &mesh.indices {
            assert!(
                idx <= max_idx,
                "{label}: index {idx} out of range (max {max_idx})"
            );
        }

        // All normals must be approximately unit length.
        for (i, &nrm) in mesh.normals.iter().enumerate() {
            let len = (nrm[0] * nrm[0] + nrm[1] * nrm[1] + nrm[2] * nrm[2]).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-4,
                "{label}: normal[{i}] has length {len:.6}, expected ~1.0"
            );
        }
    }

    #[test]
    fn sphere_mesh_valid() {
        let mesh = placeholder_sphere(50_000.0, 16);
        validate_mesh(&mesh, "sphere(16)");
    }

    #[test]
    fn sphere_mesh_min_subdivision() {
        let mesh = placeholder_sphere(1.0, 1); // should clamp to 4
        validate_mesh(&mesh, "sphere(1→4)");
    }

    #[test]
    fn sphere_inward_reverses_winding() {
        let out = placeholder_sphere(50_000.0, 16);
        let inw = sphere_inward(50_000.0, 16);
        validate_mesh(&inw, "sphere_inward(16)");
        assert_eq!(inw.indices.len(), out.indices.len());
        // Each triangle's winding is reversed (first/last swapped).
        for tri in 0..out.indices.len() / 3 {
            let b = tri * 3;
            assert_eq!(inw.indices[b], out.indices[b + 2]);
            assert_eq!(inw.indices[b + 1], out.indices[b + 1]);
            assert_eq!(inw.indices[b + 2], out.indices[b]);
        }
    }
}

//! enki-voxel — on-demand voxel/SDF terrain mesher (transvoxel).
//!
//! Self-contained, feature-gated (`voxel`) in enki-app, deletable — mirrors the
//! enki-nanite / enki-flora modularity contract (lower crates never reference it).
//!
//! **Phase 1 scope:** mesh ONE static cubic chunk of the planet surface from a
//! density field derived from the existing [`HeightField`]. No octree, no
//! streaming, no caves, no edits yet (Phases 2-4).
//!
//! ## Density
//! `density(p) = surface_radius(p̂) − |p|` — positive *below* the surface (solid),
//! zero *at* the surface, negative *above* (air). This is the radial generalization
//! of the flat-terrain `sdf = y − height(x,z)`; the transvoxel isosurface (threshold
//! 0) reproduces the heightfield surface exactly when there are no caves/edits.
//!
//! ## Precision
//! transvoxel meshes a block in `f32` *local* coordinates, but the density closure
//! converts local→world in **f64** before sampling, so the planet-radius magnitude
//! (~50 km) never touches f32 — no catastrophic cancellation. The returned
//! [`ChunkMeshArrays::positions`] are therefore LOCAL block coords; the caller draws
//! them camera-relative via `ChunkPush::camera_relative(origin, camera, …)` with
//! `origin == center_world`, exactly like the classic quadtree chunks.

use enki_planet::height::HeightField;
use enki_planet::noise::Noise3D;
use enki_planet::ChunkMeshArrays;
use glam::DVec3;
use transvoxel::prelude::*;

/// Re-exported so callers (the Phase 2 octree) can mark which block faces border a
/// coarser-LOD neighbour without depending on `transvoxel` directly. A flagged face
/// gets transvoxel transition cells → the seam to the coarser neighbour is crack-free.
pub use transvoxel::structs::transition_sides::{TransitionSide, TransitionSides};

fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Cave tuning. Caves carve air pockets BELOW the surface where 3-D noise is high
/// and we're at least `margin` metres deep — so the surface stays intact (no holes)
/// except where the player digs in. All distances in metres. ponytail: USER-tuned
/// in the GUI; these defaults are a starting point, not load-bearing.
#[derive(Clone, Copy, Debug)]
pub struct CaveParams {
    /// Max density reduction (≈ cave radius scale, m).
    pub strength: f64,
    /// Cave noise spatial frequency (1/m). Smaller = bigger caverns.
    pub freq: f64,
    /// Surface-protect depth (m): no carving within this of the surface.
    pub margin: f64,
    /// Smoothstep ramp width below `margin` (m).
    pub falloff: f64,
    /// Noise threshold 0..1 — only carve where the noise exceeds it (lower = more caves).
    pub threshold: f64,
    pub seed: u32,
}

impl Default for CaveParams {
    fn default() -> Self {
        Self { strength: 90.0, freq: 0.004, margin: 8.0, falloff: 60.0, threshold: 0.55, seed: 1234 }
    }
}

/// A player terrain edit: a sphere CSG op on the density field. `dig` carves air
/// inside the sphere; otherwise it fills solid. Edits are world-space and shared by
/// all leaves, so a sphere straddling a seam stays crack-free (both leaves apply it).
#[derive(Clone, Copy, Debug)]
pub struct Edit {
    pub center: DVec3,
    pub radius: f64,
    pub dig: bool,
}

/// A cave density source: 3-D noise + [`CaveParams`]. Held by the terrain manager.
pub struct CaveField {
    noise: Noise3D,
    pub params: CaveParams,
}

impl CaveField {
    pub fn new(params: CaveParams) -> Self {
        Self { noise: Noise3D::new(params.seed), params }
    }

    /// 3 octaves of fbm in ~[0,1] at a world point.
    fn fbm(&self, world: DVec3) -> f64 {
        let mut v = 0.0;
        let mut a = 0.5;
        let mut f = self.params.freq;
        for _ in 0..3 {
            v += a * (self.noise.sample(world.x * f, world.y * f, world.z * f) * 0.5 + 0.5);
            a *= 0.5;
            f *= 2.0;
        }
        v
    }

    /// Carve amount (≥0) at a world point; `base = surface_r − |world|` (m below
    /// surface). Subtracting this from `base` opens air where deep + noisy.
    fn carve(&self, world: DVec3, base: f64) -> f64 {
        if self.params.strength <= 0.0 {
            return 0.0;
        }
        let mask = smoothstep(self.params.margin, self.params.margin + self.params.falloff, base);
        if mask <= 0.0 {
            return 0.0;
        }
        let thr = self.params.threshold;
        let c = ((self.fbm(world) - thr) / (1.0 - thr).max(1e-3)).clamp(0.0, 1.0);
        mask * self.params.strength * c
    }
}

/// Surface radius (metres) at a unit direction: base radius + amplified height.
#[inline]
pub fn surface_radius(
    hf: &dyn HeightField,
    dir: DVec3,
    radius: f64,
    height_scale: f64,
    level: u8,
) -> f64 {
    radius + hf.height(dir, level) * height_scale
}

/// Signed density at a world point. `> 0` solid (below surface), `0` surface,
/// `< 0` air. Evaluated in f64 so planet-scale magnitudes never hit f32.
#[inline]
pub fn density(
    world: DVec3,
    hf: &dyn HeightField,
    radius: f64,
    height_scale: f64,
    level: u8,
) -> f64 {
    let r = world.length();
    if r == 0.0 {
        // Planet centre: deep solid. Avoids normalize() of a zero vector.
        return radius;
    }
    surface_radius(hf, world / r, radius, height_scale, level) - r
}

/// Mesh one cubic block with NO transition cells (all neighbours at the same LOD).
/// Convenience wrapper over [`mesh_block`] — used by the Phase 1 static chunk.
pub fn mesh_chunk(
    hf: &dyn HeightField,
    center_world: DVec3,
    size: f64,
    subdivisions: usize,
    radius: f64,
    height_scale: f64,
    level: u8,
) -> ChunkMeshArrays {
    mesh_block(
        hf, center_world, size, subdivisions, radius, height_scale, level,
        TransitionSide::none(),
    )
}

/// Mesh one cubic block centred on `center_world` (world f64), edge `size` metres,
/// `subdivisions` cells per axis. `transitions` flags the faces that border a
/// COARSER-LOD neighbour — transvoxel emits transition cells there so the seam is
/// crack-free (Phase 2 computes the flags from the quadtree neighbour levels).
/// Returns camera-relative-ready [`ChunkMeshArrays`] whose `origin` is `center_world`
/// and whose `positions` are LOCAL block coords.
///
/// `level` is the heightfield LOD (more octaves at higher levels); pick it to match
/// the cell size (`size / subdivisions`) — it's a tuning knob, not load-bearing.
pub fn mesh_block(
    hf: &dyn HeightField,
    center_world: DVec3,
    size: f64,
    subdivisions: usize,
    radius: f64,
    height_scale: f64,
    level: u8,
    transitions: TransitionSides,
) -> ChunkMeshArrays {
    let half = (size * 0.5) as f32;

    // local (f32, small) -> world (f64) -> density (f64) -> f32. The big magnitude
    // lives only in `center_world` (f64); local stays near the origin.
    let field = |lx: f32, ly: f32, lz: f32| -> f32 {
        let world = center_world + DVec3::new(lx as f64, ly as f64, lz as f64);
        density(world, hf, radius, height_scale, level) as f32
    };

    let block = Block::new([-half, -half, -half], size as f32, subdivisions);
    let builder = extract_from_field(
        &field,
        FieldCaching::CacheNothing,
        block,
        transitions,
        0f32,
        GenericMeshBuilder::new(),
    );
    let mesh = builder.build();

    let n_verts = mesh.positions.len() / 3;
    let mut positions = Vec::with_capacity(n_verts);
    let mut normals = Vec::with_capacity(n_verts);
    for v in 0..n_verts {
        let b = v * 3;
        positions.push([mesh.positions[b], mesh.positions[b + 1], mesh.positions[b + 2]]);
        // transvoxel derives normals from the density gradient. Our field is
        // positive-inside like its README sphere example, so its normals point
        // outward already. ponytail: if lighting looks INVERTED in-app, negate
        // these three components — that's the whole fix.
        normals.push([mesh.normals[b], mesh.normals[b + 1], mesh.normals[b + 2]]);
    }
    let indices: Vec<u32> = mesh.triangle_indices.iter().map(|&i| i as u32).collect();

    // ponytail: flat debug grey — biome/plate color comes in a later phase. Lit
    // shading (material_mode 0) still reveals the surface relief via the normals.
    let colors = vec![[0.55_f32, 0.55, 0.55]; n_verts];
    let plate_colors = Some(colors.clone());

    ChunkMeshArrays {
        positions,
        normals,
        colors,
        plate_colors,
        indices,
        origin: center_world,
    }
}

/// Outward normal = −∇(density), the gradient of the SCALAR FIELD (IQ 4-tap
/// tetrahedron). Correct for any isosurface — including cave overhangs where the
/// surface isn't a radial graph — and shared-vertex-consistent across leaf seams
/// (same world point → same gradient → no crease). `h` (metres) must be f64-safe at
/// planet scale; `outward_ref` (radial up) only rescues the degenerate flat case.
fn gradient_normal<F: Fn(DVec3) -> f64>(f: &F, p: DVec3, outward_ref: DVec3, h: f64) -> DVec3 {
    let kxyy = DVec3::new(1.0, -1.0, -1.0);
    let kyyx = DVec3::new(-1.0, -1.0, 1.0);
    let kyxy = DVec3::new(-1.0, 1.0, -1.0);
    let kxxx = DVec3::new(1.0, 1.0, 1.0);
    let g = kxyy * f(p + kxyy * h)
        + kyyx * f(p + kyyx * h)
        + kyxy * f(p + kyxy * h)
        + kxxx * f(p + kxxx * h);
    // Density grows INTO solid, so the outward normal is −∇density.
    let n = (-g).normalize_or_zero();
    if n.length_squared() < 0.5 {
        outward_ref.normalize_or_zero()
    } else {
        n
    }
}

/// Mesh one cube-sphere quadtree leaf `(face, level, ix, iy)` as a transvoxel block
/// in cube-face PARAMETER space `(cu, cv, r)`. Because the block is meshed in the
/// shared `(cu, cv)` parameterization (not a per-leaf tangent box), same-level
/// neighbours share the exact edge and stay crack-free; `transitions` flags faces
/// bordering a COARSER neighbour so transvoxel transition cells bridge the 2× jump.
///
/// `subdivisions` = cells per axis. Returns camera-relative-ready [`ChunkMeshArrays`]
/// (positions = world − leaf-centre origin; analytic dir-normals).
#[allow(clippy::too_many_arguments)]
pub fn mesh_leaf(
    hf: &dyn HeightField,
    face: u8,
    level: u8,
    ix: u32,
    iy: u32,
    radius: f64,
    height_scale: f64,
    subdivisions: usize,
    transitions: TransitionSides,
    caves: Option<&CaveField>,
    edits: &[Edit],
    // Data debug view: 0 = normal biome color; 1 Height / 2 Material / 3 Wetness /
    // 4 Volcano bake that heightfield channel into vertex color instead (terrain*.wgsl
    // shows it raw for view modes 7–10). 0 is byte-identical to the pre-debug mesher.
    dbg_field: u8,
) -> ChunkMeshArrays {
    use enki_planet::face_bases::{cube_to_sphere, patch_center_dir, FACE_BASES};

    let b = &FACE_BASES[face as usize];
    let scale = 1.0 / (1u64 << level) as f64;
    // Cube-face [-1,1] extent of this leaf.
    let cu0 = ix as f64 * scale * 2.0 - 1.0;
    let cu1 = (ix as f64 + 1.0) * scale * 2.0 - 1.0;
    let cv0 = iy as f64 * scale * 2.0 - 1.0;
    let cv1 = (iy as f64 + 1.0) * scale * 2.0 - 1.0;
    let dir_of = |cu: f64, cv: f64| -> DVec3 { cube_to_sphere(b.n + b.u * cu + b.v * cv).normalize() };

    // Radial band on a GLOBAL per-level lattice (radial_cell depends only on `level`,
    // never the leaf) so adjacent same-level leaves share radial samples on their
    // common face → crack-free even with caves (where density is NOT linear in r).
    let node_size = std::f64::consts::FRAC_PI_2 * radius * scale; // patch width (m)
    // Radial cell = node-size-derived at fine levels, but CAPPED to the terrain relief
    // (~height_scale) at coarse levels. Without the cap the band is ~1.5·node_size —
    // tens of km at coarse LOD, which would extend through r=0 and sample NaN
    // directions (`world/|world|`), crashing the mesher. The cap is a function of
    // `level` only, so same-level neighbours still share the radial lattice (caves
    // stay crack-free). 3·height_scale comfortably brackets the full ±height_scale relief.
    let radial_cell = (1.5 * node_size).min(3.0 * height_scale) / subdivisions as f64;
    let band = radial_cell * subdivisions as f64;
    let (mut r_min, mut r_max) = (f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..=8 {
        for j in 0..=8 {
            let cu = cu0 + (cu1 - cu0) * (i as f64 / 8.0);
            let cv = cv0 + (cv1 - cv0) * (j as f64 / 8.0);
            let sr = radius + hf.height(dir_of(cu, cv), level) * height_scale;
            r_min = r_min.min(sr);
            r_max = r_max.max(sr);
        }
    }
    // Snap the band base to the global lattice (multiples of radial_cell), centred on
    // the patch's surface mid so the band brackets the relief.
    let r_mid = 0.5 * (r_min + r_max);
    let r_lo = ((r_mid - 0.5 * band) / radial_cell).floor() * radial_cell;
    let r_hi = r_lo + band;

    let center_dir = patch_center_dir(face, level, ix, iy);
    let origin = center_dir * (radius + hf.height(center_dir, level) * height_scale);

    // World-space signed density (with optional caves). >0 below the surface (solid).
    let density_at = |world: DVec3| -> f64 {
        let r = world.length();
        if r < 1.0 {
            return radius; // near the planet centre: deep solid; avoids dir = world/0 = NaN
        }
        let dir = world / r;
        let base = radius + hf.height(dir, level) * height_scale - r;
        let mut dens = match caves {
            Some(cf) => base - cf.carve(world, base),
            None => base,
        };
        // Player edits: sphere CSG. dig → air inside (min with the sphere SDF);
        // add → solid inside (max with the negated SDF). Both keep seams crack-free
        // because every leaf applies the same world-space edits.
        for e in edits {
            let sd = world.distance(e.center) - e.radius;
            if e.dig {
                dens = dens.min(sd);
            } else {
                dens = dens.max(-sd);
            }
        }
        dens
    };
    let world_of = |cu: f64, cv: f64, r: f64| -> DVec3 { dir_of(cu, cv) * r };
    // Density in block-local [0,1]^3 → (cu,cv,r) → world.
    let field = |lx: f32, ly: f32, lz: f32| -> f32 {
        let cu = cu0 + (cu1 - cu0) * lx as f64;
        let cv = cv0 + (cv1 - cv0) * ly as f64;
        let r = r_lo + (r_hi - r_lo) * lz as f64;
        density_at(world_of(cu, cv, r)) as f32
    };
    let block = Block::new([0.0_f32, 0.0, 0.0], 1.0, subdivisions);
    let mesh = extract_from_field(
        &field,
        FieldCaching::CacheNothing,
        block,
        transitions,
        0f32,
        GenericMeshBuilder::new(),
    )
    .build();

    let step = (radial_cell * 0.25).max(0.05); // finite-diff step (m), f64-safe at planet scale
    let n_verts = mesh.positions.len() / 3;
    let mut positions = Vec::with_capacity(n_verts);
    let mut normals = Vec::with_capacity(n_verts);
    let mut colors = Vec::with_capacity(n_verts);
    let mut plate = Vec::with_capacity(n_verts);
    for v in 0..n_verts {
        let lx = mesh.positions[v * 3] as f64;
        let ly = mesh.positions[v * 3 + 1] as f64;
        let lz = mesh.positions[v * 3 + 2] as f64;
        let cu = cu0 + (cu1 - cu0) * lx;
        let cv = cv0 + (cv1 - cv0) * ly;
        let r = r_lo + (r_hi - r_lo) * lz;
        let dir = dir_of(cu, cv);
        let world = dir * r;
        let normal = gradient_normal(&density_at, world, dir, step);
        positions.push((world - origin).as_vec3().to_array());
        normals.push(normal.as_vec3().to_array());

        // Biome color — same recipe as the classic mesher (enki_planet::mesher):
        // slope from the normal, climate from the heightfield, biome ramp keyed on
        // the RAW height (∈[-1,1]; <0 ocean floor, >0 land), not the radius.
        let height = hf.height(dir, level);
        let (temp, moisture) = hf.climate(dir, height);
        let slope = (1.0 - normal.dot(dir)).clamp(0.0, 1.0) as f32;
        // Data debug views bake one heightfield channel into vertex color (terrain*.wgsl
        // shows it raw for modes 7–10); 0 = biome color. A jet colormap (blue→red) reveals
        // narrow-band + sparse fields a grayscale ramp washed out: t=0 is blue (not black),
        // so zero areas stay visible and rivers/cones pop. Height also gets 40 m contour
        // bands (hue = global elevation, bands = local relief, visible at any zoom).
        // ponytail: hardcoded gain/contour spacing — baked per-vertex, so a GUI slider
        // would remesh on every drag; tune the constants here if a scale reads wrong.
        let jet = |t: f32| {
            let t = t.clamp(0.0, 1.0);
            [
                (1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0),
                (1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0),
                (1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0),
            ]
        };
        colors.push(match dbg_field {
            1 => {
                let hue = jet((height as f32 * 2.0 + 0.5).clamp(0.0, 1.0));
                let band = 0.55 + 0.45 * (height as f32 * height_scale as f32 / 40.0).fract().abs();
                [hue[0] * band, hue[1] * band, hue[2] * band]
            }
            2 => jet(hf.material(dir)),         // rock hardness 0..1 (full range)
            3 => jet(hf.wetness(dir).sqrt()),   // river mask — sparse; sqrt lifts thin lines
            4 => jet(hf.volcanism(dir).sqrt()), // volcano cones — sparse
            _ => enki_planet::coloring::biome_color(temp, moisture, height as f32, slope),
        });
        plate.push(hf.plate_color(dir));
    }
    // The cube-face basis (u,v,n) is LEFT-handed ((u×v)·n = -1), so the param-space
    // (cu,cv,r)→world map is orientation-reversing: transvoxel emits CCW-from-outside
    // triangles, which become CW in world space and get back-face-culled (pipeline is
    // cull BACK / front = CCW), so we'd see the inside/far side. Swap two indices per
    // triangle to restore CCW-outward winding. (Normals are already outward = −∇density.)
    let mut indices: Vec<u32> = mesh.triangle_indices.iter().map(|&i| i as u32).collect();
    for tri in indices.chunks_exact_mut(3) {
        tri.swap(1, 2);
    }
    let plate_colors = Some(plate);

    ChunkMeshArrays { positions, normals, colors, plate_colors, indices, origin }
}

#[cfg(test)]
mod tests {
    use super::*;
    use enki_planet::noise::Noise3D;
    use enki_planet::simple_height::SimpleHeightField;

    const RADIUS: f64 = 50_000.0;
    const HEIGHT_SCALE: f64 = 1_200.0;

    /// Flat field → perfect sphere of radius RADIUS. Every iso-surface vertex must
    /// lie on that sphere within one cell — proves the local→world→density→isosurface
    /// pipeline is geometrically correct.
    struct FlatHF;
    impl HeightField for FlatHF {
        fn height(&self, _dir: DVec3, _level: u8) -> f64 {
            0.0
        }
    }

    #[test]
    fn isosurface_sits_on_a_flat_sphere() {
        let hf = FlatHF;
        let dir = DVec3::new(1.0, 1.0, 1.0).normalize();
        let size = 200.0;
        let subdiv = 32;
        let center = dir * RADIUS;

        let mesh = mesh_chunk(&hf, center, size, subdiv, RADIUS, HEIGHT_SCALE, 0);

        assert!(!mesh.positions.is_empty(), "chunk produced no geometry");
        assert_eq!(mesh.positions.len(), mesh.normals.len(), "pos/normal count mismatch");
        assert_eq!(mesh.indices.len() % 3, 0, "indices not a triangle list");

        let cell = size / subdiv as f64; // 6.25 m
        let mut max_err = 0.0_f64;
        for p in &mesh.positions {
            let world = center + DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
            // flat HF → surface_radius == RADIUS everywhere.
            max_err = max_err.max((world.length() - RADIUS).abs());
        }
        assert!(max_err < cell, "max radial error {max_err:.4} m exceeded cell {cell:.4} m");
    }

    /// Varying terrain (bake-free SimpleHeightField): each vertex must lie within a
    /// couple cells of `surface_radius(dir)` re-sampled at that vertex's direction —
    /// proves the isosurface tracks real height variation, not just a sphere.
    #[test]
    fn isosurface_tracks_varying_terrain() {
        let hf = SimpleHeightField { noise: Noise3D::new(42) };
        let dir = DVec3::new(0.3, 0.8, 0.5).normalize();
        let size = 200.0;
        let subdiv = 32;
        let level = 8u8;
        let center = dir * surface_radius(&hf, dir, RADIUS, HEIGHT_SCALE, level);

        let mesh = mesh_chunk(&hf, center, size, subdiv, RADIUS, HEIGHT_SCALE, level);

        assert!(!mesh.positions.is_empty(), "chunk produced no geometry");
        let cell = size / subdiv as f64;
        let mut max_err = 0.0_f64;
        for p in &mesh.positions {
            let world = center + DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64);
            let want = surface_radius(&hf, world.normalize(), RADIUS, HEIGHT_SCALE, level);
            max_err = max_err.max((world.length() - want).abs());
        }
        assert!(max_err < 2.0 * cell, "max surface error {max_err:.4} m exceeded 2 cells {:.4} m", 2.0 * cell);
    }

    /// A block flagged with transition sides still meshes to valid, non-empty
    /// geometry (smoke test for the Phase 2 crack-free seam path).
    #[test]
    fn transition_side_still_meshes() {
        let hf = SimpleHeightField { noise: Noise3D::new(7) };
        let dir = DVec3::new(0.2, 0.9, 0.3).normalize();
        let level = 8u8;
        let center = dir * surface_radius(&hf, dir, RADIUS, HEIGHT_SCALE, level);
        let sides = TransitionSide::LowX | TransitionSide::HighZ;
        let mesh = mesh_block(&hf, center, 200.0, 32, RADIUS, HEIGHT_SCALE, level, sides);
        assert!(!mesh.positions.is_empty(), "transition-flagged block produced no geometry");
        assert_eq!(mesh.indices.len() % 3, 0, "indices not a triangle list");
    }

    /// Reconstruct a leaf's world-space vertices from camera-relative positions.
    fn world_verts(m: &ChunkMeshArrays) -> Vec<DVec3> {
        m.positions
            .iter()
            .map(|p| m.origin + DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64))
            .collect()
    }

    /// A meshed leaf's vertices lie on the heightfield surface (within a cell).
    #[test]
    fn leaf_isosurface_on_surface() {
        let hf = SimpleHeightField { noise: Noise3D::new(11) };
        let (face, level, ix, iy, subdiv) = (2u8, 5u8, 9u32, 14u32, 24usize);
        let m = mesh_leaf(&hf, face, level, ix, iy, RADIUS, HEIGHT_SCALE, subdiv, TransitionSide::none(), None, &[], 0);
        assert!(!m.positions.is_empty(), "leaf produced no geometry");

        // node_size = (π/2 · R)/2^level; cell ≈ node_size / subdiv.
        let node_size = std::f64::consts::FRAC_PI_2 * RADIUS / (1u64 << level) as f64;
        let cell = node_size / subdiv as f64;
        let mut max_err = 0.0_f64;
        for w in world_verts(&m) {
            let want = surface_radius(&hf, w.normalize(), RADIUS, HEIGHT_SCALE, level);
            max_err = max_err.max((w.length() - want).abs());
        }
        assert!(max_err < cell, "leaf radial error {max_err:.3} m exceeded cell {cell:.3} m");
    }

    /// THE seam safety-net: two same-level edge-adjacent leaves must share their
    /// boundary vertices exactly (crack-free). Counts A-vertices that have a
    /// coincident B-vertex; a crack would drop this to ~0.
    #[test]
    fn same_level_neighbors_are_crack_free() {
        let hf = SimpleHeightField { noise: Noise3D::new(3) };
        let (face, level, iy, subdiv) = (2u8, 5u8, 12u32, 16usize);
        let a = mesh_leaf(&hf, face, level, 9, iy, RADIUS, HEIGHT_SCALE, subdiv, TransitionSide::none(), None, &[], 0);
        let b = mesh_leaf(&hf, face, level, 10, iy, RADIUS, HEIGHT_SCALE, subdiv, TransitionSide::none(), None, &[], 0);
        let (aw, bw) = (world_verts(&a), world_verts(&b));
        assert!(!aw.is_empty() && !bw.is_empty(), "a neighbour produced no geometry");

        // Coincidence within 1 mm — shared-edge vertices are derived from the SAME
        // (cu,cv) line and exact-r crossing, so they should match to float precision.
        let eps = 1e-3;
        let coincident = aw
            .iter()
            .filter(|pa| bw.iter().any(|pb| pa.distance(*pb) < eps))
            .count();
        // The shared edge has ~subdiv+1 surface crossings; require the whole seam.
        assert!(
            coincident >= subdiv,
            "only {coincident} coincident boundary verts (< {subdiv}) — seam has a crack"
        );
    }

    /// Caves (non-linear density) must NOT crack the same-level seam — the global
    /// radial lattice keeps adjacent leaves' radial samples aligned on the shared face.
    #[test]
    fn caves_keep_same_level_crack_free() {
        let hf = SimpleHeightField { noise: Noise3D::new(3) };
        let caves = CaveField::new(CaveParams {
            strength: 120.0,
            freq: 0.01,
            margin: 5.0,
            falloff: 40.0,
            threshold: 0.4,
            seed: 9,
        });
        let (face, level, iy, subdiv) = (2u8, 5u8, 12u32, 16usize);
        let a = mesh_leaf(&hf, face, level, 9, iy, RADIUS, HEIGHT_SCALE, subdiv, TransitionSide::none(), Some(&caves), &[], 0);
        let b = mesh_leaf(&hf, face, level, 10, iy, RADIUS, HEIGHT_SCALE, subdiv, TransitionSide::none(), Some(&caves), &[], 0);
        let (aw, bw) = (world_verts(&a), world_verts(&b));
        assert!(!aw.is_empty() && !bw.is_empty(), "a cave neighbour produced no geometry");
        let eps = 1e-3;
        let coincident = aw
            .iter()
            .filter(|pa| bw.iter().any(|pb| pa.distance(*pb) < eps))
            .count();
        assert!(
            coincident >= subdiv,
            "caves cracked the same-level seam: {coincident} coincident < {subdiv}"
        );
    }

    /// Coarse leaves (levels 0–3, requested first when the camera is far) must
    /// produce geometry — these are what render at startup. If this is empty, the
    /// planet shows nothing from orbit.
    #[test]
    fn coarse_leaves_mesh() {
        let hf = SimpleHeightField { noise: Noise3D::new(1) };
        for level in [0u8, 1, 2, 3] {
            let m = mesh_leaf(&hf, 0, level, 0, 0, RADIUS, HEIGHT_SCALE, 24, TransitionSide::none(), None, &[], 0);
            assert!(
                !m.positions.is_empty() && !m.indices.is_empty(),
                "level {level} coarse leaf produced no geometry ({} verts, {} idx)",
                m.positions.len(),
                m.indices.len()
            );
        }
    }

    /// A dig edit carves air: the dug leaf reaches deeper than the un-dug surface.
    #[test]
    fn dig_edit_carves_air() {
        let hf = SimpleHeightField { noise: Noise3D::new(5) };
        let (face, level, ix, iy, subdiv) = (2u8, 6u8, 20u32, 33u32, 24usize);
        let center_dir = enki_planet::face_bases::patch_center_dir(face, level, ix, iy);
        let surf = surface_radius(&hf, center_dir, RADIUS, HEIGHT_SCALE, level);
        let edit = Edit { center: center_dir * (surf - 5.0), radius: 25.0, dig: true };

        let min_r = |m: &ChunkMeshArrays| {
            m.positions
                .iter()
                .map(|p| (m.origin + DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64)).length())
                .fold(f64::INFINITY, f64::min)
        };
        let plain = mesh_leaf(&hf, face, level, ix, iy, RADIUS, HEIGHT_SCALE, subdiv, TransitionSide::none(), None, &[], 0);
        let dug = mesh_leaf(&hf, face, level, ix, iy, RADIUS, HEIGHT_SCALE, subdiv, TransitionSide::none(), None, &[edit], 0);
        assert!(!dug.positions.is_empty(), "dug leaf produced no geometry");
        assert!(
            min_r(&dug) < min_r(&plain) - 1.0,
            "dig did not carve a pit (dug min_r {:.1} not below plain {:.1})",
            min_r(&dug),
            min_r(&plain)
        );
    }
}

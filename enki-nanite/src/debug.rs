//! Debug visualization for the baked cluster DAG.
//!
//! Mirrors Unreal's Nanite visualization modes — flat-color each triangle or
//! cluster to reveal micro-poly density and cluster structure. This is the
//! **offline** (CPU) counterpart that exports a colored PLY for inspection in
//! Blender/MeshLab/CloudCompare before the GPU runtime exists; the **live**
//! in-engine version (N2) reuses [`id_color`] in WGSL, keyed by the
//! `(cluster_id, triangle_id)` the vertex-pulling shader already computes.

use std::io::{self, Write};
use std::path::Path;

use crate::cluster::ClusterAsset;

/// How to color the exported geometry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorMode {
    /// Every triangle a distinct color (Unreal "Triangles" view) — micro-poly density.
    Triangle,
    /// Every cluster a distinct color (Unreal "Clusters" view) — meshlet structure.
    Cluster,
    /// Color by DAG level — see the LOD bands.
    Lod,
}

/// Deterministic distinct color for an integer id: hash → hue → RGB.
/// The same function is mirrored in the N2 WGSL debug shader so offline and
/// live views agree.
pub fn id_color(id: u32) -> [f32; 3] {
    // Integer hash (xorshift-multiply) for good bit diffusion.
    let mut h = id.wrapping_mul(0x9E37_79B1);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA77);
    h ^= h >> 13;
    let hue = (h % 3600) as f32 / 10.0; // 0..360, 0.1° steps
    let sat = 0.55 + ((h >> 8) & 0xFF) as f32 / 255.0 * 0.30; // 0.55..0.85
    let val = 0.80 + ((h >> 16) & 0xFF) as f32 / 255.0 * 0.20; // 0.80..1.00
    hsv_to_rgb(hue, sat, val)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> [f32; 3] {
    let c = v * s;
    let hp = h / 60.0;
    let x = c * (1.0 - (hp % 2.0 - 1.0).abs());
    let (r, g, b) = match hp as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    [r + m, g + m, b + m]
}

fn rgb8(c: [f32; 3]) -> [u8; 3] {
    [
        (c[0].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        (c[1].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
        (c[2].clamp(0.0, 1.0) * 255.0 + 0.5) as u8,
    ]
}

fn mix2(a: u32, b: u32) -> u32 {
    let mut h = a.wrapping_mul(0x9E37_79B1) ^ b.wrapping_mul(0x85EB_CA77);
    h ^= h >> 15;
    h
}

/// Write the baked clusters to an ASCII PLY with per-vertex colors.
///
/// `lod = Some(l)` exports only DAG level `l`; `None` exports every level
/// overlaid (geometry will overlap — useful only with [`ColorMode::Lod`]).
/// Positions are patch-origin-relative (already centered near the origin).
pub fn export_ply(
    asset: &ClusterAsset,
    lod: Option<u8>,
    mode: ColorMode,
    path: impl AsRef<Path>,
) -> io::Result<()> {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[u8; 3]> = Vec::new();
    let mut faces: Vec<[u32; 3]> = Vec::new();

    for (gi, c) in asset.clusters.iter().enumerate() {
        if let Some(l) = lod {
            if c.lod != l {
                continue;
            }
        }
        match mode {
            ColorMode::Triangle => {
                for (ti, t) in c.triangles.iter().enumerate() {
                    let col = rgb8(id_color(mix2(gi as u32, ti as u32)));
                    let base = positions.len() as u32;
                    for &li in t {
                        positions.push(c.vertices[li as usize].position);
                        colors.push(col);
                    }
                    faces.push([base, base + 1, base + 2]);
                }
            }
            ColorMode::Cluster | ColorMode::Lod => {
                let col = rgb8(match mode {
                    ColorMode::Cluster => id_color(gi as u32),
                    _ => id_color(c.lod as u32 + 1),
                });
                let base = positions.len() as u32;
                for v in &c.vertices {
                    positions.push(v.position);
                    colors.push(col);
                }
                for t in &c.triangles {
                    faces.push([base + t[0] as u32, base + t[1] as u32, base + t[2] as u32]);
                }
            }
        }
    }

    let mut w = io::BufWriter::new(std::fs::File::create(path)?);
    writeln!(w, "ply")?;
    writeln!(w, "format ascii 1.0")?;
    writeln!(w, "comment enki-nanite debug export ({mode:?})")?;
    writeln!(w, "element vertex {}", positions.len())?;
    writeln!(w, "property float x")?;
    writeln!(w, "property float y")?;
    writeln!(w, "property float z")?;
    writeln!(w, "property uchar red")?;
    writeln!(w, "property uchar green")?;
    writeln!(w, "property uchar blue")?;
    writeln!(w, "element face {}", faces.len())?;
    writeln!(w, "property list uchar int vertex_indices")?;
    writeln!(w, "end_header")?;
    for (p, c) in positions.iter().zip(colors.iter()) {
        writeln!(w, "{} {} {} {} {} {}", p[0], p[1], p[2], c[0], c[1], c[2])?;
    }
    for f in &faces {
        writeln!(w, "3 {} {} {}", f[0], f[1], f[2])?;
    }
    w.flush()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::{bake_patch, PatchParams};
    use enki_planet::noise::Noise3D;
    use enki_planet::simple_height::SimpleHeightField;

    fn asset() -> ClusterAsset {
        bake_patch(
            &PatchParams {
                face: 2,
                level: 4,
                ix: 5,
                iy: 6,
                resolution: 32,
                radius: 6_371_000.0,
                height_scale: 8_848.0,
            },
            &SimpleHeightField { noise: Noise3D::new(7) },
        )
    }

    #[test]
    fn id_color_in_range_and_distinct() {
        for id in 0..1000u32 {
            let c = id_color(id);
            for ch in c {
                assert!((0.0..=1.0).contains(&ch), "color channel out of range: {ch}");
            }
        }
        assert_ne!(id_color(0), id_color(1), "adjacent ids should differ");
    }

    #[test]
    fn ply_cluster_export_is_valid() {
        let a = asset();
        let path = std::env::temp_dir().join("enki_nanite_clusters.ply");
        export_ply(&a, Some(0), ColorMode::Cluster, &path).unwrap();
        let s = std::fs::read_to_string(&path).unwrap();
        assert!(s.starts_with("ply\n"));
        // Cluster mode keeps cluster-local verts, so face count == lod0 tri count.
        let lod0_tris: usize = a.clusters.iter().filter(|c| c.lod == 0).map(|c| c.triangles.len()).sum();
        assert!(s.contains(&format!("element face {lod0_tris}")), "wrong face count");
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn ply_triangle_export_duplicates_verts() {
        let a = asset();
        let path = std::env::temp_dir().join("enki_nanite_tris.ply");
        export_ply(&a, Some(0), ColorMode::Triangle, &path).unwrap();
        let s = std::fs::read_to_string(&path).unwrap();
        let lod0_tris: usize = a.clusters.iter().filter(|c| c.lod == 0).map(|c| c.triangles.len()).sum();
        // Triangle mode emits 3 unique verts per triangle.
        assert!(s.contains(&format!("element vertex {}", lod0_tris * 3)));
        assert!(s.contains(&format!("element face {lod0_tris}")));
        std::fs::remove_file(&path).ok();
    }
}

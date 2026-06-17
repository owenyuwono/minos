//! Bake a terrain patch and export colored debug meshes (PLY) for inspection.
//!
//! Run: `cargo run -p enki-nanite --example export_debug --release`
//! Output: `target/nanite-debug/*.ply` — open in Blender / MeshLab / CloudCompare.
//!
//! - `lod0_triangles.ply` — every triangle a distinct color (micro-poly density)
//! - `lod0_clusters.ply`  — every cluster a distinct color (meshlet structure)
//! - `coarsest_clusters.ply` — the root level, cluster-colored (LOD reduction)

use enki_nanite::bake::{bake_patch, PatchParams};
use enki_nanite::debug::{export_ply, ColorMode};
use enki_planet::noise::Noise3D;
use enki_planet::simple_height::SimpleHeightField;

fn main() -> std::io::Result<()> {
    let hf = SimpleHeightField { noise: Noise3D::new(1337) };
    let p = PatchParams {
        face: 2,
        level: 4,
        ix: 5,
        iy: 6,
        resolution: 128,
        radius: 6_371_000.0,
        height_scale: 8_848.0,
    };

    let asset = bake_patch(&p, &hf);
    let max_lod = asset.clusters.iter().map(|c| c.lod).max().unwrap_or(0);

    std::fs::create_dir_all("target/nanite-debug")?;
    export_ply(&asset, Some(0), ColorMode::Triangle, "target/nanite-debug/lod0_triangles.ply")?;
    export_ply(&asset, Some(0), ColorMode::Cluster, "target/nanite-debug/lod0_clusters.ply")?;
    export_ply(&asset, Some(max_lod), ColorMode::Cluster, "target/nanite-debug/coarsest_clusters.ply")?;

    println!(
        "Baked {} clusters / {} tris / {} LOD levels.",
        asset.cluster_count(),
        asset.total_triangles(),
        asset.lod_levels()
    );
    println!("Wrote PLYs to target/nanite-debug/ — open in Blender/MeshLab/CloudCompare.");
    Ok(())
}

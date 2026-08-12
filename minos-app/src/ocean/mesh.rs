//! Projected-grid mesh helpers for the FFT ocean.
//!
//! [`ndc_grid`] is a uniform screen-space lattice; each frame every vertex is
//! ray-cast onto the sea sphere ([`project_to_sphere`], CPU f64) so the waves fill
//! the whole view with no fixed patch.

use glam::DVec3;

/// A uniform `(res+1)²` grid of NDC positions in `[-1, 1]²` (the screen-space
/// lattice projected onto the sea sphere each frame) + its triangle indices.
pub fn ndc_grid(res: u32) -> (Vec<[f32; 2]>, Vec<u32>) {
    let n = res + 1;
    let mut ndc = Vec::with_capacity((n * n) as usize);
    for j in 0..n {
        for i in 0..n {
            ndc.push([i as f32 / res as f32 * 2.0 - 1.0, j as f32 / res as f32 * 2.0 - 1.0]);
        }
    }
    let mut indices = Vec::with_capacity((res * res * 6) as usize);
    for j in 0..res {
        for i in 0..res {
            let a = j * n + i;
            let b = a + 1;
            let c = a + n;
            let d = c + 1;
            indices.extend_from_slice(&[a, b, c, b, d, c]);
        }
    }
    (ndc, indices)
}

/// Ray-cast a camera ray onto the sea-level sphere, all camera-relative (camera at
/// the origin). `center_rel` = planet centre − camera; `radius` = sea radius.
/// Returns the near hit; on a miss (ray above the horizon) returns the nearest
/// point ON the sphere (the limb silhouette) so the mesh's top edge sits on the
/// horizon with no NaNs. `dir` need not be normalized.
pub fn project_to_sphere(dir: DVec3, center_rel: DVec3, radius: f64, near: f64) -> DVec3 {
    let d = dir.normalize();
    let b = d.dot(center_rel);
    let c = center_rel.length_squared() - radius * radius;
    let disc = b * b - c;
    if disc >= 0.0 {
        let t = b - disc.sqrt();
        if t >= near {
            return d * t;
        }
    }
    // Miss (or sphere behind near plane) → snap to the sphere point nearest the ray.
    let closest = d * b;
    let to = closest - center_rel;
    let len = to.length();
    if len > 1e-6 {
        center_rel + to / len * radius
    } else {
        d * near
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_hits_sea_below_camera() {
        // Camera 1000 m above the sea, looking straight down → hit 1000 m below.
        let r = 50_000.0;
        let camera = DVec3::new(0.0, 0.0, r + 1000.0);
        let center_rel = -camera;
        let hit = project_to_sphere(DVec3::new(0.0, 0.0, -1.0), center_rel, r, 0.5);
        assert!((hit - DVec3::new(0.0, 0.0, -1000.0)).length() < 1e-3, "got {hit:?}");
    }

    #[test]
    fn project_miss_lands_on_sphere() {
        // A ray pointing away from the planet → snapped onto the sphere (limb).
        let r = 50_000.0;
        let camera = DVec3::new(0.0, 0.0, r + 1000.0);
        let center_rel = -camera;
        let hit = project_to_sphere(DVec3::new(0.0, 1.0, 1.0), center_rel, r, 0.5);
        let world = hit + camera;
        assert!((world.length() - r).abs() < 1.0, "limb point off sphere: {}", world.length());
    }
}

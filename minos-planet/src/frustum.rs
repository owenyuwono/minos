//! View-frustum culling in planet-local space.

use glam::{DVec3, Mat4};

/// Six-plane view frustum for culling sphere-bounded chunks.
pub struct Frustum {
    // 6 planes in [nx, ny, nz, d] form — opaque to callers.
    planes: [[f32; 4]; 6],
}

impl Frustum {
    /// Extract frustum planes from a combined view-projection matrix.
    ///
    /// Uses Gribb-Hartmann plane extraction: each plane is derived from a
    /// row sum/difference of the matrix, then normalized to unit normal length.
    pub fn from_view_proj(m: Mat4) -> Self {
        let r0 = m.row(0);
        let r1 = m.row(1);
        let r2 = m.row(2);
        let r3 = m.row(3);

        // Raw planes: [nx, ny, nz, d]
        let raw = [
            r3 + r0, // left
            r3 - r0, // right
            r3 + r1, // bottom
            r3 - r1, // top
            r3 + r2, // near
            r3 - r2, // far
        ];

        let mut planes = [[0f32; 4]; 6];
        for (i, p) in raw.iter().enumerate() {
            let nx = p.x;
            let ny = p.y;
            let nz = p.z;
            let d = p.w;
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            if len > 1e-10 {
                planes[i] = [nx / len, ny / len, nz / len, d / len];
            } else {
                planes[i] = [nx, ny, nz, d];
            }
        }

        Self { planes }
    }

    /// Returns true if the sphere is not fully outside all planes.
    ///
    /// A sphere is outside the frustum if, for any plane, the signed distance
    /// from the sphere center to the plane is less than `-radius`. We cast
    /// center to f32 before the dot product since planes are f32 and the
    /// precision is sufficient for culling.
    pub fn intersects_sphere(&self, center: DVec3, radius: f64) -> bool {
        let c = center.as_vec3();
        let r = radius as f32;

        for plane in &self.planes {
            let [nx, ny, nz, d] = *plane;
            let dist = nx * c.x + ny * c.y + nz * c.z + d;
            if dist < -r {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Mat4, Vec3};

    fn perspective_looking_neg_z() -> Mat4 {
        // Simple perspective looking down -Z (standard OpenGL convention)
        Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0)
    }

    #[test]
    fn intersects_sphere_in_front() {
        let proj = perspective_looking_neg_z();
        let frustum = Frustum::from_view_proj(proj);
        // Sphere at (0,0,-5) with radius 1 — directly in front
        let visible = frustum.intersects_sphere(DVec3::new(0.0, 0.0, -5.0), 1.0);
        assert!(visible, "sphere in front should be visible");
    }

    #[test]
    fn intersects_sphere_behind() {
        let proj = perspective_looking_neg_z();
        let frustum = Frustum::from_view_proj(proj);
        // Sphere at (0,0,+5) with radius 0.1 — behind the camera
        let visible = frustum.intersects_sphere(DVec3::new(0.0, 0.0, 5.0), 0.1);
        assert!(!visible, "sphere behind camera should not be visible");
    }

    #[test]
    fn intersects_sphere_on_edge() {
        let proj = perspective_looking_neg_z();
        let frustum = Frustum::from_view_proj(proj);
        // Sphere just barely inside the near plane
        let visible = frustum.intersects_sphere(DVec3::new(0.0, 0.0, -0.15), 0.5);
        assert!(visible, "sphere near edge should be visible");
    }

    #[test]
    fn from_view_proj_planes_are_normalized() {
        let proj = perspective_looking_neg_z();
        let frustum = Frustum::from_view_proj(proj);
        for plane in &frustum.planes {
            let [nx, ny, nz, _] = *plane;
            let len = (nx * nx + ny * ny + nz * nz).sqrt();
            assert!(
                (len - 1.0).abs() < 1e-5,
                "plane normal length {len} is not unit"
            );
        }
    }

    #[test]
    fn intersects_sphere_view_proj_combined() {
        // Build a view-projection that looks at the origin from (0,0,10)
        let view = Mat4::look_at_rh(Vec3::new(0.0, 0.0, 10.0), Vec3::ZERO, Vec3::Y);
        let proj = Mat4::perspective_rh(std::f32::consts::FRAC_PI_2, 1.0, 0.1, 100.0);
        let vp = proj * view;
        let frustum = Frustum::from_view_proj(vp);

        // Origin should be visible
        assert!(frustum.intersects_sphere(DVec3::ZERO, 1.0), "origin should be visible");

        // Point behind the camera (at z=20, camera is at z=10 looking toward origin)
        let behind = frustum.intersects_sphere(DVec3::new(0.0, 0.0, 20.0), 0.1);
        assert!(!behind, "point behind camera should not be visible");
    }
}

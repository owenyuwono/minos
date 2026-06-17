//! `surface_picker` — ray–sphere intersection and screen-to-world ray building.
#![allow(dead_code)]
//!
//! All math is f64 for planet-scale precision.  Returned normals are outward
//! (from sphere centre toward the hit point).

use enki_render::camera::Camera;
use glam::{DVec3, Mat4, Vec4, Vec4Swizzles};

// ── SurfaceHit ────────────────────────────────────────────────────────────

/// A successful ray–sphere intersection result.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceHit {
    /// World-space intersection point on the sphere surface.
    pub point: DVec3,
    /// Outward surface normal at the hit point (unit vector).
    pub normal: DVec3,
}

// ── Ray–sphere intersection ───────────────────────────────────────────────

/// Intersect a ray with a sphere, returning the nearest positive-t hit.
///
/// Uses the smaller positive root of the quadratic `|o + t·d|² = r²`
/// (i.e. the entry point when outside, the exit point when inside from the
/// *perspective of distance from origin*).
///
/// Returns `None` if the ray misses the sphere entirely or all intersections
/// are behind the ray origin (negative t).
///
/// # Parameters
/// - `ray_origin` — world-space ray origin.
/// - `ray_dir`    — world-space ray direction (need not be unit length, but
///   unit length gives t in metres).
/// - `sphere_center` — world-space sphere centre.
/// - `sphere_radius` — sphere radius (m).
pub fn pick(
    ray_origin: DVec3,
    ray_dir: DVec3,
    sphere_center: DVec3,
    sphere_radius: f64,
) -> Option<SurfaceHit> {
    // Offset ray so the sphere is centred at the origin
    let oc = ray_origin - sphere_center;
    let a = ray_dir.dot(ray_dir);
    let b = 2.0 * oc.dot(ray_dir);
    let c = oc.dot(oc) - sphere_radius * sphere_radius;

    let discriminant = b * b - 4.0 * a * c;
    if discriminant < 0.0 {
        return None; // no intersection
    }

    let sqrt_disc = discriminant.sqrt();
    let t0 = (-b - sqrt_disc) / (2.0 * a);
    let t1 = (-b + sqrt_disc) / (2.0 * a);

    // Choose the smallest positive root
    let t = if t0 > 1e-12 {
        t0
    } else if t1 > 1e-12 {
        t1
    } else {
        return None; // both roots behind the ray origin
    };

    let point = ray_origin + ray_dir * t;
    let normal = (point - sphere_center).normalize();

    Some(SurfaceHit { point, normal })
}

// ── Screen-to-world ray ───────────────────────────────────────────────────

/// Build a world-space ray from a `Camera` and normalised device coordinates.
///
/// Returns `(ray_origin, ray_dir)` where `ray_origin` is the camera eye
/// position and `ray_dir` is a unit vector into the scene.
///
/// # Parameters
/// - `camera`  — current camera state.
/// - `ndc_x`   — normalised screen X in `[-1, 1]` (left → right).
/// - `ndc_y`   — normalised screen Y in `[-1, 1]` (bottom → top, OpenGL convention).
/// - `aspect`  — viewport width / height.
pub fn camera_ray(camera: &Camera, ndc_x: f32, ndc_y: f32, aspect: f32) -> (DVec3, DVec3) {
    // Reconstruct the projection matrix (reversed-Z, right-handed)
    let proj = Mat4::perspective_rh(camera.fov_y_radians, aspect, camera.near, camera.far);
    let view = camera.view_matrix();

    // Unproject the near-plane NDC point into world space.
    // We use z = -1.0 (the near plane in OpenGL NDC).
    let inv_vp = (proj * view).inverse();

    let ndc_near = Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
    let world_near = inv_vp * ndc_near;
    let world_near = world_near.xyz() / world_near.w;

    // Ray direction: from eye toward the unprojected point on the near plane
    let eye = camera.position.as_vec3();
    let ray_dir_f32 = (world_near - eye).normalize();

    (camera.position, ray_dir_f32.as_dvec3())
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::controls::globe::PLANET_RADIUS;
    use glam::{DVec3, Vec3};

    const CENTER: DVec3 = DVec3::ZERO;
    const R: f64 = PLANET_RADIUS;

    // ── Ray–sphere hits ───────────────────────────────────────────────────

    #[test]
    fn ray_hits_sphere_from_outside() {
        // Ray from directly above the north pole, heading down
        let origin = DVec3::new(0.0, R * 2.0, 0.0);
        let dir    = DVec3::new(0.0, -1.0,    0.0);
        let hit = pick(origin, dir, CENTER, R).expect("should hit");
        // Should hit the north pole
        assert!((hit.point.y - R).abs() < 1e-6, "hit.y = {}", hit.point.y);
        assert!((hit.normal - DVec3::new(0.0, 1.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn ray_misses_sphere() {
        // Ray going away from the sphere entirely
        let origin = DVec3::new(0.0, R * 2.0, 0.0);
        let dir    = DVec3::new(0.0,  1.0,    0.0); // heading away
        assert!(pick(origin, dir, CENTER, R).is_none());
    }

    #[test]
    fn ray_tangent_to_sphere() {
        // Ray tangent: discriminant ≈ 0, should still return a hit (one touch point)
        let origin = DVec3::new(R,     R * 2.0, 0.0);
        let dir    = DVec3::new(0.0,  -1.0,     0.0);
        // The closest approach is exactly R from the centre
        let hit = pick(origin, dir, CENTER, R);
        // Discriminant exactly zero — could be None or a hit at the equator rim
        if let Some(h) = hit {
            let dist = (h.point - CENTER).length();
            assert!((dist - R).abs() < 1.0, "tangent hit too far from surface: {dist}");
        }
        // A miss is also acceptable for exact tangent
    }

    #[test]
    fn ray_from_inside_sphere_returns_exit_point() {
        // Ray origin inside the sphere — should return the exit (far) root
        let origin = DVec3::new(0.0, 0.0, 0.0); // dead centre
        let dir    = DVec3::new(1.0, 0.0, 0.0);
        let hit = pick(origin, dir, CENTER, R).expect("should exit sphere");
        assert!((hit.point.x - R).abs() < 1e-6, "exit x = {}", hit.point.x);
        assert!((hit.normal - DVec3::new(1.0, 0.0, 0.0)).length() < 1e-6);
    }

    #[test]
    fn ray_behind_origin_returns_none() {
        // Ray pointing away, sphere is behind
        let origin = DVec3::new(R * 3.0, 0.0, 0.0);
        let dir    = DVec3::new(1.0,     0.0, 0.0); // pointing further away
        assert!(pick(origin, dir, CENTER, R).is_none());
    }

    #[test]
    fn normal_is_outward_unit_vector() {
        let origin = DVec3::new(0.0, R * 5.0, 0.0);
        let dir    = DVec3::new(0.0, -1.0,    0.0);
        let hit = pick(origin, dir, CENTER, R).unwrap();
        // Normal must be unit length
        assert!((hit.normal.length() - 1.0).abs() < 1e-9);
        // Normal must point outward from centre
        let outward = (hit.point - CENTER).normalize();
        assert!((hit.normal - outward).length() < 1e-9);
    }

    #[test]
    fn offset_sphere_centre() {
        // Sphere not at origin
        let sphere_c = DVec3::new(1000.0, 2000.0, 3000.0);
        let origin = sphere_c + DVec3::new(0.0, R * 2.0, 0.0);
        let dir    = DVec3::new(0.0, -1.0, 0.0);
        let hit = pick(origin, dir, sphere_c, R).expect("should hit");
        let dist_to_centre = (hit.point - sphere_c).length();
        assert!((dist_to_centre - R).abs() < 1e-3);
    }

    // ── camera_ray sanity ─────────────────────────────────────────────────

    #[test]
    fn camera_ray_is_unit_direction() {
        let cam = Camera {
            position:       DVec3::new(0.0, R * 2.0, 0.0),
            orientation:    glam::Quat::IDENTITY,
            fov_y_radians:  60_f32.to_radians(),
            near:           0.5,
            far:            750_000.0,
        };
        let (_origin, dir) = camera_ray(&cam, 0.0, 0.0, 16.0 / 9.0);
        assert!((dir.length() - 1.0).abs() < 1e-5, "ray dir length = {}", dir.length());
    }

    #[test]
    fn camera_ray_origin_is_camera_position() {
        let cam = Camera::default_orbit();
        let (origin, _) = camera_ray(&cam, 0.3, -0.2, 1.7);
        assert!((origin - cam.position).length() < 1e-9);
    }

    #[test]
    fn camera_ray_hits_planet() {
        // Camera above north pole looking at planet — centre NDC should hit
        let cam = Camera {
            position:       DVec3::new(0.0, R * 3.0, 0.0),
            orientation: {
                // Looking down: forward = -Y, up = +Z
                let forward = Vec3::new(0.0, -1.0, 0.0);
                let up      = Vec3::new(0.0,  0.0,  1.0);
                let right   = forward.cross(up).normalize();
                let cam_up  = right.cross(forward).normalize();
                glam::Quat::from_mat3(&glam::Mat3::from_cols(right, cam_up, -forward))
            },
            fov_y_radians: 60_f32.to_radians(),
            near: 0.5,
            far:  750_000.0,
        };
        let (origin, dir) = camera_ray(&cam, 0.0, 0.0, 1.0);
        let hit = pick(origin, dir, CENTER, R);
        assert!(hit.is_some(), "centre NDC ray should hit the planet");
    }
}

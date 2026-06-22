//! `space` — free-flight camera for interplanetary space (`NavMode::Space`).
//!
//! Heliocentric f64 position + yaw/pitch look. `camera()` produces the engine `Camera`
//! with a **Terra-relative** position, so the existing camera-relative render path
//! draws Terra (Nanite) + the sun + planets unchanged — you just see them from a free
//! camera, and Terra visibly orbits as the system advances. Wide far plane for
//! system-scale distances.

use enki_render::camera::Camera;
use glam::{DVec3, Quat, Vec3};

const SPACE_NEAR: f32 = 50.0;
const SPACE_FAR: f32 = 2.0e9;

pub struct FreeCam {
    pub position: DVec3, // heliocentric
    pub yaw: f32,        // radians, around +Y
    pub pitch: f32,
    pub speed: f64, // m/s
}

impl FreeCam {
    /// Start at `position`, looking toward `target` (both heliocentric).
    pub fn looking_at(position: DVec3, target: DVec3) -> Self {
        // forward = rot_y(yaw)·rot_x(pitch)·(-Z) = (-cosθ·sinψ, sinθ, -cosθ·cosψ).
        let d = (target - position).normalize_or_zero();
        let yaw = (-(d.x as f32)).atan2(-(d.z as f32));
        let pitch = (d.y as f32).clamp(-1.0, 1.0).asin();
        Self { position, yaw, pitch, speed: 30_000.0 }
    }

    pub fn orientation(&self) -> Quat {
        Quat::from_rotation_y(self.yaw) * Quat::from_rotation_x(self.pitch)
    }

    /// Mouse-look (raw pixel deltas).
    pub fn look(&mut self, dx: f32, dy: f32) {
        const SENS: f32 = 0.0022;
        self.yaw -= dx * SENS;
        self.pitch = (self.pitch - dy * SENS).clamp(-1.54, 1.54);
    }

    /// Scroll adjusts fly speed (geometric, per notch).
    pub fn adjust_speed(&mut self, scroll: f32) {
        let f = (1.0 + scroll as f64 * 0.15).clamp(0.5, 2.0);
        self.speed = (self.speed * f).clamp(1.0e3, 5.0e7);
    }

    /// WASD fly in camera-local axes (forward includes pitch). `boost` = sprint ×.
    pub fn fly(&mut self, forward: f32, right: f32, boost: f32, dt: f32) {
        let dir = self.orientation() * Vec3::new(right, 0.0, -forward);
        if dir.length_squared() > 1e-6 {
            self.position += dir.normalize().as_dvec3() * (self.speed * boost as f64 * dt as f64);
        }
    }

    /// Engine camera: Terra-relative position (cancels into the camera-relative path)
    /// + wide far plane for system-scale distances.
    pub fn camera(&self, terra_world: DVec3) -> Camera {
        Camera {
            position: self.position - terra_world,
            orientation: self.orientation(),
            fov_y_radians: 60_f32.to_radians(),
            near: SPACE_NEAR,
            far: SPACE_FAR,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fly_moves_toward_look_and_camera_is_terra_relative() {
        let terra = DVec3::new(2.0e6, 0.0, 0.0);
        let mut cam = FreeCam::looking_at(DVec3::new(2.5e6, 0.0, 0.0), terra); // looking -X
        let p0 = cam.position;
        cam.fly(1.0, 0.0, 1.0, 1.0); // forward 1 s
        assert!(cam.position.x < p0.x, "flew toward Terra (-X)");
        // Engine camera position is heliocentric minus Terra.
        let c = cam.camera(terra);
        assert!((c.position - (cam.position - terra)).length() < 1e-6);
    }
}

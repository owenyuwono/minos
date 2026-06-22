//! Wind streakline particle sim — pure CPU, headless, unit-tested.
//!
//! Particles ride the planet's baked wind velocity field (`HeightField::wind_at`,
//! a unit world-space tangent vector + speed) on a cap around the sub-camera
//! point. Each carries a short trail; `build_ribbons` turns the trails into
//! camera-facing ribbon geometry. A cheap time-varying "gust" rotates the wind
//! direction so the flow shimmers without any fluid solver (Tier B / living
//! overlay). Nothing here touches Vulkan — the GPU glue lives in `mod.rs`.

use enki_planet::climate::WindSample;
use glam::{DVec3, Vec3};

/// Number of live particles (fixed — reseeding keeps the set full, so vertex /
/// index buffers are a constant size). ponytail: const, not GUI-live; bump it
/// here if density needs tuning (a live count would force a buffer realloc).
pub const MAX_PARTICLES: usize = 6000;
/// Trail nodes per particle (ribbon = a strip through these).
pub const TRAIL_LEN: usize = 16;
/// Vertices emitted per particle (two per trail node).
pub const VERTS_PER_PARTICLE: usize = TRAIL_LEN * 2;
/// Total ribbon vertices (constant).
pub const VERT_COUNT: usize = MAX_PARTICLES * VERTS_PER_PARTICLE;

/// Sim seconds between trail-node samples (decouples trail length from frame rate).
const TRAIL_DT: f32 = 0.05;
/// Particle lifetime range (seconds) — randomized so reseeds desync over time.
const LIFE_LO: f32 = 2.5;
const LIFE_HI: f32 = 6.0;
/// Birth/death fade ramp (seconds) so particles don't pop in/out.
const FADE_EDGE: f32 = 0.6;

/// Live-tunable knobs (GUI). `cap` is set by the GPU layer each frame (adaptive
/// to altitude); the rest are user sliders.
#[derive(Debug, Clone, Copy)]
pub struct WindParams {
    /// Angular advection rate multiplier (rad/s at full wind speed).
    pub speed: f32,
    /// Ribbon half-width as a fraction of camera distance (≈ screen-constant px).
    pub width: f32,
    /// Streak altitude above sea level (metres) — floats the flow off the terrain.
    pub altitude: f32,
    /// Gust strength: max direction wobble (radians) from the time-varying field.
    pub gust: f32,
    /// Global brightness / opacity scale.
    pub intensity: f32,
    /// Seed cap half-angle around the sub-camera point (radians). Set per-frame.
    pub cap: f32,
}

impl Default for WindParams {
    fn default() -> Self {
        Self { speed: 0.12, width: 0.0016, altitude: 1500.0, gust: 0.5, intensity: 1.0, cap: 0.45 }
    }
}

#[derive(Clone, Copy)]
struct Particle {
    /// Current unit position (world direction on the sphere).
    pos: DVec3,
    /// Recent positions, newest at `[0]` (head) → oldest at `[LEN-1]` (tail).
    trail: [DVec3; TRAIL_LEN],
    /// Last sampled wind speed 0..1 (drives streak color).
    speed01: f32,
    age: f32,
    life: f32,
}

pub struct WindSim {
    particles: Vec<Particle>,
    accum: f32,
    rng: u32,
}

impl WindSim {
    pub fn new() -> Self {
        // All start expired → the first `step` reseeds them around the real nadir.
        let seed = Particle {
            pos: DVec3::Y,
            trail: [DVec3::Y; TRAIL_LEN],
            speed01: 0.0,
            age: f32::MAX,
            life: 1.0,
        };
        Self { particles: vec![seed; MAX_PARTICLES], accum: 0.0, rng: 0x9e37_79b9 }
    }

    /// Advance the sim by `dt` seconds. `nadir` is the unit sub-camera direction
    /// (planet-centre → camera). `wind(dir) -> WindSample` samples the baked field.
    pub fn step(
        &mut self,
        dt: f32,
        time: f32,
        nadir: DVec3,
        p: &WindParams,
        wind: impl Fn(DVec3) -> WindSample,
    ) {
        let dt = dt.min(0.1).max(0.0);
        let nadir = nadir.normalize_or(DVec3::Y);
        let (t1, t2) = tangent_basis(nadir);
        // Reseed when a particle leaves the cap (+margin) or dies.
        let cos_cap_out = ((p.cap * 1.3) as f64).cos();

        // Disjoint field borrows so the RNG and particles can be mutated together.
        let Self { particles, accum, rng } = self;

        *accum += dt;
        let push = *accum >= TRAIL_DT;
        if push {
            *accum %= TRAIL_DT;
        }

        for part in particles.iter_mut() {
            part.age += dt;
            if part.age >= part.life || part.pos.dot(nadir) < cos_cap_out {
                reseed(part, rng, nadir, t1, t2, p);
            } else {
                advect(part, dt, time, p, &wind);
            }
            if push {
                for i in (1..TRAIL_LEN).rev() {
                    part.trail[i] = part.trail[i - 1];
                }
                part.trail[0] = part.pos;
            }
        }
    }

    /// Emit camera-relative ribbon geometry for the current trails. `radius` is the
    /// streak shell radius (sea radius + altitude); `camera_pos` is the f64 camera
    /// position (the big coordinate is subtracted here in f64 for precision).
    /// Always fills exactly [`VERT_COUNT`] vertices (degenerate where invisible).
    pub fn build_ribbons(
        &self,
        camera_pos: DVec3,
        radius: f64,
        p: &WindParams,
        pos_out: &mut Vec<[f32; 3]>,
        col_out: &mut Vec<[f32; 3]>,
    ) {
        pos_out.clear();
        col_out.clear();
        for part in &self.particles {
            // Birth/death envelope.
            let life_env = smoothstep(0.0, FADE_EDGE, part.age)
                * smoothstep(0.0, FADE_EDGE, part.life - part.age);
            for i in 0..TRAIL_LEN {
                let nd = part.trail[i];
                let rel = (nd * radius - camera_pos).as_vec3();
                let dist = rel.length();
                let view = (-rel).normalize_or_zero(); // surface → camera (cam at origin)

                // Tangent of the trail at this node (direction only; translation cancels).
                let segw: DVec3 = if i + 1 < TRAIL_LEN {
                    part.trail[i + 1] - part.trail[i]
                } else {
                    part.trail[i] - part.trail[i - 1]
                };
                let segd = segw.as_vec3();

                // Camera-facing offset; zero (→ degenerate, invisible) when the trail
                // is collapsed (fresh particle) or the segment is edge-on.
                let mut off = Vec3::ZERO;
                if segd.length() > 1e-7 && dist > 1e-4 {
                    let o = segd.normalize().cross(view);
                    if o.length() > 1e-7 {
                        off = o.normalize() * (p.width * dist);
                    }
                }

                // Head (newest) bright → tail (oldest) transparent.
                let trail_bright = 1.0 - (i as f32) / ((TRAIL_LEN - 1) as f32);
                let fade = (trail_bright * life_env * p.intensity).clamp(0.0, 1.0);
                let col = [fade, part.speed01, 0.0];

                let a = rel - off;
                let b = rel + off;
                pos_out.push([a.x, a.y, a.z]);
                col_out.push(col);
                pos_out.push([b.x, b.y, b.z]);
                col_out.push(col);
            }
        }
    }
}

impl Default for WindSim {
    fn default() -> Self {
        Self::new()
    }
}

/// Static index buffer: two triangles per trail segment, fixed for all particles.
/// Each segment is emitted in BOTH windings → double-sided: the shared pipeline
/// culls BACK and these are camera-facing billboards whose winding we can't orient
/// reliably, so doubling guarantees the streak is visible from either side (the
/// same trick flora uses for its leaf cards). ponytail: drop to one winding if a
/// cull-NONE pipeline flag ever lands.
pub fn build_indices() -> Vec<u32> {
    let mut idx = Vec::with_capacity(MAX_PARTICLES * (TRAIL_LEN - 1) * 12);
    for pidx in 0..MAX_PARTICLES {
        let base = (pidx * VERTS_PER_PARTICLE) as u32;
        for s in 0..(TRAIL_LEN - 1) as u32 {
            let b = base + s * 2; // node i verts (b, b+1), node i+1 verts (b+2, b+3).
            idx.extend_from_slice(&[
                b, b + 1, b + 2, b + 2, b + 1, b + 3, // front
                b, b + 2, b + 1, b + 2, b + 3, b + 1, // back
            ]);
        }
    }
    idx
}

// ── helpers ────────────────────────────────────────────────────────────────

fn advect(part: &mut Particle, dt: f32, time: f32, p: &WindParams, wind: &impl Fn(DVec3) -> WindSample) {
    let ws = wind(part.pos);
    let dir = DVec3::new(ws.x as f64, ws.y as f64, ws.z as f64);
    let dlen = dir.length();
    if dlen <= 1e-6 {
        part.speed01 = 0.0; // calm zone — stall
        return;
    }
    let mut v = dir / dlen; // unit world tangent
    // Gust: rotate the wind about the local normal (keeps it tangent) by a smooth
    // time-varying angle — swirling gusts, no fluid solve.
    let ang = p.gust as f64 * gust_field(part.pos, time);
    v = rotate_about(v, part.pos, ang);
    let step = p.speed as f64 * ws.speed as f64 * dt as f64;
    part.pos = (part.pos + v * step).normalize();
    part.speed01 = ws.speed;
}

fn reseed(part: &mut Particle, rng: &mut u32, nadir: DVec3, t1: DVec3, t2: DVec3, p: &WindParams) {
    let cap = p.cap as f64;
    let cos_cap = cap.cos();
    let u = cos_cap + (1.0 - cos_cap) * rand01(rng); // cosθ uniform on the cap
    let phi = std::f64::consts::TAU * rand01(rng);
    let s = (1.0 - u * u).max(0.0).sqrt();
    let pos = (nadir * u + (t1 * phi.cos() + t2 * phi.sin()) * s).normalize();
    part.pos = pos;
    part.trail = [pos; TRAIL_LEN];
    part.age = 0.0;
    part.life = LIFE_LO + (LIFE_HI - LIFE_LO) * rand01(rng) as f32;
    part.speed01 = 0.0;
}

/// Smooth, cheap, divergence-free-ish gust field in [-1, 1] (sum of plane waves).
fn gust_field(pos: DVec3, t: f32) -> f64 {
    let t = t as f64;
    let k1 = DVec3::new(0.71, 0.34, -0.62);
    let k2 = DVec3::new(-0.27, 0.86, 0.43);
    let k3 = DVec3::new(0.52, -0.48, 0.71);
    let a = (pos.dot(k1) * 3.0 + t * 0.7).sin()
        + (pos.dot(k2) * 5.0 - t * 1.1).sin() * 0.5
        + (pos.dot(k3) * 9.0 + t * 1.7).sin() * 0.25;
    a / 1.75
}

/// Rotate `v` about unit `axis` by `ang` (Rodrigues). For tangent v (v⊥axis) this
/// is a pure in-plane rotation.
fn rotate_about(v: DVec3, axis: DVec3, ang: f64) -> DVec3 {
    let (s, c) = ang.sin_cos();
    v * c + axis.cross(v) * s + axis * (axis.dot(v) * (1.0 - c))
}

fn tangent_basis(n: DVec3) -> (DVec3, DVec3) {
    let r = if n.y.abs() < 0.99 { DVec3::Y } else { DVec3::X };
    let t1 = r.cross(n).normalize();
    let t2 = n.cross(t1);
    (t1, t2)
}

/// Tiny deterministic PRNG (no dep). Advances `*s` and returns a value in [0, 1).
fn rand01(s: &mut u32) -> f64 {
    let mut x = *s;
    x ^= x >> 16;
    x = x.wrapping_mul(0x7feb_352d);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846c_a68b);
    x ^= x >> 16;
    *s = x;
    (x >> 8) as f64 / (1u32 << 24) as f64
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A constant eastward wind (world +X tangent on the equator) at full speed.
    fn const_wind(_dir: DVec3) -> WindSample {
        WindSample { x: 1.0, y: 0.0, z: 0.0, speed: 1.0 }
    }

    #[test]
    fn particles_stay_on_sphere() {
        let mut sim = WindSim::new();
        let nadir = DVec3::new(1.0, 0.2, 0.0).normalize();
        let p = WindParams::default();
        for k in 0..400 {
            sim.step(0.016, k as f32 * 0.016, nadir, &p, const_wind);
        }
        for part in &sim.particles {
            assert!((part.pos.length() - 1.0).abs() < 1e-9, "pos drifted off sphere");
            assert!(part.pos.is_finite(), "non-finite particle position");
        }
    }

    #[test]
    fn count_is_stable() {
        let mut sim = WindSim::new();
        let p = WindParams::default();
        sim.step(0.016, 0.0, DVec3::X, &p, const_wind);
        assert_eq!(sim.particles.len(), MAX_PARTICLES);
    }

    #[test]
    fn ribbon_vertex_count_matches_indices() {
        let mut sim = WindSim::new();
        let p = WindParams::default();
        sim.step(0.016, 0.0, DVec3::X, &p, const_wind);
        let (mut pos, mut col) = (Vec::new(), Vec::new());
        sim.build_ribbons(DVec3::new(60_000.0, 0.0, 0.0), 50_300.0, &p, &mut pos, &mut col);
        assert_eq!(pos.len(), VERT_COUNT);
        assert_eq!(col.len(), VERT_COUNT);
        // Every index must address a real vertex.
        let idx = build_indices();
        assert_eq!(idx.len(), MAX_PARTICLES * (TRAIL_LEN - 1) * 12);
        assert!(*idx.iter().max().unwrap() < VERT_COUNT as u32);
        // No NaNs leaked into the geometry.
        assert!(pos.iter().flatten().all(|f| f.is_finite()));
    }

    #[test]
    fn calm_zone_produces_no_nan() {
        let mut sim = WindSim::new();
        let p = WindParams::default();
        let calm = |_d: DVec3| WindSample { x: 0.0, y: 0.0, z: 0.0, speed: 0.0 };
        for k in 0..50 {
            sim.step(0.016, k as f32 * 0.016, DVec3::Z, &p, calm);
        }
        for part in &sim.particles {
            assert!(part.pos.is_finite());
        }
    }
}

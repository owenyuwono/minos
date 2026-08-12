//! Ocean **openness** (fetch proxy) + a lightly-smoothed **wind** cube — the
//! per-vertex inputs that drive wave intensity + direction.
//!
//! Baked once when the heightfield loads (cheap nearest height samples + a per-face
//! box blur), then sampled **bilinearly** per ocean vertex so the result is smooth
//! (no blocky cube texels):
//! - `openness`: 1 − "land nearby" density. Wind is scaled by this so water enclosed
//!   by land stays calm while open ocean (incl. open coasts) keeps full energy.
//! - `wind`: the planet wind (speed + direction) re-sampled + blurred so the sharp
//!   climate shear bands fall off gradually instead of abruptly.

use minos_planet::height::HeightField;
use glam::DVec3;

/// Cube resolution per face (coarse — smoothed by the blur + bilinear sampling).
pub const OPEN_RES: usize = 48;
/// Openness blur passes → enclosure radius (wide: reaches the centre of a medium
/// sea so it counts land on several sides; open-ocean centres stay far from land).
const OPEN_BLUR_PASSES: usize = 12;
/// Wind blur passes → softens the climate shear bands (gentle falloff, not erased).
const WIND_BLUR_PASSES: usize = 4;
/// LOD level for the land/sea test (coarse continents, no fine detail).
const MASK_LEVEL: u8 = 6;

// ── Cube addressing ─────────────────────────────────────────────────────────────

/// Texel center → unit direction (standard cube convention; inverse of [`face_uv`]).
fn texel_to_dir(face: usize, x: usize, y: usize, res: usize) -> DVec3 {
    let sc = (x as f64 + 0.5) / res as f64 * 2.0 - 1.0;
    let tc = (y as f64 + 0.5) / res as f64 * 2.0 - 1.0;
    let d = match face {
        0 => DVec3::new(1.0, -tc, -sc),
        1 => DVec3::new(-1.0, -tc, sc),
        2 => DVec3::new(sc, 1.0, tc),
        3 => DVec3::new(sc, -1.0, -tc),
        4 => DVec3::new(sc, -tc, 1.0),
        _ => DVec3::new(-sc, -tc, -1.0),
    };
    d.normalize()
}

/// Unit direction → (face, u, v) with u,v ∈ [0,1] (inverse of [`texel_to_dir`]).
fn face_uv(dir: DVec3) -> (usize, f64, f64) {
    let a = dir.abs();
    let (face, sc, tc, ma) = if a.x >= a.y && a.x >= a.z {
        if dir.x > 0.0 { (0, -dir.z, -dir.y, a.x) } else { (1, dir.z, -dir.y, a.x) }
    } else if a.y >= a.z {
        if dir.y > 0.0 { (2, dir.x, dir.z, a.y) } else { (3, dir.x, -dir.z, a.y) }
    } else if dir.z > 0.0 {
        (4, dir.x, -dir.y, a.z)
    } else {
        (5, -dir.x, -dir.y, a.z)
    };
    (face, (sc / ma * 0.5 + 0.5).clamp(0.0, 1.0), (tc / ma * 0.5 + 0.5).clamp(0.0, 1.0))
}

/// Bilinear texel corners + fractions for (u,v) in a `res`-face (clamped at edges).
fn bilinear(u: f64, v: f64, res: usize) -> (usize, usize, usize, usize, f32, f32) {
    let fx = u * res as f64 - 0.5;
    let fy = v * res as f64 - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = (fx - x0) as f32;
    let ty = (fy - y0) as f32;
    let c = |i: f64| (i as i64).clamp(0, res as i64 - 1) as usize;
    (c(x0), c(x0 + 1.0), c(y0), c(y0 + 1.0), tx, ty)
}

/// Bilinear lookup of a scalar cube field at a unit direction.
pub fn sample(field: &[f32], res: usize, dir: DVec3) -> f32 {
    let (face, u, v) = face_uv(dir);
    let (xa, xb, ya, yb, tx, ty) = bilinear(u, v, res);
    let b = face * res * res;
    let top = field[b + ya * res + xa] * (1.0 - tx) + field[b + ya * res + xb] * tx;
    let bot = field[b + yb * res + xa] * (1.0 - tx) + field[b + yb * res + xb] * tx;
    top * (1.0 - ty) + bot * ty
}

/// Bilinear lookup of a vec3 cube field at a unit direction.
pub fn sample_vec3(field: &[[f32; 3]], res: usize, dir: DVec3) -> [f32; 3] {
    let (face, u, v) = face_uv(dir);
    let (xa, xb, ya, yb, tx, ty) = bilinear(u, v, res);
    let b = face * res * res;
    let mut out = [0.0f32; 3];
    for k in 0..3 {
        let top = field[b + ya * res + xa][k] * (1.0 - tx) + field[b + ya * res + xb][k] * tx;
        let bot = field[b + yb * res + xa][k] * (1.0 - tx) + field[b + yb * res + xb][k] * tx;
        out[k] = top * (1.0 - ty) + bot * ty;
    }
    out
}

// ── Per-face box blur ───────────────────────────────────────────────────────────

fn blur_scalar(field: &mut Vec<f32>, res: usize, passes: usize) {
    let n = res * res;
    let mut tmp = field.clone();
    for _ in 0..passes {
        for face in 0..6 {
            let base = face * n;
            for y in 0..res {
                for x in 0..res {
                    let mut s = 0.0;
                    let mut c = 0.0;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let xx = x as i32 + dx;
                            let yy = y as i32 + dy;
                            if xx >= 0 && xx < res as i32 && yy >= 0 && yy < res as i32 {
                                s += field[base + yy as usize * res + xx as usize];
                                c += 1.0;
                            }
                        }
                    }
                    tmp[base + y * res + x] = s / c;
                }
            }
        }
        std::mem::swap(field, &mut tmp);
    }
}

fn blur_vec3(field: &mut Vec<[f32; 3]>, res: usize, passes: usize) {
    let n = res * res;
    let mut tmp = field.clone();
    for _ in 0..passes {
        for face in 0..6 {
            let base = face * n;
            for y in 0..res {
                for x in 0..res {
                    let mut s = [0.0f32; 3];
                    let mut c = 0.0;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let xx = x as i32 + dx;
                            let yy = y as i32 + dy;
                            if xx >= 0 && xx < res as i32 && yy >= 0 && yy < res as i32 {
                                let p = field[base + yy as usize * res + xx as usize];
                                s[0] += p[0]; s[1] += p[1]; s[2] += p[2];
                                c += 1.0;
                            }
                        }
                    }
                    tmp[base + y * res + x] = [s[0] / c, s[1] / c, s[2] / c];
                }
            }
        }
        std::mem::swap(field, &mut tmp);
    }
}

// ── Bakes ───────────────────────────────────────────────────────────────────────

/// Bake the openness cube (0 = sheltered/landlocked, 1 = open ocean). `sea_level_norm`
/// is the waterline in the heightfield's normalized units.
pub fn bake_openness(hf: &dyn HeightField, sea_level_norm: f64) -> Vec<f32> {
    let res = OPEN_RES;
    let n = res * res;

    // Land mask: 1 = land (above sea level), 0 = ocean → blur into a "land nearby"
    // density (per-face; cross-face seams ignored — fine for a smooth field).
    let mut m = vec![0.0f32; 6 * n];
    for face in 0..6 {
        for y in 0..res {
            for x in 0..res {
                let dir = texel_to_dir(face, x, y, res);
                let land = hf.height(dir, MASK_LEVEL) > sea_level_norm;
                m[face * n + y * res + x] = if land { 1.0 } else { 0.0 };
            }
        }
    }
    blur_scalar(&mut m, res, OPEN_BLUR_PASSES);

    // openness = 1 − land-proximity. High threshold so an OPEN coast (land on one
    // side → density ~0.5) stays full energy, and only water surrounded by land on
    // several sides (high density → enclosed sea/strait) goes calm.
    // ponytail: blind-tuned — adjust the 0.5..0.85 band by eye.
    for v in m.iter_mut() {
        *v = 1.0 - smoothstep(0.5, 0.85, *v);
    }
    m
}

/// Bake the wind cube: planet wind speed (0..1) + world tangent direction, lightly
/// blurred so the sharp climate shear bands fall off gradually.
pub fn bake_wind(hf: &dyn HeightField) -> (Vec<f32>, Vec<[f32; 3]>) {
    let res = OPEN_RES;
    let n = res * res;
    let mut speed = vec![0.0f32; 6 * n];
    let mut dir = vec![[0.0f32; 3]; 6 * n];
    for face in 0..6 {
        for y in 0..res {
            for x in 0..res {
                let d = texel_to_dir(face, x, y, res);
                let w = hf.wind_at(d);
                let i = face * n + y * res + x;
                speed[i] = w.speed;
                dir[i] = [w.x, w.y, w.z];
            }
        }
    }
    blur_scalar(&mut speed, res, WIND_BLUR_PASSES);
    blur_vec3(&mut dir, res, WIND_BLUR_PASSES);
    (speed, dir)
}

fn smoothstep(lo: f32, hi: f32, x: f32) -> f32 {
    let t = ((x - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `sample` must invert `texel_to_dir` — a value at one texel is read back at
    /// that texel's direction (bilinear at a texel centre = that texel's value).
    #[test]
    fn cube_round_trips() {
        let res = OPEN_RES;
        let n = res * res;
        let mut field = vec![0.0f32; 6 * n];
        let probes = [(0, 3, 5), (1, 40, 2), (2, 24, 24), (3, 10, 31), (4, 47, 0), (5, 1, 19)];
        for (i, &(f, x, y)) in probes.iter().enumerate() {
            field[f * n + y * res + x] = (i + 1) as f32;
        }
        for (i, &(f, x, y)) in probes.iter().enumerate() {
            let dir = texel_to_dir(f, x, y, res);
            assert!((sample(&field, res, dir) - (i + 1) as f32).abs() < 1e-4, "probe {i} face {f}");
        }
    }
}

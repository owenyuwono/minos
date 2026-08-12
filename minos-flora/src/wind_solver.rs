//! Hierarchical skeletal wind solver — 1:1 Rust port of dryad `windSolver.js`.
//!
//! Pure math (no RHI / no GPU): given a `bones_wind` table (from
//! [`crate::mesh::build_branch_mesh`]) and the per-frame wind parameters, it
//! produces ONE column-major `mat4` per bone by:
//!   1. Computing a per-bone LOCAL rotation angle from `time + per-bone phase ×
//!      stiffness` (a primary gust + a turbulence term).
//!   2. Composing TOP-DOWN: `world_c = world_parent(c) · local_c`, where
//!      `local_c = T(pivot_c) · R(axis, angle_c) · T(-pivot_c)`.
//!
//! Output layout: a flat `Vec<f32>`, `out[c*16 .. c*16+16]` = the mat4 for bone
//! `c`, COLUMN-MAJOR (the same layout a WGSL `mat4x4<f32>` storage element wants).
//!
//! Guarantees (mirroring dryad):
//!   - `strength == 0` ⇒ every bone matrix is the IDENTITY (the calm guarantee →
//!     the shader's `bones[c] · restPos == restPos` → byte-identical rest pose).
//!   - deterministic: same `(bones_wind, time, strength, dir)` ⇒ identical out.
//!   - `parent[c] < c` is the precondition for the single ascending pass; the
//!     mesher guarantees it (and [`solve_into`] panics if it is violated, like
//!     dryad's constructor `throw`).
//!   - rigid bones (`is_rigid == 1`) always produce an identity local rotation.
//!
//! ponytail: `f64` scratch internally, narrowed to `f32` on output — matches
//! dryad (`Float64Array` worldMats → `Float32Array` out) and keeps determinism.

use crate::mesh::WindBone;

// ---------------------------------------------------------------------------
// Per-bone phase: integer Wang-hash of (index, quantised pivot) → [0, 2π).
// Bit-faithful port of windSolver.js wangHash/quantisePivot/bonePhase. JS
// `Math.imul` → `u32::wrapping_mul`; `>>> n` → `>> n` on u32.
// ---------------------------------------------------------------------------

#[inline]
fn wang_hash(mut x: u32) -> u32 {
    x = (x ^ (x >> 4)).wrapping_mul(0x08D8_74F5);
    x = (x ^ (x >> 9)).wrapping_mul(0x6546_B0DF);
    x ^ (x >> 16)
}

/// Snap a float pivot coord to 0.5 mm integer buckets (`Math.round(v*2000)|0`).
#[inline]
fn quantise_pivot(v: f32) -> u32 {
    // JS `| 0` truncates toward zero into an i32, then we reinterpret as u32 (XOR
    // operand). `f64::round` matches `Math.round` (round-half-away, but the XOR
    // masks below only care about the bit pattern).
    ((v as f64).round() * 2000.0).round() as i32 as u32
}

/// Three-input hash → per-bone phase in `[0, 2π)` (windSolver.js:41-50).
#[inline]
fn bone_phase(c: u32, px: f32, py: f32, pz: f32) -> f64 {
    let hc = wang_hash(c);
    let hx = wang_hash(quantise_pivot(px) ^ 0x1234_5678);
    let hy = wang_hash(quantise_pivot(py) ^ 0xDEAD_BEEF);
    let hz = wang_hash(quantise_pivot(pz) ^ 0x9E37_79B9);
    let combined = hc ^ hx ^ hy ^ hz;
    (combined as f64 / 4_294_967_296.0) * std::f64::consts::TAU
}

// ---------------------------------------------------------------------------
// Wind formula constants (windSolver.js:146-149). SMALL on purpose: per-bone
// angles COMPOSE down the hierarchy, so a few degrees per bone is the canonical
// SpeedTree / GPU-Gems sway.
// ---------------------------------------------------------------------------

const A1: f64 = 0.06; // primary gust amplitude (rad) ≈ 3.4° per bone
const A2: f64 = 0.03; // turbulence amplitude (rad) ≈ 1.7° per bone
const F1: f64 = 1.2; // primary frequency (rad/s)
const F2: f64 = 2.73; // turbulence frequency (irrational ratio to F1)

// ---------------------------------------------------------------------------
// 4×4 column-major matrix ops on f64 scratch (mirror windSolver.js term-for-term).
// ---------------------------------------------------------------------------

#[inline]
fn set_identity(m: &mut [f64], off: usize) {
    m[off] = 1.0; m[off + 1] = 0.0; m[off + 2] = 0.0; m[off + 3] = 0.0;
    m[off + 4] = 0.0; m[off + 5] = 1.0; m[off + 6] = 0.0; m[off + 7] = 0.0;
    m[off + 8] = 0.0; m[off + 9] = 0.0; m[off + 10] = 1.0; m[off + 11] = 0.0;
    m[off + 12] = 0.0; m[off + 13] = 0.0; m[off + 14] = 0.0; m[off + 15] = 1.0;
}

/// Build `T(pivot)·R(axis,angle)·T(-pivot)` into `dst[0..16]` (column-major).
/// `axis` must be normalised. Rodrigues form, identical to `buildPivotRotation`.
#[allow(clippy::too_many_arguments)]
fn build_pivot_rotation(dst: &mut [f64; 16], ax: f64, ay: f64, az: f64, angle: f64, px: f64, py: f64, pz: f64) {
    let c = angle.cos();
    let s = angle.sin();
    let t = 1.0 - c;

    let r00 = t * ax * ax + c;
    let r10 = t * ax * ay + s * az;
    let r20 = t * ax * az - s * ay;

    let r01 = t * ax * ay - s * az;
    let r11 = t * ay * ay + c;
    let r21 = t * ay * az + s * ax;

    let r02 = t * ax * az + s * ay;
    let r12 = t * ay * az - s * ax;
    let r22 = t * az * az + c;

    // col 3 = pivot - R·pivot.
    let tx = px - (r00 * px + r01 * py + r02 * pz);
    let ty = py - (r10 * px + r11 * py + r12 * pz);
    let tz = pz - (r20 * px + r21 * py + r22 * pz);

    dst[0] = r00; dst[1] = r10; dst[2] = r20; dst[3] = 0.0;
    dst[4] = r01; dst[5] = r11; dst[6] = r21; dst[7] = 0.0;
    dst[8] = r02; dst[9] = r12; dst[10] = r22; dst[11] = 0.0;
    dst[12] = tx; dst[13] = ty; dst[14] = tz; dst[15] = 1.0;
}

/// `result = A · B` (column-major 16-element flats). `result` must alias neither
/// `a` nor `b` (we always pass distinct slices below).
fn mul4x4(result: &mut [f64], a: &[f64], b: &[f64]) {
    let a00 = a[0]; let a01 = a[4]; let a02 = a[8]; let a03 = a[12];
    let a10 = a[1]; let a11 = a[5]; let a12 = a[9]; let a13 = a[13];
    let a20 = a[2]; let a21 = a[6]; let a22 = a[10]; let a23 = a[14];
    let a30 = a[3]; let a31 = a[7]; let a32 = a[11]; let a33 = a[15];

    let b00 = b[0]; let b01 = b[4]; let b02 = b[8]; let b03 = b[12];
    let b10 = b[1]; let b11 = b[5]; let b12 = b[9]; let b13 = b[13];
    let b20 = b[2]; let b21 = b[6]; let b22 = b[10]; let b23 = b[14];
    let b30 = b[3]; let b31 = b[7]; let b32 = b[11]; let b33 = b[15];

    result[0] = a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30;
    result[1] = a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30;
    result[2] = a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30;
    result[3] = a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30;

    result[4] = a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31;
    result[5] = a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31;
    result[6] = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31;
    result[7] = a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31;

    result[8] = a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32;
    result[9] = a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32;
    result[10] = a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32;
    result[11] = a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32;

    result[12] = a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33;
    result[13] = a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33;
    result[14] = a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33;
    result[15] = a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33;
}

// ---------------------------------------------------------------------------
// WindSolver — owns the per-bone phases + reusable scratch (no per-frame alloc).
// ---------------------------------------------------------------------------

/// A reusable hierarchical wind solver bound to one `bones_wind` table.
///
/// Construct once per tree (`new`), then call [`WindSolver::solve_into`] each
/// frame into a caller-owned `&mut [f32]` of length `>= count*16`.
pub struct WindSolver {
    count: usize,
    // Cloned bone tables (the mesh's `bones_wind` is short-lived; the solver
    // outlives the mesh on the view, so it owns its own copy).
    parent: Vec<i32>,
    pivot: Vec<[f32; 3]>,
    stiffness: Vec<f32>,
    is_rigid: Vec<u8>,
    phases: Vec<f64>,
    // Scratch: composed world matrices in f64 + one local-rotation buffer.
    world: Vec<f64>,
    local: [f64; 16],
}

impl WindSolver {
    /// Build a solver for `bones`. Panics if the parent ordering invariant
    /// (`parent[c] < c`) is violated — that is a mesher bug, not a runtime input.
    pub fn new(bones: &[WindBone]) -> Self {
        let count = bones.len();
        let mut parent = Vec::with_capacity(count);
        let mut pivot = Vec::with_capacity(count);
        let mut stiffness = Vec::with_capacity(count);
        let mut is_rigid = Vec::with_capacity(count);
        let mut phases = Vec::with_capacity(count);

        for (c, b) in bones.iter().enumerate() {
            assert!(
                b.parent == -1 || ((b.parent as usize) < c),
                "windSolver: bones_wind.parent[{c}]={} violates parent < child ordering",
                b.parent
            );
            parent.push(b.parent);
            pivot.push(b.pivot);
            stiffness.push(b.stiffness);
            is_rigid.push(b.is_rigid);
            phases.push(bone_phase(c as u32, b.pivot[0], b.pivot[1], b.pivot[2]));
        }

        WindSolver {
            count,
            parent,
            pivot,
            stiffness,
            is_rigid,
            phases,
            world: vec![0.0; count * 16],
            local: [0.0; 16],
        }
    }

    /// Number of bones (== output mat4 count).
    pub fn bone_count(&self) -> usize {
        self.count
    }

    /// Solve this frame's bone matrices into `out` (column-major flat,
    /// `out.len() >= count*16`). `dir_x`/`dir_z` are the horizontal wind
    /// direction (need not be normalised). `strength == 0` ⇒ all identities.
    pub fn solve_into(&mut self, out: &mut [f32], time: f64, strength: f64, dir_x: f64, dir_z: f64) {
        debug_assert!(out.len() >= self.count * 16, "wind out buffer too small");

        // Wind axis = cross(up, windDir) = (dirZ, 0, -dirX), normalised in XZ.
        let mut ax = dir_z;
        let ay = 0.0_f64;
        let mut az = -dir_x;
        let len = (ax * ax + az * az).sqrt();
        if len > 1e-9 {
            ax /= len;
            az /= len;
        } else {
            ax = 1.0;
            az = 0.0;
        }

        for c in 0..self.count {
            let p = self.parent[c];
            let w_off = c * 16;
            let px = self.pivot[c][0] as f64;
            let py = self.pivot[c][1] as f64;
            let pz = self.pivot[c][2] as f64;

            // Per-bone local rotation angle (0 when calm or rigid).
            let mut angle = 0.0_f64;
            if strength != 0.0 && self.is_rigid[c] != 1 {
                let ph = self.phases[c];
                let s = self.stiffness[c] as f64;
                angle = strength
                    * s
                    * (A1 * (time * F1 + ph).sin()
                        + A2 * (time * F2 + ph * 2.17 + 1.3).sin());
            }

            if angle == 0.0 {
                // Local is identity → world = parent world (or identity at a root).
                if p == -1 {
                    set_identity(&mut self.world, w_off);
                } else {
                    let p_off = p as usize * 16;
                    // Copy parent's 16 floats forward (p < c ⇒ already written).
                    let (head, tail) = self.world.split_at_mut(w_off);
                    tail[..16].copy_from_slice(&head[p_off..p_off + 16]);
                }
            } else {
                build_pivot_rotation(&mut self.local, ax, ay, az, angle, px, py, pz);
                if p == -1 {
                    self.world[w_off..w_off + 16].copy_from_slice(&self.local);
                } else {
                    let p_off = p as usize * 16;
                    // world_c = world_parent · local_c. p_off < w_off (parent<child),
                    // so split at w_off keeps parent in `head`, target in `tail`.
                    let (head, tail) = self.world.split_at_mut(w_off);
                    mul4x4(&mut tail[..16], &head[p_off..p_off + 16], &self.local);
                }
            }

            for i in 0..16 {
                out[w_off + i] = self.world[w_off + i] as f32;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bones() -> Vec<WindBone> {
        // A small valid hierarchy: rigid trunk (0), two children (1,2), grandchild (3).
        vec![
            WindBone { parent: -1, pivot: [0.0, 0.0, 0.0], axis: [0.0, 1.0, 0.0], stiffness: 0.0, branch_level: 0, is_rigid: 1 },
            WindBone { parent: 0, pivot: [0.1, 1.0, 0.0], axis: [0.0, 1.0, 0.0], stiffness: 0.5, branch_level: 1, is_rigid: 0 },
            WindBone { parent: 0, pivot: [-0.1, 1.2, 0.2], axis: [0.0, 1.0, 0.0], stiffness: 0.8, branch_level: 1, is_rigid: 0 },
            WindBone { parent: 1, pivot: [0.3, 1.6, 0.1], axis: [0.0, 1.0, 0.0], stiffness: 1.0, branch_level: 2, is_rigid: 0 },
        ]
    }

    fn is_identity(m: &[f32]) -> bool {
        const ID: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        m.iter().zip(ID.iter()).all(|(a, b)| (a - b).abs() < 1e-6)
    }

    // Test convenience: allocate + solve into a fresh `Vec<f32>` (the app uses
    // `solve_into` directly to reuse a buffer).
    fn solve(s: &mut WindSolver, time: f64, strength: f64, dir_x: f64, dir_z: f64) -> Vec<f32> {
        let mut out = vec![0.0_f32; s.count * 16];
        s.solve_into(&mut out, time, strength, dir_x, dir_z);
        out
    }

    #[test]
    fn strength_zero_is_all_identity() {
        let bones = sample_bones();
        let mut solver = WindSolver::new(&bones);
        let out = solve(&mut solver, 3.21, 0.0, 1.0, 0.0);
        assert_eq!(out.len(), bones.len() * 16);
        for c in 0..bones.len() {
            assert!(is_identity(&out[c * 16..c * 16 + 16]), "bone {c} not identity at strength 0");
        }
    }

    #[test]
    fn rigid_bone_stays_identity_when_windy() {
        let bones = sample_bones();
        let mut solver = WindSolver::new(&bones);
        let out = solve(&mut solver, 1.7, 1.0, 1.0, 0.0);
        // Bone 0 is rigid → its WORLD matrix is identity (root + identity local).
        assert!(is_identity(&out[0..16]), "rigid root must stay identity");
        // A flexing bone moves: its matrix is NOT identity under wind.
        assert!(!is_identity(&out[1 * 16..1 * 16 + 16]), "flex bone should move");
    }

    #[test]
    fn deterministic_same_inputs() {
        let bones = sample_bones();
        let mut s1 = WindSolver::new(&bones);
        let mut s2 = WindSolver::new(&bones);
        let a = solve(&mut s1, 2.5, 0.7, 0.3, -0.9);
        let b = solve(&mut s2, 2.5, 0.7, 0.3, -0.9);
        assert_eq!(a, b, "solver not deterministic");
        // Re-solving the SAME solver instance with the same inputs is stable too.
        let c = solve(&mut s1, 2.5, 0.7, 0.3, -0.9);
        assert_eq!(a, c, "re-solve differs");
    }

    #[test]
    fn output_is_finite_and_affine() {
        let bones = sample_bones();
        let mut solver = WindSolver::new(&bones);
        let out = solve(&mut solver, 0.42, 1.0, 1.0, 0.5);
        for &x in &out {
            assert!(x.is_finite());
        }
        // A valid affine transform has m[3][3] == 1 (column-major index 15).
        for c in 0..bones.len() {
            assert!((out[c * 16 + 15] - 1.0).abs() < 1e-5, "bone {c} m[3][3] != 1");
        }
    }

    #[test]
    fn tip_displacement_is_parallel_to_wind_dir() {
        // The solver rotates each bone about axis = cross(+Y, windDir), so a point
        // ABOVE a bone's pivot must displace ALONG windDir (a vertical bob aside).
        // This is the invariant flora_view::solve_wind relies on: it pre-rotates
        // the world wind dir into the tree-local frame (m3⁻¹·worldDir) so that,
        // after the model basis m3 is applied in the vertex shader, the branch
        // sway lands back on the WORLD wind direction. (The bug: feeding the world
        // dir straight in made branches sway m3·worldDir ≈ 90° off — leaves were
        // fine only because they add their gust in world space, after m3.)
        let bones = vec![
            WindBone { parent: -1, pivot: [0.0, 0.0, 0.0], axis: [0.0, 1.0, 0.0], stiffness: 0.0, branch_level: 0, is_rigid: 1 },
            WindBone { parent: 0,  pivot: [0.0, 1.0, 0.0], axis: [0.0, 1.0, 0.0], stiffness: 1.0, branch_level: 1, is_rigid: 0 },
        ];
        let mut solver = WindSolver::new(&bones);
        for &(dx, dz) in &[(1.0_f32, 0.0_f32), (0.0, 1.0), (0.7, -0.3), (-0.4, -0.9)] {
            // time 0.6 → bone 1's angle is non-zero, so it actually moves.
            let out = solve(&mut solver, 0.6, 1.0, dx as f64, dz as f64);
            let m = &out[16..32]; // bone 1's mat4 (column-major); parent is identity.
            // A point on the bone, above its pivot (0,1,0).
            let (px, py, pz) = (0.0_f32, 1.5, 0.0);
            let tx = m[0] * px + m[4] * py + m[8] * pz + m[12];
            let ty = m[1] * px + m[5] * py + m[9] * pz + m[13];
            let tz = m[2] * px + m[6] * py + m[10] * pz + m[14];
            let (hx, hz) = (tx - px, tz - pz); // horizontal displacement
            let _ = ty;
            let hmag = (hx * hx + hz * hz).sqrt();
            assert!(hmag > 1e-4, "bone should sway for dir ({dx},{dz}); got hmag={hmag}");
            // Parallel to (dx,dz): the 2D cross product is ~0.
            let dmag = (dx * dx + dz * dz).sqrt();
            let cross = (hx * dz - hz * dx) / (hmag * dmag);
            assert!(cross.abs() < 1e-4, "sway not parallel to wind dir ({dx},{dz}); cross={cross}");
        }
    }

    #[test]
    #[should_panic(expected = "parent < child")]
    fn forward_parent_panics() {
        let bad = vec![
            WindBone { parent: 1, pivot: [0.0; 3], axis: [0.0, 1.0, 0.0], stiffness: 0.0, branch_level: 0, is_rigid: 1 },
            WindBone { parent: -1, pivot: [0.0; 3], axis: [0.0, 1.0, 0.0], stiffness: 0.0, branch_level: 0, is_rigid: 0 },
        ];
        WindSolver::new(&bad);
    }
}

//! Stage B — tectonic plate simulation.
//!
//! Deterministic spherical crack-propagation plate-tectonics model.
//! Pure port of `demiurge/src/planet/tectonics.ts`. All randomness is
//! derived from splitmix32 sub-seeds; no system time or external entropy.
//!
//! Pipeline (all in `Tectonics::new`):
//!   1. Trace cracks         — sphere-walking walkers marking a crack bitmask.
//!   2. Fragmentation loop   — add cracks until ≥ target significant components.
//!   3. Flood fill           — BFS over non-crack texels → component ids.
//!   4. Merge to target      — absorb small/excess components; re-index.
//!   5. Plate properties     — types, omegas, seed-dirs, elevations.
//!   6. Distance + neighbour — Dijkstra on the cube-map grid.
//!   7. Conv/shear bake      — bake convergence + shear into blurred texel fields.
//!   8. Crust SDF            — continental crust mask + signed Dijkstra. (PHASE 2)
//!   9. Paleo-orogen network — fossil crack network + Dijkstra. (PHASE 2)
//!  10. Volcano bake         — arc + hotspot volcanoes. (PHASE 2)

use crate::noise::{fbm, Noise3D};
use glam::DVec3;

// ---------------------------------------------------------------------------
// Cube-map constants
// ---------------------------------------------------------------------------

// RES=512 to match ki (tectonics.ts:128) EXACTLY. The crack/arc band constants are
// absolute radians (STEP, SIN_HALF, ARC_CRUST_WIDTH=0.090, …), so the crack-walker →
// flood-fill → sliver-merge pipeline only rasterizes to ki's continents per-seed when
// RES equals ki's. RES=256 was an angularly-similar down-scale but produced a
// *different* set of continents. (Coupled: conv/shear blur is 12 passes at this RES.)
const RES: usize = 512;
const TOTAL_TEXELS: usize = 6 * RES * RES;
// Pre-computed texel angular size: texAng(512) = (π/2)/512 ≈ 0.003068 rad
// Matches demiurge cubemap.ts: texAng(res) = (Math.PI * 0.5) / res
const TEX_ANG: f64 = std::f64::consts::FRAC_PI_2 / 512.0;

// ---------------------------------------------------------------------------
// PRNG helpers — same mixer as noise.rs (splitmix32)
// ---------------------------------------------------------------------------

/// Derive a child seed from a master seed and a stream index.
/// Mirrors `deriveSeed` in tectonics.ts.
fn derive_seed(master_seed: u32, stream: u32) -> u32 {
    let s = master_seed ^ (stream.wrapping_add(1).wrapping_mul(0xdead_beef));
    // one splitmix32 step
    let mut z = s.wrapping_add(0x9e37_79b9);
    z = (z ^ (z >> 16)).wrapping_mul(0x85eb_ca6b);
    z = (z ^ (z >> 13)).wrapping_mul(0xc2b2_ae35);
    z ^ (z >> 16)
}

/// Make a [0,1) f64 rng from a seed (mirrors `makeRng` in tectonics.ts).
///
/// IMPORTANT: `makeRng` in tectonics.ts stores the FULL MIXED OUTPUT as state
/// (s = splitmix32Step(s)), while `splitmix32` in noise.ts stores the counter
/// (s += 0x9e3779b9) as state. These diverge from the 2nd call onward.
/// We must match tectonics.ts's hash-chain variant exactly.
fn make_rng(seed: u32) -> impl FnMut() -> f64 {
    let mut s: u32 = seed;
    move || {
        // Mirror: s = splitmix32Step(s) where splitmix32Step adds 0x9e3779b9 then hashes,
        // storing the hashed value back into s (not just the incremented counter).
        s = s.wrapping_add(0x9e3779b9);
        let mut z = s;
        z = (z ^ (z >> 16)).wrapping_mul(0x85ebca6b);
        z = (z ^ (z >> 13)).wrapping_mul(0xc2b2ae35);
        z = z ^ (z >> 16);
        s = z;  // store mixed output (matches JS `s = splitmix32Step(s)`)
        z as f64 / 0x1_0000_0000u64 as f64
    }
}

// ---------------------------------------------------------------------------
// Domain-warp constants (unchanged from tectonics.ts)
// ---------------------------------------------------------------------------

const WARP_AMP: f64 = 0.08;
const WARP_FREQ: f64 = 2.2;
const WARP_OCTAVES: u32 = 4;

// ---------------------------------------------------------------------------
// Walker / crack constants
// ---------------------------------------------------------------------------

const STEP: f64 = 0.005;
const BRANCH_PROB: f64 = 0.003_5;
/// cos(10°) — kill walkers entering polar caps
const POLAR_COS: f64 = 0.984_807_753_012_208;
/// cos(12°) — reject seed spawns inside polar caps
const SEED_POLAR_COS: f64 = 0.978_147_600_733_806;
const SELF_GRACE: usize = 16;
const MIN_BUDGET: f64 = 0.6;
const MAX_BUDGET: f64 = 3.0;
const MAX_STEPS_HARD: usize = 1200;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Per-plate data (mirroring `Plate` interface in tectonics.ts).
#[derive(Debug, Clone)]
pub struct Plate {
    pub id: usize,
    pub seed_dir: DVec3,
    pub plate_type: PlateType,
    pub base_elevation: f64,
    /// Angular velocity vector (ω), rad/step.
    pub omega: DVec3,
    /// RGB colour for debug/visualisation.
    pub color: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlateType {
    Oceanic,
    Continental,
}

/// Output from `Tectonics::query`.
///
/// Every field that is populated in Phase 1 is filled from real data.
/// Fields deferred to Phase 2 are zeroed and marked with `// PHASE 2`.
#[derive(Debug, Clone, Default)]
pub struct TectonicQuery {
    /// 0-based plate id for the queried direction.
    pub plate_id: usize,
    /// 0-based neighbour plate id (nearest across boundary).
    pub neighbor_id: usize,
    /// Distance to nearest plate boundary in radians.
    pub boundary_dist: f64,
    /// Relative convergence ≈ [-1, 1]: positive = converging.
    pub convergence: f64,
    /// Tangential (along-boundary) relative velocity component ≈ [-1, 1].
    pub shear: f64,
    /// Smoothed per-plate base elevation (no Voronoi step) — C1-sampled blurred field.
    pub base_elevation: f64,
    /// Signed distance: +inside crust, -outside crust (radians). // PHASE 2
    pub crust_dist: f64,
    /// Distance to nearest paleo/fossil crack boundary (radians). // PHASE 2
    pub paleo_dist: f64,
    /// crustDist mirrored to neighbour side (coarse, no fine warp). // PHASE 2
    pub other_crust_dist: f64,
    /// Substrate rock hardness [0,1]: 1=hard fresh igneous/volcanic, 0=soft sediment.
    /// C1-sampled from the baked rock_hardness field. Consumed in Phase-3 erosion + Phase-4 sampler.
    pub rock_hardness: f64,
}

// ---------------------------------------------------------------------------
// Cube-map helpers (inline — deliberately kept local to tectonics, per ki source)
// These are direct ports of cubemap.ts: dirToTexel, texelToDir,
// neighborTexel, texelIndex, sampleC1.
// ---------------------------------------------------------------------------

/// (face, x, y) texel coordinate.
#[derive(Debug, Clone, Copy, Default)]
struct Texel {
    face: usize,
    x: usize,
    y: usize,
}

/// `texelIndex(face, x, y, RES)` → flat array index.
#[inline(always)]
fn texel_index(face: usize, x: usize, y: usize) -> usize {
    face * RES * RES + y * RES + x
}

/// Map a unit direction to a cube-map texel.
/// Mirrors `dirToTexel` in cubemap.ts (using FACE_BASES UV convention).
///
/// FACE_BASES tangent axes (u=texel-x direction, v=texel-y direction):
///   Face 0 (+X): u = +Z, v = +Y  → sc = dir.z,  tc = dir.y
///   Face 1 (-X): u = -Z, v = +Y  → sc = -dir.z, tc = dir.y
///   Face 2 (+Y): u = +X, v = +Z  → sc = dir.x,  tc = dir.z
///   Face 3 (-Y): u = +X, v = -Z  → sc = dir.x,  tc = -dir.z
///   Face 4 (+Z): u = -X, v = +Y  → sc = -dir.x, tc = dir.y
///   Face 5 (-Z): u = +X, v = +Y  → sc = dir.x,  tc = dir.y
fn dir_to_texel(d: DVec3) -> Texel {
    let ax = d.x.abs();
    let ay = d.y.abs();
    let az = d.z.abs();

    let (face, u, v) = if ax >= ay && ax >= az {
        if d.x > 0.0 {
            (0,  d.z / d.x,  d.y / d.x) // +X: u=z, v=y
        } else {
            (1, -d.z / ax,   d.y / ax)   // -X: u=-z/|x|, v=y/|x|
        }
    } else if ay >= ax && ay >= az {
        if d.y > 0.0 {
            (2,  d.x / d.y,  d.z / d.y) // +Y: u=x, v=z
        } else {
            (3,  d.x / ay,  -d.z / ay)  // -Y: u=x/|y|, v=-z/|y|
        }
    } else if d.z > 0.0 {
        (4, -d.x / d.z,  d.y / d.z)    // +Z: u=-x, v=y
    } else {
        (5,  d.x / az,   d.y / az)     // -Z: u=x/|z|, v=y/|z|
    };

    let hres = (RES as f64) * 0.5;
    let xi = ((u + 1.0) * hres).floor() as isize;
    let yi = ((v + 1.0) * hres).floor() as isize;
    let x = xi.clamp(0, (RES as isize) - 1) as usize;
    let y = yi.clamp(0, (RES as isize) - 1) as usize;
    Texel { face, x, y }
}

/// Map a cube-map texel back to a unit direction.
/// Mirrors `texelToDir` in cubemap.ts (FACE_BASES: cube = normal + u*tangU + v*tangV).
///
///   Face 0 (+X): normal=(1,0,0) tangU=(0,0,1) tangV=(0,1,0) → (1,  v,  u)
///   Face 1 (-X): normal=(-1,0,0) tangU=(0,0,-1) tangV=(0,1,0) → (-1, v, -u)
///   Face 2 (+Y): normal=(0,1,0) tangU=(1,0,0) tangV=(0,0,1) → (u,  1,  v)
///   Face 3 (-Y): normal=(0,-1,0) tangU=(1,0,0) tangV=(0,0,-1) → (u, -1, -v)
///   Face 4 (+Z): normal=(0,0,1) tangU=(-1,0,0) tangV=(0,1,0) → (-u, v,  1)
///   Face 5 (-Z): normal=(0,0,-1) tangU=(1,0,0) tangV=(0,1,0) → (u,  v, -1)
fn texel_to_dir(face: usize, x: usize, y: usize) -> DVec3 {
    let hres = (RES as f64) * 0.5;
    let u = (x as f64 + 0.5) / hres - 1.0;
    let v = (y as f64 + 0.5) / hres - 1.0;

    let raw = match face {
        0 => DVec3::new(1.0,  v,  u),  // +X: (1, v, u)
        1 => DVec3::new(-1.0, v, -u),  // -X: (-1, v, -u)
        2 => DVec3::new(u,  1.0,  v),  // +Y: (u, 1, v)
        3 => DVec3::new(u, -1.0, -v),  // -Y: (u, -1, -v)
        4 => DVec3::new(-u,  v,  1.0), // +Z: (-u, v, 1)
        _ => DVec3::new(u,   v, -1.0), // -Z: (u, v, -1)
    };
    raw.normalize()
}

/// 4/8-connected cube-map neighbour with cross-face wrapping.
/// Mirrors `neighborTexel` in cubemap.ts (FACE_BASES UV tangent nudge).
/// `dx`, `dy` must each be in {-1, 0, 1}.
///
/// Cross-face: nudge the texel-centre direction along the face's U tangent
/// (for dx) and V tangent (for dy), then re-project — identical to demiurge's
/// `neighborTexel` which calls `texelToDir` then nudges along `b.ux/vx` etc.
fn neighbor_texel(face: usize, x: usize, y: usize, dx: i32, dy: i32) -> Texel {
    let nx = x as i32 + dx;
    let ny = y as i32 + dy;

    if nx >= 0 && nx < RES as i32 && ny >= 0 && ny < RES as i32 {
        return Texel { face, x: nx as usize, y: ny as usize };
    }

    // Cross-face: dir-roundtrip via 3D tangent nudge — matches demiurge's
    // `neighborTexel` in cubemap.ts exactly:
    //   1. texelToDir(face, x, y) → base direction d
    //   2. d += U_tangent * dx * texAng(RES) * 1.2  (for dx≠0)
    //   3. d += V_tangent * dy * texAng(RES) * 1.2  (for dy≠0)
    //   4. normalize → dirToTexel
    //
    // FACE_BASES tangent vectors from faceBases.ts:
    //   0 (+X):  U=(0,0,1),  V=(0,1,0)
    //   1 (-X):  U=(0,0,-1), V=(0,1,0)
    //   2 (+Y):  U=(1,0,0),  V=(0,0,1)
    //   3 (-Y):  U=(1,0,0),  V=(0,0,-1)
    //   4 (+Z):  U=(-1,0,0), V=(0,1,0)
    //   5 (-Z):  U=(1,0,0),  V=(0,1,0)
    let d = texel_to_dir(face, x, y);
    let ang = TEX_ANG * 1.2;

    let mut nx_f = d.x;
    let mut ny_f = d.y;
    let mut nz_f = d.z;

    if dx != 0 {
        let df = dx as f64 * ang;
        let (ux, uy, uz): (f64, f64, f64) = match face {
            0 => (0.0, 0.0,  1.0),
            1 => (0.0, 0.0, -1.0),
            2 => (1.0, 0.0,  0.0),
            3 => (1.0, 0.0,  0.0),
            4 => (-1.0, 0.0, 0.0),
            _ => (1.0, 0.0,  0.0),
        };
        nx_f += ux * df;
        ny_f += uy * df;
        nz_f += uz * df;
    }
    if dy != 0 {
        let df = dy as f64 * ang;
        let (vx, vy, vz): (f64, f64, f64) = match face {
            0 => (0.0, 1.0, 0.0),
            1 => (0.0, 1.0, 0.0),
            2 => (0.0, 0.0, 1.0),
            3 => (0.0, 0.0, -1.0),
            4 => (0.0, 1.0, 0.0),
            _ => (0.0, 1.0, 0.0),
        };
        nx_f += vx * df;
        ny_f += vy * df;
        nz_f += vz * df;
    }
    dir_to_texel(DVec3::new(nx_f, ny_f, nz_f).normalize())
}

/// C1 smoothstep bilinear sample of a Float32 cubemap field at a unit direction.
/// Plain bilinear except smoothstep is applied to the fractional texel coords
/// before the bilinear lerp (`fu = fu*fu*(3-2*fu)`, same for `fv`),
/// giving zero slope at cell edges — no C0 kinks / boundary staircasing.
/// Mirrors `_sampleC1` in tectonics.ts (lines ~3118-3186).
/// NOT safe for sentinel-valued fields (comp_id / neighbor_id — use direct lookup).
fn sample_c1(field: &[f32], dir: DVec3) -> f64 {
    let ax = dir.x.abs();
    let ay = dir.y.abs();
    let az = dir.z.abs();

    let (face, sc, tc, mc) = if ax >= ay && ax >= az {
        if dir.x > 0.0 { (0usize, dir.z,  dir.y,  ax) }
        else            { (1,     -dir.z,  dir.y,  ax) }
    } else if ay >= ax && ay >= az {
        if dir.y > 0.0 { (2,  dir.x,  dir.z,  ay) }
        else            { (3,  dir.x, -dir.z,  ay) }
    } else if dir.z > 0.0 {
        (4, -dir.x,  dir.y,  az)
    } else {
        (5,  dir.x,  dir.y,  az)
    };

    let hres = (RES as f64) * 0.5;
    let fx_cont = (sc / mc + 1.0) * hres - 0.5;
    let fy_cont = (tc / mc + 1.0) * hres - 0.5;

    let x0i = fx_cont.floor() as i32;
    let y0i = fy_cont.floor() as i32;
    let mut fu = fx_cont - fx_cont.floor();
    let mut fv = fy_cont - fy_cont.floor();
    // smoothstep for C1 continuity — zero slope at cell edges, no kinks
    fu = fu * fu * (3.0 - 2.0 * fu);
    fv = fv * fv * (3.0 - 2.0 * fv);
    let x1i = x0i + 1;
    let y1i = y0i + 1;

    let fetch = |xc: usize, yc: usize, dx: i32, dy: i32| -> f64 {
        let t = neighbor_texel(face, xc, yc, dx, dy);
        field[texel_index(t.face, t.x, t.y)] as f64
    };

    let v00: f64;
    let v10: f64;
    let v01: f64;
    let v11: f64;

    if x0i >= 0 && y0i >= 0 && x1i < RES as i32 && y1i < RES as i32 {
        let base = face * RES * RES;
        let x0 = x0i as usize; let y0 = y0i as usize;
        let x1 = x1i as usize; let y1 = y1i as usize;
        v00 = field[base + y0 * RES + x0] as f64;
        v10 = field[base + y0 * RES + x1] as f64;
        v01 = field[base + y1 * RES + x0] as f64;
        v11 = field[base + y1 * RES + x1] as f64;
    } else {
        let axx = x0i.clamp(0, (RES as i32) - 1) as usize;
        let ayy = y0i.clamp(0, (RES as i32) - 1) as usize;
        let ox0 = x0i - axx as i32;
        let ox1 = x1i - axx as i32;
        let oy0 = y0i - ayy as i32;
        let oy1 = y1i - ayy as i32;
        v00 = fetch(axx, ayy, ox0, oy0);
        v10 = fetch(axx, ayy, ox1, oy0);
        v01 = fetch(axx, ayy, ox0, oy1);
        v11 = fetch(axx, ayy, ox1, oy1);
    }

    let top = v00 + (v10 - v00) * fu;
    let bot = v01 + (v11 - v01) * fu;
    top + (bot - top) * fv
}

// ---------------------------------------------------------------------------
// Crack-mask dilation (4-connected)
// Mirrors `dilate4` in tectonics.ts.
// ---------------------------------------------------------------------------

fn dilate4(src: &[u8]) -> Vec<u8> {
    let mut dst = src.to_vec();
    for (i, &val) in src.iter().enumerate() {
        if val != 1 { continue; }
        let face = i / (RES * RES);
        let rem  = i - face * RES * RES;
        let y    = rem / RES;
        let x    = rem % RES;
        for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
            let nb = neighbor_texel(face, x, y, dx, dy);
            dst[texel_index(nb.face, nb.x, nb.y)] = 1;
        }
    }
    dst
}

// ---------------------------------------------------------------------------
// Flood fill — BFS → component ids.
// Mirrors `floodFill` in tectonics.ts.
// ---------------------------------------------------------------------------

struct FloodFillResult {
    comp_id: Vec<u16>,
    comp_count: usize,
}

fn flood_fill(crack_mask: &[u8]) -> FloodFillResult {
    let mut comp_id = vec![0u16; TOTAL_TEXELS];
    let mut queue: Vec<usize> = Vec::with_capacity(TOTAL_TEXELS);
    let mut next_comp: u16 = 1;

    // Phase 1: assign components to non-crack texels.
    for seed in 0..TOTAL_TEXELS {
        if crack_mask[seed] != 0 || comp_id[seed] != 0 { continue; }

        let comp = next_comp;
        next_comp += 1;
        comp_id[seed] = comp;
        queue.clear();
        queue.push(seed);
        let mut head = 0;

        while head < queue.len() {
            let idx = queue[head];
            head += 1;
            let face = idx / (RES * RES);
            let rem  = idx - face * RES * RES;
            let y    = rem / RES;
            let x    = rem % RES;
            for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let nb = neighbor_texel(face, x, y, dx, dy);
                let ni = texel_index(nb.face, nb.x, nb.y);
                if crack_mask[ni] == 0 && comp_id[ni] == 0 {
                    comp_id[ni] = comp;
                    queue.push(ni);
                }
            }
        }
    }

    // Phase 2: assign crack texels to nearest component via multi-source BFS.
    queue.clear();
    for idx in 0..TOTAL_TEXELS {
        if crack_mask[idx] != 0 || comp_id[idx] == 0 { continue; }
        let face = idx / (RES * RES);
        let rem  = idx - face * RES * RES;
        let y    = rem / RES;
        let x    = rem % RES;
        for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
            let nb = neighbor_texel(face, x, y, dx, dy);
            let ni = texel_index(nb.face, nb.x, nb.y);
            if crack_mask[ni] != 0 && comp_id[ni] == 0 {
                comp_id[ni] = comp_id[idx];
                queue.push(ni);
            }
        }
    }
    {
        let mut head = 0;
        while head < queue.len() {
            let idx = queue[head];
            head += 1;
            let comp = comp_id[idx];
            let face = idx / (RES * RES);
            let rem  = idx - face * RES * RES;
            let y    = rem / RES;
            let x    = rem % RES;
            for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let nb = neighbor_texel(face, x, y, dx, dy);
                let ni = texel_index(nb.face, nb.x, nb.y);
                if comp_id[ni] == 0 {
                    comp_id[ni] = comp;
                    queue.push(ni);
                }
            }
        }
    }

    // Any remaining unassigned texel gets comp 1 (degenerate guard).
    for v in comp_id.iter_mut() {
        if *v == 0 { *v = 1; }
    }

    FloodFillResult { comp_id, comp_count: (next_comp - 1) as usize }
}

// ---------------------------------------------------------------------------
// Minimal binary min-heap for Dijkstra
// Mirrors `BinaryHeap` class in tectonics.ts.
// ---------------------------------------------------------------------------

// Mirrors `BinaryHeap` class in tectonics.ts.
// JS uses Float32Array for keys, so all heap key comparisons and accumulated
// distance propagation are at f32 precision. Rust must match exactly.
struct BinaryHeap {
    keys: Vec<f32>,
    vals: Vec<usize>,
}

impl BinaryHeap {
    fn new(capacity: usize) -> Self {
        BinaryHeap {
            keys: Vec::with_capacity(capacity),
            vals: Vec::with_capacity(capacity),
        }
    }

    fn is_empty(&self) -> bool { self.keys.is_empty() }

    /// Push with key truncated to f32 to match JS Float32Array heap behaviour.
    fn push(&mut self, key: f64, val: usize) {
        self.keys.push(key as f32);
        self.vals.push(val);
        let mut i = self.keys.len() - 1;
        while i > 0 {
            let p = (i - 1) >> 1;
            if self.keys[p] <= self.keys[i] { break; }
            self.keys.swap(p, i);
            self.vals.swap(p, i);
            i = p;
        }
    }

    /// Pop returns key as f64 (f32 promoted), matching JS Float32Array read behaviour.
    fn pop_min(&mut self) -> (f64, usize) {
        let key = self.keys[0] as f64;
        let val = self.vals[0];
        let last = self.keys.len() - 1;
        self.keys[0] = self.keys[last];
        self.vals[0] = self.vals[last];
        self.keys.pop();
        self.vals.pop();
        if !self.keys.is_empty() {
            self.sift_down(0);
        }
        (key, val)
    }

    fn sift_down(&mut self, mut i: usize) {
        let n = self.keys.len();
        loop {
            let l = 2 * i + 1;
            let r = 2 * i + 2;
            let mut smallest = i;
            if l < n && self.keys[l] < self.keys[smallest] { smallest = l; }
            if r < n && self.keys[r] < self.keys[smallest] { smallest = r; }
            if smallest == i { break; }
            self.keys.swap(i, smallest);
            self.vals.swap(i, smallest);
            i = smallest;
        }
    }
}

// ---------------------------------------------------------------------------
// Great-circle crack injection
// Mirrors `crackGreatCircle` in tectonics.ts.
// ---------------------------------------------------------------------------

fn crack_great_circle(
    crack_mask: &mut [u8],
    normal: DVec3,
    comp_id: Option<&[u16]>,
    target_comp: u16,
) {
    let sin_half = 0.020f64; // ~1.1° band ≈ 2 texels wide
    let n = normal.normalize();
    for f in 0..6 {
        for ty in 0..RES {
            for tx in 0..RES {
                let d = texel_to_dir(f, tx, ty);
                let dot = d.dot(n);
                if dot.abs() < sin_half {
                    let idx = texel_index(f, tx, ty);
                    if let Some(cid) = comp_id {
                        if cid[idx] != target_comp { continue; }
                    }
                    crack_mask[idx] = 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Crack walker — sphere-walking walkers.
// Mirrors `traceCracks` in tectonics.ts.
// ---------------------------------------------------------------------------

struct Walker {
    pos: DVec3,
    tan: DVec3,
    budget: f64,
    steps: usize,
    bias: f64,
    trail: [usize; SELF_GRACE],
    trail_head: usize,
}

/// Run one round of crack tracing.
/// `rng` must be a &mut dyn FnMut()->f64.
/// `spawn_filter`: optional closure (texel_idx -> bool); only spawn positions
/// that pass it are accepted.
fn trace_cracks(
    crack_mask: &mut [u8],
    rng: &mut impl FnMut() -> f64,
    seed_count: usize,
    max_walkers: usize,
    spawn_filter: Option<&dyn Fn(usize) -> bool>,
) {
    let mut walkers: Vec<Walker> = Vec::new();

    let spawn_walker = |walkers: &mut Vec<Walker>,
                             rng: &mut dyn FnMut() -> f64,
                             pos: DVec3,
                             heading: DVec3,
                             parent_trail: Option<([usize; SELF_GRACE], usize)>| {
        if walkers.len() >= max_walkers { return; }
        let budget = MIN_BUDGET + rng() * (MAX_BUDGET - MIN_BUDGET);
        let bias   = (rng() - 0.5) * 0.08;
        let mut trail = [usize::MAX; SELF_GRACE];
        let mut trail_head = 0usize;
        if let Some((pt, ph)) = parent_trail {
            trail = pt;
            trail_head = ph;
        }
        walkers.push(Walker { pos, tan: heading, budget, steps: 0, bias, trail, trail_head });
    };

    // Seed K starting positions.
    for _ in 0..seed_count {
        let max_attempts = if spawn_filter.is_some() { 200 } else { 50 };
        let mut px = 0.0f64; let mut py = 0.0f64; let mut pz = 0.0f64;
        'spawn: for _ in 0..max_attempts {
            let rx = rng() * 2.0 - 1.0;
            let ry = rng() * 2.0 - 1.0;
            let rz = rng() * 2.0 - 1.0;
            let len = (rx * rx + ry * ry + rz * rz).sqrt();
            if len <= 0.01 || len >= 1.0 { continue; }
            let nx = rx / len; let ny = ry / len; let nz = rz / len;
            if ny.abs() > SEED_POLAR_COS { continue; }
            if let Some(filter) = spawn_filter {
                let t = dir_to_texel(DVec3::new(nx, ny, nz));
                let idx = texel_index(t.face, t.x, t.y);
                if !filter(idx) { continue; }
            }
            px = nx; py = ny; pz = nz;
            break 'spawn;
        }
        if px == 0.0 && py == 0.0 && pz == 0.0 { continue; }
        let pos = DVec3::new(px, py, pz);

        // Random tangent ⊥ pos
        let arb = if px.abs() < 0.9 { DVec3::new(1.0, 0.0, 0.0) } else { DVec3::new(0.0, 1.0, 0.0) };
        let heading = pos.cross(arb).normalize();
        spawn_walker(&mut walkers, rng, pos, heading, None);
    }

    // Process walkers sequentially.
    // wi stays at 0: remove(0) shifts remaining walkers forward, so we always
    // re-check index 0 (the next walker after the removed one).
    let wi = 0;
    while wi < walkers.len() {
        // Polar-cap kill
        if walkers[wi].pos.y.abs() > POLAR_COS {
            walkers.remove(wi);
            continue;
        }
        // Budget/step cap
        if walkers[wi].budget <= 0.0 || walkers[wi].steps >= MAX_STEPS_HARD {
            walkers.remove(wi);
            continue;
        }

        let cur_t = dir_to_texel(walkers[wi].pos);
        let cur_idx = texel_index(cur_t.face, cur_t.x, cur_t.y);

        // Self-collision check: die if texel is crack-marked by someone else.
        if crack_mask[cur_idx] == 1 {
            let trail_len = walkers[wi].trail_head.min(SELF_GRACE);
            let own_mark = walkers[wi].trail[..trail_len].contains(&cur_idx);
            if !own_mark {
                walkers.remove(wi);
                continue;
            }
        }

        let prev_t = cur_t;
        crack_mask[cur_idx] = 1;
        {
            let th = walkers[wi].trail_head % SELF_GRACE;
            walkers[wi].trail[th] = cur_idx;
            walkers[wi].trail_head += 1;
        }

        // Advance position
        let step = DVec3::new(
            walkers[wi].pos.x + STEP * walkers[wi].tan.x,
            walkers[wi].pos.y + STEP * walkers[wi].tan.y,
            walkers[wi].pos.z + STEP * walkers[wi].tan.z,
        );
        walkers[wi].pos = step.normalize();

        // Re-orthonormalise tan against new pos
        let tdot = walkers[wi].tan.dot(walkers[wi].pos);
        let to = (walkers[wi].tan - tdot * walkers[wi].pos).normalize();
        walkers[wi].tan = to;

        // Rodrigues curvature rotation — mirrors JS exactly:
        // w.tan.set(tan * cosc + cross * sinc) without re-normalizing.
        // JS leaves the result un-normalized (mathematically unit, but FP drift
        // accumulates). The next step's re-orthonormalize handles any drift.
        let curv = (rng() - 0.5) * 0.18 + walkers[wi].bias;
        let cosc = curv.cos();
        let sinc = curv.sin();
        let c = walkers[wi].pos.cross(walkers[wi].tan);
        walkers[wi].tan = walkers[wi].tan * cosc + c * sinc;

        // 4-connectivity diagonal fix
        let new_t = dir_to_texel(walkers[wi].pos);
        if new_t.face == prev_t.face {
            let fdx = new_t.x as i32 - prev_t.x as i32;
            let fdy = new_t.y as i32 - prev_t.y as i32;
            if fdx.abs() == 1 && fdy.abs() == 1 {
                let bx = (prev_t.x as i32 + fdx).clamp(0, (RES as i32) - 1) as usize;
                let by2 = prev_t.y;
                if bx < RES && by2 < RES {
                    let bridge_idx = texel_index(new_t.face, bx, by2);
                    crack_mask[bridge_idx] = 1;
                    let th = walkers[wi].trail_head % SELF_GRACE;
                    walkers[wi].trail[th] = bridge_idx;
                    walkers[wi].trail_head += 1;
                }
            }
        }

        walkers[wi].budget -= STEP;
        walkers[wi].steps += 1;

        // Branching — mirror JS exactly: branch_tan is NOT re-normalized.
        if rng() < BRANCH_PROB && walkers.len() < max_walkers {
            let branch_angle = (100.0 + rng() * 40.0) * (std::f64::consts::PI / 180.0);
            let sign = if rng() < 0.5 { 1.0 } else { -1.0 };
            let ba = branch_angle * sign;
            let bc = walkers[wi].pos.cross(walkers[wi].tan);
            let branch_tan = walkers[wi].tan * ba.cos() + bc * ba.sin();
            let branch_pos = walkers[wi].pos;
            let parent = (walkers[wi].trail, walkers[wi].trail_head);
            spawn_walker(&mut walkers, rng, branch_pos, branch_tan, Some(parent));
        }
        // Do NOT advance wi; walker re-processed until it dies (splice-equivalent).
    }
}

// ---------------------------------------------------------------------------
// Pole-of-inaccessibility helper
// Mirrors `poleOfInaccessibility` in tectonics.ts.
// ---------------------------------------------------------------------------

fn pole_of_inaccessibility(plate_id: u16, comp_id: &[u16], dist_field: &[f32]) -> DVec3 {
    let mut max_dist = -1.0f32;
    let mut best_idx = 0usize;
    for i in 0..TOTAL_TEXELS {
        if comp_id[i] == plate_id && dist_field[i] > max_dist {
            max_dist = dist_field[i];
            best_idx = i;
        }
    }
    let face = best_idx / (RES * RES);
    let rem  = best_idx - face * RES * RES;
    let y    = rem / RES;
    let x    = rem % RES;
    texel_to_dir(face, x, y)
}

// ---------------------------------------------------------------------------
// Merge to target — Step 4
// Mirrors `mergeToTarget` in tectonics.ts.
// ---------------------------------------------------------------------------

/// Compact component ids to 0-based with pole plates first (id 0 = north, id 1 = south).
fn remap_components(comp_id: &mut [u16], id_map: &std::collections::HashMap<u16, u16>) {
    for v in comp_id.iter_mut() {
        if let Some(&mapped) = id_map.get(v) {
            *v = mapped;
        }
    }
}

fn merge_to_target(
    comp_id: &mut [u16],
    comp_count: usize,
    target_count: usize,
    pole_comp_n: u16,
    pole_comp_s: u16,
    max_share: f64,
) -> usize {
    let sliver_thresh = (TOTAL_TEXELS as f64 * 0.001) as usize;
    let mut current_count = comp_count;

    // Helper: compute areas
    let compute_areas = |cc: usize, cid: &[u16]| -> Vec<usize> {
        let mut areas = vec![0usize; cc + 2];
        for &v in cid.iter() {
            if (v as usize) <= cc { areas[v as usize] += 1; }
        }
        areas
    };

    // Helper: build adjacency (comp -> sorted map of adjacent comps with boundary count)
    // Uses BTreeMap for deterministic iteration order (matches JS Map<> insertion-sorted behaviour
    // when we also sort candidates by area; avoids HashMap random-seed non-determinism).
    let build_adjacency = |cid: &[u16]| -> Vec<std::collections::BTreeMap<u16, u32>> {
        let max_id = *cid.iter().max().unwrap_or(&0) as usize;
        let mut adj: Vec<std::collections::BTreeMap<u16, u32>> = vec![std::collections::BTreeMap::new(); max_id + 1];
        for idx in 0..TOTAL_TEXELS {
            let ca = cid[idx];
            let face = idx / (RES * RES);
            let rem  = idx - face * RES * RES;
            let y    = rem / RES;
            let x    = rem % RES;
            for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let nb = neighbor_texel(face, x, y, dx, dy);
                let ni = texel_index(nb.face, nb.x, nb.y);
                let cb = cid[ni];
                if cb != ca {
                    let ca_usize = ca as usize;
                    if ca_usize < adj.len() {
                        *adj[ca_usize].entry(cb).or_insert(0) += 1;
                    }
                }
            }
        }
        adj
    };

    // (a) Sliver cleanup — 3 passes
    #[allow(clippy::needless_range_loop)]
    for _ in 0..3 {
        let areas = compute_areas(comp_count, comp_id);
        let adj = build_adjacency(comp_id);
        let mut changed = false;
        for c in 1..=comp_count {
            if areas[c] == 0 { continue; }
            let cu = c as u16;
            if cu == pole_comp_n || cu == pole_comp_s { continue; }
            if areas[c] < sliver_thresh {
                // Find best neighbour (longest boundary; tie-break by smaller id for determinism)
                let best_nbr = adj.get(c).and_then(|m| {
                    m.iter().max_by(|&(ka, &va), &(kb, &vb)| {
                        va.cmp(&vb).then_with(|| kb.cmp(ka)) // higher count wins; tie → smaller id
                    }).map(|(&nb, _)| nb)
                });
                if let Some(nbr) = best_nbr {
                    if nbr > 0 {
                        for v in comp_id.iter_mut() {
                            if *v == cu { *v = nbr; }
                        }
                        changed = true;
                    }
                }
            }
        }
        if !changed { break; }
        let live: std::collections::BTreeSet<u16> = comp_id.iter().cloned().collect();
        current_count = live.len();
    }

    // (b) Union-find reduce to targetCount
    let mut parent: Vec<u16> = (0..=(comp_count as u16 + 1)).collect();
    let mut areas_uf = compute_areas(comp_count, comp_id);

    fn uf_find(parent: &mut [u16], mut x: u16) -> u16 {
        while parent[x as usize] != x {
            let gp = parent[parent[x as usize] as usize];
            parent[x as usize] = gp;
            x = gp;
        }
        x
    }

    // Build root neighbors from adjacency using BTreeSet for deterministic iteration
    let adj_full = build_adjacency(comp_id);
    let mut root_nbrs: Vec<std::collections::BTreeSet<u16>> = vec![std::collections::BTreeSet::new(); comp_count + 2];
    for ca in 1..=comp_count {
        if let Some(m) = adj_full.get(ca) {
            for &cb in m.keys() {
                root_nbrs[ca].insert(cb);
                let cb_usize = cb as usize;
                if cb_usize < root_nbrs.len() {
                    root_nbrs[cb_usize].insert(ca as u16);
                }
            }
        }
    }

    let cap_texels = (max_share * TOTAL_TEXELS as f64) as usize;

    // Collect candidates sorted by area ascending
    let mut candidates: Vec<u16> = (1..=(comp_count as u16)).collect();
    candidates.sort_by_key(|&c| areas_uf[c as usize]);

    for cand in candidates {
        if current_count <= target_count { break; }
        let rc = uf_find(&mut parent, cand);
        if areas_uf[rc as usize] == 0 { continue; }
        if rc == uf_find(&mut parent, pole_comp_n) || rc == uf_find(&mut parent, pole_comp_s) { continue; }

        let nbrs = &root_nbrs[rc as usize];
        if nbrs.is_empty() { continue; }

        let resolved: Vec<u16> = nbrs.iter()
            .map(|&n| uf_find(&mut parent, n))
            .filter(|&rn| rn != rc)
            .collect();
        let mut unique_roots: std::collections::BTreeSet<u16> = resolved.into_iter().collect();
        unique_roots.remove(&rc);

        if unique_roots.is_empty() { continue; }

        let area_s = areas_uf[rc as usize];

        // Pick best candidate: smallest area under cap; tie-break by smaller id (matches JS `rn < bestRoot`).
        // Fallback: smallest combined area; same tie-break.
        let mut best_root: Option<u16> = None;
        let mut best_area = usize::MAX;
        for &rn in &unique_roots {
            let combined = area_s + areas_uf[rn as usize];
            let rn_area = areas_uf[rn as usize];
            if combined <= cap_texels {
                if rn_area < best_area || (rn_area == best_area && best_root.map_or(true, |br| rn < br)) {
                    best_area = rn_area;
                    best_root = Some(rn);
                }
            }
        }
        if best_root.is_none() {
            let mut best_combined = usize::MAX;
            for &rn in &unique_roots {
                let combined = area_s + areas_uf[rn as usize];
                if combined < best_combined || (combined == best_combined && best_root.map_or(true, |br| rn < br)) {
                    best_combined = combined;
                    best_root = Some(rn);
                }
            }
        }

        let dst = match best_root { Some(r) => r, None => continue };
        let src = rc;

        // Union: src into dst
        parent[src as usize] = dst;
        areas_uf[dst as usize] += areas_uf[src as usize];
        areas_uf[src as usize] = 0;

        // Merge neighbor sets (BTreeSet iteration is sorted, so deterministic)
        let src_nbrs: Vec<u16> = root_nbrs[src as usize].iter().cloned().collect();
        for n in src_nbrs {
            let rn = uf_find(&mut parent, n);
            if rn == dst { continue; }
            root_nbrs[dst as usize].insert(rn);
            root_nbrs[rn as usize].remove(&src);
            root_nbrs[rn as usize].insert(dst);
        }
        root_nbrs[src as usize].clear();
        current_count -= 1;
    }

    let pole_root_n = uf_find(&mut parent, pole_comp_n);
    let pole_root_s = uf_find(&mut parent, pole_comp_s);

    // Apply union-find
    for v in comp_id.iter_mut() {
        *v = uf_find(&mut parent, *v);
    }

    // Collect live ids (BTreeSet for deterministic sorted order)
    let live_ids: std::collections::BTreeSet<u16> = comp_id.iter().cloned().collect();

    // Build id_map: pole first, then sorted
    let mut id_map: std::collections::HashMap<u16, u16> = std::collections::HashMap::new();
    let mut next_id: u16 = 0;
    if pole_root_n == pole_root_s {
        id_map.insert(pole_root_n, next_id);
        next_id += 1;
    } else {
        id_map.insert(pole_root_n, next_id); next_id += 1;
        id_map.insert(pole_root_s, next_id); next_id += 1;
    }
    let mut others: Vec<u16> = live_ids.iter()
        .cloned()
        .filter(|&c| c != pole_root_n && c != pole_root_s)
        .collect();
    others.sort_unstable();
    for c in others {
        id_map.insert(c, next_id);
        next_id += 1;
    }

    remap_components(comp_id, &id_map);
    id_map.len()
}

// ---------------------------------------------------------------------------
// Distance field — Dijkstra
// Mirrors `buildDistanceField` in tectonics.ts.
// ---------------------------------------------------------------------------

struct DistanceFieldResult {
    dist_field: Vec<f32>,
    neighbor_id: Vec<u16>,
}

fn build_distance_field(comp_id: &[u16], plate_count: usize) -> DistanceFieldResult {
    let inf = 1.0e30f64;
    let mut dist_field = vec![inf as f32; TOTAL_TEXELS];
    let mut neighbor_id = vec![0u16; TOTAL_TEXELS];
    let mut nearest_boundary = vec![0usize; TOTAL_TEXELS];
    let mut settled = vec![false; TOTAL_TEXELS];

    let diag_cost = TEX_ANG * std::f64::consts::SQRT_2;
    let orth_cost = TEX_ANG;

    let mut heap = BinaryHeap::new(TOTAL_TEXELS * 2);

    // Seed: boundary texels — those with a 4-neighbour of different plateId
    for idx in 0..TOTAL_TEXELS {
        let face = idx / (RES * RES);
        let rem  = idx - face * RES * RES;
        let y    = rem / RES;
        let x    = rem % RES;
        let pid  = comp_id[idx];
        for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
            let nb = neighbor_texel(face, x, y, dx, dy);
            let ni = texel_index(nb.face, nb.x, nb.y);
            if comp_id[ni] != pid {
                if dist_field[idx] > 0.0 {
                    dist_field[idx] = 0.0;
                    nearest_boundary[idx] = idx;
                    heap.push(0.0, idx);
                }
                break;
            }
        }
    }

    // Dijkstra — 8-connected expansion
    const DX8: [i32; 8] = [1, -1, 0, 0, 1, -1, 1, -1];
    const DY8: [i32; 8] = [0, 0, 1, -1, 1, 1, -1, -1];

    while !heap.is_empty() {
        let (dist, idx) = heap.pop_min();
        if settled[idx] { continue; }
        settled[idx] = true;

        let face = idx / (RES * RES);
        let rem  = idx - face * RES * RES;
        let y    = rem / RES;
        let x    = rem % RES;

        for d in 0..8 {
            let nb = neighbor_texel(face, x, y, DX8[d], DY8[d]);
            let ni = texel_index(nb.face, nb.x, nb.y);
            if settled[ni] { continue; }
            let cost = if DX8[d] != 0 && DY8[d] != 0 { diag_cost } else { orth_cost };
            let new_dist = dist + cost;
            if new_dist < dist_field[ni] as f64 {
                dist_field[ni] = new_dist as f32;
                nearest_boundary[ni] = nearest_boundary[idx];
                heap.push(new_dist, ni);
            }
        }
    }

    // Build neighborId from nearest_boundary
    for idx in 0..TOTAL_TEXELS {
        let own_plate = comp_id[idx];
        let src = nearest_boundary[idx];

        let src_face = src / (RES * RES);
        let src_rem  = src - src_face * RES * RES;
        let src_y    = src_rem / RES;
        let src_x    = src_rem % RES;

        let mut found = own_plate; // fallback: own plate
        if comp_id[src] != own_plate {
            found = comp_id[src];
        } else {
            for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let nb = neighbor_texel(src_face, src_x, src_y, dx, dy);
                let ni = texel_index(nb.face, nb.x, nb.y);
                if comp_id[ni] != own_plate {
                    found = comp_id[ni];
                    break;
                }
            }
        }
        // Clamp to valid range
        neighbor_id[idx] = (found as usize).min(plate_count.saturating_sub(1)) as u16;
    }

    DistanceFieldResult { dist_field, neighbor_id }
}

// ---------------------------------------------------------------------------
// Box blur helpers
// Mirrors `blurField` / `blurFieldN` in tectonics.ts.
// ---------------------------------------------------------------------------

fn blur_field(src: &[f32]) -> Vec<f32> {
    let mut dst = vec![0.0f32; TOTAL_TEXELS];
    for idx in 0..TOTAL_TEXELS {
        let face = idx / (RES * RES);
        let rem  = idx - face * RES * RES;
        let y    = rem / RES;
        let x    = rem % RES;
        let mut sum = src[idx] as f64 * 4.0;
        for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
            let nb = neighbor_texel(face, x, y, dx, dy);
            sum += src[texel_index(nb.face, nb.x, nb.y)] as f64;
        }
        dst[idx] = (sum / 8.0) as f32;
    }
    dst
}

fn blur_field_n(mut src: Vec<f32>, n: usize) -> Vec<f32> {
    for _ in 0..n { src = blur_field(&src); }
    src
}

// ---------------------------------------------------------------------------
// HSL → RGB (for plate colours)
// Mirrors `hslToRgb` / `hue2rgb` in tectonics.ts.
// ---------------------------------------------------------------------------

fn hue2rgb(p: f64, q: f64, mut t: f64) -> f64 {
    if t < 0.0 { t += 1.0; }
    if t > 1.0 { t -= 1.0; }
    if t < 1.0 / 6.0 { return p + (q - p) * 6.0 * t; }
    if t < 0.5 { return q; }
    if t < 2.0 / 3.0 { return p + (q - p) * (2.0 / 3.0 - t) * 6.0; }
    p
}

fn hsl_to_rgb(h: f64, s: f64, l: f64) -> [f64; 3] {
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    [hue2rgb(p, q, h + 1.0 / 3.0), hue2rgb(p, q, h), hue2rgb(p, q, h - 1.0 / 3.0)]
}

// ---------------------------------------------------------------------------
// Phase 2 helpers — smoothstep + gaussian (mirrors _ss / _g in tectonics.ts)
// ---------------------------------------------------------------------------

/// GLSL-style smoothstep.  Mirrors `_ss(edge0, edge1, x)` in tectonics.ts.
#[inline(always)]
fn ss(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Gaussian.  Mirrors `_g(x, center, width)` in tectonics.ts.
#[inline(always)]
fn gauss(x: f64, center: f64, width: f64) -> f64 {
    let t = (x - center) / width;
    (-t * t).exp()
}

// ---------------------------------------------------------------------------
// Crust SDF — Dijkstra signed distance field from a crust mask
// Mirrors Step 6b crust-SDF block in tectonics.ts constructor.
// Input:  crustMask[i] = 1 if inside continental crust, 0 otherwise.
// Output: signed distance: positive inside crust, negative outside (radians).
//         Capped at ±1.5 for uniform/degenerate inputs.
// ---------------------------------------------------------------------------

fn build_crust_sdf(crust_mask: &[u8]) -> Vec<f32> {
    let total_crust: usize = crust_mask.iter().filter(|&&v| v == 1).count();

    if total_crust == 0 {
        return vec![-1.5f32; TOTAL_TEXELS];
    }
    if total_crust == TOTAL_TEXELS {
        return vec![1.5f32; TOTAL_TEXELS];
    }

    let inf = 1.0e30f64;
    let mut c_dist   = vec![inf as f32; TOTAL_TEXELS];
    let mut settled  = vec![false; TOTAL_TEXELS];
    let mut heap     = BinaryHeap::new(TOTAL_TEXELS * 2);

    let diag_cost = TEX_ANG * std::f64::consts::SQRT_2;
    let orth_cost = TEX_ANG;

    // Seed: crust edge texels (crustMask differs from any 4-neighbour)
    for idx in 0..TOTAL_TEXELS {
        let face = idx / (RES * RES);
        let rem  = idx - face * RES * RES;
        let cy   = rem / RES;
        let cx   = rem % RES;
        let mine = crust_mask[idx];
        let mut is_edge = false;
        for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
            let nb = neighbor_texel(face, cx, cy, dx, dy);
            let ni = texel_index(nb.face, nb.x, nb.y);
            if crust_mask[ni] != mine {
                is_edge = true;
                break;
            }
        }
        if is_edge && c_dist[idx] > 0.0 {
            c_dist[idx] = 0.0;
            heap.push(0.0, idx);
        }
    }

    const DX8: [i32; 8] = [1, -1,  0,  0,  1, -1,  1, -1];
    const DY8: [i32; 8] = [0,  0,  1, -1,  1,  1, -1, -1];

    while !heap.is_empty() {
        let (dist, idx) = heap.pop_min();
        if settled[idx] { continue; }
        settled[idx] = true;
        let face = idx / (RES * RES);
        let rem  = idx - face * RES * RES;
        let cy   = rem / RES;
        let cx   = rem % RES;
        for d in 0..8 {
            let nb = neighbor_texel(face, cx, cy, DX8[d], DY8[d]);
            let ni = texel_index(nb.face, nb.x, nb.y);
            if settled[ni] { continue; }
            let cost = if DX8[d] != 0 && DY8[d] != 0 { diag_cost } else { orth_cost };
            let nd = dist + cost;
            if nd < c_dist[ni] as f64 {
                c_dist[ni] = nd as f32;
                heap.push(nd, ni);
            }
        }
    }

    // Sign: positive inside crust, negative outside
    let mut result = vec![0.0f32; TOTAL_TEXELS];
    for t in 0..TOTAL_TEXELS {
        result[t] = if crust_mask[t] == 1 { c_dist[t] } else { -c_dist[t] };
    }
    result
}

// ---------------------------------------------------------------------------
// bake_hardness — substrate rock-hardness cubemap field.
// Verbatim port of `_bakeHardness` in tectonics.ts (lines ~2401-2542).
//
// crust_age [0,1]: 0 = fresh/divergent/volcanic crust, 1 = old stable craton.
// rock_hardness [0,1]: 1 = hard fresh igneous, 0 = soft sediment/basin.
// Reads conv_field, dist_field (boundary distance), crust_dist (signed crust SDF).
// Pure baked-grid output — main-thread & worker identical (no live noise post-bake).
// ---------------------------------------------------------------------------

fn bake_hardness(
    conv_field: &[f32],
    dist_field: &[f32],
    crust_dist: &[f32],
    hardness_noise: &Noise3D,
    composition: f64,
) -> Vec<f32> {
    // --- Constants (50 km reference scale; angular constants are scale-invariant) ---
    const DIV_THRESH: f64   = -0.05; // conv_field below this → divergent
    const DIV_DIST: f64     = 0.08;  // radians — must be near a plate boundary
    const AGE_SCALE: f64    = 0.40;  // radians — half-sphere worth of "age"
    const VOLC_THRESH: f64  = 0.15;  // conv_field above this → arc/volcanic zone
    const HARD_FBM_AMP: f64  = 0.12; // ±0.12 hardness noise amplitude
    const HARD_FBM_FREQ: f64 = 2.8;  // spatial frequency on unit sphere

    // ---- Step 1: Multi-source Dijkstra from divergent-boundary texels → divergent_dist ----
    let inf = 1.0e30f64;
    let mut divergent_dist = vec![inf as f32; TOTAL_TEXELS];
    let mut settled        = vec![false; TOTAL_TEXELS];
    let mut heap           = BinaryHeap::new(TOTAL_TEXELS * 2);

    let diag_cost = TEX_ANG * std::f64::consts::SQRT_2;
    let orth_cost = TEX_ANG;
    const DX8: [i32; 8] = [1, -1,  0,  0,  1, -1,  1, -1];
    const DY8: [i32; 8] = [0,  0,  1, -1,  1,  1, -1, -1];

    // Seed: texels near divergent boundaries (spreading ridges).
    for idx in 0..TOTAL_TEXELS {
        let conv = conv_field[idx] as f64;
        let dist = dist_field[idx] as f64;
        if conv < DIV_THRESH && dist < DIV_DIST {
            divergent_dist[idx] = 0.0;
            heap.push(0.0, idx);
        }
    }

    // If no divergent texels (non-tectonic case), leave divergent_dist at INF → crust_age = 1.
    while !heap.is_empty() {
        let (dist, idx) = heap.pop_min();
        if settled[idx] { continue; }
        settled[idx] = true;
        let face = idx / (RES * RES);
        let rem  = idx - face * RES * RES;
        let cy   = rem / RES;
        let cx   = rem % RES;
        for d in 0..8 {
            let nb = neighbor_texel(face, cx, cy, DX8[d], DY8[d]);
            let ni = texel_index(nb.face, nb.x, nb.y);
            if settled[ni] { continue; }
            let cost = if DX8[d] != 0 && DY8[d] != 0 { diag_cost } else { orth_cost };
            let nd = dist + cost;
            if nd < divergent_dist[ni] as f64 {
                divergent_dist[ni] = nd as f32;
                heap.push(nd, ni);
            }
        }
    }

    // ---- Step 2: Build crust_age field (0 = freshest divergent, 1 = oldest craton) ----
    let mut crust_age = vec![0.0f32; TOTAL_TEXELS];
    for i in 0..TOTAL_TEXELS {
        let dv = divergent_dist[i] as f64;
        crust_age[i] = if dv >= inf * 0.5 { 1.0 } else { (dv / AGE_SCALE).min(1.0) as f32 };
    }
    // Light blur for C1-friendliness.
    let crust_age = blur_field_n(crust_age, 2);

    // ---- Step 3: Build rock_hardness (4-term blend) ----
    let mut hardness_raw = vec![0.0f32; TOTAL_TEXELS];
    for i in 0..TOTAL_TEXELS {
        let face = i / (RES * RES);
        let rem  = i - face * RES * RES;
        let ty   = rem / RES;
        let tx   = rem % RES;
        let dir  = texel_to_dir(face, tx, ty);

        // (a) Age term.
        let age = crust_age[i] as f64;
        let hard_age = if age < 0.5 {
            0.55 + (0.85 - 0.55) * (1.0 - age * 2.0)
        } else {
            0.55 - (0.55 - 0.35) * ((age - 0.5) * 2.0)
        };

        // (b) Volcanic-arc proximity (conv_field proxy).
        let conv = conv_field[i] as f64;
        let volc_frac = ((conv - VOLC_THRESH) / (1.0 - VOLC_THRESH)).clamp(0.0, 1.0);
        let hard_volc = 0.90;
        let hard_b = hard_age + volc_frac * (hard_volc - hard_age);

        // (c) Crust-type offset: oceanic basalt harder than continental sediment cover.
        let cd = crust_dist[i] as f64;
        let crust_offset = if cd < 0.0 { 0.08 } else if cd > 0.05 { -0.08 } else { 0.0 };
        let hard_c = hard_b + crust_offset;

        // (d) FBM break-up (stream 19; freq folded into coords, octaves 4).
        let fbm_val = fbm(
            hardness_noise,
            dir.x * HARD_FBM_FREQ,
            dir.y * HARD_FBM_FREQ,
            dir.z * HARD_FBM_FREQ,
            4, 1.0, 2.0, 0.5,
        );
        let hard_d = hard_c + HARD_FBM_AMP * fbm_val;

        hardness_raw[i] = hard_d.clamp(0.0, 1.0) as f32;
    }

    // ---- Step 4: Box blur for C1-friendliness (3 passes) ----
    let hardness_blurred = blur_field_n(hardness_raw, 3);

    // ---- Step 5: Scale contrast by composition ----
    let contrast_scale = 0.25 + 0.75 * composition;
    const HARD_MEAN: f64 = 0.5;
    let mut rock_hardness = vec![0.0f32; TOTAL_TEXELS];
    for i in 0..TOTAL_TEXELS {
        let h = HARD_MEAN + (hardness_blurred[i] as f64 - HARD_MEAN) * contrast_scale;
        rock_hardness[i] = h.clamp(0.0, 1.0) as f32;
    }
    rock_hardness
}

// ---------------------------------------------------------------------------
// Paleo-orogen Dijkstra — multi-source SDF from all dilated paleo crack texels.
// Mirrors Step 6c paleo-dist block in tectonics.ts constructor.
// Returns distance field; INF replaced with PI (finite cap for consumers).
// ---------------------------------------------------------------------------

fn build_paleo_dist(paleo_dilated: &[u8]) -> Vec<f32> {
    let inf = 1.0e30f64;
    let mut p_dist   = vec![inf as f32; TOTAL_TEXELS];
    let mut settled  = vec![false; TOTAL_TEXELS];
    let mut heap     = BinaryHeap::new(TOTAL_TEXELS * 2);

    let diag_cost = TEX_ANG * std::f64::consts::SQRT_2;
    let orth_cost = TEX_ANG;

    // Seed: all dilated paleo crack texels
    for idx in 0..TOTAL_TEXELS {
        if paleo_dilated[idx] == 1 {
            p_dist[idx] = 0.0;
            heap.push(0.0, idx);
        }
    }

    const DX8: [i32; 8] = [1, -1,  0,  0,  1, -1,  1, -1];
    const DY8: [i32; 8] = [0,  0,  1, -1,  1,  1, -1, -1];

    while !heap.is_empty() {
        let (dist, idx) = heap.pop_min();
        if settled[idx] { continue; }
        settled[idx] = true;
        let face = idx / (RES * RES);
        let rem  = idx - face * RES * RES;
        let py   = rem / RES;
        let px   = rem % RES;
        for d in 0..8 {
            let nb = neighbor_texel(face, px, py, DX8[d], DY8[d]);
            let ni = texel_index(nb.face, nb.x, nb.y);
            if settled[ni] { continue; }
            let cost = if DX8[d] != 0 && DY8[d] != 0 { diag_cost } else { orth_cost };
            let nd = dist + cost;
            if nd < p_dist[ni] as f64 {
                p_dist[ni] = nd as f32;
                heap.push(nd, ni);
            }
        }
    }

    // Replace INF with PI cap
    let pi32 = std::f32::consts::PI;
    for v in p_dist.iter_mut() {
        if *v >= 1.0e29 { *v = pi32; }
    }
    p_dist
}

// ---------------------------------------------------------------------------
// Conv/shear bake
// Mirrors `bakeConvShear` in tectonics.ts.
// ---------------------------------------------------------------------------

fn bake_conv_shear(
    dist_field: &[f32],
    comp_id: &[u16],
    neighbor_id: &[u16],
    omega: &[(f64, f64, f64)],
    plate_count: usize,
) -> (Vec<f32>, Vec<f32>) {
    let mut conv_field  = vec![0.0f32; TOTAL_TEXELS];
    let mut shear_field = vec![0.0f32; TOTAL_TEXELS];

    for idx in 0..TOTAL_TEXELS {
        let face = idx / (RES * RES);
        let rem  = idx - face * RES * RES;
        let ty   = rem / RES;
        let tx   = rem % RES;

        let c_dir = texel_to_dir(face, tx, ty);

        let nt0 = neighbor_texel(face, tx, ty,  1,  0);
        let nt1 = neighbor_texel(face, tx, ty, -1,  0);
        let nt2 = neighbor_texel(face, tx, ty,  0,  1);
        let nt3 = neighbor_texel(face, tx, ty,  0, -1);

        let d0 = dist_field[texel_index(nt0.face, nt0.x, nt0.y)] as f64;
        let d1 = dist_field[texel_index(nt1.face, nt1.x, nt1.y)] as f64;
        let d2 = dist_field[texel_index(nt2.face, nt2.x, nt2.y)] as f64;
        let d3 = dist_field[texel_index(nt3.face, nt3.x, nt3.y)] as f64;

        let dir0 = texel_to_dir(nt0.face, nt0.x, nt0.y);
        let dir1 = texel_to_dir(nt1.face, nt1.x, nt1.y);
        let dir2 = texel_to_dir(nt2.face, nt2.x, nt2.y);
        let dir3 = texel_to_dir(nt3.face, nt3.x, nt3.y);

        let grad_u = (d0 - d1) * 0.5;
        let grad_v = (d2 - d3) * 0.5;
        let ua = dir0 - dir1;
        let va = dir2 - dir3;

        let gw = ua * grad_u + va * grad_v;
        let gdot = gw.dot(c_dir);
        let pt = gw - gdot * c_dir;
        let g_len = pt.length();

        if g_len < 1.0e-7 { continue; }

        let t_hat = -pt / g_len;

        let pid = comp_id[idx] as usize;
        let nid = (neighbor_id[idx] as usize).min(plate_count.saturating_sub(1));

        let (oax, oay, oaz) = omega[pid];
        let (obx, oby, obz) = omega[nid];
        let cx = c_dir.x; let cy = c_dir.y; let cz = c_dir.z;

        let va_x = oay * cz - oaz * cy;
        let va_y = oaz * cx - oax * cz;
        let va_z = oax * cy - oay * cx;
        let vb_x = oby * cz - obz * cy;
        let vb_y = obz * cx - obx * cz;
        let vb_z = obx * cy - oby * cx;

        let diff = DVec3::new(va_x - vb_x, va_y - vb_y, va_z - vb_z);

        let raw_conv = diff.dot(t_hat) / 2.0;
        conv_field[idx] = raw_conv.clamp(-1.0, 1.0) as f32;

        let b = t_hat.cross(c_dir);
        let b_len = b.length();
        if b_len > 1.0e-7 {
            // Match ki's exact float order (tectonics.ts:1368): (diff·b)/bLen computed
            // as Σ(diffᵢ·bᵢ/bLen), multiply-then-divide per term — not diff·(b/bLen).
            let raw_shear = (diff.x * b.x / b_len + diff.y * b.y / b_len + diff.z * b.z / b_len) / 2.0;
            shear_field[idx] = raw_shear.clamp(-1.0, 1.0) as f32;
        }
    }

    // 12 blur passes (ki tectonics.ts:1378-1379). Blur radius ∝ √passes·cellSize, so
    // at RES=512 the angular σ needs 12 passes (was 3 at RES=256) to preserve ki's
    // smoothing; eliminates gradient sign-flip artefacts at saddle points.
    let conv_blurred  = blur_field_n(conv_field,  12);
    let shear_blurred = blur_field_n(shear_field, 12);
    (conv_blurred, shear_blurred)
}

// ---------------------------------------------------------------------------
// Phase 2 — non-tectonic continental mask constants
// Mirrors CONT_FREQ / CONT_THRESH in tectonics.ts.
// ---------------------------------------------------------------------------

const CONT_FREQ:   f64 = 1.1;
const CONT_THRESH: f64 = 0.10;

// ---------------------------------------------------------------------------
// Phase 2 — crust mask bake constants (Step 6b in tectonics.ts)
// ---------------------------------------------------------------------------

const CRUST_A:      f64 = 0.60;
const CRUST_B:      f64 = 0.20;
const CRUST_C:      f64 = 0.45;
const CRUST_T:      f64 = 0.44;
const CRUST_LEAN:   f64 = 0.25;
const CRUST_RP_LO:  f64 = 0.55;
const CRUST_RP_HI:  f64 = 0.95;

// Arc-crust bake constants (mirrors tectonics.ts)
const ARC_CONV_THRESH:  f64 = 0.14;
const ARC_OFFSET:       f64 = 0.020;
const ARC_CRUST_WIDTH:  f64 = 0.090;

// ---------------------------------------------------------------------------
// Tectonics — public struct
// ---------------------------------------------------------------------------

/// Fully-baked tectonic simulation.
///
/// After `new()` all plate-level query fields are populated. Crust/paleo/volcano
/// fields are Phase-2 work and are zeroed (see `// PHASE 2` markers).
#[allow(dead_code)] // some baked fields are retained for parity but not yet queried
pub struct Tectonics {
    pub plates: Vec<Plate>,

    // Construction parameters retained for reference / debug.
    seed: u32,
    arc_density: f64,
    hotspot_count: u32,
    hotspot_intensity: f64,
    /// Crust composition [0,1]: 0 = icy/sediment (soft), 1 = rocky/basaltic (hard).
    /// Feeds Phase-1 substrate-hardness contrast.
    composition: f64,

    // Cube-map tables (read-only after construction).
    comp_id:    Vec<u16>,   // per-texel plate id (0-based)
    dist_field: Vec<f32>,   // per-texel distance to boundary (radians)
    neighbor_id: Vec<u16>,  // per-texel neighbour plate id
    conv_field:  Vec<f32>,  // per-texel baked convergence (blurred)
    shear_field: Vec<f32>,  // per-texel baked shear (blurred)
    /// Per-texel blurred plate base elevation (eliminates the Voronoi base-elev cliff).
    base_elev_field: Vec<f32>,
    /// Wide-smoothed uplift field = blur_field_n(conv_field, 3). Consumed in Phase 4
    /// broad-deform; currently only exposed via `uplift_at`.
    uplift_field: Vec<f32>,
    /// Per-texel substrate rock hardness [0,1]. Consumed in Phase-3 erosion + Phase-4 sampler.
    rock_hardness: Vec<f32>,

    // PHASE 2 fields (zeroed in Phase 1).
    crust_dist: Vec<f32>,   // per-texel signed crust SDF
    paleo_dist: Vec<f32>,   // per-texel distance to paleo-orogen network

    // Domain-warp noise (streams 5 and 13).
    warp_noise:      Noise3D,
    fine_warp_noise: Noise3D,
    // Volcano roughening noise (stream 18).
    vol_noise:       Noise3D,
    // Substrate-hardness break-up noise (stream 19). Only read during bake_hardness,
    // kept for parity with the constructor's stream set.
    #[allow(dead_code)]
    hardness_noise:  Noise3D,

    // Volcano flat arrays (populated by bake_volcanoes in Phase 2).
    vol_pos:         Vec<f32>,  // 3*nVol xyz interleaved
    vol_base_radius: Vec<f32>,
    vol_height:      Vec<f32>,
    vol_crater_frac: Vec<f32>,
    vol_kind:        Vec<u8>,   // 0=arc, 1=hotspot
    vol_intensity:   Vec<f32>,
    // Spatial bucket index (CSR).
    vol_bucket_res:   u32,
    vol_bucket_start: Vec<i32>,
    vol_bucket_idx:   Vec<i32>,
    // Hotspot flat arrays.
    hotspot_pos:           Vec<f32>,
    hotspot_intensity_arr: Vec<f32>,
    hotspot_is_chain:      Vec<u8>,
}

/// Result of `bake_volcanoes` — all flat volcano arrays.
struct VolcanoBaked {
    vol_pos:         Vec<f32>,
    vol_base_radius: Vec<f32>,
    vol_height:      Vec<f32>,
    vol_crater_frac: Vec<f32>,
    vol_kind:        Vec<u8>,
    vol_intensity:   Vec<f32>,
    vol_bucket_res:  u32,
    vol_bucket_start: Vec<i32>,
    vol_bucket_idx:   Vec<i32>,
    hotspot_pos:           Vec<f32>,
    hotspot_intensity_arr: Vec<f32>,
    hotspot_is_chain:      Vec<u8>,
}

impl VolcanoBaked {
    fn empty() -> Self {
        VolcanoBaked {
            vol_pos:         Vec::new(),
            vol_base_radius: Vec::new(),
            vol_height:      Vec::new(),
            vol_crater_frac: Vec::new(),
            vol_kind:        Vec::new(),
            vol_intensity:   Vec::new(),
            vol_bucket_res:  0,
            vol_bucket_start: Vec::new(),
            vol_bucket_idx:   Vec::new(),
            hotspot_pos:           Vec::new(),
            hotspot_intensity_arr: Vec::new(),
            hotspot_is_chain:      Vec::new(),
        }
    }
}

impl Tectonics {
    /// Construct a new deterministic tectonic simulation.
    ///
    /// Parameters mirror the TS constructor:
    /// - `seed`              — master seed (u32)
    /// - `plate_count`       — target number of plates (default 16, clamped 0–48)
    /// - `arc_density`       — arc volcano density multiplier (default 1.0, clamped 0.2–3.0)
    /// - `hotspot_count`     — number of mantle hotspots (default 6, clamped 0–20)
    /// - `hotspot_intensity` — hotspot height multiplier (default 1.0, clamped 0–3)
    pub fn new(
        seed: u32,
        plate_count: usize,
        arc_density_raw: f64,
        hotspot_count_raw: u32,
        hotspot_intensity_raw: f64,
        composition_raw: f64,
    ) -> Self {
        let arc_density       = arc_density_raw.clamp(0.2, 3.0);
        let hotspot_count     = hotspot_count_raw.min(20);
        let hotspot_intensity = hotspot_intensity_raw.clamp(0.0, 3.0);
        // composition: 0 = icy/sediment (soft, low contrast), 1 = rocky/basaltic (hard).
        let composition       = composition_raw.clamp(0.0, 1.0);
        let plate_count_clamped = plate_count.min(48);
        let target = plate_count_clamped;

        // Noise streams (mirrors constructor)
        let warp_noise      = Noise3D::new(derive_seed(seed, 5));
        let fine_warp_noise = Noise3D::new(derive_seed(seed, 13));
        let vol_noise       = Noise3D::new(derive_seed(seed, 18));
        let hardness_noise  = Noise3D::new(derive_seed(seed, 19));

        // Non-tectonic branch: plateCount <= 1 → single plate, no boundaries.
        if plate_count_clamped <= 1 {
            return Self::compute_non_tectonic(
                seed, arc_density, hotspot_count, hotspot_intensity, composition,
                warp_noise, fine_warp_noise, vol_noise, hardness_noise,
            );
        }

        // RNG streams
        let mut elev_rng  = make_rng(derive_seed(seed, 2));
        let mut axis_rng  = make_rng(derive_seed(seed, 3));
        let mut speed_rng = make_rng(derive_seed(seed, 4));

        // -------------------------------------------------------------------------
        // Step 1 — Trace cracks
        // -------------------------------------------------------------------------
        let mut crack_mask = vec![0u8; TOTAL_TEXELS];

        let k     = 8.max((target as f64 * 2.0).round() as usize);
        let max_w = 8 * k;

        let mut walker_rng = make_rng(derive_seed(seed, 7));
        trace_cracks(&mut crack_mask, &mut walker_rng, k, max_w, None);

        // Circumpolar ring cracks at ±74°–76° latitude.
        let ring_lo = ((74.0f64).to_radians()).sin();
        let ring_hi = ((76.0f64).to_radians()).sin();
        for f in 0..6 {
            for ty in 0..RES {
                for tx in 0..RES {
                    let d = texel_to_dir(f, tx, ty);
                    let ay = d.y.abs();
                    if ay >= ring_lo && ay <= ring_hi {
                        crack_mask[texel_index(f, tx, ty)] = 1;
                    }
                }
            }
        }

        // -------------------------------------------------------------------------
        // Step 2 — Fragmentation loop
        // -------------------------------------------------------------------------
        let max_share_base = 0.22f64;
        let max_share = max_share_base.max(1.6 / target as f64);
        let sliver_thresh_count = (TOTAL_TEXELS as f64 * 0.001) as usize;

        #[allow(clippy::needless_range_loop)]
        let compute_stats = |cid: &[u16], cc: usize| -> (usize, f64, u16) {
            let mut areas = vec![0usize; cc + 2];
            for &v in cid.iter() { if (v as usize) <= cc { areas[v as usize] += 1; } }
            let mut sig = 0usize;
            let mut largest_area = 0usize;
            let mut largest_comp = 1u16;
            for c in 1..=cc {
                if areas[c] >= sliver_thresh_count {
                    sig += 1;
                    if areas[c] > largest_area { largest_area = areas[c]; largest_comp = c as u16; }
                }
            }
            (sig, largest_area as f64 / TOTAL_TEXELS as f64, largest_comp)
        };

        let fr = flood_fill(&dilate4(&crack_mask));
        let mut comp_count = fr.comp_count;
        let mut comp_id    = fr.comp_id;

        let mut extra_rng = make_rng(derive_seed(seed, 8));

        for _ in 0..24 {
            let (sig_count, largest_share, largest_comp) = compute_stats(&comp_id, comp_count);
            if sig_count >= target && largest_share <= max_share { break; }

            if largest_share > max_share {
                let num_cuts = ((target as f64 * 0.5).ceil() as usize).max(1);
                for _ in 0..num_cuts {
                    let mut gx = 0.0f64; let mut gy = 0.0f64; let mut gz = 0.0f64;
                    for _ in 0..20 {
                        let rx = extra_rng() * 2.0 - 1.0;
                        let ry = extra_rng() * 2.0 - 1.0;
                        let rz = extra_rng() * 2.0 - 1.0;
                        let rlen = (rx * rx + ry * ry + rz * rz).sqrt();
                        if rlen > 0.01 && rlen < 1.0 { gx = rx / rlen; gy = ry / rlen; gz = rz / rlen; break; }
                    }
                    if gx != 0.0 || gy != 0.0 || gz != 0.0 {
                        crack_great_circle(&mut crack_mask, DVec3::new(gx, gy, gz), Some(&comp_id), largest_comp);
                    }
                }
                let lc = largest_comp;
                let snapshot = comp_id.clone();
                let extra_k = ((target as f64 * 0.5).ceil() as usize).max(1);
                trace_cracks(&mut crack_mask, &mut extra_rng, extra_k, max_w,
                    Some(&|idx| snapshot[idx] == lc));
            } else {
                let extra_k = ((target as f64 * 0.6).ceil() as usize).max(1);
                trace_cracks(&mut crack_mask, &mut extra_rng, extra_k, max_w, None);
            }

            let refill = flood_fill(&dilate4(&crack_mask));
            comp_count = refill.comp_count;
            comp_id    = refill.comp_id;
        }

        // -------------------------------------------------------------------------
        // Step 3 — Identify pole components
        // -------------------------------------------------------------------------
        let north_t   = dir_to_texel(DVec3::new(0.0, 1.0, 0.0));
        let north_idx = texel_index(north_t.face, north_t.x, north_t.y);
        let pole_comp_n = comp_id[north_idx];

        let south_t   = dir_to_texel(DVec3::new(0.0, -1.0, 0.0));
        let south_idx = texel_index(south_t.face, south_t.x, south_t.y);
        let pole_comp_s = comp_id[south_idx];

        // -------------------------------------------------------------------------
        // Step 4 — Merge to target
        // -------------------------------------------------------------------------
        let final_count = merge_to_target(
            &mut comp_id, comp_count, target,
            pole_comp_n, pole_comp_s, max_share,
        );

        // -------------------------------------------------------------------------
        // Step 6 — Distance + neighbour fields
        // -------------------------------------------------------------------------
        let df_result = build_distance_field(&comp_id, final_count);
        let dist_field_raw = df_result.dist_field;
        let neighbor_id    = df_result.neighbor_id;

        let dist_field = blur_field_n(dist_field_raw.clone(), 2);

        // -------------------------------------------------------------------------
        // Step 6a — Omega pre-derivation
        // -------------------------------------------------------------------------
        let mut omega: Vec<(f64, f64, f64)> = Vec::with_capacity(final_count);
        for _ in 0..final_count {
            let ax = axis_rng() * 2.0 - 1.0;
            let ay = axis_rng() * 2.0 - 1.0;
            let az = axis_rng() * 2.0 - 1.0;
            // Mirror JS: `const axisLen = Math.sqrt(...) || 1`
            // JS `|| 1` returns 1 only when axisLen === 0 (exactly), otherwise axisLen.
            // We use f64::max(1.0) only on the zero branch; typical axisLen is in (0.1, 1.73).
            let axis_len_raw = (ax * ax + ay * ay + az * az).sqrt();
            let axis_len = if axis_len_raw == 0.0 { 1.0 } else { axis_len_raw };
            let speed = 0.4 + speed_rng() * 0.6;
            omega.push((ax / axis_len * speed, ay / axis_len * speed, az / axis_len * speed));
        }

        let (conv_field, shear_field) = bake_conv_shear(
            &dist_field, &comp_id, &neighbor_id, &omega, final_count,
        );

        // -------------------------------------------------------------------------
        // Step 5 — Plate properties
        // -------------------------------------------------------------------------
        // Types: id 0 = oceanic (north pole), id 1 = continental (south pole),
        // rest from typeRng (stream 1).
        let mut type_rng = make_rng(derive_seed(seed, 1));
        let phi_gold = 0.618_033_988_75f64;

        // Pre-compute seed dirs (pole of inaccessibility)
        let plates: Vec<Plate> = (0..final_count).map(|i| {
            let plate_type = if i == 0 {
                PlateType::Oceanic
            } else if i == 1 {
                PlateType::Continental
            } else {
                if type_rng() < 0.60 { PlateType::Oceanic } else { PlateType::Continental }
            };

            let base_elevation = -0.12 + elev_rng() * 0.24;

            let (ox, oy, oz) = omega[i];

            let hue = ((i as f64 * phi_gold) % 1.0 + 1.0) % 1.0;
            let color = match plate_type {
                PlateType::Oceanic      => hsl_to_rgb(hue, 0.55, 0.34),
                PlateType::Continental  => hsl_to_rgb(hue, 0.60, 0.55),
            };

            let seed_dir = pole_of_inaccessibility(i as u16, &comp_id, &dist_field_raw);

            Plate {
                id: i,
                seed_dir,
                plate_type,
                base_elevation,
                omega: DVec3::new(ox, oy, oz),
                color,
            }
        }).collect();

        // -------------------------------------------------------------------------
        // Step 6b — Crust mask bake + crust SDF
        // Mirrors Step 6b in tectonics.ts constructor.
        // -------------------------------------------------------------------------
        let crust_dist = Self::bake_crust_sdf(
            seed, &comp_id, &dist_field_raw, &dist_field, &neighbor_id, &conv_field,
            &plates, final_count, arc_density,
        );

        // -------------------------------------------------------------------------
        // Step 6c — Paleo-orogen network
        // Mirrors Step 6c in tectonics.ts constructor.
        // -------------------------------------------------------------------------
        let paleo_dist = Self::bake_paleo_dist(seed);

        // -------------------------------------------------------------------------
        // Step 8 — Volcano bake
        // -------------------------------------------------------------------------
        let vol_baked = Self::bake_volcanoes(
            seed, arc_density, hotspot_count, hotspot_intensity,
            &comp_id, &neighbor_id, &dist_field, &crust_dist,
            &conv_field, &shear_field, &plates, &warp_noise, &fine_warp_noise,
        );

        // Smooth base-elevation field: nearest-neighbour reads of
        // plates[comp_id[i]].base_elevation produce a hard cliff at Voronoi plate
        // boundaries; blurring 6× converts it to a ~1 km gradient (tectonics.ts:2154-2160).
        let base_elev_field = {
            let mut raw = vec![0.0f32; TOTAL_TEXELS];
            for (i, slot) in raw.iter_mut().enumerate() {
                *slot = plates[comp_id[i] as usize].base_elevation as f32;
            }
            blur_field_n(raw, 6)
        };
        // Wide-smoothed uplift field for broad-belt deformation (bakeUpliftField: tectonics.ts:1255-1260).
        let uplift_field = blur_field_n(conv_field.clone(), 3);

        // Substrate rock-hardness field (_bakeHardness): reads conv/dist/crust_dist + stream-19 FBM.
        let rock_hardness = bake_hardness(&conv_field, &dist_field, &crust_dist, &hardness_noise, composition);

        Tectonics {
            plates,
            seed,
            arc_density,
            hotspot_count,
            hotspot_intensity,
            composition,
            comp_id,
            dist_field,
            neighbor_id,
            conv_field,
            shear_field,
            base_elev_field,
            uplift_field,
            rock_hardness,
            crust_dist,
            paleo_dist,
            warp_noise,
            fine_warp_noise,
            vol_noise,
            hardness_noise,
            vol_pos:         vol_baked.vol_pos,
            vol_base_radius: vol_baked.vol_base_radius,
            vol_height:      vol_baked.vol_height,
            vol_crater_frac: vol_baked.vol_crater_frac,
            vol_kind:        vol_baked.vol_kind,
            vol_intensity:   vol_baked.vol_intensity,
            vol_bucket_res:  vol_baked.vol_bucket_res,
            vol_bucket_start: vol_baked.vol_bucket_start,
            vol_bucket_idx:   vol_baked.vol_bucket_idx,
            hotspot_pos:           vol_baked.hotspot_pos,
            hotspot_intensity_arr: vol_baked.hotspot_intensity_arr,
            hotspot_is_chain:      vol_baked.hotspot_is_chain,
        }
    }

    /// Non-tectonic (stagnant-lid, plateCount ≤ 1) branch.
    /// Mirrors `_computeNonTectonic` in tectonics.ts.
    fn compute_non_tectonic(
        seed: u32,
        arc_density: f64,
        hotspot_count: u32,
        hotspot_intensity: f64,
        composition: f64,
        warp_noise: Noise3D,
        fine_warp_noise: Noise3D,
        vol_noise: Noise3D,
        hardness_noise: Noise3D,
    ) -> Self {
        // Single plate covers the whole sphere.
        let plates = vec![Plate {
            id: 0,
            seed_dir: DVec3::new(0.0, 1.0, 0.0),
            plate_type: PlateType::Continental,
            base_elevation: 0.0,
            omega: DVec3::ZERO,
            color: hsl_to_rgb(0.0, 0.0, 0.55),
        }];

        let comp_id     = vec![0u16; TOTAL_TEXELS];
        let neighbor_id = vec![0u16; TOTAL_TEXELS];
        let dist_field  = vec![std::f32::consts::PI; TOTAL_TEXELS];
        let conv_field  = vec![0.0f32; TOTAL_TEXELS];
        let shear_field = vec![0.0f32; TOTAL_TEXELS];

        // ---- 1. Continental mask from noise (stream 15) ----
        let cont_noise = Noise3D::new(derive_seed(seed, 15));
        let mut crust_mask = vec![0u8; TOTAL_TEXELS];
        for face in 0..6 {
            for ty in 0..RES {
                for tx in 0..RES {
                    let d = texel_to_dir(face, tx, ty);
                    let c = fbm(&cont_noise, d.x * CONT_FREQ, d.y * CONT_FREQ, d.z * CONT_FREQ,
                                5, 1.0, 2.0, 0.5);
                    crust_mask[texel_index(face, tx, ty)] = if c > CONT_THRESH { 1 } else { 0 };
                }
            }
        }

        // ---- 2. Crust SDF via Dijkstra ----
        let crust_dist_raw = build_crust_sdf(&crust_mask);
        let crust_dist = blur_field_n(crust_dist_raw, 3);

        // ---- 3. Paleo orogens (identical to Step 6c in tectonic path) ----
        let paleo_dist = Self::bake_paleo_dist(seed);

        // ---- 4. Volcano bake (non-tectonic path — no arc candidates) ----
        let vol_baked = Self::bake_volcanoes(
            seed, arc_density, hotspot_count, hotspot_intensity,
            &comp_id, &neighbor_id, &dist_field, &crust_dist,
            &conv_field, &shear_field, &plates, &warp_noise, &fine_warp_noise,
        );

        // Base-elev field: single plate (base_elevation 0.0) → uniform; blur is a no-op.
        let base_elev_field = {
            let mut raw = vec![0.0f32; TOTAL_TEXELS];
            for (i, slot) in raw.iter_mut().enumerate() {
                *slot = plates[comp_id[i] as usize].base_elevation as f32;
            }
            blur_field_n(raw, 6)
        };
        // Uplift field: conv_field is all-zero in the stagnant-lid branch → stays zero.
        let uplift_field = blur_field_n(conv_field.clone(), 3);

        // Substrate rock-hardness: no divergent texels (conv all-zero) → crust_age = 1 everywhere.
        let rock_hardness = bake_hardness(&conv_field, &dist_field, &crust_dist, &hardness_noise, composition);

        Tectonics {
            plates,
            seed,
            arc_density,
            hotspot_count,
            hotspot_intensity,
            composition,
            comp_id,
            dist_field,
            neighbor_id,
            conv_field,
            shear_field,
            base_elev_field,
            uplift_field,
            rock_hardness,
            crust_dist,
            paleo_dist,
            warp_noise,
            fine_warp_noise,
            vol_noise,
            hardness_noise,
            vol_pos:         vol_baked.vol_pos,
            vol_base_radius: vol_baked.vol_base_radius,
            vol_height:      vol_baked.vol_height,
            vol_crater_frac: vol_baked.vol_crater_frac,
            vol_kind:        vol_baked.vol_kind,
            vol_intensity:   vol_baked.vol_intensity,
            vol_bucket_res:  vol_baked.vol_bucket_res,
            vol_bucket_start: vol_baked.vol_bucket_start,
            vol_bucket_idx:   vol_baked.vol_bucket_idx,
            hotspot_pos:           vol_baked.hotspot_pos,
            hotspot_intensity_arr: vol_baked.hotspot_intensity_arr,
            hotspot_is_chain:      vol_baked.hotspot_is_chain,
        }
    }

    // ---------------------------------------------------------------------------
    // bake_paleo_dist — Step 6c
    // Mirrors the paleo-orogen section of the tectonics.ts constructor.
    // ---------------------------------------------------------------------------

    fn bake_paleo_dist(seed: u32) -> Vec<f32> {
        let mut paleo_mask = vec![0u8; TOTAL_TEXELS];
        let mut paleo_rng = make_rng(derive_seed(seed, 12));
        const PALEO_SEED_COUNT:   usize = 14;
        const PALEO_MAX_WALKERS:  usize = 8 * PALEO_SEED_COUNT;
        trace_cracks(&mut paleo_mask, &mut paleo_rng, PALEO_SEED_COUNT, PALEO_MAX_WALKERS, None);

        let paleo_dilated = dilate4(&paleo_mask);
        let p_dist_raw = build_paleo_dist(&paleo_dilated);
        // 1 blur pass (mirrors `blurField(paleoDist)` in tectonics.ts)
        blur_field_n(p_dist_raw, 1)
    }

    // ---------------------------------------------------------------------------
    // bake_crust_sdf — Step 6b
    // Mirrors the crust-mask + crust-SDF block in the tectonics.ts constructor.
    // Only called on the tectonic path (plate_count > 1).
    // ---------------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn bake_crust_sdf(
        seed: u32,
        comp_id: &[u16],
        dist_field_raw: &[f32],       // unblurred — JS local `distField` variable
        dist_field_blurred: &[f32],   // blurred 2× — JS `this._distField` (used for arc-crust bd)
        neighbor_id: &[u16],
        conv_field: &[f32],           // already blurred
        plates: &[Plate],
        final_count: usize,
        arc_density: f64,
    ) -> Vec<f32> {
        // --- Per-plate area share ---
        let mut plate_area_share = vec![0.0f64; final_count];
        for &pid in comp_id.iter() { plate_area_share[pid as usize] += 1.0; }
        for a in plate_area_share.iter_mut() { *a /= TOTAL_TEXELS as f64; }

        // --- Pre-derive type array (same stream 1 as Step 5) ---
        let mut type_rng2 = make_rng(derive_seed(seed, 1));
        let mut types_pre = Vec::with_capacity(final_count);
        for i in 0..final_count {
            let t = if i == 0 { PlateType::Oceanic }
                    else if i == 1 { PlateType::Continental }
                    else { if type_rng2() < 0.60 { PlateType::Oceanic } else { PlateType::Continental } };
            types_pre.push(t);
        }

        // --- Per-plate R_P and anchorDir ---
        let mut rp_rng    = make_rng(derive_seed(seed, 10));
        let mut micro_rng = make_rng(derive_seed(seed, 11));
        let crust_noise   = Noise3D::new(derive_seed(seed, 9));

        let mut plate_rp       = vec![0.0f64; final_count];
        let mut plate_anchor   = vec![[0.0f64; 3]; final_count];
        let mut plate_micro_r  = vec![0.0f64; final_count];
        let mut plate_micro_c  = vec![[0.0f64; 3]; final_count];

        // Pre-compute seedDirs from unblurred distField
        let mut plate_seed_dir = vec![[0.0f64; 3]; final_count];
        #[allow(clippy::needless_range_loop)]
        for i in 0..final_count {
            let sd = pole_of_inaccessibility(i as u16, comp_id, dist_field_raw);
            plate_seed_dir[i] = [sd.x, sd.y, sd.z];
        }

        for i in 0..final_count {
            let omega = plates[i].omega;
            let [sx, sy, sz] = plate_seed_dir[i];
            if types_pre[i] == PlateType::Continental {
                let area = plate_area_share[i];
                let rp_base = CRUST_RP_LO * (area / 0.0625).sqrt();
                let rp = (CRUST_RP_LO + (CRUST_RP_HI - CRUST_RP_LO) * rp_rng())
                       * rp_base.clamp(CRUST_RP_LO, CRUST_RP_HI);
                plate_rp[i] = rp;
                // anchorDir = normalize(seedDir + LEAN * velDir), velDir = omega × seedDir
                let vx = omega.y * sz - omega.z * sy;
                let vy = omega.z * sx - omega.x * sz;
                let vz = omega.x * sy - omega.y * sx;
                let vlen = (vx * vx + vy * vy + vz * vz).sqrt();
                if vlen < 1.0e-6 {
                    plate_anchor[i] = [sx, sy, sz];
                } else {
                    let vhx = vx / vlen; let vhy = vy / vlen; let vhz = vz / vlen;
                    let ax = sx + CRUST_LEAN * vhx;
                    let ay = sy + CRUST_LEAN * vhy;
                    let az = sz + CRUST_LEAN * vhz;
                    let alen = (ax * ax + ay * ay + az * az).sqrt().max(1.0e-9);
                    plate_anchor[i] = [ax / alen, ay / alen, az / alen];
                }
            } else {
                rp_rng(); // consume rpRng for oceanic plates (same stream order)
                if micro_rng() < 0.22 {
                    // Find interior seed
                    let mut candidates: Vec<usize> = Vec::new();
                    for t in 0..TOTAL_TEXELS {
                        if comp_id[t] as usize == i && dist_field_raw[t] > 0.05 {
                            candidates.push(t);
                        }
                    }
                    if !candidates.is_empty() {
                        let pick_idx = (micro_rng() * candidates.len() as f64).floor() as usize;
                        let pick_idx = pick_idx.min(candidates.len() - 1);
                        let pick = candidates[pick_idx];
                        let mface = pick / (RES * RES);
                        let mrem  = pick - mface * RES * RES;
                        let my    = mrem / RES;
                        let mx    = mrem % RES;
                        let md = texel_to_dir(mface, mx, my);
                        plate_micro_c[i] = [md.x, md.y, md.z];
                        plate_micro_r[i] = 0.04 + micro_rng() * 0.06;
                    }
                }
            }
        }

        // --- Apply crust mask ---
        let mut crust_mask = vec![0u8; TOTAL_TEXELS];
        let arc_conv_thresh_eff = ARC_CONV_THRESH / arc_density;

        for t in 0..TOTAL_TEXELS {
            let face = t / (RES * RES);
            let rem  = t - face * RES * RES;
            let ty2  = rem / RES;
            let tx2  = rem % RES;
            let d = texel_to_dir(face, tx2, ty2);
            let pid = comp_id[t] as usize;

            if types_pre[pid] == PlateType::Continental {
                let rp = plate_rp[pid];
                let [ax2, ay2, az2] = plate_anchor[pid];
                let dot_a = (d.x * ax2 + d.y * ay2 + d.z * az2).clamp(-1.0, 1.0);
                let angle = dot_a.acos();
                let falloff = 1.0 - ss(0.0, rp, angle);
                let bdist = dist_field_raw[t] as f64;
                let bdist_ss = ss(0.0, 0.10, bdist);
                let fbm_val = fbm(&crust_noise, d.x * 1.3, d.y * 1.3, d.z * 1.3,
                                  5, 1.0, 2.0, 0.5) * 0.5 + 0.5;
                let score = CRUST_A * falloff + CRUST_B * bdist_ss + CRUST_C * fbm_val;
                if score > CRUST_T { crust_mask[t] = 1; }
            } else {
                // Oceanic: microcontinent only
                let mr = plate_micro_r[pid];
                if mr > 0.0 {
                    let [mx, my2, mz] = plate_micro_c[pid];
                    let dot_m = (d.x * mx + d.y * my2 + d.z * mz).clamp(-1.0, 1.0);
                    let m_angle = dot_m.acos();
                    let fbm_val3 = fbm(&crust_noise, d.x * 3.0, d.y * 3.0, d.z * 3.0,
                                       3, 1.0, 2.0, 0.5) * 0.5 + 0.5;
                    let threshold = mr * (0.7 + 0.5 * fbm_val3);
                    if m_angle < threshold { crust_mask[t] = 1; }
                }
            }

            // --- Arc-crust pass ---
            // Mirrors JS: `bd = this._distField[t]` — uses the BLURRED dist field,
            // not the raw one. JS stores the blurred dist as `this._distField`.
            if crust_mask[t] == 0 {
                let conv_arc = sample_c1(conv_field, d);
                if conv_arc > arc_conv_thresh_eff {
                    let bd = dist_field_blurred[t] as f64;
                    if (ARC_OFFSET - ARC_CRUST_WIDTH..=ARC_OFFSET + ARC_CRUST_WIDTH).contains(&bd) {
                        let nid = (neighbor_id[t] as usize).min(final_count.saturating_sub(1));
                        let is_oc = types_pre[pid] == PlateType::Continental
                                 && types_pre[nid] == PlateType::Oceanic;
                        let is_oo = types_pre[pid] == PlateType::Oceanic
                                 && types_pre[nid] == PlateType::Oceanic
                                 && pid > nid;
                        if is_oc || is_oo {
                            crust_mask[t] = 1;
                        }
                    }
                }
            }
        }

        // --- Crust SDF via Dijkstra ---
        let crust_dist_raw = build_crust_sdf(&crust_mask);
        // 3 blur passes (mirrors `blurFieldN(crustDist, 3)` in tectonics.ts)
        blur_field_n(crust_dist_raw, 3)
    }

    // ---------------------------------------------------------------------------
    // bake_volcanoes — Step 8
    // Mirrors `_bakeVolcanoes` + `_generateHotspots` in tectonics.ts.
    // ---------------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn bake_volcanoes(
        seed: u32,
        arc_density: f64,
        hotspot_count: u32,
        hotspot_intensity: f64,
        comp_id: &[u16],
        neighbor_id: &[u16],
        dist_field: &[f32],
        crust_dist: &[f32],
        conv_field: &[f32],
        _shear_field: &[f32],
        plates: &[Plate],
        warp_noise: &Noise3D,
        fine_warp_noise: &Noise3D,
    ) -> VolcanoBaked {
        let arc_conv_thresh_eff = ARC_CONV_THRESH / arc_density;

        const ARC_BAND:      f64 = 0.006;
        const ARC_SPACING:   f64 = 0.045;
        const ARC_MAX:       usize = 2000;
        const ARC_SIZE_SPAN: f64 = 0.25;
        let arc_spacing_eff = ARC_SPACING / arc_density;

        // Helper: domain-warp a direction using warp_noise (mirrors query() preamble).
        // Offsets: WARP_O1={13.7,0,0}, WARP_O2={0,13.7,0}, WARP_O3={0,0,13.7}
        let warp_dir = |d: DVec3| -> DVec3 {
            let dx = fbm(warp_noise, d.x + 13.7, d.y,         d.z,         WARP_OCTAVES, WARP_FREQ, 2.0, 0.5);
            let dy = fbm(warp_noise, d.x,         d.y + 13.7, d.z,         WARP_OCTAVES, WARP_FREQ, 2.0, 0.5);
            let dz = fbm(warp_noise, d.x,         d.y,         d.z + 13.7, WARP_OCTAVES, WARP_FREQ, 2.0, 0.5);
            DVec3::new(d.x + WARP_AMP * dx, d.y + WARP_AMP * dy, d.z + WARP_AMP * dz).normalize()
        };

        // Helper: fine-warp a direction (mirrors crustDist fine-warp in query())
        // Fine warp for crustDist sampling. Mirrors query() EXACTLY: the fbm is sampled
        // at the UNWARPED `dir` (offsets 5.1/7.3/11.7) and added to the warped `w`. The
        // old closure sampled the fbm at `w` (warp doubly-applied) — a bake-vs-query
        // mismatch that shifted arc-cone OC/OO crust-dist classification.
        let fine_warp = |dir: DVec3, w: DVec3| -> DVec3 {
            let fw0 = fbm(fine_warp_noise, dir.x + 5.1,  dir.y + 5.1,  dir.z + 5.1,  3, 9.0, 2.0, 0.5);
            let fw1 = fbm(fine_warp_noise, dir.x + 7.3,  dir.y + 7.3,  dir.z + 7.3,  3, 9.0, 2.0, 0.5);
            let fw2 = fbm(fine_warp_noise, dir.x + 11.7, dir.y + 11.7, dir.z + 11.7, 3, 9.0, 2.0, 0.5);
            DVec3::new(w.x + 0.025 * fw0, w.y + 0.025 * fw1, w.z + 0.025 * fw2).normalize()
        };
        // Fine-dist-warp for boundary_dist sampling (mirrors query(): amp 0.006, offsets
        // 3.7/8.2/12.9, fbm at the UNWARPED dir, added to warped w). ARC_BAND half-width
        // is 0.006 — exactly this amplitude — so the front-band gate lands on ki's texels.
        let fine_dist_warp = |dir: DVec3, w: DVec3| -> DVec3 {
            let f0 = fbm(fine_warp_noise, dir.x + 3.7,  dir.y + 3.7,  dir.z + 3.7,  3, 9.0, 2.0, 0.5);
            let f1 = fbm(fine_warp_noise, dir.x + 8.2,  dir.y + 8.2,  dir.z + 8.2,  3, 9.0, 2.0, 0.5);
            let f2 = fbm(fine_warp_noise, dir.x + 12.9, dir.y + 12.9, dir.z + 12.9, 3, 9.0, 2.0, 0.5);
            DVec3::new(w.x + 0.006 * f0, w.y + 0.006 * f1, w.z + 0.006 * f2).normalize()
        };

        // Full query at a direction (mirrors the essential parts of query())
        let full_query = |dir: DVec3| -> (f64, f64, f64, f64, f64) {
            // boundary dist, conv, crust_dist_v, other_crust_dist_v, convergence
            let w = warp_dir(dir);
            let t = dir_to_texel(w);
            let idx = texel_index(t.face, t.x, t.y);
            let pid = comp_id[idx] as usize;
            let nid = (neighbor_id[idx] as usize).min(plates.len().saturating_sub(1));

            let boundary_dist = sample_c1(dist_field, fine_dist_warp(dir, w));
            let convergence   = sample_c1(conv_field, w);

            // crustDist with fine warp
            let fw = fine_warp(dir, w);
            let crust_dist_v = sample_c1(crust_dist, fw);

            // otherCrustDist: mirror through boundary
            // Compute gradient for tHat
            let dt = dir_to_texel(dir);
            let nt0 = neighbor_texel(dt.face, dt.x, dt.y,  1,  0);
            let nt1 = neighbor_texel(dt.face, dt.x, dt.y, -1,  0);
            let nt2 = neighbor_texel(dt.face, dt.x, dt.y,  0,  1);
            let nt3 = neighbor_texel(dt.face, dt.x, dt.y,  0, -1);
            let d0 = dist_field[texel_index(nt0.face, nt0.x, nt0.y)] as f64;
            let d1 = dist_field[texel_index(nt1.face, nt1.x, nt1.y)] as f64;
            let d2 = dist_field[texel_index(nt2.face, nt2.x, nt2.y)] as f64;
            let d3 = dist_field[texel_index(nt3.face, nt3.x, nt3.y)] as f64;
            let dir0 = texel_to_dir(nt0.face, nt0.x, nt0.y);
            let dir1 = texel_to_dir(nt1.face, nt1.x, nt1.y);
            let dir2 = texel_to_dir(nt2.face, nt2.x, nt2.y);
            let dir3 = texel_to_dir(nt3.face, nt3.x, nt3.y);
            let grad_u = (d0 - d1) * 0.5;
            let grad_v = (d2 - d3) * 0.5;
            let ua = dir0 - dir1; let va = dir2 - dir3;
            let gw = ua * grad_u + va * grad_v;
            let gdot = gw.dot(dir);
            let pt = gw - gdot * dir;
            let g_len = pt.length();

            let mut other_crust_dist = crust_dist_v;
            if g_len > 1.0e-7 {
                let t_hat = -pt / g_len;
                // Axis = dir × tHat
                let ax = dir.y * t_hat.z - dir.z * t_hat.y;
                let ay = dir.z * t_hat.x - dir.x * t_hat.z;
                let az = dir.x * t_hat.y - dir.y * t_hat.x;
                let a_len = (ax * ax + ay * ay + az * az).sqrt();
                if a_len > 1.0e-7 {
                    let kx = ax / a_len; let ky = ay / a_len; let kz = az / a_len;
                    let theta = (2.0 * boundary_dist).min(0.06);
                    let cos_t = theta.cos(); let sin_t = theta.sin();
                    let vx = dir.x; let vy = dir.y; let vz = dir.z;
                    let kdotv = kx * vx + ky * vy + kz * vz;
                    let cx = ky * vz - kz * vy;
                    let cy = kz * vx - kx * vz;
                    let cz = kx * vy - ky * vx;
                    let mx = vx * cos_t + cx * sin_t + kx * kdotv * (1.0 - cos_t);
                    let my = vy * cos_t + cy * sin_t + ky * kdotv * (1.0 - cos_t);
                    let mz = vz * cos_t + cz * sin_t + kz * kdotv * (1.0 - cos_t);
                    let mlen = (mx * mx + my * my + mz * mz).sqrt().max(1.0e-9);
                    let mirror = DVec3::new(mx / mlen, my / mlen, mz / mlen);
                    let mirrored = sample_c1(crust_dist, mirror);
                    let g2 = g_len * g_len;
                    let grad_strength = ss(1.0e-6, 5.0e-6, g2);
                    other_crust_dist = crust_dist_v + grad_strength * (mirrored - crust_dist_v);
                }
            }
            (boundary_dist, convergence, crust_dist_v, other_crust_dist, pid as f64 + nid as f64 * 1_000_000.0)
        };

        // ---- a. FRONT CANDIDATES ----
        #[allow(dead_code)]
        struct Candidate {
            px: f64, py: f64, pz: f64,
            conv: f64,
            front_strength: f64,
            hash: u32,
        }
        let mut candidates: Vec<Candidate> = Vec::new();

        for face in 0..6 {
            for ty in 0..RES {
                for tx in 0..RES {
                    let dir = texel_to_dir(face, tx, ty);
                    let w = warp_dir(dir);
                    // Cheap pre-reject
                    let conv_pre = sample_c1(conv_field, w);
                    if conv_pre < arc_conv_thresh_eff * 0.5 { continue; }

                    // pid/nid at the WARPED texel (ki tectonics.ts:3254-3259). `tidx`
                    // (unwarped) is kept only for the candidate hash below (ki 2623-2624).
                    let tidx = texel_index(face, tx, ty);
                    let wt = dir_to_texel(w);
                    let tidx_w = texel_index(wt.face, wt.x, wt.y);
                    let pid = comp_id[tidx_w] as usize;
                    let nid = (neighbor_id[tidx_w] as usize).min(plates.len().saturating_sub(1));

                    let boundary_dist = sample_c1(dist_field, fine_dist_warp(dir, w));
                    let convergence   = sample_c1(conv_field, w);
                    if convergence <= arc_conv_thresh_eff { continue; }
                    if !(ARC_OFFSET - ARC_BAND..=ARC_OFFSET + ARC_BAND).contains(&boundary_dist) { continue; }

                    let fw = fine_warp(dir, w);
                    let crust_dist_v = sample_c1(crust_dist, fw);

                    // otherCrustDist via mirror (simplified — just use crust_dist for now)
                    let dt = dir_to_texel(dir);
                    let nt0 = neighbor_texel(dt.face, dt.x, dt.y,  1,  0);
                    let nt1 = neighbor_texel(dt.face, dt.x, dt.y, -1,  0);
                    let nt2 = neighbor_texel(dt.face, dt.x, dt.y,  0,  1);
                    let nt3 = neighbor_texel(dt.face, dt.x, dt.y,  0, -1);
                    let d0f = dist_field[texel_index(nt0.face, nt0.x, nt0.y)] as f64;
                    let d1f = dist_field[texel_index(nt1.face, nt1.x, nt1.y)] as f64;
                    let d2f = dist_field[texel_index(nt2.face, nt2.x, nt2.y)] as f64;
                    let d3f = dist_field[texel_index(nt3.face, nt3.x, nt3.y)] as f64;
                    let dir0f = texel_to_dir(nt0.face, nt0.x, nt0.y);
                    let dir1f = texel_to_dir(nt1.face, nt1.x, nt1.y);
                    let dir2f = texel_to_dir(nt2.face, nt2.x, nt2.y);
                    let dir3f = texel_to_dir(nt3.face, nt3.x, nt3.y);
                    let gu = (d0f - d1f) * 0.5;
                    let gv = (d2f - d3f) * 0.5;
                    let ua = dir0f - dir1f; let va = dir2f - dir3f;
                    let gw2 = ua * gu + va * gv;
                    let gdot = gw2.dot(dir);
                    let pt = gw2 - gdot * dir;
                    let g_len = pt.length();
                    let mut other_crust_dist = crust_dist_v;
                    if g_len > 1.0e-7 {
                        let t_hat = -pt / g_len;
                        let ax = dir.y * t_hat.z - dir.z * t_hat.y;
                        let ay = dir.z * t_hat.x - dir.x * t_hat.z;
                        let az = dir.x * t_hat.y - dir.y * t_hat.x;
                        let a_len = (ax * ax + ay * ay + az * az).sqrt();
                        if a_len > 1.0e-7 {
                            let kx = ax / a_len; let ky = ay / a_len; let kz = az / a_len;
                            let theta = (2.0 * boundary_dist).min(0.06);
                            let cos_t = theta.cos(); let sin_t = theta.sin();
                            let vx = dir.x; let vy = dir.y; let vz = dir.z;
                            let kdotv = kx * vx + ky * vy + kz * vz;
                            let cx2 = ky * vz - kz * vy;
                            let cy2 = kz * vx - kx * vz;
                            let cz2 = kx * vy - ky * vx;
                            let mx = vx * cos_t + cx2 * sin_t + kx * kdotv * (1.0 - cos_t);
                            let my = vy * cos_t + cy2 * sin_t + ky * kdotv * (1.0 - cos_t);
                            let mz = vz * cos_t + cz2 * sin_t + kz * kdotv * (1.0 - cos_t);
                            let mlen = (mx * mx + my * my + mz * mz).sqrt().max(1.0e-9);
                            let mirror_dir = DVec3::new(mx / mlen, my / mlen, mz / mlen);
                            let mirrored = sample_c1(crust_dist, mirror_dir);
                            let g2 = g_len * g_len;
                            let gs = ss(1.0e-6, 5.0e-6, g2);
                            other_crust_dist = crust_dist_v + gs * (mirrored - crust_dist_v);
                        }
                    }

                    let w_mine  = ss(-0.10, 0.10, crust_dist_v);
                    let w_other = ss(-0.10, 0.10, other_crust_dist);
                    let side_ramp = ss(0.002, 0.015, boundary_dist);
                    let oo_subduct = 0.5 + (if pid < nid { 1.0 } else { 0.0 } - 0.5) * side_ramp;

                    let w_co = w_mine * (1.0 - w_other);
                    let w_oo = (1.0 - w_mine) * (1.0 - w_other);

                    let mut front_strength = 0.0;
                    let mut is_candidate = false;
                    if w_mine > 0.5 && w_other < 0.5 {
                        front_strength = w_co;
                        is_candidate = true;
                    } else if w_mine < 0.5 && w_other < 0.5 && pid > nid {
                        front_strength = w_oo * (1.0 - oo_subduct);
                        is_candidate = true;
                    }
                    if !is_candidate { continue; }

                    let hash = derive_seed(tidx as u32, 14);
                    candidates.push(Candidate {
                        px: dir.x, py: dir.y, pz: dir.z,
                        conv: convergence,
                        front_strength,
                        hash,
                    });
                }
            }
        }

        let mut vol_pos:        Vec<f32> = Vec::new();
        let mut vol_base_radius: Vec<f32> = Vec::new();
        let mut vol_height:     Vec<f32> = Vec::new();
        let mut vol_crater_frac: Vec<f32> = Vec::new();
        let mut vol_kind:        Vec<u8>  = Vec::new();
        let mut vol_intensity:   Vec<f32> = Vec::new();

        // ---- b. DART-THROWING ----
        if !candidates.is_empty() {
            // Sort descending by convergence, hash tie-break
            candidates.sort_by(|a, b| {
                b.conv.partial_cmp(&a.conv).unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.hash.cmp(&a.hash))
            });

            let occ_res = 2.max(((std::f64::consts::PI / 2.0) / arc_spacing_eff).ceil() as usize);
            let n_occ_cells = 6 * occ_res * occ_res;
            let mut occupied = vec![0u8; n_occ_cells];
            let occ_cell_ang = (std::f64::consts::PI / 2.0) / occ_res as f64;
            let occ_reach = arc_spacing_eff + occ_cell_ang * std::f64::consts::SQRT_2;
            let occ_search_r = ((occ_reach / occ_cell_ang).ceil() as i32) + 1;

            let texel_index_occ = |face: usize, x: usize, y: usize| -> usize {
                face * occ_res * occ_res + y * occ_res + x
            };
            let dir_to_texel_occ = |d: DVec3| -> (usize, usize, usize) {
                let ax = d.x.abs(); let ay = d.y.abs(); let az = d.z.abs();
                let (face, u, v) = if ax >= ay && ax >= az {
                    if d.x > 0.0 { (0,  d.z / d.x,  d.y / d.x) } else { (1, -d.z / ax,   d.y / ax) }
                } else if ay >= ax && ay >= az {
                    if d.y > 0.0 { (2, d.x / d.y,  d.z / d.y) } else { (3, d.x / ay, -d.z / ay) }
                } else if d.z > 0.0 {
                    (4, -d.x / d.z,  d.y / d.z)
                } else {
                    (5,  d.x / az,   d.y / az)
                };
                let hr = occ_res as f64 * 0.5;
                let xi = ((u + 1.0) * hr).floor() as isize;
                let yi = ((v + 1.0) * hr).floor() as isize;
                let x2 = xi.clamp(0, (occ_res as isize) - 1) as usize;
                let y2 = yi.clamp(0, (occ_res as isize) - 1) as usize;
                (face, x2, y2)
            };
            let texel_to_dir_occ = |face: usize, x: usize, y: usize| -> DVec3 {
                let hr = occ_res as f64 * 0.5;
                let u = (x as f64 + 0.5) / hr - 1.0;
                let v = (y as f64 + 0.5) / hr - 1.0;
                let raw = match face {
                    0 => DVec3::new(1.0,  v,  u),
                    1 => DVec3::new(-1.0, v, -u),
                    2 => DVec3::new(u,  1.0,  v),
                    3 => DVec3::new(u, -1.0, -v),
                    4 => DVec3::new(-u,  v,  1.0),
                    _ => DVec3::new(u,   v, -1.0),
                };
                raw.normalize()
            };

            for cand in &candidates {
                if vol_pos.len() / 3 >= ARC_MAX { break; }

                // Seeded jitter
                let mut jitter_rng = make_rng(derive_seed(cand.hash ^ seed, 14));
                let jitter_angle   = jitter_rng() * arc_spacing_eff * 0.4;
                let jitter_azimuth = jitter_rng() * std::f64::consts::TAU;
                let px = cand.px; let py = cand.py; let pz = cand.pz;

                // Build orthonormal tangent frame
                let (mut ux, mut uy, uz) = if py.abs() > 0.9 { (1.0, 0.0, 0.0) } else { (0.0, 1.0, 0.0) };
                let mut t1x = py * uz - pz * uy;
                let mut t1y = pz * ux - px * uz;
                let mut t1z = px * uy - py * ux;
                let t1len = (t1x * t1x + t1y * t1y + t1z * t1z).sqrt().max(1.0e-9);
                t1x /= t1len; t1y /= t1len; t1z /= t1len;
                let t2x = py * t1z - pz * t1y;
                let t2y = pz * t1x - px * t1z;
                let t2z = px * t1y - py * t1x;
                let _ = (ux, uy);
                ux = 0.0; uy = 0.0; // suppress warnings
                let _ = (ux, uy);
                let cos_az = jitter_azimuth.cos(); let sin_az = jitter_azimuth.sin();
                let tx2 = cos_az * t1x + sin_az * t2x;
                let ty2 = cos_az * t1y + sin_az * t2y;
                let tz2 = cos_az * t1z + sin_az * t2z;
                let ax_x = py * tz2 - pz * ty2;
                let ax_y = pz * tx2 - px * tz2;
                let ax_z = px * ty2 - py * tx2;
                let ax_len = (ax_x * ax_x + ax_y * ax_y + ax_z * ax_z).sqrt().max(1.0e-9);
                let kx = ax_x / ax_len; let ky = ax_y / ax_len; let kz = ax_z / ax_len;
                let cos_j = jitter_angle.cos(); let sin_j = jitter_angle.sin();
                let kdot = kx * px + ky * py + kz * pz;
                let jx = px * cos_j + (ky * pz - kz * py) * sin_j + kx * kdot * (1.0 - cos_j);
                let jy = py * cos_j + (kz * px - kx * pz) * sin_j + ky * kdot * (1.0 - cos_j);
                let jz = pz * cos_j + (kx * py - ky * px) * sin_j + kz * kdot * (1.0 - cos_j);
                let jlen = (jx * jx + jy * jy + jz * jz).sqrt().max(1.0e-9);
                let pos_x = jx / jlen; let pos_y = jy / jlen; let pos_z = jz / jlen;

                // Min-distance reject
                let pos_dir = DVec3::new(pos_x, pos_y, pos_z);
                let (occ_face, occ_x, occ_y) = dir_to_texel_occ(pos_dir);
                let my_cell = texel_index_occ(occ_face, occ_x, occ_y);
                if occupied[my_cell] != 0 { continue; }

                // Re-validate at the final position via full_query — which now reproduces
                // query()'s warp handling EXACTLY (boundary_dist fine-dist-warped, crustDist
                // fine-warped at the unwarped dir, mirrored otherCrustDist). ki's bake calls
                // this.query() here (tectonics.ts:2733-2744); full_query is our mid-bake
                // equivalent (query() reads not-yet-stored self.* fields).
                let w2 = warp_dir(pos_dir);
                let tidx2 = texel_index(dir_to_texel(w2).face, dir_to_texel(w2).x, dir_to_texel(w2).y);
                let pid2 = comp_id[tidx2] as usize;
                let nid2 = (neighbor_id[tidx2] as usize).min(plates.len().saturating_sub(1));
                let (fbd, fconv, f_crust_dist, f_other_crust, _) = full_query(pos_dir);
                if fconv <= arc_conv_thresh_eff { continue; }
                if !(ARC_OFFSET - ARC_BAND..=ARC_OFFSET + ARC_BAND).contains(&fbd) { continue; }
                let fw_mine  = ss(-0.10, 0.10, f_crust_dist);
                let fw_other = ss(-0.10, 0.10, f_other_crust);
                let side_ok = (fw_mine > 0.5 && fw_other < 0.5)
                           || (fw_mine < 0.5 && fw_other < 0.5 && pid2 > nid2);
                if !side_ok { continue; }

                // Accept
                let conv = fconv;
                let conv_strength = ((conv - arc_conv_thresh_eff) / ARC_SIZE_SPAN).clamp(0.0, 1.0);
                let r_jitter  = 1.0 + (jitter_rng() * 2.0 - 1.0) * 0.15;
                let h_jitter  = 1.0 + (jitter_rng() * 2.0 - 1.0) * 0.15;
                let cf_jitter = (jitter_rng() * 2.0 - 1.0) * 0.03;
                let base_radius       = (0.054 + (0.070 - 0.054) * conv_strength) * r_jitter;
                let height            = (0.37  + (0.62  - 0.37)  * conv_strength) * h_jitter;
                let crater_radius_frac = (0.12 + cf_jitter).max(0.05);

                vol_pos.push(pos_x as f32); vol_pos.push(pos_y as f32); vol_pos.push(pos_z as f32);
                vol_base_radius.push(base_radius as f32);
                vol_height.push(height as f32);
                vol_crater_frac.push(crater_radius_frac as f32);
                vol_kind.push(0); // arc
                vol_intensity.push(1.0);

                // Stamp occupancy
                let (c_face, c_x, c_y) = (occ_face, occ_x, occ_y);
                for df in 0..6 {
                    for dty in 0..occ_res {
                        for dtx in 0..occ_res {
                            if df == c_face && ((dtx as i32 - c_x as i32).abs() > occ_search_r
                            || (dty as i32 - c_y as i32).abs() > occ_search_r) {
                                continue;
                            }
                            let cell_dir = texel_to_dir_occ(df, dtx, dty);
                            let dot_vc = pos_x * cell_dir.x + pos_y * cell_dir.y + pos_z * cell_dir.z;
                            let ang_vc = dot_vc.clamp(-1.0, 1.0).acos();
                            if ang_vc <= arc_spacing_eff {
                                occupied[texel_index_occ(df, dtx, dty)] = 1;
                            }
                        }
                    }
                }
            }
        }

        // ---- c. HOTSPOT GENERATION ----
        let mut hotspot_pos:           Vec<f32> = Vec::new();
        let mut hotspot_intensity_arr: Vec<f32> = Vec::new();
        let mut hotspot_is_chain:      Vec<u8>  = Vec::new();

        if hotspot_count > 0 {
            let hotspot_polar_cos: f64 = ((8.0f64).to_radians()).cos();
            const HOTSPOT_V_MOVING:   f64 = 0.15;
            const HOTSPOT_CHAIN_STEP: f64 = 0.035;
            const HOTSPOT_BASE_R:     f64 = 0.06;
            const HOTSPOT_BASE_H:     f64 = 0.6;

            let mut pos_rng = make_rng(derive_seed(seed, 16));
            let mut int_rng = make_rng(derive_seed(seed, 17));

            for _ in 0..hotspot_count {
                let mut hx = 0.0; let mut hy = 0.0; let mut hz = 0.0;
                'attempt: for _ in 0..50 {
                    let rx = pos_rng() * 2.0 - 1.0;
                    let ry = pos_rng() * 2.0 - 1.0;
                    let rz = pos_rng() * 2.0 - 1.0;
                    let len = (rx * rx + ry * ry + rz * rz).sqrt();
                    if len > 0.01 && len < 1.0 {
                        let nx = rx / len; let ny = ry / len; let nz = rz / len;
                        if ny.abs() > hotspot_polar_cos { continue; }
                        hx = nx; hy = ny; hz = nz;
                        break 'attempt;
                    }
                }
                if hx == 0.0 && hy == 0.0 && hz == 0.0 { continue; }

                let u = int_rng();
                let intensity = ((0.45 + 0.55 * u * u) * hotspot_intensity).clamp(0.0, 1.0);

                // Velocity at hotspot position
                let h_dir = DVec3::new(hx, hy, hz);
                let hw = warp_dir(h_dir);
                let htidx = texel_index(dir_to_texel(hw).face, dir_to_texel(hw).x, dir_to_texel(hw).y);
                let hpid = comp_id[htidx] as usize;
                let omega = plates[hpid].omega;
                let vel = DVec3::new(
                    omega.y * hz - omega.z * hy,
                    omega.z * hx - omega.x * hz,
                    omega.x * hy - omega.y * hx,
                );
                let speed = vel.length();

                if speed > HOTSPOT_V_MOVING {
                    let v_hat = vel / speed;
                    let n_cones = 2.max(9.min((2.0 + 4.0 * intensity).round() as usize));
                    for i in 0..n_cones {
                        let age = i as f64 / (n_cones - 1).max(1) as f64;
                        let age_scale = 1.0 - 0.7 * age;
                        let step_dist = i as f64 * HOTSPOT_CHAIN_STEP;
                        let cx = hx + v_hat.x * step_dist;
                        let cy = hy + v_hat.y * step_dist;
                        let cz = hz + v_hat.z * step_dist;
                        let c_len = (cx * cx + cy * cy + cz * cz).sqrt().max(1.0e-9);
                        let (cpx, cpy, cpz) = (cx / c_len, cy / c_len, cz / c_len);
                        let base_radius = HOTSPOT_BASE_R * (0.7 + 0.6 * intensity) * age_scale;
                        let height      = HOTSPOT_BASE_H * (0.6 + 0.8 * intensity) * age_scale;
                        vol_pos.push(cpx as f32); vol_pos.push(cpy as f32); vol_pos.push(cpz as f32);
                        vol_base_radius.push(base_radius as f32);
                        vol_height.push(height as f32);
                        vol_crater_frac.push(0.12f32);
                        vol_kind.push(1); // hotspot
                        vol_intensity.push(intensity as f32);
                    }
                    hotspot_pos.push(hx as f32); hotspot_pos.push(hy as f32); hotspot_pos.push(hz as f32);
                    hotspot_intensity_arr.push(intensity as f32);
                    hotspot_is_chain.push(1);
                } else {
                    let base_radius = HOTSPOT_BASE_R * 1.6 * (0.7 + 0.6 * intensity);
                    let height      = HOTSPOT_BASE_H * 1.3 * (0.6 + 0.8 * intensity);
                    vol_pos.push(hx as f32); vol_pos.push(hy as f32); vol_pos.push(hz as f32);
                    vol_base_radius.push(base_radius as f32);
                    vol_height.push(height as f32);
                    vol_crater_frac.push(0.15f32);
                    vol_kind.push(1);
                    vol_intensity.push(intensity as f32);
                    hotspot_pos.push(hx as f32); hotspot_pos.push(hy as f32); hotspot_pos.push(hz as f32);
                    hotspot_intensity_arr.push(intensity as f32);
                    hotspot_is_chain.push(0);
                }
            }
        }

        if vol_pos.is_empty() {
            return VolcanoBaked::empty();
        }

        // ---- d. BUILD SPATIAL INDEX ----
        let n_vol = vol_pos.len() / 3;
        let bucket_res: usize = 24;
        let n_bucket_cells = 6 * bucket_res * bucket_res;
        let cell_ang_size = (std::f64::consts::PI / 2.0) / bucket_res as f64;
        let cell_diag = cell_ang_size * std::f64::consts::SQRT_2;

        let dir_to_texel_bkt = |d: DVec3| -> (usize, usize, usize) {
            let ax = d.x.abs(); let ay = d.y.abs(); let az = d.z.abs();
            let (face, u, v) = if ax >= ay && ax >= az {
                if d.x > 0.0 { (0,  d.z / d.x,  d.y / d.x) } else { (1, -d.z / ax,   d.y / ax) }
            } else if ay >= ax && ay >= az {
                if d.y > 0.0 { (2, d.x / d.y,  d.z / d.y) } else { (3, d.x / ay, -d.z / ay) }
            } else if d.z > 0.0 {
                (4, -d.x / d.z,  d.y / d.z)
            } else {
                (5,  d.x / az,   d.y / az)
            };
            let hr = bucket_res as f64 * 0.5;
            let xi = ((u + 1.0) * hr).floor() as isize;
            let yi = ((v + 1.0) * hr).floor() as isize;
            let x2 = xi.clamp(0, (bucket_res as isize) - 1) as usize;
            let y2 = yi.clamp(0, (bucket_res as isize) - 1) as usize;
            (face, x2, y2)
        };
        let texel_to_dir_bkt = |face: usize, x: usize, y: usize| -> DVec3 {
            let hr = bucket_res as f64 * 0.5;
            let u = (x as f64 + 0.5) / hr - 1.0;
            let v = (y as f64 + 0.5) / hr - 1.0;
            let raw = match face {
                0 => DVec3::new(1.0,  v,  u),
                1 => DVec3::new(-1.0, v, -u),
                2 => DVec3::new(u,  1.0,  v),
                3 => DVec3::new(u, -1.0, -v),
                4 => DVec3::new(-u,  v,  1.0),
                _ => DVec3::new(u,   v, -1.0),
            };
            raw.normalize()
        };
        let texel_idx_bkt = |face: usize, x: usize, y: usize| -> usize {
            face * bucket_res * bucket_res + y * bucket_res + x
        };

        let mut cell_count = vec![0i32; n_bucket_cells];
        let mut vol_cells: Vec<Vec<usize>> = Vec::with_capacity(n_vol);

        for vi in 0..n_vol {
            let vx = vol_pos[vi * 3] as f64;
            let vy = vol_pos[vi * 3 + 1] as f64;
            let vz = vol_pos[vi * 3 + 2] as f64;
            let reach = vol_base_radius[vi] as f64 + cell_diag;
            let v_dir = DVec3::new(vx, vy, vz);
            let (c_face, c_x, c_y) = dir_to_texel_bkt(v_dir);
            let search_r = ((reach / cell_ang_size).ceil() as i32) + 1;

            let mut my_cells: Vec<usize> = Vec::new();
            for df in 0..6 {
                for dty in 0..bucket_res {
                    for dtx in 0..bucket_res {
                        if df == c_face && ((dtx as i32 - c_x as i32).abs() > search_r
                        || (dty as i32 - c_y as i32).abs() > search_r) {
                            continue;
                        }
                        let cell_dir = texel_to_dir_bkt(df, dtx, dty);
                        let dot_vc = vx * cell_dir.x + vy * cell_dir.y + vz * cell_dir.z;
                        let ang_vc = dot_vc.clamp(-1.0, 1.0).acos();
                        if ang_vc <= reach {
                            let bc = texel_idx_bkt(df, dtx, dty);
                            my_cells.push(bc);
                            cell_count[bc] += 1;
                        }
                    }
                }
            }
            vol_cells.push(my_cells);
        }

        // Build CSR offsets
        let mut bucket_start = vec![0i32; n_bucket_cells + 1];
        for c in 0..n_bucket_cells {
            bucket_start[c + 1] = bucket_start[c] + cell_count[c];
        }
        let total_inserts = bucket_start[n_bucket_cells] as usize;
        let mut bucket_idx = vec![0i32; total_inserts];
        let mut write_cursor = vec![0i32; n_bucket_cells];
        #[allow(clippy::needless_range_loop)]
        for vi in 0..n_vol {
            for &cell in &vol_cells[vi] {
                let wp = bucket_start[cell] + write_cursor[cell];
                bucket_idx[wp as usize] = vi as i32;
                write_cursor[cell] += 1;
            }
        }

        VolcanoBaked {
            vol_pos, vol_base_radius, vol_height, vol_crater_frac,
            vol_kind, vol_intensity,
            vol_bucket_res: bucket_res as u32,
            vol_bucket_start: bucket_start,
            vol_bucket_idx: bucket_idx,
            hotspot_pos, hotspot_intensity_arr, hotspot_is_chain,
        }
    }

    // ---------------------------------------------------------------------------
    // Query plate ownership and boundary info at surface point `dir` (unit vector).
    ///
    /// All fields are now populated from real data (Phase 2 complete).
    pub fn query(&self, dir: DVec3) -> TectonicQuery {
        // Domain-warp the query direction.
        // Mirrors TS query() lines:
        //   dx = fbm(wn, x+13.7, y+0.0, z+0.0, ...)  ← WARP_O1
        //   dy = fbm(wn, x+0.0,  y+13.7,z+0.0, ...)  ← WARP_O2
        //   dz = fbm(wn, x+0.0,  y+0.0, z+13.7,...)  ← WARP_O3
        let dx = fbm(&self.warp_noise, dir.x + 13.7, dir.y,         dir.z,
                     WARP_OCTAVES, WARP_FREQ, 2.0, 0.5);
        let dy = fbm(&self.warp_noise, dir.x,         dir.y + 13.7, dir.z,
                     WARP_OCTAVES, WARP_FREQ, 2.0, 0.5);
        let dz = fbm(&self.warp_noise, dir.x,         dir.y,         dir.z + 13.7,
                     WARP_OCTAVES, WARP_FREQ, 2.0, 0.5);
        let w = DVec3::new(
            dir.x + WARP_AMP * dx,
            dir.y + WARP_AMP * dy,
            dir.z + WARP_AMP * dz,
        ).normalize();

        // Table lookups
        let t = dir_to_texel(w);
        let idx       = texel_index(t.face, t.x, t.y);
        let plate_id  = self.comp_id[idx] as usize;
        let neighbor_id = (self.neighbor_id[idx] as usize)
            .min(self.plates.len().saturating_sub(1));

        // Smooth boundary distance.
        // Fine warp for dist_field (tectonics.ts:3265-3271): breaks the 177 m staircase
        // isoline without large profile drift. Amp 0.006 ≈ 3 cells; freq 9 ≈ 2° wavelength;
        // offsets 3.7/8.2/12.9 differ from the crust_dist warp (5.1/7.3/11.7) so the two
        // warps are uncorrelated. Same fine_warp_noise basis as the crust_dist warp below.
        let fdw0 = fbm(&self.fine_warp_noise, dir.x + 3.7,  dir.y + 3.7,  dir.z + 3.7,  3, 9.0, 2.0, 0.5);
        let fdw1 = fbm(&self.fine_warp_noise, dir.x + 8.2,  dir.y + 8.2,  dir.z + 8.2,  3, 9.0, 2.0, 0.5);
        let fdw2 = fbm(&self.fine_warp_noise, dir.x + 12.9, dir.y + 12.9, dir.z + 12.9, 3, 9.0, 2.0, 0.5);
        let fd_dir = DVec3::new(
            w.x + 0.006 * fdw0,
            w.y + 0.006 * fdw1,
            w.z + 0.006 * fdw2,
        ).normalize();
        let boundary_dist = sample_c1(&self.dist_field, fd_dir);

        // Convergence and shear from pre-baked blurred fields
        let convergence = sample_c1(&self.conv_field,  w);
        let shear       = sample_c1(&self.shear_field, w);

        // paleoDist: sample using the standard warped direction w
        let paleo_dist = sample_c1(&self.paleo_dist, w);

        // baseElevation: blurred per-plate base elevation, C1-sampled (no Voronoi step)
        let base_elevation = sample_c1(&self.base_elev_field, w);

        // rockHardness: C1 sample from baked grid using the same domain-warped dir w.
        let rock_hardness = sample_c1(&self.rock_hardness, w);

        // Fine warp for coastline complexity — applied ONLY to crustDist sampling
        let fw0 = fbm(&self.fine_warp_noise, dir.x + 5.1,  dir.y + 5.1,  dir.z + 5.1,  3, 9.0, 2.0, 0.5);
        let fw1 = fbm(&self.fine_warp_noise, dir.x + 7.3,  dir.y + 7.3,  dir.z + 7.3,  3, 9.0, 2.0, 0.5);
        let fw2 = fbm(&self.fine_warp_noise, dir.x + 11.7, dir.y + 11.7, dir.z + 11.7, 3, 9.0, 2.0, 0.5);
        let fw_dir = DVec3::new(
            w.x + 0.025 * fw0,
            w.y + 0.025 * fw1,
            w.z + 0.025 * fw2,
        ).normalize();
        let crust_dist = sample_c1(&self.crust_dist, fw_dir);

        // otherCrustDist: mirror through boundary to the neighbor side via Rodrigues rotation.
        // Mirrors the `otherCrustDist` block in tectonics.ts query().
        // We compute the gradient of the dist_field at the UNWARPED dir to get tHat,
        // then rotate dir by 2*boundaryDist about (dir × tHat).
        let t_orig = dir_to_texel(dir);
        let nt0 = neighbor_texel(t_orig.face, t_orig.x, t_orig.y,  1,  0);
        let nt1 = neighbor_texel(t_orig.face, t_orig.x, t_orig.y, -1,  0);
        let nt2 = neighbor_texel(t_orig.face, t_orig.x, t_orig.y,  0,  1);
        let nt3 = neighbor_texel(t_orig.face, t_orig.x, t_orig.y,  0, -1);
        let d0 = self.dist_field[texel_index(nt0.face, nt0.x, nt0.y)] as f64;
        let d1 = self.dist_field[texel_index(nt1.face, nt1.x, nt1.y)] as f64;
        let d2 = self.dist_field[texel_index(nt2.face, nt2.x, nt2.y)] as f64;
        let d3 = self.dist_field[texel_index(nt3.face, nt3.x, nt3.y)] as f64;
        let gdir0 = texel_to_dir(nt0.face, nt0.x, nt0.y);
        let gdir1 = texel_to_dir(nt1.face, nt1.x, nt1.y);
        let gdir2 = texel_to_dir(nt2.face, nt2.x, nt2.y);
        let gdir3 = texel_to_dir(nt3.face, nt3.x, nt3.y);
        let grad_u = (d0 - d1) * 0.5;
        let grad_v = (d2 - d3) * 0.5;
        let ua = gdir0 - gdir1;
        let va = gdir2 - gdir3;
        let gw = ua * grad_u + va * grad_v;
        let gdot = gw.dot(dir);
        let pt = gw - gdot * dir;
        let g_len = pt.length();

        let other_crust_dist = if g_len > 1.0e-7 {
            let t_hat = -pt / g_len;
            // axis = dir × tHat
            let ax = dir.y * t_hat.z - dir.z * t_hat.y;
            let ay = dir.z * t_hat.x - dir.x * t_hat.z;
            let az = dir.x * t_hat.y - dir.y * t_hat.x;
            let a_len = (ax * ax + ay * ay + az * az).sqrt();
            if a_len > 1.0e-7 {
                let kx = ax / a_len; let ky = ay / a_len; let kz = az / a_len;
                let theta = (2.0 * boundary_dist).min(0.06);
                let cos_t = theta.cos(); let sin_t = theta.sin();
                let vx = dir.x; let vy = dir.y; let vz = dir.z;
                let kdotv = kx * vx + ky * vy + kz * vz;
                let cx = ky * vz - kz * vy;
                let cy = kz * vx - kx * vz;
                let cz = kx * vy - ky * vx;
                let mx = vx * cos_t + cx * sin_t + kx * kdotv * (1.0 - cos_t);
                let my = vy * cos_t + cy * sin_t + ky * kdotv * (1.0 - cos_t);
                let mz = vz * cos_t + cz * sin_t + kz * kdotv * (1.0 - cos_t);
                let mlen = (mx * mx + my * my + mz * mz).sqrt().max(1.0e-9);
                let mirror_dir = DVec3::new(mx / mlen, my / mlen, mz / mlen);
                let mirrored = sample_c1(&self.crust_dist, mirror_dir);
                let g2 = g_len * g_len;
                let grad_strength = ss(1.0e-6, 5.0e-6, g2);
                crust_dist + grad_strength * (mirrored - crust_dist)
            } else {
                crust_dist
            }
        } else {
            crust_dist
        };

        TectonicQuery {
            plate_id,
            neighbor_id,
            boundary_dist,
            convergence,
            shear,
            base_elevation,
            crust_dist,
            paleo_dist,
            other_crust_dist,
            rock_hardness,
        }
    }

    /// Number of plates generated.
    pub fn plate_count(&self) -> usize { self.plates.len() }

    /// Wide-smoothed uplift field value at `dir` (C1). Positive = broad orogen,
    /// negative = broad basin. Uses the same domain-warp as `query`.
    /// Mirrors `upliftAt` in tectonics.ts (lines ~3193-3210). Consumed by Phase-4
    /// broad-deform; currently otherwise unused.
    #[allow(dead_code)]
    pub fn uplift_at(&self, dir: DVec3) -> f64 {
        let dx = fbm(&self.warp_noise, dir.x + 13.7, dir.y,         dir.z,
                     WARP_OCTAVES, WARP_FREQ, 2.0, 0.5);
        let dy = fbm(&self.warp_noise, dir.x,         dir.y + 13.7, dir.z,
                     WARP_OCTAVES, WARP_FREQ, 2.0, 0.5);
        let dz = fbm(&self.warp_noise, dir.x,         dir.y,         dir.z + 13.7,
                     WARP_OCTAVES, WARP_FREQ, 2.0, 0.5);
        let w = DVec3::new(
            dir.x + WARP_AMP * dx,
            dir.y + WARP_AMP * dy,
            dir.z + WARP_AMP * dz,
        ).normalize();
        sample_c1(&self.uplift_field, w)
    }

    /// Substrate rock hardness at `dir` (C1). Returns [0,1]: 1 = hard fresh igneous/volcanic,
    /// 0 = soft sediment/basin. Pure baked-grid lookup (identical main-thread/worker); uses
    /// the same domain-warp as `query`. Mirrors `hardnessAt` in tectonics.ts (lines ~3402-3419).
    /// Consumed by Phase-3 erosion + Phase-4 sampler; otherwise unused for now.
    #[allow(dead_code)]
    pub fn hardness_at(&self, dir: DVec3) -> f64 {
        let dx = fbm(&self.warp_noise, dir.x + 13.7, dir.y,         dir.z,
                     WARP_OCTAVES, WARP_FREQ, 2.0, 0.5);
        let dy = fbm(&self.warp_noise, dir.x,         dir.y + 13.7, dir.z,
                     WARP_OCTAVES, WARP_FREQ, 2.0, 0.5);
        let dz = fbm(&self.warp_noise, dir.x,         dir.y,         dir.z + 13.7,
                     WARP_OCTAVES, WARP_FREQ, 2.0, 0.5);
        let w = DVec3::new(
            dir.x + WARP_AMP * dx,
            dir.y + WARP_AMP * dy,
            dir.z + WARP_AMP * dz,
        ).normalize();
        sample_c1(&self.rock_hardness, w)
    }

    /// Volcano elevation contribution at `dir`.
    /// Cone profile: f(x) = (1 - x²)² with caldera dimple. Plus flank roughness.
    /// Mirrors `volcanoElevation` in tectonics.ts.
    pub fn volcano_elevation(&self, dir: DVec3) -> f64 {
        let n_vol = self.vol_pos.len() / 3;
        if n_vol == 0 { return 0.0; }
        let bucket_res = self.vol_bucket_res as usize;
        if bucket_res == 0 { return 0.0; }

        // Find bucket cell
        let ax = dir.x.abs(); let ay = dir.y.abs(); let az = dir.z.abs();
        let (face_b, ub, vb) = if ax >= ay && ax >= az {
            if dir.x > 0.0 { (0,  dir.z / dir.x,  dir.y / dir.x) }
            else            { (1, -dir.z / ax,      dir.y / ax) }
        } else if ay >= ax && ay >= az {
            if dir.y > 0.0 { (2, dir.x / dir.y,  dir.z / dir.y) }
            else            { (3, dir.x / ay,     -dir.z / ay) }
        } else if dir.z > 0.0 {
            (4, -dir.x / dir.z,  dir.y / dir.z)
        } else {
            (5,  dir.x / az,     dir.y / az)
        };
        let hr_b = bucket_res as f64 * 0.5;
        let xb = ((ub + 1.0) * hr_b).floor() as isize;
        let yb = ((vb + 1.0) * hr_b).floor() as isize;
        let xb = xb.clamp(0, (bucket_res as isize) - 1) as usize;
        let yb = yb.clamp(0, (bucket_res as isize) - 1) as usize;
        let cell = face_b * bucket_res * bucket_res + yb * bucket_res + xb;

        let start = self.vol_bucket_start[cell] as usize;
        let end   = self.vol_bucket_start[cell + 1] as usize;
        if start == end { return 0.0; }

        let dx = dir.x; let dy2 = dir.y; let dz = dir.z;

        // Perturbation 1 (base warp) — single noise sample at dir
        const VOLC_BASE_WARP_FREQ: f64 = 14.0;
        const VOLC_BASE_WARP:      f64 = 0.32;
        const VOLC_FLANK_ROUGH:    f64 = 0.22;
        const VOLC_FLANK_FREQ:     f64 = 26.0;

        let n_warp = self.vol_noise.sample(
            dx * VOLC_BASE_WARP_FREQ,
            dy2 * VOLC_BASE_WARP_FREQ,
            dz * VOLC_BASE_WARP_FREQ,
        );

        let mut sum = 0.0f64;
        for i in start..end {
            let vi = self.vol_bucket_idx[i] as usize;
            let vx = self.vol_pos[vi * 3] as f64;
            let vy = self.vol_pos[vi * 3 + 1] as f64;
            let vz = self.vol_pos[vi * 3 + 2] as f64;
            let dot_val = (dx * vx + dy2 * vy + dz * vz).clamp(-1.0, 1.0);
            let a = dot_val.acos();
            let br = self.vol_base_radius[vi] as f64;

            // Warp base radius
            let br_eff = (br * (1.0 + VOLC_BASE_WARP * n_warp)).max(br * 0.3);
            if a >= br_eff { continue; }

            let x = a / br_eff;
            let x2 = x * x;
            let mut cone = self.vol_height[vi] as f64 * (1.0 - x2) * (1.0 - x2);

            // Perturbation 2: flank roughness
            let flank_mask = (std::f64::consts::PI * x).sin();
            let rough = fbm(&self.vol_noise,
                dx * VOLC_FLANK_FREQ, dy2 * VOLC_FLANK_FREQ, dz * VOLC_FLANK_FREQ,
                3, 1.0, 2.0, 0.5);
            cone *= 1.0 + VOLC_FLANK_ROUGH * flank_mask * rough;

            let cf = self.vol_crater_frac[vi] as f64;
            if x < cf {
                let xc = x / cf;
                let xc2 = xc * xc;
                let crater_depth = 0.18 * self.vol_height[vi] as f64;
                sum += cone - crater_depth * (1.0 - xc2) * (1.0 - xc2);
            } else {
                sum += cone;
            }
        }
        sum
    }
}

// ---------------------------------------------------------------------------
// boundary_relief — pure function, mirrors `boundaryRelief` in tectonics.ts.
// Returns a height contribution in normalized units ≈[-1,1].
// ---------------------------------------------------------------------------

// Profile shape constants (matches tectonics.ts exactly)
const BR_TRENCH_DEPTH:         f64 = 0.60;
const BR_FLEX_BULGE:           f64 = 0.05;
const BR_FLEX_POS:             f64 = 0.030;
const BR_FLEX_WIDTH:           f64 = 0.012;
const BR_TRENCH_WIDTH:         f64 = 0.020;
const BR_CORD_HEIGHT:          f64 = 0.72;
const BR_CORD_POS:             f64 = 0.038;
const BR_CORD_WIDTH:           f64 = 0.024;
const BR_CORD_RIDGE:           f64 = 0.60;
const BR_CORD_RIDGE_BASE:      f64 = 0.55;
const BR_SHELF_DIP:            f64 = 0.06;
const BR_SHELF_WIDTH:          f64 = 0.006;
const BR_OO_TRENCH_DEPTH:      f64 = 0.50;
const BR_OO_TRENCH_WIDTH:      f64 = 0.016;
const BR_ARC_HEIGHT:           f64 = 0.55;
const BR_ARC_POS:              f64 = 0.026;
const BR_ARC_WIDTH:            f64 = 0.014;
const BR_ARC_RIDGE_THRESH:     f64 = 0.45;
const BR_ARC_RIDGE_SCALE:      f64 = 1.0 / 0.55;
const BR_BELT_HEIGHT:          f64 = 0.82;
const BR_BELT_WIDTH:           f64 = 0.075;
const BR_BELT_RIDGE:           f64 = 0.40;
const BR_BELT_RIDGE_BASE:      f64 = 0.58;
const BR_PLATEAU_HEIGHT:       f64 = 0.24;
const BR_PLATEAU_INNER:        f64 = 0.05;
const BR_PLATEAU_OUTER:        f64 = 0.13;
const BR_PLATEAU_NEAR:         f64 = 0.015;
const BR_PLATEAU_FAR:          f64 = 0.05;
const BR_RIDGE_HEIGHT:         f64 = 0.22;
const BR_RIDGE_WIDTH:          f64 = 0.05;
const BR_NOTCH_DEPTH:          f64 = 0.14;
const BR_NOTCH_WIDTH:          f64 = 0.012;
const BR_RIFT_SHOULDER:        f64 = 0.18;
const BR_RIFT_SHOULDER_POS:    f64 = 0.022;
const BR_RIFT_SHOULDER_WIDTH:  f64 = 0.012;
const BR_GRABEN_DEPTH:         f64 = 0.24;
const BR_GRABEN_WIDTH:         f64 = 0.020;
const BR_SCARP_DEPTH:          f64 = 0.10;
const BR_SCARP_WIDTH:          f64 = 0.008;
const BR_RIDGE2_HEIGHT:        f64 = 0.05;
const BR_RIDGE2_POS:           f64 = 0.012;
const BR_RIDGE2_WIDTH:         f64 = 0.008;

/// Asymmetric tectonic boundary relief profiles.
///
/// Returns a height contribution in normalized units ≈[-1,1].
/// `ridged_at(dir, freq, octaves)` should return ridged multifractal noise ≈[0,1].
/// Mirrors `boundaryRelief` in tectonics.ts exactly.
pub fn boundary_relief(
    q: &TectonicQuery,
    plates: &[Plate],
    dir: DVec3,
    ridged_at: &dyn Fn(DVec3, f64, u32) -> f64,
    base: f64,
) -> f64 {
    let d         = q.boundary_dist;
    let side_ramp = ss(0.002, 0.015, d);
    let own       = &plates[q.plate_id];
    let other     = &plates[q.neighbor_id];
    let conv      = q.convergence;
    let sh        = q.shear;

    // Convergent envelope
    let cp  = conv.max(0.0);
    let cp2 = cp * ss(0.04, 0.15, cp);

    // Divergent envelope
    let abs_conv = conv.abs();
    let dp  = (-conv).max(0.0) * ss(0.04, 0.15, abs_conv);

    // Transform envelope
    let abs_sh = sh.abs();
    let tw  = abs_sh * ss(0.10, 0.30, abs_sh) * (1.0 - ss(0.05, 0.15, conv.abs()));

    // No cross-boundary ramp (conv/shear from blurred fields)
    let cp2r = cp2;
    let dpr  = dp;

    // Continuous regime weights (0=oceanic, 1=continental)
    let w_mine  = ss(-0.10, 0.10, q.crust_dist);
    let w_other = ss(-0.10, 0.10, q.other_crust_dist);
    let cord_scale_factor = w_mine;

    let mut relief = 0.0f64;

    if cp2r > 0.0 {
        let w_oc = (1.0 - w_mine) * w_other;
        let w_co = w_mine * (1.0 - w_other);
        let w_oo = (1.0 - w_mine) * (1.0 - w_other);
        let w_cc = w_mine * w_other;

        // OC: Mariana Trench
        relief += w_oc * (-BR_TRENCH_DEPTH * cp2r * gauss(d, 0.0, BR_TRENCH_WIDTH)
                         + BR_FLEX_BULGE  * cp2r * gauss(d, BR_FLEX_POS, BR_FLEX_WIDTH));

        // CO: Andes / Cascades
        let ridge_factor = BR_CORD_RIDGE_BASE + BR_CORD_RIDGE * ridged_at(dir, 6.0, 4);
        relief += w_co * (BR_CORD_HEIGHT  * cp2r * gauss(d, BR_CORD_POS, BR_CORD_WIDTH) * ridge_factor * cord_scale_factor
                         - BR_SHELF_DIP  * cp2r * gauss(d, 0.004, BR_SHELF_WIDTH));

        // OO: Japan arc
        let oo_subduct = 0.5 + (if own.id < other.id { 1.0 } else { 0.0 } - 0.5) * side_ramp;
        let arc_ridged = (ridged_at(dir, 9.0, 4) - BR_ARC_RIDGE_THRESH).max(0.0) * BR_ARC_RIDGE_SCALE;
        // Sea-level emergence clamp (mirrors ki): cone tip can only just breach where
        // the underlying base is near/below sea level. Keeps island arcs submarine
        // except for the strongest cones.
        // ponytail: the `arcClusterGate` along-strike envelope from ki is a separate
        // drift not flagged for this phase; left out to keep this diff scoped.
        const SEA_MARGIN: f64 = 0.02;
        const SEA_ISLAND_HEADROOM: f64 = 0.05;
        let arc_relief = (BR_ARC_HEIGHT * cp2r * gauss(d, BR_ARC_POS, BR_ARC_WIDTH) * arc_ridged)
            .min((-base - SEA_MARGIN).max(0.0) + SEA_ISLAND_HEADROOM);
        relief += w_oo * (
              oo_subduct      * (-BR_OO_TRENCH_DEPTH * cp2r * gauss(d, 0.0, BR_OO_TRENCH_WIDTH))
            + (1.0 - oo_subduct) * arc_relief
        );

        // CC: Himalaya / Alps collision belt
        let belt_ridged = BR_BELT_RIDGE_BASE + BR_BELT_RIDGE * ridged_at(dir, 6.0, 5);
        relief += w_cc * (BR_BELT_HEIGHT * cp2r * gauss(d, 0.0, BR_BELT_WIDTH) * belt_ridged * cord_scale_factor);

        // Tibetan Plateau
        let cc_plateau_side = 0.5 + (if own.id > other.id { 1.0 } else { 0.0 } - 0.5) * side_ramp;
        let plateau_mask = (1.0 - ss(BR_PLATEAU_INNER, BR_PLATEAU_OUTER, d))
                          * ss(BR_PLATEAU_NEAR, BR_PLATEAU_FAR, d);
        relief += w_cc * cc_plateau_side * (BR_PLATEAU_HEIGHT * cp2r * plateau_mask * cord_scale_factor);
    }

    if dpr > 0.0 {
        // Mid-ocean ridge — ridgeCap suppresses emergence so it stays submarine.
        let ridge_cap = 1.0 - ss(-0.10, -0.02, base);
        relief += (1.0 - w_mine) * ridge_cap * (BR_RIDGE_HEIGHT * dpr * gauss(d, 0.0, BR_RIDGE_WIDTH)
                                   - BR_NOTCH_DEPTH  * dpr * gauss(d, 0.0, BR_NOTCH_WIDTH));
        // East African Rift — shoulderSuppress damps the rift shoulder on high base.
        let shoulder_suppress = 1.0 - ss(0.05, 0.25, base);
        relief += w_mine * (BR_RIFT_SHOULDER * shoulder_suppress * dpr * gauss(d, BR_RIFT_SHOULDER_POS, BR_RIFT_SHOULDER_WIDTH)
                          - BR_GRABEN_DEPTH  * dpr * gauss(d, 0.0, BR_GRABEN_WIDTH));
    }

    if tw > 0.0 {
        relief += -BR_SCARP_DEPTH  * tw * gauss(d, 0.0, BR_SCARP_WIDTH)
               + BR_RIDGE2_HEIGHT * tw * gauss(d, BR_RIDGE2_POS, BR_RIDGE2_WIDTH)
                                       * ridged_at(dir, 11.0, 3);
    }

    relief
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Same seed must produce identical plate count, plate ids, and query results.
    #[test]
    fn tectonics_deterministic() {
        let t1 = Tectonics::new(0xDEAD_BEEF, 8, 1.0, 4, 1.0, 0.5);
        let t2 = Tectonics::new(0xDEAD_BEEF, 8, 1.0, 4, 1.0, 0.5);

        assert_eq!(t1.plate_count(), t2.plate_count(),
            "plate count differs between identical seeds");

        // Query a handful of directions and compare each field.
        let dirs = [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.577_350_3, 0.577_350_3, 0.577_350_3),
            DVec3::new(0.0, -1.0, 0.0),
        ];
        for &dir in &dirs {
            let q1 = t1.query(dir.normalize());
            let q2 = t2.query(dir.normalize());
            assert_eq!(q1.plate_id,      q2.plate_id,      "plate_id differs at {:?}", dir);
            assert_eq!(q1.neighbor_id,   q2.neighbor_id,   "neighbor_id differs at {:?}", dir);
            assert_eq!(q1.boundary_dist.to_bits(), q2.boundary_dist.to_bits(),
                "boundary_dist differs at {:?}", dir);
            assert_eq!(q1.convergence.to_bits(), q2.convergence.to_bits(),
                "convergence differs at {:?}", dir);
            assert_eq!(q1.shear.to_bits(), q2.shear.to_bits(),
                "shear differs at {:?}", dir);
        }
    }

    /// Different seeds must produce at least one differing plate id somewhere.
    #[test]
    fn different_seeds_diverge() {
        let t1 = Tectonics::new(1, 8, 1.0, 4, 1.0, 0.5);
        let t2 = Tectonics::new(2, 8, 1.0, 4, 1.0, 0.5);
        // Check a grid of 20 directions
        let mut any_diff = false;
        for i in 0..20 {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / 20.0;
            let phi   = 2.0 * std::f64::consts::PI * (i as f64 * 0.618_033_988_75) % (2.0 * std::f64::consts::PI);
            let dir = DVec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()).normalize();
            let q1 = t1.query(dir);
            let q2 = t2.query(dir);
            if q1.plate_id != q2.plate_id { any_diff = true; break; }
        }
        assert!(any_diff, "different seeds produced identical plate ids for all test directions");
    }

    /// Every direction on the sphere must map to a valid plate id.
    #[test]
    fn every_direction_gets_valid_plate_id() {
        let t = Tectonics::new(42, 12, 1.0, 4, 1.0, 0.5);
        let n = t.plate_count();
        assert!(n > 0, "no plates generated");

        // Sample 100 directions across the sphere.
        for i in 0..100usize {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / 100.0;
            let phi   = 2.0 * std::f64::consts::PI * (i as f64 * 0.618_033_988_75) % (2.0 * std::f64::consts::PI);
            let dir = DVec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()).normalize();
            let q = t.query(dir);
            assert!(q.plate_id < n,
                "plate_id {} out of range [0, {}) at i={}", q.plate_id, n, i);
            assert!(q.neighbor_id < n,
                "neighbor_id {} out of range [0, {}) at i={}", q.neighbor_id, n, i);
        }
    }

    /// boundary_dist must be >= 0 for every queried point.
    #[test]
    fn boundary_dist_non_negative() {
        let t = Tectonics::new(7, 10, 1.0, 4, 1.0, 0.5);
        for i in 0..50usize {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / 50.0;
            let phi   = 2.0 * std::f64::consts::PI * (i as f64 * 0.381_966_011_25) % (2.0 * std::f64::consts::PI);
            let dir = DVec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()).normalize();
            let q = t.query(dir);
            assert!(q.boundary_dist >= 0.0 && q.boundary_dist.is_finite(),
                "boundary_dist {} is invalid at i={}", q.boundary_dist, i);
        }
    }

    /// Non-tectonic path (plate_count=0) must compile and return plate_id=0.
    #[test]
    fn non_tectonic_path_compiles() {
        let t = Tectonics::new(99, 0, 1.0, 0, 1.0, 0.5);
        assert_eq!(t.plate_count(), 1, "non-tectonic path should produce exactly 1 plate");
        let q = t.query(DVec3::new(1.0, 0.0, 0.0));
        assert_eq!(q.plate_id, 0);
        // dist_field stores f32::consts::PI; sample_smooth returns it as f64.
        assert!((q.boundary_dist - std::f64::consts::PI).abs() < 1.0e-6,
            "boundary_dist {} should be ≈ PI for non-tectonic path", q.boundary_dist);
    }

    // ---------------------------------------------------------------------------
    // Phase 2 tests
    // ---------------------------------------------------------------------------

    /// crust_dist and paleo_dist are now real data — same seed → identical values.
    #[test]
    fn phase2_crust_paleo_deterministic() {
        let t1 = Tectonics::new(0xCAFE_BABE, 8, 1.0, 4, 1.0, 0.5);
        let t2 = Tectonics::new(0xCAFE_BABE, 8, 1.0, 4, 1.0, 0.5);

        let dirs = [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.577_350_3, 0.577_350_3, 0.577_350_3),
            DVec3::new(-0.5, 0.3, 0.8).normalize(),
            DVec3::new(0.0, -1.0, 0.0),
        ];
        for &dir in &dirs {
            let q1 = t1.query(dir.normalize());
            let q2 = t2.query(dir.normalize());
            assert_eq!(q1.crust_dist.to_bits(), q2.crust_dist.to_bits(),
                "crust_dist differs at {:?}", dir);
            assert_eq!(q1.paleo_dist.to_bits(), q2.paleo_dist.to_bits(),
                "paleo_dist differs at {:?}", dir);
            assert_eq!(q1.other_crust_dist.to_bits(), q2.other_crust_dist.to_bits(),
                "other_crust_dist differs at {:?}", dir);
        }
    }

    /// crust_dist and paleo_dist must be finite everywhere.
    #[test]
    fn phase2_fields_finite() {
        let t = Tectonics::new(0xF00D, 8, 1.0, 4, 1.0, 0.5);
        for i in 0..50usize {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / 50.0;
            let phi   = 2.0 * std::f64::consts::PI * (i as f64 * 0.618_033_988_75) % (2.0 * std::f64::consts::PI);
            let dir = DVec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()).normalize();
            let q = t.query(dir);
            assert!(q.crust_dist.is_finite(),       "crust_dist not finite at i={}", i);
            assert!(q.paleo_dist.is_finite(),        "paleo_dist not finite at i={}", i);
            assert!(q.other_crust_dist.is_finite(),  "other_crust_dist not finite at i={}", i);
            assert!(q.paleo_dist >= 0.0,             "paleo_dist negative at i={}", i);
        }
    }

    /// boundary_relief must be finite and in reasonable range.
    #[test]
    fn boundary_relief_finite_and_bounded() {
        use crate::noise::ridged;
        let t = Tectonics::new(0xABCD_1234, 8, 1.0, 4, 1.0, 0.5);
        let ridged_noise = crate::noise::Noise3D::new(42);

        for i in 0..50usize {
            let theta = std::f64::consts::PI * (i as f64 + 0.5) / 50.0;
            let phi   = 2.0 * std::f64::consts::PI * (i as f64 * 0.618_033_988_75) % (2.0 * std::f64::consts::PI);
            let dir = DVec3::new(theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()).normalize();
            let q = t.query(dir);
            let r = boundary_relief(&q, &t.plates, dir, &|d, freq, oct| {
                ridged(&ridged_noise, d.x, d.y, d.z, oct, freq, 2.0, 0.5)
            }, 0.0);
            assert!(r.is_finite(), "boundary_relief not finite at i={}", i);
            assert!(r > -3.0 && r < 3.0, "boundary_relief {} out of plausible range at i={}", r, i);
        }
    }

    /// boundary_relief must be deterministic for same seed.
    #[test]
    fn boundary_relief_deterministic() {
        use crate::noise::ridged;
        let t1 = Tectonics::new(0xBEEF, 8, 1.0, 4, 1.0, 0.5);
        let t2 = Tectonics::new(0xBEEF, 8, 1.0, 4, 1.0, 0.5);
        let ridged_noise = crate::noise::Noise3D::new(77);

        let dirs = [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(-0.3, 0.5, 0.8).normalize(),
        ];
        for &dir in &dirs {
            let q1 = t1.query(dir.normalize());
            let q2 = t2.query(dir.normalize());
            let r1 = boundary_relief(&q1, &t1.plates, dir, &|d, freq, oct| {
                ridged(&ridged_noise, d.x, d.y, d.z, oct, freq, 2.0, 0.5)
            }, 0.0);
            let r2 = boundary_relief(&q2, &t2.plates, dir, &|d, freq, oct| {
                ridged(&ridged_noise, d.x, d.y, d.z, oct, freq, 2.0, 0.5)
            }, 0.0);
            assert_eq!(r1.to_bits(), r2.to_bits(),
                "boundary_relief differs between identical seeds at {:?}", dir);
        }
    }

    // ---------------------------------------------------------------------------
    // Phase 3 tests
    // ---------------------------------------------------------------------------

    /// Confirm the full public surface is present (no todo!/unimplemented!).
    /// Simply calling all public methods constitutes the check.
    #[test]
    fn public_surface_complete() {
        let t = Tectonics::new(0x1111, 6, 1.2, 3, 0.8, 0.5);
        let dir = DVec3::new(1.0, 0.0, 0.0);

        // new + query + volcano_elevation
        let q = t.query(dir);
        let _ = t.volcano_elevation(dir);

        // boundary_relief
        use crate::noise::ridged;
        let rn = crate::noise::Noise3D::new(99);
        let _ = boundary_relief(&q, &t.plates, dir, &|d, freq, oct| {
            ridged(&rn, d.x, d.y, d.z, oct, freq, 2.0, 0.5)
        }, 0.0);
    }

    /// volcano_elevation must be deterministic and finite.
    #[test]
    fn volcano_elevation_deterministic() {
        let t1 = Tectonics::new(0x1234, 8, 1.0, 4, 1.0, 0.5);
        let t2 = Tectonics::new(0x1234, 8, 1.0, 4, 1.0, 0.5);

        let dirs = [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.577_350_3, 0.577_350_3, 0.577_350_3),
            DVec3::new(-0.5, 0.1, 0.8).normalize(),
        ];
        for &dir in &dirs {
            let e1 = t1.volcano_elevation(dir.normalize());
            let e2 = t2.volcano_elevation(dir.normalize());
            assert_eq!(e1.to_bits(), e2.to_bits(),
                "volcano_elevation differs at {:?}", dir);
            assert!(e1.is_finite(), "volcano_elevation not finite at {:?}", dir);
            assert!(e1 >= 0.0, "volcano_elevation negative at {:?}", dir);
        }
    }
}

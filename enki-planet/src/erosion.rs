//! Global stream-power erosion field — the **B-path** (iterated uplift+erode loop).
//!
//! Pure port of the `if (opts.upliftForcing)` branch in `ki/src/planet/erosion.ts`
//! (lines ~438-823 + the query samplers ~1605-1869): `Erosion::new` bakes a
//! 256²×6 cubemap once, after which the struct is sampler-only.
//!
//! INTERNAL determinism only (not byte-match vs ki): we replicate ki's algorithm,
//! iteration order, and tie-breaks so enki's two renderers agree on `height()`.
//!
//! SKIPPED (dead on the shipping B-path): the frozen-flow else-branch, the
//! fan/floodplain/delta deposit classifier (Step 5b), and temperature-gating.
//! `depositEnv` is therefore all-zero and not stored.
//!
//! Pipeline per `bSteps` outer iteration:
//!   a. inject uplift on land cells (`workH += bUpliftRate * upliftForcing`)
//!   b. priority-flood (Barnes 2014) ε-fill → lakeLevel
//!   c. sort land cells by Hf descending (idx-descending tie-break)
//!   d. MFD routing (`slope^MFD_P`), accumulate world-tangent flow dir
//!   e. discharge accumulation upstream→downstream
//!   f. one incision+deposition sweep (Davy–Lague)
//!   g. 3 thermal-diffusion passes (talus collapse)
//! After the loop: lake flatten, compose `bDelta`, log-normalize flowAccum,
//! seam-blur (delta 3, acc 4, flowDir 4 — flowDir NOT re-normalized).

use glam::DVec3;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::cubemap::{neighbor_texel, sample_smooth, texel_index, texel_to_dir};
use crate::noise::Noise3D;
use crate::tectonics::Tectonics;

// ---------------------------------------------------------------------------
// Tuning constants (ki erosion.ts; ported VERBATIM at reference scale)
// ---------------------------------------------------------------------------

/// Fixed erosion cubemap resolution (`deriveErosionRes` is fixed 256 in ki).
const EROSION_RES: usize = 256;
/// LOD level at which the base heightFn is sampled for the H0 seed grid.
const EROSION_BAKE_LEVEL: u32 = 5;
/// Sentinel for "not a lake cell" in lakeLevel.
const LAKE_SENTINEL: f32 = -999.0;

/// Reference height scale (= ki HEIGHT_SCALE_REF; enki stays at reference scale).
const B_HEIGHT_SCALE: f64 = 1200.0;

const DEFAULT_K0: f64 = 0.35; // incision rate
const DEFAULT_M_EXP: f64 = 0.45; // discharge exponent
const DEFAULT_N_EXP: f64 = 1.0; // slope exponent
const DEFAULT_TALUS: f64 = 0.03; // thermal talus angle (normalized height)
const DEFAULT_KW_MIN: f64 = 0.05; // min rain weight

/// Deposition rate coefficient G (ki DEFAULT_DEPOSITION_G = 1.5 — but the B-path
/// Planet caller passes 3.0; the phase spec pins G=3.0, so that is our default).
const DEFAULT_DEPOSITION_G: f64 = 3.0;
const EPS_Q: f64 = 1e-10; // guard divisor in deposition term

const MFD_P: i32 = 6; // MFD slope exponent

/// Hardness → erodibility contrast: `1 + STR*(0.5 - h)`.
const HARD_ERODIBILITY_STR: f64 = 1.2;
/// Hardness → talus contrast: `talus*(1 + STR*(h - 0.5))`.
const HARD_TALUS_STR: f64 = 0.6;

/// Barnes ε-fill per-pop increment (keyed to flood ORDER, not spatial index).
const FILL_EPS: f64 = 1e-9;
/// Minimum meaningful flood raise (normalized) to count as a real lake cell.
const LAKE_MIN_DEPTH: f64 = 0.002;

// B-path specifics
const B_STEPS_DEFAULT: usize = 30;
/// Uplift magnitude per step in metres (phase-locked at 40 directly).
const B_UPLIFT_RATE_DEFAULT: f64 = 40.0;
const DT_B: f64 = 0.005;
/// Per-cell per-iteration height-change cap (metres): 0.02 * B_HEIGHT_SCALE.
const DH_CLAMP_M: f64 = 0.02 * B_HEIGHT_SCALE;
/// Final erosion delta clamp (normalized).
const B_DELTA_MAX: f64 = 0.55;

// Query-time domain-warp (stream 10; stream 9 reserved first).
const WARP_AMP: f64 = 0.03;
const WARP_FREQ: f64 = 1.8;
const WARP_OCT: u32 = 2;

// ---------------------------------------------------------------------------
// PRNG — deriveSeed mirrors climate.ts / erosion.ts (stream 9, 10)
// ---------------------------------------------------------------------------

fn splitmix32_step(a: u32) -> u32 {
    let a = a.wrapping_add(0x9e37_79b9);
    let mut t = a ^ (a >> 16);
    t = t.wrapping_mul(0x21f0_aaad);
    t = t ^ (t >> 15);
    t = t.wrapping_mul(0x735a_2d97);
    t ^ (t >> 15)
}

/// Erosion stream ids: 9 = reserved (kept so stream-10 derivation stays stable),
/// 10 = query-time domain-warp noise.
fn derive_seed(master_seed: u32, stream: u32) -> u32 {
    let s = master_seed ^ (stream.wrapping_add(1).wrapping_mul(0xdead_beef));
    splitmix32_step(s)
}

// ---------------------------------------------------------------------------
// MinHeap — BinaryHeap (max-heap) + Reverse-by-(elev ASC, idx ASC) wrapper.
// `clear()` keeps the backing allocation (matches ki's clear()-without-realloc).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct HeapItem {
    elev: f32,
    idx: u32,
}

impl PartialEq for HeapItem {
    fn eq(&self, other: &Self) -> bool {
        self.elev == other.elev && self.idx == other.idx
    }
}
impl Eq for HeapItem {}

impl Ord for HeapItem {
    /// Reverse order so `BinaryHeap` (max-heap) pops the (elev ASC, idx ASC)
    /// minimum. Ties on elev break by idx ascending — exactly ki's `_lt`.
    fn cmp(&self, other: &Self) -> Ordering {
        // We want self < other (heap "greater") when (elev,idx) is SMALLER.
        // BinaryHeap pops the greatest, so invert: greater == smaller key.
        match other.elev.partial_cmp(&self.elev) {
            Some(Ordering::Equal) | None => other.idx.cmp(&self.idx),
            Some(o) => o,
        }
    }
}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Bake-time inputs. `height_fn(dir, level)` returns normalized height in
/// roughly `[-1,1]`; `moisture_fn(dir, h_norm)` returns precipitation in `[0,1]`.
/// `ocean_coverage` is the sea-level percentile. When `tectonics` is `Some`,
/// per-cell hardness modulates erodibility + talus and an `uplift_forcing` field
/// is built from `uplift_at` × `smoothstep(crust_dist)`.
pub struct ErosionOpts<'a, H, M>
where
    H: Fn(DVec3, u32) -> f64,
    M: Fn(DVec3, f64) -> f64,
{
    pub seed: u32,
    pub height_fn: H,
    pub moisture_fn: M,
    pub ocean_coverage: f64,
    /// Optional baked tectonics for hardness + uplift forcing.
    pub tectonics: Option<&'a Tectonics>,
    /// Grid resolution override (None → `EROSION_RES`). Used by tests to bake a
    /// tiny field fast; production always passes None.
    pub res: Option<usize>,
    /// B-path uplift+erode iteration count override (None → `B_STEPS_DEFAULT`).
    /// Used by tests for a fast, low-step bake; production always passes None.
    pub b_steps: Option<usize>,
}

// ---------------------------------------------------------------------------
// Erosion struct
// ---------------------------------------------------------------------------

/// Baked erosion field: sampler-only after construction.
pub struct Erosion {
    res: usize,
    erosion_delta: Vec<f32>,
    flow_accum: Vec<f32>,
    flow_dir_x: Vec<f32>,
    flow_dir_y: Vec<f32>,
    flow_dir_z: Vec<f32>,
    lake_level: Vec<f32>,
    lake_mask: Vec<f32>,
    warp_noise: Noise3D,
}

impl Erosion {
    // -----------------------------------------------------------------------
    // Construction (B-path bake)
    // -----------------------------------------------------------------------

    /// Bake the erosion field. Always runs the B-path (iterated uplift+erode).
    pub fn new<H, M>(opts: ErosionOpts<H, M>) -> Self
    where
        H: Fn(DVec3, u32) -> f64,
        M: Fn(DVec3, f64) -> f64,
    {
        let res = opts.res.unwrap_or(EROSION_RES);
        let k0 = DEFAULT_K0;
        let m_exp = DEFAULT_M_EXP;
        let n_exp = DEFAULT_N_EXP;
        let talus = DEFAULT_TALUS;
        let kw_min = DEFAULT_KW_MIN;
        let deposition_g = DEFAULT_DEPOSITION_G;
        let b_steps = opts.b_steps.unwrap_or(B_STEPS_DEFAULT);
        let b_uplift_rate = B_UPLIFT_RATE_DEFAULT;

        // Stream 9 reserved (touch so stream-10 derivation stays stable), 10 = warp.
        let _ = derive_seed(opts.seed, 9);
        let warp_seed = derive_seed(opts.seed, 10);
        let warp_noise = Noise3D::new(warp_seed);

        let n = 6 * res * res;

        // Constants in metre-space.
        let talus_m = talus * B_HEIGHT_SCALE; // thermal talus in metres
        let lake_min_m = LAKE_MIN_DEPTH * B_HEIGHT_SCALE; // minimum lake depth (m)

        // -------------------------------------------------------------------
        // Step 0 — build upliftForcing[c] from tectonics (Planet.ts:1690-1711).
        //   uplift_forcing[c] = max(0, uplift_at(dir)) * smoothstep(-0.05,0.15,crust_dist)
        // -------------------------------------------------------------------
        let mut uplift_forcing = vec![0.0f32; n];
        if let Some(tect) = opts.tectonics {
            for face in 0..6usize {
                for y in 0..res {
                    for x in 0..res {
                        let c = texel_index(face, x, y, res);
                        let dir = texel_to_dir(face, x, y, res);
                        let u = tect.uplift_at(dir).max(0.0);
                        let cd = tect.query(dir).crust_dist;
                        let gate = smoothstep(-0.05, 0.15, cd);
                        uplift_forcing[c] = (u * gate) as f32;
                    }
                }
            }
        }

        // -------------------------------------------------------------------
        // Step 1 — H0 grid (metres) + rainfall
        // -------------------------------------------------------------------
        let mut h0 = vec![0.0f32; n];
        let mut rain = vec![0.0f32; n];
        for face in 0..6usize {
            for y in 0..res {
                for x in 0..res {
                    let c = texel_index(face, x, y, res);
                    let dir = texel_to_dir(face, x, y, res);
                    let h_norm = (opts.height_fn)(dir, EROSION_BAKE_LEVEL);
                    h0[c] = (h_norm * B_HEIGHT_SCALE) as f32;
                    let moisture = (opts.moisture_fn)(dir, h_norm);
                    rain[c] = kw_min.max(moisture) as f32;
                }
            }
        }

        // Per-cell hardness modulation (metre-space).
        let mut erodibility_b = vec![1.0f32; n];
        let mut talus_scale_b = vec![talus_m as f32; n];
        if let Some(tect) = opts.tectonics {
            for face in 0..6usize {
                for y in 0..res {
                    for x in 0..res {
                        let c = texel_index(face, x, y, res);
                        let dir = texel_to_dir(face, x, y, res);
                        let h = tect.hardness_at(dir);
                        erodibility_b[c] = (1.0 + HARD_ERODIBILITY_STR * (0.5 - h)) as f32;
                        talus_scale_b[c] = (talus_m * (1.0 + HARD_TALUS_STR * (h - 0.5))) as f32;
                    }
                }
            }
        }

        // Fixed sea level + ocean mask from seed H0 (metres).
        // ki: provSeaLevelM = sortedB[min(N-1, floor(oceanCoverage*N))].
        let prov_sea_level_m = {
            let mut sorted = h0.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
            let i = ((opts.ocean_coverage * n as f64).floor() as usize).min(n - 1);
            sorted[i]
        };

        let mut is_ocean = vec![0u8; n];
        for c in 0..n {
            if h0[c] < prov_sea_level_m {
                is_ocean[c] = 1;
            }
        }

        // -------------------------------------------------------------------
        // Step 2 — 8-neighbor table (cross-face)
        // -------------------------------------------------------------------
        let neigh: Vec<Vec<u32>> = (0..n)
            .map(|c| {
                let (face, x, y) = cell_to_face_xy(c, res);
                let mut ns: Vec<u32> = Vec::with_capacity(8);
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        if dx == 0 && dy == 0 {
                            continue;
                        }
                        let t = neighbor_texel(face, x, y, dx, dy, res);
                        let nc = texel_index(t.face, t.x, t.y, res);
                        if nc != c {
                            ns.push(nc as u32);
                        }
                    }
                }
                ns
            })
            .collect();

        // -------------------------------------------------------------------
        // Working state (allocated once, reset/rebuilt each B-step)
        // -------------------------------------------------------------------
        let mut work_h = h0.clone(); // metres
        let mut lake_level = vec![LAKE_SENTINEL; n];
        let mut lake_mask = vec![0.0f32; n];
        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(n);

        // mfd[c] = list of (neighbor, weight); rebuilt each step.
        let mut mfd: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        let mut flow_dir_x = vec![0.0f32; n];
        let mut flow_dir_y = vec![0.0f32; n];
        let mut flow_dir_z = vec![0.0f32; n];
        let mut q = vec![0.0f64; n];
        let mut hf = vec![0.0f32; n];

        let mut buf_a = vec![0.0f32; n];
        let mut buf_b = vec![0.0f32; n];
        let mut qs_b = vec![0.0f64; n];
        let mut land_cells: Vec<u32> = Vec::with_capacity(n);

        for _step in 0..b_steps {
            // a. inject uplift on land
            for c in 0..n {
                if is_ocean[c] == 0 {
                    work_h[c] += (b_uplift_rate * uplift_forcing[c] as f64) as f32;
                }
            }

            // b+c. priority-flood (Barnes) on current workH
            hf.copy_from_slice(&work_h);
            lake_level.iter_mut().for_each(|v| *v = LAKE_SENTINEL);
            let mut in_queue = vec![0u8; n];
            let mut processed = vec![0u8; n];
            heap.clear();
            for c in 0..n {
                if is_ocean[c] == 1 {
                    heap.push(HeapItem { elev: work_h[c], idx: c as u32 });
                    in_queue[c] = 1;
                }
            }
            let mut flood_counter: f64 = 0.0;
            while let Some(item) = heap.pop() {
                let idx = item.idx as usize;
                if processed[idx] == 1 {
                    continue;
                }
                processed[idx] = 1;
                flood_counter += 1.0;
                let elev = item.elev as f64;
                for &nn in &neigh[idx] {
                    let nu = nn as usize;
                    if processed[nu] == 1 {
                        continue;
                    }
                    let new_elev = (hf[nu] as f64).max(elev + FILL_EPS * flood_counter);
                    if new_elev > hf[nu] as f64 {
                        if new_elev - work_h[nu] as f64 > lake_min_m {
                            lake_level[nu] = new_elev as f32;
                        }
                        hf[nu] = new_elev as f32;
                    }
                    if in_queue[nu] == 0 {
                        heap.push(HeapItem { elev: hf[nu], idx: nn });
                        in_queue[nu] = 1;
                    }
                }
            }
            for c in 0..n {
                lake_mask[c] = if lake_level[c] != LAKE_SENTINEL { 1.0 } else { 0.0 };
            }

            // d. re-sort land cells by Hf descending (idx descending tie-break)
            land_cells.clear();
            for c in 0..n {
                if is_ocean[c] == 0 {
                    land_cells.push(c as u32);
                }
            }
            land_cells.sort_by(|&a, &b| {
                let ha = hf[a as usize];
                let hb = hf[b as usize];
                match hb.partial_cmp(&ha) {
                    Some(Ordering::Equal) | None => b.cmp(&a),
                    Some(o) => o,
                }
            });

            // e. re-route MFD on current Hf
            flow_dir_x.iter_mut().for_each(|v| *v = 0.0);
            flow_dir_y.iter_mut().for_each(|v| *v = 0.0);
            flow_dir_z.iter_mut().for_each(|v| *v = 0.0);
            for c in 0..n {
                mfd[c].clear();
                if is_ocean[c] == 1 {
                    continue;
                }
                let (fc, cx, cy) = cell_to_face_xy(c, res);
                let dir_c = texel_to_dir(fc, cx, cy, res);

                // Collect strictly-lower neighbors (lex (Hf, idx)).
                let mut lowers: Vec<(u32, f64)> = Vec::new();
                let mut w_sum = 0.0f64;
                let hfc = hf[c];
                for &nn in &neigh[c] {
                    let nu = nn as usize;
                    let hfn = hf[nu];
                    if hfn < hfc || (hfn == hfc && nu < c) {
                        let slope = (hfc - hfn) as f64;
                        let raw_w = slope.powi(MFD_P);
                        lowers.push((nn, raw_w));
                        w_sum += raw_w;
                    }
                }
                if lowers.is_empty() || w_sum < 1e-30 {
                    continue;
                }

                let mut fdx = 0.0;
                let mut fdy = 0.0;
                let mut fdz = 0.0;
                let mut cell_mfd: Vec<(u32, f64)> = Vec::with_capacity(lowers.len());
                for &(nn, raw_w) in &lowers {
                    let w = raw_w / w_sum;
                    cell_mfd.push((nn, w));
                    let (nf, nx, ny) = cell_to_face_xy(nn as usize, res);
                    let dir_n = texel_to_dir(nf, nx, ny, res);
                    let diff = dir_n - dir_c;
                    let dot = diff.dot(dir_c);
                    let t = diff - dir_c * dot;
                    let t_len = t.length();
                    if t_len > 1e-8 {
                        let inv = w / t_len;
                        fdx += inv * t.x;
                        fdy += inv * t.y;
                        fdz += inv * t.z;
                    }
                }
                mfd[c] = cell_mfd;
                let fd_len = (fdx * fdx + fdy * fdy + fdz * fdz).sqrt();
                if fd_len > 1e-8 {
                    flow_dir_x[c] = (fdx / fd_len) as f32;
                    flow_dir_y[c] = (fdy / fd_len) as f32;
                    flow_dir_z[c] = (fdz / fd_len) as f32;
                }
            }

            // f. accumulate discharge
            for c in 0..n {
                q[c] = rain[c] as f64;
            }
            for &c in &land_cells {
                let cu = c as usize;
                let qc = q[cu];
                for &(nn, w) in &mfd[cu] {
                    let nu = nn as usize;
                    if is_ocean[nu] == 0 {
                        q[nu] += qc * w;
                    }
                }
            }

            // g. one incision+deposition sweep (Davy–Lague, metre space)
            buf_a.copy_from_slice(&hf);
            buf_b.copy_from_slice(&hf);
            qs_b.iter_mut().for_each(|v| *v = 0.0);
            let slope_floor = 1e-8 * B_HEIGHT_SCALE;
            for &c in &land_cells {
                let cu = c as usize;
                if is_ocean[cu] == 1 {
                    continue;
                }
                let mut slope = 0.0f64;
                for &(nn, w) in &mfd[cu] {
                    slope += w * ((buf_a[cu] - buf_a[nn as usize]) as f64).max(0.0);
                }
                let eros = DT_B
                    * k0
                    * erodibility_b[cu] as f64
                    * q[cu].powf(m_exp)
                    * slope.max(slope_floor).powf(n_exp);
                let dep = DT_B * deposition_g * qs_b[cu] / q[cu].max(EPS_Q);
                let mut dh = dep - eros;
                if dh > DH_CLAMP_M {
                    dh = DH_CLAMP_M;
                }
                if dh < -DH_CLAMP_M {
                    dh = -DH_CLAMP_M;
                }
                buf_b[cu] = buf_a[cu] + dh as f32;

                let qs_out = qs_b[cu] + (eros - dep);
                let qs_out_c = if qs_out > 0.0 { qs_out } else { 0.0 };
                for &(nn, w) in &mfd[cu] {
                    let nu = nn as usize;
                    if is_ocean[nu] == 0 {
                        qs_b[nu] += qs_out_c * w;
                    }
                }
            }
            work_h.copy_from_slice(&buf_b);

            // h. thermal diffusion — 3 passes per step
            buf_a.copy_from_slice(&work_h);
            for _t_pass in 0..3 {
                buf_b.copy_from_slice(&buf_a);
                for c in 0..n {
                    let talus_cm = talus_scale_b[c] as f64;
                    let bc = buf_a[c] as f64;
                    for &nn in &neigh[c] {
                        let nu = nn as usize;
                        let slope_cn = bc - buf_a[nu] as f64;
                        if slope_cn > talus_cm {
                            let transfer = 0.0005 * (slope_cn - talus_cm) * 0.5;
                            buf_b[c] -= transfer as f32;
                            buf_b[nu] += transfer as f32;
                        }
                    }
                }
                std::mem::swap(&mut buf_a, &mut buf_b);
            }
            work_h.copy_from_slice(&buf_a);
        }

        // i. lake flatten (after loop)
        for c in 0..n {
            if lake_level[c] != LAKE_SENTINEL {
                work_h[c] = lake_level[c];
            }
        }

        // j. compose bDelta = (workH - H0)/B_HEIGHT_SCALE, clamped ±B_DELTA_MAX
        let mut erosion_delta = vec![0.0f32; n];
        for c in 0..n {
            let mut d = (work_h[c] as f64 - h0[c] as f64) / B_HEIGHT_SCALE;
            if d < -B_DELTA_MAX {
                d = -B_DELTA_MAX;
            }
            if d > B_DELTA_MAX {
                d = B_DELTA_MAX;
            }
            erosion_delta[c] = d as f32;
        }

        // k. flowAccum = log-normalized Q (from last routing step)
        let q_max = q.iter().cloned().fold(0.0f64, f64::max);
        let log_q_max = (1.0 + q_max).ln();
        let mut flow_accum = vec![0.0f32; n];
        for c in 0..n {
            let v = (1.0 + q[c]).ln() / if log_q_max != 0.0 { log_q_max } else { 1.0 };
            flow_accum[c] = v.clamp(0.0, 1.0) as f32;
        }

        // l. seam blur (delta 3, acc 4, flowDir 4 — flowDir NOT re-normalized)
        let erosion_delta = blur_field_n(&erosion_delta, res, 3);
        let flow_accum = blur_field_n(&flow_accum, res, 4);
        let flow_dir_x = blur_field_n(&flow_dir_x, res, 4);
        let flow_dir_y = blur_field_n(&flow_dir_y, res, 4);
        let flow_dir_z = blur_field_n(&flow_dir_z, res, 4);

        Erosion {
            res,
            erosion_delta,
            flow_accum,
            flow_dir_x,
            flow_dir_y,
            flow_dir_z,
            lake_level,
            lake_mask,
            warp_noise,
        }
    }

    // -----------------------------------------------------------------------
    // Query methods (mirror ki deltaAt/accAt/flowAt/lakeAt/lakeMaskAt)
    // -----------------------------------------------------------------------

    /// Normalized height delta at `dir`; <0 = incision, >0 = sediment. C1-sampled.
    pub fn delta_at(&self, dir: DVec3) -> f64 {
        sample_c1(&self.erosion_delta, self.warped_dir(dir), self.res)
    }

    /// Normalized log discharge 0..1 at `dir`. C1-sampled.
    pub fn acc_at(&self, dir: DVec3) -> f64 {
        sample_c1(&self.flow_accum, self.warped_dir(dir), self.res)
    }

    /// World-space downhill flow vector at `dir` (RAW bilinear — magnitude is the
    /// coherence signal, NOT re-normalized). C1-sampled per component.
    pub fn flow_at(&self, dir: DVec3) -> DVec3 {
        let dw = self.warped_dir(dir);
        let out = DVec3::new(
            sample_c1(&self.flow_dir_x, dw, self.res),
            sample_c1(&self.flow_dir_y, dw, self.res),
            sample_c1(&self.flow_dir_z, dw, self.res),
        );
        if out.is_finite() {
            out
        } else {
            DVec3::ZERO
        }
    }

    /// Lake spill elevation (metres) at `dir`; LAKE_SENTINEL where not a lake.
    /// PLAIN bilinear (not C1) to avoid smearing the LAKE_SENTINEL sentinel.
    pub fn lake_at(&self, dir: DVec3) -> f64 {
        sample_smooth(&self.lake_level, self.warped_dir(dir), self.res)
    }

    /// Lake presence mask at `dir` ∈ [0,1]. C1-sampled (no sentinel in this field).
    pub fn lake_mask_at(&self, dir: DVec3) -> f64 {
        sample_c1(&self.lake_mask, self.warped_dir(dir), self.res)
    }

    // -----------------------------------------------------------------------
    // Internal — query-time domain warp (stream 10), mirrors ki _warpedDir
    // -----------------------------------------------------------------------

    fn warped_dir(&self, dir: DVec3) -> DVec3 {
        let n = &self.warp_noise;
        let fx = dir.x * WARP_FREQ;
        let fy = dir.y * WARP_FREQ;
        let fz = dir.z * WARP_FREQ;

        // 2-octave fbm, three spatially offset samples (one per axis).
        let mut wx = 0.0;
        let mut wy = 0.0;
        let mut wz = 0.0;
        let mut amp = 1.0;
        let mut freq = 1.0;
        for _ in 0..WARP_OCT {
            wx += amp * n.sample(fx * freq, fy * freq, fz * freq);
            wy += amp * n.sample(fx * freq + 3.7, fy * freq + 2.1, fz * freq + 1.3);
            wz += amp * n.sample(fx * freq + 7.3, fy * freq + 5.9, fz * freq + 4.1);
            amp *= 0.5;
            freq *= 2.0;
        }
        // Normalize fbm output (max amp = 1 + 0.5 = 1.5 for 2 oct), scale by WARP_AMP.
        let max_amp = 1.5;
        wx = (wx / max_amp) * WARP_AMP;
        wy = (wy / max_amp) * WARP_AMP;
        wz = (wz / max_amp) * WARP_AMP;

        // Project warp perpendicular to dir, displace, normalize.
        let w = DVec3::new(wx, wy, wz);
        let dot = w.dot(dir);
        let t = w - dir * dot;
        let r = dir + t;
        let len = r.length();
        if len > 1e-8 {
            r / len
        } else {
            dir
        }
    }
}

// ---------------------------------------------------------------------------
// Module-level helpers
// ---------------------------------------------------------------------------

/// Decode flat cell index → (face, x, y). Mirrors ki `cellToFaceXY`.
#[inline]
fn cell_to_face_xy(c: usize, res: usize) -> (usize, usize, usize) {
    let rr = res * res;
    let face = c / rr;
    let rem = c % rr;
    (face, rem % res, rem / res)
}

#[inline]
fn smoothstep(e0: f64, e1: f64, x: f64) -> f64 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// `n`-pass 3×3 cross-face box blur (uniform 9-tap average). Matches ki's
/// `blurField` (sum of all 9 neighbours / 9, cross-face via `neighbor_texel`).
fn blur_field_n(src: &[f32], res: usize, passes: usize) -> Vec<f32> {
    let mut cur = src.to_vec();
    let mut tmp = vec![0.0f32; src.len()];
    for _ in 0..passes {
        for face in 0..6usize {
            for y in 0..res {
                for x in 0..res {
                    let c = texel_index(face, x, y, res);
                    let mut sum = 0.0f64;
                    let mut count = 0.0f64;
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let nb = neighbor_texel(face, x, y, dx, dy, res);
                            sum += cur[texel_index(nb.face, nb.x, nb.y, res)] as f64;
                            count += 1.0;
                        }
                    }
                    tmp[c] = (sum / count) as f32;
                }
            }
        }
        std::mem::swap(&mut cur, &mut tmp);
    }
    cur
}

/// C1 smoothstep-bilinear cubemap sampler. Identical to `sample_smooth` EXCEPT
/// smoothstep is applied to the fractional texel coords before the lerp.
/// Mirrors ki `_sampleC1`. NOT safe for sentinel-bearing fields (use
/// `sample_smooth` for lakeLevel).
fn sample_c1(data: &[f32], dir: DVec3, res: usize) -> f64 {
    let ax = dir.x.abs();
    let ay = dir.y.abs();
    let az = dir.z.abs();

    let (face, sc, tc, mc): (usize, f64, f64, f64);
    if ax >= ay && ax >= az {
        mc = ax;
        if dir.x > 0.0 {
            face = 0;
            sc = dir.z;
            tc = dir.y;
        } else {
            face = 1;
            sc = -dir.z;
            tc = dir.y;
        }
    } else if ay >= ax && ay >= az {
        mc = ay;
        if dir.y > 0.0 {
            face = 2;
            sc = dir.x;
            tc = dir.z;
        } else {
            face = 3;
            sc = dir.x;
            tc = -dir.z;
        }
    } else {
        mc = az;
        if dir.z > 0.0 {
            face = 4;
            sc = -dir.x;
            tc = dir.y;
        } else {
            face = 5;
            sc = dir.x;
            tc = dir.y;
        }
    }

    let half = res as f64 * 0.5;
    let fx = (sc / mc + 1.0) * half - 0.5;
    let fy = (tc / mc + 1.0) * half - 0.5;

    let x0i = fx.floor() as isize;
    let y0i = fy.floor() as isize;
    // smoothstep the fractional coords for C1 continuity.
    let mut fu = fx - x0i as f64;
    let mut fv = fy - y0i as f64;
    fu = fu * fu * (3.0 - 2.0 * fu);
    fv = fv * fv * (3.0 - 2.0 * fv);
    let x1i = x0i + 1;
    let y1i = y0i + 1;

    let (t00, t10, t01, t11): (f64, f64, f64, f64);
    if x0i >= 0 && y0i >= 0 && x1i < res as isize && y1i < res as isize {
        let x0 = x0i as usize;
        let y0 = y0i as usize;
        let x1 = x1i as usize;
        let y1 = y1i as usize;
        let base = face * res * res;
        t00 = data[base + y0 * res + x0] as f64;
        t10 = data[base + y0 * res + x1] as f64;
        t01 = data[base + y1 * res + x0] as f64;
        t11 = data[base + y1 * res + x1] as f64;
    } else {
        let ax_ = x0i.clamp(0, (res as isize) - 1) as usize;
        let ay_ = y0i.clamp(0, (res as isize) - 1) as usize;
        let ox0 = (x0i - ax_ as isize) as i32;
        let ox1 = (x1i - ax_ as isize) as i32;
        let oy0 = (y0i - ay_ as isize) as i32;
        let oy1 = (y1i - ay_ as isize) as i32;
        let n00 = neighbor_texel(face, ax_, ay_, ox0, oy0, res);
        let n10 = neighbor_texel(face, ax_, ay_, ox1, oy0, res);
        let n01 = neighbor_texel(face, ax_, ay_, ox0, oy1, res);
        let n11 = neighbor_texel(face, ax_, ay_, ox1, oy1, res);
        t00 = data[texel_index(n00.face, n00.x, n00.y, res)] as f64;
        t10 = data[texel_index(n10.face, n10.x, n10.y, res)] as f64;
        t01 = data[texel_index(n01.face, n01.x, n01.y, res)] as f64;
        t11 = data[texel_index(n11.face, n11.x, n11.y, res)] as f64;
    }

    let top = t00 + (t10 - t00) * fu;
    let bot = t01 + (t11 - t01) * fu;
    top + (bot - top) * fv
}

// ---------------------------------------------------------------------------
// Tests — ponytail synthetic-heightfield bake (headless, assert-based)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Tiny synthetic heightfield mirroring enki-nanite tessellate's SineHf.
    // No tectonics / no uplift_forcing.
    fn sine_height(dir: DVec3, _level: u32) -> f64 {
        // Low-amplitude bumpy field in normalized [-1,1]-ish range.
        0.3 * (dir.x * 6.0).sin() * (dir.y * 6.0).cos()
    }
    fn flat_height(_dir: DVec3, _level: u32) -> f64 {
        0.0
    }
    fn uniform_moisture(_dir: DVec3, _h: f64) -> f64 {
        0.5
    }

    fn bake(res: usize, hf: fn(DVec3, u32) -> f64) -> Erosion {
        Erosion::new(ErosionOpts {
            seed: 1234,
            height_fn: hf,
            moisture_fn: uniform_moisture,
            ocean_coverage: 0.3,
            tectonics: None,
            res: Some(res),
            b_steps: Some(4), // small step count: fast, still exercises the loop
        })
    }

    /// (1) A perfectly flat input yields ~zero delta everywhere.
    #[test]
    fn flat_input_zero_delta() {
        let e = bake(16, flat_height);
        for (i, &d) in e.erosion_delta.iter().enumerate() {
            assert!(d.abs() < 1e-4, "flat delta[{i}]={d} not ~0");
        }
    }

    /// (2) Delta is bounded within ±B_DELTA_MAX everywhere.
    #[test]
    fn delta_bounded() {
        let e = bake(16, sine_height);
        for (i, &d) in e.erosion_delta.iter().enumerate() {
            assert!(
                d as f64 >= -B_DELTA_MAX - 1e-6 && d as f64 <= B_DELTA_MAX + 1e-6,
                "delta[{i}]={d} out of ±{B_DELTA_MAX}"
            );
        }
    }

    /// (3) The bake is deterministic — two bakes produce bit-identical arrays.
    #[test]
    fn bake_deterministic() {
        let a = bake(16, sine_height);
        let b = bake(16, sine_height);
        assert_eq!(a.erosion_delta.len(), b.erosion_delta.len());
        for (i, (x, y)) in a.erosion_delta.iter().zip(b.erosion_delta.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "erosion_delta[{i}] diverged");
        }
        for (x, y) in a.flow_accum.iter().zip(b.flow_accum.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        for (x, y) in a.flow_dir_x.iter().zip(b.flow_dir_x.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
        for (x, y) in a.lake_level.iter().zip(b.lake_level.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    /// Samplers return finite values everywhere.
    #[test]
    fn samplers_finite() {
        let e = bake(16, sine_height);
        let dirs = [
            DVec3::X,
            DVec3::NEG_X,
            DVec3::Y,
            DVec3::NEG_Y,
            DVec3::Z,
            DVec3::NEG_Z,
            DVec3::new(1.0, 1.0, 1.0).normalize(),
        ];
        for dir in dirs {
            assert!(e.delta_at(dir).is_finite());
            assert!(e.acc_at(dir).is_finite());
            assert!(e.flow_at(dir).is_finite());
            assert!(e.lake_mask_at(dir).is_finite());
        }
    }
}

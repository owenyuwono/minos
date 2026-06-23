//! River carving via **priority-flood drainage routing** (a Dijkstra/A\*-family
//! priority-queue search) + **flow accumulation** — NO erosion simulation, pure
//! static analysis of the baked heightfield. Produces a proper dendritic network:
//! headwaters on the high ground, tributaries merging into trunks, every river
//! reaching the coast. We carve a V-incision along the channel cells so the un-cut
//! terrain on either side forms the banks. Baked once at `TectonicHeightField::new`;
//! `tect_height` subtracts `incision_at(dir)` (clamped so a land column never carves
//! below sea level).
//!
//! Why not "A\* each spring → nearest sea": that makes every spring beeline to its
//! own closest coast independently, so tributaries never share a trunk → short,
//! fat, disconnected coastal stubs, no branches. Drainage routing instead makes
//! flow CONVERGE down the real valleys, which is what produces branches.
//!
//! Pipeline:
//!   1. Sample routing height + moisture over a cube-sphere grid.
//!   2. **Priority-flood** from the ocean inward (min-heap on spill height): fills
//!      pits and, as a side effect, builds the drainage tree — each land cell drains
//!      to the (lower, nearer-the-sea) cell it was first reached from. Sink-free by
//!      construction, so every cell has a downhill path to the sea.
//!   3. **Flow accumulation**: push each cell's rainfall (∝ moisture) downstream in
//!      reverse spill order → upstream catchment area per cell (trunks accumulate).
//!   4. Rivers = land cells whose accumulation clears a threshold. Depth ∝ the
//!      accumulation (trunks deeper), dilated into banks, negated → incision.

use glam::DVec3;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::cubemap::{neighbor_texel, sample_smooth, texel_index, texel_to_dir};

// --- Tuning (ponytail: tunables, USER-verified on the Height view + on foot) ---
//
// SCALE CONTRACT: terrain is `radius 50_000 m`, `height_scale 1200 m` (main.rs LOD),
// so `meters = normalized · 1200`. The rendered/grounded Nanite mesh is
// `NANITE_BAKE_RES = 1024`/face ≈ 77 m/cell, so the NARROWEST feature the mesh (and
// the player's feet, which ray-cast the drawn triangles) can resolve is ~2 cells ≈
// 150 m — the carve cannot make a river thinner than that as GEOMETRY. The actual
// thin wet thread is the `rivers.rs` ribbon overlay; the carve's job is only the
// wide, gentle, mesh-sample-able VALLEY. (Why the old carve looked like a canyon on
// foot: RES=256 → ~610 m channel, and CARVE_AMP 0.10·1200 = 120 m deep with a 60 m
// floor on EVERY brook. See the river-scale-fix design.)
/// Carving cube-map resolution per face. Bake cost is `6·RES²` routing-height calls.
/// 1024 = NANITE_BAKE_RES, so one carve cell ≈ one mesh cell (~77 m) — the finest a
/// channel can faithfully render. (Below this the carve smears across mesh cells.)
const RES: usize = 1024;
/// `HeightField` LOD for the routing height (broad valleys drive the paths).
const ROUTE_LEVEL: u32 = 6;
/// Rainfall per land cell feeding accumulation = `RAIN_FLOOR + (1-floor)·moisture`,
/// so wet "aquifer" highlands spawn rivers and arid land stays mostly dry, but no
/// land is bone-dry (floor keeps a faint network everywhere).
const RAIN_FLOOR: f64 = 0.15;
/// A land cell becomes a river once this much upstream rainfall drains through it
/// (≈ catchment cells × mean rain). THE density knob — lower = finer tributaries.
/// Res-independent (keys off rainfall, not cell count) so the network is unchanged
/// across RES.
const RIVER_THRESHOLD: f64 = 35.0;
/// Accumulation (above threshold) at which a channel hits full depth — the
/// threshold→TRUNK_FULL band maps a headwater stream up to a major trunk.
const TRUNK_FULL: f64 = 1200.0;
/// Full-trunk valley-floor depth (normalized): 0.040·1200 ≈ 48 m. A major-river
/// gorge on a 50 km planet, not the old 120 m rift.
const CARVE_AMP: f64 = 0.040;
/// Depth-vs-discharge exponent (Leopold–Maddock hydraulic geometry, `d ∝ Q^~0.4`):
/// `depth = CARVE_AMP·t^DEPTH_EXP`, so tributaries cut shallow and only big trunks
/// cut deep (the old `DEPTH_FLOOR` made EVERY stream a 60 m trench).
const DEPTH_EXP: f64 = 0.40;
/// Floor depth for the faintest headwater (normalized): 0.0035·1200 ≈ 4.2 m — a
/// believable brook ditch you can step across, not a canyon.
const MIN_DEPTH: f64 = 0.0035;
/// Valley-flank width in mesh cells the bank ramps outward → a gentle, sample-able
/// valley (the only mesh-resolvable part). 2·~77 m ≈ 150 m flank each side.
const BANK_CELLS: usize = 2;
/// Depth lost per cell of bank ramp: 0.018·1200 ≈ 22 m/cell over ~77 m ≈ 1:3.5 bank,
/// so a 48 m trunk valley rises back to grade in ~2 cells (tight, not a wide dish).
const BANK_FALLOFF: f64 = 0.018;

const SINK: u32 = u32::MAX;
const NB_DX: [i32; 8] = [-1, 0, 1, -1, 1, -1, 0, 1];
const NB_DY: [i32; 8] = [-1, -1, -1, 0, 0, 1, 1, 1];

/// Baked river-valley incision field (≤0 per cell) + a 0..1 river-line mask (for the
/// debug Wetness view). Sampler-only after construction.
pub struct RiverCarve {
    res: usize,
    incision: Vec<f32>,
    river_mask: Vec<f32>,
}

/// Priority-flood node — min-heap by spill height (`BinaryHeap` is max-heap, so `Ord`
/// is reversed). Tie-break on idx for determinism.
#[derive(Clone, Copy, PartialEq)]
struct Node {
    spill: f64,
    idx: u32,
}
impl Eq for Node {}
impl Ord for Node {
    fn cmp(&self, o: &Self) -> Ordering {
        o.spill
            .partial_cmp(&self.spill)
            .unwrap_or(Ordering::Equal)
            .then_with(|| o.idx.cmp(&self.idx))
    }
}
impl PartialOrd for Node {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}

#[inline]
fn cell_to_fxy(c: u32, res: usize) -> (usize, usize, usize) {
    let c = c as usize;
    let rr = res * res;
    (c / rr, (c % rr) % res, (c % rr) / res)
}

impl RiverCarve {
    /// `route_h(dir, level)` is the terrain water routes on; `moisture(dir)` ∈ [0,1].
    pub fn new(route_h: impl Fn(DVec3, u32) -> f64, moisture: impl Fn(DVec3) -> f64) -> Self {
        let res = RES;
        let n = 6 * res * res;

        // 1. Sample routing height + moisture.
        let mut height = vec![0.0f32; n];
        let mut rain = vec![0.0f64; n];
        for face in 0..6usize {
            for y in 0..res {
                for x in 0..res {
                    let c = texel_index(face, x, y, res);
                    let dir = texel_to_dir(face, x, y, res);
                    height[c] = route_h(dir, ROUTE_LEVEL) as f32;
                    rain[c] = RAIN_FLOOR + (1.0 - RAIN_FLOOR) * moisture(dir).clamp(0.0, 1.0);
                }
            }
        }

        let empty = || Self {
            res,
            incision: vec![0.0f32; n],
            river_mask: vec![0.0f32; n],
        };
        if !height.iter().any(|&h| h < 0.0) {
            return empty(); // no ocean → nowhere to drain
        }

        // 2. Priority-flood from the ocean inward. `drains_to[c]` = the cell `c` was
        //    first reached from (lower, nearer the sea) → the drainage tree. `order`
        //    records pop sequence (increasing spill) for the accumulation sweep.
        //    Sink-free by construction: every land cell drains to the sea.
        let mut drains_to = vec![SINK; n];
        let mut visited = vec![false; n];
        let mut order: Vec<u32> = Vec::with_capacity(n);
        let mut heap: BinaryHeap<Node> = BinaryHeap::new();
        for c in 0..n {
            if height[c] < 0.0 {
                visited[c] = true;
                heap.push(Node { spill: height[c] as f64, idx: c as u32 });
            }
        }
        while let Some(Node { spill, idx: c }) = heap.pop() {
            order.push(c);
            let (f, x, y) = cell_to_fxy(c, res);
            for k in 0..8 {
                let nt = neighbor_texel(f, x, y, NB_DX[k], NB_DY[k], res);
                let nc = texel_index(nt.face, nt.x, nt.y, res);
                if !visited[nc] {
                    visited[nc] = true;
                    drains_to[nc] = c; // nc drains toward c (toward the sea)
                    // Spill = max(own height, the level we had to climb to reach here)
                    // → fills pits to their lowest outlet.
                    let s = (height[nc] as f64).max(spill);
                    heap.push(Node { spill: s, idx: nc as u32 });
                }
            }
        }

        // 3. Flow accumulation: push rainfall downstream in REVERSE spill order (high
        //    → low), so every upstream contributor has already added to a cell before
        //    we forward its total. `acc[c]` = upstream catchment rainfall through `c`.
        let mut acc = rain.clone();
        for &c in order.iter().rev() {
            let d = drains_to[c as usize];
            if d != SINK {
                acc[d as usize] += acc[c as usize];
            }
        }

        // 4. Rivers = land cells over threshold. Depth ∝ accumulation (trunks deeper).
        let mut depth = vec![0.0f32; n];
        let mut is_river = vec![false; n];
        for c in 0..n {
            if height[c] >= 0.0 && acc[c] >= RIVER_THRESHOLD {
                is_river[c] = true;
                let t = ((acc[c] - RIVER_THRESHOLD) / (TRUNK_FULL - RIVER_THRESHOLD)).clamp(0.0, 1.0);
                depth[c] = MIN_DEPTH.max(CARVE_AMP * t.powf(DEPTH_EXP)) as f32;
            }
        }

        // 5. Dilate the depth into banks (grayscale dilation keeps the floor, ramps
        //    flanks back to grade) and negate → incision.
        let spread = dilate(&depth, res, BANK_CELLS, BANK_FALLOFF as f32);
        let incision: Vec<f32> = spread.iter().map(|&d| -d).collect();

        // River-line mask (1 on a river cell, fading to its neighbours) for the debug
        // Wetness view — a high-contrast way to SEE the traced network (lines +
        // branches) regardless of how subtle the carved dip is in the height ramp.
        let mask: Vec<f32> = is_river.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
        let river_mask = dilate(&mask, res, 1, 0.5);

        Self { res, incision, river_mask }
    }

    /// Incision (≤0, normalized height) to add to the terrain at `dir`. C0 bilinear.
    pub fn incision_at(&self, dir: DVec3) -> f64 {
        sample_smooth(&self.incision, dir, self.res)
    }

    /// River-line presence 0..1 at `dir` (debug Wetness view). C0 bilinear.
    pub fn river_mask_at(&self, dir: DVec3) -> f64 {
        sample_smooth(&self.river_mask, dir, self.res)
    }
}

/// Grayscale dilation: spreads each channel's depth outward into a flank that ramps
/// DOWN by `falloff` per cell — so the channel floor keeps its full depth (a box-blur
/// would average it away) while the banks rise back to grade.
fn dilate(src: &[f32], res: usize, passes: usize, falloff: f32) -> Vec<f32> {
    let mut cur = src.to_vec();
    let mut tmp = vec![0.0f32; src.len()];
    for _ in 0..passes {
        for face in 0..6usize {
            for y in 0..res {
                for x in 0..res {
                    let c = texel_index(face, x, y, res);
                    let mut best = cur[c];
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            if dx == 0 && dy == 0 {
                                continue;
                            }
                            let nb = neighbor_texel(face, x, y, dx, dy, res);
                            best = best.max(cur[texel_index(nb.face, nb.x, nb.y, res)] - falloff);
                        }
                    }
                    tmp[c] = best;
                }
            }
        }
        std::mem::swap(&mut cur, &mut tmp);
    }
    cur
}

#[cfg(test)]
mod tests {
    use super::*;

    // North hemisphere = land sloping to an equatorial sea, + bumps so peaks
    // (headwaters) and branching valleys exist.
    fn bumpy(dir: DVec3, _level: u32) -> f64 {
        0.3 * dir.y + 0.05 * (dir.x * 8.0).sin() * (dir.z * 8.0).sin()
    }

    #[test]
    fn carves_some_valley() {
        let rc = RiverCarve::new(bumpy, |_| 0.6);
        assert!(rc.incision.iter().any(|&v| v < -1e-4), "drainage carved no river");
        for &v in &rc.incision {
            assert!(v.is_finite() && v <= 1e-6, "incision must be ≤0: {v}");
        }
    }

    #[test]
    fn no_ocean_no_carve() {
        let rc = RiverCarve::new(|_, _| 0.1, |_| 0.5); // all land, no sea
        assert!(rc.incision.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn rivers_reach_the_sea() {
        // The network must drain to the coast, not pool in a landlocked interior:
        // at least one river-core cell sits next to an ocean cell.
        let rc = RiverCarve::new(bumpy, |_| 0.6);
        let res = rc.res;
        let mut touches_sea = false;
        'scan: for face in 0..6usize {
            for y in 0..res {
                for x in 0..res {
                    if rc.river_mask[texel_index(face, x, y, res)] < 0.99 {
                        continue; // river-core cell only
                    }
                    for k in 0..8 {
                        let nt = neighbor_texel(face, x, y, NB_DX[k], NB_DY[k], res);
                        if bumpy(texel_to_dir(nt.face, nt.x, nt.y, res), 0) < 0.0 {
                            touches_sea = true;
                            break 'scan;
                        }
                    }
                }
            }
        }
        assert!(touches_sea, "no river cell reaches the coast — network is landlocked");
    }

    #[test]
    fn deterministic() {
        let a = RiverCarve::new(bumpy, |_| 0.6);
        let b = RiverCarve::new(bumpy, |_| 0.6);
        for (x, y) in a.incision.iter().zip(b.incision.iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }
}

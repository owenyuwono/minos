//! River-driven valley incision — Step 2 of rivers. Carves dendritic valleys +
//! banks into the terrain from a HEIGHT + PRECIPITATION drainage network (NO
//! erosion-sim data). Baked once at `TectonicHeightField::new`; `tect_height` adds
//! `incision_at(dir)` to the surface, so BOTH render paths (Nanite bake + quadtree
//! mesher) get carved valleys with banks aligned to the rivers.
//!
//! Pipeline (see docs/rivers-research.md):
//!   1. Sample the routing height + precipitation (`moisture`) over a cube-sphere grid.
//!   2. Priority-Flood (Barnes 2014) fills depressions and builds a drainage TREE —
//!      every land cell drains downhill to the sea (connected by construction).
//!   3. Precipitation accumulates down the tree (wetter basins → bigger rivers).
//!   4. Threshold the accumulation → channel cells; channel depth ∝ √discharge.
//!   5. Box-blur the depth field → each channel's depth spreads perpendicular into a
//!      smooth valley with banks; negate → incision (≤0) added to the terrain.

use glam::DVec3;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::cubemap::{neighbor_texel, sample_smooth, texel_index, texel_to_dir};

// --- Tuning (ponytail: tunables, USER-verified on the Height view) ---
/// Drainage/incision cube-map resolution per face. Higher = finer valleys but the
/// bake is `6·RES²` routing-height calls (the dominant cost). 128 ≈ ~3 s.
const RES: usize = 128;
/// `HeightField` LOD for the routing height (broad valleys drive drainage).
const ROUTE_LEVEL: u32 = 5;
/// Priority-Flood fill increment (monotonic gradient across flats/filled basins).
const FILL_EPS: f32 = 1e-7;
/// Channel-initiation SUPPORT AREA (Montgomery–Dietrich): a cell is a channel only
/// once its UPSTREAM drainage ≥ this many cells. THE density knob — lower = denser
/// (finer headwaters), higher = only major rivers. Thresholding the raw support
/// AREA (NOT a log-normalized value) is what keeps channels THIN dendritic lines
/// instead of broad blobs that carve the continent into an archipelago.
const SUPPORT_CELLS: f64 = 60.0;
/// Peak channel-FLOOR depth (normalized height) at full discharge — the dilation
/// preserves it. ~0.22·1200 ≈ 260 m at a trunk. Bump for deeper valleys.
const CARVE_AMP: f64 = 0.22;
/// Valley-flank width in grid cells the bank ramps outward. Kept SMALL → thin river
/// valleys (bilinear sampling already widens a 1-cell channel into a smooth V).
const BANK_CELLS: usize = 1;
/// Depth lost per cell of bank ramp.
const BANK_FALLOFF: f64 = 0.10;

const SINK: u32 = u32::MAX;
const NB_DX: [i32; 8] = [-1, 0, 1, -1, 1, -1, 0, 1];
const NB_DY: [i32; 8] = [-1, -1, -1, 0, 0, 1, 1, 1];

/// Baked river-valley incision field (≤0 per cell). Sampler-only after construction.
pub struct RiverCarve {
    res: usize,
    incision: Vec<f32>,
}

/// Priority-Flood min-heap item (pop LOWEST filled first; idx breaks ties).
#[derive(Clone, Copy, PartialEq)]
struct HeapItem {
    filled: f32,
    idx: u32,
}
impl Eq for HeapItem {}
impl Ord for HeapItem {
    fn cmp(&self, o: &Self) -> Ordering {
        o.filled
            .partial_cmp(&self.filled)
            .unwrap_or(Ordering::Equal)
            .then_with(|| o.idx.cmp(&self.idx))
    }
}
impl PartialOrd for HeapItem {
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
    /// Bake the incision field. `route_h(dir, level)` is the terrain water routes on
    /// (base + erosion, pre-incision — passed in to avoid circularity);
    /// `moisture(dir)` is precipitation in `[0,1]`.
    pub fn new(route_h: impl Fn(DVec3, u32) -> f64, moisture: impl Fn(DVec3) -> f64) -> Self {
        let res = RES;
        let n = 6 * res * res;

        // 1. Sample routing height + precipitation.
        let mut height = vec![0.0f32; n];
        let mut precip = vec![0.0f32; n];
        for face in 0..6usize {
            for y in 0..res {
                for x in 0..res {
                    let c = texel_index(face, x, y, res);
                    let dir = texel_to_dir(face, x, y, res);
                    height[c] = route_h(dir, ROUTE_LEVEL) as f32;
                    precip[c] = moisture(dir).max(0.0) as f32;
                }
            }
        }

        // 2. Priority-Flood with flow-direction (Barnes 2014). Seed the sea (h<0).
        let mut filled = height.clone();
        let mut downstream = vec![SINK; n];
        let mut processed = vec![false; n];
        let mut order: Vec<u32> = Vec::with_capacity(n);
        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::new();
        for c in 0..n {
            if height[c] < 0.0 {
                processed[c] = true;
                heap.push(HeapItem { filled: filled[c], idx: c as u32 });
            }
        }
        if heap.is_empty() {
            return Self { res, incision: vec![0.0f32; n] }; // no ocean → no rivers
        }
        while let Some(item) = heap.pop() {
            let c = item.idx as usize;
            order.push(c as u32);
            let (f, x, y) = cell_to_fxy(c as u32, res);
            for k in 0..8 {
                let nt = neighbor_texel(f, x, y, NB_DX[k], NB_DY[k], res);
                let nc = texel_index(nt.face, nt.x, nt.y, res);
                if processed[nc] {
                    continue;
                }
                processed[nc] = true;
                let nf = height[nc].max(item.filled + FILL_EPS);
                filled[nc] = nf;
                downstream[nc] = c as u32;
                heap.push(HeapItem { filled: nf, idx: nc as u32 });
            }
        }

        // 3. Accumulate BOTH unit drainage AREA (for the support-area channel
        //    threshold) and PRECIPITATION (for discharge-scaled depth) down the tree.
        let mut area: Vec<f64> = vec![1.0; n]; // each cell drains itself
        let mut flow: Vec<f64> = precip.iter().map(|&p| p as f64).collect();
        for &c in order.iter().rev() {
            let d = downstream[c as usize];
            if d != SINK {
                area[d as usize] += area[c as usize];
                flow[d as usize] += flow[c as usize];
            }
        }
        let flow_max = flow.iter().cloned().fold(0.0f64, f64::max).max(1e-9);

        // 4. Channel = land cell whose UPSTREAM AREA clears the support threshold.
        //    Floor depth ∝ √(discharge / max) so trunks cut deeper than tributaries.
        let mut depth = vec![0.0f32; n];
        for c in 0..n {
            if height[c] >= 0.0 && area[c] >= SUPPORT_CELLS {
                depth[c] = (CARVE_AMP * (flow[c] / flow_max).sqrt()) as f32;
            }
        }

        // 5. Dilate the depth outward into valley flanks: the channel FLOOR keeps its
        //    full depth and the banks ramp down over BANK_CELLS (unlike a box-blur,
        //    which would average the floor away to nothing). Negate → incision.
        let spread = dilate(&depth, res, BANK_CELLS, BANK_FALLOFF as f32);
        let incision: Vec<f32> = spread.iter().map(|&d| -d).collect();
        Self { res, incision }
    }

    /// Incision (≤0, normalized height) to add to the terrain at `dir`. C0 bilinear.
    pub fn incision_at(&self, dir: DVec3) -> f64 {
        sample_smooth(&self.incision, dir, self.res)
    }
}

/// Grayscale dilation: `passes` of "each cell = max(self, max-neighbour − falloff)",
/// cross-face via `neighbor_texel`. Spreads each channel's depth outward into a
/// flank that ramps DOWN by `falloff` per cell — so the channel floor keeps its full
/// depth (a box-blur would average it away) while the banks rise back to grade.
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

    // North hemisphere = land sloping to an equatorial sea, + bumps so drainage
    // branches. Uniform precipitation.
    fn bumpy(dir: DVec3, _level: u32) -> f64 {
        0.3 * dir.y + 0.05 * (dir.x * 8.0).sin() * (dir.z * 8.0).sin()
    }

    #[test]
    fn carves_some_valley() {
        let rc = RiverCarve::new(bumpy, |_| 0.6);
        // Something is incised, nothing is RAISED, all finite.
        assert!(rc.incision.iter().any(|&v| v < -1e-4), "no valley carved");
        for &v in &rc.incision {
            assert!(v.is_finite() && v <= 1e-6, "incision must be ≤0: {v}");
        }
    }

    #[test]
    fn no_ocean_no_carve() {
        let rc = RiverCarve::new(|_, _| 0.1, |_| 0.5); // all land, no sea seed
        assert!(rc.incision.iter().all(|&v| v == 0.0));
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

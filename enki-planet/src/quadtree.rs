//! Quadtree over the 6 cube faces for LOD management.

use std::f64::consts::PI;

use glam::DVec3;

use crate::face_bases::patch_center_dir;

/// Uniquely identifies a quadtree node: `(face, level, ix, iy)`.
pub type ChunkKey = (u8, u8, u32, u32);

/// A single node in the planet quadtree.
pub struct QuadtreeNode {
    pub face: u8,
    pub level: u8,
    pub ix: u32,
    pub iy: u32,
    pub center_dir: DVec3,
    pub world_center: DVec3,
    pub node_size: f64,
    /// Surface position (radius * center_dir + height displacement); None until computed.
    pub surface_center: Option<DVec3>,
    pub children: Option<Box<[QuadtreeNode; 4]>>,
}

impl QuadtreeNode {
    /// Construct a root or child node.
    ///
    /// - `center_dir` — unit sphere direction to the patch center.
    /// - `world_center` — `center_dir * radius`.
    /// - `node_size` — approximate arc-length side: `(π/2 · radius) / 2^level`.
    pub fn new(face: u8, level: u8, ix: u32, iy: u32, radius: f64) -> Self {
        let center_dir = patch_center_dir(face, level, ix, iy);
        let world_center = center_dir * radius;
        let node_size = (PI / 2.0 * radius) / (1u64 << level) as f64;
        Self {
            face,
            level,
            ix,
            iy,
            center_dir,
            world_center,
            node_size,
            surface_center: None,
            children: None,
        }
    }

    /// Return the canonical key for this node.
    pub fn key(&self) -> ChunkKey {
        (self.face, self.level, self.ix, self.iy)
    }

    /// Subdivide this node into 4 children (no-op if already split).
    ///
    /// Children at `level+1` cover `(2·ix+cx, 2·iy+cy)` for `cx,cy ∈ {0,1}`.
    pub fn split(&mut self, radius: f64) {
        if self.children.is_some() {
            return;
        }
        let nl = self.level + 1;
        let bx = self.ix * 2;
        let by = self.iy * 2;
        self.children = Some(Box::new([
            QuadtreeNode::new(self.face, nl, bx,     by,     radius),
            QuadtreeNode::new(self.face, nl, bx + 1, by,     radius),
            QuadtreeNode::new(self.face, nl, bx,     by + 1, radius),
            QuadtreeNode::new(self.face, nl, bx + 1, by + 1, radius),
        ]));
    }

    /// Collapse children back into this leaf (no-op if already a leaf).
    pub fn merge(&mut self) {
        self.children = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const RADIUS: f64 = 6_371_000.0;

    /// `key()` round-trips the four fields.
    #[test]
    fn key_round_trips() {
        let node = QuadtreeNode::new(3, 5, 7, 11, RADIUS);
        assert_eq!(node.key(), (3, 5, 7, 11));
    }

    /// Children get correct `(level+1, 2·ix+{0,1}, 2·iy+{0,1})`.
    #[test]
    fn split_children_coords() {
        let mut node = QuadtreeNode::new(0, 2, 3, 1, RADIUS);
        node.split(RADIUS);
        let children = node.children.as_ref().expect("children must be Some after split");
        let expected_keys: [(u8, u8, u32, u32); 4] = [
            (0, 3, 6, 2),
            (0, 3, 7, 2),
            (0, 3, 6, 3),
            (0, 3, 7, 3),
        ];
        for (i, child) in children.iter().enumerate() {
            assert_eq!(
                child.key(),
                expected_keys[i],
                "child {i} key mismatch"
            );
        }
    }

    /// Each child's `center_dir` lies within the parent's bounding sphere
    /// (radius ≈ node_size·√2 / 2 around `center_dir`).
    #[test]
    fn split_children_within_parent_bounding_sphere() {
        for face in 0u8..6 {
            let mut node = QuadtreeNode::new(face, 1, 0, 0, RADIUS);
            node.split(RADIUS);
            let children = node.children.as_ref().unwrap();
            // Bounding radius in chord distance.  For a patch at level L the
            // angular half-extent is approximately node_size / radius.  We use
            // a generous 1.5× factor to allow for the equal-area warp distortion.
            let bound = node.node_size * 2.0_f64.sqrt() / 2.0 / RADIUS * 1.5;
            for child in children.iter() {
                let chord = (child.center_dir - node.center_dir).length();
                assert!(
                    chord < bound + 1e-10,
                    "face={face}: child center_dir chord dist {chord} exceeds bound {bound}"
                );
            }
        }
    }

    /// `split` is a no-op when called twice; children array is unchanged.
    #[test]
    fn split_idempotent() {
        let mut node = QuadtreeNode::new(0, 0, 0, 0, RADIUS);
        node.split(RADIUS);
        let key0 = node.children.as_ref().unwrap()[0].key();
        node.split(RADIUS);
        assert_eq!(node.children.as_ref().unwrap()[0].key(), key0);
    }

    /// `merge` collapses children; `split` can re-expand afterward.
    #[test]
    fn merge_collapses_then_split_expands() {
        let mut node = QuadtreeNode::new(0, 0, 0, 0, RADIUS);
        node.split(RADIUS);
        assert!(node.children.is_some());
        node.merge();
        assert!(node.children.is_none());
        node.split(RADIUS);
        assert!(node.children.is_some());
    }

    /// `node_size` matches the formula `(π/2 · radius) / 2^level`.
    #[test]
    fn node_size_matches_formula() {
        for level in 0u8..=10 {
            let node = QuadtreeNode::new(0, level, 0, 0, RADIUS);
            let expected = (PI / 2.0 * RADIUS) / (1u64 << level) as f64;
            assert!(
                (node.node_size - expected).abs() < 1e-6,
                "level {level}: node_size={} expected={expected}",
                node.node_size
            );
        }
    }

    /// `world_center` equals `center_dir * radius`.
    #[test]
    fn world_center_equals_center_dir_times_radius() {
        let node = QuadtreeNode::new(2, 3, 4, 5, RADIUS);
        let expected = node.center_dir * RADIUS;
        let diff = (node.world_center - expected).length();
        assert!(diff < 1e-9, "world_center diff={diff}");
    }

    /// `surface_center` starts as `None`.
    #[test]
    fn surface_center_starts_none() {
        let node = QuadtreeNode::new(0, 0, 0, 0, RADIUS);
        assert!(node.surface_center.is_none());
    }
}

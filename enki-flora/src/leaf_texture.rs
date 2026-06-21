//! CPU leaf-cluster texture rasterizer — a faithful Rust port of dryad's
//! `leafTexture.js` `makeLeafClusterTexture` (the `leafMode === 'cluster'`
//! branch) + `buildLeafNormalMap`.
//!
//! Bakes, per genome, TWO RGBA8 images:
//!   * a COLOR map (alpha = silhouette coverage), painting a sprig of N
//!     superformula leaves with a base→tip lightness gradient + pinnate
//!     venation, and
//!   * a NORMAL map (tangent space, Sobel of the color luminance height field,
//!     alpha = the color cutout).
//!
//! These are uploaded by `flora_render.rs` as two `set1` sampled textures and
//! sampled by `fs_leaf` — replacing the in-shader Gielis silhouette with a real
//! generated leaf texture, exactly as dryad's leaf `InstancedMesh` does.
//!
//! Determinism: every random draw flows from [`crate::rng::Mulberry32`] keyed on
//! the genome seed, so reseeding rebuilds a bit-stable texture (parametric — no
//! system entropy). The DOM canvas in dryad is replaced by a software scanline
//! rasterizer here.
//!
//! # ponytail approximations vs. dryad's `<canvas>` 2D context
//! - **Polygon fill**: dryad relies on the browser's `ctx.fill()` (non-zero
//!   winding, analytic AA). We do an even-odd scanline fill with a per-pixel
//!   gradient lerp — a 1px hard edge, no sub-pixel AA on the silhouette.
//!   `buildLeafNormalMap`'s `a > 0.03` height mask + the alpha-cutout test (0.5)
//!   make this immaterial (the cutout is binary anyway).
//! - **Vein strokes**: dryad strokes round-capped lines via `ctx.stroke()`. We
//!   plot each segment as a distance-to-segment coverage splat (round caps,
//!   `globalAlpha=0.55` source-over) — visually equivalent, no canvas dep.
//! - **Compositing**: only the broadleaf cluster path is ported (the leaf
//!   genome enki feeds is a broadleaf — `leafDivision==0`, no frond crossfade,
//!   no scratch `drawImage` layers). Leaves are painted straight into the buffer
//!   with source-over alpha, matching dryad's pure-cluster direct-draw branch.

use crate::rng::Mulberry32;

/// Baked leaf-cluster texture pair. Both buffers are `size*size*4` RGBA8.
pub struct LeafTexture {
    /// Color sprite, alpha = silhouette coverage. Upload as sRGB.
    pub color: Vec<u8>,
    /// Tangent-space normal map (RGB = n*0.5+0.5), alpha = the color cutout.
    /// Upload as UNORM (linear, NoColorSpace).
    pub normal: Vec<u8>,
    /// Square edge length in texels.
    pub size: u32,
}

/// The 6 leaf-shape genes + pigment + seed that drive a cluster bake. All genes
/// are `[0,1]` (the resolved-genome convention). Mirrors the subset of
/// `makeLeafClusterTexture`'s options enki uses.
#[derive(Clone, Copy, Debug)]
pub struct LeafGenes {
    pub pigment: f64,
    pub leaf_width: f64,
    pub leaf_length: f64,
    pub leaf_tip: f64,
    pub leaf_serration: f64,
    pub leaf_lobing: f64,
    pub leaf_skew: f64,
    /// Texture seed — dryad keys the cluster RNG on the tree's leaf seed.
    pub seed: u32,
}

/// Tangent-space normal-map strength (Sobel gradient gain), dryad
/// `LEAF_NORMAL_STRENGTH` (leafTexture.js:965).
const LEAF_NORMAL_STRENGTH: f32 = 2.5;

/// ponytail: x_scale baked into the SINGLE-leaf sprite. dryad forces x_scale=1
/// there (width moved to the card aspect), but at x_scale=1 the height-normalized
/// superformula outline is a thin spike (normalized half-width ≈0.12 of height),
/// which reads as a blade, not a leaf. A broad fixed x_scale shapes the single
/// sprite into a recognizable ovate; the per-tree width gene still varies render
/// width via the card's widthFactor on the mesh side (it does NOT re-bake here).
/// Chosen so the leaf's pixel width ≈ 0.45 × its height (a typical broadleaf).
const SINGLE_SPRITE_X_SCALE: f64 = 3.8;

// ── superformula outline (leafTexture.js:44-156) ────────────────────────────

#[derive(Clone, Copy)]
struct Pt {
    x: f64,
    y: f64,
}

/// Superformula base + outline params, the output of [`leaf_base_params`] after
/// the per-leaf variant jitter. Mirrors the JS params object.
#[derive(Clone, Copy)]
struct SfParams {
    m: f64,
    n1: f64,
    n2: f64,
    n3: f64,
    a: f64,
    b: f64,
    axial_stretch: f64,
    x_scale: f64,
    serration: f64,
    serration_freq: f64,
    skew_gamma: f64,
}

/// `superformulaPoint` (leafTexture.js:44-54).
fn superformula_point(theta: f64, p: &SfParams) -> Pt {
    let t = p.m * theta / 4.0;
    let cos_t = (t.cos() / p.a).abs();
    let sin_t = (t.sin() / p.b).abs();
    let inner = cos_t.powf(p.n2) + sin_t.powf(p.n3);
    if inner == 0.0 {
        return Pt { x: 0.0, y: 0.0 };
    }
    let r = inner.powf(-1.0 / p.n1);
    if !r.is_finite() {
        return Pt { x: 0.0, y: 0.0 };
    }
    Pt {
        x: r * theta.cos(),
        y: r * theta.sin(),
    }
}

/// `superformulaOutline` (leafTexture.js:56-156). Returns the normalized outline
/// (base at y=0, tip at y=1) with serration teeth, ready for [`draw_leaf`].
fn superformula_outline(p: &SfParams, steps: usize) -> Vec<Pt> {
    // 1. Sample raw points (y pre-stretched by axialStretch).
    let mut raw: Vec<Pt> = Vec::with_capacity(steps);
    for i in 0..steps {
        let theta = (2.0 * std::f64::consts::PI * i as f64) / steps as f64;
        let pt = superformula_point(theta, p);
        raw.push(Pt {
            x: pt.x,
            y: pt.y * p.axial_stretch,
        });
    }

    // 2. tip (max y) / base (min y).
    let mut tip_idx = 0usize;
    let mut base_idx = 0usize;
    for i in 1..raw.len() {
        if raw[i].y > raw[tip_idx].y {
            tip_idx = i;
        }
        if raw[i].y < raw[base_idx].y {
            base_idx = i;
        }
    }
    let tip = raw[tip_idx];
    let base = raw[base_idx];

    // 3. Rotate so tip→base aligns with +Y.
    let dx = base.x - tip.x;
    let dy = base.y - tip.y;
    let rot = dx.atan2(dy);
    let (c, s) = (rot.cos(), rot.sin());
    let rotated: Vec<Pt> = raw
        .iter()
        .map(|q| Pt {
            x: q.x * c - q.y * s,
            y: q.x * s + q.y * c,
        })
        .collect();

    // 4. y extents.
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for q in &rotated {
        if q.y < min_y {
            min_y = q.y;
        }
        if q.y > max_y {
            max_y = q.y;
        }
    }

    // 5. Normalize: base→0, tip→1, x scaled by xScale.
    let leaf_len = max_y - min_y;
    if leaf_len == 0.0 {
        return rotated;
    }
    let mut normalized: Vec<Pt> = rotated
        .iter()
        .map(|q| Pt {
            x: (q.x / leaf_len) * p.x_scale,
            y: (q.y - min_y) / leaf_len,
        })
        .collect();

    // 5b. Skew warp y' = y^skewGamma.
    if p.skew_gamma != 1.0 {
        for q in normalized.iter_mut() {
            q.y = q.y.powf(p.skew_gamma);
        }
    }

    // 6. Serration — outward marginal teeth (read neighbours from an unmodified
    //    copy so a displaced point never cascades into the next).
    if p.serration > 0.0 {
        let n = normalized.len();
        let mut n_min_x = f64::INFINITY;
        let mut n_max_x = f64::NEG_INFINITY;
        for q in &normalized {
            if q.x < n_min_x {
                n_min_x = q.x;
            }
            if q.x > n_max_x {
                n_max_x = q.x;
            }
        }
        let half_w = ((n_max_x - n_min_x) / 2.0).max(1e-3);
        let tooth_depth = (0.30f64).min(p.serration * 3.5) * half_w;
        let src: Vec<Pt> = normalized.clone();
        for i in 0..n {
            let prev = src[(i + n - 1) % n];
            let next = src[(i + 1) % n];
            let cur = src[i];
            let edge_dx = next.x - prev.x;
            let edge_dy = next.y - prev.y;
            let len = (edge_dx * edge_dx + edge_dy * edge_dy).sqrt().max(1.0);
            let mut nx = -edge_dy / len;
            let mut ny = edge_dx / len;
            if nx * cur.x < 0.0 {
                nx = -nx;
                ny = -ny;
            }
            let tooth =
                (((i as f64) * 2.0 * std::f64::consts::PI * p.serration_freq) / n as f64).sin().max(0.0);
            let edge_fade = (cur.x.abs() / half_w).min(1.0);
            let amp = tooth_depth * tooth * edge_fade;
            normalized[i] = Pt {
                x: cur.x + nx * amp,
                y: cur.y + ny * amp,
            };
        }
    }

    normalized
}

// ── pinnate venation (leafTexture.js:298-351) ───────────────────────────────

struct VenNode {
    x: f64,
    y: f64,
    depth: u32,
}
struct Venation {
    nodes: Vec<VenNode>,
    edges: Vec<(usize, usize)>,
}

/// `pinnateVenation` (leafTexture.js:298-351). Midrib (depth 0) + `pairs`
/// secondary vein pairs (mid node depth 2, end depth 3).
fn pinnate_venation(outline: &[Pt], pairs: usize) -> Venation {
    let mut min_y = f64::INFINITY;
    let mut max_y = f64::NEG_INFINITY;
    for p in outline {
        if p.y < min_y {
            min_y = p.y;
        }
        if p.y > max_y {
            max_y = p.y;
        }
    }
    let h = (max_y - min_y).max(f64::MIN_POSITIVE);
    let band = (h / (pairs as f64 + 2.0)) * 0.75;

    let half_width_at = |y: f64| -> f64 {
        let mut w = 0.0;
        for p in outline {
            if (p.y - y).abs() <= band {
                let ax = p.x.abs();
                if ax > w {
                    w = ax;
                }
            }
        }
        w
    };

    let mut nodes: Vec<VenNode> = Vec::new();
    let mut edges: Vec<(usize, usize)> = Vec::new();

    // Midrib base→tip.
    let mid_steps = pairs + 1;
    let mut mid_idx: Vec<usize> = Vec::with_capacity(mid_steps + 1);
    for i in 0..=mid_steps {
        let t = i as f64 / mid_steps as f64;
        nodes.push(VenNode {
            x: 0.0,
            y: min_y + t * h,
            depth: 0,
        });
        let idx = nodes.len() - 1;
        if i > 0 {
            edges.push((mid_idx[i - 1], idx));
        }
        mid_idx.push(idx);
    }

    // Secondary vein pairs.
    for k in 1..=pairs {
        let mi = mid_idx[k];
        let my = nodes[mi].y;
        let tt = (my - min_y) / h;
        let hw = half_width_at(my);
        if hw < 1e-4 {
            continue;
        }
        let ex = hw * 0.82;
        let ey = my + (max_y - my) * (0.12 + 0.20 * (1.0 - tt));
        let mx = ex * 0.5;
        let mid_y = my + (ey - my) * 0.55;
        let ri = nodes.len();
        nodes.push(VenNode {
            x: mx,
            y: mid_y,
            depth: 2,
        });
        let re = nodes.len();
        nodes.push(VenNode {
            x: ex,
            y: ey,
            depth: 3,
        });
        edges.push((mi, ri));
        edges.push((ri, re));
        let li = nodes.len();
        nodes.push(VenNode {
            x: -mx,
            y: mid_y,
            depth: 2,
        });
        let le = nodes.len();
        nodes.push(VenNode {
            x: -ex,
            y: ey,
            depth: 3,
        });
        edges.push((mi, li));
        edges.push((li, le));
    }

    Venation { nodes, edges }
}

// ── color helpers (leafTexture.js:17-38, colorRamp pigment) ──────────────────

fn hsl_to_rgb(h: f64, s: f64, l: f64) -> [f64; 3] {
    let a = s * l.min(1.0 - l);
    let f = |n: f64| -> f64 {
        let k = (n + h * 12.0) % 12.0;
        let inner = (k - 3.0).min(9.0 - k).min(1.0);
        l - a * inner.max(-1.0)
    };
    [f(0.0), f(8.0), f(4.0)]
}

fn rgb_to_hsl(r: f64, g: f64, b: f64) -> [f64; 3] {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let l = (max + min) / 2.0;
    if max == min {
        return [0.0, 0.0, l];
    }
    let d = max - min;
    let s = if l > 0.5 {
        d / (2.0 - max - min)
    } else {
        d / (max + min)
    };
    let h = if max == r {
        (g - b) / d + if g < b { 6.0 } else { 0.0 }
    } else if max == g {
        (b - r) / d + 2.0
    } else {
        (r - g) / d + 4.0
    };
    [h / 6.0, s, l]
}

/// Lighten an RGB[0,1] color by `amount` in HSL lightness (leafTexture.js:556).
fn lighten(rgb: [f64; 3], amount: f64) -> [f64; 3] {
    let [h, s, l] = rgb_to_hsl(rgb[0], rgb[1], rgb[2]);
    hsl_to_rgb(h, s, (l + amount).min(1.0))
}

// ── gene → superformula params (leafTexture.js:411-458) ─────────────────────

fn leaf_base_params(g: &LeafGenes) -> SfParams {
    let m = 2.0 + g.leaf_lobing * 3.0;
    let n2 = 4.0 + g.leaf_lobing * 10.0;
    let n3 = 4.0 + g.leaf_lobing * 10.0;
    let n1_base = 0.4 + g.leaf_tip * 2.1;
    let n1 = n1_base * (1.0 - g.leaf_lobing * 0.35);
    let axial_stretch_base = 1.0 + g.leaf_length * 3.0;
    let axial_stretch = axial_stretch_base * (1.0 - g.leaf_lobing * 0.72) + 1.1 * g.leaf_lobing * 0.72;
    let x_scale = 0.4 + g.leaf_width * 1.2;
    let serration_boost = 1.0 + g.leaf_lobing * 1.8;
    let serration = g.leaf_serration * 0.12 * serration_boost;
    let serration_freq = 8.0 + g.leaf_serration * 24.0;
    let skew_clamped = g.leaf_skew.clamp(0.08, 0.92);
    let skew_gamma = skew_clamped.ln() / 0.5f64.ln();
    SfParams {
        m,
        n1,
        n2,
        n3,
        a: 1.0,
        b: 1.0,
        axial_stretch,
        x_scale,
        serration,
        serration_freq,
        skew_gamma,
    }
}

/// `leafVariantParams` (leafTexture.js:486-500) — per-leaf jitter of the base.
fn leaf_variant_params(base: &SfParams, rng: &mut Mulberry32) -> SfParams {
    SfParams {
        m: base.m,
        n1: base.n1 * (0.85 + rng.next() * 0.30),
        n2: base.n2 * (0.80 + rng.next() * 0.40),
        n3: base.n3 * (0.80 + rng.next() * 0.40),
        a: base.a,
        b: base.b + (rng.next() - 0.5) * 0.3,
        axial_stretch: base.axial_stretch * (0.85 + rng.next() * 0.30),
        x_scale: base.x_scale,
        serration: base.serration,
        serration_freq: base.serration_freq,
        skew_gamma: base.skew_gamma,
    }
}

// ── cluster layout (leafTexture.js:357-379) ─────────────────────────────────

#[derive(Clone, Copy)]
struct ClusterLeaf {
    angle: f64,
    ox: f64,
    oy: f64,
    scale: f64,
    hue_delta: f64,
    light_delta: f64,
}

/// `clusterLeafCount` (leafTexture.js:376-379): 5..8 leaves.
fn cluster_leaf_count(seed: u32) -> usize {
    let mut rng = Mulberry32::new(seed ^ 0xDEAD_BEEF);
    5 + (rng.next() * 4.0).floor() as usize
}

/// `clusterLeafParams` (leafTexture.js:357-374): a fan of `count` leaves.
fn cluster_leaf_params(seed: u32, count: usize) -> Vec<ClusterLeaf> {
    let mut rng = Mulberry32::new(seed);
    let spread_rad = (200.0 / 180.0) * std::f64::consts::PI;
    let step = spread_rad / (count as f64 - 1.0);
    let base_angle = -spread_rad / 2.0;
    (0..count)
        .map(|i| {
            let jitter = (rng.next() - 0.5) * (step * 0.35);
            let angle = base_angle + i as f64 * step + jitter;
            let ox = (rng.next() - 0.5) * 0.32;
            let oy = rng.next() * 0.14;
            let scale = 0.60 + rng.next() * 0.40;
            let hue_delta = (rng.next() - 0.5) * (20.0 / 360.0);
            let light_delta = (rng.next() - 0.5) * 0.18;
            ClusterLeaf {
                angle,
                ox,
                oy,
                scale,
                hue_delta,
                light_delta,
            }
        })
        .collect()
}

// ── software rasterizer ─────────────────────────────────────────────────────

/// Mutable RGBA8 image with source-over alpha compositing.
struct Canvas {
    size: usize,
    px: Vec<u8>, // RGBA8, straight (non-premultiplied), to mirror canvas source-over.
}

impl Canvas {
    fn new(size: usize) -> Self {
        Self {
            size,
            px: vec![0u8; size * size * 4],
        }
    }

    /// Source-over blend of an RGB color at coverage `a` (0..1) into texel (x,y).
    #[inline]
    fn blend(&mut self, x: i32, y: i32, rgb: [f64; 3], a: f64) {
        if x < 0 || y < 0 || x >= self.size as i32 || y >= self.size as i32 || a <= 0.0 {
            return;
        }
        let i = (y as usize * self.size + x as usize) * 4;
        let sa = a.clamp(0.0, 1.0);
        let da = self.px[i + 3] as f64 / 255.0;
        let out_a = sa + da * (1.0 - sa);
        if out_a <= 0.0 {
            return;
        }
        for c in 0..3 {
            let sc = (rgb[c].clamp(0.0, 1.0)) * 255.0;
            let dc = self.px[i + c] as f64;
            // straight-alpha source-over: out = (s*sa + d*da*(1-sa)) / out_a
            let oc = (sc * sa + dc * da * (1.0 - sa)) / out_a;
            self.px[i + c] = oc.round().clamp(0.0, 255.0) as u8;
        }
        self.px[i + 3] = (out_a * 255.0).round().clamp(0.0, 255.0) as u8;
    }
}

/// Affine: translate to (cx,cy) then rotate by `angle` (matches ctx.translate +
/// ctx.rotate). Maps a leaf-local point to device pixels.
#[inline]
fn xform(angle: f64, cx: f64, cy: f64, x: f64, y: f64) -> (f64, f64) {
    let (c, s) = (angle.cos(), angle.sin());
    (cx + x * c - y * s, cy + x * s + y * c)
}

/// `drawLeaf` (leafTexture.js:506-553) — fill the polygon with the base→tip
/// gradient, then stroke the veins at alpha 0.55.
#[allow(clippy::too_many_arguments)]
fn draw_leaf(
    canvas: &mut Canvas,
    cx: f64,
    cy: f64,
    leaf_h: f64,
    angle: f64,
    fill: [f64; 3],
    vein: [f64; 3],
    outline: &[Pt],
    ven: &Venation,
) {
    // Scale outline: x → x*leafH*0.5, y → -y*leafH (tip up). Keep leaf-LOCAL
    // (pre-rotation) coords so the gradient lerp can read the local y.
    let local: Vec<Pt> = outline
        .iter()
        .map(|p| Pt {
            x: p.x * leaf_h * 0.5,
            y: -p.y * leaf_h,
        })
        .collect();

    // Device-space polygon (rotated+translated) for the scanline fill bbox.
    let dev: Vec<(f64, f64)> = local.iter().map(|p| xform(angle, cx, cy, p.x, p.y)).collect();

    let grad_base = lighten(fill, 0.12); // at local y=0
    let grad_tip = fill; // at local y=-leafH

    // Even-odd scanline fill. For each scanline we collect device-x crossings and
    // fill spans; per pixel we map back to leaf-LOCAL y to lerp the gradient.
    let mut min_yf = f64::INFINITY;
    let mut max_yf = f64::NEG_INFINITY;
    for &(_, y) in &dev {
        if y < min_yf {
            min_yf = y;
        }
        if y > max_yf {
            max_yf = y;
        }
    }
    let y0 = min_yf.floor().max(0.0) as i32;
    let y1 = (max_yf.ceil() as i32).min(canvas.size as i32 - 1);
    let (ca, sa) = (angle.cos(), angle.sin());
    // Inverse rotation (about cx,cy) to recover local y for the gradient.
    let inv_local_y = |px: f64, py: f64| -> f64 {
        let dx = px - cx;
        let dy = py - cy;
        // local = R^-1 * d ; local.y = -dx*sin + dy*cos
        -dx * sa + dy * ca
    };

    let n = dev.len();
    for y in y0..=y1 {
        let yc = y as f64 + 0.5;
        // Gather x-crossings of the polygon edges at this scanline.
        let mut xs: Vec<f64> = Vec::with_capacity(8);
        for i in 0..n {
            let (x0, ya) = dev[i];
            let (x1b, yb) = dev[(i + 1) % n];
            if (ya > yc) != (yb > yc) {
                let t = (yc - ya) / (yb - ya);
                xs.push(x0 + t * (x1b - x0));
            }
        }
        if xs.len() < 2 {
            continue;
        }
        xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut k = 0;
        while k + 1 < xs.len() {
            let xa = xs[k];
            let xb = xs[k + 1];
            let sx = xa.ceil().max(0.0) as i32;
            let ex = (xb.floor() as i32).min(canvas.size as i32 - 1);
            for x in sx..=ex {
                // Local y in [-leafH, 0]; gradient t: 0 at base (y=0) → 1 at tip (y=-leafH).
                let ly = inv_local_y(x as f64 + 0.5, yc);
                let t = (-ly / leaf_h).clamp(0.0, 1.0);
                let col = [
                    grad_base[0] + (grad_tip[0] - grad_base[0]) * t,
                    grad_base[1] + (grad_tip[1] - grad_base[1]) * t,
                    grad_base[2] + (grad_tip[2] - grad_base[2]) * t,
                ];
                canvas.blend(x, y, col, 1.0);
            }
            k += 2;
        }
    }

    // Veins: thick round-capped segments at globalAlpha 0.55.
    for &(from, to) in &ven.edges {
        let fn_ = &ven.nodes[from];
        let tn = &ven.nodes[to];
        let (fx, fy) = xform(angle, cx, cy, fn_.x * leaf_h * 0.5, -fn_.y * leaf_h);
        let (tx, ty) = xform(angle, cx, cy, tn.x * leaf_h * 0.5, -tn.y * leaf_h);
        // lineWidth = max(0.4, leafH*0.012*0.6^(depth*0.25)).
        let lw = (leaf_h * 0.012 * 0.6f64.powf(tn.depth as f64 * 0.25)).max(0.4);
        stroke_segment(canvas, fx, fy, tx, ty, lw, vein, 0.55);
    }
}

/// Splat a round-capped line segment with half-width `lw/2`, coverage = how far
/// inside the stroke the texel center is (1px AA falloff at the edge), composited
/// source-over at `alpha`. Replaces canvas `ctx.stroke()`.
#[allow(clippy::too_many_arguments)]
fn stroke_segment(
    canvas: &mut Canvas,
    x0: f64,
    y0: f64,
    x1: f64,
    y1: f64,
    width: f64,
    rgb: [f64; 3],
    alpha: f64,
) {
    let r = (width * 0.5).max(0.5);
    let min_x = (x0.min(x1) - r - 1.0).floor().max(0.0) as i32;
    let max_x = ((x0.max(x1) + r + 1.0).ceil() as i32).min(canvas.size as i32 - 1);
    let min_y = (y0.min(y1) - r - 1.0).floor().max(0.0) as i32;
    let max_y = ((y0.max(y1) + r + 1.0).ceil() as i32).min(canvas.size as i32 - 1);
    let dx = x1 - x0;
    let dy = y1 - y0;
    let len2 = (dx * dx + dy * dy).max(1e-9);
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = x as f64 + 0.5;
            let py = y as f64 + 0.5;
            // distance from texel to the segment.
            let t = (((px - x0) * dx + (py - y0) * dy) / len2).clamp(0.0, 1.0);
            let cx = x0 + t * dx;
            let cy = y0 + t * dy;
            let d = ((px - cx) * (px - cx) + (py - cy) * (py - cy)).sqrt();
            // coverage: 1 inside (r-0.5), ramp to 0 at (r+0.5).
            let cov = ((r + 0.5 - d) / 1.0).clamp(0.0, 1.0);
            if cov > 0.0 {
                canvas.blend(x, y, rgb, cov * alpha);
            }
        }
    }
}

// ── normal map (leafTexture.js:973-1012) ────────────────────────────────────

/// `buildLeafNormalMap`: height = luminance where alpha>0.03 else 0, Sobel-lite
/// central diff × strength, encode RGB = n*0.5+0.5, carry the source alpha.
fn build_normal_map(color: &[u8], size: usize, strength: f32) -> Vec<u8> {
    let n = size * size;
    let mut h = vec![0f32; n];
    for i in 0..n {
        let a = color[i * 4 + 3] as f32 / 255.0;
        let lum = (0.299 * color[i * 4] as f32
            + 0.587 * color[i * 4 + 1] as f32
            + 0.114 * color[i * 4 + 2] as f32)
            / 255.0;
        h[i] = if a > 0.03 { lum } else { 0.0 };
    }
    let mut out = vec![0u8; n * 4];
    for y in 0..size {
        for x in 0..size {
            let i = y * size + x;
            let xl = h[y * size + if x > 0 { x - 1 } else { x }];
            let xr = h[y * size + if x < size - 1 { x + 1 } else { x }];
            let yu = h[if y > 0 { y - 1 } else { y } * size + x];
            let yd = h[if y < size - 1 { y + 1 } else { y } * size + x];
            let nx = (xl - xr) * strength;
            let ny = (yd - yu) * strength; // +Y up
            let nz = 1.0f32;
            let inv = 1.0 / (nx * nx + ny * ny + nz * nz).sqrt().max(1e-9);
            out[i * 4] = ((nx * inv * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            out[i * 4 + 1] = ((ny * inv * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            out[i * 4 + 2] = ((nz * inv * 0.5 + 0.5) * 255.0).round().clamp(0.0, 255.0) as u8;
            out[i * 4 + 3] = color[i * 4 + 3]; // carry the cutout
        }
    }
    out
}

// ── public bake ─────────────────────────────────────────────────────────────

/// Texture edge length. dryad uses 512 ("high") / 256 ("low"); we bake at 512.
pub const LEAF_TEX_SIZE: u32 = 512;

/// Bake the leaf-cluster COLOR + NORMAL textures from the genome leaf genes.
/// Ports `makeLeafClusterTexture` (cluster branch) + `buildLeafNormalMap`.
pub fn bake_leaf_cluster(g: &LeafGenes) -> LeafTexture {
    let size = LEAF_TEX_SIZE as usize;
    let mut canvas = Canvas::new(size);

    let base_params = leaf_base_params(g);
    let base_rgb = crate::color::pigment_to_color(g.pigment);
    let [base_h, base_s, base_l] = rgb_to_hsl(base_rgb[0], base_rgb[1], base_rgb[2]);

    let count = cluster_leaf_count(g.seed);
    let leaves = cluster_leaf_params(g.seed, count);
    let attach_x = size as f64 / 2.0;
    let attach_y = size as f64 - 8.0;
    let leaf_h = size as f64 * 0.55;

    // Draw order: outer leaves (|angle| large) FIRST, central leaves on top.
    let mut order: Vec<(usize, ClusterLeaf)> = leaves.iter().copied().enumerate().collect();
    order.sort_by(|a, b| b.1.angle.abs().partial_cmp(&a.1.angle.abs()).unwrap());

    for (i, leaf) in order {
        let mut variant_rng = Mulberry32::new((g.seed.wrapping_mul(1009)).wrapping_add(i as u32 * 37));
        let vp = leaf_variant_params(&base_params, &mut variant_rng);
        let outline = superformula_outline(&vp, 120);
        let vein_pairs = ((vp.serration_freq * 0.4).round() as i64).clamp(6, 9) as usize;
        let ven = pinnate_venation(&outline, vein_pairs);

        let h = (((base_h + leaf.hue_delta) % 1.0) + 1.0) % 1.0;
        let l = (base_l + leaf.light_delta).clamp(0.05, 0.95);
        let fill = hsl_to_rgb(h, base_s, l);
        let vein = hsl_to_rgb(h, base_s, (l + 0.20).min(1.0));

        let bx = attach_x + leaf.ox * size as f64;
        let by = attach_y - leaf.oy * size as f64;
        draw_leaf(
            &mut canvas,
            bx,
            by,
            leaf_h * leaf.scale,
            leaf.angle,
            fill,
            vein,
            &outline,
            &ven,
        );
    }

    let color = canvas.px;
    let normal = build_normal_map(&color, size, LEAF_NORMAL_STRENGTH);
    LeafTexture {
        color,
        normal,
        size: LEAF_TEX_SIZE,
    }
}

/// Bake the SINGLE-leaf COLOR + NORMAL textures — ONE centered leaf filling the
/// sprite. Ports `makeLeafClusterTexture`'s `leafMode === 'single'` branch
/// (leafTexture.js:863-894): one `drawLeaf` at angle 0, base at bottom-center,
/// drawn at UNIT xScale (the sprite is shape-only; the leaf card carries width).
///
/// ponytail: strictly simpler than [`bake_leaf_cluster`] — no per-leaf variant
/// jitter, no draw-order sort, no `cluster_leaf_params`. It is keyed only on the
/// shape genes + pigment, so it does NOT touch `g.seed` and is deterministic
/// regardless of reseed (the silhouette is the same for the same leaf genes).
pub fn bake_leaf_single(g: &LeafGenes) -> LeafTexture {
    let size = LEAF_TEX_SIZE as usize;
    let mut canvas = Canvas::new(size);

    // Single sprite is shape-only: leaf_width isn't baked here (the card aspect /
    // mesh scale carries width). dryad forces x_scale=1, but at x_scale=1 the
    // height-normalized outline is intrinsically NARROW (normalized half-width
    // ≈0.12 of height at default genes) → a thin blade. ponytail: bake the single
    // sprite at a BROAD fixed x_scale so the silhouette reads as a proper ovate
    // leaf; the per-tree width gene still varies on the card's widthFactor (mesh
    // side), so this only sets the sprite's intrinsic broad shape, not the
    // rendered width. axial_stretch still controls pointier/blunter.
    let mut bp = leaf_base_params(g);
    bp.x_scale = SINGLE_SPRITE_X_SCALE;
    let outline = superformula_outline(&bp, 160); // 160 steps (vs 120 for cluster)

    // Fit the leaf to fill the card: leaf_h bounded by BOTH height AND width so a
    // broad leaf doesn't clip the sprite edges (leafTexture.js:879-884). With the
    // broad SINGLE_SPRITE_X_SCALE above the width bound now actually engages.
    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in &outline {
        if p.x < min_x {
            min_x = p.x;
        }
        if p.x > max_x {
            max_x = p.x;
        }
    }
    let w_extent = (max_x - min_x).max(0.02); // outline y is already 0..1
    let leaf_h = (size as f64 * 0.92).min((size as f64 * 0.92) / (w_extent * 0.5));
    let cx = size as f64 / 2.0;
    let cy = size as f64 - size as f64 * 0.05; // base near the bottom edge

    let vein_pairs = ((bp.serration_freq * 0.4).round() as i64).clamp(6, 9) as usize;
    let ven = pinnate_venation(&outline, vein_pairs);

    let base_rgb = crate::color::pigment_to_color(g.pigment);
    let [base_h, base_s, base_l] = rgb_to_hsl(base_rgb[0], base_rgb[1], base_rgb[2]);
    let fill = hsl_to_rgb(base_h, base_s, base_l);
    let vein = hsl_to_rgb(base_h, base_s, (base_l + 0.20).min(1.0));

    draw_leaf(&mut canvas, cx, cy, leaf_h, 0.0, fill, vein, &outline, &ven);

    let color = canvas.px;
    let normal = build_normal_map(&color, size, LEAF_NORMAL_STRENGTH);
    LeafTexture {
        color,
        normal,
        size: LEAF_TEX_SIZE,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_genes(seed: u32) -> LeafGenes {
        LeafGenes {
            pigment: 0.33,
            leaf_width: 0.5,
            leaf_length: 0.45,
            leaf_tip: 0.4,
            leaf_serration: 0.0,
            leaf_lobing: 0.0,
            leaf_skew: 0.5,
            seed,
        }
    }

    #[test]
    fn deterministic() {
        let a = bake_leaf_cluster(&default_genes(7));
        let b = bake_leaf_cluster(&default_genes(7));
        assert_eq!(a.color, b.color, "color bake must be seed-deterministic");
        assert_eq!(a.normal, b.normal, "normal bake must be seed-deterministic");
        assert_eq!(a.size, LEAF_TEX_SIZE);
        assert_eq!(a.color.len(), (LEAF_TEX_SIZE * LEAF_TEX_SIZE * 4) as usize);
    }

    #[test]
    fn reseed_changes_texture() {
        let a = bake_leaf_cluster(&default_genes(1));
        let b = bake_leaf_cluster(&default_genes(2));
        // A different seed reshuffles the cluster layout → different pixels.
        assert_ne!(a.color, b.color, "reseed must rebuild a different texture");
    }

    #[test]
    fn alpha_has_silhouette() {
        // A real silhouette: some texels fully transparent (outside every leaf),
        // some fully opaque (inside). Both extremes must be present.
        let t = bake_leaf_cluster(&default_genes(3));
        let n = (LEAF_TEX_SIZE * LEAF_TEX_SIZE) as usize;
        let mut saw_zero = false;
        let mut saw_full = false;
        for i in 0..n {
            match t.color[i * 4 + 3] {
                0 => saw_zero = true,
                255 => saw_full = true,
                _ => {}
            }
            if saw_zero && saw_full {
                break;
            }
        }
        assert!(saw_zero, "expected fully-transparent background texels");
        assert!(saw_full, "expected fully-opaque leaf-body texels");
    }

    #[test]
    fn single_deterministic() {
        // Single bake ignores the seed entirely — same genes ⇒ identical sprite,
        // even across different seeds.
        let a = bake_leaf_single(&default_genes(7));
        let b = bake_leaf_single(&default_genes(7));
        let c = bake_leaf_single(&default_genes(99));
        assert_eq!(a.color, b.color, "single bake must be deterministic");
        assert_eq!(a.normal, b.normal, "single normal must be deterministic");
        assert_eq!(a.color, c.color, "single bake must be seed-independent");
        assert_eq!(a.size, LEAF_TEX_SIZE);
        assert_eq!(a.color.len(), (LEAF_TEX_SIZE * LEAF_TEX_SIZE * 4) as usize);
    }

    #[test]
    fn single_alpha_has_silhouette() {
        // One centered leaf: both fully-transparent background and fully-opaque
        // leaf-body texels must be present.
        let t = bake_leaf_single(&default_genes(3));
        let n = (LEAF_TEX_SIZE * LEAF_TEX_SIZE) as usize;
        let mut saw_zero = false;
        let mut saw_full = false;
        for i in 0..n {
            match t.color[i * 4 + 3] {
                0 => saw_zero = true,
                255 => saw_full = true,
                _ => {}
            }
            if saw_zero && saw_full {
                break;
            }
        }
        assert!(saw_zero, "expected fully-transparent background texels");
        assert!(saw_full, "expected fully-opaque leaf-body texels");
    }

    // ── Diagnostic sprite dump (ponytail) ───────────────────────────────────
    // Writes the baked SINGLE + CLUSTER color sprites to PNGs so the leaf shape
    // can be inspected visually (the screenshots can't show the sprite alone).
    // Uses TREE_DEFAULT leaf genes (the no-FLORA_SEED viewer default), so the
    // dumped sprite is exactly what the screenshots render. No `image` crate —
    // a tiny self-contained PNG writer (stored zlib blocks + CRC) keeps the
    // dep-free `enki-flora` clean. Run explicitly:
    //   cargo test -p enki-flora dump_leaf_sprites -- --ignored --nocapture

    /// CRC-32 (IEEE) — for PNG chunk CRCs and the zlib Adler trailer's sibling.
    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFF_FFFF;
        for &b in bytes {
            crc ^= b as u32;
            for _ in 0..8 {
                let mask = (crc & 1).wrapping_neg();
                crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
            }
        }
        !crc
    }

    /// Adler-32 over the raw (uncompressed) zlib payload.
    fn adler32(bytes: &[u8]) -> u32 {
        let (mut a, mut b): (u32, u32) = (1, 0);
        for &x in bytes {
            a = (a + x as u32) % 65521;
            b = (b + a) % 65521;
        }
        (b << 16) | a
    }

    fn png_chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        out.extend_from_slice(kind);
        out.extend_from_slice(data);
        let mut crc_in = Vec::with_capacity(4 + data.len());
        crc_in.extend_from_slice(kind);
        crc_in.extend_from_slice(data);
        out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
    }

    /// Encode RGBA8 `px` (size×size) as a PNG using only stored zlib blocks.
    fn encode_png(px: &[u8], size: u32) -> Vec<u8> {
        // Raw scanlines: each row prefixed with filter byte 0 (None).
        let w = size as usize;
        let mut raw = Vec::with_capacity((w * 4 + 1) * w);
        for y in 0..w {
            raw.push(0u8);
            raw.extend_from_slice(&px[y * w * 4..(y + 1) * w * 4]);
        }
        // zlib stream: 0x78 0x01 header, stored deflate blocks, adler trailer.
        let mut z = vec![0x78u8, 0x01];
        let mut i = 0usize;
        while i < raw.len() {
            let chunk = (raw.len() - i).min(65535);
            let last = if i + chunk >= raw.len() { 1u8 } else { 0u8 };
            z.push(last); // BFINAL, BTYPE=00 (stored)
            z.extend_from_slice(&(chunk as u16).to_le_bytes());
            z.extend_from_slice(&(!(chunk as u16)).to_le_bytes());
            z.extend_from_slice(&raw[i..i + chunk]);
            i += chunk;
        }
        z.extend_from_slice(&adler32(&raw).to_be_bytes());

        let mut out = Vec::new();
        out.extend_from_slice(&[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        let mut ihdr = Vec::with_capacity(13);
        ihdr.extend_from_slice(&size.to_be_bytes());
        ihdr.extend_from_slice(&size.to_be_bytes());
        ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit, RGBA, deflate, no filter, no interlace
        png_chunk(&mut out, b"IHDR", &ihdr);
        png_chunk(&mut out, b"IDAT", &z);
        png_chunk(&mut out, b"IEND", &[]);
        out
    }

    #[test]
    #[ignore = "diagnostic: writes PNGs to disk; run with --ignored"]
    fn dump_leaf_sprites() {
        let g = default_genes(0); // TREE_DEFAULT leaf genes (seed unused by single)
        let single = bake_leaf_single(&g);
        let cluster = bake_leaf_cluster(&g);
        let base = r"C:\Work\Prototype\enki";
        let single_path = format!(r"{base}\flora_leaf_single_sprite.png");
        let cluster_path = format!(r"{base}\flora_leaf_cluster_sprite.png");
        std::fs::write(&single_path, encode_png(&single.color, single.size))
            .expect("write single sprite PNG");
        std::fs::write(&cluster_path, encode_png(&cluster.color, cluster.size))
            .expect("write cluster sprite PNG");
        eprintln!("wrote {single_path}");
        eprintln!("wrote {cluster_path}");
    }

    #[test]
    fn normal_alpha_matches_color() {
        // The normal map carries the color cutout in its alpha channel.
        let t = bake_leaf_cluster(&default_genes(5));
        let n = (LEAF_TEX_SIZE * LEAF_TEX_SIZE) as usize;
        for i in (0..n).step_by(257) {
            assert_eq!(t.normal[i * 4 + 3], t.color[i * 4 + 3]);
        }
        // Flat background (height 0) encodes the +Z normal ≈ (128,128,255).
        // Find a transparent texel and check its encoded normal points up.
        for i in 0..n {
            if t.color[i * 4 + 3] == 0 {
                assert!(t.normal[i * 4 + 2] > 200, "flat region normal should point +Z");
                break;
            }
        }
    }
}

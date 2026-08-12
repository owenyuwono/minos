//! `clouds_advect` — the cloud field, TRANSPORTED (not warped).
//!
//! The shader used to *warp* static noise by `wind × elapsed_time` to fake motion —
//! which inevitably resets (snaps back) or shears at high wind. Instead we ADVECT the
//! cloud field itself: a density scalar `c ∈ [0,1]` on a 512×256 equirect grid, where
//! each tick back-traces every cell along the wind, bilinearly samples the previous
//! field (semi-Lagrangian), injects cloud SHAPE via an evolving clumpy noise gated by
//! moisture, and decays. The shader then samples this field DIRECTLY as the cloud
//! density — so ALL motion is advection: continuous, never resets, never shears.
//! (Advection, not a fluid sim: the wind is given/baked, we only transport.) See
//! `docs/clouds-advection-research.md`.
//!
//! Runs on a background worker (the ocean-FFT / wind-streak pattern); the render
//! thread copies `latest` into a per-FiF storage buffer the shader samples bilinearly.
//!
//! ponytail: plain bilinear SL over-diffuses slowly — the evolving noise source keeps
//! detail alive (IBFV-style) and the shader adds fine sub-grid Worley. 3D-Cartesian
//! back-trace dodges the pole singularity; latitude CLAMPS at the seam (fold +
//! MacCormack are the upgrades). Merging is gentle by design (div-free wind — the
//! convergence upgrade is in `docs/wind-convergence-research.md`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use minos_planet::height::HeightField;
use glam::{DVec3, Vec3};

pub(crate) const GRID_W: usize = 512;
pub(crate) const GRID_H: usize = 256;
const CELLS: usize = GRID_W * GRID_H;
/// Worker tick (Hz). SL is stable at any dt; this just paces CPU.
const TICK_HZ: f32 = 30.0;
/// Cloud-clump size in the injected source noise (higher = smaller clumps).
const NOISE_FREQ: f32 = 10.0;
/// Large-scale weather-region size (lower = bigger cloudy/clear regions). The baked
/// moisture saturates to ~1 over the oceans, so WITHOUT this mask clouds form everywhere
/// moist = the whole planet; this carves big drifting cloudy/clear regions instead.
const WEATHER_FREQ: f32 = 2.5;
/// How fast clumps form/dissipate (slow evolution of the injected detail).
const NOISE_EVOLVE: f32 = 0.04;

/// Live-tunable advection knobs (pushed from the GUI each frame).
#[derive(Clone, Copy)]
pub struct AdvectParams {
    /// Metres/second scale on the baked wind velocity.
    pub wind_speed: f32,
    /// `k_form` — relax toward the climate moisture target (1/s).
    pub form_rate: f32,
    /// `k_decay` — exponential sink so convergence can't ratchet to a solid sheet (1/s).
    pub decay_rate: f32,
}

impl Default for AdvectParams {
    fn default() -> Self {
        // wind_speed is multiplied by the baked unit-wind's speed∈[0,1] (typically well
        // below 1), so the EFFECTIVE drift is speed·wind_speed. At R=50 km, one 256-wide
        // longitude cell is ~1.2 km, so a Courant number of C≈1–5/step (the research-doc
        // CFL target) needs effective |u| in the kilometres-per-second range. 1500 puts a
        // full-speed cell crossing near ~0.8 s; weaker form_rate (τ≈33 s, was 10 s) lets
        // advection transport features faster than the static moisture target re-forms.
        Self { wind_speed: 1500.0, form_rate: 0.15, decay_rate: 0.05 }
    }
}

/// Handle to the advection worker. Holds nothing heavy on the render side.
pub struct CloudAdvect {
    latest: Arc<Mutex<Vec<f32>>>,
    hf_slot: Arc<Mutex<Option<Arc<dyn HeightField>>>>,
    params: Arc<Mutex<AdvectParams>>,
    active: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl CloudAdvect {
    pub fn new(base_radius: f64) -> Self {
        let latest = Arc::new(Mutex::new(vec![0.0f32; CELLS]));
        let hf_slot: Arc<Mutex<Option<Arc<dyn HeightField>>>> = Arc::new(Mutex::new(None));
        let params = Arc::new(Mutex::new(AdvectParams::default()));
        let active = Arc::new(AtomicBool::new(false));
        let running = Arc::new(AtomicBool::new(true));
        let ready = Arc::new(AtomicBool::new(false));
        let worker = {
            let (latest, hf_slot, params, active, running, ready) = (
                Arc::clone(&latest), Arc::clone(&hf_slot), Arc::clone(&params),
                Arc::clone(&active), Arc::clone(&running), Arc::clone(&ready),
            );
            std::thread::Builder::new()
                .name("cloud-advect".into())
                .spawn(move || worker_loop(base_radius as f32, latest, hf_slot, params, active, running, ready))
                .expect("spawn cloud-advect worker")
        };
        Self { latest, hf_slot, params, active, running, ready, worker: Some(worker) }
    }

    /// Hand the worker the planet so it can bake the wind + moisture-target grids.
    pub fn set_source(&self, hf: Arc<dyn HeightField>) {
        *self.hf_slot.lock().unwrap() = Some(hf);
        self.ready.store(false, Ordering::Relaxed); // force a re-bake
    }

    /// Step only while clouds are drawn (idles otherwise).
    pub fn set_active(&self, on: bool) {
        self.active.store(on, Ordering::Relaxed);
    }

    pub fn set_params(&self, p: AdvectParams) {
        *self.params.lock().unwrap() = p;
    }

    /// True once the grids are baked and the coverage field is meaningful.
    pub fn is_ready(&self) -> bool {
        self.ready.load(Ordering::Relaxed)
    }

    /// Copy the latest coverage grid into `out` (len `GRID_W*GRID_H`).
    pub fn copy_latest(&self, out: &mut [f32]) {
        if let Ok(g) = self.latest.lock() {
            out.copy_from_slice(&g);
        }
    }
}

impl Drop for CloudAdvect {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Relaxed);
        if let Some(h) = self.worker.take() {
            let _ = h.join();
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn worker_loop(
    radius: f32,
    latest: Arc<Mutex<Vec<f32>>>,
    hf_slot: Arc<Mutex<Option<Arc<dyn HeightField>>>>,
    params: Arc<Mutex<AdvectParams>>,
    active: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    ready: Arc<AtomicBool>,
) {
    let cell_dirs = precompute_cell_dirs();
    let mut wind_vel = vec![[0.0f32; 3]; CELLS];
    let mut target = vec![0.0f32; CELLS];
    let mut noise_buf = vec![0.0f32; CELLS];
    let mut c_prev = vec![0.0f32; CELLS];
    let mut c_next = vec![0.0f32; CELLS];
    let tick = Duration::from_secs_f32(1.0 / TICK_HZ);
    let mut last = Instant::now();
    let mut elapsed = 0.0f32;
    let mut steps_done = 0u64;

    while running.load(Ordering::Relaxed) {
        if !active.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(40));
            last = Instant::now();
            continue;
        }
        // (Re)bake the wind + moisture-target grids once the planet is available.
        if !ready.load(Ordering::Relaxed) {
            let hf = hf_slot.lock().unwrap().clone();
            match hf {
                Some(hf) => {
                    bake(hf.as_ref(), &cell_dirs, &mut wind_vel, &mut target);
                    // Seed empty — clouds build up from the noisy source + advect.
                    c_prev.fill(0.0);
                    c_next.fill(0.0);
                    latest.lock().unwrap().fill(0.0);
                    ready.store(true, Ordering::Relaxed);
                    elapsed = 0.0;
                    last = Instant::now();
                    let max_wind = wind_vel
                        .iter()
                        .map(|w| (w[0] * w[0] + w[1] * w[1] + w[2] * w[2]).sqrt())
                        .fold(0.0f32, f32::max);
                    let mean_target = target.iter().sum::<f32>() / CELLS as f32;
                    log::info!(
                        "cloud-advect: baked {GRID_W}x{GRID_H}, max|wind|={max_wind:.4} (×speed∈0..1), \
                         mean target coverage={mean_target:.3}"
                    );
                }
                None => {
                    std::thread::sleep(Duration::from_millis(40));
                    continue;
                }
            }
        }

        let now = Instant::now();
        let dt = (now - last).as_secs_f32().clamp(1e-4, 0.2);
        last = now;
        elapsed += dt;
        // Injected cloud-shape source: evolving clumpy noise, gated by moisture in
        // `step`. This is what gives the ADVECTED field real cloud shape (not just a
        // smooth placement mask) — the shader samples this field directly, so motion =
        // pure advection (continuous, never resets, never shears).
        let ez = elapsed * NOISE_EVOLVE;
        for idx in 0..CELLS {
            let d = cell_dirs[idx];
            // Large-scale weather mask: big, slowly-drifting cloudy/clear REGIONS so the
            // (mostly-moist) planet isn't blanketed. ~1/3 of the sky is "active weather".
            let big = fbm(Vec3::new(
                d.x * WEATHER_FREQ,
                d.y * WEATHER_FREQ,
                d.z * WEATHER_FREQ + ez * 0.5,
            ));
            let region = smoothstep(0.48, 0.72, big);
            // Medium-scale clumps within the active regions.
            let clump = fbm(Vec3::new(d.x * NOISE_FREQ, d.y * NOISE_FREQ, d.z * NOISE_FREQ + ez));
            noise_buf[idx] = region * smoothstep(0.35, 0.65, clump);
        }
        // Gentle unsteady gust so the flow never settles to a static steady state.
        let gust = 0.6 * (elapsed * 0.08).sin();
        let p = *params.lock().unwrap();
        step(radius, &cell_dirs, &wind_vel, &target, &noise_buf, &c_prev, &mut c_next, dt, gust, &p);
        std::mem::swap(&mut c_prev, &mut c_next);
        latest.lock().unwrap().copy_from_slice(&c_prev);

        // Periodic diagnostic (~every 3 s): is the field actually evolving?
        steps_done += 1;
        if steps_done % 90 == 0 {
            let mean = c_prev.iter().sum::<f32>() / CELLS as f32;
            let max_d = c_prev
                .iter()
                .zip(c_next.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            log::info!("cloud-advect: mean coverage={mean:.3}, max step delta={max_d:.4}");
        }

        let spent = now.elapsed();
        if spent < tick {
            std::thread::sleep(tick - spent);
        }
    }
}

/// Equirect cell-centre unit directions (dir = (cosLat·sinLon, sinLat, cosLat·cosLon),
/// matching `bake_grids` and the shader's `grid_lerp`).
fn precompute_cell_dirs() -> Vec<Vec3> {
    let mut v = Vec::with_capacity(CELLS);
    for j in 0..GRID_H {
        let lat = ((j as f64 + 0.5) / GRID_H as f64 - 0.5) * std::f64::consts::PI;
        let (sl, cl) = lat.sin_cos();
        for i in 0..GRID_W {
            let lon = ((i as f64 + 0.5) / GRID_W as f64 - 0.5) * std::f64::consts::TAU;
            let (so, co) = lon.sin_cos();
            v.push(DVec3::new(cl * so, sl, cl * co).normalize().as_vec3());
        }
    }
    v
}

fn bake(hf: &dyn HeightField, cell_dirs: &[Vec3], wind_vel: &mut [[f32; 3]], target: &mut [f32]) {
    for idx in 0..CELLS {
        let dir = cell_dirs[idx].as_dvec3();
        let w = hf.wind_at(dir);
        wind_vel[idx] = [w.x * w.speed, w.y * w.speed, w.z * w.speed];
        target[idx] = smoothstep(0.2, 0.7, hf.moisture(dir));
    }
}

/// One semi-Lagrangian step: backtrace (3D on the sphere) → bilinear → source/decay.
/// `gust` rotates the wind about each cell's local up — an UNSTEADY component so the
/// flow (and thus the coverage pattern) never settles into a static steady state.
#[allow(clippy::too_many_arguments)]
fn step(
    radius: f32,
    cell_dirs: &[Vec3],
    wind_vel: &[[f32; 3]],
    target: &[f32],
    noise: &[f32],
    c_prev: &[f32],
    c_next: &mut [f32],
    dt: f32,
    gust: f32,
    p: &AdvectParams,
) {
    let inv_r = 1.0 / radius;
    let (sg, cg) = gust.sin_cos();
    for idx in 0..CELLS {
        let dir = cell_dirs[idx];
        let wv = wind_vel[idx];
        // Rotate the tangent wind about the local up by `gust` (Rodrigues; wind ⊥ dir).
        let w = Vec3::new(wv[0], wv[1], wv[2]);
        let w_rot = w * cg + dir.cross(w) * sg;
        let u = w_rot * p.wind_speed; // m/s world tangent
        // Backward tangent step on the unit sphere, renormalize (pole-safe — no
        // lon/lat singularity). For our dt·|u|/R this stays well under one cell.
        let dep = dir - u * (dt * inv_r);
        let p_dep = dep.normalize_or_zero();
        let p_dep = if p_dep == Vec3::ZERO { dir } else { p_dep };
        let c_adv = bilinear(c_prev, p_dep);
        // Source = inject cloud SHAPE where humid (target) AND the evolving clump noise
        // is high, saturating at 1; decay everywhere. The advection (above) transports
        // this shaped field continuously → clouds flow + form/dissipate, never reset.
        let potential = target[idx] * noise[idx];
        let s = p.form_rate * potential * (1.0 - c_adv) - p.decay_rate * c_adv;
        c_next[idx] = (c_adv + dt * s).clamp(0.0, 1.0);
    }
}

/// Bilinear equirect lookup (longitude wraps, latitude clamps) — matches the
/// shader's `grid_lerp` so CPU advection and GPU sampling agree.
fn bilinear(grid: &[f32], dir: Vec3) -> f32 {
    let lon = dir.x.atan2(dir.z);
    let lat = dir.y.clamp(-1.0, 1.0).asin();
    let u = lon / std::f32::consts::TAU + 0.5;
    let v = lat / std::f32::consts::PI + 0.5;
    let fx = u * GRID_W as f32 - 0.5;
    let fy = v * GRID_H as f32 - 0.5;
    let x0 = fx.floor();
    let y0 = fy.floor();
    let tx = fx - x0;
    let ty = fy - y0;
    let gw = GRID_W as i32;
    let gh = GRID_H as i32;
    let ix0 = ((x0 as i32 % gw) + gw) % gw;
    let ix1 = (ix0 + 1) % gw;
    let iy0 = (y0 as i32).clamp(0, gh - 1);
    let iy1 = (y0 as i32 + 1).clamp(0, gh - 1);
    let at = |ix: i32, iy: i32| grid[iy as usize * GRID_W + ix as usize];
    let a = at(ix0, iy0) * (1.0 - tx) + at(ix1, iy0) * tx;
    let b = at(ix0, iy1) * (1.0 - tx) + at(ix1, iy1) * tx;
    a * (1.0 - ty) + b * ty
}

fn smoothstep(a: f32, b: f32, x: f32) -> f32 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

// ── Value-noise fBm (CPU, for the injected cloud-shape source) ──────────────────

fn hash13(p: Vec3) -> f32 {
    let mut q = (p * 0.1031).fract();
    q += Vec3::splat(q.dot(Vec3::new(q.z, q.y, q.x) + Vec3::splat(31.32)));
    ((q.x + q.y) * q.z).fract()
}

fn vnoise(x: Vec3) -> f32 {
    let i = x.floor();
    let f = x.fract();
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let c = |dx: f32, dy: f32, dz: f32| hash13(i + Vec3::new(dx, dy, dz));
    let x00 = lerp(c(0.0, 0.0, 0.0), c(1.0, 0.0, 0.0), u.x);
    let x10 = lerp(c(0.0, 1.0, 0.0), c(1.0, 1.0, 0.0), u.x);
    let x01 = lerp(c(0.0, 0.0, 1.0), c(1.0, 0.0, 1.0), u.x);
    let x11 = lerp(c(0.0, 1.0, 1.0), c(1.0, 1.0, 1.0), u.x);
    lerp(lerp(x00, x10, u.y), lerp(x01, x11, u.y), u.z)
}

fn fbm(p0: Vec3) -> f32 {
    // 2 octaves — cheap (this runs per cell per worker tick over the whole grid); the
    // shader adds fine sub-grid detail on top.
    let mut p = p0;
    let mut amp = 0.5;
    let mut sum = 0.0;
    for _ in 0..2 {
        sum += amp * vnoise(p);
        p *= 2.03;
        amp *= 0.5;
    }
    sum / 0.75
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: f32 = 50_000.0;

    fn zero_p(wind: f32, form: f32, decay: f32) -> AdvectParams {
        AdvectParams { wind_speed: wind, form_rate: form, decay_rate: decay }
    }

    #[test]
    fn zero_wind_zero_source_is_identity() {
        let dirs = precompute_cell_dirs();
        let wind = vec![[0.0f32; 3]; CELLS];
        let target = vec![0.0f32; CELLS];
        let mut c_prev = vec![0.0f32; CELLS];
        let bump = 100 * GRID_W + 50;
        c_prev[bump] = 1.0;
        let mut c_next = vec![0.0f32; CELLS];
        step(R, &dirs, &wind, &target, &vec![1.0f32; CELLS], &c_prev, &mut c_next, 0.1, 0.0, &zero_p(0.0, 0.0, 0.0));
        // Backtrace to the same cell centre → field unchanged.
        assert!((c_next[bump] - 1.0).abs() < 1e-4, "bump should persist, got {}", c_next[bump]);
        assert!(c_next[bump + 2] < 1e-4, "neighbour should stay empty");
    }

    #[test]
    fn uniform_field_stays_uniform() {
        // A convex combination of equal neighbours is that value — advection of a
        // constant field is the identity, for any wind.
        let dirs = precompute_cell_dirs();
        let wind = vec![[0.3f32, 0.0, 0.1]; CELLS];
        let target = vec![0.0f32; CELLS];
        let c_prev = vec![0.5f32; CELLS];
        let mut c_next = vec![0.0f32; CELLS];
        step(R, &dirs, &wind, &target, &vec![1.0f32; CELLS], &c_prev, &mut c_next, 0.1, 0.0, &zero_p(120.0, 0.0, 0.0));
        for &v in &c_next {
            assert!((v - 0.5).abs() < 1e-4, "uniform field must stay uniform, got {v}");
        }
    }

    #[test]
    fn bump_drifts_downwind() {
        // Pins the load-bearing semi-Lagrangian pieces the other three tests miss:
        // the backtrace DIRECTION (`dir - u·dt/R`) and the dt·|u|/R SCALE. A single
        // bump in an otherwise-zero, NON-uniform field is carried one cell downwind by
        // a +lon (eastward) wind. Fails if the sign flips (`+ u`, upwind) or `inv_r`
        // is dropped (no/huge motion). Source off so only advection moves the field.
        let dirs = precompute_cell_dirs();
        // Per-cell unit east tangent = ∂dir/∂lon normalized.
        // dir = (cosLat·sinLon, sinLat, cosLat·cosLon) → ∂/∂lon = (cosLat·cosLon, 0,
        // −cosLat·sinLon); for cosLat≠0 its norm is cosLat, so the unit tangent is
        // (cosLon, 0, −sinLon).
        let mut wind = vec![[0.0f32; 3]; CELLS];
        for j in 0..GRID_H {
            for i in 0..GRID_W {
                let lon = ((i as f64 + 0.5) / GRID_W as f64 - 0.5) * std::f64::consts::TAU;
                let (so, co) = lon.sin_cos();
                let t = DVec3::new(co, 0.0, -so).normalize().as_vec3();
                wind[j * GRID_W + i] = [t.x, t.y, t.z];
            }
        }
        let target = vec![0.0f32; CELLS];
        let mut c_prev = vec![0.0f32; CELLS];
        let (row, col) = (64usize, 128usize);
        c_prev[row * GRID_W + col] = 1.0;
        let mut c_next = vec![0.0f32; CELLS];
        // wind_speed chosen so dt·|u|/R == one longitude cell (TAU/GRID_W rad).
        let dt = 0.1f32;
        let ws = R * (std::f32::consts::TAU / GRID_W as f32) / dt;
        step(R, &dirs, &wind, &target, &vec![1.0f32; CELLS], &c_prev, &mut c_next, dt, 0.0, &zero_p(ws, 0.0, 0.0));
        // Backtrace steps UPWIND, so cell `col` samples its empty (col−1) neighbour and
        // the bump arrives one cell downwind at `col+1`.
        assert!(
            c_next[row * GRID_W + col] < 0.5,
            "bump should leave its origin cell, got {}",
            c_next[row * GRID_W + col]
        );
        assert!(
            c_next[row * GRID_W + col + 1] > 0.5,
            "bump should arrive one cell downwind (col+1), got {}",
            c_next[row * GRID_W + col + 1]
        );
    }

    #[test]
    fn source_relaxes_toward_target() {
        let dirs = precompute_cell_dirs();
        let wind = vec![[0.0f32; 3]; CELLS];
        let target = vec![1.0f32; CELLS];
        let c_prev = vec![0.0f32; CELLS];
        let mut c_next = vec![0.0f32; CELLS];
        // c_next = 0 + dt·(form·(1−0) − 0) = 0.1
        step(R, &dirs, &wind, &target, &vec![1.0f32; CELLS], &c_prev, &mut c_next, 0.1, 0.0, &zero_p(0.0, 1.0, 0.0));
        assert!((c_next[0] - 0.1).abs() < 1e-3, "expected 0.1, got {}", c_next[0]);
    }
}

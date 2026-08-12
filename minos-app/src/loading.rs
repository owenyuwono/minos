//! Async startup loader.
//!
//! Builds the terrain height field on a worker thread while the main loop renders
//! a progress bar — removing the multi-second blank window at startup. The shared
//! `Arc<dyn HeightField>` is then shared by the voxel terrain / `PlanetView`, so
//! it's built once.
//!
//! Only `Send` data crosses the thread boundary: the finished `Arc<dyn HeightField>`.
//! The (cheap) terrain-view construction + GPU upload happen back on the main thread
//! once the result arrives.

use std::sync::mpsc::{channel, Receiver};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use minos_planet::climate::ClimateParams;
use minos_planet::height::HeightField;

/// Progress snapshot shown by the loading screen.
#[derive(Clone)]
pub struct LoadProgress {
    pub fraction: f32,
    pub message: String,
}

/// Wall-clock breakdown of the load, shown in a one-time popup when it finishes.
/// All milliseconds.
#[derive(Debug, Clone, Copy, Default)]
pub struct LoadTimings {
    pub total_ms: f64,
    pub heightfield_ms: f64,
}

/// Parameters handed to the loader thread (all `Send`).
#[allow(dead_code)] // radius/height_scale are read only by the voxel terrain build.
pub struct LoadParams {
    pub seed: u32,
    pub use_tectonics: bool,
    pub plate_count: usize,
    pub arc_density: f64,
    pub hotspot_count: u32,
    pub hotspot_intensity: f64,
    pub climate: ClimateParams,
    pub radius: f64,
    pub height_scale: f64,
}

/// Result produced by the loader thread.
pub struct LoadOutput {
    /// The shared height field (used by the voxel terrain / `PlanetView`).
    pub hf: Arc<dyn HeightField>,
    /// Per-stage load timing for the completion popup.
    pub timings: LoadTimings,
}

/// Handle to an in-progress background load.
pub struct Loader {
    progress: Arc<Mutex<LoadProgress>>,
    rx: Receiver<LoadOutput>,
    _handle: JoinHandle<()>,
}

impl Loader {
    /// Spawn the background load and return immediately.
    pub fn spawn(params: LoadParams) -> Self {
        let progress = Arc::new(Mutex::new(LoadProgress {
            fraction: 0.0,
            message: "Initializing…".to_string(),
        }));
        let (tx, rx) = channel();
        let prog = Arc::clone(&progress);

        let handle = std::thread::spawn(move || {
            let set = |f: f32, m: &str| {
                if let Ok(mut p) = prog.lock() {
                    p.fraction = f;
                    p.message = m.to_string();
                }
            };

            set(0.05, "Building terrain heightfield…");
            // One clock for the whole load: `elapsed()` after the heightfield = HF
            // time; after the bake = total. Both feed the load-stats popup.
            let t_load = std::time::Instant::now();
            let hf: Arc<dyn HeightField> = if params.use_tectonics {
                use minos_planet::sampler::{TectonicHeightField, TectonicHeightFieldParams};
                Arc::new(TectonicHeightField::new(TectonicHeightFieldParams {
                    seed: params.seed,
                    plate_count: params.plate_count,
                    arc_density: params.arc_density,
                    hotspot_count: params.hotspot_count,
                    hotspot_intensity: params.hotspot_intensity,
                    // ponytail: composition not yet exposed in the GUI/LoadParams — pass the
                    // 0.5 default. Wire a slider through LoadParams when Phase-1 hardness needs it.
                    composition: 0.5,
                    climate: params.climate,
                }))
            } else {
                use minos_planet::noise::Noise3D;
                use minos_planet::simple_height::SimpleHeightField;
                Arc::new(SimpleHeightField { noise: Noise3D::new(params.seed) })
            };

            let heightfield_ms = t_load.elapsed().as_secs_f64() * 1e3;
            log::info!("load: heightfield built in {:.0}ms", heightfield_ms);

            set(0.95, "Uploading to GPU…");
            let timings = LoadTimings {
                total_ms: t_load.elapsed().as_secs_f64() * 1e3,
                heightfield_ms,
            };
            let _ = tx.send(LoadOutput {
                hf,
                timings,
            });
        });

        Self { progress, rx, _handle: handle }
    }

    /// Current progress snapshot for the loading screen.
    pub fn progress(&self) -> LoadProgress {
        self.progress
            .lock()
            .map(|p| p.clone())
            .unwrap_or(LoadProgress { fraction: 0.0, message: String::new() })
    }

    /// Returns the result once the background work has finished (non-blocking).
    pub fn poll(&self) -> Option<LoadOutput> {
        self.rx.try_recv().ok()
    }
}

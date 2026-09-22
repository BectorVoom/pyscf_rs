//! The K-09 image batch must never silently fall back to one image per launch.
//!
//! B-02 tied the batch budget to the AO accumulator. With
//! `accumulator_bytes = 2·nkpts·n·8` and `block_bytes = n·8`, the `1/2`
//! fraction reduces that term to exactly `nkpts` — so it binds hardest where
//! memory pressure is LOWEST, and at gamma (`nkpts = 1`) it asked for capacity
//! 1, the pre-K-09 per-image path. Nothing caught it: every bit-identity gate
//! passes at any capacity, because batching is bit-exact by construction. It
//! showed up only as a 2.3× wall-clock regression.
//!
//! Two gates here, in order of what they catch:
//!
//! 1. [`the_batch_floor_holds_at_every_k_point_count`] — DETERMINISTIC, and the
//!    one to trust. It reads the capacity the driver would pick straight out of
//!    [`image_batch_capacity`], so it cannot be flaky and it names the shape
//!    that broke.
//! 2. [`the_default_batch_beats_the_per_image_fallback`] — a wall-clock ratio
//!    against an explicitly pinned capacity 1, in ONE process. It is the
//!    end-to-end statement (the default really is faster than the fallback) and
//!    exists because a future change could keep the capacity high and still lose
//!    the batching some other way. Its margin is deliberately loose.

use std::sync::Mutex;
use std::time::Instant;

use pyscf_pbc_gto::{
    AO_IMAGE_BATCH_MIN, Cell, UniformGrids, eval_ao_kpts, image_batch_capacity, make_kpts_default,
};

mod common;

/// `PYSCF_PBC_AO_IMAGE_BATCH` is process-global and read per call, while the
/// harness runs the bodies of one binary concurrently — the same race that made
/// `hcore_fused.rs` fail at `--test-threads=4` and pass at 2.
static BATCH_SWITCH: Mutex<()> = Mutex::new(());

fn with_batch<T>(pinned: Option<&str>, f: impl FnOnce() -> T) -> T {
    const KEY: &str = "PYSCF_PBC_AO_IMAGE_BATCH";
    // A poisoned lock means another body panicked mid-test; this one sets the
    // switch again immediately, so taking the guard anyway is correct.
    let _guard = BATCH_SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    let previous = std::env::var(KEY).ok();
    // SAFETY-equivalent note: the guard makes this the only writer, and every
    // reader runs inside some `with_batch` body.
    match pinned {
        Some(v) => unsafe { std::env::set_var(KEY, v) },
        None => unsafe { std::env::remove_var(KEY) },
    }
    let out = f();
    match previous {
        Some(v) => unsafe { std::env::set_var(KEY, v) },
        None => unsafe { std::env::remove_var(KEY) },
    }
    out
}

/// What the driver passes: one image's AO block, and the two device planes.
fn capacity_for(nkpts: usize, nao: usize, ngrids: usize, comp: usize) -> usize {
    let n = comp * ngrids * nao;
    with_batch(None, || image_batch_capacity(n, 2 * nkpts * n * 8))
}

/// The floor holds at every k-point count, and hardest at gamma.
///
/// `nkpts = 1` is the case that regressed: without the floor the accumulator
/// term asks for exactly 1.
#[test]
fn the_batch_floor_holds_at_every_k_point_count() {
    // Realistic single-k and small-k shapes: diamond gth-szv (nao 8) and
    // gth-dzvp (nao 26) on meshes this tree actually runs.
    for &(nao, ngrids) in &[(8usize, 1331usize), (8, 29791), (26, 29791), (26, 68921)] {
        for nkpts in [1usize, 2, 4, 8, 27] {
            let cap = capacity_for(nkpts, nao, ngrids, 1);
            // Every shape here has a block well under the 256 MiB budget, so
            // the absolute cap is far above the floor and the floor is what
            // binds.
            assert!(
                cap >= AO_IMAGE_BATCH_MIN,
                "nkpts={nkpts} nao={nao} ngrids={ngrids}: capacity {cap} is \
                 below the floor {AO_IMAGE_BATCH_MIN}"
            );
            assert!(
                cap > 1,
                "nkpts={nkpts} nao={nao} ngrids={ngrids}: capacity fell to the \
                 per-image fallback (1); the accumulator term reduces to nkpts, \
                 so it must be floored by AO_IMAGE_BATCH_MIN"
            );
        }
    }
}

/// The floor may not push the batch past the absolute 256 MiB budget: on a
/// block so large that the budget allows fewer than [`AO_IMAGE_BATCH_MIN`]
/// images, the budget wins.
#[test]
fn the_absolute_budget_outranks_the_floor() {
    // One block of 64 MiB (8 M reals): 256 MiB / 64 MiB = 4 images, under the
    // floor of 8.
    let n = 8 * 1024 * 1024;
    let cap = with_batch(None, || image_batch_capacity(n, 2 * 27 * n * 8));
    assert_eq!(
        cap, 4,
        "the 256 MiB budget must outrank the floor on a huge block"
    );
}

/// An explicit `PYSCF_PBC_AO_IMAGE_BATCH` still wins over both.
#[test]
fn an_explicit_pin_still_overrides_the_floor() {
    let n = 8 * 1331;
    let pinned = with_batch(Some("1"), || image_batch_capacity(n, 2 * n * 8));
    assert_eq!(
        pinned, 1,
        "an explicit 1 must still select the per-image path"
    );
}

/// End-to-end: the default batch beats a pinned capacity 1 at gamma.
///
/// Ratio inside ONE process, so machine speed cancels. The margin is loose
/// (1.25×) because this is wall clock on a shared box; the measured gap on the
/// regression was 2.3×, and a default that had silently collapsed to 1 would
/// score ~1.0.
#[test]
fn the_default_batch_beats_the_per_image_fallback() {
    let cell: Cell = common::systems::diamond();
    let kpts = make_kpts_default(&cell, [1, 1, 1]).expect("gamma k-mesh");
    assert_eq!(kpts.len(), 1, "this gate is about the gamma case");
    let mesh = [21, 21, 21];
    let coords = UniformGrids::build(&cell, Some(mesh))
        .expect("uniform grid")
        .coords;

    let run = |pinned: Option<&str>| {
        with_batch(pinned, || {
            // One untimed call first: the first evaluation in the process pays
            // JIT and first-allocation costs that would swamp the ratio.
            let _ = eval_ao_kpts(&cell, "GTOval_sph", &coords, &kpts).expect("warm");
            let t = Instant::now();
            let out = eval_ao_kpts(&cell, "GTOval_sph", &coords, &kpts).expect("eval");
            let dt = t.elapsed().as_secs_f64();
            std::hint::black_box(&out);
            dt
        })
    };
    // Fallback first, so the default arm cannot be the one paying cold costs.
    let fallback = run(Some("1"));
    let default = run(None);
    let ratio = fallback / default;
    assert!(
        ratio >= 1.25,
        "the default batch ({default:.3} s) must beat the pinned per-image \
         fallback ({fallback:.3} s) by 1.25x; got {ratio:.2}x. A ratio near 1.0 \
         means the capacity formula has collapsed to 1 again — see \
         AO_IMAGE_BATCH_MIN."
    );
}

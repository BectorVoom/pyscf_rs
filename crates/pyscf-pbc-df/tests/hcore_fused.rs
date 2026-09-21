//! K-14f — the fused, on-device local contraction behind `get_hcore`.
//!
//! `Fftdf::local_vmat` has two routes that must agree BIT FOR BIT, because one
//! of them is what every existing `get_hcore` gate was written against:
//!
//! * `PYSCF_PBC_HCORE_FUSE=0` — evaluate the whole `nkpts x nao x ngrids`
//!   complex AO table, read it back, cache it, reduce it on the host;
//! * `PYSCF_PBC_HCORE_FUSE=1` — evaluate and contract on the device, the table
//!   never reaching the host;
//! * unset / `auto` (the default) — the first when the table would be cached
//!   and reused, the second when it would be built only to be thrown away.
//!
//! Verified here: the routes agree bitwise for `get_pp`, `get_nuc` and
//! `get_hcore`, at gamma and away from it; `auto` picks on `max_memory` alone
//! and gives the same bits either way; a warm AO cache takes the host route
//! under every setting; and the fused route deliberately leaves the AO cache
//! cold, which is the whole point — it must not have materialised the table
//! behind the caller's back.
//!
//! The switch is process-global and read per call, while the harness runs the
//! bodies of one test binary concurrently — so every body goes through
//! [`with_fuse`], which holds a process-wide lock for its whole duration and
//! restores the previous setting on the way out.

mod common;

use common::{diamond, he_all_electron};
use pyscf_algebra::CTensor;
use pyscf_pbc_df::{Fftdf, PeriodicDf, get_hcore};
use pyscf_pbc_gto::make_kpts_default;

/// Small enough to run unignored in CI; the identity under test is exact at
/// every mesh, so there is nothing to gain from a converged one.
const MESH: [usize; 3] = [11, 11, 11];

/// Serialises every body below.
///
/// `PYSCF_PBC_HCORE_FUSE` is process-global and read per call, while the test
/// harness runs the bodies of one binary CONCURRENTLY — so without this lock a
/// test that pins the switch to `1` flips it under a test that is measuring
/// `auto`, and the `auto` case fails with the fused route's cache count. That
/// is exactly how this file first failed at `--test-threads=4` (and passed at
/// 2, which is what makes the race worth a lock rather than a comment).
static FUSE_SWITCH: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Run `f` with `PYSCF_PBC_HCORE_FUSE` pinned, restoring it afterwards.
///
/// `mode` is the switch's own vocabulary: `"0"` never, `"1"` always, `"auto"`
/// the default routing.
fn with_fuse<T>(mode: &str, f: impl FnOnce() -> T) -> T {
    const KEY: &str = "PYSCF_PBC_HCORE_FUSE";
    // A poisoned lock means another body panicked mid-test; the switch is then
    // whatever that body left, and this one sets it again immediately, so
    // taking the guard anyway is correct and keeps one failure from cascading.
    let _guard = FUSE_SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    let previous = std::env::var(KEY).ok();
    // SAFETY-equivalent note: the guard above makes this the only writer, and
    // every reader of the switch runs inside some `with_fuse` body.
    unsafe { std::env::set_var(KEY, mode) };
    let out = f();
    match previous {
        Some(v) => unsafe { std::env::set_var(KEY, v) },
        None => unsafe { std::env::remove_var(KEY) },
    }
    out
}

/// `HOST` never fuses, `FUSED` always does — the two arms every bit-identity
/// test below compares.
const HOST: &str = "0";
const FUSED: &str = "1";
const AUTO: &str = "auto";

fn assert_bitwise(got: &[CTensor], want: &[CTensor], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: k-point count");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.re.len(), w.re.len(), "{what}: re length at k={k}");
        assert_eq!(g.im.len(), w.im.len(), "{what}: im length at k={k}");
        for (i, (a, b)) in g.re.iter().zip(&w.re).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{what}: re[{k}][{i}] fused {a:e} vs host {b:e}"
            );
        }
        for (i, (a, b)) in g.im.iter().zip(&w.im).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{what}: im[{k}][{i}] fused {a:e} vs host {b:e}"
            );
        }
    }
}

/// `get_pp` — the pseudopotential cell's local half, the path `get_hcore`
/// takes on every periodic DFT run in this tree.
#[test]
fn get_pp_fused_matches_the_host_route_bitwise() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("2x2x2 k-mesh");
    // A FRESH builder per arm: a shared one would serve the second arm the
    // first arm's cached AO table and compare the host route with itself.
    let host = with_fuse(HOST, || {
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
        df.get_pp(&kpts).expect("get_pp host")
    });
    let fused = with_fuse(FUSED, || {
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
        df.get_pp(&kpts).expect("get_pp fused")
    });
    assert_bitwise(&fused, &host, "get_pp");
}

/// `get_nuc` — the all-electron cell's local half. Its bit-exact upstream
/// route (`eval_ao_kpts_upstream`) is a different branch and is untouched;
/// this exercises the port's own one.
#[test]
fn get_nuc_fused_matches_the_host_route_bitwise() {
    let cell = he_all_electron();
    let kpts = make_kpts_default(&cell, [2, 1, 1]).expect("2x1x1 k-mesh");
    let host = with_fuse(HOST, || {
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
        df.get_nuc(&kpts).expect("get_nuc host")
    });
    let fused = with_fuse(FUSED, || {
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
        df.get_nuc(&kpts).expect("get_nuc fused")
    });
    assert_bitwise(&fused, &host, "get_nuc");
}

/// The assembled `T + V_pp` a periodic SCF actually consumes.
#[test]
fn get_hcore_fused_matches_the_host_route_bitwise() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("2x2x2 k-mesh");
    let host = with_fuse(HOST, || {
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
        get_hcore(&df, &kpts).expect("get_hcore host")
    });
    let fused = with_fuse(FUSED, || {
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
        get_hcore(&df, &kpts).expect("get_hcore fused")
    });
    assert_bitwise(&fused, &host, "get_hcore");
}

/// A gamma-only k-list: the case where the imaginary AO plane is dropped, and
/// so the one where a `* 0.0` scaling instead of a literal overwrite would
/// show up as a sign-of-zero difference.
#[test]
fn gamma_only_is_bitwise_identical_and_exactly_real() {
    let cell = diamond();
    let kpts = [[0.0_f64; 3]];
    let host = with_fuse(HOST, || {
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
        df.get_pp(&kpts).expect("get_pp host")
    });
    let fused = with_fuse(FUSED, || {
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
        df.get_pp(&kpts).expect("get_pp fused")
    });
    assert_bitwise(&fused, &host, "get_pp at gamma");
    for (i, v) in fused[0].im.iter().enumerate() {
        assert_eq!(
            v.to_bits(),
            0.0_f64.to_bits(),
            "V_pp at gamma must be exactly +0.0 imaginary, im[{i}] = {v:e}"
        );
    }
}

/// The fused route must NOT populate the AO cache — a table it materialised
/// behind the caller's back would put the bytes back that the whole change is
/// about removing. The host route, by contrast, does cache it.
#[test]
fn the_fused_route_leaves_the_ao_cache_cold() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("2x2x2 k-mesh");

    let fused_cached = with_fuse(FUSED, || {
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
        let _ = df.get_pp(&kpts).expect("get_pp fused");
        df.ao_cache_len()
    });
    assert_eq!(
        fused_cached, 0,
        "the fused route materialised and cached the AO table"
    );

    let host_cached = with_fuse(HOST, || {
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
        let _ = df.get_pp(&kpts).expect("get_pp host");
        df.ao_cache_len()
    });
    assert_eq!(
        host_cached, 1,
        "the host route is supposed to cache the table it built"
    );
}

/// With the table already cached — the state an SCF reaches as soon as
/// `get_j`/`get_k` have run — both settings take the host route, and the
/// answer is the same bits again.
#[test]
fn a_warm_cache_takes_the_host_route_under_either_setting() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("2x2x2 k-mesh");
    let warm = |mode: &str| {
        with_fuse(mode, || {
            let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
            let _ = df.ao_kpts(&kpts).expect("warm the AO cache");
            let v = df.get_pp(&kpts).expect("get_pp");
            (v, df.ao_cache_len())
        })
    };
    let (fused, fused_cached) = warm(FUSED);
    let (host, host_cached) = warm(HOST);
    assert_eq!(
        fused_cached, 1,
        "the warm cache must survive the fused call"
    );
    assert_eq!(host_cached, 1);
    assert_bitwise(&fused, &host, "get_pp with a warm cache");
}

/// The DEFAULT routing. `auto` fuses exactly when the AO table would not be
/// admitted to the cache, so the same call flips route on `max_memory` alone —
/// and gives the same bits either way.
#[test]
fn auto_routes_on_whether_the_table_would_be_cached() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("2x2x2 k-mesh");
    let run = |max_memory: f64| {
        with_fuse(AUTO, || {
            let mut df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
            df.max_memory = max_memory;
            let v = df.get_pp(&kpts).expect("get_pp");
            (v, df.ao_cache_len())
        })
    };
    // Roomy: the table is worth keeping, so `auto` takes the host route and
    // leaves it cached for `get_j`/`get_k` — no extra cold AO pass.
    let (roomy, roomy_cached) = run(64_000.0);
    assert_eq!(
        roomy_cached, 1,
        "with room to cache, auto must take the caching host route"
    );
    // Tight: the table would be built, reduced and thrown away, so `auto`
    // fuses and never materialises it.
    let (tight, tight_cached) = run(1.0);
    assert_eq!(
        tight_cached, 0,
        "with no room to cache, auto must take the fused route"
    );
    assert_bitwise(&tight, &roomy, "get_pp under auto routing");

    let host = with_fuse(HOST, || {
        let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
        df.get_pp(&kpts).expect("get_pp host")
    });
    assert_bitwise(&roomy, &host, "auto (roomy) vs pinned host");
}

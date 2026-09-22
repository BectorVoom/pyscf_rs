//! B-03d — the grid-blocked local contraction behind `get_hcore`.
//!
//! `eval_ao_kpts_local_vmat_blocked` must reproduce `eval_ao_kpts_local_vmat`
//! BIT FOR BIT at every block size, because B-04's decision — blocking ON by
//! default at a fixed `BLK = 8192` — is only valid if blocking is invisible in
//! the result.
//!
//! Why bitwise is achievable here — and only here: the contraction accumulates
//! `sr += term(g)` in a serial loop over `g`, one lane per `(k, p, q)`, so a
//! carried accumulator over `[0,b₁) ∪ [b₁,b₂)` is the identical sequence of
//! IEEE additions as the whole-grid run, provided blocks run in increasing `g`
//! order, serially, into one carried output. Summing blocks independently and
//! merging afterwards would reassociate and move the last bits — that is STOP
//! condition §0.5.6, and this gate must NOT be relaxed to a tolerance.
//!
//! The `PYSCF_PBC_AO_GRID_BLOCK` arms below go through `get_pp` under
//! `PYSCF_PBC_HCORE_FUSE=1` (the fused route is the only one that reads the
//! block switch); the direct arms call the blocked driver with an explicit
//! `blk` and are meaningful even before B-03e wires the switch.

mod common;

use common::{diamond, he_all_electron};
use pyscf_algebra::CTensor;
use pyscf_pbc_df::{Fftdf, PeriodicDf};
use pyscf_pbc_gto::{eval_ao_kpts_local_vmat, eval_ao_kpts_local_vmat_blocked, make_kpts_default};

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

/// Serialises the block-size arms below.
///
/// `PYSCF_PBC_AO_GRID_BLOCK` is read per call in `eval_gto.rs`, so the same T5
/// race applies: one body pinning `128` would flip it under a body measuring
/// whole-grid.
static BLOCK_SWITCH: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Run `f` with `PYSCF_PBC_AO_GRID_BLOCK` pinned (`None` = unset = the B-04
/// default, BLK=8192; `Some("0")` = one block = the whole-grid reference),
/// restoring the previous setting afterwards.
fn with_block<T>(mode: Option<&str>, f: impl FnOnce() -> T) -> T {
    const KEY: &str = "PYSCF_PBC_AO_GRID_BLOCK";
    // Same poisoned-lock reasoning as `with_fuse`.
    let _guard = BLOCK_SWITCH.lock().unwrap_or_else(|e| e.into_inner());
    let previous = std::env::var(KEY).ok();
    // SAFETY-equivalent note: the guard above makes this the only writer, and
    // every reader of the switch runs inside some `with_block` body.
    match mode {
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

fn assert_bitwise(got: &[CTensor], want: &[CTensor], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: k-point count");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.re.len(), w.re.len(), "{what}: re length at k={k}");
        assert_eq!(g.im.len(), w.im.len(), "{what}: im length at k={k}");
        for (i, (a, b)) in g.re.iter().zip(&w.re).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{what}: re[{k}][{i}] blocked {a:e} vs whole-grid {b:e}"
            );
        }
        for (i, (a, b)) in g.im.iter().zip(&w.im).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{what}: im[{k}][{i}] blocked {a:e} vs whole-grid {b:e}"
            );
        }
    }
}

/// One case of the gate: sibling vs blocked driver on the same inputs.
fn direct_case(cell: &pyscf_pbc_gto::Cell, kpts: &[[f64; 3]], blks: &[usize], what: &str) {
    // A FRESH builder per call: `Fftdf::with_mesh` fixes the grid the driver
    // evaluates on.
    let df = Fftdf::with_mesh(cell.clone(), kpts, MESH).expect("FFTDF");
    let ngrids = df.ngrids();
    // Any `vr` proves the contraction identity — both arms contract the same
    // potential, so the comparison is pure accumulation order.
    let vr = vec![1.0f64; ngrids];
    // Pin the whole-grid path for the reference: with the switch unset the
    // driver blocks at the default `BLK = 8192`, which would make the reference
    // itself a blocked run.
    let want = with_block(Some("0"), || {
        eval_ao_kpts_local_vmat(cell, &df.grids.coords, kpts, &vr).expect("sibling")
    });
    for &blk in blks {
        let got = eval_ao_kpts_local_vmat_blocked(cell, &df.grids.coords, kpts, &vr, blk)
            .expect("blocked");
        assert_bitwise(&got, &want, &format!("{what} blk={blk}"));
    }
}

/// Direct driver gate: blocked vs sibling, bit-identical at every block size.
///
/// `MESH` 11³ is 1331 points: `blk` 8192 is one block, 1024 is two
/// (1024 + 307, a ragged tail), 128 is eleven (10 × 128 + 51).
#[test]
fn blocked_local_vmat_is_bit_identical() {
    // Gamma-only k-list, pseudopotential cell.
    let cell = diamond();
    let kpts = vec![[0.0f64; 3]];
    direct_case(&cell, &kpts, &[8192, 1024, 128], "diamond gamma");
    // 2×2×2 k-mesh, pseudopotential cell.
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("2x2x2 k-mesh");
    direct_case(&cell, &kpts, &[8192, 1024, 128], "diamond 2x2x2");
    // Gamma-only k-list, all-electron cell.
    let cell = he_all_electron();
    let kpts = vec![[0.0f64; 3]];
    direct_case(&cell, &kpts, &[8192, 1024, 128], "he gamma");
    // 2×2×2 k-mesh, all-electron cell.
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("2x2x2 k-mesh");
    direct_case(&cell, &kpts, &[8192, 1024, 128], "he 2x2x2");
}

/// End-to-end gate: `get_pp` is bit-identical across `PYSCF_PBC_AO_GRID_BLOCK`
/// ∈ {whole-grid, 8192, 1024, 128}, under the fused route (the only route that
/// reads the switch).
///
/// A FRESH builder per arm: a shared one would serve the second arm the first
/// arm's cached AO table (T2).
#[test]
fn get_pp_is_bit_identical_across_grid_blocks() {
    for (cell, kpts, what) in [
        (diamond(), vec![[0.0f64; 3]], "diamond gamma"),
        (
            diamond(),
            make_kpts_default(&diamond(), [2, 2, 2]).expect("2x2x2 k-mesh"),
            "diamond 2x2x2",
        ),
        (he_all_electron(), vec![[0.0f64; 3]], "he gamma"),
        (
            he_all_electron(),
            make_kpts_default(&he_all_electron(), [2, 2, 2]).expect("2x2x2 k-mesh"),
            "he 2x2x2",
        ),
    ] {
        // The reference arm first: explicit one-block mode — the whole-grid
        // path. (`None` would take the B-04 default, BLK=8192, which is one
        // of the arms under test, not the reference.)
        let want = with_block(Some("0"), || {
            with_fuse("1", || {
                let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
                df.get_pp(&kpts).expect("get_pp whole-grid")
            })
        });
        for mode in ["8192", "1024", "128"] {
            let got = with_block(Some(mode), || {
                with_fuse("1", || {
                    let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
                    df.get_pp(&kpts).expect("get_pp blocked")
                })
            });
            assert_bitwise(&got, &want, &format!("{what} block={mode}"));
        }
    }
}

/// The DEFAULT (`PYSCF_PBC_AO_GRID_BLOCK` unset) is the fixed B-04 `BLK`, and
/// is bit-identical to the whole-grid reference. 33³ = 35 937
/// points, so the default 8192 really splits the grid (five blocks, ragged
/// tail) rather than degenerating to one.
#[test]
fn default_get_pp_is_bit_identical_to_whole_grid() {
    let cell = diamond();
    let kpts = vec![[0.0f64; 3]];
    let mesh = [33, 33, 33];
    let run = |mode: Option<&str>| {
        with_block(mode, || {
            with_fuse("1", || {
                let df = Fftdf::with_mesh(cell.clone(), &kpts, mesh).expect("FFTDF");
                df.get_pp(&kpts).expect("get_pp")
            })
        })
    };
    let want = run(Some("0"));
    assert_bitwise(&run(None), &want, "default (unset)");
    assert_bitwise(&run(Some("8192")), &want, "explicit 8192");
}

/// Finding 3: a `blk` that is zero (which used to loop forever) or not a
/// multiple of 128 (which used to panic) is refused with an error.
#[test]
fn a_bad_block_size_is_an_error_not_a_hang() {
    let cell = diamond();
    let kpts = vec![[0.0f64; 3]];
    let df = Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("FFTDF");
    let vr = vec![1.0f64; df.ngrids()];
    for blk in [0usize, 100, 129] {
        assert!(
            eval_ao_kpts_local_vmat_blocked(&cell, &df.grids.coords, &kpts, &vr, blk).is_err(),
            "blk = {blk} must be refused"
        );
    }
}

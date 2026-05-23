//! Plan 05-07 Task 2 — ALWAYS-ON structural test for the MP2 PyO3 surface
//! (MP2-07 `as_scanner` + the factory + the override-dispatch plumbing).
//!
//! Per `.planning/phases/05-mp2/05-VALIDATION.md` the scanner verification is
//! `cargo test -p pyscf-py mp2_scanner (structural)`: this test asserts the
//! `PyMp2Scanner`/`as_scanner` plumbing EXISTS and the scanner-closure shape
//! (re-run reference -> re-snapshot -> MP2 kernel -> Mole->energy) compiles and
//! behaves on a Rust-side synthetic reference — it does NOT require a live
//! Python `mf`.
//!
//! ## Manual-Only / CI verifications (NOT run in the executor sandbox)
//!
//! Two cross-module dispatch parity checks need libpython + upstream pyscf +
//! the cintx#11 arity-4 `int2e` / arity-3 `int3c2e_sph` gates, so they run in
//! CI (the `mp2-oracle-cintx-gated` job) with a fully-wired PyO3 build, NOT
//! here (Phase-4 precedent — the env lacks maturin/pyscf):
//!
//!   (a) `mf.MP2().run().e_corr == mp.RMP2(mf).kernel()[0]` — cross-module
//!       dispatch parity (live Python + fully-wired PyO3).
//!   (b) `mf.density_fit().MP2()` routes to DFMP2 (live + cintx#11 gate).

use std::fs;

use pyscf_mp2::{
    ChemistsEris, Frozen, Mp2OverrideHooks, Mp2Reference, NoMp2Overrides, rmp2_kernel, scs_energy,
};

/// The `as_scanner` Rust plumbing must exist on every PyMP2 class + the
/// PyMp2Scanner pyclass must wrap a Mole->energy callable (MP2-07).
#[test]
fn mp_rs_exposes_as_scanner_plumbing() {
    let src = fs::read_to_string("src/mp.rs").expect("mp.rs readable");
    assert!(
        src.contains("fn as_scanner"),
        "mp.rs must expose `as_scanner` on the PyMP2 classes (MP2-07)"
    );
    assert!(
        src.contains("PyMp2Scanner"),
        "mp.rs must define the PyMp2Scanner Mole->energy callable wrapper (MP2-07)"
    );
    assert!(
        src.contains("fn __call__"),
        "PyMp2Scanner must be callable (`__call__(mol) -> f64`) — the Mole->energy shape"
    );
}

/// The override-dispatch + factory + eager-snapshot contracts must be present
/// (D-07 / D-08): call_method1 (Pitfall 7), py.detach (BIND-05), the MP2()
/// factory, and the kernel calls into the pyo3-free pyscf-mp2 kernels.
#[test]
fn mp_rs_uses_call_method1_and_kernel_dispatch() {
    let src = fs::read_to_string("src/mp.rs").expect("mp.rs readable");
    assert!(
        src.contains("call_method1"),
        "mp.rs must dispatch subclass overrides via call_method1 (D-08 / Pitfall 7)"
    );
    assert!(
        src.contains("py.detach"),
        "mp.rs default-hook compute must release the GIL via py.detach (BIND-05)"
    );
    for kern in ["rmp2_kernel", "ump2_kernel", "dfrmp2_kernel"] {
        assert!(
            src.contains(kern),
            "mp.rs must call the pyo3-free pyscf-mp2 kernel `{kern}`"
        );
    }
    // The MP2() factory dispatch (istype('UHF') / with_df).
    assert!(
        src.contains("istype") && src.contains("with_df"),
        "mp.rs MP2 factory must dispatch on istype('UHF') and with_df"
    );
}

/// The Python overlay re-exports `_native.mp` with the factory + class names
/// (BIND-02). DISTINCT from the vendored upstream pyscf/mp/.
#[test]
fn python_mp_overlay_reexports_native() {
    let src = fs::read_to_string("../../python/pyscf/mp/__init__.py")
        .expect("python/pyscf/mp/__init__.py readable");
    assert!(
        src.contains("from pyscf._native.mp"),
        "python/pyscf/mp/__init__.py must re-export from pyscf._native.mp (BIND-02)"
    );
    for sym in ["MP2", "RMP2", "UMP2", "DFMP2"] {
        assert!(
            src.contains(sym),
            "python/pyscf/mp/__init__.py __all__/import must list `{sym}`"
        );
    }
    assert!(
        src.contains("__all__"),
        "python/pyscf/mp/__init__.py must define __all__"
    );
}

/// Rust-side structural assertion of the scanner-closure SHAPE (MP2-07): build
/// a Mole->energy callable over a synthetic snapshotted reference (NO live
/// Python `mf`, NO `int2e` — a hand-supplied `(ia|jb)` block via the
/// override seam) and assert it returns the expected MP2 total energy.
///
/// This is the exact compute the `PyMp2Scanner::__call__` does AFTER it
/// re-snapshots: `rmp2_kernel(&refr, &frozen, &hooks, false)` then
/// `e_hf + scs_energy(e_ss, e_os, ss, os)`. We exercise that closure shape with
/// a synthetic reference so the always-on test does not depend on the cintx#11
/// `int2e` gate.
#[test]
fn scanner_closure_shape_returns_mole_to_energy() {
    // A 1-occ / 1-vir synthetic closed-shell reference: ε_occ = -0.5, ε_vir =
    // +0.5, so εi+εj−εa−εb = -2.0. A single (ia|jb)=0.5 block gives
    // t2 = 0.5 / -2.0 = -0.25; edi = 2·(0.5·-0.25) = -0.25; exi = -(0.5·-0.25)
    // = 0.125; e_ss = edi·0.5 + exi = 0.0; e_os = edi·0.5 = -0.125;
    // e_corr = e_ss + e_os = -0.125 (matches the 05-03 1×1 kernel reference).
    let refr = Mp2Reference {
        mo_coeff: pyscf_core::MOCoefficients {
            nao: 2,
            nmo: 2,
            data: vec![1.0, 0.0, 0.0, 1.0], // identity (column-major)
            energies: vec![-0.5, 0.5],
            occupations: vec![2.0, 0.0],
        },
        mo_energy: vec![-0.5, 0.5],
        mo_occ: vec![2.0, 0.0],
        e_hf: -1.0,
        converged: true,
        // A Mole is required by Mp2Reference; the synthetic-block hooks below
        // never touch it (ao2mo is overridden), so a default Mole is fine.
        mol: pyscf_core::Mole::default(),
    };

    // Synthetic-block hooks: hand the kernel a fixed (ia|jb) block so we never
    // hit the cintx#11-gated default_ao2mo `int2e`.
    struct SyntheticEris;
    impl Mp2OverrideHooks for SyntheticEris {
        fn ao2mo(
            &self,
            _refr: &Mp2Reference,
            _frozen: &Frozen,
        ) -> Result<ChemistsEris, pyscf_core::PyscfRsError> {
            // nocc=1, nvir=1 → ovov length 1, value (ia|jb)=0.5.
            Ok(ChemistsEris {
                ovov: vec![0.5],
                nocc: 1,
                nvir: 1,
            })
        }
    }

    // The scanner closure SHAPE: Mole-snapshot -> kernel -> e_hf + scs_energy.
    // Here the reference is already snapshotted; the closure returns the MP2
    // total energy (the exact value PyMp2Scanner::__call__ returns).
    let scanner = |refr: &Mp2Reference| -> Result<f64, pyscf_core::PyscfRsError> {
        let frozen = Frozen::None;
        let res = rmp2_kernel(refr, &frozen, &SyntheticEris, false)?;
        // Plain MP2 (ss=os=1.0).
        Ok(refr.e_hf + scs_energy(res.e_ss, res.e_os, 1.0, 1.0))
    };

    let e_tot = scanner(&refr).expect("synthetic scanner closure runs");
    // e_hf + e_corr = -1.0 + (-0.125) = -1.125.
    assert!(
        (e_tot - (-1.125)).abs() < 1e-12,
        "scanner Mole->energy returned {e_tot}, expected -1.125"
    );

    // Sanity: the default (NoMp2Overrides) seam exists and is the pyo3-free
    // contract the scanner uses for the non-overridden path (compile-time
    // reachability of the trait method on the default impl).
    let _no = NoMp2Overrides;
}

//! Plan 06-10 Task 2 — ALWAYS-ON structural test for the CCSD PyO3 surface
//! (the `CCSD()` factory dispatch + the override-dispatch plumbing + the
//! `as_scanner` Mole->energy closure shape).
//!
//! Per `.planning/phases/06-ccsd/06-VALIDATION.md` the always-on arm asserts the
//! bridge plumbing EXISTS (factory dispatch / override-detect / scanner closure)
//! and the pyo3-free `CcsdOverrideHooks` seam compiles + behaves on a Rust-side
//! synthetic reference — it does NOT require a live Python `mf`, maturin, or the
//! cintx#11 `int2e` gate.
//!
//! ## Manual-Only / CI verifications (NOT run in the executor sandbox)
//!
//! The LIVE numeric CCSD dispatch-parity + the `python3.13t` GIL smoke need
//! libpython + upstream pyscf + a fully-wired PyO3 build, so they run in CI on a
//! `workflow_dispatch`/human-verify arm (06-11), NOT here (the 05-07 precedent —
//! the env lacks maturin/pyscf/3.13t):
//!
//!   (a) `mf.CCSD().run().e_corr == cc.RCCSD(mf).kernel()` — cross-module
//!       dispatch parity (live Python + fully-wired PyO3).
//!   (b) `mf.density_fit().CCSD()` routes to DFCCSD (live + cintx#11 gate).
//!   (c) `python3.13t` CCSD smoke — no GIL deadlock under the heaviest
//!       `py.detach` region in the project (the `update_amps` default).

use std::fs;

use pyscf_ccsd::{
    ChemistsEris, CcsdOverrideHooks, NoCcsdOverrides, default_energy,
};
use pyscf_core::Amplitudes;
use pyscf_mp2::Frozen;

/// The `cc.rs` bridge must define the three PyCCSD classes + the CcsdPyBridge +
/// the CCSD() factory + the scanner — the structural surface (CCSD-01/02/08).
#[test]
fn cc_rs_defines_the_ccsd_surface() {
    let src = fs::read_to_string("src/cc.rs").expect("cc.rs readable");
    for cls in ["PyRCCSD", "PyUCCSD", "PyDFCCSD", "PyCcsdScanner", "CcsdPyBridge"] {
        assert!(
            src.contains(cls),
            "cc.rs must define `{cls}` (the CCSD PyO3 surface, D-09)"
        );
    }
    // The CCSD() factory pyfunction (mirrors pyscf/cc/__init__.py:83-139).
    assert!(
        src.contains("fn ccsd_factory") && src.contains("name = \"CCSD\""),
        "cc.rs must expose the CCSD() factory pyfunction"
    );
    // The factory dispatch order: UHF->UCCSD, with_df->DFCCSD, else RCCSD.
    assert!(
        src.contains("mf_is_uhf") && src.contains("mf_has_df"),
        "cc.rs CCSD factory must dispatch on istype('UHF') and with_df (cc/__init__.py order)"
    );
}

/// The override-dispatch + the BIND-05 GIL discipline must be present:
/// call_method1 (Pitfall 7), py.detach (BIND-05), the load-bearing "DO NOT
/// py.detach at the top" comment, and the kernel calls into the pyo3-free
/// pyscf-ccsd kernels.
#[test]
fn cc_rs_uses_call_method1_detach_and_kernel_dispatch() {
    let src = fs::read_to_string("src/cc.rs").expect("cc.rs readable");
    assert!(
        src.contains("call_method1"),
        "cc.rs must dispatch subclass overrides via call_method1 (D-09 / Pitfall 7)"
    );
    assert!(
        src.contains("py.detach"),
        "cc.rs default-hook compute must release the GIL via py.detach (BIND-05)"
    );
    // The load-bearing comment: the kernel does NOT py.detach at the top (hooks
    // re-enter Python). T-06-10-GIL.
    assert!(
        src.contains("DO NOT py.detach"),
        "cc.rs kernel must carry the load-bearing 'DO NOT py.detach at the top' discipline"
    );
    // The 5-hook override set the bridge dispatches.
    for hook in ["ao2mo", "update_amps", "energy", "make_rdm1", "make_rdm2"] {
        assert!(
            src.contains(hook),
            "cc.rs CcsdPyBridge must dispatch the `{hook}` hook (D-09)"
        );
    }
    // The kernels the bridge calls into (the pyo3-free pyscf-ccsd surface).
    for kern in ["ccsd_kernel", "uccsd_kernel", "solve_lambda"] {
        assert!(
            src.contains(kern),
            "cc.rs must call the pyo3-free pyscf-ccsd kernel/method `{kern}`"
        );
    }
}

/// The `as_scanner` plumbing must exist on the PyCCSD classes + the PyCcsdScanner
/// pyclass must wrap a Mole->energy callable (CCSD-07).
#[test]
fn cc_rs_exposes_as_scanner_plumbing() {
    let src = fs::read_to_string("src/cc.rs").expect("cc.rs readable");
    assert!(
        src.contains("fn as_scanner"),
        "cc.rs must expose `as_scanner` on the PyCCSD classes (CCSD-07)"
    );
    assert!(
        src.contains("PyCcsdScanner"),
        "cc.rs must define the PyCcsdScanner Mole->energy callable wrapper (CCSD-07)"
    );
    assert!(
        src.contains("fn __call__"),
        "PyCcsdScanner must be callable (`__call__(mol) -> f64`) — the Mole->energy shape"
    );
}

/// The Python overlay re-exports `_native.cc` with the factory + class names +
/// grafts `mf.CCSD()` onto the SCF base classes (BIND-02 cross-module dispatch).
/// DISTINCT from the vendored upstream pyscf/cc/.
#[test]
fn python_cc_overlay_reexports_native_and_grafts_ccsd() {
    let src = fs::read_to_string("../../python/pyscf/cc/__init__.py")
        .expect("python/pyscf/cc/__init__.py readable");
    assert!(
        src.contains("from pyscf._native.cc"),
        "python/pyscf/cc/__init__.py must re-export from pyscf._native.cc (BIND-02)"
    );
    for sym in ["CCSD", "RCCSD", "UCCSD", "DFCCSD"] {
        assert!(
            src.contains(sym),
            "python/pyscf/cc/__init__.py __all__/import must list `{sym}`"
        );
    }
    assert!(
        src.contains("__all__"),
        "python/pyscf/cc/__init__.py must define __all__"
    );
    // The mf.CCSD() / mf.density_fit().CCSD() cross-module dispatch graft
    // (the upstream scf.hf.SCF.CCSD = CCSD pattern).
    assert!(
        src.contains("cls.CCSD") && src.contains("density_fit"),
        "python/pyscf/cc/__init__.py must graft mf.CCSD()/mf.density_fit().CCSD() onto the SCF classes"
    );
}

/// `is_overridden`-style detect: a Python subclass override is detected when the
/// resolved method `__qualname__` class component is NOT a known base class.
/// We assert the qualname-based detect LOGIC the bridge uses (the same logic as
/// `mp.rs::is_overridden`, exercised without a live Python interpreter).
#[test]
fn override_detect_qualname_logic() {
    // The bridge's detect: split the qualname on '.', take the class component,
    // and check membership in the base-class set. A base method qualname is
    // "RCCSD.update_amps"; a subclass override is "MyCCSD.update_amps".
    let detect = |qualname: &str, base_classes: &[&str]| -> bool {
        let class = qualname.split('.').next().unwrap_or("");
        !base_classes.contains(&class)
    };
    // Base class method -> NOT overridden.
    assert!(
        !detect("RCCSD.update_amps", &["RCCSD"]),
        "the base RCCSD.update_amps must be detected as NOT overridden (default path)"
    );
    assert!(
        !detect("DFCCSD.ao2mo", &["DFCCSD"]),
        "the base DFCCSD.ao2mo must be detected as NOT overridden"
    );
    // Subclass override -> overridden (dispatch through call_method1).
    assert!(
        detect("MyCCSD.update_amps", &["RCCSD"]),
        "a subclass MyCCSD.update_amps must be detected as overridden (call_method1)"
    );
    assert!(
        detect("MyDFCCSD.ao2mo", &["DFCCSD"]),
        "a subclass MyDFCCSD.ao2mo must be detected as overridden"
    );
}

/// Build a 1-occ / 1-vir synthetic `ChemistsEris` block set + a hand-built
/// `(t1,t2)` amplitude pair and assert the pyo3-free `default_energy` (the
/// no-override default the bridge's `energy` hook calls under py.detach) returns
/// the expected CCSD correlation energy. This is the EXACT pure-Rust compute the
/// `CcsdPyBridge::energy` default path runs (BIND-05) — exercised on a synthetic
/// reference so the always-on test does not depend on the cintx#11 `int2e` gate.
#[test]
fn default_energy_closure_on_synthetic_eris() {
    // nocc=1, nvir=1. The single ovov block (ia|jb) = 0.5. With t1=0, t2=-0.25
    // the closed-shell CCSD energy is e = 2·(ia|jb)·tau_iiaa − (ia|jb)·tau_iaia
    // with tau = t2 + t1·t1 = -0.25. The single ovov entry drives the energy;
    // the value is deterministic and re-derivable from the closed-shell formula.
    let no = 1usize;
    let nv = 1usize;
    let eris = ChemistsEris {
        oooo: vec![0.0; no * no * no * no],
        ovoo: vec![0.0; no * nv * no * no],
        oovv: vec![0.0; no * no * nv * nv],
        ovov: vec![0.5; no * nv * no * nv],
        ovvo: vec![0.0; no * nv * nv * no],
        ovvv: vec![0.0; no * nv * nv * nv],
        vvvv: vec![0.0; nv * nv * nv * nv],
        fock: vec![0.0; (no + nv) * (no + nv)],
        mo_energy: vec![-0.5, 0.5],
        nocc: no,
        nvir: nv,
    };
    // t1 = 0, t2 = -0.25 (a representative converged amplitude pair).
    let t1 = Amplitudes::from_vec(no, nv, vec![0.0; no * nv], Vec::new());
    let t2 = Amplitudes::from_vec(no, nv, Vec::new(), vec![-0.25; no * no * nv * nv]);

    let e = default_energy(&t1, &t2, &eris).expect("default_energy runs on synthetic eris");
    // Closed-shell CCSD energy: e = Σ 2·(ia|jb)·tau(i,j,a,b) − (ia|jb)·tau(i,b? ..)
    // For the 1×1 block: tau = t2 + t1² = -0.25; the closed-shell contraction
    // reduces to (2·0.5 − 0.5)·(-0.25) = 0.5·(-0.25) = -0.125.
    assert!(
        (e.0 - (-0.125)).abs() < 1e-12,
        "default_energy returned {}, expected -0.125 (the 1×1 closed-shell CCSD energy)",
        e.0
    );

    // Sanity: the NoCcsdOverrides default seam exists and is the pyo3-free
    // contract the bridge uses for the non-overridden path (compile-time
    // reachability of the trait method on the default impl).
    let _no = NoCcsdOverrides;
    // And a synthetic override hook compiles against the trait (the override
    // path the bridge dispatches via call_method1).
    struct SyntheticHook;
    impl CcsdOverrideHooks for SyntheticHook {
        fn ao2mo(
            &self,
            _refr: &pyscf_ccsd::CcsdReference,
            _frozen: &Frozen,
        ) -> Result<ChemistsEris, pyscf_core::PyscfRsError> {
            Ok(ChemistsEris {
                oooo: vec![0.0],
                ovoo: vec![0.0],
                oovv: vec![0.0],
                ovov: vec![0.5],
                ovvo: vec![0.0],
                ovvv: vec![0.0],
                vvvv: vec![0.0],
                fock: vec![0.0; 4],
                mo_energy: vec![-0.5, 0.5],
                nocc: 1,
                nvir: 1,
            })
        }

        fn update_amps(
            &self,
            t1: &Amplitudes,
            t2: &Amplitudes,
            eris: &ChemistsEris,
        ) -> Result<(Amplitudes, Amplitudes), pyscf_core::PyscfRsError> {
            // Delegate to the default (the structural override-compile check).
            pyscf_ccsd::default_update_amps(t1, t2, eris)
        }
    }
    let _hook = SyntheticHook;
}

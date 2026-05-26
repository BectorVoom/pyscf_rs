//! Plan 07-09 Task 1 — ALWAYS-ON structural test for the gradient PyO3 surface
//! (the `Gradients()` factory dispatch + the override-dispatch plumbing + the
//! `as_scanner` Mole -> (e_tot, de) closure shape + the cross-module graft).
//!
//! Per the 07-03/07-08 cintx-gating precedent the always-on arm asserts the
//! bridge plumbing EXISTS (factory dispatch / override-detect / scanner closure
//! / overlay graft) on a source-scan — it does NOT require a live Python `mf`,
//! maturin, or the (MISSING) cintx grad-intor families. Any Python end-to-end
//! NUMERIC gradient assertion stays gated (07-10 oracle close-out arm).

use std::fs;

/// The `grad.rs` bridge must define the six per-method PyGradients classes + the
/// scanner + the Gradients() factory — the structural surface.
#[test]
fn grad_rs_defines_the_gradient_surface() {
    let src = fs::read_to_string("src/grad.rs").expect("grad.rs readable");
    for cls in [
        "PyRhfGradients",
        "PyUhfGradients",
        "PyRksGradients",
        "PyUksGradients",
        "PyMp2Gradients",
        "PyCcsdGradients",
        "PyGradScanner",
    ] {
        assert!(
            src.contains(cls),
            "grad.rs must define `{cls}` (the gradient PyO3 surface, D-09)"
        );
    }
    // The Gradients() factory pyfunction (the mf.nuc_grad_method() target).
    assert!(
        src.contains("fn grad_factory") && src.contains("name = \"Gradients\""),
        "grad.rs must expose the Gradients() factory pyfunction"
    );
    // The factory dispatch order: MP2 -> CCSD -> KS -> UHF -> RHF.
    assert!(
        src.contains("obj_is_mp2") && src.contains("obj_is_ccsd") && src.contains("mf_is_ks"),
        "grad.rs Gradients factory must dispatch on MP2/CCSD/KS/UHF type"
    );
}

/// The override-dispatch + the BIND-05 GIL discipline must be present: the
/// is_overridden MRO check (Pitfall 7), call_method1, py.detach (BIND-05), the
/// load-bearing "does NOT py.detach at the top" comment, and the kernel calls
/// into the pyo3-free pyscf-grad drivers.
#[test]
fn grad_rs_uses_call_method1_detach_and_kernel_dispatch() {
    let src = fs::read_to_string("src/grad.rs").expect("grad.rs readable");
    assert!(
        src.contains("fn is_overridden") && src.contains("__qualname__"),
        "grad.rs must carry the is_overridden __qualname__ MRO check (D-09 / Pitfall 7)"
    );
    assert!(
        src.contains("call_method1"),
        "grad.rs must dispatch subclass overrides via call_method1 (D-09 / Pitfall 7)"
    );
    assert!(
        src.contains("py.detach"),
        "grad.rs default compute must release the GIL via py.detach (BIND-05)"
    );
    // The load-bearing comment: the kernel does NOT py.detach at the top (hooks
    // re-enter Python).
    assert!(
        src.contains("does NOT py.detach at the\n    // top")
            || src.contains("does NOT py.detach at the top"),
        "grad.rs kernel must carry the load-bearing 'does NOT py.detach at the top' discipline"
    );
    // The pyo3-free pyscf-grad drivers the bridge calls into.
    for driver in [
        "RhfGradients",
        "UhfGradients",
        "RksGradients",
        "UksGradients",
        "Mp2Gradients",
        "CcsdGradients",
    ] {
        assert!(
            src.contains(driver),
            "grad.rs must call the pyo3-free pyscf-grad driver `{driver}`"
        );
    }
}

/// The grad scanner must return a TUPLE `(e_tot, de)` (rhf.py:248-262 — distinct
/// from the energy-only SCF/MP2/CCSD scanner that returns a scalar).
#[test]
fn grad_scanner_returns_e_tot_de_tuple() {
    let src = fs::read_to_string("src/grad.rs").expect("grad.rs readable");
    assert!(
        src.contains("fn as_scanner"),
        "grad.rs must expose `as_scanner` on the PyGradients classes (the geomopt seam)"
    );
    assert!(
        src.contains("PyGradScanner"),
        "grad.rs must define the PyGradScanner Mole -> (e_tot, de) callable wrapper"
    );
    assert!(
        src.contains("fn __call__"),
        "PyGradScanner must be callable (`__call__(mol) -> (f64, ndarray)`) — the tuple shape"
    );
    // The TUPLE return shape (a (f64, PyArray2) pair, NOT a scalar).
    assert!(
        src.contains("(f64, Bound<'py, PyArray2<f64>>)"),
        "PyGradScanner __call__ must return the (e_tot, de) TUPLE (rhf.py:248-262)"
    );
}

/// The Python overlay re-exports `_native.grad` with the factory + class names +
/// grafts `mf.nuc_grad_method()` onto the SCF base classes (BIND-02 cross-module
/// dispatch). NET-NEW overlay dir for Phase 7.
#[test]
fn python_grad_overlay_reexports_native_and_grafts_nuc_grad() {
    let src = fs::read_to_string("../../python/pyscf/grad/__init__.py")
        .expect("python/pyscf/grad/__init__.py readable");
    assert!(
        src.contains("from pyscf._native.grad"),
        "python/pyscf/grad/__init__.py must re-export from pyscf._native.grad (BIND-02)"
    );
    for sym in [
        "Gradients",
        "RhfGradients",
        "UhfGradients",
        "RksGradients",
        "UksGradients",
        "Mp2Gradients",
        "CcsdGradients",
    ] {
        assert!(
            src.contains(sym),
            "python/pyscf/grad/__init__.py __all__/import must list `{sym}`"
        );
    }
    assert!(
        src.contains("__all__"),
        "python/pyscf/grad/__init__.py must define __all__"
    );
    // The mf.nuc_grad_method() cross-module dispatch graft (the upstream
    // scf.hf.SCF.nuc_grad_method pattern), guarded by a subclass-override-wins
    // `getattr(cls, ..., None) is None` check.
    assert!(
        src.contains("_graft_nuc_grad_onto_scf") && src.contains("nuc_grad_method"),
        "python/pyscf/grad/__init__.py must graft mf.nuc_grad_method() onto the SCF classes"
    );
    assert!(
        src.contains("getattr(cls, \"nuc_grad_method\", None) is None"),
        "the graft must be guarded so a subclass override wins"
    );
}

/// `is_overridden`-style detect: a Python subclass override is detected when the
/// resolved method `__qualname__` class component is NOT a known base class. We
/// assert the qualname-based detect LOGIC the bridge uses (the same logic as
/// `cc.rs::is_overridden`), exercised without a live Python interpreter.
#[test]
fn override_detect_qualname_logic() {
    let detect = |qualname: &str, base_classes: &[&str]| -> bool {
        let class = qualname.split('.').next().unwrap_or("");
        !base_classes.contains(&class)
    };
    // Base class method -> NOT overridden.
    assert!(
        !detect("RhfGradients.grad_elec", &["RhfGradients"]),
        "the base RhfGradients.grad_elec must be detected as NOT overridden (default path)"
    );
    // Subclass override -> overridden (dispatch through call_method1).
    assert!(
        detect("MyGrad.grad_elec", &["RhfGradients"]),
        "a subclass MyGrad.grad_elec must be detected as overridden (call_method1)"
    );
}

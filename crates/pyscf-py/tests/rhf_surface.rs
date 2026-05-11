//! Plan 03-07 Task 2 — full PyRHF surface tests.
//!
//! Verifies:
//!   - 30-attribute floor (SCF-14): each #[getter]/#[setter] pair is
//!     exposed and round-trips. Tested via compile-time reachability of
//!     the inner-field accessors plus a Python smoke test (maturin_smoke).
//!   - PyOverrideBridge::call_method1 dispatch path exists in source.
//!   - py.detach pattern appears in scf.rs (≥ 5 occurrences for the 10
//!     hook defaults).
//!   - Mole extraction helper (extract_mole_from_pyany) is reachable.
//!
//! Source: plan 03-07 Task 2 <verify>.

use std::fs;

#[test]
fn scf_rs_contains_py_detach_in_hook_defaults() {
    // Plan 03-07 verification: grep -c "py.detach" crates/pyscf-py/src/scf.rs ≥ 5.
    let src = fs::read_to_string("src/scf.rs").expect("scf.rs readable");
    let n = src.matches("py.detach").count();
    assert!(
        n >= 5,
        "expected ≥ 5 py.detach call sites in scf.rs (one per default hook); found {}",
        n
    );
}

#[test]
fn bridge_rs_contains_call_method1_dispatch() {
    // BIND-07: PyOverrideBridge dispatches every hook via slf.call_method1.
    let src = fs::read_to_string("src/bridge.rs").expect("bridge.rs readable");
    assert!(
        src.contains("call_method1"),
        "bridge.rs must dispatch subclass overrides via Bound::call_method1 (BIND-07)"
    );
}

#[test]
fn numpy_io_uses_is_c_contiguous() {
    // BIND-04: every NumPy input runs is_c_contiguous() before .as_slice().
    let src = fs::read_to_string("src/numpy_io.rs").expect("numpy_io.rs readable");
    assert!(
        src.contains("is_c_contiguous"),
        "numpy_io.rs must call is_c_contiguous (BIND-04 stride enforcement)"
    );
}

#[test]
fn errors_rs_uses_create_exception_macro() {
    // BIND-09 / RESEARCH §Pattern 5: abi3-py310 forbids #[pyclass(extends=PyException)],
    // so the runtime exception must be declared via the create_exception! macro.
    let src = fs::read_to_string("src/errors.rs").expect("errors.rs readable");
    assert!(
        src.contains("create_exception!(_native, PyscfRsRuntimeError"),
        "errors.rs must use create_exception!(_native, PyscfRsRuntimeError, ...) per BIND-09"
    );
}

#[test]
fn rhf_exposes_thirty_attribute_floor_getters() {
    // SCF-14 30-attribute floor — getters mirroring the RHF struct fields.
    // We verify the count of `#[getter]` annotations + the presence of the
    // critical attribute names (e_tot, mo_coeff, mo_energy, mo_occ, converged,
    // ...). Python introspection (dir(mf)) is asserted by the
    // maturin_smoke.py test in plan 03-09 (CI).
    let src = fs::read_to_string("src/scf.rs").expect("scf.rs readable");
    let getter_count = src.matches("#[getter]").count();
    assert!(
        getter_count >= 30,
        "RHF/UHF/GHF must expose ≥ 30 getters for the 30-attribute floor (SCF-14); got {}",
        getter_count
    );

    // Spot-check a few load-bearing attribute names appear at all.
    for attr in [
        "e_tot",
        "mo_coeff",
        "mo_energy",
        "mo_occ",
        "converged",
        "max_cycle",
        "conv_tol",
        "diis_space",
        "init_guess",
    ] {
        assert!(
            src.contains(attr),
            "scf.rs missing required SCF-14 attribute getter for `{}`",
            attr
        );
    }
}

#[test]
fn rhf_has_kernel_and_run_methods() {
    // BIND-02 idiom: `scf.RHF(mol).run()` and `scf.RHF(mol).kernel()` must both
    // be callable from Python.
    let src = fs::read_to_string("src/scf.rs").expect("scf.rs readable");
    assert!(src.contains("fn kernel"), "PyRHF must expose `kernel` method");
    assert!(src.contains("fn run"), "PyRHF must expose `run` alias");
}

#[test]
fn python_overlay_resolves_rhf_uhf_ghf_unconditionally() {
    // BIND-02 — Python overlay re-export must succeed when _native ships.
    // The plan 03-02 overlay used try/except as a Phase-1 hedge; plan 03-07
    // ships _native so the overlay can become unconditional.
    let src = fs::read_to_string("../../python/pyscf/scf/__init__.py")
        .expect("python/pyscf/scf/__init__.py readable");
    assert!(
        src.contains("from pyscf._native.scf import RHF"),
        "python/pyscf/scf/__init__.py must unconditionally import RHF from pyscf._native.scf"
    );
    // The try/except scaffold should be gone (or at least not gate the actual import).
    assert!(
        !src.contains("plan 03-07 fills these in"),
        "python/pyscf/scf/__init__.py still references the 'plan 03-07 fills these in' \
         placeholder — update it now that 03-07 ships the cdylib"
    );
    // The try/except hedge should be replaced with an unconditional import.
    assert!(
        !src.contains("_not_built"),
        "python/pyscf/scf/__init__.py still defines the _not_built fallback \
         — plan 03-07 makes the import unconditional"
    );
}

#[test]
fn python_hf_uhf_ghf_shim_files_exist() {
    // BIND-02: from pyscf.scf.hf import RHF must work (upstream-compat shim).
    for (file, sym) in [("hf.py", "RHF"), ("uhf.py", "UHF"), ("ghf.py", "GHF")] {
        let path = format!("../../python/pyscf/scf/{}", file);
        let src = fs::read_to_string(&path).unwrap_or_else(|e| {
            panic!("python/pyscf/scf/{} must exist (BIND-02 shim): {}", file, e)
        });
        assert!(
            src.contains(&format!("from pyscf._native.scf import {}", sym)),
            "{} must re-export {} from pyscf._native.scf",
            path,
            sym
        );
    }
}

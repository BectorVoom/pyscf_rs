//! MP2-08 — the verbatim CCSD import contract.
//!
//! Upstream `pyscf/cc/ccsd.py:35` does:
//! ```python
//! from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
//! ```
//! This test asserts the EXACT same symbol set is exported from `pyscf-mp2`
//! (the Python `_mo_without_core` maps to the Rust `mo_without_core`). The
//! always-on arm proves the symbols are importable AND callable; the numeric
//! arm (asserting the RESULTS) is `#[ignore]`d until the helper bodies land in
//! plan 05-03.

// The exact CCSD import symbol set (cc/ccsd.py:35), proving they are exported.
use pyscf_mp2::{get_e_hf, get_frozen_mask, get_nmo, get_nocc, mo_without_core};

/// Always-on: the five symbols exist, are importable, and are callable. We
/// take their function pointers (proving the signatures resolve) and invoke
/// each once on a toy closed-shell input. The bodies currently return
/// `Err(NotYetImplemented)`, which is fine — this arm asserts the CONTRACT
/// (symbol existence + callability), not the numeric result.
#[test]
fn ccsd_import_symbols_exist_and_are_callable() {
    let mo_occ = [2.0_f64, 2.0, 0.0, 0.0];
    let mo_energy = [-0.5_f64, -0.4, 0.2, 0.3];
    let mo_coeff = [1.0_f64, 0.0, 0.0, 1.0];
    let mask = [true, true, false, false];

    // Bind each as a function pointer so a missing/renamed symbol fails to
    // compile — this is the contract guard.
    let _f_nocc: fn(&[f64], usize) -> _ = get_nocc;
    let _f_nmo: fn(&[f64], usize) -> _ = get_nmo;
    let _f_frozen_mask: fn(&[f64], usize) -> _ = get_frozen_mask;
    let _f_e_hf: fn(&[f64], &[f64]) -> _ = get_e_hf;
    let _f_mo_wo_core: fn(&[f64], &[bool]) -> _ = mo_without_core;

    // Callable smoke — invoke each once. Results are not asserted here.
    let _ = get_nocc(&mo_occ, 0);
    let _ = get_nmo(&mo_occ, 0);
    let _ = get_frozen_mask(&mo_occ, 0);
    let _ = get_e_hf(&mo_energy, &mo_occ);
    let _ = mo_without_core(&mo_coeff, &mask);
}

/// Numeric arm — asserts the helper RESULTS match upstream semantics. Ignored
/// until the bodies land in plan 05-03.
#[test]
#[ignore = "numeric body lands in plan 05-03"]
fn ccsd_import_symbols_return_upstream_values() {
    let mo_occ = [2.0_f64, 2.0, 0.0, 0.0];
    // 2 doubly-occupied orbitals, no frozen core → nocc == 2.
    assert_eq!(get_nocc(&mo_occ, 0).expect("nocc"), 2);
    // 4 MOs, no frozen → nmo == 4.
    assert_eq!(get_nmo(&mo_occ, 0).expect("nmo"), 4);
}

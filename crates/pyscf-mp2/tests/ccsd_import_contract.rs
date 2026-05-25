//! MP2-08 — the verbatim CCSD import contract.
//!
//! Upstream `pyscf/cc/ccsd.py:35` does:
//! ```python
//! from pyscf.mp.mp2 import get_nocc, get_nmo, get_frozen_mask, get_e_hf, _mo_without_core
//! ```
//! This test asserts the EXACT same symbol set is exported from `pyscf-mp2`
//! (the Python `_mo_without_core` maps to the Rust `mo_without_core`). Both
//! arms are ALWAYS-ON (no cintx, no integrals): the contract proves the symbols
//! are importable AND that they compute the upstream-matching helper results.

// The exact CCSD import symbol set (cc/ccsd.py:35), proving they are exported.
use pyscf_mp2::{Frozen, get_e_hf, get_frozen_mask, get_nmo, get_nocc, mo_without_core};

/// Always-on: the five symbols exist, are importable, and resolve to the
/// expected signatures. Binding each as a function pointer is the contract
/// guard — a missing/renamed symbol or a signature drift fails to compile.
#[test]
fn ccsd_import_symbols_exist_and_are_callable() {
    let mo_occ = [2.0_f64, 2.0, 0.0, 0.0];

    // Bind each as a function pointer so a missing/renamed symbol — or a
    // signature change away from the (mo_occ, &Frozen) contract — fails to
    // compile. This is the hard CCSD interface guard.
    let _f_nocc: fn(&[f64], &Frozen) -> _ = get_nocc;
    let _f_nmo: fn(&[f64], &Frozen) -> _ = get_nmo;
    let _f_frozen_mask: fn(&[f64], &Frozen) -> _ = get_frozen_mask;
    let _f_e_hf: fn(f64) -> _ = get_e_hf;
    let _f_mo_wo_core: fn(&pyscf_core::MOCoefficients, &Frozen) -> _ = mo_without_core;

    // Callable smoke.
    let _ = get_nocc(&mo_occ, &Frozen::None);
    let _ = get_e_hf(-76.0);
}

/// Always-on numeric contract — the helper RESULTS match upstream semantics.
/// Toy closed-shell reference `mo_occ = [2, 2, 0, 0]`.
#[test]
fn ccsd_import_symbols_return_upstream_values() {
    let mo_occ = [2.0_f64, 2.0, 0.0, 0.0];

    // Frozen::None → 2 occupied, 4 total, all active.
    assert_eq!(get_nocc(&mo_occ, &Frozen::None).expect("nocc"), 2);
    assert_eq!(get_nmo(&mo_occ, &Frozen::None).expect("nmo"), 4);
    assert_eq!(
        get_frozen_mask(&mo_occ, &Frozen::None).expect("mask"),
        vec![true, true, true, true]
    );

    // Frozen::Count(1) → freeze the inner-most orbital: nocc 1, nmo 3,
    // mask [false, true, true, true].
    assert_eq!(get_nocc(&mo_occ, &Frozen::Count(1)).expect("nocc"), 1);
    assert_eq!(get_nmo(&mo_occ, &Frozen::Count(1)).expect("nmo"), 3);
    assert_eq!(
        get_frozen_mask(&mo_occ, &Frozen::Count(1)).expect("mask"),
        vec![false, true, true, true]
    );

    // get_e_hf is the reference SCF e_tot passthrough.
    assert_eq!(get_e_hf(-76.026_760_737), -76.026_760_737);
}

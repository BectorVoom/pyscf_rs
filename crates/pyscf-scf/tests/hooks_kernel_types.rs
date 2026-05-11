//! Task 1 RED — hooks + kernel scaffolding tests.
//!
//! These tests validate the trait scaffolding only. The bodies that
//! `OverrideHooks` dispatches into are `unimplemented!()` stubs that
//! plan 03-11 will fill, so the trait-dispatch test exercises ONLY
//! type-resolution / vtable wiring (no actual invocation of methods
//! that would panic).

use pyscf_scf::{InitGuessMode, KernelConfig, NoOverrides, OverrideHooks, ScfResult};

/// Test 1: `NoOverrides` is instantiable and implements `OverrideHooks`.
/// We don't actually call any method (those panic via `unimplemented!()`
/// until plan 03-11) — we only confirm the trait is implemented by
/// taking a `&dyn OverrideHooks` reference and naming a trait method.
#[test]
fn no_overrides_implements_override_hooks() {
    let h = NoOverrides;
    // If `NoOverrides` does not implement `OverrideHooks`, this fails at
    // compile time. We never call `_h.get_hcore(...)` here because that
    // would `unimplemented!()`-panic until plan 03-11.
    let _h: &dyn OverrideHooks = &h;
}

/// Test 2: `KernelConfig::default()` returns the upstream PySCF defaults.
/// Source-of-truth: pyscf/scf/hf.py:1689-1759.
#[test]
fn kernel_config_default_matches_upstream() {
    let cfg = KernelConfig::default();
    assert_eq!(cfg.conv_tol, 1e-9, "conv_tol upstream default");
    assert_eq!(cfg.max_cycle, 50, "max_cycle upstream default");
    assert_eq!(cfg.diis_space, 8, "diis_space upstream default");
    assert_eq!(cfg.diis_start_cycle, 1, "diis_start_cycle upstream default");
    assert!(cfg.diis, "diis enabled by default");
    assert_eq!(cfg.level_shift, 0.0);
    assert_eq!(cfg.damp, 0.0);
    assert_eq!(cfg.diis_damp, 0.0);
    assert!(cfg.direct_scf, "direct_scf on by default");
    assert!(matches!(cfg.init_guess, InitGuessMode::Minao));
}

/// Test 3: `ScfResult` is `Clone` + `Debug` derivable.
#[test]
fn scf_result_is_clone_and_debug() {
    let r = ScfResult {
        e_tot: pyscf_core::Energy(0.0),
        mo_coeff: pyscf_core::MOCoefficients::default(),
        mo_energy: vec![],
        mo_occ: vec![],
        converged: false,
        cycles: 0,
    };
    let _ = r.clone();
    let s = format!("{:?}", r);
    assert!(s.contains("ScfResult"));
}

/// Test 4: `OverrideHooks` is object-safe (we already exercised it
/// above via `&dyn OverrideHooks`, but explicitly assert via type alias).
#[test]
fn override_hooks_is_object_safe() {
    fn _accept_dyn(_h: &dyn OverrideHooks) {}
    _accept_dyn(&NoOverrides);
}

//! Shape smoke test for `cholesky_eri`.
//!
//! Asserts `DfIntegrals { b_uvq, naux, nao }` is constructable, `nao` matches
//! `mol.nao_nr`, and `b_uvq.len() == nao*nao*naux`. Bit-exact DF energy vs
//! upstream PySCF is the CI-gated/human-verify arm.
//!
//! History: this was `#[ignore]`d for "int3c2e_sph base symbol missing"
//! (resolved by 05-08 — cintx ships int3c2e_sph, see int3c2e_auxmol.rs), then
//! still failed on the ill-conditioned `cc-pvdz-jkfit` `(P|Q)` metric. 05-09
//! added the rank-revealing eigh fallback (`pyscf_algebra::df_metric_fit`) to
//! `cholesky_eri`, so the metric is now handled and this test is ALWAYS-ON.
//! `naux` may be the effective rank after linear-dependence removal.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

#[test]
fn h2o_cc_pvdz_df_integrals_shape() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("O 0 0 0; H 0 1 0; H 1 0 0".into()),
        basis: BasisInput::Name("cc-pvdz".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("H2O/cc-pVDZ");
    let df = pyscf_df::cholesky_eri(&mol, "cc-pvdz-jkfit").expect("cholesky_eri");
    assert_eq!(df.nao, mol.nao_nr);
    assert!(df.naux > 0, "effective auxiliary rank must be positive");
    assert_eq!(df.b_uvq.len(), df.nao * df.nao * df.naux);
    assert!(
        df.b_uvq.iter().all(|v| v.is_finite()),
        "DF B-tensor must be finite"
    );
    assert!(
        df.b_uvq.iter().any(|&v| v.abs() > 1e-12),
        "DF B-tensor must be non-zero"
    );
}

/// The previously-failing real auxiliary metrics (cc-pvdz-jkfit AND the weigend
/// universal aux) now build a real, finite DF B-tensor via the 05-09
/// rank-revealing eigh fallback — no more rank-deficient `SingularAux` error.
#[test]
fn ill_conditioned_metrics_build_via_rank_revealing_fallback() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::String("O 0 0 0; H 0 1 0; H 1 0 0".into()),
        basis: BasisInput::Name("cc-pvdz".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("H2O/cc-pVDZ");
    for aux in ["cc-pvdz-jkfit", "weigend"] {
        let df = pyscf_df::cholesky_eri(&mol, aux)
            .unwrap_or_else(|e| panic!("cholesky_eri({aux}) must now succeed: {e}"));
        assert_eq!(df.nao, mol.nao_nr, "{aux}: nao");
        assert!(df.naux > 0, "{aux}: effective rank > 0");
        assert_eq!(df.b_uvq.len(), df.nao * df.nao * df.naux, "{aux}: shape");
        assert!(df.b_uvq.iter().all(|v| v.is_finite()), "{aux}: finite");
        assert!(df.b_uvq.iter().any(|&v| v.abs() > 1e-12), "{aux}: non-zero");
    }
}

/// Negative-path: cholesky_eri's body is reachable even when the inner
/// dispatcher is partially stubbed. Asserts the `DfIntegrals` struct surface
/// (`b_uvq`, `naux`, `nao`) is wired without requiring real numerical
/// integrals.
#[test]
fn df_integrals_struct_fields_compile() {
    // This is a pure-compile check — the struct must declare the trio of
    // public fields. If it fails, the auxbasis_defaults test won't compile
    // either, so this is belt-and-suspenders.
    let df = pyscf_df::DfIntegrals {
        b_uvq: vec![0.0; 6],
        naux: 2,
        nao: 1,
    };
    assert_eq!(df.b_uvq.len(), 6);
    assert_eq!(df.naux, 2);
    assert_eq!(df.nao, 1);
}

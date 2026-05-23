//! GTO-06 (plan 02-05) smoke tests: `mol.intor(name)` end-to-end.
//!
//! These tests prove the dispatcher correctly:
//!   1. Drives `cintx_rs::SessionRequest` over a real Mole → produces
//!      a finite, F-order, `nao × nao` output for `int1e_ovlp_sph`.
//!   2. Honours upstream `_add_suffix` (`pyscf/gto/mole.py:945+`).
//!   3. Routes ECP names (`int1e_ecp*`, `ECPscalar*`) through the
//!      `EcpEngine`. For an ECP-LESS molecule (these smoke tests use
//!      H/STO-3G) the cintx-backed engine (02-10) returns
//!      `PyscfRsError::EcpEngineNotAvailable` via its `mol._ecp.is_empty()`
//!      guard — so the user-facing "ECP on a molecule without one" error
//!      contract is unchanged from the 02-07 stub. Real ECP evaluation on a
//!      molecule WITH an ECP (Cu/LANL2DZ) is covered by
//!      `tests/ecp_int1e_oracle.rs`.
//!   4. Returns a clear error for unknown names AND for unbuilt Mole.
//!
//! NOTE on the H2/STO-3G overlap assertion (structural-only, by design):
//!
//!   The plan's behaviour table calls for `S_00 ≈ 1.0` / `S_01 ≈ 0.6593`
//!   (the analytical H2/STO-3G overlap). cintx-rs now performs REAL integral
//!   evaluation — `SessionRequest::evaluate` runs `CubeClExecutor` over the
//!   linked kernels (`cintx/crates/cintx-rs/src/api.rs`); the old synthetic
//!   staging placeholder is gone (the `fill_staging_values` symbol no longer
//!   exists anywhere in the cintx workspace). The sibling
//!   `tests/ecp_int1e_oracle.rs` (added by 02-10) likewise asserts a real,
//!   symmetric, non-zero ECP matrix — only meaningful under real evaluation.
//!
//!   These in-tree smoke tests deliberately assert only the STRUCTURAL
//!   contract — `out.shape == [nao, nao]`, `out.values.len() == nao * nao`,
//!   all values finite, Hermitian when symmetry is expected — as an always-on
//!   regression gate that needs no upstream venv (same de-risk pattern as
//!   02-01's wave0_smoke.rs). Numerical byte-identity vs upstream PySCF (the
//!   0.6593 / 1.0 numbers) is delegated to the venv-gated oracle harness
//!   (`tests/oracle/test_intor_oracle.py`, 02-09), so a CI run without the
//!   oracle venv still exercises the layout / naming / routing contract.

mod common;

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, intor};

fn h2_args() -> MoleBuildArgs {
    MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    }
}

fn h_args() -> MoleBuildArgs {
    MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    }
}

#[test]
fn h2_sto3g_int1e_ovlp_sph_returns_finite_2x2_matrix() {
    let mol = M(h2_args()).expect("H2/STO-3G should build");
    assert_eq!(mol.nao_nr, 2, "H2/STO-3G has 2 spherical AOs");

    let out = intor(&mol, "int1e_ovlp_sph").expect("int1e_ovlp_sph dispatch");
    assert_eq!(out.shape, vec![2, 2], "shape should be [nao, nao] = [2, 2]");
    assert_eq!(out.values.len(), 4, "2x2 matrix has 4 elements");

    // Every entry must be finite — catches uninitialised buffers / NaN
    // propagation regardless of whether values are real or the synthetic
    // staging pattern.
    for (i, v) in out.values.iter().enumerate() {
        assert!(v.is_finite(), "out.values[{i}] = {v} must be finite");
    }
}

#[test]
fn h2_sto3g_int1e_ovlp_sph_is_hermitian() {
    // Even with synthetic staging values, the dispatcher fills the upper
    // and lower triangle from the SAME shell-pair block (block (i,j)
    // and (j,i) come from the same evaluator on a symmetric operator).
    // Asserting hermiticity catches block-stitch indexing bugs even when
    // raw values are synthetic.
    let mol = M(h2_args()).expect("H2/STO-3G should build");
    let out = intor(&mol, "int1e_ovlp_sph").expect("dispatch");
    assert_eq!(out.shape, vec![2, 2]);
    // F-order: out.values[i + j * nao] = S[i, j]; for nao=2:
    //   S[0,0] = values[0]; S[1,0] = values[1]; S[0,1] = values[2]; S[1,1] = values[3]
    let nao = 2usize;
    for i in 0..nao {
        for j in 0..nao {
            let s_ij = out.values[i + j * nao];
            let s_ji = out.values[j + i * nao];
            // Hermitian within tight tolerance — the dispatcher must place
            // the (i,j) block transpose at (j,i) for real symmetric ops.
            assert!(
                (s_ij - s_ji).abs() < 1e-12,
                "Hermiticity violated: S[{i},{j}]={s_ij} vs S[{j},{i}]={s_ji}",
            );
        }
    }
}

#[test]
fn intor_without_suffix_appends_sph_for_default_mol() {
    let mol = M(h2_args()).expect("build");
    let out_sph = intor(&mol, "int1e_ovlp_sph").expect("dispatch sph");
    let out_no = intor(&mol, "int1e_ovlp").expect("dispatch unsuffixed");
    assert_eq!(
        out_sph.shape, out_no.shape,
        "unsuffixed should resolve to _sph for cart=false"
    );
    assert_eq!(
        out_sph.values, out_no.values,
        "unsuffixed call should produce identical values to _sph",
    );
}

#[test]
fn intor_already_suffixed_does_not_double_suffix() {
    // If add_suffix double-suffixed, the lookup would search for
    // "int1e_ovlp_sph_sph" which is not in the layout table → unknown error.
    let mol = M(h2_args()).expect("build");
    let res = intor(&mol, "int1e_ovlp_sph");
    assert!(
        res.is_ok(),
        "already-suffixed name must NOT double-suffix: {res:?}",
    );
}

#[test]
fn unknown_intor_returns_invalid_molecule_error() {
    let mol = M(h_args()).expect("build");
    let r = intor(&mol, "int1e_totally_fake");
    match r {
        Err(pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(msg))) => {
            assert!(
                msg.contains("unknown intor"),
                "msg should mention 'unknown intor': {msg}"
            );
            // Post-suffix name in error message — gives the user the
            // actual lookup key the layout_table didn't contain.
            assert!(
                msg.contains("int1e_totally_fake_sph"),
                "error should include post-suffix name: {msg}",
            );
        }
        other => panic!("expected InvalidMolecule, got {other:?}"),
    }
}

#[test]
fn ecp_int1e_route_on_ecpless_mol_returns_engine_not_available() {
    // H/STO-3G has no ECP; the cintx-backed engine (02-10) guards on
    // mol._ecp.is_empty() and returns the canonical EcpEngineNotAvailable.
    let mol = M(h_args()).expect("build");
    let r = intor(&mol, "int1e_ecp");
    assert!(
        matches!(r, Err(pyscf_core::PyscfRsError::EcpEngineNotAvailable)),
        "int1e_ecp on ECP-less mol must route to EcpEngineNotAvailable; got {r:?}",
    );
}

#[test]
fn ecp_ecpscalar_route_on_ecpless_mol_returns_engine_not_available() {
    let mol = M(h_args()).expect("build");
    let r = intor(&mol, "ECPscalar");
    assert!(
        matches!(r, Err(pyscf_core::PyscfRsError::EcpEngineNotAvailable)),
        "ECPscalar* on ECP-less mol must route to EcpEngineNotAvailable; got {r:?}",
    );
}

#[test]
fn unbuilt_mol_errors() {
    let mol = pyscf_core::Mole::default();
    let r = intor(&mol, "int1e_ovlp_sph");
    match r {
        Err(pyscf_core::PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(msg))) => {
            assert!(
                msg.to_lowercase().contains("not built") || msg.contains("build()"),
                "msg should explain unbuilt: {msg}",
            );
        }
        other => panic!("expected unbuilt error, got {other:?}"),
    }
}

//! GTO-05 evaluation half (gap-closure plan 02-10): `int1e_ecp` in-tree
//! gate on Cu/LANL2DZ.
//!
//! Proves the cintx-backed `CintxEcpEngine` (returned by
//! `pyscf_gto::ecp_engine()` since 02-10) actually evaluates the ECP
//! integral end-to-end through `intor("int1e_ecp")`:
//!   1. the dispatcher routes `int1e_ecp` → the cintx engine,
//!   2. the engine builds an ECP-augmented `cintx_core::BasisSet`
//!      (so the cintx safe-API ECP preflight does NOT trip
//!      `MissingEcpBasis`),
//!   3. cintx evaluates the Type-1 + Type-2 projector integral and
//!      returns a finite, non-zero `nao × nao` matrix.
//!
//! The BYTE-IDENTITY assertion vs upstream PySCF lives in the Python-side
//! oracle harness `tests/oracle/test_ecp_int1e.py` (release-oracle, gated on
//! an upstream-pyscf venv). This in-tree test is the always-on regression
//! gate: it must exit 0 in plain `cargo test -p pyscf-gto`.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, EcpInput, M, MoleBuildArgs, intor};

fn cu_lanl2dz() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("Cu 0 0 0".into()),
        basis: BasisInput::Name("lanl2dz".into()),
        ecp: EcpInput::Name("lanl2dz".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .expect("Cu/LANL2DZ build")
}

#[test]
fn cu_lanl2dz_int1e_ecp_returns_finite_matrix() {
    let mol = cu_lanl2dz();
    let nao = mol.nao_nr;
    assert!(nao > 0, "Cu/LANL2DZ must have a non-empty AO basis");

    let out = intor(&mol, "int1e_ecp").expect("int1e_ecp must evaluate via the cintx ECP engine");

    // Shape contract: square nao × nao 1e matrix, F-order.
    assert_eq!(
        out.shape,
        vec![nao, nao],
        "int1e_ecp shape must be [nao, nao]"
    );
    assert_eq!(
        out.values.len(),
        nao * nao,
        "int1e_ecp value buffer must be nao*nao"
    );

    // Every value finite (catches NaN/uninitialised-buffer regressions).
    for (idx, &v) in out.values.iter().enumerate() {
        assert!(v.is_finite(), "int1e_ecp values[{idx}] = {v} must be finite");
    }

    // Non-zero: a regression to EcpEngineNotAvailable (all-zeros) or a
    // synthetic zero-fill stub must fail here. The Cu ECP couples the
    // valence AOs, so the matrix carries at least one non-zero element.
    let nonzero = out.values.iter().filter(|&&v| v.abs() > 1e-18).count();
    assert!(
        nonzero > 0,
        "int1e_ecp matrix is all-zeros — cintx ECP not delivering real values \
         (regression to stub / MissingEcpBasis / zero-fill?)"
    );
}

#[test]
fn cu_lanl2dz_int1e_ecp_is_symmetric() {
    // int1e_ecp is a scalar (Type-1 + Type-2) operator → the AO matrix is
    // symmetric. Asserting symmetry catches block-stitch indexing bugs.
    let mol = cu_lanl2dz();
    let nao = mol.nao_nr;
    let out = intor(&mol, "int1e_ecp").expect("int1e_ecp dispatch");

    // F-order: out.values[i + j * nao] = M[i, j].
    for i in 0..nao {
        for j in 0..nao {
            let m_ij = out.values[i + j * nao];
            let m_ji = out.values[j + i * nao];
            assert!(
                (m_ij - m_ji).abs() < 1e-10,
                "int1e_ecp symmetry violated: M[{i},{j}]={m_ij} vs M[{j},{i}]={m_ji}",
            );
        }
    }
}

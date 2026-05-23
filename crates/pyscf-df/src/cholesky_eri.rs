//! `cholesky_eri` — build the DF B-integrals tensor.
//!
//! Source: `pyscf/df/incore.py:cholesky_eri` — verbatim algorithm port.
//!
//! Algorithm (per upstream lines ~33-110):
//!   1. Build `auxmol` — a `Mole` with the auxiliary basis on the same atoms.
//!   2. `int3c = (μν|P)` via `pyscf_gto::intor_with_auxmol(mol, "int3c2e_sph", auxmol)`.
//!      Shape `[nao, nao, naux]`, F-order.
//!   3. `int2c = (P|Q)` via `pyscf_gto::intor_with_auxmol(mol, "int2c2e_sph", auxmol)`.
//!      Shape `[naux, naux]`, F-order.
//!   4. Cholesky: `(P|Q) = L · L^T` (lower-triangular L).
//!   5. Forward-substitute: for every `(μ, ν)`, solve `L · b_{μν} = int3c[μ, ν, :]`
//!      to get `B^Q_{μν}`. Pack into `b_uvq` row-major.
//!
//! Phase 3 limitation per D-11: in-memory only. Phase 6 CCSD-08 wires
//! HDF5 spill (tensor-arena from CCSD day one).
//!
//! Cholesky implementation note: `pyscf_algebra::cholesky` is Tensor-API
//! and `NotYetImplemented{phase:3}`. We inline a host-only
//! Cholesky-Banachiewicz algorithm here (row-major, ~O(naux³)) — Phase 3
//! corpus has bounded naux (≤ ~245 for benzene/6-31G*) so the algorithmic
//! cost is negligible vs the int3c2e workload. When the Tensor-API
//! `pyscf_algebra::cholesky` lands, the signature here stays stable —
//! only the body swaps.
//!
//! Source links:
//!   - upstream: `pyscf/df/incore.py:cholesky_eri` (Apache-2.0)
//!   - upstream auxmol pattern: `pyscf/df/incore.py:23-31`
//!     (`auxmol = mol.copy(); auxmol.basis = auxbasis; auxmol.build()`)

use crate::error::DfError;
use pyscf_core::{Mole, PyscfRsError};

/// Public DF B-integrals output shape — consumed by `get_jk_df` (df_jk.rs)
/// and `pyscf-scf::df_scf::DfHooks` (plan 03-05 Task 2).
///
/// Layout: `b_uvq[mu * nao * naux + nu * naux + q]` (row-major outer,
/// component q innermost). Phase 3 ships row-major; if profiling shows
/// F-order is better for the J/K contraction kernels, plan 03-10's
/// optimization round can switch.
#[derive(Debug, Clone)]
pub struct DfIntegrals {
    /// B-integrals: flattened `[nao, nao, naux]` row-major (`mu`-major).
    pub b_uvq: Vec<f64>,
    /// Number of auxiliary basis functions.
    pub naux: usize,
    /// Number of orbital basis functions (must match `mol.nao_nr`).
    pub nao: usize,
}

/// Build the DF B-integrals for orbital Mole `mol` against auxiliary basis
/// `auxbasis`.
///
/// `auxbasis` is the **basis NAME** (e.g., `"cc-pvdz-jkfit"`, `"weigend"`);
/// resolution to a file path goes through pyscf-gto's `BasisInput::Name`
/// ALIAS lookup. The caller can use `default_jkfit(orbital_basis)` to
/// fetch the appropriate aux name.
///
/// # Errors
/// - `PyscfRsError::Core` if `mol` isn't built, or auxmol construction fails.
/// - `PyscfRsError::BasisLoad` if `auxbasis` doesn't resolve to a known basis.
/// - `PyscfRsError::Core(InvalidMolecule("(P|Q) singular ..."))` (via
///   `DfError::SingularAux` → `From<DfError>`) on rank-deficient (P|Q).
pub fn cholesky_eri(mol: &Mole, auxbasis: &str) -> Result<DfIntegrals, PyscfRsError> {
    use pyscf_core::{CoreError, Unit};
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};

    if !mol._built {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "cholesky_eri: mol not built — call pyscf_gto::M(args) first".into(),
        )));
    }
    let nao = mol.nao_nr;

    // Step 1: build auxmol — same atoms (in Bohr, since mol._atom invariant),
    // basis = auxbasis. Mirrors upstream `auxmol = mol.copy(); auxmol.basis =
    // auxbasis; auxmol.build()` at incore.py:23-31.
    let auxmol = M(MoleBuildArgs {
        atom: AtomInput::Tuples(mol._atom.clone()),
        basis: BasisInput::Name(auxbasis.to_string()),
        unit: Unit::Bohr,
        charge: mol.charge,
        spin: mol.spin,
        cart: mol.cart,
        ..Default::default()
    })?;
    let naux = auxmol.nao_nr;
    if naux == 0 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cholesky_eri: auxmol built with basis '{}' produced naux=0",
            auxbasis
        ))));
    }

    tracing::debug!(
        target: "pyscf_df::cholesky_eri",
        nao = nao,
        naux = naux,
        auxbasis = auxbasis,
        "building DF B-integrals"
    );

    // Step 2: (μν|P) — 3-center 2-electron integrals.
    // NOTE: int3c2e_sph base op is currently a cintx-ops upstream gap;
    // pyscf_gto::intor_with_auxmol returns a zero-filled buffer of the
    // correct shape until cintx lands the base symbol. B integrals built
    // here will be all-zero — sufficient for shape/wiring tests, NOT for
    // bit-exact DF-HF energy (plan 03-10 gates that).
    let int3c = pyscf_gto::intor_with_auxmol(mol, "int3c2e_sph", &auxmol)?;
    if int3c.values.len() != nao * nao * naux {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cholesky_eri: int3c2e_sph returned {} elements, expected {}",
            int3c.values.len(),
            nao * nao * naux
        ))));
    }

    // Step 3: (P|Q) — 2-center 2-electron integrals.
    let int2c = pyscf_gto::intor_with_auxmol(mol, "int2c2e_sph", &auxmol)?;
    if int2c.values.len() != naux * naux {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cholesky_eri: int2c2e_sph returned {} elements, expected {}",
            int2c.values.len(),
            naux * naux
        ))));
    }

    // Step 4: Cholesky-Banachiewicz of (P|Q) — lower-triangular L with
    // L · L^T = (P|Q). Inline host implementation (pyscf_algebra::cholesky
    // is Tensor-API and NotYetImplemented{phase:3}; see module-level doc).
    // Source: standard textbook Cholesky-Banachiewicz, row-major.
    let l = cholesky_banachiewicz_lower(&int2c.values, naux)
        .map_err(<DfError as Into<PyscfRsError>>::into)?;

    // Step 5: forward-substitute L · b_{μν} = int3c[μ, ν, :] for every (μ, ν).
    // int3c is F-order shape [nao, nao, naux]:
    //   element (mu, nu, q) lives at int3c.values[mu + nu * nao + q * nao * nao].
    // b_uvq is row-major [nao, nao, naux]:
    //   element (mu, nu, q) lives at b_uvq[mu * nao * naux + nu * naux + q].
    let mut b_uvq = vec![0.0_f64; nao * nao * naux];
    let mut rhs = vec![0.0_f64; naux];
    for mu in 0..nao {
        for nu in 0..nao {
            // `q` drives multi-dim flat-array offsets (int3c F-order).
            #[allow(clippy::needless_range_loop)]
            for q in 0..naux {
                rhs[q] = int3c.values[mu + nu * nao + q * nao * nao];
            }
            forward_substitute(&l, &rhs, naux, &mut b_uvq, mu * nao * naux + nu * naux);
        }
    }

    Ok(DfIntegrals { b_uvq, naux, nao })
}

/// Cholesky-Banachiewicz lower-triangular factorisation of an `n × n`
/// symmetric positive-definite matrix `a` (row-major flat slice). Returns
/// `L` (also row-major flat) such that `L · L^T = A`. The upper triangle
/// of L is zero on output.
///
/// Returns `DfError::SingularAux` on negative or zero pivot.
fn cholesky_banachiewicz_lower(a: &[f64], n: usize) -> Result<Vec<f64>, DfError> {
    if a.len() != n * n {
        return Err(DfError::Algebra(
            pyscf_algebra::AlgebraError::ShapeMismatch {
                expected: format!("{}x{}", n, n),
                actual: format!("len={}", a.len()),
            },
        ));
    }
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    return Err(DfError::SingularAux);
                }
                l[i * n + i] = s.sqrt();
            } else {
                let diag = l[j * n + j];
                if diag == 0.0 || !diag.is_finite() {
                    return Err(DfError::SingularAux);
                }
                l[i * n + j] = s / diag;
            }
        }
    }
    Ok(l)
}

/// Forward-substitute: solve `L · x = b` for lower-triangular `L` stored
/// row-major (n × n). Writes result into `out[out_offset..out_offset+n]`.
///
/// Silently writes zeros at indices where `L[i, i] == 0` (handled by caller's
/// SingularAux check in `cholesky_banachiewicz_lower` — if we got here, the
/// diagonal is nonzero).
fn forward_substitute(l: &[f64], b: &[f64], n: usize, out: &mut [f64], out_offset: usize) {
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l[i * n + j] * out[out_offset + j];
        }
        let diag = l[i * n + i];
        out[out_offset + i] = if diag != 0.0 { s / diag } else { 0.0 };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_identity() {
        // Identity is its own Cholesky factor.
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let l = cholesky_banachiewicz_lower(&a, 2).expect("identity should factor");
        assert_eq!(l, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn cholesky_simple_2x2() {
        // [[4, 2], [2, 5]] = L · L^T with L = [[2, 0], [1, 2]].
        let a = vec![4.0, 2.0, 2.0, 5.0];
        let l = cholesky_banachiewicz_lower(&a, 2).expect("must factor");
        assert!((l[0] - 2.0).abs() < 1e-12);
        assert_eq!(l[1], 0.0); // upper-triangular zero
        assert!((l[2] - 1.0).abs() < 1e-12);
        assert!((l[3] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_singular_zero_diagonal_errors() {
        let a = vec![0.0, 0.0, 0.0, 1.0]; // zero pivot at (0,0)
        let err = cholesky_banachiewicz_lower(&a, 2).expect_err("must reject singular");
        assert!(matches!(err, DfError::SingularAux));
    }

    #[test]
    fn cholesky_negative_diagonal_errors() {
        // Negative diagonal after subtraction triggers SingularAux.
        let a = vec![-1.0, 0.0, 0.0, 1.0];
        let err = cholesky_banachiewicz_lower(&a, 2).expect_err("must reject neg-def");
        assert!(matches!(err, DfError::SingularAux));
    }

    #[test]
    fn forward_substitute_lower_identity() {
        // L = I, b = [3, 5] -> x = b.
        let l = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 5.0];
        let mut out = vec![0.0; 2];
        forward_substitute(&l, &b, 2, &mut out, 0);
        assert_eq!(out, vec![3.0, 5.0]);
    }

    #[test]
    fn forward_substitute_lower_2x2() {
        // L = [[2, 0], [1, 2]], b = [4, 5] -> x[0] = 2, x[1] = (5 - 1*2)/2 = 1.5.
        let l = vec![2.0, 0.0, 1.0, 2.0];
        let b = vec![4.0, 5.0];
        let mut out = vec![0.0; 2];
        forward_substitute(&l, &b, 2, &mut out, 0);
        assert!((out[0] - 2.0).abs() < 1e-12);
        assert!((out[1] - 1.5).abs() < 1e-12);
    }
}

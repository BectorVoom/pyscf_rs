//! k-point coupled-perturbed HF (`pbc/scf/cphf.py`, 176 l).
//!
//! **A `fvind`, not a solver.** The response equation is the molecular CPHF
//! equation `(1 + A)·z = b` with a k index on the `(vir × occ)` rotation
//! space (`cphf.py::solve_nos1`: per-k `e_ai = 1/(e_a − e_i)`, per-k
//! `mo1base = h1·(−e_ai)`, `moloc` stacking). This module builds the k-aware
//! matrix-free response operator and passes each k-block into the ONE solver,
//! [`pyscf_grad::cphf::solve`] (GRAD-10). It defines no `pub fn solve` of its
//! own — `xtask check-single-cphf` fails the build otherwise.
//!
//! Upstream performs one stacked Krylov solve over the `moloc`-concatenated
//! space. The CPHF `fvind` is k-diagonal (the Coulomb kernel conserves k;
//! `_response_functions._get_jk` is called per k at `kshift = 0`), so the
//! stacked solve factors into independent per-k solves of the same equation to
//! the same `tol`. This module therefore calls the single solver once per k
//! and joins the blocks — identical converged solutions, no second solver.
//!
//! PBC defaults follow `cphf.py:29` (`max_cycle = 20`, `tol = 1e-9`,
//! `hermi = false`), NOT the molecular 50-cycle default.

use pyscf_core::PyscfRsError;

/// PBC CPHF default max Krylov cycles (`pbc/scf/cphf.py:29`: 20).
pub const KCPHF_DEFAULT_MAX_CYCLE: usize = 20;

/// PBC CPHF default convergence tolerance (`pbc/scf/cphf.py:29`: 1e-9).
pub const KCPHF_DEFAULT_TOL: f64 = 1e-9;

/// k-resolved CPHF input: one `(mo_energy, mo_occ, h1)` triple per k-point.
///
/// `h1[k]` is the flattened `(nvir_k × nocc_k)` RHS in vir-major layout
/// (element `(a, i)` at `a·nocc_k + i`), matching the molecular solver's
/// contract. `mo_energy[k]`/`mo_occ[k]` carry the full per-k spectrum (the
/// solver reads the (vir, occ) split from `mo_occ`).
#[derive(Debug, Clone)]
pub struct KcphfInput {
    /// Per-k MO energies (one full spectrum per k-point).
    pub mo_energy_k: Vec<Vec<f64>>,
    /// Per-k MO occupations (occupied `> 0`, virtual `== 0`).
    pub mo_occ_k: Vec<Vec<f64>>,
    /// Per-k RHS blocks, flattened vir-major.
    pub h1_k: Vec<Vec<f64>>,
}

/// Per-k matrix-free response operator: maps a flattened `(nvir_k × nocc_k)`
/// rotation vector to the response **without** the `e_ai` scaling or identity
/// term (those live inside [`pyscf_grad::cphf::solve`]). Mirrors the molecular
/// [`pyscf_grad::cphf::Fvind`] contract, indexed by k-point.
pub type KFvind<'a> = dyn Fn(usize, &[f64]) -> Result<Vec<f64>, PyscfRsError> + 'a;

/// Run the k-point CPHF solve: one [`pyscf_grad::cphf::solve`] call per k,
/// joined in k order.
///
/// `fvind(k, z_k)` supplies the k-diagonal response for block `k`. Every
/// inner product routes through the solver's `oracle_*` reductions, so the
/// result is thread-count invariant under `release-oracle` (D-PBC-17).
///
/// # Errors
/// Propagates shape errors on ragged input and any `Err` from `fvind` or the
/// underlying solver (e.g. non-convergence) unchanged.
pub fn run_kcphf(
    input: &KcphfInput,
    fvind: &KFvind<'_>,
    max_cycle: usize,
    tol: f64,
) -> Result<Vec<Vec<f64>>, PyscfRsError> {
    let nk = input.mo_energy_k.len();
    if input.mo_occ_k.len() != nk || input.h1_k.len() != nk {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!(
                "run_kcphf: ragged k-input (mo_energy={}, mo_occ={}, h1={})",
                nk,
                input.mo_occ_k.len(),
                input.h1_k.len()
            ),
        )));
    }
    let mut out = Vec::with_capacity(nk);
    for k in 0..nk {
        let z_k = pyscf_grad::cphf::solve(
            &|z| fvind(k, z),
            &input.mo_energy_k[k],
            &input.mo_occ_k[k],
            &input.h1_k[k],
            None,
            max_cycle,
            tol,
            false,
            pyscf_grad::cphf::DEFAULT_LEVEL_SHIFT,
        )?;
        out.push(z_k);
    }
    Ok(out)
}

/// Build a dense-kernel `KFvind` for testing and for drivers whose response is
/// explicitly materialized: `fvind(k, z) = Kmat[k]·z` (row-major dense, host
/// loop; the oracle ordering lives in the solver).
pub fn dense_kvind<'a>(kmat_k: &'a [Vec<f64>]) -> impl Fn(usize, &[f64]) -> Result<Vec<f64>, PyscfRsError> + 'a {
    move |k: usize, z: &[f64]| -> Result<Vec<f64>, PyscfRsError> {
        let m = kmat_k.get(k).ok_or_else(|| {
            PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
                "dense_kvind: k={k} out of range (nk={})",
                kmat_k.len()
            )))
        })?;
        let ndim = z.len();
        if m.len() != ndim * ndim {
            return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                format!(
                    "dense_kvind: k={k} kernel len {} != ndim² ({ndim})",
                    m.len()
                ),
            )));
        }
        let mut out = vec![0.0f64; ndim];
        for i in 0..ndim {
            let mut acc = 0.0f64;
            for j in 0..ndim {
                acc += m[i * ndim + j] * z[j];
            }
            out[i] = acc;
        }
        Ok(out)
    }
}

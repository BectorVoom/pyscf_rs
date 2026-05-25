//! make_rdm1 = C · diag(occ) · C^T (RHF closed-shell). Body filled in plan 03-11.
//!
//! Source: `pyscf/scf/hf.py:1517-1538` — `def make_rdm1(mo_coeff,
//! mo_occ)`. Reductions go through `pyscf_algebra::oracle_sum` for
//! bit-exact reproducibility (Pitfall 9 — reduction-order drift voids
//! bit-exact-with-PySCF under release-oracle).

use pyscf_core::{Density, MOCoefficients, PyscfRsError};

/// Default closed-shell RDM1 builder. `mo.data` is F-order
/// (column-major): element (mu, i) lives at `mo.data[mu + i*nao]`.
/// Output `D` is row-major `nao×nao` per `Density` documentation
/// (`pyscf-core/src/density.rs`: "Row-major AO density matrix
/// flattened").
pub fn default_make_rdm1(mo: &MOCoefficients) -> Result<Density, PyscfRsError> {
    let nao = mo.nao;
    let nmo = mo.nmo;
    if mo.data.len() != nao * nmo {
        return Err(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::DimensionMismatch {
                expected: nao * nmo,
                actual: mo.data.len(),
            },
        ));
    }
    if mo.occupations.len() != nmo {
        return Err(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::DimensionMismatch {
                expected: nmo,
                actual: mo.occupations.len(),
            },
        ));
    }
    let mut data = vec![0.0_f64; nao * nao];
    // D[mu, nu] = Σ_i occ[i] · C[mu, i] · C[nu, i]
    // Reduction over i goes through oracle_sum for determinism (Pitfall 9).
    let mut terms = vec![0.0_f64; nmo];
    // Flat-array reduction: indices drive row-major C/D offsets + the
    // per-MO `terms`/`occupations` buffers.
    #[allow(clippy::needless_range_loop)]
    for mu in 0..nao {
        for nu in 0..nao {
            for i in 0..nmo {
                let c_mu_i = mo.data[mu + i * nao];
                let c_nu_i = mo.data[nu + i * nao];
                terms[i] = mo.occupations[i] * c_mu_i * c_nu_i;
            }
            data[mu * nao + nu] = pyscf_algebra::oracle_sum(&terms);
        }
    }
    Ok(Density { nao, data })
}

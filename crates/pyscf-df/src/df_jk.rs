//! Density-fitted J/K builder — `get_jk_df(dm, df) -> (J, K)`.
//!
//! Source: `pyscf/df/df_jk.py:31-148` — RHF closed-shell algorithm.
//!
//! Algorithm:
//!   `B^Q_{mu,nu}` is built once by `cholesky_eri` (DfIntegrals). Then for
//!   each SCF cycle:
//!
//! ```text
//!     rho^Q   = sum_{l,s} B^Q_{l,s} * D[l,s]       (naux-element)
//!     J[m,n]  = sum_Q     B^Q_{m,n} * rho^Q         (nao^2)
//!
//!     W^Q_{m,l} = sum_s   B^Q_{m,s} * D[s,l]       (naux * nao^2)
//!     K[m,n]    = sum_Q sum_l W^Q_{m,l} * B^Q_{n,l}
//! ```
//!
//! All reductions go through `pyscf_algebra::oracle_sum` /
//! `pyscf_algebra::oracle_dot` per Pitfall 9 (deterministic across thread
//! counts and rerun reordering). The naux × nao × nao intermediate W is
//! the canonical memory point of DF-HF; Phase 6 CCSD-08 puts a tensor-arena
//! over it, but Phase 3 keeps it in-memory per D-11.

use crate::cholesky_eri::DfIntegrals;
use pyscf_core::{CoreError, Density, PyscfRsError};

/// Build the density-fitted J and K matrices for an RHF closed-shell
/// density `dm` against the pre-computed B integrals `df`.
///
/// Returns `(J, K)` as a pair of `Density` (each nao × nao, row-major).
///
/// # Errors
/// - `PyscfRsError::Core(InvalidMolecule(...))` if `dm.nao != df.nao` or
///   if the density data buffer size doesn't match `nao²`.
pub fn get_jk_df(dm: &Density, df: &DfIntegrals) -> Result<(Density, Density), PyscfRsError> {
    let nao = df.nao;
    let naux = df.naux;
    if dm.nao != nao {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "get_jk_df: density nao={} but DF nao={}",
            dm.nao, nao
        ))));
    }
    if dm.data.len() != nao * nao {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "get_jk_df: density data len={} but expected {}",
            dm.data.len(),
            nao * nao
        ))));
    }
    if df.b_uvq.len() != nao * nao * naux {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "get_jk_df: DF b_uvq len={} but expected {}",
            df.b_uvq.len(),
            nao * nao * naux
        ))));
    }

    // ── Step 1: ρ^Q = Σ_λσ B^Q_{λσ} · D[λσ] ─────────────────────────────
    // b_uvq is row-major [nao, nao, naux]:
    //   B^Q_{lambda,sigma} = b_uvq[lambda*nao*naux + sigma*naux + Q]
    // D is row-major [nao, nao]:
    //   D[lambda,sigma]    = dm.data[lambda*nao + sigma]
    let mut rho_q = vec![0.0_f64; naux];
    let mut prod_buf = vec![0.0_f64; nao * nao];
    for q in 0..naux {
        for lambda in 0..nao {
            for sigma in 0..nao {
                let b = df.b_uvq[lambda * nao * naux + sigma * naux + q];
                let d = dm.data[lambda * nao + sigma];
                prod_buf[lambda * nao + sigma] = b * d;
            }
        }
        rho_q[q] = pyscf_algebra::oracle_sum(&prod_buf);
    }

    // ── Step 2: J[μν] = Σ_Q B^Q_{μν} · ρ^Q ──────────────────────────────
    let mut j = vec![0.0_f64; nao * nao];
    let mut row_buf = vec![0.0_f64; naux];
    for mu in 0..nao {
        for nu in 0..nao {
            for q in 0..naux {
                row_buf[q] = df.b_uvq[mu * nao * naux + nu * naux + q] * rho_q[q];
            }
            j[mu * nao + nu] = pyscf_algebra::oracle_sum(&row_buf);
        }
    }

    // ── Step 3: W^Q_{μλ} = Σ_σ B^Q_{μσ} · D[σλ] ────────────────────────
    // Layout: w[q*nao*nao + mu*nao + lambda]. Phase 6 CCSD-08 will tensor-arena this.
    let mut w = vec![0.0_f64; naux * nao * nao];
    let mut sigma_buf = vec![0.0_f64; nao];
    for q in 0..naux {
        for mu in 0..nao {
            for lambda in 0..nao {
                for sigma in 0..nao {
                    let b = df.b_uvq[mu * nao * naux + sigma * naux + q];
                    let d = dm.data[sigma * nao + lambda];
                    sigma_buf[sigma] = b * d;
                }
                w[q * nao * nao + mu * nao + lambda] = pyscf_algebra::oracle_sum(&sigma_buf);
            }
        }
    }

    // ── Step 4: K[μν] = Σ_Q Σ_λ W^Q_{μλ} · B^Q_{νλ} ────────────────────
    let mut k = vec![0.0_f64; nao * nao];
    let mut k_buf = vec![0.0_f64; naux * nao];
    for mu in 0..nao {
        for nu in 0..nao {
            for q in 0..naux {
                for lambda in 0..nao {
                    let w_val = w[q * nao * nao + mu * nao + lambda];
                    let b_val = df.b_uvq[nu * nao * naux + lambda * naux + q];
                    k_buf[q * nao + lambda] = w_val * b_val;
                }
            }
            k[mu * nao + nu] = pyscf_algebra::oracle_sum(&k_buf);
        }
    }

    Ok((
        Density { nao, data: j },
        Density { nao, data: k },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn zero_df(nao: usize, naux: usize) -> DfIntegrals {
        DfIntegrals {
            b_uvq: vec![0.0; nao * nao * naux],
            naux,
            nao,
        }
    }

    #[test]
    fn zero_b_gives_zero_jk() {
        // All-zero B integrals → all-zero J, K regardless of density.
        let df = zero_df(2, 3);
        let dm = Density {
            nao: 2,
            data: vec![1.0, 0.5, 0.5, 1.0],
        };
        let (j, k) = get_jk_df(&dm, &df).expect("get_jk_df");
        assert_eq!(j.data, vec![0.0; 4]);
        assert_eq!(k.data, vec![0.0; 4]);
    }

    #[test]
    fn nao_mismatch_errors() {
        let df = zero_df(2, 3);
        let dm = Density {
            nao: 3, // wrong
            data: vec![0.0; 9],
        };
        let err = get_jk_df(&dm, &df).expect_err("nao mismatch");
        let msg = format!("{}", err);
        assert!(msg.contains("nao"), "got: {}", msg);
    }

    #[test]
    fn output_shapes_match_nao() {
        let df = zero_df(3, 5);
        let dm = Density {
            nao: 3,
            data: vec![0.1; 9],
        };
        let (j, k) = get_jk_df(&dm, &df).expect("get_jk_df");
        assert_eq!(j.nao, 3);
        assert_eq!(j.data.len(), 9);
        assert_eq!(k.nao, 3);
        assert_eq!(k.data.len(), 9);
    }
}

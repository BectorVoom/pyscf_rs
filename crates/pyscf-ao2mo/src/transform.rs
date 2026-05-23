//! AO→MO quarter-transform internals (the four sequential index contractions).
//!
//! Ports the `eri_ao.size == nao**4` branch of upstream
//! `pyscf/ao2mo/incore.py:general` (lines 125-128):
//! `einsum('pqrs,pi,qj,rk,sl->ijkl', eri, C0.conj(), C1, C2.conj(), C3)`.
//! pyscf-rs is real-only for v1, so the `.conj()` is a no-op (FOUND-06 /
//! D-03 real-coefficient contract).
//!
//! `pyscf_algebra::gemm` is `NotYetImplemented{phase:2}` (gemm.rs:17), so the
//! contraction is written as the memory-bounded quarter-transform sequence of
//! host loops, each reduction routed through `oracle_sum`/`oracle_dot`
//! (Pitfall 1/2 + FOUND-06: materialize products into a `Vec` FIRST, then
//! reduce — never `+=` accumulate). This keeps the result bit-exact and
//! thread-count invariant.
//!
//! ## Flat-index discipline (Pitfall 3 mitigation)
//!
//! Every tensor is carried in a CONSISTENT column-major (F-order) flat layout
//! and the exact offset formula is doc-commented at every boundary:
//!
//! * Input AO ERI `eri_ao` — F-order `[nao, nao, nao, nao]`:
//!   element `(p, q, r, s)` lives at `p + q*nao + r*nao^2 + s*nao^3`
//!   (matches `pyscf-gto/src/intor.rs` `int2e` F-order convention).
//! * MO coefficient block `c_x` — column-major `[nao, nx]` (the `data` of an
//!   `MOCoefficients` column subset): element `C[ao, mo]` at `ao + mo*nao`.
//! * After the p-step the `[np, nao, nao, nao]` F-order intermediate element
//!   `(i, q, r, s)` lives at `i + q*np + r*np*nao + s*np*nao*nao`.
//! * After the q-step the `[np, nq, nao, nao]` F-order intermediate element
//!   `(i, j, r, s)` lives at `i + j*np + r*np*nq + s*np*nq*nao`.
//! * After the r-step the `[np, nq, nr, nao]` F-order intermediate element
//!   `(i, j, k, s)` lives at `i + j*np + k*np*nq + s*np*nq*nr`.
//! * After the s-step the final `[np, nq, nr, ns]` F-order MO ERI element
//!   `(i, j, k, l)` lives at `i + j*np + k*np*nq + l*np*nq*nr`.

use crate::error::Ao2moError;
use pyscf_algebra::oracle_sum;

/// AO→MO 4-index transform via the memory-bounded quarter-transform sequence
/// `(pq|rs) --C_p--> (iq|rs) --C_q--> (ij|rs) --C_r--> (ij|ks) --C_s--> (ij|kl)`.
///
/// * `eri_ao` — F-order `[nao, nao, nao, nao]` AO 2-electron integrals.
/// * `c_p`, `c_q`, `c_r`, `c_s` — column-major `[nao, n*]` MO coefficient
///   blocks (the `data` of an [`pyscf_core::mo::MOCoefficients`] column subset).
///
/// Returns the final `[np, nq, nr, ns]` F-order MO ERI flat tensor. Each of
/// the four contraction steps transforms one AO index into an MO index; every
/// reduction routes through [`oracle_sum`] (materialize-then-reduce — no `+=`
/// accumulation), so the result is bit-exact and thread-count invariant.
///
/// # Errors
/// Returns [`Ao2moError::ShapeMismatch`] when
/// `eri_ao.len() != nao*nao*nao*nao` or any `c_x.len() != nao * n*`.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub(crate) fn quarter_transform(
    eri_ao: &[f64],
    nao: usize,
    c_p: &[f64],
    np: usize,
    c_q: &[f64],
    nq: usize,
    c_r: &[f64],
    nr: usize,
    c_s: &[f64],
    ns: usize,
) -> Result<Vec<f64>, Ao2moError> {
    // T-05-02-SHAPE: validate every untrusted shape input at entry; never
    // index OOB or panic.
    let expected_eri = nao * nao * nao * nao;
    if eri_ao.len() != expected_eri {
        return Err(Ao2moError::ShapeMismatch {
            expected: expected_eri,
            got: eri_ao.len(),
        });
    }
    for (c, n) in [(c_p, np), (c_q, nq), (c_r, nr), (c_s, ns)] {
        if c.len() != nao * n {
            return Err(Ao2moError::ShapeMismatch {
                expected: nao * n,
                got: c.len(),
            });
        }
    }

    // --- p-step: (p,q,r,s) --C_p--> (i,q,r,s) ----------------------------
    // out1 is F-order [np, nao, nao, nao]:
    //   (i,q,r,s) at i + q*np + r*np*nao + s*np*nao*nao.
    // For each fixed (q,r,s) and target MO i:
    //   out1[i,q,r,s] = sum_p eri_ao[p,q,r,s] * c_p[p + i*nao].
    let mut out1 = vec![0.0_f64; np * nao * nao * nao];
    let mut prod = vec![0.0_f64; nao]; // per-p product buffer (reused)
    for s in 0..nao {
        for r in 0..nao {
            for q in 0..nao {
                let eri_base = q * nao + r * nao * nao + s * nao * nao * nao;
                for i in 0..np {
                    let c_col = i * nao;
                    for p in 0..nao {
                        prod[p] = eri_ao[p + eri_base] * c_p[p + c_col];
                    }
                    out1[i + q * np + r * np * nao + s * np * nao * nao] = oracle_sum(&prod);
                }
            }
        }
    }

    // --- q-step: (i,q,r,s) --C_q--> (i,j,r,s) ----------------------------
    // out2 is F-order [np, nq, nao, nao]:
    //   (i,j,r,s) at i + j*np + r*np*nq + s*np*nq*nao.
    //   out2[i,j,r,s] = sum_q out1[i,q,r,s] * c_q[q + j*nao].
    let mut out2 = vec![0.0_f64; np * nq * nao * nao];
    prod.resize(nao, 0.0);
    for s in 0..nao {
        for r in 0..nao {
            for j in 0..nq {
                let c_col = j * nao;
                for i in 0..np {
                    for q in 0..nao {
                        // out1 (i,q,r,s) offset.
                        let o1 = i + q * np + r * np * nao + s * np * nao * nao;
                        prod[q] = out1[o1] * c_q[q + c_col];
                    }
                    out2[i + j * np + r * np * nq + s * np * nq * nao] = oracle_sum(&prod);
                }
            }
        }
    }

    // --- r-step: (i,j,r,s) --C_r--> (i,j,k,s) ----------------------------
    // out3 is F-order [np, nq, nr, nao]:
    //   (i,j,k,s) at i + j*np + k*np*nq + s*np*nq*nr.
    //   out3[i,j,k,s] = sum_r out2[i,j,r,s] * c_r[r + k*nao].
    let mut out3 = vec![0.0_f64; np * nq * nr * nao];
    for s in 0..nao {
        for k in 0..nr {
            let c_col = k * nao;
            for j in 0..nq {
                for i in 0..np {
                    for r in 0..nao {
                        // out2 (i,j,r,s) offset.
                        let o2 = i + j * np + r * np * nq + s * np * nq * nao;
                        prod[r] = out2[o2] * c_r[r + c_col];
                    }
                    out3[i + j * np + k * np * nq + s * np * nq * nr] = oracle_sum(&prod);
                }
            }
        }
    }

    // --- s-step: (i,j,k,s) --C_s--> (i,j,k,l) ----------------------------
    // out4 is the final F-order [np, nq, nr, ns]:
    //   (i,j,k,l) at i + j*np + k*np*nq + l*np*nq*nr.
    //   out4[i,j,k,l] = sum_s out3[i,j,k,s] * c_s[s + l*nao].
    let mut out4 = vec![0.0_f64; np * nq * nr * ns];
    for l in 0..ns {
        let c_col = l * nao;
        for k in 0..nr {
            for j in 0..nq {
                for i in 0..np {
                    for s in 0..nao {
                        // out3 (i,j,k,s) offset.
                        let o3 = i + j * np + k * np * nq + s * np * nq * nr;
                        prod[s] = out3[o3] * c_s[s + c_col];
                    }
                    out4[i + j * np + k * np * nq + l * np * nq * nr] = oracle_sum(&prod);
                }
            }
        }
    }

    Ok(out4)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Longhand reference: the SAME four sequential quarter-transform
    /// contractions as [`quarter_transform`], but every per-index reduction is
    /// a strict left-to-right scalar fold (`acc += …`) instead of
    /// [`oracle_sum`]'s pairwise tree.
    ///
    /// This is an INDEPENDENT oracle: it never calls `oracle_sum`, so a match
    /// proves the production loop nesting / flat-index arithmetic is correct.
    /// Because the two paths reduce in different orders (pairwise tree vs
    /// left fold), agreement is to a tight numeric tolerance, not bit-exact —
    /// a single naive global einsum fold and the staged quarter-transform
    /// differ by ~1 ULP on `f64`, which is exactly why production routes
    /// through `oracle_sum` (deterministic, thread-count invariant) rather
    /// than a naive accumulator (Pitfall 1/2).
    ///
    /// All tensors F-order (same layout as [`quarter_transform`]).
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    fn reference_einsum(
        eri_ao: &[f64],
        nao: usize,
        c_p: &[f64],
        np: usize,
        c_q: &[f64],
        nq: usize,
        c_r: &[f64],
        nr: usize,
        c_s: &[f64],
        ns: usize,
    ) -> Vec<f64> {
        // p-step → [np, nao, nao, nao]
        let mut o1 = vec![0.0_f64; np * nao * nao * nao];
        for s in 0..nao {
            for r in 0..nao {
                for q in 0..nao {
                    let base = q * nao + r * nao * nao + s * nao * nao * nao;
                    for i in 0..np {
                        let mut acc = 0.0_f64;
                        for p in 0..nao {
                            acc += eri_ao[p + base] * c_p[p + i * nao];
                        }
                        o1[i + q * np + r * np * nao + s * np * nao * nao] = acc;
                    }
                }
            }
        }
        // q-step → [np, nq, nao, nao]
        let mut o2 = vec![0.0_f64; np * nq * nao * nao];
        for s in 0..nao {
            for r in 0..nao {
                for j in 0..nq {
                    for i in 0..np {
                        let mut acc = 0.0_f64;
                        for q in 0..nao {
                            acc += o1[i + q * np + r * np * nao + s * np * nao * nao]
                                * c_q[q + j * nao];
                        }
                        o2[i + j * np + r * np * nq + s * np * nq * nao] = acc;
                    }
                }
            }
        }
        // r-step → [np, nq, nr, nao]
        let mut o3 = vec![0.0_f64; np * nq * nr * nao];
        for s in 0..nao {
            for k in 0..nr {
                for j in 0..nq {
                    for i in 0..np {
                        let mut acc = 0.0_f64;
                        for r in 0..nao {
                            acc += o2[i + j * np + r * np * nq + s * np * nq * nao]
                                * c_r[r + k * nao];
                        }
                        o3[i + j * np + k * np * nq + s * np * nq * nr] = acc;
                    }
                }
            }
        }
        // s-step → [np, nq, nr, ns]
        let mut out = vec![0.0_f64; np * nq * nr * ns];
        for l in 0..ns {
            for k in 0..nr {
                for j in 0..nq {
                    for i in 0..np {
                        let mut acc = 0.0_f64;
                        for s in 0..nao {
                            acc += o3[i + j * np + k * np * nq + s * np * nq * nr]
                                * c_s[s + l * nao];
                        }
                        out[i + j * np + k * np * nq + l * np * nq * nr] = acc;
                    }
                }
            }
        }
        out
    }

    /// Build a deterministic, non-symmetric synthetic ERI `[nao,nao,nao,nao]`
    /// (F-order). The values are an arbitrary but fixed function of the indices
    /// so the test is reproducible.
    fn synthetic_eri(nao: usize) -> Vec<f64> {
        let n = nao;
        let mut eri = vec![0.0_f64; n * n * n * n];
        for s in 0..n {
            for r in 0..n {
                for q in 0..n {
                    for p in 0..n {
                        let v = (p as f64) * 1.0
                            + (q as f64) * 0.25
                            + (r as f64) * 0.0625
                            + (s as f64) * 0.015_625
                            + 0.5;
                        eri[p + q * n + r * n * n + s * n * n * n] = v;
                    }
                }
            }
        }
        eri
    }

    /// Identity-transform invariant: with an identity MO coefficient matrix on
    /// all four indices, the transform returns the AO ERI unchanged.
    #[test]
    fn identity_transform_returns_input_unchanged() {
        let nao = 2;
        let eri = synthetic_eri(nao);
        // Column-major [nao, nao] identity: C[a,b] = (a==b) at a + b*nao.
        let mut ident = vec![0.0_f64; nao * nao];
        for i in 0..nao {
            ident[i + i * nao] = 1.0;
        }
        let out = quarter_transform(
            &eri, nao, &ident, nao, &ident, nao, &ident, nao, &ident, nao,
        )
        .expect("identity transform");
        assert_eq!(out.len(), eri.len());
        for (a, b) in out.iter().zip(eri.iter()) {
            assert_eq!(a, b, "identity transform must be bit-exact unchanged");
        }
    }

    /// Non-trivial MO block: the quarter-transform must match the longhand
    /// reference einsum to BIT-EXACTNESS (delta == 0.0) under release-oracle.
    #[test]
    fn nontrivial_transform_matches_reference_bit_exact() {
        let nao = 3;
        let eri = synthetic_eri(nao);
        // Four DISTINCT non-trivial column-major [nao, n*] coefficient blocks
        // (rectangular: n* may differ from nao).
        let np = 2;
        let nq = 3;
        let nr = 2;
        let ns = 1;
        let mk = |seed: f64, n: usize| -> Vec<f64> {
            let mut c = vec![0.0_f64; nao * n];
            for mo in 0..n {
                for ao in 0..nao {
                    c[ao + mo * nao] = seed + (ao as f64) * 0.5 - (mo as f64) * 0.3;
                }
            }
            c
        };
        let c_p = mk(1.1, np);
        let c_q = mk(0.7, nq);
        let c_r = mk(-0.4, nr);
        let c_s = mk(0.9, ns);

        let got = quarter_transform(
            &eri, nao, &c_p, np, &c_q, nq, &c_r, nr, &c_s, ns,
        )
        .expect("nontrivial transform");
        let want = reference_einsum(
            &eri, nao, &c_p, np, &c_q, nq, &c_r, nr, &c_s, ns,
        );
        assert_eq!(got.len(), np * nq * nr * ns);
        assert_eq!(got.len(), want.len());
        for (idx, (a, b)) in got.iter().zip(want.iter()).enumerate() {
            // Bit-exact (delta == 0.0) is the contract for this synthetic
            // small case where the oracle_sum tree and the reference left-fold
            // visit the same finite values in a commutative-enough order; if
            // this ever drifts, fall back to a 1e-12 tolerance and document.
            assert_eq!(a, b, "mismatch at flat index {idx}: got {a}, want {b}");
        }
    }

    /// Thread-count invariance: `oracle_sum` is thread-count invariant by
    /// construction (fixed pairwise chunk), so the whole transform is
    /// deterministic. Run the SAME transform twice and assert bit-identical
    /// output — a determinism contract that holds regardless of
    /// `RAYON_NUM_THREADS` (the reduction never consults the thread count).
    #[test]
    fn transform_is_deterministic() {
        let nao = 3;
        let eri = synthetic_eri(nao);
        let mut c = vec![0.0_f64; nao * nao];
        for mo in 0..nao {
            for ao in 0..nao {
                c[ao + mo * nao] = 0.3 + (ao as f64) * 0.2 - (mo as f64) * 0.1;
            }
        }
        let run = || {
            quarter_transform(&eri, nao, &c, nao, &c, nao, &c, nao, &c, nao)
                .expect("deterministic transform")
        };
        let a = run();
        let b = run();
        assert_eq!(a, b, "transform must be bit-identical across runs");
    }
}

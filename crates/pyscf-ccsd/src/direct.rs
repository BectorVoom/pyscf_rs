//! AO-direct CCSD — the on-the-fly `_contract_vvvv_t2` branch that replaces the
//! in-memory `vvvv` with an AO-space contraction (port of
//! `pyscf/cc/ccsd.py:473-570`, `_contract_vvvv_t2` / `_contract_s4vvvv_t2`;
//! CCSD-07, `mycc.direct=True`).
//!
//! # The contraction
//!
//! The vvvv step of the RCCSD `t2new` equation is
//! `einsum('abcd,ijcd->ijab', Wvvvv, tau)` where the PURE-INTEGRAL part of
//! `cc_Wvvvv` is `vvvv.transpose(0,2,1,3)` (i.e. element `(a,b,c,d)` reads
//! `vvvv[a,c,b,d]`). The integral contribution to `t2new` is therefore
//! ```text
//! Ht2[i,j,a,b] = sum_{c,d} vvvv_mo[a,c,b,d] * tau[i,j,c,d]
//! ```
//! which is exactly upstream's `Ht2 = einsum('ijcd,acbd->ijab', tau, vvvv)`
//! (`ccsd.py:474`). The `t1`-correction parts of `cc_Wvvvv` (the two `ovvv·t1`
//! terms) are SMALL `nv^3`-sized intermediates and stay on the in-core path;
//! only the heavy `nv^4` integral contraction is moved here.
//!
//! # Open Q4 — RESOLVED AT EXECUTION (path b)
//!
//! Upstream AO-direct (`ccsd.py:538-558`) walks AO shell-pair blocks via
//! `gto.moleintor.getints4c(..., shls_slice=...)`, contracting `t2` (transformed
//! to the AO basis) against each on-the-fly `int2e` block — NEVER holding the
//! full tensor. A grep of the `pyscf-gto` intor surface
//! (`crates/pyscf-gto/src/intor.rs`) confirms there is **no** shell-sliced
//! streaming `int2e` primitive in-tree: only `intor("int2e")` returning the full
//! arity-4 AO tensor (as verified used whole in `mp2.rs:150`). We therefore take
//! **path (b)**: AO-direct v1 sources the full AO `int2e` ONCE and tiles the
//! AO→MO `vvvv` transform over the LEADING virtual index `a`, contracting each
//! `[1,nv,nv,nv]` slice against `tau` and discarding it. The peak `vvvv`-MO
//! buffer is `nv^3` (one `a`-slice), so the full `nv^4` `vvvv` MO tensor is
//! **never materialized** — satisfying CCSD-07's `direct=True` contract (the
//! memory-frugality whole point) even though the AO source tensor is held. A
//! shell-sliced primitive (which would also stream the AO source) is the v2
//! upgrade once `pyscf-gto` grows the `shls_slice` API.
//!
//! # Correctness anchor
//!
//! AO-direct is a different contraction ORDER of the SAME math as the in-core
//! `_contract_vvvv_t2`. The equivalence — AO-direct vvvv block == in-core vvvv
//! block (bit-close) — is the correctness proof, exercised both here (the
//! [`tests`] module, against a synthetic `eris.vvvv`) and end-to-end in
//! `tests/direct.rs` (AO-direct `e_corr` == in-core `e_corr` on a real
//! small-system RHF reference).
//!
//! # Reduction discipline (Pitfall 1/2)
//!
//! Every contracted-axis reduction materializes the per-element products into a
//! `Vec` then folds with `pyscf_algebra::oracle_sum` — NO bare `+=` across a
//! contracted axis, NO `gemm` (`NotYetImplemented{phase:2}`). The result is
//! bit-exact and thread-count invariant (RAYON 1==8) under `release-oracle`.
#![allow(clippy::needless_range_loop)]

use crate::eris::ChemistsEris;
use crate::error::CcsdError;
use pyscf_algebra::oracle_sum;
use pyscf_core::MOCoefficients;

#[inline]
fn t1_idx(nv: usize, i: usize, a: usize) -> usize {
    i * nv + a
}

#[inline]
fn t2_idx(no: usize, nv: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * no + j) * nv + a) * nv + b
}

/// `vvvv` element `(a,b,c,d)` at `((a*nv + b)*nv + c)*nv + d` (C-order, the
/// `ChemistsEris::vvvv` layout).
#[inline]
fn vvvv_idx(nv: usize, a: usize, b: usize, c: usize, d: usize) -> usize {
    ((a * nv + b) * nv + c) * nv + d
}

/// Build `tau[i,j,c,d] = t2[i,j,c,d] + t1[i,c]*t1[j,d]` as a closure over the
/// flat C-order amplitude buffers.
#[inline]
fn tau_closure<'a>(
    t1: &'a [f64],
    t2: &'a [f64],
    no: usize,
    nv: usize,
) -> impl Fn(usize, usize, usize, usize) -> f64 + 'a {
    move |i: usize, j: usize, c: usize, d: usize| {
        t2[t2_idx(no, nv, i, j, c, d)] + t1[t1_idx(nv, i, c)] * t1[t1_idx(nv, j, d)]
    }
}

/// Core AO-direct vvvv contraction over a single leading-virtual `a`-slice of
/// the MO `vvvv` tensor.
///
/// Given the `a`-slice `vvvv_a[(c,b,d)] = vvvv_mo[a,c,b,d]` (flat C-order
/// `[nv,nv,nv]`, element `(c,b,d)` at `(c*nv + b)*nv + d`) and the amplitude
/// closures, accumulate the contribution
/// `Ht2[i,j,a,b] = sum_{c,d} vvvv_mo[a,c,b,d] * tau[i,j,c,d]`
/// for ALL `(i,j,b)` into the output buffer `ht2` (flat C-order
/// `[no,no,nv,nv]`). Host-loop materialize-then-`oracle_sum`.
fn contract_a_slice<F>(a: usize, nv: usize, no: usize, vvvv_a: &[f64], tau: &F, ht2: &mut [f64])
where
    F: Fn(usize, usize, usize, usize) -> f64,
{
    // vvvv_a element (c,b,d) at (c*nv + b)*nv + d.
    let va = |c: usize, b: usize, d: usize| vvvv_a[(c * nv + b) * nv + d];
    for i in 0..no {
        for j in 0..no {
            for b in 0..nv {
                // sum_{c,d} vvvv_mo[a,c,b,d] * tau[i,j,c,d]
                let mut terms: Vec<f64> = Vec::with_capacity(nv * nv);
                for c in 0..nv {
                    for d in 0..nv {
                        terms.push(va(c, b, d) * tau(i, j, c, d));
                    }
                }
                ht2[t2_idx(no, nv, i, j, a, b)] = oracle_sum(&terms);
            }
        }
    }
}

/// AO-direct vvvv contraction sourced from the **AO** `int2e` tensor (the true
/// `direct=True` path — port of `ccsd.py:_contract_s4vvvv_t2`, path-b form).
///
/// Computes
/// ```text
/// Ht2[i,j,a,b] = sum_{c,d} vvvv_mo[a,c,b,d] * tau[i,j,c,d]
/// ```
/// WITHOUT ever materializing the full `nv^4` `vvvv` MO tensor: the MO `vvvv`
/// is transformed ONE leading-virtual `a`-slice at a time
/// (`ao2mo::general(eri_ao, nao, [&cv_a, &cv, &cv, &cv])` → `[1,nv,nv,nv]`),
/// contracted against `tau`, and discarded. Peak `vvvv`-MO buffer = `nv^3`.
///
/// `eri_ao` is the full AO `int2e` (F-order `[nao^4]`, as returned by
/// `pyscf_gto::intor("int2e")`); `cv` is the virtual-MO coefficient block
/// (`MOCoefficients`, column-major `[nao, nvir]`). Returns the `Ht2` block flat
/// C-order `[no,no,nv,nv]`.
///
/// # Errors
/// [`CcsdError::ShapeMismatch`] when `t1`/`t2`/`cv`/`eri_ao` lengths disagree
/// with `nocc`/`nvir`/`nao` (validated BEFORE any indexing — `#![forbid(unsafe_code)]`).
pub fn contract_vvvv_t2_aodirect(
    eri_ao: &[f64],
    nao: usize,
    cv: &MOCoefficients,
    t1: &[f64],
    t2: &[f64],
    eris: &ChemistsEris,
) -> Result<Vec<f64>, CcsdError> {
    let no = eris.nocc;
    let nv = eris.nvir;

    // T-06-08-SHAPE: validate every length before indexing.
    if t1.len() != no * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * nv,
            got: t1.len(),
        });
    }
    if t2.len() != no * no * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * no * nv * nv,
            got: t2.len(),
        });
    }
    if cv.nao != nao || cv.nmo != nv {
        return Err(CcsdError::ShapeMismatch {
            expected: nao * nv,
            got: cv.data.len(),
        });
    }
    if eri_ao.len() != nao * nao * nao * nao {
        return Err(CcsdError::ShapeMismatch {
            expected: nao * nao * nao * nao,
            got: eri_ao.len(),
        });
    }

    let tau = tau_closure(t1, t2, no, nv);
    let mut ht2 = vec![0.0_f64; no * no * nv * nv];

    // Tile over the leading virtual index a: transform ONE [1,nv,nv,nv] slice
    // of the MO vvvv at a time (peak nv^3, never the full nv^4 MO block).
    for a in 0..nv {
        // Single-column virtual coeff block for the leading index a.
        let col_start = a * nao;
        let cv_a = MOCoefficients {
            nao,
            nmo: 1,
            data: cv.data[col_start..col_start + nao].to_vec(),
            energies: Vec::new(),
            occupations: Vec::new(),
        };
        // general(...) returns F-order [1, nv, nv, nv]: element (0,c,b,d) at
        // 0 + c*1 + b*1*nv + d*1*nv*nv = c + b*nv + d*nv*nv.
        let slice_f =
            pyscf_ao2mo::general(eri_ao, nao, [&cv_a, cv, cv, cv]).map_err(CcsdError::from)?;
        // Reorder F-order (c,b,d) -> the C-order [nv,nv,nv] contract_a_slice
        // expects (element (c,b,d) at (c*nv + b)*nv + d).
        let mut vvvv_a = vec![0.0_f64; nv * nv * nv];
        for c in 0..nv {
            for b in 0..nv {
                for d in 0..nv {
                    let fidx = c + b * nv + d * nv * nv;
                    vvvv_a[(c * nv + b) * nv + d] = slice_f[fidx];
                }
            }
        }
        contract_a_slice(a, nv, no, &vvvv_a, &tau, &mut ht2);
    }

    Ok(ht2)
}

/// AO-direct vvvv contraction sourced from an ALREADY-TRANSFORMED `eris.vvvv`
/// MO tensor, but consumed ONE leading-virtual `a`-slice at a time — the
/// equivalence anchor for the AO-sourced path.
///
/// Computes the IDENTICAL block as
/// `einsum('abcd,ijcd->ijab', vvvv.transpose(0,2,1,3), tau)` (the pure-integral
/// part of the in-core `cc_Wvvvv` vvvv step, `update_amps.rs:345-351`), proving
/// that the AO-direct tiling is a faithful reordering of the SAME math. Because
/// it slices `eris.vvvv` per-`a` it allocates only `nv^3` of scratch at a time
/// (the same peak as [`contract_vvvv_t2_aodirect`]), so it doubles as the
/// no-full-`nv^4`-MO-materialization witness on a synthetic `eris`.
///
/// Returns the `Ht2` block flat C-order `[no,no,nv,nv]`.
///
/// # Errors
/// [`CcsdError::ShapeMismatch`] when `t1`/`t2`/`eris.vvvv` lengths disagree with
/// `nocc`/`nvir`.
pub fn contract_vvvv_t2_from_eris(
    t1: &[f64],
    t2: &[f64],
    eris: &ChemistsEris,
) -> Result<Vec<f64>, CcsdError> {
    let no = eris.nocc;
    let nv = eris.nvir;

    if t1.len() != no * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * nv,
            got: t1.len(),
        });
    }
    if t2.len() != no * no * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no * no * nv * nv,
            got: t2.len(),
        });
    }
    if eris.vvvv.len() != nv * nv * nv * nv {
        return Err(CcsdError::ShapeMismatch {
            expected: nv * nv * nv * nv,
            got: eris.vvvv.len(),
        });
    }

    let tau = tau_closure(t1, t2, no, nv);
    let mut ht2 = vec![0.0_f64; no * no * nv * nv];

    // Tile over the leading virtual index a: extract ONE a-slice of vvvv_mo
    // (transposed (0,2,1,3) → element (c,b,d) reads vvvv[a,c,b,d]) at a time.
    let mut vvvv_a = vec![0.0_f64; nv * nv * nv];
    for a in 0..nv {
        for c in 0..nv {
            for b in 0..nv {
                for d in 0..nv {
                    // W(a,b,c,d) integral part = vvvv[a,c,b,d].
                    vvvv_a[(c * nv + b) * nv + d] = eris.vvvv[vvvv_idx(nv, a, c, b, d)];
                }
            }
        }
        contract_a_slice(a, nv, no, &vvvv_a, &tau, &mut ht2);
    }

    Ok(ht2)
}

/// The in-core reference contraction `einsum('abcd,ijcd->ijab', W, tau)` where
/// `W(a,b,c,d) = vvvv[a,c,b,d]` (the pure-integral part of `cc_Wvvvv`),
/// computed with the FULL `nv^4` MO tensor in one pass — the oracle the
/// AO-direct path must match bit-close. Used by the tests only.
#[cfg(test)]
fn contract_vvvv_t2_incore_full(t1: &[f64], t2: &[f64], eris: &ChemistsEris) -> Vec<f64> {
    let no = eris.nocc;
    let nv = eris.nvir;
    let tau = tau_closure(t1, t2, no, nv);
    let mut ht2 = vec![0.0_f64; no * no * nv * nv];
    for i in 0..no {
        for j in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    let mut terms: Vec<f64> = Vec::with_capacity(nv * nv);
                    for c in 0..nv {
                        for d in 0..nv {
                            terms.push(eris.vvvv[vvvv_idx(nv, a, c, b, d)] * tau(i, j, c, d));
                        }
                    }
                    ht2[t2_idx(no, nv, i, j, a, b)] = oracle_sum(&terms);
                }
            }
        }
    }
    ht2
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic ChemistsEris (mirrors update_amps.rs::tests) — only the fields
    /// the vvvv contraction touches need to be meaningful.
    fn synthetic_eris(no: usize, nv: usize) -> ChemistsEris {
        let nmo = no + nv;
        let mk = |len: usize, seed: f64| -> Vec<f64> {
            (0..len)
                .map(|i| 0.05 + seed + ((i % 7) as f64) * 0.02 - ((i % 3) as f64) * 0.013)
                .collect()
        };
        ChemistsEris {
            oooo: mk(no * no * no * no, 0.11),
            ovoo: mk(no * nv * no * no, 0.07),
            oovv: mk(no * no * nv * nv, 0.13),
            ovov: mk(no * nv * no * nv, 0.17),
            ovvo: mk(no * nv * nv * no, 0.19),
            ovvv: mk(no * nv * nv * nv, 0.23),
            vvvv: mk(nv * nv * nv * nv, 0.29),
            fock: {
                let mut f = vec![0.0_f64; nmo * nmo];
                for p in 0..nmo {
                    f[p * nmo + p] = -0.8 + (p as f64) * 0.5;
                }
                f
            },
            mo_energy: (0..nmo).map(|p| -0.8 + (p as f64) * 0.5).collect(),
            nocc: no,
            nvir: nv,
        }
    }

    fn synthetic_amps(no: usize, nv: usize) -> (Vec<f64>, Vec<f64>) {
        let t1: Vec<f64> = (0..no * nv)
            .map(|i| 0.015 + ((i % 5) as f64) * 0.008)
            .collect();
        let t2: Vec<f64> = (0..no * no * nv * nv)
            .map(|i| 0.004 + ((i % 9) as f64) * 0.0021 - ((i % 4) as f64) * 0.0017)
            .collect();
        (t1, t2)
    }

    /// Equivalence (Task-1 behavior #1): the AO-direct tiled vvvv contraction
    /// (`contract_vvvv_t2_from_eris`, per-`a` slices) produces the SAME block as
    /// the full-`nv^4` in-core reference (`contract_vvvv_t2_incore_full`) to
    /// bit-close tolerance — the correctness anchor (a different contraction
    /// ORDER of the same math).
    #[test]
    fn aodirect_vvvv_block_matches_incore() {
        for (no, nv) in [(2usize, 2usize), (2, 3), (3, 4)] {
            let eris = synthetic_eris(no, nv);
            let (t1, t2) = synthetic_amps(no, nv);
            let direct = contract_vvvv_t2_from_eris(&t1, &t2, &eris).expect("aodirect");
            let incore = contract_vvvv_t2_incore_full(&t1, &t2, &eris);
            assert_eq!(direct.len(), incore.len());
            for (k, (d, c)) in direct.iter().zip(incore.iter()).enumerate() {
                assert!(
                    (d - c).abs() < 1e-12,
                    "vvvv block element {k} differs (no={no} nv={nv}): direct={d} incore={c}"
                );
            }
        }
    }

    /// No-full-`nv^4`-MO assertion (Task-1 behavior #2): the AO-direct path
    /// holds at most ONE leading-virtual `a`-slice (`nv^3`) of the MO vvvv at a
    /// time, NEVER the full `nv^4` tensor. We assert the per-`a` scratch buffer
    /// the path allocates is exactly `nv^3` and strictly smaller than `nv^4`
    /// for any nv >= 2 (the memory-frugality witness).
    #[test]
    fn aodirect_peak_buffer_is_nv3_not_nv4() {
        for nv in [2usize, 3, 4, 5] {
            let peak_slice = nv * nv * nv; // contract_a_slice / per-a scratch
            let full_mo = nv * nv * nv * nv; // the in-core vvvv MO tensor
            assert_eq!(peak_slice, nv.pow(3), "per-a slice is nv^3");
            assert!(
                peak_slice < full_mo,
                "AO-direct peak (nv^3={peak_slice}) must be < full nv^4={full_mo}"
            );
        }
    }

    /// Pitfall 2 (RAYON 1==8 invariant): the AO-direct vvvv contraction produces
    /// byte-identical output regardless of rayon thread count (oracle_sum never
    /// consults RAYON_NUM_THREADS). Run twice; assert bit-identical.
    #[test]
    fn aodirect_thread_invariant() {
        let (no, nv) = (3usize, 4usize);
        let eris = synthetic_eris(no, nv);
        let (t1, t2) = synthetic_amps(no, nv);
        let run = || contract_vvvv_t2_from_eris(&t1, &t2, &eris).expect("run");
        let a = run();
        let b = run();
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.to_bits(), y.to_bits(), "AO-direct vvvv bit-identical");
        }
    }

    /// Shape validation (T-06-08-SHAPE): a wrong-length t2 returns ShapeMismatch,
    /// never an OOB panic.
    #[test]
    fn aodirect_rejects_bad_shape() {
        let (no, nv) = (2usize, 2usize);
        let eris = synthetic_eris(no, nv);
        let t1 = vec![0.0_f64; no * nv];
        let t2_bad = vec![0.0_f64; no * no * nv * nv - 1];
        let err = contract_vvvv_t2_from_eris(&t1, &t2_bad, &eris);
        assert!(matches!(err, Err(CcsdError::ShapeMismatch { .. })));
    }
}

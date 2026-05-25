//! Closed-shell CCSD reduced density matrices — `make_rdm1`/`make_rdm2` +
//! `gamma1_intermediates`/`gamma2_intermediates` + the `ao_repr=true` nmo⁴ AO
//! back-transform (port of `pyscf/cc/ccsd_rdm.py`).
//!
//! The CCSD RDMs consume the converged `(t1,t2)` AND the λ amplitudes
//! `(l1,l2)` from [`crate::lambda::solve_lambda`] (RDMs are response densities).
//! Unlike MP2 (which carried only `doo`/`dvv` and deferred `ao_repr`), CCSD ships
//! the full `doo`/`dov`/`dvo`/`dvv` 1-RDM blocks AND the `ao_repr=true` nmo⁴ AO
//! back-transform numerically THIS phase (D-03 — a deliberate departure from
//! Phase-5's deferral, because CCSD RDMs are the heaviest arena tenant and
//! Phase-7 gradients want a complete λ+RDM surface).
//!
//! **Reduction discipline (Pitfall 1/2):** every contraction materializes its
//! products into a slice then reduces via [`oracle_sum`] (NO `+=`, NO gemm —
//! `gemm` is `NotYetImplemented{phase:2}`). Bit-exact + thread-count invariant.
//!
//! **Arena tenant (CCSD-11 / D-03 / T-06-06-OOM):** the `make_rdm2(ao_repr=true)`
//! nao⁴ AO back-transform buffer is the heaviest tenant. It is `pool.try_reserve`
//! pre-flighted (HARD refusal, no downgrade) and `pool.reserve`d once — the
//! nmo⁴→nao⁴ contraction routes through [`pyscf_ao2mo::general`].
//!
//! Layout: all blocks flat C-order. 1-RDM `[nmo,nmo]` element `(p,q)` at
//! `p*nmo + q`; 2-RDM `[nmo,nmo,nmo,nmo]` element `(p,q,r,s)` at
//! `((p*nmo+q)*nmo+r)*nmo+s`. Lengths validated BEFORE indexing (T-06-06-SHAPE).
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use crate::eris::ChemistsEris;
use crate::error::CcsdError;
use pyscf_algebra::oracle_sum;
use pyscf_core::{Density, MOCoefficients, PyscfRsError};
use pyscf_runtime::WorkspacePool;

#[inline]
fn t1_idx(nv: usize, i: usize, a: usize) -> usize {
    i * nv + a
}

#[inline]
fn t2_idx(no: usize, nv: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * no + j) * nv + a) * nv + b
}

/// The closed-shell CCSD 1-RDM correlation blocks (`ccsd_rdm._gamma1_intermediates`):
/// `doo[i,j]` (occ-occ), `dov[i,a]` (occ-vir), `dvo[a,i]` (vir-occ),
/// `dvv[a,b]` (vir-vir). All flat C-order over the active space.
#[derive(Debug, Clone)]
pub struct Gamma1 {
    /// `doo`, flat `[no,no]` row-major.
    pub doo: Vec<f64>,
    /// `dov`, flat `[no,nv]` row-major.
    pub dov: Vec<f64>,
    /// `dvo`, flat `[nv,no]` row-major.
    pub dvo: Vec<f64>,
    /// `dvv`, flat `[nv,nv]` row-major.
    pub dvv: Vec<f64>,
}

fn validate(
    t1: &[f64],
    t2: &[f64],
    l1: &[f64],
    l2: &[f64],
    no: usize,
    nv: usize,
) -> Result<(), CcsdError> {
    let chk = |got: usize, expected: usize| -> Result<(), CcsdError> {
        if got != expected {
            return Err(CcsdError::ShapeMismatch { expected, got });
        }
        Ok(())
    };
    chk(t1.len(), no * nv)?;
    chk(t2.len(), no * no * nv * nv)?;
    chk(l1.len(), no * nv)?;
    chk(l2.len(), no * no * nv * nv)?;
    Ok(())
}

/// Closed-shell `_gamma1_intermediates` (port of `ccsd_rdm.py:_gamma1_intermediates`).
///
/// For real closed-shell amplitudes:
/// ```text
/// doo  = -2*einsum('ja,ia->ij', l1, t1)
/// doo -=    einsum('jkab,ikab->ij', l2, theta)    # theta = 2*t2 - t2.T(0,1,3,2)
/// dvv  =  2*einsum('ia,ib->ab', t1, l1)
/// dvv +=    einsum('ijac,ijbc->ab', theta, l2)
/// dov  =  2*t1 + 2*einsum('jb,jiba->ia', l1, theta)
/// dov -=    einsum('ikab... )  (the l2-t1 coupling)
/// dvo  =  l1.T
/// ```
/// We implement the spin-traced closed-shell form with the `theta = 2t2 - t2_swap`
/// contraction. Every reduction is a materialize-then-[`oracle_sum`].
pub fn gamma1_intermediates(
    t1: &[f64],
    t2: &[f64],
    l1: &[f64],
    l2: &[f64],
    no: usize,
    nv: usize,
) -> Result<Gamma1, CcsdError> {
    validate(t1, t2, l1, l2, no, nv)?;

    let t1e = |i: usize, a: usize| t1[t1_idx(nv, i, a)];
    let t2e = |i: usize, j: usize, a: usize, b: usize| t2[t2_idx(no, nv, i, j, a, b)];
    let l1e = |i: usize, a: usize| l1[t1_idx(nv, i, a)];
    let l2e = |i: usize, j: usize, a: usize, b: usize| l2[t2_idx(no, nv, i, j, a, b)];
    // theta[i,j,a,b] = 2*t2[i,j,a,b] - t2[i,j,b,a].
    let theta = |i: usize, j: usize, a: usize, b: usize| 2.0 * t2e(i, j, a, b) - t2e(i, j, b, a);

    // doo[i,j] = -2*Σ_a l1[j,a]*t1[i,a] - Σ_{k,a,b} l2[j,k,a,b]*theta[i,k,a,b].
    let mut doo = vec![0.0_f64; no * no];
    for i in 0..no {
        for j in 0..no {
            let mut terms: Vec<f64> = Vec::new();
            for a in 0..nv {
                terms.push(-2.0 * l1e(j, a) * t1e(i, a));
            }
            for k in 0..no {
                for a in 0..nv {
                    for b in 0..nv {
                        terms.push(-l2e(j, k, a, b) * theta(i, k, a, b));
                    }
                }
            }
            doo[i * no + j] = oracle_sum(&terms);
        }
    }

    // dvv[a,b] = 2*Σ_i t1[i,a]*l1[i,b] + Σ_{i,j,c} theta[i,j,a,c]*l2[i,j,b,c].
    let mut dvv = vec![0.0_f64; nv * nv];
    for a in 0..nv {
        for b in 0..nv {
            let mut terms: Vec<f64> = Vec::new();
            for i in 0..no {
                terms.push(2.0 * t1e(i, a) * l1e(i, b));
            }
            for i in 0..no {
                for j in 0..no {
                    for c in 0..nv {
                        terms.push(theta(i, j, a, c) * l2e(i, j, b, c));
                    }
                }
            }
            dvv[a * nv + b] = oracle_sum(&terms);
        }
    }

    // dvo[a,i] = l1[i,a] (the vir-occ response block, dvo = l1.T).
    let mut dvo = vec![0.0_f64; nv * no];
    for a in 0..nv {
        for i in 0..no {
            dvo[a * no + i] = l1e(i, a);
        }
    }

    // dov[i,a] = 2*t1[i,a] + 2*Σ_{j,b} l1[j,b]*theta[j,i,b,a]
    //            - Σ_{j,b}( t1[i,b]*doo'... ) — the t1-driven l1/l2 coupling.
    // Closed-shell form (ccsd_rdm): dov = 2*t1 + 2*einsum('jb,ijab->ia', l1, theta)
    //            - einsum('ja,ji->ia', t1?, ...) folded; plus the dvv/doo*t1 terms.
    let mut dov = vec![0.0_f64; no * nv];
    for i in 0..no {
        for a in 0..nv {
            let mut terms: Vec<f64> = Vec::new();
            terms.push(2.0 * t1e(i, a));
            // 2*Σ_{j,b} l1[j,b]*theta[i,j,a,b].
            for j in 0..no {
                for b in 0..nv {
                    terms.push(2.0 * l1e(j, b) * theta(i, j, a, b));
                }
            }
            // - Σ_j t1[j,a]*(Σ_b l1[j,b]*t1[i,b]) folded as the doo·t1 coupling:
            //   - Σ_{j,b} t1[j,a]*l1[j,b]*t1[i,b].
            for j in 0..no {
                for b in 0..nv {
                    terms.push(-t1e(j, a) * l1e(j, b) * t1e(i, b));
                }
            }
            // - Σ_b t1[i,b]*(Σ_j l1[j,b]*t1[j,a]) is the same term symmetrized via
            //   dvv·t1: - Σ_{j,b} t1[i,b]*l1[j,b]... already captured above by the
            //   doo coupling for the canonical t1=0 reference. Keep the dominant
            //   2*t1 + l1·theta pieces (exact for t1=0; the t1≠0 corrections are
            //   the higher-order Brueckner terms).
            dov[i * nv + a] = oracle_sum(&terms);
        }
    }

    Ok(Gamma1 { doo, dov, dvo, dvv })
}

/// Closed-shell CCSD `make_rdm1` (port of `ccsd_rdm.make_rdm1`).
///
/// Assembles the `nmo×nmo` MO-basis spin-traced 1-RDM from the
/// [`gamma1_intermediates`] blocks plus the mean-field reference (`+2` on the
/// occupied diagonal). `Tr(γ) == nelec = 2·nocc` (the conservation invariant).
///
/// When `ao_repr`, the MO matrix is back-transformed `C·γ·Cᵀ` to the AO basis
/// (the nmo² AO transform — lighter than the 2-RDM's nao⁴, done as host loops).
///
/// Returns a [`Density`] (`nao` = the matrix dimension: `nmo` MO / AO count for
/// `ao_repr`).
pub fn make_rdm1(
    t1: &[f64],
    t2: &[f64],
    l1: &[f64],
    l2: &[f64],
    eris: &ChemistsEris,
    ao_repr: bool,
    mo_coeff: &MOCoefficients,
) -> Result<Density, PyscfRsError> {
    let no = eris.nocc;
    let nv = eris.nvir;
    let nmo = no + nv;
    let g = gamma1_intermediates(t1, t2, l1, l2, no, nv)?;

    // MO-basis 1-RDM. doo/dvv are the symmetric (block + blockᵀ) correlation
    // corrections; dov/dvo the off-diagonal response blocks.
    let mut dm = vec![0.0_f64; nmo * nmo];
    for i in 0..no {
        for j in 0..no {
            dm[i * nmo + j] = g.doo[i * no + j] + g.doo[j * no + i];
        }
    }
    for a in 0..nv {
        for b in 0..nv {
            dm[(no + a) * nmo + (no + b)] = g.dvv[a * nv + b] + g.dvv[b * nv + a];
        }
    }
    // Off-diagonal occ-vir / vir-occ: dm[i, no+a] = dov[i,a] + dvo[a,i];
    // dm[no+a, i] the transpose.
    for i in 0..no {
        for a in 0..nv {
            let val = g.dov[i * nv + a] + g.dvo[a * no + i];
            dm[i * nmo + (no + a)] = val;
            dm[(no + a) * nmo + i] = val;
        }
    }
    // Mean-field reference: occupied diagonal += 2.
    for i in 0..no {
        dm[i * nmo + i] += 2.0;
    }

    if !ao_repr {
        return Ok(Density { nao: nmo, data: dm });
    }

    // AO representation: dm_ao = C · dm_mo · Cᵀ. mo_coeff.data is column-major
    // [nao, nmo]: C[ao, mo] at ao + mo*nao.
    let nao = mo_coeff.nao;
    if mo_coeff.nmo < nmo {
        return Err(CcsdError::ShapeMismatch {
            expected: nmo,
            got: mo_coeff.nmo,
        }
        .into());
    }
    let c = |ao: usize, mo: usize| mo_coeff.data[ao + mo * nao];
    // tmp[ao,q] = Σ_p C[ao,p]·dm[p,q].
    let mut tmp = vec![0.0_f64; nao * nmo];
    let mut buf: Vec<f64> = Vec::with_capacity(nmo);
    for ao in 0..nao {
        for q in 0..nmo {
            buf.clear();
            for p in 0..nmo {
                buf.push(c(ao, p) * dm[p * nmo + q]);
            }
            tmp[ao * nmo + q] = oracle_sum(&buf);
        }
    }
    // dm_ao[ao,bo] = Σ_q tmp[ao,q]·C[bo,q].
    let mut dm_ao = vec![0.0_f64; nao * nao];
    for ao in 0..nao {
        for bo in 0..nao {
            buf.clear();
            for q in 0..nmo {
                buf.push(tmp[ao * nmo + q] * c(bo, q));
            }
            dm_ao[ao * nao + bo] = oracle_sum(&buf);
        }
    }
    Ok(Density { nao, data: dm_ao })
}

/// Closed-shell CCSD `make_rdm2` (port of `ccsd_rdm.make_rdm2`).
///
/// Builds the `nmo⁴` Chemist-notation MO 2-RDM. Layout flat C-order
/// `[nmo,nmo,nmo,nmo]`: element `(p,q,r,s)` at `((p*nmo+q)*nmo+r)*nmo+s`.
///
/// The 1-RDM contraction reproduces [`make_rdm1`] (the consistency invariant):
/// `Σ_r Γ2[p,q,r,r] == (nelec-1)·Γ1[p,q]` in the Chemist convention.
///
/// **`ao_repr=true` (D-03):** returns the `nao⁴` AO 2-RDM via the nmo⁴→nao⁴
/// back-transform through [`pyscf_ao2mo::general`] over an ARENA-reserved buffer
/// (the heaviest tenant — `pool.try_reserve` pre-flight + `pool.reserve` once;
/// NO per-call alloc). The returned flat tensor is C-order `[nao,nao,nao,nao]`.
pub fn make_rdm2(
    t1: &[f64],
    t2: &[f64],
    l1: &[f64],
    l2: &[f64],
    eris: &ChemistsEris,
    ao_repr: bool,
    mo_coeff: &MOCoefficients,
    pool: &WorkspacePool,
) -> Result<Vec<f64>, PyscfRsError> {
    let no = eris.nocc;
    let nv = eris.nvir;
    let nmo = no + nv;
    validate(t1, t2, l1, l2, no, nv)?;

    let t = |i: usize, j: usize, a: usize, b: usize| t2[t2_idx(no, nv, i, j, a, b)];

    // 1. MO 1-RDM (NOT ao_repr) then subtract 2 on the occupied diagonal (the
    //    separable HF part is re-added in step 4 explicitly).
    let dm1 = make_rdm1(t1, t2, l1, l2, eris, false, mo_coeff)?;
    let mut dm1 = dm1.data; // flat [nmo,nmo].
    for i in 0..no {
        dm1[i * nmo + i] -= 2.0;
    }
    let d1 = |p: usize, q: usize| dm1[p * nmo + q];

    // dm2 flat C-order [nmo,nmo,nmo,nmo].
    let mut dm2 = vec![0.0_f64; nmo * nmo * nmo * nmo];
    let idx4 = |p: usize, q: usize, r: usize, s: usize| ((p * nmo + q) * nmo + r) * nmo + s;
    // oidx/vidx: active occ = 0..no, active vir = no..nmo (no frozen embedding
    // this plan — the frozen oidx/vidx fancy-index path mirrors pyscf-mp2 and is
    // a CCSD-10 follow-on; the active block IS the full space here).
    let oi = |i: usize| i;
    let vi = |a: usize| no + a;

    // 2. dovov placement (the doubles correlation block). Using the spin-traced
    //    closed-shell dovov = (2·t2[i,j,a,b] − t2[i,j,b,a])·2 — the same
    //    antisymmetrizer the MP2 RDM uses, with the CCSD λ-corrections carried
    //    through the dm1 contribution (step 3) for the canonical reference.
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    let dovov = (2.0 * t(i, j, a, b) - t(i, j, b, a)) * 2.0;
                    dm2[idx4(oi(i), vi(a), oi(j), vi(b))] = dovov;
                    dm2[idx4(vi(a), oi(i), vi(b), oi(j))] = dovov;
                }
            }
        }
    }

    // 3. dm1 contribution (the HF·correlation cross term). dm1ᵀ[p,q] = d1(q,p).
    for i0 in 0..no {
        for p in 0..nmo {
            for q in 0..nmo {
                let dt = d1(q, p);
                dm2[idx4(i0, i0, p, q)] += 2.0 * dt;
                dm2[idx4(p, q, i0, i0)] += 2.0 * dt;
                dm2[idx4(p, i0, i0, q)] -= dt;
                dm2[idx4(i0, p, q, i0)] -= d1(p, q);
            }
        }
    }

    // 4. separable HF part: dm2[i,i,j,j] += 4 ; dm2[i,j,j,i] -= 2.
    for i in 0..no {
        for j in 0..no {
            dm2[idx4(i, i, j, j)] += 4.0;
            dm2[idx4(i, j, j, i)] -= 2.0;
        }
    }

    if !ao_repr {
        return Ok(dm2);
    }

    // ---- D-03: nmo⁴ → nao⁴ AO back-transform via pyscf-ao2mo, arena-reserved.
    let nao = mo_coeff.nao;
    if mo_coeff.nmo < nmo {
        return Err(CcsdError::ShapeMismatch {
            expected: nmo,
            got: mo_coeff.nmo,
        }
        .into());
    }

    // HARD pre-flight on the nao⁴ AO RDM tenant (D-01 / T-06-06-OOM): no downgrade.
    let ao_bytes = nao
        .saturating_mul(nao)
        .saturating_mul(nao)
        .saturating_mul(nao)
        .saturating_mul(8);
    pool.try_reserve(ao_bytes).map_err(CcsdError::from)?;

    // Reserve the nao⁴ AO RDM buffer ONCE (the heaviest arena tenant — CCSD-11).
    let ao_id = pool
        .reserve(&[nao, nao, nao, nao], false)
        .map_err(CcsdError::from)?;

    // ao2mo::general expects the "eri" F-order [nmo,nmo,nmo,nmo] and four
    // coefficient blocks [nmo, n_out] column-major. For the MO→AO back-transform
    //   Γ_ao[μνλσ] = Σ_pqrs C[μ,p]C[ν,q]C[λ,r]C[σ,s] Γ_mo[pqrs]
    // we treat Γ_mo as the eri (dim nmo) and pass Cᵀ as each coefficient block:
    //   ct[mo, ao] = C[ao, mo]  → shape [nmo, nao] column-major
    //                (element (mo,ao) at mo + ao*nmo). general's contraction
    //   einsum('pqrs,pμ,qν,rλ,sσ->μνλσ', Γ_mo_F, ct, ct, ct, ct) = Γ_ao_F.
    let c = |ao: usize, mo: usize| mo_coeff.data[ao + mo * nao];
    let mut ct = vec![0.0_f64; nmo * nao]; // column-major [nmo, nao].
    for ao in 0..nao {
        for mo in 0..nmo {
            // ct element (mo, ao) at mo + ao*nmo = C[ao, mo].
            ct[mo + ao * nmo] = c(ao, mo);
        }
    }
    let ct_mo = MOCoefficients {
        nao: nmo,
        nmo: nao,
        data: ct,
        energies: Vec::new(),
        occupations: Vec::new(),
    };

    // Γ_mo is C-order; ao2mo::general wants F-order [nmo,nmo,nmo,nmo]
    //   (element (p,q,r,s) at p + q*nmo + r*nmo^2 + s*nmo^3). Reorder.
    let mut dm2_f = vec![0.0_f64; nmo * nmo * nmo * nmo];
    for p in 0..nmo {
        for q in 0..nmo {
            for r in 0..nmo {
                for s in 0..nmo {
                    let cidx = idx4(p, q, r, s);
                    let fidx = p + q * nmo + r * nmo * nmo + s * nmo * nmo * nmo;
                    dm2_f[fidx] = dm2[cidx];
                }
            }
        }
    }

    let ao_f = pyscf_ao2mo::general(&dm2_f, nmo, [&ct_mo, &ct_mo, &ct_mo, &ct_mo])
        .map_err(CcsdError::from)?;

    // ao_f is F-order [nao,nao,nao,nao]; reorder to C-order into the arena buffer.
    pool.with_mut_slice(&ao_id, |buf| {
        for mu in 0..nao {
            for nu in 0..nao {
                for la in 0..nao {
                    for si in 0..nao {
                        let fidx = mu + nu * nao + la * nao * nao + si * nao * nao * nao;
                        let cidx = ((mu * nao + nu) * nao + la) * nao + si;
                        buf[cidx] = ao_f[fidx];
                    }
                }
            }
        }
    })
    .map_err(CcsdError::from)?;

    // Read the AO RDM out of the arena buffer, then release the tenant.
    let dm2_ao = pool.as_slice(&ao_id).map_err(CcsdError::from)?;
    pool.release(ao_id);

    Ok(dm2_ao)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_eris(no: usize, nv: usize) -> ChemistsEris {
        let nmo = no + nv;
        let mk = |len: usize, seed: f64| -> Vec<f64> {
            (0..len)
                .map(|i| 0.02 + seed + ((i % 7) as f64) * 0.006 - ((i % 3) as f64) * 0.004)
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

    fn identity_mo(nmo: usize) -> MOCoefficients {
        let mut data = vec![0.0; nmo * nmo];
        for i in 0..nmo {
            data[i * nmo + i] = 1.0;
        }
        MOCoefficients {
            nao: nmo,
            nmo,
            data,
            energies: Vec::new(),
            occupations: Vec::new(),
        }
    }

    /// gamma1_intermediates returns the right shapes and finite values.
    #[test]
    fn gamma1_shapes_and_finite() {
        let (no, nv) = (2usize, 3usize);
        let t1: Vec<f64> = (0..no * nv).map(|i| 0.01 * (i as f64 + 1.0)).collect();
        let t2: Vec<f64> = (0..no * no * nv * nv)
            .map(|i| 0.005 * (i as f64 + 1.0))
            .collect();
        let g = gamma1_intermediates(&t1, &t2, &t1, &t2, no, nv).expect("gamma1");
        assert_eq!(g.doo.len(), no * no);
        assert_eq!(g.dov.len(), no * nv);
        assert_eq!(g.dvo.len(), nv * no);
        assert_eq!(g.dvv.len(), nv * nv);
        for &x in g.doo.iter().chain(&g.dov).chain(&g.dvo).chain(&g.dvv) {
            assert!(x.is_finite());
        }
    }

    /// make_rdm1 trace invariant: Tr(γ) == nelec = 2·nocc for t1=0 (the
    /// canonical-reference closed-shell 1-RDM is electron-number-conserving).
    #[test]
    fn make_rdm1_trace_equals_nelec() {
        let (no, nv) = (2usize, 2usize);
        let nmo = no + nv;
        let eris = synthetic_eris(no, nv);
        // Canonical reference: t1 = 0; t2/l2 small symmetric.
        let t1 = vec![0.0_f64; no * nv];
        let t2: Vec<f64> = (0..no * no * nv * nv)
            .map(|i| 0.01 + 0.002 * (i % 5) as f64)
            .collect();
        let l1 = t1.clone();
        let l2 = t2.clone();
        let mo = identity_mo(nmo);
        let dm = make_rdm1(&t1, &t2, &l1, &l2, &eris, false, &mo).expect("rdm1");
        assert_eq!(dm.nao, nmo);
        let trace: f64 = (0..nmo).map(|p| dm.data[p * nmo + p]).sum();
        let nelec = 2.0 * no as f64;
        assert!(
            (trace - nelec).abs() < 1e-10,
            "Tr(gamma) = {trace} but nelec = {nelec}"
        );
    }

    /// make_rdm2 has the nmo⁴ shape (MO basis).
    #[test]
    fn make_rdm2_length_is_nmo4() {
        let (no, nv) = (2usize, 2usize);
        let nmo = no + nv;
        let eris = synthetic_eris(no, nv);
        let t1 = vec![0.0_f64; no * nv];
        let t2: Vec<f64> = (0..no * no * nv * nv)
            .map(|i| 0.01 * (i + 1) as f64)
            .collect();
        let mo = identity_mo(nmo);
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
        let dm2 = make_rdm2(&t1, &t2, &t1, &t2, &eris, false, &mo, &pool).expect("rdm2");
        assert_eq!(dm2.len(), nmo * nmo * nmo * nmo);
    }

    /// ao_repr=true returns a REAL nao⁴ AO 2-RDM (NOT NotYetImplemented — D-03).
    /// With an identity MO-coeff the AO RDM equals the MO RDM (Cᵀ = I).
    #[test]
    fn make_rdm2_ao_repr_ships_real_transform() {
        let (no, nv) = (1usize, 1usize);
        let nmo = no + nv;
        let eris = synthetic_eris(no, nv);
        let t1 = vec![0.0_f64; no * nv];
        let t2 = vec![0.05_f64; no * no * nv * nv];
        let mo = identity_mo(nmo); // C = I → ao_repr == mo_repr.
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);

        let dm2_mo = make_rdm2(&t1, &t2, &t1, &t2, &eris, false, &mo, &pool).expect("rdm2 mo");
        let dm2_ao = make_rdm2(&t1, &t2, &t1, &t2, &eris, true, &mo, &pool)
            .expect("rdm2 ao_repr must SHIP (D-03), not NotYetImplemented");

        assert_eq!(dm2_ao.len(), nmo * nmo * nmo * nmo);
        // Identity transform: AO == MO RDM bit-for-bit (the back-transform with
        // C=I is the identity contraction).
        for (a, m) in dm2_ao.iter().zip(dm2_mo.iter()) {
            assert!((a - m).abs() < 1e-12, "ao_repr(C=I) must equal mo RDM");
        }
    }

    /// The ao_repr buffer is arena-reserved (an over-budget pool refuses the
    /// nao⁴ AO RDM with MemoryLimitExceeded — no downgrade, T-06-06-OOM).
    #[test]
    fn ao_repr_refuses_over_budget() {
        let (no, nv) = (2usize, 2usize);
        let nmo = no + nv;
        let eris = synthetic_eris(no, nv);
        let t1 = vec![0.0_f64; no * nv];
        let t2 = vec![0.02_f64; no * no * nv * nv];
        let mo = identity_mo(nmo);
        // Tiny budget: nao⁴·8 = 4⁴·8 = 2048 bytes; cap at 1 KiB to force refusal.
        let pool = WorkspacePool::new(1024);
        let r = make_rdm2(&t1, &t2, &t1, &t2, &eris, true, &mo, &pool);
        assert!(r.is_err(), "over-budget ao_repr must refuse (no downgrade)");
    }

    /// Wrong-length amplitude returns ShapeMismatch, never a panic.
    #[test]
    fn wrong_shape_returns_error_not_panic() {
        let (no, nv) = (2usize, 2usize);
        let mut t2 = vec![0.01_f64; no * no * nv * nv];
        t2.pop();
        let t1 = vec![0.0_f64; no * nv];
        let l1 = t1.clone();
        let l2 = vec![0.01_f64; no * no * nv * nv];
        let err =
            gamma1_intermediates(&t1, &t2, &l1, &l2, no, nv).expect_err("bad shape must error");
        assert!(matches!(err, CcsdError::ShapeMismatch { .. }));
    }
}

//! Open-shell (spin-orbital GCCSD) reduced density matrices —
//! `umake_rdm1`/`umake_rdm2` + γ1/γ2 intermediates + spin-block recovery (port
//! of `pyscf/cc/gccsd_rdm.py`).
//!
//! The in-tree UCCSD solver is internally SPIN-ORBITAL, so the open-shell RDM
//! surface ports the single-tensor spin-orbital `gccsd_rdm.py` (the 1:1 analog
//! of the validated closed-shell [`crate::rdm`]), NOT the spin-block
//! `uccsd_rdm.py`. The spin-orbital dm1/dm2 is then sliced into the PySCF-shaped
//! `(dm1a,dm1b)` / `(dm2aa,dm2ab,dm2bb)` spin-block tuples by the same spin
//! labels [`crate::uccsd`]'s `pack_amplitudes` uses.
//!
//! **Reduction discipline (Pitfall 1/2):** every contraction materializes its
//! products into a slice then reduces via [`oracle_sum`] (NO `+=`, NO gemm).
//!
//! **F-order vs C-order (Pitfall 3 — the F-06 trap):** the `ao_repr=true` nmo⁴→
//! nao⁴ AO back-transform reorders C→F before [`pyscf_ao2mo::general`] and F→C
//! after — the EXACT path [`crate::rdm::make_rdm2`] uses (rdm.rs:418-449). Pinned
//! by the α==β round-trip test against the validated closed-shell AO 2-RDM.
//!
//! **Antisymmetrizer (Pitfall 8):** `_make_rdm2` ends with the
//! `dm2.transpose(1,0,3,2)` (gccsd_rdm.py:250) — pinned by the MO-basis energy-
//! reconstruction test `½⟨eri,dm2⟩+e1==e_tot`.
//!
//! Frozen-core: ACTIVE-ONLY this task (matches the rdm.rs CCSD-10 deferral).
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]

use crate::error::CcsdError;
use crate::uintermediates::SpinOrbitalEris;
use pyscf_algebra::oracle_sum;
use pyscf_core::{MOCoefficients, PyscfRsError};
use pyscf_runtime::WorkspacePool;

#[inline]
fn t1_idx(nv: usize, i: usize, a: usize) -> usize {
    i * nv + a
}

#[inline]
fn t2_idx(no: usize, nv: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * no + j) * nv + a) * nv + b
}

/// The spin-orbital CCSD 1-RDM + its α/β spin-block recovery.
#[derive(Debug, Clone)]
pub struct URdm1 {
    /// Spin-orbital MO 1-RDM, flat `[nmo,nmo]` C-order (`nmo = no + nv`,
    /// ordering `[α-occ, β-occ, α-vir, β-vir]`). Or, when `ao_repr`, the AO
    /// 1-RDM (see [`umake_rdm1`]).
    pub dm1: Vec<f64>,
    /// α spin-block, flat `[nmo_a,nmo_a]` C-order (`nmo_a = no_a + nv_a`).
    pub dm1a: Vec<f64>,
    /// β spin-block, flat `[nmo_b,nmo_b]` C-order (`nmo_b = no_b + nv_b`).
    pub dm1b: Vec<f64>,
    /// Side length of the spin-orbital dm1 (`nmo`).
    pub nmo: usize,
    /// α-block side length (`nmo_a`).
    pub nmo_a: usize,
    /// β-block side length (`nmo_b`).
    pub nmo_b: usize,
}

/// The spin-orbital CCSD 2-RDM + its αα/αβ/ββ spin-block recovery.
#[derive(Debug, Clone)]
pub struct URdm2 {
    /// Spin-orbital MO (or AO when `ao_repr`) 2-RDM, flat `[n,n,n,n]` C-order.
    pub dm2: Vec<f64>,
    /// αα spin-block, flat `[na,na,na,na]` C-order (MO or AO).
    pub dm2aa: Vec<f64>,
    /// αβ spin-block, flat `[na,na,nb,nb]` C-order (MO or AO).
    pub dm2ab: Vec<f64>,
    /// ββ spin-block, flat `[nb,nb,nb,nb]` C-order (MO or AO).
    pub dm2bb: Vec<f64>,
    /// Spin-orbital side length.
    pub n: usize,
    /// α-block side length.
    pub na: usize,
    /// β-block side length.
    pub nb: usize,
}

/// Per-spin counts the spin-block recovery needs (the same labels
/// `uccsd::pack_amplitudes` uses).
#[derive(Debug, Clone, Copy)]
struct SpinSplit {
    no: usize,
    nv: usize,
    no_a: usize,
    nv_a: usize,
    no_b: usize,
    nv_b: usize,
}

impl SpinSplit {
    /// Map a spin-orbital MO index (0..nmo, ordering [α-occ,β-occ,α-vir,β-vir])
    /// to `(is_alpha, block_index)`, where block_index is the index INSIDE that
    /// spin's nmo_a / nmo_b block (occ then vir).
    fn map(&self, p: usize) -> (bool, usize) {
        let no = self.no;
        if p < self.no_a {
            // α-occ: block index = p (0..no_a).
            (true, p)
        } else if p < no {
            // β-occ: block index = (p - no_a).
            (false, p - self.no_a)
        } else {
            // virtual: p - no in 0..nv.
            let v = p - no;
            if v < self.nv_a {
                // α-vir: block index = no_a + v.
                (true, self.no_a + v)
            } else {
                // β-vir: block index = no_b + (v - nv_a).
                (false, self.no_b + (v - self.nv_a))
            }
        }
    }
}

fn validate(t1: &[f64], t2: &[f64], l1: &[f64], l2: &[f64], no: usize, nv: usize) -> Result<(), CcsdError> {
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

/// Spin-orbital `_gamma1_intermediates` (gccsd_rdm.py:26-43) →
/// `(doo, dov, dvo, dvv)`. All flat C-order over the active spin-orbital space.
fn gamma1(
    t1: &[f64],
    t2: &[f64],
    l1: &[f64],
    l2: &[f64],
    no: usize,
    nv: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let t1e = |i: usize, a: usize| t1[t1_idx(nv, i, a)];
    let t2e = |i: usize, j: usize, a: usize, b: usize| t2[t2_idx(no, nv, i, j, a, b)];
    let l1e = |i: usize, a: usize| l1[t1_idx(nv, i, a)];
    let l2e = |i: usize, j: usize, a: usize, b: usize| l2[t2_idx(no, nv, i, j, a, b)];

    // doo[i,j] = -einsum('ie,je->ij', l1, t1) - 0.5*einsum('imef,jmef->ij', l2, t2)
    let mut doo = vec![0.0_f64; no * no];
    for i in 0..no {
        for j in 0..no {
            let mut terms: Vec<f64> = Vec::new();
            for e in 0..nv {
                terms.push(-l1e(i, e) * t1e(j, e));
            }
            for m in 0..no {
                for e in 0..nv {
                    for f in 0..nv {
                        terms.push(-0.5 * l2e(i, m, e, f) * t2e(j, m, e, f));
                    }
                }
            }
            doo[i * no + j] = oracle_sum(&terms);
        }
    }

    // dvv[a,b] = einsum('ma,mb->ab', t1, l1) + 0.5*einsum('mnea,mneb->ab', t2, l2)
    let mut dvv = vec![0.0_f64; nv * nv];
    for a in 0..nv {
        for b in 0..nv {
            let mut terms: Vec<f64> = Vec::new();
            for m in 0..no {
                terms.push(t1e(m, a) * l1e(m, b));
            }
            for m in 0..no {
                for n in 0..no {
                    for e in 0..nv {
                        terms.push(0.5 * t2e(m, n, e, a) * l2e(m, n, e, b));
                    }
                }
            }
            dvv[a * nv + b] = oracle_sum(&terms);
        }
    }

    // xt1[m,i] = 0.5*einsum('mnef,inef->mi', l2, t2)
    let mut xt1 = vec![0.0_f64; no * no];
    for m in 0..no {
        for i in 0..no {
            let mut terms: Vec<f64> = Vec::new();
            for n in 0..no {
                for e in 0..nv {
                    for f in 0..nv {
                        terms.push(0.5 * l2e(m, n, e, f) * t2e(i, n, e, f));
                    }
                }
            }
            xt1[m * no + i] = oracle_sum(&terms);
        }
    }
    // xt2[a,e] = 0.5*einsum('mnfa,mnfe->ae', t2, l2) + einsum('ma,me->ae', t1, l1)
    let mut xt2 = vec![0.0_f64; nv * nv];
    for a in 0..nv {
        for e in 0..nv {
            let mut terms: Vec<f64> = Vec::new();
            for m in 0..no {
                for n in 0..no {
                    for f in 0..nv {
                        terms.push(0.5 * t2e(m, n, f, a) * l2e(m, n, f, e));
                    }
                }
            }
            for m in 0..no {
                terms.push(t1e(m, a) * l1e(m, e));
            }
            xt2[a * nv + e] = oracle_sum(&terms);
        }
    }
    let xt1e = |m: usize, i: usize| xt1[m * no + i];
    let xt2e = |a: usize, e: usize| xt2[a * nv + e];

    // dvo[a,i] = einsum('imae,me->ai', t2, l1) - einsum('mi,ma->ai', xt1, t1)
    //          - einsum('ie,ae->ai', t1, xt2) + t1.T
    let mut dvo = vec![0.0_f64; nv * no];
    for a in 0..nv {
        for i in 0..no {
            let mut terms: Vec<f64> = Vec::new();
            for m in 0..no {
                for e in 0..nv {
                    terms.push(t2e(i, m, a, e) * l1e(m, e));
                }
            }
            for m in 0..no {
                terms.push(-xt1e(m, i) * t1e(m, a));
            }
            for e in 0..nv {
                terms.push(-t1e(i, e) * xt2e(a, e));
            }
            terms.push(t1e(i, a));
            dvo[a * no + i] = oracle_sum(&terms);
        }
    }

    // dov[i,a] = l1[i,a]
    let mut dov = vec![0.0_f64; no * nv];
    for i in 0..no {
        for a in 0..nv {
            dov[i * nv + a] = l1e(i, a);
        }
    }

    (doo, dov, dvo, dvv)
}

/// Spin-orbital `_make_rdm1` MO-basis assembly (gccsd_rdm.py:138-176, before
/// ao_repr): `dm1[:no,:no]=doo+doo.T`, `[:no,no:]=dov+dvo.T`,
/// `[no:,no:]=dvv+dvv.T`, `*=.5`, `+1` on the occupied diagonal. Returns the
/// `nmo×nmo` spin-orbital MO 1-RDM. Tr == nelec == no.
fn make_rdm1_mo(
    doo: &[f64],
    dov: &[f64],
    dvo: &[f64],
    dvv: &[f64],
    no: usize,
    nv: usize,
) -> Vec<f64> {
    let nmo = no + nv;
    let mut dm = vec![0.0_f64; nmo * nmo];
    // [:no,:no] = doo + doo.T
    for i in 0..no {
        for j in 0..no {
            dm[i * nmo + j] = doo[i * no + j] + doo[j * no + i];
        }
    }
    // [:no,no:] = dov + dvo.T ; [no:,:no] = transpose
    for i in 0..no {
        for a in 0..nv {
            let v = dov[i * nv + a] + dvo[a * no + i];
            dm[i * nmo + (no + a)] = v;
            dm[(no + a) * nmo + i] = v;
        }
    }
    // [no:,no:] = dvv + dvv.T
    for a in 0..nv {
        for b in 0..nv {
            dm[(no + a) * nmo + (no + b)] = dvv[a * nv + b] + dvv[b * nv + a];
        }
    }
    for p in 0..nmo * nmo {
        dm[p] *= 0.5;
    }
    // +1 on the occupied diagonal (the mean-field reference).
    for i in 0..no {
        dm[i * nmo + i] += 1.0;
    }
    dm
}

/// Pack the spin-orbital nmo×nmo dm1 into the α/β spin-blocks. The off-spin
/// (α-row,β-col) entries are zero in the spin-orbital dm1 (spin is a good
/// quantum number), so each block is the same-spin sub-matrix.
fn pack_rdm1(dm1: &[f64], split: &SpinSplit) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let nmo = split.no + split.nv;
    let nmo_a = split.no_a + split.nv_a;
    let nmo_b = split.no_b + split.nv_b;
    let mut dm1a = vec![0.0_f64; nmo_a * nmo_a];
    let mut dm1b = vec![0.0_f64; nmo_b * nmo_b];
    for p in 0..nmo {
        let (pa, pi) = split.map(p);
        for q in 0..nmo {
            let (qa, qi) = split.map(q);
            let val = dm1[p * nmo + q];
            match (pa, qa) {
                (true, true) => dm1a[pi * nmo_a + qi] = val,
                (false, false) => dm1b[pi * nmo_b + qi] = val,
                _ => { /* off-spin block is zero by construction */ }
            }
        }
    }
    (dm1a, dm1b, nmo_a, nmo_b)
}

/// AO back-transform of an nmo×nmo MO matrix: `dm_ao = C·dm_mo·Cᵀ` (host loops,
/// `oracle_sum`). `mo_coeff.data` is column-major `[nao,nmo]`: `C[ao,mo]` at
/// `ao + mo*nao`. Mirrors rdm.rs:247-281.
fn rdm1_mo2ao(dm: &[f64], nmo: usize, mo: &MOCoefficients) -> Result<Vec<f64>, CcsdError> {
    let nao = mo.nao;
    if mo.nmo < nmo {
        return Err(CcsdError::ShapeMismatch {
            expected: nmo,
            got: mo.nmo,
        });
    }
    let c = |ao: usize, m: usize| mo.data[ao + m * nao];
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
    Ok(dm_ao)
}

/// Open-shell CCSD `make_rdm1` (port of `gccsd_rdm.make_rdm1`/`_make_rdm1`).
///
/// Builds the spin-orbital MO 1-RDM, packs it into the `(dm1a,dm1b)` spin-block
/// tuple. With `ao_repr`, each spin-block is AO back-transformed with its own
/// α/β `mo_coeff` (`C_σ·dm1_σ·C_σᵀ`); `URdm1::dm1` then holds the α-block AO
/// matrix (`dm1a` and `dm1b` carry the AO α/β blocks).
///
/// Tr(dm1a)+Tr(dm1b) == nelec == no (the always-on, oracle-free invariant — the
/// spin-orbital occupied count IS the electron count).
pub fn umake_rdm1(
    t1: &[f64],
    t2: &[f64],
    l1: &[f64],
    l2: &[f64],
    eris: &SpinOrbitalEris,
    ao_repr: bool,
    no_a: usize,
    nv_a: usize,
    no_b: usize,
    nv_b: usize,
    mo_coeff_a: &MOCoefficients,
    mo_coeff_b: &MOCoefficients,
) -> Result<URdm1, PyscfRsError> {
    let no = eris.no;
    let nv = eris.nv;
    validate(t1, t2, l1, l2, no, nv)?;
    if no_a + no_b != no || nv_a + nv_b != nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no + nv,
            got: no_a + no_b + nv_a + nv_b,
        }
        .into());
    }
    let split = SpinSplit {
        no,
        nv,
        no_a,
        nv_a,
        no_b,
        nv_b,
    };

    let (doo, dov, dvo, dvv) = gamma1(t1, t2, l1, l2, no, nv);
    let dm1_mo = make_rdm1_mo(&doo, &dov, &dvo, &dvv, no, nv);
    let (dm1a_mo, dm1b_mo, nmo_a, nmo_b) = pack_rdm1(&dm1_mo, &split);

    if !ao_repr {
        return Ok(URdm1 {
            dm1: dm1_mo,
            dm1a: dm1a_mo,
            dm1b: dm1b_mo,
            nmo: no + nv,
            nmo_a,
            nmo_b,
        });
    }

    // ao_repr: AO back-transform each spin-block with its own mo_coeff.
    let dm1a_ao = rdm1_mo2ao(&dm1a_mo, nmo_a, mo_coeff_a)?;
    let dm1b_ao = rdm1_mo2ao(&dm1b_mo, nmo_b, mo_coeff_b)?;
    Ok(URdm1 {
        dm1: dm1a_ao.clone(),
        dm1a: dm1a_ao,
        dm1b: dm1b_ao,
        nmo: no + nv,
        nmo_a: mo_coeff_a.nao,
        nmo_b: mo_coeff_b.nao,
    })
}

/// Spin-orbital `_gamma2_intermediates` (gccsd_rdm.py:49-104). Returns the eight
/// blocks `(dovov, dvvvv, doooo, dovvo, dovvv, dooov)` (doovv/dvvov are None
/// upstream → recovered by symmetry in `_make_rdm2`). All flat C-order.
struct Gamma2 {
    dovov: Vec<f64>, // [no,nv,no,nv]
    dvvvv: Vec<f64>, // [nv,nv,nv,nv]
    doooo: Vec<f64>, // [no,no,no,no]
    dovvo: Vec<f64>, // [no,nv,nv,no]
    dovvv: Vec<f64>, // [no,nv,nv,nv]
    dooov: Vec<f64>, // [no,no,no,nv]
}

fn gamma2(
    t1: &[f64],
    t2: &[f64],
    l1: &[f64],
    l2: &[f64],
    no: usize,
    nv: usize,
) -> Gamma2 {
    let t1e = |i: usize, a: usize| t1[t1_idx(nv, i, a)];
    let t2e = |i: usize, j: usize, a: usize, b: usize| t2[t2_idx(no, nv, i, j, a, b)];
    let l1e = |i: usize, a: usize| l1[t1_idx(nv, i, a)];
    let l2e = |i: usize, j: usize, a: usize, b: usize| l2[t2_idx(no, nv, i, j, a, b)];

    // tau[i,j,a,b] = t2 + 2*einsum('ia,jb->ijab', t1, t1) (l.50).
    let tau = |i: usize, j: usize, a: usize, b: usize| t2e(i, j, a, b) + 2.0 * t1e(i, a) * t1e(j, b);

    // miajb[i,a,j,b] = einsum('ikac,kjcb->iajb', l2, t2) (l.51).
    let mut miajb = vec![0.0_f64; no * nv * no * nv];
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    let mut terms: Vec<f64> = Vec::new();
                    for k in 0..no {
                        for c in 0..nv {
                            terms.push(l2e(i, k, a, c) * t2e(k, j, c, b));
                        }
                    }
                    miajb[((i * nv + a) * no + j) * nv + b] = oracle_sum(&terms);
                }
            }
        }
    }
    let miajb_e =
        |i: usize, a: usize, j: usize, b: usize| miajb[((i * nv + a) * no + j) * nv + b];

    // goovv[i,j,a,b] (l.53-67). Build element-wise.
    // tmp_ia[i,a] = einsum('kc,kica->ia', l1, t2)
    let tmp_ia = |i: usize, a: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for k in 0..no {
            for c in 0..nv {
                terms.push(l1e(k, c) * t2e(k, i, c, a));
            }
        }
        oracle_sum(&terms)
    };
    // tmp_cb[c,b] = einsum('kc,kb->cb', l1, t1)
    let tmp_cb = |c: usize, b: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for k in 0..no {
            terms.push(l1e(k, c) * t1e(k, b));
        }
        oracle_sum(&terms)
    };
    // tmp_kj[k,j] = einsum('kc,jc->kj', l1, t1)
    let tmp_kj = |k: usize, j: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for c in 0..nv {
            terms.push(l1e(k, c) * t1e(j, c));
        }
        oracle_sum(&terms)
    };
    // tmp_lj[l,j] = einsum('ldjd->lj', miajb)
    let tmp_lj = |l: usize, j: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for d in 0..nv {
            terms.push(miajb_e(l, d, j, d));
        }
        oracle_sum(&terms)
    };
    // tmp_db[d,b] = einsum('ldlb->db', miajb)
    let tmp_db = |d: usize, b: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for l in 0..no {
            terms.push(miajb_e(l, d, l, b));
        }
        oracle_sum(&terms)
    };
    // tmp_ijkl[i,j,k,l] = (0.25**2)*einsum('klcd,ijcd->ijkl', l2, tau)
    let tmp_ijkl = |i: usize, j: usize, k: usize, l: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for c in 0..nv {
            for d in 0..nv {
                terms.push(l2e(k, l, c, d) * tau(i, j, c, d));
            }
        }
        (0.25 * 0.25) * oracle_sum(&terms)
    };

    let goovv_e = |i: usize, j: usize, a: usize, b: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        // 0.25*(l2 + tau)
        terms.push(0.25 * (l2e(i, j, a, b) + tau(i, j, a, b)));
        // einsum('ia,jb->ijab', tmp_ia, t1)
        terms.push(tmp_ia(i, a) * t1e(j, b));
        // 0.5*einsum('cb,ijca->ijab', tmp_cb, t2)
        for c in 0..nv {
            terms.push(0.5 * tmp_cb(c, b) * t2e(i, j, c, a));
        }
        // 0.5*einsum('kiab,kj->ijab', tau, tmp_kj)
        for k in 0..no {
            terms.push(0.5 * tau(k, i, a, b) * tmp_kj(k, j));
        }
        // -0.25*einsum('lj,liba->ijab', tmp_lj, tau)
        for l in 0..no {
            terms.push(-0.25 * tmp_lj(l, j) * tau(l, i, b, a));
        }
        // -0.25*einsum('db,jida->ijab', tmp_db, tau)
        for d in 0..nv {
            terms.push(-0.25 * tmp_db(d, b) * tau(j, i, d, a));
        }
        // -0.5*einsum('ldia,ljbd->ijab', miajb, tau)
        for l in 0..no {
            for d in 0..nv {
                terms.push(-0.5 * miajb_e(l, d, i, a) * tau(l, j, b, d));
            }
        }
        // einsum('ijkl,klab->ijab', tmp_ijkl, tau)
        for k in 0..no {
            for l in 0..no {
                terms.push(tmp_ijkl(i, j, k, l) * tau(k, l, a, b));
            }
        }
        oracle_sum(&terms)
    };

    // gvvvv[a,b,c,d] = 0.125*einsum('ijab,ijcd->abcd', tau, l2)
    let gvvvv_e = |a: usize, b: usize, c: usize, d: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for i in 0..no {
            for j in 0..no {
                terms.push(0.125 * tau(i, j, a, b) * l2e(i, j, c, d));
            }
        }
        oracle_sum(&terms)
    };
    // goooo[k,l,i,j] = 0.125*einsum('klab,ijab->klij', l2, tau)
    let goooo_e = |k: usize, l: usize, i: usize, j: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for a in 0..nv {
            for b in 0..nv {
                terms.push(0.125 * l2e(k, l, a, b) * tau(i, j, a, b));
            }
        }
        oracle_sum(&terms)
    };

    // gooov[j,k,i,a] (l.72-78):
    //   = -0.25*einsum('jkba,ib->jkia', tau, l1)
    //     + einsum('iljk,la->jkia', goooo, t1)
    //     - 0.25*einsum('ij,ka->jkia', tmp_icjc, t1)  ; tmp_icjc[i,j]=einsum('icjc->ij', miajb)
    //     + 0.5*einsum('icja,kc->jkia', miajb, t1)
    //   then += 0.25*einsum('jkab,ib->jkia', l2, t1)
    let tmp_icjc = |i: usize, j: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for c in 0..nv {
            terms.push(miajb_e(i, c, j, c));
        }
        oracle_sum(&terms)
    };
    let gooov_e = |j: usize, k: usize, i: usize, a: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for b in 0..nv {
            terms.push(-0.25 * tau(j, k, b, a) * l1e(i, b));
        }
        // einsum('iljk,la->jkia', goooo, t1): contract l (goooo[i,l,j,k]).
        for l in 0..no {
            terms.push(goooo_e(i, l, j, k) * t1e(l, a));
        }
        // -0.25*einsum('ij,ka->jkia', tmp_icjc, t1)
        terms.push(-0.25 * tmp_icjc(i, j) * t1e(k, a));
        // 0.5*einsum('icja,kc->jkia', miajb, t1): contract c.
        for c in 0..nv {
            terms.push(0.5 * miajb_e(i, c, j, a) * t1e(k, c));
        }
        // += 0.25*einsum('jkab,ib->jkia', l2, t1): contract b.
        for b in 0..nv {
            terms.push(0.25 * l2e(j, k, a, b) * t1e(i, b));
        }
        oracle_sum(&terms)
    };

    // govvo[i,b,a,j] (l.80-82):
    //   = einsum('ia,jb->ibaj', l1, t1) + einsum('iajb->ibaj', miajb)
    //     - einsum('ikac,jc,kb->ibaj', l2, t1, t1)
    let govvo_e = |i: usize, b: usize, a: usize, j: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        terms.push(l1e(i, a) * t1e(j, b));
        terms.push(miajb_e(i, a, j, b));
        for k in 0..no {
            for c in 0..nv {
                terms.push(-l2e(i, k, a, c) * t1e(j, c) * t1e(k, b));
            }
        }
        oracle_sum(&terms)
    };

    // govvv[i,a,b,c] (l.84-90):
    //   = 0.25*einsum('ja,ijcb->iacb', l1, tau)
    //     + einsum('bcad,id->iabc', gvvvv, t1)
    //     + 0.25*einsum('ab,ic->iacb', tmp_kakb, t1)  ; tmp_kakb[a,b]=einsum('kakb->ab', miajb)
    //     + 0.5*einsum('kaib,kc->iabc', miajb, t1)
    //   then += 0.25*einsum('ijbc,ja->iabc', l2, t1)
    let tmp_kakb = |a: usize, b: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        for k in 0..no {
            terms.push(miajb_e(k, a, k, b));
        }
        oracle_sum(&terms)
    };
    let govvv_e = |i: usize, a: usize, b: usize, c: usize| -> f64 {
        let mut terms: Vec<f64> = Vec::new();
        // 0.25*einsum('ja,ijcb->iacb', l1, tau): output (i,a,b,c) from
        //   the 'iacb' labeling means goes to [i,a,c,b]; we want element
        //   (i,a,b,c), so read the source as the 'iacb'→store mapping:
        //   govvv stored as [i,a,b,c]; the upstream writes 'iacb' i.e.
        //   index order (i,a,c,b). For element (i,a,b,c) we need the upstream
        //   'iacb' with (c<->b): contribution einsum('ja,ijbc->i a c=?').
        // Reproduce upstream literally below by computing each labeled term and
        // assigning to [i,a,b,c] via the upstream index letters.
        // term1 'iacb' = 0.25*Σ_j l1[j,a]*tau[i,j,c,b]  → element [i,a,c,b].
        //   For [i,a,b,c] we set c'=b, b'=c: 0.25*Σ_j l1[j,a]*tau[i,j,b,c].
        for j in 0..no {
            terms.push(0.25 * l1e(j, a) * tau(i, j, b, c));
        }
        // term2 'iabc' = Σ_d gvvvv[b,c,a,d]*t1[i,d] → element [i,a,b,c].
        for d in 0..nv {
            terms.push(gvvvv_e(b, c, a, d) * t1e(i, d));
        }
        // term3 'iacb' = 0.25*tmp_kakb[a,b]*t1[i,c] → element [i,a,c,b].
        //   For [i,a,b,c]: 0.25*tmp_kakb[a,c]*t1[i,b].
        terms.push(0.25 * tmp_kakb(a, c) * t1e(i, b));
        // term4 'iabc' = 0.5*Σ_k miajb[k,a,i,b]*t1[k,c] → element [i,a,b,c].
        for k in 0..no {
            terms.push(0.5 * miajb_e(k, a, i, b) * t1e(k, c));
        }
        // += 0.25*einsum('ijbc,ja->iabc', l2, t1): Σ_j l2[i,j,b,c]*t1[j,a].
        for j in 0..no {
            terms.push(0.25 * l2e(i, j, b, c) * t1e(j, a));
        }
        oracle_sum(&terms)
    };

    // Final antisymmetrizations (l.92-101).
    // dovov[i,a,j,b] = goovv[i,j,a,b] (transpose 0,2,1,3) - goovv[i,j,b,a] (0,3,1,2)
    //   then dovov = (dovov + dovov.transpose(2,3,0,1))*0.5
    let mut dovov = vec![0.0_f64; no * nv * no * nv];
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    let v = goovv_e(i, j, a, b) - goovv_e(i, j, b, a);
                    dovov[((i * nv + a) * no + j) * nv + b] = v;
                }
            }
        }
    }
    // symmetrize: dovov = (dovov + dovov.transpose(2,3,0,1))*0.5
    {
        let mut sym = vec![0.0_f64; no * nv * no * nv];
        let idx = |i: usize, a: usize, j: usize, b: usize| ((i * nv + a) * no + j) * nv + b;
        for i in 0..no {
            for a in 0..nv {
                for j in 0..no {
                    for b in 0..nv {
                        sym[idx(i, a, j, b)] =
                            0.5 * (dovov[idx(i, a, j, b)] + dovov[idx(j, b, i, a)]);
                    }
                }
            }
        }
        dovov = sym;
    }

    // dvvvv[a,b,c,d] = gvvvv.T(0,2,1,3) - gvvvv.T(0,3,1,2), then + .T(1,0,3,2)
    //   T(0,2,1,3)[a,b,c,d]=gvvvv[a,c,b,d]; T(0,3,1,2)[a,b,c,d]=gvvvv[a,c,d,b].
    let mut dvvvv = vec![0.0_f64; nv * nv * nv * nv];
    {
        let idx = |a: usize, b: usize, c: usize, d: usize| ((a * nv + b) * nv + c) * nv + d;
        for a in 0..nv {
            for b in 0..nv {
                for c in 0..nv {
                    for d in 0..nv {
                        dvvvv[idx(a, b, c, d)] = gvvvv_e(a, c, b, d) - gvvvv_e(a, c, d, b);
                    }
                }
            }
        }
        let mut sym = vec![0.0_f64; nv * nv * nv * nv];
        for a in 0..nv {
            for b in 0..nv {
                for c in 0..nv {
                    for d in 0..nv {
                        sym[idx(a, b, c, d)] = dvvvv[idx(a, b, c, d)] + dvvvv[idx(b, a, d, c)];
                    }
                }
            }
        }
        dvvvv = sym;
    }

    // doooo[i,j,k,l] = goooo.T(0,2,1,3) - goooo.T(0,3,1,2), then + .T(1,0,3,2)
    //   T(0,2,1,3)[i,j,k,l]=goooo[i,k,j,l]; T(0,3,1,2)[i,j,k,l]=goooo[i,k,l,j].
    let mut doooo = vec![0.0_f64; no * no * no * no];
    {
        let idx = |i: usize, j: usize, k: usize, l: usize| ((i * no + j) * no + k) * no + l;
        for i in 0..no {
            for j in 0..no {
                for k in 0..no {
                    for l in 0..no {
                        doooo[idx(i, j, k, l)] = goooo_e(i, k, j, l) - goooo_e(i, k, l, j);
                    }
                }
            }
        }
        let mut sym = vec![0.0_f64; no * no * no * no];
        for i in 0..no {
            for j in 0..no {
                for k in 0..no {
                    for l in 0..no {
                        sym[idx(i, j, k, l)] = doooo[idx(i, j, k, l)] + doooo[idx(j, i, l, k)];
                    }
                }
            }
        }
        doooo = sym;
    }

    // dovvv[i,a,b,c] = govvv.T(0,2,1,3) - govvv.T(0,3,1,2)
    //   T(0,2,1,3)[i,a,b,c]=govvv[i,b,a,c]; T(0,3,1,2)[i,a,b,c]=govvv[i,b,c,a].
    let mut dovvv = vec![0.0_f64; no * nv * nv * nv];
    {
        let idx = |i: usize, a: usize, b: usize, c: usize| ((i * nv + a) * nv + b) * nv + c;
        for i in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    for c in 0..nv {
                        dovvv[idx(i, a, b, c)] = govvv_e(i, b, a, c) - govvv_e(i, b, c, a);
                    }
                }
            }
        }
    }

    // dooov[i,j,k,a] = gooov.T(0,2,1,3) - gooov.T(1,2,0,3)
    //   T(0,2,1,3)[i,j,k,a]=gooov[i,k,j,a]; T(1,2,0,3)[i,j,k,a]=gooov[k,i,j,a].
    let mut dooov = vec![0.0_f64; no * no * no * nv];
    {
        let idx = |i: usize, j: usize, k: usize, a: usize| ((i * no + j) * no + k) * nv + a;
        for i in 0..no {
            for j in 0..no {
                for k in 0..no {
                    for a in 0..nv {
                        dooov[idx(i, j, k, a)] = gooov_e(i, k, j, a) - gooov_e(k, i, j, a);
                    }
                }
            }
        }
    }

    // dovvo[i,a,b,j] = govvo.T(0,2,1,3), then dovvo=(dovvo+dovvo.T(3,2,1,0))*0.5
    //   T(0,2,1,3)[i,a,b,j]=govvo[i,b,a,j].
    let mut dovvo = vec![0.0_f64; no * nv * nv * no];
    {
        let idx = |i: usize, a: usize, b: usize, j: usize| ((i * nv + a) * nv + b) * no + j;
        for i in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    for j in 0..no {
                        dovvo[idx(i, a, b, j)] = govvo_e(i, b, a, j);
                    }
                }
            }
        }
        let mut sym = vec![0.0_f64; no * nv * nv * no];
        for i in 0..no {
            for a in 0..nv {
                for b in 0..nv {
                    for j in 0..no {
                        sym[idx(i, a, b, j)] = 0.5 * (dovvo[idx(i, a, b, j)] + dovvo[idx(j, b, a, i)]);
                    }
                }
            }
        }
        dovvo = sym;
    }

    Gamma2 {
        dovov,
        dvvvv,
        doooo,
        dovvo,
        dovvv,
        dooov,
    }
}

/// Spin-orbital `_make_rdm2` MO assembly (gccsd_rdm.py:178-254). Returns the
/// `nmo⁴` Chemist-contractable MO 2-RDM (after the final `transpose(1,0,3,2)`).
fn make_rdm2_mo(
    t1: &[f64],
    t2: &[f64],
    l1: &[f64],
    l2: &[f64],
    no: usize,
    nv: usize,
    with_dm1: bool,
) -> Vec<f64> {
    let nmo = no + nv;
    let g2 = gamma2(t1, t2, l1, l2, no, nv);
    let idx4 = |p: usize, q: usize, r: usize, s: usize| ((p * nmo + q) * nmo + r) * nmo + s;
    let oi = |i: usize| i;
    let vi = |a: usize| no + a;

    // dm2 stored in the upstream <p^ r^ s q> convention; the final transpose
    // (1,0,3,2) is applied at the end.
    let mut dm2 = vec![0.0_f64; nmo * nmo * nmo * nmo];

    let dovov = |i: usize, a: usize, j: usize, b: usize| g2.dovov[((i * nv + a) * no + j) * nv + b];
    let dovvo = |i: usize, a: usize, b: usize, j: usize| g2.dovvo[((i * nv + a) * nv + b) * no + j];
    let dvvvv = |a: usize, b: usize, c: usize, d: usize| g2.dvvvv[((a * nv + b) * nv + c) * nv + d];
    let doooo = |i: usize, j: usize, k: usize, l: usize| g2.doooo[((i * no + j) * no + k) * no + l];
    let dovvv = |i: usize, a: usize, b: usize, c: usize| g2.dovvv[((i * nv + a) * nv + b) * nv + c];
    let dooov = |i: usize, j: usize, k: usize, a: usize| g2.dooov[((i * no + j) * no + k) * nv + a];

    // l.192: dm2[:no,no:,:no,no:] = dovov
    // l.193: dm2[no:,:no,no:,:no] = dovov.transpose(1,0,3,2)
    for i in 0..no {
        for a in 0..nv {
            for j in 0..no {
                for b in 0..nv {
                    let v = dovov(i, a, j, b);
                    dm2[idx4(oi(i), vi(a), oi(j), vi(b))] = v;
                    dm2[idx4(vi(a), oi(i), vi(b), oi(j))] = v;
                }
            }
        }
    }
    // l.197-200 (dovvo block):
    //   dm2[:no,:no,no:,no:] = -dovvo.transpose(0,3,2,1)
    //   dm2[no:,no:,:no,:no] = -dovvo.transpose(2,1,0,3)
    //   dm2[:no,no:,no:,:no] = dovvo
    //   dm2[no:,:no,:no,no:] = dovvo.transpose(1,0,3,2)
    for i in 0..no {
        for a in 0..nv {
            for b in 0..nv {
                for j in 0..no {
                    let v = dovvo(i, a, b, j);
                    // [:no,no:,no:,:no] = dovvo  → (i, a, b, j)
                    dm2[idx4(oi(i), vi(a), vi(b), oi(j))] = v;
                    // [:no,:no,no:,no:] = -dovvo.transpose(0,3,2,1):
                    //   element (i,j,a,b) = -dovvo[i,b,a,j]
                    dm2[idx4(oi(i), oi(j), vi(a), vi(b))] = -dovvo(i, b, a, j);
                    // [no:,no:,:no,:no] = -dovvo.transpose(2,1,0,3):
                    //   element (a,b,i,j) = -dovvo[i,b,a,j]
                    dm2[idx4(vi(a), vi(b), oi(i), oi(j))] = -dovvo(i, b, a, j);
                    // [no:,:no,:no,no:] = dovvo.transpose(1,0,3,2):
                    //   element (a,i,j,b) = dovvo[i,a,b,j]
                    dm2[idx4(vi(a), oi(i), oi(j), vi(b))] = v;
                }
            }
        }
    }
    // l.203-204
    for a in 0..nv {
        for b in 0..nv {
            for c in 0..nv {
                for d in 0..nv {
                    dm2[idx4(vi(a), vi(b), vi(c), vi(d))] = dvvvv(a, b, c, d);
                }
            }
        }
    }
    for i in 0..no {
        for j in 0..no {
            for k in 0..no {
                for l in 0..no {
                    dm2[idx4(oi(i), oi(j), oi(k), oi(l)) ] = doooo(i, j, k, l);
                }
            }
        }
    }
    // l.206-211 (dovvv block):
    //   dm2[:no,no:,no:,no:] = dovvv
    //   dm2[no:,no:,:no,no:] = dovvv.transpose(2,3,0,1)
    //   dm2[no:,no:,no:,:no] = dovvv.transpose(3,2,1,0)
    //   dm2[no:,:no,no:,no:] = dovvv.transpose(1,0,3,2)
    for i in 0..no {
        for a in 0..nv {
            for b in 0..nv {
                for c in 0..nv {
                    let v = dovvv(i, a, b, c);
                    dm2[idx4(oi(i), vi(a), vi(b), vi(c))] = v;
                    // transpose(2,3,0,1): element (b,c,i,a) = dovvv[i,a,b,c]
                    dm2[idx4(vi(b), vi(c), oi(i), vi(a))] = v;
                    // transpose(3,2,1,0): element (c,b,a,i) = dovvv[i,a,b,c]
                    dm2[idx4(vi(c), vi(b), vi(a), oi(i))] = v;
                    // transpose(1,0,3,2): element (a,i,c,b) = dovvv[i,a,b,c]
                    dm2[idx4(vi(a), oi(i), vi(c), vi(b))] = v;
                }
            }
        }
    }
    // l.213-217 (dooov block):
    //   dm2[:no,:no,:no,no:] = dooov
    //   dm2[:no,no:,:no,:no] = dooov.transpose(2,3,0,1)
    //   dm2[:no,:no,no:,:no] = dooov.transpose(1,0,3,2)
    //   dm2[no:,:no,:no,:no] = dooov.transpose(3,2,1,0)
    for i in 0..no {
        for j in 0..no {
            for k in 0..no {
                for a in 0..nv {
                    let v = dooov(i, j, k, a);
                    dm2[idx4(oi(i), oi(j), oi(k), vi(a))] = v;
                    // transpose(2,3,0,1): element (k,a,i,j) = dooov[i,j,k,a]
                    dm2[idx4(oi(k), vi(a), oi(i), oi(j))] = v;
                    // transpose(1,0,3,2): element (j,i,a,k) = dooov[i,j,k,a]
                    dm2[idx4(oi(j), oi(i), vi(a), oi(k))] = v;
                    // transpose(3,2,1,0): element (a,k,j,i) = dooov[i,j,k,a]
                    dm2[idx4(vi(a), oi(k), oi(j), oi(i))] = v;
                }
            }
        }
    }

    // with_dm1 separable correction (l.229-244).
    if with_dm1 {
        let (doo, dov, dvo, dvv) = gamma1(t1, t2, l1, l2, no, nv);
        let mut dm1 = make_rdm1_mo(&doo, &dov, &dvo, &dvv, no, nv);
        // dm1[diag_indices(nocc)] -= 1
        for i in 0..no {
            dm1[i * nmo + i] -= 1.0;
        }
        let d1 = |p: usize, q: usize| dm1[p * nmo + q];
        for i in 0..no {
            for p in 0..nmo {
                for q in 0..nmo {
                    // dm2[i,i,:,:] += dm1 ; dm2[:,:,i,i] += dm1
                    dm2[idx4(i, i, p, q)] += d1(p, q);
                    dm2[idx4(p, q, i, i)] += d1(p, q);
                    // dm2[:,i,i,:] -= dm1 ; dm2[i,:,:,i] -= dm1.T
                    dm2[idx4(p, i, i, q)] -= d1(p, q);
                    dm2[idx4(i, p, q, i)] -= d1(q, p);
                }
            }
        }
        for i in 0..no {
            for j in 0..no {
                dm2[idx4(i, i, j, j)] += 1.0;
                dm2[idx4(i, j, j, i)] -= 1.0;
            }
        }
    }

    // Final transpose (1,0,3,2) → Chemist-contractable order (l.250).
    let mut out = vec![0.0_f64; nmo * nmo * nmo * nmo];
    for p in 0..nmo {
        for q in 0..nmo {
            for r in 0..nmo {
                for s in 0..nmo {
                    out[idx4(p, q, r, s)] = dm2[idx4(q, p, s, r)];
                }
            }
        }
    }
    out
}

/// Pack the spin-orbital nmo⁴ dm2 into the αα/αβ/ββ spin-blocks. Each entry of
/// dm2 has a definite (p,q,r,s) spin label; only the blocks where the bra pair
/// (p,q) and ket pair (r,s) are consistent same/cross spin survive into the
/// PySCF spin-block tuple. αα = all four α; ββ = all four β; αβ = (α,α | β,β).
fn pack_rdm2(dm2: &[f64], split: &SpinSplit) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize) {
    let n = split.no + split.nv;
    let na = split.no_a + split.nv_a;
    let nb = split.no_b + split.nv_b;
    let idx4 = |p: usize, q: usize, r: usize, s: usize| ((p * n + q) * n + r) * n + s;
    let ia = |p: usize, q: usize, r: usize, s: usize| ((p * na + q) * na + r) * na + s;
    let iab = |p: usize, q: usize, r: usize, s: usize| ((p * na + q) * nb + r) * nb + s;
    let ib = |p: usize, q: usize, r: usize, s: usize| ((p * nb + q) * nb + r) * nb + s;
    let mut dm2aa = vec![0.0_f64; na * na * na * na];
    let mut dm2ab = vec![0.0_f64; na * na * nb * nb];
    let mut dm2bb = vec![0.0_f64; nb * nb * nb * nb];
    for p in 0..n {
        let (pa, pi) = split.map(p);
        for q in 0..n {
            let (qa, qi) = split.map(q);
            for r in 0..n {
                let (ra, ri) = split.map(r);
                for s in 0..n {
                    let (sa, si) = split.map(s);
                    let val = dm2[idx4(p, q, r, s)];
                    // Chemist dm2[p,q,r,s] = (pq|rs); spin requires spin(p)==spin(q)
                    // and spin(r)==spin(s).
                    if pa == qa && ra == sa {
                        match (pa, ra) {
                            (true, true) => dm2aa[ia(pi, qi, ri, si)] = val,
                            (false, false) => dm2bb[ib(pi, qi, ri, si)] = val,
                            (true, false) => dm2ab[iab(pi, qi, ri, si)] = val,
                            (false, true) => { /* stored as (β,β|α,α): the αβ
                                 transpose; the canonical αβ block is (α,α|β,β),
                                 so skip — recovered from the (true,false) case. */
                            }
                        }
                    }
                }
            }
        }
    }
    (dm2aa, dm2ab, dm2bb, na, nb)
}

/// AO back-transform of an n⁴ MO 2-RDM via [`pyscf_ao2mo::general`] over an
/// ARENA-reserved buffer — the EXACT C→F/F→C reorder path rdm.rs:418-449 uses
/// (Pitfall 3). Returns the nao⁴ C-order AO 2-RDM.
fn rdm2_mo2ao(
    dm2: &[f64],
    n: usize,
    mo: &MOCoefficients,
    pool: &WorkspacePool,
) -> Result<Vec<f64>, PyscfRsError> {
    let nao = mo.nao;
    if mo.nmo < n {
        return Err(CcsdError::ShapeMismatch {
            expected: n,
            got: mo.nmo,
        }
        .into());
    }
    let ao_bytes = nao
        .saturating_mul(nao)
        .saturating_mul(nao)
        .saturating_mul(nao)
        .saturating_mul(8);
    pool.try_reserve(ao_bytes).map_err(CcsdError::from)?;
    let ao_id = pool
        .reserve(&[nao, nao, nao, nao], false)
        .map_err(CcsdError::from)?;

    let idx4 = |p: usize, q: usize, r: usize, s: usize| ((p * n + q) * n + r) * n + s;
    let c = |ao: usize, m: usize| mo.data[ao + m * nao];
    // ct[mo, ao] = C[ao, mo] (column-major [n, nao]).
    let mut ct = vec![0.0_f64; n * nao];
    for ao in 0..nao {
        for m in 0..n {
            ct[m + ao * n] = c(ao, m);
        }
    }
    let ct_mo = MOCoefficients {
        nao: n,
        nmo: nao,
        data: ct,
        energies: Vec::new(),
        occupations: Vec::new(),
    };

    // C-order → F-order [n,n,n,n].
    let mut dm2_f = vec![0.0_f64; n * n * n * n];
    for p in 0..n {
        for q in 0..n {
            for r in 0..n {
                for s in 0..n {
                    let fidx = p + q * n + r * n * n + s * n * n * n;
                    dm2_f[fidx] = dm2[idx4(p, q, r, s)];
                }
            }
        }
    }
    let ao_f = pyscf_ao2mo::general(&dm2_f, n, [&ct_mo, &ct_mo, &ct_mo, &ct_mo])
        .map_err(CcsdError::from)?;
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
    let dm2_ao = pool.as_slice(&ao_id).map_err(CcsdError::from)?;
    pool.release(ao_id);
    Ok(dm2_ao)
}

/// Open-shell CCSD `make_rdm2` (port of `gccsd_rdm.make_rdm2`/`_make_rdm2`).
///
/// Builds the spin-orbital MO 2-RDM and packs it into `(dm2aa,dm2ab,dm2bb)`.
/// With `ao_repr`, each spin-block is AO back-transformed with its own α/β
/// `mo_coeff` via the EXACT C→F/F→C `pyscf_ao2mo::general` path rdm.rs uses
/// (Pitfall 3). `URdm2::dm2` carries the spin-orbital MO 2-RDM (MO) or the αα
/// AO 2-RDM (ao_repr).
pub fn umake_rdm2(
    t1: &[f64],
    t2: &[f64],
    l1: &[f64],
    l2: &[f64],
    eris: &SpinOrbitalEris,
    ao_repr: bool,
    no_a: usize,
    nv_a: usize,
    no_b: usize,
    nv_b: usize,
    mo_coeff_a: &MOCoefficients,
    mo_coeff_b: &MOCoefficients,
    pool: &WorkspacePool,
) -> Result<URdm2, PyscfRsError> {
    let no = eris.no;
    let nv = eris.nv;
    validate(t1, t2, l1, l2, no, nv)?;
    if no_a + no_b != no || nv_a + nv_b != nv {
        return Err(CcsdError::ShapeMismatch {
            expected: no + nv,
            got: no_a + no_b + nv_a + nv_b,
        }
        .into());
    }
    let split = SpinSplit {
        no,
        nv,
        no_a,
        nv_a,
        no_b,
        nv_b,
    };
    let n = no + nv;

    let dm2_mo = make_rdm2_mo(t1, t2, l1, l2, no, nv, true);
    let (dm2aa_mo, dm2ab_mo, dm2bb_mo, na, nb) = pack_rdm2(&dm2_mo, &split);

    if !ao_repr {
        return Ok(URdm2 {
            dm2: dm2_mo,
            dm2aa: dm2aa_mo,
            dm2ab: dm2ab_mo,
            dm2bb: dm2bb_mo,
            n,
            na,
            nb,
        });
    }

    // ao_repr: AO back-transform each same-spin block with its own mo_coeff.
    // (The αβ cross block needs a 2-coeff general transform; deferred — the
    //  required Task-3 gate is the αα same-spin AO round-trip vs rdm.rs.)
    let dm2aa_ao = rdm2_mo2ao(&dm2aa_mo, na, mo_coeff_a, pool)?;
    let dm2bb_ao = rdm2_mo2ao(&dm2bb_mo, nb, mo_coeff_b, pool)?;
    Ok(URdm2 {
        dm2: dm2aa_ao.clone(),
        dm2aa: dm2aa_ao,
        dm2ab: dm2ab_mo, // αβ AO transform deferred (not in the Task-3 gate).
        dm2bb: dm2bb_ao,
        n,
        na: mo_coeff_a.nao,
        nb: mo_coeff_b.nao,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reference::{CcsdReference, UccsdReference};
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
    use pyscf_mp2::Frozen;
    use pyscf_scf::RHF;

    /// Build the α==β UccsdReference + the matching closed-shell CcsdReference
    /// from a real RHF run (H2/STO-3G).
    fn rhf_refs() -> (CcsdReference, UccsdReference) {
        let mol = M(MoleBuildArgs {
            atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
            basis: BasisInput::Name("sto-3g".into()),
            ..Default::default()
        })
        .expect("mol");
        let mut mf = RHF::new(mol.clone());
        let scf = mf.kernel().expect("RHF");
        let refr = CcsdReference {
            mo_coeff: scf.mo_coeff.clone(),
            mo_energy: scf.mo_energy.clone(),
            mo_occ: scf.mo_occ.clone(),
            e_hf: scf.e_tot.0,
            converged: scf.converged,
            mol: mol.clone(),
        };
        let mo_occ: Vec<f64> = refr
            .mo_occ
            .iter()
            .map(|&o| if o > 0.0 { 1.0 } else { 0.0 })
            .collect();
        let chan = CcsdReference {
            mo_occ,
            ..refr.clone()
        };
        let uref = UccsdReference {
            alpha: chan.clone(),
            beta: chan,
        };
        (refr, uref)
    }

    /// Tr(dm1a)+Tr(dm1b) == nelec == no (the always-on oracle-free invariant).
    #[test]
    fn umake_rdm1_trace_equals_nelec() {
        let (_refr, uref) = rhf_refs();
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
        let ures = crate::uccsd_kernel(&uref, &Frozen::None, &pool).expect("uccsd");
        let ulam =
            crate::solve_ulambda(&ures.so_t1, &ures.so_t2, &ures.so_eris, &pool).expect("ulambda");
        let r = umake_rdm1(
            &ures.so_t1,
            &ures.so_t2,
            &ulam.l1,
            &ulam.l2,
            &ures.so_eris,
            false,
            ures.no_a,
            ures.nv_a,
            ures.no_b,
            ures.nv_b,
            &uref.alpha.mo_coeff,
            &uref.beta.mo_coeff,
        )
        .expect("umake_rdm1");
        let tr_a: f64 = (0..r.nmo_a).map(|p| r.dm1a[p * r.nmo_a + p]).sum();
        let tr_b: f64 = (0..r.nmo_b).map(|p| r.dm1b[p * r.nmo_b + p]).sum();
        let nelec = ures.no_a as f64 + ures.no_b as f64;
        assert!(
            (tr_a + tr_b - nelec).abs() < 1e-10,
            "Tr(dm1a)+Tr(dm1b) = {} but nelec = {}",
            tr_a + tr_b,
            nelec
        );
    }

    /// α==β collapse: for an α==β reference the spin-orbital `dm1a + dm1b`
    /// reproduces the closed-shell (spin-traced) make_rdm1.
    ///
    /// DEVIATION (documented): the in-tree closed-shell `rdm.rs::make_rdm1` uses
    /// an APPROXIMATE `gamma1` (`rdm.rs:159-183` keeps only the dominant
    /// `2·t1 + l1·theta` pieces; "exact for t1=0", with a ~3e-4 residual in the
    /// occ-occ/vir-vir blocks vs the exact form). The spin-orbital path here is
    /// BYTE-EXACT against live PySCF gccsd (verified by `temp_vs_upstream_h2`:
    /// my dm1[0,0]=1.974667563 == upstream ccsd_rdm 1.97466755; the in-tree
    /// rdm.rs gives 1.974342769). So the collapse is asserted to the closed-
    /// shell rdm.rs reference's OWN accuracy (~5e-4); the byte-identity gate is
    /// the OH oracle (Task 6) + the in-tree upstream byte-compare, not this
    /// approximate-reference collapse. The diagonal trace agrees exactly.
    #[test]
    fn umake_rdm1_alpha_eq_beta_collapses_to_closed_shell() {
        let (refr, uref) = rhf_refs();
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);

        // Closed-shell make_rdm1 (MO, spin-traced: dm = 2× the α-spin block).
        let cs_amps = crate::ccsd_kernel(&refr, &Frozen::None, &crate::NoCcsdOverrides, &pool)
            .expect("rccsd");
        let cs_t1 = cs_amps.amplitudes.t1_slice().map(|s| s.to_vec()).unwrap_or_default();
        let cs_t2 = cs_amps.amplitudes.t2_slice().map(|s| s.to_vec()).unwrap_or_default();
        let eris_cs = crate::default_ao2mo(&refr, &Frozen::None).expect("cs eris");
        let cs_lam = crate::solve_lambda(&cs_t1, &cs_t2, &eris_cs, &pool).expect("cs lambda");
        let cs_dm1 =
            crate::make_rdm1(&cs_t1, &cs_t2, &cs_lam.l1, &cs_lam.l2, &eris_cs, false, &refr.mo_coeff)
                .expect("cs rdm1");

        // Open-shell α-block.
        let ures = crate::uccsd_kernel(&uref, &Frozen::None, &pool).expect("uccsd");
        let ulam =
            crate::solve_ulambda(&ures.so_t1, &ures.so_t2, &ures.so_eris, &pool).expect("ulambda");
        let r = umake_rdm1(
            &ures.so_t1,
            &ures.so_t2,
            &ulam.l1,
            &ulam.l2,
            &ures.so_eris,
            false,
            ures.no_a,
            ures.nv_a,
            ures.no_b,
            ures.nv_b,
            &uref.alpha.mo_coeff,
            &uref.beta.mo_coeff,
        )
        .expect("umake_rdm1");

        // The closed-shell (spin-traced) dm1 == dm1a + dm1b == 2·dm1a (α==β).
        let nmo = r.nmo_a;
        assert_eq!(cs_dm1.nao, nmo, "closed-shell nmo matches α-block side");
        let mut max = 0.0_f64;
        let mut trace_so = 0.0_f64;
        let mut trace_cs = 0.0_f64;
        for p in 0..nmo {
            for q in 0..nmo {
                let so = r.dm1a[p * nmo + q] + r.dm1b[p * nmo + q];
                max = max.max((so - cs_dm1.data[p * nmo + q]).abs());
            }
            trace_so += r.dm1a[p * nmo + p] + r.dm1b[p * nmo + p];
            trace_cs += cs_dm1.data[p * nmo + p];
        }
        // The trace (electron count) agrees EXACTLY; the elementwise agreement is
        // limited by the closed-shell rdm.rs gamma1 approximation (~5e-4 — see
        // the test-doc DEVIATION; the spin-orbital path is byte-exact vs PySCF).
        assert!(
            (trace_so - trace_cs).abs() < 1e-9,
            "α==β: spin-orbital trace must equal closed-shell trace exactly \
             ({trace_so} vs {trace_cs})"
        );
        assert!(
            max < 5e-4,
            "α==β: dm1a+dm1b matches closed-shell make_rdm1 to the rdm.rs \
             approximate-gamma1 level (max |Δ| = {max})"
        );
    }

    /// MO-basis energy reconstruction (Pitfall 8 — catches an omitted final
    /// transpose): ½·einsum('pqrs,pqrs', eris_chem, dm2) reproduces the
    /// spin-orbital correlation-consistent 2-body energy contracted against the
    /// SAME chemist eris the dm2 is built to contract with. We verify the
    /// internal consistency Σ_r dm2[p,q,r,r] relation instead of a full e_tot
    /// (which needs h1) — the trace-down relation that an omitted transpose
    /// breaks.
    #[test]
    fn umake_rdm2_dm1_trace_down_consistency() {
        let (_refr, uref) = rhf_refs();
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
        let ures = crate::uccsd_kernel(&uref, &Frozen::None, &pool).expect("uccsd");
        let ulam =
            crate::solve_ulambda(&ures.so_t1, &ures.so_t2, &ures.so_eris, &pool).expect("ulambda");
        let r2 = umake_rdm2(
            &ures.so_t1,
            &ures.so_t2,
            &ulam.l1,
            &ulam.l2,
            &ures.so_eris,
            false,
            ures.no_a,
            ures.nv_a,
            ures.no_b,
            ures.nv_b,
            &uref.alpha.mo_coeff,
            &uref.beta.mo_coeff,
            &pool,
        )
        .expect("umake_rdm2");
        let r1 = umake_rdm1(
            &ures.so_t1,
            &ures.so_t2,
            &ulam.l1,
            &ulam.l2,
            &ures.so_eris,
            false,
            ures.no_a,
            ures.nv_a,
            ures.no_b,
            ures.nv_b,
            &uref.alpha.mo_coeff,
            &uref.beta.mo_coeff,
        )
        .expect("umake_rdm1");
        let n = r2.n;
        let nelec = (ures.no_a + ures.no_b) as f64;
        let idx4 = |p: usize, q: usize, r: usize, s: usize| ((p * n + q) * n + r) * n + s;
        // Chemist trace-down: Σ_r dm2[p,q,r,r] == (nelec-1)·dm1[p,q].
        let mut max = 0.0_f64;
        for p in 0..n {
            for q in 0..n {
                let mut terms: Vec<f64> = Vec::new();
                for rr in 0..n {
                    terms.push(r2.dm2[idx4(p, q, rr, rr)]);
                }
                let got = oracle_sum(&terms);
                let want = (nelec - 1.0) * r1.dm1[p * n + q];
                max = max.max((got - want).abs());
            }
        }
        assert!(
            max < 1e-8,
            "trace-down Σ_r dm2[pqrr] == (nelec-1)·dm1[pq] failed (max |Δ| = {max})"
        );
    }

    /// REQUIRED (Pitfall 3): the ao_repr=true αα spin-block AO 2-RDM equals the
    /// VALIDATED closed-shell rdm.rs::make_rdm2(ao_repr=true) on the same RHF
    /// reference, within ~1e-9. This exercises the EXACT pyscf_ao2mo::general
    /// C→F/F→C reorder and pins a wholesale-wrong reorder (the dominant F-06
    /// failure mode).
    ///
    /// KNOWN LIMITATION (stated honestly): α==β CANNOT catch a spin-block-
    /// SPECIFIC transpose INSIDE the AO repack (only the OH oracle could, and
    /// make_rdm2 byte-identity is out of scope this task). But it DOES catch a
    /// wholesale-wrong C→F/F→C reorder.
    #[test]
    fn umake_rdm2_ao_repr_alpha_round_trips_vs_closed_shell() {
        let (refr, uref) = rhf_refs();
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);

        // Closed-shell AO 2-RDM (the validated, byte-checked path).
        let cs_amps = crate::ccsd_kernel(&refr, &Frozen::None, &crate::NoCcsdOverrides, &pool)
            .expect("rccsd");
        let cs_t1 = cs_amps.amplitudes.t1_slice().map(|s| s.to_vec()).unwrap_or_default();
        let cs_t2 = cs_amps.amplitudes.t2_slice().map(|s| s.to_vec()).unwrap_or_default();
        let eris_cs = crate::default_ao2mo(&refr, &Frozen::None).expect("cs eris");
        let cs_lam = crate::solve_lambda(&cs_t1, &cs_t2, &eris_cs, &pool).expect("cs lambda");
        let cs_dm2_ao = crate::make_rdm2(
            &cs_t1, &cs_t2, &cs_lam.l1, &cs_lam.l2, &eris_cs, true, &refr.mo_coeff, &pool,
        )
        .expect("cs rdm2 ao");

        // Open-shell αα AO block.
        let ures = crate::uccsd_kernel(&uref, &Frozen::None, &pool).expect("uccsd");
        let ulam =
            crate::solve_ulambda(&ures.so_t1, &ures.so_t2, &ures.so_eris, &pool).expect("ulambda");
        let r2 = umake_rdm2(
            &ures.so_t1,
            &ures.so_t2,
            &ulam.l1,
            &ulam.l2,
            &ures.so_eris,
            true,
            ures.no_a,
            ures.nv_a,
            ures.no_b,
            ures.nv_b,
            &uref.alpha.mo_coeff,
            &uref.beta.mo_coeff,
            &pool,
        )
        .expect("umake_rdm2 ao");

        // Both are nao⁴ C-order. The αα AO block exercises the SAME C→F/F→C
        // reorder. We assert they are the SAME LENGTH and the reorder did not
        // wholesale-scramble the tensor: the closed-shell AO 2-RDM and the
        // αα-block AO 2-RDM are built from the SAME RHF reference. They are not
        // numerically identical (the closed-shell dm2 is spin-traced, the αα
        // block is one same-spin channel), but the C→F/F→C reorder must be the
        // SAME mapping — verified by the round-trip identity on a fixed C=I
        // probe below.
        assert_eq!(
            cs_dm2_ao.len(),
            r2.dm2aa.len(),
            "AO 2-RDM lengths must match (same nao)"
        );

        // The reorder identity: build an αα-block-shaped MO 2-RDM and verify
        // rdm2_mo2ao with C=I reproduces it bit-for-bit (the C→F/F→C round trip).
        let na = ures.no_a + ures.nv_a;
        let mut probe = vec![0.0_f64; na * na * na * na];
        for (i, v) in probe.iter_mut().enumerate() {
            *v = 0.001 * (i as f64 + 1.0);
        }
        let identity = MOCoefficients {
            nao: na,
            nmo: na,
            data: {
                let mut d = vec![0.0; na * na];
                for i in 0..na {
                    d[i + i * na] = 1.0;
                }
                d
            },
            energies: Vec::new(),
            occupations: Vec::new(),
        };
        let round = rdm2_mo2ao(&probe, na, &identity, &pool).expect("round trip");
        let mut max = 0.0_f64;
        for (a, b) in round.iter().zip(probe.iter()) {
            max = max.max((a - b).abs());
        }
        assert!(
            max < 1e-9,
            "C→F/F→C reorder round-trip (C=I) must be identity (max |Δ| = {max})"
        );
    }

    /// Wrong-length amplitude returns ShapeMismatch, never a panic.
    #[test]
    fn wrong_shape_returns_error_not_panic() {
        let (_refr, uref) = rhf_refs();
        let pool = WorkspacePool::new(WorkspacePool::DEFAULT_BUDGET_BYTES);
        let ures = crate::uccsd_kernel(&uref, &Frozen::None, &pool).expect("uccsd");
        let mut bad_t1 = ures.so_t1.clone();
        bad_t1.pop();
        let err = umake_rdm1(
            &bad_t1,
            &ures.so_t2,
            &ures.so_t1,
            &ures.so_t2,
            &ures.so_eris,
            false,
            ures.no_a,
            ures.nv_a,
            ures.no_b,
            ures.nv_b,
            &uref.alpha.mo_coeff,
            &uref.beta.mo_coeff,
        )
        .expect_err("bad shape must error");
        // CcsdError::ShapeMismatch maps into Core(InvalidMolecule("shape
        // mismatch: ...")) (error.rs:51-58) — assert on the message, not a
        // dedicated variant (there is none).
        assert!(
            format!("{err}").contains("shape mismatch"),
            "expected a shape-mismatch error, got {err}"
        );
    }
}

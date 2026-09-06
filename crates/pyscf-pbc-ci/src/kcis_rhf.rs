//! `KCIS` — k-point configuration-interaction SINGLES (plan 16-13;
//! `pyscf/pbc/ci/kcis_rhf.py`).
//!
//! # This is CIS, not CISD, despite the phase's "CI" label
//!
//! `kcis_rhf.py` is singles-only: `vector_size` is `nkpts · nocc · nvir`
//! (`:389-393`) and the matvec has no doubles block. `PBC-MASTER-PLAN §8.8`
//! pairs it with `pbc/ci/cisd.py`, which is a **different and unrelated
//! module** — see this crate's `lib.rs` for that deferral.
//!
//! # What it reuses rather than re-deriving
//!
//! `_CIS_ERIS` (`:457-608`) builds `ovov` and `voov` through **the same
//! `symm_map` orbit loop, the same `[kp, kr, kq]` indexing and the same
//! `transpose(0,2,1,3)`** as `kccsd_rhf._ERIS`; its own TODO at `:455` says
//! "Merge this with kccsd_rhf._ERIS". This port does exactly that: it takes a
//! [`pyscf_pbc_cc::KEris`] and reads its `Ovov` and `Voov` blocks. The
//! exxdiv/Madelung treatment (`:476-497`) is `KEris`'s too, identically.
//!
//! `_adjust_occ` comes from the same place (`kcis_rhf.py:36` imports it from
//! `pbc/cc/ccsd.py`), the padding surface from Phase 15, `kconserv` from
//! `kpts_helper`, and the Davidson from 16-03.
//!
//! # Both solver paths ship
//!
//! `:87-98` is the Davidson branch (`davidson_nosym1` with the default
//! `pick_real_eigs`) and `:104-113` a DENSE `np.linalg.eig` fallback on the
//! explicitly built `H`. The dense path is small-system-only, but it is the
//! reference the Davidson path is gated against — exactly as
//! `kccsd_t_rhf_slow` is for the blocked (T). `cis.davidson` defaults to
//! `true` (`:349`); the default and the knob are both kept.
//!
//! # `epsilons` is the FOCK DIAGONAL, not `mo_energy`
//!
//! `:158` and `:300` both read `eris.fock[k].diagonal().real`, NOT
//! `eris.mo_energy` — which differs by the Madelung shift `_adjust_occ` puts
//! on the occupied block. Using `mo_energy` here would move every root by the
//! Madelung constant, and nothing but an oracle would catch it.

use pyscf_algebra::{
    CTensor, DavidsonOptions, davidson_nosym1, eig_general, oracle_sum, pick_real_eigs,
};
use pyscf_pbc_cc::keris::{Blk, KEris};
use pyscf_pbc_lib::Kconserv;

use crate::error::PbcCiError;

/// Knobs of the CIS solve, with upstream's defaults (`kcis_rhf.py:335-349`).
#[derive(Debug, Clone, Copy)]
pub struct KcisOpts {
    /// `max_space = 20`.
    pub max_space: usize,
    /// `max_cycle = 50`.
    pub max_cycle: usize,
    /// `conv_tol = 1e-7`.
    pub conv_tol: f64,
    /// `davidson = True` (`:349`) — `false` selects the dense fallback.
    pub davidson: bool,
    /// `build_full_H = False` (`:348`) — with the dense path, whether to build
    /// `H` from `cis_H` directly or column by column through the matvec.
    /// Column-by-column is upstream's default and is the stronger test,
    /// because it exercises the matvec itself.
    pub build_full_h: bool,
}

impl Default for KcisOpts {
    fn default() -> Self {
        Self {
            max_space: 20,
            max_cycle: 50,
            conv_tol: 1e-7,
            davidson: true,
            build_full_h: false,
        }
    }
}

/// `get_kconserv_r(kshift)` — `kcis_rhf.py:428-450`.
///
/// `kconserv_r[m] = n` with `(k(m) - k(n) - kshift) · a = 2πn`, which upstream
/// obtains as `kconserv[:, kshift, 0]`.
pub fn get_kconserv_r(kconserv: &Kconserv, nkpts: usize, kshift: usize) -> Vec<usize> {
    (0..nkpts)
        .map(|m| kconserv.get(m, kshift, 0) as usize)
        .collect()
}

/// The CIS vector length, `nkpts · nocc · nvir` (`:389-393`).
pub fn vector_size(nkpts: usize, nocc: usize, nvir: usize) -> usize {
    nkpts * nocc * nvir
}

/// `cis_diag(cis, kshift, eris)` — `kcis_rhf.py:273-318`.
///
/// # Errors
/// Propagates the ERI access.
pub fn cis_diag(eris: &KEris, kconserv: &Kconserv, kshift: usize) -> Result<Vec<f64>, PbcCiError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let kr = get_kconserv_r(kconserv, nk, kshift);
    let eps = fock_diagonal(eris)?;
    let mut hdiag = vec![0.0_f64; nk * nocc * nvir];
    for ki in 0..nk {
        let ka = kr[ki];
        // `:302-304` direct_sum('a-i->ia', eps[ka][nocc:], eps[ki][:nocc])
        for i in 0..nocc {
            for a in 0..nvir {
                hdiag[(ki * nocc + i) * nvir + a] = eps[ka][nocc + a] - eps[ki][i];
            }
        }
        // `:309-310` — two DIAGONAL extractions (`'aiia->ia'`, `'iaia->ia'`),
        // written as loops because a repeated index inside one operand is a
        // diagonal, which the `einsum` in `pyscf-pbc-cc` deliberately rejects.
        let voov = eris
            .blk(Blk::Voov, ka, ki, ki)
            .map_err(|e| PbcCiError::Shape(e.to_string()))?;
        let ovov = eris
            .blk(Blk::Ovov, ki, ka, ki)
            .map_err(|e| PbcCiError::Shape(e.to_string()))?;
        for i in 0..nocc {
            for a in 0..nvir {
                let (r1, _) = voov
                    .at(&[a, i, i, a])
                    .map_err(|e| PbcCiError::Shape(e.to_string()))?;
                let (r2, _) = ovov
                    .at(&[i, a, i, a])
                    .map_err(|e| PbcCiError::Shape(e.to_string()))?;
                hdiag[(ki * nocc + i) * nvir + a] += 2.0 * r1 - r2;
            }
        }
    }
    Ok(hdiag)
}

/// `cis_matvec_singlet(cis, vector, kshift, eris)` — `kcis_rhf.py:128-187`,
/// the `cis.direct == False` branch.
///
/// # Errors
/// Propagates the ERI access and every shape check.
pub fn cis_matvec(
    eris: &KEris,
    kconserv: &Kconserv,
    kshift: usize,
    r: &CTensor,
) -> Result<CTensor, PbcCiError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let n = vector_size(nk, nocc, nvir);
    if r.re.len() != n {
        return Err(PbcCiError::Shape(format!(
            "CIS vector of length {} for a {n}-element space",
            r.re.len()
        )));
    }
    let kr = get_kconserv_r(kconserv, nk, kshift);
    let eps = fock_diagonal(eris)?;
    let mut hr = CTensor::zeros(n);
    let at = |k: usize, i: usize, a: usize| (k * nocc + i) * nvir + a;

    for ki in 0..nk {
        let ka = kr[ki];
        // `:163-164` — the orbital-energy difference.
        for i in 0..nocc {
            for a in 0..nvir {
                let f = at(ki, i, a);
                let d = eps[ka][nocc + a] - eps[ki][i];
                hr.re[f] += r.re[f] * d;
                hr.im[f] += r.im[f] * d;
            }
        }
        // `:169-171` — 2 * einsum("xjb,xajib->ia", r, eris.voov[ka,:,ki])
        //             -     einsum("xjb,xjaib->ia", r, eris.ovov[:,ka,ki])
        //
        // **The free k-index is the SECOND on `voov` and the FIRST on
        // `ovov`.** They produce the same shape, so swapping them is a
        // plausible wrong number no shape check catches — the defect 16-07's
        // `cc_Wovvo` actually shipped for one cycle.
        for kj in 0..nk {
            let kb = kr[kj];
            let voov = eris
                .blk(Blk::Voov, ka, kj, ki)
                .map_err(|e| PbcCiError::Shape(e.to_string()))?;
            let ovov = eris
                .blk(Blk::Ovov, kj, ka, ki)
                .map_err(|e| PbcCiError::Shape(e.to_string()))?;
            for i in 0..nocc {
                for a in 0..nvir {
                    let mut acc_re: Vec<f64> = Vec::with_capacity(2 * nocc * nvir);
                    let mut acc_im: Vec<f64> = Vec::with_capacity(2 * nocc * nvir);
                    for j in 0..nocc {
                        for b in 0..nvir {
                            let f = at(kj, j, b);
                            let (rr, ri) = (r.re[f], r.im[f]);
                            // <aj|ib>
                            let (vr, vi) = voov
                                .at(&[a, j, i, b])
                                .map_err(|e| PbcCiError::Shape(e.to_string()))?;
                            acc_re.push(2.0 * (rr * vr - ri * vi));
                            acc_im.push(2.0 * (rr * vi + ri * vr));
                            // <ja|ib>
                            let (or_, oi) = ovov
                                .at(&[j, a, i, b])
                                .map_err(|e| PbcCiError::Shape(e.to_string()))?;
                            acc_re.push(-(rr * or_ - ri * oi));
                            acc_im.push(-(rr * oi + ri * or_));
                        }
                    }
                    let f = at(ki, i, a);
                    hr.re[f] += oracle_sum(&acc_re);
                    hr.im[f] += oracle_sum(&acc_im);
                }
            }
        }
    }
    Ok(hr)
}

/// `cis_H` built COLUMN BY COLUMN through the matvec — `kcis_rhf.py:105-111`,
/// upstream's `build_full_H = False` default.
///
/// Returns the `n × n` matrix column-major, ready for [`eig_general`].
///
/// # Errors
/// Propagates the matvec.
pub fn cis_h_from_matvec(
    eris: &KEris,
    kconserv: &Kconserv,
    kshift: usize,
) -> Result<Vec<faer::c64>, PbcCiError> {
    let n = vector_size(eris.nkpts, eris.nocc, eris.nvir);
    let mut h = vec![faer::c64::new(0.0, 0.0); n * n];
    for col in 0..n {
        let mut e = CTensor::zeros(n);
        e.re[col] = 1.0;
        let v = cis_matvec(eris, kconserv, kshift, &e)?;
        for row in 0..n {
            h[col * n + row] = faer::c64::new(v.re[row], v.im[row]);
        }
    }
    Ok(h)
}

/// The CIS roots at one `kshift` — `kernel(cis, nroots, eris, kptlist)`
/// (`kcis_rhf.py:42-125`), for ONE `kshift`.
///
/// # Errors
/// Propagates the ERI access, the matvec and the eigensolve.
pub fn kernel_at_kshift(
    eris: &KEris,
    kconserv: &Kconserv,
    kshift: usize,
    nroots: usize,
    opts: &KcisOpts,
) -> Result<Vec<f64>, PbcCiError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let size = vector_size(nk, nocc, nvir);
    let nroots = nroots.min(size);
    let diag = cis_diag(eris, kconserv, kshift)?;
    if diag.len() != size {
        return Err(PbcCiError::Shape(format!(
            "the CIS diagonal has {} elements, the vector space {size}",
            diag.len()
        )));
    }

    if !opts.davidson {
        // `:104-116` — the DENSE fallback. `np.linalg.eig`, sorted ascending,
        // the lowest `nroots` taken.
        let h = cis_h_from_matvec(eris, kconserv, kshift)?;
        let (w, _) = eig_general(&h, size)
            .map_err(|e| PbcCiError::Algebra(format!("dense CIS eigensolve: {e}")))?;
        let mut re: Vec<f64> = w.iter().map(|c| c.re).collect();
        re.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        re.truncate(nroots);
        return Ok(re);
    }

    // `:87-98` — the Davidson branch, with upstream's own preconditioner
    // `r / (e0 - diag + 1e-12)` (`:95-96`) and the default `pick_real_eigs`.
    let guess = get_init_guess(&diag, nroots, size);
    let opts_d = DavidsonOptions {
        nroots,
        max_space: opts.max_space,
        max_cycle: opts.max_cycle,
        tol: opts.conv_tol,
        ..Default::default()
    };
    // The Davidson's `aop` cannot return a Result, so a matvec failure is
    // captured here and re-raised after the solve rather than swallowed into
    // a zero vector — a silent zero would be a wrong number, which is the one
    // outcome this project never accepts.
    let failure: std::cell::RefCell<Option<PbcCiError>> = std::cell::RefCell::new(None);
    let res = davidson_nosym1(
        |xs: &[CTensor]| {
            xs.iter()
                .map(|x| match cis_matvec(eris, kconserv, kshift, x) {
                    Ok(v) => v,
                    Err(e) => {
                        *failure.borrow_mut() = Some(e);
                        CTensor::zeros(size)
                    }
                })
                .collect()
        },
        guess,
        |dx: &CTensor, e0: f64, _x0: &CTensor| {
            let mut out = CTensor::zeros(size);
            for i in 0..size {
                let d = e0 - diag[i] + 1e-12;
                out.re[i] = dx.re[i] / d;
                out.im[i] = dx.im[i] / d;
            }
            out
        },
        &opts_d,
        pick_real_eigs,
    )
    .map_err(|e| PbcCiError::Algebra(format!("CIS Davidson: {e}")))?;
    if let Some(e) = failure.into_inner() {
        return Err(e);
    }
    Ok(res.e)
}

/// `get_init_guess(nroots, diag)` — `kcis_rhf.py:415-426`: unit vectors on the
/// `nroots` smallest diagonal elements.
pub fn get_init_guess(diag: &[f64], nroots: usize, size: usize) -> Vec<CTensor> {
    let mut idx: Vec<usize> = (0..diag.len()).collect();
    idx.sort_by(|&a, &b| diag[a].partial_cmp(&diag[b]).unwrap_or(std::cmp::Ordering::Equal));
    idx.into_iter()
        .take(nroots.min(size))
        .map(|i| {
            let mut g = CTensor::zeros(size);
            g.re[i] = 1.0;
            g
        })
        .collect()
}

/// `eris.fock[k].diagonal().real` — see the module doc for why this is NOT
/// `eris.mo_energy`.
fn fock_diagonal(eris: &KEris) -> Result<Vec<Vec<f64>>, PbcCiError> {
    (0..eris.nkpts)
        .map(|k| {
            let f = eris
                .fock_at(k)
                .map_err(|e| PbcCiError::Shape(e.to_string()))?;
            (0..eris.nmo)
                .map(|p| {
                    f.at(&[p, p])
                        .map(|v| v.0)
                        .map_err(|e| PbcCiError::Shape(e.to_string()))
                })
                .collect()
        })
        .collect()
}

/// `_init_cis_df_eris`'s `dimension == 2` refusal — `kcis_rhf.py:630-637`.
///
/// Upstream's reason, quoted: 2-D ERIs are not positive definite, the 3-index
/// tensor is stored as a positive and a negative part, and "the negative part
/// is not considered in the DF-driven CCSD implementation". `graphene` is a
/// `§9.2` reference cell, so this refusal is reachable, not theoretical.
///
/// # Errors
/// Always, when `dimension == 2`. That is the point.
pub fn check_dimension_for_direct_df(dimension: u8) -> Result<(), PbcCiError> {
    if dimension == 2 {
        return Err(PbcCiError::NotImplementedUpstream {
            upstream: "pyscf/pbc/ci/kcis_rhf.py:637",
            what: "the direct-DF CIS path at cell.dimension == 2: 2-D ERIs are not \
                   positive definite and the 3-index tensor's negative part is not \
                   handled",
        });
    }
    Ok(())
}

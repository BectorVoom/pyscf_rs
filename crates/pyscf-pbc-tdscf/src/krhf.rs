//! k-point restricted TDA/TDHF (`pbc/tdscf/krhf.py`, 537 l).
//!
//! The phase headline: `ROADMAP:465`'s gate names the KRHF-TDA lowest
//! excitation (now at 19-01's measured floor, Gate A1: 1e-5 in the asserted
//! unit, per shift).
//!
//! Structure vs 19-06: the response matrix at fixed transferred momentum
//! `kshift` couples `(ki, ia)` pairs across the k-mesh —
//! `A[(ki,ia)][(kj,jb)]` over `ka = kconserv[ki]`, `kb = kconserv[kj]` — so
//! roots BELONG to a shift. They are reported with their shift and sorted
//! within it, never across: a global sort mixes shifts and is close-but-wrong
//! on symmetric cells, wrong elsewhere (pinned by test on the two-shift
//! diamond fixture whose shifts differ at 10.957 vs 11.042 eV).
//!
//! Arithmetic is COMPLEX (k-point ERI blocks are complex128): the einsums are
//! complex-LINEAR (plain products, no conjugation — exactly as upstream
//! writes them), and `A` is Hermitian by the blocks' own symmetries
//! (measured 8e-17 on upstream's matrix — asserted here, not assumed).
//! TDA solves the Hermitian problem with [`pyscf_algebra::zeigh_gen`].
//!
//! TDHF at k is genuinely NON-HERMITIAN (`B ≠ B†` at 5e-4 — measured, so no
//! symmetric reduction applies): real-valued shifts delegate to the shared
//! [`crate::rhf::symm_tdhf`] core; genuinely-complex shifts are REFUSED
//! loudly. A complex-nosym 2n×2n dense solve would silently halve the
//! spectrum if done by real embedding without care — refused instead.
//!
//! Matrix build (`get_ab:add_hf_`, `ao2mo_7d([mo,orbo,mo,mo])` blocks with
//! axes `(all-MO, occ, all-MO, all-MO)`, `weight = 1/nkpts`):
//!
//! * `A[(ki,ia)][(kj,jb)] = δ·(e_vir[ka][a] − e_occ[ki][i])`
//!   `+ 2·eri[ka,ki,kj][nocc+a, i, j, nocc+b]`
//!   `− hyb·eri[kj,ki,ka][j, i, nocc+a, nocc+b]`;
//! * `B[(ki,ia)][(kj,jb)] = 2·eri[ka,ki,kb][nocc+a, i, nocc+b, j]`
//!   `− hyb·eri[ka,kj,kb][nocc+a, j, nocc+b, i]`
//!   with `kb = kconserv[kj]`.
//!
//! Triplet drops both Coulomb terms (the 19-06 convention, verified through
//! triplet roots).

use pyscf_algebra::{CTensor, zeigh_gen};

use crate::error::PbcTdscfError;
use crate::types::{TdaConfig, TdaResult};

/// Boundary: k-shift problems with `max|im| > COMPLEX_TDHF_BOUND` refuse the
/// real TDHF path (see module docs).
pub const COMPLEX_TDHF_BOUND: f64 = 1e-10;

/// Build the k-shift Casida `A`/`B` as [`CTensor`] (row-major `(nk·dim)²`,
/// `dim = nocc·nvir`, `[(ki,ia)][(kj,jb)]` with `ia = i·nvir + a`).
///
/// `eri7[(kp·nk+kq)·nk+kr]` holds the full `(nmo,nocc,nmo,nmo)` block;
/// `kconserv[ki] = ka` is the shift's 1-D conservation map; `weight` is
/// `1/nkpts` (applied here, documented — not twice). All products are
/// complex-linear (no conjugation).
#[allow(clippy::too_many_arguments)]
pub fn build_kab(
    eri7_re: &[Vec<f64>],
    eri7_im: &[Vec<f64>],
    e_occ_k: &[Vec<f64>],
    e_vir_k: &[Vec<f64>],
    kconserv: &[usize],
    nkpts: usize,
    nocc: usize,
    nmo: usize,
    singlet: bool,
    hyb: f64,
    weight: f64,
) -> Result<(CTensor, CTensor), PbcTdscfError> {
    let nvir = nmo - nocc;
    let dim = nocc * nvir;
    let nkd = nkpts * dim;
    if eri7_re.len() != nkpts * nkpts * nkpts
        || eri7_im.len() != nkpts * nkpts * nkpts
        || kconserv.len() != nkpts
    {
        return Err(PbcTdscfError::ShapeMismatch { expected: nkpts, got: kconserv.len() });
    }
    if e_occ_k.len() != nkpts || e_vir_k.len() != nkpts {
        return Err(PbcTdscfError::ShapeMismatch { expected: nkpts, got: e_occ_k.len() });
    }
    // eri7 block index (length-checked once up front so the hot loop stays
    // infallible on indexing).
    for (bi, b) in eri7_re.iter().chain(eri7_im.iter()).enumerate() {
        if b.len() != nmo * nocc * nmo * nmo {
            return Err(PbcTdscfError::ShapeMismatch {
                expected: nmo * nocc * nmo * nmo,
                got: b.len(),
            });
        }
        let _ = bi;
    }
    let at = |v: &[Vec<f64>], p: usize, q: usize, r: usize, a: usize, b: usize, c: usize, d: usize| -> f64 {
        v[(p * nkpts + q) * nkpts + r][((a * nocc + b) * nmo + c) * nmo + d]
    };
    let mut are = vec![0.0f64; nkd * nkd];
    let mut aim = vec![0.0f64; nkd * nkd];
    let mut bre = vec![0.0f64; nkd * nkd];
    let mut bim = vec![0.0f64; nkd * nkd];
    for ki in 0..nkpts {
        let ka = kconserv[ki];
        if ka >= nkpts {
            return Err(PbcTdscfError::ShapeMismatch { expected: nkpts, got: ka });
        }
        for kj in 0..nkpts {
            let kb = kconserv[kj];
            if kb >= nkpts {
                return Err(PbcTdscfError::ShapeMismatch { expected: nkpts, got: kb });
            }
            for i in 0..nocc {
                for aj in 0..nvir {
                    let ia = ki * dim + i * nvir + aj;
                    for j in 0..nocc {
                        for bj in 0..nvir {
                            let jb = kj * dim + j * nvir + bj;
                            // Complex-linear products (no conjugation).
                            let jare = at(eri7_re, ka, ki, kj, nocc + aj, i, j, nocc + bj);
                            let jaim = at(eri7_im, ka, ki, kj, nocc + aj, i, j, nocc + bj);
                            let kaare = at(eri7_re, kj, ki, ka, j, i, nocc + aj, nocc + bj);
                            let kaaim = at(eri7_im, kj, ki, ka, j, i, nocc + aj, nocc + bj);
                            let jb2re = at(eri7_re, ka, ki, kb, nocc + aj, i, nocc + bj, j);
                            let jb2im = at(eri7_im, ka, ki, kb, nocc + aj, i, nocc + bj, j);
                            let kb2re = at(eri7_re, ka, kj, kb, nocc + aj, j, nocc + bj, i);
                            let kb2im = at(eri7_im, ka, kj, kb, nocc + aj, j, nocc + bj, i);
                            if singlet {
                                are[ia * nkd + jb] = weight * (2.0 * jare - hyb * kaare);
                                aim[ia * nkd + jb] = weight * (2.0 * jaim - hyb * kaaim);
                                bre[ia * nkd + jb] = weight * (2.0 * jb2re - hyb * kb2re);
                                bim[ia * nkd + jb] = weight * (2.0 * jb2im - hyb * kb2im);
                            } else {
                                are[ia * nkd + jb] = weight * (-hyb * kaare);
                                aim[ia * nkd + jb] = weight * (-hyb * kaaim);
                                bre[ia * nkd + jb] = weight * (-hyb * kb2re);
                                bim[ia * nkd + jb] = weight * (-hyb * kb2im);
                            }
                        }
                    }
                }
            }
        }
    }
    // Diagonal ONCE (upstream initializes it via diag() before the loops —
    // folding it into the kj loop would add the gap nk times).
    for ki in 0..nkpts {
        let ka = kconserv[ki];
        for i in 0..nocc {
            for aj in 0..nvir {
                let ia = ki * dim + i * nvir + aj;
                are[ia * nkd + ia] += e_vir_k[ka][aj] - e_occ_k[ki][i];
            }
        }
    }
    Ok((CTensor { re: are, im: aim }, CTensor { re: bre, im: bim }))
}

/// Assert (complex) Hermiticity: `max|A − A†|` against a relative bound.
fn assert_hermitian(a: &CTensor, nkd: usize) -> Result<(), PbcTdscfError> {
    let mut asym = 0.0f64;
    for i in 0..nkd {
        for j in 0..nkd {
            asym = asym.max((a.re[i * nkd + j] - a.re[j * nkd + i]).abs());
            asym = asym.max((a.im[i * nkd + j] + a.im[j * nkd + i]).abs());
        }
    }
    let scale: f64 = a.re.iter().chain(a.im.iter()).map(|x| x.abs()).fold(0.0, f64::max);
    if asym > 1e-8 * scale.max(1.0) {
        return Err(PbcTdscfError::ShapeMismatch { expected: 0, got: (asym * 1e12) as usize });
    }
    Ok(())
}

/// k-point RHF-TDA driver for one shift: build the Hermitian `A`, solve with
/// [`zeigh_gen`], report WITH the shift (never pooled across shifts).
#[allow(clippy::too_many_arguments)]
pub fn kernel_krhf_tda(
    eri7_re: &[Vec<f64>],
    eri7_im: &[Vec<f64>],
    e_occ_k: &[Vec<f64>],
    e_vir_k: &[Vec<f64>],
    kconserv: &[usize],
    nkpts: usize,
    nocc: usize,
    nmo: usize,
    hyb: f64,
    kshift: usize,
    cfg: &TdaConfig,
) -> Result<TdaResult, PbcTdscfError> {
    let (a, _b) = build_kab(
        eri7_re, eri7_im, e_occ_k, e_vir_k, kconserv, nkpts, nocc, nmo, cfg.singlet, hyb,
        1.0 / nkpts as f64,
    )?;
    let nkd = nkpts * nocc * (nmo - nocc);
    if cfg.nroots > nkd {
        return Err(PbcTdscfError::TooManyRoots { nroots: cfg.nroots, dim: nkd });
    }
    assert_hermitian(&a, nkd)?;
    let ident = CTensor {
        re: {
            let mut s = vec![0.0f64; nkd * nkd];
            for i in 0..nkd {
                s[i * nkd + i] = 1.0;
            }
            s
        },
        im: vec![0.0f64; nkd * nkd],
    };
    let (evals, _) = zeigh_gen(&a, &ident, nkd).map_err(|e| {
        PbcTdscfError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!("krhf-tda zeigh failed: {e}")),
        ))
    })?;
    Ok(TdaResult {
        energies: evals[..cfg.nroots].to_vec(),
        oscillator: vec![0.0f64; cfg.nroots],
        kshift,
        converged: true,
    })
}

/// k-point RHF-TDHF driver for one shift.
///
/// Real-valued shifts delegate to the shared [`crate::rhf::symm_tdhf`] core;
/// genuinely-complex shifts are REFUSED (non-Hermitian complex TDHF needs the
/// complex-nosym dense solve — a named non-goal here, never a silent
/// real-part truncation).
#[allow(clippy::too_many_arguments)]
pub fn kernel_krhf_tdhf(
    eri7_re: &[Vec<f64>],
    eri7_im: &[Vec<f64>],
    e_occ_k: &[Vec<f64>],
    e_vir_k: &[Vec<f64>],
    kconserv: &[usize],
    nkpts: usize,
    nocc: usize,
    nmo: usize,
    hyb: f64,
    kshift: usize,
    cfg: &TdaConfig,
) -> Result<TdaResult, PbcTdscfError> {
    let (a, b) = build_kab(
        eri7_re, eri7_im, e_occ_k, e_vir_k, kconserv, nkpts, nocc, nmo, cfg.singlet, hyb,
        1.0 / nkpts as f64,
    )?;
    let nkd = nkpts * nocc * (nmo - nocc);
    let max_im: f64 = a
        .im
        .iter()
        .chain(b.im.iter())
        .map(|x| x.abs())
        .fold(0.0, f64::max);
    if max_im > COMPLEX_TDHF_BOUND {
        return Err(PbcTdscfError::NotYetImplemented {
            module: "tdscf/krhf complex-valued TDHF (non-Hermitian complex dense solve)",
        });
    }
    let mut out = crate::rhf::symm_tdhf(&a.re, &b.re, nkd, cfg.nroots)?;
    out.kshift = kshift;
    Ok(out)
}

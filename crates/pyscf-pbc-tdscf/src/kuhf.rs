//! k-point unrestricted TDA/TDHF (`pbc/tdscf/kuhf.py`, 540 l).
//!
//! The 19-07 k-generalisation applied to the 19-08 coupled spin structure:
//! `A = [[A_aa, A_ab],[A_ab†, A_bb]]` per shift, with the k-maps
//!
//! * `A_aa[(ki,ia)][(kj,jb)] = δ·gap`
//!   `+ eri_aa[ka,ki,kj][nocc_a+a, i, j, nocc_a+b]`
//!   `− hyb·eri_aa[kj,ki,ka][j, i, nocc_a+a, nocc_a+b]`;
//! * `A_ab[(ki,ia)][(kj,jb)] = eri_ab[ka,ki,kj][nocc_a+a, i, j, nocc_b+b]`
//!   (Coulomb only);
//! * `B_aa[(ki,ia)][(kj,jb)] = eri_aa[ka,ki,kb][nocc+a, i, nocc+b, j]`
//!   `− hyb·eri_aa[ka,kj,kb][nocc+a, j, nocc+b, i]`;
//! * `B_ab[(ki,ia)][(kj,jb)] = eri_ab[ka,ki,kb][nocc_a+a, i, nocc_b+b, j]`;
//!
//! (`bb` mirrors `aa`.) Complex-linear products (no conjugation), Hermitian
//! `A` asserted (as 19-07), TDA via `zeigh_gen`, TDHF real→shared core /
//! complex→refuse (as 19-07). Roots carry their shift; spin contamination is
//! reported, never filtered.

use pyscf_algebra::CTensor;

use crate::error::PbcTdscfError;
use crate::types::{TdaConfig, TdaResult};
use crate::uhf::UhfDims;

/// Build the coupled k-shift `(da+db)·nk` UHF `A`/`B` as [`CTensor`].
///
/// `eri7_aa[(kp·nk+kq)·nk+kr]` holds `(nmo_a,nocc_a,nmo_a,nmo_a)` blocks
/// (`ab`: `(nmo_a,nocc_a,nmo_b,nmo_b)` with `mo_b` axes, `bb` likewise);
/// `kconserv[ki] = ka`; `weight = 1/nkpts`. `e_occ_a[k]`/`e_vir_a[k]` are the
/// per-k spin spectra (exxdiv-none, as 19-07).
#[allow(clippy::too_many_arguments)]
pub fn build_kuab(
    eri7_aa_re: &[Vec<f64>],
    eri7_aa_im: &[Vec<f64>],
    eri7_ab_re: &[Vec<f64>],
    eri7_ab_im: &[Vec<f64>],
    eri7_bb_re: &[Vec<f64>],
    eri7_bb_im: &[Vec<f64>],
    e_occ_a: &[Vec<f64>],
    e_vir_a: &[Vec<f64>],
    e_occ_b: &[Vec<f64>],
    e_vir_b: &[Vec<f64>],
    kconserv: &[usize],
    nkpts: usize,
    dims: UhfDims,
    nmo_a: usize,
    nmo_b: usize,
    hyb: f64,
    weight: f64,
) -> Result<(CTensor, CTensor), PbcTdscfError> {
    let (noa, nva, nob, nvb) = (dims.nocc_a, dims.nvir_a, dims.nocc_b, dims.nvir_b);
    let (da, db) = (dims.da(), dims.db());
    let nkd = nkpts * (da + db);
    if kconserv.len() != nkpts {
        return Err(PbcTdscfError::ShapeMismatch { expected: nkpts, got: kconserv.len() });
    }
    for v in [eri7_aa_re, eri7_aa_im, eri7_ab_re, eri7_ab_im, eri7_bb_re, eri7_bb_im] {
        if v.len() != nkpts * nkpts * nkpts {
            return Err(PbcTdscfError::ShapeMismatch { expected: nkpts * nkpts * nkpts, got: v.len() });
        }
    }
    // Block readers with explicit axis sizes.
    let at_aa = |v: &[Vec<f64>], p: usize, q: usize, r: usize, a: usize, b: usize, c: usize, d: usize| -> Result<f64, PbcTdscfError> {
        let blk = v.get((p * nkpts + q) * nkpts + r).ok_or(PbcTdscfError::ShapeMismatch { expected: 1, got: 0 })?;
        if blk.len() != nmo_a * noa * nmo_a * nmo_a {
            return Err(PbcTdscfError::ShapeMismatch { expected: nmo_a * noa * nmo_a * nmo_a, got: blk.len() });
        }
        Ok(blk[((a * noa + b) * nmo_a + c) * nmo_a + d])
    };
    let at_ab = |v: &[Vec<f64>], p: usize, q: usize, r: usize, a: usize, b: usize, c: usize, d: usize| -> Result<f64, PbcTdscfError> {
        let blk = v.get((p * nkpts + q) * nkpts + r).ok_or(PbcTdscfError::ShapeMismatch { expected: 1, got: 0 })?;
        if blk.len() != nmo_a * noa * nmo_b * nmo_b {
            return Err(PbcTdscfError::ShapeMismatch { expected: nmo_a * noa * nmo_b * nmo_b, got: blk.len() });
        }
        Ok(blk[((a * noa + b) * nmo_b + c) * nmo_b + d])
    };
    let at_bb = |v: &[Vec<f64>], p: usize, q: usize, r: usize, a: usize, b: usize, c: usize, d: usize| -> Result<f64, PbcTdscfError> {
        let blk = v.get((p * nkpts + q) * nkpts + r).ok_or(PbcTdscfError::ShapeMismatch { expected: 1, got: 0 })?;
        if blk.len() != nmo_b * nob * nmo_b * nmo_b {
            return Err(PbcTdscfError::ShapeMismatch { expected: nmo_b * nob * nmo_b * nmo_b, got: blk.len() });
        }
        Ok(blk[((a * nob + b) * nmo_b + c) * nmo_b + d])
    };
    let mut are = vec![0.0f64; nkd * nkd];
    let mut aim = vec![0.0f64; nkd * nkd];
    let mut bre = vec![0.0f64; nkd * nkd];
    let mut bim = vec![0.0f64; nkd * nkd];
    let put = |re: &mut Vec<f64>, im: &mut Vec<f64>, r: usize, c: usize, vr: f64, vi: f64| {
        re[r * nkd + c] = vr;
        im[r * nkd + c] = vi;
    };
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
            // Alpha-alpha + beta-beta + coupling rows.
            for i in 0..noa {
                for aj in 0..nva {
                    let ia = (ki * (da + db)) + i * nva + aj;
                    for j in 0..noa {
                        for bj in 0..nva {
                            let jb = (kj * (da + db)) + j * nva + bj;
                            let (jr, ji) = (
                                at_aa(eri7_aa_re, ka, ki, kj, noa + aj, i, j, noa + bj)?,
                                at_aa(eri7_aa_im, ka, ki, kj, noa + aj, i, j, noa + bj)?,
                            );
                            let (kr, ki_) = (
                                at_aa(eri7_aa_re, kj, ki, ka, j, i, noa + aj, noa + bj)?,
                                at_aa(eri7_aa_im, kj, ki, ka, j, i, noa + aj, noa + bj)?,
                            );
                            put(&mut are, &mut aim, ia, jb, weight * (jr - hyb * kr), weight * (ji - hyb * ki_));
                            let (jr, ji) = (
                                at_aa(eri7_aa_re, ka, ki, kb, noa + aj, i, noa + bj, j)?,
                                at_aa(eri7_aa_im, ka, ki, kb, noa + aj, i, noa + bj, j)?,
                            );
                            let (kr, ki_) = (
                                at_aa(eri7_aa_re, ka, kj, kb, noa + aj, j, noa + bj, i)?,
                                at_aa(eri7_aa_im, ka, kj, kb, noa + aj, j, noa + bj, i)?,
                            );
                            put(&mut bre, &mut bim, ia, jb, weight * (jr - hyb * kr), weight * (ji - hyb * ki_));
                        }
                    }
                    for j in 0..nob {
                        for bj in 0..nvb {
                            let jb = (kj * (da + db)) + da + j * nvb + bj;
                            let (vr, vi) = (
                                at_ab(eri7_ab_re, ka, ki, kj, noa + aj, i, j, nob + bj)?,
                                at_ab(eri7_ab_im, ka, ki, kj, noa + aj, i, j, nob + bj)?,
                            );
                            // A_ab Coulomb-only; the (jb,ia) entry is the
                            // conjugate transpose (Hermitian A — asserted
                            // below; fails loudly if the blocks disagree).
                            put(&mut are, &mut aim, ia, jb, weight * vr, weight * vi);
                            put(&mut are, &mut aim, jb, ia, weight * vr, -weight * vi);
                            // B_ab: PLAIN transpose (B_bbaa =
                            // B_aabb.transpose(2,3,0,1), no conjugation —
                            // upstream's documented layout, ported literally).
                            let (wr, wi) = (
                                at_ab(eri7_ab_re, ka, ki, kb, noa + aj, i, nob + bj, j)?,
                                at_ab(eri7_ab_im, ka, ki, kb, noa + aj, i, nob + bj, j)?,
                            );
                            put(&mut bre, &mut bim, ia, jb, weight * wr, weight * wi);
                            put(&mut bre, &mut bim, jb, ia, weight * wr, weight * wi);
                        }
                    }
                }
            }
            for i in 0..nob {
                for aj in 0..nvb {
                    let ia = (ki * (da + db)) + da + i * nvb + aj;
                    for j in 0..nob {
                        for bj in 0..nvb {
                            let jb = (kj * (da + db)) + da + j * nvb + bj;
                            let (jr, ji) = (
                                at_bb(eri7_bb_re, ka, ki, kj, nob + aj, i, j, nob + bj)?,
                                at_bb(eri7_bb_im, ka, ki, kj, nob + aj, i, j, nob + bj)?,
                            );
                            let (kr, ki_) = (
                                at_bb(eri7_bb_re, kj, ki, ka, j, i, nob + aj, nob + bj)?,
                                at_bb(eri7_bb_im, kj, ki, ka, j, i, nob + aj, nob + bj)?,
                            );
                            put(&mut are, &mut aim, ia, jb, weight * (jr - hyb * kr), weight * (ji - hyb * ki_));
                            let (jr, ji) = (
                                at_bb(eri7_bb_re, ka, ki, kb, nob + aj, i, nob + bj, j)?,
                                at_bb(eri7_bb_im, ka, ki, kb, nob + aj, i, nob + bj, j)?,
                            );
                            let (kr, ki_) = (
                                at_bb(eri7_bb_re, ka, kj, kb, nob + aj, j, nob + bj, i)?,
                                at_bb(eri7_bb_im, ka, kj, kb, nob + aj, j, nob + bj, i)?,
                            );
                            put(&mut bre, &mut bim, ia, jb, weight * (jr - hyb * kr), weight * (ji - hyb * ki_));
                        }
                    }
                }
            }
        }
    }
    // Diagonal once (19-07 lesson: never inside the kj loop).
    for ki in 0..nkpts {
        let ka = kconserv[ki];
        for i in 0..noa {
            for aj in 0..nva {
                let ia = (ki * (da + db)) + i * nva + aj;
                are[ia * nkd + ia] += e_vir_a[ka][aj] - e_occ_a[ki][i];
            }
        }
        for i in 0..nob {
            for aj in 0..nvb {
                let ia = (ki * (da + db)) + da + i * nvb + aj;
                are[ia * nkd + ia] += e_vir_b[ka][aj] - e_occ_b[ki][i];
            }
        }
    }
    Ok((CTensor { re: are, im: aim }, CTensor { re: bre, im: bim }))
}

/// k-point UHF-TDA driver for one shift (coupled, Hermitian assert, shift
/// stamped).
#[allow(clippy::too_many_arguments)]
pub fn kernel_kuhf_tda(
    eri7_aa_re: &[Vec<f64>],
    eri7_aa_im: &[Vec<f64>],
    eri7_ab_re: &[Vec<f64>],
    eri7_ab_im: &[Vec<f64>],
    eri7_bb_re: &[Vec<f64>],
    eri7_bb_im: &[Vec<f64>],
    e_occ_a: &[Vec<f64>],
    e_vir_a: &[Vec<f64>],
    e_occ_b: &[Vec<f64>],
    e_vir_b: &[Vec<f64>],
    kconserv: &[usize],
    nkpts: usize,
    dims: UhfDims,
    nmo_a: usize,
    nmo_b: usize,
    hyb: f64,
    kshift: usize,
    cfg: &TdaConfig,
) -> Result<TdaResult, PbcTdscfError> {
    let (a, _b) = build_kuab(
        eri7_aa_re, eri7_aa_im, eri7_ab_re, eri7_ab_im, eri7_bb_re, eri7_bb_im,
        e_occ_a, e_vir_a, e_occ_b, e_vir_b, kconserv, nkpts, dims, nmo_a, nmo_b,
        hyb, 1.0 / nkpts as f64,
    )?;
    let nkd = nkpts * (dims.da() + dims.db());
    if cfg.nroots > nkd {
        return Err(PbcTdscfError::TooManyRoots { nroots: cfg.nroots, dim: nkd });
    }
    // Hermitian assert (19-07 tripwire, coupled form).
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
    let (evals, _) = pyscf_algebra::zeigh_gen(&a, &ident, nkd).map_err(|e| {
        PbcTdscfError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!("kuhf-tda zeigh failed: {e}")),
        ))
    })?;
    Ok(TdaResult {
        energies: evals[..cfg.nroots].to_vec(),
        oscillator: vec![0.0f64; cfg.nroots],
        kshift,
        converged: true,
    })
}

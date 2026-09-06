//! `kintermediates_rhf` — the k-point restricted CC intermediates
//! (plan 16-04; `pyscf/pbc/cc/kintermediates_rhf.py`, Hirata et al.,
//! J. Chem. Phys. **120**, 2581 (2004), Eqs. (37)-(45)).
//!
//! The molecular sibling is `crates/pyscf-ccsd/src/rintermediates.rs`; this
//! module mirrors its shape and shares none of its arithmetic — that one is f64
//! throughout and carries no k-index (`16-CONTEXT §1.3`).
//!
//! # The primitive at every contraction site (`16-CONTEXT §3.2`)
//!
//! Every contraction below is written as an [`einsum`] subscript string
//! transcribed from the upstream line above it. **`numpy.einsum` / `lib.einsum`
//! never conjugate an operand**, so every one of these is
//! `oracle_zsum(Π operands)` — the UNCONJUGATED ordered complex sum — and never
//! `zdotc`. That is the property `15-REVIEW.md D-15-R-02` found could not be
//! stated safely in prose: there, "route through `oracle_dot`" silently
//! produced `Σ conj(x)·y` where `Σ x·y` was meant. Here the subscript string
//! carries the meaning, `crates/pyscf-pbc-cc/tests/zarr.rs`'s
//! `einsum_does_not_conjugate` pins the direction, and the FOUR places in this
//! phase that DO conjugate are explicit `.conj()` calls, exactly as upstream
//! writes them.
//!
//! # Storage tiers
//!
//! `Woooo`, `Wvoov`, `Wvovo` and `Wvvvv` are `nkpts³`-shaped and come back as
//! [`KBlocks`], whose tier is chosen from an exact byte count against the
//! caller's budget — upstream's `:132-137` / `:179-192` / `:423-455` branch,
//! with `_mem_usage` replaced (D-PBC-29 clause 4). `Wvvvv` at `nkpts³ · nvir⁴`
//! is the phase's largest single tensor: 2.0 MiB on diamond `gth-szv` 2×2×2,
//! **1.79 GiB** on `gth-dzvp` 2×2×2 and **68.7 GiB** on `gth-dzvp` 3×3×3.
//!
//! `t1`/`t2` are NOT tiered, and that matches upstream: `kccsd_rhf.py:553-554`
//! allocates them as plain arrays and never spills them.

use std::sync::Arc;

use pyscf_pbc_lib::{KIdx, Kconserv, get_kconserv3};
use pyscf_runtime::ZWorkspacePool;

use crate::eom_kccsd_ghf::{KLattice, Padding};
use crate::error::PbcCcError;
use crate::keris::{Blk, KEris};
use crate::ktensor::KBlocks;
use crate::zarr::{ZArr, einsum, einsum_scaled};

/// `t1` — `[nkpts, nocc, nvir]`.
pub type T1 = ZArr;
/// `t2` — `[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]`, indexed
/// `[ki][kj][ka]`.
pub type T2 = ZArr;

/// `2 * eris.oovv[k0,k1,k2] - eris.oovv[k0,k1,k3].transpose(0,1,3,2)` — the
/// spin-summed `Soovv` that appears in `cc_Foo`, `cc_Fvv` and `cc_Fov`.
fn soovv(eris: &KEris, k0: usize, k1: usize, k2: usize, k3: usize) -> Result<ZArr, PbcCcError> {
    let mut s = eris.blk(Blk::Oovv, k0, k1, k2)?;
    s.scale(2.0);
    let t = eris.blk(Blk::Oovv, k0, k1, k3)?.transpose(&[0, 1, 3, 2])?;
    s.sub_assign(&t)?;
    Ok(s)
}

/// `cc_Foo` — Eq. (37) `kappa`, `kintermediates_rhf.py:38-52`.
///
/// Returns `[nkpts, nocc, nocc]`.
///
/// # Errors
/// Propagates the ERI access and every shape check.
pub fn cc_foo(t1: &T1, t2: &T2, eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc) = (eris.nkpts, eris.nocc);
    let mut fki = ZArr::zeros(&[nkpts, nocc, nocc]);
    for ki in 0..nkpts {
        let kk = ki;
        let mut acc = eris.foo(ki)?;
        for kl in 0..nkpts {
            for kc in 0..nkpts {
                let kd = kconserv.get(kk, kc, kl) as usize;
                let s = soovv(eris, kk, kl, kc, kd)?;
                // `:47` einsum('klcd,ilcd->ki', Soovv, t2[ki,kl,kc])
                let t = t2.slice_leading(&[ki, kl, kc])?;
                acc.add_assign(&einsum("klcd,ilcd->ki", &[&s, &t])?)?;
            }
            // `:50-51` — the kc == ki term, with two t1 factors.
            let kd = kconserv.get(kk, ki, kl) as usize;
            let s = soovv(eris, kk, kl, ki, kd)?;
            let t1i = t1.slice_leading(&[ki])?;
            let t1l = t1.slice_leading(&[kl])?;
            acc.add_assign(&einsum("klcd,ic,ld->ki", &[&s, &t1i, &t1l])?)?;
        }
        fki.set_leading(&[ki], &acc)?;
    }
    Ok(fki)
}

/// `cc_Fvv` — Eq. (38), `kintermediates_rhf.py:55-69`.
///
/// Returns `[nkpts, nvir, nvir]`.
///
/// # Errors
/// As [`cc_foo`].
pub fn cc_fvv(t1: &T1, t2: &T2, eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nkpts, nvir) = (eris.nkpts, eris.nvir);
    let mut fac = ZArr::zeros(&[nkpts, nvir, nvir]);
    for ka in 0..nkpts {
        let kc = ka;
        let mut acc = eris.fvv(ka)?;
        for kl in 0..nkpts {
            for kk in 0..nkpts {
                let kd = kconserv.get(kk, kc, kl) as usize;
                let s = soovv(eris, kk, kl, kc, kd)?;
                // `:64` einsum('klcd,klad->ac', Soovv, t2[kk,kl,ka]), NEGATED
                let t = t2.slice_leading(&[kk, kl, ka])?;
                acc.sub_assign(&einsum("klcd,klad->ac", &[&s, &t])?)?;
            }
            // `:66-68` — the kk == ka term.
            let kd = kconserv.get(ka, kc, kl) as usize;
            let s = soovv(eris, ka, kl, kc, kd)?;
            let t1a = t1.slice_leading(&[ka])?;
            let t1l = t1.slice_leading(&[kl])?;
            acc.sub_assign(&einsum("klcd,ka,ld->ac", &[&s, &t1a, &t1l])?)?;
        }
        fac.set_leading(&[ka], &acc)?;
    }
    Ok(fac)
}

/// `cc_Fov` — Eq. (39), `kintermediates_rhf.py:72-80`.
///
/// Returns `[nkpts, nocc, nvir]`.
///
/// # Errors
/// As [`cc_foo`].
pub fn cc_fov(t1: &T1, _t2: &T2, eris: &KEris) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mut fkc = ZArr::zeros(&[nkpts, nocc, nvir]);
    for kk in 0..nkpts {
        let mut acc = eris.fov(kk)?;
        for kl in 0..nkpts {
            // `:78` Soovv = 2*oovv[kk,kl,kk] - oovv[kk,kl,kl].transpose(0,1,3,2)
            let s = soovv(eris, kk, kl, kk, kl)?;
            let t1l = t1.slice_leading(&[kl])?;
            acc.add_assign(&einsum("klcd,ld->kc", &[&s, &t1l])?)?;
        }
        fkc.set_leading(&[kk], &acc)?;
    }
    Ok(fkc)
}

/// `Loo` — Eq. (40) `lambda`, `kintermediates_rhf.py:84-93`.
///
/// # Errors
/// As [`cc_foo`].
pub fn loo(t1: &T1, t2: &T2, eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let nkpts = eris.nkpts;
    let mut lki = cc_foo(t1, t2, eris, kconserv)?;
    for ki in 0..nkpts {
        let mut acc = lki.slice_leading(&[ki])?;
        let fov = eris.fov(ki)?;
        let t1i = t1.slice_leading(&[ki])?;
        acc.add_assign(&einsum("kc,ic->ki", &[&fov, &t1i])?)?;
        for kl in 0..nkpts {
            let t1l = t1.slice_leading(&[kl])?;
            let a = eris.blk(Blk::Ooov, ki, kl, ki)?;
            acc.add_assign(&einsum_scaled("klic,lc->ki", &[&a, &t1l], 2.0)?)?;
            let b = eris.blk(Blk::Ooov, kl, ki, ki)?;
            acc.sub_assign(&einsum("lkic,lc->ki", &[&b, &t1l])?)?;
        }
        lki.set_leading(&[ki], &acc)?;
    }
    Ok(lki)
}

/// `Lvv` — Eq. (41), `kintermediates_rhf.py:95-104`.
///
/// # Errors
/// As [`cc_foo`].
pub fn lvv(t1: &T1, t2: &T2, eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let nkpts = eris.nkpts;
    let mut lac = cc_fvv(t1, t2, eris, kconserv)?;
    for ka in 0..nkpts {
        let mut acc = lac.slice_leading(&[ka])?;
        let fov = eris.fov(ka)?;
        let t1a = t1.slice_leading(&[ka])?;
        acc.sub_assign(&einsum("kc,ka->ac", &[&fov, &t1a])?)?;
        for kk in 0..nkpts {
            // `:102` Svovv = 2*vovv[ka,kk,ka] - vovv[ka,kk,kk].transpose(0,1,3,2)
            let mut s = eris.blk(Blk::Vovv, ka, kk, ka)?;
            s.scale(2.0);
            let t = eris.blk(Blk::Vovv, ka, kk, kk)?.transpose(&[0, 1, 3, 2])?;
            s.sub_assign(&t)?;
            let t1k = t1.slice_leading(&[kk])?;
            acc.add_assign(&einsum("akcd,kd->ac", &[&s, &t1k])?)?;
        }
        lac.set_leading(&[ka], &acc)?;
    }
    Ok(lac)
}

/// `cc_Woooo` — Eq. (42) `chi`, `kintermediates_rhf.py:108-141`.
///
/// **`kl` runs only to `kk`, and the mirror `Wklij[kl,kk,kj]` is written in a
/// SECOND loop after all the others are made** (upstream's own comment at
/// `:136`). Fusing the two loops reads the mirror before it is written.
///
/// # Errors
/// As [`cc_foo`], plus the arena's HARD refusal.
pub fn cc_woooo(
    pool: &Arc<ZWorkspacePool>,
    t1: &T1,
    t2: &T2,
    eris: &KEris,
    kconserv: &Kconserv,
    max_memory_bytes: usize,
) -> Result<KBlocks, PbcCcError> {
    let (nkpts, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let out = KBlocks::with_budget(pool, nkpts, &[nocc, nocc, nocc, nocc], max_memory_bytes)?;
    for kk in 0..nkpts {
        for kl in 0..=kk {
            for ki in 0..nkpts {
                let kj = kconserv.get(kk, ki, kl) as usize;
                let t1j = t1.slice_leading(&[kj])?;
                let t1i = t1.slice_leading(&[ki])?;
                let a = eris.blk(Blk::Ooov, kk, kl, ki)?;
                let mut oooo = einsum("klic,jc->klij", &[&a, &t1j])?;
                let b = eris.blk(Blk::Ooov, kl, kk, kj)?;
                oooo.add_assign(&einsum("lkjc,ic->klij", &[&b, &t1i])?)?;
                oooo.add_assign(&eris.blk(Blk::Oooo, kk, kl, ki)?)?;

                // `:120-131` — the vectorised form of
                //   Σ_kc einsum('klcd,ijcd->klij', oovv[kk,kl,kc], t2[ki,kj,kc])
                //   + einsum('klcd,ic,jd->klij', oovv[kk,kl,ki], t1[ki], t1[kj])
                // written with the two t1 factors folded into `t2t[ki]`.
                let vvoo = eris
                    .blk_free2(Blk::Oovv, kk, kl)?
                    .transpose(&[0, 3, 4, 1, 2])?
                    .reshape(&[nkpts * nvir, nvir, nocc, nocc])?;
                let mut t2t = ZArr::zeros(&[nkpts, nvir, nvir, nocc, nocc]);
                for kc in 0..nkpts {
                    let b = t2.slice_leading(&[ki, kj, kc])?.transpose(&[2, 3, 0, 1])?;
                    t2t.set_leading(&[kc], &b)?;
                }
                let mut ti = t2t.slice_leading(&[ki])?;
                ti.add_assign(&einsum("ic,jd->cdij", &[&t1i, &t1j])?)?;
                t2t.set_leading(&[ki], &ti)?;
                let t2t = t2t.reshape(&[nkpts * nvir, nvir, nocc, nocc])?;
                oooo.add_assign(&einsum("cdkl,cdij->klij", &[&vvoo, &t2t])?)?;
                out.set([kk, kl, ki], &oooo)?;
            }
        }
        // `:136-140` — the mirror, only after every member above exists.
        for kl in 0..=kk {
            for ki in 0..nkpts {
                let kj = kconserv.get(kk, ki, kl) as usize;
                let v = out.get([kk, kl, ki])?.transpose(&[1, 0, 3, 2])?;
                out.set([kl, kk, kj], &v)?;
            }
        }
    }
    Ok(out)
}

/// `cc_Wvvvv` — Eq. (43), `kintermediates_rhf.py:144-162`.
///
/// The phase's largest tensor. Same two-pass structure as [`cc_woooo`].
///
/// # Errors
/// As [`cc_woooo`].
pub fn cc_wvvvv(
    pool: &Arc<ZWorkspacePool>,
    t1: &T1,
    _t2: &T2,
    eris: &KEris,
    kconserv: &Kconserv,
    max_memory_bytes: usize,
) -> Result<KBlocks, PbcCcError> {
    let (nkpts, nvir) = (eris.nkpts, eris.nvir);
    let out = KBlocks::with_budget(pool, nkpts, &[nvir, nvir, nvir, nvir], max_memory_bytes)?;
    for ka in 0..nkpts {
        for kb in 0..=ka {
            for kc in 0..nkpts {
                let kd = kconserv.get(ka, kc, kb) as usize;
                let t1b = t1.slice_leading(&[kb])?;
                let t1a = t1.slice_leading(&[ka])?;
                let x = eris.blk(Blk::Vovv, ka, kb, kc)?;
                let mut vvvv = einsum_scaled("akcd,kb->abcd", &[&x, &t1b], -1.0)?;
                let y = eris.blk(Blk::Vovv, kb, ka, kd)?;
                vvvv.add_assign(&einsum_scaled("bkdc,ka->abcd", &[&y, &t1a], -1.0)?)?;
                vvvv.add_assign(&eris.blk(Blk::Vvvv, ka, kb, kc)?)?;
                out.set([ka, kb, kc], &vvvv)?;
            }
        }
        for kb in 0..=ka {
            for kc in 0..nkpts {
                let kd = kconserv.get(ka, kc, kb) as usize;
                let v = out.get([ka, kb, kc])?.transpose(&[1, 0, 3, 2])?;
                out.set([kb, ka, kd], &v)?;
            }
        }
    }
    Ok(out)
}

/// `cc_Wvoov` — Eq. (44), `kintermediates_rhf.py:165-195`.
///
/// # Errors
/// As [`cc_woooo`].
pub fn cc_wvoov(
    pool: &Arc<ZWorkspacePool>,
    t1: &T1,
    t2: &T2,
    eris: &KEris,
    kconserv: &Kconserv,
    max_memory_bytes: usize,
) -> Result<KBlocks, PbcCcError> {
    let (nkpts, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let out = KBlocks::with_budget(pool, nkpts, &[nvir, nocc, nocc, nvir], max_memory_bytes)?;
    for ka in 0..nkpts {
        for kk in 0..nkpts {
            // `:170-172` — the three k-batched terms, `x` running over the
            // free third k-index.
            let vovv = eris.blk_free2(Blk::Vovv, ka, kk)?;
            let mut voov_i = einsum("xakdc,xid->xakic", &[&vovv, t1])?;
            let ooov = eris.blk_free2(Blk::Ooov, ka, kk)?;
            let t1a = t1.slice_leading(&[ka])?;
            voov_i.sub_assign(&einsum("xlkic,la->xakic", &[&ooov, &t1a])?)?;
            voov_i.add_assign(&eris.blk_free2(Blk::Voov, ka, kk)?)?;

            for ki in 0..nkpts {
                let kc = kconserv.get(ka, ki, kk) as usize;
                let kd = kconserv.get(ka, kc, kk) as usize;

                // `:186-188` — tau = t2[:,ki,ka] with the t1 t1 term at ka.
                let mut tau = ZArr::zeros(&[nkpts, nocc, nocc, nvir, nvir]);
                for kl in 0..nkpts {
                    tau.set_leading(&[kl], &t2.slice_leading(&[kl, ki, ka])?)?;
                }
                let t1d = t1.slice_leading(&[kd])?;
                let mut ta = tau.slice_leading(&[ka])?;
                ta.add_assign(&einsum_scaled("id,la->liad", &[&t1d, &t1a], 2.0)?)?;
                tau.set_leading(&[ka], &ta)?;

                // `:187` `oovv_tmp = np.array(eris.oovv[kk,:,kc])` — the free
                // index is the SECOND, so the gather runs over `kl`. Getting
                // this wrong is silent: the shapes agree either way.
                let mut oovv_tmp = ZArr::zeros(&[nkpts, nocc, nocc, nvir, nvir]);
                for kl in 0..nkpts {
                    oovv_tmp.set_leading(&[kl], &eris.blk(Blk::Oovv, kk, kl, kc)?)?;
                }
                let mut vi = voov_i.slice_leading(&[ki])?;
                vi.sub_assign(&einsum_scaled(
                    "xklcd,xliad->akic",
                    &[&oovv_tmp, &tau],
                    0.5,
                )?)?;

                // `:190-191` Soovv_tmp = 2*oovv_tmp - oovv[:,kk,kc].transpose(0,2,1,3,4)
                let mut soovv_tmp = oovv_tmp.clone();
                soovv_tmp.scale(2.0);
                let mut other = ZArr::zeros(&[nkpts, nocc, nocc, nvir, nvir]);
                for kl in 0..nkpts {
                    other.set_leading(&[kl], &eris.blk(Blk::Oovv, kl, kk, kc)?)?;
                }
                soovv_tmp.sub_assign(&other.transpose(&[0, 2, 1, 3, 4])?)?;
                let mut t2ia = ZArr::zeros(&[nkpts, nocc, nocc, nvir, nvir]);
                for kl in 0..nkpts {
                    t2ia.set_leading(&[kl], &t2.slice_leading(&[ki, kl, ka])?)?;
                }
                vi.add_assign(&einsum_scaled(
                    "xklcd,xilad->akic",
                    &[&soovv_tmp, &t2ia],
                    0.5,
                )?)?;
                voov_i.set_leading(&[ki], &vi)?;
            }
            for ki in 0..nkpts {
                out.set([ka, kk, ki], &voov_i.slice_leading(&[ki])?)?;
            }
        }
    }
    Ok(out)
}

/// `cc_Wvovo` — Eq. (45), `kintermediates_rhf.py:197-226`.
///
/// # Errors
/// As [`cc_woooo`].
pub fn cc_wvovo(
    pool: &Arc<ZWorkspacePool>,
    t1: &T1,
    t2: &T2,
    eris: &KEris,
    kconserv: &Kconserv,
    max_memory_bytes: usize,
) -> Result<KBlocks, PbcCcError> {
    let (nkpts, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let out = KBlocks::with_budget(pool, nkpts, &[nvir, nocc, nvir, nocc], max_memory_bytes)?;
    for ka in 0..nkpts {
        for kk in 0..nkpts {
            for kc in 0..nkpts {
                let ki = kconserv.get(ka, kc, kk) as usize;
                let t1i = t1.slice_leading(&[ki])?;
                let t1a = t1.slice_leading(&[ka])?;
                let x = eris.blk(Blk::Vovv, ka, kk, kc)?;
                let mut vovo = einsum("akcd,id->akci", &[&x, &t1i])?;
                let y = eris.blk(Blk::Ooov, kk, ka, ki)?;
                vovo.sub_assign(&einsum("klic,la->akci", &[&y, &t1a])?)?;
                vovo.add_assign(&eris.blk(Blk::Ovov, kk, ka, ki)?.transpose(&[1, 0, 3, 2])?)?;

                // `:214-224` — the vectorised tau-like term over `kl`.
                let mut oovvf = ZArr::zeros(&[nkpts, nocc, nocc, nvir, nvir]);
                for kl in 0..nkpts {
                    oovvf.set_leading(&[kl], &eris.blk(Blk::Oovv, kl, kk, kc)?)?;
                }
                let oovvf = oovvf.reshape(&[nkpts * nocc, nocc, nvir, nvir])?;
                let mut t2f = ZArr::zeros(&[nkpts, nocc, nocc, nvir, nvir]);
                for kl in 0..nkpts {
                    t2f.set_leading(&[kl], &t2.slice_leading(&[kl, ki, ka])?)?;
                }
                let kd = kconserv.get(ka, kc, kk) as usize;
                let t1d = t1.slice_leading(&[kd])?;
                let mut ta = t2f.slice_leading(&[ka])?;
                ta.add_assign(&einsum_scaled("id,la->liad", &[&t1d, &t1a], 2.0)?)?;
                t2f.set_leading(&[ka], &ta)?;
                let t2f = t2f.reshape(&[nkpts * nocc, nocc, nvir, nvir])?;
                vovo.sub_assign(&einsum_scaled("lkcd,liad->akci", &[&oovvf, &t2f], 0.5)?)?;
                out.set([ka, kk, kc], &vovo)?;
            }
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// The EOM intermediates (`kintermediates_rhf.py:229-455`, plan 16-10 Task 1).
//
// `eom_kccsd_rhf._IMDS` builds `Loo`/`Lvv`/`cc_Fov` (shared 1e),
// `Wovov`/`Wovvo` (shared 2e), `Woooo`/`Wooov`/`Wovoo` (IP) and
// `Wvovv`/`Wvvvv`/`Wvvvo` (EA).
//
// `W1ovvo`/`W2ovvo` and `W1ovov`/`W2ovov` are separate upstream because the
// `1` halves are reused on their own by `Wvvvo` and `Wovoo` (`:382-383`,
// `:424-426`); building only the sums would mean recomputing them.
// ---------------------------------------------------------------------------

/// `Wooov` — `kintermediates_rhf.py:229-238`.
///
/// # Errors
/// Propagates the ERI access.
pub fn wooov(t1: &T1, eris: &KEris) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mut w = ZArr::zeros(&[nk, nk, nk, nocc, nocc, nocc, nvir]);
    for kk in 0..nk {
        for kl in 0..nk {
            for ki in 0..nk {
                let mut v = einsum(
                    "ic,klcd->klid",
                    &[&t1.slice_leading(&[ki])?, &eris.blk(Blk::Oovv, kk, kl, ki)?],
                )?;
                v.add_assign(&eris.blk(Blk::Ooov, kk, kl, ki)?)?;
                w.set_leading(&[kk, kl, ki], &v)?;
            }
        }
    }
    Ok(w)
}

/// `Wvovv` — `:240-249`.
///
/// # Errors
/// Propagates the ERI access.
pub fn wvovv(t1: &T1, eris: &KEris) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mut w = ZArr::zeros(&[nk, nk, nk, nvir, nocc, nvir, nvir]);
    for ka in 0..nk {
        for kl in 0..nk {
            for kc in 0..nk {
                let mut v = einsum_scaled(
                    "ka,klcd->alcd",
                    &[&t1.slice_leading(&[ka])?, &eris.blk(Blk::Oovv, ka, kl, kc)?],
                    -1.0,
                )?;
                v.add_assign(&eris.blk(Blk::Vovv, ka, kl, kc)?)?;
                w.set_leading(&[ka, kl, kc], &v)?;
            }
        }
    }
    Ok(w)
}

/// `W1ovvo` — `:251-266`. The `t2`-only half of `Wovvo`.
///
/// # Errors
/// As [`cc_foo`].
pub fn w1ovvo(t2: &T2, eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mut w = ZArr::zeros(&[nk, nk, nk, nocc, nvir, nvir, nocc]);
    for kk in 0..nk {
        for ka in 0..nk {
            for kc in 0..nk {
                let ki = kconserv.get(kk, kc, ka) as usize;
                // `:259` — `ovvo[kk,ka,kc,ki]` IS `voov[ka,kk,ki,kc]`
                // transposed; upstream writes the comment because the two have
                // the same shape and swapping them is silent.
                let mut v = eris.blk(Blk::Voov, ka, kk, ki)?.transpose(&[1, 0, 3, 2])?;
                for kl in 0..nk {
                    let kd = kconserv.get(ki, ka, kl) as usize;
                    // `St2 = 2 t2[ki,kl,ka] - t2[kl,ki,ka].transpose(1,0,2,3)`
                    let mut st2 = t2.slice_leading(&[ki, kl, ka])?;
                    st2.scale(2.0);
                    st2.sub_assign(&t2.slice_leading(&[kl, ki, ka])?.transpose(&[1, 0, 2, 3])?)?;
                    v.add_assign(&einsum(
                        "klcd,ilad->kaci",
                        &[&eris.blk(Blk::Oovv, kk, kl, kc)?, &st2],
                    )?)?;
                    v.sub_assign(&einsum(
                        "kldc,ilad->kaci",
                        &[
                            &eris.blk(Blk::Oovv, kk, kl, kd)?,
                            &t2.slice_leading(&[ki, kl, ka])?,
                        ],
                    )?)?;
                }
                w.set_leading(&[kk, ka, kc], &v)?;
            }
        }
    }
    Ok(w)
}

/// `W2ovvo` — `:268-279`. The `t1` half.
///
/// # Errors
/// As [`cc_foo`].
pub fn w2ovvo(t1: &T1, eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let ww = wooov(t1, eris)?;
    let mut w = ZArr::zeros(&[nk, nk, nk, nocc, nvir, nvir, nocc]);
    for kk in 0..nk {
        for ka in 0..nk {
            for kc in 0..nk {
                let ki = kconserv.get(kk, kc, ka) as usize;
                let mut v = einsum_scaled(
                    "la,lkic->kaci",
                    &[&t1.slice_leading(&[ka])?, &ww.slice_leading(&[ka, kk, ki])?],
                    -1.0,
                )?;
                v.add_assign(&einsum(
                    "akdc,id->kaci",
                    &[&eris.blk(Blk::Vovv, ka, kk, ki)?, &t1.slice_leading(&[ki])?],
                )?)?;
                w.set_leading(&[kk, ka, kc], &v)?;
            }
        }
    }
    Ok(w)
}

/// `Wovvo = W1ovvo + W2ovvo` — `:281-285`.
///
/// # Errors
/// As [`cc_foo`].
pub fn wovvo(t1: &T1, t2: &T2, eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let mut w = w1ovvo(t2, eris, kconserv)?;
    w.add_assign(&w2ovvo(t1, eris, kconserv)?)?;
    Ok(w)
}

/// `W1ovov` — `:287-301`.
///
/// # Errors
/// As [`cc_foo`].
pub fn w1ovov(t2: &T2, eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mut w = ZArr::zeros(&[nk, nk, nk, nocc, nvir, nocc, nvir]);
    for kk in 0..nk {
        for kb in 0..nk {
            for ki in 0..nk {
                let kd = kconserv.get(kk, ki, kb) as usize;
                let mut v = eris.blk(Blk::Ovov, kk, kb, ki)?;
                for kl in 0..nk {
                    let kc = kconserv.get(kk, kd, kl) as usize;
                    v.sub_assign(&einsum(
                        "klcd,ilcb->kbid",
                        &[
                            &eris.blk(Blk::Oovv, kk, kl, kc)?,
                            &t2.slice_leading(&[ki, kl, kc])?,
                        ],
                    )?)?;
                }
                w.set_leading(&[kk, kb, ki], &v)?;
            }
        }
    }
    Ok(w)
}

/// `W2ovov` — `:303-314`.
///
/// # Errors
/// As [`cc_foo`].
pub fn w2ovov(t1: &T1, eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let ww = wooov(t1, eris)?;
    let mut w = ZArr::zeros(&[nk, nk, nk, nocc, nvir, nocc, nvir]);
    for kk in 0..nk {
        for kb in 0..nk {
            for ki in 0..nk {
                let kd = kconserv.get(kk, ki, kb) as usize;
                let mut v = einsum_scaled(
                    "klid,lb->kbid",
                    &[&ww.slice_leading(&[kk, kb, ki])?, &t1.slice_leading(&[kb])?],
                    -1.0,
                )?;
                v.add_assign(&einsum(
                    "bkdc,ic->kbid",
                    &[&eris.blk(Blk::Vovv, kb, kk, kd)?, &t1.slice_leading(&[ki])?],
                )?)?;
                w.set_leading(&[kk, kb, ki], &v)?;
            }
        }
    }
    Ok(w)
}

/// `Wovov = W1ovov + W2ovov` — `:316-320`.
///
/// # Errors
/// As [`cc_foo`].
pub fn wovov(t1: &T1, t2: &T2, eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let mut w = w1ovov(t2, eris, kconserv)?;
    w.add_assign(&w2ovov(t1, eris, kconserv)?)?;
    Ok(w)
}

/// `Woooo` — `:322-337`. NOT [`cc_woooo`]: this one carries the `t1·t1`,
/// `ooov·t1` and full `oovv·t2` terms the ground-state version does not need.
///
/// # Errors
/// As [`cc_foo`].
pub fn eom_woooo(t1: &T1, t2: &T2, eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let _ = nvir;
    let mut w = ZArr::zeros(&[nk, nk, nk, nocc, nocc, nocc, nocc]);
    for kk in 0..nk {
        for kl in 0..nk {
            for ki in 0..nk {
                let kj = kconserv.get(kk, ki, kl) as usize;
                let mut v = einsum(
                    "klcd,ic,jd->klij",
                    &[
                        &eris.blk(Blk::Oovv, kk, kl, ki)?,
                        &t1.slice_leading(&[ki])?,
                        &t1.slice_leading(&[kj])?,
                    ],
                )?;
                v.add_assign(&einsum(
                    "klid,jd->klij",
                    &[&eris.blk(Blk::Ooov, kk, kl, ki)?, &t1.slice_leading(&[kj])?],
                )?)?;
                v.add_assign(&einsum(
                    "lkjc,ic->klij",
                    &[&eris.blk(Blk::Ooov, kl, kk, kj)?, &t1.slice_leading(&[ki])?],
                )?)?;
                v.add_assign(&eris.blk(Blk::Oooo, kk, kl, ki)?)?;
                for kc in 0..nk {
                    v.add_assign(&einsum(
                        "klcd,ijcd->klij",
                        &[
                            &eris.blk(Blk::Oovv, kk, kl, kc)?,
                            &t2.slice_leading(&[ki, kj, kc])?,
                        ],
                    )?)?;
                }
                w.set_leading(&[kk, kl, ki], &v)?;
            }
        }
    }
    Ok(w)
}

/// `get_Wvvvv(t1, t2, eris, kconserv, ka, kb, kc)` — `:348-369`, the
/// non-`Lpv` branch.
///
/// The `Lpv` branch (`:351-358`) builds `Wvvvv` on the fly from GDF's
/// three-index tensors. `crate::keris` does not carry `Lpv` — see
/// `crate::kueris`'s module doc for the same deferral on the unrestricted
/// side — so the `else` at `:360` is the only route here, and it produces the
/// same numbers.
///
/// # Errors
/// As [`cc_foo`].
pub fn get_wvvvv(
    t1: &T1,
    t2: &T2,
    eris: &KEris,
    kconserv: &Kconserv,
    ka: usize,
    kb: usize,
    kc: usize,
) -> Result<ZArr, PbcCcError> {
    let nk = eris.nkpts;
    let kd = kconserv.get(ka, kc, kb) as usize;
    let mut v = einsum(
        "klcd,ka,lb->abcd",
        &[
            &eris.blk(Blk::Oovv, ka, kb, kc)?,
            &t1.slice_leading(&[ka])?,
            &t1.slice_leading(&[kb])?,
        ],
    )?;
    v.sub_assign(&einsum(
        "alcd,lb->abcd",
        &[&eris.blk(Blk::Vovv, ka, kb, kc)?, &t1.slice_leading(&[kb])?],
    )?)?;
    v.sub_assign(&einsum(
        "bkdc,ka->abcd",
        &[&eris.blk(Blk::Vovv, kb, ka, kd)?, &t1.slice_leading(&[ka])?],
    )?)?;
    v.add_assign(&eris.blk(Blk::Vvvv, ka, kb, kc)?)?;
    for kk in 0..nk {
        let kl = kconserv.get(kc, kk, kd) as usize;
        v.add_assign(&einsum(
            "klcd,klab->abcd",
            &[
                &eris.blk(Blk::Oovv, kk, kl, kc)?,
                &t2.slice_leading(&[kk, kl, ka])?,
            ],
        )?)?;
    }
    Ok(v)
}

/// `Wvvvv` — `:339-346`, every `(ka, kb, kc)` of [`get_wvvvv`].
///
/// # Errors
/// As [`cc_foo`].
pub fn eom_wvvvv(t1: &T1, t2: &T2, eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nk, nvir) = (eris.nkpts, eris.nvir);
    let mut w = ZArr::zeros(&[nk, nk, nk, nvir, nvir, nvir, nvir]);
    for ka in 0..nk {
        for kb in 0..nk {
            for kc in 0..nk {
                w.set_leading(
                    &[ka, kb, kc],
                    &get_wvvvv(t1, t2, eris, kconserv, ka, kb, kc)?,
                )?;
            }
        }
    }
    Ok(w)
}

/// `Wvvvo` — `:371-419`.
///
/// `ww_vvvv` is the caller's `Wvvvv` when it has one (`_IMDS.make_ea` passes
/// its own at `eom_kccsd_rhf.py:1624`); `None` rebuilds it per k-triple, which
/// is what upstream does at `:414`.
///
/// # Errors
/// As [`cc_foo`].
pub fn wvvvo(
    t1: &T1,
    t2: &T2,
    eris: &KEris,
    kconserv: &Kconserv,
    ww_vvvv: Option<&ZArr>,
) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let w1ov = w1ovov(t2, eris, kconserv)?;
    let w1vo = w1ovvo(t2, eris, kconserv)?;
    let ffov = cc_fov(t1, t2, eris)?;
    let mut w = ZArr::zeros(&[nk, nk, nk, nvir, nvir, nvir, nocc]);

    for ka in 0..nk {
        for kb in 0..nk {
            for kc in 0..nk {
                let kj = kconserv.get(ka, kc, kb) as usize;
                // `:383` — `Wvovo[ka,kl,kc,kj]` IS `Wovov[kl,ka,kj,kc]`
                // transposed `(1,0,3,2)`; upstream states it as a comment
                // because the two have the same shape.
                let alcj = w1ov
                    .slice_leading(&[kb, ka, kj])?
                    .transpose(&[1, 0, 3, 2])?;
                let mut v =
                    einsum_scaled("alcj,lb->abcj", &[&alcj, &t1.slice_leading(&[kb])?], -1.0)?;
                v.sub_assign(&einsum(
                    "kbcj,ka->abcj",
                    &[
                        &w1vo.slice_leading(&[ka, kb, kc])?,
                        &t1.slice_leading(&[ka])?,
                    ],
                )?)?;
                // `:387` — `vovv[kc,kj,ka].transpose(2,3,0,1).conj()`
                v.add_assign(
                    &eris
                        .blk(Blk::Vovv, kc, kj, ka)?
                        .transpose(&[2, 3, 0, 1])?
                        .conj(),
                )?;

                for kl in 0..nk {
                    let kd = kconserv.get(ka, kc, kl) as usize;
                    let mut st2 = t2.slice_leading(&[kl, kj, kd])?;
                    st2.scale(2.0);
                    st2.sub_assign(&t2.slice_leading(&[kl, kj, kb])?.transpose(&[0, 1, 3, 2])?)?;
                    v.add_assign(&einsum(
                        "alcd,ljdb->abcj",
                        &[&eris.blk(Blk::Vovv, ka, kl, kc)?, &st2],
                    )?)?;
                    v.sub_assign(&einsum(
                        "aldc,ljdb->abcj",
                        &[
                            &eris.blk(Blk::Vovv, ka, kl, kd)?,
                            &t2.slice_leading(&[kl, kj, kd])?,
                        ],
                    )?)?;
                    let kd = kconserv.get(kb, kc, kl) as usize;
                    v.sub_assign(&einsum(
                        "bldc,jlda->abcj",
                        &[
                            &eris.blk(Blk::Vovv, kb, kl, kd)?,
                            &t2.slice_leading(&[kj, kl, kd])?,
                        ],
                    )?)?;
                    let kk = kconserv.get(kb, kl, ka) as usize;
                    v.add_assign(&einsum(
                        "lkjc,lkba->abcj",
                        &[
                            &eris.blk(Blk::Ooov, kl, kk, kj)?,
                            &t2.slice_leading(&[kl, kk, kb])?,
                        ],
                    )?)?;
                }
                v.add_assign(&einsum(
                    "lkjc,lb,ka->abcj",
                    &[
                        &eris.blk(Blk::Ooov, kb, ka, kj)?,
                        &t1.slice_leading(&[kb])?,
                        &t1.slice_leading(&[ka])?,
                    ],
                )?)?;
                v.sub_assign(&einsum(
                    "lc,ljab->abcj",
                    &[
                        &ffov.slice_leading(&[kc])?,
                        &t2.slice_leading(&[kc, kj, ka])?,
                    ],
                )?)?;
                w.set_leading(&[ka, kb, kc], &v)?;
            }
        }
    }

    // `:408-418` — the `Wvvvv·t1` term, skipped entirely when `t1` is zero.
    // That is upstream's own optimisation with its own comment ("don't make
    // vvvv if you can avoid it"), and it is a real one: `Wvvvv` is the phase's
    // largest tensor.
    let t1_is_zero =
        t1.data().re.iter().all(|v| *v == 0.0) && t1.data().im.iter().all(|v| *v == 0.0);
    if !t1_is_zero {
        for ka in 0..nk {
            for kb in 0..nk {
                for kc in 0..nk {
                    let kj = kconserv.get(ka, kc, kb) as usize;
                    let wv = match ww_vvvv {
                        Some(w4) => w4.slice_leading(&[ka, kb, kc])?,
                        None => get_wvvvv(t1, t2, eris, kconserv, ka, kb, kc)?,
                    };
                    let add = einsum("abcd,jd->abcj", &[&wv, &t1.slice_leading(&[kj])?])?;
                    let mut cur = w.slice_leading(&[ka, kb, kc])?;
                    cur.add_assign(&add)?;
                    w.set_leading(&[ka, kb, kc], &cur)?;
                }
            }
        }
    }
    Ok(w)
}

/// `Wovoo` — `:421-455`.
///
/// # Errors
/// As [`cc_foo`].
pub fn wovoo(t1: &T1, t2: &T2, eris: &KEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let w1ov = w1ovov(t2, eris, kconserv)?;
    let woooo_ = eom_woooo(t1, t2, eris, kconserv)?;
    let w1vo = w1ovvo(t2, eris, kconserv)?;
    let ffov = cc_fov(t1, t2, eris)?;
    let mut w = ZArr::zeros(&[nk, nk, nk, nocc, nvir, nocc, nocc]);

    for kk in 0..nk {
        for kb in 0..nk {
            for ki in 0..nk {
                let kj = kconserv.get(kk, ki, kb) as usize;
                let mut v = einsum(
                    "kbid,jd->kbij",
                    &[
                        &w1ov.slice_leading(&[kk, kb, ki])?,
                        &t1.slice_leading(&[kj])?,
                    ],
                )?;
                v.sub_assign(&einsum(
                    "klij,lb->kbij",
                    &[
                        &woooo_.slice_leading(&[kk, kb, ki])?,
                        &t1.slice_leading(&[kb])?,
                    ],
                )?)?;
                v.add_assign(&einsum(
                    "kbcj,ic->kbij",
                    &[
                        &w1vo.slice_leading(&[kk, kb, ki])?,
                        &t1.slice_leading(&[ki])?,
                    ],
                )?)?;
                // `:429` — `ooov[ki,kj,kk].transpose(2,3,0,1).conj()`
                v.add_assign(
                    &eris
                        .blk(Blk::Ooov, ki, kj, kk)?
                        .transpose(&[2, 3, 0, 1])?
                        .conj(),
                )?;

                for kd in 0..nk {
                    let kl = kconserv.get(ki, kk, kd) as usize;
                    let mut st2 = t2.slice_leading(&[kl, kj, kd])?;
                    st2.scale(2.0);
                    st2.sub_assign(&t2.slice_leading(&[kj, kl, kd])?.transpose(&[1, 0, 2, 3])?)?;
                    v.add_assign(&einsum(
                        "klid,ljdb->kbij",
                        &[&eris.blk(Blk::Ooov, kk, kl, ki)?, &st2],
                    )?)?;
                    v.sub_assign(&einsum(
                        "lkid,ljdb->kbij",
                        &[
                            &eris.blk(Blk::Ooov, kl, kk, ki)?,
                            &t2.slice_leading(&[kl, kj, kd])?,
                        ],
                    )?)?;
                    let kl = kconserv.get(kb, ki, kd) as usize;
                    v.sub_assign(&einsum(
                        "lkjd,libd->kbij",
                        &[
                            &eris.blk(Blk::Ooov, kl, kk, kj)?,
                            &t2.slice_leading(&[kl, ki, kb])?,
                        ],
                    )?)?;
                    v.add_assign(&einsum(
                        "bkdc,jidc->kbij",
                        &[
                            &eris.blk(Blk::Vovv, kb, kk, kd)?,
                            &t2.slice_leading(&[kj, ki, kd])?,
                        ],
                    )?)?;
                }
                v.add_assign(&einsum(
                    "bkdc,jd,ic->kbij",
                    &[
                        &eris.blk(Blk::Vovv, kb, kk, kj)?,
                        &t1.slice_leading(&[kj])?,
                        &t1.slice_leading(&[ki])?,
                    ],
                )?)?;
                v.add_assign(&einsum(
                    "kc,ijcb->kbij",
                    &[
                        &ffov.slice_leading(&[kk])?,
                        &t2.slice_leading(&[ki, kj, kk])?,
                    ],
                )?)?;
                w.set_leading(&[kk, kb, ki], &v)?;
            }
        }
    }
    Ok(w)
}

// ---------------------------------------------------------------------------
// T3[2] — the (T)a intermediates `EOMIP_Ta` / `EOMEA_Ta` are built on
// (`:465-655`)
// ---------------------------------------------------------------------------

/// The full spin-adapted `T3[2]` array, one `[nocc³, nvir³]` block per
/// `(ki,kj,kk,ka,kb)` — upstream's rank-11 `tmp_t3` (`:499`).
pub struct T3p2 {
    nkpts: usize,
    blocks: Vec<ZArr>,
}

impl T3p2 {
    fn flat(&self, k: [usize; 5]) -> usize {
        let n = self.nkpts;
        ((((k[0] * n + k[1]) * n + k[2]) * n + k[3]) * n) + k[4]
    }

    /// `tmp_t3[ki,kj,kk,ka,kb]`.
    #[must_use]
    pub fn get(&self, k: [usize; 5]) -> &ZArr {
        &self.blocks[self.flat(k)]
    }
}

/// What `get_t3p2_imds_slow` returns (`:655`).
pub struct T3p2Imds {
    /// `energy(pt1, pt2) − energy(t1, t2)`.
    pub delta_ccsd_energy: f64,
    /// The perturbatively corrected `t1`.
    pub pt1: ZArr,
    /// The perturbatively corrected `t2`.
    pub pt2: ZArr,
    /// `Wmcik` — the `[nkpts³, nocc, nvir, nocc, nocc]` addition to `Wovoo`.
    pub wovoo: ZArr,
    /// `Wacek` — the `[nkpts³, nvir, nvir, nvir, nocc]` addition to `Wvvvo`.
    pub wvvvo: ZArr,
}

/// `get_t3p2_imds_slow(cc, t1, t2, eris)` (`:465-655`) — the SPIN-ADAPTED
/// `T1[2]`/`T2[2]` corrections and the two `(T)a` intermediates.
///
/// # Why the `_slow` form and not `get_t3p2_imds`
///
/// Upstream has two: this loop-explicit one and a blocked `get_t3p2_imds`
/// (`:703-925`) that `_IMDS.make_t3p2_ip` actually calls. **They do not agree
/// to machine precision** — measured on diamond `gth-szv` `[1,1,2]` at the
/// pinned mesh, upstream's own two implementations differ by `1.3e-9` on
/// `pt1`, `6.1e-10` on `pt2` and `9.4e-10` on both `W` intermediates, against
/// magnitudes of `9.9e-3`, `0.106` and `6.2e-4`. So neither can be gated
/// against the other more tightly than that, and this port takes the
/// loop-explicit one for the same reason `16-08` took the loop-explicit `(T)`
/// first: it is the form a transcription can be checked against line by line.
/// The oracle emits both and reports the gap.
///
/// # Errors
/// Propagates every block access and shape check.
#[allow(clippy::too_many_lines)]
pub fn get_t3p2_imds_slow(
    t1: &ZArr,
    t2: &ZArr,
    eris: &KEris,
    kconserv: &Kconserv,
    padding: &Padding,
    lat: &KLattice<'_>,
) -> Result<T3p2Imds, PbcCcError> {
    let (nz_o, nz_v) = (&padding.occupied, &padding.virtuals);
    let (a_lat, kpts) = (lat.a, lat.kpts);
    let (nkpts, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mo_e_o: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[..nocc].to_vec()).collect();
    let mo_e_v: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[nocc..].to_vec()).collect();
    let ccsd_energy = crate::kccsd_rhf::energy(t1, t2, eris, kconserv)?;

    let kc_of = |ki: usize, kj: usize, kk: usize, ka: usize, kb: usize| -> usize {
        get_kconserv3(
            a_lat,
            kpts,
            &[
                KIdx::One(ki),
                KIdx::One(kj),
                KIdx::One(kk),
                KIdx::One(ka),
                KIdx::One(kb),
            ],
        )
        .data[0] as usize
    };

    /// `get_w(ki,kj,kk,ka,kb,kc)` (`:501-506`). `kc` is unused in the body,
    /// exactly as upstream leaves it.
    #[allow(clippy::too_many_arguments)]
    fn w(
        t2: &ZArr,
        eris: &KEris,
        kconserv: &Kconserv,
        ki: usize,
        kj: usize,
        kk: usize,
        ka: usize,
        kb: usize,
    ) -> Result<ZArr, PbcCcError> {
        let kf = kconserv.get(ka, ki, kb) as usize;
        let kc = kconserv.get(kk, kf, kj) as usize;
        let mut ret = einsum(
            "fiba,kjcf->ijkabc",
            &[
                &eris.blk(Blk::Vovv, kf, ki, kb)?.conj(),
                &t2.slice_leading(&[kk, kj, kc])?,
            ],
        )?;
        let km = kconserv.get(kc, kk, kb) as usize;
        ret.sub_assign(&einsum(
            "jima,mkbc->ijkabc",
            &[
                &eris.blk(Blk::Ooov, kj, ki, km)?.conj(),
                &t2.slice_leading(&[km, kk, kb])?,
            ],
        )?)?;
        Ok(ret)
    }

    // `:508-528` — six permutations, then the denominator.
    let n5 = nkpts.pow(5);
    let mut t3 = T3p2 {
        nkpts,
        blocks: vec![ZArr::zeros(&[0]); n5],
    };
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for kk in 0..nkpts {
                let eijk = crate::kccsd_t::epqr3(&mo_e_o, nz_o, ki, kj, kk, nocc, 1.0);
                for ka in 0..nkpts {
                    for kb in 0..nkpts {
                        let kc = kc_of(ki, kj, kk, ka, kb);
                        let mut blk = w(t2, eris, kconserv, ki, kj, kk, ka, kb)?;
                        blk.add_assign(
                            &w(t2, eris, kconserv, ki, kk, kj, ka, kc)?
                                .transpose(&[0, 2, 1, 3, 5, 4])?,
                        )?;
                        blk.add_assign(
                            &w(t2, eris, kconserv, kj, ki, kk, kb, ka)?
                                .transpose(&[1, 0, 2, 4, 3, 5])?,
                        )?;
                        blk.add_assign(
                            &w(t2, eris, kconserv, kj, kk, ki, kb, kc)?
                                .transpose(&[2, 0, 1, 5, 3, 4])?,
                        )?;
                        blk.add_assign(
                            &w(t2, eris, kconserv, kk, ki, kj, kc, ka)?
                                .transpose(&[1, 2, 0, 4, 5, 3])?,
                        )?;
                        blk.add_assign(
                            &w(t2, eris, kconserv, kk, kj, ki, kc, kb)?
                                .transpose(&[2, 1, 0, 5, 4, 3])?,
                        )?;
                        let eabc = crate::kccsd_t::epqr3(&mo_e_v, nz_v, ka, kb, kc, nvir, -1.0);
                        let mut f = 0;
                        for &e_ijk in &eijk {
                            for &e_abc in &eabc {
                                let d = e_ijk + e_abc;
                                blk.data_mut().re[f] /= d;
                                blk.data_mut().im[f] /= d;
                                f += 1;
                            }
                        }
                        let i = t3.flat([ki, kj, kk, ka, kb]);
                        t3.blocks[i] = blk;
                    }
                }
            }
        }
    }

    // `:530-538` — `pt1`, through the spin-adapted `2·oovv − oovvᵀ` and the
    // antisymmetrised `St3`.
    let mut pt1 = ZArr::zeros(&[nkpts, nocc, nvir]);
    for ki in 0..nkpts {
        let mut acc = ZArr::zeros(&[nocc, nvir]);
        for km in 0..nkpts {
            for kn in 0..nkpts {
                for ke in 0..nkpts {
                    let kf = kconserv.get(km, ke, kn) as usize;
                    let mut soovv = eris.blk(Blk::Oovv, km, kn, ke)?;
                    soovv.scale(2.0);
                    soovv
                        .sub_assign(&eris.blk(Blk::Oovv, km, kn, kf)?.transpose(&[0, 1, 3, 2])?)?;
                    let mut st3 = t3.get([ki, km, kn, ki, ke]).clone();
                    st3.sub_assign(
                        &t3.get([ki, km, kn, ke, ki])
                            .transpose(&[0, 1, 2, 4, 3, 5])?,
                    )?;
                    acc.add_assign(&einsum("mnef,imnaef->ia", &[&soovv, &st3])?)?;
                }
            }
        }
        pt1.set_leading(&[ki], &acc)?;
    }

    // `:540-608` — `pt2`.
    let mut pt2 = ZArr::zeros(&[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let mut acc = ZArr::zeros(&[nocc, nocc, nvir, nvir]);
                for km in 0..nkpts {
                    for kn in 0..nkpts {
                        // `:545-556` — `(ia,jb) -> (ia,jb)`.
                        let ke = kconserv.get(km, kj, kn) as usize;
                        acc.add_assign(&einsum_scaled(
                            "imnabe,mnje->ijab",
                            &[
                                t3.get([ki, km, kn, ka, kb]),
                                &eris.blk(Blk::Ooov, km, kn, kj)?,
                            ],
                            -2.0,
                        )?)?;
                        acc.add_assign(&einsum(
                            "imnabe,nmje->ijab",
                            &[
                                t3.get([ki, km, kn, ka, kb]),
                                &eris.blk(Blk::Ooov, kn, km, kj)?,
                            ],
                        )?)?;
                        acc.add_assign(&einsum(
                            "inmeab,mnje->ijab",
                            &[
                                t3.get([ki, kn, km, ke, ka]),
                                &eris.blk(Blk::Ooov, km, kn, kj)?,
                            ],
                        )?)?;
                        // `:558-566` — `(ia,jb) -> (jb,ia)`.
                        let ke = kconserv.get(km, ki, kn) as usize;
                        acc.add_assign(&einsum_scaled(
                            "jmnbae,mnie->ijab",
                            &[
                                t3.get([kj, km, kn, kb, ka]),
                                &eris.blk(Blk::Ooov, km, kn, ki)?,
                            ],
                            -2.0,
                        )?)?;
                        acc.add_assign(&einsum(
                            "jmnbae,nmie->ijab",
                            &[
                                t3.get([kj, km, kn, kb, ka]),
                                &eris.blk(Blk::Ooov, kn, km, ki)?,
                            ],
                        )?)?;
                        acc.add_assign(&einsum(
                            "jnmeba,mnie->ijab",
                            &[
                                t3.get([kj, kn, km, ke, kb]),
                                &eris.blk(Blk::Ooov, km, kn, ki)?,
                            ],
                        )?)?;
                    }

                    // `:568-582` — the `fov` terms. Note `t3[…,ka,km]` and
                    // `t3[…,kb,km]`: the LAST k-index is `km`, not a
                    // conserved one, which is upstream as written.
                    acc.add_assign(&einsum(
                        "ijmabe,me->ijab",
                        &[t3.get([ki, kj, km, ka, kb]), &eris.fov(km)?],
                    )?)?;
                    acc.sub_assign(&einsum(
                        "ijmaeb,me->ijab",
                        &[t3.get([ki, kj, km, ka, km]), &eris.fov(km)?],
                    )?)?;
                    acc.add_assign(&einsum(
                        "jimbae,me->ijab",
                        &[t3.get([kj, ki, km, kb, ka]), &eris.fov(km)?],
                    )?)?;
                    acc.sub_assign(&einsum(
                        "jimbea,me->ijab",
                        &[t3.get([kj, ki, km, kb, km]), &eris.fov(km)?],
                    )?)?;

                    for ke in 0..nkpts {
                        // `:584-596` — `(ia,jb) -> (ia,jb)`.
                        let kf = kconserv.get(km, ke, kb) as usize;
                        acc.add_assign(&einsum_scaled(
                            "ijmaef,bmef->ijab",
                            &[
                                t3.get([ki, kj, km, ka, ke]),
                                &eris.blk(Blk::Vovv, kb, km, ke)?,
                            ],
                            2.0,
                        )?)?;
                        acc.sub_assign(&einsum(
                            "ijmaef,bmfe->ijab",
                            &[
                                t3.get([ki, kj, km, ka, ke]),
                                &eris.blk(Blk::Vovv, kb, km, kf)?,
                            ],
                        )?)?;
                        acc.sub_assign(&einsum(
                            "imjfae,bmef->ijab",
                            &[
                                t3.get([ki, km, kj, kf, ka]),
                                &eris.blk(Blk::Vovv, kb, km, ke)?,
                            ],
                        )?)?;
                        // `:598-608` — `(ia,jb) -> (jb,ia)`.
                        let kf = kconserv.get(km, ke, ka) as usize;
                        acc.add_assign(&einsum_scaled(
                            "jimbef,amef->ijab",
                            &[
                                t3.get([kj, ki, km, kb, ke]),
                                &eris.blk(Blk::Vovv, ka, km, ke)?,
                            ],
                            2.0,
                        )?)?;
                        acc.sub_assign(&einsum(
                            "jimbef,amfe->ijab",
                            &[
                                t3.get([kj, ki, km, kb, ke]),
                                &eris.blk(Blk::Vovv, ka, km, kf)?,
                            ],
                        )?)?;
                        acc.sub_assign(&einsum(
                            "jmifbe,amef->ijab",
                            &[
                                t3.get([kj, km, ki, kf, kb]),
                                &eris.blk(Blk::Vovv, ka, km, ke)?,
                            ],
                        )?)?;
                    }
                }
                pt2.set_leading(&[ki, kj, ka], &acc)?;
            }
        }
    }

    // `:610-616` — `pt1 /= eia`, at `ka == ki`.
    for ki in 0..nkpts {
        let eia = crate::kccsd_t::epq_ov(&mo_e_o, nz_o, ki, &mo_e_v, nz_v, ki, nocc, nvir);
        let mut blk = pt1.slice_leading(&[ki])?;
        for (f, d) in eia.iter().enumerate() {
            blk.data_mut().re[f] /= d;
            blk.data_mut().im[f] /= d;
        }
        pt1.set_leading(&[ki], &blk)?;
    }

    // `:618-628` — `pt2 /= eijab`.
    for ki in 0..nkpts {
        for ka in 0..nkpts {
            let eia = crate::kccsd_t::epq_ov(&mo_e_o, nz_o, ki, &mo_e_v, nz_v, ka, nocc, nvir);
            for kj in 0..nkpts {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let ejb = crate::kccsd_t::epq_ov(&mo_e_o, nz_o, kj, &mo_e_v, nz_v, kb, nocc, nvir);
                let mut blk = pt2.slice_leading(&[ki, kj, ka])?;
                let mut f = 0;
                for i in 0..nocc {
                    for j in 0..nocc {
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let d = eia[i * nvir + a] + ejb[j * nvir + b];
                                blk.data_mut().re[f] /= d;
                                blk.data_mut().im[f] /= d;
                                f += 1;
                            }
                        }
                    }
                }
                pt2.set_leading(&[ki, kj, ka], &blk)?;
            }
        }
    }

    // `:630-631`
    pt1.add_assign(t1)?;
    pt2.add_assign(t2)?;

    // `:633-641` — `Wmcik`, the spin-adapted `2·ijk − jik − kji`.
    let mut wovoo = ZArr::zeros(&[nkpts, nkpts, nkpts, nocc, nvir, nocc, nocc]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for kk in 0..nkpts {
                for ka in 0..nkpts {
                    for kb in 0..nkpts {
                        let kc = kc_of(ki, kj, kk, ka, kb);
                        let km = kconserv.get(kc, ki, ka) as usize;
                        let oovv = eris.blk(Blk::Oovv, km, ki, kc)?;
                        let mut w = wovoo.slice_leading(&[km, kb, kk])?;
                        w.add_assign(&einsum_scaled(
                            "ijkabc,mica->mbkj",
                            &[t3.get([ki, kj, kk, ka, kb]), &oovv],
                            2.0,
                        )?)?;
                        w.sub_assign(&einsum(
                            "jikabc,mica->mbkj",
                            &[t3.get([kj, ki, kk, ka, kb]), &oovv],
                        )?)?;
                        w.sub_assign(&einsum(
                            "kjiabc,mica->mbkj",
                            &[t3.get([kk, kj, ki, ka, kb]), &oovv],
                        )?)?;
                        wovoo.set_leading(&[km, kb, kk], &w)?;
                    }
                }
            }
        }
    }

    // `:643-651` — `Wacek`.
    let mut wvvvo = ZArr::zeros(&[nkpts, nkpts, nkpts, nvir, nvir, nvir, nocc]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for kk in 0..nkpts {
                for ka in 0..nkpts {
                    for kb in 0..nkpts {
                        let kc = kc_of(ki, kj, kk, ka, kb);
                        let ke = kconserv.get(ki, ka, kk) as usize;
                        let oovv = eris.blk(Blk::Oovv, ki, kk, ka)?;
                        let mut w = wvvvo.slice_leading(&[kc, kb, ke])?;
                        w.add_assign(&einsum_scaled(
                            "ijkabc,ikae->cbej",
                            &[t3.get([ki, kj, kk, ka, kb]), &oovv],
                            -2.0,
                        )?)?;
                        w.add_assign(&einsum(
                            "jikabc,ikae->cbej",
                            &[t3.get([kj, ki, kk, ka, kb]), &oovv],
                        )?)?;
                        w.add_assign(&einsum(
                            "kjiabc,ikae->cbej",
                            &[t3.get([kk, kj, ki, ka, kb]), &oovv],
                        )?)?;
                        wvvvo.set_leading(&[kc, kb, ke], &w)?;
                    }
                }
            }
        }
    }

    let delta_ccsd_energy = crate::kccsd_rhf::energy(&pt1, &pt2, eris, kconserv)? - ccsd_energy;
    Ok(T3p2Imds {
        delta_ccsd_energy,
        pt1,
        pt2,
        wovoo,
        wvvvo,
    })
}

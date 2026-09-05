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

use pyscf_pbc_lib::Kconserv;
use pyscf_runtime::ZWorkspacePool;

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

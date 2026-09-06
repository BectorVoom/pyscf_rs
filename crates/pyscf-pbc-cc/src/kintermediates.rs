//! `kintermediates` — the SPIN-ORBITAL k-point CC intermediates
//! (plan 16-07; `pyscf/pbc/cc/kintermediates.py`, following
//! J. Gauss and J. F. Stanton, J. Chem. Phys. **103**, 3561 (1995), Table III).
//!
//! Everything here is in the ANTISYMMETRISED PHYSICIST convention `<pq||rs>`
//! that `crate::kccsd`'s ERI build produces — see
//! `pyscf_ccsd::gccsd::PhysicistsEris`' doc for why the convention is named
//! rather than assumed.
//!
//! The primitive at every site is the same as in
//! [`crate::kintermediates_rhf`]: an [`einsum`] subscript string transcribed
//! from the upstream line, hence UNCONJUGATED by construction, with the
//! conjugations appearing as explicit `.conj()` calls where upstream writes
//! them.

use pyscf_pbc_lib::Kconserv;

use crate::error::PbcCcError;
use crate::kccsd::{GBlk, KgEris};
use crate::zarr::{ZArr, einsum, einsum_scaled};

/// `make_tau(cc, t2, t1, t1p, kconserv, fac)` —
/// `kintermediates.py:42-60`.
///
/// The four `if` guards are momentum conditions, not optimisations: each term
/// contributes only when its two k-points coincide, and dropping one is a
/// different program.
///
/// # Errors
/// Shape violations only.
pub fn make_tau(
    t2: &ZArr,
    t1: &ZArr,
    t1p: &ZArr,
    kconserv: &Kconserv,
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    fac: f64,
) -> Result<ZArr, PbcCcError> {
    let mut tau = t2.clone();
    for ki in 0..nkpts {
        for ka in 0..nkpts {
            for kj in 0..nkpts {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let mut tmp = ZArr::zeros(&[nocc, nocc, nvir, nvir]);
                if ki == ka && kj == kb {
                    tmp.add_assign(&einsum(
                        "ia,jb->ijab",
                        &[&t1.slice_leading(&[ki])?, &t1p.slice_leading(&[kj])?],
                    )?)?;
                }
                if ki == kb && kj == ka {
                    tmp.sub_assign(&einsum(
                        "ib,ja->ijab",
                        &[&t1.slice_leading(&[ki])?, &t1p.slice_leading(&[kj])?],
                    )?)?;
                }
                if kj == ka && ki == kb {
                    tmp.sub_assign(&einsum(
                        "ja,ib->ijab",
                        &[&t1.slice_leading(&[kj])?, &t1p.slice_leading(&[ki])?],
                    )?)?;
                }
                if kj == kb && ki == ka {
                    tmp.add_assign(&einsum(
                        "jb,ia->ijab",
                        &[&t1.slice_leading(&[kj])?, &t1p.slice_leading(&[ki])?],
                    )?)?;
                }
                let mut blk = tau.slice_leading(&[ki, kj, ka])?;
                blk.zip_assign(&tmp, fac * 0.5)?;
                tau.set_leading(&[ki, kj, ka], &blk)?;
            }
        }
    }
    Ok(tau)
}

/// `eris_vovv = -eris.ovvv.transpose(1,0,2,4,3,5,6)` — `kintermediates.py:67`.
///
/// Note the transpose acts on the K-AXES (0 and 1 swap) as well as the orbital
/// axes (3 and 4 swap), because `<ov||vv>` and `<vo||vv>` differ by exchanging
/// the bra pair, which exchanges their k-points too.
///
/// # Errors
/// Propagates the ERI access.
pub fn eris_vovv(eris: &KgEris) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mut out = ZArr::zeros(&[nk, nk, nk, nvir, nocc, nvir, nvir]);
    for k0 in 0..nk {
        for k1 in 0..nk {
            for k2 in 0..nk {
                let mut b = eris.blk(GBlk::Ovvv, k1, k0, k2)?.transpose(&[1, 0, 2, 3])?;
                b.scale(-1.0);
                out.set_leading(&[k0, k1, k2], &b)?;
            }
        }
    }
    Ok(out)
}

/// `cc_Fvv` — `kintermediates.py:62-80`.
///
/// # Errors
/// Propagates the ERI access and every shape check.
pub fn cc_fvv(t1: &ZArr, t2: &ZArr, eris: &KgEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let vovv = eris_vovv(eris)?;
    let tau_tilde = make_tau(t2, t1, t1, kconserv, nk, nocc, nvir, 0.5)?;
    let mut fae = ZArr::zeros(&[nk, nvir, nvir]);
    for ka in 0..nk {
        let mut acc = eris.fvv(ka)?;
        acc.add_assign(&einsum_scaled(
            "me,ma->ae",
            &[&eris.fov(ka)?, &t1.slice_leading(&[ka])?],
            -0.5,
        )?)?;
        for km in 0..nk {
            acc.add_assign(&einsum(
                "mf,amef->ae",
                &[&t1.slice_leading(&[km])?, &vovv.slice_leading(&[ka, km, ka])?],
            )?)?;
            for kn in 0..nk {
                acc.add_assign(&einsum_scaled(
                    "mnaf,mnef->ae",
                    &[
                        &tau_tilde.slice_leading(&[km, kn, ka])?,
                        &eris.blk(GBlk::Oovv, km, kn, ka)?,
                    ],
                    -0.5,
                )?)?;
            }
        }
        fae.set_leading(&[ka], &acc)?;
    }
    Ok(fae)
}

/// `cc_Foo` — `kintermediates.py:82-96`.
///
/// # Errors
/// As [`cc_fvv`].
pub fn cc_foo(t1: &ZArr, t2: &ZArr, eris: &KgEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let tau_tilde = make_tau(t2, t1, t1, kconserv, nk, nocc, nvir, 0.5)?;
    let mut fmi = ZArr::zeros(&[nk, nocc, nocc]);
    for km in 0..nk {
        let mut acc = eris.foo(km)?;
        acc.add_assign(&einsum_scaled(
            "me,ie->mi",
            &[&eris.fov(km)?, &t1.slice_leading(&[km])?],
            0.5,
        )?)?;
        for kn in 0..nk {
            acc.add_assign(&einsum(
                "ne,mnie->mi",
                &[
                    &t1.slice_leading(&[kn])?,
                    &eris.blk(GBlk::Ooov, km, kn, km)?,
                ],
            )?)?;
            for ke in 0..nk {
                acc.add_assign(&einsum_scaled(
                    "inef,mnef->mi",
                    &[
                        &tau_tilde.slice_leading(&[km, kn, ke])?,
                        &eris.blk(GBlk::Oovv, km, kn, ke)?,
                    ],
                    0.5,
                )?)?;
            }
        }
        fmi.set_leading(&[km], &acc)?;
    }
    Ok(fmi)
}

/// `cc_Fov` — `kintermediates.py:98-106`.
///
/// # Errors
/// As [`cc_fvv`].
pub fn cc_fov(t1: &ZArr, eris: &KgEris) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mut fme = ZArr::zeros(&[nk, nocc, nvir]);
    for km in 0..nk {
        let mut acc = eris.fov(km)?;
        for kf in 0..nk {
            let kn = kf;
            acc.sub_assign(&einsum(
                "nf,mnfe->me",
                &[
                    &t1.slice_leading(&[kf])?,
                    &eris.blk(GBlk::Oovv, km, kn, kf)?,
                ],
            )?)?;
        }
        fme.set_leading(&[km], &acc)?;
    }
    Ok(fme)
}

/// `cc_Woooo` — `kintermediates.py:108-131`.
///
/// # Errors
/// As [`cc_fvv`].
pub fn cc_woooo(
    t1: &ZArr,
    t2: &ZArr,
    eris: &KgEris,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let tau = make_tau(t2, t1, t1, kconserv, nk, nocc, nvir, 1.0)?;
    let mut w = ZArr::zeros(&[nk, nk, nk, nocc, nocc, nocc, nocc]);
    for km in 0..nk {
        for kn in 0..nk {
            for ki in 0..nk {
                w.set_leading(&[km, kn, ki], &eris.blk(GBlk::Oooo, km, kn, ki)?)?;
            }
            // `:120-121` tmp = einsum('xje,ymnie->yxmnij', t1, ooov[km,kn]);
            //            tmp -= tmp.transpose(1,0,2,3,5,4)
            let ooov = eris.blk_free2(GBlk::Ooov, km, kn)?;
            let mut tmp = einsum("xje,ymnie->yxmnij", &[t1, &ooov])?;
            tmp.sub_assign(&tmp.clone().transpose(&[1, 0, 2, 3, 5, 4])?)?;

            // `:126` Wmnij[km,kn,:] += 0.25*einsum('yxijef,xmnef->ymnij', tau[kij], oovv[km,kn])
            // `kij` gathers tau at (ki, kj) for every ki, with kj = kconserv[km,ki,kn].
            let mut taukij = ZArr::zeros(&[nk, nk, nocc, nocc, nvir, nvir]);
            for ki in 0..nk {
                let kj = kconserv.get(km, ki, kn) as usize;
                for kx in 0..nk {
                    taukij.set_leading(&[ki, kx], &tau.slice_leading(&[ki, kj, kx])?)?;
                }
            }
            let oovv = eris.blk_free2(GBlk::Oovv, km, kn)?;
            let quarter = einsum_scaled("yxijef,xmnef->ymnij", &[&taukij, &oovv], 0.25)?;
            for ki in 0..nk {
                let kj = kconserv.get(km, ki, kn) as usize;
                let mut blk = w.slice_leading(&[km, kn, ki])?;
                blk.add_assign(&quarter.slice_leading(&[ki])?)?;
                blk.add_assign(&tmp.slice_leading(&[ki, kj])?)?;
                w.set_leading(&[km, kn, ki], &blk)?;
            }
        }
    }
    Ok(w)
}

/// `cc_Wvvvv` — `kintermediates.py:133-155`.
///
/// The phase's largest spin-orbital tensor: `nkpts³ · nvir⁴` with `nvir`
/// DOUBLED relative to the RHF case, i.e. **16×** the RHF `Wvvvv`
/// (`16-REVIEW.md §2.3`).
///
/// # Errors
/// As [`cc_fvv`].
pub fn cc_wvvvv(
    t1: &ZArr,
    t2: &ZArr,
    eris: &KgEris,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let vovv = eris_vovv(eris)?;
    let tau = make_tau(t2, t1, t1, kconserv, nk, nocc, nvir, 1.0)?;
    let mut w = ZArr::zeros(&[nk, nk, nk, nvir, nvir, nvir, nvir]);
    for ka in 0..nk {
        for kb in 0..nk {
            for ke in 0..nk {
                w.set_leading(&[ka, kb, ke], &eris.blk(GBlk::Vvvv, ka, kb, ke)?)?;
            }
            // `:144` — the `(km, kn)` pair list with kn = kconserv[ka,km,kb].
            let mut taumn = ZArr::zeros(&[nk, nocc, nocc, nvir, nvir]);
            let mut oovvmn = ZArr::zeros(&[nk, nk, nocc, nocc, nvir, nvir]);
            for km in 0..nk {
                let kn = kconserv.get(ka, km, kb) as usize;
                taumn.set_leading(&[km], &tau.slice_leading(&[km, kn, ka])?)?;
                for ky in 0..nk {
                    oovvmn.set_leading(&[km, ky], &eris.blk(GBlk::Oovv, km, kn, ky)?)?;
                }
            }
            let quarter = einsum_scaled("xmnab,xymnef->yabef", &[&taumn, &oovvmn], 0.25)?;
            for ke in 0..nk {
                let mut blk = w.slice_leading(&[ka, kb, ke])?;
                blk.add_assign(&quarter.slice_leading(&[ke])?)?;
                let mut tmp = einsum(
                    "mb,amef->abef",
                    &[
                        &t1.slice_leading(&[kb])?,
                        &vovv.slice_leading(&[ka, kb, ke])?,
                    ],
                )?;
                tmp.sub_assign(&einsum(
                    "ma,bmef->abef",
                    &[
                        &t1.slice_leading(&[ka])?,
                        &vovv.slice_leading(&[kb, ka, ke])?,
                    ],
                )?)?;
                blk.zip_assign(&tmp, -1.0)?;
                w.set_leading(&[ka, kb, ke], &blk)?;
            }
        }
    }
    Ok(w)
}

/// `eris_ovvo` and `eris_oovo` — `kintermediates.py:159-171`, also used
/// directly by `kccsd.py`'s `update_amps`.
///
/// # Errors
/// Propagates the ERI access.
pub fn eris_ovvo_oovo(eris: &KgEris, kconserv: &Kconserv) -> Result<(ZArr, ZArr), PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mut ovvo = ZArr::zeros(&[nk, nk, nk, nocc, nvir, nvir, nocc]);
    let mut oovo = ZArr::zeros(&[nk, nk, nk, nocc, nocc, nvir, nocc]);
    for km in 0..nk {
        for kb in 0..nk {
            for ke in 0..nk {
                let kj = kconserv.get(km, ke, kb) as usize;
                // <mb||je> -> -<mb||ej>
                let mut a = eris.blk(GBlk::Ovov, km, kb, kj)?.transpose(&[0, 1, 3, 2])?;
                a.scale(-1.0);
                ovvo.set_leading(&[km, kb, ke], &a)?;
                // <mn||je> -> -<mn||ej>, with kb standing in for kn
                let mut b = eris.blk(GBlk::Ooov, km, kb, kj)?.transpose(&[0, 1, 3, 2])?;
                b.scale(-1.0);
                oovo.set_leading(&[km, kb, ke], &b)?;
            }
        }
    }
    Ok((ovvo, oovo))
}

/// `cc_Wovvo` — `kintermediates.py:157-184`.
///
/// # Errors
/// As [`cc_fvv`].
pub fn cc_wovvo(
    t1: &ZArr,
    t2: &ZArr,
    eris: &KgEris,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let (ovvo, oovo) = eris_ovvo_oovo(eris, kconserv)?;
    let mut w = ovvo.clone();
    for km in 0..nk {
        for kb in 0..nk {
            for ke in 0..nk {
                let kj = kconserv.get(km, ke, kb) as usize;
                let mut blk = w.slice_leading(&[km, kb, ke])?;
                blk.add_assign(&einsum(
                    "jf,mbef->mbej",
                    &[
                        &t1.slice_leading(&[kj])?,
                        &eris.blk(GBlk::Ovvv, km, kb, ke)?,
                    ],
                )?)?;
                blk.sub_assign(&einsum(
                    "nb,mnej->mbej",
                    &[
                        &t1.slice_leading(&[kb])?,
                        &oovo.slice_leading(&[km, kb, ke])?,
                    ],
                )?)?;
                // `:177-182` — the k-batched `temp` over `kn`.
                let mut temp = ZArr::zeros(&[nk, nocc, nocc, nvir, nvir]);
                for kn in 0..nk {
                    let kf = kconserv.get(km, ke, kn) as usize;
                    let mut t = t2.slice_leading(&[kj, kn, kf])?;
                    t.scale(-0.5);
                    if kn == kb && kf == kj {
                        t.sub_assign(&einsum(
                            "jf,nb->jnfb",
                            &[&t1.slice_leading(&[kj])?, &t1.slice_leading(&[kn])?],
                        )?)?;
                    }
                    temp.set_leading(&[kn], &t)?;
                }
                // `:183` `eris.oovv[km,:,ke]` — the free index is the
                // MIDDLE one. `blk_free1` produces the same SHAPE from
                // `oovv[:,km,ke]`, so getting this wrong is a plausible wrong
                // number that no shape check catches; it cost one debug cycle
                // here and is why the two accessors are named apart.
                let oovv = eris.blk_free_mid(GBlk::Oovv, km, ke)?;
                blk.add_assign(&einsum("xjnfb,xmnef->mbej", &[&temp, &oovv])?)?;
                w.set_leading(&[km, kb, ke], &blk)?;
            }
        }
    }
    Ok(w)
}

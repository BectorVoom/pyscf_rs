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

use pyscf_pbc_lib::{KIdx, Kconserv, get_kconserv3};

use crate::eom_kccsd_ghf::{KLattice, Padding};
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
pub fn cc_fvv(
    t1: &ZArr,
    t2: &ZArr,
    eris: &KgEris,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
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
                &[
                    &t1.slice_leading(&[km])?,
                    &vovv.slice_leading(&[ka, km, ka])?,
                ],
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
pub fn cc_foo(
    t1: &ZArr,
    t2: &ZArr,
    eris: &KgEris,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
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

// ---------------------------------------------------------------------------
// The EOM intermediates (`kintermediates.py:206-353`, plan 16-09 Task 1).
//
// These are NOT the `cc_*` ones with a different name: each starts from its
// `cc_*` sibling and adds the terms the ground-state equations do not need
// because they cancel there. `_IMDS._make_shared` (`eom_kccsd_ghf.py:1863`)
// builds `Foo`/`Fvv`/`Fov`/`Wovvo`; `make_ip` adds `Woooo`/`Wooov`/`Wovoo`;
// `make_ea` adds `Woooo`/`Wvovv`/`Wvvvv`/`Wvvvo`.
// ---------------------------------------------------------------------------

/// `Fvv` — `kintermediates.py:206-212`. `cc_Fvv` minus a half `t1·Fov`.
///
/// # Errors
/// As [`cc_fvv`].
pub fn fvv(t1: &ZArr, t2: &ZArr, eris: &KgEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let cc_fov_ = cc_fov(t1, eris)?;
    let mut fae = cc_fvv(t1, t2, eris, kconserv)?;
    for km in 0..eris.nkpts {
        let v = einsum_scaled(
            "ma,me->ae",
            &[&t1.slice_leading(&[km])?, &cc_fov_.slice_leading(&[km])?],
            0.5,
        )?;
        let mut blk = fae.slice_leading(&[km])?;
        blk.sub_assign(&v)?;
        fae.set_leading(&[km], &blk)?;
    }
    Ok(fae)
}

/// `Foo` — `:214-220`. `cc_Foo` PLUS a half `t1·Fov` (the sign differs from
/// [`fvv`]'s, which is why both are written out).
///
/// # Errors
/// As [`cc_fvv`].
pub fn foo(t1: &ZArr, t2: &ZArr, eris: &KgEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let cc_fov_ = cc_fov(t1, eris)?;
    let mut fmi = cc_foo(t1, t2, eris, kconserv)?;
    for km in 0..eris.nkpts {
        let v = einsum_scaled(
            "ie,me->mi",
            &[&t1.slice_leading(&[km])?, &cc_fov_.slice_leading(&[km])?],
            0.5,
        )?;
        let mut blk = fmi.slice_leading(&[km])?;
        blk.add_assign(&v)?;
        fmi.set_leading(&[km], &blk)?;
    }
    Ok(fmi)
}

/// `Fov` — `:222-225`. Identical to `cc_Fov`; upstream keeps the alias so the
/// EOM code reads uniformly, and so does this port.
///
/// # Errors
/// As [`cc_fov`].
pub fn fov(t1: &ZArr, eris: &KgEris) -> Result<ZArr, PbcCcError> {
    cc_fov(t1, eris)
}

/// `Woooo` — `:227-237`. `cc_Woooo` plus `0.25 tau·oovv`.
///
/// # Errors
/// As [`cc_woooo`].
pub fn woooo(t1: &ZArr, t2: &ZArr, eris: &KgEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let tau = make_tau(t2, t1, t1, kconserv, nk, nocc, nvir, 1.0)?;
    let mut w = cc_woooo(t1, t2, eris, kconserv)?;
    for km in 0..nk {
        for kn in 0..nk {
            for ki in 0..nk {
                let kj = kconserv.get(km, ki, kn) as usize;
                let mut blk = w.slice_leading(&[km, kn, ki])?;
                for kx in 0..nk {
                    blk.add_assign(&einsum_scaled(
                        "ijef,mnef->mnij",
                        &[
                            &tau.slice_leading(&[ki, kj, kx])?,
                            &eris.blk(GBlk::Oovv, km, kn, kx)?,
                        ],
                        0.25,
                    )?)?;
                }
                w.set_leading(&[km, kn, ki], &blk)?;
            }
        }
    }
    Ok(w)
}

/// `Wvvvv` — `:239-249`. `cc_Wvvvv` plus `0.25 tau·oovv`.
///
/// The phase's largest tensor; see [`cc_wvvvv`].
///
/// # Errors
/// As [`cc_wvvvv`].
pub fn wvvvv(t1: &ZArr, t2: &ZArr, eris: &KgEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let tau = make_tau(t2, t1, t1, kconserv, nk, nocc, nvir, 1.0)?;
    let mut w = cc_wvvvv(t1, t2, eris, kconserv)?;
    for ka in 0..nk {
        for kb in 0..nk {
            for ke in 0..nk {
                let mut blk = w.slice_leading(&[ka, kb, ke])?;
                for km in 0..nk {
                    // `:245` — `kn` is NOT a free index: it is fixed by
                    // `kconserv[ka, km, kb]`.
                    let kn = kconserv.get(ka, km, kb) as usize;
                    blk.add_assign(&einsum_scaled(
                        "mnab,mnef->abef",
                        &[
                            &tau.slice_leading(&[km, kn, ka])?,
                            &eris.blk(GBlk::Oovv, km, kn, ke)?,
                        ],
                        0.25,
                    )?)?;
                }
                w.set_leading(&[ka, kb, ke], &blk)?;
            }
        }
    }
    Ok(w)
}

/// `Wovvo` — `:251-262`. `cc_Wovvo` minus `0.5 t2·oovv`.
///
/// # Errors
/// As [`cc_wovvo`].
pub fn wovvo(t1: &ZArr, t2: &ZArr, eris: &KgEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let nk = eris.nkpts;
    let mut w = cc_wovvo(t1, t2, eris, kconserv)?;
    for km in 0..nk {
        for kb in 0..nk {
            for ke in 0..nk {
                let kj = kconserv.get(km, ke, kb) as usize;
                let mut blk = w.slice_leading(&[km, kb, ke])?;
                for kn in 0..nk {
                    let kf = kconserv.get(km, ke, kn) as usize;
                    blk.sub_assign(&einsum_scaled(
                        "jnfb,mnef->mbej",
                        &[
                            &t2.slice_leading(&[kj, kn, kf])?,
                            &eris.blk(GBlk::Oovv, km, kn, ke)?,
                        ],
                        0.5,
                    )?)?;
                }
                w.set_leading(&[km, kb, ke], &blk)?;
            }
        }
    }
    Ok(w)
}

/// `Wooov` — `:266-272`. `eris.ooov` plus one `t1·oovv`.
///
/// # Errors
/// Propagates the ERI access.
pub fn wooov(t1: &ZArr, eris: &KgEris) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mut w = ZArr::zeros(&[nk, nk, nk, nocc, nocc, nocc, nvir]);
    for km in 0..nk {
        for kn in 0..nk {
            for ki in 0..nk {
                // `:270` — `kf = ki`, stated as an assignment upstream and
                // kept as one here rather than folded into the index.
                let kf = ki;
                let mut blk = eris.blk(GBlk::Ooov, km, kn, ki)?;
                blk.add_assign(&einsum(
                    "if,mnfe->mnie",
                    &[
                        &t1.slice_leading(&[ki])?,
                        &eris.blk(GBlk::Oovv, km, kn, kf)?,
                    ],
                )?)?;
                w.set_leading(&[km, kn, ki], &blk)?;
            }
        }
    }
    Ok(w)
}

/// `Wvovv` — `:274-281`. `-ovvv.transpose(1,0,2,3)` minus one `t1·oovv`.
///
/// # Errors
/// Propagates the ERI access.
pub fn wvovv(t1: &ZArr, eris: &KgEris) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mut w = ZArr::zeros(&[nk, nk, nk, nvir, nocc, nvir, nvir]);
    for ka in 0..nk {
        for km in 0..nk {
            for ke in 0..nk {
                let kn = ka;
                let mut blk = eris.blk(GBlk::Ovvv, km, ka, ke)?.transpose(&[1, 0, 2, 3])?;
                blk.scale(-1.0);
                blk.sub_assign(&einsum(
                    "na,nmef->amef",
                    &[
                        &t1.slice_leading(&[kn])?,
                        &eris.blk(GBlk::Oovv, kn, km, ke)?,
                    ],
                )?)?;
                w.set_leading(&[ka, km, ke], &blk)?;
            }
        }
    }
    Ok(w)
}

/// `Wovoo` — `:283-315`.
///
/// The `P(ij)` antisymmetrisation is written out term by term upstream rather
/// than as a transpose, and it is transcribed the same way here: the two halves
/// contract DIFFERENT k-blocks (`ooov[km,kn,ki]` against `ooov[km,kn,kj]`), so
/// a transpose of the assembled result is not the same program.
///
/// # Errors
/// As [`woooo`].
pub fn wovoo(t1: &ZArr, t2: &ZArr, eris: &KgEris, kconserv: &Kconserv) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mut w = ZArr::zeros(&[nk, nk, nk, nocc, nvir, nocc, nocc]);
    for km in 0..nk {
        for kb in 0..nk {
            for ki in 0..nk {
                let kj = kconserv.get(km, ki, kb) as usize;
                let mut blk = eris.blk(GBlk::Ovoo, km, kb, ki)?;
                for kn in 0..nk {
                    blk.add_assign(&einsum(
                        "mnie,jnbe->mbij",
                        &[
                            &eris.blk(GBlk::Ooov, km, kn, ki)?,
                            &t2.slice_leading(&[kj, kn, kb])?,
                        ],
                    )?)?;
                }
                // `:292` — `mbej` is `-ovov[km,kb,kj].transpose(0,1,3,2)`,
                // built inline upstream.
                let mut ovvo = eris.blk(GBlk::Ovov, km, kb, kj)?.transpose(&[0, 1, 3, 2])?;
                ovvo.scale(-1.0);
                blk.add_assign(&einsum(
                    "ie,mbej->mbij",
                    &[&t1.slice_leading(&[ki])?, &ovvo],
                )?)?;
                for kf in 0..nk {
                    let kn = kconserv.get(kb, kj, kf) as usize;
                    blk.sub_assign(&einsum(
                        "ie,njbf,mnef->mbij",
                        &[
                            &t1.slice_leading(&[ki])?,
                            &t2.slice_leading(&[kn, kj, kb])?,
                            &eris.blk(GBlk::Oovv, km, kn, ki)?,
                        ],
                    )?)?;
                }
                // P(ij)
                for kn in 0..nk {
                    blk.sub_assign(&einsum(
                        "mnje,inbe->mbij",
                        &[
                            &eris.blk(GBlk::Ooov, km, kn, kj)?,
                            &t2.slice_leading(&[ki, kn, kb])?,
                        ],
                    )?)?;
                }
                let mut ovvo = eris.blk(GBlk::Ovov, km, kb, ki)?.transpose(&[0, 1, 3, 2])?;
                ovvo.scale(-1.0);
                blk.sub_assign(&einsum(
                    "je,mbei->mbij",
                    &[&t1.slice_leading(&[kj])?, &ovvo],
                )?)?;
                for kf in 0..nk {
                    let kn = kconserv.get(kb, ki, kf) as usize;
                    blk.add_assign(&einsum(
                        "je,nibf,mnef->mbij",
                        &[
                            &t1.slice_leading(&[kj])?,
                            &t2.slice_leading(&[kn, ki, kb])?,
                            &eris.blk(GBlk::Oovv, km, kn, kj)?,
                        ],
                    )?)?;
                }
                w.set_leading(&[km, kb, ki], &blk)?;
            }
        }
    }

    // `:308-315` — a SECOND pass, because `Fov`, `Woooo` and `tau` are built
    // after the first loop upstream.
    let ffov = fov(t1, eris)?;
    let wwoooo = woooo(t1, t2, eris, kconserv)?;
    let tau = make_tau(t2, t1, t1, kconserv, nk, nocc, nvir, 1.0)?;
    for km in 0..nk {
        for kb in 0..nk {
            for ki in 0..nk {
                let kj = kconserv.get(km, ki, kb) as usize;
                let mut blk = w.slice_leading(&[km, kb, ki])?;
                blk.sub_assign(&einsum(
                    "me,ijbe->mbij",
                    &[
                        &ffov.slice_leading(&[km])?,
                        &t2.slice_leading(&[ki, kj, kb])?,
                    ],
                )?)?;
                blk.sub_assign(&einsum(
                    "nb,mnij->mbij",
                    &[
                        &t1.slice_leading(&[kb])?,
                        &wwoooo.slice_leading(&[km, kb, ki])?,
                    ],
                )?)?;
                for kx in 0..nk {
                    blk.add_assign(&einsum_scaled(
                        "mbef,ijef->mbij",
                        &[
                            &eris.blk(GBlk::Ovvv, km, kb, kx)?,
                            &tau.slice_leading(&[ki, kj, kx])?,
                        ],
                        0.5,
                    )?)?;
                }
                w.set_leading(&[km, kb, ki], &blk)?;
            }
        }
    }
    Ok(w)
}

/// `Wvvvo` — `:317-353`.
///
/// `ww_vvvv` is `Wvvvv` when the caller already has it (`_IMDS.make_ee` passes
/// its own at `eom_kccsd_ghf.py:1966`); `None` builds it.
///
/// # Errors
/// As [`wvvvv`].
pub fn wvvvo(
    t1: &ZArr,
    t2: &ZArr,
    eris: &KgEris,
    kconserv: &Kconserv,
    ww_vvvv: Option<&ZArr>,
) -> Result<ZArr, PbcCcError> {
    let (nk, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let ffov = fov(t1, eris)?;
    let owned;
    let wwvvvv = match ww_vvvv {
        Some(w) => w,
        None => {
            owned = wvvvv(t1, t2, eris, kconserv)?;
            &owned
        }
    };

    // `:324-330` — `eris_ovvo[km,kb,ke] = -ovov[km,kb,kj].transpose(0,1,3,2)`.
    let mut eris_ovvo = ZArr::zeros(&[nk, nk, nk, nocc, nvir, nvir, nocc]);
    for km in 0..nk {
        for kb in 0..nk {
            for ke in 0..nk {
                let kj = kconserv.get(km, ke, kb) as usize;
                let mut v = eris.blk(GBlk::Ovov, km, kb, kj)?.transpose(&[0, 1, 3, 2])?;
                v.scale(-1.0);
                eris_ovvo.set_leading(&[km, kb, ke], &v)?;
            }
        }
    }

    let mut tmp1 = ZArr::zeros(&[nk, nk, nk, nvir, nvir, nvir, nocc]);
    let mut tmp2 = ZArr::zeros(&[nk, nk, nk, nvir, nvir, nvir, nocc]);
    for ka in 0..nk {
        for kb in 0..nk {
            for ke in 0..nk {
                let ki = kconserv.get(ka, ke, kb) as usize;
                let mut a2 = einsum(
                    "ma,mbei->abei",
                    &[
                        &t1.slice_leading(&[ka])?,
                        &eris_ovvo.slice_leading(&[ka, kb, ke])?,
                    ],
                )?;
                for kn in 0..nk {
                    a2.sub_assign(&einsum(
                        "ma,nibf,mnef->abei",
                        &[
                            &t1.slice_leading(&[ka])?,
                            &t2.slice_leading(&[kn, ki, kb])?,
                            &eris.blk(GBlk::Oovv, ka, kn, ke)?,
                        ],
                    )?)?;
                }
                tmp2.set_leading(&[ka, kb, ke], &a2)?;

                let mut a1 = ZArr::zeros(&[nvir, nvir, nvir, nocc]);
                for km in 0..nk {
                    a1.add_assign(&einsum(
                        "mbef,miaf->abei",
                        &[
                            &eris.blk(GBlk::Ovvv, km, kb, ke)?,
                            &t2.slice_leading(&[km, ki, ka])?,
                        ],
                    )?)?;
                }
                tmp1.set_leading(&[ka, kb, ke], &a1)?;
            }
        }
    }
    let tau = make_tau(t2, t1, t1, kconserv, nk, nocc, nvir, 1.0)?;

    // `:344-345` — `Wabei = -tmp1 + tmp1.transpose(1,0,2,4,3,5,6)`
    //                      - tmp2 + tmp2.transpose(1,0,2,4,3,5,6)
    // i.e. at `[ka,kb,ke]` the mirrored term reads `[kb,ka,ke]` and swaps the
    // two virtual axes.
    let mut w = ZArr::zeros(&[nk, nk, nk, nvir, nvir, nvir, nocc]);
    for ka in 0..nk {
        for kb in 0..nk {
            for ke in 0..nk {
                let mut blk = tmp1.slice_leading(&[ka, kb, ke])?;
                blk.scale(-1.0);
                blk.add_assign(
                    &tmp1
                        .slice_leading(&[kb, ka, ke])?
                        .transpose(&[1, 0, 2, 3])?,
                )?;
                blk.sub_assign(&tmp2.slice_leading(&[ka, kb, ke])?)?;
                blk.add_assign(
                    &tmp2
                        .slice_leading(&[kb, ka, ke])?
                        .transpose(&[1, 0, 2, 3])?,
                )?;
                w.set_leading(&[ka, kb, ke], &blk)?;
            }
        }
    }

    for ka in 0..nk {
        for kb in 0..nk {
            for ke in 0..nk {
                let ki = kconserv.get(ka, ke, kb) as usize;
                let mut blk = w.slice_leading(&[ka, kb, ke])?;
                // `:348` — `ovvv[ki,ke,kb].conj().transpose(3,2,1,0)`
                blk.add_assign(
                    &eris
                        .blk(GBlk::Ovvv, ki, ke, kb)?
                        .conj()
                        .transpose(&[3, 2, 1, 0])?,
                )?;
                blk.add_assign(&einsum(
                    "if,abef->abei",
                    &[
                        &t1.slice_leading(&[ki])?,
                        &wwvvvv.slice_leading(&[ka, kb, ke])?,
                    ],
                )?)?;
                blk.sub_assign(&einsum(
                    "me,miab->abei",
                    &[
                        &ffov.slice_leading(&[ke])?,
                        &t2.slice_leading(&[ke, ki, ka])?,
                    ],
                )?)?;
                for km in 0..nk {
                    let kn = kconserv.get(ka, km, kb) as usize;
                    blk.add_assign(&einsum_scaled(
                        "nmie,mnab->abei",
                        &[
                            &eris.blk(GBlk::Ooov, kn, km, ki)?,
                            &tau.slice_leading(&[km, kn, ka])?,
                        ],
                        0.5,
                    )?)?;
                }
                w.set_leading(&[ka, kb, ke], &blk)?;
            }
        }
    }
    Ok(w)
}

// ---------------------------------------------------------------------------
// T3[2] — the (T)a intermediates `EOMIP_Ta` / `EOMEA_Ta` are built on
// (`:355-414`, `:416-529`)
// ---------------------------------------------------------------------------

/// The full `T3[2]` array, one `[nocc³, nvir³]` block per `(ki,kj,kk,ka,kb)`.
///
/// Upstream holds it as a single rank-11 `ndarray` (`:385`), which is what its
/// own docstring calls "the entire T3[2] array in memory" and what the
/// `_slow` in [`get_t3p2_imds_slow`] refers to. This keeps the same total —
/// `nkpts⁵·nocc³·nvir³` complex — as `nkpts⁵` separate blocks, so the
/// `P(ijk)` step is a per-block permutation rather than a rank-11 one, and
/// the k-axes are addressed by arithmetic instead of by striding.
pub struct T3p2 {
    nkpts: usize,
    blocks: Vec<ZArr>,
}

impl T3p2 {
    fn flat(&self, k: [usize; 5]) -> usize {
        let n = self.nkpts;
        ((((k[0] * n + k[1]) * n + k[2]) * n + k[3]) * n) + k[4]
    }

    /// `t3[ki,kj,kk,ka,kb]`, the `[nocc,nocc,nocc,nvir,nvir,nvir]` block.
    #[must_use]
    pub fn get(&self, k: [usize; 5]) -> &ZArr {
        &self.blocks[self.flat(k)]
    }

    fn set(&mut self, k: [usize; 5], v: ZArr) {
        let i = self.flat(k);
        self.blocks[i] = v;
    }
}

/// `get_full_t3p2(mycc, t1, t2, eris)` (`:355-414`).
///
/// # `t1` is not read
///
/// Upstream's signature takes it and its body never uses it — `get_wijkabc`
/// (`:364-370`) is built from `t2` and the eris alone. It is dropped from this
/// signature rather than accepted and ignored.
///
/// # Errors
/// Propagates every block access and shape check.
pub fn get_full_t3p2(
    t2: &ZArr,
    eris: &KgEris,
    kconserv: &Kconserv,
    padding: &Padding,
    lat: &KLattice<'_>,
) -> Result<T3p2, PbcCcError> {
    let (nz_o, nz_v) = (&padding.occupied, &padding.virtuals);
    let (a_lat, kpts) = (lat.a, lat.kpts);
    let (nkpts, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mo_e_o: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[..nocc].to_vec()).collect();
    let mo_e_v: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[nocc..].to_vec()).collect();

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

    /// `get_wijkabc` (`:364-370`).
    #[allow(clippy::too_many_arguments)]
    fn w(
        t2: &ZArr,
        eris: &KgEris,
        kconserv: &Kconserv,
        ki: usize,
        kj: usize,
        kk: usize,
        ka: usize,
        kb: usize,
        kc: usize,
    ) -> Result<ZArr, PbcCcError> {
        let km = kconserv.get(kc, kk, kb) as usize;
        let kf = kconserv.get(kk, kc, kj) as usize;
        let mut ret = einsum(
            "kjcf,ifab->ijkabc",
            &[
                &t2.slice_leading(&[kk, kj, kc])?,
                &eris.blk(GBlk::Ovvv, ki, kf, ka)?.conj(),
            ],
        )?;
        ret.sub_assign(&einsum(
            "jima,mkbc->ijkabc",
            &[
                &eris.blk(GBlk::Ooov, kj, ki, km)?.conj(),
                &t2.slice_leading(&[km, kk, kb])?,
            ],
        )?)?;
        Ok(ret)
    }

    let n5 = nkpts.pow(5);
    let mut raw = T3p2 {
        nkpts,
        blocks: vec![ZArr::zeros(&[0]); n5],
    };
    // `:387-393` — `P(abc)` on each block.
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for kk in 0..nkpts {
                for ka in 0..nkpts {
                    for kb in 0..nkpts {
                        let kc = kc_of(ki, kj, kk, ka, kb);
                        let mut blk = w(t2, eris, kconserv, ki, kj, kk, ka, kb, kc)?;
                        blk.add_assign(
                            &w(t2, eris, kconserv, ki, kj, kk, kb, kc, ka)?
                                .transpose(&[0, 1, 2, 5, 3, 4])?,
                        )?;
                        blk.add_assign(
                            &w(t2, eris, kconserv, ki, kj, kk, kc, ka, kb)?
                                .transpose(&[0, 1, 2, 4, 5, 3])?,
                        )?;
                        raw.set([ki, kj, kk, ka, kb], blk);
                    }
                }
            }
        }
    }

    // `:396-398` — `P(ijk)` over the WHOLE array, k-axes included. Written out
    // per block: upstream's `transpose(1,2,0,3,4,6,7,5,8,9,10)` sends
    // `[ki,kj,kk,ka,kb][i,j,k,…]` to `[kk,ki,kj,ka,kb][k,i,j,…]`, and
    // `transpose(2,0,1,3,4,7,5,6,8,9,10)` to `[kj,kk,ki,ka,kb][j,k,i,…]`.
    let mut t3 = T3p2 {
        nkpts,
        blocks: vec![ZArr::zeros(&[0]); n5],
    };
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for kk in 0..nkpts {
                for ka in 0..nkpts {
                    for kb in 0..nkpts {
                        let mut blk = raw.get([ki, kj, kk, ka, kb]).clone();
                        blk.add_assign(
                            &raw.get([kk, ki, kj, ka, kb])
                                .transpose(&[1, 2, 0, 3, 4, 5])?,
                        )?;
                        blk.add_assign(
                            &raw.get([kj, kk, ki, ka, kb])
                                .transpose(&[2, 0, 1, 3, 4, 5])?,
                        )?;
                        t3.set([ki, kj, kk, ka, kb], blk);
                    }
                }
            }
        }
    }
    drop(raw);

    // `:400-412` — the denominator.
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for kk in 0..nkpts {
                let eijk = crate::kccsd_t::epqr3(&mo_e_o, nz_o, ki, kj, kk, nocc, 1.0);
                for ka in 0..nkpts {
                    for kb in 0..nkpts {
                        let kc = kc_of(ki, kj, kk, ka, kb);
                        let eabc = crate::kccsd_t::epqr3(&mo_e_v, nz_v, ka, kb, kc, nvir, -1.0);
                        let i = t3.flat([ki, kj, kk, ka, kb]);
                        let blk = &mut t3.blocks[i];
                        let mut f = 0;
                        for &e_ijk in &eijk {
                            for &e_abc in &eabc {
                                let d = e_ijk + e_abc;
                                blk.data_mut().re[f] /= d;
                                blk.data_mut().im[f] /= d;
                                f += 1;
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(t3)
}

/// What `get_t3p2_imds_slow` returns (`:529`).
pub struct T3p2Imds {
    /// `energy(pt1, pt2) − energy(t1, t2)`, upstream's `delta_ccsd_energy`.
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

/// `get_t3p2_imds_slow(cc, t1, t2, eris)` (`:416-529`) — the `T1[2]`/`T2[2]`
/// corrections and the two `(T)a` intermediates, after
/// D. A. Matthews and J. F. Stanton, JCP **145**, 124102 (2016), Eq. 14.
///
/// # This builds the entire `T3[2]` array
///
/// `nkpts⁵·nocc³·nvir³` complex, upstream's own shape and the reason its name
/// ends in `_slow`. There is no blocked spin-orbital form upstream to port
/// instead: `kintermediates.py` has only this one.
///
/// # Errors
/// Propagates every block access and shape check.
pub fn get_t3p2_imds_slow(
    t1: &ZArr,
    t2: &ZArr,
    eris: &KgEris,
    kconserv: &Kconserv,
    padding: &Padding,
    lat: &KLattice<'_>,
) -> Result<T3p2Imds, PbcCcError> {
    let (nz_o, nz_v) = (&padding.occupied, &padding.virtuals);
    let (a_lat, kpts) = (lat.a, lat.kpts);
    let (nkpts, nocc, nvir) = (eris.nkpts, eris.nocc, eris.nvir);
    let mo_e_o: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[..nocc].to_vec()).collect();
    let mo_e_v: Vec<Vec<f64>> = eris.mo_energy.iter().map(|e| e[nocc..].to_vec()).collect();
    let ccsd_energy = crate::kccsd::energy(t1, t2, eris)?;

    let t3 = get_full_t3p2(t2, eris, kconserv, padding, lat)?;

    // `:480-488` — `pt1`.
    let mut pt1 = ZArr::zeros(&[nkpts, nocc, nvir]);
    for ki in 0..nkpts {
        let ka = ki;
        let mut acc = ZArr::zeros(&[nocc, nvir]);
        for km in 0..nkpts {
            for kn in 0..nkpts {
                for ke in 0..nkpts {
                    acc.add_assign(&einsum_scaled(
                        "mnef,imnaef->ia",
                        &[
                            &eris.blk(GBlk::Oovv, km, kn, ke)?,
                            t3.get([ki, km, kn, ka, ke]),
                        ],
                        0.25,
                    )?)?;
                }
            }
        }
        let eia = crate::kccsd_t::epq_ov(&mo_e_o, nz_o, ki, &mo_e_v, nz_v, ka, nocc, nvir);
        for (f, d) in eia.iter().enumerate() {
            acc.data_mut().re[f] /= d;
            acc.data_mut().im[f] /= d;
        }
        pt1.set_leading(&[ki], &acc)?;
    }

    // `:490-512` — `pt2`.
    let mut pt2 = ZArr::zeros(&[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let mut acc = ZArr::zeros(&[nocc, nocc, nvir, nvir]);
                for km in 0..nkpts {
                    acc.add_assign(&einsum(
                        "ijmabe,me->ijab",
                        &[t3.get([ki, kj, km, ka, kb]), &eris.fov(km)?],
                    )?)?;
                    for ke in 0..nkpts {
                        let kf = kconserv.get(km, ke, kb) as usize;
                        acc.add_assign(&einsum_scaled(
                            "ijmaef,mbfe->ijab",
                            &[
                                t3.get([ki, kj, km, ka, ke]),
                                &eris.blk(GBlk::Ovvv, km, kb, kf)?,
                            ],
                            0.5,
                        )?)?;
                        let kf = kconserv.get(km, ke, ka) as usize;
                        acc.add_assign(&einsum_scaled(
                            "ijmbef,mafe->ijab",
                            &[
                                t3.get([ki, kj, km, kb, ke]),
                                &eris.blk(GBlk::Ovvv, km, ka, kf)?,
                            ],
                            -0.5,
                        )?)?;
                    }
                    for kn in 0..nkpts {
                        acc.add_assign(&einsum_scaled(
                            "inmabe,nmje->ijab",
                            &[
                                t3.get([ki, kn, km, ka, kb]),
                                &eris.blk(GBlk::Ooov, kn, km, kj)?,
                            ],
                            -0.5,
                        )?)?;
                        acc.add_assign(&einsum_scaled(
                            "jnmabe,nmie->ijab",
                            &[
                                t3.get([kj, kn, km, ka, kb]),
                                &eris.blk(GBlk::Ooov, kn, km, ki)?,
                            ],
                            0.5,
                        )?)?;
                    }
                }
                let eia = crate::kccsd_t::epq_ov(&mo_e_o, nz_o, ki, &mo_e_v, nz_v, ka, nocc, nvir);
                let ejb = crate::kccsd_t::epq_ov(&mo_e_o, nz_o, kj, &mo_e_v, nz_v, kb, nocc, nvir);
                let mut f = 0;
                for i in 0..nocc {
                    for j in 0..nocc {
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let d = eia[i * nvir + a] + ejb[j * nvir + b];
                                acc.data_mut().re[f] /= d;
                                acc.data_mut().im[f] /= d;
                                f += 1;
                            }
                        }
                    }
                }
                pt2.set_leading(&[ki, kj, ka], &acc)?;
            }
        }
    }

    // `:514-515`
    pt1.add_assign(t1)?;
    pt2.add_assign(t2)?;

    // `:517-525` — the two `(T)a` intermediates.
    let mut wovoo = ZArr::zeros(&[nkpts, nkpts, nkpts, nocc, nvir, nocc, nocc]);
    let mut wvvvo = ZArr::zeros(&[nkpts, nkpts, nkpts, nvir, nvir, nvir, nocc]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for kk in 0..nkpts {
                for ka in 0..nkpts {
                    for kb in 0..nkpts {
                        let kc = get_kconserv3(
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
                        .data[0] as usize;
                        let tmp = t3.get([ki, kj, kk, ka, kb]);
                        let km = kconserv.get(ki, kc, kk) as usize;
                        let ke = kconserv.get(ka, kk, kc) as usize;

                        let mut w = wovoo.slice_leading(&[km, kc, ki])?;
                        w.add_assign(&einsum_scaled(
                            "ijkabc,mjab->mcik",
                            &[tmp, &eris.blk(GBlk::Oovv, km, kj, ka)?],
                            0.5,
                        )?)?;
                        wovoo.set_leading(&[km, kc, ki], &w)?;

                        let mut w = wvvvo.slice_leading(&[ka, kc, ke])?;
                        w.add_assign(&einsum_scaled(
                            "ijkabc,ijeb->acek",
                            &[tmp, &eris.blk(GBlk::Oovv, ki, kj, ke)?],
                            -0.5,
                        )?)?;
                        wvvvo.set_leading(&[ka, kc, ke], &w)?;
                    }
                }
            }
        }
    }

    let delta_ccsd_energy = crate::kccsd::energy(&pt1, &pt2, eris)? - ccsd_energy;
    Ok(T3p2Imds {
        delta_ccsd_energy,
        pt1,
        pt2,
        wovoo,
        wvvvo,
    })
}

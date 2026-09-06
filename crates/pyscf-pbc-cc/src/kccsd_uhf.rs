//! `KUCCSD` — unrestricted k-point coupled cluster singles and doubles
//! (plan 16-06 Tasks 3-5; `pyscf/pbc/cc/kccsd_uhf.py:61-777`).
//!
//! # Why this is a third implementation and not a parameterisation
//!
//! The repository now carries three k-point CCSD ground states: [`crate::kccsd_rhf`]
//! (spin-restricted, one amplitude pair), [`crate::kccsd`] (spin-ORBITAL, one
//! amplitude pair over a doubled basis) and this one (spin-UNRESTRICTED, five
//! amplitude arrays over three spin channels). Upstream keeps them apart for
//! the reason `16-CONTEXT §1.3` gives: `kccsd.py` costs `(2N)⁶ = 64 N⁶` where
//! this costs `≈ 3 N⁶`, and `kccsd_rhf.py` cannot represent an open shell at
//! all. Folding any two together would mean re-deriving equations that upstream
//! states independently — the `16-CONTEXT §3.4` prohibition.
//!
//! # `t1` is `(t1a, t1b)` and `t2` is `(t2aa, t2ab, t2bb)`
//!
//! There is no `t2ba`: `t2ab[ki,kj,ka][i,J,a,B]` carries it, and every place
//! upstream needs the `ba` ordering it re-indexes `t2ab` rather than storing a
//! fourth array (e.g. `:113` `einsum('yxymiea,yme->xia', t2ab, Fov_)`). This
//! port keeps that, because a `t2ba` would have to be kept consistent by hand.
//!
//! # The one place upstream's own equations are asymmetric, and it is not a bug
//!
//! `u2aa` and `u2bb` are antisymmetrised at the end (`:331-333`, `:361-363`)
//! while `Ht2ab` is not, because the `ab` channel has no exchange partner
//! inside itself — its `P(ij)`/`P(ab)` images live in `Ht2ab` at MIRRORED
//! k-addresses, which is why `:261-262` writes `Ht2ab[kj,:,kb]` and `:279`
//! writes `Ht2ab[:,kj,kb]`. Both are transcribed here as explicit k-loops.
//!
//! # DEFERRED, explicitly
//!
//! `_make_df_eris` (`:1017`) — see [`crate::kueris`]'s module doc. The
//! `cc.direct` branch of `add_vvvv_` (`:562-590`) that reads its `Lpv`/`LPV` is
//! therefore not reachable, and [`add_vvvv`] takes the `cc_Wvvvv_half` route
//! unconditionally, which is upstream's `else` at `:591-594` and produces the
//! same numbers.

use std::sync::Arc;

use pyscf_algebra::oracle_sum;
use pyscf_diis::Diis;
use pyscf_pbc_lib::Kconserv;
use pyscf_pbc_mp::PaddedMos;
use pyscf_runtime::ZWorkspacePool;

use crate::error::PbcCcError;
use crate::kccsd_rhf::{KrccsdOpts, get_eia, split_padding};
use crate::kintermediates_uhf::{
    UT1, UT2, UWovvo, b, cc_foo, cc_fov, cc_fvv, cc_woooo, cc_wovvo, cc_wvvvv_half, make_tau,
};
use crate::ktensor::KBlocks;
use crate::kueris::KuEris;
use crate::zarr::{ZArr, einsum};

/// What [`kernel`] returns.
#[derive(Debug, Clone)]
pub struct KuccsdResult {
    pub e_corr: f64,
    pub emp2: f64,
    pub converged: bool,
    pub cycles: usize,
    pub t1: UT1,
    pub t2: UT2,
}

/// `energy(cc, t1, t2, eris)` — `kccsd_uhf.py:436-470`.
///
/// # Errors
/// Propagates the ERI access and every shape check.
pub fn energy(t1: &UT1, t2: &UT2, eris: &KuEris) -> Result<f64, PbcCcError> {
    let nkpts = eris.nkpts;
    let mut re: Vec<f64> = Vec::new();
    let mut im: Vec<f64> = Vec::new();

    // `:446-449` — the one-body term.
    for ki in 0..nkpts {
        let (r, i) = einsum(
            "ia,ia->",
            &[&eris.fov(false, ki)?, &t1.0.slice_leading(&[ki])?],
        )?
        .at(&[])?;
        re.push(r);
        im.push(i);
        let (r, i) = einsum(
            "ia,ia->",
            &[&eris.fov(true, ki)?, &t1.1.slice_leading(&[ki])?],
        )?
        .at(&[])?;
        re.push(r);
        im.push(i);
    }

    // `:450-462` — tau = t2 + 2 t1 t1 (same spin) / t2 + t1 t1 (mixed), all at
    // `ka == ki`. NOT `make_tau`: the factors differ (2 against ½·2).
    let mut tau = (t2.0.clone(), t2.1.clone(), t2.2.clone());
    for ki in 0..nkpts {
        let (ia, ib) = (t1.0.slice_leading(&[ki])?, t1.1.slice_leading(&[ki])?);
        for kj in 0..nkpts {
            let (ja, jb) = (t1.0.slice_leading(&[kj])?, t1.1.slice_leading(&[kj])?);
            add_at(
                &mut tau.0,
                [ki, kj, ki],
                &einsum("ia,jb->ijab", &[&ia, &ja])?,
                2.0,
            )?;
            add_at(
                &mut tau.1,
                [ki, kj, ki],
                &einsum("ia,jb->ijab", &[&ia, &jb])?,
                1.0,
            )?;
            add_at(
                &mut tau.2,
                [ki, kj, ki],
                &einsum("ia,jb->ijab", &[&ib, &jb])?,
                2.0,
            )?;
        }
    }

    let (dr, di) = pair_energy(&tau, eris)?;
    re.extend(dr);
    im.extend(di);
    let e_re = oracle_sum(&re) / nkpts as f64;
    let e_im = oracle_sum(&im) / nkpts as f64;
    if e_im.abs() > 1e-4 {
        tracing::warn!(
            imaginary = e_im,
            "non-zero imaginary part in the KUCCSD energy (kccsd_uhf.py:468)"
        );
    }
    Ok(e_re)
}

/// `:463-467` / `:743-748` — the two-body energy of a `tau`-shaped triple,
/// returned as UNSUMMED ordered terms so the caller controls the reduction
/// (D-PBC-17).
fn pair_energy(tau: &UT2, eris: &KuEris) -> Result<(Vec<f64>, Vec<f64>), PbcCcError> {
    let nkpts = eris.nkpts;
    let mut re = Vec::new();
    let mut im = Vec::new();
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            for kz in 0..nkpts {
                let aa = tau.0.slice_leading(&[kx, ky, kz])?;
                let ab = tau.1.slice_leading(&[kx, ky, kz])?;
                let bb = tau.2.slice_leading(&[kx, ky, kz])?;
                let mut push = |v: (f64, f64), f: f64| {
                    re.push(f * v.0);
                    im.push(f * v.1);
                };
                push(
                    einsum("iajb,ijab->", &[&eris.blk(b::ovov, kx, kz, ky)?, &aa])?.at(&[])?,
                    0.25,
                );
                push(
                    einsum("jaib,ijab->", &[&eris.blk(b::ovov, ky, kz, kx)?, &aa])?.at(&[])?,
                    -0.25,
                );
                push(
                    einsum("iajb,ijab->", &[&eris.blk(b::ovOV, kx, kz, ky)?, &ab])?.at(&[])?,
                    1.0,
                );
                push(
                    einsum("iajb,ijab->", &[&eris.blk(b::OVOV, kx, kz, ky)?, &bb])?.at(&[])?,
                    0.25,
                );
                push(
                    einsum("jaib,ijab->", &[&eris.blk(b::OVOV, ky, kz, kx)?, &bb])?.at(&[])?,
                    -0.25,
                );
            }
        }
    }
    Ok((re, im))
}

/// The `(occ, vir)` energy differences of both spins, `[ki][ka]`-indexed, with
/// PADDED entries at [`crate::kccsd_rhf::LARGE_DENOM`] (`:718-733`).
struct Denoms {
    a: Vec<Vec<Vec<f64>>>,
    b: Vec<Vec<Vec<f64>>>,
}

fn denoms(
    eris: &KuEris,
    padded: (&PaddedMos, &PaddedMos),
    level_shift: f64,
) -> Result<Denoms, PbcCcError> {
    let nkpts = eris.nkpts;
    let mut out = Denoms {
        a: Vec::new(),
        b: Vec::new(),
    };
    for (spin, pad) in [(0usize, padded.0), (1usize, padded.1)] {
        let (nocc, energies) = if spin == 0 {
            (eris.nocc.0, &eris.mo_energy.0)
        } else {
            (eris.nocc.1, &eris.mo_energy.1)
        };
        let mo_e_o: Vec<Vec<f64>> = energies.iter().map(|e| e[..nocc].to_vec()).collect();
        let mo_e_v: Vec<Vec<f64>> = energies
            .iter()
            .map(|e| e[nocc..].iter().map(|x| x + level_shift).collect())
            .collect();
        let (nz_o, nz_v) = split_padding(pad)?;
        let mut per_ki = Vec::with_capacity(nkpts);
        for ki in 0..nkpts {
            let mut per_ka = Vec::with_capacity(nkpts);
            for ka in 0..nkpts {
                per_ka.push(get_eia(&mo_e_o, &mo_e_v, ki, ka, &nz_o, &nz_v));
            }
            per_ki.push(per_ka);
        }
        if spin == 0 {
            out.a = per_ki;
        } else {
            out.b = per_ki;
        }
    }
    Ok(out)
}

/// `x / (eia[i,a] + ejb[j,b])`, with independent occupied and virtual extents
/// per index pair so the mixed-spin `ab` channel is expressible.
fn divide(
    x: &ZArr,
    eia: &[f64],
    ejb: &[f64],
    o1: usize,
    o2: usize,
    v1: usize,
    v2: usize,
) -> Result<ZArr, PbcCcError> {
    let mut out = x.clone();
    for i in 0..o1 {
        for j in 0..o2 {
            for a in 0..v1 {
                for b_ in 0..v2 {
                    let d = eia[i * v1 + a] + ejb[j * v2 + b_];
                    let f = ((i * o2 + j) * v1 + a) * v2 + b_;
                    out.data_mut().re[f] /= d;
                    out.data_mut().im[f] /= d;
                }
            }
        }
    }
    Ok(out)
}

/// `KUCCSD.init_amps(eris)` — `:684-753`. Returns `(emp2, t1, t2)`.
///
/// # Errors
/// Propagates the ERI access and every shape check.
pub fn init_amps(
    eris: &KuEris,
    padded: (&PaddedMos, &PaddedMos),
    kconserv: &Kconserv,
) -> Result<(f64, UT1, UT2), PbcCcError> {
    let nkpts = eris.nkpts;
    let (oa, ob) = (eris.nocc.0, eris.nocc.1);
    let (va, vb) = (eris.nvir.0, eris.nvir.1);
    let t1 = (ZArr::zeros(&[nkpts, oa, va]), ZArr::zeros(&[nkpts, ob, vb]));
    let mut t2 = (
        ZArr::zeros(&[nkpts, nkpts, nkpts, oa, oa, va, va]),
        ZArr::zeros(&[nkpts, nkpts, nkpts, oa, ob, va, vb]),
        ZArr::zeros(&[nkpts, nkpts, nkpts, ob, ob, vb, vb]),
    );
    // `:711` — `init_amps` does NOT apply `level_shift`; `update_amps` does
    // (`:88-89`). The two denominators genuinely differ.
    let d = denoms(eris, padded, 0.0)?;

    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let (eia, eib) = (&d.a[ki][ka], &d.b[ki][ka]);
                let (eja, ejb) = (&d.a[kj][kb], &d.b[kj][kb]);

                // `:739-743`
                let x = eris
                    .blk(b::ovov, ki, ka, kj)?
                    .conj()
                    .transpose(&[0, 2, 1, 3])?;
                let mut aa = divide(&x, eia, eja, oa, oa, va, va)?;
                let y = eris
                    .blk(b::ovov, kj, ka, ki)?
                    .conj()
                    .transpose(&[2, 0, 1, 3])?;
                aa.sub_assign(&divide(&y, eia, eja, oa, oa, va, va)?)?;
                t2.0.set_leading(&[ki, kj, ka], &aa)?;

                let x = eris
                    .blk(b::ovOV, ki, ka, kj)?
                    .conj()
                    .transpose(&[0, 2, 1, 3])?;
                t2.1.set_leading(&[ki, kj, ka], &divide(&x, eia, ejb, oa, ob, va, vb)?)?;

                let x = eris
                    .blk(b::OVOV, ki, ka, kj)?
                    .conj()
                    .transpose(&[0, 2, 1, 3])?;
                let mut bb = divide(&x, eib, ejb, ob, ob, vb, vb)?;
                let y = eris
                    .blk(b::OVOV, kj, ka, ki)?
                    .conj()
                    .transpose(&[2, 0, 1, 3])?;
                bb.sub_assign(&divide(&y, eib, ejb, ob, ob, vb, vb)?)?;
                t2.2.set_leading(&[ki, kj, ka], &bb)?;
            }
        }
    }

    let (re, _) = pair_energy(&t2, eris)?;
    Ok((oracle_sum(&re) / nkpts as f64, t1, t2))
}

/// `add_vvvv_(cc, Ht2, t1, t2, eris)` — `:550-635`, `else` branch only.
///
/// # Errors
/// Propagates the intermediate build and every shape check.
pub(crate) fn add_vvvv(
    ht2: &mut UT2,
    t1: &UT1,
    t2: &UT2,
    w: &(KBlocks, KBlocks, KBlocks),
    nkpts: usize,
    kconserv: &Kconserv,
) -> Result<(), PbcCcError> {
    for ka in 0..nkpts {
        for kb in 0..nkpts {
            for kc in 0..nkpts {
                let kd = kconserv.get(ka, kc, kb) as usize;
                let wvvvv = w.0.get([ka, kc, kb])?;
                let wvvvvab = w.1.get([ka, kc, kb])?;
                let wvvvvbb = w.2.get([ka, kc, kb])?;
                for ki in 0..nkpts {
                    let kj = kconserv.get(ka, ki, kb) as usize;
                    let mut tauaa = t2.0.slice_leading(&[ki, kj, kc])?;
                    let mut tauab = t2.1.slice_leading(&[ki, kj, kc])?;
                    let mut taubb = t2.2.slice_leading(&[ki, kj, kc])?;
                    let (ia, ib) = (t1.0.slice_leading(&[ki])?, t1.1.slice_leading(&[ki])?);
                    let (ja, jb) = (t1.0.slice_leading(&[kj])?, t1.1.slice_leading(&[kj])?);
                    if ki == kc && kj == kd {
                        tauaa.add_assign(&einsum("ic,jd->ijcd", &[&ia, &ja])?)?;
                        tauab.add_assign(&einsum("ic,jd->ijcd", &[&ia, &jb])?)?;
                        taubb.add_assign(&einsum("ic,jd->ijcd", &[&ib, &jb])?)?;
                    }
                    if ki == kd && kj == kc {
                        tauaa.sub_assign(&einsum("id,jc->ijcd", &[&ia, &ja])?)?;
                        taubb.sub_assign(&einsum("id,jc->ijcd", &[&ib, &jb])?)?;
                    }

                    let mut tmp = einsum("acbd,ijcd->ijab", &[&wvvvv, &tauaa])?;
                    tmp.scale(0.5);
                    add_at(&mut ht2.0, [ki, kj, ka], &tmp, 1.0)?;
                    add_at(
                        &mut ht2.0,
                        [ki, kj, kb],
                        &tmp.transpose(&[0, 1, 3, 2])?,
                        -1.0,
                    )?;

                    let mut tmp = einsum("acbd,ijcd->ijab", &[&wvvvvbb, &taubb])?;
                    tmp.scale(0.5);
                    add_at(&mut ht2.2, [ki, kj, ka], &tmp, 1.0)?;
                    add_at(
                        &mut ht2.2,
                        [ki, kj, kb],
                        &tmp.transpose(&[0, 1, 3, 2])?,
                        -1.0,
                    )?;

                    let tmp = einsum("acbd,ijcd->ijab", &[&wvvvvab, &tauab])?;
                    add_at(&mut ht2.1, [ki, kj, ka], &tmp, 1.0)?;
                }
            }
        }
    }
    Ok(())
}

/// [`add_vvvv`] under a name the integration test can reach.
///
/// It is `pub` for one reason: 16-06 test 3b bisects a `t2new` mismatch by
/// running each intermediate separately, and `add_vvvv_` is the one step that
/// is neither an intermediate nor the whole update. Upstream exposes it as a
/// module-level function (`kccsd_uhf.py:550`) for the same reason.
///
/// # Errors
/// As [`update_amps`].
pub fn add_vvvv_for_test(
    ht2: &mut UT2,
    t1: &UT1,
    t2: &UT2,
    w: &(KBlocks, KBlocks, KBlocks),
    nkpts: usize,
    kconserv: &Kconserv,
) -> Result<(), PbcCcError> {
    add_vvvv(ht2, t1, t2, w, nkpts, kconserv)
}

/// [`wovvo_terms`] under a name the integration test can reach — the sibling of
/// [`add_vvvv_for_test`], and for the same reason: it is the largest single
/// block of the doubles equation (`kccsd_uhf.py:230-386`) and a mismatch in the
/// assembled `t2new` has to be attributable to it or to something else.
///
/// # Errors
/// As [`update_amps`].
pub fn wovvo_terms_for_test(
    ht2: &mut UT2,
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    w: &UWovvo,
    kconserv: &Kconserv,
) -> Result<(), PbcCcError> {
    wovvo_terms(ht2, t1, t2, eris, w, kconserv)
}

/// `:205-226` — the bare `ovov` driver and the `Woooo` stage.
///
/// Upstream writes these as two separate `loop_kkk`s (`:216-220` completing the
/// intermediate, `:222-226` contracting it). They are merged here into one
/// loop, which is exact: the completing write and the contracting read are both
/// at `[km,ki,kn]`, so no iteration reads an address another iteration writes.
///
/// # Errors
/// Propagates every ERI access and shape check.
#[allow(clippy::too_many_arguments)]
pub(crate) fn woooo_terms(
    ht2: &mut UT2,
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    pool: &Arc<ZWorkspacePool>,
    budget: usize,
    kconserv: &Kconserv,
) -> Result<(), PbcCcError> {
    let nkpts = eris.nkpts;
    // `:203-206` — the bare `ovov` driver, per block rather than as three
    // whole-array transposes.
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let mut aa = eris.blk(b::ovov, ki, ka, kj)?.transpose(&[0, 2, 1, 3])?;
                aa.sub_assign(&eris.blk(b::ovov, kj, ka, ki)?.transpose(&[2, 0, 1, 3])?)?;
                add_at(&mut ht2.0, [ki, kj, ka], &aa.conj(), 1.0)?;

                let mut bb = eris.blk(b::OVOV, ki, ka, kj)?.transpose(&[0, 2, 1, 3])?;
                bb.sub_assign(&eris.blk(b::OVOV, kj, ka, ki)?.transpose(&[2, 0, 1, 3])?)?;
                add_at(&mut ht2.2, [ki, kj, ka], &bb.conj(), 1.0)?;

                let ab = eris.blk(b::ovOV, ki, ka, kj)?.transpose(&[0, 2, 1, 3])?;
                add_at(&mut ht2.1, [ki, kj, ka], &ab.conj(), 1.0)?;
            }
        }
    }

    // `:208-224` — Woooo, with the `ovov·tau` completion folded in by
    // `cc_woooo` itself (upstream adds it at `:212-215` and `cc_Woooo`
    // adds a `0.25`/`0.5` copy at `:218-220`; the two together are the
    // `0.5`/`1.0` of the driver).
    let tau = make_tau(t2, t1, t1, 1.0)?;
    let (waa, wab, wbb) = cc_woooo(pool, budget, t1, t2, eris, kconserv)?;
    for km in 0..nkpts {
        for ki in 0..nkpts {
            for kn in 0..nkpts {
                let kj = kconserv.get(km, ki, kn) as usize;
                let mut aa = waa.get([km, ki, kn])?;
                let mut ab = wab.get([km, ki, kn])?;
                let mut bb = wbb.get([km, ki, kn])?;
                // `:212-215` — the SECOND half-weight `ovov·tau` term, added to
                // the intermediate in `update_amps` itself.
                for kx in 0..nkpts {
                    let mut c = einsum(
                        "menf,ijef->minj",
                        &[
                            &eris.blk(b::ovov, km, kx, kn)?,
                            &tau.0.slice_leading(&[ki, kj, kx])?,
                        ],
                    )?;
                    c.scale(0.5);
                    aa.add_assign(&c)?;
                    let mut c = einsum(
                        "MENF,IJEF->MINJ",
                        &[
                            &eris.blk(b::OVOV, km, kx, kn)?,
                            &tau.2.slice_leading(&[ki, kj, kx])?,
                        ],
                    )?;
                    c.scale(0.5);
                    bb.add_assign(&c)?;
                    let mut c = einsum(
                        "meNF,iJeF->miNJ",
                        &[
                            &eris.blk(b::ovOV, km, kx, kn)?,
                            &tau.1.slice_leading(&[ki, kj, kx])?,
                        ],
                    )?;
                    c.scale(0.5);
                    ab.add_assign(&c)?;
                }
                // `:219-222`
                for kw in 0..nkpts {
                    let mut c = einsum(
                        "minj,mnab->ijab",
                        &[&aa, &tau.0.slice_leading(&[km, kn, kw])?],
                    )?;
                    c.scale(0.5);
                    add_at(&mut ht2.0, [ki, kj, kw], &c, 1.0)?;
                    let mut c = einsum(
                        "MINJ,MNAB->IJAB",
                        &[&bb, &tau.2.slice_leading(&[km, kn, kw])?],
                    )?;
                    c.scale(0.5);
                    add_at(&mut ht2.2, [ki, kj, kw], &c, 1.0)?;
                    let c = einsum(
                        "miNJ,mNaB->iJaB",
                        &[&ab, &tau.1.slice_leading(&[km, kn, kw])?],
                    )?;
                    add_at(&mut ht2.1, [ki, kj, kw], &c, 1.0)?;
                }
            }
        }
    }
    waa.release();
    wab.release();
    wbb.release();
    Ok(())
}

/// [`woooo_terms`] under a name the integration test can reach — the sibling of
/// [`add_vvvv_for_test`] and [`wovvo_terms_for_test`].
///
/// # Errors
/// As [`update_amps`].
pub fn woooo_terms_for_test(
    ht2: &mut UT2,
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    pool: &Arc<ZWorkspacePool>,
    budget: usize,
    kconserv: &Kconserv,
) -> Result<(), PbcCcError> {
    woooo_terms(ht2, t1, t2, eris, pool, budget, kconserv)
}

/// `:65-202` — the intermediates, the SINGLES equation and the `Fvv`/`Foo`
/// doubles driving loop.
///
/// The three are one stage because they share `Fvv_`/`Foo_`/`Fov_` and the
/// diagonal shift that moves the orbital energies to the other side of the
/// equation (`:96-100`): splitting them would mean building the intermediates
/// twice or passing four more arguments.
///
/// # Errors
/// Propagates every ERI access, intermediate build and shape check.
pub(crate) fn fock_terms(
    ht1: &mut UT1,
    ht2: &mut UT2,
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    kconserv: &Kconserv,
    opts: &KrccsdOpts,
) -> Result<(), PbcCcError> {
    let nkpts = eris.nkpts;
    let (oa, ob) = eris.nocc;
    // `:92-101` — the intermediates, with the orbital energies moved to the
    // other side of the equation.
    let (mut fvv_a, mut fvv_b) = cc_fvv(t1, t2, eris, kconserv)?;
    let (mut foo_a, mut foo_b) = cc_foo(t1, t2, eris, kconserv)?;
    let (fov_a, fov_b) = cc_fov(t1, eris)?;
    for k in 0..nkpts {
        shift_diag(&mut fvv_a, k, &eris.mo_energy.0[k][oa..], opts.level_shift)?;
        shift_diag(&mut fvv_b, k, &eris.mo_energy.1[k][ob..], opts.level_shift)?;
        shift_diag(&mut foo_a, k, &eris.mo_energy.0[k][..oa], 0.0)?;
        shift_diag(&mut foo_b, k, &eris.mo_energy.1[k][..ob], 0.0)?;
    }

    // ---------------------------------------------------------------- T1
    // `:107-108`
    for k in 0..nkpts {
        add_at1(&mut ht1.0, k, &eris.fov(false, k)?.conj(), 1.0)?;
        add_at1(&mut ht1.1, k, &eris.fov(true, k)?.conj(), 1.0)?;
    }
    // `:109-112`
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            let fa = fov_a.slice_leading(&[ky])?;
            let fb = fov_b.slice_leading(&[ky])?;
            add_at1(
                &mut ht1.0,
                kx,
                &einsum("imae,me->ia", &[&t2.0.slice_leading(&[kx, ky, kx])?, &fa])?,
                1.0,
            )?;
            add_at1(
                &mut ht1.0,
                kx,
                &einsum("imae,me->ia", &[&t2.1.slice_leading(&[kx, ky, kx])?, &fb])?,
                1.0,
            )?;
            add_at1(
                &mut ht1.1,
                kx,
                &einsum("imae,me->ia", &[&t2.2.slice_leading(&[kx, ky, kx])?, &fb])?,
                1.0,
            )?;
            add_at1(
                &mut ht1.1,
                kx,
                &einsum("miea,me->ia", &[&t2.1.slice_leading(&[ky, kx, ky])?, &fa])?,
                1.0,
            )?;
        }
    }
    // `:113-121` — the `ooov` terms. `x` and `y` are summed; `z` is the output
    // k-index, and the ERI is addressed `[kx, kz, ky]`, NOT `[kx, ky, kz]`.
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            for kz in 0..nkpts {
                let kw = kconserv.get(kx, kz, ky) as usize;
                add_at1(
                    &mut ht1.0,
                    kz,
                    &einsum(
                        "mnae,mine->ia",
                        &[
                            &t2.0.slice_leading(&[kx, ky, kz])?,
                            &eris.blk(b::ooov, kx, kz, ky)?,
                        ],
                    )?,
                    -1.0,
                )?;
                add_at1(
                    &mut ht1.0,
                    kz,
                    &einsum(
                        "mNaE,miNE->ia",
                        &[
                            &t2.1.slice_leading(&[kx, ky, kz])?,
                            &eris.blk(b::ooOV, kx, kz, ky)?,
                        ],
                    )?,
                    -1.0,
                )?;
                add_at1(
                    &mut ht1.1,
                    kz,
                    &einsum(
                        "mnae,mine->ia",
                        &[
                            &t2.2.slice_leading(&[kx, ky, kz])?,
                            &eris.blk(b::OOOV, kx, kz, ky)?,
                        ],
                    )?,
                    -1.0,
                )?;
                add_at1(
                    &mut ht1.1,
                    kz,
                    &einsum(
                        "nmea,mine->ia",
                        &[
                            &t2.1.slice_leading(&[ky, kx, kw])?,
                            &eris.blk(b::OOov, kx, kz, ky)?,
                        ],
                    )?,
                    -1.0,
                )?;
            }
        }
    }

    // `:123-145`
    for ka in 0..nkpts {
        let (t1aa, t1ba) = (t1.0.slice_leading(&[ka])?, t1.1.slice_leading(&[ka])?);
        add_at1(
            &mut ht1.0,
            ka,
            &einsum("ie,ae->ia", &[&t1aa, &fvv_a.slice_leading(&[ka])?])?,
            1.0,
        )?;
        add_at1(
            &mut ht1.1,
            ka,
            &einsum("ie,ae->ia", &[&t1ba, &fvv_b.slice_leading(&[ka])?])?,
            1.0,
        )?;
        add_at1(
            &mut ht1.0,
            ka,
            &einsum("ma,mi->ia", &[&t1aa, &foo_a.slice_leading(&[ka])?])?,
            -1.0,
        )?;
        add_at1(
            &mut ht1.1,
            ka,
            &einsum("ma,mi->ia", &[&t1ba, &foo_b.slice_leading(&[ka])?])?,
            -1.0,
        )?;

        for km in 0..nkpts {
            let (t1am, t1bm) = (t1.0.slice_leading(&[km])?, t1.1.slice_leading(&[km])?);
            add_at1(
                &mut ht1.0,
                ka,
                &einsum("mf,aimf->ia", &[&t1am, &eris.blk(b::voov, ka, ka, km)?])?,
                1.0,
            )?;
            add_at1(
                &mut ht1.0,
                ka,
                &einsum("mf,miaf->ia", &[&t1am, &eris.blk(b::oovv, km, ka, ka)?])?,
                -1.0,
            )?;
            add_at1(
                &mut ht1.0,
                ka,
                &einsum("MF,aiMF->ia", &[&t1bm, &eris.blk(b::voOV, ka, ka, km)?])?,
                1.0,
            )?;
            add_at1(
                &mut ht1.1,
                ka,
                &einsum("MF,AIMF->IA", &[&t1bm, &eris.blk(b::VOOV, ka, ka, km)?])?,
                1.0,
            )?;
            add_at1(
                &mut ht1.1,
                ka,
                &einsum("MF,MIAF->IA", &[&t1bm, &eris.blk(b::OOVV, km, ka, ka)?])?,
                -1.0,
            )?;
            add_at1(
                &mut ht1.1,
                ka,
                &einsum(
                    "mf,fmIA->IA",
                    &[&t1am, &eris.blk(b::voOV, km, km, ka)?.conj()],
                )?,
                1.0,
            )?;

            for kf in 0..nkpts {
                let ki = ka;
                let ke = kconserv.get(ki, kf, km) as usize;
                add_at1(
                    &mut ht1.0,
                    ka,
                    &einsum(
                        "imef,fmea->ia",
                        &[
                            &t2.0.slice_leading(&[ki, km, ke])?,
                            &eris.blk(b::vovv, kf, km, ke)?.conj(),
                        ],
                    )?,
                    1.0,
                )?;
                add_at1(
                    &mut ht1.0,
                    ka,
                    &einsum(
                        "iMeF,FMea->ia",
                        &[
                            &t2.1.slice_leading(&[ki, km, ke])?,
                            &eris.blk(b::VOvv, kf, km, ke)?.conj(),
                        ],
                    )?,
                    1.0,
                )?;
                add_at1(
                    &mut ht1.1,
                    ka,
                    &einsum(
                        "IMEF,FMEA->IA",
                        &[
                            &t2.2.slice_leading(&[ki, km, ke])?,
                            &eris.blk(b::VOVV, kf, km, ke)?.conj(),
                        ],
                    )?,
                    1.0,
                )?;
                add_at1(
                    &mut ht1.1,
                    ka,
                    &einsum(
                        "mIfE,fmEA->IA",
                        &[
                            &t2.1.slice_leading(&[km, ki, kf])?,
                            &eris.blk(b::voVV, kf, km, ke)?.conj(),
                        ],
                    )?,
                    1.0,
                )?;
            }
        }
    }

    // ---------------------------------------------------------------- T2
    // `:147-197` — the `Fvv`/`Foo` driving terms.
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let ftmp = |f: &ZArr,
                            fov: &ZArr,
                            t: &ZArr,
                            k: usize,
                            s: f64,
                            spec: &str|
                 -> Result<ZArr, PbcCcError> {
                    let mut x = f.slice_leading(&[k])?;
                    let mut c =
                        einsum(spec, &[&t.slice_leading(&[k])?, &fov.slice_leading(&[k])?])?;
                    c.scale(s);
                    x.add_assign(&c)?;
                    Ok(x)
                };
                // `:150-156`
                let fa_kb = ftmp(&fvv_a, &fov_a, &t1.0, kb, -0.5, "mb,me->be")?;
                let fb_kb = ftmp(&fvv_b, &fov_b, &t1.1, kb, -0.5, "MB,ME->BE")?;
                let fa_ka = ftmp(&fvv_a, &fov_a, &t1.0, ka, -0.5, "mb,me->be")?;
                let fb_ka = ftmp(&fvv_b, &fov_b, &t1.1, ka, -0.5, "MB,ME->BE")?;

                add_at(
                    &mut ht2.0,
                    [ki, kj, ka],
                    &einsum(
                        "ijae,be->ijab",
                        &[&t2.0.slice_leading(&[ki, kj, ka])?, &fa_kb],
                    )?,
                    1.0,
                )?;
                add_at(
                    &mut ht2.2,
                    [ki, kj, ka],
                    &einsum(
                        "IJAE,BE->IJAB",
                        &[&t2.2.slice_leading(&[ki, kj, ka])?, &fb_kb],
                    )?,
                    1.0,
                )?;
                add_at(
                    &mut ht2.1,
                    [ki, kj, ka],
                    &einsum(
                        "iJaE,BE->iJaB",
                        &[&t2.1.slice_leading(&[ki, kj, ka])?, &fb_kb],
                    )?,
                    1.0,
                )?;
                add_at(
                    &mut ht2.1,
                    [ki, kj, ka],
                    &einsum(
                        "iJeB,ae->iJaB",
                        &[&t2.1.slice_leading(&[ki, kj, ka])?, &fa_ka],
                    )?,
                    1.0,
                )?;
                // `:174-179` P(ab)
                add_at(
                    &mut ht2.0,
                    [ki, kj, ka],
                    &einsum(
                        "ijbe,ae->ijab",
                        &[&t2.0.slice_leading(&[ki, kj, kb])?, &fa_ka],
                    )?,
                    -1.0,
                )?;
                add_at(
                    &mut ht2.2,
                    [ki, kj, ka],
                    &einsum(
                        "IJBE,AE->IJAB",
                        &[&t2.2.slice_leading(&[ki, kj, kb])?, &fb_ka],
                    )?,
                    -1.0,
                )?;

                // `:182-186`
                let ga_kj = ftmp(&foo_a, &fov_a, &t1.0, kj, 0.5, "je,me->mj")?;
                let gb_kj = ftmp(&foo_b, &fov_b, &t1.1, kj, 0.5, "JE,ME->MJ")?;
                let ga_ki = ftmp(&foo_a, &fov_a, &t1.0, ki, 0.5, "je,me->mj")?;
                let gb_ki = ftmp(&foo_b, &fov_b, &t1.1, ki, 0.5, "JE,ME->MJ")?;

                add_at(
                    &mut ht2.0,
                    [ki, kj, ka],
                    &einsum(
                        "imab,mj->ijab",
                        &[&t2.0.slice_leading(&[ki, kj, ka])?, &ga_kj],
                    )?,
                    -1.0,
                )?;
                add_at(
                    &mut ht2.2,
                    [ki, kj, ka],
                    &einsum(
                        "IMAB,MJ->IJAB",
                        &[&t2.2.slice_leading(&[ki, kj, ka])?, &gb_kj],
                    )?,
                    -1.0,
                )?;
                add_at(
                    &mut ht2.1,
                    [ki, kj, ka],
                    &einsum(
                        "iMaB,MJ->iJaB",
                        &[&t2.1.slice_leading(&[ki, kj, ka])?, &gb_kj],
                    )?,
                    -1.0,
                )?;
                add_at(
                    &mut ht2.1,
                    [ki, kj, ka],
                    &einsum(
                        "mJaB,mi->iJaB",
                        &[&t2.1.slice_leading(&[ki, kj, ka])?, &ga_ki],
                    )?,
                    -1.0,
                )?;
                // `:191-197` P(ij)
                add_at(
                    &mut ht2.0,
                    [ki, kj, ka],
                    &einsum(
                        "jmab,mi->ijab",
                        &[&t2.0.slice_leading(&[kj, ki, ka])?, &ga_ki],
                    )?,
                    1.0,
                )?;
                add_at(
                    &mut ht2.2,
                    [ki, kj, ka],
                    &einsum(
                        "JMAB,MI->IJAB",
                        &[&t2.2.slice_leading(&[kj, ki, ka])?, &gb_ki],
                    )?,
                    1.0,
                )?;
            }
        }
    }
    Ok(())
}

/// [`fock_terms`] under a name the integration test can reach — the last of the
/// four stage hooks (`add_vvvv_for_test`, `woooo_terms_for_test`,
/// `wovvo_terms_for_test`).
///
/// # Errors
/// As [`update_amps`].
pub fn fock_terms_for_test(
    ht1: &mut UT1,
    ht2: &mut UT2,
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    kconserv: &Kconserv,
    opts: &KrccsdOpts,
) -> Result<(), PbcCcError> {
    fock_terms(ht1, ht2, t1, t2, eris, kconserv, opts)
}

/// `update_amps(cc, t1, t2, eris)` — `:61-424`.
///
/// # Errors
/// Propagates every ERI access, intermediate build and shape check.
#[allow(clippy::too_many_lines)]
pub fn update_amps(
    pool: &Arc<ZWorkspacePool>,
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    padded: (&PaddedMos, &PaddedMos),
    kconserv: &Kconserv,
    opts: &KrccsdOpts,
) -> Result<(UT1, UT2), PbcCcError> {
    let nkpts = eris.nkpts;
    let (oa, ob) = (eris.nocc.0, eris.nocc.1);
    let (va, vb) = (eris.nvir.0, eris.nvir.1);
    let budget = (opts.max_memory * 1e6).max(0.0) as usize;

    let mut ht1 = (ZArr::zeros(&[nkpts, oa, va]), ZArr::zeros(&[nkpts, ob, vb]));
    let mut ht2 = (
        ZArr::zeros(&[nkpts, nkpts, nkpts, oa, oa, va, va]),
        ZArr::zeros(&[nkpts, nkpts, nkpts, oa, ob, va, vb]),
        ZArr::zeros(&[nkpts, nkpts, nkpts, ob, ob, vb, vb]),
    );

    fock_terms(&mut ht1, &mut ht2, t1, t2, eris, kconserv, opts)?;

    woooo_terms(&mut ht2, t1, t2, eris, pool, budget, kconserv)?;

    // `:224` add_vvvv_
    let wv = cc_wvvvv_half(pool, budget, t1, eris, kconserv)?;
    add_vvvv(&mut ht2, t1, t2, &wv, nkpts, kconserv)?;
    wv.0.release();
    wv.1.release();
    wv.2.release();

    // `:226-227` Wovvo and everything that reads it.
    let w = cc_wovvo(pool, budget, t1, t2, eris, kconserv)?;
    wovvo_terms(&mut ht2, t1, t2, eris, &w, kconserv)?;
    w.aa.release();
    w.ab.release();
    w.ba.release();
    w.bb.release();
    w.abba.release();
    w.baab.release();

    // `:387-424` — the denominators, with `level_shift` on the virtuals.
    let d = denoms(eris, padded, opts.level_shift)?;
    for ki in 0..nkpts {
        let mut x = ht1.0.slice_leading(&[ki])?;
        divide_in_place(&mut x, &d.a[ki][ki]);
        ht1.0.set_leading(&[ki], &x)?;
        let mut x = ht1.1.slice_leading(&[ki])?;
        divide_in_place(&mut x, &d.b[ki][ki]);
        ht1.1.set_leading(&[ki], &x)?;
    }
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(ki, ka, kj) as usize;
                let (eia, eib) = (&d.a[ki][ka], &d.b[ki][ka]);
                let (eja, ejb) = (&d.a[kj][kb], &d.b[kj][kb]);
                let x = ht2.0.slice_leading(&[ki, kj, ka])?;
                ht2.0
                    .set_leading(&[ki, kj, ka], &divide(&x, eia, eja, oa, oa, va, va)?)?;
                let x = ht2.1.slice_leading(&[ki, kj, ka])?;
                ht2.1
                    .set_leading(&[ki, kj, ka], &divide(&x, eia, ejb, oa, ob, va, vb)?)?;
                let x = ht2.2.slice_leading(&[ki, kj, ka])?;
                ht2.2
                    .set_leading(&[ki, kj, ka], &divide(&x, eib, ejb, ob, ob, vb, vb)?)?;
            }
        }
    }

    Ok((ht1, ht2))
}

/// `:229-345` — every term that contracts a `Wovvo`-family intermediate, plus
/// the `t1·t1·eri` companions and the two `u2` antisymmetrisations.
#[allow(clippy::too_many_lines)]
pub(crate) fn wovvo_terms(
    ht2: &mut UT2,
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    w: &UWovvo,
    kconserv: &Kconserv,
) -> Result<(), PbcCcError> {
    let nkpts = eris.nkpts;
    let (oa, ob) = (eris.nocc.0, eris.nocc.1);
    let (va, vb) = (eris.nvir.0, eris.nvir.1);

    // `:233-241`
    for kx in 0..nkpts {
        for kw in 0..nkpts {
            for kz in 0..nkpts {
                let kv = kconserv.get(kx, kz, kw) as usize;
                for ku in 0..nkpts {
                    let ky = kconserv.get(kw, kv, ku) as usize;
                    add_at(
                        &mut ht2.1,
                        [kx, ky, kz],
                        &einsum(
                            "imae,mebj->ijab",
                            &[
                                &t2.0.slice_leading(&[kx, kw, kz])?,
                                &w.ab.get([kw, kv, ku])?,
                            ],
                        )?,
                        1.0,
                    )?;
                    add_at(
                        &mut ht2.1,
                        [kx, ky, kz],
                        &einsum(
                            "imae,mebj->ijab",
                            &[
                                &t2.1.slice_leading(&[kx, kw, kz])?,
                                &w.bb.get([kw, kv, ku])?,
                            ],
                        )?,
                        1.0,
                    )?;
                }
            }
        }
    }
    // `:249` — `Ht2ab -= einsum('xie,yma,xyzemjb->xzyijab', t1a, t1a, voOV.conj())`
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            for kz in 0..nkpts {
                add_at(
                    &mut ht2.1,
                    [kx, kz, ky],
                    &einsum(
                        "ie,ma,emjb->ijab",
                        &[
                            &t1.0.slice_leading(&[kx])?,
                            &t1.0.slice_leading(&[ky])?,
                            &eris.blk(b::voOV, kx, ky, kz)?.conj(),
                        ],
                    )?,
                    -1.0,
                )?;
            }
        }
    }
    // `:260-263`
    for km in 0..nkpts {
        for ke in 0..nkpts {
            for kb_ in 0..nkpts {
                let kj = kconserv.get(km, ke, kb_) as usize;
                let waa = w.aa.get([km, ke, kb_])?;
                let wba = w.ba.get([km, ke, kb_])?;
                for kx in 0..nkpts {
                    add_at(
                        &mut ht2.1,
                        [kj, kx, kb_],
                        &einsum(
                            "miea,mebj->jiba",
                            &[&t2.1.slice_leading(&[km, kx, ke])?, &waa],
                        )?,
                        1.0,
                    )?;
                    add_at(
                        &mut ht2.1,
                        [kj, kx, kb_],
                        &einsum(
                            "miea,mebj->jiba",
                            &[&t2.2.slice_leading(&[km, kx, ke])?, &wba],
                        )?,
                        1.0,
                    )?;
                }
            }
        }
    }
    // `:266-269`
    for kz in 0..nkpts {
        for ku in 0..nkpts {
            for kw in 0..nkpts {
                let kx = kconserv.get(kz, kw, ku) as usize;
                let ky = kconserv.get(kz, kx, ku) as usize;
                add_at(
                    &mut ht2.1,
                    [ky, kx, ku],
                    &einsum(
                        "ie,ma,bjme->jiba",
                        &[
                            &t1.1.slice_leading(&[kx])?,
                            &t1.1.slice_leading(&[kz])?,
                            &eris.blk(b::voOV, ku, kw, kz)?,
                        ],
                    )?,
                    -1.0,
                )?;
            }
        }
    }
    // `:278-280`
    for km in 0..nkpts {
        for ke in 0..nkpts {
            for kb_ in 0..nkpts {
                let kj = kconserv.get(km, ke, kb_) as usize;
                let wbaab = w.baab.get([km, ke, kb_])?;
                for kx in 0..nkpts {
                    add_at(
                        &mut ht2.1,
                        [kx, kj, kb_],
                        &einsum(
                            "imea,mebj->ijba",
                            &[&t2.1.slice_leading(&[kx, km, ke])?, &wbaab],
                        )?,
                        1.0,
                    )?;
                }
            }
        }
    }
    // `:283-286`
    for kz in 0..nkpts {
        for ku in 0..nkpts {
            for kw in 0..nkpts {
                let kx = kconserv.get(kz, kw, ku) as usize;
                let ky = kconserv.get(kz, kx, ku) as usize;
                add_at(
                    &mut ht2.1,
                    [kx, ky, ku],
                    &einsum(
                        "ie,ma,mjbe->ijba",
                        &[
                            &t1.0.slice_leading(&[kx])?,
                            &t1.1.slice_leading(&[kz])?,
                            &eris.blk(b::OOvv, kz, kw, ku)?,
                        ],
                    )?,
                    -1.0,
                )?;
            }
        }
    }
    // `:290-296`
    for kx in 0..nkpts {
        for kw in 0..nkpts {
            for kz in 0..nkpts {
                let kv = kconserv.get(kx, kz, kw) as usize;
                for ku in 0..nkpts {
                    let ky = kconserv.get(kw, kv, ku) as usize;
                    add_at(
                        &mut ht2.1,
                        [ky, kx, kz],
                        &einsum(
                            "miae,mebj->jiab",
                            &[
                                &t2.1.slice_leading(&[kw, kx, kz])?,
                                &w.abba.get([kw, kv, ku])?,
                            ],
                        )?,
                        1.0,
                    )?;
                }
            }
        }
    }
    // `:298-301`
    for kz in 0..nkpts {
        for ku in 0..nkpts {
            for kw in 0..nkpts {
                let kx = kconserv.get(kz, kw, ku) as usize;
                let ky = kconserv.get(kz, kx, ku) as usize;
                add_at(
                    &mut ht2.1,
                    [ky, kx, kz],
                    &einsum(
                        "ie,ma,mjbe->jiab",
                        &[
                            &t1.1.slice_leading(&[kx])?,
                            &t1.0.slice_leading(&[kz])?,
                            &eris.blk(b::ooVV, kz, kw, ku)?,
                        ],
                    )?,
                    -1.0,
                )?;
            }
        }
    }

    // ---- u2aa (`:303-333`)
    let mut u2aa = ZArr::zeros(&[nkpts, nkpts, nkpts, oa, oa, va, va]);
    for kx in 0..nkpts {
        for kw in 0..nkpts {
            for kz in 0..nkpts {
                let kv = kconserv.get(kx, kz, kw) as usize;
                for ku in 0..nkpts {
                    let ky = kconserv.get(kw, kv, ku) as usize;
                    add_at(
                        &mut u2aa,
                        [kx, ky, kz],
                        &einsum(
                            "imae,mebj->ijab",
                            &[
                                &t2.0.slice_leading(&[kx, kw, kz])?,
                                &w.aa.get([kw, kv, ku])?,
                            ],
                        )?,
                        1.0,
                    )?;
                    add_at(
                        &mut u2aa,
                        [kx, ky, kz],
                        &einsum(
                            "imae,mebj->ijab",
                            &[
                                &t2.1.slice_leading(&[kx, kw, kz])?,
                                &w.ba.get([kw, kv, ku])?,
                            ],
                        )?,
                        1.0,
                    )?;
                }
            }
        }
    }
    for kz in 0..nkpts {
        for ku in 0..nkpts {
            for kw in 0..nkpts {
                let kx = kconserv.get(kz, kw, ku) as usize;
                let ky = kconserv.get(kz, kx, ku) as usize;
                let (ix, mz) = (t1.0.slice_leading(&[kx])?, t1.0.slice_leading(&[kz])?);
                add_at(
                    &mut u2aa,
                    [kx, ky, kz],
                    &einsum(
                        "ie,ma,mjbe->ijab",
                        &[&ix, &mz, &eris.blk(b::oovv, kz, kw, ku)?],
                    )?,
                    1.0,
                )?;
                add_at(
                    &mut u2aa,
                    [kx, ky, kz],
                    &einsum(
                        "ie,ma,bjme->ijab",
                        &[&ix, &mz, &eris.blk(b::voov, ku, kw, kz)?],
                    )?,
                    -1.0,
                )?;
            }
        }
    }
    for ky in 0..nkpts {
        for kx in 0..nkpts {
            for ku in 0..nkpts {
                let kz = kconserv.get(ky, ku, kx) as usize;
                add_at(
                    &mut u2aa,
                    [kx, ky, kz],
                    &einsum(
                        "ie,bjae->ijab",
                        &[&t1.0.slice_leading(&[kx])?, &eris.blk(b::vovv, ku, ky, kz)?],
                    )?,
                    1.0,
                )?;
                add_at(
                    &mut u2aa,
                    [kx, ky, kz],
                    &einsum(
                        "ma,imjb->ijab",
                        &[
                            &t1.0.slice_leading(&[kz])?,
                            &eris.blk(b::ooov, kx, kz, ky)?.conj(),
                        ],
                    )?,
                    -1.0,
                )?;
            }
        }
    }
    antisymmetrise(&mut u2aa, nkpts, kconserv)?;
    ht2.0.add_assign(&u2aa)?;

    // ---- u2bb (`:335-363`)
    let mut u2bb = ZArr::zeros(&[nkpts, nkpts, nkpts, ob, ob, vb, vb]);
    for kx in 0..nkpts {
        for kw in 0..nkpts {
            for kz in 0..nkpts {
                let kv = kconserv.get(kx, kz, kw) as usize;
                for ku in 0..nkpts {
                    let ky = kconserv.get(kw, kv, ku) as usize;
                    add_at(
                        &mut u2bb,
                        [kx, ky, kz],
                        &einsum(
                            "imae,mebj->ijab",
                            &[
                                &t2.2.slice_leading(&[kx, kw, kz])?,
                                &w.bb.get([kw, kv, ku])?,
                            ],
                        )?,
                        1.0,
                    )?;
                    add_at(
                        &mut u2bb,
                        [kx, ky, kz],
                        &einsum(
                            "miea,mebj->ijab",
                            &[
                                &t2.1.slice_leading(&[kw, kx, kv])?,
                                &w.ab.get([kw, kv, ku])?,
                            ],
                        )?,
                        1.0,
                    )?;
                }
            }
        }
    }
    for kz in 0..nkpts {
        for ku in 0..nkpts {
            for kw in 0..nkpts {
                let kx = kconserv.get(kz, kw, ku) as usize;
                let ky = kconserv.get(kz, kx, ku) as usize;
                let (ix, mz) = (t1.1.slice_leading(&[kx])?, t1.1.slice_leading(&[kz])?);
                add_at(
                    &mut u2bb,
                    [kx, ky, kz],
                    &einsum(
                        "ie,ma,mjbe->ijab",
                        &[&ix, &mz, &eris.blk(b::OOVV, kz, kw, ku)?],
                    )?,
                    1.0,
                )?;
                add_at(
                    &mut u2bb,
                    [kx, ky, kz],
                    &einsum(
                        "ie,ma,bjme->ijab",
                        &[&ix, &mz, &eris.blk(b::VOOV, ku, kw, kz)?],
                    )?,
                    -1.0,
                )?;
            }
        }
    }
    for ky in 0..nkpts {
        for kx in 0..nkpts {
            for ku in 0..nkpts {
                let kz = kconserv.get(ky, ku, kx) as usize;
                add_at(
                    &mut u2bb,
                    [kx, ky, kz],
                    &einsum(
                        "ie,bjae->ijab",
                        &[&t1.1.slice_leading(&[kx])?, &eris.blk(b::VOVV, ku, ky, kz)?],
                    )?,
                    1.0,
                )?;
            }
        }
    }
    // `:357` — written whole-array upstream, at the `[kx,kz,ky]` ERI address.
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            for kz in 0..nkpts {
                add_at(
                    &mut u2bb,
                    [kx, ky, kz],
                    &einsum(
                        "ma,imjb->ijab",
                        &[
                            &t1.1.slice_leading(&[kz])?,
                            &eris.blk(b::OOOV, kx, kz, ky)?.conj(),
                        ],
                    )?,
                    -1.0,
                )?;
            }
        }
    }
    antisymmetrise(&mut u2bb, nkpts, kconserv)?;
    ht2.2.add_assign(&u2bb)?;

    // ---- the four remaining `ab` one-particle terms (`:365-385`)
    for ky in 0..nkpts {
        for kx in 0..nkpts {
            for ku in 0..nkpts {
                let kz = kconserv.get(ky, ku, kx) as usize;
                add_at(
                    &mut ht2.1,
                    [kx, ky, kz],
                    &einsum(
                        "ie,bjae->ijab",
                        &[&t1.0.slice_leading(&[kx])?, &eris.blk(b::VOvv, ku, ky, kz)?],
                    )?,
                    1.0,
                )?;
                add_at(
                    &mut ht2.1,
                    [kx, ky, kz],
                    &einsum(
                        "je,aibe->ijab",
                        &[&t1.1.slice_leading(&[ky])?, &eris.blk(b::voVV, kz, kx, ku)?],
                    )?,
                    1.0,
                )?;
            }
        }
    }
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            for kz in 0..nkpts {
                add_at(
                    &mut ht2.1,
                    [kx, ky, kz],
                    &einsum(
                        "ma,imjb->ijab",
                        &[
                            &t1.0.slice_leading(&[kz])?,
                            &eris.blk(b::ooOV, kx, kz, ky)?.conj(),
                        ],
                    )?,
                    -1.0,
                )?;
            }
        }
    }
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            for ku in 0..nkpts {
                let kz = kconserv.get(kx, ku, ky) as usize;
                add_at(
                    &mut ht2.1,
                    [kx, ky, kz],
                    &einsum(
                        "mb,jmia->ijab",
                        &[
                            &t1.1.slice_leading(&[ku])?,
                            &eris.blk(b::OOov, ky, ku, kx)?.conj(),
                        ],
                    )?,
                    -1.0,
                )?;
            }
        }
    }
    Ok(())
}

/// `u = u - u.transpose(1,0,2,4,3,5,6)` then
/// `u = u - einsum('xyzijab,xyzu->xyuijba', u, P)` — `:331-332` / `:361-362`.
///
/// Both read PRE-update values on the right, so each pass works from a
/// complete copy.
fn antisymmetrise(u: &mut ZArr, nkpts: usize, kconserv: &Kconserv) -> Result<(), PbcCcError> {
    let src = u.clone();
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            for kz in 0..nkpts {
                let t = src.slice_leading(&[ky, kx, kz])?.transpose(&[1, 0, 2, 3])?;
                add_at(u, [kx, ky, kz], &t, -1.0)?;
            }
        }
    }
    let src = u.clone();
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            for ku in 0..nkpts {
                // `P[x,y,z,u] = 1` iff `u == kconserv[x,z,y]`, i.e.
                // `z == kconserv[x,u,y]`.
                let kz = kconserv.get(kx, ku, ky) as usize;
                let t = src.slice_leading(&[kx, ky, kz])?.transpose(&[0, 1, 3, 2])?;
                add_at(u, [kx, ky, ku], &t, -1.0)?;
            }
        }
    }
    Ok(())
}

/// `f[k][np.diag_indices(n)] -= e + shift` — `:96-100`.
fn shift_diag(f: &mut ZArr, k: usize, e: &[f64], shift: f64) -> Result<(), PbcCcError> {
    let n = e.len();
    let mut blk = f.slice_leading(&[k])?;
    for p in 0..n {
        blk.data_mut().re[p * n + p] -= e[p] + shift;
    }
    f.set_leading(&[k], &blk)
}

fn divide_in_place(x: &mut ZArr, e: &[f64]) {
    for i in 0..x.len() {
        x.data_mut().re[i] /= e[i];
        x.data_mut().im[i] /= e[i];
    }
}

fn add_at(t: &mut ZArr, k: [usize; 3], v: &ZArr, s: f64) -> Result<(), PbcCcError> {
    let mut cur = t.slice_leading(&k)?;
    cur.zip_assign(v, s)?;
    t.set_leading(&k, &cur)
}

fn add_at1(t: &mut ZArr, k: usize, v: &ZArr, s: f64) -> Result<(), PbcCcError> {
    let mut cur = t.slice_leading(&[k])?;
    cur.zip_assign(v, s)?;
    t.set_leading(&[k], &cur)
}

/// The five amplitude arrays packed for CDIIS, `[re…, im…]` per array in
/// `amplitudes_to_vector`'s order (`:529-531`).
#[derive(Clone)]
struct KuAmplitudes {
    flat: Vec<f64>,
}

impl KuAmplitudes {
    fn from_amplitudes(t1: &UT1, t2: &UT2) -> Self {
        let mut flat = Vec::new();
        for a in [&t1.0, &t1.1] {
            flat.extend_from_slice(&a.data().re);
            flat.extend_from_slice(&a.data().im);
        }
        for a in [&t2.0, &t2.1, &t2.2] {
            flat.extend_from_slice(&a.data().re);
            flat.extend_from_slice(&a.data().im);
        }
        Self { flat }
    }

    fn to_amplitudes(&self, t1: &UT1, t2: &UT2) -> (UT1, UT2) {
        let mut off = 0_usize;
        let mut take = |src: &ZArr| -> ZArr {
            let n = src.len();
            let mut out = src.clone();
            out.data_mut().re.copy_from_slice(&self.flat[off..off + n]);
            out.data_mut()
                .im
                .copy_from_slice(&self.flat[off + n..off + 2 * n]);
            off += 2 * n;
            out
        };
        let a1 = take(&t1.0);
        let b1 = take(&t1.1);
        let aa = take(&t2.0);
        let ab = take(&t2.1);
        let bb = take(&t2.2);
        ((a1, b1), (aa, ab, bb))
    }

    fn residual(&self, prev: &Self) -> Vec<f64> {
        self.flat
            .iter()
            .zip(&prev.flat)
            .map(|(a, b)| a - b)
            .collect()
    }
}

impl pyscf_diis::DiisStorable for KuAmplitudes {
    fn as_flat(&self) -> &[f64] {
        &self.flat
    }
    fn from_flat(&mut self, slice: &[f64]) {
        self.flat.copy_from_slice(slice);
    }
    fn dot(&self, other: &Self) -> f64 {
        pyscf_algebra::oracle_dot(&self.flat, &other.flat)
    }
    fn len(&self) -> usize {
        self.flat.len()
    }
}

/// `pyscf.cc.ccsd.kernel` driven with the unrestricted k-point `update_amps`
/// and `energy`.
///
/// # Errors
/// Propagates every intermediate build, and the DIIS solve.
pub fn kernel(
    pool: &Arc<ZWorkspacePool>,
    eris: &KuEris,
    padded: (&PaddedMos, &PaddedMos),
    kconserv: &Kconserv,
    opts: &KrccsdOpts,
) -> Result<KuccsdResult, PbcCcError> {
    let (emp2, mut t1, mut t2) = init_amps(eris, padded, kconserv)?;
    let mut eccsd = energy(&t1, &t2, eris)?;
    let mut converged = false;
    let mut cycles = 0_usize;

    let mut diis: Option<Diis<KuAmplitudes>> = if opts.diis {
        Some(Diis::new(opts.diis_space))
    } else {
        None
    };

    for istep in 0..opts.max_cycle {
        cycles = istep + 1;
        let (mut t1new, mut t2new) = update_amps(pool, &t1, &t2, eris, padded, kconserv, opts)?;

        let cur = KuAmplitudes::from_amplitudes(&t1new, &t2new);
        let prev = KuAmplitudes::from_amplitudes(&t1, &t2);
        let res = cur.residual(&prev);
        let normt = pyscf_algebra::oracle_dot(&res, &res).sqrt();

        if opts.iterative_damping < 1.0 {
            let a = opts.iterative_damping;
            for (n, o) in [(&mut t1new.0, &t1.0), (&mut t1new.1, &t1.1)] {
                n.scale(a);
                n.zip_assign(o, 1.0 - a)?;
            }
            for (n, o) in [
                (&mut t2new.0, &t2.0),
                (&mut t2new.1, &t2.1),
                (&mut t2new.2, &t2.2),
            ] {
                n.scale(a);
                n.zip_assign(o, 1.0 - a)?;
            }
        }

        t1 = t1new;
        t2 = t2new;

        if let Some(stack) = diis.as_mut()
            && istep >= opts.diis_start_cycle
        {
            let cur = KuAmplitudes::from_amplitudes(&t1, &t2);
            let err = cur.residual(&prev);
            let extrap = stack
                .extrapolate(cur, err)
                .map_err(|e| PbcCcError::Algebra(format!("amplitude DIIS: {e}")))?;
            let (a, b_) = extrap.to_amplitudes(&t1, &t2);
            t1 = a;
            t2 = b_;
        }

        let eold = eccsd;
        eccsd = energy(&t1, &t2, eris)?;
        if (eccsd - eold).abs() < opts.conv_tol && normt < opts.conv_tol_normt {
            converged = true;
            break;
        }
    }

    Ok(KuccsdResult {
        e_corr: eccsd,
        emp2,
        converged,
        cycles,
        t1,
        t2,
    })
}

/// `KUCCSD` — the object `kccsd_uhf.py:638` declares, tying a converged
/// UNRESTRICTED k-point mean field to the amplitude iteration.
///
/// The restricted sibling is [`crate::kccsd_rhf::Krccsd`] and the surface is
/// deliberately the same shape: `new` / `ao2mo` / `kernel` / `kernel_with`.
/// What differs is that everything is a pair — two `PaddedMos`, two density
/// matrices, two Fock matrices — because `KUHF` genuinely has two independent
/// orbital sets, not one set used twice.
#[derive(Debug)]
pub struct Kuccsd<'a> {
    pub with_df: &'a dyn pyscf_pbc_df::PeriodicDf,
    pub khelper: pyscf_pbc_lib::KptsHelper,
    /// `(alpha, beta)` padded MO sets.
    pub padded: (PaddedMos, PaddedMos),
    pub frozen: pyscf_pbc_mp::FrozenK,
    pub opts: KrccsdOpts,
    pub eris_opts: crate::keris::KErisOpts,
    dm: (Vec<pyscf_algebra::CTensor>, Vec<pyscf_algebra::CTensor>),
    e_hf: f64,
    converged: bool,
}

impl<'a> Kuccsd<'a> {
    /// Build from a converged UNRESTRICTED k-point SCF.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] if the SCF is not two spin channels over this
    /// builder's k-points, or if the padding surface refuses.
    pub fn new(
        scf: &'a pyscf_pbc_scf::KScfResult,
        with_df: &'a dyn pyscf_pbc_df::PeriodicDf,
    ) -> Result<Self, PbcCcError> {
        if scf.nset != 2 || scf.nkpts != with_df.kpts().len() {
            return Err(PbcCcError::Shape(
                "KUCCSD needs two unrestricted SCF channels over with_df.kpts()".into(),
            ));
        }
        let cell = with_df.cell();
        let nao = cell.mol.nao_nr;
        let frozen = pyscf_pbc_mp::FrozenK::default();
        let mut padded = Vec::with_capacity(2);
        let mut dm = Vec::with_capacity(2);
        for set in 0..2 {
            let mf = pyscf_pbc_mp::spin_block(scf, set)
                .map_err(|e| PbcCcError::Shape(format!("spin_block: {e}")))?;
            let raw: Result<Vec<pyscf_pbc_df::MoCoeff>, _> = mf
                .mo_coeff
                .iter()
                .zip(mf.mo_occ)
                .map(|(c, occ)| pyscf_pbc_mp::mo_coeff_from_kscf(c, nao, occ.len()))
                .collect();
            let raw = raw.map_err(|e| PbcCcError::Shape(format!("mo_coeff_from_kscf: {e}")))?;
            padded.push(
                pyscf_pbc_mp::add_padding(&raw, mf.mo_energy, mf.mo_occ, &frozen)
                    .map_err(|e| PbcCcError::Shape(format!("add_padding: {e}")))?,
            );
            // As in the restricted driver, the density matrix comes from the
            // mean field's OWN unpadded orbitals (`kccsd_uhf.py:857`).
            dm.push(pyscf_pbc_scf::krdm::make_rdm1(mf.mo_coeff, mf.mo_occ, nao));
        }
        let mut padded = padded.into_iter();
        let mut dm = dm.into_iter();
        let pa = padded.next().ok_or_else(|| shape("no alpha MOs"))?;
        let pb = padded.next().ok_or_else(|| shape("no beta MOs"))?;
        let da = dm.next().ok_or_else(|| shape("no alpha density"))?;
        let db = dm.next().ok_or_else(|| shape("no beta density"))?;
        Ok(Self {
            with_df,
            khelper: pyscf_pbc_lib::KptsHelper::without_symm_map(&cell.a, with_df.kpts()),
            padded: (pa, pb),
            frozen,
            opts: KrccsdOpts::default(),
            eris_opts: crate::keris::KErisOpts::default(),
            dm: (da, db),
            e_hf: scf.e_tot,
            converged: scf.converged,
        })
    }

    /// `cc.ao2mo()` — build the twenty-six blocks.
    ///
    /// # Errors
    /// Propagates the density-fitting builder and the arena.
    pub fn ao2mo(&self) -> Result<KuEris, PbcCcError> {
        let (fock, mo_energy, madelung) = KuEris::build_fock(
            self.with_df.cell(),
            self.with_df,
            (&self.padded.0, &self.padded.1),
            (&self.dm.0, &self.dm.1),
            self.eris_opts,
        )?;
        KuEris::from_parts(
            self.with_df,
            &self.khelper.kconserv,
            (&self.padded.0, &self.padded.1),
            fock,
            mo_energy,
            madelung,
            self.eris_opts,
        )
    }

    /// Run the amplitude iteration.
    ///
    /// # Errors
    /// [`PbcCcError::NotConverged`] if the reference SCF did not converge;
    /// otherwise propagates the kernel.
    pub fn kernel(&self) -> Result<KuccsdResult, PbcCcError> {
        if !self.converged {
            return Err(PbcCcError::NotConverged {
                what: "the reference k-point SCF",
                detail: "KUCCSD refuses an unconverged mean field".into(),
            });
        }
        let eris = self.ao2mo()?;
        self.kernel_with(&eris)
    }

    /// Run the amplitude iteration on an already-built [`KuEris`].
    ///
    /// # Errors
    /// Propagates the kernel.
    pub fn kernel_with(&self, eris: &KuEris) -> Result<KuccsdResult, PbcCcError> {
        let pool = Arc::new(ZWorkspacePool::new(
            (self.opts.max_memory * 1e6).max(0.0) as usize
        ));
        kernel(
            &pool,
            eris,
            (&self.padded.0, &self.padded.1),
            &self.khelper.kconserv,
            &self.opts,
        )
    }

    /// The mean-field total energy the correlation energy adds to.
    pub fn e_hf(&self) -> f64 {
        self.e_hf
    }
}

fn shape(msg: &str) -> PbcCcError {
    PbcCcError::Shape(msg.to_string())
}

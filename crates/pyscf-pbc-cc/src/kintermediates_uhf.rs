//! `kintermediates_uhf` — the k-point UNRESTRICTED CC intermediates
//! (plan 16-06 Task 2; `pyscf/pbc/cc/kintermediates_uhf.py:26-580`).
//!
//! Only the eight functions `KUCCSD`'s ground state contracts against are here:
//! [`make_tau`], [`make_tau2`], [`cc_fvv`], [`cc_foo`], [`cc_fov`],
//! [`cc_woooo`], [`cc_wvvvv_half`] and [`cc_wovvo`]. The twelve below
//! `kintermediates_uhf.py:590` (`Foo`, `Fvv`, `Fov`, `Wvvov`, `Wvvvo`, `Woooo`,
//! `Woovo`, `Wooov`, `Wovvo`, `W1oovv`, `W2oovv`, `Woovv`) belong to
//! EOM-KUCCSD, and `_eri_spin2spatial` / `_eri_spatial2spin` to its ERI
//! plumbing; they are plan 16-11's, not this one's, and are NOT stubbed here.
//!
//! # Three transcription rules, all of them load-bearing
//!
//! **1. The k-index of a block is the plain `(kp, kq, kr)` of the chemists'
//! `(pq|rs)`** — see [`crate::kueris`]'s module doc. Nothing in this file
//! applies `KEris`'s `(0,2,1,3)`.
//!
//! **2. `numpy.einsum` never conjugates** (`16-CONTEXT §3.2`), so every
//! [`einsum`] call below is the UNCONJUGATED ordered complex sum, and the
//! conjugations are the explicit `.conj()` calls upstream writes — twenty of
//! them, all in [`cc_fvv`], [`cc_wvvvv_half`] and [`cc_wovvo`].
//!
//! **3. Upstream's whole-array `einsum`s over free k-axes are rewritten as
//! explicit k-loops, never as "gather the free axis and contract".** That is
//! not a style preference: 16-05 lost half a day to `cc_Wovvo` gathering
//! `oovv[:,km,ke]` where `oovv[km,:,ke]` was meant — the two have the SAME
//! SHAPE, so nothing failed until `t2new` came out `7.7e-4` wrong. A `for kx`
//! loop with the k-index written at every use site cannot make that mistake.

use std::sync::Arc;

use pyscf_pbc_lib::Kconserv;
use pyscf_runtime::ZWorkspacePool;

use crate::error::PbcCcError;
use crate::ktensor::KBlocks;
use crate::kueris::KuEris;
use crate::zarr::{ZArr, einsum};

/// Upstream's block names, spelled exactly as `kccsd_uhf.py` spells them.
#[allow(non_upper_case_globals)]
pub mod b {
    use crate::kueris::{UBlk, UKind::*, UPass::*};

    pub const oooo: UBlk = UBlk::Pair(Aaaa, Oooo);
    pub const ooov: UBlk = UBlk::Pair(Aaaa, Ooov);
    pub const oovv: UBlk = UBlk::Pair(Aaaa, Oovv);
    pub const ovov: UBlk = UBlk::Pair(Aaaa, Ovov);
    pub const voov: UBlk = UBlk::Pair(Aaaa, Voov);
    pub const vovv: UBlk = UBlk::Pair(Aaaa, Vovv);

    pub const OOOO: UBlk = UBlk::Pair(Bbbb, Oooo);
    pub const OOOV: UBlk = UBlk::Pair(Bbbb, Ooov);
    pub const OOVV: UBlk = UBlk::Pair(Bbbb, Oovv);
    pub const OVOV: UBlk = UBlk::Pair(Bbbb, Ovov);
    pub const VOOV: UBlk = UBlk::Pair(Bbbb, Voov);
    pub const VOVV: UBlk = UBlk::Pair(Bbbb, Vovv);

    pub const ooOO: UBlk = UBlk::Pair(AaBb, Oooo);
    pub const ooOV: UBlk = UBlk::Pair(AaBb, Ooov);
    pub const ooVV: UBlk = UBlk::Pair(AaBb, Oovv);
    pub const ovOV: UBlk = UBlk::Pair(AaBb, Ovov);
    pub const voOV: UBlk = UBlk::Pair(AaBb, Voov);
    pub const voVV: UBlk = UBlk::Pair(AaBb, Vovv);

    pub const OOov: UBlk = UBlk::Pair(BbAa, Ooov);
    pub const OOvv: UBlk = UBlk::Pair(BbAa, Oovv);
    pub const OVov: UBlk = UBlk::Pair(BbAa, Ovov);
    pub const VOov: UBlk = UBlk::Pair(BbAa, Voov);
    pub const VOvv: UBlk = UBlk::Pair(BbAa, Vovv);

    pub const vvvv: UBlk = UBlk::Quad(Aaaa);
    pub const VVVV: UBlk = UBlk::Quad(Bbbb);
    pub const vvVV: UBlk = UBlk::Quad(AaBb);
}

/// `(t1a, t1b)`, each `[nkpts, nocc, nvir]`.
pub type UT1 = (ZArr, ZArr);
/// `(t2aa, t2ab, t2bb)`, each `[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]`.
pub type UT2 = (ZArr, ZArr, ZArr);

/// `make_tau(cc, t2, t1, t1p, fac)` — `kintermediates_uhf.py:26-47`.
///
/// # Errors
/// Propagates every shape check.
pub fn make_tau(t2: &UT2, t1: &UT1, t1p: &UT1, fac: f64) -> Result<UT2, PbcCcError> {
    let nkpts = t2.0.shape()[0];
    let (mut tauaa, mut tauab, mut taubb) = (t2.0.clone(), t2.1.clone(), t2.2.clone());
    let h = 0.5 * fac;
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let (ia, ib) = (t1.0.slice_leading(&[ki])?, t1.1.slice_leading(&[ki])?);
            let (ja, jb) = (t1.0.slice_leading(&[kj])?, t1.1.slice_leading(&[kj])?);
            let (pia, pib) = (t1p.0.slice_leading(&[ki])?, t1p.1.slice_leading(&[ki])?);
            let (pja, pjb) = (t1p.0.slice_leading(&[kj])?, t1p.1.slice_leading(&[kj])?);

            // `:35-38`
            add_at(
                &mut tauaa,
                [ki, kj, ki],
                &einsum("ia,jb->ijab", &[&ia, &pja])?,
                h,
            )?;
            add_at(
                &mut tauaa,
                [ki, kj, kj],
                &einsum("ib,ja->ijab", &[&ia, &pja])?,
                -h,
            )?;
            add_at(
                &mut tauaa,
                [ki, kj, kj],
                &einsum("ja,ib->ijab", &[&ja, &pia])?,
                -h,
            )?;
            add_at(
                &mut tauaa,
                [ki, kj, ki],
                &einsum("jb,ia->ijab", &[&ja, &pia])?,
                h,
            )?;
            // `:40-43`
            add_at(
                &mut taubb,
                [ki, kj, ki],
                &einsum("ia,jb->ijab", &[&ib, &pjb])?,
                h,
            )?;
            add_at(
                &mut taubb,
                [ki, kj, kj],
                &einsum("ib,ja->ijab", &[&ib, &pjb])?,
                -h,
            )?;
            add_at(
                &mut taubb,
                [ki, kj, kj],
                &einsum("ja,ib->ijab", &[&jb, &pib])?,
                -h,
            )?;
            add_at(
                &mut taubb,
                [ki, kj, ki],
                &einsum("jb,ia->ijab", &[&jb, &pib])?,
                h,
            )?;
            // `:45-46`
            add_at(
                &mut tauab,
                [ki, kj, ki],
                &einsum("ia,jb->ijab", &[&ia, &pjb])?,
                h,
            )?;
            add_at(
                &mut tauab,
                [ki, kj, ki],
                &einsum("jb,ia->ijab", &[&jb, &pia])?,
                h,
            )?;
        }
    }
    Ok((tauaa, tauab, taubb))
}

/// `make_tau2(cc, t2, t1, t1p, fac)` — `:49-66`. The two "exchange" terms
/// [`make_tau`] carries are absent; that is the whole difference.
///
/// # Errors
/// Propagates every shape check.
pub fn make_tau2(t2: &UT2, t1: &UT1, t1p: &UT1, fac: f64) -> Result<UT2, PbcCcError> {
    let nkpts = t2.0.shape()[0];
    let (mut tauaa, mut tauab, mut taubb) = (t2.0.clone(), t2.1.clone(), t2.2.clone());
    let h = 0.5 * fac;
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let (ia, ib) = (t1.0.slice_leading(&[ki])?, t1.1.slice_leading(&[ki])?);
            let (ja, jb) = (t1.0.slice_leading(&[kj])?, t1.1.slice_leading(&[kj])?);
            let (pia, pib) = (t1p.0.slice_leading(&[ki])?, t1p.1.slice_leading(&[ki])?);
            let (pja, pjb) = (t1p.0.slice_leading(&[kj])?, t1p.1.slice_leading(&[kj])?);
            add_at(
                &mut tauaa,
                [ki, kj, ki],
                &einsum("ia,jb->ijab", &[&ia, &pja])?,
                h,
            )?;
            add_at(
                &mut tauaa,
                [ki, kj, ki],
                &einsum("jb,ia->ijab", &[&ja, &pia])?,
                h,
            )?;
            add_at(
                &mut taubb,
                [ki, kj, ki],
                &einsum("ia,jb->ijab", &[&ib, &pjb])?,
                h,
            )?;
            add_at(
                &mut taubb,
                [ki, kj, ki],
                &einsum("jb,ia->ijab", &[&jb, &pib])?,
                h,
            )?;
            add_at(
                &mut tauab,
                [ki, kj, ki],
                &einsum("ia,jb->ijab", &[&ia, &pjb])?,
                h,
            )?;
            add_at(
                &mut tauab,
                [ki, kj, ki],
                &einsum("jb,ia->ijab", &[&jb, &pia])?,
                h,
            )?;
        }
    }
    Ok((tauaa, tauab, taubb))
}

/// `cc_Fvv` — `:68-110`. Returns `(fa, fb)`, `[nkpts, nvir, nvir]` each.
///
/// # Errors
/// Propagates the ERI access and every shape check.
pub fn cc_fvv(
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    kconserv: &Kconserv,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let nkpts = eris.nkpts;
    let mut fa = ZArr::zeros(&[nkpts, eris.nvir.0, eris.nvir.0]);
    let mut fb = ZArr::zeros(&[nkpts, eris.nvir.1, eris.nvir.1]);
    let tt = make_tau(t2, t1, t1, 0.5)?;

    for ka in 0..nkpts {
        let mut acc_a = eris.fvv(false, ka)?;
        let mut acc_b = eris.fvv(true, ka)?;
        // `:87-88`
        acc_a.sub_assign(&einsum_half(
            "me,ma->ae",
            &[&eris.fov(false, ka)?, &t1.0.slice_leading(&[ka])?],
        )?)?;
        acc_b.sub_assign(&einsum_half(
            "me,ma->ae",
            &[&eris.fov(true, ka)?, &t1.1.slice_leading(&[ka])?],
        )?)?;
        for km in 0..nkpts {
            let (t1am, t1bm) = (t1.0.slice_leading(&[km])?, t1.1.slice_leading(&[km])?);
            // `:90-92`
            acc_a.add_assign(&einsum(
                "mf,fmea->ae",
                &[&t1am, &eris.blk(b::vovv, km, km, ka)?.conj()],
            )?)?;
            acc_a.sub_assign(&einsum(
                "mf,emfa->ae",
                &[&t1am, &eris.blk(b::vovv, ka, km, km)?.conj()],
            )?)?;
            acc_a.add_assign(&einsum(
                "mf,fmea->ae",
                &[&t1bm, &eris.blk(b::VOvv, km, km, ka)?.conj()],
            )?)?;
            // `:94-96`
            acc_b.add_assign(&einsum(
                "mf,fmea->ae",
                &[&t1bm, &eris.blk(b::VOVV, km, km, ka)?.conj()],
            )?)?;
            acc_b.sub_assign(&einsum(
                "mf,emfa->ae",
                &[&t1bm, &eris.blk(b::VOVV, ka, km, km)?.conj()],
            )?)?;
            acc_b.add_assign(&einsum(
                "mf,fmea->ae",
                &[&t1am, &eris.blk(b::voVV, km, km, ka)?.conj()],
            )?)?;

            for kn in 0..nkpts {
                let kf = kconserv.get(km, ka, kn) as usize;
                // `:99-102`
                let mut tmp = eris.blk(b::ovov, km, ka, kn)?;
                tmp.sub_assign(&eris.blk(b::ovov, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;
                acc_a.sub_assign(&einsum_half(
                    "mnaf,menf->ae",
                    &[&tt.0.slice_leading(&[km, kn, ka])?, &tmp],
                )?)?;
                acc_a.sub_assign(&einsum(
                    "mNaF,meNF->ae",
                    &[
                        &tt.1.slice_leading(&[km, kn, ka])?,
                        &eris.blk(b::ovOV, km, ka, kn)?,
                    ],
                )?)?;
                // `:104-106`
                let mut tmp = eris.blk(b::OVOV, km, ka, kn)?;
                tmp.sub_assign(&eris.blk(b::OVOV, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;
                acc_b.sub_assign(&einsum_half(
                    "mnaf,menf->ae",
                    &[&tt.2.slice_leading(&[km, kn, ka])?, &tmp],
                )?)?;
                acc_b.sub_assign(&einsum(
                    "MnFa,MFne->ae",
                    &[
                        &tt.1.slice_leading(&[km, kn, kf])?,
                        &eris.blk(b::ovOV, km, kf, kn)?,
                    ],
                )?)?;
            }
        }
        fa.set_leading(&[ka], &acc_a)?;
        fb.set_leading(&[ka], &acc_b)?;
    }
    Ok((fa, fb))
}

/// `cc_Foo` — `:113-157`. Returns `(fa, fb)`, `[nkpts, nocc, nocc]` each.
///
/// # Errors
/// As [`cc_fvv`].
pub fn cc_foo(
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    kconserv: &Kconserv,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let nkpts = eris.nkpts;
    let mut fa = ZArr::zeros(&[nkpts, eris.nocc.0, eris.nocc.0]);
    let mut fb = ZArr::zeros(&[nkpts, eris.nocc.1, eris.nocc.1]);
    let tt = make_tau(t2, t1, t1, 0.5)?;

    // `:132-144` — the one-body half, indexed by `ka` upstream (which is an
    // OCCUPIED label here despite the name; `:133` writes `fa[ka] += foo[ka]`).
    let mut accs_a = Vec::with_capacity(nkpts);
    let mut accs_b = Vec::with_capacity(nkpts);
    for ka in 0..nkpts {
        let mut acc_a = eris.foo(false, ka)?;
        let mut acc_b = eris.foo(true, ka)?;
        acc_a.add_assign(&einsum_half(
            "me,ne->mn",
            &[&eris.fov(false, ka)?, &t1.0.slice_leading(&[ka])?],
        )?)?;
        acc_b.add_assign(&einsum_half(
            "me,ne->mn",
            &[&eris.fov(true, ka)?, &t1.1.slice_leading(&[ka])?],
        )?)?;
        for km in 0..nkpts {
            let (t1am, t1bm) = (t1.0.slice_leading(&[km])?, t1.1.slice_leading(&[km])?);
            acc_a.add_assign(&einsum(
                "oa,mnoa->mn",
                &[&t1am, &eris.blk(b::ooov, ka, ka, km)?],
            )?)?;
            acc_a.add_assign(&einsum(
                "oa,mnoa->mn",
                &[&t1bm, &eris.blk(b::ooOV, ka, ka, km)?],
            )?)?;
            acc_a.sub_assign(&einsum(
                "oa,onma->mn",
                &[&t1am, &eris.blk(b::ooov, km, ka, ka)?],
            )?)?;
            acc_b.add_assign(&einsum(
                "oa,mnoa->mn",
                &[&t1bm, &eris.blk(b::OOOV, ka, ka, km)?],
            )?)?;
            acc_b.add_assign(&einsum(
                "oa,mnoa->mn",
                &[&t1am, &eris.blk(b::OOov, ka, ka, km)?],
            )?)?;
            acc_b.sub_assign(&einsum(
                "oa,onma->mn",
                &[&t1bm, &eris.blk(b::OOOV, km, ka, ka)?],
            )?)?;
        }
        accs_a.push(acc_a);
        accs_b.push(acc_b);
    }

    // `:146-157` — the two-body half, a SEPARATE triple loop upstream.
    for km in 0..nkpts {
        for kn in 0..nkpts {
            for ke in 0..nkpts {
                let kf = kconserv.get(km, ke, kn) as usize;
                let mut tmp = eris.blk(b::ovov, km, ke, kn)?;
                tmp.sub_assign(&eris.blk(b::ovov, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;
                accs_a[km].add_assign(&einsum_half(
                    "inef,menf->mi",
                    &[&tt.0.slice_leading(&[km, kn, ke])?, &tmp],
                )?)?;
                accs_a[km].add_assign(&einsum(
                    "iNeF,meNF->mi",
                    &[
                        &tt.1.slice_leading(&[km, kn, ke])?,
                        &eris.blk(b::ovOV, km, ke, kn)?,
                    ],
                )?)?;

                let mut tmp = eris.blk(b::OVOV, km, ke, kn)?;
                tmp.sub_assign(&eris.blk(b::OVOV, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;
                accs_b[km].add_assign(&einsum_half(
                    "INEF,MENF->MI",
                    &[&tt.2.slice_leading(&[km, kn, ke])?, &tmp],
                )?)?;
                accs_b[km].add_assign(&einsum(
                    "nIeF,neMF->MI",
                    &[
                        &tt.1.slice_leading(&[kn, km, ke])?,
                        &eris.blk(b::ovOV, kn, ke, km)?,
                    ],
                )?)?;
            }
        }
    }

    for k in 0..nkpts {
        fa.set_leading(&[k], &accs_a[k])?;
        fb.set_leading(&[k], &accs_b[k])?;
    }
    Ok((fa, fb))
}

/// `cc_Fov` — `:160-182`. Returns `(fa, fb)`, `[nkpts, nocc, nvir]` each.
///
/// # Errors
/// As [`cc_fvv`].
pub fn cc_fov(t1: &UT1, eris: &KuEris) -> Result<(ZArr, ZArr), PbcCcError> {
    let nkpts = eris.nkpts;
    let mut fa = ZArr::zeros(&[nkpts, eris.nocc.0, eris.nvir.0]);
    let mut fb = ZArr::zeros(&[nkpts, eris.nocc.1, eris.nvir.1]);
    for km in 0..nkpts {
        let mut acc_a = eris.fov(false, km)?;
        let mut acc_b = eris.fov(true, km)?;
        for kn in 0..nkpts {
            let (t1an, t1bn) = (t1.0.slice_leading(&[kn])?, t1.1.slice_leading(&[kn])?);
            acc_a.add_assign(&einsum(
                "nf,menf->me",
                &[&t1an, &eris.blk(b::ovov, km, km, kn)?],
            )?)?;
            acc_a.add_assign(&einsum(
                "nf,menf->me",
                &[&t1bn, &eris.blk(b::ovOV, km, km, kn)?],
            )?)?;
            acc_a.sub_assign(&einsum(
                "nf,mfne->me",
                &[&t1an, &eris.blk(b::ovov, km, kn, kn)?],
            )?)?;
            acc_b.add_assign(&einsum(
                "nf,menf->me",
                &[&t1bn, &eris.blk(b::OVOV, km, km, kn)?],
            )?)?;
            acc_b.add_assign(&einsum(
                "nf,nfme->me",
                &[&t1an, &eris.blk(b::ovOV, kn, kn, km)?],
            )?)?;
            acc_b.sub_assign(&einsum(
                "nf,mfne->me",
                &[&t1bn, &eris.blk(b::OVOV, km, kn, kn)?],
            )?)?;
        }
        fa.set_leading(&[km], &acc_a)?;
        fb.set_leading(&[km], &acc_b)?;
    }
    Ok((fa, fb))
}

/// `cc_Woooo` — `:185-223`. Returns `(Woooo, WooOO, WOOOO)`, tiered.
///
/// The two same-spin blocks are antisymmetrised at the end by
/// `W - W.transpose(2,1,0,5,4,3,6)` (`:222-223`), i.e.
/// `W[km,ki,kn] -= W_old[kn,ki,km].transpose(2,1,0,3)`. That reads the
/// PRE-antisymmetrisation values on both sides, so the two members of each
/// `{km, kn}` pair are updated together from a saved copy — an in-place
/// sequential sweep would feed a already-updated block back in.
///
/// # Errors
/// As [`cc_fvv`], plus the arena's HARD refusal.
pub fn cc_woooo(
    pool: &Arc<ZWorkspacePool>,
    max_memory_bytes: usize,
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    kconserv: &Kconserv,
) -> Result<(KBlocks, KBlocks, KBlocks), PbcCcError> {
    let nkpts = eris.nkpts;
    let (na, nb) = (eris.nocc.0, eris.nocc.1);
    let waa = KBlocks::with_budget(pool, nkpts, &[na, na, na, na], max_memory_bytes)?;
    let wab = KBlocks::with_budget(pool, nkpts, &[na, na, nb, nb], max_memory_bytes)?;
    let wbb = KBlocks::with_budget(pool, nkpts, &[nb, nb, nb, nb], max_memory_bytes)?;
    let tau = make_tau(t2, t1, t1, 1.0)?;

    for km in 0..nkpts {
        for kn in 0..nkpts {
            for ki in 0..nkpts {
                let kj = kconserv.get(km, ki, kn) as usize;
                let (t1ai, t1bi) = (t1.0.slice_leading(&[ki])?, t1.1.slice_leading(&[ki])?);
                let (t1aj, t1bj) = (t1.0.slice_leading(&[kj])?, t1.1.slice_leading(&[kj])?);

                // `:206-208` the bare integral
                let mut aa = eris.blk(b::oooo, km, ki, kn)?;
                let mut ab = eris.blk(b::ooOO, km, ki, kn)?;
                let mut bb = eris.blk(b::OOOO, km, ki, kn)?;

                // `:214` `tmp_aaaaJ[ki,kj]`, with the `(1,0,2,5,4,3)`
                // antisymmetrisation of `:201` folded in.
                aa.add_assign(&einsum(
                    "je,mine->minj",
                    &[&t1aj, &eris.blk(b::ooov, km, ki, kn)?],
                )?)?;
                aa.sub_assign(&einsum(
                    "ie,mjne->minj",
                    &[&t1ai, &eris.blk(b::ooov, km, kj, kn)?],
                )?)?;
                // `:215`
                bb.add_assign(&einsum(
                    "je,mine->minj",
                    &[&t1bj, &eris.blk(b::OOOV, km, ki, kn)?],
                )?)?;
                bb.sub_assign(&einsum(
                    "ie,mjne->minj",
                    &[&t1bi, &eris.blk(b::OOOV, km, kj, kn)?],
                )?)?;
                // `:216`
                ab.add_assign(&einsum(
                    "je,mine->minj",
                    &[&t1bj, &eris.blk(b::ooOV, km, ki, kn)?],
                )?)?;

                // `:218-220` the tau terms, summed over the free k-axis.
                for kx in 0..nkpts {
                    aa.add_assign(&einsum_scale(
                        "ijef,menf->minj",
                        &[
                            &tau.0.slice_leading(&[ki, kj, kx])?,
                            &eris.blk(b::ovov, km, kx, kn)?,
                        ],
                        0.25,
                    )?)?;
                    bb.add_assign(&einsum_scale(
                        "ijef,menf->minj",
                        &[
                            &tau.2.slice_leading(&[ki, kj, kx])?,
                            &eris.blk(b::OVOV, km, kx, kn)?,
                        ],
                        0.25,
                    )?)?;
                    ab.add_assign(&einsum_scale(
                        "ijef,menf->minj",
                        &[
                            &tau.1.slice_leading(&[ki, kj, kx])?,
                            &eris.blk(b::ovOV, km, kx, kn)?,
                        ],
                        0.5,
                    )?)?;
                }

                acc(&waa, [km, ki, kn], &aa, 1.0)?;
                acc(&wbb, [km, ki, kn], &bb, 1.0)?;
                acc(&wab, [km, ki, kn], &ab, 1.0)?;

                // `:217` — the ONE term that writes the mixed block at the
                // MIRRORED address `[kn, ki, km]`, from `tmp_baabJ`'s
                // `transpose(0,3,2,1,4)` on the five-axis fancy-indexed array.
                let t = einsum("ie,mjne->nimj", &[&t1ai, &eris.blk(b::OOov, km, kj, kn)?])?;
                acc(&wab, [kn, ki, km], &t, 1.0)?;
            }
        }
    }

    // `:222-223` — `W - W.transpose(2,1,0,5,4,3,6)`, pairwise so that both
    // sides read pre-antisymmetrisation values.
    for w in [&waa, &wbb] {
        for ki in 0..nkpts {
            for km in 0..nkpts {
                for kn in km..nkpts {
                    let a = w.get([km, ki, kn])?;
                    let b_ = w.get([kn, ki, km])?;
                    let mut na_ = a.clone();
                    na_.sub_assign(&b_.transpose(&[2, 1, 0, 3])?)?;
                    if km == kn {
                        w.set([km, ki, kn], &na_)?;
                    } else {
                        let mut nb_ = b_;
                        nb_.sub_assign(&a.transpose(&[2, 1, 0, 3])?)?;
                        w.set([km, ki, kn], &na_)?;
                        w.set([kn, ki, km], &nb_)?;
                    }
                }
            }
        }
    }
    Ok((waa, wab, wbb))
}

/// `cc_Wvvvv_half` — `:270-309`. Returns `(Wvvvv, WvvVV, WVVVV)`, tiered.
///
/// "Half" is upstream's word: unlike `cc_Wvvvv` (`:226`) it does NOT
/// antisymmetrise, because `add_vvvv_` (`kccsd_uhf.py:600-607`) does that on
/// the CONTRACTED result instead — `Ht2aa[ki,kj,kb] -= tmp.transpose(0,1,3,2)`
/// — which is `nvir²·nocc²` work in place of `nvir⁴`.
///
/// # Errors
/// As [`cc_woooo`].
pub fn cc_wvvvv_half(
    pool: &Arc<ZWorkspacePool>,
    max_memory_bytes: usize,
    t1: &UT1,
    eris: &KuEris,
    kconserv: &Kconserv,
) -> Result<(KBlocks, KBlocks, KBlocks), PbcCcError> {
    let nkpts = eris.nkpts;
    let (va, vb) = (eris.nvir.0, eris.nvir.1);
    let waa = KBlocks::with_budget(pool, nkpts, &[va, va, va, va], max_memory_bytes)?;
    let wab = KBlocks::with_budget(pool, nkpts, &[va, va, vb, vb], max_memory_bytes)?;
    let wbb = KBlocks::with_budget(pool, nkpts, &[vb, vb, vb, vb], max_memory_bytes)?;

    for ka in 0..nkpts {
        for kb in 0..nkpts {
            for ke in 0..nkpts {
                let kf = kconserv.get(ka, ke, kb) as usize;
                let (t1aa, t1ab) = (t1.0.slice_leading(&[ka])?, t1.0.slice_leading(&[kb])?);
                let t1bb = t1.1.slice_leading(&[kb])?;

                // `:279-285`
                let mut aebf = eris.blk(b::vvvv, ka, ke, kb)?;
                aebf.add_assign(&einsum(
                    "mb,emfa->aebf",
                    &[&t1ab, &eris.blk(b::vovv, ke, kb, kf)?.conj()],
                )?)?;
                aebf.sub_assign(&einsum(
                    "mb,fmea->aebf",
                    &[&t1ab, &eris.blk(b::vovv, kf, kb, ke)?.conj()],
                )?)?;
                waa.set([ka, ke, kb], &aebf)?;

                // `:290-297`
                let mut aebf = eris.blk(b::vvVV, ka, ke, kb)?;
                aebf.sub_assign(&einsum(
                    "ma,emfb->aebf",
                    &[&t1aa, &eris.blk(b::voVV, ke, ka, kf)?.conj()],
                )?)?;
                aebf.sub_assign(&einsum(
                    "mb,fmea->aebf",
                    &[&t1bb, &eris.blk(b::VOvv, kf, kb, ke)?.conj()],
                )?)?;
                wab.set([ka, ke, kb], &aebf)?;

                // `:303-308`
                let mut aebf = eris.blk(b::VVVV, ka, ke, kb)?;
                aebf.add_assign(&einsum(
                    "mb,emfa->aebf",
                    &[&t1bb, &eris.blk(b::VOVV, ke, kb, kf)?.conj()],
                )?)?;
                aebf.sub_assign(&einsum(
                    "mb,fmea->aebf",
                    &[&t1bb, &eris.blk(b::VOVV, kf, kb, ke)?.conj()],
                )?)?;
                wbb.set([ka, ke, kb], &aebf)?;
            }
        }
    }
    Ok((waa, wab, wbb))
}

/// The six `Wovvo`-family intermediates of `cc_Wovvo` (`:392-479`), in
/// upstream's return order.
pub struct UWovvo {
    /// `Wovvo`, `[nocca, nvira, nvira, nocca]`.
    pub aa: KBlocks,
    /// `WovVO`, `[nocca, nvira, nvirb, noccb]`.
    pub ab: KBlocks,
    /// `WOVvo`, `[noccb, nvirb, nvira, nocca]`.
    pub ba: KBlocks,
    /// `WOVVO`, `[noccb, nvirb, nvirb, noccb]`.
    pub bb: KBlocks,
    /// `WoVVo`, `[nocca, nvirb, nvirb, nocca]`.
    pub abba: KBlocks,
    /// `WOvvO`, `[noccb, nvira, nvira, noccb]`.
    pub baab: KBlocks,
}

/// `cc_Wovvo` — `:392-479`.
///
/// # Errors
/// As [`cc_woooo`].
#[allow(clippy::too_many_lines)]
pub fn cc_wovvo(
    pool: &Arc<ZWorkspacePool>,
    max_memory_bytes: usize,
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    kconserv: &Kconserv,
) -> Result<UWovvo, PbcCcError> {
    let nkpts = eris.nkpts;
    let (oa, ob) = (eris.nocc.0, eris.nocc.1);
    let (va, vb) = (eris.nvir.0, eris.nvir.1);
    let w = UWovvo {
        aa: KBlocks::with_budget(pool, nkpts, &[oa, va, va, oa], max_memory_bytes)?,
        ab: KBlocks::with_budget(pool, nkpts, &[oa, va, vb, ob], max_memory_bytes)?,
        ba: KBlocks::with_budget(pool, nkpts, &[ob, vb, va, oa], max_memory_bytes)?,
        bb: KBlocks::with_budget(pool, nkpts, &[ob, vb, vb, ob], max_memory_bytes)?,
        abba: KBlocks::with_budget(pool, nkpts, &[oa, vb, vb, oa], max_memory_bytes)?,
        baab: KBlocks::with_budget(pool, nkpts, &[ob, va, va, ob], max_memory_bytes)?,
    };

    // `:407-419` — the bare integrals.
    for ka in 0..nkpts {
        for ki in 0..nkpts {
            for kj in 0..nkpts {
                let kb = kconserv.get(ka, ki, kj) as usize;
                acc(
                    &w.aa,
                    [ki, ka, kb],
                    &eris
                        .blk(b::voov, ka, ki, kj)?
                        .conj()
                        .transpose(&[1, 0, 3, 2])?,
                    1.0,
                )?;
                acc(
                    &w.ab,
                    [ki, ka, kb],
                    &eris
                        .blk(b::voOV, ka, ki, kj)?
                        .conj()
                        .transpose(&[1, 0, 3, 2])?,
                    1.0,
                )?;
                acc(
                    &w.ba,
                    [ki, ka, kb],
                    &eris.blk(b::voOV, kb, kj, ki)?.transpose(&[2, 3, 0, 1])?,
                    1.0,
                )?;
                acc(
                    &w.bb,
                    [ki, ka, kb],
                    &eris
                        .blk(b::VOOV, ka, ki, kj)?
                        .conj()
                        .transpose(&[1, 0, 3, 2])?,
                    1.0,
                )?;

                let kb = kconserv.get(ki, kj, ka) as usize;
                acc(
                    &w.aa,
                    [ki, kb, ka],
                    &eris.blk(b::oovv, ki, kj, ka)?.transpose(&[0, 3, 2, 1])?,
                    -1.0,
                )?;
                acc(
                    &w.bb,
                    [ki, kb, ka],
                    &eris.blk(b::OOVV, ki, kj, ka)?.transpose(&[0, 3, 2, 1])?,
                    -1.0,
                )?;
                acc(
                    &w.abba,
                    [ki, kb, ka],
                    &eris.blk(b::ooVV, ki, kj, ka)?.transpose(&[0, 3, 2, 1])?,
                    -1.0,
                )?;
                acc(
                    &w.baab,
                    [ki, kb, ka],
                    &eris.blk(b::OOvv, ki, kj, ka)?.transpose(&[0, 3, 2, 1])?,
                    -1.0,
                )?;
            }
        }
    }

    let tau = make_tau2(t2, t1, t1, 2.0)?;
    for km in 0..nkpts {
        for kb in 0..nkpts {
            for ke in 0..nkpts {
                let kj = kconserv.get(km, ke, kb) as usize;
                let vovv_c = eris.blk(b::vovv, ke, km, kj)?.conj();
                let vovv_bb = eris.blk(b::VOVV, ke, km, kj)?.conj();
                let vovv_ab = eris.blk(b::voVV, ke, km, kj)?.conj();
                let vovv_ba = eris.blk(b::VOvv, ke, km, kj)?.conj();
                let (t1aj, t1bj) = (t1.0.slice_leading(&[kj])?, t1.1.slice_leading(&[kj])?);
                let (t1ae, t1be) = (t1.0.slice_leading(&[ke])?, t1.1.slice_leading(&[ke])?);
                let (t1ab_, t1bb_) = (t1.0.slice_leading(&[kb])?, t1.1.slice_leading(&[kb])?);

                // `:427-430`
                acc(
                    &w.aa,
                    [km, ke, kb],
                    &einsum("jf,emfb->mebj", &[&t1aj, &vovv_c])?,
                    1.0,
                )?;
                acc(
                    &w.bb,
                    [km, ke, kb],
                    &einsum("jf,emfb->mebj", &[&t1bj, &vovv_bb])?,
                    1.0,
                )?;
                acc(
                    &w.ab,
                    [km, ke, kb],
                    &einsum("jf,emfb->mebj", &[&t1bj, &vovv_ab])?,
                    1.0,
                )?;
                acc(
                    &w.ba,
                    [km, ke, kb],
                    &einsum("jf,emfb->mebj", &[&t1aj, &vovv_ba])?,
                    1.0,
                )?;

                // `:432-435` — note the DIFFERENT address `[km, kj, kb]`, and
                // that `WOvvO` takes `VOvv` while `WoVVo` takes `voVV`.
                acc(
                    &w.aa,
                    [km, kj, kb],
                    &einsum("je,emfb->mfbj", &[&t1ae, &vovv_c])?,
                    -1.0,
                )?;
                acc(
                    &w.bb,
                    [km, kj, kb],
                    &einsum("je,emfb->mfbj", &[&t1be, &vovv_bb])?,
                    -1.0,
                )?;
                acc(
                    &w.baab,
                    [km, kj, kb],
                    &einsum("je,emfb->mfbj", &[&t1be, &vovv_ba])?,
                    -1.0,
                )?;
                acc(
                    &w.abba,
                    [km, kj, kb],
                    &einsum("je,emfb->mfbj", &[&t1ae, &vovv_ab])?,
                    -1.0,
                )?;

                // `:437-438`
                acc(
                    &w.ba,
                    [km, ke, kb],
                    &einsum("nb,njme->mebj", &[&t1ab_, &eris.blk(b::ooOV, kb, kj, km)?])?,
                    -1.0,
                )?;
                acc(
                    &w.ab,
                    [km, ke, kb],
                    &einsum("nb,njme->mebj", &[&t1bb_, &eris.blk(b::OOov, kb, kj, km)?])?,
                    -1.0,
                )?;
                // `:440-441`
                acc(
                    &w.baab,
                    [km, ke, kb],
                    &einsum("nb,mjne->mebj", &[&t1ab_, &eris.blk(b::OOov, km, kj, kb)?])?,
                    1.0,
                )?;
                acc(
                    &w.abba,
                    [km, ke, kb],
                    &einsum("nb,mjne->mebj", &[&t1bb_, &eris.blk(b::ooOV, km, kj, kb)?])?,
                    1.0,
                )?;

                // `:443-448`
                let mut ooov_temp = eris.blk(b::ooov, kb, kj, km)?;
                ooov_temp.sub_assign(&eris.blk(b::ooov, km, kj, kb)?.transpose(&[2, 1, 0, 3])?)?;
                acc(
                    &w.aa,
                    [km, ke, kb],
                    &einsum("nb,njme->mebj", &[&t1ab_, &ooov_temp])?,
                    -1.0,
                )?;
                let mut oooo_temp = eris.blk(b::OOOV, kb, kj, km)?;
                oooo_temp.sub_assign(&eris.blk(b::OOOV, km, kj, kb)?.transpose(&[2, 1, 0, 3])?)?;
                acc(
                    &w.bb,
                    [km, ke, kb],
                    &einsum("nb,njme->mebj", &[&t1bb_, &oooo_temp])?,
                    -1.0,
                )?;

                // `:450-452` and `:465-479` — every remaining term sums over a
                // free k-index. Written as an explicit `for kx`, per rule 3.
                for kx in 0..nkpts {
                    let kf = kconserv.get(km, ke, kx) as usize;

                    // `:450`
                    acc(
                        &w.aa,
                        [km, ke, kb],
                        &einsum_scale(
                            "jnbf,menf->mebj",
                            &[
                                &t2.1.slice_leading(&[kj, kx, kb])?,
                                &eris.blk(b::ovOV, km, ke, kx)?,
                            ],
                            0.5,
                        )?,
                        1.0,
                    )?;
                    // `:451`
                    acc(
                        &w.baab,
                        [km, ke, kb],
                        &einsum_scale(
                            "njbf,nemf->mebj",
                            &[
                                &tau.1.slice_leading(&[kx, kj, kb])?,
                                &eris.blk(b::ovOV, kx, ke, km)?,
                            ],
                            0.5,
                        )?,
                        1.0,
                    )?;
                    // `:452`
                    acc(
                        &w.ab,
                        [km, ke, kb],
                        &einsum_scale(
                            "njbf,menf->mebj",
                            &[
                                &tau.2.slice_leading(&[kx, kj, kb])?,
                                &eris.blk(b::ovOV, km, ke, kx)?,
                            ],
                            0.5,
                        )?,
                        -1.0,
                    )?;

                    // `:454-461` — `temp_ovOV_1[kn] = ovOV[kn, kf, km]`,
                    // `temp_ovOV_2[kn] = ovOV[km, kf, kn]`, `kf` the CONSERVED
                    // partner of the free index, not a loop bound.
                    let ov1 = eris.blk(b::ovOV, kx, kf, km)?;
                    let ov2 = eris.blk(b::ovOV, km, kf, kx)?;
                    // `:465`
                    acc(
                        &w.bb,
                        [km, ke, kb],
                        &einsum_scale(
                            "njfb,nfme->mebj",
                            &[&t2.1.slice_leading(&[kx, kj, kf])?, &ov1],
                            0.5,
                        )?,
                        1.0,
                    )?;
                    // `:466`
                    acc(
                        &w.ba,
                        [km, ke, kb],
                        &einsum_scale(
                            "njbf,nfme->mebj",
                            &[&tau.0.slice_leading(&[kx, kj, kb])?, &ov1],
                            0.5,
                        )?,
                        -1.0,
                    )?;
                    // `:467`
                    acc(
                        &w.abba,
                        [km, ke, kb],
                        &einsum_scale(
                            "jnfb,mfne->mebj",
                            &[&tau.1.slice_leading(&[kj, kx, kf])?, &ov2],
                            0.5,
                        )?,
                        1.0,
                    )?;

                    // `:469-472`
                    let mut temp_ovov_b = eris.blk(b::OVOV, km, ke, kx)?;
                    temp_ovov_b
                        .sub_assign(&eris.blk(b::OVOV, kx, ke, km)?.transpose(&[2, 1, 0, 3])?)?;
                    acc(
                        &w.bb,
                        [km, ke, kb],
                        &einsum_scale(
                            "njbf,menf->mebj",
                            &[&tau.2.slice_leading(&[kx, kj, kb])?, &temp_ovov_b],
                            0.5,
                        )?,
                        -1.0,
                    )?;
                    acc(
                        &w.ba,
                        [km, ke, kb],
                        &einsum_scale(
                            "jnbf,menf->mebj",
                            &[&t2.1.slice_leading(&[kj, kx, kb])?, &temp_ovov_b],
                            0.5,
                        )?,
                        1.0,
                    )?;

                    // `:475-478`
                    let mut temp_ovov_a = eris.blk(b::ovov, kx, ke, km)?;
                    temp_ovov_a
                        .sub_assign(&eris.blk(b::ovov, km, ke, kx)?.transpose(&[2, 1, 0, 3])?)?;
                    acc(
                        &w.aa,
                        [km, ke, kb],
                        &einsum_scale(
                            "njbf,nemf->mebj",
                            &[&tau.0.slice_leading(&[kx, kj, kb])?, &temp_ovov_a],
                            0.5,
                        )?,
                        1.0,
                    )?;
                    acc(
                        &w.ab,
                        [km, ke, kb],
                        &einsum_scale(
                            "njfb,nemf->mebj",
                            &[&t2.1.slice_leading(&[kx, kj, kf])?, &temp_ovov_a],
                            0.5,
                        )?,
                        -1.0,
                    )?;
                }
            }
        }
    }
    Ok(w)
}

/// `kconserv_mat(nkpts, kconserv)` — `:581-588`. `P[ki,kj,ka,kb] = 1` when
/// `kb == kconserv[ki,ka,kj]`.
///
/// Nothing in this port contracts against `P`: every place upstream uses it,
/// the k-loop below carries the same constraint directly. It is here because
/// three of `update_amps`'s commented-out reference einsums are written with
/// it, and a reader checking those against this port needs the definition.
pub fn kconserv_mat(nkpts: usize, kconserv: &Kconserv) -> Vec<f64> {
    let mut p = vec![0.0; nkpts * nkpts * nkpts * nkpts];
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(ki, ka, kj) as usize;
                p[((ki * nkpts + kj) * nkpts + ka) * nkpts + kb] = 1.0;
            }
        }
    }
    p
}

/// `w[k] += s * v`.
fn acc(w: &KBlocks, k: [usize; 3], v: &ZArr, s: f64) -> Result<(), PbcCcError> {
    let mut cur = w.get(k)?;
    cur.zip_assign(v, s)?;
    w.set(k, &cur)
}

/// `t[k0,k1,k2] += s * v`, on a plain `nkpts³`-leading [`ZArr`].
fn add_at(t: &mut ZArr, k: [usize; 3], v: &ZArr, s: f64) -> Result<(), PbcCcError> {
    let mut cur = t.slice_leading(&k)?;
    cur.zip_assign(v, s)?;
    t.set_leading(&k, &cur)
}

fn einsum_half(spec: &str, ops: &[&ZArr]) -> Result<ZArr, PbcCcError> {
    einsum_scale(spec, ops, 0.5)
}

fn einsum_scale(spec: &str, ops: &[&ZArr], f: f64) -> Result<ZArr, PbcCcError> {
    let mut r = einsum(spec, ops)?;
    r.scale(f);
    Ok(r)
}

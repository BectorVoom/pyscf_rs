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

// ---------------------------------------------------------------------------
// The EOM intermediates (`kintermediates_uhf.py:311-353`, `:590-1120`,
// plan 16-11 Task 1).
//
// `eom_kccsd_uhf._IMDS` builds `Foo`/`Fvv`/`Fov`, `Wovvo`, `Woovv` (shared),
// `Woooo`/`Wooov`/`Woovo` (IP) and `Wvvov`/`Wvvvv`/`Wvvvo` (EA). Every one
// returns THREE or FOUR spin blocks, and the blocks are not related by any
// symmetry this port could exploit — `WooVO` and `WOOvo` are different tensors
// with different shapes, not transposes of one another.
// ---------------------------------------------------------------------------

/// The four spin blocks an EOM-UHF intermediate returns, in upstream's order.
pub type UQuad = (ZArr, ZArr, ZArr, ZArr);
/// The three blocks `Wvvvv` returns.
pub type UTriple = (ZArr, ZArr, ZArr);

/// `Foo` — `:590-600`. `cc_Foo` plus a half `t1·Fov`, per spin.
///
/// # Errors
/// As [`cc_foo`].
pub fn foo(
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    kconserv: &Kconserv,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let (fova, fovb) = cc_fov(t1, eris)?;
    let (mut fa, mut fb) = cc_foo(t1, t2, eris, kconserv)?;
    for ki in 0..eris.nkpts {
        add_at1(
            &mut fa,
            ki,
            &einsum(
                "ie,me->mi",
                &[&t1.0.slice_leading(&[ki])?, &fova.slice_leading(&[ki])?],
            )?,
            0.5,
        )?;
        add_at1(
            &mut fb,
            ki,
            &einsum(
                "ie,me->mi",
                &[&t1.1.slice_leading(&[ki])?, &fovb.slice_leading(&[ki])?],
            )?,
            0.5,
        )?;
    }
    Ok((fa, fb))
}

/// `Fvv` — `:602-612`. `cc_Fvv` MINUS a half `Fov·t1` (the operand order and
/// the sign both differ from [`foo`]'s, which is why both are written out).
///
/// # Errors
/// As [`cc_fvv`].
pub fn fvv(
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    kconserv: &Kconserv,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let (fova, fovb) = cc_fov(t1, eris)?;
    let (mut fa, mut fb) = cc_fvv(t1, t2, eris, kconserv)?;
    for ka in 0..eris.nkpts {
        add_at1(
            &mut fa,
            ka,
            &einsum(
                "me,ma->ae",
                &[&fova.slice_leading(&[ka])?, &t1.0.slice_leading(&[ka])?],
            )?,
            -0.5,
        )?;
        add_at1(
            &mut fb,
            ka,
            &einsum(
                "me,ma->ae",
                &[&fovb.slice_leading(&[ka])?, &t1.1.slice_leading(&[ka])?],
            )?,
            -0.5,
        )?;
    }
    Ok((fa, fb))
}

/// `Fov` — `:614-616`. Identical to `cc_Fov`; the alias is upstream's.
///
/// # Errors
/// As [`cc_fov`].
pub fn fov(t1: &UT1, eris: &KuEris) -> Result<(ZArr, ZArr), PbcCcError> {
    cc_fov(t1, eris)
}

/// `Wooov` — `:996-1009`.
///
/// Written whole-array upstream; the two same-spin blocks carry a
/// `transpose(2,1,0,5,4,3,6)` antisymmetrisation, which per k-block is
/// `W[kx,ky,kz][m,i,n,e] -= X[kz,ky,kx][n,i,m,e]`.
///
/// # Errors
/// Propagates the ERI access.
pub fn wooov(t1: &UT1, eris: &KuEris) -> Result<UQuad, PbcCcError> {
    let nk = eris.nkpts;
    let (oa, ob) = eris.nocc;
    let (va, vb) = eris.nvir;
    let mut w_aaaa = ZArr::zeros(&[nk, nk, nk, oa, oa, oa, va]);
    let mut w_aabb = ZArr::zeros(&[nk, nk, nk, oa, oa, ob, vb]);
    let mut w_bbaa = ZArr::zeros(&[nk, nk, nk, ob, ob, oa, va]);
    let mut w_bbbb = ZArr::zeros(&[nk, nk, nk, ob, ob, ob, vb]);
    for kx in 0..nk {
        for ky in 0..nk {
            for kz in 0..nk {
                let (t1ay, t1by) = (t1.0.slice_leading(&[ky])?, t1.1.slice_leading(&[ky])?);

                let mut a = eris.blk(b::ooov, kx, ky, kz)?;
                a.sub_assign(&eris.blk(b::ooov, kz, ky, kx)?.transpose(&[2, 1, 0, 3])?)?;
                a.add_assign(&einsum(
                    "if,mfne->mine",
                    &[&t1ay, &eris.blk(b::ovov, kx, ky, kz)?],
                )?)?;
                a.sub_assign(&einsum(
                    "if,nfme->mine",
                    &[&t1ay, &eris.blk(b::ovov, kz, ky, kx)?],
                )?)?;
                w_aaaa.set_leading(&[kx, ky, kz], &a)?;

                let mut a = eris.blk(b::ooOV, kx, ky, kz)?;
                a.add_assign(&einsum(
                    "if,mfNE->miNE",
                    &[&t1ay, &eris.blk(b::ovOV, kx, ky, kz)?],
                )?)?;
                w_aabb.set_leading(&[kx, ky, kz], &a)?;

                let mut a = eris.blk(b::OOov, kx, ky, kz)?;
                a.add_assign(&einsum(
                    "IF,MFne->MIne",
                    &[&t1by, &eris.blk(b::OVov, kx, ky, kz)?],
                )?)?;
                w_bbaa.set_leading(&[kx, ky, kz], &a)?;

                let mut a = eris.blk(b::OOOV, kx, ky, kz)?;
                a.sub_assign(&eris.blk(b::OOOV, kz, ky, kx)?.transpose(&[2, 1, 0, 3])?)?;
                a.add_assign(&einsum(
                    "IF,MFNE->MINE",
                    &[&t1by, &eris.blk(b::OVOV, kx, ky, kz)?],
                )?)?;
                a.sub_assign(&einsum(
                    "IF,NFME->MINE",
                    &[&t1by, &eris.blk(b::OVOV, kz, ky, kx)?],
                )?)?;
                w_bbbb.set_leading(&[kx, ky, kz], &a)?;
            }
        }
    }
    Ok((w_aaaa, w_aabb, w_bbaa, w_bbbb))
}

/// `Wovvo` — `:1011-1038`. `cc_Wovvo`'s FIRST FOUR blocks plus the `t2·ovov`
/// completions. `WoVVo`/`WOvvO` are dropped: `_IMDS._make_shared` (`:1069`)
/// keeps only four.
///
/// # Errors
/// As [`cc_wovvo`].
pub fn wovvo(
    pool: &Arc<ZWorkspacePool>,
    max_memory_bytes: usize,
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    kconserv: &Kconserv,
) -> Result<UQuad, PbcCcError> {
    let nk = eris.nkpts;
    let w = cc_wovvo(pool, max_memory_bytes, t1, t2, eris, kconserv)?;
    let mut aa = gather3(&w.aa)?;
    let mut ab = gather3(&w.ab)?;
    let mut ba = gather3(&w.ba)?;
    let mut bb = gather3(&w.bb)?;
    w.aa.release();
    w.ab.release();
    w.ba.release();
    w.bb.release();
    w.abba.release();
    w.baab.release();

    for km in 0..nk {
        for kb_ in 0..nk {
            for ke in 0..nk {
                let kj = kconserv.get(km, ke, kb_) as usize;
                for kn in 0..nk {
                    let kf = kconserv.get(km, ke, kn) as usize;
                    // `:1023-1025`
                    add_at(
                        &mut aa,
                        [km, ke, kb_],
                        &einsum(
                            "jnbf,menf->mebj",
                            &[
                                &t2.0.slice_leading(&[kj, kn, kb_])?,
                                &eris.blk(b::ovov, km, ke, kn)?,
                            ],
                        )?,
                        0.5,
                    )?;
                    add_at(
                        &mut aa,
                        [km, ke, kb_],
                        &einsum(
                            "jnbf,mfne->mebj",
                            &[
                                &t2.0.slice_leading(&[kj, kn, kb_])?,
                                &eris.blk(b::ovov, km, kf, kn)?,
                            ],
                        )?,
                        -0.5,
                    )?;
                    add_at(
                        &mut aa,
                        [km, ke, kb_],
                        &einsum(
                            "jNbF,meNF->mebj",
                            &[
                                &t2.1.slice_leading(&[kj, kn, kb_])?,
                                &eris.blk(b::ovOV, km, ke, kn)?,
                            ],
                        )?,
                        0.5,
                    )?;
                    // `:1027-1029`
                    add_at(
                        &mut ba,
                        [km, ke, kb_],
                        &einsum(
                            "jNbF,MENF->MEbj",
                            &[
                                &t2.1.slice_leading(&[kj, kn, kb_])?,
                                &eris.blk(b::OVOV, km, ke, kn)?,
                            ],
                        )?,
                        0.5,
                    )?;
                    add_at(
                        &mut ba,
                        [km, ke, kb_],
                        &einsum(
                            "jNbF,MFNE->MEbj",
                            &[
                                &t2.1.slice_leading(&[kj, kn, kb_])?,
                                &eris.blk(b::OVOV, km, kf, kn)?,
                            ],
                        )?,
                        -0.5,
                    )?;
                    add_at(
                        &mut ba,
                        [km, ke, kb_],
                        &einsum(
                            "jnbf,MEnf->MEbj",
                            &[
                                &t2.0.slice_leading(&[kj, kn, kb_])?,
                                &eris.blk(b::OVov, km, ke, kn)?,
                            ],
                        )?,
                        0.5,
                    )?;
                    // `:1031-1033`
                    add_at(
                        &mut ab,
                        [km, ke, kb_],
                        &einsum(
                            "nJfB,menf->meBJ",
                            &[
                                &t2.1.slice_leading(&[kn, kj, kf])?,
                                &eris.blk(b::ovov, km, ke, kn)?,
                            ],
                        )?,
                        0.5,
                    )?;
                    add_at(
                        &mut ab,
                        [km, ke, kb_],
                        &einsum(
                            "nJfB,mfne->meBJ",
                            &[
                                &t2.1.slice_leading(&[kn, kj, kf])?,
                                &eris.blk(b::ovov, km, kf, kn)?,
                            ],
                        )?,
                        -0.5,
                    )?;
                    add_at(
                        &mut ab,
                        [km, ke, kb_],
                        &einsum(
                            "JNBF,meNF->meBJ",
                            &[
                                &t2.2.slice_leading(&[kj, kn, kb_])?,
                                &eris.blk(b::ovOV, km, ke, kn)?,
                            ],
                        )?,
                        0.5,
                    )?;
                    // `:1035-1037`
                    add_at(
                        &mut bb,
                        [km, ke, kb_],
                        &einsum(
                            "JNBF,MENF->MEBJ",
                            &[
                                &t2.2.slice_leading(&[kj, kn, kb_])?,
                                &eris.blk(b::OVOV, km, ke, kn)?,
                            ],
                        )?,
                        0.5,
                    )?;
                    add_at(
                        &mut bb,
                        [km, ke, kb_],
                        &einsum(
                            "JNBF,MFNE->MEBJ",
                            &[
                                &t2.2.slice_leading(&[kj, kn, kb_])?,
                                &eris.blk(b::OVOV, km, kf, kn)?,
                            ],
                        )?,
                        -0.5,
                    )?;
                    add_at(
                        &mut bb,
                        [km, ke, kb_],
                        &einsum(
                            "nJfB,MEnf->MEBJ",
                            &[
                                &t2.1.slice_leading(&[kn, kj, kf])?,
                                &eris.blk(b::OVov, km, ke, kn)?,
                            ],
                        )?,
                        0.5,
                    )?;
                }
            }
        }
    }
    Ok((aa, ab, ba, bb))
}

/// A tiered `KBlocks` gathered into one `[nkpts, nkpts, nkpts, ...]` array.
fn gather3(blocks: &KBlocks) -> Result<ZArr, PbcCcError> {
    let nk = blocks.nkpts();
    let bs = blocks.block_shape().to_vec();
    let mut shape = vec![nk, nk, nk];
    shape.extend_from_slice(&bs);
    let mut out = ZArr::zeros(&shape);
    for k0 in 0..nk {
        for k1 in 0..nk {
            for k2 in 0..nk {
                out.set_leading(&[k0, k1, k2], &blocks.get([k0, k1, k2])?)?;
            }
        }
    }
    Ok(out)
}

/// `t[k] += s * v`, on a plain `nkpts`-leading [`ZArr`].
fn add_at1(t: &mut ZArr, k: usize, v: &ZArr, s: f64) -> Result<(), PbcCcError> {
    let mut cur = t.slice_leading(&[k])?;
    cur.zip_assign(v, s)?;
    t.set_leading(&[k], &cur)
}

/// `W1oovv` — `:1040-1074`.
///
/// # Errors
/// As [`cc_foo`].
pub fn w1oovv(t2: &UT2, eris: &KuEris, kconserv: &Kconserv) -> Result<UQuad, PbcCcError> {
    let nk = eris.nkpts;
    let (oa, ob) = eris.nocc;
    let (va, vb) = eris.nvir;
    let mut w_aa = ZArr::zeros(&[nk, nk, nk, oa, oa, va, va]);
    let mut w_ab = ZArr::zeros(&[nk, nk, nk, oa, oa, vb, vb]);
    let mut w_ba = ZArr::zeros(&[nk, nk, nk, ob, ob, va, va]);
    let mut w_bb = ZArr::zeros(&[nk, nk, nk, ob, ob, vb, vb]);
    for kk in 0..nk {
        for ki in 0..nk {
            for kb_ in 0..nk {
                let kd = kconserv.get(kk, ki, kb_) as usize;
                let mut aa = eris.blk(b::oovv, kk, ki, kb_)?;
                aa.sub_assign(&eris.blk(b::voov, kb_, ki, kk)?.transpose(&[2, 1, 0, 3])?)?;
                let mut ab = eris.blk(b::ooVV, kk, ki, kb_)?;
                let mut ba = eris.blk(b::OOvv, kk, ki, kb_)?;
                let mut bb = eris.blk(b::OOVV, kk, ki, kb_)?;
                bb.sub_assign(&eris.blk(b::VOOV, kb_, ki, kk)?.transpose(&[2, 1, 0, 3])?)?;

                for kl in 0..nk {
                    let kc = kconserv.get(ki, kb_, kl) as usize;
                    aa.sub_assign(&einsum(
                        "lckd,ilbc->kibd",
                        &[
                            &eris.blk(b::ovov, kl, kc, kk)?,
                            &t2.0.slice_leading(&[ki, kl, kb_])?,
                        ],
                    )?)?;
                    aa.add_assign(&einsum(
                        "ldkc,ilbc->kibd",
                        &[
                            &eris.blk(b::ovov, kl, kd, kk)?,
                            &t2.0.slice_leading(&[ki, kl, kb_])?,
                        ],
                    )?)?;
                    aa.sub_assign(&einsum(
                        "LCkd,iLbC->kibd",
                        &[
                            &eris.blk(b::OVov, kl, kc, kk)?,
                            &t2.1.slice_leading(&[ki, kl, kb_])?,
                        ],
                    )?)?;

                    ab.sub_assign(&einsum(
                        "kcLD,iLcB->kiBD",
                        &[
                            &eris.blk(b::ovOV, kk, kc, kl)?,
                            &t2.1.slice_leading(&[ki, kl, kc])?,
                        ],
                    )?)?;
                    ba.sub_assign(&einsum(
                        "KCld,lIbC->KIbd",
                        &[
                            &eris.blk(b::OVov, kk, kc, kl)?,
                            &t2.1.slice_leading(&[kl, ki, kb_])?,
                        ],
                    )?)?;

                    bb.sub_assign(&einsum(
                        "LCKD,ILBC->KIBD",
                        &[
                            &eris.blk(b::OVOV, kl, kc, kk)?,
                            &t2.2.slice_leading(&[ki, kl, kb_])?,
                        ],
                    )?)?;
                    bb.add_assign(&einsum(
                        "LDKC,ILBC->KIBD",
                        &[
                            &eris.blk(b::OVOV, kl, kd, kk)?,
                            &t2.2.slice_leading(&[ki, kl, kb_])?,
                        ],
                    )?)?;
                    bb.sub_assign(&einsum(
                        "lcKD,lIcB->KIBD",
                        &[
                            &eris.blk(b::ovOV, kl, kc, kk)?,
                            &t2.1.slice_leading(&[kl, ki, kc])?,
                        ],
                    )?)?;
                }
                w_aa.set_leading(&[kk, ki, kb_], &aa)?;
                w_ab.set_leading(&[kk, ki, kb_], &ab)?;
                w_ba.set_leading(&[kk, ki, kb_], &ba)?;
                w_bb.set_leading(&[kk, ki, kb_], &bb)?;
            }
        }
    }
    Ok((w_aa, w_ab, w_ba, w_bb))
}

/// `W2oovv` — `:1076-1105`.
///
/// # Errors
/// As [`cc_foo`].
pub fn w2oovv(t1: &UT1, eris: &KuEris, kconserv: &Kconserv) -> Result<UQuad, PbcCcError> {
    let nk = eris.nkpts;
    let (oa, ob) = eris.nocc;
    let (va, vb) = eris.nvir;
    let (ww_aa, ww_ab, ww_ba, ww_bb) = wooov(t1, eris)?;
    let mut w_aa = ZArr::zeros(&[nk, nk, nk, oa, oa, va, va]);
    let mut w_ab = ZArr::zeros(&[nk, nk, nk, oa, oa, vb, vb]);
    let mut w_ba = ZArr::zeros(&[nk, nk, nk, ob, ob, va, va]);
    let mut w_bb = ZArr::zeros(&[nk, nk, nk, ob, ob, vb, vb]);
    for kk in 0..nk {
        for ki in 0..nk {
            for kb_ in 0..nk {
                let kd = kconserv.get(kk, ki, kb_) as usize;
                let (t1ab_, t1bb_) = (t1.0.slice_leading(&[kb_])?, t1.1.slice_leading(&[kb_])?);
                let (t1ai, t1bi) = (t1.0.slice_leading(&[ki])?, t1.1.slice_leading(&[ki])?);

                let mut aa = einsum_scale(
                    "kild,lb->kibd",
                    &[&ww_aa.slice_leading(&[kk, ki, kb_])?, &t1ab_],
                    -1.0,
                )?;
                aa.add_assign(&einsum(
                    "ckdb,ic->kibd",
                    &[&eris.blk(b::vovv, ki, kk, kd)?.conj(), &t1ai],
                )?)?;
                aa.sub_assign(&einsum(
                    "dkcb,ic->kibd",
                    &[&eris.blk(b::vovv, kd, kk, ki)?.conj(), &t1ai],
                )?)?;
                w_aa.set_leading(&[kk, ki, kb_], &aa)?;

                let mut ab = einsum_scale(
                    "kiLD,LB->kiBD",
                    &[&ww_ab.slice_leading(&[kk, ki, kb_])?, &t1bb_],
                    -1.0,
                )?;
                ab.add_assign(&einsum(
                    "ckDB,ic->kiBD",
                    &[&eris.blk(b::voVV, ki, kk, kd)?.conj(), &t1ai],
                )?)?;
                w_ab.set_leading(&[kk, ki, kb_], &ab)?;

                let mut ba = einsum_scale(
                    "KIld,lb->KIbd",
                    &[&ww_ba.slice_leading(&[kk, ki, kb_])?, &t1ab_],
                    -1.0,
                )?;
                ba.add_assign(&einsum(
                    "CKdb,IC->KIbd",
                    &[&eris.blk(b::VOvv, ki, kk, kd)?.conj(), &t1bi],
                )?)?;
                w_ba.set_leading(&[kk, ki, kb_], &ba)?;

                let mut bb = einsum_scale(
                    "KILD,LB->KIBD",
                    &[&ww_bb.slice_leading(&[kk, ki, kb_])?, &t1bb_],
                    -1.0,
                )?;
                bb.add_assign(&einsum(
                    "CKDB,IC->KIBD",
                    &[&eris.blk(b::VOVV, ki, kk, kd)?.conj(), &t1bi],
                )?)?;
                bb.sub_assign(&einsum(
                    "DKCB,IC->KIBD",
                    &[&eris.blk(b::VOVV, kd, kk, ki)?.conj(), &t1bi],
                )?)?;
                w_bb.set_leading(&[kk, ki, kb_], &bb)?;
            }
        }
    }
    Ok((w_aa, w_ab, w_ba, w_bb))
}

/// `Woovv = W1oovv + W2oovv` — `:1107-1117`.
///
/// # Errors
/// As [`cc_foo`].
pub fn woovv(t1: &UT1, t2: &UT2, eris: &KuEris, kconserv: &Kconserv) -> Result<UQuad, PbcCcError> {
    let (mut a, mut b_, mut c, mut d) = w1oovv(t2, eris, kconserv)?;
    let (x, y, z, w) = w2oovv(t1, eris, kconserv)?;
    a.add_assign(&x)?;
    b_.add_assign(&y)?;
    c.add_assign(&z)?;
    d.add_assign(&w)?;
    Ok((a, b_, c, d))
}

/// `Woooo` — `:826-877`. Returns `(Woooo, WooOO, WOOOO)`; upstream's fourth
/// return is `WOOoo = None` (`:876`).
///
/// **Not [`cc_woooo`].** The `t1` terms coincide, but the antisymmetrisation
/// happens BEFORE the `tau` terms here and AFTER them there, and the `tau`
/// terms use the ANTISYMMETRISED `ovov` with factors `0.5 / 0.5 / 1.0` rather
/// than the plain `ovov` with `0.25 / 0.25 / 0.5`. Reusing one for the other
/// would be silently wrong.
///
/// # Errors
/// As [`cc_woooo`].
pub fn eom_woooo(
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    kconserv: &Kconserv,
) -> Result<UTriple, PbcCcError> {
    let nk = eris.nkpts;
    let (oa, ob) = eris.nocc;
    let mut waa = ZArr::zeros(&[nk, nk, nk, oa, oa, oa, oa]);
    let mut wab = ZArr::zeros(&[nk, nk, nk, oa, oa, ob, ob]);
    let mut wbb = ZArr::zeros(&[nk, nk, nk, ob, ob, ob, ob]);

    // `:838-861` — the bare integral plus the `t1` terms.
    for km in 0..nk {
        for kn in 0..nk {
            for ki in 0..nk {
                let kj = kconserv.get(km, ki, kn) as usize;
                let (t1ai, t1bi) = (t1.0.slice_leading(&[ki])?, t1.1.slice_leading(&[ki])?);
                let (t1aj, t1bj) = (t1.0.slice_leading(&[kj])?, t1.1.slice_leading(&[kj])?);

                let mut aa = eris.blk(b::oooo, km, ki, kn)?;
                aa.add_assign(&einsum(
                    "je,mine->minj",
                    &[&t1aj, &eris.blk(b::ooov, km, ki, kn)?],
                )?)?;
                aa.sub_assign(&einsum(
                    "ie,mjne->minj",
                    &[&t1ai, &eris.blk(b::ooov, km, kj, kn)?],
                )?)?;
                add_at(&mut waa, [km, ki, kn], &aa, 1.0)?;

                let mut bb = eris.blk(b::OOOO, km, ki, kn)?;
                bb.add_assign(&einsum(
                    "je,mine->minj",
                    &[&t1bj, &eris.blk(b::OOOV, km, ki, kn)?],
                )?)?;
                bb.sub_assign(&einsum(
                    "ie,mjne->minj",
                    &[&t1bi, &eris.blk(b::OOOV, km, kj, kn)?],
                )?)?;
                add_at(&mut wbb, [km, ki, kn], &bb, 1.0)?;

                let mut ab = eris.blk(b::ooOO, km, ki, kn)?;
                ab.add_assign(&einsum(
                    "je,mine->minj",
                    &[&t1bj, &eris.blk(b::ooOV, km, ki, kn)?],
                )?)?;
                add_at(&mut wab, [km, ki, kn], &ab, 1.0)?;

                // `:857` — the mirrored `[kn, ki, km]` write.
                let t = einsum("ie,mjne->nimj", &[&t1ai, &eris.blk(b::OOov, km, kj, kn)?])?;
                add_at(&mut wab, [kn, ki, km], &t, 1.0)?;
            }
        }
    }

    // `:860-861` — `W - W.transpose(2,1,0,5,4,3,6)`, pairwise so both sides
    // read pre-antisymmetrisation values.
    for w in [&mut waa, &mut wbb] {
        let src = w.clone();
        for km in 0..nk {
            for ki in 0..nk {
                for kn in 0..nk {
                    let t = src.slice_leading(&[kn, ki, km])?.transpose(&[2, 1, 0, 3])?;
                    add_at(w, [km, ki, kn], &t, -1.0)?;
                }
            }
        }
    }

    // `:863-875` — the `tau` terms, AFTER the antisymmetrisation, with the
    // ANTISYMMETRISED `ovov`.
    let tau = make_tau(t2, t1, t1, 1.0)?;
    for km in 0..nk {
        for ki in 0..nk {
            for kn in 0..nk {
                let kj = kconserv.get(km, ki, kn) as usize;
                for ke in 0..nk {
                    let kf = kconserv.get(km, ke, kn) as usize;
                    let mut ovov = eris.blk(b::ovov, km, ke, kn)?;
                    ovov.sub_assign(&eris.blk(b::ovov, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;
                    let mut ovov_b = eris.blk(b::OVOV, km, ke, kn)?;
                    ovov_b.sub_assign(&eris.blk(b::OVOV, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;

                    add_at(
                        &mut waa,
                        [km, ki, kn],
                        &einsum(
                            "ijef,menf->minj",
                            &[&tau.0.slice_leading(&[ki, kj, ke])?, &ovov],
                        )?,
                        0.5,
                    )?;
                    add_at(
                        &mut wbb,
                        [km, ki, kn],
                        &einsum(
                            "IJEF,MENF->MINJ",
                            &[&tau.2.slice_leading(&[ki, kj, ke])?, &ovov_b],
                        )?,
                        0.5,
                    )?;
                    add_at(
                        &mut wab,
                        [km, ki, kn],
                        &einsum(
                            "iJeF,meNF->miNJ",
                            &[
                                &tau.1.slice_leading(&[ki, kj, ke])?,
                                &eris.blk(b::ovOV, km, ke, kn)?,
                            ],
                        )?,
                        1.0,
                    )?;
                }
            }
        }
    }
    Ok((waa, wab, wbb))
}

/// `Wvvvv` — `:311-323`. `cc_Wvvvv` plus the `tau·ovov` completions.
///
/// # Errors
/// As [`cc_wvvvv_half`].
pub fn eom_wvvvv(
    pool: &Arc<ZWorkspacePool>,
    max_memory_bytes: usize,
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    kconserv: &Kconserv,
) -> Result<UTriple, PbcCcError> {
    let nk = eris.nkpts;
    let tau = make_tau(t2, t1, t1, 1.0)?;
    // `cc_Wvvvv` (`:226`) is `cc_Wvvvv_half` PLUS its antisymmetrisation
    // (`:263-268`); `cc_wvvvv_half` is what this port has, so the
    // antisymmetrisation is applied here.
    let half = cc_wvvvv_half(pool, max_memory_bytes, t1, eris, kconserv)?;
    let mut aa = gather3(&half.0)?;
    let mut ab = gather3(&half.1)?;
    let mut bb = gather3(&half.2)?;
    half.0.release();
    half.1.release();
    half.2.release();

    // `:263-268` — `W = W - W.transpose(2,1,0,5,4,3,6)` for the same-spin
    // blocks only; the mixed one is not antisymmetrised.
    for w in [&mut aa, &mut bb] {
        let src = w.clone();
        for ka in 0..nk {
            for ke in 0..nk {
                for kb_ in 0..nk {
                    let t = src
                        .slice_leading(&[kb_, ke, ka])?
                        .transpose(&[2, 1, 0, 3])?;
                    add_at(w, [ka, ke, kb_], &t, -1.0)?;
                }
            }
        }
    }

    for ka in 0..nk {
        for kb_ in 0..nk {
            for ke in 0..nk {
                for km in 0..nk {
                    let kn = kconserv.get(ka, km, kb_) as usize;
                    add_at(
                        &mut aa,
                        [ka, ke, kb_],
                        &einsum(
                            "mnab,menf->aebf",
                            &[
                                &tau.0.slice_leading(&[km, kn, ka])?,
                                &eris.blk(b::ovov, km, ke, kn)?,
                            ],
                        )?,
                        1.0,
                    )?;
                    add_at(
                        &mut ab,
                        [ka, ke, kb_],
                        &einsum(
                            "mNaB,meNF->aeBF",
                            &[
                                &tau.1.slice_leading(&[km, kn, ka])?,
                                &eris.blk(b::ovOV, km, ke, kn)?,
                            ],
                        )?,
                        1.0,
                    )?;
                    add_at(
                        &mut bb,
                        [ka, ke, kb_],
                        &einsum(
                            "mnab,menf->aebf",
                            &[
                                &tau.2.slice_leading(&[km, kn, ka])?,
                                &eris.blk(b::OVOV, km, ke, kn)?,
                            ],
                        )?,
                        1.0,
                    )?;
                }
            }
        }
    }
    Ok((aa, ab, bb))
}

/// `Wvvov` — `:618-645`.
///
/// # Errors
/// As [`cc_foo`].
pub fn wvvov(t1: &UT1, eris: &KuEris, kconserv: &Kconserv) -> Result<UQuad, PbcCcError> {
    let nk = eris.nkpts;
    let (oa, ob) = eris.nocc;
    let (va, vb) = eris.nvir;
    let mut w_aa = ZArr::zeros(&[nk, nk, nk, va, va, oa, va]);
    let mut w_ab = ZArr::zeros(&[nk, nk, nk, va, va, ob, vb]);
    let mut w_ba = ZArr::zeros(&[nk, nk, nk, vb, vb, oa, va]);
    let mut w_bb = ZArr::zeros(&[nk, nk, nk, vb, vb, ob, vb]);
    for kn in 0..nk {
        for km in 0..nk {
            for ke in 0..nk {
                let kf = kconserv.get(kn, ke, km) as usize;
                let ka = kn;
                let (t1an, t1bn) = (t1.0.slice_leading(&[kn])?, t1.1.slice_leading(&[kn])?);

                // `:632-635` — four `.conj().transpose(...)` of `vovv`-family
                // blocks, with two DIFFERENT axis permutations.
                let mut aa = eris
                    .blk(b::vovv, kf, km, ke)?
                    .transpose(&[3, 2, 1, 0])?
                    .conj();
                aa.sub_assign(
                    &eris
                        .blk(b::vovv, ke, km, kf)?
                        .transpose(&[3, 0, 1, 2])?
                        .conj(),
                )?;
                let ba = eris
                    .blk(b::voVV, kf, km, ke)?
                    .transpose(&[3, 2, 1, 0])?
                    .conj();
                let ab = eris
                    .blk(b::VOvv, kf, km, ke)?
                    .transpose(&[3, 2, 1, 0])?
                    .conj();
                let mut bb = eris
                    .blk(b::VOVV, kf, km, ke)?
                    .transpose(&[3, 2, 1, 0])?
                    .conj();
                bb.sub_assign(
                    &eris
                        .blk(b::VOVV, ke, km, kf)?
                        .transpose(&[3, 0, 1, 2])?
                        .conj(),
                )?;
                let mut w_aa_blk = aa;
                let mut w_ba_blk = ba;
                let mut w_ab_blk = ab;
                let mut w_bb_blk = bb;

                let mut ovov = eris.blk(b::ovov, kn, ke, km)?;
                ovov.sub_assign(&eris.blk(b::ovov, kn, kf, km)?.transpose(&[0, 3, 2, 1])?)?;
                let mut ovov_b = eris.blk(b::OVOV, kn, ke, km)?;
                ovov_b.sub_assign(&eris.blk(b::OVOV, kn, kf, km)?.transpose(&[0, 3, 2, 1])?)?;

                w_aa_blk.sub_assign(&einsum("na,nemf->aemf", &[&t1an, &ovov])?)?;
                w_ab_blk.sub_assign(&einsum(
                    "na,neMF->aeMF",
                    &[&t1an, &eris.blk(b::ovOV, kn, ke, km)?],
                )?)?;
                w_ba_blk.sub_assign(&einsum(
                    "NA,NEmf->AEmf",
                    &[&t1bn, &eris.blk(b::OVov, kn, ke, km)?],
                )?)?;
                w_bb_blk.sub_assign(&einsum("NA,NEMF->AEMF", &[&t1bn, &ovov_b])?)?;

                add_at(&mut w_aa, [ka, ke, km], &w_aa_blk, 1.0)?;
                add_at(&mut w_ab, [ka, ke, km], &w_ab_blk, 1.0)?;
                add_at(&mut w_ba, [ka, ke, km], &w_ba_blk, 1.0)?;
                add_at(&mut w_bb, [ka, ke, km], &w_bb_blk, 1.0)?;
            }
        }
    }
    Ok((w_aa, w_ab, w_ba, w_bb))
}

/// `get_Wvvvv(cc, t1, t2, eris, ka, kb, kc)` — `:325-390`, the non-`Lpv`
/// branch. Returns `(vvvv, vvVV, VVVV)` for ONE k-triple.
///
/// The `Lpv` branch builds these from GDF's three-index tensors; this port does
/// not carry `Lpv` (see [`crate::kueris`]'s module doc), so the `else` at
/// `:361` is the only route and it produces the same numbers.
///
/// # Errors
/// As [`cc_foo`].
pub fn get_wvvvv(
    t1: &UT1,
    t2: &UT2,
    eris: &KuEris,
    kconserv: &Kconserv,
    ka: usize,
    kb_: usize,
    kc: usize,
) -> Result<UTriple, PbcCcError> {
    let nk = eris.nkpts;
    let kd = kconserv.get(ka, kc, kb_) as usize;
    let (t1aa, t1ab) = (t1.0.slice_leading(&[ka])?, t1.0.slice_leading(&[kb_])?);
    let (t1ba, t1bb) = (t1.1.slice_leading(&[ka])?, t1.1.slice_leading(&[kb_])?);

    // `:362-370`
    let mut vvvv = einsum(
        "emfa,mb->aebf",
        &[&eris.blk(b::vovv, kc, kb_, kd)?.conj(), &t1ab],
    )?;
    vvvv.sub_assign(&einsum(
        "fmea,mb->aebf",
        &[&eris.blk(b::vovv, kd, kb_, kc)?.conj(), &t1ab],
    )?)?;
    vvvv.sub_assign(&einsum(
        "emfb,ma->aebf",
        &[&eris.blk(b::vovv, kc, ka, kd)?.conj(), &t1aa],
    )?)?;
    vvvv.add_assign(&einsum(
        "fmeb,ma->aebf",
        &[&eris.blk(b::vovv, kd, ka, kc)?.conj(), &t1aa],
    )?)?;
    vvvv.add_assign(&eris.blk(b::vvvv, ka, kc, kb_)?)?;
    vvvv.sub_assign(&eris.blk(b::vvvv, kb_, kc, ka)?.transpose(&[2, 1, 0, 3])?)?;
    vvvv.add_assign(&einsum(
        "mcnf,ma,nb->acbf",
        &[&eris.blk(b::ovov, ka, kc, kb_)?, &t1aa, &t1ab],
    )?)?;
    vvvv.sub_assign(&einsum(
        "mcnf,mb,na->acbf",
        &[&eris.blk(b::ovov, kb_, kc, ka)?, &t1ab, &t1aa],
    )?)?;

    // `:372-375`
    let mut vv_vv = einsum_scale(
        "emfb,ma->aebf",
        &[&eris.blk(b::voVV, kc, ka, kd)?.conj(), &t1aa],
        -1.0,
    )?;
    vv_vv.add_assign(&einsum_scale(
        "fmea,mb->aebf",
        &[&eris.blk(b::VOvv, kd, kb_, kc)?.conj(), &t1bb],
        -1.0,
    )?)?;
    vv_vv.add_assign(&einsum(
        "mcnf,ma,nb->acbf",
        &[&eris.blk(b::ovOV, ka, kc, kb_)?, &t1aa, &t1bb],
    )?)?;
    vv_vv.add_assign(&eris.blk(b::vvVV, ka, kc, kb_)?)?;

    // `:377-385`
    let mut v4 = einsum(
        "emfa,mb->aebf",
        &[&eris.blk(b::VOVV, kc, kb_, kd)?.conj(), &t1bb],
    )?;
    v4.sub_assign(&einsum(
        "fmea,mb->aebf",
        &[&eris.blk(b::VOVV, kd, kb_, kc)?.conj(), &t1bb],
    )?)?;
    v4.sub_assign(&einsum(
        "emfb,ma->aebf",
        &[&eris.blk(b::VOVV, kc, ka, kd)?.conj(), &t1ba],
    )?)?;
    v4.add_assign(&einsum(
        "fmeb,ma->aebf",
        &[&eris.blk(b::VOVV, kd, ka, kc)?.conj(), &t1ba],
    )?)?;
    v4.add_assign(&eris.blk(b::VVVV, ka, kc, kb_)?)?;
    v4.sub_assign(&eris.blk(b::VVVV, kb_, kc, ka)?.transpose(&[2, 1, 0, 3])?)?;
    v4.add_assign(&einsum(
        "mcnf,ma,nb->acbf",
        &[&eris.blk(b::OVOV, ka, kc, kb_)?, &t1ba, &t1bb],
    )?)?;
    v4.sub_assign(&einsum(
        "mcnf,mb,na->acbf",
        &[&eris.blk(b::OVOV, kb_, kc, ka)?, &t1bb, &t1ba],
    )?)?;

    // `:387-391`
    for km in 0..nk {
        let kn = kconserv.get(ka, km, kb_) as usize;
        vvvv.add_assign(&einsum(
            "mnab,mcnf->acbf",
            &[
                &t2.0.slice_leading(&[km, kn, ka])?,
                &eris.blk(b::ovov, km, kc, kn)?,
            ],
        )?)?;
        vv_vv.add_assign(&einsum(
            "mNaB,mcNF->acBF",
            &[
                &t2.1.slice_leading(&[km, kn, ka])?,
                &eris.blk(b::ovOV, km, kc, kn)?,
            ],
        )?)?;
        v4.add_assign(&einsum(
            "mnab,mcnf->acbf",
            &[
                &t2.2.slice_leading(&[km, kn, ka])?,
                &eris.blk(b::OVOV, km, kc, kn)?,
            ],
        )?)?;
    }
    Ok((vvvv, vv_vv, v4))
}

/// `Woovo` — `:879-994`.
///
/// The `P(ij)` antisymmetriser is written out term by term upstream, and the
/// two halves contract DIFFERENT k-blocks, so a transpose of the assembled
/// result is not the same program (the same point [`crate::kintermediates::wovoo`]
/// makes).
///
/// # Errors
/// As [`cc_foo`].
#[allow(clippy::too_many_lines)]
pub fn woovo(t1: &UT1, t2: &UT2, eris: &KuEris, kconserv: &Kconserv) -> Result<UQuad, PbcCcError> {
    let nk = eris.nkpts;
    let (oa, ob) = eris.nocc;
    let (va, vb) = eris.nvir;
    let mut w_aa = ZArr::zeros(&[nk, nk, nk, oa, oa, va, oa]);
    let mut w_ab = ZArr::zeros(&[nk, nk, nk, oa, oa, vb, ob]);
    let mut w_ba = ZArr::zeros(&[nk, nk, nk, ob, ob, va, oa]);
    let mut w_bb = ZArr::zeros(&[nk, nk, nk, ob, ob, vb, ob]);

    for km in 0..nk {
        for kb_ in 0..nk {
            for ki in 0..nk {
                let kj = kconserv.get(km, ki, kb_) as usize;
                // `:893-898`
                let mut aa = eris
                    .blk(b::ooov, ki, km, kj)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                aa.sub_assign(
                    &eris
                        .blk(b::ooov, kj, km, ki)?
                        .transpose(&[1, 2, 3, 0])?
                        .conj(),
                )?;
                let mut ab = eris
                    .blk(b::ooOV, ki, km, kj)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                let mut ba = eris
                    .blk(b::OOov, ki, km, kj)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                let mut bb = eris
                    .blk(b::OOOV, ki, km, kj)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                bb.sub_assign(
                    &eris
                        .blk(b::OOOV, kj, km, ki)?
                        .transpose(&[1, 2, 3, 0])?
                        .conj(),
                )?;

                for kn in 0..nk {
                    let ke = kconserv.get(km, ki, kn) as usize;
                    let mut ooov = eris.blk(b::ooov, km, ki, kn)?;
                    ooov.sub_assign(&eris.blk(b::ooov, kn, ki, km)?.transpose(&[2, 1, 0, 3])?)?;
                    let mut ooov_b = eris.blk(b::OOOV, km, ki, kn)?;
                    ooov_b.sub_assign(&eris.blk(b::OOOV, kn, ki, km)?.transpose(&[2, 1, 0, 3])?)?;

                    aa.add_assign(&einsum(
                        "mine,jnbe->mibj",
                        &[&ooov, &t2.0.slice_leading(&[kj, kn, kb_])?],
                    )?)?;
                    aa.add_assign(&einsum(
                        "miNE,jNbE->mibj",
                        &[
                            &eris.blk(b::ooOV, km, ki, kn)?,
                            &t2.1.slice_leading(&[kj, kn, kb_])?,
                        ],
                    )?)?;
                    ab.add_assign(&einsum(
                        "mine,nJeB->miBJ",
                        &[&ooov, &t2.1.slice_leading(&[kn, kj, ke])?],
                    )?)?;
                    ab.add_assign(&einsum(
                        "miNE,JNBE->miBJ",
                        &[
                            &eris.blk(b::ooOV, km, ki, kn)?,
                            &t2.2.slice_leading(&[kj, kn, kb_])?,
                        ],
                    )?)?;
                    ba.add_assign(&einsum(
                        "MINE,jNbE->MIbj",
                        &[&ooov_b, &t2.1.slice_leading(&[kj, kn, kb_])?],
                    )?)?;
                    ba.add_assign(&einsum(
                        "MIne,jnbe->MIbj",
                        &[
                            &eris.blk(b::OOov, km, ki, kn)?,
                            &t2.0.slice_leading(&[kj, kn, kb_])?,
                        ],
                    )?)?;
                    bb.add_assign(&einsum(
                        "MINE,JNBE->MIBJ",
                        &[&ooov_b, &t2.2.slice_leading(&[kj, kn, kb_])?],
                    )?)?;
                    bb.add_assign(&einsum(
                        "MIne,nJeB->MIBJ",
                        &[
                            &eris.blk(b::OOov, km, ki, kn)?,
                            &t2.1.slice_leading(&[kn, kj, ke])?,
                        ],
                    )?)?;

                    // P(ij) — `:913-925`
                    let ke = kconserv.get(km, kj, kn) as usize;
                    let mut ooov = eris.blk(b::ooov, km, kj, kn)?;
                    ooov.sub_assign(&eris.blk(b::ooov, kn, kj, km)?.transpose(&[2, 1, 0, 3])?)?;
                    let mut ooov_b = eris.blk(b::OOOV, km, kj, kn)?;
                    ooov_b.sub_assign(&eris.blk(b::OOOV, kn, kj, km)?.transpose(&[2, 1, 0, 3])?)?;

                    aa.sub_assign(&einsum(
                        "mjne,inbe->mibj",
                        &[&ooov, &t2.0.slice_leading(&[ki, kn, kb_])?],
                    )?)?;
                    aa.sub_assign(&einsum(
                        "mjNE,iNbE->mibj",
                        &[
                            &eris.blk(b::ooOV, km, kj, kn)?,
                            &t2.1.slice_leading(&[ki, kn, kb_])?,
                        ],
                    )?)?;
                    ab.sub_assign(&einsum(
                        "NJme,iNeB->miBJ",
                        &[
                            &eris.blk(b::OOov, kn, kj, km)?,
                            &t2.1.slice_leading(&[ki, kn, ke])?,
                        ],
                    )?)?;
                    ba.sub_assign(&einsum(
                        "njME,nIbE->MIbj",
                        &[
                            &eris.blk(b::ooOV, kn, kj, km)?,
                            &t2.1.slice_leading(&[kn, ki, kb_])?,
                        ],
                    )?)?;
                    bb.sub_assign(&einsum(
                        "MJNE,INBE->MIBJ",
                        &[&ooov_b, &t2.2.slice_leading(&[ki, kn, kb_])?],
                    )?)?;
                    bb.sub_assign(&einsum(
                        "MJne,nIeB->MIBJ",
                        &[
                            &eris.blk(b::OOov, km, kj, kn)?,
                            &t2.1.slice_leading(&[kn, ki, ke])?,
                        ],
                    )?)?;
                }

                // `:927-937`
                let (t1ai, t1bi) = (t1.0.slice_leading(&[ki])?, t1.1.slice_leading(&[ki])?);
                let (t1aj, t1bj) = (t1.0.slice_leading(&[kj])?, t1.1.slice_leading(&[kj])?);
                let mut ovvo = eris
                    .blk(b::voov, ki, km, kj)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                ovvo.sub_assign(&eris.blk(b::oovv, km, kj, kb_)?.transpose(&[0, 3, 2, 1])?)?;
                let mut ovvo_b = eris
                    .blk(b::VOOV, ki, km, kj)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                ovvo_b.sub_assign(&eris.blk(b::OOVV, km, kj, kb_)?.transpose(&[0, 3, 2, 1])?)?;
                let ov_vo = eris
                    .blk(b::voOV, ki, km, kj)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                let ovv_o = eris
                    .blk(b::VOov, ki, km, kj)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                aa.add_assign(&einsum("ie,mebj->mibj", &[&t1ai, &ovvo])?)?;
                ab.add_assign(&einsum("ie,meBJ->miBJ", &[&t1ai, &ov_vo])?)?;
                ba.add_assign(&einsum("IE,MEbj->MIbj", &[&t1bi, &ovv_o])?)?;
                bb.add_assign(&einsum("IE,MEBJ->MIBJ", &[&t1bi, &ovvo_b])?)?;

                // P(ij) — `:939-945`
                let mut ovvo = eris
                    .blk(b::voov, kj, km, ki)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                ovvo.sub_assign(&eris.blk(b::oovv, km, ki, kb_)?.transpose(&[0, 3, 2, 1])?)?;
                let mut ovvo_b = eris
                    .blk(b::VOOV, kj, km, ki)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                ovvo_b.sub_assign(&eris.blk(b::OOVV, km, ki, kb_)?.transpose(&[0, 3, 2, 1])?)?;
                aa.sub_assign(&einsum("je,mebi->mibj", &[&t1aj, &ovvo])?)?;
                // `:942-943` — the DOUBLE negative upstream writes as
                // `-= -einsum(...)`, kept as an `add` here.
                ab.add_assign(&einsum(
                    "JE,miBE->miBJ",
                    &[&t1bj, &eris.blk(b::ooVV, km, ki, kb_)?],
                )?)?;
                ba.add_assign(&einsum(
                    "je,MIbe->MIbj",
                    &[&t1aj, &eris.blk(b::OOvv, km, ki, kb_)?],
                )?)?;
                bb.sub_assign(&einsum("JE,MEBI->MIBJ", &[&t1bj, &ovvo_b])?)?;

                // `:948-975`
                for kf in 0..nk {
                    let kn = kconserv.get(kb_, kj, kf) as usize;
                    let mut ovov = eris.blk(b::ovov, km, ki, kn)?;
                    ovov.sub_assign(&eris.blk(b::ovov, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;
                    let mut ovov_b = eris.blk(b::OVOV, km, ki, kn)?;
                    ovov_b.sub_assign(&eris.blk(b::OVOV, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;

                    aa.sub_assign(&einsum(
                        "ie,njbf,menf->mibj",
                        &[&t1ai, &t2.0.slice_leading(&[kn, kj, kb_])?, &ovov],
                    )?)?;
                    aa.add_assign(&einsum(
                        "ie,jNbF,meNF->mibj",
                        &[
                            &t1ai,
                            &t2.1.slice_leading(&[kj, kn, kb_])?,
                            &eris.blk(b::ovOV, km, ki, kn)?,
                        ],
                    )?)?;
                    ab.add_assign(&einsum(
                        "ie,nJfB,menf->miBJ",
                        &[&t1ai, &t2.1.slice_leading(&[kn, kj, kf])?, &ovov],
                    )?)?;
                    ab.sub_assign(&einsum(
                        "ie,NJBF,meNF->miBJ",
                        &[
                            &t1ai,
                            &t2.2.slice_leading(&[kn, kj, kb_])?,
                            &eris.blk(b::ovOV, km, ki, kn)?,
                        ],
                    )?)?;
                    ba.add_assign(&einsum(
                        "IE,jNbF,MENF->MIbj",
                        &[&t1bi, &t2.1.slice_leading(&[kj, kn, kb_])?, &ovov_b],
                    )?)?;
                    ba.sub_assign(&einsum(
                        "IE,njbf,MEnf->MIbj",
                        &[
                            &t1bi,
                            &t2.0.slice_leading(&[kn, kj, kb_])?,
                            &eris.blk(b::OVov, km, ki, kn)?,
                        ],
                    )?)?;
                    bb.sub_assign(&einsum(
                        "IE,NJBF,MENF->MIBJ",
                        &[&t1bi, &t2.2.slice_leading(&[kn, kj, kb_])?, &ovov_b],
                    )?)?;
                    bb.add_assign(&einsum(
                        "IE,nJfB,MEnf->MIBJ",
                        &[
                            &t1bi,
                            &t2.1.slice_leading(&[kn, kj, kf])?,
                            &eris.blk(b::OVov, km, ki, kn)?,
                        ],
                    )?)?;

                    // P(ij) — `:967-975`
                    let kn = kconserv.get(kb_, ki, kf) as usize;
                    let mut ovov = eris.blk(b::ovov, km, kj, kn)?;
                    ovov.sub_assign(&eris.blk(b::ovov, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;
                    let mut ovov_b = eris.blk(b::OVOV, km, kj, kn)?;
                    ovov_b.sub_assign(&eris.blk(b::OVOV, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;

                    aa.add_assign(&einsum(
                        "je,nibf,menf->mibj",
                        &[&t1aj, &t2.0.slice_leading(&[kn, ki, kb_])?, &ovov],
                    )?)?;
                    aa.sub_assign(&einsum(
                        "je,iNbF,meNF->mibj",
                        &[
                            &t1aj,
                            &t2.1.slice_leading(&[ki, kn, kb_])?,
                            &eris.blk(b::ovOV, km, kj, kn)?,
                        ],
                    )?)?;
                    ab.sub_assign(&einsum(
                        "JE,iNfB,mfNE->miBJ",
                        &[
                            &t1bj,
                            &t2.1.slice_leading(&[ki, kn, kf])?,
                            &eris.blk(b::ovOV, km, kf, kn)?,
                        ],
                    )?)?;
                    ba.sub_assign(&einsum(
                        "je,nIbF,MFne->MIbj",
                        &[
                            &t1aj,
                            &t2.1.slice_leading(&[kn, ki, kb_])?,
                            &eris.blk(b::OVov, km, kf, kn)?,
                        ],
                    )?)?;
                    bb.add_assign(&einsum(
                        "JE,NIBF,MENF->MIBJ",
                        &[&t1bj, &t2.2.slice_leading(&[kn, ki, kb_])?, &ovov_b],
                    )?)?;
                    bb.sub_assign(&einsum(
                        "JE,nIfB,MEnf->MIBJ",
                        &[
                            &t1bj,
                            &t2.1.slice_leading(&[kn, ki, kf])?,
                            &eris.blk(b::OVov, km, kj, kn)?,
                        ],
                    )?)?;
                }
                add_at(&mut w_aa, [km, ki, kb_], &aa, 1.0)?;
                add_at(&mut w_ab, [km, ki, kb_], &ab, 1.0)?;
                add_at(&mut w_ba, [km, ki, kb_], &ba, 1.0)?;
                add_at(&mut w_bb, [km, ki, kb_], &bb, 1.0)?;
            }
        }
    }

    // `:977-991` — a SECOND pass, because `Fov`, `Woooo` and `tau` are built
    // after the first loop upstream.
    let (fme, fme_b) = fov(t1, eris)?;
    let (wminj, wmi_nj, w_minj) = eom_woooo(t1, t2, eris, kconserv)?;
    let tau = make_tau(t2, t1, t1, 1.0)?;
    for km in 0..nk {
        for kb_ in 0..nk {
            for ki in 0..nk {
                let kj = kconserv.get(km, ki, kb_) as usize;
                add_at(
                    &mut w_aa,
                    [km, ki, kb_],
                    &einsum(
                        "me,ijbe->mibj",
                        &[
                            &fme.slice_leading(&[km])?,
                            &t2.0.slice_leading(&[ki, kj, kb_])?,
                        ],
                    )?,
                    -1.0,
                )?;
                // `:980` `-= -einsum(...)`
                add_at(
                    &mut w_ab,
                    [km, ki, kb_],
                    &einsum(
                        "me,iJeB->miBJ",
                        &[
                            &fme.slice_leading(&[km])?,
                            &t2.1.slice_leading(&[ki, kj, km])?,
                        ],
                    )?,
                    1.0,
                )?;
                add_at(
                    &mut w_ba,
                    [km, ki, kb_],
                    &einsum(
                        "ME,jIbE->MIbj",
                        &[
                            &fme_b.slice_leading(&[km])?,
                            &t2.1.slice_leading(&[kj, ki, kb_])?,
                        ],
                    )?,
                    1.0,
                )?;
                add_at(
                    &mut w_bb,
                    [km, ki, kb_],
                    &einsum(
                        "ME,IJBE->MIBJ",
                        &[
                            &fme_b.slice_leading(&[km])?,
                            &t2.2.slice_leading(&[ki, kj, kb_])?,
                        ],
                    )?,
                    -1.0,
                )?;

                add_at(
                    &mut w_aa,
                    [km, ki, kb_],
                    &einsum(
                        "nb,minj->mibj",
                        &[
                            &t1.0.slice_leading(&[kb_])?,
                            &wminj.slice_leading(&[km, ki, kb_])?,
                        ],
                    )?,
                    -1.0,
                )?;
                add_at(
                    &mut w_ab,
                    [km, ki, kb_],
                    &einsum(
                        "NB,miNJ->miBJ",
                        &[
                            &t1.1.slice_leading(&[kb_])?,
                            &wmi_nj.slice_leading(&[km, ki, kb_])?,
                        ],
                    )?,
                    -1.0,
                )?;
                // `:989` — `WmiNJ[kb, kj, km]`, a DIFFERENT k-address from the
                // line above it.
                add_at(
                    &mut w_ba,
                    [km, ki, kb_],
                    &einsum(
                        "nb,njMI->MIbj",
                        &[
                            &t1.0.slice_leading(&[kb_])?,
                            &wmi_nj.slice_leading(&[kb_, kj, km])?,
                        ],
                    )?,
                    -1.0,
                )?;
                add_at(
                    &mut w_bb,
                    [km, ki, kb_],
                    &einsum(
                        "NB,MINJ->MIBJ",
                        &[
                            &t1.1.slice_leading(&[kb_])?,
                            &w_minj.slice_leading(&[km, ki, kb_])?,
                        ],
                    )?,
                    -1.0,
                )?;
            }
        }
    }

    // `:993-1005`
    for km in 0..nk {
        for kb_ in 0..nk {
            for ki in 0..nk {
                let kj = kconserv.get(km, ki, kb_) as usize;
                for ke in 0..nk {
                    let kf = kconserv.get(km, ke, kb_) as usize;
                    let mut ovvv = eris
                        .blk(b::vovv, ke, km, kf)?
                        .transpose(&[1, 0, 3, 2])?
                        .conj();
                    ovvv.sub_assign(
                        &eris
                            .blk(b::vovv, kf, km, ke)?
                            .transpose(&[1, 2, 3, 0])?
                            .conj(),
                    )?;
                    let mut ovvv_b = eris
                        .blk(b::VOVV, ke, km, kf)?
                        .transpose(&[1, 0, 3, 2])?
                        .conj();
                    ovvv_b.sub_assign(
                        &eris
                            .blk(b::VOVV, kf, km, ke)?
                            .transpose(&[1, 2, 3, 0])?
                            .conj(),
                    )?;
                    let ov_vv = eris
                        .blk(b::voVV, ke, km, kf)?
                        .transpose(&[1, 0, 3, 2])?
                        .conj();
                    let ovv_v = eris
                        .blk(b::VOvv, ke, km, kf)?
                        .transpose(&[1, 0, 3, 2])?
                        .conj();

                    add_at(
                        &mut w_aa,
                        [km, ki, kb_],
                        &einsum(
                            "mebf,ijef->mibj",
                            &[&ovvv, &tau.0.slice_leading(&[ki, kj, ke])?],
                        )?,
                        0.5,
                    )?;
                    add_at(
                        &mut w_ab,
                        [km, ki, kb_],
                        &einsum(
                            "meBF,iJeF->miBJ",
                            &[&ov_vv, &tau.1.slice_leading(&[ki, kj, ke])?],
                        )?,
                        1.0,
                    )?;
                    add_at(
                        &mut w_ba,
                        [km, ki, kb_],
                        &einsum(
                            "MEbf,jIfE->MIbj",
                            &[&ovv_v, &tau.1.slice_leading(&[kj, ki, kf])?],
                        )?,
                        1.0,
                    )?;
                    add_at(
                        &mut w_bb,
                        [km, ki, kb_],
                        &einsum(
                            "MEBF,IJEF->MIBJ",
                            &[&ovvv_b, &tau.2.slice_leading(&[ki, kj, ke])?],
                        )?,
                        0.5,
                    )?;
                }
            }
        }
    }
    Ok((w_aa, w_ab, w_ba, w_bb))
}

/// `Wvvvo` — `:647-824`. The phase's densest single intermediate: four spin
/// blocks, three separate `(ka, ke, kb)` sweeps and two `P(ab)` writes that
/// land at the MIRRORED `[kb, ke, ka]` address.
///
/// # Errors
/// As [`cc_foo`].
#[allow(clippy::too_many_lines)]
pub fn wvvvo(t1: &UT1, t2: &UT2, eris: &KuEris, kconserv: &Kconserv) -> Result<UQuad, PbcCcError> {
    let nk = eris.nkpts;
    let (oa, ob) = eris.nocc;
    let (va, vb) = eris.nvir;
    let (fova, fovb) = cc_fov(t1, eris)?;
    let tau = make_tau(t2, t1, t1, 1.0)?;

    let mut w_aa = ZArr::zeros(&[nk, nk, nk, va, va, va, oa]);
    let mut w_ab = ZArr::zeros(&[nk, nk, nk, va, va, vb, ob]);
    let mut w_ba = ZArr::zeros(&[nk, nk, nk, vb, vb, va, oa]);
    let mut w_bb = ZArr::zeros(&[nk, nk, nk, vb, vb, vb, ob]);

    // ---- sweep 1 (`:660-724`)
    for ka in 0..nk {
        for ke in 0..nk {
            for kb_ in 0..nk {
                let ki = kconserv.get(ka, ke, kb_) as usize;
                for km in 0..nk {
                    let kf = kconserv.get(km, ke, kb_) as usize;
                    let mut ovvv = eris
                        .blk(b::vovv, ke, km, kf)?
                        .transpose(&[1, 0, 3, 2])?
                        .conj();
                    ovvv.sub_assign(
                        &eris
                            .blk(b::vovv, kf, km, ke)?
                            .transpose(&[1, 2, 3, 0])?
                            .conj(),
                    )?;
                    let ovv_v = eris
                        .blk(b::VOvv, ke, km, kf)?
                        .transpose(&[1, 0, 3, 2])?
                        .conj();
                    let ov_vv = eris
                        .blk(b::voVV, ke, km, kf)?
                        .transpose(&[1, 0, 3, 2])?
                        .conj();
                    let mut ovvv_b = eris
                        .blk(b::VOVV, ke, km, kf)?
                        .transpose(&[1, 0, 3, 2])?
                        .conj();
                    ovvv_b.sub_assign(
                        &eris
                            .blk(b::VOVV, kf, km, ke)?
                            .transpose(&[1, 2, 3, 0])?
                            .conj(),
                    )?;

                    let mut aebi = einsum(
                        "mebf,miaf->aebi",
                        &[&ovvv, &t2.0.slice_leading(&[km, ki, ka])?],
                    )?;
                    aebi.add_assign(&einsum(
                        "MFbe,iMaF->aebi",
                        &[
                            &eris
                                .blk(b::VOvv, kf, km, ke)?
                                .transpose(&[1, 0, 3, 2])?
                                .conj(),
                            &t2.1.slice_leading(&[ki, km, ka])?,
                        ],
                    )?)?;
                    add_at(&mut w_aa, [ka, ke, kb_], &aebi, -1.0)?;
                    // P(ab), at the MIRRORED address.
                    add_at(
                        &mut w_aa,
                        [kb_, ke, ka],
                        &aebi.transpose(&[2, 1, 0, 3])?,
                        1.0,
                    )?;

                    add_at(
                        &mut w_ba,
                        [ka, ke, kb_],
                        &einsum(
                            "MEbf,iMfA->AEbi",
                            &[&ovv_v, &t2.1.slice_leading(&[ki, km, kf])?],
                        )?,
                        -1.0,
                    )?;
                    add_at(
                        &mut w_ab,
                        [ka, ke, kb_],
                        &einsum(
                            "meBF,mIaF->aeBI",
                            &[&ov_vv, &t2.1.slice_leading(&[km, ki, ka])?],
                        )?,
                        -1.0,
                    )?;

                    let mut aebi_b = einsum(
                        "MEBF,MIAF->AEBI",
                        &[&ovvv_b, &t2.2.slice_leading(&[km, ki, ka])?],
                    )?;
                    aebi_b.add_assign(&einsum(
                        "mfBE,mIfA->AEBI",
                        &[
                            &eris
                                .blk(b::voVV, kf, km, ke)?
                                .transpose(&[1, 0, 3, 2])?
                                .conj(),
                            &t2.1.slice_leading(&[km, ki, kf])?,
                        ],
                    )?)?;
                    add_at(&mut w_bb, [ka, ke, kb_], &aebi_b, -1.0)?;
                    add_at(
                        &mut w_bb,
                        [kb_, ke, ka],
                        &aebi_b.transpose(&[2, 1, 0, 3])?,
                        1.0,
                    )?;
                }

                // `:686-724` — `km = ka`.
                let km = ka;
                let mut ovvo = eris
                    .blk(b::voov, ke, km, ki)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                ovvo.sub_assign(&eris.blk(b::oovv, km, ki, kb_)?.transpose(&[0, 3, 2, 1])?)?;
                let ovv_o = eris
                    .blk(b::VOov, ke, km, ki)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                let ov_vo = eris
                    .blk(b::voOV, ke, km, ki)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                let mut ovvo_b = eris
                    .blk(b::VOOV, ke, km, ki)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                ovvo_b.sub_assign(&eris.blk(b::OOVV, km, ki, kb_)?.transpose(&[0, 3, 2, 1])?)?;

                let mut t_aa = ZArr::zeros(&[oa, va, va, oa]);
                let mut t_ab = ZArr::zeros(&[oa, va, vb, ob]);
                let mut t_ba = ZArr::zeros(&[ob, vb, va, oa]);
                let mut t_bb = ZArr::zeros(&[ob, vb, vb, ob]);
                for kn in 0..nk {
                    let kf = kconserv.get(km, ke, kn) as usize;
                    let mut ovov = eris.blk(b::ovov, km, ke, kn)?;
                    ovov.sub_assign(&eris.blk(b::ovov, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;
                    let ov_ov = eris.blk(b::OVov, km, ke, kn)?;
                    let ovov_x = eris.blk(b::ovOV, km, ke, kn)?;
                    let mut ovov_b = eris.blk(b::OVOV, km, ke, kn)?;
                    ovov_b.sub_assign(&eris.blk(b::OVOV, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;

                    t_aa.sub_assign(&einsum(
                        "nibf,menf->mebi",
                        &[&t2.0.slice_leading(&[kn, ki, kb_])?, &ovov],
                    )?)?;
                    t_aa.add_assign(&einsum(
                        "iNbF,meNF->mebi",
                        &[&t2.1.slice_leading(&[ki, kn, kb_])?, &ovov_x],
                    )?)?;
                    t_ab.add_assign(&einsum(
                        "nIfB,menf->meBI",
                        &[&t2.1.slice_leading(&[kn, ki, kf])?, &ovov],
                    )?)?;
                    t_ab.sub_assign(&einsum(
                        "NIBF,meNF->meBI",
                        &[&t2.2.slice_leading(&[kn, ki, kb_])?, &ovov_x],
                    )?)?;
                    t_ba.add_assign(&einsum(
                        "iNbF,MENF->MEbi",
                        &[&t2.1.slice_leading(&[ki, kn, kb_])?, &ovov_b],
                    )?)?;
                    t_ba.sub_assign(&einsum(
                        "nibf,MEnf->MEbi",
                        &[&t2.0.slice_leading(&[kn, ki, kb_])?, &ov_ov],
                    )?)?;
                    t_bb.sub_assign(&einsum(
                        "NIBF,MENF->MEBI",
                        &[&t2.2.slice_leading(&[kn, ki, kb_])?, &ovov_b],
                    )?)?;
                    t_bb.add_assign(&einsum(
                        "nIfB,MEnf->MEBI",
                        &[&t2.1.slice_leading(&[kn, ki, kf])?, &ov_ov],
                    )?)?;
                }
                let mut x = ovvo.clone();
                x.add_assign(&t_aa)?;
                add_at(
                    &mut w_aa,
                    [ka, ke, kb_],
                    &einsum("ma,mebi->aebi", &[&t1.0.slice_leading(&[km])?, &x])?,
                    -1.0,
                )?;
                let mut x = ovv_o.clone();
                x.add_assign(&t_ba)?;
                add_at(
                    &mut w_ba,
                    [ka, ke, kb_],
                    &einsum("MA,MEbi->AEbi", &[&t1.1.slice_leading(&[km])?, &x])?,
                    -1.0,
                )?;
                let mut x = ov_vo.clone();
                x.add_assign(&t_ab)?;
                add_at(
                    &mut w_ab,
                    [ka, ke, kb_],
                    &einsum("ma,meBI->aeBI", &[&t1.0.slice_leading(&[km])?, &x])?,
                    -1.0,
                )?;
                let mut x = ovvo_b.clone();
                x.add_assign(&t_bb)?;
                add_at(
                    &mut w_bb,
                    [ka, ke, kb_],
                    &einsum("MA,MEBI->AEBI", &[&t1.1.slice_leading(&[km])?, &x])?,
                    -1.0,
                )?;
            }
        }
    }

    // ---- sweep 2 (`:727-785`)
    for ka in 0..nk {
        for ke in 0..nk {
            for kb_ in 0..nk {
                let ki = kconserv.get(ka, ke, kb_) as usize;
                for km in 0..nk {
                    let kf = kconserv.get(km, ke, ka) as usize;
                    let mut ovvv_b = eris
                        .blk(b::VOVV, ke, km, kf)?
                        .transpose(&[1, 0, 3, 2])?
                        .conj();
                    ovvv_b.sub_assign(
                        &eris
                            .blk(b::VOVV, kf, km, ke)?
                            .transpose(&[1, 2, 3, 0])?
                            .conj(),
                    )?;
                    let mut ovvv = eris
                        .blk(b::vovv, ke, km, kf)?
                        .transpose(&[1, 0, 3, 2])?
                        .conj();
                    ovvv.sub_assign(
                        &eris
                            .blk(b::vovv, kf, km, ke)?
                            .transpose(&[1, 2, 3, 0])?
                            .conj(),
                    )?;

                    add_at(
                        &mut w_ba,
                        [ka, ke, kb_],
                        &einsum(
                            "mfAE,mibf->AEbi",
                            &[
                                &eris
                                    .blk(b::voVV, kf, km, ke)?
                                    .transpose(&[1, 0, 3, 2])?
                                    .conj(),
                                &t2.0.slice_leading(&[km, ki, kb_])?,
                            ],
                        )?,
                        -1.0,
                    )?;
                    add_at(
                        &mut w_ba,
                        [ka, ke, kb_],
                        &einsum(
                            "MEAF,iMbF->AEbi",
                            &[&ovvv_b, &t2.1.slice_leading(&[ki, km, kb_])?],
                        )?,
                        -1.0,
                    )?;
                    add_at(
                        &mut w_ab,
                        [ka, ke, kb_],
                        &einsum(
                            "MFae,MIBF->aeBI",
                            &[
                                &eris
                                    .blk(b::VOvv, kf, km, ke)?
                                    .transpose(&[1, 0, 3, 2])?
                                    .conj(),
                                &t2.2.slice_leading(&[km, ki, kb_])?,
                            ],
                        )?,
                        -1.0,
                    )?;
                    add_at(
                        &mut w_ab,
                        [ka, ke, kb_],
                        &einsum(
                            "meaf,mIfB->aeBI",
                            &[&ovvv, &t2.1.slice_leading(&[km, ki, kf])?],
                        )?,
                        -1.0,
                    )?;
                }

                // `:747-785` — `km = kb`.
                let km = kb_;
                let mut ovvo = eris
                    .blk(b::voov, ke, km, ki)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                ovvo.sub_assign(&eris.blk(b::oovv, km, ki, ka)?.transpose(&[0, 3, 2, 1])?)?;
                let mut ovvo_b = eris
                    .blk(b::VOOV, ke, km, ki)?
                    .transpose(&[1, 0, 3, 2])?
                    .conj();
                ovvo_b.sub_assign(&eris.blk(b::OOVV, km, ki, ka)?.transpose(&[0, 3, 2, 1])?)?;

                let mut t_aa = ZArr::zeros(&[oa, va, va, oa]);
                let mut t_ab = ZArr::zeros(&[ob, va, va, ob]);
                let mut t_ba = ZArr::zeros(&[oa, vb, vb, oa]);
                let mut t_bb = ZArr::zeros(&[ob, vb, vb, ob]);
                for kn in 0..nk {
                    let kf = kconserv.get(km, ke, kn) as usize;
                    let mut ovov = eris.blk(b::ovov, km, ke, kn)?;
                    ovov.sub_assign(&eris.blk(b::ovov, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;
                    let ov_ov = eris.blk(b::OVov, km, ke, kn)?;
                    let ovov_x = eris.blk(b::ovOV, km, ke, kn)?;
                    let mut ovov_b = eris.blk(b::OVOV, km, ke, kn)?;
                    ovov_b.sub_assign(&eris.blk(b::OVOV, km, kf, kn)?.transpose(&[0, 3, 2, 1])?)?;

                    t_aa.sub_assign(&einsum(
                        "niaf,menf->meai",
                        &[&t2.0.slice_leading(&[kn, ki, ka])?, &ovov],
                    )?)?;
                    t_aa.add_assign(&einsum(
                        "iNaF,meNF->meai",
                        &[&t2.1.slice_leading(&[ki, kn, ka])?, &ovov_x],
                    )?)?;
                    // `:772` and `:774` read `OVov`/`ovOV` at `kf`, NOT at `ke`.
                    t_ab.add_assign(&einsum(
                        "nIaF,MFne->MeaI",
                        &[
                            &t2.1.slice_leading(&[kn, ki, ka])?,
                            &eris.blk(b::OVov, km, kf, kn)?,
                        ],
                    )?)?;
                    t_ba.add_assign(&einsum(
                        "iNfA,mfNE->mEAi",
                        &[
                            &t2.1.slice_leading(&[ki, kn, kf])?,
                            &eris.blk(b::ovOV, km, kf, kn)?,
                        ],
                    )?)?;
                    t_bb.sub_assign(&einsum(
                        "NIAF,MENF->MEAI",
                        &[&t2.2.slice_leading(&[kn, ki, ka])?, &ovov_b],
                    )?)?;
                    t_bb.add_assign(&einsum(
                        "nIfA,MEnf->MEAI",
                        &[&t2.1.slice_leading(&[kn, ki, kf])?, &ov_ov],
                    )?)?;
                }
                let mut x = ovvo.clone();
                x.add_assign(&t_aa)?;
                add_at(
                    &mut w_aa,
                    [ka, ke, kb_],
                    &einsum("mb,meai->aebi", &[&t1.0.slice_leading(&[km])?, &x])?,
                    1.0,
                )?;
                let mut x = eris.blk(b::ooVV, km, ki, ka)?.transpose(&[0, 3, 2, 1])?;
                x.scale(-1.0);
                x.add_assign(&t_ba)?;
                add_at(
                    &mut w_ba,
                    [ka, ke, kb_],
                    &einsum("mb,mEAi->AEbi", &[&t1.0.slice_leading(&[km])?, &x])?,
                    1.0,
                )?;
                let mut x = eris.blk(b::OOvv, km, ki, ka)?.transpose(&[0, 3, 2, 1])?;
                x.scale(-1.0);
                x.add_assign(&t_ab)?;
                add_at(
                    &mut w_ab,
                    [ka, ke, kb_],
                    &einsum("MB,MeaI->aeBI", &[&t1.1.slice_leading(&[km])?, &x])?,
                    1.0,
                )?;
                let mut x = ovvo_b.clone();
                x.add_assign(&t_bb)?;
                add_at(
                    &mut w_bb,
                    [ka, ke, kb_],
                    &einsum("MB,MEAI->AEBI", &[&t1.1.slice_leading(&[km])?, &x])?,
                    1.0,
                )?;
            }
        }
    }

    // ---- sweep 3, "Remaining terms" (`:787-822`)
    for ka in 0..nk {
        for ke in 0..nk {
            for kb_ in 0..nk {
                let ki = kconserv.get(ka, ke, kb_) as usize;
                let kf = ki;

                let mut v = eris.blk(b::vovv, kb_, ki, ka)?.transpose(&[2, 3, 0, 1])?;
                v.sub_assign(&eris.blk(b::vovv, ka, ki, kb_)?.transpose(&[0, 3, 2, 1])?)?;
                add_at(&mut w_aa, [ka, ke, kb_], &v, 1.0)?;
                add_at(
                    &mut w_ba,
                    [ka, ke, kb_],
                    &eris.blk(b::voVV, kb_, ki, ka)?.transpose(&[2, 3, 0, 1])?,
                    1.0,
                )?;
                add_at(
                    &mut w_ab,
                    [ka, ke, kb_],
                    &eris.blk(b::VOvv, kb_, ki, ka)?.transpose(&[2, 3, 0, 1])?,
                    1.0,
                )?;
                let mut v = eris.blk(b::VOVV, kb_, ki, ka)?.transpose(&[2, 3, 0, 1])?;
                v.sub_assign(&eris.blk(b::VOVV, ka, ki, kb_)?.transpose(&[0, 3, 2, 1])?)?;
                add_at(&mut w_bb, [ka, ke, kb_], &v, 1.0)?;

                add_at(
                    &mut w_aa,
                    [ka, ke, kb_],
                    &einsum(
                        "me,miab->aebi",
                        &[
                            &fova.slice_leading(&[ke])?,
                            &t2.0.slice_leading(&[ke, ki, ka])?,
                        ],
                    )?,
                    -1.0,
                )?;
                add_at(
                    &mut w_ba,
                    [ka, ke, kb_],
                    &einsum(
                        "ME,iMbA->AEbi",
                        &[
                            &fovb.slice_leading(&[ke])?,
                            &t2.1.slice_leading(&[ki, ke, kb_])?,
                        ],
                    )?,
                    -1.0,
                )?;
                add_at(
                    &mut w_ab,
                    [ka, ke, kb_],
                    &einsum(
                        "me,mIaB->aeBI",
                        &[
                            &fova.slice_leading(&[ke])?,
                            &t2.1.slice_leading(&[ke, ki, ka])?,
                        ],
                    )?,
                    -1.0,
                )?;
                add_at(
                    &mut w_bb,
                    [ka, ke, kb_],
                    &einsum(
                        "ME,MIAB->AEBI",
                        &[
                            &fovb.slice_leading(&[ke])?,
                            &t2.2.slice_leading(&[ke, ki, ka])?,
                        ],
                    )?,
                    -1.0,
                )?;

                let (g_aa, g_ab, g_bb) = get_wvvvv(t1, t2, eris, kconserv, ka, kb_, ke)?;
                add_at(
                    &mut w_aa,
                    [ka, ke, kb_],
                    &einsum("if,aebf->aebi", &[&t1.0.slice_leading(&[ki])?, &g_aa])?,
                    1.0,
                )?;
                // `:812` — a DIFFERENT k-address, `[kb, kf, ka]` with `kf = ki`.
                add_at(
                    &mut w_ba,
                    [kb_, kf, ka],
                    &einsum("ie,aeBF->BFai", &[&t1.0.slice_leading(&[ke])?, &g_ab])?,
                    1.0,
                )?;
                add_at(
                    &mut w_ab,
                    [ka, ke, kb_],
                    &einsum("IF,aeBF->aeBI", &[&t1.1.slice_leading(&[ki])?, &g_ab])?,
                    1.0,
                )?;
                add_at(
                    &mut w_bb,
                    [ka, ke, kb_],
                    &einsum("IF,AEBF->AEBI", &[&t1.1.slice_leading(&[ki])?, &g_bb])?,
                    1.0,
                )?;

                for km in 0..nk {
                    let kn = kconserv.get(ka, km, kb_) as usize;
                    let mut ovoo = eris.blk(b::ooov, kn, ki, km)?.transpose(&[2, 3, 0, 1])?;
                    ovoo.sub_assign(&eris.blk(b::ooov, km, ki, kn)?.transpose(&[0, 3, 2, 1])?)?;
                    let ov_oo = eris.blk(b::OOov, kn, ki, km)?.transpose(&[2, 3, 0, 1])?;
                    let oo_ov = eris.blk(b::ooOV, km, ki, kn)?;
                    let oo_ov_b = eris.blk(b::OOov, km, ki, kn)?;
                    let ovoo_x = eris.blk(b::ooOV, kn, ki, km)?.transpose(&[2, 3, 0, 1])?;
                    let mut ovoo_b = eris.blk(b::OOOV, kn, ki, km)?.transpose(&[2, 3, 0, 1])?;
                    ovoo_b.sub_assign(&eris.blk(b::OOOV, km, ki, kn)?.transpose(&[0, 3, 2, 1])?)?;

                    add_at(
                        &mut w_aa,
                        [ka, ke, kb_],
                        &einsum(
                            "meni,mnab->aebi",
                            &[&ovoo, &tau.0.slice_leading(&[km, kn, ka])?],
                        )?,
                        0.5,
                    )?;
                    add_at(
                        &mut w_ba,
                        [ka, ke, kb_],
                        &einsum(
                            "MEni,nMbA->AEbi",
                            &[&ovoo_x, &tau.1.slice_leading(&[kn, km, kb_])?],
                        )?,
                        0.5,
                    )?;
                    add_at(
                        &mut w_ba,
                        [ka, ke, kb_],
                        &einsum(
                            "miNE,mNbA->AEbi",
                            &[&oo_ov, &tau.1.slice_leading(&[km, kn, kb_])?],
                        )?,
                        0.5,
                    )?;
                    add_at(
                        &mut w_ab,
                        [ka, ke, kb_],
                        &einsum(
                            "meNI,mNaB->aeBI",
                            &[&ov_oo, &tau.1.slice_leading(&[km, kn, ka])?],
                        )?,
                        0.5,
                    )?;
                    add_at(
                        &mut w_ab,
                        [ka, ke, kb_],
                        &einsum(
                            "MIne,nMaB->aeBI",
                            &[&oo_ov_b, &tau.1.slice_leading(&[kn, km, ka])?],
                        )?,
                        0.5,
                    )?;
                    add_at(
                        &mut w_bb,
                        [ka, ke, kb_],
                        &einsum(
                            "MENI,MNAB->AEBI",
                            &[&ovoo_b, &tau.2.slice_leading(&[km, kn, ka])?],
                        )?,
                        0.5,
                    )?;
                }
            }
        }
    }
    Ok((w_aa, w_ab, w_ba, w_bb))
}

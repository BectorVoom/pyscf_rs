//! The k-SYMMETRIC restricted CC intermediates —
//! `pyscf/pbc/cc/kintermediates_rhf_ksymm.py` (265 l), plan 17-09.
//!
//! # What differs from `kintermediates_rhf`, and why it is not a wrapper
//!
//! Every intermediate here is built on the **`kqrts_ibz` k-quartets only** and
//! then unfolded to the full zone by the `KsymmArray` rotation. Two
//! consequences that have no analogue in the non-symmetric module:
//!
//! * **The rank-2 intermediates are SYMMETRISED, not just computed.** `cc_Foo`
//!   accumulates `einsum('ki,km,in->mn', fock, R.conj(), R)` over the
//!   STABILISER of each IBZ quartet (`kintermediates_rhf_ksymm.py:44-46`) — the
//!   operations that fix the quartet. That is a projection onto the symmetric
//!   part, and dropping it silently gives an intermediate that is right only
//!   where the little co-group is trivial.
//! * **`cc_Fvv` and `update_amps`' second T1 term scatter to a DIFFERENT
//!   k-point.** `ka_prim = kpts.k2opk[ka, op_group]` (`:94`) sends the quartet's
//!   contribution to every IBZ representative reachable from `ka`, not back to
//!   `ka`. Using the stabiliser there instead — the shape the other terms have —
//!   would drop every contribution whose image leaves `ka`.
//!
//! Both are transcribed with the upstream line beside them for that reason.
//!
//! # `cc_Woooo` / `cc_Wvvvv` skip half their quartets on purpose
//!
//! `_s2_index` (`:140-146`) finds the pairs `(i, j)`, `i < j`, whose IBZ
//! quartets are each other's dummy-index interchange `[1,0,3,2]`. The `j` of
//! each pair is SKIPPED in the build loop and then filled by transposing block
//! `i` (`:177-179`, `:205-207`). It is a genuine halving of the `oooo` and
//! `vvvv` work, and it is exact — `W[kl,kk,kj] == W[kk,kl,ki].transpose(1,0,3,2)`
//! is a symmetry of the intermediate itself, not an approximation.

use pyscf_pbc_symm::ktensor::{KsymmArray, OrbSpace};

use crate::error::PbcCcError;
use crate::kccsd_rhf_ksymm::KsymEris;
use crate::keris::Blk;
use crate::ksymm_common::{
    KsymCtx, add_stored2, rot, set_stored2, set_stored4, to_dense2, to_dense4,
};
use crate::zarr::{ZArr, einsum, einsum_scaled};

// =====================================================================
// The rank-2 intermediates (`:25-137`)
// =====================================================================

/// `cc_Foo` (`:25-47`) as the `KsymmArray` upstream keeps, so `Loo` can add to
/// it and `update_amps` can subtract the orbital energies from its diagonal
/// IN PLACE (upstream's `Foo[ki][np.diag_indices(nocc)] -= mo_e_o[ki]` works
/// because `transform_2d` returns a numpy VIEW of the store at a
/// representative, `ktensor.py:269-270`).
///
/// # Errors
/// Propagates every ERI access, rotation lookup and shape check.
pub fn cc_foo_arr<'a>(
    ctx: &'a KsymCtx<'a>,
    t1: &ZArr,
    t2: &ZArr,
    eris: &KsymEris,
) -> Result<KsymmArray<'a>, PbcCcError> {
    let no = ctx.nocc;
    let mut f = ctx.empty2([no, no], &ctx.labels.oo, &ctx.labels.cn)?;
    for i in 0..ctx.nkpts_ibz() {
        let ki = ctx.kpts.ibz2bz[i];
        set_stored2(&mut f, ctx.kpts, ki, &eris.foo(ki)?)?;
    }
    for (i, kq) in ctx.kqrts_ibz().iter().enumerate() {
        let [ki, kl, kc, kd] = *kq;
        let kk = ki;
        let mut soovv = eris.blk(Blk::Oovv, kk, kl, kc)?;
        soovv.scale(2.0);
        soovv.sub_assign(&eris.blk(Blk::Oovv, kk, kl, kd)?.transpose(&[0, 1, 3, 2])?)?;
        let mut fock = einsum(
            "klcd,ilcd->ki",
            &[&soovv, &t2.slice_leading(&[ki, kl, kc])?],
        )?;
        if ki == kc {
            fock.add_assign(&einsum(
                "klcd,ic,ld->ki",
                &[&soovv, &t1.slice_leading(&[ki])?, &t1.slice_leading(&[kl])?],
            )?)?;
        }
        for (_, iop) in ctx.kqrts.loop_stabilizer(i) {
            let r = rot(ctx.rmat, OrbSpace::Occ, ki, iop)?;
            let d = einsum("ki,km,in->mn", &[&fock, &r.conj(), &r])?;
            add_stored2(&mut f, ctx.kpts, ki, &d)?;
        }
    }
    Ok(f)
}

/// `cc_Fov` (`:49-70`).
///
/// # Errors
/// As [`cc_foo_arr`].
pub fn cc_fov_arr<'a>(
    ctx: &'a KsymCtx<'a>,
    t1: &ZArr,
    _t2: &ZArr,
    eris: &KsymEris,
) -> Result<KsymmArray<'a>, PbcCcError> {
    let (no, nv) = (ctx.nocc, ctx.nvir);
    let mut f = ctx.empty2([no, nv], &ctx.labels.ov, &ctx.labels.cn)?;
    for i in 0..ctx.nkpts_ibz() {
        let ki = ctx.kpts.ibz2bz[i];
        set_stored2(&mut f, ctx.kpts, ki, &eris.fov(ki)?)?;
    }
    for (i, kq) in ctx.kqrts_ibz().iter().enumerate() {
        let [kk, kl, kc, kd] = *kq;
        if !(kc == kk && kl == kd) {
            continue;
        }
        let mut soovv = eris.blk(Blk::Oovv, kk, kl, kk)?;
        soovv.scale(2.0);
        soovv.sub_assign(&eris.blk(Blk::Oovv, kk, kl, kl)?.transpose(&[0, 1, 3, 2])?)?;
        let fock = einsum("klcd,ld->kc", &[&soovv, &t1.slice_leading(&[kl])?])?;
        for (_, iop) in ctx.kqrts.loop_stabilizer(i) {
            let roo = rot(ctx.rmat, OrbSpace::Occ, kk, iop)?;
            let rvv = rot(ctx.rmat, OrbSpace::Vir, kk, iop)?;
            let d = einsum("kc,km,cb->mb", &[&fock, &roo.conj(), &rvv])?;
            add_stored2(&mut f, ctx.kpts, kk, &d)?;
        }
    }
    Ok(f)
}

/// `cc_Fvv` (`:72-99`).
///
/// **This one SCATTERS.** `ka_prim = kpts.k2opk[ka, op_group]` (`:94`) is the
/// image of `ka` under every operation of the quartet's star, and the
/// contribution lands at whichever of those images is an IBZ representative —
/// not back at `ka`. Substituting the stabiliser loop the other rank-2
/// intermediates use would drop every term whose image leaves `ka`.
///
/// # Errors
/// As [`cc_foo_arr`].
pub fn cc_fvv_arr<'a>(
    ctx: &'a KsymCtx<'a>,
    t1: &ZArr,
    t2: &ZArr,
    eris: &KsymEris,
) -> Result<KsymmArray<'a>, PbcCcError> {
    let nv = ctx.nvir;
    let mut f = ctx.empty2([nv, nv], &ctx.labels.vv, &ctx.labels.cn)?;
    for i in 0..ctx.nkpts_ibz() {
        let ki = ctx.kpts.ibz2bz[i];
        set_stored2(&mut f, ctx.kpts, ki, &eris.fvv(ki)?)?;
    }
    for (i, kq) in ctx.kqrts_ibz().iter().enumerate() {
        let [kk, kl, ka, kd] = *kq;
        let kc = ka;
        let mut soovv = eris.blk(Blk::Oovv, kk, kl, kc)?;
        soovv.scale(2.0);
        soovv.sub_assign(&eris.blk(Blk::Oovv, kk, kl, kd)?.transpose(&[0, 1, 3, 2])?)?;
        let mut fock = einsum_scaled(
            "klcd,klad->ac",
            &[&soovv, &t2.slice_leading(&[kk, kl, ka])?],
            -1.0,
        )?;
        if kk == ka {
            fock.add_assign(&einsum_scaled(
                "klcd,ka,ld->ac",
                &[&soovv, &t1.slice_leading(&[ka])?, &t1.slice_leading(&[kl])?],
                -1.0,
            )?)?;
        }
        for &iop in &ctx.kqrts.stars_ops[i] {
            let ka_p = ctx.kpts.k2opk[ka][iop];
            if ka_p < 0 {
                continue;
            }
            let ka_p = ka_p as usize;
            if crate::ksymm_common::ibz2_index(ctx.kpts, ka_p).is_none() {
                continue; // `np.isin(ka_prim, ka_ibz_bz)` (`:95`)
            }
            let rvv = rot(ctx.rmat, OrbSpace::Vir, ka, iop)?;
            let d = einsum("ac,ae,cf->ef", &[&fock, &rvv.conj(), &rvv])?;
            add_stored2(&mut f, ctx.kpts, ka_p, &d)?;
        }
    }
    Ok(f)
}

/// `Loo` (`:101-118`).
///
/// # Errors
/// As [`cc_foo_arr`].
pub fn loo_arr<'a>(
    ctx: &'a KsymCtx<'a>,
    t1: &ZArr,
    t2: &ZArr,
    eris: &KsymEris,
) -> Result<KsymmArray<'a>, PbcCcError> {
    let no = ctx.nocc;
    let mut l = cc_foo_arr(ctx, t1, t2, eris)?;
    for i in 0..ctx.nkpts_ibz() {
        let ki = ctx.kpts.ibz2bz[i];
        let d = einsum("kc,ic->ki", &[&eris.fov(ki)?, &t1.slice_leading(&[ki])?])?;
        add_stored2(&mut l, ctx.kpts, ki, &d)?;
    }
    for (i, kq) in ctx.kqrts_ibz().iter().enumerate() {
        let [ki, kl, ka, _kb] = *kq;
        if ki != ka {
            continue;
        }
        let t1l = t1.slice_leading(&[kl])?;
        let mut fock = einsum_scaled(
            "klic,lc->ki",
            &[&eris.blk(Blk::Ooov, ki, kl, ki)?, &t1l],
            2.0,
        )?;
        fock.sub_assign(&einsum(
            "lkic,lc->ki",
            &[&eris.blk(Blk::Ooov, kl, ki, ki)?, &t1l],
        )?)?;
        for (_, iop) in ctx.kqrts.loop_stabilizer(i) {
            let r = rot(ctx.rmat, OrbSpace::Occ, ki, iop)?;
            let d = einsum("ki,km,in->mn", &[&fock, &r.conj(), &r])?;
            add_stored2(&mut l, ctx.kpts, ki, &d)?;
        }
    }
    let _ = no;
    Ok(l)
}

/// `Lvv` (`:120-137`).
///
/// # Errors
/// As [`cc_foo_arr`].
pub fn lvv_arr<'a>(
    ctx: &'a KsymCtx<'a>,
    t1: &ZArr,
    t2: &ZArr,
    eris: &KsymEris,
) -> Result<KsymmArray<'a>, PbcCcError> {
    let mut l = cc_fvv_arr(ctx, t1, t2, eris)?;
    for i in 0..ctx.nkpts_ibz() {
        let ka = ctx.kpts.ibz2bz[i];
        let d = einsum_scaled(
            "kc,ka->ac",
            &[&eris.fov(ka)?, &t1.slice_leading(&[ka])?],
            -1.0,
        )?;
        add_stored2(&mut l, ctx.kpts, ka, &d)?;
    }
    for (i, kq) in ctx.kqrts_ibz().iter().enumerate() {
        let [ka, kk, kc, _kl] = *kq;
        if ka != kc {
            continue;
        }
        let mut svovv = eris.blk(Blk::Vovv, ka, kk, ka)?;
        svovv.scale(2.0);
        svovv.sub_assign(&eris.blk(Blk::Vovv, ka, kk, kk)?.transpose(&[0, 1, 3, 2])?)?;
        let fock = einsum("akcd,kd->ac", &[&svovv, &t1.slice_leading(&[kk])?])?;
        for (_, iop) in ctx.kqrts.loop_stabilizer(i) {
            let rvv = rot(ctx.rmat, OrbSpace::Vir, ka, iop)?;
            let d = einsum("ac,ae,cf->ef", &[&fock, &rvv.conj(), &rvv])?;
            add_stored2(&mut l, ctx.kpts, ka, &d)?;
        }
    }
    Ok(l)
}

/// Dense wrappers, for the places `update_amps` calls `.todense()`.
///
/// # Errors
/// As the `_arr` builders.
pub fn cc_fov(
    ctx: &KsymCtx<'_>,
    t1: &ZArr,
    t2: &ZArr,
    eris: &KsymEris,
) -> Result<ZArr, PbcCcError> {
    let a = cc_fov_arr(ctx, t1, t2, eris)?;
    to_dense2(&a, ctx.nkpts(), [ctx.nocc, ctx.nvir])
}

// =====================================================================
// The rank-4 intermediates (`:140-265`)
// =====================================================================

/// `_s2_index` (`:140-146`) — the indices `j` of the pairs `(i, j)`, `i < j`,
/// whose IBZ quartets are each other's `[1,0,3,2]` interchange.
///
/// Upstream's `np.where` returns the pairs in row-major order of the
/// `(i, j)` grid, and it keeps only `i < j`; this scan reproduces both.
pub fn s2_index(kqrts_ibz: &[[usize; 4]]) -> Vec<usize> {
    let mut out = Vec::new();
    for (i, q) in kqrts_ibz.iter().enumerate() {
        let t = [q[1], q[0], q[3], q[2]];
        for (j, r) in kqrts_ibz.iter().enumerate() {
            if *r == t && i < j {
                out.push(j);
            }
        }
    }
    out
}

/// `cc_Woooo` (`:148-180`).
///
/// # Errors
/// As [`cc_foo_arr`].
pub fn cc_woooo(
    ctx: &KsymCtx<'_>,
    t1: &ZArr,
    t2: &ZArr,
    eris: &KsymEris,
) -> Result<ZArr, PbcCcError> {
    let (no, nv, nk) = (ctx.nocc, ctx.nvir, ctx.nkpts());
    let mut w = ctx.empty4([no, no, no, no], &ctx.labels.oooo, &ctx.labels.ccnn)?;
    let skip = s2_index(ctx.kqrts_ibz());

    for (i, kq) in ctx.kqrts_ibz().iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        let [kk, kl, ki, kj] = *kq;
        let t1i = t1.slice_leading(&[ki])?;
        let t1j = t1.slice_leading(&[kj])?;
        let mut oooo = einsum("klic,jc->klij", &[&eris.blk(Blk::Ooov, kk, kl, ki)?, &t1j])?;
        oooo.add_assign(&einsum(
            "lkjc,ic->klij",
            &[&eris.blk(Blk::Ooov, kl, kk, kj)?, &t1i],
        )?)?;
        oooo.add_assign(&eris.blk(Blk::Oooo, kk, kl, ki)?)?;

        // `:170-174` — the merged `(ka, c)` axis, done as an explicit `ka`
        // loop: `vvoo = eris.oovv[kk,kl].transpose(0,3,4,1,2)` and
        // `t2t = t2[ki,kj].transpose(0,3,4,1,2)` share that free axis.
        for ka in 0..nk {
            let vvoo = eris.blk(Blk::Oovv, kk, kl, ka)?.transpose(&[2, 3, 0, 1])?;
            let mut t2t = t2.slice_leading(&[ki, kj, ka])?.transpose(&[2, 3, 0, 1])?;
            if ka == ki {
                t2t.add_assign(&einsum("ic,jd->cdij", &[&t1i, &t1j])?)?;
            }
            oooo.add_assign(&einsum("cdkl,cdij->klij", &[&vvoo, &t2t])?)?;
        }
        set_stored4(&mut w, ctx.kpts, ctx.kqrts, [kk, kl, ki], &oooo)?;
    }

    for &j in &skip {
        let [kl, kk, kj, ki] = ctx.kqrts_ibz()[j];
        let src = crate::ksymm_common::get_stored4(
            &w,
            ctx.kpts,
            ctx.kqrts,
            [kk, kl, ki],
            [no, no, no, no],
        )?;
        set_stored4(
            &mut w,
            ctx.kpts,
            ctx.kqrts,
            [kl, kk, kj],
            &src.transpose(&[1, 0, 3, 2])?,
        )?;
    }
    let _ = nv;
    to_dense4(&w, nk, [no, no, no, no])
}

/// `cc_Wvvvv` (`:182-208`).
///
/// # Errors
/// As [`cc_foo_arr`].
pub fn cc_wvvvv(
    ctx: &KsymCtx<'_>,
    t1: &ZArr,
    _t2: &ZArr,
    eris: &KsymEris,
) -> Result<ZArr, PbcCcError> {
    let (nv, nk) = (ctx.nvir, ctx.nkpts());
    let mut w = ctx.empty4([nv, nv, nv, nv], &ctx.labels.vvvv, &ctx.labels.ccnn)?;
    let skip = s2_index(ctx.kqrts_ibz());

    for (i, kq) in ctx.kqrts_ibz().iter().enumerate() {
        if skip.contains(&i) {
            continue;
        }
        let [ka, kb, kc, kd] = *kq;
        let mut vvvv = einsum_scaled(
            "akcd,kb->abcd",
            &[&eris.blk(Blk::Vovv, ka, kb, kc)?, &t1.slice_leading(&[kb])?],
            -1.0,
        )?;
        vvvv.add_assign(&einsum_scaled(
            "bkdc,ka->abcd",
            &[&eris.blk(Blk::Vovv, kb, ka, kd)?, &t1.slice_leading(&[ka])?],
            -1.0,
        )?)?;
        vvvv.add_assign(&eris.blk(Blk::Vvvv, ka, kb, kc)?)?;
        set_stored4(&mut w, ctx.kpts, ctx.kqrts, [ka, kb, kc], &vvvv)?;
    }

    for &j in &skip {
        let [kb, ka, kd, kc] = ctx.kqrts_ibz()[j];
        let src = crate::ksymm_common::get_stored4(&w, ctx.kpts, ctx.kqrts, [ka, kb, kc], [nv; 4])?;
        set_stored4(
            &mut w,
            ctx.kpts,
            ctx.kqrts,
            [kb, ka, kd],
            &src.transpose(&[1, 0, 3, 2])?,
        )?;
    }
    to_dense4(&w, nk, [nv; 4])
}

/// `cc_Wvoov` (`:210-237`).
///
/// # Errors
/// As [`cc_foo_arr`].
pub fn cc_wvoov(
    ctx: &KsymCtx<'_>,
    t1: &ZArr,
    t2: &ZArr,
    eris: &KsymEris,
) -> Result<ZArr, PbcCcError> {
    let (no, nv, nk) = (ctx.nocc, ctx.nvir, ctx.nkpts());
    let mut w = ctx.empty4([nv, no, no, nv], &ctx.labels.voov, &ctx.labels.ccnn)?;

    for kq in ctx.kqrts_ibz().iter() {
        let [ka, kk, ki, kc] = *kq;
        let t1i = t1.slice_leading(&[ki])?;
        let t1a = t1.slice_leading(&[ka])?;
        let mut voov = einsum("akdc,id->akic", &[&eris.blk(Blk::Vovv, ka, kk, ki)?, &t1i])?;
        voov.sub_assign(&einsum(
            "lkic,la->akic",
            &[&eris.blk(Blk::Ooov, ka, kk, ki)?, &t1a],
        )?)?;
        voov.add_assign(&eris.blk(Blk::Voov, ka, kk, ki)?)?;

        // `:228-234` — the free `x` axis, explicit.
        let kd = ki;
        let t1d = t1.slice_leading(&[kd])?;
        for x in 0..nk {
            let mut tau = t2.slice_leading(&[x, ki, ka])?;
            if x == ka {
                tau.add_assign(&einsum_scaled("id,la->liad", &[&t1d, &t1a], 2.0)?)?;
            }
            let oovv_x = eris.blk(Blk::Oovv, kk, x, kc)?;
            voov.add_assign(&einsum_scaled("klcd,liad->akic", &[&oovv_x, &tau], -0.5)?)?;

            let mut soovv = oovv_x.clone();
            soovv.scale(2.0);
            soovv.sub_assign(&eris.blk(Blk::Oovv, x, kk, kc)?.transpose(&[1, 0, 2, 3])?)?;
            voov.add_assign(&einsum_scaled(
                "klcd,ilad->akic",
                &[&soovv, &t2.slice_leading(&[ki, x, ka])?],
                0.5,
            )?)?;
        }
        set_stored4(&mut w, ctx.kpts, ctx.kqrts, [ka, kk, ki], &voov)?;
    }
    to_dense4(&w, nk, [nv, no, no, nv])
}

/// `cc_Wvovo` (`:239-265`).
///
/// # Errors
/// As [`cc_foo_arr`].
pub fn cc_wvovo(
    ctx: &KsymCtx<'_>,
    t1: &ZArr,
    t2: &ZArr,
    eris: &KsymEris,
) -> Result<ZArr, PbcCcError> {
    let (no, nv, nk) = (ctx.nocc, ctx.nvir, ctx.nkpts());
    let mut w = ctx.empty4([nv, no, nv, no], &ctx.labels.vovo, &ctx.labels.ccnn)?;

    for kq in ctx.kqrts_ibz().iter() {
        let [ka, kk, kc, ki] = *kq;
        let t1i = t1.slice_leading(&[ki])?;
        let t1a = t1.slice_leading(&[ka])?;
        let mut vovo = einsum("akcd,id->akci", &[&eris.blk(Blk::Vovv, ka, kk, kc)?, &t1i])?;
        vovo.sub_assign(&einsum(
            "klic,la->akci",
            &[&eris.blk(Blk::Ooov, kk, ka, ki)?, &t1a],
        )?)?;
        vovo.add_assign(&eris.blk(Blk::Ovov, kk, ka, ki)?.transpose(&[1, 0, 3, 2])?)?;

        // `:257-262` — the merged `(x, l)` axis, explicit.
        for x in 0..nk {
            let mut t2f = t2.slice_leading(&[x, ki, ka])?;
            if x == ka {
                t2f.add_assign(&einsum_scaled("id,la->liad", &[&t1i, &t1a], 2.0)?)?;
            }
            vovo.add_assign(&einsum_scaled(
                "lkcd,liad->akci",
                &[&eris.blk(Blk::Oovv, x, kk, kc)?, &t2f],
                -0.5,
            )?)?;
        }
        set_stored4(&mut w, ctx.kpts, ctx.kqrts, [ka, kk, kc], &vovo)?;
    }
    to_dense4(&w, nk, [nv, no, nv, no])
}

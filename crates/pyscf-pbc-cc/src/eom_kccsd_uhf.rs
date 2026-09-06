//! `eom_kccsd_uhf` — equation-of-motion CCSD over UNRESTRICTED k-point orbitals
//! (plan 16-11; `pyscf/pbc/cc/eom_kccsd_uhf.py`, 1275 l).
//!
//! # IP and EA are the whole surface
//!
//! `eom_kccsd_uhf` declares **no `EOMEE` class at all** and its
//! `_IMDS.make_ee` (`:1120`) is a bare `raise NotImplementedError`
//! (`16-CONTEXT §1.5`). Nothing is deferred here that upstream implements.
//!
//! # `r2` is FOUR blocks, and two of them carry a triangle
//!
//! `amplitudes_to_vector_ip` (`:43-57`) packs `r1a`, `r1b`, the strict lower
//! triangle of `r2aaa` over the composite `(k, occ)` index, ALL of `r2baa` and
//! `r2abb`, then the strict lower triangle of `r2bbb`. The two same-spin blocks
//! are antisymmetric in `(ki,i) <-> (kj,j)`; the two mixed ones are not, and
//! storing them triangularly would lose information.
//!
//! `r2baa` is `[nkpts, nkpts, noccb, nocca, nvira]` and `r2abb` is
//! `[nkpts, nkpts, nocca, noccb, nvirb]` — different shapes, not transposes.

use std::sync::Arc;

use pyscf_algebra::CTensor;
use pyscf_pbc_lib::Kconserv;
use pyscf_runtime::ZWorkspacePool;

use crate::error::PbcCcError;
use crate::kintermediates_uhf::{self as uimd, UT1, UT2, b};
use crate::kueris::KuEris;
use crate::zarr::{ZArr, einsum, einsum_scaled};

/// The intermediates `eom_kccsd_uhf._IMDS` caches (`:1040-1130`).
pub struct UhfEomImds<'a> {
    pub eris: &'a KuEris,
    pub t1: UT1,
    pub t2: UT2,
    /// `(Foo, FOO)`.
    pub foo: (ZArr, ZArr),
    /// `(Fvv, FVV)`.
    pub fvv: (ZArr, ZArr),
    /// `(Fov, FOV)`.
    pub fov: (ZArr, ZArr),
    /// `(Wovvo, WovVO, WOVvo, WOVVO)`.
    pub wovvo: uimd::UQuad,
    /// `(Woovv, WooVV, WOOvv, WOOVV)`.
    pub woovv: uimd::UQuad,
    /// `Wovov = ovov - ovov.transpose(2,1,0,5,4,3,6)` (`:1071`).
    pub wovov: ZArr,
    /// `WOVOV`, likewise (`:1072`).
    pub wovov_b: ZArr,
    /// `WovOV = eris.ovOV` — an ALIAS, not a built intermediate (`:1073`).
    pub wov_ov: ZArr,
    /// IP: `(Woooo, WooOO, WOOOO)`.
    pub woooo: Option<uimd::UTriple>,
    /// IP: `(Wooov, WooOV, WOOov, WOOOV)`.
    pub wooov: Option<uimd::UQuad>,
    /// IP: `(Woovo, WooVO, WOOvo, WOOVO)`.
    pub woovo: Option<uimd::UQuad>,
    /// EA: `(Wvvov, WvvOV, WVVov, WVVOV)`.
    pub wvvov: Option<uimd::UQuad>,
    /// EA: `(Wvvvv, WvvVV, WVVVV)`.
    pub wvvvv: Option<uimd::UTriple>,
    /// EA: `(Wvvvo, WvvVO, WVVvo, WVVVO)`.
    pub wvvvo: Option<uimd::UQuad>,
}

impl<'a> UhfEomImds<'a> {
    /// `_IMDS._make_shared` (`:1060-1078`).
    ///
    /// # Errors
    /// Propagates every intermediate build.
    pub fn make_shared(
        pool: &Arc<ZWorkspacePool>,
        budget: usize,
        t1: &UT1,
        t2: &UT2,
        eris: &'a KuEris,
        kconserv: &Kconserv,
    ) -> Result<Self, PbcCcError> {
        let nk = eris.nkpts;
        // `:1071-1073` — the two same-spin `Wovov` are antisymmetrised copies
        // of the ERI blocks; `WovOV` is the block itself.
        let mut wovov = gather_blk(eris, b::ovov)?;
        let mut wovov_b = gather_blk(eris, b::OVOV)?;
        for (w, blk) in [(&mut wovov, b::ovov), (&mut wovov_b, b::OVOV)] {
            for kx in 0..nk {
                for ky in 0..nk {
                    for kz in 0..nk {
                        let t = eris.blk(blk, kz, ky, kx)?.transpose(&[2, 1, 0, 3])?;
                        let mut cur = w.slice_leading(&[kx, ky, kz])?;
                        cur.sub_assign(&t)?;
                        w.set_leading(&[kx, ky, kz], &cur)?;
                    }
                }
            }
        }
        Ok(Self {
            eris,
            t1: (t1.0.clone(), t1.1.clone()),
            t2: (t2.0.clone(), t2.1.clone(), t2.2.clone()),
            foo: uimd::foo(t1, t2, eris, kconserv)?,
            fvv: uimd::fvv(t1, t2, eris, kconserv)?,
            fov: uimd::fov(t1, eris)?,
            wovvo: uimd::wovvo(pool, budget, t1, t2, eris, kconserv)?,
            woovv: uimd::woovv(t1, t2, eris, kconserv)?,
            wovov,
            wovov_b,
            wov_ov: gather_blk(eris, b::ovOV)?,
            woooo: None,
            wooov: None,
            woovo: None,
            wvvov: None,
            wvvvv: None,
            wvvvo: None,
        })
    }

    /// `_IMDS.make_ip` (`:1080-1096`).
    ///
    /// # Errors
    /// Propagates every intermediate build.
    pub fn make_ip(mut self, kconserv: &Kconserv) -> Result<Self, PbcCcError> {
        let (t1, t2) = (
            (self.t1.0.clone(), self.t1.1.clone()),
            (self.t2.0.clone(), self.t2.1.clone(), self.t2.2.clone()),
        );
        self.woooo = Some(uimd::eom_woooo(&t1, &t2, self.eris, kconserv)?);
        self.wooov = Some(uimd::wooov(&t1, self.eris)?);
        self.woovo = Some(uimd::woovo(&t1, &t2, self.eris, kconserv)?);
        Ok(self)
    }

    /// `_IMDS.make_ea` (`:1098-1117`).
    ///
    /// # Errors
    /// Propagates every intermediate build.
    pub fn make_ea(
        mut self,
        pool: &Arc<ZWorkspacePool>,
        budget: usize,
        kconserv: &Kconserv,
    ) -> Result<Self, PbcCcError> {
        let (t1, t2) = (
            (self.t1.0.clone(), self.t1.1.clone()),
            (self.t2.0.clone(), self.t2.1.clone(), self.t2.2.clone()),
        );
        self.wvvov = Some(uimd::wvvov(&t1, self.eris, kconserv)?);
        self.wvvvv = Some(uimd::eom_wvvvv(
            pool, budget, &t1, &t2, self.eris, kconserv,
        )?);
        self.wvvvo = Some(uimd::wvvvo(&t1, &t2, self.eris, kconserv)?);
        Ok(self)
    }

    fn need<'w, T>(&self, w: &'w Option<T>, what: &'static str) -> Result<&'w T, PbcCcError> {
        w.as_ref()
            .ok_or_else(|| PbcCcError::Shape(format!("{what} was not built; call make_ip/make_ea")))
    }
}

fn gather_blk(eris: &KuEris, blk: crate::kueris::UBlk) -> Result<ZArr, PbcCcError> {
    let nk = eris.nkpts;
    let d = blk.dims(eris.nocc, eris.nvir);
    let mut out = ZArr::zeros(&[nk, nk, nk, d[0], d[1], d[2], d[3]]);
    for k0 in 0..nk {
        for k1 in 0..nk {
            for k2 in 0..nk {
                out.set_leading(&[k0, k1, k2], &eris.blk(blk, k0, k1, k2)?)?;
            }
        }
    }
    Ok(out)
}

/// `EOMIP.vector_size` (`:526-535`).
pub fn ip_vector_size(nkpts: usize, nocc: (usize, usize), nvir: (usize, usize)) -> usize {
    let (oa, ob) = nocc;
    let (va, vb) = nvir;
    let na = nkpts * oa;
    let nb = nkpts * ob;
    oa + ob
        + na * (na - 1) * va / 2
        + nkpts * nkpts * ob * oa * va
        + nkpts * nkpts * oa * ob * vb
        + nb * (nb - 1) * vb / 2
}

/// `vector_to_amplitudes_ip` (`:59-91`). Returns `(r1, r2)` with
/// `r2 = (r2aaa, r2baa, r2abb, r2bbb)`.
///
/// # Errors
/// [`PbcCcError::Shape`] on a length mismatch.
pub fn vector_to_amplitudes_ip(
    vector: &ZArr,
    nkpts: usize,
    nocc: (usize, usize),
    nvir: (usize, usize),
) -> Result<(UT1, uimd::UQuad), PbcCcError> {
    let (oa, ob) = nocc;
    let (va, vb) = nvir;
    let want = ip_vector_size(nkpts, nocc, nvir);
    if vector.len() != want {
        return Err(PbcCcError::Shape(format!(
            "UHF IP vector of {} elements, expected {want}",
            vector.len()
        )));
    }
    let mut cur = 0_usize;
    let mut take1 = |n: usize| -> ZArr {
        let mut a = ZArr::zeros(&[n]);
        a.data_mut()
            .re
            .copy_from_slice(&vector.data().re[cur..cur + n]);
        a.data_mut()
            .im
            .copy_from_slice(&vector.data().im[cur..cur + n]);
        cur += n;
        a
    };
    let r1a = take1(oa);
    let r1b = take1(ob);

    let mut r2aaa = ZArr::zeros(&[nkpts, nkpts, oa, oa, va]);
    fill_tri(&mut r2aaa, vector, &mut cur, nkpts, oa, va);
    let n_baa = nkpts * nkpts * ob * oa * va;
    let mut r2baa = ZArr::zeros(&[nkpts, nkpts, ob, oa, va]);
    r2baa
        .data_mut()
        .re
        .copy_from_slice(&vector.data().re[cur..cur + n_baa]);
    r2baa
        .data_mut()
        .im
        .copy_from_slice(&vector.data().im[cur..cur + n_baa]);
    cur += n_baa;
    let n_abb = nkpts * nkpts * oa * ob * vb;
    let mut r2abb = ZArr::zeros(&[nkpts, nkpts, oa, ob, vb]);
    r2abb
        .data_mut()
        .re
        .copy_from_slice(&vector.data().re[cur..cur + n_abb]);
    r2abb
        .data_mut()
        .im
        .copy_from_slice(&vector.data().im[cur..cur + n_abb]);
    cur += n_abb;
    let mut r2bbb = ZArr::zeros(&[nkpts, nkpts, ob, ob, vb]);
    fill_tri(&mut r2bbb, vector, &mut cur, nkpts, ob, vb);

    Ok(((r1a, r1b), (r2aaa, r2baa, r2abb, r2bbb)))
}

/// Unpack a strict lower triangle over the composite `(k, occ)` index into an
/// antisymmetric `[nkpts, nkpts, nocc, nocc, nvir]` array.
fn fill_tri(r2: &mut ZArr, vector: &ZArr, cur: &mut usize, nkpts: usize, nocc: usize, nvir: usize) {
    let n = nkpts * nocc;
    for p in 0..n {
        for q in 0..p {
            let (kp, i) = (p / nocc, p % nocc);
            let (kq, j) = (q / nocc, q % nocc);
            for a in 0..nvir {
                let (re, im) = (vector.data().re[*cur], vector.data().im[*cur]);
                *cur += 1;
                let f = (((kp * nkpts + kq) * nocc + i) * nocc + j) * nvir + a;
                r2.data_mut().re[f] = re;
                r2.data_mut().im[f] = im;
                let g = (((kq * nkpts + kp) * nocc + j) * nocc + i) * nvir + a;
                r2.data_mut().re[g] = -re;
                r2.data_mut().im[g] = -im;
            }
        }
    }
}

/// `amplitudes_to_vector_ip` (`:43-57`).
///
/// # Errors
/// [`PbcCcError::Shape`] on a shape mismatch.
pub fn amplitudes_to_vector_ip(
    r1: &UT1,
    r2: &uimd::UQuad,
    nkpts: usize,
) -> Result<ZArr, PbcCcError> {
    let oa = r1.0.shape()[0];
    let ob = r1.1.shape()[0];
    let va = r2.0.shape()[4];
    let vb = r2.3.shape()[4];
    let mut v = ZArr::zeros(&[ip_vector_size(nkpts, (oa, ob), (va, vb))]);
    let mut cur = 0_usize;
    for a in [&r1.0, &r1.1] {
        v.data_mut().re[cur..cur + a.len()].copy_from_slice(&a.data().re);
        v.data_mut().im[cur..cur + a.len()].copy_from_slice(&a.data().im);
        cur += a.len();
    }
    read_tri(&r2.0, &mut v, &mut cur, nkpts, oa, va);
    for a in [&r2.1, &r2.2] {
        v.data_mut().re[cur..cur + a.len()].copy_from_slice(&a.data().re);
        v.data_mut().im[cur..cur + a.len()].copy_from_slice(&a.data().im);
        cur += a.len();
    }
    read_tri(&r2.3, &mut v, &mut cur, nkpts, ob, vb);
    Ok(v)
}

fn read_tri(r2: &ZArr, v: &mut ZArr, cur: &mut usize, nkpts: usize, nocc: usize, nvir: usize) {
    let n = nkpts * nocc;
    for p in 0..n {
        for q in 0..p {
            let (kp, i) = (p / nocc, p % nocc);
            let (kq, j) = (q / nocc, q % nocc);
            for a in 0..nvir {
                let f = (((kp * nkpts + kq) * nocc + i) * nocc + j) * nvir + a;
                v.data_mut().re[*cur] = r2.data().re[f];
                v.data_mut().im[*cur] = r2.data().im[f];
                *cur += 1;
            }
        }
    }
}

/// `ipccsd_matvec` (`:93-307`).
///
/// # Errors
/// Propagates every intermediate access and shape check.
#[allow(clippy::too_many_lines)]
pub fn ipccsd_matvec(
    vector: &ZArr,
    kshift: usize,
    imds: &UhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let nk = imds.eris.nkpts;
    let (oa, ob) = imds.eris.nocc;
    let (va, vb) = imds.eris.nvir;
    let (r1, r2) = vector_to_amplitudes_ip(vector, nk, (oa, ob), (va, vb))?;
    let (r1a, r1b) = (&r1.0, &r1.1);
    let (r2aaa, r2baa, r2abb, r2bbb) = (&r2.0, &r2.1, &r2.2, &r2.3);
    let woooo = imds.need(&imds.woooo, "Woooo")?;
    let wooov = imds.need(&imds.wooov, "Wooov")?;
    let woovo = imds.need(&imds.woovo, "Woovo")?;

    // `:127-129` Foo
    let mut hr1a = einsum_scaled(
        "mi,m->i",
        &[&imds.foo.0.slice_leading(&[kshift])?, r1a],
        -1.0,
    )?;
    let mut hr1b = einsum_scaled(
        "MI,M->I",
        &[&imds.foo.1.slice_leading(&[kshift])?, r1b],
        -1.0,
    )?;
    // `:133-137` Fov
    for km in 0..nk {
        hr1a.add_assign(&einsum(
            "me,mie->i",
            &[
                &imds.fov.0.slice_leading(&[km])?,
                &r2aaa.slice_leading(&[km, kshift])?,
            ],
        )?)?;
        hr1a.sub_assign(&einsum(
            "ME,iME->i",
            &[
                &imds.fov.1.slice_leading(&[km])?,
                &r2abb.slice_leading(&[kshift, km])?,
            ],
        )?)?;
        hr1b.add_assign(&einsum(
            "ME,MIE->I",
            &[
                &imds.fov.1.slice_leading(&[km])?,
                &r2bbb.slice_leading(&[km, kshift])?,
            ],
        )?)?;
        hr1b.sub_assign(&einsum(
            "me,Ime->I",
            &[
                &imds.fov.0.slice_leading(&[km])?,
                &r2baa.slice_leading(&[kshift, km])?,
            ],
        )?)?;
    }
    // `:142-148` Wooov
    for km in 0..nk {
        for kn in 0..nk {
            hr1a.add_assign(&einsum_scaled(
                "nime,mne->i",
                &[
                    &wooov.0.slice_leading(&[kn, kshift, km])?,
                    &r2aaa.slice_leading(&[km, kn])?,
                ],
                -0.5,
            )?)?;
            hr1b.add_assign(&einsum(
                "NIme,Nme->I",
                &[
                    &wooov.2.slice_leading(&[kn, kshift, km])?,
                    &r2baa.slice_leading(&[kn, km])?,
                ],
            )?)?;
            hr1b.add_assign(&einsum_scaled(
                "NIME,MNE->I",
                &[
                    &wooov.3.slice_leading(&[kn, kshift, km])?,
                    &r2bbb.slice_leading(&[km, kn])?,
                ],
                -0.5,
            )?)?;
            hr1a.add_assign(&einsum(
                "niME,nME->i",
                &[
                    &wooov.1.slice_leading(&[kn, kshift, km])?,
                    &r2abb.slice_leading(&[kn, km])?,
                ],
            )?)?;
        }
    }

    let mut h_aaa = ZArr::zeros(&[nk, nk, oa, oa, va]);
    let mut h_baa = ZArr::zeros(&[nk, nk, ob, oa, va]);
    let mut h_abb = ZArr::zeros(&[nk, nk, oa, ob, vb]);
    let mut h_bbb = ZArr::zeros(&[nk, nk, ob, ob, vb]);

    // `:158-165` Fvv
    for kb_ in 0..nk {
        for ki in 0..nk {
            let kj = kconserv.get(kshift, ki, kb_) as usize;
            add2(
                &mut h_aaa,
                [ki, kj],
                &einsum(
                    "be,ije->ijb",
                    &[
                        &imds.fvv.0.slice_leading(&[kb_])?,
                        &r2aaa.slice_leading(&[ki, kj])?,
                    ],
                )?,
                1.0,
            )?;
            add2(
                &mut h_abb,
                [ki, kj],
                &einsum(
                    "BE,iJE->iJB",
                    &[
                        &imds.fvv.1.slice_leading(&[kb_])?,
                        &r2abb.slice_leading(&[ki, kj])?,
                    ],
                )?,
                1.0,
            )?;
            add2(
                &mut h_bbb,
                [ki, kj],
                &einsum(
                    "BE,IJE->IJB",
                    &[
                        &imds.fvv.1.slice_leading(&[kb_])?,
                        &r2bbb.slice_leading(&[ki, kj])?,
                    ],
                )?,
                1.0,
            )?;
            add2(
                &mut h_baa,
                [ki, kj],
                &einsum(
                    "be,Ije->Ijb",
                    &[
                        &imds.fvv.0.slice_leading(&[kb_])?,
                        &r2baa.slice_leading(&[ki, kj])?,
                    ],
                )?,
                1.0,
            )?;
        }
    }

    // `:174-187` Foo
    for ki in 0..nk {
        for kj in 0..nk {
            let mut tmpa = einsum(
                "mi,mjb->ijb",
                &[
                    &imds.foo.0.slice_leading(&[ki])?,
                    &r2aaa.slice_leading(&[ki, kj])?,
                ],
            )?;
            tmpa.sub_assign(&einsum(
                "mj,mib->ijb",
                &[
                    &imds.foo.0.slice_leading(&[kj])?,
                    &r2aaa.slice_leading(&[kj, ki])?,
                ],
            )?)?;
            add2(&mut h_aaa, [ki, kj], &tmpa, -1.0)?;

            add2(
                &mut h_abb,
                [ki, kj],
                &einsum(
                    "mi,mJB->iJB",
                    &[
                        &imds.foo.0.slice_leading(&[ki])?,
                        &r2abb.slice_leading(&[ki, kj])?,
                    ],
                )?,
                -1.0,
            )?;
            add2(
                &mut h_abb,
                [ki, kj],
                &einsum(
                    "MJ,iMB->iJB",
                    &[
                        &imds.foo.1.slice_leading(&[kj])?,
                        &r2abb.slice_leading(&[ki, kj])?,
                    ],
                )?,
                -1.0,
            )?;
            add2(
                &mut h_baa,
                [ki, kj],
                &einsum(
                    "MI,Mjb->Ijb",
                    &[
                        &imds.foo.1.slice_leading(&[ki])?,
                        &r2baa.slice_leading(&[ki, kj])?,
                    ],
                )?,
                -1.0,
            )?;
            add2(
                &mut h_baa,
                [ki, kj],
                &einsum(
                    "mj,Imb->Ijb",
                    &[
                        &imds.foo.0.slice_leading(&[kj])?,
                        &r2baa.slice_leading(&[ki, kj])?,
                    ],
                )?,
                -1.0,
            )?;

            let mut tmpb = einsum(
                "MI,MJB->IJB",
                &[
                    &imds.foo.1.slice_leading(&[ki])?,
                    &r2bbb.slice_leading(&[ki, kj])?,
                ],
            )?;
            tmpb.sub_assign(&einsum(
                "MJ,MIB->IJB",
                &[
                    &imds.foo.1.slice_leading(&[kj])?,
                    &r2bbb.slice_leading(&[kj, ki])?,
                ],
            )?)?;
            add2(&mut h_bbb, [ki, kj], &tmpb, -1.0)?;
        }
    }

    // `:192-198` Wovoo
    for ki in 0..nk {
        for kj in 0..nk {
            let kb_ = kconserv.get(ki, kshift, kj) as usize;
            add2(
                &mut h_aaa,
                [ki, kj],
                &einsum(
                    "mjbi,m->ijb",
                    &[&woovo.0.slice_leading(&[kshift, kj, kb_])?, r1a],
                )?,
                -1.0,
            )?;
            add2(
                &mut h_abb,
                [ki, kj],
                &einsum(
                    "miBJ,m->iJB",
                    &[&woovo.1.slice_leading(&[kshift, ki, kb_])?, r1a],
                )?,
                1.0,
            )?;
            add2(
                &mut h_baa,
                [ki, kj],
                &einsum(
                    "MIbj,M->Ijb",
                    &[&woovo.2.slice_leading(&[kshift, ki, kb_])?, r1b],
                )?,
                1.0,
            )?;
            add2(
                &mut h_bbb,
                [ki, kj],
                &einsum(
                    "MJBI,M->IJB",
                    &[&woovo.3.slice_leading(&[kshift, kj, kb_])?, r1b],
                )?,
                -1.0,
            )?;
        }
    }

    // `:203-207` Woooo
    for ki in 0..nk {
        for kj in 0..nk {
            for kn in 0..nk {
                let km = kconserv.get(kj, kn, ki) as usize;
                add2(
                    &mut h_aaa,
                    [ki, kj],
                    &einsum(
                        "minj,mnb->ijb",
                        &[
                            &woooo.0.slice_leading(&[km, ki, kn])?,
                            &r2aaa.slice_leading(&[km, kn])?,
                        ],
                    )?,
                    0.5,
                )?;
                add2(
                    &mut h_abb,
                    [ki, kj],
                    &einsum(
                        "miNJ,mNB->iJB",
                        &[
                            &woooo.1.slice_leading(&[km, ki, kn])?,
                            &r2abb.slice_leading(&[km, kn])?,
                        ],
                    )?,
                    1.0,
                )?;
                add2(
                    &mut h_bbb,
                    [ki, kj],
                    &einsum(
                        "MINJ,MNB->IJB",
                        &[
                            &woooo.2.slice_leading(&[km, ki, kn])?,
                            &r2bbb.slice_leading(&[km, kn])?,
                        ],
                    )?,
                    0.5,
                )?;
                // `:207` — `WooOO[kn, kj, km]`, a DIFFERENT k-address, and the
                // `njMI` subscript reads it with the two index pairs swapped.
                add2(
                    &mut h_baa,
                    [ki, kj],
                    &einsum(
                        "njMI,Mnb->Ijb",
                        &[
                            &woooo.1.slice_leading(&[kn, kj, km])?,
                            &r2baa.slice_leading(&[km, kn])?,
                        ],
                    )?,
                    1.0,
                )?;
            }
        }
    }

    // `:215-221` — the four `nvir`-vector intermediates.
    let mut tmp_aaa = ZArr::zeros(&[va]);
    let mut tmp_bbb = ZArr::zeros(&[vb]);
    let mut tmp_abb = ZArr::zeros(&[va]);
    let mut tmp_baa = ZArr::zeros(&[vb]);
    for kx in 0..nk {
        for ky in 0..nk {
            tmp_aaa.add_assign(&einsum(
                "menf,mnf->e",
                &[
                    &imds.wovov.slice_leading(&[kx, kshift, ky])?,
                    &r2aaa.slice_leading(&[kx, ky])?,
                ],
            )?)?;
            tmp_bbb.add_assign(&einsum(
                "MENF,MNF->E",
                &[
                    &imds.wovov_b.slice_leading(&[kx, kshift, ky])?,
                    &r2bbb.slice_leading(&[kx, ky])?,
                ],
            )?)?;
            tmp_abb.add_assign(&einsum(
                "meNF,mNF->e",
                &[
                    &imds.wov_ov.slice_leading(&[kx, kshift, ky])?,
                    &r2abb.slice_leading(&[kx, ky])?,
                ],
            )?)?;
            // `:220-221` — `WovOV[kn, kf, km]` with `kf = kconserv[kn,kshift,km]`;
            // this one is a GATHER, not a plane, so upstream writes it as a loop
            // and so does this port.
            let kf = kconserv.get(ky, kshift, kx) as usize;
            tmp_baa.add_assign(&einsum(
                "nfME,Mnf->E",
                &[
                    &imds.wov_ov.slice_leading(&[ky, kf, kx])?,
                    &r2baa.slice_leading(&[kx, ky])?,
                ],
            )?)?;
        }
    }

    // `:224-237`
    for ki in 0..nk {
        for kj in 0..nk {
            let kb_ = kconserv.get(ki, kshift, kj) as usize;
            add2(
                &mut h_aaa,
                [ki, kj],
                &einsum(
                    "e,jibe->ijb",
                    &[&tmp_aaa, &imds.t2.0.slice_leading(&[kj, ki, kb_])?],
                )?,
                -0.5,
            )?;
            add2(
                &mut h_aaa,
                [ki, kj],
                &einsum(
                    "e,jibe->ijb",
                    &[&tmp_abb, &imds.t2.0.slice_leading(&[kj, ki, kb_])?],
                )?,
                -1.0,
            )?;
            add2(
                &mut h_abb,
                [ki, kj],
                &einsum(
                    "e,iJeB->iJB",
                    &[&tmp_aaa, &imds.t2.1.slice_leading(&[ki, kj, kshift])?],
                )?,
                -0.5,
            )?;
            add2(
                &mut h_abb,
                [ki, kj],
                &einsum(
                    "e,iJeB->iJB",
                    &[&tmp_abb, &imds.t2.1.slice_leading(&[ki, kj, kshift])?],
                )?,
                -1.0,
            )?;
            add2(
                &mut h_baa,
                [ki, kj],
                &einsum(
                    "E,jIbE->Ijb",
                    &[&tmp_bbb, &imds.t2.1.slice_leading(&[kj, ki, kb_])?],
                )?,
                -0.5,
            )?;
            add2(
                &mut h_baa,
                [ki, kj],
                &einsum(
                    "E,jIbE->Ijb",
                    &[&tmp_baa, &imds.t2.1.slice_leading(&[kj, ki, kb_])?],
                )?,
                -1.0,
            )?;
            add2(
                &mut h_bbb,
                [ki, kj],
                &einsum(
                    "E,JIBE->IJB",
                    &[&tmp_bbb, &imds.t2.2.slice_leading(&[kj, ki, kb_])?],
                )?,
                -0.5,
            )?;
            add2(
                &mut h_bbb,
                [ki, kj],
                &einsum(
                    "E,JIBE->IJB",
                    &[&tmp_baa, &imds.t2.2.slice_leading(&[kj, ki, kb_])?],
                )?,
                -1.0,
            )?;
        }
    }

    // `:247-295` — the `Wovvo` ring terms.
    for ki in 0..nk {
        for kj in 0..nk {
            let kb_ = kconserv.get(ki, kshift, kj) as usize;
            for km in 0..nk {
                let ke = kconserv.get(km, kshift, ki) as usize;
                add2(
                    &mut h_aaa,
                    [ki, kj],
                    &einsum(
                        "mebj,ime->ijb",
                        &[
                            &imds.wovvo.0.slice_leading(&[km, ke, kb_])?,
                            &r2aaa.slice_leading(&[ki, km])?,
                        ],
                    )?,
                    1.0,
                )?;
                add2(
                    &mut h_aaa,
                    [ki, kj],
                    &einsum(
                        "MEbj,iME->ijb",
                        &[
                            &imds.wovvo.2.slice_leading(&[km, ke, kb_])?,
                            &r2abb.slice_leading(&[ki, km])?,
                        ],
                    )?,
                    1.0,
                )?;
                // P(ij)
                let ke = kconserv.get(km, kshift, kj) as usize;
                add2(
                    &mut h_aaa,
                    [ki, kj],
                    &einsum(
                        "mebi,jme->ijb",
                        &[
                            &imds.wovvo.0.slice_leading(&[km, ke, kb_])?,
                            &r2aaa.slice_leading(&[kj, km])?,
                        ],
                    )?,
                    -1.0,
                )?;
                add2(
                    &mut h_aaa,
                    [ki, kj],
                    &einsum(
                        "MEbi,jME->ijb",
                        &[
                            &imds.wovvo.2.slice_leading(&[km, ke, kb_])?,
                            &r2abb.slice_leading(&[kj, km])?,
                        ],
                    )?,
                    -1.0,
                )?;

                let ke = kconserv.get(km, kshift, ki) as usize;
                add2(
                    &mut h_abb,
                    [ki, kj],
                    &einsum(
                        "meBJ,ime->iJB",
                        &[
                            &imds.wovvo.1.slice_leading(&[km, ke, kb_])?,
                            &r2aaa.slice_leading(&[ki, km])?,
                        ],
                    )?,
                    1.0,
                )?;
                add2(
                    &mut h_abb,
                    [ki, kj],
                    &einsum(
                        "MEBJ,iME->iJB",
                        &[
                            &imds.wovvo.3.slice_leading(&[km, ke, kb_])?,
                            &r2abb.slice_leading(&[ki, km])?,
                        ],
                    )?,
                    1.0,
                )?;
                add2(
                    &mut h_abb,
                    [ki, kj],
                    &einsum(
                        "miBE,mJE->iJB",
                        &[
                            &imds.woovv.1.slice_leading(&[km, ki, kb_])?,
                            &r2abb.slice_leading(&[km, kj])?,
                        ],
                    )?,
                    -1.0,
                )?;

                let ke = kconserv.get(km, kshift, ki) as usize;
                add2(
                    &mut h_baa,
                    [ki, kj],
                    &einsum(
                        "MEbj,IME->Ijb",
                        &[
                            &imds.wovvo.2.slice_leading(&[km, ke, kb_])?,
                            &r2bbb.slice_leading(&[ki, km])?,
                        ],
                    )?,
                    1.0,
                )?;
                add2(
                    &mut h_baa,
                    [ki, kj],
                    &einsum(
                        "mebj,Ime->Ijb",
                        &[
                            &imds.wovvo.0.slice_leading(&[km, ke, kb_])?,
                            &r2baa.slice_leading(&[ki, km])?,
                        ],
                    )?,
                    1.0,
                )?;
                add2(
                    &mut h_baa,
                    [ki, kj],
                    &einsum(
                        "MIbe,Mje->Ijb",
                        &[
                            &imds.woovv.2.slice_leading(&[km, ki, kb_])?,
                            &r2baa.slice_leading(&[km, kj])?,
                        ],
                    )?,
                    -1.0,
                )?;

                let ke = kconserv.get(km, kshift, ki) as usize;
                add2(
                    &mut h_bbb,
                    [ki, kj],
                    &einsum(
                        "MEBJ,IME->IJB",
                        &[
                            &imds.wovvo.3.slice_leading(&[km, ke, kb_])?,
                            &r2bbb.slice_leading(&[ki, km])?,
                        ],
                    )?,
                    1.0,
                )?;
                add2(
                    &mut h_bbb,
                    [ki, kj],
                    &einsum(
                        "meBJ,Ime->IJB",
                        &[
                            &imds.wovvo.1.slice_leading(&[km, ke, kb_])?,
                            &r2baa.slice_leading(&[ki, km])?,
                        ],
                    )?,
                    1.0,
                )?;
                // P(ij)
                let ke = kconserv.get(km, kshift, kj) as usize;
                add2(
                    &mut h_bbb,
                    [ki, kj],
                    &einsum(
                        "MEBI,JME->IJB",
                        &[
                            &imds.wovvo.3.slice_leading(&[km, ke, kb_])?,
                            &r2bbb.slice_leading(&[kj, km])?,
                        ],
                    )?,
                    -1.0,
                )?;
                add2(
                    &mut h_bbb,
                    [ki, kj],
                    &einsum(
                        "meBI,Jme->IJB",
                        &[
                            &imds.wovvo.1.slice_leading(&[km, ke, kb_])?,
                            &r2baa.slice_leading(&[kj, km])?,
                        ],
                    )?,
                    -1.0,
                )?;
            }
        }
    }

    amplitudes_to_vector_ip(&(hr1a, hr1b), &(h_aaa, h_baa, h_abb, h_bbb), nk)
}

fn add2(t: &mut ZArr, k: [usize; 2], v: &ZArr, s: f64) -> Result<(), PbcCcError> {
    let mut cur = t.slice_leading(&k)?;
    cur.zip_assign(v, s)?;
    t.set_leading(&k, &cur)
}

/// `ipccsd_diag` (`:309-389`), the `partition = None` branch.
///
/// The `'mp'` branch (`:325`) opens with `raise Exception("MP diag is not
/// tested")` followed by code upstream itself has commented out; it is not
/// ported, and that is not a deferral — upstream refuses it too.
///
/// # Errors
/// As [`ipccsd_matvec`].
#[allow(clippy::too_many_lines)]
pub fn ipccsd_diag(
    kshift: usize,
    imds: &UhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let nk = imds.eris.nkpts;
    let (oa, ob) = imds.eris.nocc;
    let (va, vb) = imds.eris.nvir;
    let woooo = imds.need(&imds.woooo, "Woooo")?;

    let mut hr1a = ZArr::zeros(&[oa]);
    let f = imds.foo.0.slice_leading(&[kshift])?;
    for i in 0..oa {
        let (re, im) = f.at(&[i, i])?;
        hr1a.data_mut().re[i] = -re;
        hr1a.data_mut().im[i] = -im;
    }
    let mut hr1b = ZArr::zeros(&[ob]);
    let f = imds.foo.1.slice_leading(&[kshift])?;
    for i in 0..ob {
        let (re, im) = f.at(&[i, i])?;
        hr1b.data_mut().re[i] = -re;
        hr1b.data_mut().im[i] = -im;
    }

    let mut h_aaa = ZArr::zeros(&[nk, nk, oa, oa, va]);
    let mut h_baa = ZArr::zeros(&[nk, nk, ob, oa, va]);
    let mut h_abb = ZArr::zeros(&[nk, nk, oa, ob, vb]);
    let mut h_bbb = ZArr::zeros(&[nk, nk, ob, ob, vb]);

    // `:348-362` — the Fock diagonals. `kj = kconserv[kshift, ki, ka]`, i.e.
    // the loop is over `(ka, ki)` and `kj` follows, NOT the other way round.
    for ka in 0..nk {
        for ki in 0..nk {
            let kj = kconserv.get(kshift, ki, ka) as usize;
            let fa = imds.fvv.0.slice_leading(&[ka])?;
            let fb = imds.fvv.1.slice_leading(&[ka])?;
            let oi_a = imds.foo.0.slice_leading(&[ki])?;
            let oj_a = imds.foo.0.slice_leading(&[kj])?;
            let oi_b = imds.foo.1.slice_leading(&[ki])?;
            let oj_b = imds.foo.1.slice_leading(&[kj])?;
            diag3(&mut h_aaa, [ki, kj], oa, oa, va, &oi_a, &oj_a, &fa)?;
            diag3(&mut h_abb, [ki, kj], oa, ob, vb, &oi_a, &oj_b, &fb)?;
            diag3(&mut h_baa, [ki, kj], ob, oa, va, &oi_b, &oj_a, &fa)?;
            diag3(&mut h_bbb, [ki, kj], ob, ob, vb, &oi_b, &oj_b, &fb)?;
        }
    }

    for ki in 0..nk {
        for kj in 0..nk {
            // `:365-368` — `einsum('iijj->ij', ...)`, a DOUBLE diagonal.
            oooo_diag(
                &mut h_aaa,
                [ki, kj],
                &woooo.0.slice_leading(&[ki, ki, kj])?,
                oa,
                oa,
                va,
                false,
            )?;
            oooo_diag(
                &mut h_abb,
                [ki, kj],
                &woooo.1.slice_leading(&[ki, ki, kj])?,
                oa,
                ob,
                vb,
                false,
            )?;
            oooo_diag(
                &mut h_bbb,
                [ki, kj],
                &woooo.2.slice_leading(&[ki, ki, kj])?,
                ob,
                ob,
                vb,
                false,
            )?;
            // `:368` — `WooOO[kj, kj, ki]` read as `jjII->Ij`, i.e. the two
            // occupied labels come back SWAPPED relative to the three above.
            oooo_diag(
                &mut h_baa,
                [ki, kj],
                &woooo.1.slice_leading(&[kj, kj, ki])?,
                ob,
                oa,
                va,
                true,
            )?;

            let kb_ = kconserv.get(ki, kshift, kj) as usize;
            add2(
                &mut h_aaa,
                [ki, kj],
                &einsum(
                    "iejb,jibe->ijb",
                    &[
                        &imds.wovov.slice_leading(&[ki, kshift, kj])?,
                        &imds.t2.0.slice_leading(&[kj, ki, kb_])?,
                    ],
                )?,
                -1.0,
            )?;
            add2(
                &mut h_abb,
                [ki, kj],
                &einsum(
                    "ieJB,iJeB->iJB",
                    &[
                        &imds.wov_ov.slice_leading(&[ki, kshift, kj])?,
                        &imds.t2.1.slice_leading(&[ki, kj, kshift])?,
                    ],
                )?,
                -1.0,
            )?;
            // `:372` — `WovOV[kj, kb, ki]`, at `kb` not `kshift`.
            add2(
                &mut h_baa,
                [ki, kj],
                &einsum(
                    "jbIE,jIbE->Ijb",
                    &[
                        &imds.wov_ov.slice_leading(&[kj, kb_, ki])?,
                        &imds.t2.1.slice_leading(&[kj, ki, kb_])?,
                    ],
                )?,
                -1.0,
            )?;
            add2(
                &mut h_bbb,
                [ki, kj],
                &einsum(
                    "IEJB,JIBE->IJB",
                    &[
                        &imds.wovov_b.slice_leading(&[ki, kshift, kj])?,
                        &imds.t2.2.slice_leading(&[kj, ki, kb_])?,
                    ],
                )?,
                -1.0,
            )?;

            // `:375-385` — the `ibbi` / `jbbj` diagonals, broadcast over the
            // OTHER occupied index.
            let wi = imds.wovvo.0.slice_leading(&[ki, kb_, kb_])?;
            let wj = imds.wovvo.0.slice_leading(&[kj, kb_, kb_])?;
            let wi_b = imds.wovvo.3.slice_leading(&[ki, kb_, kb_])?;
            let wj_b = imds.wovvo.3.slice_leading(&[kj, kb_, kb_])?;
            ovvo_diag(&mut h_aaa, [ki, kj], &wi, oa, oa, va, true, 1.0)?;
            ovvo_diag(&mut h_aaa, [ki, kj], &wj, oa, oa, va, false, 1.0)?;
            ovvo_diag(&mut h_baa, [ki, kj], &wj, ob, oa, va, false, 1.0)?;
            ovvo_diag(&mut h_abb, [ki, kj], &wj_b, oa, ob, vb, false, 1.0)?;
            ovvo_diag(&mut h_bbb, [ki, kj], &wi_b, ob, ob, vb, true, 1.0)?;
            ovvo_diag(&mut h_bbb, [ki, kj], &wj_b, ob, ob, vb, false, 1.0)?;
            // `:379` / `:382` — `WOOvv[ki,ki,kb]` as `IIbb->Ib` and
            // `WooVV[ki,ki,kb]` as `iiBB->iB`: a double diagonal on BOTH pairs.
            oovv_diag(
                &mut h_baa,
                [ki, kj],
                &imds.woovv.2.slice_leading(&[ki, ki, kb_])?,
                ob,
                oa,
                va,
                -1.0,
            )?;
            oovv_diag(
                &mut h_abb,
                [ki, kj],
                &imds.woovv.1.slice_leading(&[ki, ki, kb_])?,
                oa,
                ob,
                vb,
                -1.0,
            )?;
        }
    }

    amplitudes_to_vector_ip(&(hr1a, hr1b), &(h_aaa, h_baa, h_abb, h_bbb), nk)
}

/// `H[ki,kj][i,j,b] += fvv[b,b] - foo_i[i,i] - foo_j[j,j]`.
#[allow(clippy::too_many_arguments)]
fn diag3(
    h: &mut ZArr,
    k: [usize; 2],
    ni: usize,
    nj: usize,
    nb: usize,
    oi: &ZArr,
    oj: &ZArr,
    fv: &ZArr,
) -> Result<(), PbcCcError> {
    let mut blk = h.slice_leading(&k)?;
    for i in 0..ni {
        for j in 0..nj {
            for b_ in 0..nb {
                let f = (i * nj + j) * nb + b_;
                let (r, m) = fv.at(&[b_, b_])?;
                blk.data_mut().re[f] += r;
                blk.data_mut().im[f] += m;
                let (r, m) = oi.at(&[i, i])?;
                blk.data_mut().re[f] -= r;
                blk.data_mut().im[f] -= m;
                let (r, m) = oj.at(&[j, j])?;
                blk.data_mut().re[f] -= r;
                blk.data_mut().im[f] -= m;
            }
        }
    }
    h.set_leading(&k, &blk)
}

/// `H[ki,kj][i,j,b] += einsum('iijj->ij', W)`, or `('jjii->ij')` when `swap`.
#[allow(clippy::too_many_arguments)]
fn oooo_diag(
    h: &mut ZArr,
    k: [usize; 2],
    w: &ZArr,
    ni: usize,
    nj: usize,
    nb: usize,
    swap: bool,
) -> Result<(), PbcCcError> {
    let mut blk = h.slice_leading(&k)?;
    for i in 0..ni {
        for j in 0..nj {
            let (r, m) = if swap {
                w.at(&[j, j, i, i])?
            } else {
                w.at(&[i, i, j, j])?
            };
            for b_ in 0..nb {
                let f = (i * nj + j) * nb + b_;
                blk.data_mut().re[f] += r;
                blk.data_mut().im[f] += m;
            }
        }
    }
    h.set_leading(&k, &blk)
}

/// `H[ki,kj][i,j,b] += s * einsum('ibbi->ib', W)` (`first`) or `('jbbj->jb')`.
#[allow(clippy::too_many_arguments)]
fn ovvo_diag(
    h: &mut ZArr,
    k: [usize; 2],
    w: &ZArr,
    ni: usize,
    nj: usize,
    nb: usize,
    first: bool,
    s: f64,
) -> Result<(), PbcCcError> {
    let mut blk = h.slice_leading(&k)?;
    for i in 0..ni {
        for j in 0..nj {
            for b_ in 0..nb {
                let idx = if first { i } else { j };
                let (r, m) = w.at(&[idx, b_, b_, idx])?;
                let f = (i * nj + j) * nb + b_;
                blk.data_mut().re[f] += s * r;
                blk.data_mut().im[f] += s * m;
            }
        }
    }
    h.set_leading(&k, &blk)
}

/// `H[ki,kj][i,j,b] += s * einsum('iibb->ib', W)`.
#[allow(clippy::too_many_arguments)]
fn oovv_diag(
    h: &mut ZArr,
    k: [usize; 2],
    w: &ZArr,
    ni: usize,
    nj: usize,
    nb: usize,
    s: f64,
) -> Result<(), PbcCcError> {
    let mut blk = h.slice_leading(&k)?;
    for i in 0..ni {
        for b_ in 0..nb {
            let (r, m) = w.at(&[i, i, b_, b_])?;
            for j in 0..nj {
                let f = (i * nj + j) * nb + b_;
                blk.data_mut().re[f] += s * r;
                blk.data_mut().im[f] += s * m;
            }
        }
    }
    h.set_leading(&k, &blk)
}

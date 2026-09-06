//! `eom_kccsd_rhf` — equation-of-motion CCSD over SPIN-ADAPTED k-point orbitals
//! (plan 16-10; `pyscf/pbc/cc/eom_kccsd_rhf.py`, 1716 l).
//!
//! # This is not [`crate::eom_kccsd_ghf`] with `nocc` halved
//!
//! `eom_kccsd_rhf.py:25` imports the GHF module and its `EOMIP`/`EOMEA` inherit
//! from it, but only the DRIVER is shared: the matvecs, the diagonals, the
//! intermediates and the vector packing are all different. The spin-adapted
//! equations carry the `2·X − Xᵀ` combinations that a spin-orbital treatment
//! gets from antisymmetry, and they appear here as explicit `St2`/`SWooov`/
//! `SWovvo`/`SWoovv` terms — thirteen of them, each transcribed from the
//! upstream line above it.
//!
//! # The packing is FLAT, unlike the spin-orbital one
//!
//! `EOMIP.ip_vector_desc` (`:390-393`) is `[(nocc,), (nkpts, nkpts, nocc, nocc,
//! nvir)]` and `nested_to_vector` simply concatenates. There is no triangle and
//! no `kshift` dependence: `vector_size` is `nocc + nkpts²·nocc²·nvir`
//! (`:409-413`). The spin-orbital module's careful `tril` packing exists
//! because ITS `r2` is antisymmetric; this one's is not.
//!
//! # `partition` is not ported
//!
//! Both matvecs branch three ways on `eom.partition` (`'mp'`, `'full'`, `None`).
//! Nothing in this phase sets it, and `lipccsd_matvec`/`leaccsd_matvec` both
//! `assert eom.partition is None` outright (`:107`, `:510`) — so the `None`
//! branch is the only one the left matvecs admit at all. Porting the other two
//! would mean shipping arithmetic no test can reach.

use pyscf_pbc_lib::Kconserv;

use crate::error::PbcCcError;
use crate::keris::{Blk, KEris};
use crate::kintermediates_rhf as imd;
use crate::zarr::{ZArr, einsum, einsum_scaled};

/// The intermediates `eom_kccsd_rhf._IMDS` caches (`:1497-1716`).
pub struct RhfEomImds<'a> {
    pub eris: &'a KEris,
    pub t1: ZArr,
    pub t2: ZArr,
    /// `[nkpts, nocc, nocc]` — `Loo`, NOT `cc_Foo` (`:1524`).
    pub loo: ZArr,
    /// `[nkpts, nvir, nvir]`.
    pub lvv: ZArr,
    /// `[nkpts, nocc, nvir]`.
    pub fov: ZArr,
    /// Shared 2e.
    pub wovov: ZArr,
    pub wovvo: ZArr,
    /// IP.
    pub woooo: Option<ZArr>,
    pub wooov: Option<ZArr>,
    pub wovoo: Option<ZArr>,
    /// EA.
    pub wvovv: Option<ZArr>,
    pub wvvvv: Option<ZArr>,
    pub wvvvo: Option<ZArr>,
}

impl<'a> RhfEomImds<'a> {
    /// `_make_shared_1e` + `_make_shared_2e` (`:1520-1548`).
    ///
    /// # Errors
    /// Propagates every intermediate build.
    pub fn make_shared(
        t1: &ZArr,
        t2: &ZArr,
        eris: &'a KEris,
        kconserv: &Kconserv,
    ) -> Result<Self, PbcCcError> {
        Ok(Self {
            eris,
            t1: t1.clone(),
            t2: t2.clone(),
            loo: imd::loo(t1, t2, eris, kconserv)?,
            lvv: imd::lvv(t1, t2, eris, kconserv)?,
            fov: imd::cc_fov(t1, t2, eris)?,
            wovov: imd::wovov(t1, t2, eris, kconserv)?,
            wovvo: imd::wovvo(t1, t2, eris, kconserv)?,
            woooo: None,
            wooov: None,
            wovoo: None,
            wvovv: None,
            wvvvv: None,
            wvvvo: None,
        })
    }

    /// `_IMDS.make_ip` (`:1550-1577`).
    ///
    /// # Errors
    /// Propagates every intermediate build.
    pub fn make_ip(mut self, kconserv: &Kconserv) -> Result<Self, PbcCcError> {
        let (t1, t2) = (self.t1.clone(), self.t2.clone());
        if self.woooo.is_none() {
            self.woooo = Some(imd::eom_woooo(&t1, &t2, self.eris, kconserv)?);
        }
        self.wooov = Some(imd::wooov(&t1, self.eris)?);
        self.wovoo = Some(imd::wovoo(&t1, &t2, self.eris, kconserv)?);
        Ok(self)
    }

    /// `_IMDS.make_ea` (`:1595-1626`).
    ///
    /// # Errors
    /// Propagates every intermediate build.
    pub fn make_ea(mut self, kconserv: &Kconserv) -> Result<Self, PbcCcError> {
        let (t1, t2) = (self.t1.clone(), self.t2.clone());
        self.wvovv = Some(imd::wvovv(&t1, self.eris)?);
        let w4 = imd::eom_wvvvv(&t1, &t2, self.eris, kconserv)?;
        self.wvvvo = Some(imd::wvvvo(&t1, &t2, self.eris, kconserv, Some(&w4))?);
        self.wvvvv = Some(w4);
        Ok(self)
    }

    /// `_IMDS.get_Wvvvv(ka, kb, kc)` (`:1708-1716`) — the cached block when
    /// `Wvvvv` was built, otherwise rebuilt on the fly.
    ///
    /// # Errors
    /// Propagates the rebuild.
    pub fn get_wvvvv(
        &self,
        ka: usize,
        kb: usize,
        kc: usize,
        kconserv: &Kconserv,
    ) -> Result<ZArr, PbcCcError> {
        match &self.wvvvv {
            Some(w) => w.slice_leading(&[ka, kb, kc]),
            None => imd::get_wvvvv(&self.t1, &self.t2, self.eris, kconserv, ka, kb, kc),
        }
    }

    fn need<'w>(&self, w: &'w Option<ZArr>, what: &'static str) -> Result<&'w ZArr, PbcCcError> {
        w.as_ref()
            .ok_or_else(|| PbcCcError::Shape(format!("{what} was not built; call make_ip/make_ea")))
    }
}

/// `EOMIP.vector_size` (`:409-413`).
pub fn ip_vector_size(nkpts: usize, nocc: usize, nvir: usize) -> usize {
    nocc + nkpts * nkpts * nocc * nocc * nvir
}

/// `EOMEA.vector_size` (`:810-814`).
pub fn ea_vector_size(nkpts: usize, nocc: usize, nvir: usize) -> usize {
    nvir + nkpts * nkpts * nocc * nvir * nvir
}

/// `vector_to_nested(vec, ip_vector_desc)` (`:390-401`) — a flat split.
///
/// # Errors
/// [`PbcCcError::Shape`] on a length mismatch.
pub fn vector_to_amplitudes_ip(
    vector: &ZArr,
    nkpts: usize,
    nocc: usize,
    nvir: usize,
) -> Result<(ZArr, ZArr), PbcCcError> {
    split(vector, &[nocc], &[nkpts, nkpts, nocc, nocc, nvir])
}

/// `nested_to_vector((r1, r2))` for IP.
///
/// # Errors
/// [`PbcCcError::Shape`] on a shape mismatch.
pub fn amplitudes_to_vector_ip(r1: &ZArr, r2: &ZArr) -> Result<ZArr, PbcCcError> {
    join(r1, r2)
}

/// `vector_to_nested(vec, ea_vector_desc)` (`:790-801`).
///
/// # Errors
/// [`PbcCcError::Shape`] on a length mismatch.
pub fn vector_to_amplitudes_ea(
    vector: &ZArr,
    nkpts: usize,
    nocc: usize,
    nvir: usize,
) -> Result<(ZArr, ZArr), PbcCcError> {
    split(vector, &[nvir], &[nkpts, nkpts, nocc, nvir, nvir])
}

/// `nested_to_vector((r1, r2))` for EA.
///
/// # Errors
/// [`PbcCcError::Shape`] on a shape mismatch.
pub fn amplitudes_to_vector_ea(r1: &ZArr, r2: &ZArr) -> Result<ZArr, PbcCcError> {
    join(r1, r2)
}

fn split(v: &ZArr, s1: &[usize], s2: &[usize]) -> Result<(ZArr, ZArr), PbcCcError> {
    let n1: usize = s1.iter().product();
    let n2: usize = s2.iter().product();
    if v.len() != n1 + n2 {
        return Err(PbcCcError::Shape(format!(
            "EOM vector of {} elements, expected {}",
            v.len(),
            n1 + n2
        )));
    }
    let mut r1 = ZArr::zeros(s1);
    r1.data_mut().re.copy_from_slice(&v.data().re[..n1]);
    r1.data_mut().im.copy_from_slice(&v.data().im[..n1]);
    let mut r2 = ZArr::zeros(s2);
    r2.data_mut().re.copy_from_slice(&v.data().re[n1..]);
    r2.data_mut().im.copy_from_slice(&v.data().im[n1..]);
    Ok((r1, r2))
}

fn join(r1: &ZArr, r2: &ZArr) -> Result<ZArr, PbcCcError> {
    let mut v = ZArr::zeros(&[r1.len() + r2.len()]);
    v.data_mut().re[..r1.len()].copy_from_slice(&r1.data().re);
    v.data_mut().im[..r1.len()].copy_from_slice(&r1.data().im);
    v.data_mut().re[r1.len()..].copy_from_slice(&r2.data().re);
    v.data_mut().im[r1.len()..].copy_from_slice(&r2.data().im);
    Ok(v)
}

/// `ipccsd_matvec` (`:39-104`), the `partition = None` branch.
///
/// **The caller is responsible for the frozen mask.** Upstream applies
/// `mask_frozen(..., const=0.0)` to both the input vector and the result
/// (`:48`, `:104`); [`crate::eom_kccsd_ghf::eom_kernel`] does that around the
/// matvec, so doing it here as well would mask twice.
///
/// # Errors
/// Propagates every intermediate access and shape check.
pub fn ipccsd_matvec(
    vector: &ZArr,
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let (r1, r2) = vector_to_amplitudes_ip(vector, nkpts, nocc, nvir)?;
    let woooo = imds.need(&imds.woooo, "Woooo")?;
    let wooov = imds.need(&imds.wooov, "Wooov")?;
    let wovoo = imds.need(&imds.wovoo, "Wovoo")?;

    // `:52-61`
    let mut hr1 = einsum_scaled("ki,k->i", &[&imds.loo.slice_leading(&[kshift])?, &r1], -1.0)?;
    for kl in 0..nkpts {
        hr1.add_assign(&einsum_scaled(
            "ld,ild->i",
            &[
                &imds.fov.slice_leading(&[kl])?,
                &r2.slice_leading(&[kshift, kl])?,
            ],
            2.0,
        )?)?;
        hr1.sub_assign(&einsum(
            "ld,lid->i",
            &[
                &imds.fov.slice_leading(&[kl])?,
                &r2.slice_leading(&[kl, kshift])?,
            ],
        )?)?;
        for kk in 0..nkpts {
            hr1.add_assign(&einsum_scaled(
                "klid,kld->i",
                &[
                    &wooov.slice_leading(&[kk, kl, kshift])?,
                    &r2.slice_leading(&[kk, kl])?,
                ],
                -2.0,
            )?)?;
            hr1.add_assign(&einsum(
                "lkid,kld->i",
                &[
                    &wooov.slice_leading(&[kl, kk, kshift])?,
                    &r2.slice_leading(&[kk, kl])?,
                ],
            )?)?;
        }
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nocc, nvir]);
    // `:63-69`
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let kb = kconserv.get(ki, kshift, kj) as usize;
            let v = einsum_scaled(
                "kbij,k->ijb",
                &[&wovoo.slice_leading(&[kshift, kb, ki])?, &r1],
                -1.0,
            )?;
            hr2.set_leading(&[ki, kj], &v)?;
        }
    }
    // `:87-102`
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let kb = kconserv.get(ki, kshift, kj) as usize;
            let mut blk = hr2.slice_leading(&[ki, kj])?;
            blk.add_assign(&einsum(
                "bd,ijd->ijb",
                &[
                    &imds.lvv.slice_leading(&[kb])?,
                    &r2.slice_leading(&[ki, kj])?,
                ],
            )?)?;
            blk.sub_assign(&einsum(
                "li,ljb->ijb",
                &[
                    &imds.loo.slice_leading(&[ki])?,
                    &r2.slice_leading(&[ki, kj])?,
                ],
            )?)?;
            blk.sub_assign(&einsum(
                "lj,ilb->ijb",
                &[
                    &imds.loo.slice_leading(&[kj])?,
                    &r2.slice_leading(&[ki, kj])?,
                ],
            )?)?;
            for kl in 0..nkpts {
                let kk = kconserv.get(ki, kl, kj) as usize;
                blk.add_assign(&einsum(
                    "klij,klb->ijb",
                    &[
                        &woooo.slice_leading(&[kk, kl, ki])?,
                        &r2.slice_leading(&[kk, kl])?,
                    ],
                )?)?;
                let kd = kconserv.get(kl, kj, kb) as usize;
                blk.add_assign(&einsum_scaled(
                    "lbdj,ild->ijb",
                    &[
                        &imds.wovvo.slice_leading(&[kl, kb, kd])?,
                        &r2.slice_leading(&[ki, kl])?,
                    ],
                    2.0,
                )?)?;
                blk.sub_assign(&einsum(
                    "lbdj,lid->ijb",
                    &[
                        &imds.wovvo.slice_leading(&[kl, kb, kd])?,
                        &r2.slice_leading(&[kl, ki])?,
                    ],
                )?)?;
                // `:97` carries upstream's own `# typo in Ref` comment: the
                // published equation has a different index here and upstream's
                // code is the correct one.
                blk.sub_assign(&einsum(
                    "lbjd,ild->ijb",
                    &[
                        &imds.wovov.slice_leading(&[kl, kb, kj])?,
                        &r2.slice_leading(&[ki, kl])?,
                    ],
                )?)?;
                let _kd = kconserv.get(kl, ki, kb) as usize;
                blk.sub_assign(&einsum(
                    "lbid,ljd->ijb",
                    &[
                        &imds.wovov.slice_leading(&[kl, kb, ki])?,
                        &r2.slice_leading(&[kl, kj])?,
                    ],
                )?)?;
            }
            hr2.set_leading(&[ki, kj], &blk)?;
        }
    }
    // `:100-102` — the spin-adapted `2·W − Wᵀ` contraction into one `nvir`
    // vector, then broadcast back through `t2`.
    let mut tmp = ZArr::zeros(&[nvir]);
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            tmp.add_assign(&einsum_scaled(
                "klcd,kld->c",
                &[
                    &imds.eris.blk(Blk::Oovv, kx, ky, kshift)?,
                    &r2.slice_leading(&[kx, ky])?,
                ],
                2.0,
            )?)?;
            tmp.sub_assign(&einsum(
                "lkcd,kld->c",
                &[
                    &imds.eris.blk(Blk::Oovv, ky, kx, kshift)?,
                    &r2.slice_leading(&[kx, ky])?,
                ],
            )?)?;
        }
    }
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            let mut blk = hr2.slice_leading(&[kx, ky])?;
            blk.sub_assign(&einsum(
                "c,ijcb->ijb",
                &[&tmp, &imds.t2.slice_leading(&[kx, ky, kshift])?],
            )?)?;
            hr2.set_leading(&[kx, ky], &blk)?;
        }
    }

    amplitudes_to_vector_ip(&hr1, &hr2)
}

/// `lipccsd_matvec` (`:106-164`). `partition` must be `None` — upstream
/// asserts it (`:110`).
///
/// # Errors
/// As [`ipccsd_matvec`].
pub fn lipccsd_matvec(
    vector: &ZArr,
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let (r1, r2) = vector_to_amplitudes_ip(vector, nkpts, nocc, nvir)?;
    let woooo = imds.need(&imds.woooo, "Woooo")?;
    let wooov = imds.need(&imds.wooov, "Wooov")?;
    let wovoo = imds.need(&imds.wovoo, "Wovoo")?;

    let mut hr1 = einsum_scaled("ki,i->k", &[&imds.loo.slice_leading(&[kshift])?, &r1], -1.0)?;
    for ki in 0..nkpts {
        for kb in 0..nkpts {
            let kj = kconserv.get(kshift, ki, kb) as usize;
            hr1.sub_assign(&einsum(
                "kbij,ijb->k",
                &[
                    &wovoo.slice_leading(&[kshift, kb, ki])?,
                    &r2.slice_leading(&[ki, kj])?,
                ],
            )?)?;
        }
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nocc, nvir]);
    // `:124-133`
    for kl in 0..nkpts {
        for kk in 0..nkpts {
            let kd = kconserv.get(kk, kshift, kl) as usize;
            let mut sw = wooov.slice_leading(&[kk, kl, kshift])?;
            sw.scale(2.0);
            sw.sub_assign(
                &wooov
                    .slice_leading(&[kl, kk, kshift])?
                    .transpose(&[1, 0, 2, 3])?,
            )?;
            let mut blk = hr2.slice_leading(&[kk, kl])?;
            blk.sub_assign(&einsum("klid,i->kld", &[&sw, &r1])?)?;
            hr2.set_leading(&[kk, kl], &blk)?;

            if kk == kd {
                let v = einsum("kd,l->kld", &[&imds.fov.slice_leading(&[kk])?, &r1])?;
                let mut b = hr2.slice_leading(&[kk, kshift])?;
                b.sub_assign(&v)?;
                hr2.set_leading(&[kk, kshift], &b)?;
            }
            if kl == kd {
                let v = einsum_scaled("ld,k->kld", &[&imds.fov.slice_leading(&[kl])?, &r1], 2.0)?;
                let mut b = hr2.slice_leading(&[kshift, kl])?;
                b.add_assign(&v)?;
                hr2.set_leading(&[kshift, kl], &b)?;
            }
        }
    }

    // `:135-155`
    for kl in 0..nkpts {
        for kk in 0..nkpts {
            let kd = kconserv.get(kk, kshift, kl) as usize;
            let mut blk = hr2.slice_leading(&[kk, kl])?;
            blk.sub_assign(&einsum(
                "ki,ild->kld",
                &[
                    &imds.loo.slice_leading(&[kk])?,
                    &r2.slice_leading(&[kk, kl])?,
                ],
            )?)?;
            blk.sub_assign(&einsum(
                "lj,kjd->kld",
                &[
                    &imds.loo.slice_leading(&[kl])?,
                    &r2.slice_leading(&[kk, kl])?,
                ],
            )?)?;
            blk.add_assign(&einsum(
                "bd,klb->kld",
                &[
                    &imds.lvv.slice_leading(&[kd])?,
                    &r2.slice_leading(&[kk, kl])?,
                ],
            )?)?;
            for kj in 0..nkpts {
                let kb = kconserv.get(kd, kl, kj) as usize;
                let mut sw = imds.wovvo.slice_leading(&[kl, kb, kd])?;
                sw.scale(2.0);
                sw.sub_assign(
                    &imds
                        .wovov
                        .slice_leading(&[kl, kb, kj])?
                        .transpose(&[0, 1, 3, 2])?,
                )?;
                blk.add_assign(&einsum(
                    "lbdj,kjb->kld",
                    &[&sw, &r2.slice_leading(&[kk, kj])?],
                )?)?;

                let kb = kconserv.get(kd, kk, kj) as usize;
                blk.sub_assign(&einsum(
                    "kbdj,ljb->kld",
                    &[
                        &imds.wovvo.slice_leading(&[kk, kb, kd])?,
                        &r2.slice_leading(&[kl, kj])?,
                    ],
                )?)?;
                blk.sub_assign(&einsum(
                    "kbjd,jlb->kld",
                    &[
                        &imds.wovov.slice_leading(&[kk, kb, kj])?,
                        &r2.slice_leading(&[kj, kl])?,
                    ],
                )?)?;

                let ki = kconserv.get(kk, kj, kl) as usize;
                blk.add_assign(&einsum(
                    "klji,jid->kld",
                    &[
                        &woooo.slice_leading(&[kk, kl, kj])?,
                        &r2.slice_leading(&[kj, ki])?,
                    ],
                )?)?;
            }
            hr2.set_leading(&[kk, kl], &blk)?;
        }
    }

    // `:157-166`
    let mut tmp = ZArr::zeros(&[nvir]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            tmp.add_assign(&einsum(
                "ijcb,ijb->c",
                &[
                    &imds.t2.slice_leading(&[ki, kj, kshift])?,
                    &r2.slice_leading(&[ki, kj])?,
                ],
            )?)?;
        }
    }
    for kl in 0..nkpts {
        for kk in 0..nkpts {
            let kd = kconserv.get(kk, kshift, kl) as usize;
            let mut sw = imds.eris.blk(Blk::Oovv, kl, kk, kd)?;
            sw.scale(2.0);
            sw.sub_assign(
                &imds
                    .eris
                    .blk(Blk::Oovv, kk, kl, kd)?
                    .transpose(&[1, 0, 2, 3])?,
            )?;
            let mut blk = hr2.slice_leading(&[kk, kl])?;
            blk.sub_assign(&einsum("lkdc,c->kld", &[&sw, &tmp])?)?;
            hr2.set_leading(&[kk, kl], &blk)?;
        }
    }

    amplitudes_to_vector_ip(&hr1, &hr2)
}

/// `ipccsd_diag` (`:166-212`), the `partition = None` branch.
///
/// # Errors
/// As [`ipccsd_matvec`].
pub fn ipccsd_diag(
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let woooo = imds.need(&imds.woooo, "Woooo")?;

    let mut hr1 = ZArr::zeros(&[nocc]);
    let l = imds.loo.slice_leading(&[kshift])?;
    for i in 0..nocc {
        let (re, im) = l.at(&[i, i])?;
        hr1.data_mut().re[i] = -re;
        hr1.data_mut().im[i] = -im;
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nocc, nvir]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let kb = kconserv.get(ki, kshift, kj) as usize;
            let mut blk = ZArr::zeros(&[nocc, nocc, nvir]);
            let lb = imds.lvv.slice_leading(&[kb])?;
            let li = imds.loo.slice_leading(&[ki])?;
            let lj = imds.loo.slice_leading(&[kj])?;
            for i in 0..nocc {
                for j in 0..nocc {
                    for b in 0..nvir {
                        let f = (i * nocc + j) * nvir + b;
                        let (r, m) = lb.at(&[b, b])?;
                        blk.data_mut().re[f] += r;
                        blk.data_mut().im[f] += m;
                        let (r, m) = li.at(&[i, i])?;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                        let (r, m) = lj.at(&[j, j])?;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                    }
                }
            }
            if ki == kconserv.get(ki, kj, kj) as usize {
                let w = woooo.slice_leading(&[ki, kj, ki])?;
                for i in 0..nocc {
                    for j in 0..nocc {
                        let (r, m) = w.at(&[i, j, i, j])?;
                        for b in 0..nvir {
                            let f = (i * nocc + j) * nvir + b;
                            blk.data_mut().re[f] += r;
                            blk.data_mut().im[f] += m;
                        }
                    }
                }
            }
            // `:196` `-einsum('jbjb->jb', Wovov[kj,kb,kj])`, broadcast over i.
            let w = imds.wovov.slice_leading(&[kj, kb, kj])?;
            for j in 0..nocc {
                for b in 0..nvir {
                    let (r, m) = w.at(&[j, b, j, b])?;
                    for i in 0..nocc {
                        let f = (i * nocc + j) * nvir + b;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                    }
                }
            }
            // `:198-201` — `2 Wovvo`, then a MINUS on the `i == j` diagonal
            // when `ki == kj`. That last line is the one an index-free port
            // silently drops.
            let w = imds.wovvo.slice_leading(&[kj, kb, kb])?;
            for j in 0..nocc {
                for b in 0..nvir {
                    let (r, m) = w.at(&[j, b, b, j])?;
                    for i in 0..nocc {
                        let f = (i * nocc + j) * nvir + b;
                        blk.data_mut().re[f] += 2.0 * r;
                        blk.data_mut().im[f] += 2.0 * m;
                    }
                    if ki == kj {
                        let f = (j * nocc + j) * nvir + b;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                    }
                }
            }
            // `:203` `-einsum('ibib->ib', Wovov[ki,kb,ki])`, broadcast over j.
            let w = imds.wovov.slice_leading(&[ki, kb, ki])?;
            for i in 0..nocc {
                for b in 0..nvir {
                    let (r, m) = w.at(&[i, b, i, b])?;
                    for j in 0..nocc {
                        let f = (i * nocc + j) * nvir + b;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                    }
                }
            }
            // `:205-207`
            let kd = kconserv.get(kj, kshift, ki) as usize;
            blk.sub_assign(&einsum_scaled(
                "ijcb,jibc->ijb",
                &[
                    &imds.t2.slice_leading(&[ki, kj, kshift])?,
                    &imds.eris.blk(Blk::Oovv, kj, ki, kd)?,
                ],
                2.0,
            )?)?;
            blk.add_assign(&einsum(
                "ijcb,ijbc->ijb",
                &[
                    &imds.t2.slice_leading(&[ki, kj, kshift])?,
                    &imds.eris.blk(Blk::Oovv, ki, kj, kd)?,
                ],
            )?)?;
            hr2.set_leading(&[ki, kj], &blk)?;
        }
    }
    amplitudes_to_vector_ip(&hr1, &hr2)
}

/// `eaccsd_matvec` (`:430-505`), the `partition = None` branch.
///
/// # Errors
/// As [`ipccsd_matvec`].
#[allow(clippy::too_many_lines)]
pub fn eaccsd_matvec(
    vector: &ZArr,
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let (r1, r2) = vector_to_amplitudes_ea(vector, nkpts, nocc, nvir)?;
    let wvovv = imds.need(&imds.wvovv, "Wvovv")?;
    let wvvvo = imds.need(&imds.wvvvo, "Wvvvo")?;

    // `:442-451`
    let mut hr1 = einsum("ac,c->a", &[&imds.lvv.slice_leading(&[kshift])?, &r1])?;
    for kl in 0..nkpts {
        hr1.add_assign(&einsum_scaled(
            "ld,lad->a",
            &[
                &imds.fov.slice_leading(&[kl])?,
                &r2.slice_leading(&[kl, kshift])?,
            ],
            2.0,
        )?)?;
        hr1.sub_assign(&einsum(
            "ld,lda->a",
            &[
                &imds.fov.slice_leading(&[kl])?,
                &r2.slice_leading(&[kl, kl])?,
            ],
        )?)?;
        for kc in 0..nkpts {
            let kd = kconserv.get(kshift, kc, kl) as usize;
            hr1.add_assign(&einsum_scaled(
                "alcd,lcd->a",
                &[
                    &wvovv.slice_leading(&[kshift, kl, kc])?,
                    &r2.slice_leading(&[kl, kc])?,
                ],
                2.0,
            )?)?;
            hr1.sub_assign(&einsum(
                "aldc,lcd->a",
                &[
                    &wvovv.slice_leading(&[kshift, kl, kd])?,
                    &r2.slice_leading(&[kl, kc])?,
                ],
            )?)?;
        }
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nvir, nvir]);
    // `:455-460`
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let _kb = kconserv.get(kshift, ka, kj) as usize;
            let v = einsum(
                "abcj,c->jab",
                &[&wvvvo.slice_leading(&[ka, _kb, kshift])?, &r1],
            )?;
            hr2.set_leading(&[kj, ka], &v)?;
        }
    }
    // `:478-500`
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let kb = kconserv.get(kshift, ka, kj) as usize;
            let mut blk = hr2.slice_leading(&[kj, ka])?;
            blk.sub_assign(&einsum(
                "lj,lab->jab",
                &[
                    &imds.loo.slice_leading(&[kj])?,
                    &r2.slice_leading(&[kj, ka])?,
                ],
            )?)?;
            blk.add_assign(&einsum(
                "ac,jcb->jab",
                &[
                    &imds.lvv.slice_leading(&[ka])?,
                    &r2.slice_leading(&[kj, ka])?,
                ],
            )?)?;
            blk.add_assign(&einsum(
                "bd,jad->jab",
                &[
                    &imds.lvv.slice_leading(&[kb])?,
                    &r2.slice_leading(&[kj, ka])?,
                ],
            )?)?;
            for kd in 0..nkpts {
                let kc = kconserv.get(ka, kd, kb) as usize;
                let w4 = imds.get_wvvvv(ka, kb, kc, kconserv)?;
                blk.add_assign(&einsum(
                    "abcd,jcd->jab",
                    &[&w4, &r2.slice_leading(&[kj, kc])?],
                )?)?;
                let kl = kconserv.get(kd, kb, kj) as usize;
                blk.add_assign(&einsum_scaled(
                    "lbdj,lad->jab",
                    &[
                        &imds.wovvo.slice_leading(&[kl, kb, kd])?,
                        &r2.slice_leading(&[kl, ka])?,
                    ],
                    2.0,
                )?)?;
                // `:492` — `Wvovo[kb,kl,kd,kj]` IS `Wovov[kl,kb,kj,kd]`
                // transposed `(1,0,3,2)`; upstream's comment says so.
                blk.sub_assign(&einsum(
                    "bldj,lad->jab",
                    &[
                        &imds
                            .wovov
                            .slice_leading(&[kl, kb, kj])?
                            .transpose(&[1, 0, 3, 2])?,
                        &r2.slice_leading(&[kl, ka])?,
                    ],
                )?)?;
                blk.sub_assign(&einsum(
                    "bljd,lda->jab",
                    &[
                        &imds
                            .wovvo
                            .slice_leading(&[kl, kb, kd])?
                            .transpose(&[1, 0, 3, 2])?,
                        &r2.slice_leading(&[kl, kd])?,
                    ],
                )?)?;
                let kl = kconserv.get(kd, ka, kj) as usize;
                blk.sub_assign(&einsum(
                    "aldj,ldb->jab",
                    &[
                        &imds
                            .wovov
                            .slice_leading(&[kl, ka, kj])?
                            .transpose(&[1, 0, 3, 2])?,
                        &r2.slice_leading(&[kl, kd])?,
                    ],
                )?)?;
            }
            hr2.set_leading(&[kj, ka], &blk)?;
        }
    }
    // `:501-503`
    let mut tmp = ZArr::zeros(&[nocc]);
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            tmp.add_assign(&einsum_scaled(
                "klcd,lcd->k",
                &[
                    &imds.eris.blk(Blk::Oovv, kshift, kx, ky)?,
                    &r2.slice_leading(&[kx, ky])?,
                ],
                2.0,
            )?)?;
            tmp.sub_assign(&einsum(
                "lkcd,lcd->k",
                &[
                    &imds.eris.blk(Blk::Oovv, kx, kshift, ky)?,
                    &r2.slice_leading(&[kx, ky])?,
                ],
            )?)?;
        }
    }
    for kx in 0..nkpts {
        for ky in 0..nkpts {
            let mut blk = hr2.slice_leading(&[kx, ky])?;
            blk.sub_assign(&einsum(
                "k,kjab->jab",
                &[&tmp, &imds.t2.slice_leading(&[kshift, kx, ky])?],
            )?)?;
            hr2.set_leading(&[kx, ky], &blk)?;
        }
    }

    amplitudes_to_vector_ea(&hr1, &hr2)
}

/// `leaccsd_matvec` (`:507-570`).
///
/// # Errors
/// As [`ipccsd_matvec`].
#[allow(clippy::too_many_lines)]
pub fn leaccsd_matvec(
    vector: &ZArr,
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let (r1, r2) = vector_to_amplitudes_ea(vector, nkpts, nocc, nvir)?;
    let wvovv = imds.need(&imds.wvovv, "Wvovv")?;
    let wvvvo = imds.need(&imds.wvvvo, "Wvvvo")?;

    // `:520-525`
    let mut hr1 = einsum("ac,a->c", &[&imds.lvv.slice_leading(&[kshift])?, &r1])?;
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let kb = kconserv.get(kj, ka, kshift) as usize;
            hr1.add_assign(&einsum(
                "abcj,jab->c",
                &[
                    &wvvvo.slice_leading(&[ka, kb, kshift])?,
                    &r2.slice_leading(&[kj, ka])?,
                ],
            )?)?;
        }
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nvir, nvir]);
    // `:527-536`
    for kl in 0..nkpts {
        for kc in 0..nkpts {
            let kd = kconserv.get(kl, kc, kshift) as usize;
            let mut blk = hr2.slice_leading(&[kl, kc])?;
            if kl == kd {
                blk.add_assign(&einsum_scaled(
                    "c,ld->lcd",
                    &[&r1, &imds.fov.slice_leading(&[kd])?],
                    2.0,
                )?)?;
            }
            if kl == kc {
                blk.sub_assign(&einsum(
                    "d,lc->lcd",
                    &[&r1, &imds.fov.slice_leading(&[kl])?],
                )?)?;
            }
            let mut sw = wvovv.slice_leading(&[kshift, kl, kc])?;
            sw.scale(2.0);
            sw.sub_assign(
                &wvovv
                    .slice_leading(&[kshift, kl, kd])?
                    .transpose(&[0, 1, 3, 2])?,
            )?;
            blk.add_assign(&einsum("a,alcd->lcd", &[&r1, &sw])?)?;
            hr2.set_leading(&[kl, kc], &blk)?;
        }
    }

    // `:538-556`
    for kl in 0..nkpts {
        for kc in 0..nkpts {
            let kd = kconserv.get(kl, kc, kshift) as usize;
            let mut blk = hr2.slice_leading(&[kl, kc])?;
            blk.add_assign(&einsum(
                "lad,ac->lcd",
                &[
                    &r2.slice_leading(&[kl, kc])?,
                    &imds.lvv.slice_leading(&[kc])?,
                ],
            )?)?;
            blk.add_assign(&einsum(
                "lcb,bd->lcd",
                &[
                    &r2.slice_leading(&[kl, kc])?,
                    &imds.lvv.slice_leading(&[kd])?,
                ],
            )?)?;
            blk.sub_assign(&einsum(
                "jcd,lj->lcd",
                &[
                    &r2.slice_leading(&[kl, kc])?,
                    &imds.loo.slice_leading(&[kl])?,
                ],
            )?)?;
            for kb in 0..nkpts {
                let kj = kconserv.get(kl, kd, kb) as usize;
                let mut sw = imds.wovvo.slice_leading(&[kl, kb, kd])?;
                sw.scale(2.0);
                sw.sub_assign(
                    &imds
                        .wovov
                        .slice_leading(&[kl, kb, kj])?
                        .transpose(&[0, 1, 3, 2])?,
                )?;
                blk.add_assign(&einsum(
                    "jcb,lbdj->lcd",
                    &[&r2.slice_leading(&[kj, kc])?, &sw],
                )?)?;

                let kj = kconserv.get(kl, kc, kb) as usize;
                blk.sub_assign(&einsum(
                    "lbjc,jbd->lcd",
                    &[
                        &imds.wovov.slice_leading(&[kl, kb, kj])?,
                        &r2.slice_leading(&[kj, kb])?,
                    ],
                )?)?;
                blk.sub_assign(&einsum(
                    "lbcj,jdb->lcd",
                    &[
                        &imds.wovvo.slice_leading(&[kl, kb, kc])?,
                        &r2.slice_leading(&[kj, kd])?,
                    ],
                )?)?;

                let ka = kconserv.get(kc, kb, kd) as usize;
                let w4 = imds.get_wvvvv(ka, kb, kc, kconserv)?;
                blk.add_assign(&einsum(
                    "lab,abcd->lcd",
                    &[&r2.slice_leading(&[kl, ka])?, &w4],
                )?)?;
            }
            hr2.set_leading(&[kl, kc], &blk)?;
        }
    }

    // `:558-568`
    let mut tmp = ZArr::zeros(&[nocc]);
    for ki in 0..nkpts {
        for kc in 0..nkpts {
            let kb = kconserv.get(ki, kc, kshift) as usize;
            tmp.add_assign(&einsum(
                "ijcb,ibc->j",
                &[
                    &imds.t2.slice_leading(&[ki, kshift, kc])?,
                    &r2.slice_leading(&[ki, kb])?,
                ],
            )?)?;
        }
    }
    for kl in 0..nkpts {
        for kc in 0..nkpts {
            let kd = kconserv.get(kl, kc, kshift) as usize;
            let mut sw = imds.eris.blk(Blk::Oovv, kl, kshift, kd)?;
            sw.scale(2.0);
            sw.sub_assign(
                &imds
                    .eris
                    .blk(Blk::Oovv, kl, kshift, kc)?
                    .transpose(&[0, 1, 3, 2])?,
            )?;
            let mut blk = hr2.slice_leading(&[kl, kc])?;
            blk.sub_assign(&einsum("ljdc,j->lcd", &[&sw, &tmp])?)?;
            hr2.set_leading(&[kl, kc], &blk)?;
        }
    }

    amplitudes_to_vector_ea(&hr1, &hr2)
}

/// `eaccsd_diag` (`:572-615`), the `partition = None` branch.
///
/// # Errors
/// As [`ipccsd_matvec`].
pub fn eaccsd_diag(
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);

    let mut hr1 = ZArr::zeros(&[nvir]);
    let l = imds.lvv.slice_leading(&[kshift])?;
    for a in 0..nvir {
        let (re, im) = l.at(&[a, a])?;
        hr1.data_mut().re[a] = re;
        hr1.data_mut().im[a] = im;
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nocc, nvir, nvir]);
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let kb = kconserv.get(kshift, ka, kj) as usize;
            let mut blk = ZArr::zeros(&[nocc, nvir, nvir]);
            let lj = imds.loo.slice_leading(&[kj])?;
            let la = imds.lvv.slice_leading(&[ka])?;
            let lb = imds.lvv.slice_leading(&[kb])?;
            let w4 = imds.get_wvvvv(ka, kb, ka, kconserv)?;
            let wjbjb = imds.wovov.slice_leading(&[kj, kb, kj])?;
            let wjbbj = imds.wovvo.slice_leading(&[kj, kb, kb])?;
            let wjaja = imds.wovov.slice_leading(&[kj, ka, kj])?;
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let f = (j * nvir + a) * nvir + b;
                        let mut acc = (0.0_f64, 0.0_f64);
                        let mut add = |v: (f64, f64), s: f64| {
                            acc.0 += s * v.0;
                            acc.1 += s * v.1;
                        };
                        add(lj.at(&[j, j])?, -1.0);
                        add(la.at(&[a, a])?, 1.0);
                        add(lb.at(&[b, b])?, 1.0);
                        // `:598` `einsum('abab->ab', Wvvvv)`
                        add(w4.at(&[a, b, a, b])?, 1.0);
                        // `:600` `-einsum('jbjb->jb', Wovov[kj,kb,kj])`
                        add(wjbjb.at(&[j, b, j, b])?, -1.0);
                        // `:601-602` `+2 einsum('jbbj->jb', Wovvo[kj,kb,kb])`
                        add(wjbbj.at(&[j, b, b, j])?, 2.0);
                        // `:605` `-einsum('jaja->ja', Wovov[kj,ka,kj])`
                        add(wjaja.at(&[j, a, j, a])?, -1.0);
                        blk.data_mut().re[f] += acc.0;
                        blk.data_mut().im[f] += acc.1;
                    }
                }
            }
            // `:603-604` — a MINUS on the `a == b` diagonal, but only when
            // `ka == kb`. Dropping the guard is silent on a fixture where the
            // two happen to coincide.
            if ka == kb {
                for j in 0..nocc {
                    for a in 0..nvir {
                        let (r, m) = wjbbj.at(&[j, a, a, j])?;
                        let f = (j * nvir + a) * nvir + a;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                    }
                }
            }
            // `:607-608`
            blk.sub_assign(&einsum_scaled(
                "ijab,ijab->jab",
                &[
                    &imds.t2.slice_leading(&[kshift, kj, ka])?,
                    &imds.eris.blk(Blk::Oovv, kshift, kj, ka)?,
                ],
                2.0,
            )?)?;
            blk.add_assign(&einsum(
                "ijab,ijba->jab",
                &[
                    &imds.t2.slice_leading(&[kshift, kj, ka])?,
                    &imds.eris.blk(Blk::Oovv, kshift, kj, kb)?,
                ],
            )?)?;
            hr2.set_leading(&[kj, ka], &blk)?;
        }
    }
    amplitudes_to_vector_ea(&hr1, &hr2)
}

/// `mask_frozen_ip` (`eom_kccsd_ghf.py:663-682`) with the RHF packing.
///
/// The MASKING is identical to the spin-orbital version — same `r2` shape,
/// same `kb = kconserv[ki, kshift, kj]`, same "replace every padded index with
/// `const`". Only the vector layout differs, which is why `EOMIP` can inherit
/// `mask_frozen` from the GHF module in Python (`:383`) and cannot here: this
/// port's packing functions are free functions, not methods on a shared base.
///
/// # Errors
/// As [`ipccsd_matvec`].
#[allow(clippy::too_many_arguments)]
pub fn mask_frozen_ip(
    vector: &ZArr,
    kshift: usize,
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    nonzero_opadding: &[Vec<usize>],
    nonzero_vpadding: &[Vec<usize>],
    kconserv: &Kconserv,
    konst: f64,
) -> Result<ZArr, PbcCcError> {
    let (r1, r2) = vector_to_amplitudes_ip(vector, nkpts, nocc, nvir)?;
    let mut new_r1 = filled(&[nocc], konst);
    for &i in &nonzero_opadding[kshift] {
        if i < nocc {
            new_r1.data_mut().re[i] = r1.data().re[i];
            new_r1.data_mut().im[i] = r1.data().im[i];
        }
    }
    let mut new_r2 = filled(&[nkpts, nkpts, nocc, nocc, nvir], konst);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let kb = kconserv.get(ki, kshift, kj) as usize;
            for &i in &nonzero_opadding[ki] {
                for &j in &nonzero_opadding[kj] {
                    for &b in &nonzero_vpadding[kb] {
                        if i >= nocc || j >= nocc || b >= nvir {
                            continue;
                        }
                        let f = (((ki * nkpts + kj) * nocc + i) * nocc + j) * nvir + b;
                        new_r2.data_mut().re[f] = r2.data().re[f];
                        new_r2.data_mut().im[f] = r2.data().im[f];
                    }
                }
            }
        }
    }
    amplitudes_to_vector_ip(&new_r1, &new_r2)
}

/// `mask_frozen_ea` (`eom_kccsd_ghf.py:1180-1199`) with the RHF packing.
///
/// # Errors
/// As [`ipccsd_matvec`].
#[allow(clippy::too_many_arguments)]
pub fn mask_frozen_ea(
    vector: &ZArr,
    kshift: usize,
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    nonzero_opadding: &[Vec<usize>],
    nonzero_vpadding: &[Vec<usize>],
    kconserv: &Kconserv,
    konst: f64,
) -> Result<ZArr, PbcCcError> {
    let (r1, r2) = vector_to_amplitudes_ea(vector, nkpts, nocc, nvir)?;
    let mut new_r1 = filled(&[nvir], konst);
    for &a in &nonzero_vpadding[kshift] {
        if a < nvir {
            new_r1.data_mut().re[a] = r1.data().re[a];
            new_r1.data_mut().im[a] = r1.data().im[a];
        }
    }
    let mut new_r2 = filled(&[nkpts, nkpts, nocc, nvir, nvir], konst);
    for kj in 0..nkpts {
        for ka in 0..nkpts {
            let kb = kconserv.get(kshift, ka, kj) as usize;
            for &j in &nonzero_opadding[kj] {
                for &a in &nonzero_vpadding[ka] {
                    for &b in &nonzero_vpadding[kb] {
                        if j >= nocc || a >= nvir || b >= nvir {
                            continue;
                        }
                        let f = (((kj * nkpts + ka) * nocc + j) * nvir + a) * nvir + b;
                        new_r2.data_mut().re[f] = r2.data().re[f];
                        new_r2.data_mut().im[f] = r2.data().im[f];
                    }
                }
            }
        }
    }
    amplitudes_to_vector_ea(&new_r1, &new_r2)
}

fn filled(shape: &[usize], v: f64) -> ZArr {
    let mut a = ZArr::zeros(shape);
    for x in a.data_mut().re.iter_mut() {
        *x = v;
    }
    a
}

/// `kernel(eom, ...)` for the SPIN-ADAPTED IP and EA (`eom_kccsd_ghf.py:40-159`
/// driving `eom_kccsd_rhf`'s matvecs).
///
/// Structurally identical to [`crate::eom_kccsd_ghf::eom_kernel`] — same guess,
/// same `LARGE_DENOM` mask, same preconditioner, same quasiparticle weight —
/// with this module's packings and matvecs substituted, which is exactly what
/// Python's inheritance does.
///
/// **Upstream masks inside the matvec** (`:48`, `:104`), not only around it.
/// That is reproduced here rather than hoisted: masking the input, computing,
/// then masking the output is not the same operation as masking once.
///
/// # Errors
/// Propagates the matvec and the Davidson solve.
pub fn eom_kernel(
    kind: crate::eom_kccsd_ghf::Excitation,
    kshift: usize,
    imds: &RhfEomImds<'_>,
    padding: &crate::eom_kccsd_ghf::Padding,
    kconserv: &Kconserv,
    opts: &crate::eom_kccsd_ghf::EomOpts,
) -> Result<crate::eom_kccsd_ghf::EomRoots, PbcCcError> {
    use crate::eom_kccsd_ghf::Excitation;
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    if kind == Excitation::Ee {
        // `EOMEESinglet` (`:1425`) is a different vector shape and a different
        // matvec; it is not ported. `EOMEETriplet` (`:1483`) and
        // `EOMEESpinFlip` (`:1489`) are SHELLS upstream — see `16-CONTEXT §1.5`.
        return Err(PbcCcError::NotImplementedUpstream {
            upstream: "pbc/cc/eom_kccsd_rhf.py:1425",
            what: "EOMEESinglet is not ported; EOMEETriplet and EOMEESpinFlip are shells upstream",
        });
    }
    let size = match kind {
        Excitation::Ip => ip_vector_size(nkpts, nocc, nvir),
        Excitation::Ea => ea_vector_size(nkpts, nocc, nvir),
        Excitation::Ee => unreachable!(),
    };

    let mask = |v: &ZArr, konst: f64| -> Result<ZArr, PbcCcError> {
        match kind {
            Excitation::Ip => mask_frozen_ip(
                v,
                kshift,
                nkpts,
                nocc,
                nvir,
                &padding.occupied,
                &padding.virtuals,
                kconserv,
                konst,
            ),
            Excitation::Ea => mask_frozen_ea(
                v,
                kshift,
                nkpts,
                nocc,
                nvir,
                &padding.occupied,
                &padding.virtuals,
                kconserv,
                konst,
            ),
            Excitation::Ee => unreachable!(),
        }
    };

    let ones = mask(&ZArr::zeros(&[size]), 1.0)?;
    let nfrozen = ones.data().re.iter().filter(|v| **v == 1.0).count();
    let nroots = opts.nroots.min(size).min(size - nfrozen).max(1);

    let diag = match kind {
        Excitation::Ip => ipccsd_diag(kshift, imds, kconserv)?,
        Excitation::Ea => eaccsd_diag(kshift, imds, kconserv)?,
        Excitation::Ee => unreachable!(),
    };
    let diag = mask(&diag, crate::kccsd_rhf::LARGE_DENOM)?;

    let mut guess = Vec::with_capacity(nroots);
    if opts.koopmans {
        let seeds: Vec<usize> = match kind {
            Excitation::Ip => padding.occupied[kshift].iter().rev().copied().collect(),
            Excitation::Ea => padding.virtuals[kshift].to_vec(),
            Excitation::Ee => unreachable!(),
        };
        for &n in seeds.iter().take(nroots) {
            let mut g = ZArr::zeros(&[size]);
            if n < size {
                g.data_mut().re[n] = 1.0;
            }
            guess.push(mask(&g, 0.0)?);
        }
    } else {
        let mut idx: Vec<usize> = (0..size).collect();
        idx.sort_by(|a, b| {
            diag.data().re[*a]
                .partial_cmp(&diag.data().re[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for &i in idx.iter().take(nroots) {
            let mut g = ZArr::zeros(&[size]);
            g.data_mut().re[i] = 1.0;
            guess.push(mask(&g, 0.0)?);
        }
    }

    let aop = |xs: &[pyscf_algebra::CTensor]| -> Vec<pyscf_algebra::CTensor> {
        xs.iter()
            .map(|x| {
                let v = ZArr::from_ctensor(&[size], x.clone()).expect("guess shape");
                let v = mask(&v, 0.0).expect("mask in");
                let out = match (kind, opts.left) {
                    (Excitation::Ip, false) => ipccsd_matvec(&v, kshift, imds, kconserv),
                    (Excitation::Ip, true) => lipccsd_matvec(&v, kshift, imds, kconserv),
                    (Excitation::Ea, false) => eaccsd_matvec(&v, kshift, imds, kconserv),
                    (Excitation::Ea, true) => leaccsd_matvec(&v, kshift, imds, kconserv),
                    (Excitation::Ee, _) => unreachable!(),
                }
                .expect("EOM matvec");
                mask(&out, 0.0).expect("mask out").into_ctensor()
            })
            .collect()
    };

    let dre: Vec<f64> = diag.data().re.clone();
    let precond = |r: &pyscf_algebra::CTensor, e0: f64, _x0: &pyscf_algebra::CTensor| {
        let mut out = r.clone();
        for i in 0..out.re.len() {
            let d = e0 - dre[i] + 1e-12;
            out.re[i] /= d;
            out.im[i] /= d;
        }
        out
    };

    let dopts = pyscf_algebra::DavidsonOptions {
        tol: opts.conv_tol,
        tol_residual: None,
        max_cycle: opts.max_cycle,
        max_space: opts.max_space,
        nroots,
        left: false,
        real_dtype: false,
        ..Default::default()
    };
    let res = pyscf_algebra::davidson_nosym1(
        aop,
        guess.iter().map(|g| g.data().clone()).collect(),
        precond,
        &dopts,
        pyscf_algebra::pick_real_eigs,
    )
    .map_err(|e| PbcCcError::Algebra(format!("EOM Davidson: {e}")))?;

    let n1 = match kind {
        Excitation::Ip => nocc,
        Excitation::Ea => nvir,
        Excitation::Ee => unreachable!(),
    };
    let mut v = Vec::with_capacity(res.x.len());
    let mut qp_weight = Vec::with_capacity(res.x.len());
    for x in &res.x {
        let vec = ZArr::from_ctensor(&[size], x.clone())?;
        let w: f64 = (0..n1)
            .map(|i| vec.data().re[i] * vec.data().re[i] + vec.data().im[i] * vec.data().im[i])
            .sum();
        qp_weight.push(w);
        v.push(vec);
    }
    Ok(crate::eom_kccsd_ghf::EomRoots {
        kshift,
        conv: res.conv,
        e: res.e,
        v,
        qp_weight,
    })
}

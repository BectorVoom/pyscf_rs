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
//! # `partition`
//!
//! Both matvecs and both diagonals branch three ways on `eom.partition`
//! (`'mp'`, `'full'`, `None`), and this module ports the two that compute
//! something:
//!
//! * `'mp'` — [`ipccsd_matvec_partition`], [`eaccsd_matvec_partition`],
//!   [`ipccsd_diag_partition`], [`eaccsd_diag_partition`], plus the
//!   intermediate skips at [`RhfEomImds::make_ip_partition`] /
//!   [`RhfEomImds::make_ea_partition`].
//! * `'full'` — REFUSED in the matvecs. `:471-474` reads `if diag is not
//!   None: diag = eom.get_diag(imds=imds)` — note the inverted guard — and
//!   then `eom.vector_to_amplitudes(diag, nmo, nocc)`, which here takes at
//!   most `(self, vector, kshift)`, so upstream raises `TypeError:
//!   EOMIP.vector_to_amplitudes() takes from 2 to 3 positional arguments but 4
//!   were given` before any arithmetic. Measured. The DIAGONALS have no
//!   `'full'` branch at all, so there `'full'` is the `None` diagonal and
//!   [`ipccsd_diag_partition`] reproduces that rather than refusing.
//! * `None` — the default, and the only branch the LEFT matvecs admit:
//!   `lipccsd_matvec`/`leaccsd_matvec` both `assert eom.partition is None`
//!   (`:109`, `:510`), as do both `*_star_contract` (`:216`, `:619`).
//!
//! **The driver refuses both non-`None` values anyway.** `EOMIP`/`EOMEA` here
//! inherit `ipccsd`/`eaccsd` from [`crate::eom_kccsd_ghf`] (`:378`, `:777`),
//! and those raise `NotImplementedError` at `eom_kccsd_ghf.py:618`/`:905`. So
//! [`eom_kernel`] refuses, exactly as upstream's does, and the branches are
//! reachable only through the `*_partition` entry points — which is the same
//! access upstream gives them (set `eom.partition` and call the matvec).

//! # EE is the SINGLET, and only the singlet
//!
//! `eeccsd` (`:831`) and `eeccsd_matvec` (`:965`) are both bare
//! `raise NotImplementedError`; `EOMEETriplet` (`:1483`) and `EOMEESpinFlip`
//! (`:1489`) declare nothing but a `vector_size` that returns `None`. The one
//! EE surface upstream has here is `EOMEESinglet` (`:1425`), driven by
//! `eomee_ccsd_singlet` (`:838`), and that is what [`eom_kernel`] runs for
//! [`Excitation::Ee`].
//!
//! Three things about it are unlike the IP/EA sides of this module:
//!
//! * **the vector size depends on `kshift` and is not a closed form**
//!   ([`ee_singlet_vector_size`]) — `r2` is packed over the composite index
//!   `(i k_i a k_a)` and a `(ki,ka,kj)` triple contributes a triangle, a full
//!   block, or nothing, according to how `ki·nkpts+ka` compares to
//!   `kj·nkpts+kb`;
//! * **the initial guess is a CIS diagonalisation** ([`ee_singlet_cis_guess`]),
//!   not the diagonal's `argsort`: `EOMEESinglet.get_init_guess` is
//!   `get_init_guess_cis` (`:1429`), which materialises the singles block of
//!   `Hbar` column by column and solves it densely;
//! * **there is no left EE.** `EOMEESinglet.gen_matvec` (`:1464`) raises for
//!   `left=True`, so there is no left matvec and hence no EE-CCSD\*.
//!
//! `_IMDS.make_ee` (`:1666-1706`) RENAMES the IP and EA sets rather than
//! building new ones; [`RhfEomImds::make_ee`] lists the eleven renames.

use pyscf_pbc_lib::{KIdx, Kconserv, get_kconserv3};

use crate::eom_kccsd_ghf::{EomOpts, Excitation, KLattice, Padding, Partition, StarPair, StarRoot};
use crate::error::PbcCcError;
use crate::keris::{Blk, KEris};
use crate::kintermediates_rhf as imd;
use crate::zarr::{ZArr, einsum, einsum_scaled};

/// `imd.get_t3p2_imds(cc, t1, t2, eris)` as the three `make_t3p2_*` call it.
///
/// **This port calls the LOOP-EXPLICIT `get_t3p2_imds_slow`**, not the blocked
/// `get_t3p2_imds` upstream's `_IMDS` reaches for. Upstream's own two
/// implementations do not agree to machine precision — see
/// [`crate::kintermediates_rhf::get_t3p2_imds_slow`] for the measured gap and
/// why the loop-explicit one is the porting target.
fn t3p2(
    t1: &ZArr,
    t2: &ZArr,
    eris: &KEris,
    kconserv: &Kconserv,
    padding: &Padding,
    lat: &KLattice<'_>,
) -> Result<imd::T3p2Imds, PbcCcError> {
    imd::get_t3p2_imds_slow(t1, t2, eris, kconserv, padding, lat)
}

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
    /// Shared 2e (`_make_shared_2e`, `:1531-1548`). `None` when the caller
    /// asked for the `'mp'` partition, which skips the whole block
    /// (`:1552-1554`, `:1597-1599`) because no `'mp'` branch reads it.
    pub wovov: Option<ZArr>,
    pub wovvo: Option<ZArr>,
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
            wovov: Some(imd::wovov(t1, t2, eris, kconserv)?),
            wovvo: Some(imd::wovvo(t1, t2, eris, kconserv)?),
            woooo: None,
            wooov: None,
            wovoo: None,
            wvovv: None,
            wvvvv: None,
            wvvvo: None,
        })
    }

    /// `_make_shared_1e` ALONE (`:1520-1530`) — `Loo`, `Lvv`, `Fov`, and none
    /// of the shared two-electron set.
    ///
    /// This is what `make_ip(ip_partition='mp')` and `make_ea(ea_partition=
    /// 'mp')` leave behind: `:1552` and `:1597` both guard `_make_shared_2e()`
    /// on `!= 'mp'`, and no `'mp'` branch reads `Wovov`, `Wovvo` or `Woovv`.
    /// [`RhfEomImds::wovov`] and [`RhfEomImds::wovvo`] refuse rather than
    /// return a wrong answer if one ever does.
    ///
    /// # Errors
    /// Propagates every intermediate build.
    pub fn make_shared_1e(
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
            wovov: None,
            wovvo: None,
            woooo: None,
            wooov: None,
            wovoo: None,
            wvovv: None,
            wvvvv: None,
            wvvvo: None,
        })
    }

    /// `_IMDS.make_ip` (`:1550-1577`), the `ip_partition = None` branch.
    ///
    /// # Errors
    /// Propagates every intermediate build.
    pub fn make_ip(self, kconserv: &Kconserv) -> Result<Self, PbcCcError> {
        self.make_ip_partition(kconserv, Partition::None)
    }

    /// `_IMDS.make_ip(ip_partition)` (`:1550-1577`) — `Woooo` is built only
    /// when the partition is not `'mp'` (`:1570-1571`); `Wooov` and `Wovoo`
    /// always are.
    ///
    /// # Errors
    /// Propagates every intermediate build; [`PbcCcError::NotImplementedUpstream`]
    /// for [`Partition::Full`], which upstream cannot build either — see
    /// [`Partition`].
    pub fn make_ip_partition(
        mut self,
        kconserv: &Kconserv,
        partition: Partition,
    ) -> Result<Self, PbcCcError> {
        Partition::refuse_full(partition)?;
        let (t1, t2) = (self.t1.clone(), self.t2.clone());
        if partition != Partition::Mp && self.woooo.is_none() {
            self.woooo = Some(imd::eom_woooo(&t1, &t2, self.eris, kconserv)?);
        }
        self.wooov = Some(imd::wooov(&t1, self.eris)?);
        self.wovoo = Some(imd::wovoo(&t1, &t2, self.eris, kconserv)?);
        Ok(self)
    }

    /// `_IMDS.make_ea` (`:1595-1626`), the `ea_partition = None` branch.
    ///
    /// # Errors
    /// Propagates every intermediate build.
    pub fn make_ea(self, kconserv: &Kconserv) -> Result<Self, PbcCcError> {
        self.make_ea_partition(kconserv, Partition::None)
    }

    /// `_IMDS.make_ea(ea_partition)` (`:1595-1626`).
    ///
    /// # The `Wvvvv` skip needs BOTH conditions
    ///
    /// `:1618` is `if ea_partition == 'mp' and np.all(t1 == 0)` — the phase's
    /// largest tensor is skipped only when the partition is `'mp'` AND `t1` is
    /// identically zero, and `Wvvvo` is then built from the eris rather than
    /// from `Wvvvv`. An `'mp'` run on a converged `t1` still pays for `Wvvvv`.
    ///
    /// # Errors
    /// Propagates every intermediate build; [`PbcCcError::NotImplementedUpstream`]
    /// for [`Partition::Full`].
    pub fn make_ea_partition(
        mut self,
        kconserv: &Kconserv,
        partition: Partition,
    ) -> Result<Self, PbcCcError> {
        Partition::refuse_full(partition)?;
        let (t1, t2) = (self.t1.clone(), self.t2.clone());
        self.wvovv = Some(imd::wvovv(&t1, self.eris)?);
        let t1_is_zero =
            t1.data().re.iter().all(|v| *v == 0.0) && t1.data().im.iter().all(|v| *v == 0.0);
        if partition == Partition::Mp && t1_is_zero {
            // `:1619` — no `Wvvvv` argument, so `Wvvvo` is built without it.
            self.wvvvo = Some(imd::wvvvo(&t1, &t2, self.eris, kconserv, None)?);
        } else {
            let w4 = imd::eom_wvvvv(&t1, &t2, self.eris, kconserv)?;
            self.wvvvo = Some(imd::wvvvo(&t1, &t2, self.eris, kconserv, Some(&w4))?);
            self.wvvvv = Some(w4);
        }
        Ok(self)
    }

    /// `_IMDS.make_ee(ee_partition)` (`:1666-1706`).
    ///
    /// # It RENAMES rather than rebuilds
    ///
    /// Every tensor `make_ee` produces already exists under an IP or EA name:
    /// `Foo ← Loo`, `Fvv ← Lvv`, `woOvV ← Woovv` (i.e. `eris.oovv`),
    /// `woVvO ← Wovvo`, `woVoV ← Wovov`, `woOoO ← Woooo`, `woOoV ← Wooov`,
    /// `woVoO ← Wovoo`, `wvOvV ← Wvovv`, `wvVvV ← Wvvvv`, `wvVvO ← Wvvvo`.
    /// Upstream builds whichever half is missing (`:1686-1706`); this is
    /// `make_ip` followed by `make_ea`, which is the union, and the two `if
    /// self.woooo.is_none()` guards make the overlap free.
    ///
    /// `ee_partition` is accepted by upstream's signature and never read — it
    /// appears in no branch of the body — so it is absent here.
    ///
    /// # Errors
    /// Propagates every intermediate build.
    pub fn make_ee(self, kconserv: &Kconserv) -> Result<Self, PbcCcError> {
        self.make_ip(kconserv)?.make_ea(kconserv)
    }

    /// `_IMDS.make_t3p2_ip(cc)` (`:1578-1593`) — the `EOMIP_Ta` (`:419-424`)
    /// intermediates.
    ///
    /// `t1`/`t2` are REPLACED by `pt1`/`pt2` and the entire set is rebuilt
    /// from them (`:1587`'s `self._made_shared_2e = False  # Force update`,
    /// and `_make_shared_1e` runs unconditionally inside `make_ip`), so
    /// `Loo`, `Lvv`, `Fov`, `Wovov`, `Wovvo`, `Woooo`, `Wooov` and `Wovoo` are
    /// all the PERTURBED ones. Only then is `Wmcik` added to `Wovoo`.
    ///
    /// Returns the intermediates and `delta_ccsd_energy`.
    ///
    /// # Errors
    /// Propagates the `T3[2]` build and every intermediate.
    pub fn make_t3p2_ip(
        t1: &ZArr,
        t2: &ZArr,
        eris: &'a KEris,
        kconserv: &Kconserv,
        padding: &Padding,
        lat: &KLattice<'_>,
    ) -> Result<(Self, f64), PbcCcError> {
        let p = t3p2(t1, t2, eris, kconserv, padding, lat)?;
        let mut imds = Self::make_shared(&p.pt1, &p.pt2, eris, kconserv)?.make_ip(kconserv)?;
        let mut w = imds.need(&imds.wovoo, "Wovoo")?.clone();
        w.add_assign(&p.wovoo)?;
        imds.wovoo = Some(w);
        Ok((imds, p.delta_ccsd_energy))
    }

    /// `_IMDS.make_t3p2_ea(cc)` (`:1629-1644`) — the `EOMEA_Ta` (`:819-824`)
    /// intermediates. As [`RhfEomImds::make_t3p2_ip`], with `Wacek` added to
    /// the rebuilt `Wvvvo`.
    ///
    /// # Errors
    /// As [`RhfEomImds::make_t3p2_ip`].
    pub fn make_t3p2_ea(
        t1: &ZArr,
        t2: &ZArr,
        eris: &'a KEris,
        kconserv: &Kconserv,
        padding: &Padding,
        lat: &KLattice<'_>,
    ) -> Result<(Self, f64), PbcCcError> {
        let p = t3p2(t1, t2, eris, kconserv, padding, lat)?;
        let mut imds = Self::make_shared(&p.pt1, &p.pt2, eris, kconserv)?.make_ea(kconserv)?;
        let mut w = imds.need(&imds.wvvvo, "Wvvvo")?.clone();
        w.add_assign(&p.wvvvo)?;
        imds.wvvvo = Some(w);
        Ok((imds, p.delta_ccsd_energy))
    }

    /// `_IMDS.make_t3p2_ip_ea(cc)` (`:1646-1664`) — BOTH sets from ONE
    /// `T3[2]` build.
    ///
    /// This module has it and the spin-orbital one does not; it exists so a
    /// caller wanting both `EOMIP_Ta` and `EOMEA_Ta` pays for `T3[2]` once.
    ///
    /// # Errors
    /// As [`RhfEomImds::make_t3p2_ip`].
    pub fn make_t3p2_ip_ea(
        t1: &ZArr,
        t2: &ZArr,
        eris: &'a KEris,
        kconserv: &Kconserv,
        padding: &Padding,
        lat: &KLattice<'_>,
    ) -> Result<(Self, f64), PbcCcError> {
        let p = t3p2(t1, t2, eris, kconserv, padding, lat)?;
        let mut imds = Self::make_shared(&p.pt1, &p.pt2, eris, kconserv)?
            .make_ip(kconserv)?
            .make_ea(kconserv)?;
        let mut w = imds.need(&imds.wovoo, "Wovoo")?.clone();
        w.add_assign(&p.wovoo)?;
        imds.wovoo = Some(w);
        let mut w = imds.need(&imds.wvvvo, "Wvvvo")?.clone();
        w.add_assign(&p.wvvvo)?;
        imds.wvvvo = Some(w);
        Ok((imds, p.delta_ccsd_energy))
    }

    /// `Wovov` — present unless the caller built an `'mp'` set.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] when it was not built.
    pub fn wovov(&self) -> Result<&ZArr, PbcCcError> {
        self.need(&self.wovov, "Wovov")
    }

    /// `Wovvo` — present unless the caller built an `'mp'` set.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] when it was not built.
    pub fn wovvo(&self) -> Result<&ZArr, PbcCcError> {
        self.need(&self.wovvo, "Wovvo")
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
    ipccsd_matvec_partition(vector, kshift, imds, kconserv, Partition::None)
}

/// `ipccsd_matvec` (`:39-104`) with `eom.partition` selected explicitly.
///
/// Only the 2h1p-2h1p block differs. Under [`Partition::Mp`] (`:69-77`) it is
/// the BARE Fock — `fvv[kb]`, `foo[ki]`, `foo[kj]` in place of `Lvv`/`Loo` —
/// and every two-body term is dropped, so the `Woooo`, `Wovvo`, `Wovov` and
/// `Woovv` contractions of the `None` branch do not run and their
/// intermediates are never read. The 1h-1h, 1h-2h1p and 2h1p-1h blocks are
/// shared by both branches and use `Loo`, `Fov` and `Wooov`/`Wovoo`, which
/// `make_ip(ip_partition='mp')` still builds.
///
/// # Errors
/// As [`ipccsd_matvec`]; [`PbcCcError::NotImplementedUpstream`] for
/// [`Partition::Full`] — see the module doc.
pub fn ipccsd_matvec_partition(
    vector: &ZArr,
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
    partition: Partition,
) -> Result<ZArr, PbcCcError> {
    Partition::refuse_full(partition)?;
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let (r1, r2) = vector_to_amplitudes_ip(vector, nkpts, nocc, nvir)?;
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
    // `:70-77` — the `'mp'` 2h1p-2h1p block: the BARE Fock, no two-body term.
    if partition == Partition::Mp {
        for ki in 0..nkpts {
            for kj in 0..nkpts {
                let kb = kconserv.get(ki, kshift, kj) as usize;
                let mut blk = hr2.slice_leading(&[ki, kj])?;
                blk.add_assign(&einsum(
                    "bd,ijd->ijb",
                    &[&imds.eris.fvv(kb)?, &r2.slice_leading(&[ki, kj])?],
                )?)?;
                blk.sub_assign(&einsum(
                    "li,ljb->ijb",
                    &[&imds.eris.foo(ki)?, &r2.slice_leading(&[ki, kj])?],
                )?)?;
                blk.sub_assign(&einsum(
                    "lj,ilb->ijb",
                    &[&imds.eris.foo(kj)?, &r2.slice_leading(&[ki, kj])?],
                )?)?;
                hr2.set_leading(&[ki, kj], &blk)?;
            }
        }
        return amplitudes_to_vector_ip(&hr1, &hr2);
    }

    let woooo = imds.need(&imds.woooo, "Woooo")?;
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
                        &imds.wovvo()?.slice_leading(&[kl, kb, kd])?,
                        &r2.slice_leading(&[ki, kl])?,
                    ],
                    2.0,
                )?)?;
                blk.sub_assign(&einsum(
                    "lbdj,lid->ijb",
                    &[
                        &imds.wovvo()?.slice_leading(&[kl, kb, kd])?,
                        &r2.slice_leading(&[kl, ki])?,
                    ],
                )?)?;
                // `:97` carries upstream's own `# typo in Ref` comment: the
                // published equation has a different index here and upstream's
                // code is the correct one.
                blk.sub_assign(&einsum(
                    "lbjd,ild->ijb",
                    &[
                        &imds.wovov()?.slice_leading(&[kl, kb, kj])?,
                        &r2.slice_leading(&[ki, kl])?,
                    ],
                )?)?;
                let _kd = kconserv.get(kl, ki, kb) as usize;
                blk.sub_assign(&einsum(
                    "lbid,ljd->ijb",
                    &[
                        &imds.wovov()?.slice_leading(&[kl, kb, ki])?,
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
                let mut sw = imds.wovvo()?.slice_leading(&[kl, kb, kd])?;
                sw.scale(2.0);
                sw.sub_assign(
                    &imds
                        .wovov()?
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
                        &imds.wovvo()?.slice_leading(&[kk, kb, kd])?,
                        &r2.slice_leading(&[kl, kj])?,
                    ],
                )?)?;
                blk.sub_assign(&einsum(
                    "kbjd,jlb->kld",
                    &[
                        &imds.wovov()?.slice_leading(&[kk, kb, kj])?,
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

/// `ipccsd_diag` (`:166-212`) with `eom.partition` selected explicitly.
///
/// `Hr1` is `−diag(Loo[kshift])` in BOTH branches (`:174`) — the partition
/// touches only the 2h1p block, where `'mp'` (`:177-186`) is the bare
/// `fvv[kb] − foo[ki] − foo[kj]` with every `Woooo`/`Wovov`/`Wovvo`/`Woovv`
/// term of the `None` branch dropped.
///
/// # Errors
/// As [`ipccsd_matvec_partition`].
pub fn ipccsd_diag_partition(
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
    partition: Partition,
) -> Result<ZArr, PbcCcError> {
    // The diagonals have NO `'full'` branch (`:177`/`:583` is `if 'mp' … else`),
    // so `'full'` here is the `None` diagonal — not a refusal.
    if partition != Partition::Mp {
        return ipccsd_diag(kshift, imds, kconserv);
    }
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);

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
            let fb = imds.eris.fvv(kb)?;
            let fi = imds.eris.foo(ki)?;
            let fj = imds.eris.foo(kj)?;
            let mut blk = ZArr::zeros(&[nocc, nocc, nvir]);
            for i in 0..nocc {
                for j in 0..nocc {
                    for b in 0..nvir {
                        let f = (i * nocc + j) * nvir + b;
                        // `:182` — an ASSIGNMENT, not an accumulation.
                        let (r, m) = fb.at(&[b, b])?;
                        blk.data_mut().re[f] = r;
                        blk.data_mut().im[f] = m;
                        let (r, m) = fi.at(&[i, i])?;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                        let (r, m) = fj.at(&[j, j])?;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                    }
                }
            }
            hr2.set_leading(&[ki, kj], &blk)?;
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
            let w = imds.wovov()?.slice_leading(&[kj, kb, kj])?;
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
            let w = imds.wovvo()?.slice_leading(&[kj, kb, kb])?;
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
            let w = imds.wovov()?.slice_leading(&[ki, kb, ki])?;
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
pub fn eaccsd_matvec(
    vector: &ZArr,
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    eaccsd_matvec_partition(vector, kshift, imds, kconserv, Partition::None)
}

/// `eaccsd_matvec` (`:430-505`) with `eom.partition` selected explicitly.
///
/// As [`ipccsd_matvec_partition`]: only the 2p1h-2p1h block differs, and the
/// [`Partition::Mp`] form (`:462-470`) is the bare `foo[kj]`/`fvv[ka]`/
/// `fvv[kb]` with every two-body term dropped — so `Wvvvv`, `Wovvo`, `Wovov`
/// and `Woovv` are never read. Note the sign pattern is `−foo + fvv + fvv`
/// here, matching the `None` branch's `−Loo + Lvv + Lvv` (`:479-483`); the
/// spin-orbital module's `'mp'` EA diagonal does NOT match its own `None`
/// branch that way — see [`crate::eom_kccsd_ghf::eaccsd_diag_mp`].
///
/// # Errors
/// As [`ipccsd_matvec_partition`].
#[allow(clippy::too_many_lines)]
pub fn eaccsd_matvec_partition(
    vector: &ZArr,
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
    partition: Partition,
) -> Result<ZArr, PbcCcError> {
    Partition::refuse_full(partition)?;
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
    // `:463-470` — the `'mp'` 2p1h-2p1h block: the BARE Fock, no two-body term.
    if partition == Partition::Mp {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(kshift, ka, kj) as usize;
                let mut blk = hr2.slice_leading(&[kj, ka])?;
                blk.sub_assign(&einsum(
                    "lj,lab->jab",
                    &[&imds.eris.foo(kj)?, &r2.slice_leading(&[kj, ka])?],
                )?)?;
                blk.add_assign(&einsum(
                    "ac,jcb->jab",
                    &[&imds.eris.fvv(ka)?, &r2.slice_leading(&[kj, ka])?],
                )?)?;
                blk.add_assign(&einsum(
                    "bd,jad->jab",
                    &[&imds.eris.fvv(kb)?, &r2.slice_leading(&[kj, ka])?],
                )?)?;
                hr2.set_leading(&[kj, ka], &blk)?;
            }
        }
        return amplitudes_to_vector_ea(&hr1, &hr2);
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
                        &imds.wovvo()?.slice_leading(&[kl, kb, kd])?,
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
                            .wovov()?
                            .slice_leading(&[kl, kb, kj])?
                            .transpose(&[1, 0, 3, 2])?,
                        &r2.slice_leading(&[kl, ka])?,
                    ],
                )?)?;
                blk.sub_assign(&einsum(
                    "bljd,lda->jab",
                    &[
                        &imds
                            .wovvo()?
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
                            .wovov()?
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
                let mut sw = imds.wovvo()?.slice_leading(&[kl, kb, kd])?;
                sw.scale(2.0);
                sw.sub_assign(
                    &imds
                        .wovov()?
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
                        &imds.wovov()?.slice_leading(&[kl, kb, kj])?,
                        &r2.slice_leading(&[kj, kb])?,
                    ],
                )?)?;
                blk.sub_assign(&einsum(
                    "lbcj,jdb->lcd",
                    &[
                        &imds.wovvo()?.slice_leading(&[kl, kb, kc])?,
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

/// `eaccsd_diag` (`:572-615`) with `eom.partition` selected explicitly.
///
/// `Hr1` is `diag(Lvv[kshift])` in both branches (`:580`). The `'mp'` 2p1h
/// block (`:583-591`) is `−foo[kj] + fvv[ka] + fvv[kb]` — the same sign
/// pattern as the `None` branch (`:595-597`), unlike the spin-orbital
/// module's, which flips `fvv[ka]` (see
/// [`crate::eom_kccsd_ghf::eaccsd_diag_mp`]).
///
/// [`Partition::Full`] is NOT refused here: the diagonal has no `'full'`
/// branch, so upstream computes the `None` diagonal for it and so does this.
///
/// # Errors
/// As [`ipccsd_matvec_partition`].
pub fn eaccsd_diag_partition(
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
    partition: Partition,
) -> Result<ZArr, PbcCcError> {
    // The diagonals have NO `'full'` branch (`:177`/`:583` is `if 'mp' … else`),
    // so `'full'` here is the `None` diagonal — not a refusal.
    if partition != Partition::Mp {
        return eaccsd_diag(kshift, imds, kconserv);
    }
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
            let fj = imds.eris.foo(kj)?;
            let fa = imds.eris.fvv(ka)?;
            let fb = imds.eris.fvv(kb)?;
            let mut blk = ZArr::zeros(&[nocc, nvir, nvir]);
            for j in 0..nocc {
                for a in 0..nvir {
                    for b in 0..nvir {
                        let f = (j * nvir + a) * nvir + b;
                        let (r, m) = fj.at(&[j, j])?;
                        blk.data_mut().re[f] -= r;
                        blk.data_mut().im[f] -= m;
                        let (r, m) = fa.at(&[a, a])?;
                        blk.data_mut().re[f] += r;
                        blk.data_mut().im[f] += m;
                        let (r, m) = fb.at(&[b, b])?;
                        blk.data_mut().re[f] += r;
                        blk.data_mut().im[f] += m;
                    }
                }
            }
            hr2.set_leading(&[kj, ka], &blk)?;
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
            let wjbjb = imds.wovov()?.slice_leading(&[kj, kb, kj])?;
            let wjbbj = imds.wovvo()?.slice_leading(&[kj, kb, kb])?;
            let wjaja = imds.wovov()?.slice_leading(&[kj, ka, kj])?;
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
    // `EOMIP`/`EOMEA` here inherit `ipccsd`/`eaccsd` from the spin-orbital
    // module (`:378`, `:777`), and those refuse both non-`None` partitions at
    // `eom_kccsd_ghf.py:618`/`:905`. The branches themselves live on in
    // `*_matvec_partition` / `*_diag_partition`.
    if opts.partition != Partition::None {
        return Err(crate::eom_kccsd_ghf::partition_refusal());
    }
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let kr2 = crate::eom_kccsd_ghf::kconserv_ee_r2(nkpts, kshift, kconserv);
    let size = match kind {
        Excitation::Ip => ip_vector_size(nkpts, nocc, nvir),
        Excitation::Ea => ea_vector_size(nkpts, nocc, nvir),
        // `EOMEESinglet` only. `EOMEETriplet` (`:1483`) and `EOMEESpinFlip`
        // (`:1489`) are SHELLS upstream — their `vector_size` is a bare
        // `return None` and they declare no matvec — so an EE run here is a
        // SINGLET run, which is what `eomee_ccsd_singlet` (`:838`) drives.
        // Upstream's own `eeccsd` (`:831`) is `raise NotImplementedError`.
        Excitation::Ee => ee_singlet_vector_size(nkpts, nocc, nvir, &kr2),
    };
    if kind == Excitation::Ee && opts.left {
        // `EOMEESinglet.gen_matvec` (`:1459-1467`) raises for `left`, with a
        // `# TODO allow left vectors to be computed`.
        return Err(PbcCcError::NotImplementedUpstream {
            upstream: "pbc/cc/eom_kccsd_rhf.py:1464",
            what: "EOMEESinglet.gen_matvec raises NotImplementedError for left=True",
        });
    }

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
            // `kernel_ee` (`eom_kccsd_ghf.py:1288-1296`) does NO masking —
            // its docstring says the `eom.mask_frozen()` parts were removed —
            // and `EOMEESinglet` declares no `mask_frozen` at all.
            Excitation::Ee => Ok(v.clone()),
        }
    };

    let ones = mask(&ZArr::zeros(&[size]), 1.0)?;
    let nfrozen = ones.data().re.iter().filter(|v| **v == 1.0).count();
    let nroots = opts.nroots.min(size).min(size - nfrozen).max(1);

    let diag = match kind {
        Excitation::Ip => ipccsd_diag(kshift, imds, kconserv)?,
        Excitation::Ea => eaccsd_diag(kshift, imds, kconserv)?,
        Excitation::Ee => eeccsd_diag(kshift, imds, kconserv)?,
    };
    let diag = mask(&diag, crate::kccsd_rhf::LARGE_DENOM)?;

    // `EOMEESinglet.get_init_guess` is `get_init_guess_cis` (`:1429`), NOT
    // the diagonal-argsort guess the IP/EA classes inherit — it diagonalises
    // the singles block of `Hbar` and pads with zeros.
    let guess = if kind == Excitation::Ee {
        ee_singlet_cis_guess(kshift, nroots, imds, kconserv)?
    } else if opts.koopmans {
        let seeds: Vec<usize> = match kind {
            Excitation::Ip => padding.occupied[kshift].iter().rev().copied().collect(),
            Excitation::Ea => padding.virtuals[kshift].to_vec(),
            Excitation::Ee => unreachable!("handled above"),
        };
        seeds
            .iter()
            .take(nroots)
            .map(|&n| {
                let mut g = ZArr::zeros(&[size]);
                if n < size {
                    g.data_mut().re[n] = 1.0;
                }
                mask(&g, 0.0)
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        let mut idx: Vec<usize> = (0..size).collect();
        idx.sort_by(|a, b| {
            diag.data().re[*a]
                .partial_cmp(&diag.data().re[*b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idx.iter()
            .take(nroots)
            .map(|&i| {
                let mut g = ZArr::zeros(&[size]);
                g.data_mut().re[i] = 1.0;
                mask(&g, 0.0)
            })
            .collect::<Result<Vec<_>, _>>()?
    };

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
                    (Excitation::Ee, false) => eeccsd_matvec_singlet(&v, kshift, imds, kconserv),
                    (Excitation::Ee, true) => unreachable!("refused above"),
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
        // The EE `r1` block is `[nkpts, nocc, nvir]`, so the singles weight is
        // the whole of it — `kernel_ee` (`eom_kccsd_ghf.py:1367`) takes
        // `‖r1‖²` over the same block.
        Excitation::Ee => nkpts * nocc * nvir,
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

// ---------------------------------------------------------------------------
// The CCSD* corrections (`:214-375`, `:617-774`)
// ---------------------------------------------------------------------------

/// The `1/2` prefactor both spin-adapted corrections end with (`:369`,
/// `:769`) — NOT the spin-orbital module's `1/12` (`eom_kccsd_ghf.py:603`).
const STAR_PREFACTOR: f64 = 0.5;

/// Upstream warns below this left-right overlap (`:325`, `:722`).
const STAR_SMALL_OVERLAP: f64 = 1e-7;

/// `<L|R> = l1·r1 + l2·r2` (`:311`), UNCONJUGATED and with NO `½` — the
/// spin-orbital module's `:519` carries a `0.5` on the doubles term and this
/// one does not, because the packings differ.
fn ldotr(l1: &ZArr, l2: &ZArr, r1: &ZArr, r2: &ZArr) -> (f64, f64) {
    let mut re = 0.0;
    let mut im = 0.0;
    for i in 0..l1.len() {
        re += l1.data().re[i] * r1.data().re[i] - l1.data().im[i] * r1.data().im[i];
        im += l1.data().re[i] * r1.data().im[i] + l1.data().im[i] * r1.data().re[i];
    }
    for i in 0..l2.len() {
        re += l2.data().re[i] * r2.data().re[i] - l2.data().im[i] * r2.data().im[i];
        im += l2.data().re[i] * r2.data().im[i] + l2.data().im[i] * r2.data().re[i];
    }
    (re, im)
}

fn scale_by_inverse(x: &mut ZArr, (re, im): (f64, f64)) {
    let d = re * re + im * im;
    x.scale_complex(re / d, -im / d);
}

/// `get_kconserv3(cell, kpts, [p, q, kshift, range(nkpts), range(nkpts)])`.
fn kklist(lat: &KLattice<'_>, p: usize, q: usize, kshift: usize, nkpts: usize) -> Vec<usize> {
    let all: Vec<usize> = (0..nkpts).collect();
    get_kconserv3(
        lat.a,
        lat.kpts,
        &[
            KIdx::One(p),
            KIdx::One(q),
            KIdx::One(kshift),
            KIdx::Many(all.clone()),
            KIdx::Many(all),
        ],
    )
    .data
    .iter()
    .map(|v| *v as usize)
    .collect()
}

fn split_mo_energy(eris: &KEris, nocc: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let o = eris.mo_energy.iter().map(|e| e[..nocc].to_vec()).collect();
    let v = eris.mo_energy.iter().map(|e| e[nocc..].to_vec()).collect();
    (o, v)
}

/// `contract_l3p(l1, l2, [ki,kj,kk,ka,kb])` (`:230-249`) — one perturbed left
/// 3h2p block, before the `P(ia|jb)` symmetrisation.
fn ip_l3p(
    l1: &ZArr,
    l2: &ZArr,
    kv: [usize; 5],
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (eris, nocc, nvir) = (imds.eris, imds.eris.nocc, imds.eris.nvir);
    let [ki, kj, kk, ka, kb] = kv;
    let mut out = ZArr::zeros(&[nocc, nocc, nocc, nvir, nvir]);
    // `:236-237` — note the `0.5`, which the spin-orbital form does not have.
    if kk == kshift && kj == kconserv.get(ka, ki, kb) as usize {
        out.add_assign(&einsum_scaled(
            "ijab,k->ijkab",
            &[&eris.blk(Blk::Oovv, ki, kj, ka)?, l1],
            0.5,
        )?)?;
    }
    // `:238-239`
    let ke = kconserv.get(kb, ki, ka) as usize;
    out.add_assign(&einsum(
        "eiba,jke->ijkab",
        &[
            &eris.blk(Blk::Vovv, ke, ki, kb)?,
            &l2.slice_leading(&[kj, kk])?,
        ],
    )?)?;
    // `:240-241`
    let km = kconserv.get(kshift, ki, ka) as usize;
    out.sub_assign(&einsum(
        "kjmb,ima->ijkab",
        &[
            &eris.blk(Blk::Ooov, kk, kj, km)?,
            &l2.slice_leading(&[ki, km])?,
        ],
    )?)?;
    // `:242-243`
    let km = kconserv.get(ki, kb, kj) as usize;
    out.sub_assign(&einsum(
        "ijmb,mka->ijkab",
        &[
            &eris.blk(Blk::Ooov, ki, kj, km)?,
            &l2.slice_leading(&[km, kk])?,
        ],
    )?)?;
    Ok(out)
}

/// `contract_r3p(r1, r2, [ki,kj,kk,ka,kb])` (`:258-281`).
fn ip_r3p(
    r1: &ZArr,
    r2: &ZArr,
    kv: [usize; 5],
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (eris, t2, nocc, nvir) = (imds.eris, &imds.t2, imds.eris.nocc, imds.eris.nvir);
    let [ki, kj, kk, ka, kb] = kv;
    let mut out = ZArr::zeros(&[nocc, nocc, nocc, nvir, nvir]);
    // `:264-265`
    let tmp = einsum("mbke,m->bke", &[&eris.blk(Blk::Ovov, kshift, kb, kk)?, r1])?;
    out.sub_assign(&einsum(
        "bke,ijae->ijkab",
        &[&tmp, &t2.slice_leading(&[ki, kj, ka])?],
    )?)?;
    // `:266-268` — `ke` is computed and never used; upstream's own dead line.
    let _ke = kconserv.get(kb, kshift, kj) as usize;
    let tmp = einsum("bmje,m->bej", &[&eris.blk(Blk::Voov, kb, kshift, kj)?, r1])?;
    out.sub_assign(&einsum(
        "bej,ikae->ijkab",
        &[&tmp, &t2.slice_leading(&[ki, kk, ka])?],
    )?)?;
    // `:269-271`
    let km = kconserv.get(ka, ki, kb) as usize;
    let tmp = einsum("mnjk,n->mjk", &[&eris.blk(Blk::Oooo, km, kshift, kj)?, r1])?;
    out.add_assign(&einsum(
        "mjk,imab->ijkab",
        &[&tmp, &t2.slice_leading(&[ki, km, ka])?],
    )?)?;
    // `:272-273`
    let ke = kconserv.get(kk, kshift, kj) as usize;
    out.add_assign(&einsum(
        "eiba,kje->ijkab",
        &[
            &eris.blk(Blk::Vovv, ke, ki, kb)?.conj(),
            &r2.slice_leading(&[kk, kj])?,
        ],
    )?)?;
    // `:274-275`
    let km = kconserv.get(kk, kb, kj) as usize;
    out.sub_assign(&einsum(
        "kjmb,mia->ijkab",
        &[
            &eris.blk(Blk::Ooov, kk, kj, km)?.conj(),
            &r2.slice_leading(&[km, ki])?,
        ],
    )?)?;
    // `:276-277`
    let km = kconserv.get(ki, kb, kj) as usize;
    out.sub_assign(&einsum(
        "ijmb,kma->ijkab",
        &[
            &eris.blk(Blk::Ooov, ki, kj, km)?.conj(),
            &r2.slice_leading(&[kk, km])?,
        ],
    )?)?;
    Ok(out)
}

/// `ipccsd_star_contract` (`:214-375`) — the SPIN-ADAPTED IP-CCSD\*.
///
/// # Three things separate this from the spin-orbital form
///
/// * **`l2` is symmetrised first** (`:315-320`): `l2 ← (l2 + 2·l2ᵀ)/3` with
///   `l2ᵀ[ki,kj] = l2[kj,ki].transpose(1,0,2)`. `r2` is NOT touched.
/// * **`P(ijk)` carries spin-adapted weights** (`:352-358`): `4, 1, 1, −2,
///   −2, −2` over the six index permutations, where the spin-orbital form has
///   three terms with weight `1`.
/// * **Only the LEFT side is permuted.** `:365` contracts `Plijkab` with the
///   BARE `rijkab`; the spin-orbital `:601` permutes both.
///
/// The prefactor is `½`, not `1/12`.
///
/// # Errors
/// Propagates every block access and shape check.
#[allow(clippy::too_many_lines)]
pub fn ipccsd_star_contract(
    pairs: &[StarPair<'_>],
    kshift: usize,
    imds: &RhfEomImds<'_>,
    padding: &Padding,
    kconserv: &Kconserv,
    lat: &KLattice<'_>,
) -> Result<Vec<StarRoot>, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let (mo_e_o, mo_e_v) = split_mo_energy(imds.eris, nocc);
    let mut out = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let (mut l1, l2) = vector_to_amplitudes_ip(pair.l, nkpts, nocc, nvir)?;
        let (r1, r2) = vector_to_amplitudes_ip(pair.r, nkpts, nocc, nvir)?;

        // `:311` — the overlap is taken on the UNSYMMETRISED `l2`.
        let dot = ldotr(&l1, &l2, &r1, &r2);
        let small = dot.0.hypot(dot.1) < STAR_SMALL_OVERLAP;

        // `:314-320` — `l2 ← (l2 + 2·l2ᵀ)/3`.
        let mut l2s = ZArr::zeros(l2.shape());
        for ki in 0..nkpts {
            for kj in 0..nkpts {
                let mut blk = l2.slice_leading(&[ki, kj])?;
                blk.zip_assign(&l2.slice_leading(&[kj, ki])?.transpose(&[1, 0, 2])?, 2.0)?;
                blk.scale(1.0 / 3.0);
                l2s.set_leading(&[ki, kj], &blk)?;
            }
        }
        let mut l2 = l2s;

        scale_by_inverse(&mut l1, dot);
        scale_by_inverse(&mut l2, dot);

        let mut delta_re = 0.0_f64;
        let mut delta_im = 0.0_f64;

        for ka in 0..nkpts {
            for kb in 0..nkpts {
                let kk_of = kklist(lat, ka, kb, kshift, nkpts);
                let shape = [nkpts, nkpts, nocc, nocc, nocc, nvir, nvir];
                let mut lijkab = ZArr::zeros(&shape);
                let mut rijkab = ZArr::zeros(&shape);

                for ki in 0..nkpts {
                    for kj in 0..nkpts {
                        let kk = kk_of[ki * nkpts + kj];
                        // `contract_pl3p` (`:251-256`) — `X + P(ia|jb) X`,
                        // where `P(ia|jb)` swaps `(i,a)` with `(j,b)`: the
                        // k-vector is permuted `[1,0,2,4,3]` and the RESULT is
                        // transposed the same way.
                        let mut l = ip_l3p(&l1, &l2, [ki, kj, kk, ka, kb], kshift, imds, kconserv)?;
                        l.add_assign(
                            &ip_l3p(&l1, &l2, [kj, ki, kk, kb, ka], kshift, imds, kconserv)?
                                .transpose(&[1, 0, 2, 4, 3])?,
                        )?;
                        lijkab.set_leading(&[ki, kj], &l)?;

                        let mut r = ip_r3p(&r1, &r2, [ki, kj, kk, ka, kb], kshift, imds, kconserv)?;
                        r.add_assign(
                            &ip_r3p(&r1, &r2, [kj, ki, kk, kb, ka], kshift, imds, kconserv)?
                                .transpose(&[1, 0, 2, 4, 3])?,
                        )?;
                        rijkab.set_leading(&[ki, kj], &r)?;
                    }
                }

                // `:349-367`
                let eab = crate::kccsd_t::epq2(&mo_e_v, &padding.virtuals, ka, kb, nvir, -1.0);
                for ki in 0..nkpts {
                    for kj in 0..nkpts {
                        let kk = kk_of[ki * nkpts + kj];
                        // `:352-358` — the spin-adapted `P(ijk)` weights.
                        let mut pl = lijkab.slice_leading(&[ki, kj])?;
                        pl.scale(4.0);
                        pl.zip_assign(
                            &lijkab
                                .slice_leading(&[kj, kk])?
                                .transpose(&[2, 0, 1, 3, 4])?,
                            1.0,
                        )?;
                        pl.zip_assign(
                            &lijkab
                                .slice_leading(&[kk, ki])?
                                .transpose(&[1, 2, 0, 3, 4])?,
                            1.0,
                        )?;
                        pl.zip_assign(
                            &lijkab
                                .slice_leading(&[ki, kk])?
                                .transpose(&[0, 2, 1, 3, 4])?,
                            -2.0,
                        )?;
                        pl.zip_assign(
                            &lijkab
                                .slice_leading(&[kk, kj])?
                                .transpose(&[2, 1, 0, 3, 4])?,
                            -2.0,
                        )?;
                        pl.zip_assign(
                            &lijkab
                                .slice_leading(&[kj, ki])?
                                .transpose(&[1, 0, 2, 3, 4])?,
                            -2.0,
                        )?;
                        // `:365` — the BARE `rijkab`, not a permuted one.
                        let pr = rijkab.slice_leading(&[ki, kj])?;

                        let eijk = crate::kccsd_t::epqr3(
                            &mo_e_o,
                            &padding.occupied,
                            ki,
                            kj,
                            kk,
                            nocc,
                            1.0,
                        );
                        let mut f = 0;
                        for &e_ijk in &eijk {
                            for &e_ab in &eab {
                                let denom = e_ijk + e_ab + pair.eval;
                                let (lr, li) = (pl.data().re[f], pl.data().im[f]);
                                let (rr, ri) = (pr.data().re[f], pr.data().im[f]);
                                delta_re += (lr * rr - li * ri) / denom;
                                delta_im += (lr * ri + li * rr) / denom;
                                f += 1;
                            }
                        }
                    }
                }
            }
        }

        let delta_e = delta_re * STAR_PREFACTOR;
        out.push(StarRoot {
            e_star: pair.eval + delta_e,
            delta_e,
            delta_e_imag: delta_im * STAR_PREFACTOR,
            ldotr: dot,
            small_overlap: small,
        });
    }
    Ok(out)
}

/// `contract_l3p(l1, l2, [ki,kj,ka,kb,kc])` (`:633-651`).
fn ea_l3p(
    l1: &ZArr,
    l2: &ZArr,
    kv: [usize; 5],
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (eris, nocc, nvir) = (imds.eris, imds.eris.nocc, imds.eris.nvir);
    let [ki, kj, ka, kb, kc] = kv;
    let mut out = ZArr::zeros(&[nocc, nocc, nvir, nvir, nvir]);
    // `:641-642`
    if kc == kshift && kb == kconserv.get(ki, ka, kj) as usize {
        out.sub_assign(&einsum_scaled(
            "ijab,c->ijabc",
            &[&eris.blk(Blk::Oovv, ki, kj, ka)?, l1],
            0.5,
        )?)?;
    }
    // `:643-644`
    let km = kconserv.get(ki, ka, kj) as usize;
    out.add_assign(&einsum(
        "jima,mbc->ijabc",
        &[
            &eris.blk(Blk::Ooov, kj, ki, km)?,
            &l2.slice_leading(&[km, kb])?,
        ],
    )?)?;
    // `:645-646`
    let ke = kconserv.get(kshift, ka, ki) as usize;
    out.sub_assign(&einsum(
        "ejcb,iae->ijabc",
        &[
            &eris.blk(Blk::Vovv, ke, kj, kc)?,
            &l2.slice_leading(&[ki, ka])?,
        ],
    )?)?;
    // `:647-648`
    let ke = kconserv.get(kshift, kc, ki) as usize;
    out.sub_assign(&einsum(
        "ejab,iec->ijabc",
        &[
            &eris.blk(Blk::Vovv, ke, kj, ka)?,
            &l2.slice_leading(&[ki, ke])?,
        ],
    )?)?;
    Ok(out)
}

/// `contract_r3p(r1, r2, [ki,kj,ka,kb,kc])` (`:665-689`).
fn ea_r3p(
    r1: &ZArr,
    r2: &ZArr,
    kv: [usize; 5],
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (eris, t2, nocc, nvir) = (imds.eris, &imds.t2, imds.eris.nocc, imds.eris.nvir);
    let [ki, kj, ka, kb, kc] = kv;
    let mut out = ZArr::zeros(&[nocc, nocc, nvir, nvir, nvir]);
    // `:673-676`
    let ke = kconserv.get(ki, ka, kj) as usize;
    let tmp = einsum("bcef,f->bce", &[&eris.blk(Blk::Vvvv, kb, kc, ke)?, r1])?;
    out.sub_assign(&einsum(
        "bce,ijae->ijabc",
        &[&tmp, &t2.slice_leading(&[ki, kj, ka])?],
    )?)?;
    // `:677-679`
    let km = kconserv.get(kshift, kc, kj) as usize;
    let tmp = einsum("mcje,e->mcj", &[&eris.blk(Blk::Ovov, km, kc, kj)?, r1])?;
    out.add_assign(&einsum(
        "mcj,imab->ijabc",
        &[&tmp, &t2.slice_leading(&[ki, km, ka])?],
    )?)?;
    // `:680-682` — `voov[kb,km,kj]` with `km = kconserv[kc,ki,ka]`.
    let km = kconserv.get(kc, ki, ka) as usize;
    let tmp = einsum("bmje,e->mbj", &[&eris.blk(Blk::Voov, kb, km, kj)?, r1])?;
    out.add_assign(&einsum(
        "mbj,imac->ijabc",
        &[&tmp, &t2.slice_leading(&[ki, km, ka])?],
    )?)?;
    // `:683-684`
    let km = kconserv.get(ki, ka, kj) as usize;
    out.add_assign(&einsum(
        "jima,mcb->ijabc",
        &[
            &eris.blk(Blk::Ooov, kj, ki, km)?.conj(),
            &r2.slice_leading(&[km, kc])?,
        ],
    )?)?;
    // `:685-686`
    let ke = kconserv.get(kshift, ka, ki) as usize;
    out.sub_assign(&einsum(
        "ejcb,iea->ijabc",
        &[
            &eris.blk(Blk::Vovv, ke, kj, kc)?.conj(),
            &r2.slice_leading(&[ki, ke])?,
        ],
    )?)?;
    // `:687-688`
    let ke = kconserv.get(kshift, kc, kj) as usize;
    out.sub_assign(&einsum(
        "eiba,jce->ijabc",
        &[
            &eris.blk(Blk::Vovv, ke, ki, kb)?.conj(),
            &r2.slice_leading(&[kj, kc])?,
        ],
    )?)?;
    Ok(out)
}

/// `eaccsd_star_contract` (`:617-774`) — the SPIN-ADAPTED EA-CCSD\*.
///
/// The same three departures from the spin-orbital form as
/// [`ipccsd_star_contract`], with EA's own index pattern: `l2ᵀ[kj,kb] =
/// l2[kj,ka].transpose(0,2,1)` where `kb = kconserv[kj,ka,kshift]` (`:716-720`)
/// — a k-index REMAP, not just an axis swap, which the IP transposition is
/// not.
///
/// # Errors
/// As [`ipccsd_star_contract`].
#[allow(clippy::too_many_lines)]
pub fn eaccsd_star_contract(
    pairs: &[StarPair<'_>],
    kshift: usize,
    imds: &RhfEomImds<'_>,
    padding: &Padding,
    kconserv: &Kconserv,
    lat: &KLattice<'_>,
) -> Result<Vec<StarRoot>, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let (mo_e_o, mo_e_v) = split_mo_energy(imds.eris, nocc);
    let mut out = Vec::with_capacity(pairs.len());

    for pair in pairs {
        let (mut l1, l2) = vector_to_amplitudes_ea(pair.l, nkpts, nocc, nvir)?;
        let (r1, r2) = vector_to_amplitudes_ea(pair.r, nkpts, nocc, nvir)?;

        let dot = ldotr(&l1, &l2, &r1, &r2);
        let small = dot.0.hypot(dot.1) < STAR_SMALL_OVERLAP;

        // `:716-720` — `l2T[kj,kb] = l2[kj,ka].transpose(0,2,1)`.
        let mut l2t = ZArr::zeros(l2.shape());
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv.get(kj, ka, kshift) as usize;
                l2t.set_leading(
                    &[kj, kb],
                    &l2.slice_leading(&[kj, ka])?.transpose(&[0, 2, 1])?,
                )?;
            }
        }
        let mut l2 = l2;
        l2.zip_assign(&l2t, 2.0)?;
        l2.scale(1.0 / 3.0);

        scale_by_inverse(&mut l1, dot);
        scale_by_inverse(&mut l2, dot);

        let mut delta_re = 0.0_f64;
        let mut delta_im = 0.0_f64;

        for ki in 0..nkpts {
            for kj in 0..nkpts {
                let kc_of = kklist(lat, ki, kj, kshift, nkpts);
                let shape = [nkpts, nkpts, nocc, nocc, nvir, nvir, nvir];
                let mut lijabc = ZArr::zeros(&shape);
                let mut rijabc = ZArr::zeros(&shape);

                for ka in 0..nkpts {
                    for kb in 0..nkpts {
                        let kc = kc_of[ka * nkpts + kb];
                        // `contract_pl3p` (`:653-663`): the k-vector permutes
                        // `[1,0,3,2,4]` and the result transposes the same.
                        let mut l = ea_l3p(&l1, &l2, [ki, kj, ka, kb, kc], kshift, imds, kconserv)?;
                        l.add_assign(
                            &ea_l3p(&l1, &l2, [kj, ki, kb, ka, kc], kshift, imds, kconserv)?
                                .transpose(&[1, 0, 3, 2, 4])?,
                        )?;
                        lijabc.set_leading(&[ka, kb], &l)?;

                        let mut r = ea_r3p(&r1, &r2, [ki, kj, ka, kb, kc], kshift, imds, kconserv)?;
                        r.add_assign(
                            &ea_r3p(&r1, &r2, [kj, ki, kb, ka, kc], kshift, imds, kconserv)?
                                .transpose(&[1, 0, 3, 2, 4])?,
                        )?;
                        rijabc.set_leading(&[ka, kb], &r)?;
                    }
                }

                // `:743-766`
                let eij = crate::kccsd_t::epq2(&mo_e_o, &padding.occupied, ki, kj, nocc, 1.0);
                for ka in 0..nkpts {
                    for kb in 0..nkpts {
                        let kc = kc_of[ka * nkpts + kb];
                        let mut pl = lijabc.slice_leading(&[ka, kb])?;
                        pl.scale(4.0);
                        pl.zip_assign(
                            &lijabc
                                .slice_leading(&[kb, kc])?
                                .transpose(&[0, 1, 4, 2, 3])?,
                            1.0,
                        )?;
                        pl.zip_assign(
                            &lijabc
                                .slice_leading(&[kc, ka])?
                                .transpose(&[0, 1, 3, 4, 2])?,
                            1.0,
                        )?;
                        pl.zip_assign(
                            &lijabc
                                .slice_leading(&[ka, kc])?
                                .transpose(&[0, 1, 2, 4, 3])?,
                            -2.0,
                        )?;
                        pl.zip_assign(
                            &lijabc
                                .slice_leading(&[kc, kb])?
                                .transpose(&[0, 1, 4, 3, 2])?,
                            -2.0,
                        )?;
                        pl.zip_assign(
                            &lijabc
                                .slice_leading(&[kb, ka])?
                                .transpose(&[0, 1, 3, 2, 4])?,
                            -2.0,
                        )?;
                        let pr = rijabc.slice_leading(&[ka, kb])?;

                        let eabc = crate::kccsd_t::epqr3(
                            &mo_e_v,
                            &padding.virtuals,
                            ka,
                            kb,
                            kc,
                            nvir,
                            -1.0,
                        );
                        let mut f = 0;
                        for &e_ij in &eij {
                            for &e_abc in &eabc {
                                let denom = e_ij + e_abc + pair.eval;
                                let (lr, li) = (pl.data().re[f], pl.data().im[f]);
                                let (rr, ri) = (pr.data().re[f], pr.data().im[f]);
                                delta_re += (lr * rr - li * ri) / denom;
                                delta_im += (lr * ri + li * rr) / denom;
                                f += 1;
                            }
                        }
                    }
                }
            }
        }

        let delta_e = delta_re * STAR_PREFACTOR;
        out.push(StarRoot {
            e_star: pair.eval + delta_e,
            delta_e,
            delta_e_imag: delta_im * STAR_PREFACTOR,
            ldotr: dot,
            small_overlap: small,
        });
    }
    Ok(out)
}

/// `perturbed_ccsd_kernel` (`eom_kccsd_ghf.py:625-648`) with this module's
/// matvecs — `EOMIP`/`EOMEA` here inherit `ipccsd_star`/`eaccsd_star` from the
/// spin-orbital module and override only `ccsd_star_contract` (`:382`,
/// `:781`).
///
/// # Errors
/// Propagates both Davidson solves and the contraction.
pub fn perturbed_ccsd_kernel(
    kind: Excitation,
    kshift: usize,
    imds: &RhfEomImds<'_>,
    padding: &Padding,
    kconserv: &Kconserv,
    lat: &KLattice<'_>,
    opts: &EomOpts,
) -> Result<Vec<StarRoot>, PbcCcError> {
    if kind == Excitation::Ee {
        return Err(PbcCcError::NotImplementedUpstream {
            upstream: "pbc/cc/eom_kccsd_rhf.py:1406",
            what: "EOMEE declares no ccsd_star_contract; there is no EE-CCSD* correction",
        });
    }
    let right = eom_kernel(
        kind,
        kshift,
        imds,
        padding,
        kconserv,
        &EomOpts {
            left: false,
            ..*opts
        },
    )?;
    let left = eom_kernel(
        kind,
        kshift,
        imds,
        padding,
        kconserv,
        &EomOpts {
            left: true,
            ..*opts
        },
    )?;
    let pairs = crate::eom_kccsd_ghf::sort_left_right_eigensystem(&right, &left, 1e-6);
    match kind {
        Excitation::Ip => ipccsd_star_contract(&pairs, kshift, imds, padding, kconserv, lat),
        Excitation::Ea => eaccsd_star_contract(&pairs, kshift, imds, padding, kconserv, lat),
        Excitation::Ee => unreachable!("refused above"),
    }
}

// ---------------------------------------------------------------------------
// EOM-EE-CCSD, SINGLET (`:826-1481`)
// ---------------------------------------------------------------------------

/// `EOMEESinglet.vector_size(kshift)` (`:1432-1458`).
///
/// # The size depends on `kshift`, and not by a closed form
///
/// `r1` contributes `nkpts·nocc·nvir`. `r2` is packed over the composite index
/// `(i k_i a k_a)`, and a `(ki,ka,kj)` triple contributes `nov(nov+1)/2` when
/// `ki·nkpts+ka == kj·nkpts+kb`, `nov²` when it is GREATER, and nothing when
/// it is smaller — so the count has to be summed over the triples with the
/// `kconserv_r2` of that shift. `16-REVIEW §5` recorded exactly this: the EE
/// size "is *not* a closed form".
#[must_use]
pub fn ee_singlet_vector_size(
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    kconserv_r2: &[usize],
) -> usize {
    let nov = nocc * nvir;
    let mut size = nkpts * nov;
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kconserv_r2[(ki * nkpts + ka) * nkpts + kj];
                let (kika, kjkb) = (ki * nkpts + ka, kj * nkpts + kb);
                if kika == kjkb {
                    size += nov * (nov + 1) / 2;
                } else if kika > kjkb {
                    size += nov * nov;
                }
            }
        }
    }
    size
}

/// The `(kika, kjkb)` classification the singlet packing turns on, and the
/// `(ki, ka, kj)` ORDER the offsets run in (`:894`, `:927`).
///
/// Upstream's two packing functions both iterate `for ki, ka, kj in
/// loop_kkk(nkpts)` — note the middle index is `ka`, not `kj` — while
/// `vector_size` iterates `for ki, kj, ka`. The counts agree because the
/// condition is symmetric under relabelling, but the OFFSETS do not: a port
/// that used the `vector_size` order would pack the same number of elements
/// in a different order. This iterator is the packing order.
fn singlet_pairs(
    nkpts: usize,
    kconserv_r2: &[usize],
) -> impl Iterator<Item = (usize, usize, usize, usize)> + '_ {
    (0..nkpts).flat_map(move |ki| {
        (0..nkpts).flat_map(move |ka| {
            (0..nkpts).map(move |kj| {
                let kb = kconserv_r2[(ki * nkpts + ka) * nkpts + kj];
                (ki, ka, kj, kb)
            })
        })
    })
}

/// `vector_to_amplitudes_singlet(vector, nkpts, nmo, nocc, kconserv)`
/// (`:849-893`).
///
/// Returns `(r1, r2)` with `r1` shaped `[nkpts, nocc, nvir]` and `r2` shaped
/// `[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]` indexed
/// `[ki, kj, ka, i, j, a, b]` — upstream's final layout after its
/// `reshape(...).transpose(0,2,1,3,5,4,6)` (`:891`), which this builds
/// directly rather than materialising the intermediate `(ki,ka)`-composite
/// form.
///
/// # Errors
/// [`PbcCcError::Shape`] if the vector is not [`ee_singlet_vector_size`] long.
pub fn vector_to_amplitudes_singlet(
    vector: &ZArr,
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    kconserv_r2: &[usize],
) -> Result<(ZArr, ZArr), PbcCcError> {
    let want = ee_singlet_vector_size(nkpts, nocc, nvir, kconserv_r2);
    if vector.len() != want {
        return Err(PbcCcError::Shape(format!(
            "EE singlet vector of {} elements, expected {want}",
            vector.len()
        )));
    }
    let nov = nocc * nvir;
    let n1 = nkpts * nov;
    let mut r1 = ZArr::zeros(&[nkpts, nocc, nvir]);
    r1.data_mut().re.copy_from_slice(&vector.data().re[..n1]);
    r1.data_mut().im.copy_from_slice(&vector.data().im[..n1]);

    let mut r2 = ZArr::zeros(&[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]);
    // `r2[ki,kj,ka][i,j,a,b] <- tmp[ia, jb]` and, in the same breath,
    // `r2[kj,ki,kb][j,i,b,a] <- tmp[ia, jb]` — upstream's paired assignment at
    // `:882-883` / `:887-888`.
    let put = |r2: &mut ZArr, ki, kj, ka, kb, ia: usize, jb: usize, v: (f64, f64)| {
        let (i, a) = (ia / nvir, ia % nvir);
        let (j, b) = (jb / nvir, jb % nvir);
        let s1 =
            ((((ki * nkpts + kj) * nkpts + ka) * nocc + i) * nocc + j) * nvir * nvir + a * nvir + b;
        r2.data_mut().re[s1] = v.0;
        r2.data_mut().im[s1] = v.1;
        let s2 =
            ((((kj * nkpts + ki) * nkpts + kb) * nocc + j) * nocc + i) * nvir * nvir + b * nvir + a;
        r2.data_mut().re[s2] = v.0;
        r2.data_mut().im[s2] = v.1;
    };

    let mut off = n1;
    for (ki, ka, kj, kb) in singlet_pairs(nkpts, kconserv_r2) {
        let (kika, kjkb) = (ki * nkpts + ka, kj * nkpts + kb);
        if kika == kjkb {
            // `kika == kjkb` forces `ki == kj` and `ka == kb`, so the two
            // assignments fill the (p,q) and (q,p) halves of ONE block.
            for p in 0..nov {
                for q in 0..=p {
                    let v = (vector.data().re[off], vector.data().im[off]);
                    put(&mut r2, ki, kj, ka, kb, p, q, v);
                    off += 1;
                }
            }
        } else if kika > kjkb {
            for p in 0..nov {
                for q in 0..nov {
                    let v = (vector.data().re[off], vector.data().im[off]);
                    put(&mut r2, ki, kj, ka, kb, p, q, v);
                    off += 1;
                }
            }
        }
    }
    debug_assert_eq!(off, want);
    Ok((r1, r2))
}

/// `amplitudes_to_vector_singlet(r1, r2, kconserv)` (`:897-935`), the inverse
/// of [`vector_to_amplitudes_singlet`].
///
/// # Errors
/// [`PbcCcError::Shape`] on a rank or extent mismatch.
pub fn amplitudes_to_vector_singlet(
    r1: &ZArr,
    r2: &ZArr,
    nkpts: usize,
    nocc: usize,
    nvir: usize,
    kconserv_r2: &[usize],
) -> Result<ZArr, PbcCcError> {
    let nov = nocc * nvir;
    let n1 = nkpts * nov;
    let size = ee_singlet_vector_size(nkpts, nocc, nvir, kconserv_r2);
    let mut v = ZArr::zeros(&[size]);
    if r1.len() != n1 {
        return Err(PbcCcError::Shape(format!(
            "EE singlet r1 of {} elements, expected {n1}",
            r1.len()
        )));
    }
    v.data_mut().re[..n1].copy_from_slice(&r1.data().re);
    v.data_mut().im[..n1].copy_from_slice(&r1.data().im);

    let at = |ki: usize, kj: usize, ka: usize, ia: usize, jb: usize| -> usize {
        let (i, a) = (ia / nvir, ia % nvir);
        let (j, b) = (jb / nvir, jb % nvir);
        ((((ki * nkpts + kj) * nkpts + ka) * nocc + i) * nocc + j) * nvir * nvir + a * nvir + b
    };

    let mut off = n1;
    for (ki, ka, kj, kb) in singlet_pairs(nkpts, kconserv_r2) {
        let (kika, kjkb) = (ki * nkpts + ka, kj * nkpts + kb);
        if kika == kjkb {
            for p in 0..nov {
                for q in 0..=p {
                    let s = at(ki, kj, ka, p, q);
                    v.data_mut().re[off] = r2.data().re[s];
                    v.data_mut().im[off] = r2.data().im[s];
                    off += 1;
                }
            }
        } else if kika > kjkb {
            for p in 0..nov {
                for q in 0..nov {
                    let s = at(ki, kj, ka, p, q);
                    v.data_mut().re[off] = r2.data().re[s];
                    v.data_mut().im[off] = r2.data().im[s];
                    off += 1;
                }
            }
        }
    }
    debug_assert_eq!(off, size);
    Ok(v)
}

/// `eeccsd_matvec_singlet(eom, vector, kshift, imds)` (`:969-1222`).
///
/// # The intermediate names
///
/// `_IMDS.make_ee` (`:1666-1706`) RENAMES the set rather than rebuilding it:
/// `Foo ← Loo`, `Fvv ← Lvv`, `woOvV ← Woovv` (which is `eris.oovv`),
/// `woVvO ← Wovvo`, `woVoV ← Wovov`, `woOoO ← Woooo`, `woOoV ← Wooov`,
/// `woVoO ← Wovoo`, `wvOvV ← Wvovv`, `wvVvV ← Wvvvv`, `wvVvO ← Wvvvo`. This
/// port keeps the [`RhfEomImds`] field names and quotes the upstream name at
/// each use, so a reader can follow either.
///
/// # The four antisymmetrised tensors
///
/// `:993-1027` builds `r2bar`, `woOoV_bar`, `wvOvV_bar` and `woVvO_bar` up
/// front. Only `woVvO_bar` mixes two DIFFERENT intermediates — it is
/// `2·woVvO − woVoV` transposed, not `2·X − X` — which is the one place a
/// transcription can go wrong silently.
///
/// # Errors
/// Propagates every intermediate access and shape check.
#[allow(clippy::too_many_lines)]
pub fn eeccsd_matvec_singlet(
    vector: &ZArr,
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let kr1 = crate::eom_kccsd_ghf::kconserv_ee_r1(nkpts, kshift, kconserv);
    let kr2 = crate::eom_kccsd_ghf::kconserv_ee_r2(nkpts, kshift, kconserv);
    let k2 = |ki: usize, ka: usize, kj: usize| kr2[(ki * nkpts + ka) * nkpts + kj];

    let (r1, r2) = vector_to_amplitudes_singlet(vector, nkpts, nocc, nvir, &kr2)?;

    let woooo = imds.need(&imds.woooo, "woOoO")?;
    let wooov = imds.need(&imds.wooov, "woOoV")?;
    let wovoo = imds.need(&imds.wovoo, "woVoO")?;
    let wvovv = imds.need(&imds.wvovv, "wvOvV")?;
    let wvvvv = imds.need(&imds.wvvvv, "wvVvV")?;
    let wvvvo = imds.need(&imds.wvvvo, "wvVvO")?;
    let wovvo = imds.wovvo()?;
    let wovov = imds.wovov()?;

    // `:993-1027` — the four antisymmetrised tensors.
    let mut r2bar = ZArr::zeros(r2.shape());
    let mut wooov_bar = ZArr::zeros(wooov.shape());
    let mut wvovv_bar = ZArr::zeros(wvovv.shape());
    let mut wovvo_bar = ZArr::zeros(wovvo.shape());
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                // `:1002-1004` — `rbar_ijab = 2 r_ijab − r_ijba`.
                let kb = k2(ki, ka, kj);
                let mut b = r2.slice_leading(&[ki, kj, ka])?;
                b.scale(2.0);
                b.sub_assign(&r2.slice_leading(&[ki, kj, kb])?.transpose(&[0, 1, 3, 2])?)?;
                r2bar.set_leading(&[ki, kj, ka], &b)?;

                // `:1013` — `wbar_nmie = 2 W_nmie − W_mnie^T`, at
                // `(wkn, wkm, wki) = (ki, kj, ka)`.
                let mut b = wooov.slice_leading(&[ki, kj, ka])?;
                b.scale(2.0);
                b.sub_assign(
                    &wooov
                        .slice_leading(&[kj, ki, ka])?
                        .transpose(&[1, 0, 2, 3])?,
                )?;
                wooov_bar.set_leading(&[ki, kj, ka], &b)?;

                // `:1021` — `wbar_amfe = 2 W_amfe − W_amef`, at
                // `(wka, wkm, wkf) = (ki, kj, ka)` and `wke = kconserv[wka, wkf, wkm]`.
                let wke = kconserv.get(ki, ka, kj) as usize;
                let mut b = wvovv.slice_leading(&[ki, kj, ka])?;
                b.scale(2.0);
                b.sub_assign(
                    &wvovv
                        .slice_leading(&[ki, kj, wke])?
                        .transpose(&[0, 1, 3, 2])?,
                )?;
                wvovv_bar.set_leading(&[ki, kj, ka], &b)?;

                // `:1027` — `wbar_mbej = 2 W_mbej − W_mbje`, and the second
                // term comes from `woVoV`, NOT from `woVvO`. At
                // `(wkm, wkb, wke) = (ki, kj, ka)`, `wkj = kconserv[wkm, wke, wkb]`.
                let wkj = kconserv.get(ki, ka, kj) as usize;
                let mut b = wovvo.slice_leading(&[ki, kj, ka])?;
                b.scale(2.0);
                b.sub_assign(
                    &wovov
                        .slice_leading(&[ki, kj, wkj])?
                        .transpose(&[0, 1, 3, 2])?,
                )?;
                wovvo_bar.set_leading(&[ki, kj, ka], &b)?;
            }
        }
    }

    // `:1029-1060` — `Hr1`.
    let mut hr1 = ZArr::zeros(&[nkpts, nocc, nvir]);
    for ki in 0..nkpts {
        let ka = kr1[ki];
        let mut acc = ZArr::zeros(&[nocc, nvir]);
        acc.sub_assign(&einsum(
            "mi,ma->ia",
            &[&imds.loo.slice_leading(&[ki])?, &r1.slice_leading(&[ki])?],
        )?)?;
        acc.add_assign(&einsum(
            "ac,ic->ia",
            &[&imds.lvv.slice_leading(&[ka])?, &r1.slice_leading(&[ki])?],
        )?)?;
        for (km, &ke) in kr1.iter().enumerate() {
            acc.add_assign(&einsum_scaled(
                "maei,me->ia",
                &[
                    &wovvo.slice_leading(&[km, ka, ke])?,
                    &r1.slice_leading(&[km])?,
                ],
                2.0,
            )?)?;
            acc.sub_assign(&einsum(
                "maie,me->ia",
                &[
                    &wovov.slice_leading(&[km, ka, ki])?,
                    &r1.slice_leading(&[km])?,
                ],
            )?)?;
            acc.add_assign(&einsum_scaled(
                "me,imae->ia",
                &[
                    &imds.fov.slice_leading(&[km])?,
                    &r2.slice_leading(&[ki, km, ka])?,
                ],
                2.0,
            )?)?;
            acc.sub_assign(&einsum(
                "me,miae->ia",
                &[
                    &imds.fov.slice_leading(&[km])?,
                    &r2.slice_leading(&[km, ki, ka])?,
                ],
            )?)?;
            for ke in 0..nkpts {
                acc.add_assign(&einsum_scaled(
                    "amef,imef->ia",
                    &[
                        &wvovv.slice_leading(&[ka, km, ke])?,
                        &r2.slice_leading(&[ki, km, ke])?,
                    ],
                    2.0,
                )?)?;
                let kf = kconserv.get(ka, ke, km) as usize;
                acc.sub_assign(&einsum(
                    "amfe,imef->ia",
                    &[
                        &wvovv.slice_leading(&[ka, km, kf])?,
                        &r2.slice_leading(&[ki, km, ke])?,
                    ],
                )?)?;
                // `:1057-1059` — the dummy `ke` is renamed `kn` here.
                let kn = ke;
                acc.add_assign(&einsum_scaled(
                    "mnie,mnae->ia",
                    &[
                        &wooov.slice_leading(&[km, kn, ki])?,
                        &r2.slice_leading(&[km, kn, ka])?,
                    ],
                    -2.0,
                )?)?;
                acc.add_assign(&einsum(
                    "mnie,nmae->ia",
                    &[
                        &wooov.slice_leading(&[km, kn, ki])?,
                        &r2.slice_leading(&[kn, km, ka])?,
                    ],
                )?)?;
            }
        }
        hr1.set_leading(&[ki], &acc)?;
    }

    // `:1062-1129` — `Hr2`.
    let mut hr2 = ZArr::zeros(r2.shape());
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = k2(ki, ka, kj);
                let mut acc = hr2.slice_leading(&[ki, kj, ka])?;
                acc.sub_assign(&einsum(
                    "mj,imab->ijab",
                    &[
                        &imds.loo.slice_leading(&[kj])?,
                        &r2.slice_leading(&[ki, kj, ka])?,
                    ],
                )?)?;
                acc.sub_assign(&einsum(
                    "mi,jmba->ijab",
                    &[
                        &imds.loo.slice_leading(&[ki])?,
                        &r2.slice_leading(&[kj, ki, kb])?,
                    ],
                )?)?;
                acc.add_assign(&einsum(
                    "be,ijae->ijab",
                    &[
                        &imds.lvv.slice_leading(&[kb])?,
                        &r2.slice_leading(&[ki, kj, ka])?,
                    ],
                )?)?;
                acc.add_assign(&einsum(
                    "ae,jibe->ijab",
                    &[
                        &imds.lvv.slice_leading(&[ka])?,
                        &r2.slice_leading(&[kj, ki, kb])?,
                    ],
                )?)?;

                let ke = kr1[ki];
                acc.add_assign(&einsum(
                    "abej,ie->ijab",
                    &[
                        &wvvvo.slice_leading(&[ka, kb, ke])?,
                        &r1.slice_leading(&[ki])?,
                    ],
                )?)?;
                let ke = kr1[kj];
                acc.add_assign(&einsum(
                    "baei,je->ijab",
                    &[
                        &wvvvo.slice_leading(&[kb, ka, ke])?,
                        &r1.slice_leading(&[kj])?,
                    ],
                )?)?;
                let km = kconserv.get(ki, kb, kj) as usize;
                acc.sub_assign(&einsum(
                    "mbij,ma->ijab",
                    &[
                        &wovoo.slice_leading(&[km, kb, ki])?,
                        &r1.slice_leading(&[km])?,
                    ],
                )?)?;
                let km = kconserv.get(ki, ka, kj) as usize;
                acc.sub_assign(&einsum(
                    "maji,mb->ijab",
                    &[
                        &wovoo.slice_leading(&[km, ka, kj])?,
                        &r1.slice_leading(&[km])?,
                    ],
                )?)?;
                hr2.set_leading(&[ki, kj, ka], &acc)?;

                // `:1096-1112` — `tmp`, then the SAME `tmp` transposed into
                // the `[kj, ki, kb]` slot. That second write is why `Hr2` is
                // accumulated globally rather than per block.
                let mut tmp = ZArr::zeros(&[nocc, nocc, nvir, nvir]);
                for km in 0..nkpts {
                    let ke = kconserv.get(km, kj, kb) as usize;
                    tmp.add_assign(&einsum(
                        "mbej,imae->ijab",
                        &[
                            &wovvo_bar.slice_leading(&[km, kb, ke])?,
                            &r2.slice_leading(&[ki, km, ka])?,
                        ],
                    )?)?;
                    tmp.sub_assign(&einsum(
                        "mbej,imea->ijab",
                        &[
                            &wovvo.slice_leading(&[km, kb, ke])?,
                            &r2.slice_leading(&[ki, km, ke])?,
                        ],
                    )?)?;
                    let ke = kconserv.get(km, kj, ka) as usize;
                    tmp.sub_assign(&einsum(
                        "maje,imeb->ijab",
                        &[
                            &wovov.slice_leading(&[km, ka, kj])?,
                            &r2.slice_leading(&[ki, km, ke])?,
                        ],
                    )?)?;
                }
                let mut acc = hr2.slice_leading(&[ki, kj, ka])?;
                acc.add_assign(&tmp)?;
                hr2.set_leading(&[ki, kj, ka], &acc)?;
                let mut acc = hr2.slice_leading(&[kj, ki, kb])?;
                acc.add_assign(&tmp.transpose(&[1, 0, 3, 2])?)?;
                hr2.set_leading(&[kj, ki, kb], &acc)?;

                // `:1114-1122` — the two four-index ladders. The loop variable
                // is `km` upstream and renamed `ke` inside the first term.
                let mut acc = hr2.slice_leading(&[ki, kj, ka])?;
                for km in 0..nkpts {
                    let ke = km;
                    acc.add_assign(&einsum(
                        "abef,ijef->ijab",
                        &[
                            &wvvvv.slice_leading(&[ka, kb, ke])?,
                            &r2.slice_leading(&[ki, kj, ke])?,
                        ],
                    )?)?;
                    let kn = kconserv.get(ki, km, kj) as usize;
                    acc.add_assign(&einsum(
                        "mnij,mnab->ijab",
                        &[
                            &woooo.slice_leading(&[km, kn, ki])?,
                            &r2.slice_leading(&[km, kn, ka])?,
                        ],
                    )?)?;
                }
                hr2.set_leading(&[ki, kj, ka], &acc)?;
            }
        }
    }

    // `:1131-1180` — the four `M = W·r` intermediates.
    //
    // Upstream writes these with a FREE k-axis inside the einsum
    // (`imds.woOvV[km]`, `imds.woOvV[:, :, ke]`, `woOoV_bar[kn, :, ki]`), and
    // `16-VERIFICATION` records that free k-axis gathers are shape-silent —
    // `oovv[:,km,ke]` and `oovv[km,:,ke]` have the same shape. They are
    // written here as explicit k loops instead.
    let mut wr2_oo = ZArr::zeros(&[nkpts, nocc, nocc]);
    let mut wr2_vv = ZArr::zeros(&[nkpts, nvir, nvir]);
    let mut wr1_oo = ZArr::zeros(&[nkpts, nocc, nocc]);
    let mut wr1_vv = ZArr::zeros(&[nkpts, nvir, nvir]);
    for kx in 0..nkpts {
        // `:1141-1146` — `Wr2_jm = W_mnef rbar_jnef`, `km = kconserv_r1[kj]`.
        let kj = kx;
        let km = kr1[kj];
        let mut acc = ZArr::zeros(&[nocc, nocc]);
        for kn in 0..nkpts {
            for ke in 0..nkpts {
                acc.add_assign(&einsum(
                    "mnef,jnef->jm",
                    &[
                        &imds.eris.blk(Blk::Oovv, km, kn, ke)?,
                        &r2bar.slice_leading(&[kj, kn, ke])?,
                    ],
                )?)?;
            }
        }
        wr2_oo.set_leading(&[kj], &acc)?;

        // `:1148-1156` — `Wr2_eb = W_mnef rbar_mnbf`, `kb = kconserv_r1[ke]`.
        let ke = kx;
        let kb = kr1[ke];
        let mut acc = ZArr::zeros(&[nvir, nvir]);
        for km in 0..nkpts {
            for kn in 0..nkpts {
                acc.add_assign(&einsum(
                    "mnef,mnbf->eb",
                    &[
                        &imds.eris.blk(Blk::Oovv, km, kn, ke)?,
                        &r2bar.slice_leading(&[km, kn, kb])?,
                    ],
                )?)?;
            }
        }
        wr2_vv.set_leading(&[ke], &acc)?;

        // `:1158-1166` — `Wr1_in = wbar_nmie r_me`, `kn = kconserv_r1[ki]`.
        let ki = kx;
        let kn = kr1[ki];
        let mut acc = ZArr::zeros(&[nocc, nocc]);
        for km in 0..nkpts {
            acc.add_assign(&einsum(
                "nmie,me->in",
                &[
                    &wooov_bar.slice_leading(&[kn, km, ki])?,
                    &r1.slice_leading(&[km])?,
                ],
            )?)?;
        }
        wr1_oo.set_leading(&[ki], &acc)?;

        // `:1168-1176` — `Wr1_fa = wbar_amfe r_me`, `ka = kconserv_r1[kf]`.
        let kf = kx;
        let ka = kr1[kf];
        let mut acc = ZArr::zeros(&[nvir, nvir]);
        for km in 0..nkpts {
            acc.add_assign(&einsum(
                "amfe,me->fa",
                &[
                    &wvovv_bar.slice_leading(&[ka, km, kf])?,
                    &r1.slice_leading(&[km])?,
                ],
            )?)?;
        }
        wr1_vv.set_leading(&[kf], &acc)?;
    }

    // `:1180-1216` — the eight contractions against `t2`.
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = k2(ki, ka, kj);
                let mut acc = hr2.slice_leading(&[ki, kj, ka])?;
                let km = kr1[kj];
                acc.sub_assign(&einsum(
                    "jm,imab->ijab",
                    &[
                        &wr2_oo.slice_leading(&[kj])?,
                        &imds.t2.slice_leading(&[ki, km, ka])?,
                    ],
                )?)?;
                let km = kr1[ki];
                acc.sub_assign(&einsum(
                    "im,jmba->ijab",
                    &[
                        &wr2_oo.slice_leading(&[ki])?,
                        &imds.t2.slice_leading(&[kj, km, kb])?,
                    ],
                )?)?;
                let ke = kconserv.get(ki, ka, kj) as usize;
                acc.sub_assign(&einsum(
                    "eb,ijae->ijab",
                    &[
                        &wr2_vv.slice_leading(&[ke])?,
                        &imds.t2.slice_leading(&[ki, kj, ka])?,
                    ],
                )?)?;
                let ke = kconserv.get(kj, kb, ki) as usize;
                acc.sub_assign(&einsum(
                    "ea,jibe->ijab",
                    &[
                        &wr2_vv.slice_leading(&[ke])?,
                        &imds.t2.slice_leading(&[kj, ki, kb])?,
                    ],
                )?)?;

                let kn = kr1[ki];
                acc.sub_assign(&einsum(
                    "in,jnba->ijab",
                    &[
                        &wr1_oo.slice_leading(&[ki])?,
                        &imds.t2.slice_leading(&[kj, kn, kb])?,
                    ],
                )?)?;
                let kn = kr1[kj];
                acc.sub_assign(&einsum(
                    "jn,inab->ijab",
                    &[
                        &wr1_oo.slice_leading(&[kj])?,
                        &imds.t2.slice_leading(&[ki, kn, ka])?,
                    ],
                )?)?;
                let kf = kconserv.get(kj, kb, ki) as usize;
                acc.add_assign(&einsum(
                    "fa,jibf->ijab",
                    &[
                        &wr1_vv.slice_leading(&[kf])?,
                        &imds.t2.slice_leading(&[kj, ki, kb])?,
                    ],
                )?)?;
                let kf = kconserv.get(ki, ka, kj) as usize;
                acc.add_assign(&einsum(
                    "fb,ijaf->ijab",
                    &[
                        &wr1_vv.slice_leading(&[kf])?,
                        &imds.t2.slice_leading(&[ki, kj, ka])?,
                    ],
                )?)?;
                hr2.set_leading(&[ki, kj, ka], &acc)?;
            }
        }
    }

    amplitudes_to_vector_singlet(&hr1, &hr2, nkpts, nocc, nvir, &kr2)
}

/// `eeccsd_diag(eom, kshift, imds)` (`:1225-1280`).
///
/// `partition == 'mp'` is REFUSED — upstream's own branch is a bare
/// `raise NotImplementedError` behind a `# TODO Allow partition='mp'`
/// (`:1243-1244`).
///
/// # Errors
/// As [`eeccsd_matvec_singlet`].
#[allow(clippy::too_many_lines)]
pub fn eeccsd_diag(
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let kr1 = crate::eom_kccsd_ghf::kconserv_ee_r1(nkpts, kshift, kconserv);
    let kr2 = crate::eom_kccsd_ghf::kconserv_ee_r2(nkpts, kshift, kconserv);
    let woooo = imds.need(&imds.woooo, "woOoO")?;
    let wvvvv = imds.need(&imds.wvvvv, "wvVvV")?;
    let wovvo = imds.wovvo()?;
    let wovov = imds.wovov()?;

    let mut hr1 = ZArr::zeros(&[nkpts, nocc, nvir]);
    for (ki, &ka) in kr1.iter().enumerate() {
        let f_oo = imds.loo.slice_leading(&[ki])?;
        let f_vv = imds.lvv.slice_leading(&[ka])?;
        let wo = wovvo.slice_leading(&[ki, ka, ka])?;
        let wv = wovov.slice_leading(&[ki, ka, ki])?;
        let mut blk = ZArr::zeros(&[nocc, nvir]);
        for i in 0..nocc {
            for a in 0..nvir {
                let f = i * nvir + a;
                let (r, m) = f_oo.at(&[i, i])?;
                blk.data_mut().re[f] -= r;
                blk.data_mut().im[f] -= m;
                let (r, m) = f_vv.at(&[a, a])?;
                blk.data_mut().re[f] += r;
                blk.data_mut().im[f] += m;
                let (r, m) = wo.at(&[i, a, a, i])?;
                blk.data_mut().re[f] += r;
                blk.data_mut().im[f] += m;
                let (r, m) = wv.at(&[i, a, i, a])?;
                blk.data_mut().re[f] -= r;
                blk.data_mut().im[f] -= m;
            }
        }
        hr1.set_leading(&[ki], &blk)?;
    }

    let mut hr2 = ZArr::zeros(&[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]);
    for ki in 0..nkpts {
        for kj in 0..nkpts {
            for ka in 0..nkpts {
                let kb = kr2[(ki * nkpts + ka) * nkpts + kj];
                let fi = imds.loo.slice_leading(&[ki])?;
                let fj = imds.loo.slice_leading(&[kj])?;
                let fa = imds.lvv.slice_leading(&[ka])?;
                let fb = imds.lvv.slice_leading(&[kb])?;
                let w_jbbj = wovvo.slice_leading(&[kj, kb, kb])?;
                let w_jbjb = wovov.slice_leading(&[kj, kb, kj])?;
                let w_jaja = wovov.slice_leading(&[kj, ka, kj])?;
                let w_ibib = wovov.slice_leading(&[ki, kb, ki])?;
                let w_iaai = wovvo.slice_leading(&[ki, ka, ka])?;
                let w_iaia = wovov.slice_leading(&[ki, ka, ki])?;
                let w_abab = wvvvv.slice_leading(&[ka, kb, ka])?;
                let w_ijij = woooo.slice_leading(&[ki, kj, ki])?;

                // `:1268-1279` — the four `Woovv·t2` traces. Each contracts a
                // DIFFERENT pair of k-indices; the `km`/`ke` above each is
                // upstream's own derivation of it.
                let km1 = kconserv.get(ka, ki, kb) as usize;
                let a1 = imds.eris.blk(Blk::Oovv, ki, km1, ka)?;
                let b1 = imds.t2.slice_leading(&[ki, km1, ka])?;
                let km2 = kconserv.get(ka, kj, kb) as usize;
                let a2 = imds.eris.blk(Blk::Oovv, km2, kj, ka)?;
                let b2 = imds.t2.slice_leading(&[km2, kj, ka])?;
                let a3 = imds.eris.blk(Blk::Oovv, ki, kj, ka)?;
                let b3 = imds.t2.slice_leading(&[ki, kj, ka])?;
                let ke4 = kconserv.get(ki, kb, kj) as usize;
                let a4 = imds.eris.blk(Blk::Oovv, ki, kj, ke4)?;
                let b4 = imds.t2.slice_leading(&[ki, kj, ke4])?;
                // `iab`, `jab`, `ija`, `ijb` — each summed over the axis the
                // trace leaves free.
                let t1_iab = einsum("imab,imab->iab", &[&a1, &b1])?;
                let t2_jab = einsum("mjab,mjab->jab", &[&a2, &b2])?;
                let t3_ija = einsum("ijae,ijae->ija", &[&a3, &b3])?;
                let t4_ijb = einsum("ijeb,ijeb->ijb", &[&a4, &b4])?;

                let mut blk = ZArr::zeros(&[nocc, nocc, nvir, nvir]);
                let mut f = 0;
                for i in 0..nocc {
                    for j in 0..nocc {
                        for a in 0..nvir {
                            for b in 0..nvir {
                                let mut re = 0.0;
                                let mut im = 0.0;
                                let mut add = |v: (f64, f64), s: f64| {
                                    re += s * v.0;
                                    im += s * v.1;
                                };
                                add(fi.at(&[i, i])?, -1.0);
                                add(fj.at(&[j, j])?, -1.0);
                                add(fa.at(&[a, a])?, 1.0);
                                add(fb.at(&[b, b])?, 1.0);
                                add(w_jbbj.at(&[j, b, b, j])?, 1.0);
                                add(w_jbjb.at(&[j, b, j, b])?, -1.0);
                                add(w_jaja.at(&[j, a, j, a])?, -1.0);
                                add(w_ibib.at(&[i, b, i, b])?, -1.0);
                                add(w_iaai.at(&[i, a, a, i])?, 1.0);
                                add(w_iaia.at(&[i, a, i, a])?, -1.0);
                                add(w_abab.at(&[a, b, a, b])?, 1.0);
                                add(w_ijij.at(&[i, j, i, j])?, 1.0);
                                add(t1_iab.at(&[i, a, b])?, -1.0);
                                add(t2_jab.at(&[j, a, b])?, -1.0);
                                add(t3_ija.at(&[i, j, a])?, -1.0);
                                add(t4_ijb.at(&[i, j, b])?, -1.0);
                                blk.data_mut().re[f] = re;
                                blk.data_mut().im[f] = im;
                                f += 1;
                            }
                        }
                    }
                }
                hr2.set_leading(&[ki, kj, ka], &blk)?;
            }
        }
    }

    amplitudes_to_vector_singlet(&hr1, &hr2, nkpts, nocc, nvir, &kr2)
}

/// `eeccsd_matvec_singlet_Hr1(eom, vector, kshift, imds)` (`:1282-1314`) —
/// `Hbar·r1` alone, the block [`ee_singlet_cis_guess`] diagonalises.
///
/// # Errors
/// [`PbcCcError::Shape`] if the vector is not `nkpts·nocc·nvir` long.
pub fn eeccsd_matvec_singlet_hr1(
    vector: &ZArr,
    kshift: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<ZArr, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let n1 = nkpts * nocc * nvir;
    if vector.len() != n1 {
        return Err(PbcCcError::Shape(format!(
            "EE singlet r1 vector of {} elements, expected {n1}",
            vector.len()
        )));
    }
    let kr1 = crate::eom_kccsd_ghf::kconserv_ee_r1(nkpts, kshift, kconserv);
    let r1 = vector.reshape(&[nkpts, nocc, nvir])?;
    let wovvo = imds.wovvo()?;
    let wovov = imds.wovov()?;

    let mut hr1 = ZArr::zeros(&[nkpts, nocc, nvir]);
    for ki in 0..nkpts {
        let ka = kr1[ki];
        let mut acc = ZArr::zeros(&[nocc, nvir]);
        acc.sub_assign(&einsum(
            "mi,ma->ia",
            &[&imds.loo.slice_leading(&[ki])?, &r1.slice_leading(&[ki])?],
        )?)?;
        acc.add_assign(&einsum(
            "ac,ic->ia",
            &[&imds.lvv.slice_leading(&[ka])?, &r1.slice_leading(&[ki])?],
        )?)?;
        for (km, &ke) in kr1.iter().enumerate() {
            acc.add_assign(&einsum_scaled(
                "maei,me->ia",
                &[
                    &wovvo.slice_leading(&[km, ka, ke])?,
                    &r1.slice_leading(&[km])?,
                ],
                2.0,
            )?)?;
            acc.sub_assign(&einsum(
                "maie,me->ia",
                &[
                    &wovov.slice_leading(&[km, ka, ki])?,
                    &r1.slice_leading(&[km])?,
                ],
            )?)?;
        }
        hr1.set_leading(&[ki], &acc)?;
    }
    hr1.reshape(&[n1])
}

/// `get_init_guess_cis(eom, kshift, nroots, imds)` (`:1351-1369`), built on
/// `eeccsd_cis_approx_slow` (`:1317-1348`).
///
/// # This is not the IP/EA guess
///
/// `EOMEESinglet` overrides `get_init_guess` (`:1429`) with a CIS-like one:
/// it materialises the `r1`-block of `Hbar` COLUMN BY COLUMN — `nkpts·nocc·
/// nvir` matvecs of the singles-only [`eeccsd_matvec_singlet_hr1`] — and
/// diagonalises it. Upstream's own docstring explains why: "such evaluation
/// has N³ cost, but error free (because matvec() has been proven correct)".
/// The lowest `nroots` eigenvectors are then padded with zeros in the doubles
/// block.
///
/// Note this is a NON-Hermitian dense eigenproblem, sorted by the eigenvalue's
/// `argsort` (`:1339`), which for a complex array sorts by real part then
/// imaginary — numpy's lexicographic complex ordering, reproduced here.
///
/// # Errors
/// Propagates the matvecs and the dense eigensolve.
pub fn ee_singlet_cis_guess(
    kshift: usize,
    nroots: usize,
    imds: &RhfEomImds<'_>,
    kconserv: &Kconserv,
) -> Result<Vec<ZArr>, PbcCcError> {
    let (nkpts, nocc, nvir) = (imds.eris.nkpts, imds.eris.nocc, imds.eris.nvir);
    let kr2 = crate::eom_kccsd_ghf::kconserv_ee_r2(nkpts, kshift, kconserv);
    let n1 = nkpts * nocc * nvir;
    let size = ee_singlet_vector_size(nkpts, nocc, nvir, &kr2);

    // `:1327-1331` — one matvec per column, stored COLUMN-major for the
    // eigensolver: `h[col*n1 + row]` is `H1[row, col]`.
    let mut h = pyscf_algebra::CTensor {
        re: vec![0.0; n1 * n1],
        im: vec![0.0; n1 * n1],
    };
    for col in 0..n1 {
        let mut e = ZArr::zeros(&[n1]);
        e.data_mut().re[col] = 1.0;
        let c = eeccsd_matvec_singlet_hr1(&e, kshift, imds, kconserv)?;
        for row in 0..n1 {
            h.re[col * n1 + row] = c.data().re[row];
            h.im[col * n1 + row] = c.data().im[row];
        }
    }

    let (evals, evecs) = pyscf_algebra::zeig_general(&h, n1)
        .map_err(|e| PbcCcError::Algebra(format!("EE CIS guess eigensolve: {e}")))?;
    // `:1333-1336` — `eigval.argsort()[:nroots]`, numpy's complex ordering
    // (real part first, imaginary part as the tie-break).
    let mut idx: Vec<usize> = (0..n1).collect();
    idx.sort_by(|&x, &y| {
        (evals.re[x], evals.im[x])
            .partial_cmp(&(evals.re[y], evals.im[y]))
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut guess = Vec::with_capacity(nroots.min(n1));
    for &i in idx.iter().take(nroots.min(n1)) {
        let mut g = ZArr::zeros(&[size]);
        for row in 0..n1 {
            g.data_mut().re[row] = evecs.re[i * n1 + row];
            g.data_mut().im[row] = evecs.im[i * n1 + row];
        }
        guess.push(g);
    }
    Ok(guess)
}

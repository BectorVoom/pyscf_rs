//! `rccsd` — the MOLECULAR, COMPLEX-CAPABLE restricted CCSD
//! (`pyscf/cc/rccsd.py`, 432 l, and the part of `pyscf/cc/rintermediates.py`
//! it contracts against).
//!
//! # Why this lives in `pyscf-pbc-cc` and not in `pyscf-ccsd`
//!
//! `16-CONTEXT §1.2` records it as absent and as the one thing blocking the
//! Γ-point shim: *"`pyscf/cc/rccsd.py` (432 l) — used by `pbc/cc/ccsd.py:24`
//! `class RCCSD(rccsd.RCCSD)`; `pbc/ci/cisd.py:33` `rccsd._make_eris_incore`.
//! **absent** (the port has `ccsd.CCSD`, not the complex-capable
//! `rccsd.RCCSD`)."*
//!
//! `crates/pyscf-ccsd` is `f64` by construction — its `ChemistsEris`, its
//! intermediates and its DIIS all are — and upstream's own file header says
//! what separates this one: *"Intermediates for restricted CCSD. **Complex
//! integrals are supported.**"* (`rintermediates.py:17`). The planar-complex
//! [`ZArr`] and the deterministic [`einsum`] every contraction here is written
//! in live in THIS crate, so the arithmetic lives with its substrate, exactly
//! as `pyscf_ccsd::gccsd`'s doc reasons for the spin-orbital case: the narrow
//! molecular surface is declared there, the k-point-shaped arithmetic is
//! here.
//!
//! **The move is recorded, not hidden.** When a phase needs molecular complex
//! correlation for its own sake — `pbc/ci/cisd.py` is the next caller — this
//! module and `ZArr` belong in `pyscf-ccsd` together. Nothing here depends on
//! anything periodic: [`ChemistsErisZ`] is fed by [`crate::ccsd`]'s Γ-point
//! transform, but it would take a molecular one just as well.
//!
//! # This is NOT `kccsd_rhf` at `nkpts = 1`
//!
//! It would give the same numbers — and `oracle_gamma.rs` measures that it
//! does — but it is a different set of expressions: seven CHEMIST blocks
//! `oooo/ovoo/ovov/oovv/ovvo/ovvv/vvvv` against `kccsd_rhf`'s
//! `oooo/ooov/oovv/ovov/voov/vovv/vvvv`, no `kconserv`, and the
//! `cc2` branch (`rccsd.py:96-113`) which the k-point module has no analogue
//! of. Each function below is transcribed from its own upstream line.

use pyscf_algebra::CTensor;

use crate::error::PbcCcError;
use crate::zarr::{ZArr, einsum, einsum_scaled};

/// `_ChemistsERIs` (`rccsd.py:230-237`, on `ccsd._ChemistsERIs`) — the seven
/// MO blocks in CHEMISTS' notation `(pq|rs)`, complex.
///
/// The block set is NOT [`crate::keris::Blk`]'s. `_make_eris_incore`
/// (`:239-263`) slices `oooo`, `ovoo`, `ovov`, `oovv`, `ovvo`, `ovvv` and
/// `vvvv` out of one full `[nmo]⁴` tensor, and `update_amps` reads exactly
/// those seven — the k-point module's `ooov`/`voov`/`vovv` are the same
/// integrals under different index orders and are NOT interchangeable at a
/// call site.
#[derive(Debug, Clone)]
pub struct ChemistsErisZ {
    pub nocc: usize,
    pub nvir: usize,
    /// `[nmo, nmo]` — the Fock matrix in the MO basis.
    pub fock: ZArr,
    /// `mo_energy`, `nmo` long. For the Γ shim this carries the Madelung
    /// re-add on the occupied block (`pbc/cc/ccsd.py:146-150`).
    pub mo_energy: Vec<f64>,
    pub oooo: ZArr,
    pub ovoo: ZArr,
    pub ovov: ZArr,
    pub oovv: ZArr,
    pub ovvo: ZArr,
    pub ovvv: ZArr,
    pub vvvv: ZArr,
}

impl ChemistsErisZ {
    /// `_make_eris_incore(mycc, mo_coeff, ao2mofn)` (`:239-263`) — slice the
    /// seven blocks out of a full `[nmo]⁴` chemists' tensor.
    ///
    /// `eri` is `(pq|rs)` row-major `[p][q][r][s]`.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] if `eri` is not `nmo⁴` long or `fock` is not
    /// `[nmo, nmo]`.
    pub fn from_full(
        eri: &ZArr,
        fock: ZArr,
        mo_energy: Vec<f64>,
        nocc: usize,
    ) -> Result<Self, PbcCcError> {
        let nmo = mo_energy.len();
        let nvir = nmo - nocc;
        if eri.len() != nmo * nmo * nmo * nmo {
            return Err(PbcCcError::Shape(format!(
                "_make_eris_incore: eri has {} elements, expected nmo^4 = {}",
                eri.len(),
                nmo * nmo * nmo * nmo
            )));
        }
        if fock.shape() != [nmo, nmo] {
            return Err(PbcCcError::Shape(format!(
                "_make_eris_incore: fock is {:?}, expected [{nmo}, {nmo}]",
                fock.shape()
            )));
        }
        let e = eri.reshape(&[nmo, nmo, nmo, nmo])?;
        let o = (0, nocc);
        let v = (nocc, nmo);
        Ok(Self {
            nocc,
            nvir,
            fock,
            mo_energy,
            oooo: e.slice_axes(&[o, o, o, o])?,
            ovoo: e.slice_axes(&[o, v, o, o])?,
            ovov: e.slice_axes(&[o, v, o, v])?,
            oovv: e.slice_axes(&[o, o, v, v])?,
            ovvo: e.slice_axes(&[o, v, v, o])?,
            ovvv: e.slice_axes(&[o, v, v, v])?,
            vvvv: e.slice_axes(&[v, v, v, v])?,
        })
    }

    /// `eris.fock[:nocc,:nocc]`.
    ///
    /// # Errors
    /// Propagates the slice.
    pub fn foo(&self) -> Result<ZArr, PbcCcError> {
        self.fock.slice_axes(&[(0, self.nocc), (0, self.nocc)])
    }
    /// `eris.fock[:nocc,nocc:]`.
    ///
    /// # Errors
    /// Propagates the slice.
    pub fn fov(&self) -> Result<ZArr, PbcCcError> {
        let nmo = self.nocc + self.nvir;
        self.fock.slice_axes(&[(0, self.nocc), (self.nocc, nmo)])
    }
    /// `eris.fock[nocc:,nocc:]`.
    ///
    /// # Errors
    /// Propagates the slice.
    pub fn fvv(&self) -> Result<ZArr, PbcCcError> {
        let nmo = self.nocc + self.nvir;
        self.fock.slice_axes(&[(self.nocc, nmo), (self.nocc, nmo)])
    }
}

// ---------------------------------------------------------------------------
// `pyscf/cc/rintermediates.py` — the COMPLEX-capable restricted intermediates
// ---------------------------------------------------------------------------

/// `cc_Foo(t1, t2, eris)` — `rintermediates.py:30-38`, Hirata Eq. (37).
///
/// # Errors
/// Propagates every contraction.
pub fn cc_foo(t1: &ZArr, t2: &ZArr, eris: &ChemistsErisZ) -> Result<ZArr, PbcCcError> {
    let ovov = &eris.ovov;
    let mut f = einsum_scaled("kcld,ilcd->ki", &[ovov, t2], 2.0)?;
    f.sub_assign(&einsum("kdlc,ilcd->ki", &[ovov, t2])?)?;
    f.add_assign(&einsum_scaled("kcld,ic,ld->ki", &[ovov, t1, t1], 2.0)?)?;
    f.sub_assign(&einsum("kdlc,ic,ld->ki", &[ovov, t1, t1])?)?;
    f.add_assign(&eris.foo()?)?;
    Ok(f)
}

/// `cc_Fvv(t1, t2, eris)` — `:40-48`, Eq. (38).
///
/// # Errors
/// Propagates every contraction.
pub fn cc_fvv(t1: &ZArr, t2: &ZArr, eris: &ChemistsErisZ) -> Result<ZArr, PbcCcError> {
    let ovov = &eris.ovov;
    let mut f = einsum_scaled("kcld,klad->ac", &[ovov, t2], -2.0)?;
    f.add_assign(&einsum("kdlc,klad->ac", &[ovov, t2])?)?;
    f.add_assign(&einsum_scaled("kcld,ka,ld->ac", &[ovov, t1, t1], -2.0)?)?;
    f.add_assign(&einsum("kdlc,ka,ld->ac", &[ovov, t1, t1])?)?;
    f.add_assign(&eris.fvv()?)?;
    Ok(f)
}

/// `cc_Fov(t1, t2, eris)` — `:50-57`, Eq. (39). `t2` is unread upstream too.
///
/// # Errors
/// Propagates every contraction.
pub fn cc_fov(t1: &ZArr, eris: &ChemistsErisZ) -> Result<ZArr, PbcCcError> {
    let ovov = &eris.ovov;
    let mut f = einsum_scaled("kcld,ld->kc", &[ovov, t1], 2.0)?;
    f.sub_assign(&einsum("kdlc,ld->kc", &[ovov, t1])?)?;
    f.add_assign(&eris.fov()?)?;
    Ok(f)
}

/// `Loo(t1, t2, eris)` — `:61-68`, Eq. (40).
///
/// # Errors
/// Propagates every contraction.
pub fn loo(t1: &ZArr, t2: &ZArr, eris: &ChemistsErisZ) -> Result<ZArr, PbcCcError> {
    let mut l = cc_foo(t1, t2, eris)?;
    l.add_assign(&einsum("kc,ic->ki", &[&eris.fov()?, t1])?)?;
    l.add_assign(&einsum_scaled("lcki,lc->ki", &[&eris.ovoo, t1], 2.0)?)?;
    l.sub_assign(&einsum("kcli,lc->ki", &[&eris.ovoo, t1])?)?;
    Ok(l)
}

/// `Lvv(t1, t2, eris)` — `:70-77`, Eq. (41).
///
/// # Errors
/// Propagates every contraction.
pub fn lvv(t1: &ZArr, t2: &ZArr, eris: &ChemistsErisZ) -> Result<ZArr, PbcCcError> {
    let mut l = cc_fvv(t1, t2, eris)?;
    l.sub_assign(&einsum("kc,ka->ac", &[&eris.fov()?, t1])?)?;
    l.add_assign(&einsum_scaled("kdac,kd->ac", &[&eris.ovvv, t1], 2.0)?)?;
    l.sub_assign(&einsum("kcad,kd->ac", &[&eris.ovvv, t1])?)?;
    Ok(l)
}

/// `cc_Woooo(t1, t2, eris)` — `:81-89`, Eq. (42).
///
/// # Errors
/// Propagates every contraction.
pub fn cc_woooo(t1: &ZArr, t2: &ZArr, eris: &ChemistsErisZ) -> Result<ZArr, PbcCcError> {
    let mut w = einsum("lcki,jc->klij", &[&eris.ovoo, t1])?;
    w.add_assign(&einsum("kclj,ic->klij", &[&eris.ovoo, t1])?)?;
    w.add_assign(&einsum("kcld,ijcd->klij", &[&eris.ovov, t2])?)?;
    w.add_assign(&einsum("kcld,ic,jd->klij", &[&eris.ovov, t1, t1])?)?;
    w.add_assign(&eris.oooo.transpose(&[0, 2, 1, 3])?)?;
    Ok(w)
}

/// `cc_Wvvvv(t1, t2, eris)` — `:91-97`, Eq. (43). `t2` is unread upstream too.
///
/// # Errors
/// Propagates every contraction.
pub fn cc_wvvvv(t1: &ZArr, eris: &ChemistsErisZ) -> Result<ZArr, PbcCcError> {
    let mut w = einsum_scaled("kdac,kb->abcd", &[&eris.ovvv, t1], -1.0)?;
    w.sub_assign(&einsum("kcbd,ka->abcd", &[&eris.ovvv, t1])?)?;
    // `_get_vvvv(eris)` (`:228-237`) is `eris.vvvv` itself for the incore,
    // four-index form `_make_eris_incore` builds.
    w.add_assign(&eris.vvvv.transpose(&[0, 2, 1, 3])?)?;
    Ok(w)
}

/// `cc_Wvoov(t1, t2, eris)` — `:99-110`, Eq. (44).
///
/// # Errors
/// Propagates every contraction.
pub fn cc_wvoov(t1: &ZArr, t2: &ZArr, eris: &ChemistsErisZ) -> Result<ZArr, PbcCcError> {
    let mut w = einsum("kcad,id->akic", &[&eris.ovvv, t1])?;
    w.sub_assign(&einsum("kcli,la->akic", &[&eris.ovoo, t1])?)?;
    w.add_assign(&eris.ovvo.transpose(&[2, 0, 3, 1])?)?;
    let ovov = &eris.ovov;
    w.add_assign(&einsum_scaled("ldkc,ilda->akic", &[ovov, t2], -0.5)?)?;
    w.add_assign(&einsum_scaled("lckd,ilad->akic", &[ovov, t2], -0.5)?)?;
    w.sub_assign(&einsum("ldkc,id,la->akic", &[ovov, t1, t1])?)?;
    w.add_assign(&einsum("ldkc,ilad->akic", &[ovov, t2])?)?;
    Ok(w)
}

/// `cc_Wvovo(t1, t2, eris)` — `:112-120`, Eq. (45).
///
/// # Errors
/// Propagates every contraction.
pub fn cc_wvovo(t1: &ZArr, t2: &ZArr, eris: &ChemistsErisZ) -> Result<ZArr, PbcCcError> {
    let mut w = einsum("kdac,id->akci", &[&eris.ovvv, t1])?;
    w.sub_assign(&einsum("lcki,la->akci", &[&eris.ovoo, t1])?)?;
    w.add_assign(&eris.oovv.transpose(&[2, 0, 3, 1])?)?;
    let ovov = &eris.ovov;
    w.add_assign(&einsum_scaled("lckd,ilda->akci", &[ovov, t2], -0.5)?)?;
    w.sub_assign(&einsum("lckd,id,la->akci", &[ovov, t1, t1])?)?;
    Ok(w)
}

// ---------------------------------------------------------------------------
// `pyscf/cc/rccsd.py`
// ---------------------------------------------------------------------------

/// `energy(cc, t1, t2, eris)` — `rccsd.py:146-162`.
///
/// # The imaginary part is DISCARDED, with upstream's own threshold
///
/// `:160-161` warns when `|Im(e)| > 1e-4` and returns `e.real` regardless.
/// This returns `(re, im)` so the caller applies its own judgement;
/// [`kernel`] keeps upstream's behaviour.
///
/// # Errors
/// Propagates every contraction.
pub fn energy(t1: &ZArr, t2: &ZArr, eris: &ChemistsErisZ) -> Result<(f64, f64), PbcCcError> {
    let mut tau = einsum("ia,jb->ijab", &[t1, t1])?;
    tau.add_assign(t2)?;
    let mut e = einsum_scaled("ia,ia->", &[&eris.fov()?, t1], 2.0)?;
    e.add_assign(&einsum_scaled("ijab,iajb->", &[&tau, &eris.ovov], 2.0)?)?;
    e.sub_assign(&einsum("ijab,ibja->", &[&tau, &eris.ovov])?)?;
    let (re, im) = e.at(&[])?;
    Ok((re, im))
}

/// `ccsd.CCSDBase.init_amps(eris)` — `pyscf/cc/ccsd.py:1050-1077`, complex.
///
/// Returns `(emp2, t1, t2)`.
///
/// # Errors
/// Propagates every contraction.
pub fn init_amps(eris: &ChemistsErisZ) -> Result<(f64, ZArr, ZArr), PbcCcError> {
    let (nocc, nvir) = (eris.nocc, eris.nvir);
    let e = &eris.mo_energy;
    let eia: Vec<f64> = (0..nocc)
        .flat_map(|i| (0..nvir).map(move |a| (i, a)))
        .map(|(i, a)| e[i] - e[nocc + a])
        .collect();

    let mut t1 = eris.fov()?;
    for (f, d) in eia.iter().enumerate() {
        t1.data_mut().re[f] /= d;
        t1.data_mut().im[f] /= d;
    }

    // `:1068` — `t2 = ovov.transpose(0,2,1,3).conj() / eijab`.
    let mut t2 = eris.ovov.transpose(&[0, 2, 1, 3])?.conj();
    let mut f = 0;
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let d = eia[i * nvir + a] + eia[j * nvir + b];
                    t2.data_mut().re[f] /= d;
                    t2.data_mut().im[f] /= d;
                    f += 1;
                }
            }
        }
    }

    // `:1070-1071` — note the SECOND term contracts `t2[j,i,a,b]`, not
    // `t2[i,j,a,b]`: upstream writes `'jiab,iajb'`.
    let mut emp2 = einsum_scaled("ijab,iajb->", &[&t2, &eris.ovov], 2.0)?;
    emp2.sub_assign(&einsum("jiab,iajb->", &[&t2, &eris.ovov])?)?;
    Ok((emp2.at(&[])?.0, t1, t2))
}

/// `update_amps(cc, t1, t2, eris)` — `rccsd.py:43-144`, the `cc2 = False`
/// branch (`:113-138`).
///
/// # `cc2` is not ported
///
/// `:96-112` is a second branch selected by `cc.cc2`, which
/// `pbc/cc/ccsd.py`'s `RCCSD` never sets and `ccsd.CCSD.__init__` defaults to
/// `False`. It is a different approximation, not a different implementation of
/// this one, and porting it would ship arithmetic no caller in this workspace
/// reaches. [`update_amps_cc2_refusal`] names the line.
///
/// # Errors
/// Propagates every contraction.
#[allow(clippy::too_many_lines)]
pub fn update_amps(
    t1: &ZArr,
    t2: &ZArr,
    eris: &ChemistsErisZ,
    level_shift: f64,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let (nocc, nvir) = (eris.nocc, eris.nvir);
    let mo_e_o: Vec<f64> = eris.mo_energy[..nocc].to_vec();
    let mo_e_v: Vec<f64> = eris.mo_energy[nocc..]
        .iter()
        .map(|x| x + level_shift)
        .collect();

    let fov = eris.fov()?;

    let mut foo_i = cc_foo(t1, t2, eris)?;
    let mut fvv_i = cc_fvv(t1, t2, eris)?;
    let fov_i = cc_fov(t1, eris)?;

    // `:61-62` — move the energy terms to the other side.
    for (i, e) in mo_e_o.iter().enumerate() {
        foo_i.data_mut().re[i * nocc + i] -= e;
    }
    for (a, e) in mo_e_v.iter().enumerate() {
        fvv_i.data_mut().re[a * nvir + a] -= e;
    }

    // ---- T1 (`:64-83`)
    let mut t1new = einsum_scaled("kc,ka,ic->ia", &[&fov, t1, t1], -2.0)?;
    t1new.add_assign(&einsum("ac,ic->ia", &[&fvv_i, t1])?)?;
    t1new.sub_assign(&einsum("ki,ka->ia", &[&foo_i, t1])?)?;
    t1new.add_assign(&einsum_scaled("kc,kica->ia", &[&fov_i, t2], 2.0)?)?;
    t1new.sub_assign(&einsum("kc,ikca->ia", &[&fov_i, t2])?)?;
    t1new.add_assign(&einsum("kc,ic,ka->ia", &[&fov_i, t1, t1])?)?;
    t1new.add_assign(&fov.conj())?;
    t1new.add_assign(&einsum_scaled("kcai,kc->ia", &[&eris.ovvo, t1], 2.0)?)?;
    t1new.sub_assign(&einsum("kiac,kc->ia", &[&eris.oovv, t1])?)?;
    let ovvv = &eris.ovvv;
    t1new.add_assign(&einsum_scaled("kdac,ikcd->ia", &[ovvv, t2], 2.0)?)?;
    t1new.sub_assign(&einsum("kcad,ikcd->ia", &[ovvv, t2])?)?;
    t1new.add_assign(&einsum_scaled("kdac,kd,ic->ia", &[ovvv, t1, t1], 2.0)?)?;
    t1new.sub_assign(&einsum("kcad,kd,ic->ia", &[ovvv, t1, t1])?)?;
    let ovoo = &eris.ovoo;
    t1new.add_assign(&einsum_scaled("lcki,klac->ia", &[ovoo, t2], -2.0)?)?;
    t1new.add_assign(&einsum("kcli,klac->ia", &[ovoo, t2])?)?;
    t1new.add_assign(&einsum_scaled("lcki,lc,ka->ia", &[ovoo, t1, t1], -2.0)?)?;
    t1new.add_assign(&einsum("kcli,lc,ka->ia", &[ovoo, t1, t1])?)?;

    // ---- T2 (`:85-138`)
    // `:86-90` — `tmp2 = -oovv·t1 + ovvv*.transpose(1,3,0,2)`, then
    // `t2new = tmp + tmpᵀ` with the `(ij|ab)` swap.
    let mut tmp2 = einsum_scaled("kibc,ka->abic", &[&eris.oovv, t1], -1.0)?;
    tmp2.add_assign(&ovvv.conj().transpose(&[1, 3, 0, 2])?)?;
    let tmp = einsum("abic,jc->ijab", &[&tmp2, t1])?;
    let mut t2new = tmp.clone();
    t2new.add_assign(&tmp.transpose(&[1, 0, 3, 2])?)?;

    let mut tmp2 = einsum("kcai,jc->akij", &[&eris.ovvo, t1])?;
    tmp2.add_assign(&ovoo.transpose(&[1, 3, 0, 2])?.conj())?;
    let tmp = einsum("akij,kb->ijab", &[&tmp2, t1])?;
    t2new.sub_assign(&tmp)?;
    t2new.sub_assign(&tmp.transpose(&[1, 0, 3, 2])?)?;
    t2new.add_assign(&eris.ovov.conj().transpose(&[0, 2, 1, 3])?)?;

    let mut loo_i = loo(t1, t2, eris)?;
    let mut lvv_i = lvv(t1, t2, eris)?;
    for (i, e) in mo_e_o.iter().enumerate() {
        loo_i.data_mut().re[i * nocc + i] -= e;
    }
    for (a, e) in mo_e_v.iter().enumerate() {
        lvv_i.data_mut().re[a * nvir + a] -= e;
    }

    let woooo = cc_woooo(t1, t2, eris)?;
    let wvoov = cc_wvoov(t1, t2, eris)?;
    let wvovo = cc_wvovo(t1, t2, eris)?;
    let wvvvv = cc_wvvvv(t1, eris)?;

    let mut tau = einsum("ia,jb->ijab", &[t1, t1])?;
    tau.add_assign(t2)?;
    t2new.add_assign(&einsum("klij,klab->ijab", &[&woooo, &tau])?)?;
    t2new.add_assign(&einsum("abcd,ijcd->ijab", &[&wvvvv, &tau])?)?;

    let tmp = einsum("ac,ijcb->ijab", &[&lvv_i, t2])?;
    t2new.add_assign(&tmp)?;
    t2new.add_assign(&tmp.transpose(&[1, 0, 3, 2])?)?;
    let tmp = einsum("ki,kjab->ijab", &[&loo_i, t2])?;
    t2new.sub_assign(&tmp)?;
    t2new.sub_assign(&tmp.transpose(&[1, 0, 3, 2])?)?;

    let mut tmp = einsum_scaled("akic,kjcb->ijab", &[&wvoov, t2], 2.0)?;
    tmp.sub_assign(&einsum("akci,kjcb->ijab", &[&wvovo, t2])?)?;
    t2new.add_assign(&tmp)?;
    t2new.add_assign(&tmp.transpose(&[1, 0, 3, 2])?)?;
    let tmp = einsum("akic,kjbc->ijab", &[&wvoov, t2])?;
    t2new.sub_assign(&tmp)?;
    t2new.sub_assign(&tmp.transpose(&[1, 0, 3, 2])?)?;
    let tmp = einsum("bkci,kjac->ijab", &[&wvovo, t2])?;
    t2new.sub_assign(&tmp)?;
    t2new.sub_assign(&tmp.transpose(&[1, 0, 3, 2])?)?;

    // `:140-143` — the denominators.
    let eia: Vec<f64> = (0..nocc)
        .flat_map(|i| (0..nvir).map(move |a| (i, a)))
        .map(|(i, a)| mo_e_o[i] - mo_e_v[a])
        .collect();
    for (f, d) in eia.iter().enumerate() {
        t1new.data_mut().re[f] /= d;
        t1new.data_mut().im[f] /= d;
    }
    let mut f = 0;
    for i in 0..nocc {
        for j in 0..nocc {
            for a in 0..nvir {
                for b in 0..nvir {
                    let d = eia[i * nvir + a] + eia[j * nvir + b];
                    t2new.data_mut().re[f] /= d;
                    t2new.data_mut().im[f] /= d;
                    f += 1;
                }
            }
        }
    }
    Ok((t1new, t2new))
}

/// The refusal `cc2 = True` gets — see [`update_amps`].
#[must_use]
pub fn update_amps_cc2_refusal() -> PbcCcError {
    PbcCcError::NotImplementedUpstream {
        upstream: "cc/rccsd.py:96",
        what: "the cc2 branch of update_amps is a different approximation; \
               pbc/cc/ccsd.py's RCCSD never sets cc.cc2",
    }
}

/// What [`kernel`] returns.
#[derive(Debug, Clone)]
pub struct RccsdResult {
    pub converged: bool,
    pub e_corr: f64,
    pub emp2: f64,
    pub t1: ZArr,
    pub t2: ZArr,
    pub niter: usize,
    /// The largest `|Im(E_corr)|` seen. `energy` (`:160`) warns above `1e-4`.
    pub max_imag: f64,
}

/// Options for [`kernel`], with `ccsd.CCSDBase`'s own defaults
/// (`pyscf/cc/ccsd.py:915-930`).
#[derive(Debug, Clone, Copy)]
pub struct RccsdOpts {
    /// `max_cycle` (`:920` — 50).
    pub max_cycle: usize,
    /// `conv_tol` (`:917` — 1e-7).
    pub conv_tol: f64,
    /// `conv_tol_normt` (`:918` — 1e-5).
    pub conv_tol_normt: f64,
    /// `level_shift`, added to the VIRTUAL orbital energies inside
    /// [`update_amps`] (`:49`).
    pub level_shift: f64,
    /// `diis` on/off.
    pub diis: bool,
    /// `diis_space` (`:926` — 6, not SCF's 8).
    pub diis_space: usize,
    /// `diis_start_cycle`.
    pub diis_start_cycle: usize,
}

impl Default for RccsdOpts {
    fn default() -> Self {
        Self {
            max_cycle: 50,
            conv_tol: 1e-7,
            conv_tol_normt: 1e-5,
            level_shift: 0.0,
            diis: true,
            diis_space: 6,
            diis_start_cycle: 0,
        }
    }
}

/// `ccsd.CCSDBase.ccsd(self, t1, t2, eris)` driven with this module's
/// `update_amps` and `energy` — `rccsd.RCCSD.ccsd` (`:173-189`) delegates to
/// it.
///
/// The DIIS is [`crate::kccsd_rhf`]'s `KAmplitudeSubspace`, which packs
/// `[re…, im…]` and reuses the Phase-3 `pyscf-diis` body; a real linear
/// combination of complex iterates is exactly the same operation applied to
/// each plane.
///
/// # Errors
/// Propagates every amplitude update and the DIIS solve. An unconverged run is
/// RETURNED with `converged = false`, not refused — upstream's behaviour.
pub fn kernel(eris: &ChemistsErisZ, opts: &RccsdOpts) -> Result<RccsdResult, PbcCcError> {
    use pyscf_diis::Diis;

    let (emp2, mut t1, mut t2) = init_amps(eris)?;
    let (mut e_corr, mut max_imag) = energy(&t1, &t2, eris)?;
    max_imag = max_imag.abs();
    let mut converged = false;
    let mut niter = 0;
    let mut diis = if opts.diis {
        Some(Diis::<crate::kccsd_rhf::KAmplitudeSubspace>::new(
            opts.diis_space,
        ))
    } else {
        None
    };

    for istep in 0..opts.max_cycle {
        niter = istep + 1;
        let (t1new, t2new) = update_amps(&t1, &t2, eris, opts.level_shift)?;
        let cur = crate::kccsd_rhf::KAmplitudeSubspace::from_amplitudes(&t1new, &t2new);
        let prev = crate::kccsd_rhf::KAmplitudeSubspace::from_amplitudes(&t1, &t2);
        let res = cur.residual(&prev);
        let normt = pyscf_algebra::oracle_dot(&res, &res).sqrt();
        t1 = t1new;
        t2 = t2new;

        if let Some(stack) = diis.as_mut()
            && istep >= opts.diis_start_cycle
        {
            let cur = crate::kccsd_rhf::KAmplitudeSubspace::from_amplitudes(&t1, &t2);
            let err = cur.residual(&prev);
            let extrap = stack
                .extrapolate(cur, err)
                .map_err(|e| PbcCcError::Algebra(format!("amplitude DIIS: {e}")))?;
            let (a, b) = extrap.to_amplitudes(&t1, &t2);
            t1 = a;
            t2 = b;
        }

        let eold = e_corr;
        let (e, im) = energy(&t1, &t2, eris)?;
        e_corr = e;
        max_imag = max_imag.max(im.abs());
        if (e_corr - eold).abs() < opts.conv_tol && normt < opts.conv_tol_normt {
            converged = true;
            break;
        }
    }
    Ok(RccsdResult {
        converged,
        e_corr,
        emp2,
        t1,
        t2,
        niter,
        max_imag,
    })
}

/// The planar-complex `CTensor` behind a `ZArr`, for callers that want to hand
/// amplitudes to another crate.
#[must_use]
pub fn as_ctensor(z: &ZArr) -> CTensor {
    z.data().clone()
}

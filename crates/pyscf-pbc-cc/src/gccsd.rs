//! `gccsd` — the MOLECULAR, complex-capable spin-orbital CCSD
//! (`pyscf/cc/gccsd.py` and the six `cc_*` of `pyscf/cc/gintermediates.py`).
//!
//! # Why this lives here, and what it is NOT
//!
//! Same reason as [`crate::rccsd`]: `crates/pyscf-ccsd` is `f64` by
//! construction, and `pyscf_ccsd::gccsd` is a deliberately PARTIAL port whose
//! own doc says so — it declares the `<pq||rs>` convention and leaves the
//! arithmetic to whatever needs it. `pbc/cc/ccsd.py:94`'s `GCCSD` needs it,
//! and the planar-complex substrate is in this crate.
//!
//! **This is not [`crate::kccsd`] at `nkpts = 1`.** It would give the same
//! numbers, and `oracle_gamma_ug.rs` measures that it does, but
//! `pbc/cc/kccsd.py:68` replaces the molecular `update_amps` WHOLESALE — the
//! k-point form carries `kconserv` through every contraction and builds ten
//! EOM-shaped intermediates; this one is six `cc_*` and fifty lines. Each
//! function below is transcribed from its own upstream line.
//!
//! # `<pq||rs>`, and the `ovvo` block nothing reads
//!
//! `_PhysicistsERIs` (`:297-312`) declares seven blocks. `update_amps` and
//! the six intermediates read only SIX of them — `oooo`, `ooov`, `oovv`,
//! `ovov`, `ovvv`, `vvvv`. `cc_Wovvo` (`gintermediates.py:76-77`) derives what
//! it needs from `ovov` and `ooov` by antisymmetry (`<ia||bj> = −<ia||jb>`)
//! rather than reading `eris.ovvo`, so `ovvo` is stored for API shape only and
//! is not carried here.

use crate::error::PbcCcError;
use crate::zarr::{ZArr, einsum, einsum_scaled};

/// `_PhysicistsERIs` (`gccsd.py:297-312`) — the ANTISYMMETRISED
/// `<pq||rs> = <pq|rs> − <pq|sr>` blocks, complex.
#[derive(Debug, Clone)]
pub struct PhysicistsErisZ {
    pub nocc: usize,
    pub nvir: usize,
    /// `[nmo, nmo]`, the MO-basis Fock matrix.
    pub fock: ZArr,
    /// `mo_energy`. `_common_init_:341` takes it as `fock.diagonal().real`;
    /// the Γ shim then applies the Madelung re-add to the occupied block
    /// (`pbc/cc/ccsd.py:142`).
    pub mo_energy: Vec<f64>,
    pub oooo: ZArr,
    pub ooov: ZArr,
    pub oovv: ZArr,
    pub ovov: ZArr,
    pub ovvv: ZArr,
    pub vvvv: ZArr,
}

impl PhysicistsErisZ {
    /// `_make_eris_incore(mycc, mo_coeff, ao2mofn)` (`:344-386`) from a full
    /// `[nmo]⁴` CHEMISTS' tensor `(pq|rs)`.
    ///
    /// `:378` is the whole conversion:
    ///
    /// ```text
    /// eri = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)
    /// ```
    ///
    /// — chemist to physicist, then antisymmetrised. This project has paid
    /// once for confusing the two (14-05's `decompose_j2c`,
    /// `16-CONTEXT §3.4`), so the operation is written out rather than folded
    /// into the slicing.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] if `eri` is not `nmo⁴` long or `fock` is not
    /// `[nmo, nmo]`.
    pub fn from_full_chemists(
        eri: &ZArr,
        fock: ZArr,
        mo_energy: Vec<f64>,
        nocc: usize,
    ) -> Result<Self, PbcCcError> {
        let nmo = mo_energy.len();
        let nvir = nmo - nocc;
        if eri.len() != nmo * nmo * nmo * nmo {
            return Err(PbcCcError::Shape(format!(
                "gccsd::_make_eris_incore: eri has {} elements, expected nmo^4 = {}",
                eri.len(),
                nmo * nmo * nmo * nmo
            )));
        }
        if fock.shape() != [nmo, nmo] {
            return Err(PbcCcError::Shape(format!(
                "gccsd::_make_eris_incore: fock is {:?}, expected [{nmo}, {nmo}]",
                fock.shape()
            )));
        }
        let e = eri.reshape(&[nmo, nmo, nmo, nmo])?;
        let mut anti = e.transpose(&[0, 2, 1, 3])?;
        anti.sub_assign(&e.transpose(&[0, 2, 3, 1])?)?;
        let o = (0, nocc);
        let v = (nocc, nmo);
        Ok(Self {
            nocc,
            nvir,
            fock,
            mo_energy,
            oooo: anti.slice_axes(&[o, o, o, o])?,
            ooov: anti.slice_axes(&[o, o, o, v])?,
            oovv: anti.slice_axes(&[o, o, v, v])?,
            ovov: anti.slice_axes(&[o, v, o, v])?,
            ovvv: anti.slice_axes(&[o, v, v, v])?,
            vvvv: anti.slice_axes(&[v, v, v, v])?,
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
// `pyscf/cc/gintermediates.py` — the six `cc_*`
// ---------------------------------------------------------------------------

/// `make_tau(t2, t1a, t1b, fac)` — `gintermediates.py:27-32`.
///
/// The antisymmetrised `t1·t1` product: `t1t1 − t1t1ᵀ_{ij}`, then that minus
/// its `ab` transpose, then `+ t2`.
///
/// # Errors
/// Propagates every contraction.
pub fn make_tau(t2: &ZArr, t1a: &ZArr, t1b: &ZArr, fac: f64) -> Result<ZArr, PbcCcError> {
    let mut t1t1 = einsum_scaled("ia,jb->ijab", &[t1a, t1b], fac * 0.5)?;
    let sw = t1t1.transpose(&[1, 0, 2, 3])?;
    t1t1.sub_assign(&sw)?;
    let mut tau = t1t1.clone();
    tau.sub_assign(&t1t1.transpose(&[0, 1, 3, 2])?)?;
    tau.add_assign(t2)?;
    Ok(tau)
}

/// `cc_Fvv(t1, t2, eris)` — `:34-43`.
///
/// # Errors
/// Propagates every contraction.
pub fn cc_fvv(t1: &ZArr, t2: &ZArr, eris: &PhysicistsErisZ) -> Result<ZArr, PbcCcError> {
    // `:38` — `eris_vovv = ovvv.transpose(1,0,3,2)`.
    let vovv = eris.ovvv.transpose(&[1, 0, 3, 2])?;
    let tau_tilde = make_tau(t2, t1, t1, 0.5)?;
    let mut f = eris.fvv()?;
    f.sub_assign(&einsum_scaled("me,ma->ae", &[&eris.fov()?, t1], 0.5)?)?;
    f.add_assign(&einsum("mf,amef->ae", &[t1, &vovv])?)?;
    f.sub_assign(&einsum_scaled(
        "mnaf,mnef->ae",
        &[&tau_tilde, &eris.oovv],
        0.5,
    )?)?;
    Ok(f)
}

/// `cc_Foo(t1, t2, eris)` — `:45-53`.
///
/// # Errors
/// Propagates every contraction.
pub fn cc_foo(t1: &ZArr, t2: &ZArr, eris: &PhysicistsErisZ) -> Result<ZArr, PbcCcError> {
    let tau_tilde = make_tau(t2, t1, t1, 0.5)?;
    let mut f = eris.foo()?;
    f.add_assign(&einsum_scaled("me,ie->mi", &[&eris.fov()?, t1], 0.5)?)?;
    f.add_assign(&einsum("ne,mnie->mi", &[t1, &eris.ooov])?)?;
    f.add_assign(&einsum_scaled(
        "inef,mnef->mi",
        &[&tau_tilde, &eris.oovv],
        0.5,
    )?)?;
    Ok(f)
}

/// `cc_Fov(t1, t2, eris)` — `:55-59`. `t2` is unread upstream too.
///
/// # Errors
/// Propagates every contraction.
pub fn cc_fov(t1: &ZArr, eris: &PhysicistsErisZ) -> Result<ZArr, PbcCcError> {
    let mut f = eris.fov()?;
    f.add_assign(&einsum("nf,mnef->me", &[t1, &eris.oovv])?)?;
    Ok(f)
}

/// `cc_Woooo(t1, t2, eris)` — `:61-66`.
///
/// # Errors
/// Propagates every contraction.
pub fn cc_woooo(t1: &ZArr, t2: &ZArr, eris: &PhysicistsErisZ) -> Result<ZArr, PbcCcError> {
    let tau = make_tau(t2, t1, t1, 1.0)?;
    let tmp = einsum("je,mnie->mnij", &[t1, &eris.ooov])?;
    let mut w = eris.oooo.clone();
    w.add_assign(&tmp)?;
    w.sub_assign(&tmp.transpose(&[0, 1, 3, 2])?)?;
    w.add_assign(&einsum_scaled(
        "ijef,mnef->mnij",
        &[&tau, &eris.oovv],
        0.25,
    )?)?;
    Ok(w)
}

/// `cc_Wvvvv(t1, t2, eris)` — `:68-74`.
///
/// # Errors
/// Propagates every contraction.
pub fn cc_wvvvv(t1: &ZArr, t2: &ZArr, eris: &PhysicistsErisZ) -> Result<ZArr, PbcCcError> {
    let tau = make_tau(t2, t1, t1, 1.0)?;
    let tmp = einsum("mb,mafe->bafe", &[t1, &eris.ovvv])?;
    let mut w = eris.vvvv.clone();
    w.sub_assign(&tmp)?;
    w.add_assign(&tmp.transpose(&[1, 0, 2, 3])?)?;
    w.add_assign(&einsum_scaled(
        "mnab,mnef->abef",
        &[&tau, &eris.oovv],
        0.25,
    )?)?;
    Ok(w)
}

/// `cc_Wovvo(t1, t2, eris)` — `:76-85`.
///
/// # The two derived blocks
///
/// `:77-78` builds `ovvo` and `oovo` from `ovov` and `ooov` by antisymmetry —
/// `<ia||bj> = −<ia||jb>` and `<ij||ak> = −<ij||ka>` — rather than reading
/// `eris.ovvo`. Reproduced, so no permutational identity is assumed anywhere
/// this port does not state it.
///
/// # Errors
/// Propagates every contraction.
pub fn cc_wovvo(t1: &ZArr, t2: &ZArr, eris: &PhysicistsErisZ) -> Result<ZArr, PbcCcError> {
    let mut ovvo = eris.ovov.transpose(&[0, 1, 3, 2])?;
    ovvo.scale(-1.0);
    let mut oovo = eris.ooov.transpose(&[0, 1, 3, 2])?;
    oovo.scale(-1.0);

    let mut w = einsum("jf,mbef->mbej", &[t1, &eris.ovvv])?;
    w.sub_assign(&einsum("nb,mnej->mbej", &[t1, &oovo])?)?;
    w.sub_assign(&einsum_scaled("jnfb,mnef->mbej", &[t2, &eris.oovv], 0.5)?)?;
    w.sub_assign(&einsum("jf,nb,mnef->mbej", &[t1, t1, &eris.oovv])?)?;
    w.add_assign(&ovvo)?;
    Ok(w)
}

// ---------------------------------------------------------------------------
// `pyscf/cc/gccsd.py`
// ---------------------------------------------------------------------------

/// `energy(cc, t1, t2, eris)` — `gccsd.py:95-106`.
///
/// Returns `(re, im)`; `:104` warns above `|Im| > 1e-4` and returns `e.real`
/// regardless, which is what [`kernel`] does with it.
///
/// # Errors
/// Propagates every contraction.
pub fn energy(t1: &ZArr, t2: &ZArr, eris: &PhysicistsErisZ) -> Result<(f64, f64), PbcCcError> {
    let mut e = einsum("ia,ia->", &[&eris.fov()?, t1])?;
    e.add_assign(&einsum_scaled("ijab,ijab->", &[t2, &eris.oovv], 0.25)?)?;
    e.add_assign(&einsum_scaled("ia,jb,ijab->", &[t1, t1, &eris.oovv], 0.5)?)?;
    e.at(&[])
}

/// `GCCSD.init_amps(eris)` — `gccsd.py:122-136`.
///
/// Returns `(emp2, t1, t2)`. Note `t2 = oovv.conj() / eijab` and
/// `emp2 = 0.25·<t2, oovv>` — the spin-orbital factors, not the spin-adapted
/// `2·… − …` of [`crate::rccsd::init_amps`].
///
/// # Errors
/// Propagates every contraction.
pub fn init_amps(eris: &PhysicistsErisZ) -> Result<(f64, ZArr, ZArr), PbcCcError> {
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

    let mut t2 = eris.oovv.conj();
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
    let emp2 = einsum_scaled("ijab,ijab->", &[&t2, &eris.oovv], 0.25)?;
    Ok((emp2.at(&[])?.0, t1, t2))
}

/// `update_amps(cc, t1, t2, eris)` — `gccsd.py:36-92`.
///
/// # Errors
/// Propagates every contraction.
#[allow(clippy::too_many_lines)]
pub fn update_amps(
    t1: &ZArr,
    t2: &ZArr,
    eris: &PhysicistsErisZ,
    level_shift: f64,
) -> Result<(ZArr, ZArr), PbcCcError> {
    let (nocc, nvir) = (eris.nocc, eris.nvir);
    let mo_e_o: Vec<f64> = eris.mo_energy[..nocc].to_vec();
    let mo_e_v: Vec<f64> = eris.mo_energy[nocc..]
        .iter()
        .map(|x| x + level_shift)
        .collect();
    let fov = eris.fov()?;

    let tau = make_tau(t2, t1, t1, 1.0)?;
    let mut f_vv = cc_fvv(t1, t2, eris)?;
    let mut f_oo = cc_foo(t1, t2, eris)?;
    let fov_i = cc_fov(t1, eris)?;
    let woooo = cc_woooo(t1, t2, eris)?;
    let wvvvv = cc_wvvvv(t1, t2, eris)?;
    let wovvo = cc_wovvo(t1, t2, eris)?;

    // `:56-57` — move the energy terms to the other side.
    for (a, e) in mo_e_v.iter().enumerate() {
        f_vv.data_mut().re[a * nvir + a] -= e;
    }
    for (i, e) in mo_e_o.iter().enumerate() {
        f_oo.data_mut().re[i * nocc + i] -= e;
    }

    // ---- T1 (`:59-67`)
    let mut t1new = einsum("ie,ae->ia", &[t1, &f_vv])?;
    t1new.sub_assign(&einsum("ma,mi->ia", &[t1, &f_oo])?)?;
    t1new.add_assign(&einsum("imae,me->ia", &[t2, &fov_i])?)?;
    t1new.sub_assign(&einsum("nf,naif->ia", &[t1, &eris.ovov])?)?;
    t1new.add_assign(&einsum_scaled("imef,maef->ia", &[t2, &eris.ovvv], -0.5)?)?;
    t1new.add_assign(&einsum_scaled("mnae,mnie->ia", &[t2, &eris.ooov], -0.5)?)?;
    t1new.add_assign(&fov.conj())?;

    // ---- T2 (`:69-87`)
    let mut ftmp = f_vv.clone();
    ftmp.sub_assign(&einsum_scaled("mb,me->be", &[t1, &fov_i], 0.5)?)?;
    let tmp = einsum("ijae,be->ijab", &[t2, &ftmp])?;
    let mut t2new = tmp.clone();
    t2new.sub_assign(&tmp.transpose(&[0, 1, 3, 2])?)?;

    let mut ftmp = f_oo.clone();
    ftmp.add_assign(&einsum_scaled("je,me->mj", &[t1, &fov_i], 0.5)?)?;
    let tmp = einsum("imab,mj->ijab", &[t2, &ftmp])?;
    t2new.sub_assign(&tmp)?;
    t2new.add_assign(&tmp.transpose(&[1, 0, 2, 3])?)?;

    t2new.add_assign(&eris.oovv.conj())?;
    t2new.add_assign(&einsum_scaled("mnab,mnij->ijab", &[&tau, &woooo], 0.5)?)?;
    t2new.add_assign(&einsum_scaled("ijef,abef->ijab", &[&tau, &wvvvv], 0.5)?)?;

    // `:79-83` — the `P(ij)P(ab)` block. `:80`'s `-= -einsum(...)` is a DOUBLE
    // negation upstream writes literally; it is an addition.
    let mut tmp = einsum("imae,mbej->ijab", &[t2, &wovvo])?;
    tmp.add_assign(&einsum("ie,ma,mbje->ijab", &[t1, t1, &eris.ovov])?)?;
    let a = tmp.transpose(&[1, 0, 2, 3])?;
    tmp.sub_assign(&a)?;
    let a = tmp.transpose(&[0, 1, 3, 2])?;
    tmp.sub_assign(&a)?;
    t2new.add_assign(&tmp)?;

    let tmp = einsum("ie,jeba->ijab", &[t1, &eris.ovvv.conj()])?;
    t2new.add_assign(&tmp)?;
    t2new.sub_assign(&tmp.transpose(&[1, 0, 2, 3])?)?;
    let tmp = einsum("ma,ijmb->ijab", &[t1, &eris.ooov.conj()])?;
    t2new.sub_assign(&tmp)?;
    t2new.add_assign(&tmp.transpose(&[0, 1, 3, 2])?)?;

    // `:89-92` — the denominators.
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

/// What [`kernel`] returns — the same shape as [`crate::rccsd::RccsdResult`].
#[derive(Debug, Clone)]
pub struct GccsdResult {
    pub converged: bool,
    pub e_corr: f64,
    pub emp2: f64,
    pub t1: ZArr,
    pub t2: ZArr,
    pub niter: usize,
    /// The largest `|Im(E_corr)|` seen. `energy` (`:104`) warns above `1e-4`.
    pub max_imag: f64,
}

/// `ccsd.CCSDBase.ccsd` driven with this module's `update_amps` and `energy`.
///
/// `GCCSD`'s own defaults differ from the restricted ones: `conv_tol = 1e-7`
/// but `conv_tol_normt = 1e-6`, not `1e-5` (`gccsd.py:116-117`). That is
/// carried by [`crate::rccsd::RccsdOpts`]'s caller, not silently here.
///
/// # Errors
/// Propagates every amplitude update and the DIIS solve.
pub fn kernel(
    eris: &PhysicistsErisZ,
    opts: &crate::rccsd::RccsdOpts,
) -> Result<GccsdResult, PbcCcError> {
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
    Ok(GccsdResult {
        converged,
        e_corr,
        emp2,
        t1,
        t2,
        niter,
        max_imag,
    })
}

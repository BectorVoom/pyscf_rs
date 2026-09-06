//! `ccsd` — the Γ-point (single-k-point) `pbc/cc/ccsd.py` shim.
//!
//! # What the shim actually is
//!
//! `pbc/cc/ccsd.py` is 157 lines and holds no CC equations at all. Its three
//! classes subclass the MOLECULAR `rccsd.RCCSD` / `uccsd.UCCSD` /
//! `gccsd.GCCSD` and override exactly two methods each:
//!
//! * `ccsd(..., mbpt2=False)` (`:25-34`) — a `warn_pbc2d_eri` call, an
//!   `mbpt2` short-circuit that returns MP2 amplitudes instead of running the
//!   CC iteration, and otherwise the molecular kernel unchanged;
//! * `ao2mo(mo_coeff)` (`:36-59`) — the periodic AO→MO transform, with the
//!   Fock built under `exxdiv = None` and the Madelung constant added back to
//!   the OCCUPIED orbital energies afterwards (`_adjust_occ`, `:146-150`).
//!
//! That second point is `16-CONTEXT §3.5`, and this port already implements
//! both halves: [`crate::keris::KEris::build_fock`] builds the Fock under
//! `exxdiv = None` and applies [`crate::keris::adjust_occ`], whose doc cites
//! `pbc/cc/ccsd.py:146-150` directly. The shim here is therefore the
//! single-k-point face of machinery this crate already had, plus the
//! molecular complex CCSD in [`crate::rccsd`] — the piece `16-VERIFICATION
//! §6.1` recorded as blocking it.
//!
//! # The three classes, and where each one's arithmetic lives
//!
//! | shim | molecular base | this crate |
//! |---|---|---|
//! | [`GammaRccsd`] (`:24-59`) | `cc/rccsd.py` | [`crate::rccsd`] |
//! | [`GammaUccsd`] (`:61-92`) | `cc/uccsd.py` | [`crate::uccsd`] |
//! | [`GammaGccsd`] (`:94-144`) | `cc/gccsd.py` | [`crate::gccsd`] |
//!
//! # `mbpt2=True` is Γ-only for TWO of the three
//!
//! Each `ccsd(..., mbpt2=True)` constructs the matching `pbc.mp` class, and
//! those do not agree with each other: `RMP2.__init__` (`pbc/mp/mp2.py:21-23`)
//! and `UMP2.__init__` (`:35-37`) both open with
//! `if abs(mf.kpt).max() > 1e-9: raise NotImplementedError`, and
//! `GMP2.__init__` (`:47-51`) does NOT. So `GammaGccsd::mbpt2` runs at a
//! shifted k-point where its two siblings refuse. Measured, and reproduced.
//!
//! # The `_ERI` block set differs from the k-point one
//!
//! `rccsd._make_eris_incore` slices SEVEN chemists' blocks out of one full
//! `[nmo]⁴` tensor. [`crate::keris::KEris`] holds a different seven. They are
//! the same integrals under different index orders, and this module builds the
//! molecular set from `ao2mo_7d`'s single `[0,0,0]` block rather than
//! re-deriving it from `KEris`, so no permutational-symmetry assumption enters.

use pyscf_algebra::CTensor;
use pyscf_pbc_df::PeriodicDf;
use pyscf_pbc_mp::PaddedMos;

use crate::error::PbcCcError;
use crate::rccsd::{ChemistsErisZ, RccsdOpts, RccsdResult};
use crate::zarr::ZArr;

fn shape(m: impl Into<String>) -> PbcCcError {
    PbcCcError::Shape(m.into())
}

/// `class RCCSD(rccsd.RCCSD)` (`pbc/cc/ccsd.py:24-59`) at a SINGLE k-point.
pub struct GammaRccsd<'a> {
    with_df: &'a dyn PeriodicDf,
    padded: PaddedMos,
    dm: Vec<CTensor>,
    /// `mf.e_tot`.
    pub e_hf: f64,
    /// Whether the reference mean field converged.
    pub converged: bool,
    /// The CC iteration's knobs.
    pub opts: RccsdOpts,
    /// The `_ERIS` build's knobs — `keep_exxdiv` MUST stay `false`, which is
    /// `:47`'s `with lib.temporary_env(self._scf, exxdiv=None)`.
    pub eris_opts: crate::keris::KErisOpts,
}

impl<'a> GammaRccsd<'a> {
    /// `RCCSD(mf)` on a converged single-k-point restricted mean field.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] if the mean field is not one restricted channel
    /// over exactly one k-point.
    pub fn new(
        scf: &pyscf_pbc_scf::KScfResult,
        with_df: &'a dyn PeriodicDf,
    ) -> Result<Self, PbcCcError> {
        if with_df.kpts().len() != 1 {
            return Err(shape(format!(
                "pbc/cc/ccsd.py's RCCSD is the SINGLE-k-point shim; \
                 with_df carries {} k-points — use KRCCSD",
                with_df.kpts().len()
            )));
        }
        if scf.nset != 1 || scf.nkpts != 1 {
            return Err(shape(
                "the Gamma-point RCCSD needs one restricted SCF channel at one k-point",
            ));
        }
        let cell = with_df.cell();
        let nao = cell.mol.nao_nr;
        let mf = pyscf_pbc_mp::spin_block(scf, 0).map_err(|e| shape(format!("spin_block: {e}")))?;
        let raw: Result<Vec<pyscf_pbc_df::MoCoeff>, _> = mf
            .mo_coeff
            .iter()
            .zip(mf.mo_occ)
            .map(|(c, occ)| pyscf_pbc_mp::mo_coeff_from_kscf(c, nao, occ.len()))
            .collect();
        let raw = raw.map_err(|e| shape(format!("mo_coeff_from_kscf: {e}")))?;
        let frozen = pyscf_pbc_mp::FrozenK::default();
        let padded = pyscf_pbc_mp::add_padding(&raw, mf.mo_energy, mf.mo_occ, &frozen)
            .map_err(|e| shape(format!("add_padding: {e}")))?;
        let dm = pyscf_pbc_scf::krdm::make_rdm1(mf.mo_coeff, mf.mo_occ, nao);
        Ok(Self {
            with_df,
            padded,
            dm,
            e_hf: scf.e_tot,
            converged: scf.converged,
            opts: RccsdOpts::default(),
            eris_opts: crate::keris::KErisOpts::default(),
        })
    }

    /// `RCCSD.ao2mo(mo_coeff)` (`:36-59`).
    ///
    /// Two things happen that a molecular `_make_eris_incore` does not do, and
    /// both are `16-CONTEXT §3.5`:
    ///
    /// 1. the Fock is built with `exxdiv = None` (`:47`), so the HF exchange
    ///    divergence correction is OUT of `eris.fock`;
    /// 2. the Madelung constant is then added back to the occupied orbital
    ///    energies alone (`:58`, `_adjust_occ` at `:146-150`). Upstream's own
    ///    comment says why: without it "MP2 energy may be largely off the
    ///    correct value" for low-dimensional systems whose occupied and
    ///    virtual energies overlap.
    ///
    /// [`crate::keris::KEris::build_fock`] does both; this takes its `fock`
    /// and `mo_energy` and pairs them with the full `[nmo]⁴` chemists' tensor.
    ///
    /// # Errors
    /// Propagates the Fock build, the AO→MO transform and every shape check.
    pub fn ao2mo(&self) -> Result<ChemistsErisZ, PbcCcError> {
        let cell = self.with_df.cell();
        let (fock, mo_energy, _madelung) = crate::keris::KEris::build_fock(
            cell,
            self.with_df,
            &self.padded,
            &self.dm,
            self.eris_opts,
        )?;
        let nmo = self.padded.nmo;
        let nocc = self.padded.nocc;
        // One k-point: `fock` is `[1, nmo, nmo]` and `mo_energy` has one row.
        let fock = fock.slice_leading(&[0])?;
        let mo_energy = mo_energy
            .into_iter()
            .next()
            .ok_or_else(|| shape("build_fock produced no mo_energy"))?;

        // `ao2mofn = mp.mp2._gen_ao2mofn(self._scf)` (`:38`) — the periodic
        // transform. At ONE k-point `ao2mo_7d` is that transform, and its
        // `1/nkpts` factor is 1.
        let mos: Vec<pyscf_pbc_df::MoCoeff> = self.padded.mo_coeff.clone();
        let e7 = self
            .with_df
            .ao2mo_7d([&mos, &mos, &mos, &mos], 1.0)
            .map_err(|e| shape(format!("ao2mo_7d: {e}")))?;
        let n4 = nmo * nmo * nmo * nmo;
        if e7.data.re.len() != n4 {
            return Err(shape(format!(
                "ao2mo_7d returned {} elements at one k-point, expected nmo^4 = {n4}",
                e7.data.re.len()
            )));
        }
        let eri = ZArr::from_ctensor(&[nmo, nmo, nmo, nmo], e7.data)?;
        ChemistsErisZ::from_full(&eri, fock, mo_energy, nocc)
    }

    /// `RCCSD.ccsd(t1, t2, eris, mbpt2=False)` (`:25-34`) — the CC iteration.
    ///
    /// # Errors
    /// Propagates the transform and the amplitude iteration.
    pub fn kernel(&self) -> Result<RccsdResult, PbcCcError> {
        let eris = self.ao2mo()?;
        self.kernel_with(&eris)
    }

    /// [`GammaRccsd::kernel`] on an already-built `_ERIS`.
    ///
    /// # Errors
    /// Propagates the amplitude iteration.
    pub fn kernel_with(&self, eris: &ChemistsErisZ) -> Result<RccsdResult, PbcCcError> {
        if !self.converged {
            return Err(PbcCcError::NotConverged {
                what: "the reference Gamma-point SCF",
                detail: "RCCSD refuses an unconverged mean field".into(),
            });
        }
        crate::rccsd::kernel(eris, &self.opts)
    }

    /// `RCCSD.ccsd(..., mbpt2=True)` (`:28-33`) — the one-shot MBPT2
    /// short-circuit: `t2` are the MP2 amplitudes and `t1` is ZERO.
    ///
    /// # This is `init_amps`, and that is not a shortcut
    ///
    /// Upstream runs `mp.RMP2(...).kernel(eris=eris)` on the SAME `_ERIS` the
    /// CC would use. `MP2`'s amplitudes and correlation energy are built from
    /// `eris.ovov` and `eris.mo_energy` by the identical expression
    /// `ccsd.CCSDBase.init_amps` uses for its starting guess, so
    /// [`crate::rccsd::init_amps`] IS that kernel — and `oracle_gamma.rs`
    /// gates `e_corr` against upstream's `mbpt2=True` rather than asserting
    /// the identity.
    ///
    /// # It is refused away from Γ, and that is upstream's refusal
    ///
    /// `:29` constructs `mp.RMP2(self._scf, …)`, and `pbc/mp/mp2.py:21-23`
    /// opens with `if abs(mf.kpt).max() > 1e-9: raise NotImplementedError`.
    /// So `mbpt2=True` exists only at Γ even though the CC branch beside it
    /// runs at any single k-point — measured, and reproduced here.
    ///
    /// # Errors
    /// [`PbcCcError::NotImplementedUpstream`] away from Γ; otherwise
    /// propagates the transform.
    pub fn mbpt2(&self) -> Result<(f64, ZArr, ZArr), PbcCcError> {
        let eris = self.ao2mo()?;
        self.mbpt2_with(&eris)
    }

    /// Whether the single k-point this shim runs at is Γ, to
    /// `pbc/mp/mp2.py:22`'s own `1e-9` threshold.
    #[must_use]
    pub fn is_gamma(&self) -> bool {
        self.with_df.kpts()[0].iter().all(|v| v.abs() <= 1e-9)
    }

    /// [`GammaRccsd::mbpt2`] on an already-built `_ERIS`.
    ///
    /// # Errors
    /// [`PbcCcError::NotImplementedUpstream`] away from Γ; otherwise
    /// propagates `init_amps`.
    pub fn mbpt2_with(&self, eris: &ChemistsErisZ) -> Result<(f64, ZArr, ZArr), PbcCcError> {
        if !self.is_gamma() {
            return Err(PbcCcError::NotImplementedUpstream {
                upstream: "pbc/mp/mp2.py:22",
                what: "RMP2.__init__ raises NotImplementedError away from Gamma, so \
                       RCCSD.ccsd(mbpt2=True) is a Gamma-only branch",
            });
        }
        let (emp2, _t1, t2) = crate::rccsd::init_amps(eris)?;
        // `:32` — `self.t1 = numpy.zeros((nocc, nvir))`.
        let t1 = ZArr::zeros(&[eris.nocc, eris.nvir]);
        Ok((emp2, t1, t2))
    }
}

/// `class GCCSD(gccsd.GCCSD)` (`pbc/cc/ccsd.py:94-144`) at a SINGLE k-point.
///
/// # The `ao2mofn` is the four-spin-block sum
///
/// `:107-131` has two branches. Without an `orbspin` tag on the coefficients
/// it is
///
/// ```text
/// eri  = with_df.ao2mo(mo_a, kpt)          # (aa|aa)
/// eri += with_df.ao2mo(mo_b, kpt)          # (bb|bb)
/// eri1 = with_df.ao2mo((mo_a,mo_a,mo_b,mo_b), kpt)
/// eri += eri1; eri += eri1.T               # (aa|bb) + (bb|aa)
/// ```
///
/// and WITH one it transforms `mo_a + mo_b` once and zeroes the spin-forbidden
/// entries. This port takes the first, which is also what
/// `pbc/cc/kccsd.py:577-596` does and what [`crate::kccsd::KgEris::from_parts`]
/// already implements at every k-point — the two agree because a GHF
/// coefficient block that HAS a clean `orbspin` gives zero for exactly the
/// entries the second branch zeroes.
pub struct GammaGccsd<'a> {
    with_df: &'a dyn PeriodicDf,
    /// The padded spin-orbital MO coefficients, `2·nao × nmo`.
    pub mo_coeff: Vec<pyscf_pbc_df::MoCoeff>,
    /// `[nmo, nmo]` — the MO Fock, `exxdiv`-suppressed.
    pub fock: ZArr,
    /// `mo_energy`, with the Madelung re-add on the occupied block.
    pub mo_energy: Vec<f64>,
    pub nocc: usize,
    pub nmo: usize,
    /// `mf.e_tot`.
    pub e_hf: f64,
    /// Whether the reference mean field converged.
    pub converged: bool,
    /// The CC iteration's knobs. `GCCSD`'s `conv_tol_normt` default is `1e-6`,
    /// not the restricted `1e-5` (`cc/gccsd.py:117`) — set by
    /// [`GammaGccsd::default_opts`].
    pub opts: RccsdOpts,
}

impl<'a> GammaGccsd<'a> {
    /// `cc/gccsd.py:116-117`'s own convergence defaults, which differ from the
    /// restricted ones.
    #[must_use]
    pub fn default_opts() -> RccsdOpts {
        RccsdOpts {
            conv_tol: 1e-7,
            conv_tol_normt: 1e-6,
            ..RccsdOpts::default()
        }
    }

    /// `GCCSD(mf)` on a converged single-k-point GHF mean field.
    ///
    /// The Fock, the orbital energies and the Madelung re-add come from
    /// [`crate::kccsd::Kgccsd::new`], which is `pbc/cc/kccsd.py:538-555` — the
    /// SAME two-step treatment `pbc/cc/ccsd.py:136-143` applies, at one
    /// k-point.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] if the mean field is not one GHF channel at one
    /// k-point; otherwise propagates the Fock build.
    pub fn new(
        scf: &pyscf_pbc_scf::KScfResult,
        mf: &mut pyscf_pbc_scf::Kghf,
        with_df: &'a dyn PeriodicDf,
    ) -> Result<Self, PbcCcError> {
        if with_df.kpts().len() != 1 || scf.nkpts != 1 {
            return Err(shape(format!(
                "pbc/cc/ccsd.py's GCCSD is the SINGLE-k-point shim; \
                 the mean field carries {} k-points — use KGCCSD",
                scf.nkpts
            )));
        }
        let kg = crate::kccsd::Kgccsd::new(scf, mf)?;
        let nmo = kg.nmo;
        Ok(Self {
            with_df,
            mo_coeff: kg.mo_coeff.clone(),
            fock: kg.fock.slice_leading(&[0])?,
            mo_energy: kg
                .mo_energy
                .into_iter()
                .next()
                .ok_or_else(|| shape("Kgccsd produced no mo_energy"))?,
            nocc: kg.nocc,
            nmo,
            e_hf: scf.e_tot,
            converged: scf.converged,
            opts: Self::default_opts(),
        })
    }

    /// `GCCSD.ao2mo(mo_coeff)` (`:101-144`).
    ///
    /// # Errors
    /// Propagates the AO→MO transform and every shape check.
    pub fn ao2mo(&self) -> Result<crate::gccsd::PhysicistsErisZ, PbcCcError> {
        let eri = self.chemists_eri()?;
        crate::gccsd::PhysicistsErisZ::from_full_chemists(
            &eri,
            self.fock.clone(),
            self.mo_energy.clone(),
            self.nocc,
        )
    }

    /// The `ao2mofn` of `:107-131`: the full `[nmo]⁴` CHEMISTS' tensor over
    /// the spin-orbital basis, as the four-spin-block sum.
    ///
    /// # Errors
    /// Propagates the transform.
    pub fn chemists_eri(&self) -> Result<ZArr, PbcCcError> {
        let m = &self.mo_coeff[0];
        let nmo = self.nmo;
        if !m.nao.is_multiple_of(2) {
            return Err(shape(format!(
                "GCCSD needs a spin-orbital MO block with an even row count, got {}",
                m.nao
            )));
        }
        let nao = m.nao / 2;
        // `:110-111` — the two spin halves.
        let split = |top: bool| -> pyscf_pbc_df::MoCoeff {
            let off = usize::from(top) * nao;
            let mut c = CTensor::zeros(nao * nmo);
            for a in 0..nao {
                for p in 0..nmo {
                    c.re[a * nmo + p] = m.c.re[(a + off) * nmo + p];
                    c.im[a * nmo + p] = m.c.im[(a + off) * nmo + p];
                }
            }
            pyscf_pbc_df::MoCoeff::new(nao, nmo, c)
        };
        let mo_a = split(false);
        let mo_b = split(true);

        let mut acc = ZArr::zeros(&[nmo, nmo, nmo, nmo]);
        for (x, y) in [
            (&mo_a, &mo_a),
            (&mo_b, &mo_b),
            (&mo_a, &mo_b),
            (&mo_b, &mo_a),
        ] {
            let e = self
                .with_df
                .ao2mo([x, x, y, y], [0, 0, 0, 0], false)
                .map_err(|e| shape(format!("ao2mo: {e}")))?
                .restore_s1();
            acc.add_assign(&ZArr::from_ctensor(&[nmo, nmo, nmo, nmo], e.data)?)?;
        }
        Ok(acc)
    }

    /// `GCCSD.ccsd(t1, t2, eris, mbpt2=False)` (`:95-105`).
    ///
    /// # Errors
    /// Propagates the transform and the amplitude iteration.
    pub fn kernel(&self) -> Result<crate::gccsd::GccsdResult, PbcCcError> {
        let eris = self.ao2mo()?;
        self.kernel_with(&eris)
    }

    /// [`GammaGccsd::kernel`] on an already-built `_PhysicistsERIs`.
    ///
    /// # Errors
    /// Propagates the amplitude iteration.
    pub fn kernel_with(
        &self,
        eris: &crate::gccsd::PhysicistsErisZ,
    ) -> Result<crate::gccsd::GccsdResult, PbcCcError> {
        if !self.converged {
            return Err(PbcCcError::NotConverged {
                what: "the reference Gamma-point KGHF",
                detail: "GCCSD refuses an unconverged mean field".into(),
            });
        }
        crate::gccsd::kernel(eris, &self.opts)
    }

    /// `GCCSD.ccsd(..., mbpt2=True)` (`:98-104`).
    ///
    /// # This one is NOT Γ-only, and its siblings are
    ///
    /// It goes through `pbc.mp.GMP2`, whose `__init__` (`pbc/mp/mp2.py:47-51`)
    /// carries only `warn_pbc2d_eri` — no k-point guard — while `RMP2`
    /// (`:21-23`) and `UMP2` (`:35-37`) both refuse away from Γ. So of the
    /// three Γ shims only this one's `mbpt2` runs at a shifted k-point.
    /// Measured, not read.
    ///
    /// # Errors
    /// Propagates the transform.
    pub fn mbpt2(&self) -> Result<(f64, ZArr, ZArr), PbcCcError> {
        let eris = self.ao2mo()?;
        self.mbpt2_with(&eris)
    }

    /// [`GammaGccsd::mbpt2`] on an already-built `_PhysicistsERIs`.
    ///
    /// # Errors
    /// Propagates `init_amps`.
    pub fn mbpt2_with(
        &self,
        eris: &crate::gccsd::PhysicistsErisZ,
    ) -> Result<(f64, ZArr, ZArr), PbcCcError> {
        let (emp2, _t1, t2) = crate::gccsd::init_amps(eris)?;
        let t1 = ZArr::zeros(&[eris.nocc, eris.nvir]);
        Ok((emp2, t1, t2))
    }
}

/// `class UCCSD(uccsd.UCCSD)` (`pbc/cc/ccsd.py:61-92`) at a SINGLE k-point.
pub struct GammaUccsd<'a> {
    with_df: &'a dyn PeriodicDf,
    /// The padded alpha and beta MO coefficients.
    pub mo_coeff: (pyscf_pbc_df::MoCoeff, pyscf_pbc_df::MoCoeff),
    /// `(focka, fockb)`, each `[nmo, nmo]`, `exxdiv`-suppressed.
    pub fock: (ZArr, ZArr),
    /// `(mo_energy_a, mo_energy_b)`, with the Madelung re-add on each occupied
    /// block — `:89-91` applies `_adjust_occ` to BOTH spins.
    pub mo_energy: (Vec<f64>, Vec<f64>),
    pub nocc: (usize, usize),
    pub nmo: (usize, usize),
    /// `mf.e_tot`.
    pub e_hf: f64,
    /// Whether the reference mean field converged.
    pub converged: bool,
    /// The CC iteration's knobs.
    pub opts: RccsdOpts,
}

impl<'a> GammaUccsd<'a> {
    /// `UCCSD(mf)` on a converged single-k-point UHF mean field.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] if the mean field is not two unrestricted
    /// channels at one k-point; otherwise propagates the Fock build.
    pub fn new(
        scf: &pyscf_pbc_scf::KScfResult,
        with_df: &'a dyn PeriodicDf,
    ) -> Result<Self, PbcCcError> {
        if with_df.kpts().len() != 1 || scf.nkpts != 1 {
            return Err(shape(format!(
                "pbc/cc/ccsd.py's UCCSD is the SINGLE-k-point shim; \
                 the mean field carries {} k-points — use KUCCSD",
                scf.nkpts
            )));
        }
        if scf.nset != 2 {
            return Err(shape(
                "the Gamma-point UCCSD needs two unrestricted SCF channels",
            ));
        }
        let cell = with_df.cell();
        let nao = cell.mol.nao_nr;
        let frozen = pyscf_pbc_mp::FrozenK::default();
        let mut padded = Vec::with_capacity(2);
        let mut dm = Vec::with_capacity(2);
        for set in 0..2 {
            let mf = pyscf_pbc_mp::spin_block(scf, set)
                .map_err(|e| shape(format!("spin_block: {e}")))?;
            let raw: Result<Vec<pyscf_pbc_df::MoCoeff>, _> = mf
                .mo_coeff
                .iter()
                .zip(mf.mo_occ)
                .map(|(c, occ)| pyscf_pbc_mp::mo_coeff_from_kscf(c, nao, occ.len()))
                .collect();
            let raw = raw.map_err(|e| shape(format!("mo_coeff_from_kscf: {e}")))?;
            padded.push(
                pyscf_pbc_mp::add_padding(&raw, mf.mo_energy, mf.mo_occ, &frozen)
                    .map_err(|e| shape(format!("add_padding: {e}")))?,
            );
            dm.push(pyscf_pbc_scf::krdm::make_rdm1(mf.mo_coeff, mf.mo_occ, nao));
        }
        let pb = padded.pop().ok_or_else(|| shape("no beta MOs"))?;
        let pa = padded.pop().ok_or_else(|| shape("no alpha MOs"))?;
        let db = dm.pop().ok_or_else(|| shape("no beta density"))?;
        let da = dm.pop().ok_or_else(|| shape("no alpha density"))?;

        // `:78-80` — the Fock under `exxdiv = None`; `:89-91` — the Madelung
        // re-add on BOTH occupied blocks. Both are in `build_fock`.
        let ((fa, fb), (ea, eb), _mad) = crate::kueris::KuEris::build_fock(
            cell,
            with_df,
            (&pa, &pb),
            (&da, &db),
            crate::keris::KErisOpts::default(),
        )?;
        Ok(Self {
            with_df,
            mo_coeff: (pa.mo_coeff[0].clone(), pb.mo_coeff[0].clone()),
            fock: (fa.slice_leading(&[0])?, fb.slice_leading(&[0])?),
            mo_energy: (
                ea.into_iter()
                    .next()
                    .ok_or_else(|| shape("no alpha mo_energy"))?,
                eb.into_iter()
                    .next()
                    .ok_or_else(|| shape("no beta mo_energy"))?,
            ),
            nocc: (pa.nocc, pb.nocc),
            nmo: (pa.nmo, pb.nmo),
            e_hf: scf.e_tot,
            converged: scf.converged,
            opts: RccsdOpts::default(),
        })
    }

    /// `UCCSD.ao2mo(mo_coeff)` (`:74-92`).
    ///
    /// # Errors
    /// Propagates the AO→MO transform and every shape check.
    pub fn ao2mo(&self) -> Result<crate::uccsd::ChemistsErisU, PbcCcError> {
        let (a, b) = (&self.mo_coeff.0, &self.mo_coeff.1);
        // `uccsd.py:886-888` — `ao2mofn(moa)`, `ao2mofn(mob)` and
        // `ao2mofn((moa,moa,mob,mob))`, the periodic transform each time.
        let one =
            |x: &pyscf_pbc_df::MoCoeff, y: &pyscf_pbc_df::MoCoeff| -> Result<ZArr, PbcCcError> {
                let e = self
                    .with_df
                    .ao2mo([x, x, y, y], [0, 0, 0, 0], false)
                    .map_err(|e| shape(format!("ao2mo: {e}")))?
                    .restore_s1();
                let (nx, ny) = (x.nmo, y.nmo);
                ZArr::from_ctensor(&[nx, nx, ny, ny], e.data)
            };
        let eri_aa = one(a, a)?;
        let eri_bb = one(b, b)?;
        let eri_ab = one(a, b)?;
        crate::uccsd::ChemistsErisU::from_full_chemists(
            &eri_aa,
            &eri_bb,
            &eri_ab,
            self.fock.0.clone(),
            self.fock.1.clone(),
            self.mo_energy.clone(),
            self.nocc,
        )
    }

    /// `UCCSD.ccsd(t1, t2, eris, mbpt2=False)` (`:62-73`).
    ///
    /// # Errors
    /// Propagates the transform and the amplitude iteration.
    pub fn kernel(&self) -> Result<crate::uccsd::UccsdResult, PbcCcError> {
        let eris = self.ao2mo()?;
        self.kernel_with(&eris)
    }

    /// [`GammaUccsd::kernel`] on an already-built `_ChemistsERIs`.
    ///
    /// # Errors
    /// Propagates the amplitude iteration.
    pub fn kernel_with(
        &self,
        eris: &crate::uccsd::ChemistsErisU,
    ) -> Result<crate::uccsd::UccsdResult, PbcCcError> {
        if !self.converged {
            return Err(PbcCcError::NotConverged {
                what: "the reference Gamma-point KUHF",
                detail: "UCCSD refuses an unconverged mean field".into(),
            });
        }
        crate::uccsd::kernel(eris, &self.opts)
    }

    /// Whether the single k-point this shim runs at is Γ, to
    /// `pbc/mp/mp2.py:36`'s own `1e-9` threshold.
    #[must_use]
    pub fn is_gamma(&self) -> bool {
        self.with_df.kpts()[0].iter().all(|v| v.abs() <= 1e-9)
    }

    /// `UCCSD.ccsd(..., mbpt2=True)` (`:65-72`) — Γ-ONLY, through
    /// `pbc.mp.UMP2` (`pbc/mp/mp2.py:35-37`).
    ///
    /// # Errors
    /// [`PbcCcError::NotImplementedUpstream`] away from Γ; otherwise
    /// propagates the transform.
    pub fn mbpt2(
        &self,
    ) -> Result<
        (
            f64,
            crate::kintermediates_uhf::UT1,
            crate::kintermediates_uhf::UT2,
        ),
        PbcCcError,
    > {
        let eris = self.ao2mo()?;
        self.mbpt2_with(&eris)
    }

    /// [`GammaUccsd::mbpt2`] on an already-built `_ChemistsERIs`.
    ///
    /// # Errors
    /// [`PbcCcError::NotImplementedUpstream`] away from Γ; otherwise
    /// propagates `init_amps`.
    pub fn mbpt2_with(
        &self,
        eris: &crate::uccsd::ChemistsErisU,
    ) -> Result<
        (
            f64,
            crate::kintermediates_uhf::UT1,
            crate::kintermediates_uhf::UT2,
        ),
        PbcCcError,
    > {
        if !self.is_gamma() {
            return Err(PbcCcError::NotImplementedUpstream {
                upstream: "pbc/mp/mp2.py:36",
                what: "UMP2.__init__ raises NotImplementedError away from Gamma, so \
                       UCCSD.ccsd(mbpt2=True) is a Gamma-only branch",
            });
        }
        let (emp2, _t1, t2) = crate::uccsd::init_amps(eris)?;
        // `:71` — `self.t1 = (zeros(nocca,nvira), zeros(noccb,nvirb))`.
        let t1 = (
            ZArr::zeros(&[eris.nocca, eris.nvira]),
            ZArr::zeros(&[eris.noccb, eris.nvirb]),
        );
        Ok((emp2, t1, t2))
    }
}

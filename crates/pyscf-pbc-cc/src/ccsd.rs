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
//! # RCCSD only
//!
//! `UCCSD` (`:61-92`) and `GCCSD` (`:94-144`) need molecular complex-capable
//! `uccsd` and `gccsd`, which this port does not have any more than it had
//! `rccsd` before this module — see [`uccsd_refusal`] and [`gccsd_refusal`],
//! which name the upstream lines rather than pretending the surface exists.
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

/// The refusal `pbc/cc/ccsd.py`'s `UCCSD` gets.
#[must_use]
pub fn uccsd_refusal() -> PbcCcError {
    PbcCcError::NotImplementedUpstream {
        upstream: "pbc/cc/ccsd.py:61",
        what: "the Gamma-point UCCSD shim subclasses the molecular complex-capable \
               cc/uccsd.py, which this port does not have (16-CONTEXT §1.2)",
    }
}

/// The refusal `pbc/cc/ccsd.py`'s `GCCSD` gets.
#[must_use]
pub fn gccsd_refusal() -> PbcCcError {
    PbcCcError::NotImplementedUpstream {
        upstream: "pbc/cc/ccsd.py:94",
        what: "the Gamma-point GCCSD shim subclasses the molecular complex-capable \
               cc/gccsd.py, of which pyscf_ccsd::gccsd is a deliberately partial port",
    }
}

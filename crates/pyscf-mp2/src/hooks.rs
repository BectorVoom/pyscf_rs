//! `Mp2OverrideHooks` trait + `NoMp2Overrides` default impl (D-08): the
//! subclass-override seam that DF-MP2 uses to swap the `ao2mo` ERI source
//! (mirrors the Phase-3 SCF `OverrideHooks` precedent, but a smaller method
//! set — MP2 only overrides the integral source, the energy, and the RDMs).
//!
//! pyo3-free (D-07/D-08): the PyO3 bridge + subclass-override dispatch lives
//! exclusively in `pyscf-py`. Each `NoMp2Overrides` method delegates to a
//! `crate::mp2::default_*` free function — the same pattern as
//! `pyscf-scf/src/hooks.rs::NoOverrides`.

use crate::frozen::Frozen;
use crate::mp2::Mp2Reference;
use pyscf_core::{Density, Energy, PyscfRsError};

/// Chemist's-notation `(occ,vir|occ,vir)` ERI block — the `(ia|jb)` integrals
/// the closed-form RMP2 kernel consumes.
///
/// `ovov` is the flat `(i,a,j,b)` C-order block: element `(i,a,j,b)` lives at
/// `((i*nvir + a)*nocc + j)*nvir + b`. (Upstream stores
/// `eris.ovov[i*nvir:(i+1)*nvir]` then reshapes `(nvir,nocc,nvir)`; this flat
/// layout is the same data in `(i,a,j,b)` order.)
#[derive(Debug, Clone)]
pub struct ChemistsEris {
    /// Flat `(i,a,j,b)` C-order `(ia|jb)` block, length `nocc*nvir*nocc*nvir`.
    pub ovov: Vec<f64>,
    /// Number of (active) occupied orbitals.
    pub nocc: usize,
    /// Number of (active) virtual orbitals.
    pub nvir: usize,
}

/// MP2 subclass-override seam (D-08). pyo3-free. DF-MP2 (plan 05-05) implements
/// this to swap `ao2mo` for the density-fitted `(ia|jb)` source; the RDM
/// methods are fleshed out in plan 05-04.
pub trait Mp2OverrideHooks {
    /// Build the `(ia|jb)` block. The main override point — DF-MP2 swaps the
    /// integral source here. The default impl (via `NoMp2Overrides`) routes
    /// through `crate::mp2::default_ao2mo` (in-core `int2e` → `ao2mo.general`).
    fn ao2mo(&self, refr: &Mp2Reference, frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError>;

    /// SCS-scaled correlation energy from a precomputed block. Default routes
    /// through `crate::mp2::default_energy`. (`ss_factor`/`os_factor` are the
    /// SCS factors; `1.0`/`1.0` reproduces plain MP2.)
    fn energy(
        &self,
        eris: &ChemistsEris,
        refr: &Mp2Reference,
        frozen: &Frozen,
        ss_factor: f64,
        os_factor: f64,
    ) -> Result<Energy, PyscfRsError> {
        let e = crate::mp2::default_energy(eris, refr, frozen, ss_factor, os_factor)?;
        Ok(Energy(e))
    }

    /// MP2 one-particle reduced density matrix. Body lands in plan 05-04.
    fn make_rdm1(&self, _refr: &Mp2Reference, _frozen: &Frozen) -> Result<Density, PyscfRsError> {
        Err(crate::error::Mp2Error::NotYetImplemented { plan: 4 }.into())
    }

    /// MP2 two-particle reduced density matrix. Body lands in plan 05-04.
    fn make_rdm2(&self, _refr: &Mp2Reference, _frozen: &Frozen) -> Result<Density, PyscfRsError> {
        Err(crate::error::Mp2Error::NotYetImplemented { plan: 4 }.into())
    }
}

/// The no-override default. Every method delegates to a `crate::mp2::default_*`
/// free function (mirrors `pyscf-scf::NoOverrides`).
pub struct NoMp2Overrides;

impl Mp2OverrideHooks for NoMp2Overrides {
    fn ao2mo(&self, refr: &Mp2Reference, frozen: &Frozen) -> Result<ChemistsEris, PyscfRsError> {
        crate::mp2::default_ao2mo(refr, frozen)
    }
}

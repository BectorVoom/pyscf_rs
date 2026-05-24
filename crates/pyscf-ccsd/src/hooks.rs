//! `CcsdOverrideHooks` trait + `NoCcsdOverrides` default impl (D-09): the
//! subclass-override seam CCSD uses to swap the ERI source (`ao2mo`), the
//! amplitude-equation core (`update_amps`), the energy, and the RDMs.
//!
//! Mirrors `crates/pyscf-mp2/src/hooks.rs` (the closest one-to-one analog in
//! the phase), with a richer hook set: MP2 overrides `ao2mo`/`energy`/`make_rdm1`/
//! `make_rdm2`; CCSD adds `update_amps` (the per-iteration amplitude update —
//! the biggest `py.detach` region in the project once the pyscf-py bridge lands
//! in Wave 5).
//!
//! pyo3-free (D-09): the PyO3 bridge + subclass-override dispatch lives
//! exclusively in `pyscf-py`. Each `NoCcsdOverrides` method delegates to a
//! `crate::ccsd::default_*` / `crate::update_amps::default_update_amps` free
//! function — the same pattern as `pyscf_mp2::NoMp2Overrides`.
//!
//! The `ChemistsEris` container itself lives in [`crate::eris`] (it carries far
//! more blocks than MP2's single `ovov`); it is re-exported from this module's
//! parent so the `lib.rs` flat re-export shape matches MP2's.

use crate::eris::ChemistsEris;
use crate::reference::CcsdReference;
use pyscf_core::{Amplitudes, Density, Energy, PyscfRsError};
use pyscf_mp2::Frozen;

/// CCSD subclass-override seam (D-09). pyo3-free. DF-CCSD (Wave 4) implements
/// `ao2mo` to swap the in-core `int2e` block source for the density-fitted
/// `vvL` B-tensor; a Python subclass overrides `update_amps`/`make_rdm1`/
/// `make_rdm2`/`energy` through the pyscf-py bridge (Wave 5).
pub trait CcsdOverrideHooks {
    /// Build the full `ChemistsEris` block set. The main ERI override point —
    /// DF-CCSD swaps the integral source here. The default (via
    /// `NoCcsdOverrides`) routes through [`crate::ccsd::default_ao2mo`].
    fn ao2mo(
        &self,
        refr: &CcsdReference,
        frozen: &Frozen,
    ) -> Result<ChemistsEris, PyscfRsError>;

    /// The amplitude-equation core: one `(t1, t2) → (t1new, t2new)` update.
    ///
    /// The amplitude handle is `pyscf_core::Amplitudes` for this scaffold;
    /// Wave 2's opaque-`Tensor` upgrade (D-01, the `WorkspacePool`-backed
    /// spillable handle) flows through this signature when it lands. The
    /// default (via `NoCcsdOverrides`) routes through
    /// [`crate::update_amps::default_update_amps`].
    fn update_amps(
        &self,
        t1: &Amplitudes,
        t2: &Amplitudes,
        eris: &ChemistsEris,
    ) -> Result<(Amplitudes, Amplitudes), PyscfRsError>;

    /// CCSD correlation energy from amplitudes + a precomputed block set.
    /// Default-delegates to [`crate::ccsd::default_energy`].
    fn energy(
        &self,
        t1: &Amplitudes,
        t2: &Amplitudes,
        eris: &ChemistsEris,
    ) -> Result<Energy, PyscfRsError> {
        crate::ccsd::default_energy(t1, t2, eris)
    }

    /// CCSD one-particle reduced density matrix.
    ///
    /// The RDM math ships in Wave 3 as [`crate::rdm::make_rdm1`] (consuming the
    /// converged amplitudes + the λ multipliers). This trait-default seam
    /// returns `NotYetImplemented { wave: 3 }`.
    fn make_rdm1(
        &self,
        _refr: &CcsdReference,
        _frozen: &Frozen,
    ) -> Result<Density, PyscfRsError> {
        Err(crate::error::CcsdError::NotYetImplemented { wave: 3 }.into())
    }

    /// CCSD two-particle reduced density matrix. See [`make_rdm1`] — the math
    /// ships as [`crate::rdm::make_rdm2`] in Wave 3; this trait-default seam is
    /// a `NotYetImplemented { wave: 3 }` stub.
    ///
    /// [`make_rdm1`]: CcsdOverrideHooks::make_rdm1
    fn make_rdm2(
        &self,
        _refr: &CcsdReference,
        _frozen: &Frozen,
    ) -> Result<Density, PyscfRsError> {
        Err(crate::error::CcsdError::NotYetImplemented { wave: 3 }.into())
    }
}

/// The no-override default. Every method delegates to a `crate::ccsd::default_*`
/// / `crate::update_amps::default_update_amps` free function (mirrors
/// `pyscf_mp2::NoMp2Overrides`).
pub struct NoCcsdOverrides;

impl CcsdOverrideHooks for NoCcsdOverrides {
    fn ao2mo(
        &self,
        refr: &CcsdReference,
        frozen: &Frozen,
    ) -> Result<ChemistsEris, PyscfRsError> {
        crate::ccsd::default_ao2mo(refr, frozen)
    }

    fn update_amps(
        &self,
        t1: &Amplitudes,
        t2: &Amplitudes,
        eris: &ChemistsEris,
    ) -> Result<(Amplitudes, Amplitudes), PyscfRsError> {
        crate::update_amps::default_update_amps(t1, t2, eris)
    }
}

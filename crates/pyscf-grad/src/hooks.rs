//! `GradOverrideHooks` trait + `NoGradOverrides` default impl (D-09): the
//! subclass-override seam the gradient methods use to swap the 2-electron
//! response term (`get_veff` — DF-grad swaps the in-core `int2e_ip1` source
//! for the density-fitted `int3c2e_ip1` path) and to inject any method-
//! specific `extra_force` (e.g. the grid-weight-derivative term in KS-grad, or
//! a pulled-back point-charge force).
//!
//! Mirrors `crates/pyscf-ccsd/src/hooks.rs:29-97` — the
//! `trait XOverrideHooks { ... }` + `struct NoXOverrides` default-delegation
//! pattern: each trait method default-delegates to a `crate::*::default_*`
//! free function, and `NoGradOverrides` is the zero-override marker.
//!
//! pyo3-free (D-09): the PyO3 bridge + subclass-override dispatch lives
//! exclusively in `pyscf-py`. The bodies these hooks delegate to land in the
//! per-method waves (07-03..07-08); for THIS plan they are
//! `NotYetImplemented { wave }` seams, so the override contract is fixed from
//! day one (D-09) without committing to a method body.

use crate::error::GradError;
use pyscf_core::{Density, Mole, PyscfRsError};

/// Gradient subclass-override seam (D-09). pyo3-free. DF-grad (Wave 5)
/// implements `get_veff` to swap the `int2e_ip1` block source for the
/// density-fitted `int3c2e_ip1` B-tensor derivative; a Python subclass
/// overrides `get_veff` / `extra_force` through the pyscf-py bridge.
pub trait GradOverrideHooks {
    /// The 2-electron response term of the gradient — `get_jk` over
    /// `int2e_ip1`, returning a component-leading `[3, nao, nao]` array
    /// (the `vhf` in `pyscf/grad/rhf.py:149-183`). The main override point:
    /// DF-grad swaps the integral source here. The trait default (via
    /// `NoGradOverrides`) is a `NotYetImplemented { wave: 3 }` seam until the
    /// RHF body lands.
    fn get_veff(&self, _mol: &Mole, _dm: &Density) -> Result<Density, PyscfRsError> {
        Err(GradError::NotYetImplemented { wave: 3 }.into())
    }

    /// Any method-specific additional force to add to the assembled gradient
    /// (e.g. the KS grid-weight-derivative term, or a pulled-back external
    /// point-charge contribution). Defaults to a zero contribution (no extra
    /// force) — the common case for plain HF, where the gradient is fully
    /// captured by the Hellmann-Feynman + Pulay terms.
    fn extra_force(&self, _atom_id: usize, _mol: &Mole) -> Result<[f64; 3], PyscfRsError> {
        Ok([0.0, 0.0, 0.0])
    }
}

/// The no-override default. `get_veff` delegates to the per-method default
/// (a `NotYetImplemented` seam until the body lands); `extra_force` returns a
/// zero contribution. Mirrors `pyscf_ccsd::NoCcsdOverrides`.
pub struct NoGradOverrides;

impl GradOverrideHooks for NoGradOverrides {}

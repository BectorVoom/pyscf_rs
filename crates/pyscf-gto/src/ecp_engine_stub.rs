//! GTO-05 (engine half): `EcpEngineNotAvailable` stub per D-06 sequencing.
//!
//! Returns `PyscfRsError::EcpEngineNotAvailable` for every method.
//! Gap-closure plan 02-10 swaps the stub for the cintx-backed impl when
//! cintx ECP merges; the user-facing API surface (`mol.intor("int1e_ecp")`)
//! is unaffected by the swap.

use pyscf_core::{Density, EcpEngine, Mole, PyscfRsError};

/// The Phase 2 ECP engine: every method errs with `EcpEngineNotAvailable`.
///
/// `pyscf_gto::ecp_engine()` returns this until 02-10 wires the
/// cintx-backed impl. The default `ecp_int1e_ipnuc` impl (declared in
/// `pyscf_core::traits::EcpEngine`) returns `NotYetImplemented{phase:7}`
/// — that's the right error for the Phase 2 stub since the gradient
/// path is gated on Phase 7 GRAD-07 regardless of cintx ECP availability.
#[derive(Debug, Default, Clone, Copy)]
pub struct EcpEngineNotAvailable;

impl EcpEngine for EcpEngineNotAvailable {
    fn ecp_int1e(&self, _mol: &Mole, _name: &str) -> Result<Density, PyscfRsError> {
        Err(PyscfRsError::EcpEngineNotAvailable)
    }
}

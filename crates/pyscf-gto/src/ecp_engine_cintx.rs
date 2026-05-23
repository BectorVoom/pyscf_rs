//! GTO-05 (eval half): `EcpEngine` impl backed by cintx ECP integral
//! evaluation. Closes the evaluation half of GTO-05 per gap-closure plan
//! 02-10 (the cintx Phase 19 workstream shipped `int1e_ecp_{cart,sph}`
//! Type-1 + Type-2 projector integrals byte-identical to upstream PySCF
//! `nr_ecp`, verified by `cintx-oracle/tests/safe_api_ecp_parity.rs` at
//! `atol=1e-12`).
//!
//! Replaces `EcpEngineNotAvailable` as the default returned by
//! `pyscf_gto::ecp_engine()`. The stub stays in the codebase
//! (`ecp_engine_stub`) for documentation of the pre-merge state and so the
//! `EcpEngineNotAvailable` error path remains testable.
//!
//! ## How this mirrors the non-ECP intor path
//!
//! `crate::intor::evaluate_arity2` iterates `(i, j)` AO shell pairs over
//! the Mole's typed `cintx_core::BasisSet`, drives one
//! `cintx_rs::SessionRequest` per pair, and stitches each block into an
//! F-order `nao × nao` buffer. This engine does the same — the ONLY
//! difference is that the ECP safe-API preflight
//! (`SessionRequest::query_workspace`) requires the `BasisSet` to carry ECP
//! shells (`basis.ecp_shells()` non-empty), else `FacadeError::MissingEcpBasis`.
//! The `BasisSet` stored in `mol.basis_set` is built ECP-free
//! (`projection::build_cintx_basis_set` → `BasisSet::try_new`), so this
//! engine builds an ECP-augmented view on demand via
//! `projection::build_cintx_basis_set_with_ecp`.
//!
//! ## Operator resolution
//!
//! The dispatcher hands names already `_add_suffix`-normalised
//! (`int1e_ecp` → `int1e_ecp_sph` for spherical molecules). The four typed
//! cintx ECP operators live at manifest positions 26..=29
//! (`OperatorId::INT1E_ECP_{CART,SPH,IPNUC_*}`). We resolve scalar names
//! (`int1e_ecp_sph`, `int1e_ecp_cart`, and the `ECPscalar*` aliases) to the
//! scalar operator for the active representation; the `ipnuc` gradient arm
//! is gated to Phase 7 GRAD-07 via the trait's `ecp_int1e_ipnuc` default.

use cintx_core::{BasisSet as CintxBasisSet, OperatorId, Representation};
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use pyscf_core::{CoreError, Density, EcpEngine, Mole, PyscfRsError};

/// `EcpEngine` backed by cintx ECP. Routes `int1e_ecp*` / `ECPscalar*`
/// names through `cintx_rs::SessionRequest` over an ECP-augmented
/// `cintx_core::BasisSet`.
#[derive(Debug, Default, Clone, Copy)]
pub struct CintxEcpEngine;

impl EcpEngine for CintxEcpEngine {
    fn ecp_int1e(&self, mol: &Mole, name: &str) -> Result<Density, PyscfRsError> {
        if !mol._built {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                "CintxEcpEngine: Mole not built — call pyscf_gto::M(args) or mol.build() first"
                    .into(),
            )));
        }

        // A molecule with no ECP entries cannot evaluate an ECP integral —
        // the cintx safe API would (correctly) refuse with MissingEcpBasis.
        // Surface the canonical EcpEngineNotAvailable so callers see a stable
        // error rather than a low-level facade message.
        if mol._ecp.is_empty() {
            return Err(PyscfRsError::EcpEngineNotAvailable);
        }

        // Representation per the (already suffix-normalised) name. For the
        // `ECPscalar*` aliases (no _sph/_cart suffix) fall back to the Mole's
        // cart flag, matching upstream `_add_suffix` semantics.
        let representation = if name.ends_with("_cart") {
            Representation::Cart
        } else if name.ends_with("_sph") {
            Representation::Spheric
        } else if mol.cart {
            Representation::Cart
        } else {
            Representation::Spheric
        };

        let operator = match representation {
            Representation::Cart => OperatorId::INT1E_ECP_CART,
            Representation::Spheric => OperatorId::INT1E_ECP_SPH,
            Representation::Spinor => {
                return Err(PyscfRsError::NotYetImplemented {
                    phase: 3,
                    what: "spinor ECP integrals (out of v1 scope)",
                });
            }
        };

        // Build the ECP-augmented typed BasisSet (AO shells identical to the
        // Mole's stored basis_set; ECP shells projected from mol._ecp).
        let basis_arc = crate::projection::build_cintx_basis_set_with_ecp(
            &mol._atom,
            &mol._basis,
            &mol._ecp,
            mol.cart,
        )?;
        let basis_ref: &CintxBasisSet = &basis_arc;

        if basis_ref.ecp_shells().is_empty() {
            // mol._ecp was non-empty but projected to zero ECP shells (e.g.
            // an ECP entry with no channels for any present atom). Treat as
            // "no ECP available" rather than letting the facade error leak.
            return Err(PyscfRsError::EcpEngineNotAvailable);
        }

        let nbas = basis_ref.shells().len();
        let nao = basis_ref.meta().total_ao;

        // Cache per-shell AO offsets + counts from BasisMeta.
        let meta = basis_ref.meta();
        let shell_offsets: Vec<usize> = (0..nbas)
            .map(|s| meta.shell_offset(s).unwrap_or(0))
            .collect();
        let shell_counts: Vec<usize> = (0..nbas).map(|s| meta.ao_count(s).unwrap_or(0)).collect();

        // F-order nao × nao output buffer (matches every other arity-2 1e
        // intor; the dispatcher promises this layout).
        let mut out = vec![0.0_f64; nao * nao];

        for i in 0..nbas {
            for j in 0..nbas {
                let shells = basis_ref.shell_tuple_for_indices([i, j]).map_err(|e| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "ECP shell_tuple_for_indices(i={i}, j={j}) failed for '{name}': {e}",
                    )))
                })?;

                let request = SessionRequest::new(
                    operator,
                    representation,
                    basis_ref,
                    shells,
                    ExecutionOptions::default(),
                );
                let outcome = request
                    .query_workspace()
                    .map_err(|e| {
                        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                            "cintx ECP workspace query failed for '{name}' shell pair ({i},{j}): {e}",
                        )))
                    })?
                    .evaluate()
                    .map_err(|e| {
                        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                            "cintx ECP evaluate failed for '{name}' shell pair ({i},{j}): {e}",
                        )))
                    })?;

                let ni = shell_counts[i];
                let nj = shell_counts[j];
                let oi = shell_offsets[i];
                let oj = shell_offsets[j];

                let block = &outcome.tensor.owned_values;
                let expected = ni * nj;
                if block.len() != expected {
                    return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "cintx ECP returned block of {} elements for shell pair ({i},{j}) of \
                         '{name}', expected ni*nj = {expected} (ni={ni}, nj={nj}, extents={:?})",
                        block.len(),
                        outcome.tensor.extents,
                    ))));
                }

                // Per the arity-2 contract (cintx-oracle safe_api_ecp_parity.rs
                // collector): the per-pair block is row-major within the shell
                // pair — block[ii * nj + jj] is element (ii, jj). Stitch into
                // the global F-order buffer at (oi+ii, oj+jj):
                //   out[(oi+ii) + (oj+jj) * nao] = block[ii * nj + jj].
                for ii in 0..ni {
                    for jj in 0..nj {
                        out[(oi + ii) + (oj + jj) * nao] = block[ii * nj + jj];
                    }
                }
            }
        }

        Ok(Density::from_flat(nao, out))
    }
}

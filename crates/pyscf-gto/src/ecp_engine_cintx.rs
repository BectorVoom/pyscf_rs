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
//! (`int1e_ecp_sph`, `int1e_ecp_cart`, and the `ECPscalar` aliases) to the
//! scalar operator for the active representation. The scalar `ecp_int1e`
//! entry-point still rejects derivative/gradient names up front (so a
//! gradient name can never silently resolve to the scalar operator — 02-10
//! code review WR-01); but the rejection is now a clean cintx-availability
//! error, NOT `NotYetImplemented{phase:7}` (Phase 7 GRAD-07 owns this seam).
//!
//! ## ECP gradient (GRAD-07 / Phase 7, plan 07-01)
//!
//! `ecp_int1e_ipnuc` resolves `OperatorId::INT1E_ECP_IPNUC_{CART,SPH}`
//! (manifest positions 28..=29 — `int1e_ecp_ipnuc_{cart,sph}`, present on
//! cintx main today per 07-RESEARCH §Gradient-Integral Availability Matrix)
//! and returns a component-leading `[3, nao, nao]` buffer (the `c × nao × nao`
//! analog of the scalar `nao × nao` stitch). This is the `ECPscalar_ipnuc`
//! grad term consumed by `pyscf/grad/rhf.py:116-117` (`get_hcore`).
//!
//! `int1e_ecp_iprinv` / `ECPscalar_iprinv` (the per-atom `hcore_deriv` term,
//! `pyscf/grad/rhf.py:139-140`) is **MISSING from every cintx branch today**
//! and is not yet on any scheduled cintx workstream — it routes to a clean
//! cintx-availability error (NOT `NotYetImplemented{phase:7}`); numeric
//! un-gating waits on a future cintx workstream landing the family.

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

        // WR-01 (02-10 code review): the scalar `ecp_int1e` entry-point wires
        // ONLY the scalar ECP operator. Derivative/gradient ECP names —
        // int1e_ecp_ipnuc / int1e_ecp_iprinv and the ECPscalar_ip* aliases —
        // must NOT fall through to the scalar OperatorId (that would return a
        // wrong-shaped scalar matrix mislabeled as a gradient). Reject anything
        // whose representation-suffix-stripped core is not the bare scalar name,
        // BEFORE the _ecp guard so the contract does not depend on guard
        // ordering. GRAD-07 (plan 07-01) wires the gradient `ipnuc` through the
        // dedicated `ecp_int1e_ipnuc` override below; this scalar path therefore
        // rejects ALL derivative names with a clean cintx-availability error
        // (never `NotYetImplemented{phase:7}` — that disposition is closed).
        let core = name
            .strip_suffix("_sph")
            .or_else(|| name.strip_suffix("_cart"))
            .or_else(|| name.strip_suffix("_spinor"))
            .unwrap_or(name);
        if core != "int1e_ecp" && core != "ECPscalar" {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "ECP derivative/gradient name '{name}' is not a scalar ECP integral; \
                 ECP gradients route through `ecp_int1e_ipnuc` (GRAD-07). The `ipnuc` \
                 family is cintx-ready; `iprinv`/ECPscalar_iprinv is MISSING from cintx \
                 today (no scheduled cintx workstream) — un-gates when that family lands."
            ))));
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

        // Cache per-shell AO offsets + counts from BasisMeta. WR-02 (02-10
        // code review): a missing offset/count for a valid shell index is an
        // invariant break — surface it as a loud internal error rather than
        // silently substituting 0 (which would corrupt the stitch into M[0,0]
        // and still pass the oracle's finite/non-zero checks).
        let meta = basis_ref.meta();
        let shell_offsets: Vec<usize> = (0..nbas)
            .map(|s| {
                meta.shell_offset(s).ok_or_else(|| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "ECP basis meta missing shell_offset for shell {s} of {nbas}"
                    )))
                })
            })
            .collect::<Result<_, _>>()?;
        let shell_counts: Vec<usize> = (0..nbas)
            .map(|s| {
                meta.ao_count(s).ok_or_else(|| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "ECP basis meta missing ao_count for shell {s} of {nbas}"
                    )))
                })
            })
            .collect::<Result<_, _>>()?;

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

    /// GRAD-07 (plan 07-01): ECP gradient `int1e_ecp_ipnuc` / `ECPscalar_ipnuc`.
    ///
    /// Resolves `OperatorId::INT1E_ECP_IPNUC_{CART,SPH}` (manifest positions
    /// 28..=29 — present on cintx main today, 07-RESEARCH §Availability Matrix)
    /// and returns a component-leading `[3, nao, nao]` F-order buffer (the
    /// `c × nao × nao` analog of the scalar `nao × nao` stitch). This is the
    /// `ECPscalar_ipnuc` grad term consumed by `pyscf/grad/rhf.py:116-117`.
    ///
    /// The returned `Density` carries `nao` (the AO-axis dimension) and a
    /// `3 * nao * nao` flat buffer laid out component-leading F-order:
    /// `data[comp + p*3 + q*3*nao]` is `∂/∂R_comp ⟨p|V_ECP|q⟩`.
    ///
    /// D-02: `int1e_ecp_iprinv` / `ECPscalar_iprinv` (the per-atom
    /// `with_rinv_at_nucleus` `hcore_deriv` term, `pyscf/grad/rhf.py:139-140`)
    /// is MISSING from cintx and routes to a clean cintx-availability error
    /// (never `NotYetImplemented{phase:7}`).
    fn ecp_int1e_ipnuc(&self, mol: &Mole, name: &str) -> Result<Density, PyscfRsError> {
        if !mol._built {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                "CintxEcpEngine::ecp_int1e_ipnuc: Mole not built — call \
                 pyscf_gto::M(args) or mol.build() first"
                    .into(),
            )));
        }

        // Reject the still-missing iprinv family with a clean cintx-availability
        // error (NOT NotYetImplemented{phase:7}). The suffix-stripped core must
        // be exactly the ipnuc gradient name.
        let core = name
            .strip_suffix("_sph")
            .or_else(|| name.strip_suffix("_cart"))
            .or_else(|| name.strip_suffix("_spinor"))
            .unwrap_or(name);
        if core != "int1e_ecp_ipnuc" && core != "ECPscalar_ipnuc" {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "ECP gradient name '{name}' is not the cintx-ready ipnuc family. \
                 `ECPscalar_iprinv`/`int1e_ecp_iprinv` is MISSING from every cintx branch \
                 today (no scheduled cintx workstream) — numeric un-gates when that family \
                 lands; only `ECPscalar_ipnuc`/`int1e_ecp_ipnuc` is ready."
            ))));
        }

        if mol._ecp.is_empty() {
            return Err(PyscfRsError::EcpEngineNotAvailable);
        }

        // Representation per the (already suffix-normalised) name; ECPscalar_*
        // aliases fall back to the Mole's cart flag (matches `ecp_int1e`).
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
            Representation::Cart => OperatorId::INT1E_ECP_IPNUC_CART,
            Representation::Spheric => OperatorId::INT1E_ECP_IPNUC_SPH,
            Representation::Spinor => {
                return Err(PyscfRsError::NotYetImplemented {
                    phase: 3,
                    what: "spinor ECP gradient integrals (out of v1 scope)",
                });
            }
        };

        // ECP-augmented typed BasisSet (same build as the scalar path).
        let basis_arc = crate::projection::build_cintx_basis_set_with_ecp(
            &mol._atom,
            &mol._basis,
            &mol._ecp,
            mol.cart,
        )?;
        let basis_ref: &CintxBasisSet = &basis_arc;

        if basis_ref.ecp_shells().is_empty() {
            return Err(PyscfRsError::EcpEngineNotAvailable);
        }

        let nbas = basis_ref.shells().len();
        let nao = basis_ref.meta().total_ao;
        const COMPONENTS: usize = 3; // x/y/z derivative components.

        let meta = basis_ref.meta();
        let shell_offsets: Vec<usize> = (0..nbas)
            .map(|s| {
                meta.shell_offset(s).ok_or_else(|| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "ECP basis meta missing shell_offset for shell {s} of {nbas}"
                    )))
                })
            })
            .collect::<Result<_, _>>()?;
        let shell_counts: Vec<usize> = (0..nbas)
            .map(|s| {
                meta.ao_count(s).ok_or_else(|| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "ECP basis meta missing ao_count for shell {s} of {nbas}"
                    )))
                })
            })
            .collect::<Result<_, _>>()?;

        // Component-leading F-order `[3, nao, nao]` buffer:
        // out[comp + p*3 + q*3*nao] = ∂/∂R_comp ⟨p|V_ECP|q⟩.
        let nao_c = nao * COMPONENTS;
        let mut out = vec![0.0_f64; COMPONENTS * nao * nao];

        for i in 0..nbas {
            for j in 0..nbas {
                let shells = basis_ref.shell_tuple_for_indices([i, j]).map_err(|e| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "ECP-grad shell_tuple_for_indices(i={i}, j={j}) failed for '{name}': {e}",
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
                            "cintx ECP-grad workspace query failed for '{name}' shell pair ({i},{j}): {e}",
                        )))
                    })?
                    .evaluate()
                    .map_err(|e| {
                        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                            "cintx ECP-grad evaluate failed for '{name}' shell pair ({i},{j}): {e}",
                        )))
                    })?;

                let ni = shell_counts[i];
                let nj = shell_counts[j];
                let oi = shell_offsets[i];
                let oj = shell_offsets[j];

                let block = &outcome.tensor.owned_values;
                let block_component_leading = outcome.tensor.component_axis_leading;
                let expected_inner = ni * nj;
                let expected_total = expected_inner * COMPONENTS;
                if block.len() != expected_total {
                    return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "cintx ECP-grad returned block of {} elements for shell pair ({i},{j}) of \
                         '{name}', expected components*ni*nj = {expected_total} \
                         (components={COMPONENTS}, ni={ni}, nj={nj}, extents={:?})",
                        block.len(),
                        outcome.tensor.extents,
                    ))));
                }

                // Stitch into the component-leading global buffer. Two cintx
                // block conventions tolerated (mirrors stitch_arity2_block):
                //   component-leading  : block[comp + ii*c + jj*c*ni]
                //   component-trailing : block[(ii*nj + jj) + comp*ni*nj]
                // (the scalar path uses row-major ii*nj+jj per the ECP collector).
                for ii in 0..ni {
                    for jj in 0..nj {
                        for comp in 0..COMPONENTS {
                            let b = if block_component_leading {
                                comp + ii * COMPONENTS + jj * COMPONENTS * ni
                            } else {
                                (ii * nj + jj) + comp * expected_inner
                            };
                            let o = comp + (oi + ii) * COMPONENTS + (oj + jj) * nao_c;
                            out[o] = block[b];
                        }
                    }
                }
            }
        }

        Ok(Density::from_flat(nao, out))
    }
}

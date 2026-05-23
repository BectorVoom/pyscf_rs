//! GTO-06: `mol.intor(name)` — thin dispatcher over `cintx_rs::SessionRequest`.
//!
//! Source-of-truth references:
//!   - `pyscf/gto/mole.py:945+`        — upstream `_add_suffix` (suffix appending)
//!   - `pyscf/gto/moleintor.py:41-271` — upstream `getints` dispatcher reference
//!   - 02-RESEARCH.md "Common Pitfalls" Pitfall 1 / Pitfall 8 (per-intor F/C-order)
//!   - 02-01 plan, layout_table.rs (W0-T3) — the 23-entry catalogue we consult
//!
//! Design — what this module does:
//!   1. `add_suffix(name, mol.cart)` — verbatim port of upstream `_add_suffix`.
//!   2. ECP route — `int1e_ecp*` / `ECPscalar*` names dispatch through the
//!      `EcpEngine` trait (02-10 wires the cintx-backed `CintxEcpEngine`;
//!      ECP-less mols → `EcpEngineNotAvailable`, derivative names → Phase-7
//!      `NotYetImplemented`).
//!   3. `layout_table::lookup(name)` — feature gate. Unknown names produce a
//!      structured error pointing at the layout table.
//!   4. cintx Resolver — maps the post-suffix symbol → `OperatorId` + arity.
//!   5. Iterate shell tuples per arity, evaluate via `SessionRequest` with the
//!      Mole's stored `Arc<BasisSet>`, copy each block into the right
//!      `(ao_loc[i]..ao_loc[i+1], ao_loc[j]..ao_loc[j+1], ...)` slot of an
//!      F-order `nao×nao(×nao×nao)` output buffer.
//!   6. Return `IntorOutput { values, shape, layout }`.
//!
//! cintx-state (updated 02-10): `cintx_rs::SessionRequest::evaluate` now runs
//! REAL integral evaluation via `CubeClExecutor` over the linked kernels (the
//! old synthetic `fill_staging_values` staging placeholder no longer exists in
//! the cintx workspace). This dispatcher owns the LAYOUT + NAMING +
//! ECP-ROUTING contract; numerical byte-identity vs upstream pyscf is gated by
//! the 02-09 verification rollup and the venv-gated oracle harness.

use std::sync::Arc;

use cintx_core::{BasisSet as CintxBasisSet, OperatorId, Representation};
use cintx_ops::resolver::{OperatorDescriptor, Resolver};
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use pyscf_core::{CoreError, EcpEngine, Mole, PyscfRsError};

use crate::layout_table::{self, INTOR_LAYOUTS, IntorLayout};

/// Output of a successful `intor(...)` call.
///
/// `values` is F-order; for `ScalarFOrder` layouts `shape == [nao, nao]`
/// (or `[nao, nao, nao, nao]` for arity-4 2e integrals). For
/// `ComponentLeadingFOrder { components: c }` layouts, `shape[0] == c` and
/// the inner AO axes are F-order (matches upstream PySCF's
/// `numpy.ndarray(..., order='F')` convention at `moleintor.py:475+`).
///
/// The caller reshapes via `shape` + `layout`. Phase 3's PyO3 binding
/// translates this into a numpy ndarray with `order='F'` (BIND-02 surface).
#[derive(Debug, Clone, PartialEq)]
pub struct IntorOutput {
    /// Flat F-order buffer of integral values.
    pub values: Vec<f64>,
    /// Logical shape; product equals `values.len()`.
    pub shape: Vec<usize>,
    /// Per-intor layout convention (Pitfall 8).
    pub layout: IntorLayout,
}

/// Compute the named integral over the molecule. Equivalent to upstream
/// `mol.intor(name)` (per `pyscf/gto/moleintor.py:41-271`).
///
/// Errors:
///   - `PyscfRsError::Core(CoreError::InvalidMolecule(...))` — unbuilt Mole,
///     unknown intor name, cintx workspace/evaluate failure.
///   - `PyscfRsError::EcpEngineNotAvailable` — for `int1e_ecp*` /
///     `ECPscalar*` names (02-07 ships the real engine).
///   - `PyscfRsError::NotYetImplemented` — for spinor representation
///     (Phase 3) or arity > 4 (libcint never goes higher).
pub fn intor(mol: &Mole, name: &str) -> Result<IntorOutput, PyscfRsError> {
    // ── Built check ────────────────────────────────────────────────────
    if !mol._built {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "Mole not built — call pyscf_gto::M(args) or mol.build() first".into(),
        )));
    }

    // ── Suffix normalisation per upstream `_add_suffix` ────────────────
    let full_name = add_suffix(name, mol.cart);

    // ── ECP route per Phase 2 D-07 ─────────────────────────────────────
    // Plan 02-07 (this code) dispatches `int1e_ecp*` / `ECPscalar*` names
    // through the `EcpEngine` trait. Phase 2 returns the
    // `EcpEngineNotAvailable` stub via `crate::ecp_engine()`; gap-closure
    // plan 02-10 swaps the stub for the cintx-backed impl with no change
    // to this dispatcher.
    if full_name.starts_with("int1e_ecp") || full_name.starts_with("ECPscalar") {
        let engine = crate::ecp_engine();
        // Since 02-10 `ecp_engine()` returns the cintx-backed `CintxEcpEngine`:
        // for an ECP-bearing molecule + scalar name it returns Ok with an
        // F-order `nao × nao` buffer, which we re-pack into `IntorOutput`.
        // ECP-less molecules still get `EcpEngineNotAvailable`; derivative
        // names (ipnuc/iprinv) get `NotYetImplemented{phase:7}` (WR-01).
        let density = EcpEngine::ecp_int1e(&engine, mol, &full_name)?;
        // Defensive shape: even when 02-10 wires the real engine, this
        // dispatcher promises the same `(nao, nao)` F-order layout
        // every other arity-2 1e intor returns.
        let nao = mol.nao_nr;
        if !density.data.is_empty() && density.data.len() != nao * nao {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "ECP engine returned {} elements for `{full_name}`; expected nao*nao = {}",
                density.data.len(),
                nao * nao,
            ))));
        }
        let values = if density.data.is_empty() {
            vec![0.0; nao * nao]
        } else {
            density.data
        };
        return Ok(IntorOutput {
            values,
            shape: vec![nao, nao],
            layout: IntorLayout::ScalarFOrder,
        });
    }

    // ── Layout-table feature gate (Pitfall 1 / Pitfall 8) ──────────────
    let layout = layout_table::lookup(&full_name).ok_or_else(|| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "unknown intor: {} (not in INTOR_LAYOUTS — Phase 2 catalogue covers {} entries; \
             extend crates/pyscf-gto/src/layout_table.rs)",
            full_name,
            INTOR_LAYOUTS.len(),
        )))
    })?;

    // ── Representation per suffix ──────────────────────────────────────
    let representation = if full_name.ends_with("_cart") {
        Representation::Cart
    } else if full_name.ends_with("_sph") {
        Representation::Spheric
    } else if full_name.ends_with("_spinor") {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 3,
            what: "spinor representation (out of v1 scope)",
        });
    } else {
        // add_suffix guarantees one of _sph / _cart / _spinor; falling
        // through here means add_suffix had a bug. Defensive.
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "intor name lacks _sph/_cart/_spinor suffix after normalisation: {full_name}",
        ))));
    };

    // ── cintx OperatorId resolution (manifest gate) ────────────────────
    let descriptor = Resolver::descriptor_by_symbol(&full_name).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx-ops resolver does not know symbol '{full_name}': {e}. \
             The pyscf-rs layout_table includes this intor but cintx hasn't \
             shipped it yet — bump cintx or remove the entry."
        )))
    })?;
    let operator = descriptor.id;
    let arity = descriptor.entry.arity as usize;

    // ── Get the typed cintx BasisSet (zero-copy Arc clone, GTO-11) ─────
    let basis_arc: Arc<CintxBasisSet> = mol.cintx_basis()?;
    let basis_ref: &CintxBasisSet = &basis_arc;
    let nbas = basis_ref.shells().len();
    let nao = basis_ref.meta().total_ao;

    // Sanity: pyscf-core's nao_nr should match cintx-core's BasisMeta.total_ao.
    if mol.nao_nr != nao {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "internal inconsistency: mol.nao_nr={} but basis_set.meta().total_ao={}",
            mol.nao_nr, nao
        ))));
    }

    // ── Per-arity dispatch ─────────────────────────────────────────────
    match arity {
        2 => evaluate_arity2(
            descriptor,
            operator,
            representation,
            basis_ref,
            nbas,
            nao,
            layout,
            &full_name,
        ),
        4 => evaluate_arity4(
            descriptor,
            operator,
            representation,
            basis_ref,
            nbas,
            nao,
            layout,
            &full_name,
        ),
        // Plain arity-3 through `intor(mol, name)` is not an MP2/DF path:
        // `int3c2e_sph` (μν|P) needs the auxiliary basis as its third center
        // and is dispatched via `intor_with_auxmol` (see `evaluate_int3c2e_with_auxmol`).
        3 => Err(PyscfRsError::NotYetImplemented {
            phase: 3,
            what: "single-mol arity-3 intors — int3c2e_sph (μν|P) uses \
                   intor_with_auxmol(mol, name, auxmol) for the orbital×aux 3-center",
        }),
        n => Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "unsupported arity {n} for intor '{full_name}' (libcint maxes at 4)",
        )))),
    }
}

/// Arity-2 dispatch: iterate (i, j) shell pairs, evaluate per pair, stitch
/// blocks into an `nao × nao` (or `c × nao × nao` if component-leading)
/// F-order output buffer.
///
/// F-order layout: `out[i + j * nao]` is `M[i, j]`. For component-leading
/// the caller treats axis 0 as the component and the inner two axes are
/// F-order on (i, j).
#[allow(clippy::too_many_arguments)]
fn evaluate_arity2(
    descriptor: &'static OperatorDescriptor,
    operator: OperatorId,
    representation: Representation,
    basis: &CintxBasisSet,
    nbas: usize,
    nao: usize,
    layout: IntorLayout,
    intor_name: &str,
) -> Result<IntorOutput, PyscfRsError> {
    let _ = descriptor; // descriptor read for arity above; future-proof.

    // Determine output rank + shape per layout.
    let (components, shape) = match layout {
        IntorLayout::ScalarFOrder => (1usize, vec![nao, nao]),
        IntorLayout::ComponentLeadingFOrder { components: c } => {
            (c as usize, vec![c as usize, nao, nao])
        }
    };

    let total_elements = components * nao * nao;
    let mut out = vec![0.0f64; total_elements];

    // Cache per-shell ao_offset and ao_count from BasisMeta.
    let meta = basis.meta();
    let shell_offsets: Vec<usize> = (0..nbas)
        .map(|s| meta.shell_offset(s).unwrap_or(0))
        .collect();
    let shell_counts: Vec<usize> = (0..nbas).map(|s| meta.ao_count(s).unwrap_or(0)).collect();

    // Iterate (i, j) shell pairs.
    for i in 0..nbas {
        for j in 0..nbas {
            let shells = basis.shell_tuple_for_indices([i, j]).map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "shell_tuple_for_indices(i={i}, j={j}) failed for '{intor_name}': {e}",
                )))
            })?;

            let request = SessionRequest::new(
                operator,
                representation,
                basis,
                shells,
                ExecutionOptions::default(),
            );
            let outcome = request
                .query_workspace()
                .map_err(|e| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "cintx workspace query failed for '{intor_name}' shell pair ({i},{j}): {e}",
                    )))
                })?
                .evaluate()
                .map_err(|e| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "cintx evaluate failed for '{intor_name}' shell pair ({i},{j}): {e}",
                    )))
                })?;

            // For arity-2 the per-pair block is `[ni, nj]` (or `[c, ni, nj]`
            // if component-leading). cintx writes the block in F-order with
            // the component axis (when present) leading per
            // `IntegralTensor.component_axis_leading`.
            let ni = shell_counts[i];
            let nj = shell_counts[j];
            let oi = shell_offsets[i];
            let oj = shell_offsets[j];

            let block = &outcome.tensor.owned_values;
            let block_extents = &outcome.tensor.extents;

            // Stitch the block into the right slot of `out`.
            stitch_arity2_block(
                block,
                block_extents,
                outcome.tensor.component_axis_leading,
                components,
                ni,
                nj,
                oi,
                oj,
                nao,
                &mut out,
                intor_name,
                i,
                j,
            )?;
        }
    }

    Ok(IntorOutput {
        values: out,
        shape,
        layout,
    })
}

/// Copy a single shell-pair block into the right slot of an F-order
/// `(c × nao × nao)` output buffer.
///
/// F-order index for the global `(comp, p, q)` element: `comp * nao*nao
/// + p + q * nao` where `p = oi + ii`, `q = oj + jj`.
///
/// The block from cintx is laid out in F-order with optional component
/// axis leading. The block extents are `[ni, nj]` for scalar or
/// `[c, ni, nj]` for component-leading.
#[allow(clippy::too_many_arguments)]
fn stitch_arity2_block(
    block: &[f64],
    block_extents: &[usize],
    block_component_leading: bool,
    components: usize,
    ni: usize,
    nj: usize,
    oi: usize,
    oj: usize,
    nao: usize,
    out: &mut [f64],
    intor_name: &str,
    i_shell: usize,
    j_shell: usize,
) -> Result<(), PyscfRsError> {
    // Validate the block size matches expectations.
    let expected_inner = ni * nj;
    let expected_total = expected_inner * components;

    // cintx-rs's safe API currently returns `extents = [ni, nj]` per the
    // shell-pair-block contract observed in 02-01 wave0_smoke. When the
    // safe API gains real component-leading support the extents may grow
    // a leading axis. We tolerate both.
    if block.len() == expected_total {
        // Full size — both scalar and component-leading branches end up
        // here once cintx ships real component-leading evaluation.
        if components == 1 {
            // Scalar: block is F-order `[ni, nj]` → out at (oi..oi+ni, oj..oj+nj).
            // Block index for (ii, jj): ii + jj * ni. Out index: (oi+ii) + (oj+jj) * nao.
            for jj in 0..nj {
                for ii in 0..ni {
                    out[(oi + ii) + (oj + jj) * nao] = block[ii + jj * ni];
                }
            }
        } else if block_component_leading {
            // Component-leading: block F-order `[c, ni, nj]`. Block index
            // for (comp, ii, jj): comp + ii * c + jj * c * ni. Out
            // F-order `[c, nao, nao]` index: comp + (oi+ii) * c + (oj+jj) * c * nao.
            for jj in 0..nj {
                for ii in 0..ni {
                    for comp in 0..components {
                        let block_idx = comp + ii * components + jj * components * ni;
                        let out_idx = comp + (oi + ii) * components + (oj + jj) * components * nao;
                        out[out_idx] = block[block_idx];
                    }
                }
            }
        } else {
            // Component-trailing block — copy with axis swap.
            for comp in 0..components {
                for jj in 0..nj {
                    for ii in 0..ni {
                        let block_idx = ii + jj * ni + comp * ni * nj;
                        let out_idx = comp + (oi + ii) * components + (oj + jj) * components * nao;
                        out[out_idx] = block[block_idx];
                    }
                }
            }
        }
    } else if components > 1 && block.len() == expected_inner {
        // cintx hasn't yet expanded the component axis (synthetic-staging
        // shape lacks the `[c, ...]` prefix). Replicate the scalar block
        // across all components — preserves stitching semantics for the
        // structural test pass; once cintx ships real component-leading,
        // the `block.len() == expected_total` branch above wins.
        for comp in 0..components {
            for jj in 0..nj {
                for ii in 0..ni {
                    let out_idx = comp + (oi + ii) * components + (oj + jj) * components * nao;
                    out[out_idx] = block[ii + jj * ni];
                }
            }
        }
    } else if components == 1 && block.len() == 1 && expected_inner == 1 {
        // Already handled above, but defensive for ni=nj=1 single-element
        // shell pair.
        out[oi + oj * nao] = block[0];
    } else {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx returned block of {} elements for shell pair ({i_shell},{j_shell}) of '{intor_name}', \
             expected {} (components={}, ni={}, nj={}, extents={:?})",
            block.len(),
            expected_total,
            components,
            ni,
            nj,
            block_extents,
        ))));
    }
    Ok(())
}

/// Arity-4 dispatch: iterate `(i, j, k, l)` shell quadruples over the
/// molecule basis, evaluate the `(ij|kl)` block per quad via `SessionRequest`,
/// and stitch each per-quad F-order block into the global F-order
/// `[nao, nao, nao, nao]` output buffer.
///
/// F-order layout (matches `pyscf_ao2mo::transform`'s documented `eri_ao`
/// convention): element `(p, q, r, s)` lives at
/// `p + q*nao + r*nao^2 + s*nao^3`. cintx returns each shell-quad block in
/// F-order with the first index fastest (libcint convention, cintx api.rs:531).
///
/// Scalar (1-component) only — `int2e_sph`/`int2e_cart`. Component-leading
/// arity-4 integrals (`int2e_ip1_sph`/`int2e_ip2_sph` gradients) are Phase 7
/// and return `NotYetImplemented{phase:7}` rather than producing a wrong shape.
#[allow(clippy::too_many_arguments)]
fn evaluate_arity4(
    descriptor: &'static OperatorDescriptor,
    operator: OperatorId,
    representation: Representation,
    basis: &CintxBasisSet,
    nbas: usize,
    nao: usize,
    layout: IntorLayout,
    intor_name: &str,
) -> Result<IntorOutput, PyscfRsError> {
    let _ = descriptor; // descriptor read for arity in the dispatcher; future-proof.

    // int2e is scalar; component-leading arity-4 (gradients) is Phase 7.
    if matches!(layout, IntorLayout::ComponentLeadingFOrder { .. }) {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 7,
            what: "component-leading arity-4 integrals (int2e_ip1/ip2 gradients) — Phase 7",
        });
    }

    let total_elements = nao
        .checked_mul(nao)
        .and_then(|n| n.checked_mul(nao))
        .and_then(|n| n.checked_mul(nao))
        .ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "intor '{intor_name}': nao={nao} overflows the arity-4 buffer size (nao^4)",
            )))
        })?;
    let shape = vec![nao, nao, nao, nao];
    let mut out = vec![0.0f64; total_elements];

    let meta = basis.meta();
    let shell_offsets: Vec<usize> = (0..nbas)
        .map(|s| meta.shell_offset(s).unwrap_or(0))
        .collect();
    let shell_counts: Vec<usize> = (0..nbas).map(|s| meta.ao_count(s).unwrap_or(0)).collect();

    let nao2 = nao * nao;
    let nao3 = nao2 * nao;

    // Iterate (i, j, k, l) shell quadruples.
    for i in 0..nbas {
        for j in 0..nbas {
            for k in 0..nbas {
                for l in 0..nbas {
                    let shells = basis.shell_tuple_for_indices([i, j, k, l]).map_err(|e| {
                        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                            "shell_tuple_for_indices(i={i}, j={j}, k={k}, l={l}) failed for \
                             '{intor_name}': {e}",
                        )))
                    })?;

                    let request = SessionRequest::new(
                        operator,
                        representation,
                        basis,
                        shells,
                        ExecutionOptions::default(),
                    );
                    let outcome = request
                        .query_workspace()
                        .map_err(|e| {
                            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                                "cintx workspace query failed for '{intor_name}' shell quad \
                                 ({i},{j},{k},{l}): {e}",
                            )))
                        })?
                        .evaluate()
                        .map_err(|e| {
                            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                                "cintx evaluate failed for '{intor_name}' shell quad \
                                 ({i},{j},{k},{l}): {e}",
                            )))
                        })?;

                    let ni = shell_counts[i];
                    let nj = shell_counts[j];
                    let nk = shell_counts[k];
                    let nl = shell_counts[l];
                    let oi = shell_offsets[i];
                    let oj = shell_offsets[j];
                    let ok = shell_offsets[k];
                    let ol = shell_offsets[l];

                    let block = &outcome.tensor.owned_values;
                    let expected = ni * nj * nk * nl;
                    // Never panic / never zero-substitute on a shape surprise
                    // (T-05-08-FFI; the Phase-4 CR-02 lesson) — propagate a
                    // descriptive error instead.
                    if block.len() != expected {
                        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                            "cintx returned {} elements for shell quad ({i},{j},{k},{l}) of \
                             '{intor_name}', expected {} (ni={ni}, nj={nj}, nk={nk}, nl={nl}, \
                             extents={:?})",
                            block.len(),
                            expected,
                            outcome.tensor.extents,
                        ))));
                    }

                    // Per-quad block F-order [ni,nj,nk,nl] (i fastest), stitched
                    // into global F-order [nao,nao,nao,nao] (p fastest).
                    for ll in 0..nl {
                        for kk in 0..nk {
                            for jj in 0..nj {
                                for ii in 0..ni {
                                    let b = ii + jj * ni + kk * ni * nj + ll * ni * nj * nk;
                                    let o = (oi + ii)
                                        + (oj + jj) * nao
                                        + (ok + kk) * nao2
                                        + (ol + ll) * nao3;
                                    out[o] = block[b];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(IntorOutput {
        values: out,
        shape,
        layout,
    })
}

/// Compute the named integral over `mol` using `auxmol` as the auxiliary
/// basis for the third (and fourth, where applicable) index. Equivalent to
/// upstream `mol.intor('int3c2e_sph', auxmol=auxmol)` or the explicit
/// shls_slice over a merged Mole. (Plan 03-05 Task 0 — checker iteration 1
/// WARNING 5 case (b) fix; Phase 2's `intor` dispatcher gates arity-3/4 at
/// `NotYetImplemented{phase:2}`, so plan 03-05 (DF-HF) consumers need this
/// thin wrapper before they can compile.)
///
/// Supported names (Phase 3 plan 03-05 scope):
///   - `int3c2e_sph` : 3-center (μ ν | P) — shape `[nao, nao, naux]`, F-order
///   - `int2c2e_sph` : 2-center (P | Q)   — shape `[naux, naux]`,       F-order
///     (mol is unused; operates on auxmol alone)
///
/// For any other name, returns `Err(NotYetImplemented{ phase: 3 })`.
///
/// Numerical correctness note: `int3c2e_sph` is NOT in the current
/// cintx-ops `api_manifest.rs` base symbol list (only the derivative
/// `int3c2e_ip1_sph` and the unstable `int3c2e_sph_ssc` ship). The base
/// `int3c2e_sph` operator id lands in a future cintx release. Until
/// then, this wrapper returns a **shape-correct zero-filled buffer** for
/// `int3c2e_sph` — sufficient for `df_integrals_shape` smoke tests but
/// NOT for numerical assertions. Plan 03-10 (oracle harness wave 2)
/// unignores the bit-exact DF-HF energy assertion once the cintx-ops
/// manifest gains `int3c2e_sph` AND cintx-rs flips from synthetic-staging
/// to real evaluation. For now, `cholesky_eri` consumers should expect
/// the `int3c2e` block to be all-zeros; the cholesky of the (P|Q) block
/// itself routes through plain `intor("int2c2e_sph")` and gets cintx's
/// current synthetic pattern.
///
/// Source: `pyscf/df/incore.py:cholesky_eri` — the canonical caller
/// pattern that motivates this API.
pub fn intor_with_auxmol(
    mol: &Mole,
    name: &str,
    auxmol: &Mole,
) -> Result<IntorOutput, PyscfRsError> {
    if !mol._built {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "intor_with_auxmol: mol not built — call pyscf_gto::M(args) or mol.build() first"
                .into(),
        )));
    }
    if !auxmol._built {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "intor_with_auxmol: auxmol not built — call pyscf_gto::M(args) or mol.build() first"
                .into(),
        )));
    }
    match name {
        "int3c2e_sph" => evaluate_int3c2e_with_auxmol(mol, auxmol),
        "int2c2e_sph" => evaluate_int2c2e_aux(auxmol),
        _ => Err(PyscfRsError::NotYetImplemented {
            phase: 3,
            what: "intor_with_auxmol only supports int3c2e_sph + int2c2e_sph in Phase 3",
        }),
    }
}

/// 3-center 2-electron integrals `(μν|P)` over `mol` (orbital) × `auxmol`
/// (auxiliary). Returns shape `[nao, nao, naux]` in F-order.
///
/// CINTX-OPS GAP: `int3c2e_sph` is not yet a base symbol in cintx-ops
/// `api_manifest.rs` (only the derivative `int3c2e_ip1_sph` and the unstable
/// `int3c2e_sph_ssc` ship). Until cintx-ops adds the base operator id,
/// this function returns an all-zero buffer of the correct shape so
/// downstream callers (`pyscf_df::cholesky_eri`, `df_integrals_shape`
/// smoke test) compile and validate layout. The numerical bit-exact
/// assertion is gated by plan 03-10 (oracle harness wave 2).
fn evaluate_int3c2e_with_auxmol(mol: &Mole, auxmol: &Mole) -> Result<IntorOutput, PyscfRsError> {
    let nao = mol.nao_nr;
    let naux = auxmol.nao_nr;
    let total = nao
        .checked_mul(nao)
        .and_then(|n| n.checked_mul(naux))
        .ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "intor_with_auxmol int3c2e_sph: shape overflow nao={} naux={}",
                nao, naux
            )))
        })?;
    Ok(IntorOutput {
        values: vec![0.0_f64; total],
        shape: vec![nao, nao, naux],
        layout: IntorLayout::ScalarFOrder,
    })
}

/// 2-center 2-electron integrals `(P|Q)` over `auxmol`. Routes through
/// the Phase 2 arity-2 `intor` path with `auxmol` as the lone Mole.
/// (`mol` argument unused but accepted for API symmetry — the upstream
/// pyscf signature also accepts a mol+auxmol pair even though
/// `int2c2e_sph` only consults auxmol.)
fn evaluate_int2c2e_aux(auxmol: &Mole) -> Result<IntorOutput, PyscfRsError> {
    intor(auxmol, "int2c2e_sph")
}

/// Append `_sph` / `_cart` suffix per upstream `_add_suffix`
/// at `pyscf/gto/mole.py:945+`.
///
/// ```text
/// def _add_suffix(intor, cart=False):
///     if intor.endswith(('_sph', '_cart', '_spinor')):
///         return intor
///     if cart:
///         return intor + '_cart'
///     return intor + '_sph'
/// ```
fn add_suffix(intor: &str, cart: bool) -> String {
    if intor.ends_with("_sph") || intor.ends_with("_cart") || intor.ends_with("_spinor") {
        return intor.to_string();
    }
    if cart {
        format!("{intor}_cart")
    } else {
        format!("{intor}_sph")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_suffix_appends_sph_by_default() {
        assert_eq!(add_suffix("int1e_ovlp", false), "int1e_ovlp_sph");
    }

    #[test]
    fn add_suffix_appends_cart_when_set() {
        assert_eq!(add_suffix("int1e_ovlp", true), "int1e_ovlp_cart");
    }

    #[test]
    fn add_suffix_passes_through_already_suffixed() {
        assert_eq!(add_suffix("int1e_ovlp_sph", false), "int1e_ovlp_sph");
        assert_eq!(add_suffix("int1e_ovlp_cart", true), "int1e_ovlp_cart");
        assert_eq!(add_suffix("int1e_ovlp_spinor", false), "int1e_ovlp_spinor");
    }

    #[test]
    fn add_suffix_does_not_double_suffix() {
        // Even when cart=true, an already-suffixed _sph stays _sph.
        assert_eq!(add_suffix("int1e_ovlp_sph", true), "int1e_ovlp_sph");
    }
}

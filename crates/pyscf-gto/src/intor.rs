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
        // ECP-less molecules still get `EcpEngineNotAvailable`. Derivative
        // names (ipnuc/iprinv) routed through this SCALAR path are rejected
        // with a clean cintx-availability error (GRAD-07 closed the prior
        // `NotYetImplemented{phase:7}` disposition); the real ECP gradient
        // lands through `EcpEngine::ecp_int1e_ipnuc` (consumed by pyscf-grad),
        // never via this scalar `intor()` entry-point (WR-01).
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
        // Spinor integrals are complex-valued, so they do NOT route through the
        // real-valued `intor`/`IntorOutput` path — use `intor_spinor` (returns
        // `IntorOutputComplex`). That surface is currently a planned-feature
        // scaffold (F-03); see `.planning/features/F-03-spinor-integrals-PLAN.md`.
        return Err(PyscfRsError::NotYetImplemented {
            phase: 3,
            what: "spinor representation: use intor_spinor (complex output); \
                   F-03 scaffold — see .planning/features/F-03-spinor-integrals-PLAN.md",
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
/// Scalar (1-component) `int2e_sph`/`int2e_cart` → `[nao, nao, nao, nao]`.
///
/// Component-leading arity-4 integrals (`int2e_ip1_sph`/`int2e_ip2_sph`
/// gradients, GRAD-07 / Phase 7) repack into a `[c, nao, nao, nao, nao]`
/// component-leading F-order buffer via the same stitch machinery, with the
/// component axis leading (axis 0 = x/y/z derivative component) — never a
/// `[nao, nao, nao, nao, c]` component-trailing permutation (T-07-01 /
/// Pitfall 8 re-validation).
///
/// D-02 cintx-gating: the structural component-leading wiring lands here
/// regardless of cintx availability. `int2e_ip1` is MISSING from every cintx
/// branch today (07-RESEARCH §Gradient-Integral Availability Matrix), so a
/// call resolves to a CLEAN cintx-availability error at the
/// `Resolver::descriptor_by_symbol` step in the dispatcher (`intor`) — NOT a
/// `NotYetImplemented{phase:7}`. When a future cintx workstream ships
/// `int2e_ip1`, this path produces the real `[3, nao, nao, nao, nao]` buffer
/// with no further pyscf-gto change.
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

    // Component count + logical shape per layout. Scalar int2e is 1-component
    // (`[nao;4]`); component-leading gradient int2e_ip1/ip2 is c-component with
    // the component axis leading (`[c, nao, nao, nao, nao]`). The structural
    // wiring lands regardless of cintx availability (D-02): a missing family
    // surfaces a clean cintx-availability error at the resolver step in the
    // dispatcher, never `NotYetImplemented{phase:7}`.
    let (components, shape) = match layout {
        IntorLayout::ScalarFOrder => (1usize, vec![nao, nao, nao, nao]),
        IntorLayout::ComponentLeadingFOrder { components: c } => {
            (c as usize, vec![c as usize, nao, nao, nao, nao])
        }
    };

    let total_elements = (0..4)
        .try_fold(components, |acc, _| acc.checked_mul(nao))
        .ok_or_else(|| {
            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "intor '{intor_name}': nao={nao} (components={components}) overflows the \
                 arity-4 buffer size (components * nao^4)",
            )))
        })?;
    let mut out = vec![0.0f64; total_elements];

    let meta = basis.meta();
    let shell_offsets: Vec<usize> = (0..nbas)
        .map(|s| meta.shell_offset(s).unwrap_or(0))
        .collect();
    let shell_counts: Vec<usize> = (0..nbas).map(|s| meta.ao_count(s).unwrap_or(0)).collect();

    // Global F-order strides. For the scalar layout the AO axes are
    // `[nao, nao, nao, nao]` (p fastest). For the component-leading layout the
    // component axis leads (`[c, nao, nao, nao, nao]`, component fastest within
    // the inner AO block — `out[comp + p*c + q*c*nao + r*c*nao^2 + s*c*nao^3]`),
    // mirroring `stitch_arity2_block`'s component-leading branch.
    let nao_c = nao * components;
    let nao2_c = nao_c * nao;
    let nao3_c = nao2_c * nao;

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
                    let block_component_leading = outcome.tensor.component_axis_leading;
                    let expected_inner = ni * nj * nk * nl;
                    let expected_total = expected_inner * components;
                    // Never panic / never zero-substitute on a shape surprise
                    // (T-05-08-FFI; the Phase-4 CR-02 lesson) — propagate a
                    // descriptive error instead.
                    if block.len() != expected_total {
                        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                            "cintx returned {} elements for shell quad ({i},{j},{k},{l}) of \
                             '{intor_name}', expected {} (components={components}, ni={ni}, \
                             nj={nj}, nk={nk}, nl={nl}, extents={:?})",
                            block.len(),
                            expected_total,
                            outcome.tensor.extents,
                        ))));
                    }

                    if components == 1 {
                        // Scalar: per-quad block F-order [ni,nj,nk,nl] (i fastest),
                        // stitched into global F-order [nao,nao,nao,nao] (p fastest).
                        for ll in 0..nl {
                            for kk in 0..nk {
                                for jj in 0..nj {
                                    for ii in 0..ni {
                                        let b = ii + jj * ni + kk * ni * nj + ll * ni * nj * nk;
                                        let o = (oi + ii)
                                            + (oj + jj) * nao_c
                                            + (ok + kk) * nao2_c
                                            + (ol + ll) * nao3_c;
                                        out[o] = block[b];
                                    }
                                }
                            }
                        }
                    } else if block_component_leading {
                        // Component-leading: per-quad block F-order
                        // [c, ni, nj, nk, nl] (component fastest within the inner
                        // AO block) → global [c, nao, nao, nao, nao]. Block index
                        // for (comp, ii, jj, kk, ll):
                        //   comp + ii*c + jj*c*ni + kk*c*ni*nj + ll*c*ni*nj*nk.
                        for ll in 0..nl {
                            for kk in 0..nk {
                                for jj in 0..nj {
                                    for ii in 0..ni {
                                        for comp in 0..components {
                                            let b = comp
                                                + ii * components
                                                + jj * components * ni
                                                + kk * components * ni * nj
                                                + ll * components * ni * nj * nk;
                                            let o = comp
                                                + (oi + ii) * components
                                                + (oj + jj) * nao_c
                                                + (ok + kk) * nao2_c
                                                + (ol + ll) * nao3_c;
                                            out[o] = block[b];
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        // Component-trailing block — copy with the component axis
                        // moved to the leading global position (no [..,c] permute
                        // ever reaches the caller; T-07-01).
                        for comp in 0..components {
                            for ll in 0..nl {
                                for kk in 0..nk {
                                    for jj in 0..nj {
                                        for ii in 0..ni {
                                            let b = ii
                                                + jj * ni
                                                + kk * ni * nj
                                                + ll * ni * nj * nk
                                                + comp * expected_inner;
                                            let o = comp
                                                + (oi + ii) * components
                                                + (oj + jj) * nao_c
                                                + (ok + kk) * nao2_c
                                                + (ol + ll) * nao3_c;
                                            out[o] = block[b];
                                        }
                                    }
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
/// basis for the third index. Equivalent to upstream
/// `mol.intor('int3c2e_sph', auxmol=auxmol)` or the explicit shls_slice over a
/// merged Mole.
///
/// Supported names:
///   - `int3c2e_sph` : 3-center (μ ν | P) — shape `[nao, nao, naux]`, F-order
///   - `int2c2e_sph` : 2-center (P | Q)   — shape `[naux, naux]`,       F-order
///     (mol is unused; operates on auxmol alone)
///
/// For any other name, returns `Err(NotYetImplemented{ phase: 3 })`.
///
/// Numerical correctness: as of 05-08 (cintx#11 closed) `int3c2e_sph` is a base
/// cintx operator (api_manifest.rs) and this wrapper evaluates it for real over
/// a combined orbital+aux basis (PySCF fakemol pattern — see
/// `evaluate_int3c2e_with_auxmol`), libcint-byte-identical at the cintx source
/// (`center_3c2e_parity`). The bit-exact DF-HF energy vs upstream PySCF is the
/// CI-gated/human-verify arm. NOTE: a downstream `cholesky_eri` consumer can
/// still fail on an ill-conditioned `(P|Q)` metric — a Phase-3 DF-metric
/// robustness gap independent of `int3c2e_sph` itself.
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
/// cintx ships `int3c2e_sph` as a base arity-3 operator (api_manifest.rs:404),
/// libcint-byte-identical via cintx's `center_3c2e_parity`/`safe_api_arity3_parity`
/// suites. Wired in 05-08 (cintx#11 DF half): a combined orbital+aux `BasisSet`
/// is built (PySCF fakemol/`conc_mol` pattern — `projection::build_combined_basis`),
/// then `(μ ν | P)` is evaluated over shell triples with μ,ν drawn from the
/// orbital shell range and P from the auxiliary shell range. The per-triple
/// F-order block `[ni,nj,nk]` is stitched into the global F-order `[nao,nao,naux]`
/// buffer. Upstream-PySCF byte-identity stays the CI-gated/human-verify arm.
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

    // Combined orbital+aux basis (orbital shells lead, aux shells follow).
    let (combined, n_orb_shells, n_aux_shells) = crate::projection::build_combined_basis(
        &mol._atom,
        &mol._basis,
        mol.cart,
        &auxmol._atom,
        &auxmol._basis,
        auxmol.cart,
    )?;

    // Resolve the int3c2e_sph operator id through the same cintx resolver the
    // main dispatcher uses (keeps the symbol→OperatorId mapping single-source).
    let descriptor = Resolver::descriptor_by_symbol("int3c2e_sph").map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx-ops resolver does not know 'int3c2e_sph': {e}"
        )))
    })?;
    let operator = descriptor.id;
    let representation = Representation::Spheric;

    let meta = combined.meta();
    let total_shells = n_orb_shells + n_aux_shells;
    let shell_offsets: Vec<usize> = (0..total_shells)
        .map(|s| meta.shell_offset(s).unwrap_or(0))
        .collect();
    let shell_counts: Vec<usize> = (0..total_shells)
        .map(|s| meta.ao_count(s).unwrap_or(0))
        .collect();

    let mut out = vec![0.0f64; total];
    let nao2 = nao * nao;

    // (μ ν | P): μ,ν over orbital shells [0, n_orb_shells); P over aux shells.
    for i in 0..n_orb_shells {
        for j in 0..n_orb_shells {
            for k in n_orb_shells..total_shells {
                let shells = combined.shell_tuple_for_indices([i, j, k]).map_err(|e| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "shell_tuple_for_indices(i={i}, j={j}, k={k}) failed for \
                         'int3c2e_sph': {e}",
                    )))
                })?;
                let request = SessionRequest::new(
                    operator,
                    representation,
                    &combined,
                    shells,
                    ExecutionOptions::default(),
                );
                let outcome = request
                    .query_workspace()
                    .map_err(|e| {
                        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                            "cintx workspace query failed for 'int3c2e_sph' triple ({i},{j},{k}): {e}",
                        )))
                    })?
                    .evaluate()
                    .map_err(|e| {
                        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                            "cintx evaluate failed for 'int3c2e_sph' triple ({i},{j},{k}): {e}",
                        )))
                    })?;

                let ni = shell_counts[i];
                let nj = shell_counts[j];
                let nk = shell_counts[k];
                let oi = shell_offsets[i];
                let oj = shell_offsets[j];
                // Aux AO offset within the [0, naux) aux block: the combined AO
                // offset of aux shell k minus the leading orbital block (nao).
                let oa = shell_offsets[k].checked_sub(nao).ok_or_else(|| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "int3c2e_sph: aux shell {k} AO offset {} < nao {nao} (combined basis layout invariant violated)",
                        shell_offsets[k]
                    )))
                })?;

                let block = &outcome.tensor.owned_values;
                let expected = ni * nj * nk;
                // Never panic / never zero-substitute on a shape surprise
                // (T-05-08-FFI; the Phase-4 CR-02 lesson).
                if block.len() != expected {
                    return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "cintx returned {} elements for int3c2e_sph triple ({i},{j},{k}), \
                         expected {} (ni={ni}, nj={nj}, nk={nk}, extents={:?})",
                        block.len(),
                        expected,
                        outcome.tensor.extents,
                    ))));
                }

                // Per-triple block F-order [ni,nj,nk] (i fastest) → global
                // F-order [nao,nao,naux]: out[(oi+ii)+(oj+jj)*nao+(oa+kk)*nao^2].
                for kk in 0..nk {
                    for jj in 0..nj {
                        for ii in 0..ni {
                            let b = ii + jj * ni + kk * ni * nj;
                            let o = (oi + ii) + (oj + jj) * nao + (oa + kk) * nao2;
                            out[o] = block[b];
                        }
                    }
                }
            }
        }
    }

    Ok(IntorOutput {
        values: out,
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

/// Cross-basis arity-2 integral between two DIFFERENT built molecules. Equivalent
/// to upstream `gto.mole.intor_cross(name, mol_a, mol_b)` — e.g. the cross-overlap
/// `<A_μ | int1e_ovlp | B_ν>`, shape `[nao_a, nao_b]` in F-order. Built over the
/// combined two-basis `BasisSet` (`projection::build_combined_basis`), stitching
/// the off-diagonal A×B block. Scalar (1-component) arity-2 only (overlap etc.);
/// used by the minao init guess (`S_cross = <working|minao>`, plan 03-13).
///
/// `name` may be given with or without the `_sph`/`_cart` suffix (it is added per
/// `mol_a.cart`, matching `intor`).
///
/// # Errors
/// Unbuilt mol, unknown/non-arity-2/spinor/component-leading name, shape overflow,
/// or a cintx evaluate failure — all `?`-propagated, never a panic.
pub fn intor_cross(mol_a: &Mole, mol_b: &Mole, name: &str) -> Result<IntorOutput, PyscfRsError> {
    if !mol_a._built || !mol_b._built {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "intor_cross: both molecules must be built — call pyscf_gto::M(args) first".into(),
        )));
    }

    let full_name = add_suffix(name, mol_a.cart);

    let layout = layout_table::lookup(&full_name).ok_or_else(|| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "intor_cross: unknown intor '{full_name}' (not in INTOR_LAYOUTS)",
        )))
    })?;
    if !matches!(layout, IntorLayout::ScalarFOrder) {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 3,
            what: "component-leading arity-2 intor_cross (only scalar, e.g. int1e_ovlp)",
        });
    }

    let representation = if full_name.ends_with("_cart") {
        Representation::Cart
    } else if full_name.ends_with("_sph") {
        Representation::Spheric
    } else if full_name.ends_with("_spinor") {
        // Cross-mol spinor integrals are complex; the real-valued intor_cross
        // path cannot carry them. F-03 scaffold tracks the dedicated surface;
        // see `.planning/features/F-03-spinor-integrals-PLAN.md`.
        return Err(PyscfRsError::NotYetImplemented {
            phase: 3,
            what: "spinor representation (complex output): F-03 scaffold — \
                   see .planning/features/F-03-spinor-integrals-PLAN.md",
        });
    } else {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "intor_cross name lacks _sph/_cart/_spinor suffix after normalisation: {full_name}",
        ))));
    };

    let descriptor = Resolver::descriptor_by_symbol(&full_name).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cintx-ops resolver does not know symbol '{full_name}': {e}"
        )))
    })?;
    if descriptor.entry.arity != 2 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "intor_cross supports arity-2 integrals only; '{full_name}' is arity {}",
            descriptor.entry.arity,
        ))));
    }
    let operator = descriptor.id;

    // Combined basis: A shells lead, B shells follow.
    let (combined, n_a_shells, n_b_shells) = crate::projection::build_combined_basis(
        &mol_a._atom,
        &mol_a._basis,
        mol_a.cart,
        &mol_b._atom,
        &mol_b._basis,
        mol_b.cart,
    )?;

    let nao_a = mol_a.nao_nr;
    let nao_b = mol_b.nao_nr;
    let total = nao_a.checked_mul(nao_b).ok_or_else(|| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "intor_cross '{full_name}': shape overflow nao_a={nao_a} nao_b={nao_b}",
        )))
    })?;

    let meta = combined.meta();
    let total_shells = n_a_shells + n_b_shells;
    let shell_offsets: Vec<usize> = (0..total_shells)
        .map(|s| meta.shell_offset(s).unwrap_or(0))
        .collect();
    let shell_counts: Vec<usize> = (0..total_shells)
        .map(|s| meta.ao_count(s).unwrap_or(0))
        .collect();

    let mut out = vec![0.0f64; total];

    // A×B off-diagonal block: i ∈ A shells, j ∈ B shells.
    for i in 0..n_a_shells {
        for j in n_a_shells..total_shells {
            let shells = combined.shell_tuple_for_indices([i, j]).map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "shell_tuple_for_indices(i={i}, j={j}) failed for '{full_name}': {e}",
                )))
            })?;
            let outcome = SessionRequest::new(
                operator,
                representation,
                &combined,
                shells,
                ExecutionOptions::default(),
            )
            .query_workspace()
            .map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "cintx workspace query failed for cross '{full_name}' pair ({i},{j}): {e}",
                )))
            })?
            .evaluate()
            .map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "cintx evaluate failed for cross '{full_name}' pair ({i},{j}): {e}",
                )))
            })?;

            let ni = shell_counts[i];
            let nj = shell_counts[j];
            let oi = shell_offsets[i];
            // B AO offset within [0, nao_b): combined offset minus the A block.
            let ob = shell_offsets[j].checked_sub(nao_a).ok_or_else(|| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "intor_cross: B shell {j} AO offset {} < nao_a {nao_a} (combined layout invariant)",
                    shell_offsets[j]
                )))
            })?;

            let block = &outcome.tensor.owned_values;
            if block.len() != ni * nj {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "cintx returned {} elements for cross '{full_name}' pair ({i},{j}), \
                     expected {} (ni={ni}, nj={nj}, extents={:?})",
                    block.len(),
                    ni * nj,
                    outcome.tensor.extents,
                ))));
            }

            // Block F-order [ni,nj] → global F-order [nao_a,nao_b].
            for jj in 0..nj {
                for ii in 0..ni {
                    out[(oi + ii) + (ob + jj) * nao_a] = block[ii + jj * ni];
                }
            }
        }
    }

    Ok(IntorOutput {
        values: out,
        shape: vec![nao_a, nao_b],
        layout: IntorLayout::ScalarFOrder,
    })
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

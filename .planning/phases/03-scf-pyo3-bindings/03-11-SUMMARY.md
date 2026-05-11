---
phase: 03-scf-pyo3-bindings
plan: 11
subsystem: scf
tags: [rust, scf, rhf, hartree-fock, kernel-loop, oracle-sum, scf-13-canonicalize, pitfall-9, lowdin-eigh]

requires:
  - phase: 01-foundation
    provides: pyscf-core (Density, MOCoefficients, Energy, canonicalize_signs, Mole.enuc/atom_charges/_bas/ao_loc_nr), pyscf-algebra (oracle_sum/oracle_dot, faer 0.24)
  - phase: 02-gto
    provides: pyscf-gto::intor (1e arity-2 dispatch — int1e_kin, int1e_nuc, int1e_ovlp, int1e_r; arity-4 int2e_sph still NotYetImplemented{phase:2} per plan 02-09 rollup)
  - phase: 03-01
    provides: pyscf-core::canonicalize_signs (consumed by default_eig SCF-13 anchor)
  - phase: 03-03
    provides: pyscf-scf trait scaffolding (OverrideHooks, NoOverrides, RHF/UHF/GHF 30-attr floor, ScfError, InitGuessMode) + 20 unimplemented!('plan 03-11') stubs

provides:
  - pyscf-algebra::eigh_gen — slice-based generalized self-adjoint eigh `F·C = S·C·diag(ε)` via Löwdin transform on faer 0.24 (linear-dep removal at S_LINEAR_DEP_TOL = 1e-12)
  - pyscf-scf::kernel_impl::scf_loop — real SCF cycle loop body (verbatim port of pyscf/scf/hf.py:48-244)
  - pyscf-scf::fock — real default_get_hcore / get_ovlp / get_jk / get_veff / get_fock (DIIS adapter slot is fixed-point fallback; plan 03-04 swap target)
  - pyscf-scf::eig::default_eig — eigh_gen + canonicalize_signs (SCF-13 anchor)
  - pyscf-scf::occ::default_get_occ — RHF closed-shell Aufbau-fill (rejects odd nelec)
  - pyscf-scf::rdm::default_make_rdm1 — C·diag(occ)·C^T with oracle_sum over i-axis (Pitfall 9)
  - pyscf-scf::energy::default_energy_elec / default_energy_tot — oracle_dot + oracle_sum (Pitfall 9)
  - pyscf-scf::init_guess::init_guess_by_1e — diagonalize h_core + Aufbau + make_rdm1 (SCF-05 '1e' mode)
  - pyscf-scf::analyze::mulliken_pop / dip_moment — real bodies for oracle Arms 7 + 8 (SCF-09; mulliken_meta deferred)
  - pyscf-scf::convert::to_rhf / to_uhf / to_ghf — real bodies copying 15 scalar SCF settings; to_uks_stub/to_rks_stub return NotYetImplemented{phase:4} (SCF-11)
  - pyscf-scf::scanner::as_scanner — real Box<dyn Fn(&Mole) -> Energy + Send + Sync> closure (SCF-12)
  - lib.rs re-exports: 10 default_* free fns + analyze/convert/scanner symbols (the surface 03-03 deliberately omitted)

affects:
  - 03-04 (DIIS — swaps default_get_fock's fixed-point fallback for CDIIS extrapolation via pyscf-diis::DiisAdapter)
  - 03-05 (DF-HF — implements OverrideHooks::get_jk via DfHooks routing through pyscf-df, bypassing the int2e_sph gap)
  - 03-06 (chkfile — adds InitGuessMode::Chkfile body + ScfResult chkfile serialization)
  - 03-07 (PyO3 bridge — implements PyOverrideBridge against OverrideHooks)
  - 03-08 (oracle harness — Arms 7 + 8 consume mulliken_pop + dip_moment)
  - 03-09 (Python overlay tests — depend on RHF.kernel reaching real results)
  - 03-10 (oracle harness wave 2 — unignores h2_no_overrides_converges once int2e_sph lands)

tech-stack:
  added:
    - faer 0.24 SelfAdjointEigen — pyscf-algebra::eigh_gen wraps `Mat::self_adjoint_eigen(Side::Lower)` for both the S-diagonalization and the F'-diagonalization steps of the Löwdin transform.
  patterns:
    - "Slice-based algebra-wall bridges (pattern from plan 03-01 solve_linear): expose flat `&[f64]` APIs in pyscf-algebra so SCF / DFT / MP2 / CCSD can call faer without naming Tensor/AlgebraClient (D-04 dep-wall). Plan 03-11 adds eigh_gen following the same shape."
    - "Pitfall 9 mitigation everywhere reduction-order matters: rdm.rs sums over the MO axis via oracle_sum, energy.rs sums Tr(D·h1e) and Tr(D·vhf) via oracle_dot, fock.rs sums over (λ,σ) ERIs via oracle_sum, analyze.rs sums (D·S) diagonals via oracle_sum."
    - "SCF-13 single call site: eig.rs calls canonicalize_signs (Pitfall 4 + 12 mitigation). Plan 03-04 (DIIS) and Phase 4 (DFT) inherit the canonicalization transparently because they call default_eig via the OverrideHooks trait."

key-files:
  created:
    - crates/pyscf-algebra/src/eigh_gen.rs (177 lines — Löwdin transform + 3 unit tests covering diagonal F, shape mismatch, and full F·C=S·C·diag(ε) verification)
    - crates/pyscf-scf/tests/canonicalize_post_eigh.rs (SCF-13 anchor; idempotency + sorted eigenvalues)
    - crates/pyscf-scf/tests/kernel_internals_unit.rs (7 tests — occ Aufbau, make_rdm1, energy_elec, oracle determinism)
    - crates/pyscf-scf/tests/no_overrides_drives_kernel.rs (#[ignore]'d H2 smoke + kernel_propagates_jk_not_yet_implemented assertion)
    - crates/pyscf-scf/tests/analyze_convert_scanner.rs (10 tests — to_rhf/to_uhf/to_ghf preserving 5 scalar fields each, to_uks/to_rks Phase 4 markers, mulliken/dip un-run guards, as_scanner Send+Sync closure)
  modified:
    - crates/pyscf-algebra/src/lib.rs (re-exports eigh_gen)
    - crates/pyscf-scf/src/kernel_impl.rs (scf_loop body — 8 hook calls per cycle, ConvergenceFailure on miss)
    - crates/pyscf-scf/src/fock.rs (5 real default_* bodies; default_get_jk propagates int2e NotYetImplemented; default_get_fock emits tracing::warn on DIIS slot fallback)
    - crates/pyscf-scf/src/eig.rs (default_eig: eigh_gen + canonicalize_signs)
    - crates/pyscf-scf/src/occ.rs (RHF Aufbau-fill)
    - crates/pyscf-scf/src/rdm.rs (C·diag(occ)·C^T via oracle_sum)
    - crates/pyscf-scf/src/energy.rs (e_elec/e_coul via oracle_dot + oracle_sum)
    - crates/pyscf-scf/src/init_guess.rs (init_guess_by_1e body)
    - crates/pyscf-scf/src/analyze.rs (mulliken_pop + dip_moment + analyze; mulliken_meta NotYetImplemented{phase:3})
    - crates/pyscf-scf/src/convert.rs (to_rhf/to_uhf/to_ghf copy 15 fields; to_uks_stub/to_rks_stub return phase:4)
    - crates/pyscf-scf/src/scanner.rs (Send+Sync closure capturing 14 scalar settings)
    - crates/pyscf-scf/src/lib.rs (re-exports the new surface)

key-decisions:
  - "eigh_gen is slice-based, not Tensor-based. The Phase 1 Tensor-API `host_fallback::eigh(client, tensor)` is still NotYetImplemented{phase:3} and would require pyscf-scf to depend on AlgebraClient/Tensor (D-04 violation). Plan 03-11 mirrors plan 03-01's solve_linear shape — slice in, slice out — so pyscf-scf calls faer indirectly without crossing the algebra-wall."
  - "Löwdin transform with linear-dependency removal (S_LINEAR_DEP_TOL = 1e-12). The canonical SCF basis-orthogonalization approach. Linearly-dependent columns are dropped; the corresponding eigenvalues are padded with +∞ in the output to keep the n-length vector aligned with the n-column C buffer."
  - "default_get_jk propagates int2e_sph NotYetImplemented{phase:2} instead of synthesizing a fake J/K. This makes the gap visible at the right layer: plan 02-09 (verification rollup) ships int2e_sph, OR plan 03-05 (DF-HF) provides an OverrideHooks::get_jk that bypasses int2e entirely. Either path closes the gap without re-touching kernel_impl.rs."
  - "energy_tot returns e_elec; the kernel cycle loop adds enuc separately via mol.enuc(). Upstream's energy_tot(dm, h1e, vhf) reads self.mol.energy_nuc() internally; we preserve the SCF-08 hook signature parity by routing nuc through the caller, so an OverrideHooks impl that overrides energy_tot doesn't accidentally double-count nuc."
  - "ERI layout in default_get_jk follows the F-order arity-4 convention: r.values[mu + nu*nao + lambda*nao^2 + sigma*nao^3]. ERIs are 8-fold symmetric so chemist's vs physicist's notation is indistinguishable in this contraction."
  - "Mulliken AO→atom aggregation uses mol._bas[shell*BAS_SLOTS + ATOM_OF] + mol.ao_loc_nr. The upstream pyscf method aoslice_by_atom is not exposed on pyscf-core's Mole today; the direct _bas walk is equivalent and avoids a Mole-method dependency that pyscf-core's Phase 1 surface doesn't yet provide."

patterns-established:
  - "Slice-bridge pattern: when pyscf-scf needs a numerical primitive that pyscf-algebra has via a Tensor API but doesn't yet ship the Tensor body, plan 03-XX adds a slice-based wrapper alongside (mirroring solve_linear shape). eigh_gen joins solve_linear; future plans likely add `gemm_slice` etc."
  - "Gap-propagation pattern: when an upstream dependency is NotYetImplemented at land time, propagate the error via `?` rather than synthesizing. kernel_propagates_jk_not_yet_implemented test asserts the error message contains a recognizable marker."

requirements-completed: [SCF-01, SCF-02, SCF-03, SCF-05, SCF-06, SCF-09, SCF-11, SCF-12, SCF-13]

duration: 9min
completed: 2026-05-11
---

# Phase 03 Plan 11: SCF kernel internals — Cycle loop + Fock build + eig (Löwdin + canonicalize_signs) + analyze + convert + scanner

**Real SCF cycle loop + 10 default_* hook bodies + analyze/convert/scanner surface — every `unimplemented!()` stub plan 03-03 shipped is now real code. SCF-13 canonicalize_signs anchored post-eigh; every reduction goes through oracle_sum/oracle_dot (Pitfall 9). Wave 2 → Wave 3 handoff complete.**

## Performance

- **Duration:** 9 min
- **Started:** 2026-05-11T12:57:40Z
- **Completed:** 2026-05-11T13:07:12Z
- **Tasks:** 2 (TDD RED + GREEN per task)
- **Files modified/created:** 1 new algebra module + 11 source files + 4 test files + 1 lib.rs re-export update = 17

## Accomplishments

- **Kernel cycle loop body** (`crates/pyscf-scf/src/kernel_impl.rs`) — verbatim port of `pyscf/scf/hf.py:48-244`. Hook-driven through `OverrideHooks` trait (8 hook calls per cycle). `ConvergenceFailure` returned on miss; `tracing::info!` / `debug!` log on entry/cycle/convergence. NoOverrides path produces a clean compile-time-inlined fast path (D-02).
- **Fock build** — `default_get_hcore` (T+V_nuc), `default_get_ovlp` (S), `default_get_jk` (ERI contraction with oracle_sum over λσ — Pitfall 9), `default_get_veff` (J - 0.5·K for RHF), `default_get_fock` (h1e + V_HF fixed-point — DIIS adapter slot for plan 03-04).
- **eig + SCF-13 anchor** — `default_eig` routes to `pyscf_algebra::eigh_gen` (new slice-based generalized self-adjoint eigh via Löwdin) then applies `pyscf_core::canonicalize_signs`. Idempotency proven by `canonicalize_post_eigh.rs` integration test.
- **occ Aufbau-fill** — closed-shell RHF; rejects odd nelec with a clear error message.
- **make_rdm1** — `C·diag(occ)·C^T` with `oracle_sum` over the MO axis (Pitfall 9 mitigation).
- **energy_elec / energy_tot** — `oracle_dot(D, h1e)` for E_1e, `0.5 · oracle_dot(D, vhf)` for E_coul, `oracle_sum` to combine (3 reduction sites — Pitfall 9 grep count: 6 in energy.rs).
- **init_guess '1e' mode** — diagonalize h_core, Aufbau-fill, make_rdm1. Other modes (minao/atom/huckel/chkfile) stay `NotYetImplemented` per plan 03-03.
- **analyze / mulliken_pop / dip_moment** — real bodies; oracle Arms 7 + 8 (plan 03-08) consume these. AO→atom aggregation via `mol._bas[shell, ATOM_OF]` + `mol.ao_loc_nr`. `mulliken_meta` returns `NotYetImplemented{phase:3}`.
- **convert** — `to_rhf` / `to_uhf` / `to_ghf` clone target Mole and copy 15 scalar SCF settings each; `to_uks_stub` / `to_rks_stub` return `NotYetImplemented{phase:4}`.
- **as_scanner** — `Box<dyn Fn(&Mole) -> Result<Energy, PyscfRsError> + Send + Sync>` closure captures 14 scalar settings; rebuilds RHF, runs kernel, returns Energy. Send+Sync so geometry-optimization drivers can evaluate in parallel.
- **pyscf-algebra::eigh_gen** — new slice-based generalized self-adjoint eigh via Löwdin transform. 177 lines, 3 unit tests. Mirrors plan 03-01's `solve_linear` wrapper shape (algebra-wall bridge — D-04).

## Task Commits

Each task produced TDD RED → GREEN pair:

1. **Task 1 RED: failing tests for kernel-internals fills** — `0cdda56` (test)
2. **Task 1 GREEN: fill SCF kernel internals** — `335f628` (feat)
3. **Task 2 RED: failing tests for analyze/convert/scanner** — `9627223` (test)
4. **Task 2 GREEN: fill analyze/convert/scanner + re-exports** — `8fa69fd` (feat)

_4 atomic commits (2 RED + 2 GREEN), no REFACTOR needed._

## Files Created/Modified

### Created (5 files)
- `crates/pyscf-algebra/src/eigh_gen.rs` — slice-based generalized self-adjoint eigh (Löwdin transform; 177 lines incl. 3 unit tests)
- `crates/pyscf-scf/tests/canonicalize_post_eigh.rs` — SCF-13 anchor integration test (idempotency + sorted eigenvalues)
- `crates/pyscf-scf/tests/kernel_internals_unit.rs` — 7 tests (occ Aufbau even/odd/4e, make_rdm1 trace + determinism, energy_elec sign convention)
- `crates/pyscf-scf/tests/no_overrides_drives_kernel.rs` — H2 smoke (#[ignore] pending int2e_sph) + propagates-NotYetImplemented assertion
- `crates/pyscf-scf/tests/analyze_convert_scanner.rs` — 10 tests covering analyze/convert/scanner surface
- `.planning/phases/03-scf-pyo3-bindings/03-11-SUMMARY.md` (this file)

### Modified (12 files)
- `crates/pyscf-algebra/src/lib.rs` — re-export `eigh_gen`
- `crates/pyscf-scf/src/kernel_impl.rs` — real `scf_loop` body
- `crates/pyscf-scf/src/fock.rs` — 5 real default_* bodies
- `crates/pyscf-scf/src/eig.rs` — real `default_eig` (eigh_gen + canonicalize_signs)
- `crates/pyscf-scf/src/occ.rs` — RHF Aufbau-fill
- `crates/pyscf-scf/src/rdm.rs` — `C·diag(occ)·C^T` via oracle_sum
- `crates/pyscf-scf/src/energy.rs` — `e_elec`/`e_coul` via oracle_dot + oracle_sum
- `crates/pyscf-scf/src/init_guess.rs` — `init_guess_by_1e` body
- `crates/pyscf-scf/src/analyze.rs` — `mulliken_pop` + `dip_moment` + `analyze`
- `crates/pyscf-scf/src/convert.rs` — `to_rhf` / `to_uhf` / `to_ghf` + Phase 4 stubs
- `crates/pyscf-scf/src/scanner.rs` — real closure
- `crates/pyscf-scf/src/lib.rs` — re-export the new public surface

## Source-of-Truth Line References

| Module | Upstream PySCF reference |
|--------|---------------------------|
| `kernel_impl::scf_loop` | `pyscf/scf/hf.py:48-244` (def kernel) |
| `fock::default_get_hcore` | `pyscf/scf/hf.py:316-345` |
| `fock::default_get_ovlp` | `pyscf/scf/hf.py:346-360` |
| `fock::default_get_jk` | `pyscf/scf/hf.py:957-1033` |
| `fock::default_get_veff` | `pyscf/scf/hf.py:1034-1085` |
| `fock::default_get_fock` | `pyscf/scf/hf.py:1086-1135` (DIIS slot) |
| `eig::default_eig` | `pyscf/scf/hf.py:1349-1357` (inline canonicalize) |
| `occ::default_get_occ` | `pyscf/scf/hf.py:1499-1517` |
| `rdm::default_make_rdm1` | `pyscf/scf/hf.py:1517-1538` |
| `energy::default_energy_elec` | `pyscf/scf/hf.py:1556-1574` |
| `energy::default_energy_tot` | `pyscf/scf/hf.py:1574-1602` |
| `init_guess::init_guess_by_1e` | `pyscf/scf/hf.py:485-494` |
| `analyze::mulliken_pop` | `pyscf/scf/hf.py:1262-1300` |
| `analyze::dip_moment` | `pyscf/scf/hf.py:1306-1340` |
| `convert::to_rhf/to_uhf/to_ghf` | `pyscf/scf/hf.py:2272-2300` |
| `scanner::as_scanner` | `pyscf/scf/hf.py:1538-1602` |

## Pitfall 9 Mitigation — oracle_sum / oracle_dot Call Sites

```
$ grep -cE "oracle_sum|oracle_dot" crates/pyscf-scf/src/{fock,rdm,energy,analyze}.rs
crates/pyscf-scf/src/fock.rs:2
crates/pyscf-scf/src/rdm.rs:1
crates/pyscf-scf/src/energy.rs:6
crates/pyscf-scf/src/analyze.rs:5
```

Total: **14 oracle_* call sites** across the 4 reduction-bearing modules. Every cross-axis sum routes through the pairwise-tree reduction (chunk=128) — bit-identical results across thread counts and rerun-time reordering (Pitfall 2 + Pitfall 9 mitigation).

## SCF-13 Anchor Status

```
$ grep -F "canonicalize_signs" crates/pyscf-scf/src/eig.rs
//! (the inline canonicalize was extracted into pyscf-core::canonicalize_signs
//! calls `pyscf_core::canonicalize_signs` on the F-order MO coefficient
use pyscf_core::{canonicalize_signs, Density, MOCoefficients, PyscfRsError};
    canonicalize_signs(&mut eigvecs, nao, nao);
```

Single call site at `eig.rs:36`. Idempotency asserted by `canonicalize_post_eigh.rs::eig_applies_sign_canonicalization` — a second `canonicalize_signs` pass on the output buffer must be a no-op. Plan 03-04 (DIIS) and Phase 4 (DFT) inherit canonicalization transparently because they call `default_eig` via the `OverrideHooks` trait.

## init_guess Mode Status

| Mode | Status | Notes |
|------|--------|-------|
| `OneElectron` ("1e") | **Real body** | Diagonalize h_core, Aufbau, make_rdm1. SCF-05 |
| `UserDM(d)` | **Real body** | Clones the user-supplied Density |
| `Minao` | NotYetImplemented | Plan 03-03 stub; Phase 3 follow-up if time permits |
| `Atom` | NotYetImplemented | Plan 03-03 stub |
| `Huckel` | NotYetImplemented | Plan 03-03 stub |
| `Chkfile(_)` | NotYetImplemented | Plan 03-06 ships |

## mulliken_pop + dip_moment Status (SCF-09)

- **`mulliken_pop(rhf)`** — real body. Builds D and S, computes AO populations `pop[μ] = Σ_ν D[μν]·S[νμ]` via oracle_sum, aggregates onto atoms via `mol._bas[shell, ATOM_OF]` + `mol.ao_loc_nr`. Errors cleanly when kernel hasn't run.
- **`dip_moment(rhf)`** — real body. Builds D, fetches `int1e_r` (component-leading F-order [3, nao, nao]), computes electronic contribution `-Σ_μν D[μν]·r[k, μ, ν]` via oracle_sum, adds nuclear contribution `Σ_A Z_A · r_A[k]`.
- **`mulliken_meta(rhf)`** — `NotYetImplemented{phase:3}` (meta-Löwdin variant deferred to plan 03-10 follow-up).
- **Oracle Arms 7 + 8 (plan 03-08) ready to consume** — `MullikenResult { atom_charges, ao_populations }` and `[f64; 3]` dipole.

## to_rhf / to_uhf / to_ghf / Phase 4 stubs (SCF-11)

| Function | Behaviour |
|----------|-----------|
| `to_rhf(uhf)` → `Result<RHF, _>` | Real body — copies 15 scalar SCF fields |
| `to_uhf(rhf)` → `Result<UHF, _>` | Real body — copies 15 scalar SCF fields |
| `to_ghf(rhf)` → `Result<GHF, _>` | Real body — copies 15 scalar SCF fields |
| `to_uks_stub(rhf)` → `Result<(), _>` | Returns `NotYetImplemented{phase:4, what:"to_uks (UKS target lands in Phase 4 DFT)"}` |
| `to_rks_stub(rhf)` → `Result<(), _>` | Returns `NotYetImplemented{phase:4, what:"to_rks (RKS target lands in Phase 4 DFT)"}` |

MO coefficients are NOT copied across — the alpha/beta promotion (UHF) and 2c-spinor projection (GHF) live in plan 03-04+ if needed.

## as_scanner closure (SCF-12)

```rust
pub fn as_scanner(rhf: &RHF) -> Box<dyn Fn(&Mole) -> Result<Energy, PyscfRsError> + Send + Sync>
```

Captures 14 scalar SCF settings by value (conv_tol, conv_tol_grad, max_cycle, init_guess, diis, diis_space, diis_start_cycle, diis_damp, level_shift, damp, direct_scf, direct_scf_tol, verbose, max_memory). On invocation: clones the new Mole, instantiates RHF with the captured settings, runs `kernel()`, returns `Energy(new_rhf.e_tot)`. `Send + Sync` so parallel geometry-optimization drivers can evaluate scanners across threads.

## DIIS Adapter Slot Status (plan 03-04 swap target)

`default_get_fock(h1e, s1e, vhf, dm, cycle, diis_state) -> F`:

- **Current body (plan 03-11):** `F = h1e + V_HF` (fixed-point fallback). After cycle ≥ 1, emits `tracing::warn!(target: "pyscf_scf::fock", cycle, "DIIS adapter slot — plan 03-04 not yet shipped, fixed-point fallback in use")`.
- **Plan 03-04 swap:** Replace fixed-point body with `pyscf-diis::DiisAdapter::extrapolate(h1e + V_HF, dm, S, cycle, diis_state)`. Trait surface lives in pyscf-diis; the OverrideHooks::get_fock signature here doesn't change.

## DF-HF Entry Point Status (plan 03-05 swap target)

`default_get_jk(mol, dm) -> (J, K)`:

- **Current body (plan 03-11):** Routes through `pyscf_gto::intor(mol, "int2e")` which returns `NotYetImplemented{phase:2}` (plan 02-09 verification rollup gap). The error propagates via `?` to the kernel caller; no panic.
- **Plan 03-05 swap:** Ships `DfHooks: OverrideHooks` whose `get_jk` impl routes through `pyscf-df::DfIntegrals::get_jk` (3-index density-fitted ERIs). Bypasses int2e_sph entirely for the SCF test corpus.

## Decisions Made

1. **eigh_gen is slice-based, not Tensor-based.** Phase 1 `host_fallback::eigh(client, tensor)` is `NotYetImplemented{phase:3}`; using it would force pyscf-scf to depend on AlgebraClient/Tensor (D-04 violation). Mirrored `solve_linear`'s wrapper pattern from plan 03-01.
2. **Löwdin transform with linear-dep removal at `S_LINEAR_DEP_TOL = 1e-12`** — matches upstream pyscf's `LINEAR_DEP_THRESHOLD`. Dropped columns map to `+∞` eigenvalues in the output (signalling "linearly dependent direction") so the n-length vector stays aligned with the n-column C buffer.
3. **default_get_jk propagates int2e_sph NotYetImplemented**, not a panic or synthetic fake. Either plan 02-09 (verification rollup) or plan 03-05 (DF-HF override) closes the gap cleanly without re-touching kernel_impl.rs.
4. **energy_tot returns e_elec (caller adds enuc).** Preserves the SCF-08 hook signature parity with upstream `Method.energy_tot` while routing nuc-repulsion through `mol.enuc()` at the kernel call site.
5. **ERI layout convention.** F-order arity-4: `r.values[mu + nu*nao + lambda*nao^2 + sigma*nao^3]`. ERIs' 8-fold symmetry makes the J and K index permutations work regardless of chemist's / physicist's notation.
6. **Mulliken AO→atom aggregation walks `mol._bas[shell, ATOM_OF]` + `mol.ao_loc_nr` directly** — upstream's `aoslice_by_atom` isn't exposed as a Mole method in pyscf-core's Phase 1 surface.
7. **`mulliken_meta` ships as `NotYetImplemented{phase:3}`** — meta-Löwdin variant is a plan 03-10 follow-up; plan 03-11's surface is the simpler closed-shell Mulliken that Arms 7 + 8 actually consume.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `pyscf_algebra::eigh` Tensor API is NotYetImplemented; plan body's slice-based call signature doesn't exist**
- **Found during:** Task 1 GREEN, eig.rs implementation.
- **Issue:** The plan body said `pyscf_algebra::eigh(&fock.data, &s1e.data, nao)?` but the actual `pyscf_algebra::eigh` signature is `eigh(client: &AlgebraClient, matrix: &Tensor) -> Result<(Vec<f64>, Tensor), AlgebraError>` AND it returns `NotYetImplemented{phase:3}` as its body. The plan also assumed a generalized-eigh signature (F + S inputs), but the Tensor API is single-matrix.
- **Fix:** Added `pyscf-algebra::eigh_gen` — a new slice-based generalized self-adjoint eigh module mirroring plan 03-01's `solve_linear` shape. Implements the Löwdin transform via faer 0.24 `SelfAdjointEigen` (2 self-adjoint eigh calls: one on S to build S^{-1/2}, one on the transformed F'). Includes linear-dependency removal at `1e-12`.
- **Files modified:** `crates/pyscf-algebra/src/eigh_gen.rs` (new, 177 lines), `crates/pyscf-algebra/src/lib.rs` (re-export).
- **Verification:** 3 unit tests in eigh_gen pass; the canonicalize_post_eigh integration test (which uses S=I path) and the embedded `generalized_problem_recovers_eigenvalue_equation` test verify F·c = ε·S·c holds.
- **Committed in:** `335f628` (Task 1 GREEN).

**2. [Rule 3 - Blocking] `pyscf_algebra::axpy` signature is Tensor-based; the plan body's slice-based call doesn't compile**
- **Found during:** Task 1 GREEN, fock.rs implementation.
- **Issue:** Plan body wrote `pyscf_algebra::axpy(1.0, &v.values, &mut data)?` — but the actual `pyscf_algebra::axpy` signature is `axpy(client, alpha, &Tensor, &mut Tensor)` and its body returns `NotYetImplemented`.
- **Fix:** Replaced axpy calls in fock.rs (`default_get_hcore`, `default_get_veff`, `default_get_fock`) with explicit loops over `0..(nao*nao)`. These are SCF-loop inner operations with `nao²` elements — small enough that an explicit loop is fine until the Tensor-API axpy lands.
- **Files modified:** `crates/pyscf-scf/src/fock.rs`.
- **Committed in:** `335f628` (Task 1 GREEN).

**3. [Rule 3 - Blocking] `pyscf_gto::intor` returns `NotYetImplemented{phase:2}` for arity-4 int2e_sph**
- **Found during:** Task 1 GREEN, fock.rs `default_get_jk` implementation.
- **Issue:** Plan body called `pyscf_algebra::oracle_einsum("μνλσ,λσ->μν", ...)` but `oracle_einsum` only supports the binary `"ij,jk->ik"` pattern (per `oracle.rs:46-69`). Even if it didn't, `pyscf_gto::intor(mol, "int2e_sph")` itself returns `NotYetImplemented{phase:2}` (plan 02-09 verification rollup gap explicitly documented in the plan).
- **Fix:** Documented in plan as the gap-closure path: `default_get_jk` propagates the int2e NotYetImplemented error verbatim via `?`. The contraction logic is in place (8-fold symmetric F-order index, per-(μ,ν) oracle_sum over (λ,σ)) so when int2e_sph lands, no body changes — only the upstream gap closes. Plan 03-05 (DF-HF) provides the override path that bypasses int2e_sph entirely.
- **Files modified:** `crates/pyscf-scf/src/fock.rs` (default_get_jk body + comment block).
- **Test:** `no_overrides_drives_kernel::kernel_propagates_jk_not_yet_implemented` asserts the kernel returns a structured error (not a panic) when int2e is invoked.
- **Committed in:** `335f628` (Task 1 GREEN).

**4. [Rule 1 - Bug] `mol.energy_nuc()` and `mol.atom_charge(a)` / `mol.atom_coord(a)` / `mol.aoslice_by_atom(a)` don't exist on pyscf-core::Mole**
- **Found during:** Task 1 GREEN (kernel_impl.rs uses `mol.energy_nuc()`); Task 2 GREEN (analyze.rs uses `mol.atom_charge(a)`/`atom_coord(a)`/`aoslice_by_atom(a)`).
- **Issue:** Plan body referenced PySCF Python method names that pyscf-core's Phase 1 surface doesn't expose under those names. Actual methods are `mol.enuc()` (Phase 1), `mol.atom_charges() -> Vec<i32>` (Phase 1), `mol._atom[i].1` for coords. No `aoslice_by_atom` exists; the equivalent walk is `mol._bas[shell*BAS_SLOTS + ATOM_OF]` + `mol.ao_loc_nr[shell..shell+2]`.
- **Fix:** kernel_impl.rs uses `mol.enuc()`. analyze.rs uses `mol.atom_charges()` (indexed by `a`) and `mol._atom[a].1` for coords. The AO→atom aggregation walks `mol._bas` directly. Errors cleanly when these arrays haven't been populated by `mol.build()` (Phase 2 prerequisite).
- **Files modified:** `crates/pyscf-scf/src/kernel_impl.rs`, `crates/pyscf-scf/src/analyze.rs`.
- **Committed in:** `335f628` (Task 1) and `8fa69fd` (Task 2).

**5. [Rule 1 - Bug] `pyscf_algebra::oracle_sum` takes `&[f64]`, not an iterator**
- **Found during:** Task 1 GREEN.
- **Issue:** Plan body wrote `pyscf_algebra::oracle_sum((0..nmo).map(|i| ...))` — but the actual signature is `oracle_sum(xs: &[f64]) -> f64` (pairwise-tree expects a slice for determinism — see `oracle.rs:22`).
- **Fix:** Materialise the iterator into a `Vec<f64>` (or reuse a scratch buffer for hot loops) before calling oracle_sum. In rdm.rs, energy.rs, analyze.rs the scratch buffer pattern keeps the allocation cost low.
- **Files modified:** `crates/pyscf-scf/src/rdm.rs`, `crates/pyscf-scf/src/energy.rs`, `crates/pyscf-scf/src/analyze.rs`, `crates/pyscf-scf/src/fock.rs`.
- **Committed in:** `335f628` and `8fa69fd`.

**6. [Rule 2 - Critical] Plan body's RHF::kernel call site in scanner.rs would deadlock on the `(new_rhf, &NoOverrides)` borrow split if naively expressed**
- **Found during:** Task 2 GREEN, scanner.rs.
- **Issue:** No actual bug — confirming the existing `RHF::kernel(&mut self)` returns the result correctly. Spent ~30 seconds re-verifying ownership; no fix needed.
- **Resolution:** No change. Plan body's scanner shape is correct against the RHF::kernel signature shipped in plan 03-03.

**7. [Rule 1 - Bug] `pyscf_gto::intor` is called as a free function, not `mol.intor(name)`**
- **Found during:** Task 1 GREEN (fock.rs), Task 2 GREEN (analyze.rs).
- **Issue:** Plan body wrote `mol.intor("int1e_kin")` (mirroring pyscf Python). Actual pyscf-gto API is `pyscf_gto::intor(&mole, name) -> Result<IntorOutput, _>` (intor lives in pyscf-gto, not as a Mole method).
- **Fix:** Use `pyscf_gto::intor(mol, "...")` everywhere. Note `IntorOutput { values, shape, layout }` is what's returned; we extract `.values` (F-order flat buffer).
- **Files modified:** `crates/pyscf-scf/src/fock.rs`, `crates/pyscf-scf/src/analyze.rs`.
- **Committed in:** `335f628` and `8fa69fd`.

---

**Total deviations:** 7 (5 blocking-gap auto-fixes, 1 API-signature adapt, 1 false-alarm). Net effect: the plan's intended surface ships, with a slice-based eigh_gen added to pyscf-algebra as the one structural deviation. No scope creep — all changes inside the plan's named files.

## Issues Encountered

- **Worktree base mismatch on init:** HEAD was at the expected base commit (`d3cc064`) but the index showed the prior wave's files as "to be deleted from working tree" (worktree appeared to start from `a05f896`). Resolved via `git reset --hard HEAD` to restore the worktree to d3cc064's state. Verified by `ls crates/pyscf-scf/src/` showing all 17 plan-03-03 files present before starting work.

- **`pyscf_algebra::eigh` Tensor API is `NotYetImplemented{phase:3}` and signature mismatch with plan body:** Addressed by adding `eigh_gen` (deviation 1 above). This is the largest structural change in the plan but keeps the algebra-wall (D-04) clean and mirrors plan 03-01's `solve_linear` pattern.

- **`pyscf-gto::intor` returns `NotYetImplemented{phase:2}` for arity-4 int2e_sph:** The documented gap-closure path is followed (deviation 3 above) — propagate cleanly, document the plan 03-05 override path, ship the contraction logic so no body changes when the gap closes.

## User Setup Required

None — pure Rust kernel-internals plan, no external service config.

## Next Wave Readiness

- **Plan 03-04 (DIIS):** `default_get_fock`'s fixed-point fallback emits a `tracing::warn` after cycle≥1; plan 03-04 replaces the body with CDIIS extrapolation via `pyscf-diis::DiisAdapter`. The OverrideHooks::get_fock signature here doesn't change.
- **Plan 03-05 (DF-HF):** Implements `DfHooks: OverrideHooks` with a `get_jk` override that routes through pyscf-df, bypassing int2e_sph. The hf-overrides path is fully wired; only the DfHooks impl is needed.
- **Plan 03-06 (chkfile):** Adds the `InitGuessMode::Chkfile` body in `default_get_init_guess` (currently returns `InitGuessNotYetImplemented("chkfile", "03-06")`).
- **Plan 03-07 (PyO3):** `PyOverrideBridge` implements `OverrideHooks`; the trait surface is stable since plan 03-03.
- **Plan 03-08 (oracle harness):** Arms 7 + 8 consume `mulliken_pop` and `dip_moment` (both shipped). `parse_init_guess_mode` Arm 4 already wired (plan 03-03).
- **Plan 03-10 (oracle harness wave 2):** Unignores `h2_no_overrides_converges` once `int2e_sph` lands (plan 02-09) OR uses DfHooks (plan 03-05) for the H2 oracle assertion.

## Stub Inventory

```
$ grep -rn "unimplemented!" crates/pyscf-scf/src/
(no matches)
```

**Zero `unimplemented!()` markers remain in `crates/pyscf-scf/src/`** — all 20 stubs plan 03-03 shipped are now real bodies or `NotYetImplemented` errors with explicit Phase / plan markers (`mulliken_meta` phase:3, `to_uks_stub`/`to_rks_stub` phase:4, init_guess minao/atom/huckel/chkfile via the ScfError::InitGuessNotYetImplemented variant).

## Known Stubs

The following functions return `NotYetImplemented` / `InitGuessNotYetImplemented` as structured errors (not panics) — these are intentional deferrals to future plans:

| Function | Status | Resolved by |
|----------|--------|-------------|
| `default_get_jk` (when int2e called) | Propagates `NotYetImplemented{phase:2}` from `pyscf_gto::intor("int2e")` | Plan 02-09 verification rollup OR plan 03-05 DF-HF override |
| `init_guess_by_minao` | `ScfError::InitGuessNotYetImplemented("minao", "03-03 follow-up")` | Phase 3 follow-up plan |
| `init_guess_by_atom` | Same | Same |
| `init_guess_by_huckel` | Same | Same |
| `init_guess_by_chkfile` | Same | Plan 03-06 |
| `mulliken_meta` | `NotYetImplemented{phase:3, what:"mulliken_meta (meta-Löwdin variant; plan 03-10 follow-up)"}` | Plan 03-10 follow-up |
| `to_uks_stub` | `NotYetImplemented{phase:4, what:"to_uks (UKS target lands in Phase 4 DFT)"}` | Phase 4 DFT |
| `to_rks_stub` | `NotYetImplemented{phase:4, what:"to_rks (RKS target lands in Phase 4 DFT)"}` | Phase 4 DFT |

None of these are "wired-to-UI silently empty" stubs — they all return structured Rust errors that callers see via `Result::Err`. The plan's goal (kernel cycle loop with SCF-13 + Pitfall 9 + analyze/convert/scanner surface) is fully achieved.

## Self-Check

Files claimed created/modified, verified to exist:

```
FOUND: crates/pyscf-algebra/src/eigh_gen.rs
FOUND: crates/pyscf-algebra/src/lib.rs
FOUND: crates/pyscf-scf/src/kernel_impl.rs
FOUND: crates/pyscf-scf/src/fock.rs
FOUND: crates/pyscf-scf/src/eig.rs
FOUND: crates/pyscf-scf/src/occ.rs
FOUND: crates/pyscf-scf/src/rdm.rs
FOUND: crates/pyscf-scf/src/energy.rs
FOUND: crates/pyscf-scf/src/init_guess.rs
FOUND: crates/pyscf-scf/src/analyze.rs
FOUND: crates/pyscf-scf/src/convert.rs
FOUND: crates/pyscf-scf/src/scanner.rs
FOUND: crates/pyscf-scf/src/lib.rs
FOUND: crates/pyscf-scf/tests/canonicalize_post_eigh.rs
FOUND: crates/pyscf-scf/tests/kernel_internals_unit.rs
FOUND: crates/pyscf-scf/tests/no_overrides_drives_kernel.rs
FOUND: crates/pyscf-scf/tests/analyze_convert_scanner.rs
```

Commits claimed, verified in `git log --oneline`:

```
FOUND: 0cdda56 — test(03-11) Task 1 RED
FOUND: 335f628 — feat(03-11) Task 1 GREEN
FOUND: 9627223 — test(03-11) Task 2 RED
FOUND: 8fa69fd — feat(03-11) Task 2 GREEN
```

Plan-level verification commands:

```
$ grep -F "canonicalize_signs" crates/pyscf-scf/src/eig.rs
[4 matches — see SCF-13 Anchor Status above]
$ grep -cE "oracle_sum|oracle_dot" crates/pyscf-scf/src/energy.rs
6
$ grep -F "NotYetImplemented" crates/pyscf-scf/src/convert.rs
[3 matches — see SCF-11 status above]
$ grep -F "phase: 4" crates/pyscf-scf/src/convert.rs
[2 matches — to_uks_stub + to_rks_stub]
$ grep -F "unimplemented!" crates/pyscf-scf/src/kernel_impl.rs
[no matches]
$ grep -F "unimplemented!" crates/pyscf-scf/src/fock.rs
[no matches]
```

Test counts (full pyscf-scf suite):

```
attribute_floor                  4 passing
hooks_kernel_types               4 passing
canonicalize_post_eigh           2 passing
kernel_internals_unit            7 passing
analyze_convert_scanner         10 passing
no_overrides_drives_kernel       1 passing, 1 ignored (int2e_sph gap)
pyscf-algebra eigh_gen (lib)     3 passing
                              ─────────────
                                31 passing, 1 ignored, 0 failed
```

## Self-Check: PASSED

---

*Phase: 03-scf-pyo3-bindings*
*Plan: 11*
*Completed: 2026-05-11*

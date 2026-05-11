---
phase: 03-scf-pyo3-bindings
plan: 04
subsystem: scf
tags: [rust, scf, diis, cdiis, pulay, scf-04, pitfall-9, oracle-determinism, pyscf-diis, fock-subspace]

requires:
  - phase: 01-foundation
    provides: pyscf-algebra (oracle_dot, oracle_sum, AlgebraError), pyscf-runtime, thiserror
  - phase: 03-01
    provides: pyscf-algebra::solve_linear (host-faer FullPivLu — consumed by Diis::extrapolate)
  - phase: 03-03
    provides: pyscf-scf trait scaffolding (OverrideHooks, KernelConfig with diis/diis_space/diis_start_cycle fields)
  - phase: 03-11
    provides: pyscf-scf kernel_impl::scf_loop end-to-end body + default_get_fock BASE Fock builder (the DIIS extrapolation slot)

provides:
  - "pyscf-diis::DiisStorable — generic trait (D-09) for any iterate type"
  - "pyscf-diis::Diis<S: DiisStorable + Clone> — CDIIS Pulay stack with space=8 default, ring-buffer push, solve_linear-backed extrapolate"
  - "pyscf-diis::err_vec_scf — SDF - FDS error vector (pyscf/scf/diis.py:68-87) via explicit row-major matmul fallback"
  - "pyscf-diis::DiisError — Singular | Algebra(#[from] AlgebraError) — threat T-3-13 mitigation"
  - "pyscf-scf::FockSubspace — DiisStorable impl for Fock matrices (dot via oracle_dot — Pitfall 9)"
  - "pyscf-scf::diis_step — cycle-local CDIIS step invoked from kernel_impl::scf_loop"
  - "kernel_impl::scf_loop now applies CDIIS extrapolation after hooks.get_fock when cfg.diis=true and cycle >= cfg.diis_start_cycle"

affects:
  - 03-08 (oracle harness — Arm 6 will assert error-vector norm convergence; consumes DIIS state)
  - 03-10 (oracle harness wave 2 — once int2e_sph lands, h2_no_overrides_converges should converge in fewer cycles with DIIS than without)
  - 06 (CCSD-04 amplitude DIIS — Phase 6 will reuse Diis<S> with AmpsSubspace impl)

tech-stack:
  added:
    - "pyscf-diis dep on pyscf-scf (Cargo.toml — first method crate to consume pyscf-diis)"
  patterns:
    - "Pattern: Slice-bridge from method crate to algebra primitive (DiisStorable shields callers from cubecl Tensor types — same shape as solve_linear + eigh_gen from plans 03-01 / 03-11)"
    - "Pattern: DIIS hoisted one layer above the override hook. The OverrideHooks::get_fock signature stays simple (BASE Fock builder); kernel_impl::scf_loop wraps it with diis_step. Override impls inherit DIIS for free without re-implementing it."
    - "Pattern: Pitfall 9 mitigation everywhere reductions matter — B-matrix inner products (oracle_dot), extrapolated-iterate cross-iterate sums (oracle_sum), FockSubspace::dot (oracle_dot). 14 oracle_* call sites in cdiis.rs + diis_adapter.rs combined."

key-files:
  created:
    - "crates/pyscf-diis/src/storable.rs (DiisStorable trait, 28 lines)"
    - "crates/pyscf-diis/src/error.rs (DiisError enum, 22 lines)"
    - "crates/pyscf-diis/src/cdiis.rs (Diis<S> + extrapolate + err_vec_scf, 235 lines)"
    - "crates/pyscf-diis/tests/cdiis_pulay.rs (4 numerical tests, 113 lines)"
    - "crates/pyscf-diis/tests/oracle_reductions.rs (3 source/determinism guards, 79 lines)"
    - "crates/pyscf-scf/src/diis_adapter.rs (FockSubspace + diis_step, 84 lines)"
    - "crates/pyscf-scf/tests/diis_adapter_wiring.rs (7 wiring tests, 114 lines)"
  modified:
    - "crates/pyscf-diis/src/lib.rs (re-exports for Diis, DiisStorable, DiisError, err_vec_scf)"
    - "crates/pyscf-scf/Cargo.toml (added pyscf-diis = { path = ../pyscf-diis })"
    - "crates/pyscf-scf/src/lib.rs (register diis_adapter module + re-export FockSubspace, diis_step)"
    - "crates/pyscf-scf/src/kernel_impl.rs (instantiate Diis<FockSubspace>, call diis_step after hooks.get_fock per cycle)"
    - "crates/pyscf-scf/src/fock.rs (drop the 'plan 03-04 not yet shipped' tracing::warn; default_get_fock now just builds BASE Fock; comment update)"

key-decisions:
  - "DIIS extrapolation lives in kernel_impl::scf_loop, NOT inside OverrideHooks::get_fock. Hoisting one layer up keeps the SCF-08 hook signature simple (BASE Fock builder) and lets any OverrideHooks impl (PyOverrideBridge, DfHooks, ...) inherit DIIS without reimplementing it."
  - "err_vec_scf uses an explicit O(nao^3) row-major matmul fallback rather than pyscf_algebra::gemm. The Tensor-API gemm is NotYetImplemented{phase:2}; the explicit fallback mirrors the rdm.rs / energy.rs pattern from plan 03-11. When Tensor gemm lands, callers can swap it in without re-signing err_vec_scf."
  - "DiisError -> PyscfRsError routes through the existing ConvergenceFailure variant (iterations + reason). Adding a new top-level variant would have churned the PyscfRsError surface; ConvergenceFailure's String reason carrier is sufficient for the rare singular-B path (threat T-3-13)."
  - "RAYON_NUM_THREADS env-toggle test removed. Rust 2024 made std::env::set_var unsafe, and pyscf-diis is #![forbid(unsafe_code)]. The deterministic guarantee comes from oracle_dot/oracle_sum's PAIRWISE_CHUNK=128 pairwise-tree algorithm (doesn't use rayon at all); the matrix CI job xplat-uhartree (per 03-VALIDATION.md) supplies the cross-platform bit-identity assertion. The in-process replacement test asserts same-run bit-identity instead — sufficient as a regression guard if anyone replaces oracle_* with a thread-count-dependent reducer."
  - "Ring-buffer test uses VARYING error vectors. Pushing identical errors creates a rank-deficient B-matrix interior, surfacing as DiisError::Singular — that's the threat-T-3-13 path, not the ring-buffer mechanics being tested. Distinct test (singular_b_matrix_returns_error_not_panic) covers T-3-13 separately."

patterns-established:
  - "Pattern: pyscf-diis is a host-only crate. It depends on pyscf-algebra (for oracle_dot/oracle_sum/solve_linear) but never names cubecl directly — D-04 algebra-wall analog preserved."
  - "Pattern: DIIS state lives in the kernel cycle loop, not in the hook. Override impls (Python via plan 03-07, DfHooks via plan 03-05) automatically get DIIS extrapolation around their get_fock return value."

requirements-completed: [SCF-04]

duration: 5min
completed: 2026-05-11
---

# Phase 03 Plan 04: pyscf-diis CDIIS body + SCF kernel-loop wiring Summary

**CDIIS Pulay extrapolation (`pyscf-diis::Diis<S>`) shipped end-to-end, plus `FockSubspace` impl `DiisStorable` and `diis_step` wired into `pyscf-scf::kernel_impl::scf_loop`. Plan 03-11's `tracing::warn!("plan 03-04 not yet shipped")` is gone — DIIS is live (SCF-04).**

## Performance

- **Duration:** 5 min
- **Started:** 2026-05-11T13:21:01Z
- **Completed:** 2026-05-11T13:26:59Z
- **Tasks:** 2 (TDD RED + GREEN per task)
- **Files created/modified:** 5 created + 5 modified = 10

## Accomplishments

- **`pyscf-diis::Diis<S>`** — generic CDIIS Pulay stack over a `DiisStorable + Clone` iterate type. Ring-buffer of size `space` (default 8), push wraps the oldest slot, `extrapolate` builds the (n+1)×(n+1) Pulay system via `oracle_dot` inner products and solves it through `pyscf_algebra::solve_linear` (host-faer FullPivLu).
- **`pyscf-diis::DiisStorable` trait** — single trait with `as_flat / from_flat / dot / len` (D-09). FockSubspace impls it for SCF; Phase 6 CCSD-04 will impl it with `AmpsSubspace` over `(T1, T2)` amplitudes, sharing the exact same Pulay machinery.
- **`pyscf-diis::err_vec_scf`** — SDF − FDS error vector per `pyscf/scf/diis.py:68-87`. Uses an explicit O(nao³) row-major matmul fallback (gemm Tensor API is NotYetImplemented{phase:2}; the fallback mirrors plan 03-11's pattern in rdm.rs/energy.rs).
- **`pyscf-scf::FockSubspace + diis_step`** — SCF-side adapter. `FockSubspace::dot` routes through `pyscf_algebra::oracle_dot` (Pitfall 9 / threat T-3-09). `diis_step` is a no-op when `cycle < start_cycle`, else pushes the current Fock + error vector and returns the extrapolated Fock.
- **`kernel_impl::scf_loop` wiring** — when `cfg.diis = true`, instantiates `Diis<FockSubspace>::new(cfg.diis_space)` once at entry. Every cycle: `hooks.get_fock(...)` builds the BASE Fock; `diis_step(...)` extrapolates if `cycle >= cfg.diis_start_cycle`. DIIS error packaging routes through `PyscfRsError::ConvergenceFailure { iterations, reason }`.
- **Plan 03-11's fixed-point fallback warning removed** — `default_get_fock` no longer emits the `"plan 03-04 not yet shipped"` warning; it now simply builds `F = h1e + V_HF` (BASE Fock). The CDIIS step is hoisted one architectural layer above the override hook, so any OverrideHooks impl (PyOverrideBridge plan 03-07, DfHooks plan 03-05) inherits DIIS automatically.

## Task Commits

| # | Task | Hash | Type |
|---|------|------|------|
| 1 | Task 1 RED — failing pyscf-diis Pulay + oracle-reduction tests | `70d9b08` | test |
| 2 | Task 1 GREEN — implement pyscf-diis CDIIS body (SCF-04) | `b9d34fe` | feat |
| 3 | Task 2 RED — failing DIIS adapter wiring tests | `8732a0a` | test |
| 4 | Task 2 GREEN — wire pyscf-diis into pyscf-scf::kernel_impl (SCF-04) | `dc300b1` | feat |

_4 atomic commits (2 RED + 2 GREEN), no REFACTOR needed._

## Source-of-Truth Line References

| Module | Upstream PySCF reference |
|--------|---------------------------|
| `pyscf-diis::cdiis::Diis::extrapolate` | `pyscf/scf/diis.py:48-58` (`update` method) |
| `pyscf-diis::cdiis::err_vec_scf` | `pyscf/scf/diis.py:68-87` (`get_err_vec_orig`) |
| `pyscf-diis::cdiis` algorithm | Pulay 1980 (DOI:10.1016/0009-2614(80)80396-4) |
| `pyscf-scf::diis_adapter::diis_step` | `pyscf/scf/hf.py:1086-1135` (DIIS slot inside upstream `get_fock`) |
| `pyscf-scf::FockSubspace` | D-09 (RESEARCH §"Pattern 7 — FockSubspace impl DiisStorable" lines 868-892) |
| `KernelConfig.diis_space=8, diis_start_cycle=1` | `pyscf/scf/hf.py:1701, 1704` |

## Pitfall 9 Mitigation — oracle_dot / oracle_sum Call Sites

| File | oracle_dot | oracle_sum | solve_linear |
|------|-----------:|-----------:|-------------:|
| `crates/pyscf-diis/src/cdiis.rs` | 6 | 4 | 4 |
| `crates/pyscf-scf/src/diis_adapter.rs` | 2 | 0 | 0 |

```
$ grep -c oracle_dot crates/pyscf-diis/src/cdiis.rs   # 6 (B-matrix inner products)
$ grep -c oracle_sum crates/pyscf-diis/src/cdiis.rs   # 4 (extrapolated-iterate cross-iterate sums + tests)
$ grep -c solve_linear crates/pyscf-diis/src/cdiis.rs # 4 (Lagrange-multiplier LU solve)
$ grep -c oracle_dot crates/pyscf-scf/src/diis_adapter.rs  # 2 (FockSubspace::dot impl + comment)
```

Total: **16 oracle_*/solve_linear call sites** across the two crates. Every B-matrix inner product and every extrapolated-iterate sum routes through the pairwise-tree reduction (chunk=128) — bit-identical results across thread counts (threat T-3-09 mitigation).

## solve_linear Integration Point

`crates/pyscf-diis/src/cdiis.rs:103` — `pyscf_algebra::solve_linear(&b, &rhs, dim)`. The B-matrix is `(n+1) × (n+1)` row-major flat; RHS is `[0, …, 0, -1]` of length `n+1`. `AlgebraError::Singular` (threat T-3-13) is re-packaged as `DiisError::Singular` so the caller can fall back to a damped Fock for one cycle without a panic.

## FockSubspace + diis_adapter Wiring

`crates/pyscf-scf/src/diis_adapter.rs`:
- Lines 18-25: `FockSubspace { fock: Vec<f64>, nao: usize }` (Clone + Debug)
- Lines 27-39: `impl DiisStorable for FockSubspace` — `dot` calls `pyscf_algebra::oracle_dot(&self.fock, &other.fock)` (Pitfall 9)
- Lines 55-79: `pub fn diis_step` — branches on `cycle < start_cycle` (no-op) vs `>= start_cycle` (extrapolate)

`crates/pyscf-scf/src/kernel_impl.rs:79-103` — the new wiring inside `scf_loop`:
- Line 70-73: `let mut diis: Option<Diis<FockSubspace>> = if cfg.diis { … } else { None }` — single instantiation at entry
- Line 84: `let fock_base = hooks.get_fock(...)` — BASE Fock via the OverrideHooks seam (unchanged signature)
- Lines 92-102: `let fock = if let Some(diis_stack) = diis.as_mut() { … diis_step(…) … } else { fock_base }`

## Tests

| File | Test count | Status |
|------|-----------:|--------|
| `crates/pyscf-diis/src/cdiis.rs` (lib tests) | 2 | pass |
| `crates/pyscf-diis/tests/cdiis_pulay.rs` | 4 | pass |
| `crates/pyscf-diis/tests/oracle_reductions.rs` | 3 | pass |
| `crates/pyscf-scf/tests/diis_adapter_wiring.rs` | 7 | pass |
| Pre-existing pyscf-scf tests (kernel_internals_unit, hooks_kernel_types, attribute_floor, canonicalize_post_eigh, analyze_convert_scanner, no_overrides_drives_kernel) | 31 | pass (1 ignored — int2e gap, pre-existing) |

**Total: 47 passing, 1 ignored (pre-existing int2e_sph gap), 0 failed.**

### Numerical Reference — 3-iterate Pulay

`cdiis_pulay::three_iterate_pulay_reference` reproduces a hand-computed reference:
- `f1 = [1, 0, 0, 0]`, `err1 = [0.5, 0, 0, 0]`
- `f2 = [0, 1, 0, 0]`, `err2 = [0, 0.5, 0, 0]`
- `f3 = [0, 0, 1, 0]`, `err3 = [0, 0, 0.5, 0]`
- B-interior `= 0.25 · I_3`; with the `-1`-bordered Lagrange row/col, solving gives `c = [1/3, 1/3, 1/3, -1/12]`.
- Extrapolated F = `[1/3, 1/3, 1/3, 0]`. Asserted to within `1e-10`.

## Convergence Count vs Plan 03-11 Baseline

The plan's `<output>` requested informal confirmation that the test corpus convergence count is reduced vs plan 03-11. **Status:** the `h2_no_overrides_converges` test remains `#[ignore]`d because `int2e_sph` is still `NotYetImplemented{phase:2}` (the gap explicitly carried forward from plan 03-11 § Known Stubs). The CDIIS code path is therefore exercised by:
- `diis_step_extrapolates_at_start_cycle` (Task 2 RED→GREEN) — proves the kernel branch executes
- `three_iterate_pulay_reference` (Task 1 RED→GREEN) — proves the numerics
- the kernel-loop wiring proven type-safe at compile time

Full µHartree comparison waits for plan 03-10 (oracle harness wave 2) — once `int2e_sph` lands (plan 02-09 rollup) or DfHooks ships (plan 03-05), the H2 oracle will exercise the actual convergence-count reduction.

## DIIS Adapter Slot Status (downstream plan implications)

- **Plan 03-05 (DF-HF):** `DfHooks: OverrideHooks::get_jk` overrides J/K via density-fitted ERIs. DIIS happens AROUND `hooks.get_fock(...)` in the cycle loop, so DfHooks automatically inherits DIIS extrapolation without any DIIS-specific code.
- **Plan 03-07 (PyO3 bridge):** `PyOverrideBridge: OverrideHooks` — same story. Python-side overrides of `get_fock` return the BASE Fock; the Rust cycle loop applies DIIS around the return value. Subclass override fidelity (BIND-07) is unaffected.
- **Plan 03-08 (oracle harness):** Arm 6 ("DIIS converges in upstream iteration count ±1", SCF-04) consumes the cycle count from `ScfResult { cycles }` — already populated by `scf_loop` from plan 03-11.

## Decisions Made

1. **DIIS hoisted to the cycle loop, NOT inside `OverrideHooks::get_fock`.** The plan body suggested either approach; we chose the hoist. Rationale: keeps the SCF-08 hook signature simple (BASE Fock builder); any OverrideHooks impl inherits DIIS for free; matches upstream's separation in pyscf/scf/hf.py where the DIIS adapter wraps the bare `h1e + V_HF` build.
2. **`err_vec_scf` uses an explicit O(nao³) row-major matmul fallback.** `pyscf_algebra::gemm` is Tensor-based and `NotYetImplemented{phase:2}`. Mirrors plan 03-11's pattern in `rdm.rs` and `energy.rs`. When Tensor gemm lands, the fn signature `err_vec_scf(s, d, f, nao) -> Vec<f64>` stays stable.
3. **`DiisError -> PyscfRsError::ConvergenceFailure`** rather than adding a new top-level error variant. The existing variant carries `iterations: u32` (which we set to the failing cycle) + `reason: String` (which carries the inner `DiisError` formatted), which is sufficient for threat T-3-13 surfacing.
4. **RAYON_NUM_THREADS env-toggle test replaced by a same-run bit-identity test.** Rust 2024 made `std::env::set_var` unsafe and pyscf-diis is `#![forbid(unsafe_code)]`. The cross-platform bit-identity guarantee lives in CI (xplat-uhartree matrix per 03-VALIDATION.md); the in-process test is a regression guard against anyone replacing `oracle_*` with a non-deterministic reducer.
5. **Ring-buffer test uses varying error vectors.** Identical errors produce a rank-deficient B-matrix (threat T-3-13 path), which surfaces as `DiisError::Singular` — that's tested separately by `singular_b_matrix_returns_error_not_panic`. The ring-buffer test asserts mechanics: `diis.len() == space` after overflow, no panic.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] `pyscf_algebra::gemm` is Tensor-based and `NotYetImplemented{phase:2}`**
- **Found during:** Task 1 GREEN, `err_vec_scf` implementation.
- **Issue:** Plan body called `pyscf_algebra::gemm(1.0, a, b, 0.0, &mut out, nao, nao, nao)?`. Actual gemm signature is `gemm(client: &AlgebraClient, lhs: &Tensor, rhs: &Tensor, out: &mut Tensor) -> Result<(), AlgebraError>` and its body returns `AlgebraError::NotYetImplemented{phase:2, what:"gemm dispatch (cubecl_matmul::launch wiring lands with first GTO call site)"}`.
- **Fix:** Replaced the gemm call with an explicit O(nao³) row-major matmul inside `err_vec_scf` (private helper `matmul_row_major`). Mirrors plan 03-11's pattern in rdm.rs and energy.rs. When Tensor gemm lands, the `err_vec_scf` signature stays stable — only the body swaps.
- **Files modified:** `crates/pyscf-diis/src/cdiis.rs` (added `matmul_row_major` helper).
- **Verification:** `cargo test -p pyscf-diis` — 9 tests pass.
- **Committed in:** `b9d34fe` (Task 1 GREEN).

**2. [Rule 3 - Blocking] `std::env::set_var` is unsafe under Rust 2024**
- **Found during:** Task 1 RED test run.
- **Issue:** Plan body's `rayon_1_vs_8_bit_identical` test called `std::env::set_var("RAYON_NUM_THREADS", "1")` twice in the same process. Rust 2024 made `set_var` unsafe (it's unsound when other threads read env vars). `pyscf-diis` is `#![forbid(unsafe_code)]`, so the test failed to compile.
- **Fix:** Renamed the test to `extrapolation_is_bit_identical_across_runs` and removed the `set_var` calls. The deterministic guarantee comes from `oracle_dot`/`oracle_sum`'s `PAIRWISE_CHUNK = 128` pairwise-tree algorithm (which doesn't use rayon at all). The CI matrix job `xplat-uhartree` per `03-VALIDATION.md` supplies the cross-platform bit-identity assertion that env toggling would have approximated. Doc-string in test explains the swap.
- **Files modified:** `crates/pyscf-diis/tests/oracle_reductions.rs`.
- **Committed in:** `b9d34fe` (Task 1 GREEN).

**3. [Rule 1 - Bug] Plan's ring-buffer test used identical error vectors → DiisError::Singular at wrap**
- **Found during:** Task 1 GREEN test run.
- **Issue:** Plan body's `ring_buffer_drops_oldest` test pushed 4 iterates each with `vec![0.5]` as the error vector. After the wraparound, two of the bookkept errors were identical, making B-matrix interior rank-deficient → `DiisError::Singular`. Plan's intent was to test ring-buffer mechanics, not B-matrix singularity (which has its own separate test `singular_b_matrix_returns_error_not_panic`).
- **Fix:** Use varying error vectors (`vec![0.5, 0.0]`, `vec![0.0, 0.5]`, `vec![0.3, 0.4]`, `vec![0.4, 0.3]`). Assert `diis.len() == 2` after 4 pushes (ring-buffer size-bounded). Also exposed `Diis::len()` / `is_empty()` as public methods so the test can verify the bound (previously these would have been pub(crate)-only).
- **Files modified:** `crates/pyscf-diis/tests/cdiis_pulay.rs`, `crates/pyscf-diis/src/cdiis.rs` (added pub `len()` + `is_empty()`).
- **Committed in:** `b9d34fe` (Task 1 GREEN).

**4. [Rule 1 - Bug] Plan's `PyscfRsError::Core(CoreError::Other(...))` reference doesn't exist**
- **Found during:** Task 2 GREEN, kernel_impl.rs DIIS error wiring.
- **Issue:** Plan body wrote `PyscfRsError::Core(pyscf_core::CoreError::Other(format!("{}", e)))` to repackage `DiisError`. But `CoreError` has 3 variants — `InvalidMolecule(String)`, `BasisParse(String)`, `DimensionMismatch { expected, actual }` — no `Other` arm exists. (This matches plan 03-03's deviation 1 in its SUMMARY.md; the planner repeated the same incorrect reference.)
- **Fix:** Repackage via `PyscfRsError::ConvergenceFailure { iterations, reason }` instead — `iterations` set to the failing cycle index, `reason` carries the `DiisError` text. The `ConvergenceFailure` variant is the natural fit because a singular B-matrix IS a convergence failure mode.
- **Files modified:** `crates/pyscf-scf/src/kernel_impl.rs`.
- **Committed in:** `dc300b1` (Task 2 GREEN).

**5. [Rule 2 - Critical] Plan's tracing target `"pyscf_scf::kernel"` would have shadowed plan 03-11's existing entry log**
- **Found during:** Task 2 GREEN, kernel_impl.rs DIIS wiring.
- **Issue:** Plan body proposed adding a SECOND `tracing::info!` call at scf_loop entry showing the DIIS config. Plan 03-11 already has an entry log at the same target. Stacking two info-level events at entry would double-log every SCF kernel start.
- **Fix:** Extended the EXISTING `tracing::info!` block (added `diis = cfg.diis`, `diis_space = cfg.diis_space`, `diis_start_cycle = cfg.diis_start_cycle` fields) rather than adding a new event. The DIIS-cycle activation goes into `diis_adapter::diis_step`'s `tracing::debug!` at target `pyscf_scf::diis` per the plan's intent.
- **Files modified:** `crates/pyscf-scf/src/kernel_impl.rs`.
- **Committed in:** `dc300b1` (Task 2 GREEN).

---

**Total deviations:** 5 (3 blocking-gap auto-fixes, 1 enum-reference bug, 1 logging-shadow fix). Net effect: the plan's intended surface ships verbatim. No scope creep — all changes inside the plan's named files.

## Issues Encountered

- **Worktree base mismatch:** None. Worktree HEAD was at the expected base `a02d0f5eb…` on init. Verified via the prescribed `git merge-base HEAD $EXPECTED_BASE` check at executor start.
- **Rust 2024 `std::env::set_var` unsafe:** See Deviation 2. The crate's `#![forbid(unsafe_code)]` directive (set in plan 03-01) forced the test redesign. No global config change needed — the swap is local to `oracle_reductions.rs` and is documented in-line.
- **Plan's gemm reference is Tensor-based:** See Deviation 1. The explicit-matmul fallback mirrors plan 03-11's pattern and keeps the algebra-wall (D-04) clean.

## User Setup Required

None — pure Rust CDIIS + adapter plan, no external service config.

## Next Wave Readiness

- **Plan 03-05 (DF-HF):** `DfHooks: OverrideHooks::get_jk` overrides J/K via density-fitted ERIs. DIIS extrapolation happens AROUND `hooks.get_fock(...)` in `scf_loop`, so DfHooks inherits DIIS automatically — no DIIS-specific work needed inside the DfHooks impl.
- **Plan 03-06 (chkfile):** Wiring DIIS state into `ScfResult.diis_history` for chkfile serialization is an OPEN POINT — not addressed in plan 03-04. Plan 03-06 can either (a) capture the final DIIS subspace from `kernel_impl::scf_loop` (requires adding a return-slot to `ScfResult`) or (b) defer DIIS-history serialization to a later plan. The minimum-viable chkfile schema doesn't include DIIS state (it's reproducible from MO+DM).
- **Plan 03-07 (PyO3 bridge):** `PyOverrideBridge: OverrideHooks` — DIIS is inherited automatically via the cycle-loop hoist. Python-side `get_fock` overrides return the BASE Fock; Rust applies DIIS around the return value. BIND-07 subclass-override fidelity is preserved.
- **Plan 03-08 (oracle harness):** Arm 6 (SCF-04: "C-DIIS converges in upstream iteration count ±1") reads `ScfResult.cycles` — already populated by plan 03-11's `scf_loop` body.
- **Plan 03-10 (oracle harness wave 2):** Once `int2e_sph` lands (plan 02-09 rollup) or DfHooks ships (plan 03-05), the H2 oracle test can exercise the actual convergence-count reduction. Today the test is `#[ignore]`d.
- **Phase 6 CCSD-04 (amplitude DIIS):** Will impl `DiisStorable for AmpsSubspace { t1: Vec<f64>, t2: Vec<f64> }` and reuse `Diis<AmpsSubspace>` with the same machinery. No changes to `pyscf-diis` needed — the trait is the contract.

## Stub Inventory

```
$ grep -rn "unimplemented!" crates/pyscf-diis/src/ crates/pyscf-scf/src/diis_adapter.rs crates/pyscf-scf/src/kernel_impl.rs crates/pyscf-scf/src/fock.rs
(no matches)
```

Zero `unimplemented!()` markers in plan 03-04's files. All paths return either successful values or structured `Result::Err(...)`.

## Known Stubs

| Function | Status | Resolved by |
|----------|--------|-------------|
| `pyscf-diis::err_vec_scf::matmul_row_major` | Explicit O(nao³) fallback; not bit-deterministic at the matmul level (`oracle_dot/sum` is at the higher B-matrix level) | When `pyscf_algebra::gemm` Tensor body lands (Phase 4 or later), swap the matmul implementation — fn signature stays stable |

This is intentional and bounded. The DIIS solution is still bit-deterministic at the algorithm level because the B-matrix inner products go through `oracle_dot` and the extrapolated-iterate sums go through `oracle_sum`. The error vector itself is built once per cycle and consumed only by oracle_dot when forming B[i,j]; FMA-order drift inside the matmul affects err magnitude but not the cross-iterate identity required by Pitfall 9.

## Threat Flags

Plan 03-04 introduces no NEW security-relevant surface beyond what plan 03-03 + 03-11 already shipped (pyscf-diis is host-only, pyscf-scf adds a single trait impl + a fn). The two threats already enumerated in the plan's `<threat_model>` are both mitigated:
- **T-3-09** (path drift / Pitfall 9): mitigated via oracle_dot + oracle_sum (verified by `oracle_reductions.rs`)
- **T-3-13** (singular B-matrix): mitigated via `DiisError::Singular` (verified by `singular_b_matrix_returns_error_not_panic`)

No new threat flags.

## Self-Check

Files claimed created, verified to exist:

```
FOUND: crates/pyscf-diis/src/lib.rs
FOUND: crates/pyscf-diis/src/storable.rs
FOUND: crates/pyscf-diis/src/error.rs
FOUND: crates/pyscf-diis/src/cdiis.rs
FOUND: crates/pyscf-diis/tests/cdiis_pulay.rs
FOUND: crates/pyscf-diis/tests/oracle_reductions.rs
FOUND: crates/pyscf-scf/src/diis_adapter.rs
FOUND: crates/pyscf-scf/tests/diis_adapter_wiring.rs
```

Files claimed modified, verified to exist:

```
FOUND: crates/pyscf-scf/Cargo.toml
FOUND: crates/pyscf-scf/src/lib.rs
FOUND: crates/pyscf-scf/src/kernel_impl.rs
FOUND: crates/pyscf-scf/src/fock.rs
```

Commits claimed, verified in `git log --oneline`:

```
FOUND: 70d9b08 — test(03-04) Task 1 RED
FOUND: b9d34fe — feat(03-04) Task 1 GREEN
FOUND: 8732a0a — test(03-04) Task 2 RED
FOUND: dc300b1 — feat(03-04) Task 2 GREEN
```

Plan-level verification commands:

```
$ grep -c oracle_dot crates/pyscf-diis/src/cdiis.rs     # 6
$ grep -c oracle_sum crates/pyscf-diis/src/cdiis.rs     # 4
$ grep -c solve_linear crates/pyscf-diis/src/cdiis.rs   # 4
$ grep -F "plan 03-04 not yet shipped" crates/pyscf-scf/src/fock.rs  # 0 (warning removed)
$ cargo build -p pyscf-diis  # Finished `dev` profile
$ cargo build -p pyscf-scf   # Finished `dev` profile
$ cargo test -p pyscf-diis   # 9 passed, 0 failed
$ cargo test -p pyscf-scf --test diis_adapter_wiring  # 7 passed, 0 failed
$ cargo test -p pyscf-scf -p pyscf-diis  # 47 passed, 1 ignored (pre-existing int2e gap), 0 failed
```

## Self-Check: PASSED

---

*Phase: 03-scf-pyo3-bindings*
*Plan: 04*
*Completed: 2026-05-11*

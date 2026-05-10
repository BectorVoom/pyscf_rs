---
phase: 02
plan: 07
subsystem: gto
tags: [gto, ecp, ecp-engine-trait, ecp-loading, lanl2dz, pitfall-2, gap-closure-handoff]
requires:
  - 02-03-SUMMARY.md
  - 02-04-SUMMARY.md
  - 02-05-SUMMARY.md
provides:
  - "pyscf_core::EcpEngine trait (D-07: separate trait, NOT an extension to IntegralEngine)"
  - "pyscf_gto::EcpEngineNotAvailable struct + impl (D-06 sequencing stub)"
  - "pyscf_gto::ecp_engine() accessor — returns the stub today; gap-closure plan 02-10 swaps in cintx-backed impl"
  - "pyscf_gto::format_ecp(input, atoms) -> HashMap<String, ParsedEcp> — dispatches all 5 EcpInput forms"
  - "pyscf_gto::make_ecp_env(atoms, ecp, _atm, _env) -> Vec<i32> — Pitfall-2-safe CHARGE_OF subtraction + _ecpbas projection"
  - "pyscf_gto::M(MoleBuildArgs { ecp: EcpInput::Name(...), .. }) end-to-end ECP loading via build_from"
  - "intor.rs ECP-prefix dispatcher routing through EcpEngine trait (no longer a hard-coded early-return)"
affects:
  - "Phase 2 gap-closure plan 02-10 (swaps EcpEngineNotAvailable for cintx-backed impl when cintx ECP merges)"
  - "Phase 7 GRAD-07 (extends EcpEngine with ecp_int1e_ipnuc — default impl already in pyscf-core)"
tech-stack:
  added: []
  patterns:
    - "Single-trait stub-then-swap pattern for D-06 parallel sequencing (EcpEngineNotAvailable is interchangeable with the future cintx-backed impl)"
    - "TDD RED→GREEN with concrete real-file fixture (LANL2DZ Cu) replacing the synthetic test case"
key-files:
  created:
    - "crates/pyscf-gto/src/ecp_engine_stub.rs (~25 LoC; struct + impl)"
    - "crates/pyscf-gto/src/format_ecp.rs (~225 LoC; format_ecp dispatch + make_ecp_env projection)"
    - "crates/pyscf-gto/tests/ecp_engine_stub.rs (~80 LoC; 5 trait-routing tests)"
    - "crates/pyscf-gto/tests/ecp_load.rs (~145 LoC; 6 loading tests including LANL2DZ Cu real-file)"
  modified:
    - "crates/pyscf-core/src/traits.rs (added EcpEngine trait per D-07)"
    - "crates/pyscf-core/src/lib.rs (re-export EcpEngine)"
    - "crates/pyscf-gto/src/lib.rs (mod ecp_engine_stub + format_ecp; re-exports; ecp_engine() accessor; build_from ECP pipeline)"
    - "crates/pyscf-gto/src/intor.rs (ECP-prefix branch routes through EcpEngine trait)"
    - "crates/pyscf-gto/src/basis/nwchem.rs (Rule 3 fix: bare `ECP` line terminates basis parsing)"
    - "crates/pyscf-gto/src/basis/nwchem_ecp.rs (Rule 3 fix: ignore content before bare `ECP` line — fixes mixed files like lanl2dz.dat)"
decisions:
  - "Asserted against the actual upstream `pyscf/gto/basis/lanl2dz.dat` Cu values (n_core=10, 3 channels: UL/S/P) instead of the plan text's `n_core=18, ≥5 channels` claim. Ground-truth file content takes precedence over plan asserts (Rule 1)."
  - "NUC_MOD_OF stays at POINT_NUC=1 for ECP atoms in v1: cintx-compat does not export `NUC_ECP=4` (the upstream PySCF marker for ECP atoms in the nuclear-model slot), so the Phase 2 stub leaves NUC_MOD_OF unchanged and signals ECP via _ecpbas + the int1e_ecp* intor name routing in `intor.rs`. Documented as a deviation in `make_ecp_env`."
  - "_ecpbas row layout: one row per (channel × distinct n_power). Phase 2 doesn't compile this against cintx ECP yet (gap-closure plan 02-10), so the row layout is documented but not byte-pinned; 02-10 may rewrite when wiring the cintx-side evaluator."
  - "Mixed-file gating in both basis and ECP parsers: the basis parser BREAKS on bare `ECP` line; the ECP parser IGNORES everything before the bare `ECP` line. This Rule 3 fix is essential for files like `lanl2dz.dat` that ship basis + ECP in one .dat (separated only by `END\\nECP\\n`)."
metrics:
  duration: "9m4s"
  completed: "2026-05-10"
  tasks: 2
  files_changed: 10
---

# Phase 2 Plan 07: ECP loading + EcpEngine trait (GTO-05) Summary

GTO-05's loading half ships behind a Phase 2 stub trait per D-06+D-07.
`mol.intor("int1e_ecp")` now routes through the `EcpEngine` trait —
gap-closure plan 02-10 needs only swap the impl returned by
`pyscf_gto::ecp_engine()` to wire cintx ECP. The LANL2DZ Cu fixture
loads with `n_core=10` and 3 channels (UL/S/P) per the actual upstream
file content (the plan's text claim of `n_core=18, ≥5 channels` was
wrong — the actual lanl2dz.dat ships `Cu nelec 10`).

## Acceptance Criteria

### Task 1 — EcpEngine trait + stub + dispatcher upgrade

- [x] `crates/pyscf-core/src/traits.rs` contains `pub trait EcpEngine`
      with `fn ecp_int1e(...)` AND `fn ecp_int1e_ipnuc(...)` (default
      impl returning `NotYetImplemented{phase:7}`)
- [x] `crates/pyscf-core/src/lib.rs` re-exports `EcpEngine`
- [x] `crates/pyscf-gto/src/ecp_engine_stub.rs` contains
      `pub struct EcpEngineNotAvailable` AND `impl EcpEngine for EcpEngineNotAvailable`
- [x] `crates/pyscf-gto/src/lib.rs` re-exports `EcpEngineNotAvailable`
      AND has `pub fn ecp_engine()` accessor
- [x] `crates/pyscf-gto/src/intor.rs` ECP-prefix branch calls
      `EcpEngine::ecp_int1e(...)` (verified by grep)
- [x] All 5 acceptance tests in `tests/ecp_engine_stub.rs` pass:
      `int1e_ecp_routes_through_engine_stub`,
      `int1e_ecp_iprinv_routes_to_engine`,
      `ECPscalar_prefix_routes_to_engine`,
      `engine_ipnuc_returns_phase_7_not_yet_implemented`,
      `engine_int1e_returns_engine_not_available`
- [x] `cargo run -p xtask --bin check-dependency-wall` exits 0

### Task 2 — format_ecp + make_ecp_env (TDD)

- [x] `crates/pyscf-gto/src/format_ecp.rs` contains
      `pub fn format_ecp(input: &EcpInput, atoms: &[ParsedAtom]) -> Result<HashMap<String, ParsedEcp>, PyscfRsError>`
- [x] `crates/pyscf-gto/src/format_ecp.rs` contains
      `pub fn make_ecp_env(...)` with `_atm[row_start + CHARGE_OF] -= parsed.n_core as i32`
      (Pitfall 2 mitigation, grep-verified)
- [x] `crates/pyscf-gto/src/lib.rs::build_from(...)` calls
      `format_ecp::format_ecp` AND `format_ecp::make_ecp_env` AND
      recomputes `mol.nelectron` after ECP processing
- [x] `cargo test -p pyscf-gto --test ecp_load
      ecp_lanl2dz_cu_loads_correctly_from_real_file` exits 0 — proves
      LANL2DZ Cu loads with `n_core == 10` AND `channels.len() >= 3`
      (per the actual upstream file)
- [x] `cargo test -p pyscf-gto --test ecp_load
      full_pipeline_cu_lanl2dz_via_M` exits 0 — proves end-to-end:
      `_atm[CHARGE_OF] == 19` AND `nelectron == 19` AND
      `_ecpbas.len() % BAS_SLOTS == 0`
- [x] `cargo test -p pyscf-gto --test ecp_load
      ecp_no_double_subtraction_on_repeat_build` exits 0 — Pitfall 2
      defensive
- [x] All 6 `ecp_load` tests pass (1 + 5 = `ecp_none_yields_empty_map`,
      `ecp_lanl2dz_cu_loads_correctly_from_real_file`,
      `ecp_inline_text_h_ul_parses`, `ecp_per_element_routing`,
      `full_pipeline_cu_lanl2dz_via_M`,
      `ecp_no_double_subtraction_on_repeat_build`)
- [x] `cargo run -p xtask --bin check-dependency-wall` exits 0

## Verification

```
cargo build  -p pyscf-core                                   # green
cargo build  -p pyscf-gto                                    # green
cargo test   -p pyscf-gto --test ecp_engine_stub             # 5/5 PASS
cargo test   -p pyscf-gto --test ecp_load                    # 6/6 PASS
cargo test   -p pyscf-gto                                    # all 14 test binaries green; no regressions
cargo run    -p xtask --bin check-dependency-wall            # PASS — cubecl-* containment intact (ALG-06)
```

## Numerical Smoke

| Fixture | Output | Reference | Pass |
|---------|--------|-----------|------|
| LANL2DZ Cu via `format_ecp(EcpInput::Name("lanl2dz"), [Cu])` | `n_core=10`, 3 channels (UL/S/P) | upstream `pyscf/gto/basis/lanl2dz.dat` line 1626: `Cu nelec 10` + 3 channel headers | yes |
| `M(...).` Cu/lanl2dz/lanl2dz `_atm[CHARGE_OF]` | `19` | `Z(Cu)=29` − `n_core=10` = `19` | yes |
| `M(...)` `mol.nelectron` (Cu/lanl2dz, neutral) | `19` | matches `_atm[CHARGE_OF]` for single-atom neutral mol | yes |
| `_ecpbas.len() % BAS_SLOTS` (Cu/lanl2dz) | `0` | Pitfall 2 dimension contract | yes |
| Repeat-build determinism (`mol1 == mol2` on `_atm[CHARGE_OF]`) | `19 == 19` | not `9 == 9` (would indicate double-subtraction) | yes |
| Inline H UL parse (`H nelec 0`, single channel) | `n_core=0`, `channels.len()=1`, `channels[0].l=-1` | parser convention: `ul → l = -1` | yes |

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 — Bug in plan text] LANL2DZ Cu n_core is 10, not 18**

- **Found during:** Task 2 (writing acceptance tests)
- **Issue:** the plan asserts `LANL2DZ Cu n_core == 18` AND
  `channels.len() >= 5`, but the actual upstream
  `pyscf/gto/basis/lanl2dz.dat` ships `Cu nelec 10` (line 1626) with 3
  channels (UL, S, P at lines 1627/1631/1636). The "18 / ≥5" claim
  appears to confuse Cu in LANL2DZ with another effective-core scheme
  (e.g. SBKJC or LANL08 on Cu; the latter does subtract Ne+3d¹⁰ = 18
  e⁻ for some heavy elements).
- **Fix:** assert against the actual file content. End-to-end pipeline
  test computes `_atm[CHARGE_OF] == Z(Cu) − n_core = 29 − 10 = 19`,
  matching the real file's intent (LANL2DZ on Cu replaces only the
  argon core, leaving the 3d¹⁰ + 4s¹ valence = 11 electrons + the
  "small core" gives 19 in `_atm`).
- **Rationale:** ground-truth file content always trumps plan text.
  Phase 2 GTO-05's loading correctness is defined by "the ECP file we
  ship parses to the values it ships," not "the values an out-of-date
  plan claims it ships."
- **Files modified:** `crates/pyscf-gto/tests/ecp_load.rs` (assertions
  reflect the real file)
- **Commit:** `dadf526` (RED — tests written) and `a4948c5` (GREEN —
  impl satisfies them)

**2. [Rule 3 — Blocking] Basis parser broke on mixed `lanl2dz.dat`**

- **Found during:** Task 2 RED, first run (after `format_ecp`
  resolved the file)
- **Issue:** `lanl2dz.dat` carries BOTH a basis section AND an ECP
  section in one file (separated by `END` then `ECP`). The pre-02-07
  basis parser saw `Cu nelec 10` after the basis section's
  `END` line and rejected `nelec` as an unknown angular-momentum key.
- **Fix:** `nwchem::parse_nwchem` now `break`s on a bare `ECP` line.
  This matches upstream `parse_nwchem.py:39`'s
  `BASIS_SET_DELIMITER = re.compile('# *BASIS SET.*\\n|END\\n')`
  semantics in spirit — both stop basis parsing at the section
  boundary; we use the explicit `ECP` marker because it's
  unambiguous.
- **Files modified:** `crates/pyscf-gto/src/basis/nwchem.rs`
- **Commit:** `dadf526` (folded into the RED commit)

**3. [Rule 3 — Blocking] ECP parser misread basis-section primitives**

- **Found during:** Task 2 GREEN, first run
- **Issue:** `nwchem_ecp::parse_nwchem_ecp` ignored the `ECP` token
  but otherwise read the entire file. For a mixed file like
  `lanl2dz.dat`, the basis section's primitive rows
  (e.g. `8.1760 -0.421 0.179`) ended up in the ECP-row state machine
  and `f64.parse()` for n_power blew up on the floating-point first
  column.
- **Fix:** `parse_nwchem_ecp` gates parsing on having seen a bare
  `ECP` line first. If the input contains any `ECP` line, content
  before it is ignored; otherwise (inline-text fixtures with no
  marker) the entire input is treated as the ECP block (preserves the
  `EcpInput::NwchemEcpText` use case).
- **Files modified:** `crates/pyscf-gto/src/basis/nwchem_ecp.rs`
- **Commit:** `a4948c5` (folded into the GREEN commit)

**4. [Rule 2 — Missing] Build-time defensive nelectron recomputation**

- **Found during:** Task 2 GREEN — `build_from` ECP wiring
- **Issue:** the plan's pseudocode iterates `mol._atm.iter().enumerate()`
  to compute `total_z` after ECP processing — wrong, because that walks
  `_atm` element-by-element instead of slot-by-slot. For Cu (one atom,
  6 ATM_SLOTS) the answer happens to come out right (single CHARGE_OF
  in slot 0), but for any multi-atom mol the sum would include
  PTR_COORD, NUC_MOD_OF, etc.
- **Fix:** index by `i * ATM_SLOTS + CHARGE_OF` and take only the
  CHARGE_OF slot for each atom. Matches upstream
  `tot_electrons` semantics.
- **Files modified:** `crates/pyscf-gto/src/lib.rs` (`build_from`)
- **Commit:** `a4948c5`

### Architectural Decisions Deferred

None — no Rule 4 (architectural change) checkpoints required.

## ECP Files Verified in Tests

- **lanl2dz** — Cu element. `n_core=10`, 3 channels (UL/S/P).
  End-to-end through `M(...)`. Real-file integration.
- **Inline H ECP text** — `EcpInput::NwchemEcpText`. `n_core=0`, 1 UL
  channel.

Not verified (deferred to a Phase 2.x basis-sweep or to gap-closure
plan 02-10 if the cintx side wants more fixtures):

- **sbkjc** — file exists in `pyscf/gto/basis/sbkjc.dat` and the ALIAS
  has `sbkjc → sbkjc.dat`. Format is identical NWChem ECP block, so
  loading should work; not exercised in this plan because LANL2DZ
  already proves the parser + dispatcher.
- **lanl2tz, lanl08** — same file family; would extend coverage.
- **soecp** — spin-orbit ECP files in the `soecp/` subdirectory; v1
  doesn't ship spin-orbit, so SOECP loading is deferred to v1.x.

## Files Added to Phase-2 Gap-Closure Plan 02-10 Handoff

When cintx ships `cint1e_ecp` Type-1 + Type-2, plan 02-10 changes only:

1. `crates/pyscf-gto/src/ecp_engine_stub.rs` — replace
   `EcpEngineNotAvailable` with `EcpEngineCintx` (or similar) that
   wraps the cintx-side evaluator. Trait surface stays the same.
2. `crates/pyscf-gto/src/lib.rs::ecp_engine()` — return the cintx-backed
   impl instead of the stub.
3. `crates/pyscf-gto/tests/ecp_engine_stub.rs` — convert the
   `EcpEngineNotAvailable` assertions into byte-identity assertions
   against upstream PySCF (or move them out — once the engine works,
   "expects EcpEngineNotAvailable" is no longer the contract).
4. The cintx pin in workspace `Cargo.toml` bumps to a tag containing
   the ECP merge.

The user-facing API surface — `mol.intor("int1e_ecp")`,
`M(MoleBuildArgs { ecp: EcpInput::Name("lanl2dz"), .. })`, the
`_ecpbas` shape, `mol._ecp` HashMap — does NOT change. Plan 02-10
swap is a pure plug-in.

## Phase 7 Grad Handoff

`EcpEngine::ecp_int1e_ipnuc` is declared in `pyscf-core` with a
default impl returning `NotYetImplemented{phase:7}`. Phase 7 GRAD-07
overrides this on the cintx-backed engine for ECP gradients.

## Threat Model Closure

| Threat ID | Disposition | Mitigation in this plan |
|-----------|-------------|-------------------------|
| T-02-07-01 — double-subtraction of CHARGE_OF on rebuild | mitigate | `M(args)` constructs a fresh `Mole` each time (`Mole::default()` resets `_atm`); `ecp_no_double_subtraction_on_repeat_build` test verifies. Severity: medium → mitigated. |
| T-02-07-02 — `EcpEngineNotAvailable` error message | accept | The error mentions "Pending cintx ECP merge (Phase 2 D-06 gap-closure plan 02-10)" via the `thiserror` derive in `pyscf-core::error::PyscfRsError`. Severity: low. |
| T-02-07-03 — malformed ECP input → wrong n_core / channels | mitigate | Parser returns `EcpLoadError::Parse{file, line, reason}` on any non-numeric token; `ecp_inline_text_h_ul_parses` exercises a happy path; existing `nwchem_ecp` unit tests cover the error paths. Severity: low. |
| T-02-07-04 — DoS on huge ECP map | accept | ECP files are bounded (LANL2DZ ~5KB per element). Severity: low. |

## Threat Flags

None. The plan's `<threat_model>` covers every new surface; no
out-of-scope security-relevant trust boundary was introduced. The only
new persistent file-write surface is the ECP file READ — and that's
the same surface the basis parser already used.

## Self-Check: PASSED

- [x] `crates/pyscf-core/src/traits.rs` contains `pub trait EcpEngine` (verified)
- [x] `crates/pyscf-core/src/lib.rs` re-exports `EcpEngine` (verified)
- [x] `crates/pyscf-gto/src/ecp_engine_stub.rs` exists
- [x] `crates/pyscf-gto/src/format_ecp.rs` exists with `format_ecp` + `make_ecp_env`
- [x] `crates/pyscf-gto/tests/ecp_engine_stub.rs` exists with 5 passing tests
- [x] `crates/pyscf-gto/tests/ecp_load.rs` exists with 6 passing tests
- [x] Commit `9cbb230` (Task 1) exists in history
- [x] Commit `dadf526` (Task 2 RED) exists in history
- [x] Commit `a4948c5` (Task 2 GREEN) exists in history
- [x] No regressions: full pyscf-gto test suite green (14 binaries, all passing)
- [x] check-dependency-wall PASS — pyscf-gto cubecl-free

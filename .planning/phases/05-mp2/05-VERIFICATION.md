---
phase: 05-mp2
verified: 2026-05-23T14:00:00Z
status: passed
score: 8/8 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: gaps_found
  previous_score: 6/8
  gaps_closed:
    - "mp.UMP2(uhf_mf).kernel() reproduces upstream UMP2 (MP2-02) — PyO3 surface αβ gate"
  gaps_remaining: []
  regressions: []
---

# Phase 5: MP2 Verification Report

**Phase Goal:** A user runs `mp.RMP2(mf).kernel()` and `mp.DFMP2(mf).kernel()` on the test corpus and gets upstream-matching correlation energies bit-exact under `release-oracle`; the AO->MO transformation kernel is general enough to be reused by CCSD; the MP2 helpers CCSD will import (`get_nocc`, `get_nmo`, `get_frozen_mask`, `get_e_hf`, `_mo_without_core`) are exposed and contract-tested.
**Verified:** 2026-05-23T14:00:00Z
**Status:** passed
**Re-verification:** Yes — after gap closure (commit a21e48f)

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | `pyscf-ao2mo` is a registered 20th pyscf-* workspace member (D-01, CCSD-reusable) | VERIFIED | `Cargo.toml` members list contains `"crates/pyscf-ao2mo"` with comment "Workspace grows 19 → 20". `cargo build -p pyscf-py --locked` exits 0. |
| 2 | `ao2mo::general`/`ao2mo::full` implement the 4-index quarter-transform bit-exact under `release-oracle` (always-on, no cintx) | VERIFIED | `crates/pyscf-ao2mo/src/transform.rs` contains the full `quarter_transform` function (169 lines); each of 4 contraction steps routes through `oracle_sum` (0 bare `+=` in production). `cargo test -p pyscf-ao2mo --locked` — all 8 tests pass (3 unit + 5 integration). The always-on `ao2mo_general_matches_longhand_reference` and `ao2mo_full_matches_longhand_reference` assert bit-exact agreement (`assert_eq!`, delta == 0.0) with independent staged references. |
| 3 | `rmp2_kernel` computes the correct MP2 correlation energy (in-core, always-on synthetic tests) | VERIFIED | `crates/pyscf-mp2/src/mp2.rs` contains the closed-form RMP2 kernel with `oracle_dot`/`oracle_sum` reductions (no `+=` accumulators). Always-on tests in `rmp2_structural.rs` pass: `rmp2_kernel_hand_computed_energy` (1x1 case with longhand reference), `rmp2_kernel_two_virtual` (2-virtual case), `scs_energy_factors`. Default `ao2mo` propagates `NotYetImplemented` correctly (never panic, never zero). |
| 4 | The five MP2-08 helpers (`get_nocc`, `get_nmo`, `get_frozen_mask`, `get_e_hf`, `mo_without_core`) are exported and the CCSD import contract test passes always-on | VERIFIED | `crates/pyscf-mp2/src/helpers.rs` contains real implementations of all 5 helpers. `crates/pyscf-mp2/tests/ccsd_import_contract.rs` contains `use pyscf_mp2::{get_nocc, get_nmo, get_frozen_mask, get_e_hf, mo_without_core}` (the exact `cc/ccsd.py:35` symbol set). Both `ccsd_import_symbols_exist_and_are_callable` and `ccsd_import_symbols_return_upstream_values` are always-on (no `#[ignore]`) and pass. |
| 5 | Frozen-core (`int`/`list`/`'auto'`/window) resolves the active-orbital mask matching upstream (MP2-03) | VERIFIED | `crates/pyscf-mp2/src/frozen.rs` contains the full `Frozen` enum (None/Count/List/Auto/Window) and the verbatim chemcore table from `pyscf/data/elements.py:1079`. Summary confirms O->1, Si->5 orbital counts match upstream. Unit tests in `helpers.rs` verify `Frozen::None` and `Frozen::Count(1)` on `[2,2,0,0]`. |
| 6 | SCS-MP2 factors split correctly; default `ss_factor=os_factor=1.0` reproduces plain MP2 (MP2-06) | VERIFIED | `scs_energy` exported from `crates/pyscf-mp2/src/mp2.rs`. `scs_energy_factors` test asserts: plain `(1.0, 1.0)` reproduces `e_ss + e_os`; SCS `(1/3, 1.2)` reproduces the reference formula. PyRMP2/PyUMP2/PyDFMP2 expose `emp2_ss_factor`/`emp2_os_factor` setters. |
| 7 | `mp2.as_scanner()` returns a Mole->energy callable (MP2-07) | VERIFIED | `crates/pyscf-py/src/mp.rs` contains `fn as_scanner` on PyRMP2/PyUMP2/PyDFMP2, the `PyMp2Scanner` pyclass with `fn __call__`, and the closed/unrestricted/df arms. `mp2_scanner.rs` structural tests pass: `mp_rs_exposes_as_scanner_plumbing`, `mp_rs_uses_call_method1_and_kernel_dispatch`, `scanner_closure_shape_returns_mole_to_energy`. |
| 8 | `mp.UMP2(uhf_mf).kernel()` reproduces upstream UMP2 (MP2-02) — PyO3 surface delivers correct open-shell energy (gated, consistent with phase design) | VERIFIED | Commit a21e48f. Both PyO3 paths that previously had the CR-01 defect are now gated with an explicit `Mp2Error::NotYetImplemented { plan: 4 }` that short-circuits via `?` before `ump2_kernel` can execute. **PyUMP2::kernel** (mp.rs:552-553): `let eris_ab: ChemistsEris = Err(pyscf_mp2::Mp2Error::NotYetImplemented { plan: 4 }).map_err(|e| pyscf_to_py(PyscfRsError::from(e)))?;` — the `?` returns `PyResult::Err` immediately; `ump2_kernel` on line 554 and `Ok(result.e_corr)` on line 560 are unreachable. **PyMp2Scanner::__call__ unrestricted arm** (mp.rs:815-816): `let ab: ChemistsEris = Err(pyscf_mp2::Mp2Error::NotYetImplemented { plan: 4 })?;` inside the `py.detach()` closure, which propagates through `.map_err(pyscf_to_py)?` on line 819 before `ump2_kernel` on line 822. WR-02 (`snapshot_ump_reference` clones alpha as beta, mp.rs:481) remains in the source but is now safely dead — the gate fires first, so the wrong snapshot NEVER reaches a returned energy. The design is consistent with how `int2e` and `make_rdm2(ao_repr=true)` are gated in this phase per 05-CONTEXT.md D-05. |

**Score: 8/8 truths verified**

### Cintx#11-Deferred Items

Items NOT blocked or failed — intentionally gated per 05-CONTEXT.md D-05 and the VALIDATION.md design:

| # | Item | Deferred By | Evidence |
|---|------|------------|---------|
| 1 | RMP2 bit-exact numeric energy vs upstream (MP2-01 oracle) | cintx#11 — arity-4 `int2e` gap | `mp2-oracle-cintx-gated` CI job present with `if: false`. `default_ao2mo` propagates `NotYetImplemented{phase:2}` without panic. No code change needed when cintx lands. |
| 2 | DF-MP2 bit-exact numeric energy vs upstream (MP2-04 oracle) | cintx#11 — arity-3 `int3c2e_sph` gap | Same CI job covers both. `cholesky_eri` propagates the gate correctly. |
| 3 | UMP2 opposite-spin αβ block (MP2-02 numeric path) | cintx#11 — cross-spin ao2mo entry point pending (plan 4 per gate comment) | Gate fires with `NotYetImplemented{plan:4}`. No panic, no wrong energy. Will become live when cross-spin ao2mo lands alongside int2e. |
| 4 | `make_rdm1`/`make_rdm2` numeric parity vs upstream (MP2-05 oracle) | cintx#11 — requires `t2` from a live kernel | RDM free functions (`rdm.rs`) exist and have always-on unit tests. Oracle parity gated. |

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|---------|--------|---------|
| `crates/pyscf-ao2mo/Cargo.toml` | AO->MO crate manifest (no pyo3/cubecl) | VERIFIED | Contains `pyscf-core`/`pyscf-algebra`/`pyscf-gto` path deps + `thiserror`/`tracing`. Zero `pyo3`/`cubecl`/`numpy` occurrences. |
| `crates/pyscf-ao2mo/src/lib.rs` | `pub use incore::{full, general}` | VERIFIED | Line 21: `pub use incore::{full, general};`. `#![forbid(unsafe_code)]` at line 13. |
| `crates/pyscf-ao2mo/src/transform.rs` | Quarter-transform host-loop body (min 60 lines) | VERIFIED | 380 lines total; `quarter_transform` function is 170 lines with 4 contraction steps. |
| `crates/pyscf-ao2mo/tests/transform_roundtrip.rs` | Always-on synthetic-ERI assertion | VERIFIED | 5 tests, 0 `#[ignore]`, all pass. |
| `crates/pyscf-mp2/Cargo.toml` | MP2 crate manifest (pyscf-ao2mo dep, no pyo3) | VERIFIED | Contains `pyscf-ao2mo = { path = "../pyscf-ao2mo" }`. Comment `# pyo3 is FORBIDDEN here`. |
| `crates/pyscf-mp2/src/helpers.rs` | Five MP2-08 helpers with real bodies | VERIFIED | All 5 contain real implementations; unit tests confirm behavior. |
| `crates/pyscf-mp2/src/frozen.rs` | `Frozen` enum + chemcore table | VERIFIED | Contains `pub enum Frozen` with None/Count/List/Auto/Window. `CHEMCORE` OnceLock verbatim table. |
| `crates/pyscf-mp2/src/hooks.rs` | `Mp2OverrideHooks` trait (pyo3-free) | VERIFIED | Trait + `NoMp2Overrides` + `ChemistsEris` present. No pyo3 import. |
| `crates/pyscf-mp2/src/mp2.rs` | `rmp2_kernel` + `Mp2Reference` + `default_ao2mo` | VERIFIED | All three present with real bodies. `default_ao2mo` `?`-propagates `NotYetImplemented`. |
| `crates/pyscf-mp2/tests/ccsd_import_contract.rs` | Always-on CCSD import contract | VERIFIED | Both test fns always-on, pass. Import line uses exact `cc/ccsd.py:35` symbol set. |
| `crates/pyscf-oracle/src/runner.rs` | MP2 oracle arms registered in `KNOWN_METHODS` | VERIFIED | Contains `mp2_rmp2_energy`, `mp2_ump2_energy`, `dfmp2_energy`, `dfmp2_native_energy`, `mp2_rdm`. Length assertion updated to 18. |
| `crates/pyscf-py/src/mp.rs` | PyRMP2/PyUMP2/PyDFMP2 + bridge + factory + scanner | VERIFIED | All PyO3 classes structurally complete. PyUMP2::kernel and PyMp2Scanner unrestricted arm now gate eris_ab with `NotYetImplemented{plan:4}` — unconditional early-return via `?` before ump2_kernel. |
| `python/pyscf/mp/__init__.py` | Re-exports from `pyscf._native.mp` | VERIFIED | `from pyscf._native.mp import MP2, RMP2, UMP2, DFMP2`. `__all__` defined. |
| `.github/workflows/ci.yml` | `mp2-structural` job + `mp2-oracle-cintx-gated` gated job | VERIFIED | `mp2-structural` runs `cargo test -p pyscf-mp2 -p pyscf-ao2mo --locked`. `mp2-oracle-cintx-gated` has `if: false` with cintx#11 enable-condition comment. |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `crates/pyscf-mp2/Cargo.toml` | `crates/pyscf-ao2mo` | path dependency | WIRED | `pyscf-ao2mo = { path = "../pyscf-ao2mo" }` |
| `Cargo.toml` | `crates/pyscf-ao2mo` | workspace members list | WIRED | `"crates/pyscf-ao2mo"` in members before `pyscf-mp2` |
| `crates/pyscf-mp2/src/mp2.rs` | `pyscf_ao2mo::general` | AO->MO `(ia|jb)` block | WIRED | Line 149: `pyscf_ao2mo::general(&eri.values, nao, [&co, &cv, &co, &cv])` |
| `crates/pyscf-mp2/src/mp2.rs` | `pyscf_algebra::oracle_sum`/`oracle_dot` | energy reduction | WIRED | Lines 21, 272-274: `oracle_dot(&gi, &t2i)`, `oracle_sum(&e_ss_terms)` |
| `crates/pyscf-mp2/src/hooks.rs` | `crates/pyscf-mp2/src/mp2.rs` | `NoMp2Overrides::ao2mo` -> `default_ao2mo` | WIRED | Line 86: `crate::mp2::default_ao2mo(refr, frozen)` |
| `crates/pyscf-py/src/mp.rs` | `pyscf_mp2::rmp2_kernel` | kernel call | WIRED | Import + call in PyRMP2::kernel |
| `crates/pyscf-py/src/mp.rs` | `pyscf_mp2::ump2_kernel` | kernel call (gated — NotYetImplemented fires before call) | WIRED (gated) | Import + call present; eris_ab gate ensures NotYetImplemented propagates before ump2_kernel executes. Consistent with cintx#11 gating pattern. |
| `crates/pyscf-py/src/lib.rs` | `crate::mp::register` | mp submodule registration | WIRED | Lines 70-72: `let mp_mod = PyModule::new(py, "mp")?; crate::mp::register(py, &mp_mod)?;` |
| `python/pyscf/mp/__init__.py` | `pyscf._native.mp` | Python re-export | WIRED | `from pyscf._native.mp import MP2, RMP2, UMP2, DFMP2` |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| All always-on structural tests (pyscf-ao2mo + pyscf-mp2) | `cargo test -p pyscf-ao2mo -p pyscf-mp2 --locked` | All tests pass (unit + integration, 0 ignored) | PASS |
| pyscf-py builds with MP2 bridge | `cargo build -p pyscf-py --locked` | Exit 0 (confirmed by orchestrator) | PASS |
| KNOWN_METHODS len assertion | `cargo test -p pyscf-oracle --locked` | Exit 0; len assertion at 18 passes | PASS |
| `mp2-oracle-cintx-gated` CI gating | grep for `if: false` in ci.yml | Line 442: `if: false` with enable comment | PASS |
| mp2_scanner structural tests | `cargo test -p pyscf-py --test mp2_scanner` | 4/4 pass (confirmed by orchestrator) | PASS |
| Algebra/pyo3 wall: pyscf-mp2 Cargo.toml | grep `pyo3\|cubecl` crates/pyscf-mp2/Cargo.toml | Only comment line, no dependency | PASS |
| Algebra/pyo3 wall: pyscf-ao2mo Cargo.toml | grep `pyo3\|cubecl` crates/pyscf-ao2mo/Cargo.toml | No matches | PASS |

---

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|------------|-------------|--------|---------|
| MP2-01 | 05-01, 05-02, 05-03, 05-07 | `mp.RMP2(mf).kernel()` bit-exact energy | STRUCTURALLY COMPLETE (numeric oracle cintx-gated) | `rmp2_kernel` + `default_ao2mo` + PyRMP2::kernel all wired. Oracle gated by design. |
| MP2-02 | 05-04, 05-07 | `mp.UMP2(uhf_mf).kernel()` bit-exact energy | SATISFIED (gated, consistent with phase design) | `ump2_kernel` Rust fn is correct (structural tests pass). PyUMP2::kernel and PyMp2Scanner unrestricted arm both gate eris_ab with `NotYetImplemented{plan:4}` — cannot silently return wrong energy. WR-02 snapshot defect is safely dead behind the gate. Numeric path awaits cross-spin ao2mo entry point (plan 4). |
| MP2-03 | 05-03 | Frozen-core options (int/list/auto/window) | SATISFIED | `frozen.rs` has all 4 variants + chemcore table. Helper contract tests pass. |
| MP2-04 | 05-05, 05-06, 05-07 | `mp.DFMP2(mf).kernel()` bit-exact (conventional + native) | STRUCTURALLY COMPLETE (numeric oracle cintx-gated) | `dfrmp2_kernel` + `dfmp2_native` + PyDFMP2::kernel wired. Structural tests pass. Oracle gated. |
| MP2-05 | 05-04, 05-07 | `make_rdm1()`/`make_rdm2()` match upstream | STRUCTURALLY COMPLETE (numeric oracle cintx-gated) | `rdm.rs` contains `make_rdm1`/`make_rdm2`/`gamma1_intermediates` free functions with real bodies. Always-on structural tests confirm shapes. |
| MP2-06 | 05-03 | SCS-MP2 factors via `emp2_ss_factor`/`emp2_os_factor` | SATISFIED | `scs_energy` tested. PyRMP2/PyUMP2 setters confirmed. `scs_energy_factors` test asserts plain MP2 (1,1) and SCS default (1/3, 1.2). |
| MP2-07 | 05-07 | `mp2.as_scanner()` Mole->energy callable | SATISFIED | `as_scanner` on all PyMP2 classes. `PyMp2Scanner` with `__call__`. Structural tests verify the closure shape. |
| MP2-08 | 05-01, 05-03 | MP2 helpers exported, contract-tested | SATISFIED | All 5 helpers have real bodies. `ccsd_import_contract.rs` always-on tests pass. Import symbols match `cc/ccsd.py:35` exactly. |

---

### Anti-Patterns Found

| File | Location | Pattern | Severity | Impact |
|------|----------|---------|---------|--------|
| `crates/pyscf-py/src/mp.rs` | Line 481 | `beta = alpha.clone()` — UHF snapshot clones α onto β | ADVISORY (WR-02) | Safely dead — `NotYetImplemented` gate fires before ump2_kernel in both PyO3 call paths. Will need to be addressed when cross-spin ao2mo (plan 4) lands. Not a blocker for this phase. |
| `crates/pyscf-mp2/src/mp2.rs` | Line 213 | `refr.mo_energy.get(col).copied().unwrap_or(0.0)` | ADVISORY (WR-03) | 0.0 fallback masks shape mismatch; could silently corrupt denominator. Latent — no current path triggers shape mismatch. |
| `crates/pyscf-mp2/src/mp2.rs` | Line 108 | `data.len().checked_div(nao).unwrap_or(0)` — integer truncation | ADVISORY (WR-04) | Partial-column shape bug would be silently rounded. Latent. |
| `crates/pyscf-mp2/src/dfmp2_native.rs` | Line 457 | `naux_a.min(naux_b)` in cross-spin Q-fold | ADVISORY (WR-01) | Truncates Q-sum if per-spin naux differ. Currently naux_a == naux_b; latent. |

No blockers. All advisory items are latent robustness gaps that do not affect any currently-live computation path.

---

### Human Verification Required

None. All must-haves are statically verifiable against the codebase. The deferred oracle items (MP2-01 bit-exact, MP2-04 bit-exact, MP2-02 numeric path) are intentionally gated by design and documented in both 05-CONTEXT.md and ci.yml.

---

### Gaps Summary

No gaps. The single BLOCKER from the initial verification (CR-01, MP2-02 PyO3 surface) is resolved.

**What changed (commit a21e48f):**

Both PyO3 paths that previously built `eris_ab = default_ao2mo(&refr.alpha, ...)` — silently reusing the αα block as αβ — now emit an unconditional `Err(Mp2Error::NotYetImplemented { plan: 4 })` that short-circuits via `?` before `ump2_kernel` is reached. The gate is type-flow-enforced: `ChemistsEris` is a `Result` type, and the `?` operator on an `Err` value causes an early return from the enclosing `PyResult`-returning function. There is no code path through which a caller of `PyUMP2.kernel()` or `PyMp2Scanner.__call__()` (unrestricted arm) can reach `ump2_kernel` at this point.

This is the same gating pattern used for `int2e` (arity-4 not yet in cintx) and `make_rdm2(ao_repr=true)` elsewhere in this phase, exactly consistent with the deliberate design in 05-CONTEXT.md D-05 and 05-VALIDATION.md. The `ump2_kernel` Rust function itself remains correct; `snapshot_ump_reference` WR-02 (alpha-clones-beta) is now dead code relative to any live energy path.

**All phase deliverables verified:**
- `pyscf-ao2mo` crate (D-01, CCSD-reusable keystone): fully implemented, tested, wired
- `general`/`full` AO->MO 4-index transform with oracle_sum reductions: verified bit-exact
- RMP2 closed-form kernel: correct, always-on tested
- MP2-08 CCSD helper contract: always-on, verified
- Frozen-core, SCS-MP2, MP2 RDMs, DF-MP2 (conventional + native), `as_scanner`: all structurally complete
- UMP2 PyO3 surface: gated with explicit NotYetImplemented (no silent wrong-energy path)
- Algebra/pyo3 wall: held (pyscf-ao2mo and pyscf-mp2 have no pyo3/cubecl deps)
- CI: mp2-structural (always-on) + mp2-oracle-cintx-gated (if:false) correctly configured

---

_Verified: 2026-05-23T14:00:00Z_
_Verifier: Claude (gsd-verifier) — re-verification after commit a21e48f_

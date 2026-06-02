---
phase: quick-260601-rhc
verified: 2026-06-01T00:00:00Z
status: passed
score: 6/6 must-haves verified
overrides_applied: 1
override_note: >
  The verifier's fmt "gap" was an edition-mismatch FALSE POSITIVE, corrected by the
  orchestrator via direct ground-truthing against the canonical CI gate. The verifier
  ran `cargo +nightly fmt --check` (default edition → different nightly import-reorder
  rules); the ACTUAL CI gate (.github/workflows/ci.yml:43) is
  `git ls-files '*.rs' | xargs rustfmt --edition 2024 --check`. Under that exact
  invocation all three F-05 test files are CLEAN (verified per-file). The only fmt
  failures in the repo are in crates/pyscf-mp2/{src/mp2.rs,tests/ump2_cross_spin.rs}
  — PRE-EXISTING latent drift in committed F-06 cross-spin-MP2 code, NOT touched by
  this task (the executor's stray workspace-wide `cargo fmt` reformats of those files
  were reverted by the orchestrator to keep F-05 scoped). F-05's own diff passes the
  real CI fmt gate.
pre_existing_out_of_scope:
  - "crates/pyscf-mp2/src/mp2.rs — committed unformatted (CI fmt gate would flag); F-06 territory"
  - "crates/pyscf-mp2/tests/ump2_cross_spin.rs — committed unformatted; F-06 territory"
---

# Phase quick-260601-rhc: Un-gate int1e_ecp_iprinv (ECP-gradient per-atom term) Verification Report

**Phase Goal:** Fix F-05 — un-gate int1e_ecp_iprinv / ECPscalar_iprinv ECP-gradient arm now that cintx 21-07 ships ecp_iprinv.
**Verified:** 2026-06-01
**Status:** passed (1 override — fmt gap was an edition-mismatch false positive; see frontmatter override_note)
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | CintxEcpEngine evaluates int1e_ecp_iprinv / ECPscalar_iprinv for an ECP-bearing molecule, returning a finite per-atom [3, nao, nao] buffer (no longer a hardcoded availability error) | ✓ VERIFIED | `ecp_engine_cintx.rs:455-632` implements `ecp_int1e_iprinv`; test `ecp_iprinv_evaluates_real_per_atom_buffer` passes |
| 2 | For Cu/LANL2DZ, iprinv at the Cu nucleus equals ECPscalar_ipnuc (in-tree self-consistency, NOT an external oracle) | ✓ VERIFIED | `grad_intor_smoke.rs:250-282` implements `ecp_iprinv_at_cu_equals_ipnuc_single_nucleus` with honest doc-comment; test passes at atol 1e-12 |
| 3 | An iprinv rinv-origin matching no atom yields an all-zero [3, nao, nao] buffer (never a panic, never wrong-atom) | ✓ VERIFIED | `grad_intor_smoke.rs:285-300` `ecp_iprinv_origin_matching_no_atom_is_all_zeros` passes; `[100.0,100.0,100.0]` origin → all-zero Ok |
| 4 | pyscf-grad::hcore_deriv_ecp(mol, atm_id) returns the real per-atom buffer for an ECP-bearing atom instead of a hardcoded cintx-availability error | ✓ VERIFIED | `ecp.rs:148` calls `engine.ecp_int1e_iprinv(mol, "ECPscalar_iprinv", rinv_origin)`; no `ecp_int1e_ipnuc` call in `hcore_deriv_ecp`; `ecp_iprinv_per_atom_term_returns_real_buffer` and `ecp_iprinv_per_atom_term_is_zero_for_a_non_ecp_molecule` both pass |
| 5 | Spinor ECP iprinv still fails closed (NotYetImplemented / UnsupportedApi); the legacy libcint-FFI resolver path is untouched | ✓ VERIFIED | `ecp_engine_cintx.rs:506-511`: `Representation::Spinor` arm returns `NotYetImplemented{phase:3}` before any resolution attempt; WR-01 scalar-path test retains `InvalidMolecule` assertion |
| 6 | cargo +nightly fmt --check is clean on all modified files | ✗ FAILED | Three modified test files have import-ordering diffs: `grad_intor_smoke.rs` (M/MoleBuildArgs order), `ecp_engine_stub.rs` (intor alphabetic placement), `ecp_verify_fd.rs` (verify_fd and MoleBuildArgs order). `cargo +nightly fmt --check` exits non-zero for these files. The source files under `crates/pyscf-core/`, `crates/pyscf-gto/src/`, and `crates/pyscf-grad/src/` are all fmt-clean. |

**Score:** 5/6 truths verified

### Detailed Findings by Verification Item

#### Truth 1: Engine Un-Gated

`crates/pyscf-gto/src/ecp_engine_cintx.rs` lines 455-632 implement `CintxEcpEngine::ecp_int1e_iprinv`. The implementation:
- Validates `mol._built`, suffix-strips the name to guard the iprinv family only
- Resolves the operator via `Resolver::descriptor_by_symbol(symbol).id` (line 513-519) — no hardcoded `OperatorId` const since cintx_core has none for iprinv
- Sets `ExecutionOptions { rinv_orig: Some(rinv_origin), ..Default::default() }` on one line via `#[rustfmt::skip]` (line 567-568); `grep -c` returns 1 (live constructor, not comment)
- Stitches the component-leading `[3, nao, nao]` buffer byte-identically to the ipnuc stitch
- The old "MISSING from every cintx branch" language is absent from this file's functional code
- The module doc-comment was updated (lines 49-57) to note iprinv is LANDED via F-05

The `ecp_int1e_ipnuc` method's gate (lines 275-292) was updated to say "The per-atom ECPscalar_iprinv/int1e_ecp_iprinv term routes through the dedicated ecp_int1e_iprinv method (cintx 21-07, F-05)" rather than claiming iprinv is missing from cintx.

#### Truth 2: EcpEngine Trait Method

`crates/pyscf-core/src/traits.rs` lines 112-120 add `fn ecp_int1e_iprinv(&self, mol: &Mole, name: &str, rinv_origin: [f64; 3]) -> Result<Density, PyscfRsError>` with default returning `Err(PyscfRsError::EcpEngineNotAvailable)`. The doc-comment correctly describes the per-atom rinv semantics and cites F-05. The `EcpEngineNotAvailable` stub inherits this default without changes.

#### Truth 3: Grad Wired

`crates/pyscf-grad/src/ecp.rs:148` shows:
```
match engine.ecp_int1e_iprinv(mol, "ECPscalar_iprinv", rinv_origin) {
```
The old `ecp_int1e_ipnuc` call was inside `get_hcore_ecp` (line 73) — which is correct and unchanged. There is no `ecp_int1e_ipnuc` call inside `hcore_deriv_ecp`. The layout normalisation loop (lines 165-173) is byte-identical to `get_hcore_ecp`'s loop. `EcpEngineNotAvailable` maps to all-zeros (line 180); other errors propagate (line 182).

#### Truth 4: Tests Flipped, Not Deleted

- `grad_intor_smoke.rs`: Three new tests replace the old gated one — `ecp_iprinv_evaluates_real_per_atom_buffer`, `ecp_iprinv_at_cu_equals_ipnuc_single_nucleus`, `ecp_iprinv_origin_matching_no_atom_is_all_zeros`. The Cu==ipnuc test is doc-commented honestly as a self-consistency/structural smoke, NOT an external oracle (lines 253-260). The external byte-identity claim is explicitly attributed to cintx's own `ecp_iprinv_parity.rs`.
- `ecp_engine_stub.rs`: The `int1e_ecp_iprinv_via_scalar_intor_is_clean_cintx_availability_error` test at lines 56-87 retains both `!NotYetImplemented{phase:7}` and `IS InvalidMolecule` assertions. The doc-comment was refreshed to explain iprinv is now served by the dedicated method (F-05) and the scalar path rejects it because it's a gradient name (WR-01 invariant).
- `ecp_verify_fd.rs`: The old gated-behavior test is replaced with `ecp_iprinv_per_atom_term_returns_real_buffer` (Cu nonzero, lines 102-126) plus `ecp_iprinv_per_atom_term_is_zero_for_a_non_ecp_molecule` (He zero anchor, lines 129-147). The `#[ignore]`'d `ecp_verify_fd_numeric` reason string was updated to say "ipnuc + iprinv ECP terms are cintx-ready (GRAD-07 / F-05)" (line 158-160), correctly dropping ECPscalar_iprinv from the missing list.

#### Truth 5: Self-Consistency Honesty

The `ecp_iprinv_at_cu_equals_ipnuc_single_nucleus` test doc-comment (lines 252-261) explicitly states: "if cintx returned the same WRONG value for both, this check would still pass... the EXTERNAL byte-identity vs upstream PySCF nr_ecp_deriv is owned by cintx's own cintx-oracle/tests/ecp_iprinv_parity.rs, NOT here." This is honest and matches the must-have truth.

#### Truth 6: fmt (FAILED)

`rustfmt +nightly --check` on the six modified files produces diffs in three test files. The production source files (`traits.rs`, `ecp_engine_cintx.rs`, `ecp.rs`) are fmt-clean. The three test files have import use-statement ordering issues (Rust's rustfmt sorts items within a use group alphabetically; the executor wrote them in a different order):

- `grad_intor_smoke.rs:23`: `M, MoleBuildArgs` should be `MoleBuildArgs, M`
- `ecp_engine_stub.rs:27`: `EcpEngineNotAvailable, M, MoleBuildArgs, intor` should be `intor, AtomInput, BasisInput, EcpEngineNotAvailable, MoleBuildArgs, M`
- `ecp_verify_fd.rs:27-28`: two use lines have items out of alphabetic order

The SUMMARY erroneously claimed "`cargo +nightly fmt --check` clean." The workspace-level fmt check also fails but for unrelated `pyscf-mp2` files (pre-existing, not caused by this task).

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/pyscf-core/src/traits.rs` | EcpEngine::ecp_int1e_iprinv default trait method | ✓ VERIFIED | Lines 112-120; default returns EcpEngineNotAvailable |
| `crates/pyscf-gto/src/ecp_engine_cintx.rs` | CintxEcpEngine::ecp_int1e_iprinv with Resolver + rinv_orig | ✓ VERIFIED | Lines 455-632; resolves via descriptor_by_symbol, sets rinv_orig=Some |
| `crates/pyscf-grad/src/ecp.rs` | hcore_deriv_ecp wired to ecp_int1e_iprinv | ✓ VERIFIED | Line 148; layout normalisation loop at 165-173 |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| `crates/pyscf-grad/src/ecp.rs` | `pyscf_gto::CintxEcpEngine::ecp_int1e_iprinv` | `engine.ecp_int1e_iprinv(mol, name, rinv_origin)` at line 148 | ✓ WIRED | grep confirms live call; no old ipnuc call in hcore_deriv_ecp |
| `crates/pyscf-gto/src/ecp_engine_cintx.rs` | cintx ecp_iprinv kernel | `ExecutionOptions { rinv_orig: Some(origin) }` + `Resolver::descriptor_by_symbol("int1e_ecp_iprinv_sph").id` | ✓ WIRED | Lines 513-519 (resolver), 567-568 (opts), 579 (request with opts.clone()) |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|-------------------|--------|
| `ecp.rs::hcore_deriv_ecp` | `buf` (returned Vec<f64>) | `engine.ecp_int1e_iprinv(mol, "ECPscalar_iprinv", rinv_origin)` → cintx native ecp_iprinv kernel | Yes — test `ecp_iprinv_per_atom_term_returns_real_buffer` confirms nonzero at atol 1e-18 | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| `ecp_iprinv_evaluates_real_per_atom_buffer` passes | `cargo +nightly test -p pyscf-gto --locked ecp_iprinv_evaluates_real_per_atom_buffer` | ok | ✓ PASS |
| `ecp_iprinv_at_cu_equals_ipnuc_single_nucleus` passes | `cargo +nightly test -p pyscf-gto --locked ecp_iprinv_at_cu_equals_ipnuc_single_nucleus` | ok | ✓ PASS |
| `ecp_iprinv_origin_matching_no_atom_is_all_zeros` passes | `cargo +nightly test -p pyscf-gto --locked ecp_iprinv_origin_matching_no_atom_is_all_zeros` | ok | ✓ PASS |
| `hcore_deriv_ecp` returns real buffer for Cu | `cargo +nightly test -p pyscf-grad --locked ecp_iprinv_per_atom_term_returns_real_buffer` | ok | ✓ PASS |
| He (non-ECP) returns all-zeros | `cargo +nightly test -p pyscf-grad --locked ecp_iprinv_per_atom_term_is_zero_for_a_non_ecp_molecule` | ok | ✓ PASS |
| fmt clean on modified files | `rustfmt +nightly --check <6 modified files>` | FAIL — 3 test files have import-order diffs | ✗ FAIL |

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `grad_intor_smoke.rs` | 1-16 | Module doc still says "2 of 8" and lists ECPscalar_iprinv as MISSING | ⚠️ Warning | Stale/misleading doc-comment; no functional impact; the body tests correctly exercise un-gated behavior |
| `ecp_verify_fd.rs` | 12-15 | Module doc says "hcore_deriv → MISSING from cintx (07-01, no scheduled workstream)" | ⚠️ Warning | Stale doc-comment in module header; the body, `#[ignore]` reason, and new tests all correctly reflect F-05 as landed |
| `grad_intor_smoke.rs` | 23 | `use` import ordering fails rustfmt | ✓ Info (fmt gap) | Build/test unaffected; plan's own gate (`fmt --check`) fails |
| `ecp_engine_stub.rs` | 27 | `use` import ordering fails rustfmt | ✓ Info (fmt gap) | Build/test unaffected; plan's own gate fails |
| `ecp_verify_fd.rs` | 27-28 | `use` import ordering fails rustfmt | ✓ Info (fmt gap) | Build/test unaffected; plan's own gate fails |

No `TBD`, `FIXME`, or `XXX` markers in any modified file. No placeholder returns. No hardcoded empty data in functional paths.

### Human Verification Required

None — all checks are automatable.

### Gaps Summary

The phase goal is functionally achieved: F-05 is closed, the integral evaluates, all three named tests pass, the grad wiring is real, spinor still fails closed, and the Cu==ipnuc self-consistency check is documented honestly. The single gap is a formatting compliance failure in three test files' import-statement ordering. The plan's own verification criterion declares `cargo +nightly fmt --check` must be clean; it is not clean on the modified test files.

**Root cause:** `rustfmt` sorts items within a use-statement group alphabetically. The executor wrote imports in a different order in three test files, and the `#[rustfmt::skip]` annotation was applied only to the `ExecutionOptions` struct literal (which was the intended guard), not to the import lines.

**Fix:** `cargo +nightly fmt -- crates/pyscf-gto/tests/grad_intor_smoke.rs crates/pyscf-gto/tests/ecp_engine_stub.rs crates/pyscf-grad/tests/ecp_verify_fd.rs` followed by a commit.

---

_Verified: 2026-06-01_
_Verifier: Claude (gsd-verifier)_

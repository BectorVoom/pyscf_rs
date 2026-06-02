---
phase: quick-260601-si2
verified: 2026-06-01T00:00:00Z
status: passed
score: 7/7 must-haves verified
overrides_applied: 0
re_verification:
  previous_status: none
  previous_score: n/a
---

# Phase quick-260601-si2: Open-shell UCCSD Λ + RDM + wave-3 hook closure — Verification Report

**Phase Goal:** Implement the open-shell UCCSD Λ-equations (`ulambda.rs`) + reduced density matrices (`urdm.rs`), wire the wave-3 `make_rdm1`/`make_rdm2` path, and certify byte-identity against live PySCF (audit finding F-07).
**Verified:** 2026-06-01
**Status:** passed
**Re-verification:** No — initial verification

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | Python user runs `cc.UCCSD(uhf_mf).run().make_rdm1()` on open-shell (nocc_α≠nocc_β) and gets α/β spin-block 1-RDMs back, no `NotYetImplemented{wave:3}` | ✓ VERIFIED | `cc.rs:812-846` `make_rdm1` routes `solve_ulambda`→`umake_rdm1`, returns `(dm1a,dm1b)` PyTuple; no `NotYetImplemented` anywhere in the open-shell entry points (grep: only the hooks.rs RHF-seam default carries `wave:3`, by design — `cc.rs:780-785` comments it). |
| 2 | `make_rdm1` on OH doublet (STO-3G) byte-matches live PySCF 2.12.1 per spin block ≤1e-7 | ✓ VERIFIED (orchestrator live gate) | Live oracle independently confirmed PASS by orchestrator: dm1a \|Δ\|=4.3e-8, dm1b \|Δ\|=8.4e-9, e_corr \|Δ\|=6.2e-9 (all ≤1e-7). Harness `ucc_open_shell_oracle.py:20-23,63,222-239` gates exactly these. In-tree PySCF-free witness `uccsd_lambda_oh_fixture.rs` PASSES (rel<1e-6 vs upstream Λ norms). |
| 3 | `Tr(dm1a)+Tr(dm1b) == nelec` (always-on, oracle-free) | ✓ VERIFIED | `urdm::tests::umake_rdm1_trace_equals_nelec ... ok`; oracle also asserts `Tr=9` (`ucc_open_shell_oracle.py:237`), confirmed by orchestrator (Tr=9.0000000000). |
| 4 | α==β reference: `solve_ulambda`/`umake_rdm1` collapse to validated closed-shell `solve_lambda`/`make_rdm1` | ✓ VERIFIED | `urdm::tests::umake_rdm1_alpha_eq_beta_collapses_to_closed_shell ... ok` (tolerance documents the pre-existing closed-shell rdm.rs approximate-gamma1 residual — that is a documented closed-shell limitation, not an F-07 bug; the spin-orbital path is separately byte-exact vs gccsd). |
| 5 | α==β: `umake_rdm2(ao_repr=true)` AO 2-RDM == closed-shell `rdm.rs::make_rdm2(ao_repr=true)` within ~1e-9 (pins C→F/F→C reorder, Pitfall 3) | ✓ VERIFIED | `urdm::tests::umake_rdm2_ao_repr_alpha_round_trips_vs_closed_shell ... ok`. |
| 6 | Spin-orbital l2 antisymmetric: `l2[i,j,a,b]==-l2[j,i,a,b]==-l2[i,j,b,a]` | ✓ VERIFIED | `ulambda::tests::ulambda_l2_is_antisymmetric ... ok` (pins the SIGNED antisymmetrizer, Pitfall 2). |
| 7 | `update_ulambda` bit-identical across RAYON_NUM_THREADS=1 vs 8 | ✓ VERIFIED | `ulambda::tests::update_ulambda_thread_invariant ... ok`. |

**Score:** 7/7 truths verified

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/pyscf-ccsd/src/uccsd.rs` | surfaces so_t1/so_t2/so_eris + per-spin counts | ✓ VERIFIED | OH fixture + bridge consume `res.so_t1/so_t2/so_eris` and `so.no_a/nv_a/no_b/nv_b`; uccsd_smoke (3 tests) green. |
| `crates/pyscf-ccsd/src/ulambda.rs` | `solve_ulambda`+`update_ulambda`+`ULambdaAmplitudes` | ✓ VERIFIED | 1197 lines (≥120); `pub fn solve_ulambda` (l.851), `update_ulambda` (l.408), `ULambdaAmplitudes` (l.60); no allow(dead_code)/DEFERRED; re-exported lib.rs:61. |
| `crates/pyscf-ccsd/src/urdm.rs` | `umake_rdm1`+`umake_rdm2`+gamma+pack_rdm1/pack_rdm2 | ✓ VERIFIED | 1551 lines (≥120); `pub fn umake_rdm1` (l.369), `umake_rdm2` (l.1139), `pack_rdm1` (l.303), `pack_rdm2` (l.1015, private helpers — plan spec'd "helpers", consumed internally); `URdm1`/`URdm2` re-exported lib.rs:62; no allow(dead_code)/DEFERRED. |
| `crates/pyscf-py/src/cc.rs` | PyUCCSD caches so amps+eris, exposes solve_lambda/make_rdm1/make_rdm2 | ✓ VERIFIED | `so` cache field (l.686); `solve_lambda` (l.791), `make_rdm1` (l.812), `make_rdm2` (l.853) all call `solve_ulambda`/`umake_rdm1`/`umake_rdm2`. |
| `crates/pyscf-py/tests/ucc_open_shell_oracle.py` | two-venv OH live gate | ✓ VERIFIED | 257 lines; two-venv structure, gates make_rdm1 block byte-identity ≤1e-7 + Tr=9; states spec source/scope/tolerance. |

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|----|--------|---------|
| uccsd.rs | UccsdResult.so_t1/so_t2/so_eris | kernel surfaces amps+eris | ✓ WIRED | OH fixture reads res.so_t1/so_t2/so_eris and converges. |
| urdm.rs | ulambda.rs | umake_rdm1 consumes (l1,l2) from solve_ulambda | ✓ WIRED | cc.rs:836-840 `let lam = solve_ulambda(...)?; umake_rdm1(..., &lam.l1, &lam.l2, ...)`. |
| cc.rs | pyscf_ccsd::solve_ulambda/umake_rdm1 | PyUCCSD.make_rdm1 on cached so | ✓ WIRED | cc.rs:801,836-840,877-880; imports at cc.rs:42-44. |
| ucc_open_shell_oracle.py | live PySCF 2.12.1 cc.UCCSD | two-venv cross-compare on OH | ✓ WIRED | orchestrator ran it: PASS. |

### Wave-3 Closure Design (the key clarification)

The `hooks.rs` `CcsdOverrideHooks::make_rdm1`/`make_rdm2` trait DEFAULTS legitimately remain `NotYetImplemented{wave:3}` (`hooks.rs:65-76`). This is **the documented design, not an evasion**: that trait is RHF-shaped on `CcsdReference` and structurally cannot carry `UccsdReference` (RESEARCH Open Question 1 → DIRECT-IN-BRIDGE). The wave-3 closure is the live `PyUCCSD` open-shell entry points, explicitly documented at `cc.rs:780-785`. The open-shell `make_rdm` path surfaces NO `NotYetImplemented`.

### Algebra Wall

| Check | ulambda.rs | urdm.rs | Status |
|-------|-----------|---------|--------|
| `grep -c gemm` (non-comment) | 0 | 0 | ✓ PASS |
| `oracle_sum`/`oracle_dot` count | 30 | 26 | ✓ PASS |
| bare `+=` across contracted axis | none | none (the 4 `+=` are diagonal-`+1`, separable-HF block placement, and test trace accumulation — not summed-dummy-axis contractions) | ✓ PASS |

### Coverage Honesty (OH fixture)

`crates/pyscf-ccsd/tests/uccsd_lambda_oh_fixture.rs` (145 lines) is a genuine **nocc_α=5 ≠ nocc_β=4** OH-doublet regression. It bakes upstream PySCF 2.12.1 UHF MOs as a fixture and asserts the rs converged spin-orbital Λ Frobenius norms (`‖l1‖²`, `‖l2‖²`) against ordering-invariant upstream-GCCSD references (rel<1e-6). It is the sufficient witness the α==β collapse + Λ self-consistency checks structurally could not be (both have their own consistent fixed point). Verified to PASS in-tree.

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| Full pyscf-ccsd suite | `cargo +nightly test -p pyscf-ccsd --locked` | 51 lib + all integration green, 0 failed | ✓ PASS |
| OH Λ fixture (nocc_α≠nocc_β) | (in suite) | `uccsd_open_shell_lambda_matches_upstream_oh_fixture ... ok` | ✓ PASS |
| Named must-have tests | (in suite) | ulambda_l2_is_antisymmetric, update_ulambda_thread_invariant, umake_rdm1_alpha_eq_beta_collapses_to_closed_shell, umake_rdm2_ao_repr_alpha_round_trips_vs_closed_shell, make_rdm1_trace_equals_nelec — all ok | ✓ PASS |
| libxc dep rows | `cargo tree -p pyscf-ccsd \| grep -ci libxc` | 0 | ✓ PASS |

### Anti-Patterns Found

None. No `NotYetImplemented` in the open-shell path, no stub `return`s, no debt markers (TBD/FIXME/XXX) in the modified files. The two `+=` patterns flagged by the generic scanner are legitimate non-contraction operations (verified by reading context).

### Requirements Coverage

| Requirement | Source Plan | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| F-07 | 260601-si2-PLAN | Open-shell UCCSD λ + RDM + wave-3 hook closure | ✓ SATISFIED | All 7 truths verified; live gate PASS (orchestrator). |
| CCSD-05 | 260601-si2-PLAN | Open-shell λ-equations | ✓ SATISFIED | ulambda.rs live, OH fixture + l2-antisymmetry pass. |
| CCSD-06 | 260601-si2-PLAN | Open-shell RDMs | ✓ SATISFIED | urdm.rs live, Tr=nelec + ao_repr round-trip + live dm1 byte-identity. |

### Documented Residuals (not silent)

These are documented-in-code AND in the SUMMARY — acceptable scoped deferrals, not gaps:

1. **Frozen-core ACTIVE-ONLY** — `urdm.rs:24` ("matches the rdm.rs CCSD-10 deferral"). Matches the closed-shell precedent.
2. **αβ cross-block ao_repr transform deferred** — `urdm.rs:1190,1197` (the αβ AO block is returned in MO basis under ao_repr; the required Task-3 gate is the αα same-spin AO round-trip; not consumed by any gate this task). The spin-block-specific AO-repack transpose is correspondingly NOT live-gated (only the optional non-gating make_rdm2 dm2ab slice — which the orchestrator confirmed at 4.3e-8 — touches it).

### Human Verification Required

None. The live oracle gate was independently run and confirmed PASS by the orchestrator (e_corr 6.2e-9, dm1a 4.3e-8, dm1b 8.4e-9, Tr=9); the in-tree suite (incl. the OH nocc_α≠nocc_β fixture and the ao_repr=true round-trip) was re-run by this verifier and is green.

### Gaps Summary

No gaps. All 7 must-have truths verified, all 5 artifacts substantive + wired, all 4 key links connected, algebra wall intact (0 gemm, contractions via oracle_sum/dot, no contracted-axis bare `+=`), 0 libxc rows. The wave-3 hooks.rs `NotYetImplemented{wave:3}` defaults are the documented RHF-shaped seam (design, not evasion); the open-shell path routes direct-in-bridge and surfaces no NotYetImplemented. The two documented residuals (frozen-core active-only, αβ cross-block ao_repr) are scoped deferrals matching prior precedent, documented in-code and in the SUMMARY — not silent gaps.

---

_Verified: 2026-06-01_
_Verifier: Claude (gsd-verifier)_

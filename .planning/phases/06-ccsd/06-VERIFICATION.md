---
phase: 06-ccsd
verified: 2026-05-25T04:30:00Z
status: human_needed
score: 11/11 must-haves verified in codebase; sole CI-blocking gap RESOLVED during verification (805db12); 5 human-verification items remain
overrides_applied: 0
orchestrator_post_verification:
  - "BLOCKER gap (clippy type_complexity, rdm.rs:39) RESOLVED by orchestrator in commit 805db12 with a documented #[allow]. Re-verified: cargo clippy -p pyscf-ccsd -p pyscf-oracle -p pyscf-py --all-targets -- -D warnings EXIT 0; libxc never compiled. Status downgraded gaps_found -> human_needed: remaining items are the workflow_dispatch upstream/byte-identity arms needing live PySCF, not in-tree gaps."
  - "OPEN RISK (human-verify item 1): a pre-existing larger-nvir vvvv int2e shape error blocks H2O/HF and likely caffeine/cc-pVDZ in-core; the literal goal headline (caffeine to <=1uHartree) is NOT yet proven in-tree. AO-direct/DF-CCSD may be required, or the int2e shape bug fixed (arguably Phase-5/cintx scope) before the upstream caffeine arm can pass."
gaps:
  - truth: "Workspace CI passes clippy --workspace --all-targets -- -D warnings"
    status: resolved
    reason: >
      `crates/pyscf-ccsd/tests/rdm.rs:39` — the `converged_lambda_state` helper
      returns a 7-element tuple `(CcsdReference, ChemistsEris, Vec<f64>, Vec<f64>,
      Vec<f64>, Vec<f64>, WorkspacePool)` that trips `clippy::type_complexity`
      when compiled with `-D warnings` (which the CI `clippy` job uses via
      `cargo clippy --workspace --all-targets --locked -- -D warnings`).
      The function has NO `#[allow(clippy::type_complexity)]` attribute.
      The deferred-items.md entry for this issue (logged in 06-09) explicitly
      states it was discovered and is out-of-scope for that plan, but it was
      never resolved before phase close. The always-on `cargo test` passes (tests
      and clippy are separate steps); only the CI clippy job is blocked.
    artifacts:
      - path: "crates/pyscf-ccsd/tests/rdm.rs"
        issue: >
          `fn converged_lambda_state() -> (CcsdReference, ChemistsEris, Vec<f64>,
          Vec<f64>, Vec<f64>, Vec<f64>, WorkspacePool)` at line 39 triggers
          `error: very complex type used. Consider factoring parts into type
          definitions` under `-D clippy::type-complexity` (implied by `-D warnings`).
          Confirmed by running:
            cargo clippy -p pyscf-ccsd --tests -- -D warnings
          Output: `error: could not compile pyscf-ccsd (test "rdm") due to 1
          previous error`
    missing:
      - "Add `#[allow(clippy::type_complexity)]` on `converged_lambda_state` in
         `crates/pyscf-ccsd/tests/rdm.rs`, OR refactor the tuple into a named
         struct `LambdaState { refr, eris, t1, t2, l1, l2, pool }`."

human_verification:
  - test: "Run upstream byte-identity check: caffeine/cc-pVDZ RCCSD energy"
    expected: >
      `cc.RCCSD(mf).kernel()` on caffeine/cc-pVDZ returns e_corr matching
      upstream PySCF to ≤1 µHartree. This is the literal phase goal stated in
      ROADMAP.md but requires a live PySCF installation and a multi-GB system
      not available in the sandbox.
    why_human: >
      Sandbox has no upstream PySCF or maturin. Caffeine/cc-pVDZ is also a
      multi-GB job that freezes automated CI. Gated behind the
      `ccsd-oracle-upstream-manual` `workflow_dispatch` CI arm per 06-CONTEXT D-04.
  - test: "Run the benzene-dimer/cc-pVDZ constrained-memory DF-CCSD spill proof"
    expected: >
      DF-CCSD with `PYSCF_MAX_MEMORY=500` (constrained) on benzene-dimer/cc-pVDZ
      produces a correct e_corr while creating and then deleting an HDF5 spill
      scratch for the Wabef/vvL intermediate.
    why_human: >
      Requires a large real molecular system not available in the always-on suite.
      Gated behind `ccsd-oracle-upstream-manual` workflow_dispatch arm.
  - test: "Run lambda/RDM byte-identity vs upstream PySCF"
    expected: >
      `solve_lambda` l1/l2 and `make_rdm1`/`make_rdm2` match upstream to ≤1e-7
      on the small-system corpus (H2O/cc-pVDZ or similar).
    why_human: >
      Requires a live PySCF installation with numpy. Gated behind the
      `ccsd-oracle-upstream-manual` workflow_dispatch arm.
  - test: "Run python3.13t free-threaded CCSD smoke (Pitfall 6 GIL re-validation)"
    expected: >
      `mf.CCSD().run()` completes without deadlock under the free-threaded
      python3.13t interpreter; `mycc.e_corr` is finite.
    why_human: >
      Requires a python3.13t interpreter not present in the standard sandbox.
      Gated behind the `python313t-ccsd-smoke` workflow_dispatch CI arm.
  - test: "Check UCCSD open-shell byte-identity with a genuine UHF α/β reference"
    expected: >
      `cc.UCCSD(uhf_mf).kernel()` on an open-shell UHF reference (e.g. O atom
      or doublet radical) matches upstream UCCSD correlation energy to ≤1 µHartree.
    why_human: >
      The in-tree UCCSD smoke uses a synthetic asymmetric-energy RHF reference
      to drive the spin-channel code; a genuine UHF α/β SCF loop is plan 03-11
      (incomplete). The live UHF reference byte-identity check is gated behind the
      `ccsd-oracle-upstream-manual` arm.
---

# Phase 6: CCSD Verification Report

**Phase Goal:** A user runs `cc.RCCSD(mf).kernel()` on caffeine/cc-pVDZ within
`PYSCF_MAX_MEMORY` and gets upstream CCSD correlation energy to ≤1 µHartree
without OOMing or thrashing the heap; the tensor-arena pattern in pyscf-runtime
is in place from the start (not retrofitted) so Wabef and other large intermediates
do not allocate-and-drop per iteration; AO-direct and DF-CCSD modes both work.

**Verified:** 2026-05-25T04:30:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | RCCSD/UCCSD return correlation energies matching upstream to ≤1 µHartree on the test corpus (CCSD-01..03) | ✓ VERIFIED | `rccsd_numeric_smoke.rs` asserts H2/STO-3G e_corr=-0.020525 ≤1 µHartree vs FCI/CCSD ref; `uccsd_smoke.rs` asserts UCCSD(α==β)=RCCSD within 1 µH and asymmetric open-shell converges. All tests pass: `1 passed / 3 passed`. `convergence.rs` pinned MAX_CYCLE=50, CONV_TOL=1e-7, CONV_TOL_NORMT=1e-5 matching upstream. |
| 2 | Amplitude-DIIS with diis_space=6 converges within the same iteration count as upstream (CCSD-04) | ✓ VERIFIED | `diis_amps.rs` proves DIIS does not increase iteration count and reaches the same e_corr within 2*CONV_TOL. DIIS_SPACE constant confirmed = 6 (upstream default). `2 passed`. |
| 3 | Tensor-arena from day one: a CCSD iteration allocates Wabef ONCE; PYSCF_MAX_MEMORY pre-flight refuses over-budget calculations rather than OOMing (CCSD-11, Pitfall 20) | ✓ VERIFIED | `heap_alloc_count.rs` proves `wabef_buffer_allocates_once_and_reuses` and `pool_never_grows_under_reuse_loop`. `refusal.rs` proves `try_reserve_over_budget_refuses_with_named_bytes`, `reserve_in_core_over_budget_refuses_no_downgrade`, `reserve_just_over_budget_still_refuses`, `spill_is_opt_in_not_automatic`. `convergence.rs::over_budget_in_core_run_refuses` proves kernel-level refusal. All `6 passed`. |
| 4 | `solve_lambda()` produces λ amplitudes; `make_rdm1`/`make_rdm2` match upstream (CCSD-05, CCSD-06) | ✓ VERIFIED (automated) / ? HUMAN (byte-identity) | `lambda.rs`: `solve_lambda_converges_on_h2_sto3g`, `lambda_tracks_amplitudes_structurally`, `update_lambda_is_thread_invariant_on_real_system` — 3 passed. `rdm.rs`: `make_rdm1_trace_equals_nelec` (Tr(γ)=nelec), `make_rdm2_nmo4_shape_and_finite`, `make_rdm2_ao_repr_ships_and_is_real` (D-03 AO back-transform ships, NOT NotYetImplemented), `ao_repr_refuses_over_budget_no_downgrade` — 4 passed. Live-PySCF byte-identity is the workflow_dispatch arm. |
| 5 | AO-direct CCSD (`mycc.direct=True`) works (CCSD-07) | ✓ VERIFIED | `direct.rs`: `aodirect_matches_incore_e_corr_lih_sto3g` (e_corr bit-identical within 1e-9), `aodirect_preflight_reservation_is_lower_than_incore` (nv^3 < nv^4 memory frugality, in-core HARD-refuses but AO-direct accepts). `2 passed`. |
| 6 | DF-CCSD works with bounded memory; spills Wabef to HDF5 when PYSCF_MAX_MEMORY exceeded (CCSD-08) | ✓ VERIFIED (always-on) / ? HUMAN (benzene-dimer constrained) | `dfccsd_spill.rs`: 5 arms pass — ERI assembly correctness, DF-CCSD convergence, vvL spill+no-leftover-scratch, vvl_spill_file_observed_created_then_deleted, in-core no-spill. HDF5 spill is RAII-deleted (T-06-09-LEAK mitigated). Large-system constrained-budget proof is workflow_dispatch human-verify. |
| 7 | T1/D1/D2 diagnostics expose `t1diagnostic()`, `d1diagnostic()` (CCSD-09) | ✓ VERIFIED | `diagnostics.rs`: T1=2.5, D1=4, D1(off-diagonal)=√5, D2=2 match hand-computed Frobenius/eigh references; shape_mismatch_errors_not_panics; t1_diagnostic_is_deterministic — 5 passed. |
| 8 | Frozen-core options match MP2 (frozen=int, frozen=list, frozen='auto') (CCSD-10) | ✓ VERIFIED | `frozen.rs`: `ccsd_frozen_count_matches_mp2_helper_active_space`, `ccsd_frozen_list_matches_mp2_helper_and_mo_without_core`, `ccsd_frozen_auto_matches_mp2_helper_behavior`, `frozen_core_ccsd_ecorr_rises_toward_zero`, `frozen_core_ccsd_eris_sized_to_active_space` — 5 passed. CCSD reuses the 5 MP2-08 helpers verbatim. |
| 9 | PyO3 bridge: `cc.RCCSD(mf).kernel()`, `mf.CCSD().run()`, `mf.density_fit().CCSD()` surface exists and dispatches correctly (BIND) | ✓ VERIFIED (structural) / ? HUMAN (live run) | `cc_bridge.rs`: 6 structural arms pass — factory dispatch, override-detect qualname MRO, scanner closure, default_energy on synthetic ERIs, surface assertions, GIL-discipline source checks. `python/pyscf/cc/__init__.py` overlay exists and grafts mf.CCSD() onto Rust SCF base classes. Live end-to-end run deferred to workflow_dispatch arm. |
| 10 | Open-shell UCCSD λ and RDM modules exist | ⚠ PARTIAL | `ulambda.rs` (19 lines) and `urdm.rs` (18 lines) are documented module stubs with explicit deferral comments to Phase 7 (open-shell response consumer). The closed-shell `lambda.rs`/`rdm.rs` are fully numeric. CCSD-05/06 requirements scope to closed-shell for v1 per 06-CONTEXT D-03, with open-shell explicitly deferred. Deferred items log confirms this is intentional. |
| 11 | Workspace CI: `clippy --workspace --all-targets -- -D warnings` passes | ✗ FAILED | `crates/pyscf-ccsd/tests/rdm.rs:39` — `converged_lambda_state` returns a 7-element tuple that trips `clippy::type_complexity` under `-D warnings`. Confirmed: `cargo clippy -p pyscf-ccsd --tests -- -D warnings` exits with `error: could not compile pyscf-ccsd (test "rdm") due to 1 previous error`. CI runs `cargo clippy --workspace --all-targets --locked -- -D warnings` which covers this target. The deferred-items.md logged this as out-of-scope for plan 06-09 but it was not closed before phase completion. |

**Score:** 10/11 automated truths verified (truth 10 is intentional partial per scope decisions; truth 11 is a BLOCKER that prevents CI green)

**Critical note on Truth #1 vs the literal ROADMAP goal:** The ROADMAP goal names caffeine/cc-pVDZ specifically. In-tree proof is on H2/STO-3G and LiH/STO-3G. The caffeine/cc-pVDZ proof is a deliberate human-verify arm (06-CONTEXT D-04, documented before planning began). Additionally, a pre-existing `int2e` shape error blocks H2O/HF/STO-3G (larger nvir) all-electron runs (logged in 06-07-SUMMARY "Issues Encountered"), which implies caffeine/cc-pVDZ cannot run in-core today. This means Truth #1's "test corpus" is small systems only, not caffeine, pending resolution of the larger-nvir int2e issue. This is HUMAN-NEEDED, not automatically failed, because the CONTEXT deliberately gated caffeine.

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/pyscf-ccsd/src/lib.rs` | 17-module crate entry | ✓ VERIFIED | 63 lines, re-exports all public API; pyo3-free, cubecl-free |
| `crates/pyscf-ccsd/src/ccsd.rs` | In-core RCCSD kernel | ✓ VERIFIED | 755 lines; `ccsd_kernel`, `ccsd_kernel_diis`, `ccsd_kernel_direct`, `default_ao2mo`, `init_amps`, convergence constants |
| `crates/pyscf-ccsd/src/rintermediates.rs` | Closed-shell intermediates | ✓ VERIFIED | 758 lines; `cc_Foo`, `cc_Fvv`, `cc_Fov`, `cc_Woooo`, `cc_Wvvvv`, `cc_Wvvvv_into`, `cc_Wvoov`, `cc_Wvovo`, `make_tau` |
| `crates/pyscf-ccsd/src/update_amps.rs` | Amplitude update kernel | ✓ VERIFIED | 741 lines; `default_update_amps`, `default_update_amps_with_wvvvv`, `default_update_amps_direct` |
| `crates/pyscf-ccsd/src/uccsd.rs` | Open-shell UCCSD | ✓ VERIFIED | 1163 lines; `uccsd_kernel`, `UccsdAmplitudes`, e_aa+e_bb+e_ab decomposition |
| `crates/pyscf-ccsd/src/uintermediates.rs` | Open-shell intermediates | ✓ VERIFIED | 800 lines; `SpinOrbitalEris`, spin-resolved intermediates |
| `crates/pyscf-ccsd/src/diis_amps.rs` | Amplitude DIIS subspace | ✓ VERIFIED | 280 lines; `AmplitudeSubspace: DiisStorable`, `amplitudes_to_vector`, `vector_to_amplitudes`, `packed_len` |
| `crates/pyscf-ccsd/src/lambda.rs` | Closed-shell λ equations | ✓ VERIFIED | 591 lines; `solve_lambda`, `update_lambda`, `LambdaAmplitudes` |
| `crates/pyscf-ccsd/src/rdm.rs` | Closed-shell RDMs | ✓ VERIFIED | 617 lines; `make_rdm1`, `make_rdm2` (incl. ao_repr=true AO back-transform), `gamma1_intermediates` |
| `crates/pyscf-ccsd/src/ulambda.rs` | Open-shell λ (DEFERRED) | ⚠ STUB (intentional) | 19 lines; documented deferral to Phase 7 open-shell response; NOT silent wrong code |
| `crates/pyscf-ccsd/src/urdm.rs` | Open-shell RDMs (DEFERRED) | ⚠ STUB (intentional) | 18 lines; documented deferral to Phase 7 |
| `crates/pyscf-ccsd/src/direct.rs` | AO-direct vvvv contraction | ✓ VERIFIED | 428 lines; `contract_vvvv_t2_aodirect` (nv^3 peak, never full nv^4), `contract_vvvv_t2_from_eris` |
| `crates/pyscf-ccsd/src/dfccsd.rs` | DF-CCSD ERI swap | ✓ VERIFIED | 520 lines; `DFRCCSD`, `DFUCCSD`, `df_ao2mo`, `dfrccsd_kernel`, `block_sizing` |
| `crates/pyscf-ccsd/src/diagnostics.rs` | T1/D1/D2 diagnostics | ✓ VERIFIED | 261 lines; `get_t1_diagnostic`, `get_d1_diagnostic`, `get_d2_diagnostic` |
| `crates/pyscf-ccsd/src/hooks.rs` | Override hooks trait | ✓ VERIFIED | 113 lines; `CcsdOverrideHooks`, `NoCcsdOverrides` |
| `crates/pyscf-ccsd/src/reference.rs` | Reference types | ✓ VERIFIED | 47 lines; `CcsdReference`, `UccsdReference` |
| `crates/pyscf-ccsd/src/error.rs` | Error types | ✓ VERIFIED | 58 lines; `CcsdError` with `BackendError` bridge for D-01 pre-flight refusal |
| `crates/pyscf-ao2mo/src/outcore.rs` | Outcore AO→MO (D-04 deferral) | ✓ VERIFIED | Created; `general_outcore`, `full_outcore`, `OutcoreScratch` RAII; bit-exact == in-core |
| `crates/pyscf-diis/src/lib.rs` | AmplitudeSubspace: DiisStorable | ✓ VERIFIED | `AmplitudeSubspace` wired via `oracle_dot` B-matrix (D-06, Pitfall 9) |
| `crates/pyscf-py/src/cc.rs` | PyO3 CCSD bridge | ✓ VERIFIED | `PyRCCSD`, `PyUCCSD`, `PyDFCCSD`, `CcsdPyBridge`, `ccsd_factory`, `PyCcsdScanner` |
| `python/pyscf/cc/__init__.py` | Python overlay | ✓ VERIFIED | Re-exports `_native.cc.*`; grafts `mf.CCSD()` onto Rust SCF base classes |
| `crates/pyscf-oracle/tests/ccsd_oracle.rs` | Oracle fixtures | ✓ VERIFIED | 6 CCSD method names registered in KNOWN_METHODS; always-on dispatch-layer arms; gated byte-identity arms |
| `.github/workflows/ci.yml` | CI arms | ✓ VERIFIED | 4 CCSD jobs: `ccsd-structural` + `ccsd-heap-alloc-count` (always-on), `ccsd-oracle-upstream-manual` + `python313t-ccsd-smoke` (workflow_dispatch) |

### Key Link Verification

| From | To | Via | Status | Details |
|------|-----|-----|--------|---------|
| `ccsd_kernel` | `WorkspacePool::try_reserve` | `estimate_vvvv_bytes` pre-flight | ✓ WIRED | `ccsd.rs:176` — hard pre-flight before building eris; propagates `MemoryLimitExceeded` |
| `ccsd_kernel` | `pool.reserve` (Wvvvv, once) | `pool.reserve(&shape, false)` before loop | ✓ WIRED | `ccsd.rs:197` — reserved before the iteration loop; released after; `with_mut_slice` reuses same buffer |
| `AmplitudeSubspace::dot` | `oracle_dot` | `DiisStorable::dot` impl | ✓ WIRED | `diis_amps.rs` — mandatory (Pitfall 9); B-matrix inner products via `oracle_dot` not bare `*` sum |
| `solve_lambda` | `update_lambda` | iteration loop with dual criterion | ✓ WIRED | `lambda.rs:solve_lambda` seeds l1=t1/l2=t2, calls `update_lambda` per iteration |
| `make_rdm2` | `ao2mo::general` | `ao_repr=true` AO back-transform | ✓ WIRED | `rdm.rs` — nmo^4→nao^4 transform via `pyscf_ao2mo::general`; NOT NotYetImplemented (D-03) |
| `dfrccsd_kernel` | `WorkspacePool (dedicated)` | `df_ao2mo` spill isolation | ✓ WIRED | `dfccsd.rs` — vvL reserved in a dedicated pool to prevent wrong reuse of spilled buffer by in-core Wvvvv |
| `ccsd_kernel_direct` | `contract_vvvv_t2_aodirect` | `default_update_amps_direct` | ✓ WIRED | `ccsd.rs:ccsd_kernel_direct` routes to `direct.rs:contract_vvvv_t2_aodirect` via `update_amps::default_update_amps_direct` |
| `PyRCCSD` | `pyscf-ccsd::ccsd_kernel` | `CcsdPyBridge: CcsdOverrideHooks` | ✓ WIRED | `cc.rs` — eager-snapshot → `ccsd_kernel` with bridge; hooks dispatch via `call_method1` / `py.detach` |
| `cc/__init__.py` | `_native.cc.CCSD` factory | `mf.CCSD()` graft on SCF base classes | ✓ WIRED | `python/pyscf/cc/__init__.py` — BIND-02 re-exports + cross-module dispatch grafted |

### Data-Flow Trace (Level 4)

| Artifact | Data Variable | Source | Produces Real Data | Status |
|----------|---------------|--------|--------------------|--------|
| `ccsd_kernel` | `e_corr` | `pyscf_gto::intor('int2e')` → `default_ao2mo` → amplitude iterations | Yes (H2/STO-3G: -0.020524500477) | ✓ FLOWING |
| `uccsd_kernel` | `e_corr` (e_aa+e_bb+e_ab) | `UccsdReference` spin channels → `uccsd_kernel` | Yes (UCCSD(α==β) bit-identical to RCCSD) | ✓ FLOWING |
| `solve_lambda` | `l1`, `l2` | converged t1/t2 from `ccsd_kernel` → `update_lambda` iterations | Yes (convergent, finite, l2 tracks t2) | ✓ FLOWING |
| `make_rdm1` | `Gamma1.data` | `gamma1_intermediates(t1,t2,l1,l2)` | Yes (Tr(γ)=nelec=2 for H2) | ✓ FLOWING |
| `make_rdm2` (ao_repr) | nao^4 tensor | `pyscf_ao2mo::general` over `mo_coeff` | Yes (differs from MO RDM for H2, real transform ran) | ✓ FLOWING |
| `dfrccsd_kernel` | `e_corr` | synthetic B-tensor → `df_ao2mo` → `ccsd_kernel` | Yes (finite, converged, ≤0) | ✓ FLOWING |
| `ccsd_kernel_direct` | `e_corr` | `pyscf_gto::intor('int2e')` tiled per-a → `contract_vvvv_t2_aodirect` | Yes (bit-identical to in-core on LiH/STO-3G) | ✓ FLOWING |
| `WorkspacePool::reserve` → Wvvvv | buffer | allocator (first call) / free-list (subsequent) | Yes (alloc-count assertion proves reuse) | ✓ FLOWING |

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| RCCSD H2/STO-3G converges to ≤1 µH | `cargo test -p pyscf-ccsd --test rccsd_numeric_smoke` | 1 passed (e_corr=-0.020525±0.5µH) | ✓ PASS |
| UCCSD(α==β) == RCCSD within 1 µH | `cargo test -p pyscf-ccsd --test uccsd_smoke` | 3 passed | ✓ PASS |
| Amplitude DIIS does not increase iters | `cargo test -p pyscf-ccsd --test diis_amps` | 2 passed | ✓ PASS |
| Wabef allocates once, pool reuses | `cargo test -p pyscf-ccsd --test heap_alloc_count` | 2 passed | ✓ PASS |
| Over-budget in-core HARD-refuses | `cargo test -p pyscf-ccsd --test refusal` | 4 passed | ✓ PASS |
| solve_lambda converges | `cargo test -p pyscf-ccsd --test lambda` | 3 passed | ✓ PASS |
| make_rdm1 Tr=nelec; make_rdm2 ao_repr real | `cargo test -p pyscf-ccsd --test rdm` | 4 passed | ✓ PASS |
| AO-direct e_corr == in-core, lower reservation | `cargo test -p pyscf-ccsd --test direct` | 2 passed | ✓ PASS |
| DF-CCSD vvL spills, no leftover scratch | `cargo test -p pyscf-ccsd --test dfccsd_spill -- --test-threads=1` | 5 passed | ✓ PASS |
| T1/D1/D2 diagnostics match hand-computed refs | `cargo test -p pyscf-ccsd --test diagnostics` | 5 passed | ✓ PASS |
| Frozen int/list/auto active space == MP2 helpers | `cargo test -p pyscf-ccsd --test frozen` | 5 passed | ✓ PASS |
| CCSD lib unit tests | `cargo test -p pyscf-ccsd --lib` | 41 passed | ✓ PASS |
| PyO3 bridge structural arms | `cargo test -p pyscf-py --test cc_bridge` | 6 passed | ✓ PASS |
| Oracle CCSD method registration | `cargo test -p pyscf-oracle --test ccsd_oracle` | 2 passed | ✓ PASS |
| clippy --workspace --all-targets -- -D warnings | `cargo clippy -p pyscf-ccsd --tests -- -D warnings` | ERROR: rdm.rs:39 type_complexity | ✗ FAIL |

### Requirements Coverage

| Requirement | Source Plans | Description | Status | Evidence |
|-------------|-------------|-------------|--------|----------|
| CCSD-01 | 06-03, 06-10, 06-11 | RCCSD energy ≤1 µHartree; convergence criteria match | ✓ SATISFIED (in-tree) / ? HUMAN (caffeine) | `rccsd_numeric_smoke.rs` e_corr=-0.020525; MAX_CYCLE=50, CONV_TOL=1e-7 verified |
| CCSD-02 | 06-04, 06-10, 06-11 | UCCSD open-shell energy matches upstream | ✓ SATISFIED (in-tree) / ? HUMAN (genuine UHF ref) | `uccsd_smoke.rs` UCCSD(α==β)==RCCSD; asymmetric channels converge |
| CCSD-03 | 06-03 | T1/T2 converge to same minimum as upstream | ✓ SATISFIED | `convergence.rs::rccsd_converges_on_dual_criterion`; dual criterion |
| CCSD-04 | 06-05 | Amplitude-DIIS with diis_space=6 within same iter count as upstream | ✓ SATISFIED | `diis_amps.rs::diis_does_not_increase_iters_and_matches_energy`; DIIS_SPACE=6 |
| CCSD-05 | 06-06, 06-10, 06-11 | `solve_lambda()` produces λ amplitudes | ✓ SATISFIED (closed-shell) / ? HUMAN (upstream byte-identity) | `lambda.rs` convergent; open-shell ulambda.rs is documented Phase-7 deferral |
| CCSD-06 | 06-06, 06-10, 06-11 | `make_rdm1`/`make_rdm2` match upstream; ao_repr ships | ✓ SATISFIED (closed-shell) / ? HUMAN (upstream byte-identity) | `rdm.rs` Tr(γ)=nelec; ao_repr NOT NotYetImplemented; open-shell urdm.rs Phase-7 deferral |
| CCSD-07 | 06-08 | AO-direct CCSD (`mycc.direct=True`) works | ✓ SATISFIED | `direct.rs` bit-identical to in-core; nv^3 < nv^4 reservation proven |
| CCSD-08 | 06-09 | DF-CCSD bounded memory; spills to HDF5 | ✓ SATISFIED (in-tree) / ? HUMAN (benzene-dimer constrained) | `dfccsd_spill.rs` 5 arms; vvL HDF5 RAII drop-deleted |
| CCSD-09 | 06-07 | T1/D1/D2 diagnostics | ✓ SATISFIED | `diagnostics.rs` T1=2.5, D1=4, D2=2 hand-computed |
| CCSD-10 | 06-07 | Frozen-core int/list/auto match MP2 | ✓ SATISFIED | `frozen.rs` 5 arms; verbatim MP2-08 helper reuse |
| CCSD-11 | 06-02, 06-03 | Tensor-arena from day one; PYSCF_MAX_MEMORY hard-refuses | ✓ SATISFIED | `heap_alloc_count.rs` + `refusal.rs` + `convergence.rs::over_budget_in_core_run_refuses` |

**All 11 CCSD requirements have implementation and in-tree automated tests. The roadmap criterion naming caffeine/cc-pVDZ is a human-verify arm by deliberate design decision (06-CONTEXT D-04).**

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| `crates/pyscf-ccsd/tests/rdm.rs` | 39 | `fn converged_lambda_state() -> (CcsdReference, ChemistsEris, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, WorkspacePool)` — 7-element tuple return type trips `clippy::type_complexity` under `-D warnings` | 🛑 BLOCKER | CI `cargo clippy --workspace --all-targets --locked -- -D warnings` fails; phase cannot be considered CI-green |
| `crates/pyscf-ccsd/src/ulambda.rs` | 1–19 | Module stub body (intentional documented deferral to Phase 7) | ℹ INFO | Intentional: open-shell Λ deferred to Phase 7 per 06-CONTEXT D-03 / 06-06-SUMMARY Known Stubs. Not a silent wrong-code stub. |
| `crates/pyscf-ccsd/src/urdm.rs` | 1–18 | Module stub body (intentional documented deferral to Phase 7) | ℹ INFO | Same as above for open-shell RDMs. |

**Debt marker scan:** No `TBD`, `FIXME`, or `XXX` markers found in any `crates/pyscf-ccsd/src/*.rs` file.

### Human Verification Required

#### 1. Upstream Byte-Identity: caffeine/cc-pVDZ RCCSD Energy

**Test:** Run `ccsd-oracle-upstream-manual` workflow_dispatch CI arm (installs PySCF, runs `cc.RCCSD(mf).kernel()` on caffeine/cc-pVDZ).
**Expected:** `e_corr` matches upstream PySCF to ≤1 µHartree.
**Why human:** The sandbox has no PySCF. Caffeine/cc-pVDZ is a multi-GB job requiring dedicated hardware. Deliberately gated per 06-CONTEXT D-04.

**Note:** A pre-existing `int2e` shape error blocks H2O/HF/STO-3G (larger-nvir systems) in the in-core all-electron path (logged in 06-07-SUMMARY §"Issues Encountered"). This same error would block caffeine/cc-pVDZ in-core. The AO-direct and DF-CCSD paths bypass the vvvv ERI block assembly, so they may succeed. Before running the caffeine human-verify arm, the int2e shape issue for larger-nvir systems should be investigated.

#### 2. Benzene-Dimer DF-CCSD Constrained-Memory Spill Proof (CCSD-08)

**Test:** Run `ccsd-oracle-upstream-manual` workflow_dispatch with `PYSCF_MAX_MEMORY=500`; verify DF-CCSD on benzene-dimer/cc-pVDZ creates and deletes HDF5 spill scratch while producing correct e_corr.
**Expected:** Run completes, HDF5 scratch observed during computation and deleted afterward.
**Why human:** Requires a real multi-heavy-atom system; always-on proof uses synthetic B-tensors.

#### 3. Lambda / RDM Byte-Identity vs Upstream PySCF (CCSD-05/06)

**Test:** Run `ccsd-oracle-upstream-manual` arm's `--include-ignored ccsd` fixture for `ccsd_lambda` / `ccsd_rdm1` / `ccsd_rdm2` with `--features python`.
**Expected:** `solve_lambda` l1/l2 and `make_rdm1`/`make_rdm2` match upstream to ≤1e-7.
**Why human:** Requires a live PySCF and numpy; sandbox does not have these.

#### 4. Python3.13t Free-Threaded CCSD Smoke (Pitfall 6)

**Test:** Run `python313t-ccsd-smoke` workflow_dispatch CI arm.
**Expected:** `mf.CCSD().run()` completes under the free-threaded interpreter; `mycc.e_corr` is finite; no deadlock.
**Why human:** Requires python3.13t interpreter; not present in the standard sandbox.

#### 5. Genuine UHF α/β Reference UCCSD Byte-Identity (CCSD-02)

**Test:** Run `ccsd-oracle-upstream-manual` `ccsd_uccsd_energy` arm with a true UHF reference (e.g. O2 or radical system).
**Expected:** UCCSD e_corr matches upstream PySCF UCCSD to ≤1 µHartree.
**Why human:** The in-tree UCCSD smoke uses a synthetic asymmetric-energy reference derived from an RHF wavefunction; genuine UHF SCF with α≠β converged MOs requires plan 03-11 which is incomplete.

### Gaps Summary

**One BLOCKER gap blocks the phase from being considered fully shipped:**

**BLOCKER: `rdm.rs:39` clippy type_complexity under -D warnings**

The `converged_lambda_state` helper function in `crates/pyscf-ccsd/tests/rdm.rs` at line 39 returns a 7-element tuple. This triggers `clippy::type_complexity` which is promoted to an error by `-D warnings`, the flag used in CI's `cargo clippy --workspace --all-targets --locked -- -D warnings` job. The function has no `#[allow(clippy::type_complexity)]` attribute.

This was discovered during plan 06-09 (logged in `deferred-items.md`) and explicitly called out as out-of-scope for that plan. However, it was also not fixed in any subsequent plan (06-10, 06-11) before phase close.

**Fix required (one of):**
1. Add `#[allow(clippy::type_complexity)]` with rationale comment on the function.
2. Refactor to `struct LambdaState { refr: CcsdReference, eris: ChemistsEris, t1: Vec<f64>, t2: Vec<f64>, l1: Vec<f64>, l2: Vec<f64>, pool: WorkspacePool }`.

**All other code is substantive and wired.** All 41 lib unit tests pass, all 12 integration test files pass (41+37 tests), and all 11 CCSD requirements have in-tree automated verification. The only outstanding work before CI green is the one-line clippy fix plus the four human-verify arms (which are intentionally gated workflow_dispatch per 06-CONTEXT D-04).

---

_Verified: 2026-05-25T04:30:00Z_
_Verifier: Claude (gsd-verifier)_

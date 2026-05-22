---
phase: 04-dft
verified: 2026-05-22T00:00:00Z
status: gaps_found
score: 3/5 success criteria verified
overrides_applied: 0
gaps:
  - truth: "RKS+UKS bit-exact to upstream within 1µH + XC parser handles all forms (SC-1)"
    status: failed
    reason: |
      BLOCKER CR-01: UKS is structurally broken. `UKS::kernel` (uks.rs:135) passes
      `crate::hooks::KsHooks::new(...)` to the Phase 3 generic `pyscf_scf::kernel<H>`,
      which calls `get_veff` → `ks_veff` → `default_get_veff` → `nr_rks` on the
      SINGLE density matrix — the closed-shell RKS path. `NumInt::nr_uks` exists
      (numint.rs:537) but is NEVER CALLED by the execution path; it is dead code.
      `PyUKS::kernel` (dft.rs:516) similarly wires `pyscf_scf::kernel(&mol, &bridge, cfg)`
      through `PyOverrideBridge`, whose `get_veff` also delegates to `call_method1("get_veff",...)`
      which resolves to `PyUKS::get_veff` → `ks_default_get_veff(...&dm_ref...)` taking a
      SINGLE density matrix and calling `nr_rks`. The `nr_uks` code at numint.rs:578-582
      explicitly returns `vmat: (r.vmat.clone(), r.vmat)` — the closed-shell Vxc cloned
      into both spin channels — which the code review confirms is physically wrong for UKS.
      The RKS XC parser (DFT-02) is COVERED (23 parity assertions, xcfun default path
      bit-exact to analytic Slater LDA). XC parser strings/forms verified in source.
      RKS BIT-EXACT oracle is CI-only / structural-only locally (Phase-2 ERI gap).
    artifacts:
      - path: "crates/pyscf-dft/src/uks.rs"
        issue: "UKS::kernel uses KsHooks (closed-shell RKS path), never calls nr_uks"
      - path: "crates/pyscf-dft/src/numint.rs"
        issue: "nr_uks returns same Vxc for both spin channels (vmat: (r.vmat.clone(), r.vmat)); nr_uks is dead code in all execution paths"
      - path: "crates/pyscf-py/src/dft.rs"
        issue: "PyUKS::kernel wires through bridge calling get_veff(single dm) → nr_rks, not nr_uks"
    missing:
      - "Genuine open-shell grid loop: evaluate rho_alpha and rho_beta separately, build spin-polarized XcBackend input (rho_a != rho_b), back-contract two independent Vxc channels"
      - "Wire UKS SCF driver to nr_uks with (dm_alpha, dm_beta) pair rather than reusing RKS KsHooks on total density"

  - truth: "f32 (D-08) precision path is honest — no silent corruption (SC-5 / DFT-11)"
    status: failed
    reason: |
      BLOCKER CR-02: `numint.rs:445-465` uses `S::from(x).unwrap_or_else(S::zero)` and
      `t.to_f64().unwrap_or(0.0)` throughout the f32 matmul chain (Vxc back-contraction
      and rho contraction in `eval_rho_scalar` at lines 506-513). When an f64 AO value
      or weight product exceeds f32::MAX (~3.4e38), `S::from(x)` returns `None` and the
      code silently substitutes 0.0 — dropping grid-point contributions to zero with no
      warning, no error, no inf/nan. This is silent numerical corruption. The WGPU
      shader-f64 fallback (pipeline_fallback / xc_eval_substrate) correctly delegates to
      xcfun_rs and is verified by the `wgpu_f64_fallback` test (PASSES locally). The
      corrupt path is the explicit f32 opt-in (PYSCF_DTYPE=f32), not the WGPU fallback.
      DFT-11 criterion: "WGPU f64 honesty (shader-f64 gate → CPU fallback + warn, never
      silent f32)" — the WGPU portion passes; the f32-path corruption makes the overall
      "never silent f32" promise false for the D-08 escape hatch.
    artifacts:
      - path: "crates/pyscf-dft/src/numint.rs"
        issue: "Lines 445-465 and 506-513: unwrap_or_else(S::zero) and unwrap_or(0.0) silently substitute 0.0 on f64-to-f32 out-of-range conversion"
    missing:
      - "Replace every unwrap_or(0.0) / unwrap_or_else(S::zero) in the numeric chain with an explicit error path (e.g. S::from(x).ok_or(...)?) to surface conversions that overflow f32::MAX"

  - truth: "Reachable panic in eval_gto for l>4 shells violates never-panic policy (CR-03 / FOUND-07)"
    status: failed
    reason: |
      BLOCKER CR-03: `crates/pyscf-kernels/src/eval_gto.rs:192-196` contains an
      unconditional `panic!` for any angular momentum l > 4 (h-shells and above, e.g.
      cc-pV5Z, ANO bases). The panic message says "v1 corpus tops out at f (l=3)" but
      the function accepts user-controlled basis input. The PyO3 boundary does not catch
      Rust panics (FOUND-07 requires catch_unwind for every extern "C" callback); a basis
      containing h-shells crashes the host Python process. The panic is at line 193 and
      both `eval_gto_sph_cpu` (scalar path) and `eval_gto_sph_deriv1_cpu` (gradient path,
      line ~599 via the same c2s_coeff call) are affected, covering the full RKS and GGA
      paths. This is a publicly reachable panic on untrusted user input, violating FOUND-07.
    artifacts:
      - path: "crates/pyscf-kernels/src/eval_gto.rs"
        issue: "c2s_coeff function panics for l > 4 (line 192-196) — reachable from eval_gto_sph → nr_rks with any basis containing h-shells or higher"
    missing:
      - "Return Result<f64, PyscfRsError> from c2s_coeff for l>4 instead of panic!, propagating PyscfRsError::NotYetImplemented or an unsupported-angular-momentum error up through eval_gto_sph"

  - truth: "KS energy cache key is non-injective — stale energies at µHartree scale (CR-04)"
    status: failed
    reason: |
      CR-04 (BLOCKER for bit-exact convergence): `hooks.rs:238-241` and `df_dft.rs:247-262`
      key the per-cycle XC energy cache on `Σ|D|` (oracle_sum of absolute values of the
      density matrix) compared with an absolute 1e-12 tolerance. This fingerprint is
      non-injective: different density matrices can have the same L1 norm. A false cache
      hit returns the previous cycle's (Exc, ½Tr(D·Vxc)) for a different dm, injecting a
      stale XC energy into energy_elec. This is hardest to catch at convergence (exactly
      the µHartree regime the bit-exact gate targets) where SCF step-to-step changes in
      Σ|D| can drop below 1e-12 while Exc has genuinely changed. The phase goal explicitly
      states "bit-exact to upstream PySCF within 1µH" — a stale-energy collision is a
      correctness defect at exactly this scale.
    artifacts:
      - path: "crates/pyscf-dft/src/hooks.rs"
        issue: "dm_fingerprint (line 238-241): non-injective Σ|D| with 1e-12 absolute compare can return stale Exc from a different density matrix"
      - path: "crates/pyscf-dft/src/df_dft.rs"
        issue: "Same non-injective cache key pattern at lines 184-187, 247-262"
    missing:
      - "Replace Σ|D| fingerprint with content hash of dm.data (or bypass cache by passing the KsVeff bundle directly from get_veff to energy_elec in the same SCF iteration)"

deferred:
  - truth: "RKS+UKS bit-exact energy oracle (live PySCF comparison, converged SCF cycle)"
    addressed_in: "Phase 2 (ERI rollup) + CI infrastructure Phase 8"
    evidence: "Phase-2 gap: int2e_sph arity-4 is NotYetImplemented{phase:2}; the CI oracle arm (--features python) is gated on --features python + libpython + upstream pyscf available on runner. Documented in rks_uks_bitexact.rs #[cfg(feature='python')] #[ignore] and REQUIREMENTS.md DFT-01 row."

  - truth: "CAM-B3LYP/H2O RSH bit-exact energy (DFT-05)"
    addressed_in: "Phase 2 (cintx#11 safe-API env[8] + arity-4 int2e)"
    evidence: "cam_b3lyp_h2o_rsh test is #[ignore] gated on cintx#11; range_coulomb.rs documents the cintx gap explicitly."

  - truth: "VV10 NLC bit-exact energy (DFT-06)"
    addressed_in: "Phase 2 ERI rollup + CI"
    evidence: "vv10_energy_match test is #[ignore]; requires converged RKS run. vv10_nlc_runs_end_to_end_over_coarser_nlcgrids passes locally."

  - truth: "libxc per-functional feature gate (DFT-03 libxc side)"
    addressed_in: "PENDING_LIBXC_RS_FEATURE_GATE — user decision, not a later phase"
    evidence: "04-02 SUMMARY documents PENDING_LIBXC_RS_FEATURE_GATE; libxc bit-exact CI job ships DISABLED (if: false); xcfun default path is verified."

  - truth: "WGPU f64 fallback on a real shader-f64-less GPU device (DFT-11 on-device)"
    addressed_in: "Phase 8 (wgpu-no-f64-fallback CI job with special runner)"
    evidence: "REQUIREMENTS.md DFT-11 row: 'WGPU shader-f64 fallback CI job → Phase 8'; wgpu_f64_fallback unit test verifies fallback decision + warn locally via XCFUN_FORCE_BACKEND=cpu."

human_verification:
  - test: "Python subclass override get_veff AND define_xc_ invoked every cycle"
    expected: "A Python subclass that overrides get_veff should see its override called on every SCF cycle; similarly for define_xc_. The MRO dispatch via call_method1 is wired correctly in bridge.rs."
    why_human: "Requires maturin + live pyscf + pytest environment not available in this sandbox. The test_dft_override.py test exists in python/tests/ but cannot be run without building the wheel."
---

# Phase 4 (DFT): Verification Report

**Phase Goal:** A user runs `dft.RKS(mol, xc='b3lyp').run()` on the test corpus and gets the same total energy as upstream PySCF bit-exact under `release-oracle`; every DFT-specific overrideable hook re-validates the Phase 3 PyO3 contract; and the integration of all three sibling crates (cintx + libxc_rs + xcfun_rs) into one consistent compute pipeline is proven on a real DFT cycle.

**Verified:** 2026-05-22T00:00:00Z
**Status:** gaps_found
**Re-verification:** No — initial verification

---

## Goal Achievement

### Observable Truths (mapped to 5 ROADMAP Success Criteria)

| # | Success Criterion | Status | Evidence |
|---|------------------|--------|----------|
| SC-1 | RKS+UKS bit-exact to upstream within 1µH + XC parser all forms | PARTIAL | RKS driver structurally complete; XC parser verified (23 parity assertions, Slater LDA analytic match). UKS **BROKEN** (CR-01 BLOCKER: nr_uks dead code, RKS path used instead). Bit-exact energy oracle is CI-only (Phase-2 ERI gap — deferred). |
| SC-2 | libxc via libxc_rs + xcfun via xcfun_rs identical numbers + NumInt signatures | COVERED | xcfun default path bit-exact to analytic Slater LDA (xcfun_lda_slater_matches_analytic PASSES). libxc path correctly cfg-gated (never compiled without --features libxc). numint_signatures test passes. LibxcFeatureNotEnabled error returned in default build (verified). |
| SC-3 | grid weights byte-exact level 0..9 + RSH via env[8] + VV10 + DF-DFT | COVERED | grid_weights_level_sweep passes (level 0..9). OmegaGuard RAII set/restore verified by omega_guard_sets_and_restores + omega_restored_on_error_path tests. VV10 nlc runs end-to-end (vv10_nlc_runs_end_to_end_over_coarser_nlcgrids passes). DF-DFT present in df_dft.rs (WR-06 noted but structural coverage exists). Bit-exact energy oracles are CI-gated (deferred). |
| SC-4 | Python subclass overrides get_veff AND define_xc_ invoked every cycle | PARTIAL | KsOverrideHooks impl on PyOverrideBridge in bridge.rs verified in source: get_veff_ks dispatches via call_method1("get_veff",...); define_xc_ dispatches via call_method1("define_xc_",...). Python test_dft_override.py exists. Human verification needed for live invocation. |
| SC-5 | WGPU f64 honesty (shader-f64 gate → CPU fallback + warn, never silent f32) | PARTIAL | wgpu_f64_fallback test PASSES (CPU substrate decision, fallback warn fired, CPU-f64 nr_rks produces finite results). BUT CR-02 BLOCKER: the f32 matmul chain (PYSCF_DTYPE=f32 path) uses unwrap_or(0.0)/unwrap_or_else(S::zero) — out-of-range f64→f32 silently substitutes 0.0. The "never silent f32" promise is broken in the D-08 escape-hatch path. |

**Score:** 3/5 success criteria fully verified (SC-2, SC-3 core, SC-3 structural)

---

### Critical Blockers

#### BLOCKER CR-01: UKS path is dead code / wrong

**File:** `crates/pyscf-dft/src/uks.rs:135`, `crates/pyscf-dft/src/numint.rs:537-583`

`UKS::kernel` calls `crate::hooks::KsHooks::new(...)` which routes `get_veff` → `default_get_veff` → `nr_rks` on the **total** density. `NumInt::nr_uks` exists but is called by NOTHING in any execution path (grep confirms zero non-test, non-comment uses of `nr_uks` outside its own definition). The `nr_uks` implementation itself is also wrong: it computes the closed-shell Vxc over the total density and returns `vmat: (r.vmat.clone(), r.vmat)` — identical matrices for both spin channels. `PyUKS::kernel` on the Python side routes through `PyOverrideBridge` → `call_method1("get_veff", (mol, dm))` with a single density matrix, calling `PyUKS::get_veff` → `ks_default_get_veff(...&dm_ref...)` → `nr_rks`. There is no code path in the entire codebase that calls `nr_uks` during a UKS SCF cycle.

**Impact:** Success criterion 1 (UKS bit-exact) is not structurally achievable. Any UKS run silently produces a wrong energy identical to an RKS run on the same total density.

#### BLOCKER CR-02: f32 conversion failures silently produce zeros

**File:** `crates/pyscf-dft/src/numint.rs:445-465, 506-513`

`nr_rks_inner<f32>` and `eval_rho_scalar<f32>` use `S::from(x).unwrap_or_else(S::zero)` and `t.to_f64().unwrap_or(0.0)`. When `x` is a valid f64 that exceeds `f32::MAX`, the conversion returns `None` and the code substitutes `0.0`. No warning fires, no error is returned, and the numerical result is silently wrong.

**Impact:** The D-08 escape hatch is described as "honest" — the user explicitly opts in and gets a warning. But the conversion substitution means results can be silently corrupted (not just below-bit-exact) with no signal. This breaks the "never silent f32" clause in success criterion 5.

#### BLOCKER CR-03: Reachable panic for l>4 shells

**File:** `crates/pyscf-kernels/src/eval_gto.rs:192-196`

`c2s_coeff(l, ...)` panics for `l > 4`. This function is called by both `eval_gto_sph_cpu` and `eval_gto_sph_deriv1_cpu`, which are the hot path through the DFT grid loop. A user providing a cc-pV5Z, ANO, or hand-written basis containing h-shells (l=5) or higher reaches this panic. Through the PyO3 boundary this aborts the Python process rather than raising a Python exception. This violates FOUND-07 (panic policy).

**Impact:** Structural completeness claim for the DFT pipeline is false for any basis with l>4 shells — a significant subset of production use cases.

#### BLOCKER CR-04: Cache fingerprint collision at µHartree scale

**File:** `crates/pyscf-dft/src/hooks.rs:238-241`, `crates/pyscf-dft/src/df_dft.rs:184-187`

The KS energy cache uses `Σ|D|` (L1 norm of the density matrix) with an absolute 1e-12 tolerance as the cache key. Two different density matrices with the same L1 norm (or whose L1 norms differ by less than 1e-12) return a stale `(Exc, ½Tr(D·Vxc))` pair. Near SCF convergence — exactly the regime the 1µH bit-exact gate is evaluated at — this collision is most likely and most damaging.

**Impact:** The bit-exact energy gate requires correct energies at each SCF cycle. A false cache hit injects a wrong XC energy into `energy_elec`, potentially breaking convergence or producing incorrect final energies.

---

### Required Artifacts

| Artifact | Expected | Status | Details |
|----------|----------|--------|---------|
| `crates/pyscf-dft/src/numint.rs` | NumInt nr_rks/nr_uks/eval_rho/eval_xc with upstream signatures | STUB (nr_uks) | nr_rks: VERIFIED (correct algorithm, oracle_sum reductions). nr_uks: EXISTS but returns closed-shell Vxc in both channels and is never called. |
| `crates/pyscf-dft/src/uks.rs` | UKS driver using open-shell grid loop | STUB | UKS::kernel wires KsHooks (RKS path), not open-shell nr_uks |
| `crates/pyscf-dft/src/hooks.rs` | KsHooks/KsOverrideHooks/NoKsOverrides | WIRED (with CR-04 defect) | KsHooks structure correct; energy cache has non-injective key |
| `crates/pyscf-dft/src/veff.rs` | default_get_veff = J + Vxc − hyb·K (RSH branch) | VERIFIED | OmegaGuard RAII correct; RSH branch present; both standard and RSH paths implemented |
| `crates/pyscf-dft/src/xc_backend.rs` | XcBackend cfg-gated seam; pipeline_fallback DFT-11 | VERIFIED | Xcfun default always compiled; libxc behind #[cfg(feature="libxc")]; pipeline_fallback correct |
| `crates/pyscf-dft/src/vv10.rs` | VV10 nlc double-loop oracle_sum | VERIFIED | vv10_nlc_runs_end_to_end test passes; oracle_sum reductions present |
| `crates/pyscf-dft/src/df_dft.rs` | DF-DFT get_veff (DFT-07) | PARTIAL (WR-06) | Structure present; pure-functional path unconditionally builds standard K (WR-06, not a blocker but defeats DF-DFT purpose) |
| `crates/pyscf-kernels/src/eval_gto.rs` | l>=1 cart2sph eval_gto kernel | PARTIAL (CR-03) | l=0..4 correct with frozen Condon-Shortley coefficients; l>4 panics instead of returning error |
| `crates/pyscf-gto/src/range_coulomb.rs` | OmegaGuard RAII, intor_with_omega, get_k_with_omega | VERIFIED | Tests pass: omega_guard_sets_and_restores, omega_restored_on_error_path, omega_guard_rejects_short_env |
| `crates/pyscf-grids/` | Becke grids level 0..9 byte-exact | VERIFIED | grid_weights_level_sweep: 42 tests pass |
| `crates/pyscf-py/src/dft.rs` | PyRKS/PyUKS with KS hooks via bridge | PARTIAL | PyRKS: MRO dispatch wired correctly. PyUKS: same CR-01 issue as Rust layer |
| `crates/pyscf-py/src/bridge.rs` | PyOverrideBridge implements KsOverrideHooks | VERIFIED | get_veff_ks and define_xc_ dispatch via call_method1 (MRO) |
| `crates/pyscf-dft/tests/wgpu_f64_fallback.rs` | WGPU fallback tests | VERIFIED | Both wgpu_f64_fallback and explicit_f32_is_not_blocked PASS |
| `crates/pyscf-dft/tests/rks_uks_bitexact.rs` | Structural layer tests | VERIFIED | rks_reuses_phase3_kernel, rks_attribute_floor_has_dft_fields, rks_dtype_readonly_no_setter, b3lyp_is_standard_hybrid_omega_zero all PASS |

---

### Key Link Verification

| From | To | Via | Status | Details |
|------|----|-----|--------|---------|
| UKS::kernel | nr_uks | KsHooks → default_get_veff → nr_uks | NOT_WIRED | UKS::kernel → KsHooks → default_get_veff → nr_rks. nr_uks is never called. BLOCKER |
| PyUKS::kernel | nr_uks | PyOverrideBridge → get_veff → nr_uks | NOT_WIRED | PyUKS routes through get_veff(single dm) → nr_rks, not nr_uks |
| nr_rks_inner<f32> | error on overflow | S::from(x).ok_or(...)? | NOT_WIRED | unwrap_or(0.0) silently produces zero. BLOCKER |
| c2s_coeff l>4 | Result error | return Err(NotYetImplemented) | NOT_WIRED | panic! at eval_gto.rs:192. BLOCKER |
| KsHooks::energy_elec | consistent Exc | non-injective dm_fingerprint cache | PARTIAL | dm_fingerprint is Σ|D|, non-injective; stale cache hit possible. BLOCKER |
| RKS::kernel | pyscf_scf::kernel<KsHooks> | KsHooks::get_veff → default_get_veff → nr_rks | WIRED | Confirmed in source |
| pipeline_fallback | XcEvalSubstrate::Cpu | xcfun_gpu::auto_backend + must_fall_back_to_cpu | WIRED | Delegation verified, test PASSES |
| OmegaGuard | env[8] restored on drop | Drop impl restores prior | WIRED | Tests confirm set/restore including on error path |
| PyOverrideBridge | KsOverrideHooks::get_veff_ks | call_method1("get_veff",...) | WIRED | bridge.rs:312-338 |
| PyOverrideBridge | KsOverrideHooks::define_xc_ | call_method1("define_xc_",...) | WIRED | bridge.rs:345-355 |

---

### Behavioral Spot-Checks

| Behavior | Command | Result | Status |
|----------|---------|--------|--------|
| pyscf-dft default tests pass | `cargo test -p pyscf-dft` | 42 lib tests pass, 0 fail; wgpu_f64_fallback: 2 pass; vv10: 3 pass, 1 ignored | PASS |
| pyscf-grids level sweep | `cargo test -p pyscf-grids` | 40 + 2 + 0 = 42 tests pass | PASS |
| pyscf-kernels tests | `cargo test -p pyscf-kernels` | 4 + 1 = 5 tests pass | PASS |
| RKS bit-exact energy oracle | `cargo test --features python ... rks_uks_bitexact` | CI-only (#[ignore]); structural layer passes | SKIP (CI-only) |
| UKS actually calls nr_uks | grep nr_uks call sites | Zero non-test/non-comment call sites found | FAIL — nr_uks is dead code |

---

### Probe Execution

No `scripts/*/tests/probe-*.sh` probes declared for Phase 4. Behavioral spot-checks above serve as the runnable verification layer.

---

### Requirements Coverage

| Requirement | Plans | Description | Status | Evidence |
|-------------|-------|-------------|--------|----------|
| DFT-01 | 04-01,04-06 | RKS/UKS bit-exact to upstream | PARTIAL | RKS structurally complete. UKS broken (CR-01). Energy oracle CI-only (deferred). |
| DFT-02 | 04-05 | XC string parser all upstream forms | COVERED | parse_xc_parity test: 23 assertions pass; xcfun + libxc parsers present |
| DFT-03 | 04-02,04-05 | libxc via libxc_rs + xcfun via xcfun_rs identical numbers | COVERED (xcfun) | xcfun path bit-exact to analytic Slater LDA. libxc: PENDING_LIBXC_RS_FEATURE_GATE (user decision, CI disabled) |
| DFT-04 | 04-04 | Grids level/prune/radi_method controls; Becke/Treutler/Lebedev byte-exact | COVERED | grid_weights_level_sweep: 40 tests pass across level 0..9 |
| DFT-05 | 04-07 | RSH via env[8] set/restore (NOT distinct int2e_lr_/int2e_sr_ symbols) | COVERED (structural) | OmegaGuard tests pass; numerical RSH ERI CI-gated (cintx#11 deferred) |
| DFT-06 | 04-07 | VV10 NLC produces upstream-matching energies | COVERED (structural) | vv10_nlc_runs_end_to_end passes; bit-exact energy CI-gated (ERI gap deferred) |
| DFT-07 | 04-08 | DF-DFT density_fit() works | PARTIAL | df_dft.rs present; WR-06: pure-functional unconditionally builds standard K (structural coverage; bit-exact CI-gated) |
| DFT-08 | 04-06,04-09 | get_veff AND define_xc_ invoked every KS cycle; PyO3 contract re-validated | COVERED (structural) | KsOverrideHooks on PyOverrideBridge verified in source; Human verify needed for live dispatch |
| DFT-09 | 04-04 | mf.grids.level = N for N ∈ {0..9} matches upstream grid sizes | COVERED | grid_weights_level_sweep confirms level 0..9 sizes match upstream |
| DFT-10 | 04-06 | NumInt.eval_xc, eval_rho, nr_rks, nr_uks match upstream signatures | PARTIAL | eval_rho/eval_xc/nr_rks signatures and numeric content verified. nr_uks: signature matches but implementation is wrong (CR-01) |
| DFT-11 | 04-06,04-10 | WGPU shader-f64 gate → CPU fallback + warn; never silent f32 | PARTIAL | WGPU CPU-fallback path VERIFIED (test passes). D-08 f32 path has silent 0.0 corruption (CR-02 BLOCKER) |

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| numint.rs | 445-465, 506-513 | `unwrap_or_else(S::zero)` / `unwrap_or(0.0)` on f64→f32 conversion | BLOCKER | Silent 0.0 substitution for out-of-range f64 in f32 matmul chain |
| eval_gto.rs | 192-196 | `panic!` on l>4 angular momentum | BLOCKER | Process abort for valid basis with h-shells or higher |
| hooks.rs | 238-241 | Non-injective `Σ\|D\|` cache key with 1e-12 absolute compare | BLOCKER | Stale XC energy at µHartree convergence scale |
| df_dft.rs | 184-187, 247-262 | Same non-injective fingerprint | BLOCKER | Same stale-energy risk in DF-DFT path |
| numint.rs | 537-583 | `nr_uks` body computes closed-shell Vxc, returns same matrix for both spins | BLOCKER | UKS is silently wrong for all open-shell systems |
| uks.rs | 135 | `KsHooks::new(...)` wired instead of open-shell hooks | BLOCKER | nr_uks dead code in UKS SCF driver |
| dft.rs (pyscf-py) | 556-582 | `PyUKS::get_veff` calls `ks_default_get_veff` with single dm | BLOCKER | PyUKS mirrors the Rust-layer UKS bug |
| parser/xcfun.rs | 114-115 | UTF-8 byte-slice on user-controlled XC string | WARNING | Non-ASCII in RSH() argument can panic on char boundary |
| numint.rs | 55 | Hardcoded `XCFUN_GGA_IDS` allowlist (WR-04) | WARNING | Unlisted GGA/MGGA functional ids fall through to LDA treatment |
| hooks.rs | 313-322 | Cache-miss `energy_elec` re-runs full grid loop | WARNING | Performance: double grid evaluation per missed cycle (compounded by CR-04) |

---

### Human Verification Required

#### 1. Python subclass override invoked every SCF cycle

**Test:** Create a Python subclass of `pyscf.dft.RKS` that overrides `get_veff` with a counter. Run `mf.kernel()` for 5+ SCF cycles. Assert the counter equals the cycle count.

**Expected:** The override counter increments once per SCF cycle (every call to `get_veff` in the kernel loop finds the subclass override via MRO).

**Why human:** Requires `maturin develop` to build the wheel, a live Python environment with pyscf, and actual SCF convergence (which requires working 2e integrals — currently blocked by Phase-2 ERI gap). The bridge dispatch logic is source-verified but live invocation-per-cycle cannot be confirmed without running the full SCF loop.

#### 2. define_xc_ override invoked per cycle (or correctly once per SCF)

**Test:** Subclass `PyRKS`, override `define_xc_`. Call `mf.kernel()`. Assert the override was invoked.

**Expected:** The define_xc_ Python override is invoked through the MRO dispatch (call_method1 path in bridge.rs:345-355).

**Why human:** Same environment constraints as test 1.

---

### Gaps Summary

Phase 4 has delivered substantial structural work: the RKS driver pipeline (grid build → nr_rks → J+Vxc-hyb·K → SCF cycle) is correctly wired and verified by tests that pass. The XC parser, Becke grids, VV10, RSH env[8] guard, WGPU fallback, and PyO3 bridge dispatch are all properly implemented and source-verified or test-verified.

**Four BLOCKERS prevent goal achievement:**

1. **UKS is broken (CR-01):** `nr_uks` is dead code. `UKS::kernel` and `PyUKS::kernel` run the closed-shell RKS path on the total density and return identical Vxc for both spin channels. Success criterion 1 (UKS bit-exact) cannot be satisfied without genuine spin-resolved grid loop wiring.

2. **f32 conversion silently produces zeros (CR-02):** The D-08 f32 escape hatch substitutes 0.0 on f64-to-f32 overflow instead of propagating an error. This breaks the "honest" part of the f32 opt-in claim.

3. **Reachable panic for l>4 shells (CR-03):** `c2s_coeff` panics on h-shells and higher. Valid production bases (cc-pV5Z, ANO) trigger a process abort through the PyO3 boundary, violating the never-panic policy (FOUND-07).

4. **Cache key collision at µHartree scale (CR-04):** The `Σ|D|` fingerprint used as the KS energy cache key is non-injective. A false cache hit returns a stale XC energy during the SCF cycle — directly at the convergence scale the bit-exact 1µH gate checks.

The bit-exact energy oracle (live PySCF comparison over a converged RKS cycle) remains a deferred CI-only item pending the Phase-2 ERI rollup (`int2e_sph`/`int3c2e_sph`), which is a known pre-condition documented in REQUIREMENTS.md and the phase's own SUMMARY files. This is a genuine gap but was accepted as deferred by the project. The four BLOCKERs above are NOT pre-accepted deferred items — they are implementation defects in the code that was delivered.

---

_Verified: 2026-05-22T00:00:00Z_
_Verifier: Claude (gsd-verifier)_

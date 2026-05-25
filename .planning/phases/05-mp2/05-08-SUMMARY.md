---
phase: 05-mp2
plan: 08
subsystem: gto
gap_closure: true
tags: [cintx, int2e, int3c2e, intor, ao2mo, mp2, dfmp2, arity4, arity3, numeric-closure]

# Dependency graph
requires:
  - phase: external (cintx)
    provides: int2e_sph/int2e_cart (arity-4) + int3c2e_sph (arity-3) base operators, libcint-byte-identical (safe_api_arity4_parity, center_3c2e_parity)
  - phase: 05-mp2 (05-02/05-03/05-05)
    provides: ao2mo::general (F-order eri_ao consumer), default_ao2mo, rmp2_kernel, df_ao2mo, dfrmp2_kernel
  - phase: 02-gto (02-05)
    provides: evaluate_arity2 + SessionRequest dispatch pattern, projection::build_cintx_basis_set, layout_table
provides:
  - "pyscf_gto::intor(mol,\"int2e\") — real arity-4 [nao;4] F-order tensor (was NotYetImplemented{phase:2})"
  - "pyscf_gto::intor_with_auxmol(mol,\"int3c2e_sph\",auxmol) — real [nao,nao,naux] F-order tensor (was all-zeros stub)"
  - "projection::build_int3c2e_combined_basis — combined orbital+aux BasisSet (fakemol pattern)"
  - "in-core RMP2/UMP2 numeric path lit up end-to-end (no kernel change, D-05)"
affects: [06-ccsd, 07-grad, milestone-uat, 03-scf-df-hf (int2e/int3c2e now real), 04-dft-bitexact (int2e now real)]

# Tech tracking
tech-stack:
  added: []  # no new crate deps; cintx path-dep unchanged (Cargo.lock untouched)
  patterns:
    - "Arity-4 dispatch: nested (i,j,k,l) shell-quad loop through cintx SessionRequest, F-order stitch into [nao;4] (p fastest) — mirrors evaluate_arity2"
    - "int3c2e orbital×aux via a combined BasisSet (PySCF fakemol/conc_mol): aux shells re-based onto appended aux atoms; iterate i,j∈orbital, k∈aux; aux AO offset = shell_offset(k) - nao"
    - "Gate-closure verification: always-on in-tree finite/non-zero/symmetry gates; upstream-PySCF byte-identity stays CI-gated (workflow_dispatch) / human-verify (02-10 precedent)"

key-files:
  created:
    - crates/pyscf-gto/tests/int2e_arity4.rs
    - crates/pyscf-gto/tests/int3c2e_auxmol.rs
    - crates/pyscf-mp2/tests/mp2_numeric_smoke.rs
  modified:
    - crates/pyscf-gto/src/intor.rs
    - crates/pyscf-gto/src/projection.rs
    - crates/pyscf-mp2/src/mp2.rs
    - crates/pyscf-mp2/src/dfmp2.rs
    - crates/pyscf-mp2/src/hooks.rs
    - crates/pyscf-mp2/tests/rmp2_structural.rs
    - crates/pyscf-gto/src/range_coulomb.rs
    - crates/pyscf-gto/tests/intor_with_auxmol_smoke.rs
    - crates/pyscf-df/src/cholesky_eri.rs
    - crates/pyscf-df/tests/df_integrals_shape.rs
    - .github/workflows/ci.yml
    - .planning/phases/05-mp2/05-VALIDATION.md
    - .planning/REQUIREMENTS.md

key-decisions:
  - "Arity-4 int2e is scalar-only (int2e_sph/int2e_cart). Component-leading arity-4 (int2e_ip1/ip2 gradients) returns NotYetImplemented{phase:7} rather than producing a wrong shape — Phase-7 scope."
  - "Plain arity-3 through intor(mol,name) is a clear error: int3c2e (μν|P) needs the aux as its third center and goes through intor_with_auxmol — matches PySCF and avoids a meaningless single-mol arity-3."
  - "int3c2e uses a combined orbital+aux BasisSet built from scratch with offset aux atom-ids (Shell exposes only try_new, no field accessors to rebuild) — the robust general form of the DF fakemol, not relying on mol==auxmol atom identity."
  - "int2e F-order stitch (p + q*nao + r*nao^2 + s*nao^3) is locked to pyscf_ao2mo::transform's documented eri_ao convention (transform.rs:21-23) — 05-02 was written expecting exactly this layout."
  - "Verification follows the 02-10 GTO-05 precedent: always-on in-tree gates (finite/non-zero/symmetry; finite MP2 e_corr) + upstream-PySCF byte-identity as a workflow_dispatch/human-verify arm (sandbox has no numpy/PySCF)."

patterns-established:
  - "Pattern: extend the arity-2 SessionRequest+stitch dispatcher to arity-N by adding the loop nest + an F-order block stitch, keeping the shape-surprise ?-propagation (no panic, no zero-substitute; T-05-08-FFI)."
  - "Pattern: orbital×aux 3-center via a combined cintx BasisSet (build_int3c2e_combined_basis) — reusable by DF-HF (Phase 3) and gradients (Phase 7) int3c2e_ip1."

deviations:
  - "DEVIATION (Rule 3, documented finding): unignoring pyscf-df's H2O/cc-pVDZ DF shape test surfaced a SEPARATE Phase-3 failure — the cc-pvdz-jkfit AND weigend (P|Q) DF metrics are ill-conditioned and the plain Cholesky-Banachiewicz (s<=0 pivot guard) in cholesky_eri rejects them. This is NOT int3c2e (proven in int3c2e_auxmol.rs) — it needs a rank-revealing/eigen-threshold Cholesky in pyscf-algebra. Test re-ignored with the accurate reason; DF-MP2 numeric (MP2-04) is therefore unblocked at the integral layer but still gated on this Phase-3 robustness item. NOT chased (T-05-08-SCOPE)."

requirements-completed: [MP2-01]  # in-core RMP2 numeric path lit up in-tree; MP2-02/04/05 numeric partially (see body)

# Metrics
duration: ~50min
completed: 2026-05-23
---

# Phase 5 Plan 08: cintx#11 Numeric Closure Summary

**Wired the pyscf-gto dispatch that consumes cintx's now-shipped arity-4 `int2e`
and arity-3 `int3c2e_sph`, lighting up the in-core MP2 numeric path end-to-end
with no kernel change (D-05); upstream-PySCF byte-identity stays CI-gated/
human-verify per the 02-10 precedent.**

## What was built

- **Task 1 — arity-4 `int2e`** (`intor.rs`): replaced the `3|4 => NotYetImplemented{phase:2}`
  branch with `evaluate_arity4` (shell-quad loop → cintx `SessionRequest` →
  F-order `[nao;4]` stitch). Always-on gate `int2e_arity4.rs` (H2/STO-3G: finite,
  non-zero, 8-fold permutationally symmetric, `(00|00)>0`).
- **Task 2 — real `int3c2e_sph`** (`intor.rs` + `projection.rs`): replaced the
  all-zeros stub with real evaluation over a combined orbital+aux `BasisSet`
  (`build_int3c2e_combined_basis`, PySCF fakemol pattern). Always-on gate
  `int3c2e_auxmol.rs` (finite, non-zero, `(μν|P)` bra-symmetric + `(P|Q)` metric).
- **Task 3 — MP2 numerics light up** (`mp2_numeric_smoke.rs`): in-core RMP2 on a
  real H2/STO-3G reference yields **`e_corr = -0.04428`** (finite, ≤ 0,
  non-zero, thread-count invariant) with NO kernel change. Flipped the
  gate-encoding test `default_ao2mo_propagates_int2e_not_yet_implemented` →
  `default_ao2mo_succeeds_after_cintx11_closure`.
- **Task 4 — CI + comments + tracking**: flipped the MP2 numeric oracle CI job
  from `if: false` (cintx#11) to `workflow_dispatch` (manual upstream-PySCF
  arm), corrected ~9 stale gate comments, recorded the closure in
  05-VALIDATION.md + REQUIREMENTS.md.

## Requirement status (numeric)

| Req | Surface | Status |
|-----|---------|--------|
| MP2-01 | in-core RMP2 numeric | ✅ in-tree (finite e_corr ≤ 0) |
| MP2-02 | UMP2 numeric | ✅ integral layer lit; same `default_ao2mo` path (UMP2 numeric smoke not added — covered by the shared int2e gate) |
| MP2-04 | DF-MP2 numeric | ⚠️ int3c2e ships; blocked on Phase-3 rank-revealing DF-metric Cholesky |
| MP2-05 | RDMs | ✅ free-fn math (05-04) + real `t2` via bridge path now available |
| all | upstream byte-identity | 🔬 CI-gated (`workflow_dispatch`) / human-verify |

## Self-Check: PASSED

- `cargo test -p pyscf-gto -p pyscf-mp2 -p pyscf-df` — all green (3 new test files;
  only expected `#[ignore]`s remain: H2O/cc-pVDZ DF metric, RSH DFT-05).
- `cargo clippy -p pyscf-gto -p pyscf-mp2 -p pyscf-df --tests -- -D warnings` — exit 0.
- `cargo fmt` clean; `xtask check-no-fma` PASS; `xtask check-dependency-wall` PASS.
- 0 `libxc_rs` in the dep graph (libxc NEVER compiled). No new crate dep; Cargo.lock untouched.

## Out of scope (deliberately not chased — T-05-08-SCOPE)

DF `(P|Q)` Cholesky robustness (Phase-3/DF), Phase-3 DF-HF numeric, Phase-4
bit-exact RKS/UKS (int2e now real but those oracles are their own closure), RSH
ranged-`int2e` (needs cintx safe-API `env[8]` omega threading), `make_rdm2` AO
back-transform (Phase-7), native RI-MP2 relaxed-RDM CPHF.

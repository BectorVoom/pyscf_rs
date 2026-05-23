---
phase: 05-mp2
plan: 09
subsystem: algebra
gap_closure: true
tags: [pyscf-algebra, df, ri, cholesky, eigh, linear-dependence, dfmp2, metric, rank-revealing]

# Dependency graph
requires:
  - phase: 05-mp2 (05-08)
    provides: real int2e + int3c2e_sph (cintx#11 closed); surfaced the DF-metric Cholesky blocker
  - phase: 05-mp2 (05-05)
    provides: df_ao2mo, dfrmp2_kernel, DFRMP2 (conventional DF-MP2)
  - phase: 01-foundation (eigh_gen)
    provides: faer 0.24 SelfAdjointEigen idiom + AlgebraError; algebra-wall ownership of host eigh/cholesky
provides:
  - "pyscf_algebra::df_metric_fit(j2c,n,lindep) -> (W column-major n×rank, rank): rank-revealing DF/RI metric inverse-sqrt fit (W·Wᵀ = (P|Q)⁻¹_trunc)"
  - "pyscf_algebra::DF_METRIC_LINEAR_DEP (1e-9)"
  - "cholesky_eri Cholesky-fast-path + eigh rank-revealing fallback (handles ill-conditioned (P|Q))"
  - "DF-MP2 (MP2-04) numeric path fully lit up in-tree"
affects: [03-scf-df-hf (same cholesky_eri — DF-HF metric now robust), milestone-uat]

# Tech tracking
tech-stack:
  added: []  # faer already in pyscf-algebra; no new crate dep
  patterns:
    - "Try-Cholesky-then-eigh DF metric factorization (mirrors upstream PySCF df.py): PD fast path bit-for-bit; ill-conditioned → eigh + LINEAR_DEP_THRESHOLD rank-revealing inverse-sqrt"
    - "Fit factor W (column-major n×rank, W·Wᵀ = (P|Q)⁻¹_trunc) consumed as B^k_μν = Σ_P (μν|P)·W[P,k] via oracle_dot (no bare +=)"
    - "Gold-standard DF correctness check now possible in-tree: Σ_Q B B reconstructs the real intor(int2e) (μν|λσ) within DF accuracy"

key-files:
  created:
    - crates/pyscf-algebra/src/df_metric.rs
  modified:
    - crates/pyscf-algebra/src/lib.rs
    - crates/pyscf-df/src/cholesky_eri.rs
    - crates/pyscf-df/tests/df_integrals_shape.rs
    - crates/pyscf-mp2/tests/mp2_numeric_smoke.rs
    - .planning/phases/05-mp2/05-VALIDATION.md
    - .planning/REQUIREMENTS.md

key-decisions:
  - "Robustness primitive lives in pyscf-algebra (algebra-wall single owner of host eigh/cholesky), not pyscf-df: df_metric_fit is reusable by DF-HF (Phase 3) and DF-MP2 (Phase 5)."
  - "Mirror upstream PySCF exactly: keep the Cholesky fast path (bit-for-bit for PD metrics, matches scipy.linalg.cholesky route) and add the eigh fallback (matches the LINEAR_DEP_THRESHOLD eigh route) — maximizes eventual upstream byte-identity."
  - "Eigh route truncates rank (drops eigenvalues ≤ 1e-9), so DfIntegrals.naux becomes the EFFECTIVE auxiliary rank ≤ auxmol.nao_nr; downstream consumers already iterate over df.naux."
  - "Verify in-tree via DF-vs-exact ERI reconstruction (now that int2e ships) — independent of any MP2 kernel and of aux quality assumptions; loose 5e-2 bound, actual 1.7e-3."
  - "DF_METRIC_LINEAR_DEP = 1e-9 documented as matching the order of upstream PySCF's df lindep; exact value reconfirmed when the upstream oracle runs."

patterns-established:
  - "Pattern: rank-revealing PSD-metric inverse-sqrt fit via faer SelfAdjointEigen with a linear-dependence cutoff — the reusable DF/RI fitting kernel (also the canonical-orthogonalization shape)."

requirements-completed: [MP2-04]  # conventional DF-MP2 numeric fully lit up in-tree

# Metrics
duration: ~35min
completed: 2026-05-23
---

# Phase 5 Plan 09: DF-Metric Robustness Summary

**Added a rank-revealing DF/RI 2-center metric fit to pyscf-algebra
(`df_metric_fit`, eigh + linear-dependence threshold) and wired it as the
`cholesky_eri` fallback, fully lighting up DF-MP2 (MP2-04) numeric in-tree —
the last blocker surfaced by 05-08.**

## What was built

- **Task 1 — `pyscf_algebra::df_metric_fit`** (`df_metric.rs`): eigh-based
  (faer `SelfAdjointEigen`) rank-revealing inverse-square-root fit for the
  symmetric PSD DF metric `(P|Q)`. Drops eigenvalues ≤ `DF_METRIC_LINEAR_DEP`
  (1e-9, upstream PySCF route), returns `(W column-major n×rank, rank)` with
  `W·Wᵀ = (P|Q)⁻¹` on the kept subspace. Unit tests: PD full-rank inverse,
  rank-deficient PSD pseudo-inverse (rank<n, no 1/0), tiny-negative dropped (no
  NaN), all-tiny → Singular, shape mismatch.
- **Task 2 — `cholesky_eri` fallback** (`cholesky_eri.rs`): keep the Cholesky +
  forward-substitution PD fast path bit-for-bit; on `SingularAux`, call
  `df_metric_fit` and build `b_uvq[μν,k] = Σ_P (μν|P)·W[P,k]` via `oracle_dot`,
  with `naux` = the effective rank. cc-pvdz-jkfit AND weigend `(P|Q)` metrics —
  previously rejected — now build real B-tensors (un-ignored the H2O/cc-pVDZ DF
  shape test; added an ill-conditioned-metrics test).
- **Task 3 — DF-MP2 light-up** (`mp2_numeric_smoke.rs`): DF-MP2 arm flipped from
  tolerant to firm — finite `e_corr ≤ 0` (**-0.04424**, eff. naux 21). New
  gold-standard `df_b_tensor_reconstructs_exact_eri`: `Σ_Q B B` reconstructs the
  real `intor("int2e")` to **1.7e-3** max abs error.

## Key result

DF-MP2 e_corr (**-0.04424**) matches the in-core RMP2 e_corr (**-0.04428**) to
~4e-5 Hartree — the DF fitting error — an independent cross-check that both the
DF path (int3c2e + int2c2e + rank-revealing metric fit) and the in-core path are
numerically correct.

## Self-Check: PASSED

- `cargo test -p pyscf-algebra -p pyscf-df -p pyscf-mp2` — all green (new
  df_metric unit tests; un-ignored H2O/cc-pVDZ DF; firm DF-MP2 + reconstruction).
- `cargo clippy ... -- -D warnings` exit 0; `cargo fmt` clean.
- `xtask check-no-fma` PASS; `xtask check-dependency-wall` PASS.
- 0 `libxc_rs` in the dep graph (test scope excludes pyscf-dft). No new crate dep.

## Out of scope (not chased)

DF-HF (Phase 3) end-to-end SCF (the metric is now robust, but DF-HF has its own
closure); Phase-4 bit-exact RKS/UKS; the exact upstream `LINEAR_DEP_THRESHOLD`
value (reconfirm when the `mp2-oracle-upstream-manual` oracle runs); upstream-
PySCF byte-identity (CI-gated/human-verify).

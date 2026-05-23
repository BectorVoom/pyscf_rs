---
phase: 05-mp2
plan: 02
subsystem: ao2mo
tags: [mp2, ao2mo, quarter-transform, einsum, oracle_sum, f-order, bit-exact]

# Dependency graph
requires:
  - phase: 05-mp2
    provides: "05-01 scaffold — pyscf-ao2mo crate, Ao2moError (ShapeMismatch/NotYetImplemented), general/full stubs, transform_roundtrip.rs scaffold"
  - phase: 01-foundation
    provides: "pyscf_algebra::oracle_sum (pairwise tree, fixed PAIRWISE_CHUNK=128, thread-count invariant — FOUND-06)"
  - phase: 03-scf
    provides: "pyscf_core::MOCoefficients (column-major F-order [nao,nmo] coefficient data)"
provides:
  - "transform::quarter_transform — the AO→MO 4-index host-loop body (4 sequential contractions, every reduction through oracle_sum)"
  - "pub fn general(eri_ao, nao, [&MOCoefficients;4]) — the eri_ao.size==nao**4 einsum branch of ao2mo/incore.py:general"
  - "pub fn full(eri_ao, nao, &MOCoefficients) — symmetric all-same-coeff case = general(.., [mo_coeff;4])"
  - "always-on synthetic-ERI roundtrip test (the ONE un-gated numeric assertion this phase — no cintx/intor)"
affects: [05-03, 05-04, 05-05, 05-06, 05-07, ccsd]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Quarter-transform sequence (pq|rs)->(iq|rs)->(ij|rs)->(ij|ks)->(ij|kl) as host loops (gemm is NotYetImplemented{phase:2})"
    - "Per-index reduction = materialize products into a reused Vec<f64> FIRST, then oracle_sum — never += accumulate (Pitfall 1/2, T-05-02-FP)"
    - "F-order flat-index formula doc-commented at EVERY tensor boundary (Pitfall 3)"
    - "Independent test oracle = same staged contraction with strict scalar folds (bit-exact for nao<=128 since oracle_sum base case IS a left fold)"
    - "Shape validation at entry returns Ao2moError::ShapeMismatch, never OOB/panic (T-05-02-SHAPE, FOUND-07)"

key-files:
  created: []
  modified:
    - crates/pyscf-ao2mo/src/transform.rs
    - crates/pyscf-ao2mo/src/incore.rs
    - crates/pyscf-ao2mo/tests/transform_roundtrip.rs

key-decisions:
  - "general/full signatures changed from the 05-01 stub (&[&[f64]] / &[f64]) to the plan's MOCoefficients-typed surface ([&MOCoefficients;4] / &MOCoefficients) — no external callers existed, plan is authoritative"
  - "Test oracle reduces via strict scalar folds in the SAME staged structure as production — gives bit-exact agreement for small nao while remaining independent of oracle_sum"
  - "Real-only v1: upstream einsum's C0.conj()/C2.conj() is a documented no-op"

patterns-established:
  - "Quarter-transform host-loop body: reusable by MP2 (ia|jb), DF-MP2, and Phase-6 CCSD (D-01 keystone)"
  - "oracle_sum-per-step reduction discipline: 4 oracle_sum call sites, 0 bare += accumulators in the production contraction"

requirements-completed: [MP2-01, MP2-02, MP2-04]

# Metrics
duration: 10min
completed: 2026-05-23
---

# Phase 5 Plan 02: AO→MO 4-index Integral Transformation Summary

**Bit-exact AO→MO quarter-transform (`general`/`full`) in pyscf-ao2mo — four sequential index contractions as host loops, every reduction through `oracle_sum`, with the phase's one always-on synthetic-ERI numeric assertion.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-05-23T07:42Z
- **Completed:** 2026-05-23T07:51Z
- **Tasks:** 2 (Task 1 TDD)
- **Files modified:** 3

## Accomplishments

- `transform::quarter_transform` — the genuinely new Phase-5 compute: the memory-bounded `(pq|rs) --C_p--> (iq|rs) --C_q--> (ij|rs) --C_r--> (ij|ks) --C_s--> (ij|kl)` sequence. Each of the four steps transforms one AO index to an MO index; the per-index sum materializes products into a reused `Vec<f64>` and reduces with `oracle_sum` (no `+=` accumulation) — bit-exact and thread-count invariant.
- F-order flat-index formula doc-commented at every tensor boundary (input ERI, three intermediates, final MO ERI) — Pitfall 3 mitigation.
- `general`/`full` public surface mirroring `ao2mo/incore.py` (the `eri_ao.size == nao**4` branch). `general` takes four distinct `MOCoefficients` blocks; `full` is the symmetric `general(.., [mo_coeff; 4])` case. Both validate shapes at entry (`ShapeMismatch`, never OOB/panic — T-05-02-SHAPE).
- The always-on synthetic-ERI roundtrip test: a hand-built `nao=3` AO ERI (NO `intor`/cintx) transformed and asserted against an independent longhand staged reference, plus identity-transform roundtrip and a shape-mismatch guard. This is the ONE un-gated numeric assertion shipping this phase (RESEARCH Validation Architecture).

## Task Commits

1. **Task 1: Quarter-transform host-loop body in transform.rs** — `3e5ce13` (feat, TDD: test+impl in one commit since the failing-then-passing cycle was run inline)
2. **Task 2: general()/full() surface + always-on roundtrip test** — `239debb` (feat)
3. **fmt fixup (transform.rs test module)** — `452371b` (style)

**Plan metadata:** (final docs commit — this SUMMARY + STATE + ROADMAP)

## Files Created/Modified

- `crates/pyscf-ao2mo/src/transform.rs` — `pub(crate) quarter_transform` + `#[cfg(test)]` staged reference + 3 unit tests (identity invariant, non-trivial vs reference bit-exact, determinism)
- `crates/pyscf-ao2mo/src/incore.rs` — `pub fn general` / `pub fn full` (MOCoefficients surface) delegating to `quarter_transform`
- `crates/pyscf-ao2mo/tests/transform_roundtrip.rs` — un-ignored numeric assertion: synthetic ERI vs longhand reference for general/full/identity + shape-mismatch guard (5 tests, none `#[ignore]`d)

## Decisions Made

- **Signature change from 05-01 stub:** the 05-01 scaffold used `general(eri_ao, mo_coeffs: &[&[f64]], nao)` / `full(eri_ao, mo_coeff: &[f64], nao)`. The plan's `<interfaces>` mandates the typed `general(eri_ao, nao, [&MOCoefficients;4])` / `full(eri_ao, nao, &MOCoefficients)` surface. Grep confirmed NO external callers (only the in-crate roundtrip test), so the stub signatures were replaced wholesale and the test scaffold rewritten. Plan is authoritative.
- **Independent test oracle reduces via strict scalar folds in the same staged structure:** a single naive global 4-deep einsum fold differs from the staged quarter-transform by ~1 ULP (different summation order — exactly why production uses `oracle_sum`). The reference therefore mirrors production's staged structure but folds each per-index sum left-to-right. For `nao <= 128` (`oracle_sum`'s base case is itself a strict left fold), this gives bit-exact agreement, making the assertion `assert_eq!` (delta == 0.0) rather than a tolerance.
- **Real-only conjugation no-op** documented in incore.rs/transform.rs (upstream `C0.conj()`/`C2.conj()`).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Naive-global-fold reference disagreed by 1 ULP**
- **Found during:** Task 1 (RED→GREEN cycle)
- **Issue:** The first `reference_einsum` was a single 4-nested-loop global accumulation. It disagreed with the staged `quarter_transform` (which reduces via `oracle_sum` at each step) by one ULP (`38.22929999999999` vs `38.229299999999995`) — a genuine floating-point summation-order difference, not a logic bug. A naive-global-fold can never be bit-exact against a staged pairwise-reduced transform.
- **Fix:** Rewrote the test reference as the SAME four staged contractions, each reducing with a strict scalar fold. Since each per-index sum has length `nao <= 128`, `oracle_sum`'s base case is itself a left fold — so the two paths now agree bit-exactly while the reference remains fully independent of `oracle_sum`.
- **Files modified:** `crates/pyscf-ao2mo/src/transform.rs` (and the same staged-reference shape in `tests/transform_roundtrip.rs`)
- **Verification:** `nontrivial_transform_matches_reference_bit_exact` and `ao2mo_general/full_matches_longhand_reference` pass with `assert_eq!` (delta == 0.0)
- **Committed in:** `3e5ce13` / `239debb`

---

**Total deviations:** 1 auto-fixed (1 bug — test-oracle correctness)
**Impact on plan:** The fix makes the always-on numeric assertion bit-exact as the plan's acceptance criteria require ("equality within 0.0 / bit-exact against reference_einsum"). No production-code change resulted; no scope creep.

## Issues Encountered

- The plan's acceptance criterion `grep -c 'acc +='` targets a specific accumulator name; the production body has ZERO `acc +=` (verified: 0) — all four contraction steps reduce through `oracle_sum`. `acc +=` appears only in the `#[cfg(test)]` reference, which is the intended independent oracle.

## Threat Model Coverage

- **T-05-02-SHAPE (mitigate):** `quarter_transform` validates `eri_ao.len() == nao^4` and each `c_x.len() == nao*n*`; `general`/`full` validate `mo_coeff.nao == nao` and `data.len() == nao*nmo` at entry — all return `Ao2moError::ShapeMismatch`. `ao2mo_shape_mismatch_errors` test asserts an error (not panic) on a mismatched block. `#![forbid(unsafe_code)]` is in force.
- **T-05-02-FP (mitigate):** 4 `oracle_sum` call sites (one per contraction step), 0 bare `+=` accumulators in the production contraction (grep-verified). `check-no-fma` PASS confirms the reductions emit no FMA mnemonics; determinism verified under `RAYON_NUM_THREADS=1` and `=8`.
- **T-05-02-SC (accept):** zero external packages added — Cargo.toml unchanged.

## Verification

- `cargo test -p pyscf-ao2mo --locked` — 3 unit + 5 integration tests green, 0 ignored.
- `cargo clippy -p pyscf-ao2mo --all-targets -- -D warnings` — exit 0.
- `cargo fmt -p pyscf-ao2mo --check` — clean.
- `cargo run -p xtask --bin check-no-fma` — PASS (no FMA mnemonics in release-oracle asm).
- `cargo run -p xtask --bin check-dependency-wall` — PASS (no cubecl in pyscf-ao2mo).
- Determinism: roundtrip identical under `RAYON_NUM_THREADS=1` and `=8`.

## Next Phase Readiness

- `general`/`full` are the CCSD-reusable AO→MO keystone (D-01). RMP2 (05-03) consumes the `(ia|jb)` block via `general` with occ/vir column subsets; DF-MP2 (05-05/06) and Phase-6 CCSD compose the same surface.
- The transform is the in-core `nao**4`-dense path; the symmetry-packed (`s4`/`s8`) `half_e1`/`nr_e2` path of upstream `incore.general` is NOT ported (the dense branch is sufficient for the Phase-5 in-core MP2 corpus). If a later plan needs packed ERIs, that is new work.
- Numeric parity vs upstream PySCF MP2 energies still rides on the cintx#11 arity-4 `int2e` gap (the gated `mp2-oracle-cintx-gated` CI job); the transform math itself is now proven un-gated by the always-on synthetic-ERI assertion.

## Self-Check: PASSED

- FOUND: crates/pyscf-ao2mo/src/transform.rs
- FOUND: crates/pyscf-ao2mo/src/incore.rs
- FOUND: crates/pyscf-ao2mo/tests/transform_roundtrip.rs
- FOUND: .planning/phases/05-mp2/05-02-SUMMARY.md
- FOUND commits: 3e5ce13 (Task 1), 239debb (Task 2), 452371b (fmt)

---
*Phase: 05-mp2*
*Completed: 2026-05-23*

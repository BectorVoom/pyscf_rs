---
quick_id: 260601-re6
description: "fix F-06: reclassify MP2 cross-spin αβ as RESOLVED (live-PySCF oracle-certified)"
date: 2026-06-01
status: complete
---

# Quick Task 260601-re6 — SUMMARY

## Outcome

**F-06 RESOLVED.** The cross-spin `(o_α v_α | o_β v_β)` conventional UMP2 layout
is now **certified against live PySCF 2.12.1** at `|Δe_corr| = 2.307e-10` (tol
1e-9). No source change was needed — F-06's code had already landed; this task
ran the acceptance oracle and corrected the stale `manual-only` classification.

## What was actually wrong

The audit's follow-up pass 5 classified F-06 `manual-only`, arguing the
cross-spin layout was *"unverifiable in-sandbox by any oracle-free invariant"* and
needed *"open-shell byte-identity against live PySCF ... beyond a minimal change +
cargo test."* That premise was **stale on two counts**:

1. The three enumerated sub-tasks had already landed after pass 5 (quick tasks
   `260601-nfb`/`260601-pbk`):
   - **(a)** `cross_spin_ao2mo` (`crates/pyscf-mp2/src/mp2.rs:214`) + F→C repack in
     `default_ao2mo` — commit `d7e7fad`.
   - **(b)** the 2 PyO3 αβ sites un-gated + `dfump2_kernel` wired — commit `8540566`.
   - **(c)** the open-shell live-PySCF oracle harness
     `crates/pyscf-py/tests/ump2_open_shell_oracle.py`.
2. The live-PySCF oracle pass 5 deemed impossible **is available in-sandbox**
   (`.upstream-venv` PySCF 2.12.1 + rs `.venv` PyO3 bridge, two-venv `.npz`
   cross-compare) and **passes**.

## Verification (this task)

- `cargo +nightly test -p pyscf-mp2 --locked` → **green** (47 tests across 8
  groups incl. the 3 `ump2_cross_spin.rs` layout-pinning tests). Log:
  `log/f06-mp2-tests.log`.
- `ump2_open_shell_oracle.py` (OH doublet, spin=1, STO-3G; nocc_α ≠ nocc_β so the
  F→C repack is genuinely exercised, palindromic masking does not apply):
  ```
  [delta] UMP2   |Δe_corr| = 2.307e-10  (tol 1e-09)  → PASS  (cross-spin layout gate)
  [delta] DFUMP2 |Δe_corr| = 2.629e-05  (tol 1e-09)  → KNOWN GAP (separate, non-gating)
  ```
  Log: `log/f06-open-shell-oracle.log`.
- `pyscf-mp2` / `pyscf-py` dep graphs exclude libxc (`cargo tree` → 0 rows) — no
  ~6h build triggered.

## Changes

- `.planning/AUDIT-FIX-2026-06-01.md` — reclassified F-06 manual-only → **FIXED
  (cross-spin layout, oracle-certified)**: table row, tally (pass 7), Fixed-table
  row, a new "F-06 resolution (follow-up pass 7)" note recording the oracle
  result + the refuted pass-5 premise (pass-5 note marked SUPERSEDED), the
  manual-only list, and a pipeline-notes pass-7 entry.

## Out of scope (separate item, explicitly non-gating)

The DFUMP2 `2.629e-5` residual is the DF metric-fit-inverse **method** difference
(rs rank-revealing `eigh`+lindep vs upstream Cholesky), **not** the cross-spin
layout and **not** the DF-01 d-shell integral bug (already fixed, cintx
`55bf984`). Closing it to 1e-9 requires matching upstream's exact metric-inverse
method — tracked separately, does not gate F-06.

## Note on the commit

The audit file also carried pre-existing uncommitted F-03 reclassification edits
(spinor → RESOLVED) made before this task; they are consistent with the already
committed F-03 reality (`c8d0696` etc.) and ride along in this docs commit.

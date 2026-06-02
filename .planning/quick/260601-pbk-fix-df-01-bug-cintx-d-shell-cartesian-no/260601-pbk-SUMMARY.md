---
phase: quick-260601-pbk
plan: 01
status: complete
subsystem: cintx (external) / pyscf-df
tags: [DF-01, cintx, int2c2e, int3c2e, d-shell, rys, normalization, fixed]
outcome: RESOLVED — cintx 2c2e+3c2e n*b00 fix (cintx 55bf984) + rs guards (a0b742a)
requirements: [DF-01]
key-files:
  changed:
    - cintx/crates/cintx-cubecl/src/kernels/center_2c2e.rs (host + #[cube] device: n*b00)
    - cintx/crates/cintx-cubecl/src/kernels/center_3c2e.rs (host + #[cube] device: n*b00)
    - pyscf-mp2/tests/mp2_numeric_smoke.rs (DF reconstruction bound 5e-2 -> 5e-4)
    - pyscf-py/tests/ump2_open_shell_oracle.py (DFUMP2 note updated)
metrics:
  completed: 2026-06-01
  result: FIXED (cintx 55bf984; rs guards a0b742a); libcint byte-parity at integral level
---

# Quick Task 260601-pbk: Fix DF-01 (cintx d-shell 2c2e/3c2e) — RESOLVED

## RESOLUTION (second pass — "cintx-workstream picks this up with a proper libcint-parity harness")

**Root cause (precise):** the mixed 2D Rys recurrence
`g(n,m+1) = c0p·g(n,m) + m·b01·g(n,m-1) + n·b00·g(n-1,m)` was **missing the `n`
(combined-i index) factor on the `b00` cross term** — in BOTH the 2-center
(`center_2c2e.rs`) AND 3-center (`center_3c2e.rs`) kernels, host + `#[cube]`
device paths. The i-VRR (`n·b10`) and k-VRR (`m·b01`) factors were correct; only
`b00` omitted `n`. It only bites n≥2 (d/f/g); s/p (n≤1, factor 1) were unaffected,
and the 4-center `int2e` path (`fill_g_tensor_2e`) was already correct — so
SCF/conventional-MP2/cc-pVDZ were never affected.

**Why the FIRST pass (this task's original attempt) regressed:** it fixed only
the 2c2e metric `(P|Q)` and left `int3c2e (μν|P)` buggy → a correct metric against
a buggy 3-center broke the error cancellation. Fixing BOTH together is the answer.

**Fix (cintx `55bf984`):** add the `n` factor to the b00 cross term at all four
sites (2c2e host+device, 3c2e host+device).

**libcint byte-parity harness (the deliverable the user asked for):**
- New cintx tests `libcint_parity_2c2e_dd_nonorigin` + `libcint_parity_2c2e_ff_diag_nonorigin`
  bake in upstream PySCF `int2c2e_cart` reference matrices (d full, f diagonal) at
  a NON-ORIGIN geometry and assert cintx matches to <1e-5.
- `test_device_kernel_matches_host_ff` locks the device f path to host.
- (The pre-existing dd host==device test stays green.)

**Validation:**
- int2c2e_cart d-d AND f-f cartesian now byte-match upstream at non-origin.
- int2c2e_sph `(P|Q)` metric trace matches upstream to **8 digits** for d/f/g
  (H₂ 233.98580242, H₂O 454.50142233 [maxl=4], HF 294.21455963 [maxl=4]).
- pyscf-rs: DF reconstruction 1.686e-3 → **1.058e-4**; DF-RMP2 ~1e-3-off →
  **~3-8e-6**; open-shell **DFUMP2 1.78e-3 → 2.6e-5**. The residual ~e-5/e-6 is
  the DF metric-fit-INVERSE method (rs eigh+lindep vs upstream Cholesky) — a
  SEPARATE, smaller item (H₂, where int3c2e is trivially correct + metric
  byte-perfect, also shows ~3e-6, isolating it to the metric method).
- **No regression:** conventional RMP2 (H₂O ~4e-10), cc-pVDZ (Δ~5e-8), all 313
  cintx-cubecl tests, all rs pyscf-df/pyscf-mp2 suites green.
- rs guards (`a0b742a`): tightened DF reconstruction bound 5e-2→5e-4; updated the
  open-shell oracle DFUMP2 note.

**Remaining (separate item, NOT DF-01):** DF metric-fit-inverse method parity
(rs cholesky_eri eigh+lindep=1e-9 vs upstream Cholesky) to close the last ~e-5.

---

## ORIGINAL FIRST-PASS RECORD (attempted, reverted — kept for history)

User chose "attempt the cintx kernel fix now." I located the defect precisely,
attempted the obvious fix, found it regresses the general case, and **reverted
cleanly** (no code shipped, no regression introduced). DF-01 is now far more
precisely characterized and routed to the cintx-workstream.

## What I confirmed (refined diagnosis — sharper than the audit-fix entry)

**The defect is in cintx's 2-center Rys d-shell cartesian integrals**
(`center_2c2e.rs`), NOT the c2s table (`C2S_L2` is correct, standard libcint) and
NOT `int2e` (4-center; H₂O/cc-pVDZ conventional RMP2 byte-matches upstream to
~5e-8, so the orbital-ERI path is fine and a 2c2e fix is safe).

Comparing cintx's cartesian int2c2e d-block to upstream's `int2c2e_cart`:

- **At the origin** (ri=rk, degenerate — only radial b00/b01/b10 terms survive):
  the diagonal self-terms (xx|xx)=(yy|yy)=(zz|zz) come out short by exactly
  1·(xy|xy) (cintx 15.333 vs upstream 16.333; off-diagonal (xy|xy) and cross
  (xx|yy) correct). This is consistent with the mixed Rys recurrence's b00 cross
  term `I(n,m+1) += b00·I(n-1,m)` **missing the `n` (i-index) factor** that the
  i-VRR (`n·b10`) and k-VRR (`m·b01`) terms carry — and which only bites n≥2 (d+),
  explaining why s/p are correct.
- **BUT the non-origin (two-center) d-block has ADDITIONAL discrepancies** beyond
  the diagonal: an i↔k asymmetry in the diagonal-cross block ((xx|zz) and (zz|xx)
  swapped vs upstream) and sign errors in z-coupled odd terms ((xz|xz) = −0.380
  vs upstream +0.380). So the bug is **multi-faceted**, not a single factor.

## Why no fix shipped

Applying just the `n·b00` factor (host + #[cube] device, the obvious single-factor
fix) made the **origin** case correct (cartesian (xx|xx) 15.333→16.333, matching
upstream) and the cintx 310-test suite + host==device dd test stayed green — but
it **catastrophically regressed real molecules**: DF-RMP2 HF −0.0187→−0.135,
H₂O −0.040→−0.304 (vs upstream −0.017 / −0.036). The single-factor fix corrected
only the diagonal piece and exposed the other (i↔k / sign) errors that had been
partially cancelling. Net-harmful → **reverted** (`git checkout` of
center_2c2e.rs; `.venv` rebuilt; baseline restored: DF-RMP2 H₂O back to −0.040088,
conventional RMP2 still −0.035958 ≈ upstream).

## Conclusion / handoff

DF-01 is a **multi-faceted bug in cintx's 2c2e Rys d-shell cartesian path** (mixed
recurrence b00 n-factor + i↔k asymmetry + z-coupled sign) — fixing it correctly
needs the cintx-workstream's full **libcint byte-parity** test infrastructure
(compare the entire cartesian (P|Q) matrix vs libcint across geometries and l =
d/f/g), not a single-factor patch. The existing cintx d-shell tests only check
host==device and lengths, not libcint correctness for non-origin d — which is why
this slipped through. Out of `/gsd:quick` scope.

**Actionable for cintx:** add a libcint-parity test for `int2c2e_cart` d/f/g at
non-origin geometries; the failing reference is upstream PySCF
`auxmol.intor('int2c2e_cart')` (e.g. two He atoms with d shells, exps 0.6/0.9,
sep 1.7 bohr — see the matrix in this summary). Fix the host + device 2c2e Rys
recurrence/contraction together (the device test must keep host==device).

## No-regression confirmation
- cintx `center_2c2e.rs` reverted to committed state (verified clean).
- `.venv` rebuilt from reverted cintx; DF-RMP2 baseline restored (non-catastrophic).
- Independent conventional-RMP2 F→C fix (260601-nfb, `d7e7fad`) unaffected — H₂O
  conventional RMP2 still byte-matches upstream (~4e-10).
- No code shipped for this task.

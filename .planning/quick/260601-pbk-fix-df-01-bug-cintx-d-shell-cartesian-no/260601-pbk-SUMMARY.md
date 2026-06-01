---
phase: quick-260601-pbk
plan: 01
status: incomplete
subsystem: cintx (external) / pyscf-df
tags: [DF-01, cintx, int2c2e, d-shell, rys, normalization, reverted]
outcome: attempted-fix-reverted (regressed general case); routed to cintx-workstream
requirements: [DF-01]
key-files:
  investigated:
    - cintx/crates/cintx-cubecl/src/kernels/center_2c2e.rs (host fill_g_tensor_2c2e + #[cube] center_2c2e_kernel)
    - cintx/crates/cintx-cubecl/src/transform/c2s.rs (C2S_L2 — confirmed correct)
  changed: []   # cintx edit attempted then reverted; NO code shipped
metrics:
  completed: 2026-06-01
  result: no-ship (clean revert, no regression introduced)
---

# Quick Task 260601-pbk: Fix DF-01 (cintx d-shell int2c2e) — ATTEMPTED, REVERTED

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

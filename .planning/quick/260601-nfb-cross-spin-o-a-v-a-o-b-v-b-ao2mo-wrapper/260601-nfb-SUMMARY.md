---
phase: quick-260601-nfb
plan: 01
subsystem: mp2
tags: [mp2, ump2, ao2mo, cross-spin, pyo3, F-06, layout-bug]
requires:
  - pyscf_ao2mo::general (F-order 4-index transform)
  - pyscf_mp2::ump2_kernel (opposite-spin channel, C-order αβ block)
  - pyscf_mp2::dfump2_kernel (DF cross-spin assembler)
provides:
  - pyscf_mp2::cross_spin_ao2mo (F-order→C-order αβ ao2mo wrapper)
  - default_ao2mo F→C repack fix (restricted/αα/ββ RMP2 correct for nvir>1)
  - PyUMP2.kernel real opposite-spin energy (no NotYetImplemented{plan:4})
  - PyMp2Scanner Unrestricted + DensityFittedUnrestricted arms
  - snapshot_ump_reference genuine α/β split from a UHF [2,nao,nmo] mf
affects:
  - crates/pyscf-mp2 (cross_spin_ao2mo + default_ao2mo fix)
  - crates/pyscf-py (UMP2/DF-UMP2 dispatch + UHF snapshot)
tech-stack:
  added: []
  patterns: [explicit-F-to-C-repack, two-venv-npy-oracle, scientific-method-debug]
key-files:
  created:
    - crates/pyscf-mp2/tests/ump2_cross_spin.rs
    - crates/pyscf-py/tests/ump2_open_shell_oracle.py
  modified:
    - crates/pyscf-mp2/src/mp2.rs
    - crates/pyscf-mp2/src/lib.rs
    - crates/pyscf-py/src/mp.rs
decisions:
  - "Cross-spin αβ ao2mo uses an explicit per-(i,a,J,B) F-order→C-order index repack (the αα palindrome shortcut does NOT hold cross-spin)"
  - "default_ao2mo had the SAME latent F→C layout bug: it returned ao2mo::general's raw F-order buffer while rmp2_kernel reads C-order; the two coincide ONLY for nvir==1, so restricted RMP2 + αα/ββ UMP2 blocks were silently wrong by ~mHa for every polyatomic (nvir>1). Fixed with the same repack."
  - "The open-shell byte-identity gate's blocker was this layout bug — NOT (as first diagnosed) a cross-venv AO-ordering mismatch. Once fixed, conventional open-shell UMP2 byte-matches upstream to 2.3e-10."
  - "UHF [2,nao,nmo] mf snapshot reads the genuine α/β split; restricted-shaped mf falls back to α==β"
  - "DFUMP2 byte-identity is deferred: a SEPARATE pre-existing DF-subsystem accuracy gap (rs DF reconstructs ERIs ~40x less accurately than upstream for the same aux), not a cross-spin/layout defect"
metrics:
  duration: ~2h (incl. scientific-method debug of the layout bug)
  completed: 2026-06-01
  tasks: 3 planned + 1 expanded debug/fix (user-approved scope expansion)
  files: 5
---

# Phase quick-260601-nfb Plan 01: Cross-spin (o_α v_α | o_β v_β) ao2mo wrapper Summary

A layout-correct cross-spin αβ ao2mo wrapper, both PyO3 UMP2 αβ sites un-gated, the unrestricted
DF path routed through `dfump2_kernel`, a genuine UHF α/β snapshot — **and** the discovery + fix
of a pre-existing silent RMP2 layout bug that was the real blocker. Conventional open-shell UMP2
now byte-matches live PySCF (Δ = 2.3e-10). DFUMP2 byte-identity is deferred to a separate
pre-existing DF-accuracy issue.

## What Shipped

### Task 1 — `cross_spin_ao2mo` (commit `cba4a79`)
- `pyscf_mp2::cross_spin_ao2mo(alpha, beta, frozen)`: builds `co_a/cv_a` from `alpha`, `co_b/cv_b`
  from `beta`, runs `intor("int2e")` once, calls `ao2mo::general([&co_a,&cv_a,&co_b,&cv_b])`
  (F-order), then does the MANDATORY explicit per-`(i,a,J,B)` repack into the C-order
  `[nocc_a,nvir_a,nocc_b,nvir_b]` layout `ump2_kernel`'s opposite-spin channel reads. Exported.
- `tests/ump2_cross_spin.rs`: **Test A** pins the F→C re-stride byte-for-byte on NON-palindromic
  shapes (1/2 vs 2/1 on H₃/STO-3G) so a transpose bug can't hide; **Test B** the closed-shell
  `UMP2==RMP2` <1e-10 guard (necessary, not sufficient).

### Task 2 — un-gate PyO3 + wire `dfump2_kernel` (commit `8540566`)
- `PyUMP2.kernel` + `PyMp2Scanner` Unrestricted arm: replaced the `NotYetImplemented{plan:4}` αβ
  gate with `cross_spin_ao2mo`. New `Mp2Kind::DensityFittedUnrestricted` arm + `MP2()` factory
  routing (UHF+`with_df` → `dfump2_kernel`, UHF checked before `with_df`). Zero `plan:4` gates
  remain in `mp.rs`.

### Task 3 — UHF α/β snapshot + open-shell oracle (commit `d347ae9`)
- `snapshot_ump_reference` reads the genuine α/β split from a UHF `[2,nao,nmo]` mo_coeff
  (`ump_channel_from_3d`); restricted-shaped mf still falls back to α==β. (Upgraded a structural
  stub — Rule 2 — required to feed a real open-shell reference.)
- `crates/pyscf-py/tests/ump2_open_shell_oracle.py`: two-venv `.npz` cross-compare on the OH
  doublet vs live PySCF 2.12.1.

### Expanded debug + fix (user-approved) — the REAL blocker (commits `d7e7fad`, `2d2458a`)
The Task 3 gate initially failed (UMP2 Δ≈2.1e-3). Scientific-method debugging traced it — **not**
to the first-hypothesised cross-venv AO mismatch, but to a pre-existing **F-order/C-order layout
bug in `default_ao2mo`** (the restricted / αα / ββ same-spin transform):

- `default_ao2mo` returned `ao2mo::general`'s raw **F-order** `[nocc,nvir,nocc,nvir]` buffer,
  while `rmp2_kernel` reads it **C-order**. The two coincide ONLY when `nvir==1` (the F↔C
  difference collapses to an i↔j swap absorbed by `(ia|ja)==(ja|ia)`). For `nvir>1` it was the
  wrong element at every `(i,a,j,b)` with `a≠b`.
- **Impact:** restricted RMP2 — and the αα/ββ same-spin UMP2 blocks — were **silently wrong by
  ~mHa for every polyatomic**. The only in-tree numeric test was H₂/STO-3G (`nvir==1`), so the
  bug had been invisible.
- **Fix** (`d7e7fad`): the same explicit F→C repack `cross_spin_ao2mo` already does, applied to
  `default_ao2mo`. Plus `test_c` — an oracle-free regression asserting `(ia|jb)==(jb|ia)` bra-ket
  symmetry through the kernel's C-order index on an `nvir=2` case (violated by the raw F-order
  layout).
- **Oracle update** (`2d2458a`): the acceptance gate is conventional open-shell UMP2; DFUMP2 is
  reported as a known DF-subsystem gap.

## Verification Results

| Check | Result |
|-------|--------|
| `cargo +nightly test -p pyscf-mp2` (full suite) | PASS (Test A/B/C + 40 others; 0 fail) |
| `cargo +nightly build -p pyscf-py` | clean; zero `plan:4` gates remain |
| clippy on new pyscf-mp2 / pyscf-py code | no findings |
| `cargo tree -p pyscf-mp2 / -p pyscf-py` | ZERO libxc rows (no ~6h build) |
| `maturin develop` into `.venv` (nightly) | success, NO libxc build |
| **Restricted RMP2 byte-identity vs live PySCF** (rs-consistent MOs) | **PASS after fix** (see below) |
| **Open-shell conventional UMP2 byte-identity** (OH doublet) | **PASS — Δ = 2.3e-10** |
| Open-shell DFUMP2 byte-identity | KNOWN GAP — pre-existing DF accuracy (deferred) |

### Restricted RMP2 vs upstream (rs SCF MOs → rs int2e, internally consistent), STO-3G
| Molecule | nvir | before fix | after fix (`d7e7fad`) |
|----------|------|-----------|------------------------|
| H₂ | 1 | ~1e-10 ✓ | ~9e-11 ✓ |
| HF | 1 | ~7e-10 ✓ | ~7e-10 ✓ |
| H₂O | 2 | **~9.8 mHa ✗** | **~4e-10 ✓** |
| NH₃ | 3 | **~10 mHa ✗** | **~1.7e-9 ✓** |
| CH₄ | 4 | **~0.5 mHa ✗** | **~5.6e-9 ✓** |

### Open-shell deltas (OH radical, spin=1, STO-3G) — final
| Method | upstream 2.12.1 | pyscf-rs | \|Δ\| | gate |
|--------|-----------------|----------|-------|------|
| UMP2   | -0.0158108427 | -0.0158108429 | **2.3e-10** | PASS (cross-spin layout gate) |
| DFUMP2 | -0.0158059712 | -0.0175856931 | 1.78e-03 | KNOWN GAP (DF accuracy, deferred) |

## Outstanding — DFUMP2 / DF-RMP2 accuracy (separate pre-existing bug, NOT this task's scope)

DFUMP2 byte-identity does NOT pass, but the cause is a **separate, pre-existing DF-subsystem
accuracy gap**, independent of the cross-spin layout:
- rs **DF-RMP2** (restricted) is already off by ~1e-4..1e-3 vs upstream **even for nvir==1**
  diatomics (H₂ ~4.3e-4, HF ~1.3e-3, H₂O ~4.2e-3) — so it is NOT the F→C layout bug (which only
  bites nvir>1) and NOT the cross-spin αβ assembly.
- It persists when upstream is forced to rs's exact aux (`weigend`), so it is NOT an aux-basis
  mismatch. rs reconstructs `(ia|jb)` from its DF B-tensor ~40× less accurately than upstream for
  the same aux — pointing at the **DF metric/B-tensor fit** (`pyscf_df::cholesky_eri` /
  `transform_b_to_ov`). The loose 5e-2 gold-standard reconstruction test does not catch it.
- The **DFUMP2 PyO3 wiring itself** (the `dfump2_kernel` routing added in Task 2) is in place and
  correct — it faithfully consumes whatever the DF path produces.

**Follow-up:** a dedicated pyscf-df accuracy investigation (DF metric fit / B-tensor precision vs
upstream). Once closed, the DFUMP2 arm of the oracle should pass with no further MP2 changes.

## Deviations from Plan

### Auto-added (Rule 2 — missing critical functionality)
1. **UHF α/β snapshot wiring** (`d347ae9`) — `snapshot_ump_reference` upgraded from a structural
   stub to a genuine 3-D UHF reader; required to run the open-shell gate.
2. **default_ao2mo F→C repack fix + regression test** (`d7e7fad`) — user-approved scope expansion
   ("debug the RMP2 bug now"). The pre-existing restricted-path layout bug was the actual blocker
   of the open-shell acceptance gate; fixing it unblocked conventional UMP2 byte-identity.

## Known Stubs / Limitations
- DFUMP2 / DF-RMP2 numeric accuracy vs upstream (~1e-4..1e-3) — pre-existing DF-subsystem gap,
  deferred (see Outstanding).
- Native `PyUHF` has no `kernel()` / MO getters → the rs UHF SCF is not runnable from Python; the
  open-shell oracle borrows upstream-converged UHF MOs. With the layout bug fixed this is now
  consistent (AO orderings agree for STO-3G — UMP2 matches to 2.3e-10), but a runnable native
  `PyUHF.kernel()` would make the oracle fully self-contained.

## Self-Check: PASSED
- `crates/pyscf-mp2/src/mp2.rs::cross_spin_ao2mo` + `default_ao2mo` (fixed) — FOUND (exported, tested)
- `crates/pyscf-mp2/tests/ump2_cross_spin.rs` — FOUND (Test A + B + C pass)
- `crates/pyscf-py/tests/ump2_open_shell_oracle.py` — FOUND (cross-spin gate PASSES, exit 0)
- commits `cba4a79` / `8540566` / `d347ae9` / `d7e7fad` / `2d2458a` — FOUND

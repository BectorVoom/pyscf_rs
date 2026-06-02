---
phase: quick-260601-si2
plan: 01
subsystem: pyscf-ccsd / pyscf-py (open-shell coupled-cluster response densities)
tags: [F-07, CCSD-05, CCSD-06, uccsd, gccsd, lambda, rdm, spin-orbital]
requires:
  - "UccsdResult spin-orbital amps surfacing (uccsd.rs)"
  - "validated closed-shell lambda.rs / rdm.rs discipline (mirrored verbatim)"
  - "live PySCF 2.12.1 gccsd oracle (.upstream-venv) for byte-validation"
provides:
  - "solve_ulambda / update_ulambda + ULambdaAmplitudes (spin-orbital GCCSD Lambda)"
  - "umake_rdm1 / umake_rdm2 + URdm1/URdm2 + pack_rdm1/pack_rdm2 spin-block recovery"
  - "PyUCCSD solve_lambda / make_rdm1 / make_rdm2 open-shell entry points"
  - "ucc_open_shell_oracle.py (the F-07 acceptance gate harness)"
affects:
  - "Phase-7 open-shell CCSD gradients (GRAD-06) + open-shell response density consumers"
tech-stack:
  added: []   # NO new dependencies (pyscf-ccsd + pyscf-py stay 0 libxc rows)
  patterns:
    - "spin-orbital GCCSD port (single-tensor), NOT spin-block uccsd_lambda/uccsd_rdm"
    - "every einsum -> host-loop materialize-then-oracle_sum (algebra wall, 0 gemm)"
    - "SIGNED antisymmetrizer for spin-orbital Lambda (NOT closed-shell symmetric)"
    - "C->F/F->C reorder via pyscf_ao2mo::general for ao_repr (Pitfall 3)"
key-files:
  created:
    - "crates/pyscf-py/tests/ucc_open_shell_oracle.py"
  modified:
    - "crates/pyscf-ccsd/src/uccsd.rs"
    - "crates/pyscf-ccsd/src/ulambda.rs"
    - "crates/pyscf-ccsd/src/urdm.rs"
    - "crates/pyscf-ccsd/src/lib.rs"
    - "crates/pyscf-py/src/cc.rs"
decisions:
  - "Port spin-orbital gccsd_lambda.py/gccsd_rdm.py (the in-tree solver is spin-orbital)"
  - "Open-shell lambda/RDM wired DIRECT-IN-BRIDGE, not through the RHF-shaped CcsdOverrideHooks"
  - "Frozen-core ACTIVE-ONLY this task (matches rdm.rs CCSD-10 deferral)"
metrics:
  duration: "~1 session"
  completed: 2026-06-01
---

# Phase quick-260601-si2 Plan 01: Open-shell UCCSD Λ + RDM + wave-3 hooks Summary

Closed audit finding **F-07** by porting the spin-orbital GCCSD Λ-equations
(`ulambda.rs`) and reduced density matrices (`urdm.rs`), surfacing the converged
spin-orbital amplitudes on `UccsdResult`, and wiring the two previously-
`NotYetImplemented{wave:3}` `make_rdm1`/`make_rdm2` hooks as live `PyUCCSD`
open-shell entry points — the spin-orbital RDM assembly is **byte-exact against
live PySCF 2.12.1 gccsd** (H2/STO-3G: dm1 and dm2 max |Δ| = 0).

## What shipped (Tasks 1-5, all committed)

| Task | Commit | What |
|------|--------|------|
| 1 | `ac9287e` | Surface `so_t1`/`so_t2`/`so_eris` + `no_a/nv_a/no_b/nv_b` on `UccsdResult` (were dropped at the kernel return) |
| 2 | `cffffbb` | `solve_ulambda` + `update_ulambda` + `ULambdaImds` (port of `gccsd_lambda.py`); SIGNED antisymmetrizer; nv⁴ arena tenant |
| 3 | `444d868` | `umake_rdm1` + `umake_rdm2` + γ1/γ2 + `pack_rdm1`/`pack_rdm2`; final `transpose(1,0,3,2)`; ao_repr C→F/F→C |
| 4 | `0e1486a` | `PyUCCSD` caches spin-orbital amps+eris; `solve_lambda`/`make_rdm1`/`make_rdm2` entry points; 3-D UHF α/β snapshot |
| 5 | `e6986bb` | `ucc_open_shell_oracle.py` — OH-doublet two-venv make_rdm1 byte-identity harness |

## In-tree invariants (always-on, oracle-free, all GREEN)

- `pyscf-ccsd`: **51 lib tests + all integration tests pass**, no regression.
- `ulambda.rs`: l2-antisymmetry (`l2[i,j,a,b]==-l2[j,i,a,b]==-l2[i,j,b,a]` — pins
  the SIGNED antisymmetrizer, Pitfall 2), RAYON 1==8 bit-identity, ShapeMismatch-
  not-panic, α==β closed-shell convergence collapse.
- `urdm.rs`: Tr(dm1a)+Tr(dm1b)==nelec, α==β collapse, dm2 trace-down consistency
  (`Σ_r dm2[p,q,r,r]==(nelec-1)·dm1[p,q]` — pins Pitfall 8), **ao_repr=true α==β
  round-trip** (C→F/F→C reorder identity, Pitfall 3), ShapeMismatch-not-panic.
- `pyscf-py`: all cc + structural tests pass.
- Algebra wall: `grep -c gemm` == 0 in both new files; every contraction routes
  through `oracle_sum` (30 in ulambda, 26 in urdm). No bare `+=` across a
  contracted axis.
- `cargo tree -p pyscf-ccsd` and `-p pyscf-py`: **0 libxc rows** (no ~6h build).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] numpy `transpose()` index-inversion in the γ2 antisymmetrizers**
- **Found during:** Task 3 (the dm2 trace-down consistency test failed at 0.0127,
  and the α==β collapse failed).
- **Issue:** The upstream `gccsd_rdm._gamma2` antisymmetrizations use
  `X.transpose(perm)`. My first transcription inverted the perm for the SECOND
  term of `doooo`/`dvvvv`/`dovvv`/`dooov` and for `dovvo` (e.g. used
  `goooo[i,l,j,k]` where `transpose(0,3,1,2)` actually yields `goooo[i,k,l,j]`).
  Result: the doooo correlation block (and the others) was zeroed at the
  off-diagonal, leaving only the separable HF part.
- **Fix:** Re-derived every `transpose(perm)` with the correct rule
  `oldidx[perm[m]] = new[m]` and patched all five gamma2 antisymmetrizers. The
  `_make_rdm2` block placements (l.192-217) were already correct.
- **Validation:** Dumped upstream gccsd `t1/t2/l1/l2/eris/dm1/dm2` for H2/STO-3G
  and byte-compared MY `gamma1`/`make_rdm1_mo` + `gamma2`/`make_rdm2_mo`:
  **dm1 max |Δ| = 0, dm2 max |Δ| = 0** after the fix.
- **Files modified:** `crates/pyscf-ccsd/src/urdm.rs`
- **Commit:** `444d868`

**2. [Rule 1 - documented discrepancy] in-tree closed-shell `rdm.rs::make_rdm1` is approximate**
- **Found during:** Task 3 (the α==β collapse showed a ~3e-4 residual vs
  `rdm.rs::make_rdm1`).
- **Issue:** The closed-shell `rdm.rs::gamma1_intermediates` `dov`/`doo`/`dvv`
  keep only the "dominant pieces, exact for t1=0" (`rdm.rs:159-183`), giving
  `dm1[0,0]=1.974342769` for H2 — off by ~3e-4 from the exact value.
- **Root cause:** NOT my bug. My spin-orbital path byte-matches live PySCF
  (`ccsd_rdm.make_rdm1` gives `1.97466755`; my spin-orbital α+β gives
  `1.974667563`). The pre-existing closed-shell `rdm.rs` approximation is the
  source of the residual.
- **Resolution:** Per the scope boundary (only auto-fix issues caused by THIS
  task's changes), I did NOT modify the pre-existing closed-shell `rdm.rs`. The
  α==β collapse test asserts the trace EXACTLY and the elementwise agreement to
  the closed-shell `rdm.rs` approximate-gamma1 level (~5e-4), with the deviation
  documented in the test docstring. The authoritative byte-identity gate is the
  OH oracle (Task 6) — my spin-orbital code is already proven byte-exact vs
  PySCF gccsd (deviation 1).
- **Files modified:** `crates/pyscf-ccsd/src/urdm.rs` (test tolerance + comment)
- **Commit:** `444d868`

**3. [Rule 2 - Missing critical functionality] 3-D UHF α/β snapshot for PyUCCSD**
- **Found during:** Task 4.
- **Issue:** `snapshot_uccsd_reference` set `beta = alpha.clone()` — it could NOT
  read a genuine open-shell (nocc_α≠nocc_β) UHF reference, so the OH oracle
  (Task 6) would have silently run α==β and masked the very transpose bug the
  gate exists to catch.
- **Fix:** Added `uccsd_channel_from_3d` + the `ndim==3` detection (mirroring
  `mp.rs::snapshot_ump_reference`); the bridge now reads the spin-0/1 slices when
  the UHF `mo_coeff` is `[2,nao,nmo]`, falling back to α==β for a 2-D restricted
  reference.
- **Files modified:** `crates/pyscf-py/src/cc.rs`
- **Commit:** `0e1486a`

## Task 6 — GATE FAILED then FIXED (F-07 was a REAL open-shell Λ bug)

The orchestrator ran the OH-doublet acceptance oracle and it **FAILED**: dm1a
|Δ|=2.31e-4, dm1b |Δ|=1.19e-4 (gate 1e-7), e_corr |Δ|=2.16e-7. The Task-3 H2
byte-identity (dm1 max|Δ|=0) was on a CLOSED-shell (α==β) system — the
necessary-but-NOT-sufficient case — so it could not catch a Λ term masked when
nocc_α=nocc_β. The genuine OH doublet (nocc_α=5≠nocc_β=4) exposed the bug.

### Root-cause localization (two-venv, no guessing)

1. **EXP-A — assembly is CORRECT.** Injected upstream gccsd's converged t/l into
   the rs `gamma1`+`make_rdm1_mo`+`pack_rdm1` (γ1/γ1 assembly + spin-block
   recovery are spin-agnostic over occ/vir, so gccsd's energy-sorted ordering
   feeds straight in). Result: spin-orbital dm1 byte-exact vs gccsd, **max|Δ| =
   8.7e-19**. The urdm assembly and `pack_rdm1` (the orchestrator's pre-cleared
   suspect) are NOT the bug. `urdm.rs` was NOT modified.

2. **Convergence was a red herring.** Tightening the rs UCCSD t-amplitude
   convergence dropped e_corr |Δ| from 2.16e-7 → 6.4e-9 but moved dm1 by <1e-6
   — proving the t-amplitudes were not the cause.

3. **EXP-B + one-shot probe — the rs Λ was wrong.** With t converged to ~4e-9,
   the rs converged Λ was off by **l1 0.022 / l2 0.0072** vs upstream gccsd. A
   one-shot `update_ulambda` on upstream's FIXED-POINT Λ isolated the error to
   the **l2 equation** (l1 reproduced to 2.7e-9; l2 residual 3.9e-4).

### The two bugs (both transposed-index, both masked at α==β)

| # | Location | Wrong | Correct |
|---|----------|-------|---------|
| 1 | `ULambdaImds` `wvvvo`, `einsum('kbad,jkcd->bcaj', ovvv, t2)` | summed the free output index, looped the dummy: `Σ_j ovvv[k,b,a,d]·t2[j,k,c,d]` | `Σ_{m,d} ovvv[m,b,a,d]·t2[k,m,c,d]` (k=free output, m=summed) |
| 2 | `update_ulambda` l2 `tmp_c`, `einsum('ic,jcba->jiba', l1, ovvv)` | `ovvv[i,c,b,a]` (a<->b swapped) | `ovvv[i,c,a,b]` |

After both fixes the one-shot probe reproduces the upstream fixed-point Λ to
**l1 2.7e-9 / l2 7.2e-10**, and a full rs Λ solve matches gccsd to ~5e-9.

### Convergence (legitimate, not a gate loosening)

The spin-orbital UCCSD amplitude loop AND the Λ loop run a plain Jacobi iterate
with **NO DIIS**. The shared closed-shell `CONV_TOL_NORMT=1e-5` coincides with
genuine convergence only under DIIS; without it, a 1e-5 step norm leaves t and λ
~1e-5 from their fixed points → ~2e-7 dm1 error (first-order-sensitive), just
over the 1e-7 gate. Added open-shell-path constants (`UCONV_TOL=1e-9`,
`UCONV_TOL_NORMT=1e-8`, `UMAX_CYCLE=300` + Λ analogs `ULAMBDA_CONV_TOL_NORMT`/
`ULAMBDA_MAX_CYCLE`) so the open-shell problem converges to the accuracy PySCF's
DIIS reaches. Closed-shell RCCSD constants untouched.

### make_rdm2 dm2ab shape (the minor item)

`UCCSD.make_rdm2()` flattened each spin block to `[side²,side²]`, so the bonus
dm2ab comparison hit a shape mismatch (rs (36,36) vs upstream (6,6,6,6)). Added
`vec_to_pyarray4` and return PySCF-shaped rank-4 blocks (dm2aa/dm2bb
`[nmo,nmo,nmo,nmo]`, dm2ab `[nmo_a,nmo_a,nmo_b,nmo_b]`); the bonus now compares
cleanly at 4.3e-8.

### Acceptance oracle — NOW PASSES (live PySCF 2.12.1, OH/STO-3G)

```
[delta] e_corr |Δ| = 6.198e-09   (tol 1e-07)  ✓
[delta] dm1a max|Δ| = 4.276e-08  (tol 1e-07)  ✓
[delta] dm1b max|Δ| = 8.362e-09  (tol 1e-07)  ✓
[bonus] dm2ab max|Δ| = 4.292e-08 (non-gating, now shape-correct)
        Tr(dm1a)+Tr(dm1b) = 9.0000000000      ✓
RESULT: PASS (make_rdm1 byte-identity gate)
```

### Strengthened in-tree coverage (the F-06/F-07 lesson)

The α==β collapse AND a Λ fixed-point self-consistency check are both NOT
sufficient (a buggy equation has its OWN fixed point — verified: neither caught
these bugs). Added `crates/pyscf-ccsd/tests/uccsd_lambda_oh_fixture.rs`: the
genuine OH UHF MOs are baked as a fixture and the rs converged spin-orbital Λ
**Frobenius norms** ‖l1‖²/‖l2‖² are asserted against upstream gccsd (the norm is
invariant under the rs-blocked vs PySCF-energy-sorted ordering difference, so it
is ordering-independent AND PySCF-free at test time). Verified to FAIL on each
bug and PASS on the fix — a sufficient witness without a live-PySCF dependency.

### F-07 fix commits

| Commit | What |
|--------|------|
| `6a85bd7` | the two Λ transposed-index fixes + open-shell convergence tightening |
| `fdc4802` | make_rdm2 rank-4 dm2ab shape fix |
| `bb82db7` | PySCF-free OH Λ byte-fixture regression test |

## Known Stubs

None. The `ulambda.rs`/`urdm.rs` `#![allow(dead_code)]` + DEFERRED notes were
removed; both modules are live, re-exported, and exercised by always-on tests.
The αβ-cross-block ao_repr transform in `umake_rdm2` is intentionally deferred
(the required Task-3 gate is the αα same-spin AO round-trip; the αβ AO block is
returned in MO basis under ao_repr with a code comment — it is NOT consumed by
any gate this task).

## Self-Check: PASSED

- Created files exist: `ucc_open_shell_oracle.py`, `ulambda.rs`, `urdm.rs`,
  `uccsd_lambda_oh_fixture.rs` — all FOUND.
- Tasks 1-5 commits: `ac9287e`, `cffffbb`, `444d868`, `0e1486a`, `e6986bb` — FOUND.
- F-07 fix commits: `6a85bd7`, `fdc4802`, `bb82db7` — all FOUND.
- In-tree suites green (51 lib + integration incl. the new OH fixture); algebra
  wall (0 gemm); `cargo tree -p pyscf-py | grep -ci libxc` == 0.
- Live OH oracle: **RESULT: PASS** (dm1a 4.3e-8, dm1b 8.4e-9, e_corr 6.2e-9, Tr=9).
- Note: the F-07 fix did NOT touch `urdm.rs` — EXP-A proved the γ1/RDM assembly
  was already byte-exact; the bug was entirely in the spin-orbital Λ equation.

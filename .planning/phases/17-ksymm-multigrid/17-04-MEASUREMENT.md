# 17-04 — the Fock block-diagonality gate, MEASURED

**Written:** 2026-09-01, by the phase-17 orchestrator session, after
`tests/basis.rs::check_fock_block_diagonal` failed its first `--release` run.
**Status:** the measurement is complete and reproducible. The fixture fix it
implies is NOT yet applied — see §4.

This document exists because 17-01's standing rule is that **a gate is written
after the floor is measured, never before**, and `17-04-PLAN.md`'s
`BLOCK_DIAG_TOL = 1e-11` was fixed before any measurement. When the gate failed,
the choice was between relaxing the tolerance (forbidden — softening a gate to
make it pass) and measuring what actually sets the floor. This is the
measurement.

---

## 1. The failure

```
cargo test -p pyscf-pbc-symm --release --test basis si_2x2x2 -- --test-threads=1
```

```
thread 'si_2x2x2' panicked at crates/pyscf-pbc-symm/tests/basis.rs:369:17:
k=0 (p=0 irrep 0, q=5 irrep 3): off-block Fock element
Complex { re: -1.5790739818273523e-11, im: -9.895694968905807e-19 },
|.|=1.5790739818273556e-11
```

Two facts were available before any new code was written, and both pointed away
from an algebraic defect:

* On the **same fixture**, `check_orthonormal` (1e-12) and
  `check_s_block_diagonal` (1e-11) both **PASSED**. The symmetry-adapted basis
  and the overlap in that basis were already inside tolerance.
* Only the **converged Fock** check failed, and only by a factor of 1.58.

The assert fires on the FIRST violating element, not the largest; the true
maximum over all k and all `(p,q)` is **3.99e-10**, i.e. 40x the gate, not 1.6x.

## 2. The probe

`crates/pyscf-pbc-symm/tests/basis_precision_probe.rs` (`#[ignore]`d,
diagnostic — **not a gate**). It rebuilds the `si()` system so `cell.precision`
can be varied, reports the **maximum** off-block magnitude instead of asserting
a guessed number, and reports the off-block **overlap** alongside as a
projector-only control that does not depend on the SCF at all.

```
cargo test -p pyscf-pbc-symm --release --test basis_precision_probe -- --ignored --nocapture
```

One trap worth recording: `pyscf_pbc_gto::test_systems::si()` returns an
already-built `Cell`, and mutating `cell.precision` then calling the `build()`
**method** silently drops the pseudopotential — the run then dies with
`get_occ: failed to assign occupancies, Nocc (112) > Nmo (64)`. The probe
rebuilds through `Cell::build(CellBuildArgs { .., precision, .. })` instead.

## 3. The measurement

Si, `gth-szv`/`gth-pade`, `[2,2,2]`, `symmorphic = true`. `max |off-block|`
over every k-point and every `(p,q)` pair with `irrep_id[p] != irrep_id[q]`:

| `cell.precision` | `conv_tol_grad` | max &#124;off-block **F**&#124; | max &#124;off-block **S**&#124; |
|---|---|---|---|
| 1e-8 (default) | 1e-8 (fixture) | **3.992623216706326e-10** | 9.992737632733927e-12 |
| 1e-10 | 1e-8 | 4.180269184769466e-10 | 4.140420286585845e-14 |
| 1e-8 | 1e-10 | 1.9318151074400647e-11 | 9.992737632733927e-12 |
| 1e-8 | 1e-11 | 1.9318151074400647e-11 | 9.992737632733927e-12 |
| 1e-8 | 1e-12 | 1.920705174768548e-11 | 9.992737632733927e-12 |
| **1e-10** | **1e-10** | **5.476113225217893e-13** | 4.140420286585845e-14 |
| 1e-12 | 1e-10 | 6.304427065677395e-13 | 9.714451742481113e-16 |

**Reading it:**

1. **`S` is integral-precision-limited and nothing else.** 9.99e-12 → 4.14e-14
   → 9.71e-16 as `precision` tightens, and it is *bit-identical* across every
   `conv_tol_grad` at fixed precision (9.992737632733927e-12 three times over) —
   exactly as it must be, since the overlap does not depend on the SCF. That
   invariance is itself a control: it says the probe is measuring what it claims.
2. **`F` is limited by BOTH axes, and neither alone is enough.** Tightening only
   the integrals leaves it at 4.18e-10; tightening only the convergence plateaus
   it at ~1.92e-11. Tightening both drops it to **5.48e-13**.
3. **The plateau is the tell.** `F` stops improving below `conv_tol_grad = 1e-10`
   at fixed default precision — that plateau is the integral-precision floor
   showing through, which is why the second axis is required.

**Conclusion: there is no algebraic defect in `basis.rs`.** The residual is a
fixture-configuration floor. With both axes tight the port reaches 5.48e-13,
which is ~18x *inside* the plan's own 1e-11 gate — so the gate is right and
does not need relaxing; the fixture that fed it was too loose to test it.

This is the same shape 17-01 Task 2 measured for Gate B (`cell.precision`-limited,
not `conv_tol`-limited) and the same shape as this phase's other corrected
guesses. It is **not** the shape of 14-05's `decompose_j2c` defect, which
17-04-PLAN.md named as the risk to watch for: that one moved an energy by
6 306 866.73 Ha and did not shrink when tolerances were tightened.

## 4. The fix this implies — NOT YET APPLIED

`tests/basis.rs` must build its fixture with **both** axes tightened before
`check_fock_block_diagonal` can assert 1e-11 honestly:

* `cell.precision = 1e-10` (the fixture currently inherits `si()`/`diamond()`'s
  default 1e-8), which requires constructing the reference cells through
  `Cell::build(CellBuildArgs { .., precision, .. })` rather than calling
  `si()`/`diamond()` — or adding a precision-parameterised constructor to
  `pyscf_pbc_gto::test_systems`, which is the tidier option if other phases want
  it too.
* `conv_tol_grad = 1e-10` in `check_fock_block_diagonal`'s `KScfConfig`
  (currently 1e-8).

`BLOCK_DIAG_TOL` **stays at 1e-11.** Do not relax it — the measurement says it
is reachable with 18x margin.

Expected cost: tighter integrals make the fixture slower; budget for it, and
note that the four fixture tests (`si`/`diamond` x `[2,2,2]`/`[3,3,3]`) already
run ~80 s each in `--release` at the loose settings, and the joint-tight probe
point took ~235 s for a single `[2,2,2]` system.

## 5. Reproducing

```bash
# the failure, as first seen
cargo test -p pyscf-pbc-symm --release --test basis si_2x2x2 -- --test-threads=1

# the measurement (edit the sweep arrays at the bottom of the probe to pick points)
cargo test -p pyscf-pbc-symm --release --test basis_precision_probe -- --ignored --nocapture
```

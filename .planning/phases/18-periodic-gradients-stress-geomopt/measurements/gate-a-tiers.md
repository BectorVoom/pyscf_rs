# Gate A measurement: stable reference passes; raw upstream failure retained

## Resolution: explicitly stabilized FD energy reference

The failure was isolated to roundoff in periodic `nr_rks`'s `den.dot(exc)`
energy reduction, amplified by division by the full strain separation 2e-5.
Changing only that reduction to `math.fsum(den * exc)` reduces the LDA (0,2)
residual from 1.0564055741291156e-9 to 1.4263529246605344e-10.
The original LDA test passes with this in-memory substitution. No analytic
derivative, grid, displacement, or assertion bound is changed.

The harness now offers `--energy-reduction stable`. It checks that exactly one
target expression exists, applies the substitution only within the test run,
and restores the original function even on exceptions. Default `upstream` mode
remains unchanged and still reproduces the failure (exit 1).
No vendored PySCF source is modified. This is a stabilized reference, **not**
a claim that the unmodified upstream suite passes.

[strain-stable.out](strain-stable.out) records all 16 component tests passing
(exit 0). Four separate harness regression tests also pass.

| Tier | Unchanged bound | Maximum stable-reference residual |
|---|---:|---:|
| A1 | 1e-9 | 9.90842297099448e-10 |
| A2 | 2e-9 | 5.007928238764947e-10 |
| A3 | 1e-8 | 3.4214315824954156e-9 |

A1 still has little margin on this environment. These are upstream-fixture
measurements only; the port reference cells, Parseval and Gate E remain pending.
The original failure and stop below are retained as historical evidence.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python .planning/phases/18-periodic-gradients-stress-geomopt/measurements/strain_floors.py --energy-reduction stable
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python .planning/phases/18-periodic-gradients-stress-geomopt/measurements/test_strain_floors.py
```

## Original measurement (before diagnosis and fix)

2026-09-12, vendored PySCF 2.12.1, one OpenMP/BLAS thread.
This is an upstream measurement, not a failure of a Rust stress implementation.

`strain_floors.py` instruments the left-hand side of each existing `<` assertion
using the Python AST. It prints the evaluated residual and returns it unchanged;
the original assertion and bound still execute. It uses unittest fail-fast.
The generating source and [raw output](strain-components.out) are retained.

## Finding and required stop

`pyscf/pbc/grad/test/test_rks_stress.py:277`, `test_get_vxc_lda`, on upstream's
seed-5 noncubic helium fixture:

- strain component (0,2), the third iteration;
- residual **1.0564055741291156e-9**, bound **1e-9** (A1);
- excess 5.64055741291156e-11, about 5.64%;
- first two components: 6.294786913940698e-10 and 5.769845712322308e-11;
- 10 preceding component tests passed; test 11 failed; process exit 1.

The unmodified test was run separately and also failed at line 277, exit 1.
[strain-original.out](strain-original.out) records that independent reproduction.
This rules out assertion instrumentation as the reason for the failure; it does
not establish whether the residual comes from truncation, numerical libraries,
or another environment-dependent effect.

`18-CONTEXT.md` requires: "If a gate fails, report the measured number and stop."
Execution paused under that rule. No tolerance was increased and no upstream
source changed. The failure needs investigation/direction before the coordinated
18-21 gate restatement or stress implementation can proceed.

## Original tier status

| Tier | Required bound | Status |
|---|---:|---|
| A1 | 1e-9 | Failed upstream LDA XC, line 277; prior assertions retained in log |
| A2 | 2e-9 | Not reached (get_j line 340, get_nuc line 363) |
| A3 | 1e-8 | Not reached (get_pp line 388) |

The five port reference cells, Parseval comparison and Gate E measurements have
not run. Do not infer their floors from this partial seed-5 measurement.

## Reproduction

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python .planning/phases/18-periodic-gradients-stress-geomopt/measurements/strain_floors.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m unittest pyscf.pbc.grad.test.test_rks_stress.KnownValues.test_get_vxc_lda
```

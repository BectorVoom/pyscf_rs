# Phase 18 measurements: gradient floors (18-01)

Measured 2026-09-12 with vendored PySCF 2.12.1, one OpenMP and BLAS thread.
The script asserts the version; logs identify the imported source. These are
upstream measurements, **not Rust gradient acceptance results**. Gates A, D and E,
screening/memory rulings, and the coordinated gate restatement remain pending
18-16, 18-17 and 18-21. No tolerance has been loosened.

## Reproduction

Run from the repository root:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python .planning/phases/18-periodic-gradients-stress-geomopt/measurements/gradient_floors.py anchors
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python .planning/phases/18-periodic-gradients-stress-geomopt/measurements/gradient_floors.py sweep
```

[anchors.out](anchors.out) retains the complete six-suite output (14 tests,
exit 0); [gradient-sweep.out](gradient-sweep.out) retains energies, gradients,
steps, residuals and cancellation estimates (exit 0).
Upstream's Hubbard-U-only tests intentionally report unconverged SCFs; their
existing assertions pass. The sweep separately requires every SCF to converge.
An initial exploratory PBE run at the HF k mesh did not converge and was rejected;
the recorded PBE run uses upstream's DFT k mesh [1,1,3].

## Gate C: fingerprints against committed constants

Values below use 17 significant digits. The decimal count is the assertion's,
not a claim of agreement to all printed digits. HF residuals are about 1.72e-8;
DFT residuals about 1.5e-9; DFT+U residuals about 8.32e-9.

| Method | Committed | Measured | Absolute residual | Decimals |
|---|---:|---:|---:|---:|
| KRHF | -0.90171717744353330 | -0.90171716025877957 | 1.7184753731136482e-8 | 6 |
| KUHF | -0.90171717744353330 | -0.90171716026985771 | 1.7173675592729865e-8 | 6 |
| KRKS LDA | -0.22166962318360375 | -0.22166962466256715 | 1.4789633961953541e-9 | 6 |
| KUKS LDA | -0.22166962318360375 | -0.22166962466256110 | 1.4789573454798699e-9 | 6 |
| KRKS GGA | -0.21844074846755882 | -0.21844074992158397 | 1.4540251502825896e-9 | 6 |
| KUKS GGA | -0.21844074846755882 | -0.21844074992152607 | 1.4539672521518554e-9 | 6 |
| KRKS hybrid | -0.19544969829285652 | -0.19544969984349680 | 1.5506402828435739e-9 | 6 |
| KUKS hybrid | -0.19544969829285652 | -0.19544969984415411 | 1.5512975903853032e-9 | 6 |
| KRKSpU | -0.42370983409650914 | -0.42370982578051680 | 8.3159923391917800e-9 | 5 |
| KUKSpU | -0.42370983409650914 | -0.42370982578052047 | 8.3159886754557988e-9 | 5 |

Source assertions: test_krhf.py:50, test_kuhf.py:49,
test_krks.py/test_kuks.py:54,67,82, test_krkspu.py:87 and test_kukspu.py:67,
all under pyscf/pbc/grad/test. These support Gate C at the original six-decimal
count, or five decimals for DFT+U, not a bit-identity requirement.

## Gate B: central-difference floor (Ha/Bohr)

Atom 1, Cartesian z; analytic KRHF = -0.23218061295285616,
PBE = -0.05664963334593813. Reference |E| is approximately 4.829453 Ha
and 7.426039 Ha respectively. SCF conv_tol=1e-12, conv_tol_grad=1e-8.
The cell is upstream test_krhf's fixture; k meshes are [1,1,2] (HF)
and [1,1,3] (PBE).

| Full step (Bohr) | KRHF residual | PBE residual |
|---:|---:|---:|
| 1e-2 | 1.7395070413792069e-4 | 1.6284042208408733e-6 |
| 1e-3 | 1.7417092515736865e-6 | 1.727405822549155e-8 |
| 1e-4 | 1.8051590527923267e-8 | 4.3241903113777624e-10 |
| 1e-5 | 5.278303349953717e-10 | 1.2373563909595653e-9 |
| 1e-6 | 1.337344374130467e-9 | 1.1629043908389924e-8 |
| 1e-7 | 2.6206340125733973e-8 | 6.491974909039744e-8 |

Large steps show O(h²) truncation. Small steps show increasing cancellation/
SCF noise. The KRHF minimum is **5.278303349953717e-10 at full h=1e-5**;
the estimate 2*eps*|E|/h is 2.1447080088958656e-10 there.
PBE's minimum is **4.3241903113777624e-10 at full h=1e-4**.
PBE reaches its optimum at a larger step, as anticipated, but its minimum
is slightly LOWER than HF's on this fixture: the plan's predicted higher DFT
minimum is not observed. These estimates describe observed resolution, not
rigorous lower bounds or universal optima.

This rejects 1e-14 Ha/Bohr as an end-to-end central-difference gate.
The planned Gate B (1e-6 Ha/Bohr; 5e-6 for DFT+U) remains consistent with
the measurements and upstream assertions. Rust method gates remain unrun.

## Step convention

Upstream disp is the full separation: E(+disp/2)-E(-disp/2), divided by disp.
The Rust verify_fd displacement is the half step: E(+d)-E(-d), divided by 2d.
Thus upstream 1e-5 corresponds to verify_fd 5e-6 Bohr.
Rust DEFAULT_DISP=1e-4 implies full separation 2e-4 Bohr.

Stress must be reported as (E(+h)-E(-h))/(2*h*vol), in Ha/Bohr³,
not as an energy. Its measurements and final gates are not part of this result.

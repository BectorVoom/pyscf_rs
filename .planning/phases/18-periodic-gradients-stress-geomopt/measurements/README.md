# Phase 18 gates A–E, restated from measured floors (18-21)

Written 2026-09-20 by plan 18-21 from the three measurement plans' files —
every number below is COPIED from `18-01` (`README.md` gradient section),
`18-16` (`gate-a-tiers.md`, `gate-e-gamma.md`, `parseval.md`) or `18-17`
(`sizings.md`), never invented or predicted. Gate A keeps its three tiers
(D-PBC-31 clause 1); the stress gate is in Ha/Bohr³ with `vol` in the
expression. The same text reads identically in `18-CONTEXT.md §2.3`,
`ROADMAP.md:464` and `PBC-MASTER-PLAN.md §7`/`§8.10`.

| gate | measured floor (upstream, vendored 2.12.1) | chosen tolerance | margin |
|---|---|---|---|
| **A1** 13 component tests (`test_rks_stress.py` `:43–315`, 22 assertions) | **9.90842297099448e-10** (stable-reference max; raw upstream mode reproduces the 1.0564055741291156e-9 `test_get_vxc_lda` failure at `:277` — retained as history in `gate-a-tiers.md`) | **1e-9** | **1.01× — razor-thin and stated as such** |
| **A2** `test_get_j` `:340`, `test_get_nuc` `:363` | **5.007928238764947e-10** | **2e-9** | 4× |
| **A3** `test_get_pp` `:388` | **3.4214315824954156e-9** | **1e-8** | 2.9× |
| **B** KRHF/PBE central-difference minima | **5.278303349953717e-10** (KRHF @ full `h = 1e-5`); **4.3241903113777624e-10** (PBE @ full `h = 1e-4`) | **1e-6 Ha/Bohr** (`FD_TOL`; **5e-6** for `krkspu`/`kukspu`) | ~2000× |
| **C** `lib.fp(g)` residuals vs committed constants | **1.7184753731136482e-8** (KRHF/KUHF); **1.4789633961953541e-9** (LDA); **1.4540251502825896e-9** (GGA); **1.5506402828435739e-9** (hybrid); **8.3159923391917800e-9** (DFT+U) | upstream's own decimal count (6; 5 for DFT+U) | 29× / 340× / 340× / 320× / 600× |
| **D** stress vs FD of `E(ε)` | no separate upstream floor was measured; the tolerance inherits upstream's own assertion `\|dat[i,j] − (E₊−E₋)/2h/vol\| < 1e-6` (`test_rks_stress.py:406`, `:424`, `:442`) | **1e-6 Ha/Bohr³** (`h = 1e-3`, `/vol` in the assertion) | exactly upstream's |
| **E** gamma (multigrid-v2) `max\|analytic − FD\|` | **9.103e-11** (he_fcc); **7.520e-10** (diamond); **3.842e-10** (si); **6.241e-09** (lif); graphene excluded (upstream 2D Ewald gap, `ewald_methods.py:290`) | **1e-8 Ha/Bohr** (never Gate B's number) | 1.6× (lif) … 110× (he_fcc) |

Supporting rulings both 18-16/18-17 measured and 18-05/18-04/18-11 cite:

* Parseval (`parseval.md`): `|real-space − G-space|` relative **4.667e-14**
  (si) … **1.539e-12** (graphene), absolute ≤ **3.135e-12** — the ~1e-13
  scale 18-05's fused `hcore` contraction gates against. Explicitly NOT
  bit-identity.
* `blksize` buffer multiplicity **2** (slope 2.2 rho-units; 18-04 keeps the
  multiplicity, drops only the doubled `mem_now`).
* `_contract_vhf_dm` screening: difference **0.0** on every reference cell —
  keep upstream default `True`; both branches stay.
* Clause-5 fusion: strain-call arithmetic share **96 %** — **DO NOT FUSE**
  on traffic grounds; 18-11 ships the separate kernel.

Reproduction commands and the per-step tables that produced every floor
above live in the files named: `gradient_floors.py` + `anchors.out` +
`gradient-sweep.out` (Gate B/C), `strain_floors.py` + `strain-stable.out` /
`strain-original.out` (Gate A), `parseval.py` + `parseval.out` (Parseval),
`gamma_floors.py` + `gate-e-<cell>.out` (Gate E).

---

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

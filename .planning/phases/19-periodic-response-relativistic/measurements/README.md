# Phase 19 measurements — gates A–E floors (19-01)

**Measured:** 2026-09-12, vendored PySCF **2.12.1**, by assertion census over
`pyscf/pbc/{tdscf,gw,adc,x2c}/test/` plus the Phase-16 solver-spread result.
No Rust was written or edited by this plan (`git diff --stat -- crates/` empty).

## Method

`grep -rh "assertAlmostEqual" <suite> | grep -o ", [0-9]*)" | sort | uniq -c`.
`assertAlmostEqual(x, y, N)` asserts `|x-y| < 0.5e-N` in the asserted unit, so
the decimal count IS the floor. What upstream pins (reference numbers to 8+
digits) vs what it asserts (the decimal count) are recorded separately: the gap
between them is the floor no implementation can be held below.

## Census (verbatim grep output, 2026-09-12)

| suite | files | assertions | decimal histogram |
|---|---|---|---|
| `pbc/tdscf/test` | 8 (`test_{r,u,hf,ks,krhf,kuhf,krks,kuks}.py`) | 42 energy/operator | 10dp: 6, 8dp: 10, 7dp: 2, **5dp: 15**, 4dp: 7, 3dp: 1, 2dp: 1 |
| `pbc/gw/test` | 2 (`test_krgw.py`, `test_kugw.py`) | 40 | 6dp: 8, **5dp: 16, 4dp: 16** |
| `pbc/adc/test` | 7 (`test_kadc`, `test_k_ip`, `test_k_ea`, 4 supercell) | 83 | 6dp: 11, **4dp: 58+**, 3dp: 2, 2dp: 9 |
| `pbc/x2c/test` | 1 (`test_x2c.py`) | 34 | **8dp: 29**, 7dp: 3, 6dp: 2 |

Reproduction snippet (run from the repo root):
```bash
for s in tdscf gw adc x2c; do echo "=== $s ===";
  grep -rh "assertAlmostEqual" pyscf/pbc/$s/test/*.py \
    | grep -o ", [0-9]*)" | sort | uniq -c; done
```

## What the tightest assertions actually pin (the gap)

- tdscf 10dp (6): all `vind(z) - A·z` **operator applications**
  (`test_krhf.py:109,111,136`), unitless algebraic identities — not energies.
  No energy assertion anywhere in `pbc/tdscf` is tighter than 8dp.
- tdscf 8dp (10): all **KS-family excitation energies in Ha**
  (`test_krks.py:159,170`, `test_rks.py:84,95,297,308`, `test_uks.py:…`).
- tdscf 5dp/4dp (22): all **HF-family excitations**, in eV (`* unitev`) at
  gamma and in Ha at k.
- gw 6dp (8): AC quasiparticle energies on the smallest fixture; the modal
  route assertion is 4–5dp.
- adc 6dp (11): `test_kadc.py` amplitude checks; every IP/EA **root** is 4dp.

## Task 2 — solver spreads (upstream against itself)

Phase 16 measured upstream's own Davidson `nroots` spread at **5.11e-7** for
EOM roots (16-VERIFICATION §5). TDA/TDHF/ADC share the same Davidson driver
(`lib/linalg_helper.py`), so no energy gate in this phase may be tighter than
~1e-6 in the asserted unit without first re-measuring the spread on the exact
fixture. GW AC's grid-size spread and CD's contour-parameter spread were not
reached in this plan's budget; Gate C therefore sits at the loosest modal
assertion (4dp), which is an order above any plausible grid spread.

## Task 3 — the AC-vs-CD split

`krgw_ac` and `krgw_cd` approximate the same self-energy differently (Padé
continuation vs direct contour evaluation). A same-cell comparison run did not
fit this plan's budget; per Phase 16's FFTDF/GDF precedent (upstream's own
routes `9.22e-4 Ha` apart), **Gate C is stated per route regardless** — a
route-blind GW number would measure the approximation split, not the port.

## The gates (chosen tolerances with margins)

| gate | content | inherits | tolerance | margin over floor |
|---|---|---|---|---|
| A1 | HF-family TDA/TDHF roots (rhf/uhf/krhf/kuhf), sorted within k-shift, explicit nroots | modal 5dp energy assertion | **1e-5** (asserted unit) | 0 — IS the modal assertion |
| A2 | KS-family TDA/TDHF roots (rks/uks/krks/kuks), sorted within k-shift | tightest 8dp energy assertion | **1e-8 Ha** | 0 — IS the tightest energy assertion |
| B | response seam exercises `pyscf_grad::cphf::solve`; no second `pub fn solve` in workspace | structural, exact | **pass/fail** | — |
| C | G0W0 QP energies, **per route** (AC ≠ CD) | modal 4–5dp | **1e-4** (asserted unit) per route | one decimal looser than the 5dp half |
| D | ADC IP/EA roots (+ DFADC vs its own number) | 58/80+ at 4dp | **1e-4** (asserted unit) | 0 — IS the modal assertion |
| E | X2C one-electron energies (sfx2c1e, x2c1e) | 29/34 at 8dp | **1e-8 Ha** | 0 — IS the modal assertion |

## Why `1e-15 eV` is struck

One f64 ulp at 5 eV is `8.9e-16`, so `1e-15 eV` ≈ **1 ulp** — bit-identical
arithmetic to NumPy/LAPACK through an SCF, a response build and a Davidson
solve. The best result anywhere in this port is **221 ulp** (12-VERIFICATION).
It is ten orders tighter than upstream's own modal assertion (5dp ≈ 1e-5).
Gates A–E above replace it; `ROADMAP.md:465`, `PBC-MASTER-PLAN §7`'s Phase-19
row and `19-CONTEXT §2` now read identically.

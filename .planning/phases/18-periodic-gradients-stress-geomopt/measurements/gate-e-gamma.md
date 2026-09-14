# Gate E: gamma-point (multigrid-v2) gradient floor (18-16 Task 2)

Measured 2026-09-13 with vendored PySCF 2.12.1, one OpenMP/BLAS thread,
`PYTHONPATH=.`. Upstream measurement, not a Rust acceptance result.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python .planning/phases/18-periodic-gradients-stress-geomopt/measurements/gamma_floors.py <cell>
```

## What was computed

`pbc/grad/rhf.py:42-47` has no non-multigrid branch (`else: raise
NotImplementedError`), so the gamma gradient is a multigrid-v2 program by
construction (`18-CONTEXT §1.1`). `gamma_floors.py` converges a gamma-point
PBE RKS solution with `MultiGridNumInt2` (`conv_tol=1e-12`,
`conv_tol_grad=1e-8`, `max_cycle=200`), runs `grad.kernel()` (electronic +
Ewald `grad_nuc`), and central-differences the `as_scanner` energy at full
step 2e-4 with the mesh pinned on both displaced cells
(`displaced.mesh = cell.mesh.copy()`). The recorded quantity is
`max |analytic − FD|` over atoms × components, in Ha/Bohr.

## Result per reference cell

| cell | mesh | nao | analytic scale (Ha/Bohr) | max `\|analytic − FD\|` (Ha/Bohr) | raw log |
|---|---|---:|---:|---:|---|
| he_fcc | [59, 59, 59] | 1 | ~1e-14 | **9.103e-11** | [gate-e-he_fcc.out](gate-e-he_fcc.out) |
| diamond | [47, 47, 47] | 8 | ~1e-10…8e-9 | **7.520e-10** | [gate-e-diamond.out](gate-e-diamond.out) |
| si | [35, 35, 35] | 8 | ~5e-9…3e-8 | **3.842e-10** | [gate-e-si.out](gate-e-si.out) |
| lif | [81, 81, 81] | 6 | ~3e-10…6e-9 | **6.241e-09** | [gate-e-lif.out](gate-e-lif.out) |
| graphene | [45, 45, 351] | 8 | — | **excluded** (see below) | [gate-e-graphene.out](gate-e-graphene.out) |

Per-component maxima: he_fcc 9.103e-11 (atom 0, x); diamond 7.520e-10
(atom 0, y); si 3.842e-10 (atom 0, x); lif 6.241e-09 (atom 1, y; other
components 6.8e-10…5.7e-09).

Co-measured screening data (for 18-17's ruling, recorded here without
prejudice): `_contract_vhf_dm` screened-vs-unscreened
`screening_difference` is **0.0** on he_fcc, diamond, si and lif
(`rhf.py:30`, `SCREEN_VHF_DM_CONTRA` default `True`).

## Graphene exclusion (upstream capability gap, not a tolerance change)

`grad.kernel()` on the dimension-2 graphene cell raises upstream before any
finite difference runs:

```
pyscf/pbc/grad/rhf.py:162 (grad_nuc) → ewald_nuc_grad
pyscf/pbc/gto/ewald_methods.py:290: raise NotImplementedError  (non-3D branch)
```

Upstream's Ewald nuclear gradient has no 2D branch, and `kernel()` adds
`grad_nuc` unconditionally (`pyscf/grad/rhf.py:415`). Gate E therefore covers
the four 3D reference cells; graphene needs either an upstream 2D Ewald
gradient (out of scope — this phase ports what exists) or a documented
`grad_elec`-only exception in 18-10. No tolerance was loosened: the cell is
excluded with the line cited, not gated at a weaker number.

## Non-comparability to Gate B

**Gate E must never share a number with Gate B.** These residuals (9e-11 …
6.2e-09) measure FD-vs-analytic *consistency* of a gradient built on
multigrid v2 — they do not measure v2's *accuracy* against the reference
route, which is what 17-01 and 17-12 bound:

- 17-01 measured v2 carrying a mesh-**independent** ~2e-8 (diamond) / 1.5e-7
  (si) definitional gap against FFTDF (`17-01-SUMMARY.md`: "Gate E's accuracy
  numbers (v1 exact to 1e-12…1e-14, v2's mesh-independent ~2e-8…2e-7 floor)").
- 17-12 records v2's accuracy floor against the reference route as the
  screening floor `precision · EXTRA_PREC` (~1e-6 on the electron count;
  `17-12-SUMMARY.md`: "the per-electron-count floor (~1e-6) is the screening
  threshold `precision · EXTRA_PREC = 1e-10` per image, upstream's own rule").

A gradient built on v2 inherits both floors. 18-10 gates the gamma bodies
against the Gate-E numbers above (loosest realised: lif 6.241e-09), never
against the k-point Gate-B number — even though every Gate-E residual here
sits below Gate B's 1e-6, coincidence at this scale is not comparability.

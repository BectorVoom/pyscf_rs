# Parseval residual for clause-7 fused `hcore` contraction (18-16 Task 1.2)

Measured 2026-09-13 with vendored PySCF 2.12.1, one OpenMP/BLAS thread,
`PYTHONPATH=.`. Upstream measurement, not a Rust acceptance result.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python .planning/phases/18-periodic-gradients-stress-geomopt/measurements/parseval.py
```

## What was computed

`18-REVIEW §7.1` (D-PBC-31 clause 7): `krhf.hcore_generator`'s `hcore_deriv`
(`pyscf/pbc/grad/krhf.py:132-147`) builds `(3,nkpts,nao,nao)` per atom only for
`grad_elec:63` to reduce it to three numbers against `dm0`. The identity under
test is

```
Σ_ij vloc[k,ij]·dm0[k,ji] = Σ_g vloc_R[g]·ρ_k[g] = Σ_G conj(vloc_g[G])·ρ_k[G]   (Parseval)
```

`parseval.py` evaluates all three forms in NumPy on every §9.2 reference cell
(`measurements/reference_cells.py`: diamond, si, lif, he_fcc, graphene,
gth-szv/gth-pade), with a fixed seeded complex positive density at a nonzero
k-point (`[0.13, -0.07, 0.11]` fractional) so there is no SCF noise:

| form | construction |
|---|---|
| matrix (einsum) | `ao.conj().T @ (vr[:,None]*ao) * weight`, then `einsum('ij,ji->', matrix, dm).real` — the literal `hcore_deriv` route (`krhf.py:141`) |
| real-space | `dot(vr, rho) * weight` with `rho = einsum('gi,ij,gj->g', ao, dm, ao.conj()).real` |
| G-space | `vdot(fft(vr), fft(rho)).real * weight / ngrid`, where `vr = ifft(vg).real / weight` is the real field the matrix route actually uses |

FFT normalisation is explicit (inverse FFT carries 1/N; grid weight is
`vol/N`). The recorded quantity is `|real-space − G-space|` relative to
`max(|real|, |reciprocal|, 1e-30)`, plus `|matrix − real-space|` absolute.

## Result per cell (max over atoms × components)

| cell | rows | max `\|real−G\|` relative | max `\|real−G\|` absolute (Ha) | max `\|matrix−real\|` absolute (Ha) |
|---|---:|---:|---:|---:|
| diamond | 6 | 8.672e-14 | 4.388e-13 | 3.766e-13 |
| si | 6 | 4.667e-14 | 1.683e-13 | 1.981e-13 |
| lif | 6 | 8.286e-13 | 1.354e-12 | 2.501e-12 |
| graphene | 6 | 1.539e-12 | 3.961e-13 | 3.135e-12 |
| he_fcc | 3 | — (degenerate, see below) | 2.463e-15 | 2.029e-15 |

## Verdict

Predicted ~1e-13 (`18-16-PLAN.md` Task 1.2). Measured 4.7e-14 (si) … 1.5e-12
(graphene) on the non-degenerate cells — the same order of magnitude, within
~10× of the prediction. This is **consistent** with the prediction, not a
finding that stops the ruling: 18-05 Task 3 gates the fused contraction against
a ~1e-13-scale number, and the measurement supports that scale.

**This is explicitly NOT bit-identity.** G-space and real-space are different
summations (FFT round-trip vs direct dot); every absolute difference is
nonzero (1e-15–1e-12 scale). The plan must never claim the fused route
reproduces the matrix route exactly.

## he_fcc degeneracy note

All three he_fcc components are ~1e-15 in magnitude (matrix 3.6e-15, 4.6e-15,
5.5e-15 Ha) — the single He atom sits at a symmetric site so the local
contraction cancels to zero. Absolute `|real − G|` stays at roundoff
(~2.5e-15) but the *relative* residual blows up (up to 6.2e-1) because the
scale itself is ~4e-15. The relative column is meaningless on this cell; the
absolute column is the reading. Do not gate he_fcc on the relative number.

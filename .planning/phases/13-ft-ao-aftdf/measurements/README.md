# Phase 13 pre-implementation measurements (upstream PySCF 2.12.1)

Run every one of these as:

```bash
PYTHONPATH=. .venv/bin/python -u <script>.py
```

`PYTHONPATH=.` is mandatory — it pins `import pyscf` to the **vendored** 2.12.1
tree at `<root>/pyscf`, not site-packages 2.14 (which rewrote `fft_jk`'s exxdiv
handling; see `11-VERIFICATION.md`). `-u` is also mandatory: these runs take tens
of minutes and a buffered pipe swallows every row until exit.

| script | measures | feeds |
|---|---|---|
| `jk2.py` | Gate 1 at gamma and `k ≠ 0`; `estimate_rcut` vs `cell.rcut`; the J/K mesh sweep | Gate 1a, Gate 2 |
| `exxctl.py` | the same J/K diff with `exxdiv` on and off | risk R-15, plan 13-05 test 4 |
| `rcutlever.py` | Gate 1 and the J/K diff at `estimate_rcut` × {1.0, 1.5, 2.0} | Gate 1b/1c, D-PBC-21 |
| `gate2.py` | the KRHF energy series, both builders, `conv_tol = 1e-12` | Gate 2 |

## Results recorded 2026-08-28/29 (diamond, `gth-szv`/`gth-pade`, 2×2×2)

`cell.rcut` = 21.319 Bohr · `ft_ao.estimate_rcut` = 20.420 Bohr ·
`madelung` = 0.3400910 · default mesh `[47,47,47]` · nao 8

**Gate 1** — upstream `ft_aopair[G=0]` vs `pbc_intor("int1e_ovlp")`:
1.554e-9 at gamma, 5.322e-10 at `k ≠ 0`.

**J/K mesh sweep** (fixed init-guess density, `max|·|`):

| mesh | `dvj` | `dvk` (ewald) | `dvk` (None) |
|---|---|---|---|
| 15 | 3.827e-7 | 1.112e-6 | — |
| 21 | 2.365e-10 | 1.804e-9 | 1.690e-9 |
| 31 | 1.996e-11 | 6.487e-10 | 2.653e-11 |
| 41 | 1.996e-11 | 6.485e-10 | — |

**`rcut` lever** (mesh 31 throughout):

| `rcut` | Gate 1 | `dvj` | `dvk` |
|---|---|---|---|
| ×1.0 = 20.42 | 1.554e-9 | 1.996e-11 | 6.487e-10 |
| ×1.5 = 30.63 | 1.472e-10 | 7.727e-13 | 1.609e-10 |
| ×2.0 = 40.84 | 1.472e-10 | 7.726e-13 | 1.609e-10 |

**KRHF energies** (`conv_tol = 1e-12`; rows for mesh ≥ 31 were still running when
Phase 13 was planned — plan 13-08 Task 0 completes this table):

| mesh | `E_FFTDF` | `E_AFTDF` | abs diff |
|---|---|---|---|
| 15 | −10.93090153682113 | −10.93087523588556 | 2.630e-5 |
| 21 | −10.93087319091901 | −10.93087316555834 | 2.536e-8 |
| 31 | −10.93087316795859 | −10.93087316798466 | **2.607e-11** |
| 41 | −10.93087316795859 | −10.93087316798466 | **2.607e-11** |

## The four conclusions these force

1. **Gate 1 cannot be 1e-10 against `pbc_intor`.** Beyond `rcut` ×1.5 the FT sum
   is converged and the residual (1.472e-10) is `pbc_intor`'s own truncation. Only
   a reference built over the SAME `Ls` (`intor_cross_with_images`) isolates the
   kernel — that is Gate 1c, target 1e-13.
2. **Gate 2 is not a mesh sweep.** Mesh 41 == mesh 31 to three digits; `rcut` ×2.0
   == ×1.5 to four. Two independent floors, so the gate is a `(rcut, mesh)` ladder.
3. **The R-15 exxdiv asymmetry is first-order, not a footnote** — ~96% of `dvk` at
   mesh 31 with the SCF's default `exxdiv='ewald'`.
4. **The energy and the matrices floor at DIFFERENT levels.** At mesh 31 the
   energy difference is 2.607e-11 Ha while `max|vk_AFT − vk_FFT|` is 6.487e-10 —
   the R-15 G=0 asymmetry is a near-uniform shift that largely cancels in
   `Tr(D·vk)`. Gate them separately; do not reuse one tolerance for both.
5. **Never characterise any of this at mesh 21.** There the general screening
   residual still dominates and the two error sources partially cancel in the
   max-abs norm, which makes the exxdiv term look ~6× smaller than it is.

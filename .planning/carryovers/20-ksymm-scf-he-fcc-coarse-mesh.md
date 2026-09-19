# Carryover — ksymm KRHF vs full BZ on He-fcc at mesh 15: 1.354e-6

**Source:** `.planning/phases/20-pbc-python-bindings/20-15-SUMMARY.md` O1 and `20-13-SUMMARY.md` D5;
recorded by 20-18 (`20-VERIFICATION.md`). Not investigated in Phase 20; no tolerance changed.

## Measured

| fixture | mesh | ksymm − full BZ |
|---|---|---|
| He-fcc `6-31g` 2×2×2 KRHF, `ops_outside_kmesh_subgroup` empty, `use_ao_symmetry` on or off | `[15]*3` | **1.354e-6** |
| same | default `[99]^3` | 3.6e-14 (SCF); ksymm KMP2 still **6.84e-9** from full BZ |
| He simple-cubic `6-31g` 2×2×2 (the fixture the binding gates use) | 15 | KMP2 2.297e-11, SCF 1.8e-15 |
| He sto-3g LDA KRKS (20-13 D5) | 15³ / 16³ / 21³ / 43³ | 8.31e-7 / 5.21e-7 / 2.94e-9 / 8.39e-14 |

The monotone fall with mesh points at quadrature aliasing (rotations do not commute with an
under-resolved grid, memory `krhf-coarse-mesh-diverges`), but that is a hypothesis for the KRHF
row: J/K on FFTDF at mesh 15 was not bisected, and the default-mesh KMP2 6.84e-9 is unexplained.

## Unblock

1. Bisect the He-fcc mesh-15 KRHF gap by step (hcore / J / K / eig) as 20-04 did for GDF.
2. Explain the default-mesh KMP2 6.84e-9 (vs 2.3e-11 on simple cubic) — fcc k-mesh symmetry vs
   the `kmp2_ksymm` padding/star mapping.

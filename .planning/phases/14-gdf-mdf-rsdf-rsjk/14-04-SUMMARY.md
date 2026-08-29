# 14-04 SUMMARY — `df_jk`: J and K from `cderi`, and **Gate 1 MET**

**Status:** SHIPPED. **Date:** 2026-08-29.
**Green:** `cargo test -p pyscf-pbc-df --test df_jk_gdf` (6),
`cargo test -p pyscf-pbc-scf --test df_swap` (the phase gate).

## GATE 1 — a converged KRHF on GDF matches upstream

He-fcc/`sto-3g` 2×2×2, the ALL-ELECTRON control where `exclude_dd_block` is
provably inert (D-PBC-23):

| | |
|---|---|
| port, `KRHF` on GDF | **−2.80842508692377** |
| upstream `df.GDF`, `_prefer_ccdf = True` | **−2.80842508664874** |
| **\|dE\|** | **2.750e-10** |

One number exercising the whole phase: the auxiliary cell (14-01), the
compensating charge and `cderi` (14-02), the store and the nuclear builder
(14-03), and the J/K contraction (14-04). The residual is the SCF-accumulated
form of the 1.9e-9 `direct_scf_tol` screening difference 14-02 attributed to
upstream's own prescreen.

And the DF **fitting error** is reproduced too: `|E_GDF − E_FFTDF|` =
**6.0056e-05** against upstream's **6.006e-05**. That is a property of the
auxiliary basis, present in upstream, reachable by no implementation — and it
is now asserted, so nobody can "fix" it.

## What shipped

`gdf/jk.rs`: `get_j_kpts`, `get_k_kpts`, `get_jk`, wired into
`impl PeriodicDf for Gdf`.

```text
J:  rho[L]       = SUM_k SUM_{mu nu} cderi[k][L, mu nu] · dm[k][nu, mu]
    vj[k][mu nu] = (1/nkpts) SUM_L rho[L] · cderi[k][L, mu nu]

K:  t[i, k]      = SUM_q cderi[ki,kj][L, i q] · dm[kj][q, k]
    vk[ki][i, l] = (1/nkpts) SUM_kj SUM_L SUM_k t[i,k] · conj(cderi[ki,kj][L, l k])
```

The K shape is deliberately identical to `aft_jk::get_k_kpts`'s with `cderi`
replacing the weighted AO-pair Fourier transform, so the two read side by side.

`exxdiv` is applied to the ASSEMBLED `vk`, not folded into the kernel — the
structural difference from AFTDF, and upstream comments on it
(`df_jk.py:676-679`) because GDF's integrals are analytic.

## Two defects this plan caught

1. **GDF did not build lazily, and the SCF drivers never call `build()`.**
   Upstream builds `_cderi` on demand at the head of `get_j_kpts` /
   `get_k_kpts` (`df_jk.py:86-92`, `:292-299`); the port required an explicit
   `build()` and the driver — which hands the boxed builder straight to
   `get_jk` — failed with `call build() first`. `Gdf` now carries its `cderi`
   and its builder in `OnceLock`s and builds on first use.
2. **`get_nuc` / `get_pp` were evaluated on the COMPENSATED mesh, and that is
   wrong.** Plan 14-03 delegated them to AFTDF at `_CCNucBuilder`'s mesh, on the
   reasoning that "with `eta` chosen so the model charge is resolved by the mesh
   the two agree by construction". **That reasoning is false**: the mesh
   resolves the MODEL CHARGE, not the nuclear density. `_CCNucBuilder` splits
   `get_pp_loc_part1` into a real-space `_int_nuc_vloc` plus a reciprocal-space
   remainder precisely so the coarse mesh suffices for what is left. Measured on
   He-fcc 2×2×2:

   | mesh | `v_nuc[0,0]` |
   |---|---|
   | `[9,9,9]` (`_CCNucBuilder`'s) | −1.835938176640 |
   | `[15,15,15]` | −1.871405034120 |
   | `[21,21,21]` | −1.872891481488 |
   | `[31,31,31]` | −1.872934360277 |
   | `[43,43,43]` (the cell's) | −1.872934388301 |

   3.7e-2 per element, which took the converged `KRHF` **0.0743 Ha** away from
   upstream. Fixed by using the CELL's mesh, where AFTDF's `get_nuc` is
   oracle-gated to 2.755e-12. The cost of not porting the split is therefore
   **performance, not accuracy**.

## Deviations from the plan

* Only the density-matrix K branch. Upstream's MO-factorised `get_k_kpts`
  (`force_dm_kbuild = False`) produces the same numbers with one fewer `nao`
  factor; it is a performance variant and is Phase 17. Not selectable, so it
  cannot be reached by accident.
* `kpts_band` is refused (upstream REBUILDS `_cderi` to cover band k-points).
* `omega` is refused — it needs `GDF.range_coulomb`, plan 14-07. An ignored
  `omega` would give a plausible full-range answer to an RSH functional, which
  is exactly the class of silent error D-PBC-20 exists to prevent.

## Carry-overs

* `_CCNucBuilder._int_nuc_vloc` — the real/reciprocal split. Purely a
  performance item now that the mesh is right, but it is what makes upstream's
  `get_nuc` cheap.
* The MO-factorised K branch.
* Everything 14-02/14-03 already listed.

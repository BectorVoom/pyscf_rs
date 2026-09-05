# 15-07 — verification rollup

**Status: complete.** Phase 15 is closed, with one gate `NOT MET` and assigned
to Phase 14.

The first pass (2026-09-05) shipped three of the nine oracle parts and reported
`PARTIAL`. The follow-up pass completed the remaining six, ran the staggered
energy oracle, and measured the two open D-PBC-28 rows. Everything is in
`15-VERIFICATION.md`.

**The staggered oracle earned its cost.** It caught three defects that no
existing gate could see:

1. `Krhf::get_occ` took its k-point count from `mf.kpts` instead of from its
   `mo_energy` argument (`khf.py:191-192`). A no-op during SCF; on
   `kmp2_stagger`'s full-mesh path it left eight of sixteen k-points with no
   occupied orbital and returned `-0.3737 Ha` where upstream gives
   `-0.014029 Ha` — **a factor of 26.6**. `Kghf::get_occ` carried the identical
   divergence and is fixed with it.
2. `Kmp2Stagger::kernel` used the mean field's own DF for the four-index path.
   Upstream always builds a fresh `FFTDF(mp.cell, mp.kpts)` there
   (`kmp2_stagger.py:74`), *even on a GDF mean field* — worth `5.01e-5 Ha` on
   the H2 fixture, i.e. the port would have reported the wrong one of two
   legitimate integral approximations.
3. `new_full_mesh` hardcoded `with_df_ints = false`; upstream sets it from the
   mean field (`:279-282`) and rebuilds a GDF over the combined mesh (`:169`).

**Measured, not asserted.** FFTDF KMP2 clears its `2e-6` gate by five orders
(`5.418e-11` diamond, `2.719e-11` He/6-31g); the staggered energies land at
`1.408e-12` and `8.28e-13`; `symm_map`, `_operation` and the whole padding
surface match upstream *exactly*. MO-first AO2MO is **9.784×** the AO-ERI route
on diamond `[1,1,2]`, against §7.0's `16×` prediction — the prediction
over-states it, and saying so is the point of measuring.

**Two upstream constants no longer reproduce in 2.12.1** —
`kmp2_stagger.py:385/390/395` are `2.8e-7`…`3.5e-7` away from what the code
now produces, the same shape as the diamond anchor's `2.1e-10`. The tests gate
on the live values and `15-VERIFICATION.md §2` records both.

**What was not reached, and why:** the `[2,2,2]` and `gth-dzvp` legs of the
MO-first sweep, at ~26 s per quadruple × 512 quadruples. Reported unreached
rather than extrapolated; the test that runs them is committed.

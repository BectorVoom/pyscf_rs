# 14-03 SUMMARY — `GDF`, the `_cderi` store, and the `PeriodicDf` seam

**Status:** SHIPPED except `get_jk` (plan 14-04 owns it) and the k-point
`get_pp` path (a PRE-EXISTING gap, surfaced and pinned — see below).
**Date:** 2026-08-29.
**Green:** `cargo test -p pyscf-pbc-df --test gdf` — **11 passed, 0 failed**.
`check-dependency-wall` PASS, `check-orphan-modules` PASS.

## Task 0 — the D-07 / `hdf5-metno` question, settled

`pyscf-pbc-df` carried a **direct `hdf5-metno` dependency from the Phase-9
scaffolding and never used it** — `grep hdf5 crates/pyscf-pbc-df/src/` was
empty — which contradicted D-07 ("`pyscf-chkfile` is the sole owner"). The
`_cderi` store is this crate's first real HDF5 consumer, so the direct dep is
**removed** and the store goes through `pyscf_chkfile::hdf5`, exactly as
`pyscf-runtime`, `pyscf-ccsd`, `pyscf-grad`, `pyscf-geomopt` and `pyscf-ao2mo`
already do. `check-dependency-wall` stays green.

## What shipped

| Module | Content |
|---|---|
| `gdf/cderi_store.rs` | `sr_loop`, `get_naoaux`, `CderiFile` (HDF5 save/load, RAII delete-on-drop), the `s1 ↔ s2` repack |
| `gdf/nuc.rs` | `nuc_eta`, `nuc_eta_mesh`, `get_nuc`, `get_pp` — `_CCNucBuilder`'s own `eta`, which is NOT `_CCGDFBuilder`'s |
| `gdf/mod.rs` | the `Gdf` class, `build`, `ensure_built`, `sr_loop`, `get_naoaux`, and `impl PeriodicDf` |

The `_cderi` file uses upstream's layout (`/kpts`, `/aosym`,
`/j3c/<ki*nkpts+kj>/<step>`) so a file this port writes is one Phase 15/16 can
read.

## Numbers

| assertion | result |
|---|---|
| `get_naoaux` (He-fcc) | 23 |
| `GDF::mesh()` (He-fcc) | `[9,9,9]` — the MODEL-CHARGE mesh, against FFTDF's `[43,43,43]`. That is why GDF is cheap. |
| `_cderi` HDF5 round trip | bit-identical on every block |
| `sr_loop` `s2 ↔ s1`, incl. `ANTIHERMI` on the imaginary half | exact |
| `get_nuc` Hermiticity, He-fcc 2×2×2 | < 1e-11 |
| `get_pp` Hermiticity, diamond gamma | < 1e-9 (Phase 13 measured the 5.131e-11 screening residue this has to clear) |
| `nuc_eta` | `max(0.5/(0.5 + nkpts^(1/9)), ETA_MIN)`, exact |

`GDF` drives through `Box<dyn PeriodicDf>` with no driver change (D-PBC-22), and
both unimplemented paths REFUSE loudly: `get_jk` names plan 14-04,
`prefer_ccdf = false` names plan 14-07 and the 4.502e-06 it is worth.

## A PRE-EXISTING GAP this plan surfaced: `get_pp` is gamma-only

`GDF::get_pp` splits the local pseudopotential the way AFTDF does — part 1 in
reciprocal space, part 2 in real space — and Phase 10's
`pseudo::vloc_part2::get_pp_loc_part2` **refuses any non-gamma k-point**: its
k-resolved counterpart is upstream's `aft._IntPPBuilder`, which Phase 13 chose
not to port. **AFTDF has the same limitation**, which means Phase 13's Gate-3
`get_pp` measurement was a gamma measurement.

FFTDF is unaffected — it evaluates the WHOLE local part in G-space through
`get_vlocg` and never calls part 2 — which is exactly why Phases 11 and 12 could
run `KRHF`/`KRKS` on diamond 2×2×2 while this path cannot.

**This blocks plan 14-04's k-point pseudopotential gates.** Closing it is now
cheap and in scope: `incore::aux_e2` **is** `_IntPPBuilder`'s double lattice
sum, already k-resolved and already Bloch-phased; it only needs its `intor`
generalised from `int3c2e` to the `int3c1e_r{2,4,6}_origk` family that
`vloc_part2::PART2_INTORS` already names, and a contraction against
`fake_cell_vloc`'s `C_n` coefficients. `tests/gdf.rs` asserts the refusal names
`_IntPPBuilder`, so the gap cannot be forgotten.

## Deviations from the plan

* `get_nuc` / `get_pp` delegate to the already-shipped, oracle-gated AFTDF
  routines at `_CCNucBuilder`'s mesh rather than re-splitting
  `get_pp_loc_part1` into its own real- and reciprocal-space halves. With `eta`
  chosen so the model charge is resolved by the mesh the two agree by
  construction, and a second code path for the same number is a liability.
  Plan 14-09 should record the equivalence as a measurement.
* `CderiArray` / `_load3c` / `_KPair3CLoader` are represented by `CderiFile` +
  `sr_loop`; the rest of upstream's reader zoo is unbuilt until a consumer asks.
* `Gdf::prefer_ccdf` defaults to **`true`**, the opposite of upstream, because
  the compensated builder is the one shipped. 14-07 flips it and moves the
  committed reference energies by a documented 5.960e-07.

## Carry-overs

* **`_IntPPBuilder` / k-point `get_pp`** — the blocker above.
* `get_jk` — plan 14-04.
* `exp_to_discard` and the 2-D `cderi_negative` branch are refused, not ignored.

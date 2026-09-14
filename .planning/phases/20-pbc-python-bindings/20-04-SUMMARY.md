# 20-04 SUMMARY — GDF `KRKS` ksymm Gate C bisected: an XC-grid defect in the full-BZ arm

**Shipped:** 2026-09-14. All four tasks complete. Measurement plan — **no
production `src/` file edited**, no commit (D-20-A). Files: new
`crates/pyscf-pbc-dft/tests/gdf_ksymm_bisect.rs` (3 tests, all
`#[ignore = "T3: 20-04 bisect diagnostic ..."]`), new
`measurements/gdf-ksymm-bisect.md`, one row appended to `measurements/README.md`
§1 (row 14).

## Task 1 — non-vacuity

- `nkpts = 8`, `nkpts_ibz = 3` (integers, asserted `3 < 8`); band set
  `kpts_ibz` asserted a strict AND bitwise subset (`kpts_ibz[i]` ==
  `kpts[ibz2bz[i]]` by `to_bits`).
- One density for every arm: `D_sym = unfold(D[ibz2bz])` from one converged
  FFTDF `KRKS`; `D_sym[ibz2bz] == D_ibz` and `unfold(D_sym[ibz2bz]) == D_sym`
  both asserted at **0e0**.

## Task 2 — GDF, step by step (first > 1e-9 named)

| step | GDF `max|Δ|` |
|---|---:|
| 1a `_cderi` two full-BZ builds (64 blocks) / 1b full-BZ vs IBZ-only fit `(k,k)` | 0e0 / 0e0 |
| 2 `get_j` direct vs `kpts_band` path; 2e `ecoul` | 0e0; 9.09e-12 |
| 3 `get_k` | 0e0 |
| 4 band route vs direct, `vj` / `vk` | 0e0 / 0e0 |
| 5 `hcore` | 0e0 |
| **6 `vxc` each arm on its own grid; 6e `exc`** | **1.810e-05; 1.432e-06 — FIRST** |
| 7 `E_elec[D]` production drivers | 1.432e-06 |
| 8 / 8e / 8E same, full arm's grid matched to `cell.mesh` | 3.3e-16 / 4.0e-15 / 2.719e-10 |

## Task 3 — FFTDF control

Same harness, same density: every step ≤ **1.008e-13** (`E_elec[D]`), J/K/band/
hcore 0e0, `exc` 4.0e-15. The harness resolves 1e-13, so the 1e-6 is the
route's, not the harness's.

## Task 4 — classification: (a) DEFECT

`crates/pyscf-pbc-dft/src/krks.rs:93` builds `Krks::from_df`'s XC grid on
`with_df.mesh()`. For `Gdf` that is the RS long-range mesh **`[13,13,13]`**;
upstream (`rks.py:272`, `gen_grid.py:72`) and `KsymAdaptedKrks` use
`cell.mesh` = **`[35,35,35]`**. FFTDF passed only because its DF mesh equals
`cell.mesh`. Same line in `kuks.rs:80`, `kroks.rs:63`, `kgks.rs:63`. Not a
fitting difference: steps 1-5 are bit-identical. 17-08's "symmetry-broken GDF
fit" hypothesis is refuted.

Confirmed at gate level (`gdf_gate_c_with_and_without_matched_xc_grid`, 2544 s,
exit 0): as written `|dE| = 1.4324445718e-06` (reproduces 17-08 to 12 digits);
full arm's grid matched → **`|dE| = 2.720e-10`**. Gate (1e-8) is correct and NOT
restated.

## Deviations

- Plan named `krks_ksymm.rs` for edits; a new test file was used instead (task
  instruction, avoids a concurrent `#[ignore]` rewrite).
- Steps 5-8 (hcore, XC, energy, matched grid) added beyond the plan's four:
  the plan's four were all 0e0, so the bisect had to continue to the XC half.
- `src/` diff shows other agents' concurrent edits (Phase 18, D-20-E, and
  `gdf/*`, `mdf/*`, `rsdf_builder/*` at 09:05, after both binaries were built);
  none from 20-04.

## Hand-off (fix NOT done)

Faithful fix is `cell.mesh`, but `Fftdf::with_mesh` callers
(`pyscf-pbc-dft/tests/gate.rs:46`, `tests/modules.rs:34`,
`pyscf-bench/src/bin/krks_repro.rs:83`) currently rely on the DF mesh driving
the XC grid — pin `cell.mesh` there or keep the DF mesh for FFTDF only, re-run
those gates, then re-run `krks_ibz_energy_matches_full_bz_on_gdf` and update
`17-VERIFICATION.md:126` / §10.8.

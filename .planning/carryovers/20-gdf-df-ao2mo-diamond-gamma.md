# Carryover — GDF `df_ao2mo::get_eri_matches_upstream_on_diamond_gamma` 3.8427e-8 vs 1e-11

**Source:** `.planning/phases/20-pbc-python-bindings/measurements/gdf-ao2mo-regression.md`
§Diamond gamma; recorded by 20-18 (`20-VERIFICATION.md`). Gate not loosened.

## Measured

| | value |
|---|---|
| port CCGDF vs upstream CCGDF (route pinned `prefer_ccdf = true`, `exclude_dd_block=False`, screens 1e-14) | **3.8426711743144715e-8** FAIL vs `< 1e-11` |
| before the route pin (port RSGDF vs upstream CCGDF) | 2.3500e-7 (20-02 P3), 2.3735e-7 (P2) |
| wall | **2793 s** (46.5 min, ~12 cores), 240 MB RSS, exit 101 |

The 1e-11 gate has never been met; Phase 14 left it owed (`14-VERIFICATION.md:124-137`).
Its ignore tag says `T2`; by D-20-D it is **T3** (retag owed to the test owner). It is a KNOWN-RED
row of `pbc-oracle-full` and is left out of nightly (20-03).

The He-fcc sister gates were a harness route mismatch, fixed test-only (1.6667e-12, 1.9839e-12, PASS).

## Hypotheses (not measured)

- Diamond's GDF metric is near-singular (`eig_min = 3.17e-11`, Cholesky; Phase 14 measured
  `V j2c Vᴴ = I` at 3.094e-8, "a conditioning floor") — same order as the residual.
- D-PBC-21 `ft_aopair` screening residual 5.121e-10.
- Not D-PBC-23 (dd-block switched off on the upstream side).

## Unblock, in order

1. Attribution device on diamond: upstream `get_eri` over the port's `cderi`.
2. Raw fused `j3c`/`j2c` elementwise vs upstream
   (`gdf_builder.rs::cderi_fingerprint_matches_upstream_diamond`, T3). If both ~1e-12 while `cderi`
   ~1e-8, redesign the diamond gate at the `j3c`/`j2c` level (owner decision, not a loosening).
3. Audit every other `Gdf::new`/`Mdf::new` next to a `_prefer_ccdf = True` oracle for the same
   route pin.

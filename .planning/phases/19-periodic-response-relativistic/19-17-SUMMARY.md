# 19-17 SUMMARY — DFADC + Gate-D rollup

**Shipped:** 2026-09-12. `gate_d` 5/5 green. **Gate D closed** for IP-incore,
EA-incore, IP-DF, EA-DF, each against its own upstream number.

## Variant (`crates/pyscf-pbc-adc/src/dfadc.rs`)

- `df_ovvv_chunk` / `df_vvvv_chunk` (the 62-line helpers, fidelitous down to
  `vvvv`'s trailing `transpose(0,2,1,3)`), `build_df_blocks` (the five DF
  `L·L` einsums `/nkpts`), `sigma_ea_df` + `kernel_ea_df` (the DF-chunk EA
  branch ported LITERALLY — direct `2·icab` AND exchange `−ibac`, the
  asymmetry vs the incore branch preserved, not "fixed").
- DF `ovvv` stays unset (None upstream): NaN marker with the right shape so
  an incore-EA read propagates NaN loudly; the chunk path never reads it. A
  test pins the NaN.
- Correction during development: the `/nkpts` on DF chunks lives at
  upstream's CALL site (`...reshape(...)/nkpts`), not in `get_ovvv_df` —
  dropping it doubles every EA root on a 2-k mesh (caught by Gate D at
  4.8e-2, divided at the call sites, documented on the function).

## The route-mix finding (recorded, not absorbed)

PBC `density_fit()` defaults to **GDF**, so `kernel()` runs the DF transform
while a direct `transform_integrals()` call runs incore. The first Gate-D
attempt compared this port's incore-EA against DF-reference roots (1.45e-2
off) — root cause identified in source (DF branch has a different equation),
fixture regenerated all-incore, and the DF-vs-incore EA gap (1.5e-2) is now
its own per-route gate rather than a failure. DF IP == incore IP to 1e-15
here (no ovvv at ADC(2)) — measured, not assumed.

## Gate D (`tests/gate_d.rs`)

Chunks vs direct, DF-IP and DF-EA at 4dp vs DF numbers, rollup table with the
measured route gaps, a mechanical guard that cross-route gating cannot pass
(EA routes differ at 1.5e-2), determinism.

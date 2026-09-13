# 19-16 SUMMARY — ADC(2) EA

**Shipped:** 2026-09-12. `ea` 3/3 green + 1 ignored; **Gate D passes** for EA
roots. **No fork of the base**: 19-14's files untouched (only new `roots.rs`
is shared, used by both manifolds).

## Solver (`crates/pyscf-pbc-adc/src/ea.rs`)

- `build_m_ab` (zeroth `diag(e_vir)` + four `t2_1`–`ovov` families, two plain
  + two conjugated), `sigma_ea` (incore `ovvv` branch ported LITERALLY —
  single-`ovvv`-fetch shape included), `kernel_ea` (same nosym dense solve).
- The investigation that closed the plan: EA roots came out 1.45e-2 from the
  first fixture — root cause was a ROUTE MIX, not a code bug. PBC
  `density_fit()` defaults to **GDF**, so `kernel()` runs the DF transform
  (`ovvv=None`, chunk branch with a DIFFERENT equation: direct 2· AND
  exchange −1·) while the fixture saved INCORE blocks. Proof chain: (1) the
  incore-vs-DF equation difference identified in source; (2) with EXPLICIT
  incore eris passed everywhere, upstream's EA roots are [0.81973…] —
  matching this port's [0.81973…] to the gate; (3) the DF-vs-incore EA gap
  (0.8197 vs 0.8343) is a genuine route difference, recorded for 19-17's
  per-route Gate D. Fixture regenerated all-incore.

## Gate D (`tests/ea.rs`)

- `M_ab` vs `get_imds` (< 1e-9), sorted/counted EA roots at 4dp, determinism.
- Spec factors deferred (same `#[ignore]`d arm as IP).

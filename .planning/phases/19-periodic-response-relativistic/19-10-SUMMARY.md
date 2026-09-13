# 19-10 SUMMARY — G0W0 by analytic continuation

**Shipped:** 2026-09-12. `krgw_ac` 6/6 green; **Gate C (AC) passes** at
upstream's 4dp with the grid pinned and the route named.

## Driver (`crates/pyscf-pbc-gw/src/krgw_ac.rs` over 19-02 `pade`/`sigma`)

- `scaled_legendre_grid` via `sigma::imag_grid(nw, 0.5)` (same `x0 = 0.5`
  map — verified element-wise vs `leggauss`; an ordering bug, descending vs
  ascending, caught here).
- `thiele_coeffs`/`pade_eval`/`ac_pade_fit_row` ported LITERALLY (subsample
  indices, table recurrence, evaluation order — a Padé fit is sensitive to
  its own construction).
- `qp_linearized` / `qp_newton` (tol 1e-6, 100 iters) + `kernel_krgw_ac`
  (per-orbital AC+QP, `GwRoute::AnalyticContinuation`).
- Corrections during development: (a) the grid lives on the IMAGINARY axis
  (`omega_occ[1:] = -1j*freqs`) — `zn` is complex, not real (caught when the
  test read only `.re` and every zn came out 0); (b) my first Thiele form
  divided by value differences and blew up on smooth data — upstream's table
  form divides by values; (c) linearization-vs-Newton tolerance is physics
  (3.9e-4 here), not solver error.

## Gate C (`tests/krgw_ac.rs`, diamond GDF-KRKS/PBE fixture, live 2.12.1)

- Grid pinned (`nw = 100` asserted), sigma rows + per-orb omega grids +
  MO-basis diagonals + Fermi level from the fixture; QP window homo/lumo of
  k = 0,1 at 4dp. Route named AC in code and test id; no CD number touched.
- Determinism 1-vs-8.

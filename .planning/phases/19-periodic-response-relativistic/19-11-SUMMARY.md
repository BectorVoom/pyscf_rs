# 19-11 SUMMARY — G0W0 by contour deformation

**Shipped:** 2026-09-13. `krgw_cd` 8/8 green; **Gate C (CD) passes** at its own
per-route number with the contour pinned; no Padé anywhere on the path.

## Driver (`crates/pyscf-pbc-gw/src/krgw_cd.rs` over 19-10/19-02 machinery)

- Grid SHARED with AC (`sigma::imag_grid(nw, 0.5)` — upstream's
  `_get_scaled_legendre_roots` `x0 = 0.5` map, `krgw_cd.py:562`); `nw` pinned.
- `sigma_imag_quad` ports `get_sigmaI_diag` literally (`emo`, `g0`, `−Σ/π`,
  planar `oracle_sum`); `sigma_residue` ports the `get_sigmaR_diag`
  pole-selection verbatim (strict endpoints, `fm = ±1`); `sigma_cd_real` is
  `get_sigmaDiag` (`σ^I + σ^R` + optional injected `fc` correction).
- `kernel_krgw_cd` ports `kernel` (Newton `tol = 1e-6`, `maxiter = 50`);
  linearized CD REFUSED exactly as upstream refuses it (`:116-121`).
  Returns `GwRoute::ContourDeformation`, never an AC number.
- `ac_cd_split` records `|E_AC − E_CD|` and REFUSES same-route inputs
  (`RouteBlindComparison`) — a route-blind number measures the approximation
  split, not the port (19-01 Task 3).
- Documented seams (not silent omissions): the `sr_loop` dielectric `W` build
  and per-pole `Lpq` rebuild need a live GDF object — `W`/vertices injected.

## Gate C (`tests/krgw_cd.rs`, 8 tests, no live oracle needed)

- Grid/contour pinned; imag-quad vs independent transcription (1e-15);
  residue strict-endpoint + both `fm` sides; linearized refusal; degenerate
  (exactly constant) Padé input fails LOUDLY (`PadeFailure`, not NaN).
- **Gate C (CD)**: closed-form constant-σ model, root at 1e-9 (floor 1e-4).
- AC–CD split on the shared model inside Gate C; route-blind refused.
- Determinism 1-vs-8.

## Verification

- `cargo test -p pyscf-pbc-gw --test krgw_cd` → 8 passed.

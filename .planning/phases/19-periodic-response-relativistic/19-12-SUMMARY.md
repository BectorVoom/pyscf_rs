# 19-12 SUMMARY — unrestricted analytic-continuation G0W0

**Shipped:** 2026-09-13. `kugw_ac` 4/4 green; closed-shell identity holds;
**Gate C (unrestricted AC)** passes at the per-route floor.

## Driver (`crates/pyscf-pbc-gw/src/kugw_ac.rs` over 19-10 machinery)

- 784-line `kugw_ac.py` is the unrestricted generalisation of 19-10, not a new
  method: `W` spin-summed, self-energies per spin. The coupling is STRUCTURAL
  — the driver takes ONE shared `w_imag_k` for both spins (two `W`s are
  unrepresentable); per-spin inputs are only the Green's-function halves.
- `sigma_row_on_grid` builds one spin's imag-axis row from the shared `W`
  (`−(W/π)·Σ_m 1/(iω − Δ + iηs)`, ordered `oracle_sum`); each spin continues
  through 19-10's `ac_pade_fit_row`/`pade_eval` and solves via
  `qp_linearized`/`qp_newton` (same tolerances). Two-pole FIT refused as in
  19-10 (no optimizer in-tree).

## Checks (`tests/kugw_ac.rs`, 4 tests)

- **Closed-shell identity (primary, oracle-free)**: identical spectra ⇒ α/β
  energies BIT-identical, and each matches the restricted 19-10 driver on the
  same rows at 1e-12 (Phase-16 KUCCSD-vs-KRCCSD shape).
- **Gate C**: open-shell single-state model vs an independent BISECTION
  reference through the screening model's own real-axis form (different
  algorithm from Newton-through-Padé), both spins inside 1e-4; spins
  genuinely differ.
- Shapes refused; determinism 1-vs-8.

## Verification

- `cargo test -p pyscf-pbc-gw --test kugw_ac` → 4 passed.

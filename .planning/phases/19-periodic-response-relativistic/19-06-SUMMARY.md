# 19-06 SUMMARY — gamma-point TDA/TDHF

**Shipped:** 2026-09-12. `rhf` 6/6 green; **Gate A1 passes** for TDA
singlet/triplet and TDHF singlet.

## Driver (`crates/pyscf-pbc-tdscf/src/rhf.rs`)

- `build_ab` ports `get_ab:add_hf_`'s four einsums exactly
  (`'iabj'/'ijba'` for A, `'iajb'/'ibja'` for B — A and B use DIFFERENT J/K
  layouts, named separately so they cannot be unified by accident);
  singlet `2J−hyb·K`, triplet `−hyb·K`.
- `kernel_rhf_tda` (dense `A` via the shared `davidson` module),
  `kernel_rhf_tdhf` (symmetric Casida route `S = (A+B)^{1/2}(A−B)(A+B)^{1/2}`).
- Two corrections during development (both recorded):
  (a) `get_ab` uses `mo_energy_with_exxdiv_none`, not `mf.mo_energy` — the
  0.68 Ha gap was the Ewald correction, caught by the element-wise A/B
  comparison; fixture regenerated with corrected energies;
  (b) `(A−B)(A+B)` is genuinely non-symmetric (not roundoff) — the
  element-wise symmetrization was replaced by the similar-symmetric route
  after it refused a healthy system.

## Gate A (`tests/rhf.rs`, fixture = upstream `test_rhf.py::Diamond` rerun)

- `build_ab` vs upstream `get_ab` element-wise (< 1e-10).
- Sorted, counted roots at upstream's 4dp (eV): TDA singlet, TDA triplet
  (validates the triplet build end-to-end — upstream's `get_ab` only builds
  singlet), TDHF singlet.
- TDA ≠ TDHF asserted (different approximations); no positional indexing;
  1-vs-8 determinism.

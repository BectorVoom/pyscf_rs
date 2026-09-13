# 19-07 SUMMARY — k-point TDA/TDHF (phase headline)

**Shipped:** 2026-09-12. `krhf` 7/7 green; **Gate A1 passes per shift**
(shift 0: 10.957 eV, shift 1: 11.042 eV scale, both at upstream's 4dp).

## Driver (`crates/pyscf-pbc-tdscf/src/krhf.rs`)

- `build_kab` ports `get_ab:add_hf_` over `ao2mo_7d([mo,orbo,mo,mo])` blocks
  (axes `(all-MO, occ, all-MO, all-MO)`, `weight = 1/nkpts`), complex-linear
  (no conjugation — exactly as upstream writes the einsums), triplet drops
  Coulomb. `A` Hermitian by construction (asserted, 8e-17 upstream).
- `kernel_krhf_tda` (complex-Hermitian dense via `zeigh_gen`, shift stamped),
  `kernel_krhf_tdhf` (real shifts → shared `rhf::symm_tdhf` core, factored out
  of 19-06; genuinely-complex shifts REFUSE — `B ≠ B†` at 5e-4 measured, so
  no symmetric reduction applies and truncation would be silent).
- Correction during development: the diagonal was folded into the `kj` loop
  and added the gap `nk` times (upstream initializes via `diag()` once) —
  caught by the element-wise A/B comparison at 1.15, fixed by hoisting.

## Gate A (`tests/krhf.rs`, diamond `(2,1,1)` fixture from live 2.12.1)

- `build_kab` vs upstream `get_ab` element-wise both shifts (< 1e-10).
- Sorted, counted TDA singlet per shift + triplet at shift 0, at 4dp (eV).
- Cross-shift pooling proven catchable: shifts differ by 0.085 eV ≫ gate.
- Complex-TDHF refusal + real-path diagonal hand-check + determinism.

# 16-05 — `KRCCSD`. COMPLETE 2026-09-06. **`e_corr` matches upstream to 6.56e-9.**

`crates/pyscf-pbc-cc/src/{keris,kccsd_rhf}.rs`, gated by
`tests/oracle_phase16.rs` (3 tests) and `tests/kccsd_rhf.rs` (5 oracle-free
tests).

## The headline

| quantity | this port | upstream 2.12.1 | \|Δ\| | gate |
|---|---|---|---|---|
| `KRCCSD e_corr` | `-0.15529848504714802` | `-0.15529847848635378` | **`6.56e-9`** | G1 `1e-7` |
| `init_amps emp2` | `-0.10987845280037442` | `-0.1098784493062738` | `3.49e-9` | G1 |
| `energy()` (synthetic amps) | `0.008452497351695409` | `0.00845249413472838` | `3.22e-9` | `1e-6` |
| `update_amps t1new` / `t2new` | — | — | `1.84e-8` / `7.01e-8` | `1e-6` |

diamond `gth-szv` `[1,1,2]`, FFTDF, `cell.mesh = [15,15,15]`,
`cell.precision = 1e-8`, `conv_tol = 1e-9`, `conv_tol_normt = 1e-7`, 18 DIIS
cycles.

## Task 1 — `_ERIS`

The seven blocks through the `symm_map` ORBIT loop from the first version
(D-PBC-29 clause 3), `vvvv` via `ao2mo_7d` exactly as upstream does
(`kccsd_rhf.py:798`, where the `self.vvvv[...]` line inside the symmetry loop
is commented out), three storage tiers from an exact byte count, and **both
halves of the `exxdiv` treatment**: the Fock rebuilt with `exxdiv` suppressed
and the Madelung correction re-added through `_adjust_occ` (`16-CONTEXT §3.5`).
Either half alone is quietly wrong.

The index convention is stated once in the module doc and carried everywhere:
`kccsd_rhf.py:806-812` stores at `[kp, kr, kq]` after `transpose(0,2,1,3)`, so
the chemist's `(pq|rs)` from `ao2mo` becomes the physicist-ordered `<pr|qs>` the
intermediates read. A port that silently normalises either is 14-05's
`decompose_j2c` again (`+6 306 866.73 Ha`).

## Tasks 2-3 — `update_amps`, `energy`, `init_amps`, the kernel

`LARGE_DENOM = 1e14` fills the denominator at every padded orbital rather than
skipping it (`16-CONTEXT §3.3`); the four conjugation sites are explicit
`.conj()` calls at the upstream `.conj()` they came from and are enumerated in
the module doc. The DIIS iteration reuses the Phase-3 CDIIS ring buffer through
a `KAmplitudeSubspace` that packs `[re…, im…]` — a real linear combination of
complex vectors is the same operation applied to each plane, so no new DIIS body
was written, exactly as `pyscf-ccsd`'s `AmplitudeSubspace` does for the
molecular case.

## THE FINDING — the gates run on upstream's mean field, and why

Building the first end-to-end run surfaced a divergence Phase 16 neither owns
nor caused:

```
diamond gth-szv [1,1,2], cell.mesh = [15,15,15], exxdiv = None
  this port  KRHF e_tot  -8.652011318061934
  upstream   KRHF e_tot  -8.651997841504999      |Δ| 1.348e-05
```

Phase 15's `oracle_phase15` measured the same two agreeing to **`4.772e-11` on
the same cell at the DEFAULT mesh**, and upstream's `rcut` (21.319) and `nimgs`
([6,6,6]) are identical at both meshes — so it is the FFT-grid-evaluated part of
the mean field at a coarse mesh, not the lattice sums and not anything in
`pbc/cc`. **A correlation energy compared across two different mean fields
measures the mean fields.** Every CC oracle test therefore drives
`KEris::from_parts` with upstream's own `fock` / `mo_energy` / `mo_coeff` — the
discipline `15-VERIFICATION` used to drive `Lov` from "upstream's own padded
MOs" and get `2e-15`. The mean-field residual is PRINTED beside every result,
never absorbed into a tolerance, and is carried to 16-14 as a finding for
whichever phase owns FFTDF's coarse-mesh behaviour.

## Deviations from the plan's Task 4, each measured rather than argued

1. **Test 5's bit-identity is impossible and the plan is corrected.** 16-01
   measured upstream's OWN symmetry-loop and all-triples paths differing by up
   to `1.32e-7` (`measurements/README.md §7`): a symmetry-related k-quadruple's
   FFT transform and its transposed sibling are not the same floating-point
   computation. `tests/kccsd_rhf.rs::symm_map_loop_matches_the_all_triples_loop`
   gates at `1e-6` (G10) and asserts `vvvv` — built by `ao2mo_7d` in BOTH paths
   — IS bit-identical, which is the control that says the difference is the FFT
   and not the transposition.
2. **Test 3 needs `keep_exxdiv = true`, and that is not a workaround.**
   Upstream's own log line reads "MP2 energy (**with fock eigenvalue shift**)"
   (`:594`): with the default `keep_exxdiv = false` the CC orbital energies carry
   the Madelung re-add and `KMP2`'s do not, so the two are different quantities
   by construction and a test comparing them anyway would be asserting the
   Madelung shift is zero.
3. **Tests 1 and 2 (supercell equivalence, the Γ reduction against molecular
   RCCSD) are NOT shipped.** Both need infrastructure this phase does not have:
   a `super_cell(cell, nk) -> Cell` builder (`pyscf-pbc-tools` has
   `scale_lattice` and `super_cell_translations` but no cell builder) and
   16-12's Γ-point `pbc/cc/ccsd.py` shim. They are listed as carry-overs in
   `16-VERIFICATION.md` with the work each needs, not quietly dropped — and
   16-01 measured the numbers they would have been gated against
   (`2.97e-8` supercell, `README §2`), so whoever adds them has the target.

## Verification

* `cargo test -p pyscf-pbc-cc` green (the non-`--release` tests: `zarr` 8,
  `ktensor` 4).
* `PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-cc --test
  oracle_phase16 -- --ignored` — 3 passed.
* `check-orphan-modules` and `check-dependency-wall` PASS.

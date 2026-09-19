# Carryover — per-element pseudopotential dicts with DIFFERENT names are refused

**Source:** `.planning/phases/20-pbc-python-bindings/20-19-B-SUMMARY.md` D1; recorded by 20-18.

## Measured / current behaviour

- `crates/pyscf-py/src/pbc/gto.rs` builds a `cell.pseudo` dict only when every value names the
  same pseudopotential (case-insensitive) and every PP-covered atom is a key (or `'default'`).
- Refused with `NotImplementedError`: a dict with different names per element
  (e.g. `{'Si': 'gth-pade', 'O': 'gth-pbe'}`), a dict that leaves a PP-covered atom out (upstream
  keeps it all-electron), and parsed GTH parameter lists.
- Upstream suite: `gto/pseudo/test/test_pp.py::test_pp_int` (parsed GTH list) and
  `::test_pp_loc_part2` (mixed AE/pseudo cell) are refused; the counterfactual run hit "mixed AE/pseudo
  cell" on 26 tests (`upstream-pbc-suite-after-20-19.md`).

## Why it is a crate change

`CellBuildArgs::pseudo` / `Cell::pseudo_name` are ONE name for the whole cell. Per-element pseudos
need `pyscf-pbc-gto` changes in `types.rs`, `cell.rs:616`, `pseudo/mod.rs::resolve_pseudo`,
`dumps_loads.rs:117`, `supercell.rs:189`, `vnl.rs:275`, then gates: a two-element cell with two
different GTH files vs vendored 2.12.1 (`get_pp`, KRHF energy) and a mixed AE/PP cell (`get_nuc` +
`get_pp` split).

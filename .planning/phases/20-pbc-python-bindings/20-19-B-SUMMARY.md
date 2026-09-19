# 20-19 B SUMMARY — native `Cell` accepts upstream's input spellings

**Shipped:** 2026-09-14. **Files:** `crates/pyscf-py/src/pbc/gto.rs` (binding only) and `python/pyscf/tests/test_pbc_cell.py` (+23 cases).
**No crate outside `pyscf-py` was touched.** In particular: nothing in `pyscf-pbc-gto` (so nothing in `hcore.rs`/`pbc_intor.rs`) and nothing in `pyscf-pbc-scf`. No git add/commit/stash/restore/checkout.

## What was wrong

The 68 upstream class-5 tests (`measurements/upstream-pbc-suite.md`) failed on four input forms. None of the four was a Rust-crate gap:

- **`cell.output = ...` (31 tests).** `PyCell` had no `output` attribute, and a pyclass has no `__dict__`. `output` was only swallowed as a keyword.
- **List-form atoms (30).** `extract_atom` extracted `(String, [f64; 3])`. pyo3 tuple extraction rejects a Python *list*, so `[['He', (x, y, z)]]` failed.
- **Per-element pseudo dicts (4).** `extract_pseudo` accepted only a string.
- **Explicit shell lists (3).** `extract_basis` accepted only names. `pyscf_gto::BasisInput::Parsed(ParsedBasis)` already carries shells, and `format_basis.rs:108` resolves them.

The molecular `Mole` binding (`crates/pyscf-py/src/gto.rs` `M`) also accepts only strings, so there was no parsing to reuse. `bridge.rs::cell_from_upstream_attrs` is unchanged.

## What the binding now does (upstream semantics cited)

| form | implementation | upstream |
|---|---|---|
| `atom` list | Each entry may be:<br>• a string `'Sym x y z'` (commas allowed, `#` lines skipped)<br>• `[sym, x, y, z]` (`atom[1]` is a Python int/float)<br>• `[sym, coords]` (list, tuple or ndarray)<br><br>The symbol may be an int or a digit string (mapped through `ELEMENTS`). The result is fed as `AtomInput::TupleVec`, which applies the same `atom_symbol` normalisation as the string form (`format_atom.rs:51-57` vs `:129`). Built-cell round trips keep `Tuples`. | `mole.py:393-401`, `elements.py:1192-1199` |
| `basis` shell list | Accepted either globally (applies to every atom) or as a `{el: name \| shells}` dict, including `'default'`.<br><br>Conversion to `ParsedBasis`:<br>• empty shells are dropped<br>• shells are stably sorted by `l`<br>• primitive rows are sorted descending, lexicographically<br><br>`kappa = 0` is accepted. `kappa ≠ 0` raises `NotImplementedError`, because the non-relativistic `_bas` has `KAPPA_OF = 0`.<br><br>A list of shell lists is concatenated. A list that mixes names with shells raises `NotImplementedError`.<br><br>A multi-line string becomes `NwchemText`, or `Cp2kText` if it contains `GTH`. | `mole.py:457-464` (sort by l), `:484-503` (converter), `:995-1000` (row sort), `gto/basis/__init__.py:677-691` |
| `pseudo` dict | `CellBuildArgs::pseudo` is ONE name, applied to every element. A dict is therefore built as that name when **(a)** every value names the same pseudopotential (case-insensitive) and **(b)** after the build, every atom that got a pseudopotential is a key (or there is a `'default'` key).<br><br>Refused with `NotImplementedError`:<br>• a dict with different names per element<br>• a dict that leaves a PP-covered atom out (upstream keeps it all-electron)<br><br>A keyed atom that the file lacks raises `RuntimeError`. Parsed GTH parameter lists raise `NotImplementedError`. | `mole.py:2575-2591` (exact-label `symb in _pseudo`), `elements.py:1146`, `mole.py:3954-3964` |
| `output` / `stdout` | Both are real attributes. `build()` and `M()` open `output` (`os.devnull` for `/dev/null`) as `cell.stdout`, before parsing and unless `stdout.name == output` already. They print `output file: …` / `overwrite output file: …` when `verbose > 0`. `stdout` defaults to `sys.stdout` and can be assigned. | `mole.py:2517, 2535-2549` |

The `atom`, `basis` and `pseudo` getters return the object as it was assigned, as upstream does.

**If per-element pseudopotentials with different names are needed, that is a crate change.** `CellBuildArgs::pseudo` and `Cell::pseudo_name` would have to become per-element: `types.rs`, `cell.rs:616`, `pseudo/mod.rs::resolve_pseudo`, `dumps_loads.rs:117`, `supercell.rs:189` and `vnl.rs:275`. It was deliberately not done, because item D is editing `pyscf-pbc-gto` in a copy. Among the 68 tests, only single-element dicts occur (`tools/test_k2gamma`, `{'Li': 'GTH-PBE-q3'}`).

## Verification

| check | result |
|---|---|
| `cargo check -p pyscf-py` | exit 0, no warnings in `pbc/gto.rs` |
| `maturin develop --release --skip-install` (§2 env, 1 rebuild) | exit 0, 34 s wall; `.so` 18:49:53 |
| `test_pbc_cell.py` | **65 passed** (42 old + 23 new), 53 s |
| `test_pbc_identity_gate.py` | **18 / 18** |
| `xtask check-catch-unwind` / `check-dependency-wall` | PASS (852 files) / PASS (both walls) |
| full `.venv/bin/pytest python/pyscf/tests -q -p no:cacheprovider` | **10 failed, 407 passed, 4 skipped, 3 xfailed, 9 errors** in 1013 s (ran concurrently with the class-5 run; `target/p20-19-B-suite/full-python-suite.out`). The failing/erroring ids are exactly the known molecular set of `python-suite-triage.md` (panic ×2, cross_dispatch ×3, ghf, rhf_ccpvdz ×3, rhf_h2o; 9 `moleintor` errors). vs 20-18-PRE (384 passed): +23 = the new `test_pbc_cell` cases. **No new failures.** |

**Bitwise-equivalence tests.** Each alternative spelling is compared with its string-form twin on the following, all bit for bit:
- `atom_symbol`, `atom_coords`;
- `nao_nr` / `nbas` / `natm`;
- `basis_per_element` from `dumps()` (the parsed `_basis` shells);
- `pbc_intor('int1e_ovlp')`;
- `mesh`, `rcut`, `atom_charges`, `energy_nuc`.

The cases:
- **6 atom spellings** against `"C 0 0 0; C q q q"` (diamond, `gth-szv`/`gth-pade`).
- **Pseudo dicts** `{'C': 'gth-pade'}`, `{'C': 'GTH-PADE'}`, `{'default': …}` against `'gth-pade'`, and `{'Li': 'GTH-PBE-q3'}` against `'GTH-PBE-q3'`.
- **7 He shell-list spellings** against `'sto-3g'`: global, dict, re-ordered primitives, `kappa = 0`, a nested list, `'default'`, and ndarray rows.
- **Order independence:** `[p, s]` builds the same cell as `[s, p]`.
- **Refusals:** the refusal cases listed above.

**Against upstream.** An explicit-shell H₂ cell compared with vendored 2.12.1 (subprocess, version asserted) gives max |ΔS| = **1.8e-15**. The gate is 1e-10.

## Upstream class-5 re-run (the 68 node ids, 20-18 harness env/plugin, `--import-mode=prepend`, `--timeout=600`)

Setup:
- Script: `target/p20-19-B-suite/run_class5.sh`.
- Node ids: `target/p20-19-B-class5-ids.txt`. They were extracted from `logs/run-overlay/*.xml` by the four class-5 messages: 31 / 30 / 4 / 3.
- Logs and junit XML: `target/p20-19-B-suite/logs/p2019B-class5/`.
- Run detached under `systemd-run --user --scope -p MemoryMax=11G`.

**68 / 68 now get past Cell construction.** None of the four class-5 messages occurs anywhere in the logs. **4 pass:**
- `dft/test_krks::test_klda`
- `dft/test_kuks::test_klda`
- `scf/test_band::test_band_kscf`
- `tools/test_k2gamma::test_double_translation_indices`

The other 64 fail on their next blocker:

| next blocker | tests | route |
|---|---:|---|
| class-4a imports under the overlay: `from pyscf.scf import chkfile` (test_newton) ×15; `dft.radi` in setUpClass (test_gks) ×2 | 17 | item A |
| method missing on native SCF / post-SCF objects: `.newton` ×5, `to_hf` ×2, `reset`, `to_khf`, `mix_density_fit`, `KMP2.khelper`, `KUKS.mulliken_meta` | 12 | SCF bindings |
| **native `Cell` surface beyond the four forms:**<br>• `cell.set(**kw)` ×8 (test_mulliken_meta)<br>• `nimgs` is a method, not upstream's property, and has no setter ×2 (test_kpts_to_kmesh, test_chkfile_k_point)<br>• `set_geom_` ×1<br>• `cell.spin = …` after `M()` drops the build, where upstream keeps it ×1 (test_kuccsd_openshell) | 12 | 20-09 follow-up (cheap: `set`, `nimgs` property+setter, in-place `spin`) |
| numerical mismatch:<br>• `test_krks::test_klda8_*` ×4 (3.4e-3 / 4.3e-3 cubic, 3.3e-7 / 7.0e-7 primitive)<br>• `test_kuks_as_kuhf` 9.4e-3<br>• `test_rks::test_density_fit` 3.13 Ha, `_2d` 7.5e-3, `test_rsh_0d` 4.5e-4<br>• `test_uks::test_pp_UKS` 2.1e-4<br>• `mp/test_ksym::test_kmp2` 3.8e-10 (places=10), `test_rdm1` 1.5e-10 | 11 | triage (some may be item D's overlap precision) |
| XC tokens `WB97` / `HSE06` unknown to the PBC NumInt backend | 8 | XC |
| `NotYetImplemented`: `GDF.get_jk(omega > 0)` (test_rsh_0d_df); symmetrized k-symmetric quadrature (test_k2gamma_ksymm) | 2 | refusals |
| other: `KSCF.get_hcore(kpt=)` (test_band); `kcc.t1.todense` (test_krccsd_ksym) | 2 | bindings |

## Deviations

- **D1 — pseudo dicts with different names per element are refused, not built.** No such dict appears among the 68 tests. The crate change it needs is listed above.
- **D2 — basis lists that mix names with shells, and `kappa ≠ 0` shells, are refused** with `NotImplementedError`.
- **D3 — `max_memory`, `dump_input` and `parse_arg` are still accepted and ignored.** Upstream `print`s the output-file notice; so does the binding, and it appears in captured stdout.
- **D4 — `rustfmt --edition 2024` was run on `pbc/gto.rs` only.** `git diff --stat` against HEAD also counts earlier uncommitted 20-09…20-18 work in that file.

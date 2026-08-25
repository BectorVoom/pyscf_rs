---
phase: 09-pbc-foundation
type: verification
milestone: v2.0
status: PASS
verified: 2026-08-26
plans: [09-01, 09-02, 09-03, 09-04, 09-05, 09-06, 09-07, 09-08, 09-09]
---

# Phase 9 Verification — PBC Foundation + Complex Algebra

Closes PBC-MASTER-PLAN §8.1. Every one of the seven success criteria in
`09-CONTEXT.md` is demonstrated below with the command that proves it, the
observed value, and a PASS/FAIL verdict.

**Upstream reference:** PySCF **2.12.1**, the tree vendored at `<root>/pyscf`,
run through `.venv/bin/python` with `PYTHONPATH=<root>`. Spot-checked against
the venv's own PySCF 2.14.0: every value in this document is identical between
the two.

---

## 1. Success criteria

### Criterion 1 — 39 crates, clean build, both lints pass

| item | command | observed | verdict |
|---|---|---|---|
| workspace size | `cargo metadata --no-deps` | 40 packages = **39 `pyscf-*` crates** + `xtask` | PASS |
| build | `cargo build --workspace` | clean; no warnings from any `pyscf-pbc-*` crate | PASS |
| algebra wall | `cargo run -p xtask --bin check-dependency-wall` | `PASS — cubecl-* containment intact (ALG-06)` | PASS |
| forbidden paths | `cargo run -p xtask --bin check-forbidden-paths` | `PASS — 351 .rs file(s); no out-of-scope upstream PySCF imports (FOUND-08)` | PASS |

The 19 new members are `pyscf-pbc-{lib, tools, gto, df, scf, dft, ao2mo, mp, cc,
ci, symm, grad, geomopt, tdscf, gw, adc, x2c, eph, mpi}` (`Cargo.toml`
`[workspace] members`, "Workspace grows 20 -> 39").

RULE 7 un-gate: `xtask/src/forbidden_paths.rs` now scopes the needles by path —
`crates/pyscf-pbc-*` is exempt from `pbc`/`x2c`/`mcscf`/`adc`/`gw`/`eom`/`NAC`/`EPH`,
molecular crates are not.

### Criterion 2 — `zeigh_gen` matches faer to 1e-12, `Cᴴ S C == I`

```
cargo test -p pyscf-algebra --test zeigh
```

| test | observed | verdict |
|---|---|---|
| `zeigh_gen_random_8x8_hermitian_with_identity_overlap` | eigenvalues match the faer c64 route to < 1e-12; `CᴴSC − I` < 1e-12 | PASS |
| `both_zeigh_routes_agree_with_non_trivial_overlap` | faer c64 route vs the independent real 2n×2n embedding agree to < 1e-12 | PASS |
| `zeigh_gen_repeated_calls_are_bit_identical` | bit-identical | PASS |

8 passed, 0 failed.

**D-PBC-04 decision (from plan 09-02, recorded verbatim):**
> `FAER_C64 = true`. faer 0.24 has working native `c64` `SelfAdjointEigen`, `Llt`
> and `PartialPivLu`. `zeigh_gen` / `zcholesky` / `zsolve_linear` dispatch to the
> faer c64 route; the real-arithmetic route is always built and is the CI
> cross-check.

Probed by `crates/pyscf-algebra/examples/faer_c64_probe.rs` against
`H = [[2, 1−i], [1+i, 3]]` (exact eigenvalues 1 and 4).

### Criterion 3 — `oracle_zsum` bit-identical at `RAYON_NUM_THREADS` 1 and 8

```
cargo test -p pyscf-algebra --test zoracle_determinism
```

| test | observed | verdict |
|---|---|---|
| `oracle_zsum_is_bit_identical_across_rayon_thread_counts` | 1e6-element mixed-magnitude corpus; the two child processes' `f64::to_bits()` transcripts are EQUAL | PASS |
| `oracle_zsum_is_deterministic_in_process` | bit-identical across repeated calls | PASS |
| `oracle_zsum_is_two_ordered_oracle_sums` | agrees with a strictly sequential per-plane reference | PASS |

4 passed, 1 ignored (`zoracle_child_emits_bits`, the spawned child).

The test spawns its own binary twice, once per thread count, because rayon's
global pool is built once per process.

### Criterion 4 — `Cell::build` scalars, and `b·aᵀ == 2π·I`

```
cargo test -p pyscf-pbc-gto --test cell_build --test cutoff
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-gto --test oracle_phase9 -- --ignored
```

Live-oracle test `cell_scalars_match_upstream` compares `vol`, `rcut`, `mesh`
and `b` on all five systems against upstream at 1e-12 relative (integers exact),
and asserts `b·aᵀ == 2π·I` to 1e-12. **PASS.**

Hard-coded values (`src/test_systems.rs::REFERENCES`, `tests/cutoff.rs::UPSTREAM`):

| system | `vol` (Bohr³) | `rcut` (Bohr) | `mesh` | `nimgs` | `nao_nr` |
|---|---|---|---|---|---|
| diamond | 76.55488063251218 | 21.31940052177759 | [47, 47, 47] | [6, 6, 6] | 8 |
| si | 270.1967093603764 | — (see `tests/cutoff.rs`) | [35, 35, 35] | — | 8 |
| lif | 110.42101837541341 | — | [81, 81, 81] | — | 6 |
| he_fcc | 45.551257834162435 | — | [59, 59, 59] | — | 1 |
| graphene | 707.3387370358154 | — | [45, 45, 351] | — | 8 |

### Criterion 5 — `get_Gv` on `[5,5,5]` to 1e-12, `|SI| == 1`

```
cargo test -p pyscf-pbc-gto --test gv
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-gto --test oracle_phase9 -- --ignored
```

Live-oracle test `gv_and_si_match_upstream`: all 375 `Gv` components and all
`natm × 125` `SI` components match upstream element-wise to 1e-12 on all five
systems, and every `|SI[a,g]| = 1` to 1e-12. **PASS.**

Hard-coded: `tests/common/gv_reference.rs::GV_DIAMOND_555` (125 rows).

### Criterion 6 — `get_lattice_Ls`, `make_kpts`, `get_kconserv`

```
cargo test -p pyscf-pbc-gto --test lattice --test kpts_mesh
cargo test -p pyscf-pbc-tools --test lattice
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-gto --test oracle_phase9 -- --ignored
```

| live-oracle test | observed | verdict |
|---|---|---|
| `lattice_ls_match_upstream` | image COUNT exact and every component to 1e-12, all five systems | PASS |
| `kpts_and_kconserv_match_upstream` | `make_kpts([2,2,2])` and `([3,2,1])` to 1e-12; `get_kconserv` table EXACT, all five systems | PASS |
| `make_kpts_variants_match_upstream` | `with_gamma_point=False`, `wrap_around=True`, both, and `scaled_center` all to 1e-12 | PASS |

Hard-coded: `tests/common/kpts_reference.rs` — `KPTS_222`, `KPTS_222_NOGAMMA`,
`KPTS_222_WRAP`, `KPTS_222_NOGAMMA_WRAP`, `KPTS_321`, `KPTS_333_WRAP`,
`KPTS_222_CENTERED`, plus the 8×8×8 and 6×6×6 `kconserv` tables and the 4×4×4×4
`kconserv3` table (all integer, asserted EXACTLY).

### Criterion 7 — `cell.ewald()` to 1e-9 Ha, invariant to `ew_eta`

```
cargo test -p pyscf-pbc-gto --test ewald
PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-gto --test oracle_phase9 -- --ignored
```

| system | upstream `ewald()` (Ha) | this port | verdict |
|---|---|---|---|
| diamond | -28.771040577654524 | matches < 1e-9 | PASS |
| si | -102.88216217333321 | matches < 1e-9 | PASS |
| lif | -30.95510482656236 | matches < 1e-9 | PASS |
| he_fcc | -1.6174696832216189 | matches < 1e-9 | PASS |
| graphene | -44.57202102404764 | `NotYetImplemented { phase: 12 }` | DEFERRED (D-PBC-20) |

`ew_eta`-invariance over `[0.5η₀, 2η₀]` with `ew_cut` re-derived per `η`:
observed spread **< 1e-13 Ha** against the 1e-8 gate. **PASS.**

> **Correction to the criterion as written.** Holding `ew_cut` fixed while
> scaling `ew_eta` is not a physical invariance — weaker screening needs a longer
> real-space tail. Under that literal recipe UPSTREAM ITSELF drifts 8.1e-7 Ha at
> `0.5·η₀`, i.e. the criterion is unsatisfiable by any correct implementation.
> The gate therefore re-derives `ew_cut = _estimate_rcut(η², 0, 1., precision)`
> for each `η`, exactly as `get_ewald_params` does. See 09-08-SUMMARY deviation 1.

---

## 2. Reference values and their generating snippets

Every tier-2 literal in Phase 9 (D-PBC-19). Each table's exact Python is stored
next to the table itself, in the `UPSTREAM_SNIPPET` doc comment of the consuming
test — this index says where.

| plan | literals | file | generating snippet |
|---|---|---|---|
| 09-03 | `vol`, `natm`, `nao_nr`, `nelectron_pp`, `a_bohr` for all five systems | `crates/pyscf-pbc-gto/src/test_systems.rs::REFERENCES` | doc comment on `REFERENCES` |
| 09-04 | `rcut`, `ke_cutoff`, `mesh`, `mesh_at_ke100`, `nimgs`, `shell_radii`, `bas_radii`, `err_at_ke100`, all five systems | `crates/pyscf-pbc-gto/tests/cutoff.rs::UPSTREAM` | `tests/cutoff.rs::UPSTREAM_SNIPPET` |
| 09-05 | `GV_DIAMOND_555` (125×3) | `crates/pyscf-pbc-gto/tests/common/gv_reference.rs` | `tests/gv.rs::UPSTREAM_SNIPPET` |
| 09-06 | lattice-`Ls` counts and values; supercell tables | `crates/pyscf-pbc-gto/tests/lattice.rs`, `crates/pyscf-pbc-tools/tests/lattice.rs` | `UPSTREAM_SNIPPET` in each |
| 09-07 | 7 `make_kpts` tables, `kconserv` 8×8×8 and 6×6×6, `kconserv3` 4×4×4×4, `unique` | `crates/pyscf-pbc-gto/tests/common/kpts_reference.rs` | `tests/kpts_mesh.rs::UPSTREAM_SNIPPET` |
| 09-08 | `ew_eta`, `ew_cut`, `n_ls`, `mesh`, `ewald()` for all five; the `η` scan; the pseudised targets | `crates/pyscf-pbc-gto/tests/common/ewald_reference.rs` | `ewald_reference::UPSTREAM_SNIPPET` |

### Scalar values, inline

`cell.get_ewald_params()` and `cell.ewald()`, Bohr-specified geometry, **no
pseudopotential** (see the caveat below):

| system | `ew_eta` | `ew_cut` | `len(Ls)` | ewald mesh | `ewald()` (Ha) |
|---|---|---|---|---|---|
| diamond | 0.4852935502366724 | 14.69856051295752 | 321 | [9, 9, 9] | -28.771040577654524 |
| si | 0.39329641773158136 | 18.204925606997 | 177 | [11, 11, 11] | -102.88216217333321 |
| lif | 0.4565531833791103 | 15.640918598989309 | 429 | [9, 9, 9] | -30.95510482656236 |
| he_fcc | 0.5291554470071487 | 13.459295088515482 | 225 | [9, 9, 9] | -1.6174696832216189 |
| graphene | 0.26966050248398293 | 18.89726124565062 | 85 | [7, 7, 35] | -44.57202102404764 (deferred) |

`ew_eta` scan for diamond (`η₀ = 0.4852935502366724`):

| scale | `ew_cut` | upstream `ewald()` |
|---|---|---|
| 0.50 | 29.760860832192574 | -28.771040577654862 |
| 0.75 | 19.69887957045394 | -28.77104057765446 |
| 1.25 | 11.711800495983008 | -28.77104057765455 |
| 2.00 | 7.257622705598762 | -28.771040577654702 |

Plan 10-01's targets, once `pseudo='gth-pade'` is parsed
(`ewald_reference::PSEUDISED_EWALD`):

| system | charges | `ewald()` (Ha) |
|---|---|---|
| diamond | [4, 4] | -12.78712914562424 |
| si | [4, 4] | -8.398543850884348 |
| lif | [3, 7] | -20.463977469434052 |
| he_fcc | [2] | -1.6174696832216189 |
| graphene | [4, 4] | -19.80978712179894 (`ew_eta` also shifts to 0.2675469466398444) |

---

## 3. Two standing caveats every later phase must know

### 3.1 The CODATA gap — 4.951e-9 relative on every Angstrom-built lattice

`pyscf_core::Unit::Ang.length_in_au() = 1.8897261339213` (CODATA 2014) while
upstream `pyscf/data/nist.py BOHR = 0.52917721092` gives `1.8897261245650618`
(CODATA 2010). Every lattice this port builds from Angstrom input is **4.951e-9
relatively longer** than upstream's.

Consequences, and how Phase 9 handles them:

* Any quantity scaling as `1/length` inherits that relative error — for diamond's
  Ewald energy that is 1.4e-7 Ha, two orders above the 1e-9 Ha gate.
* Tier-2 tables for plans 09-07 and 09-08, and the whole of `oracle_phase9.rs`,
  therefore specify geometry in **Bohr**, so both sides start bit-identical and
  the gate measures the algorithm.
* The conversion path is still covered, in two places:
  `oracle_phase9::angstrom_lattices_match_upstream_within_the_codata_gap` asserts
  the Angstrom deviation is EXACTLY the CODATA gap (a two-sided bound — it fails
  if either side changes its constant), and
  `ewald::angstrom_reference_systems_match_upstream_within_the_unit_gap` does the
  same for energies.
* `crates/pyscf-core/src/mole.rs:483` still asserts
  `Unit::Ang.length_in_au() == 1.8897261339213`, and its doc comment's claim to
  "match upstream `pyscf/data/nist.py BOHR` (verbatim)" is **false**. Plan 09-03
  recorded this; it is a deliberate open item, not a bug to fix silently —
  changing the constant would move every molecular v1.0 regression baseline.

### 3.2 Charges are all-electron until plan 10-01

Upstream `cell.atom_charges()` returns the pseudopotential VALENCE charge when
`pseudo=` is set. This port stores the pseudopotential as a NAME only
(`Cell::pseudo_name`) until plan 10-01 lands the GTH parser (D-PBC-11), so
`atom_charges()` is the all-electron `Z`. Every Ewald reference above is
generated WITHOUT `pseudo=` so both sides agree; the pseudised targets are
committed for plan 10-01 to re-pin against.

---

## 4. Deferred branches (D-PBC-20 — a typed error, never a wrong number)

Each has a test asserting the exact phase number.

| what | upstream | error | owner |
|---|---|---|---|
| `get_Gv_weights` `inf_vacuum` (non-uniform Gauss-Chebyshev base) | `cell.py:558-578` | `NotYetImplemented { phase: 12 }` | plan 12-08 |
| `ewald` `dimension == 2` truncated Coulomb | `cell.py:773-800` | `NotYetImplemented { phase: 12 }` | plan 12-08 |
| `ewald` `dimension == 0` truncated Coulomb | `cell.py:802-808` | `InvalidMolecule` | upstream raises too |
| `particle_mesh_ewald` G-space sum | `ewald_methods.py:171-173` | `NotYetImplemented { phase: 11 }` | plan 11-01 (FFT) |
| `make_kpts` with symmetry (returns a `KPoints`) | `cell.py:874-883` | `NotYetImplemented { phase: 17 }` | plan 17 |
| every PBC PyO3 binding | — | not exposed; `python/pyscf/pbc/__init__.py` is an import-path shim | plan 20-05 (D-PBC-14) |

---

## 5. Carry-overs into Phase 10

* **`get_kconserv` k2gamma shortcut** — plan 09-07 ported `_get_kconserv_slow`
  only; the shortcut needs a module outside that plan's PORT block. Both paths
  were verified identical on the committed tables.
* **`Cell::tot_electrons`** returns the all-electron count; plan 10-01 makes it
  valence and re-pins `REFERENCES::nelectron_pp`.
* **cintx Wave 0.5** — the 10 moment-weighted families
  (`int3c1e_r{2,4,6}_origk`, `int1e_r{2,4}_origi`) are a hard prerequisite for
  Phase 10's GTH pseudopotentials. Not a Phase 9 item, but Phase 10 cannot start
  its plan 10-05 without them.
* **Ewald gradients** — `ewald_nuc_grad` / `get_ewald_direct_nuc_grad`
  (`ewald_methods.py:178-292`) are Phase 18. `ewald_real_space`, `ewald_self` and
  `ewald_g_space` are public so that plan can build on them.

---

## 6. Full command transcript

```bash
cargo build --workspace                                     # clean
cargo test --workspace                                      # all green
cargo run -p xtask --bin check-dependency-wall              # PASS
cargo run -p xtask --bin check-forbidden-paths              # PASS

cargo test -p pyscf-algebra  --test zeigh                   # 8 passed
cargo test -p pyscf-algebra  --test zoracle_determinism     # 4 passed, 1 ignored (child)
cargo test -p pyscf-pbc-tools                               # 30 passed
cargo test -p pyscf-pbc-gto  --test cell_build
cargo test -p pyscf-pbc-gto  --test cutoff
cargo test -p pyscf-pbc-gto  --test gv
cargo test -p pyscf-pbc-gto  --test lattice
cargo test -p pyscf-pbc-gto  --test kpts_mesh
cargo test -p pyscf-pbc-gto  --test ewald                   # 22 passed

PYSCF_ORACLE_VENV=1 \
  cargo test -p pyscf-pbc-gto --test oracle_phase9 -- --ignored   # 7 passed
```

Without `PYSCF_ORACLE_VENV` the oracle file reports `7 ignored` and never spawns
Python, so it is never a hard CI dependency.

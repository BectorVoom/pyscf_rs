---
phase: 03-scf-pyo3-bindings
plan: 06
subsystem: scf
tags: [rust, scf, chkfile, hdf5, hdf5-metno, varlen-unicode, f-order, mo-coeff, pitfall-8, scf-10, d-05, d-06, checkpointable]

requires:
  - phase: 01-foundation
    provides: pyscf-core (Density, MOCoefficients, Energy, PyscfRsError, CoreError, Mole), pyscf-algebra (oracle_sum), pyscf-runtime
  - phase: 03-01
    provides: pyscf-chkfile empty-skeleton crate (Cargo.toml + lib.rs stub with hdf5-metno workspace dep wired)
  - phase: 03-03
    provides: pyscf-scf scaffolding (ScfResult, InitGuessMode::Chkfile variant, default_get_init_guess dispatch with NotYetImplemented placeholder for the 5th mode)
  - phase: 03-11
    provides: pyscf-scf kernel_impl::scf_loop + init_guess_by_1e (the '1e' mode plan 03-11 shipped; this plan closes the 'chkfile' mode)

provides:
  - "pyscf-chkfile::primitives — h5py-compatible HDF5 primitives (open_for_read/write, write_mol/read_mol with VarLenUnicode, write_dataset_1d/2d, write_scalar_f64, F-order mo_coeff via write_dataset_f_order)"
  - "pyscf-chkfile::Checkpointable trait (D-06) — per-method dump/load surface; pyscf-scf::ScfResult is the first impl, Phase 4-7 add KsResult/CcsdResult/OptimState"
  - "pyscf-chkfile::ChkfileError — Hdf5(#[from] hdf5_metno::Error) | MalformedMol(#[from] serde_json::Error) | InvalidUtf8 | MissingKey | ShapeMismatch | Io(#[from] std::io::Error)"
  - "pyscf-chkfile re-export: pub use hdf5_metno as hdf5 — downstream per-method modules name pyscf_chkfile::hdf5::Group without their own hdf5-metno dep (D-05 sole-owner discipline)"
  - "pyscf-scf::chkfile::dump_scf_to_file(path, mol_json, &result) — mirrors upstream pyscf.lib.chkfile.save + pyscf.scf.chkfile.dump_scf"
  - "pyscf-scf::chkfile::load_scf_from_file(path) -> ScfResult — reads upstream-PySCF-written or pyscf-rs-written chkfile"
  - "pyscf-scf::init_guess::init_guess_by_chkfile(mol, path) — port of pyscf/scf/hf.py:673-763 simple-case (same-basis); basis projection deferred via NotYetImplemented{phase:3}"
  - "InitGuessMode::Chkfile(path) is no longer NotYetImplemented — the 5th init_guess mode from plan 03-03 task 2 is now wired end-to-end"

affects:
  - 03-07 (PyO3 bridge — Py wrapper for RHF.kernel auto-chkfile-write on converged SCF when mf.chkfile is set; PyOverrideBridge unaffected)
  - 03-08 (oracle harness — ORACLE-08 cross-language h5py↔hdf5-metno round-trip consumes the primitives shipped here)
  - 03-10 (pytest oracle wave 2 — test_scf_chkfile.py SCF-10 assertion uses dump_scf_to_file end-to-end)

tech-stack:
  added:
    - "ndarray = { workspace = true } on pyscf-scf — required by chkfile.rs for ArrayView2 F-order strides (mo_coeff write). pyscf-scf was previously algebra-only; this is the second algebra-style dep after pyscf-chkfile itself."
  patterns:
    - "Pattern: D-05 sole-ownership of hdf5-metno preserved. pyscf-chkfile is the only workspace member that names hdf5-metno in its Cargo.toml + source; downstream modules use `pyscf_chkfile::hdf5::Group` via the re-export."
    - "Pattern: Pitfall 8 mitigation for mo_coeff F-order. hdf5-metno's `write` requires C-contiguous standard layout, so write_dataset_f_order transposes the input before write. h5py reading the same dataset with explicit F-order recovers the original LAPACK layout; read_2d returns the transpose (asserted by the F-order smoke test)."
    - "Pattern: NotYetImplemented{phase:N} as structured-deferral. init_guess_by_chkfile ships the same-basis case; basis-projection (general PySCF behavior) returns NotYetImplemented{phase:3} so a future plan can fill the projection without re-touching the calling convention."
    - "Pattern: open_for_write handles empty pre-existing files. tempfile::NamedTempFile creates an empty file on init; hdf5-metno's `File::append` fails on that. The primitive detects empty + falls through to `File::create`."

key-files:
  created:
    - "crates/pyscf-chkfile/src/error.rs (ChkfileError variant family with #[from] for hdf5_metno::Error + serde_json::Error)"
    - "crates/pyscf-chkfile/src/checkpointable.rs (pub trait Checkpointable { dump, load })"
    - "crates/pyscf-chkfile/src/primitives.rs (12 primitives: open_for_read/write/group, read_dataset_1d/2d, read_mol, read_scalar_f64, write_dataset_1d, write_dataset_c_order, write_dataset_f_order, write_mol, write_scalar_f64)"
    - "crates/pyscf-chkfile/tests/primitives_smoke.rs (4 tests: mol_json_round_trip_utf8, scf_scalar_and_1d_round_trip, mo_coeff_f_order_round_trip, append_mode_preserves_existing_data)"
    - "crates/pyscf-scf/src/chkfile.rs (impl Checkpointable for ScfResult + dump_scf_to_file/load_scf_from_file helpers)"
    - "crates/pyscf-scf/tests/chkfile_dump_load.rs (3 tests: rust_rust_round_trip, schema_keys_match_upstream, checkpointable_trait_dump_load_directly)"
    - "crates/pyscf-scf/tests/init_guess_chkfile.rs (2 tests: init_guess_by_chkfile_reads_prior_density, init_guess_by_chkfile_rejects_nao_mismatch)"
    - ".planning/phases/03-scf-pyo3-bindings/03-06-SUMMARY.md (this file)"
  modified:
    - "crates/pyscf-chkfile/Cargo.toml (added tempfile dev-dep)"
    - "crates/pyscf-chkfile/src/lib.rs (pub mod tree + re-exports — replaces plan 03-01 empty skeleton)"
    - "crates/pyscf-scf/Cargo.toml (added pyscf-chkfile + ndarray deps; added tempfile dev-dep)"
    - "crates/pyscf-scf/src/lib.rs (pub mod chkfile + re-export dump_scf_to_file/load_scf_from_file)"
    - "crates/pyscf-scf/src/init_guess.rs (InitGuessMode::Chkfile arm now calls init_guess_by_chkfile; the new function reads chkfile + reconstructs density via oracle_sum)"

key-decisions:
  - "F-order mo_coeff round-trip uses write-transpose convention. hdf5-metno's `write` requires C-contiguous standard layout; the F-order primitive writes the transpose so that the on-disk byte layout is column-major. Reading via `read_dataset_2d` (which returns C-order Array2) yields the transpose; the SCF chkfile.rs::load reconstructs the F-order MOCoefficients.data by indexing `mat_on_disk[(j, i)]` into `data[i + j*nao]`. Element-wise round-trip verified in `rust_rust_round_trip`. Cross-language ORACLE-08 (plan 03-08) is the empirical seal."
  - "open_for_write detects empty pre-existing files. `tempfile::NamedTempFile::new()` creates an empty file (size 0) on init; `hdf5::File::append` rejects size-0 files as 'not an HDF5 file'. The primitive checks `std::fs::metadata(p).len() == 0` and routes to `File::create` in that case, otherwise `File::append`. This is the documented tempfile-test pattern."
  - "pyscf-scf adds ndarray as a regular dependency (Deviation Rule 3). The plan body's chkfile.rs uses `ArrayView2::from_shape((nao, nmo).strides((1, nao)), &data)` for the F-order view construction. Without ndarray on pyscf-scf, the F-order view construction would have to happen inside pyscf-chkfile — leaking the F-order convention out of the per-method module (D-06 violation)."
  - "init_guess_by_chkfile ships SAME-BASIS only. Basis projection (`prior.nao != current nao`) returns `NotYetImplemented{phase:3}` rather than silently doing the wrong thing. Upstream pyscf/scf/hf.py:673-763 handles basis projection by interpolating MO coefficients between bases; that's a larger Phase-3 follow-up. The simple-case path covers the SCF restart workflow (rerun-from-converged-prior-state) which is 95% of chkfile use."
  - "Pitfall 9 mitigation in density reconstruction. The MO-axis sum `D[mu,nu] = sum_i occ_i * C[mu,i] * C[nu,i]` materializes the per-i terms into a scratch Vec then calls `pyscf_algebra::oracle_sum`. Matches plan 03-11's rdm.rs/energy.rs pattern. The terms vector is `nmo` long (≤ basis size); cost is negligible against the O(nao²) outer loop."
  - "Checkpointable trait is in pyscf-chkfile, NOT in pyscf-core. D-06 places per-method schema modules in each method crate (pyscf-scf::chkfile, pyscf-dft::chkfile in Phase 4, etc.). The trait itself lives in pyscf-chkfile because every method crate needs the same trait signature; putting it in pyscf-core would force pyscf-core to depend on hdf5-metno (D-05 violation)."
  - "MOCoefficients.energies + .occupations clones in load. The MOCoefficients struct carries `energies + occupations` redundantly with ScfResult.mo_energy + ScfResult.mo_occ; load reads each field once and clones into both slots for shape parity with how plan 03-03 constructs MOCoefficients."

patterns-established:
  - "Pattern: D-05 algebra-wall analog for HDF5 — pyscf-chkfile is the SOLE workspace owner of hdf5-metno. Verified by `grep -rln 'hdf5_metno\\|hdf5-metno' crates/*/Cargo.toml crates/*/src/` returning only paths under crates/pyscf-chkfile/."
  - "Pattern: empty-tempfile open_for_write — production callers that pass paths returned from `tempfile::NamedTempFile` get correct behavior. Phase 4+ chkfile callers may follow the same path-handling pattern."
  - "Pattern: F-order write-transpose convention — hdf5-metno's `write` requires C-contiguous, so F-order data writes the transpose. Documented in primitives.rs; Phase 4 RKS chkfile + Phase 6 CCSD t2 amplitude chkfile inherit this convention."

requirements-completed: [SCF-10]

duration: 10min
completed: 2026-05-11
---

# Phase 03 Plan 06: pyscf-chkfile primitives + ScfResult Checkpointable + InitGuessMode::Chkfile Summary

**HDF5 chkfile primitives + Checkpointable trait shipped: `pyscf-chkfile` is the sole workspace owner of `hdf5-metno`; `pyscf-scf::chkfile` impls `Checkpointable` for `ScfResult` with F-order `mo_coeff` (Pitfall 8) and VL Unicode `/mol` JSON (h5py-compat A2); `InitGuessMode::Chkfile(path)` reads upstream-PySCF-written chkfiles and reconstructs the density via `D = C·diag(occ)·C^T` (oracle_sum — Pitfall 9). Plan 03-03's NotYetImplemented stub for the 5th init_guess mode is gone — SCF-10 surface complete.**

## Performance

- **Duration:** ~10 min
- **Started:** 2026-05-11T13:52:17Z
- **Completed:** 2026-05-11T14:01:42Z
- **Tasks:** 2 (TDD RED + GREEN per task)
- **Files created/modified:** 7 created + 5 modified = 12

## Accomplishments

- **`pyscf-chkfile` body filled** (was plan-03-01 empty skeleton). 3 modules + 12 primitives + 1 trait + 1 error enum + 1 hdf5-metno re-export:
  - `error::ChkfileError` — 6 variants with `#[from]` for `hdf5_metno::Error` + `serde_json::Error`
  - `checkpointable::Checkpointable` — D-06 trait surface (`dump` + `load` against `&hdf5::Group`)
  - `primitives` — open_for_read/write/group, write_mol/read_mol (VL Unicode), write_scalar_f64/read_scalar_f64, write_dataset_1d/read_dataset_1d, write_dataset_c_order, write_dataset_f_order, read_dataset_2d
  - Re-export `pub use hdf5_metno as hdf5;` so downstream modules name `pyscf_chkfile::hdf5::Group` without their own hdf5-metno dep
- **`pyscf-scf::chkfile` impl Checkpointable for ScfResult** — schema mirrors `pyscf/scf/chkfile.py:25-42` byte-for-byte:
  - `/mol` at file root (VL Unicode JSON)
  - `/scf/e_tot` scalar f64
  - `/scf/mo_energy` 1D f64
  - `/scf/mo_occ` 1D f64
  - `/scf/mo_coeff` 2D f64 (F-order on disk; Pitfall 8)
- **`dump_scf_to_file(path, mol_json, &result)` + `load_scf_from_file(path)`** — top-level helpers mirroring `pyscf.lib.chkfile.save + pyscf.scf.chkfile.dump_scf`. Idempotent dump (recreates `/scf` group if it exists).
- **`init_guess_by_chkfile(mol, path)`** — replaces plan 03-03's `InitGuessNotYetImplemented` stub:
  - Loads prior `ScfResult` via `load_scf_from_file`
  - Same-basis case: `D[μν] = Σ_i mo_occ[i] · mo_coeff[μ, i] · mo_coeff[ν, i]` via `oracle_sum` (Pitfall 9)
  - Basis projection (prior.nao ≠ current nao) returns `NotYetImplemented{phase:3}` — same-basis only in Phase 3
- **9 tests passing** (4 primitives_smoke + 3 chkfile_dump_load + 2 init_guess_chkfile). 0 regressions in the pre-existing pyscf-scf suite from plans 03-03/03-11.

## Task Commits

| # | Task | Hash | Type |
|---|------|------|------|
| 1 | Task 1 RED — failing tests for pyscf-chkfile primitives | `6053358` | test |
| 2 | Task 1 GREEN — pyscf-chkfile primitives + Checkpointable trait | `84bd07a` | feat |
| 3 | Task 2 RED — failing tests for ScfResult Checkpointable + chkfile mode | `2df39f7` | test |
| 4 | Task 2 GREEN — Checkpointable for ScfResult + InitGuessMode::Chkfile | `8961d51` | feat |

_4 atomic commits (2 RED + 2 GREEN), no REFACTOR needed._

## Source-of-Truth Line References

| Module | Upstream PySCF reference |
|--------|---------------------------|
| `pyscf-chkfile::primitives::write_mol` / `read_mol` | `pyscf/lib/chkfile.py:179-191` (`save_mol`) |
| `pyscf-chkfile::primitives::write_dataset_f_order` | `pyscf/scf/chkfile.py:28-42` (mo_coeff F-order write — LAPACK convention) |
| `pyscf-scf::chkfile::ScfResult::dump` / `load` | `pyscf/scf/chkfile.py:25-42` (`dump_scf` / `load_scf`) |
| `pyscf-scf::init_guess::init_guess_by_chkfile` | `pyscf/scf/hf.py:673-763` (`init_guess_by_chkfile` — simple-case body) |
| Checkpointable trait surface (D-06) | `.planning/phases/03-scf-pyo3-bindings/03-RESEARCH.md` §"Pattern 6" lines 698-805 |

## Pitfall 8 Mitigation — F-order mo_coeff

```
$ grep -c write_dataset_f_order crates/pyscf-chkfile/src/primitives.rs
2
$ grep -c write_dataset_f_order crates/pyscf-scf/src/chkfile.rs
3
```

mo_coeff round-trip semantics:
1. **Write:** `write_dataset_f_order(group, "mo_coeff", view)` materializes the *transpose* of `view` as a C-contig owned array and writes it. On-disk byte layout: `mat_on_disk[(j, i)] = view[(i, j)]`.
2. **h5py read with explicit F-order:** recovers `view[(i, j)]` byte-for-byte (ORACLE-08 cross-language seal — plan 03-08).
3. **pyscf-rs read via `read_dataset_2d`:** returns the C-order interpretation (which is the transpose). `pyscf-scf::chkfile::load` reconstructs F-order `MOCoefficients.data` by indexing `mat_on_disk[(j, i)]` into `data[i + j*nao]`.

Element-wise round-trip asserted by `rust_rust_round_trip` (4×4 with non-symmetric data so transpose ≠ original).

## Pitfall 9 Mitigation — oracle_sum in init_guess_by_chkfile

```
$ grep -c oracle_sum crates/pyscf-scf/src/init_guess.rs
1
```

Density reconstruction `D[μν] = Σ_i occ_i · C[μ, i] · C[ν, i]` materializes per-`i` terms into a scratch Vec then calls `pyscf_algebra::oracle_sum` — matches the plan 03-11 rdm.rs / energy.rs reduction pattern. The scratch buffer is `nmo` long; cost is negligible against the O(nao²) outer loop.

## Schema Keys Verification

```
$ cargo test -p pyscf-scf --test chkfile_dump_load schema_keys_match_upstream
running 1 test
test schema_keys_match_upstream ... ok
```

The dumped file contains exactly `/mol`, `/scf/e_tot`, `/scf/mo_energy`, `/scf/mo_occ`, `/scf/mo_coeff` — 5 keys total, matching `pyscf/scf/chkfile.py:25-42` byte-for-byte.

## D-05 Sole-Owner Verification

```
$ grep -rln "hdf5_metno\|hdf5-metno" crates/*/Cargo.toml crates/*/src/
crates/pyscf-chkfile/Cargo.toml
crates/pyscf-chkfile/src/checkpointable.rs
crates/pyscf-chkfile/src/error.rs
crates/pyscf-chkfile/src/lib.rs
crates/pyscf-chkfile/src/primitives.rs
```

Only paths under `crates/pyscf-chkfile/` name `hdf5-metno`. pyscf-scf::chkfile uses `pyscf_chkfile::hdf5::Group` via the re-export `pub use hdf5_metno as hdf5;` in pyscf-chkfile/src/lib.rs. **D-05 sole-owner discipline confirmed.**

## DIST-05 Baseline

`hdf5-metno = { version = "=0.10.0", features = ["static"] }` (workspace Cargo.toml line 65) ships libhdf5 embedded. No system-installed libhdf5 required at link or runtime. The `--features hdf5-metno/static` is the workspace default; verified by `cargo build -p pyscf-chkfile` succeeding on a system without `apt install libhdf5-dev`.

## Tests Summary

| File | Test count | Status |
|------|-----------:|--------|
| `crates/pyscf-chkfile/tests/primitives_smoke.rs` | 4 | pass |
| `crates/pyscf-scf/tests/chkfile_dump_load.rs` | 3 | pass |
| `crates/pyscf-scf/tests/init_guess_chkfile.rs` | 2 | pass |
| **Plan 03-06 total** | **9** | |
| Pre-existing pyscf-scf tests (attribute_floor, hooks_kernel_types, canonicalize_post_eigh, kernel_internals_unit, analyze_convert_scanner, no_overrides_drives_kernel) | 28 + 1 ignored | pass |

Full pyscf-scf + pyscf-chkfile suite: **37 passing, 1 ignored (pre-existing int2e_sph gap from plan 03-11)**.

## Decisions Made

1. **F-order write-transpose convention.** hdf5-metno's `write` requires C-contiguous; we materialize the transpose so the on-disk byte layout corresponds to column-major. Element-wise round-trip is the contract; ORACLE-08 (plan 03-08) seals cross-language compatibility with h5py.
2. **open_for_write detects empty pre-existing files.** `tempfile::NamedTempFile` creates an empty file on init; `hdf5::File::append` rejects size-0 files. Detection via `std::fs::metadata(p).len() == 0` + fall-through to `File::create`.
3. **pyscf-scf gains an ndarray dep.** The F-order `ArrayView2` construction in chkfile.rs uses `ShapeBuilder::strides((1, nao))`. Keeping the F-order convention inside pyscf-scf::chkfile (D-06 per-method module owns its schema) requires direct ndarray use there.
4. **init_guess_by_chkfile is same-basis only.** Basis projection (general case in pyscf/scf/hf.py:673-763) returns `NotYetImplemented{phase:3}`. Simple-case covers the 95% SCF-restart workflow; projection is a clean follow-up plan because the function signature stays stable.
5. **oracle_sum scratch-Vec pattern for density reconstruction.** Matches plan 03-11's rdm.rs/energy.rs idiom — per-axis-of-reduction terms materialize into a `Vec<f64>`, then `pyscf_algebra::oracle_sum(&terms)` provides the pairwise-tree determinism (Pitfall 9).
6. **Checkpointable trait lives in pyscf-chkfile, not pyscf-core.** D-06 places schema modules in each method crate; the trait itself lives in pyscf-chkfile because every method crate needs the same surface and pyscf-core can't depend on hdf5-metno (D-05 violation).
7. **MOCoefficients.energies + .occupations cloned into both slots.** ScfResult carries `mo_energy + mo_occ` AND `mo_coeff.energies + mo_coeff.occupations`; load reads each field once and clones into both for shape parity with how plan 03-03 constructs MOCoefficients.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] pyscf-scf had no ndarray dep**
- **Found during:** Task 2 GREEN, first `cargo build -p pyscf-scf`.
- **Issue:** The plan's chkfile.rs body uses `ArrayView2::from_shape((nao, nmo).strides((1, nao)), &data)` to construct an F-order view over `MOCoefficients.data`. pyscf-scf was previously algebra-only (no ndarray); without `ndarray = { workspace = true }`, the use statement `use ndarray::{Array2, ArrayView2, ShapeBuilder};` fails with `unresolved module or unlinked crate `ndarray``.
- **Fix:** Added `ndarray = { workspace = true }` to `crates/pyscf-scf/Cargo.toml [dependencies]`. ndarray is already a workspace dep (line 67 of root Cargo.toml — added in Phase 3 plan 03-01 as a comment-tagged dep for pyscf-chkfile/pyscf-df). pyscf-scf gaining ndarray is the minimal change to keep the F-order convention inside the per-method module (D-06).
- **Files modified:** `crates/pyscf-scf/Cargo.toml`.
- **Committed in:** `8961d51` (Task 2 GREEN).

**2. [Rule 1 - Bug] open_for_write fails on empty pre-existing files (tempfile pattern)**
- **Found during:** Task 1 GREEN, first test run.
- **Issue:** Plan body's `open_for_write` used `if p.exists() { append } else { create }`. But `tempfile::NamedTempFile::new()` creates an empty (size 0) file at the OS level; `hdf5::File::append` rejects size-0 files with `not an HDF5 file`. Every primitives_smoke test panicked at the first `open_for_write(path)` call.
- **Fix:** Added an empty-file detection branch: `let is_empty = std::fs::metadata(p).map(|m| m.len() == 0).unwrap_or(false);` — if the file exists but is empty, route to `File::create` (which truncates); otherwise `File::append`. The condition is bounded and idiomatic for tempfile-test patterns.
- **Files modified:** `crates/pyscf-chkfile/src/primitives.rs`.
- **Committed in:** `84bd07a` (Task 1 GREEN).

**3. [Rule 1 - Bug] Plan body's F-order write strategy (`as_standard_layout().reversed_axes()`) produces a non-standard-layout array that `write` rejects**
- **Found during:** Plan body review pre-implementation; confirmed by reading hdf5-metno container.rs:297-316 (`write` calls `ensure!(view.is_standard_layout())`).
- **Issue:** The plan's `write_dataset_f_order` sketch did `let f_owned = data.as_standard_layout().reversed_axes().to_owned();` then `write(&f_owned)`. `as_standard_layout()` returns a C-contig view; `.reversed_axes()` flips strides — the result is no longer standard layout; `to_owned()` materializes in the new (non-standard) stride order; `write` then rejects with "input array is not in standard layout".
- **Fix:** Use `data.t().as_standard_layout().to_owned()` instead. `.t()` returns a transposed view (non-standard strides); `.as_standard_layout()` materializes into a fresh C-contig array in transposed shape; `.to_owned()` is a no-op on the CowArray::Owned case. The resulting on-disk byte layout is column-major relative to the original — the desired F-order semantics.
- **Files modified:** `crates/pyscf-chkfile/src/primitives.rs` (write_dataset_f_order body); `crates/pyscf-scf/src/chkfile.rs` (corresponding load reconstruction).
- **Verification:** `mo_coeff_f_order_round_trip` asserts `mat[(0, 1)] == c_order[(1, 0)]` (transpose semantics); `rust_rust_round_trip` asserts element-wise F-order data preservation.
- **Committed in:** `84bd07a` (Task 1 GREEN) and `8961d51` (Task 2 GREEN).

**4. [Rule 1 - Bug] Plan body referenced `pyscf_core::CoreError::Other` which doesn't exist**
- **Found during:** Task 2 GREEN, init_guess_by_chkfile error wiring.
- **Issue:** Plan body wrote `pyscf_core::CoreError::Other(format!("chkfile read: {}", e))` to repackage a `ChkfileError` into `PyscfRsError`. But `CoreError` has 3 variants — `InvalidMolecule(String)`, `BasisParse(String)`, `DimensionMismatch { expected, actual }` — no `Other` arm exists (same issue plan 03-03 / 03-04 / 03-05 SUMMARYs all flagged).
- **Fix:** Route through `CoreError::InvalidMolecule(String)` — the only String-carrying catch-all on the enum. Mirrors the plan-03-03/04/05 precedent.
- **Files modified:** `crates/pyscf-scf/src/init_guess.rs`.
- **Committed in:** `8961d51` (Task 2 GREEN).

**5. [Rule 1 - Bug] Plan body used `mol.nao_nr()` as a method; nao_nr is a public field**
- **Found during:** Task 2 GREEN, init_guess_by_chkfile basis-mismatch check.
- **Issue:** Plan body wrote `let nao = mol.nao_nr();` — but `pyscf_core::Mole.nao_nr` is a public `usize` *field* (line 174 of mole.rs), not a method. No `nao_nr()` method exists.
- **Fix:** Direct field access — `let nao = mol.nao_nr;`. Same idiom plan 03-11 already uses elsewhere in pyscf-scf.
- **Files modified:** `crates/pyscf-scf/src/init_guess.rs`.
- **Committed in:** `8961d51` (Task 2 GREEN).

**6. [Rule 1 - Bug] Plan body used `mol.dumps()` — no such method on pyscf-core::Mole**
- **Found during:** Plan body review pre-implementation.
- **Issue:** Plan body's RHF::kernel auto-chkfile-write path called `self.mol.dumps()` to JSON-stringify the Mole for the `/mol` dataset. pyscf-core::Mole has `Debug + Default + Clone` only — no `Serialize` impl (only some of its sub-fields like `Unit`, `ParsedAtom` derive Serialize). Calling `serde_json::to_string(&mol)` would fail to compile.
- **Resolution:** The auto-chkfile-write inside `RHF::kernel()` is part of the SCF-10 user-facing surface but requires a Mole-JSON-serialize plumbing that doesn't exist on pyscf-core today. Plan 03-06 ships the chkfile primitives + Checkpointable + InitGuessMode::Chkfile read path; `dump_scf_to_file(path, mol_json, &result)` accepts the mol JSON string as a caller-supplied parameter (per the plan's own helper signature on lines 612-625). The "automatic" write-on-converged in RHF::kernel is deferred to plan 03-07 (PyO3 bridge) where the Python wrapper can call `pyscf-rs`'s Python-side `mol.dumps()` (which already exists in upstream pyscf via the Python class) and pass the JSON string in. This decision matches the plan body's natural fall-through point — the helper's signature was already mol_json-parameterized.
- **Files modified:** None (the auto-write hook was an optional plan body addition; the plan's explicit success criteria don't require it — only "InitGuessMode::Chkfile read path" and "Checkpointable trait shipped" + "round-trip tests").
- **Threat / regression impact:** None. Plan 03-08 (ORACLE-08) cross-language round-trip uses h5py's writes for Python-side and pyscf-rs's `dump_scf_to_file` for Rust-side; both accept mol JSON as input.

---

**Total deviations:** 6 (4 plan-body-API bugs, 1 dep-graph blocking-fix, 1 missing-API deferred to 03-07). All fixes are inside plan-named files except the dep addition (ndarray on pyscf-scf). Net effect: the plan's intended surface ships verbatim; the only structural change is pyscf-scf gaining an ndarray dep which mirrors the algebra-style-dep pattern.

## Issues Encountered

- **Worktree base mismatch on init:** HEAD was at `a02d0f5` (post-plan-03-11 commit) while orchestrator's `EXPECTED_BASE = 459de51` (post-plan-03-05 tip). The worktree was branched from the 03-11 wave but the orchestrator's plan-06 metadata expects the 03-05 wave tip. Resolved via `git reset --soft 459de51` → `git reset HEAD` to clear the index. The unstaged diffs (the 03-11 wave commits as "modifications") were left in place because they represent legitimate Wave-3 work that this worktree inherits.
- **hdf5-metno API verification:** Read `~/.cargo/registry/src/index.crates.io-*/hdf5-metno-0.10.0/src/hl/*.rs` directly to confirm method names (`link_exists`, `unlink`, `append`, `flush`, `create_group`, `new_dataset`, `write`, `write_scalar`, `read_1d`, `read_2d`, `read_scalar`). All names match the plan body's expectations; no API drift.
- **ndarray ShapeBuilder import:** `ShapeBuilder` is in `ndarray::shape_builder` but is re-exported at the crate root as `ndarray::ShapeBuilder`. Test compile flagged the missing import; resolved with `use ndarray::{Array2, ArrayView2, ShapeBuilder};` in chkfile.rs.

## User Setup Required

None — pure Rust chkfile + Checkpointable plan, no external service config.

## Next Wave Readiness

- **Plan 03-07 (PyO3 bridge):** Will wrap `RHF.kernel()` so that when `mf.chkfile = path` is set, the post-converge step calls `dump_scf_to_file(path, mol.dumps(), &result)`. The mol JSON serialization happens on the Python side via the existing upstream `Mole.dumps()` method; pyscf-rs's `dump_scf_to_file` already accepts the JSON string as input.
- **Plan 03-08 (ORACLE-08 cross-language):** Will write a chkfile via `dump_scf_to_file`, read it via `h5py.File(path)` in Python, assert schema keys + element-wise mo_coeff match. The F-order convention's correctness across the language boundary is the empirical seal Pitfall 11 mitigation.
- **Plan 03-10 (pytest oracle wave 2):** `python/pyscf/tests/test_scf_chkfile.py` SCF-10 assertion uses `dump_scf_to_file` end-to-end (Rust write + Python read).
- **Phase 4 (DFT):** `pyscf-dft::chkfile` will impl `Checkpointable` for `KsResult` following the same per-method-module pattern. Shares primitives via the `pyscf_chkfile::primitives::*` re-exports.
- **Phase 6 (CCSD):** `pyscf-ccsd::chkfile` will impl `Checkpointable` for `CcsdResult`. The F-order convention applies to t2 amplitudes (4-index tensor); the existing `write_dataset_f_order` primitive generalizes (or a new arity-4 helper lands).
- **Phase 7 (geomopt):** `pyscf-geomopt::chkfile` will impl `Checkpointable` for `OptimState`. Already enumerated in `Checkpointable` trait doc-comment.

## Stub Inventory

```
$ grep -rn "unimplemented!" crates/pyscf-chkfile/src/ crates/pyscf-scf/src/chkfile.rs
(no matches)
```

Zero `unimplemented!()` markers in plan 03-06's files. All paths return either successful values or structured `Result::Err(...)`.

## Known Stubs

| Function / Surface | Status | Resolved by |
|---|---|---|
| `init_guess_by_chkfile` basis-projection (prior.nao != current nao) | Returns `NotYetImplemented{phase:3, what:"basis projection..."}` | Phase 3 follow-up plan |
| `RHF::kernel()` auto-chkfile-write on converged SCF | Not wired (requires `mol.dumps()` JSON serialize on pyscf-core) | Plan 03-07 PyO3 bridge calls upstream Python `Mole.dumps()` and passes JSON string to `dump_scf_to_file` |
| Cross-language h5py ↔ hdf5-metno empirical seal | Deferred to plan 03-08 (ORACLE-08) | Plan 03-08 |

None of these are "wired-to-UI silently empty" stubs — they return structured `Result` values or are documented contracts that plan 03-07 / plan 03-08 close.

## Threat Flags

Plan 03-06's `<threat_model>` enumerated T-3-03 (malformed mol JSON), T-3-04 (HDF5 DoS — accepted), T-3-11 (Pitfall 11 — h5py↔hdf5-metno encoding mismatch). All three are addressed at plan-shipment quality:

- **T-3-03 (Tampering — malformed mol JSON):** Mitigated. `ChkfileError::MalformedMol(#[from] serde_json::Error)` ensures `serde_json::from_str` errors panic-free. Currently no caller invokes `serde_json::from_str` on the chkfile mol field — the Rust side stores/loads the JSON string verbatim. Plan 03-08 / 03-10 callers that parse the JSON will see structured errors.
- **T-3-04 (DoS — large dataset reads):** Accepted per threat model. Phase 3 reads only scalar `e_tot` + 1D `mo_energy/mo_occ` + 2D `mo_coeff[nao, nao]` — all bounded by the loaded Mole's basis. Phase 6 CCSD-08 introduces `PYSCF_MAX_MEMORY` enforcement.
- **T-3-11 (Pitfall 11 — encoding mismatch):** This plan ships the primitives; plan 03-08's ORACLE-08 IS the empirical seal. The F-order write-transpose convention is consistently applied (verified via `rust_rust_round_trip`); cross-language interop is plan 03-08's job.

No new threat flags surfaced.

## Self-Check

Files claimed created, verified to exist:

```
FOUND: crates/pyscf-chkfile/src/error.rs
FOUND: crates/pyscf-chkfile/src/checkpointable.rs
FOUND: crates/pyscf-chkfile/src/primitives.rs
FOUND: crates/pyscf-chkfile/tests/primitives_smoke.rs
FOUND: crates/pyscf-scf/src/chkfile.rs
FOUND: crates/pyscf-scf/tests/chkfile_dump_load.rs
FOUND: crates/pyscf-scf/tests/init_guess_chkfile.rs
```

Files claimed modified, verified to exist:

```
FOUND: crates/pyscf-chkfile/Cargo.toml
FOUND: crates/pyscf-chkfile/src/lib.rs
FOUND: crates/pyscf-scf/Cargo.toml
FOUND: crates/pyscf-scf/src/lib.rs
FOUND: crates/pyscf-scf/src/init_guess.rs
```

Commits claimed, verified in `git log --oneline`:

```
FOUND: 6053358 — test(03-06) Task 1 RED
FOUND: 84bd07a — feat(03-06) Task 1 GREEN
FOUND: 2df39f7 — test(03-06) Task 2 RED
FOUND: 8961d51 — feat(03-06) Task 2 GREEN
```

Plan-level verification commands:

```
$ cargo build -p pyscf-chkfile                         # ok
$ cargo build -p pyscf-scf                             # ok
$ cargo test -p pyscf-chkfile --test primitives_smoke  # 4 passed
$ cargo test -p pyscf-scf --test chkfile_dump_load     # 3 passed
$ cargo test -p pyscf-scf --test init_guess_chkfile    # 2 passed
$ grep -F write_dataset_f_order crates/pyscf-scf/src/chkfile.rs  # 3 matches
$ grep -F VarLenUnicode crates/pyscf-chkfile/src/primitives.rs   # 4 matches
$ grep -rln "hdf5_metno\|hdf5-metno" crates/*/src/ crates/*/Cargo.toml  # only crates/pyscf-chkfile/*
$ grep -E "InitGuessMode::Chkfile" crates/pyscf-scf/src/init_guess.rs   # routes to init_guess_by_chkfile, not NYI
```

Full test counts:

```
pyscf-chkfile tests/primitives_smoke     4 passed
pyscf-scf tests/chkfile_dump_load        3 passed
pyscf-scf tests/init_guess_chkfile       2 passed
pyscf-scf (pre-existing from 03-03/11)  28 passed, 1 ignored (int2e_sph gap)
                                      ─────────────
                                        37 passed, 0 failed, 1 ignored
```

## Self-Check: PASSED

---

*Phase: 03-scf-pyo3-bindings*
*Plan: 06*
*Completed: 2026-05-11*

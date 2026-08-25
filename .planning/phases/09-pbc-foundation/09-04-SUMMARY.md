---
phase: 09-pbc-foundation
plan: 04
subsystem: pbc-gto / pbc-tools
tags: [pbc, rcut, ke-cutoff, mesh, nimgs, bounding-sphere, pgf-rcut, qr]

# Dependency graph
requires:
  - phase: 09-pbc-foundation
    plan: 01
    provides: "the pyscf-pbc-gto / pyscf-pbc-tools crate scaffolds + path-scoped lint exemptions"
  - phase: 09-pbc-foundation
    plan: 03
    provides: "Cell (OWNS a Mole, Derefs to it), lattice_vectors / vol / reciprocal_vectors, Cell::build, the five §9.2 reference systems"
provides:
  - "pyscf_pbc_gto::cutoff — estimate_rcut, bas_rcut, _estimate_rcut, _extract_pgto_params, estimate_ke_cutoff, _estimate_ke_cutoff, error_for_ke_cutoff, get_bounding_sphere, get_nimgs, pgf_rcut (+ its C twin), rcut_by_shells, _mesh_inf_vaccum, estimate_mesh"
  - "pyscf_pbc_tools::mesh — cutoff_to_mesh, mesh_to_cutoff, cutoff_to_gs, gs_to_cutoff, qr_heights, qr_r22_abs"
  - "pyscf_pbc_tools::mat3 — det3 / transpose3 / inv3 / dot3 / cross3 / norm3, the single lattice-algebra owner (pyscf_pbc_gto::cell re-exports the first three)"
  - "Cell::build now fills rcut and mesh — the 09-03 RCUT_UNSET / MESH_UNSET sentinels no longer survive a build"
  - "Cell::cutoff_to_mesh / Cell::nimgs / Cell::rcut_by_shells / Cell::bas_rcut methods"
  - "Cell::use_loose_rcut (+ CellBuildArgs / CellPack round-trip)"
  - "the dimension<=2 vacuum-size warning (09-03 carry-over, closed)"
affects: [09-05, 09-06, 09-07, 09-08, 09-09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "the 3x3 lattice algebra lives in the LOWEST crate that needs it (pyscf-pbc-tools) and is re-exported upward, so there is one inversion in the workspace"
    - "port BOTH the Python function and the C twin when upstream's Python entry point actually dispatches to libpbc (pgf_rcut / pgf_rcut_c)"
    - "tier-1 invariants that pin a value to a closed form (|R22| == |det|/||c0 x c1||, qr_heights == 2*pi/||a_i||) so a reference-value test cannot be the only gate"

key-files:
  created:
    - crates/pyscf-pbc-gto/src/cutoff.rs
    - crates/pyscf-pbc-gto/tests/cutoff.rs
    - crates/pyscf-pbc-tools/src/mesh.rs
    - crates/pyscf-pbc-tools/src/mat3.rs
    - crates/pyscf-pbc-tools/tests/mesh.rs
  modified:
    - crates/pyscf-pbc-gto/src/cell.rs
    - crates/pyscf-pbc-gto/src/lib.rs
    - crates/pyscf-pbc-gto/src/types.rs
    - crates/pyscf-pbc-gto/src/dumps_loads.rs
    - crates/pyscf-pbc-gto/tests/cell_build.rs
    - crates/pyscf-pbc-gto/tests/common/systems.rs
    - crates/pyscf-pbc-tools/src/lib.rs

key-decisions:
  - "det3/transpose3/inv3 MOVED from pyscf-pbc-gto::cell to pyscf-pbc-tools::mat3 and re-exported, because `cutoff_to_mesh` needs `2*pi*inv(a.T)` and the dependency edge runs gto -> tools. One lattice inversion, no drift."
  - "`np.linalg.qr(...)[1][2,2]` is ported as a 3x3 Householder QR in the same order LAPACK dgeqrf applies its reflections; only |R[2,2]| is ever observable, so the sign convention never surfaces."
  - "BOTH pgf_rcut variants are ported: the Python one (cell.py:974-991) and the C one (lib/pbc/cell.c:30-59, with the `gmax < precision` early return). `rcut_by_shells` uses the C twin, because that is what upstream's `cell.rcut_by_shells` calls through libpbc."
  - "`use_loose_rcut` added to Cell / CellBuildArgs / CellPack — `estimate_rcut` branches on it (cell.py:430-431), so leaving it out would have made `rcut_by_shells` unreachable from the estimator."
  - "The plan's guessed acceptance numbers (rcut ~ 15.6, mesh == [15,15,15]) were REGENERATED as the plan instructs; the true upstream values are rcut = 21.319400521777592 and cutoff_to_mesh(a, 100) == [23,23,23]."

patterns-established:
  - "A reference-value test file records the exact generating Python snippet in a `const UPSTREAM_SNIPPET` doc comment (D-PBC-19)"
  - "Tolerance is chosen by PROVENANCE: basis-only quantities at 1e-12 relative, lattice-derived floats at 1e-7 (the known Unit::Ang gap), lattice-derived integers exactly"

requirements-completed: [PBC-GTO-03]

# Metrics
duration: ~1h
completed: 2026-08-25
---

# Phase 9 Plan 04: Cutoffs, `rcut`, mesh

**`pyscf-pbc-gto` gains `cutoff.rs` — every estimator of `pyscf/pbc/gto/cell.py:373-523`
and `:968-1025`, ported line by line — and `pyscf-pbc-tools` gains `mesh.rs`
(`tools/pbc.py:787-836`) plus `mat3.rs`. `Cell::build` now computes `rcut` and `mesh`,
closing plan 09-03's `RCUT_UNSET` / `MESH_UNSET` gap. All five §9.2 reference systems
reproduce upstream PySCF 2.12.1 to a relative 1e-12 on every basis-derived quantity and
EXACTLY on every mesh / image count.**

## What Shipped

### `crates/pyscf-pbc-tools/src/mat3.rs`

`det3` / `transpose3` / `inv3` moved DOWN from `pyscf-pbc-gto::cell` (plan 09-03) because
`cutoff_to_mesh` needs `b = 2*pi*inv(a.T)` and the dependency edge runs
`pyscf-pbc-gto -> pyscf-pbc-tools`. `pyscf_pbc_gto::cell` re-exports them, so
`pyscf_pbc_gto::{det3, inv3, transpose3}` and every 09-03 test are unchanged. Adds
`dot3` / `cross3` / `norm3`.

### `crates/pyscf-pbc-tools/src/mesh.rs` — `tools/pbc.py:787-836`

`cutoff_to_mesh`, `mesh_to_cutoff`, `cutoff_to_gs`, `gs_to_cutoff`, plus the two
helpers upstream inlines: `qr_heights(a)` (the three `qr(...)[1][2,2]` magnitudes) and
`qr_r22_abs` (a 3x3 Householder QR, LAPACK `dgeqrf` reflection order). Only `|R[2,2]|`
is observable at either call site — `cutoff_to_mesh` takes `np.abs`, `mesh_to_cutoff`
squares — so the Householder sign convention never surfaces.
`qr_r22_abs_closed_form` (`|det(M)| / ||c0 x c1||`) is exported as the independent
cross-check the tests use.

Deviation from upstream's silence: a negative / non-finite `cutoff` and a singular
lattice are `Err`, not a NaN mesh.

### `crates/pyscf-pbc-gto/src/cutoff.rs`

| upstream | Rust |
|---|---|
| `cell.py:373-390` `get_nimgs` | `get_nimgs` |
| `cell.py:392-407` `_estimate_rcut` | `estimate_rcut_pgto` |
| `cell.py:409-422` `bas_rcut` | `bas_rcut` (+ `Cell::bas_rcut`) |
| `cell.py:424-436` `estimate_rcut` | `estimate_rcut` |
| `cell.py:438-451` `_estimate_ke_cutoff` | `estimate_ke_cutoff_pgto` |
| `cell.py:453-464` `estimate_ke_cutoff` | `estimate_ke_cutoff` |
| `cell.py:481-500` `_extract_pgto_params` | `extract_pgto_params` + `PgtoOp` |
| `cell.py:502-515` `error_for_ke_cutoff` | `error_for_ke_cutoff` |
| `cell.py:517-543` `get_bounding_sphere` | `get_bounding_sphere` (+ `Cell::nimgs`) |
| `cell.py:968-972` `_mesh_inf_vaccum` | `mesh_inf_vacuum` |
| `cell.py:974-991` `pgf_rcut` | `pgf_rcut` |
| `lib/pbc/cell.c:30-59` `pgf_rcut` | `pgf_rcut_c` |
| `cell.py:993-1024` `rcut_by_shells` | `rcut_by_shells` / `rcut_by_shells_with_pgf` (+ `Cell::rcut_by_shells`) |
| `cell.py:1760-1768` (mesh half of `build`) | `estimate_mesh` |

Plus the Rust spelling of `mol.bas_angular` / `bas_nprim` / `bas_nctr` / `bas_exp` /
`_libcint_ctr_coeff(...).max(axis=1)` / `mol.omega` over `_bas` / `_env`. Upstream
variable names (`theta`, `a1`, `norm_ang`, `fac`, `r0`, `Ecut`, `heights_inv`, `rmin`,
`gmax`) are kept verbatim in the bodies so a reviewer can diff side by side.

`np.float64 ** x` is ported as `powf`, never `powi` — numpy casts an integer exponent to
`float64` and calls `pow`, and `powi`'s repeated-multiplication result can differ in the
last bits.

### `Cell` wiring (`cell.rs`)

`estimate_rcut` / `estimate_mesh` are now thin `Result` wrappers over `crate::cutoff`
(the signatures 09-03 froze are unchanged, so no call site or `pub use` moved).
`Cell::build`'s step list matches upstream's order exactly:
4 `rcut` -> 5 left-handed warning -> **6 vacuum-size warning** -> 7 `mesh` -> 8 `_built`.
Step 6 is the 09-03 carry-over ("needs `rcut`, add it in 09-04"), now closed.

New: `Cell::cutoff_to_mesh` (`cell.py:1952-1967`, the method form that keeps the
non-periodic axes), `Cell::nimgs`, `Cell::rcut_by_shells`, `Cell::bas_rcut`, and the
`use_loose_rcut` field on `Cell` / `CellBuildArgs` / `CellPack`.

## Verification Results

```
cargo test -p pyscf-pbc-gto --test cutoff                                   ✅ 29 passed / 0 failed
cargo test -p pyscf-pbc-tools                                               ✅ 12 passed / 0 failed
cargo test -p pyscf-pbc-gto --all-features                                  ✅ 52 passed / 0 failed
cargo clippy -p pyscf-pbc-gto -p pyscf-pbc-tools --all-targets --all-features -- -D warnings  ✅ clean
cargo build --workspace                                                     ✅ clean
cargo doc -p pyscf-pbc-gto -p pyscf-pbc-tools --no-deps --all-features      ✅ no broken links
cargo run -p xtask --bin check-dependency-wall                              ✅ PASS (ALG-06 intact)
cargo run -p xtask --bin check-forbidden-paths                              ✅ PASS
rustfmt --edition 2024 --check <all touched files>                          ✅ clean
```

### Tier 2 — upstream PySCF 2.12.1 (D-PBC-19)

`precision = 1e-8`. Every value below is reproduced by the port; basis-derived
quantities at a relative **1e-12**, mesh / image counts **exactly**.

| system | `estimate_rcut` | `estimate_ke_cutoff` | `cell.mesh` | `cutoff_to_mesh(a,100)` | `nimgs` |
|---|---:|---:|---|---|---|
| diamond | 21.319400521777592 | 422.9075470012404 | [47,47,47] | [23,23,23] | [6,6,6] |
| si | 29.960198598827567 | 108.20807384760171 | [35,35,35] | [35,35,35] | [6,6,6] |
| lif | 38.46107083110416 | 1077.5424956328934 | [81,81,81] | [27,27,27] | [9,9,9] |
| he_fcc | 16.808894871965055 | 979.7661855059968 | [59,59,59] | [21,21,21] | [6,6,6] |
| graphene | 21.319400521777592 | 422.9075470012404 | [45,45,351] | [23,23,173] | [6,6,**0**] |

Also matched: `rcut_by_shells` and `bas_rcut` per shell for all five;
`error_for_ke_cutoff(cell, 100)` spanning 25.9 (LiF) down to 5.4e-8 (Si);
`cutoff_to_mesh` at ke = 50/100/200 and `cutoff_to_gs(a, 100)` for all five lattices;
`mesh_to_cutoff(a, [15,15,15])` (see the tolerance note below).

### Tier 1 — invariants (no upstream needed)

`estimate_rcut == max_shell bas_rcut` (proves the `argmin` tie-break and the `axis=1`
coefficient reduction); `rcut` and `ke_cutoff` strictly grow as `precision` tightens;
`error_for_ke_cutoff(estimate_ke_cutoff(p)) ∈ [0.5p, 1.5p]` (the two are inverses) and
falls monotonically in the cutoff; `pgf_rcut` satisfies its own defining equation
`c*r^(l+2)*exp(-alpha*r^2) = precision` to 1e-6 over `l ∈ 0..3` × 4 alphas × 3 coeffs;
`pgf_rcut_c`'s early return is exactly `rmin` and fires exactly when `gmax < precision`;
`rcut_by_shells` (loose) never exceeds `bas_rcut` (tight); `get_bounding_sphere` zeroes
axes `>= dimension`, is monotone in `rcut`, and is `[0,0,0]` at `rcut = 0`; every mesh
axis is odd, monotone, and `cutoff_to_mesh` returns the MINIMAL sufficient odd mesh (the
mesh one step smaller provably under-resolves); `qr_r22_abs` agrees with
`|det|/||c0 x c1||` on 2000 pseudo-random matrices; `qr_heights(a)[i] == 2*pi/||a_i||`
exactly, on the reference lattices and on 500 random ones; a cubic lattice gives the
hand-computable mesh; `nbas == 0` returns upstream's literal `0.01` / `0.`; a non-zero
`omega` lowers the `ke_cutoff`; the `inf_vacuum` branch replaces only the non-periodic
axes and `mesh_inf_vacuum` is always even.

## DEVIATIONS from the plan

**1. The plan's stated acceptance numbers were wrong; regenerated as instructed.**
PBC-MASTER-PLAN §8.1 plan 09-04 guesses `rcut ~ 15.6` Bohr and
`cutoff_to_mesh(a, 100.0) == [15,15,15]` for diamond/gth-szv, with the explicit
instruction "regenerate and hard-code these before writing the test". The true upstream
values are **`rcut = 21.319400521777592`** and **`[23,23,23]`**. Those are asserted.

**2. `pgf_rcut` is ported TWICE.** The plan's PORT block names `cell.py:974-1025`
(`pgf_rcut`, `rcut_by_shells`). But upstream's `cell.rcut_by_shells` does not call the
Python `pgf_rcut` — it calls `libpbc.rcut_by_shells`, whose C twin
(`lib/pbc/cell.c:30-59`) adds a `gmax < precision` early return the Python version lacks.
Porting only the Python one would have produced a `rcut_by_shells` that cannot reproduce
upstream's numbers by construction. Both are shipped: `pgf_rcut` (Python) and
`pgf_rcut_c` (C, used by `rcut_by_shells`). For the five §9.2 systems they happen to
agree, which the tests record.

**3. `det3` / `transpose3` / `inv3` moved crates.** They were plan 09-03's, in
`pyscf-pbc-gto::cell`. `cutoff_to_mesh` needs `2*pi*inv(a.T)` and lives in
`pyscf-pbc-tools`, which is BELOW `pyscf-pbc-gto`. Rather than a second copy of the
lattice inversion, the definitions moved to `pyscf_pbc_tools::mat3` and `cell.rs`
re-exports them — `pyscf_pbc_gto::{det3, inv3, transpose3}` and every 09-03 test are
byte-for-byte unaffected.

**4. `use_loose_rcut` added to `Cell`.** The 09-03 struct spec does not list it, but
`estimate_rcut` branches on `cell.use_loose_rcut` (`cell.py:430-431`). Without the field
that branch — and therefore `rcut_by_shells`, which the same plan requires — would be
dead. Added to `Cell`, `CellBuildArgs` and `CellPack` (`#[serde(default)]`, so older
JSON still loads).

**5. `estimate_rcut` / `estimate_mesh` keep their `Result` signatures.** The real
`estimate_rcut` is infallible; the wrapper still returns `Result<f64, _>` so plan 09-03's
call sites and the `pub use` in `lib.rs` did not have to move. Documented as
"# Errors: never, today".

**6. `rcut_by_shells_with_pgf` returns a RAGGED `Vec<Vec<f64>>`.** Upstream allocates a
rectangular `(nbas, max nprim)` array and explicitly leaves the tail uninitialised
(`cell.py:1013-1015`). A ragged `Vec` carries the same information and cannot expose
uninitialised memory.

**7. `cutoff_to_mesh` validates its input.** Upstream silently produces a NaN mesh for a
negative or non-finite cutoff, and `inf`/`nan` for a singular lattice. Both are `Err`
here.

**8. Errors instead of `NotYetImplemented`.** The plan's Task 2 boilerplate mentions
deferred branches returning `NotYetImplemented { phase }`. This plan has none — every
branch of every ported function is implemented, including the `inf_vacuum` mesh path and
the `use_loose_rcut` path.

## Tolerance provenance

Tolerances are chosen by what a quantity depends on, not by what passes:

* **basis-only** (`rcut`, `ke_cutoff`, `rcut_by_shells`, `bas_rcut`,
  `error_for_ke_cutoff`) — relative **1e-12**. Exponents and libcint contraction
  coefficients are unit-independent, so the `pyscf_core::Unit::Ang` gap documented in
  09-03 cannot reach them. (Observed agreement is at the 1e-16 level; the one visible
  difference is a 1-ULP contraction coefficient, e.g. diamond's
  `0.20266288674094945` vs upstream `0.20266288674094948`.)
* **lattice-derived integers** (`mesh`, `nimgs`, `cutoff_to_gs`) — **exact**. They are
  `ceil`-ed far from a boundary, so the 4.95e-9 `Unit::Ang` gap cannot move them; a
  failure here would be a real port bug.
* **lattice-derived floats** (`mesh_to_cutoff`) — relative **1e-7**, the same bound
  `tests/cell_build.rs` uses. The test additionally asserts the residual is below
  2e-8, i.e. that it really is the known constant's gap SQUARED (~9.9e-9) and not
  something larger hiding behind a loosened bound.

## Carry-overs

- **`Cell::tot_electrons` is still the ALL-ELECTRON count**, so `mesh_inf_vacuum`
  (which reads it) will shift slightly for pseudopotential cells when plan 10-01 lands
  GTH (D-PBC-11). No §9.2 system exercises the `inf_vacuum` branch today, so no
  committed reference value moves.
- **`pyscf_core::Unit::Ang.length_in_au()` still disagrees with upstream by 4.95e-9
  relative** (09-03's carry-over, unchanged). It is why `mesh_to_cutoff` is checked
  relatively. When it is corrected, that test's bound drops to 1e-12.
- **`pyscf-gto` should still declare `serde_json`'s `float_roundtrip` feature itself**
  (09-03 carry-over, unchanged).
- **`use_loose_rcut` has no builder-level ergonomics** beyond the struct field; if a
  PyO3 kwarg is wanted it belongs with the Phase 3-style bridge work.
- **`Cell::fromstring` / `fromfile`** still not ported (09-03 carry-over).
- **`ew_eta` / `ew_cut`** still always `None` — plan 09-08's `get_ewald_params`.
- **`error_for_ke_cutoff`'s `omega` argument is `Option<f64>`**, matching upstream's
  `omega=None` default. When Phase 12 wires range-separated PBC exchange, check that
  callers pass the RSH omega rather than relying on `cell._env[8]`.

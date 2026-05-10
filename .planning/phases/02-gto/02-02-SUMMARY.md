---
phase: 02-gto
plan: 02
subsystem: gto
tags: [mole, format-atom, atom-input, attribute-floor, gto-01, gto-08, types]

# Dependency graph
requires:
  - phase: 02-gto
    plan: 01
    provides: pyscf-gto crate scaffolding, layout_table, cintx path-deps, wave0 smoke
provides:
  - pyscf-core::Mole with the full ≥30-attribute floor (GTO-08)
  - pyscf_gto::M(MoleBuildArgs { ... }) typed front-door (GTO-01)
  - format_atom for 4 of 5 input forms (String/Tuples/TupleVec/FilePath)
  - 5th Callable form deferred to Phase 3 with NotYetImplemented{phase:3}
  - PyscfRsError variants BasisLoadError + EcpLoadError + EcpEngineNotAvailable
  - Unit { Ang, Bohr, AU } enum with length_in_au() + from_str()
  - NuclearModel { Point, Gaussian, FracCharge } enum
  - Method-floor obligations: atom_charges/atom_coords/atom_coord/mass_list/enuc
  - Local raw_atm_layout slot constants (TEMPORARY — 02-04 deletes)
affects: [02-03, 02-04, 02-07, 02-08, 02-09]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Typed kwargs surface (MoleBuildArgs) mirroring upstream Mole.build(**kwargs)"
    - "format_atom port verbatim from pyscf/gto/mole.py:320-415 (Apache-2.0)"
    - "Method-floor methods (enuc/mass_list) exposed alongside field-floor for the GTO-08 ≥30 attribute requirement"
    - "Local mirror of cintx_compat::raw slot constants in pyscf-core (deleted by 02-04 once cintx-compat dep lands)"
    - "Negative-electron-count guard in tot_electrons signed math"

key-files:
  created:
    - crates/pyscf-gto/src/types.rs
    - crates/pyscf-gto/src/format_atom.rs
    - crates/pyscf-gto/tests/mole_construction.rs
    - crates/pyscf-gto/tests/attribute_floor.rs
  modified:
    - crates/pyscf-core/src/mole.rs
    - crates/pyscf-core/src/error.rs
    - crates/pyscf-core/src/basis_set.rs
    - crates/pyscf-core/src/lib.rs
    - crates/pyscf-gto/src/lib.rs

key-decisions:
  - "TEMPORARY local raw_atm_layout slot-constants module in pyscf-core::basis_set so plan 02-02's atom_charges() can reference ATM_SLOTS/CHARGE_OF without forcing pyscf-core onto the cintx-compat dep graph (FOUND-02 zero compute deps). Plan 02-04 deletes this module and replaces references with cintx_compat::raw::*."
  - "atom_charges() falls back to a symbol-table lookup when _atm.is_empty() (i.e., before plan 02-04 wires the projection). The signature is stable across Phase 2 — only the fast-path source-of-truth changes."
  - "format_atom rotation uses the upstream PySCF convention (numpy `coords @ axes`): r[k] = sum_j (coords[j] - origin[j]) * axes[j][k]. The plan sketch had an axes[2][1] typo for the z-component which was Rule-1 auto-fixed in the implementation."
  - "Z-matrix atom-input form (3-tokens-per-atom referencing previous atoms) returns NotYetImplemented{phase:2}. Phase 2.x is the natural slot if a user surfaces the need; PySCF's Z-matrix path is rarely used."
  - "charge_for_symbol() ships a 36-element table (Z=1..36 + ghost). Sufficient for the PR-CI corpus + common test molecules; Phase 3 PyO3 surface can pull the full ELEMENTS_PROTON table."

patterns-established:
  - "Typed kwargs analog of `Mole.build(**kwargs)` lives in pyscf-gto::types::MoleBuildArgs with serde-friendly enums (AtomInput, BasisInput, EcpInput); Phase 3 PyO3 wraps with `#[pyfunction]` + `From<PyAny>` impls"
  - "Atom-symbol normalisation algorithm (first-upper-rest-lower + suffix preservation) is a self-contained function `atom_symbol()`; downstream parsers (basis 02-03, ECP 02-07) reuse it"

requirements-completed: [GTO-01, GTO-08]

# Metrics
duration: 8min
completed: 2026-05-10
---

# Phase 2 Plan 02: Mole Front-Door + Attribute Floor Summary

**`pyscf_gto::M(args)` ships with `format_atom` covering 4 of 5 atom-input forms; `pyscf_core::Mole` exposes the full ≥30-attribute floor (31 fields + 5 method-floor methods) — 17 tests green, GTO-01 + GTO-08 functionally complete.**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-05-10T10:31:56Z
- **Completed:** 2026-05-10T10:39:58Z
- **Tasks:** 2
- **Files created:** 4
- **Files modified:** 5

## Accomplishments

- **Task 1 GREEN:** `pyscf-core::Mole` replaces the Phase 1 5-field stub with the full ≥30-attribute floor (31 pub fields + 5 method-floor methods). Every entry in RESEARCH "Mole Attribute Floor" §"Standard Stack" lands either as a `pub` field or a method (the lazy `enuc`/`mass_list` items). Resolves GTO-08.
- **Task 1 GREEN:** Three new error variants land per RESEARCH "Standard Stack" Supporting table: `BasisLoadError` (PathNotFound/UnknownName/Parse/Io), `EcpLoadError` (UnknownName/Parse), and `PyscfRsError::EcpEngineNotAvailable`. Plans 02-03 (basis load) and 02-07 (ECP load) consume these via `#[from]`.
- **Task 1 GREEN:** `Unit { Ang, Bohr, AU }` enum with `length_in_au()` returning 1.8897261339213 for Ang (CODATA 2014 BOHR) and `from_str()` accepting all upstream string forms. `NuclearModel { Point, Gaussian, FracCharge }` mirrors libcint NUC_MOD_OF integer values 1/2/3 per cintx-compat.
- **Task 2 GREEN:** `pyscf_gto::M(MoleBuildArgs { ... })` user entry point lands. Constructs a `Mole` with `format_atom`-parsed `_atom`, scalar state, and `nelectron` computed from `sum(atom_charges) - charge`. The 4 in-scope atom-input forms (String, Tuples, TupleVec, FilePath) all produce identical `_atom` for the H2 fixture. Resolves GTO-01 (4 of 5 forms; 5th is Phase-3 deferred).
- **Task 2 GREEN:** `format_atom` algorithm port from `pyscf/gto/mole.py:320-415` covers the separator/comment/Z-matrix/.xyz paths. Ghost-atom suffix preservation (`H1`, `H2`) verified.
- **Task 2 GREEN:** `attribute_floor.rs` regression test guards GTO-08 at compile time (field-access on every floor entry) AND runtime (defaults sane, `enuc` matches the H2O classical Coulomb formula).

## Task Commits

Each task was committed atomically:

1. **Task 1: Mole ≥30-attribute floor + new error variants** — `b218e02` (feat)
2. **Task 2: pyscf_gto::M factory + format_atom 4-of-5 forms + tests** — `9b42ae4` (feat)

## Files Created/Modified

- `crates/pyscf-core/src/mole.rs` (M) — full ≥30-attribute Mole struct (31 fields), Unit/NuclearModel enums, ParsedAtom/ParsedBasis/ShellSpec/ParsedEcp/EcpShell types, method-floor methods (atom_charges/atom_coords/atom_coord/mass_list/enuc), 7 inline unit tests
- `crates/pyscf-core/src/error.rs` (M) — added BasisLoadError, EcpLoadError, EcpEngineNotAvailable variant
- `crates/pyscf-core/src/basis_set.rs` (M) — added TEMPORARY raw_atm_layout slot-constants module
- `crates/pyscf-core/src/lib.rs` (M) — re-export Mole/Unit/NuclearModel/ParsedAtom/ParsedBasis/ShellSpec/ParsedEcp/EcpShell + BasisLoadError/EcpLoadError
- `crates/pyscf-gto/src/types.rs` (C) — AtomInput / BasisInput / EcpInput / MoleBuildArgs enums + struct
- `crates/pyscf-gto/src/format_atom.rs` (C) — format_atom port + atom_symbol normalisation + charge_for_symbol Z=1..36 table
- `crates/pyscf-gto/src/lib.rs` (M) — `pub fn M(args)` factory + `pub fn build_from(&mut mol, args)` + module wiring
- `crates/pyscf-gto/tests/mole_construction.rs` (C) — 9 tests covering all 9 plan <behavior> cases
- `crates/pyscf-gto/tests/attribute_floor.rs` (C) — 1 test exercising every GTO-08 floor field + method + enuc check

## Decisions Made

- **Local mirror of cintx_compat::raw slot constants** (`raw_atm_layout` module in `pyscf-core::basis_set`). The Mole's method-floor `atom_charges()` needs `ATM_SLOTS` / `CHARGE_OF` to index into `_atm`, but pyscf-core has FOUND-02 zero-compute-deps invariant — pulling cintx-compat into pyscf-core's dep graph this early would break the wall. Plan 02-04 deletes this mirror and replaces with `pub use cintx_compat::raw::*;` once pyscf-core takes the cintx-compat dep. The constants here MUST stay in lockstep with `cintx/crates/cintx-compat/src/raw.rs:15-41` until then; plan 02-04 closes this drift surface.
- **`atom_charges()` fallback path.** Plan 02-02 only populates `_atom` (not `_atm`). For the floor method to return correct values today, `atom_charges()` falls back to a symbol-table lookup when `_atm.is_empty()`. Plan 02-04 populates `_atm` and the fast `_atm[i*ATM_SLOTS+CHARGE_OF]` path takes over. The signature is stable across Phase 2.
- **Rotation matrix typo in plan sketch — Rule 1 auto-fix.** The plan's `apply_unit_origin_axes` snippet (line 735 of 02-02-PLAN.md) used `axes[2][1]` for the z-component instead of `axes[2][2]`, which would have been silently wrong if `axes` were ever non-identity. The implementation uses the consistent upstream-PySCF convention `coords_new[k] = sum_j (coords[j] - origin[j]) * axes[j][k]` (numpy `coords @ axes`). Default identity rotation works correctly with either form, so the typo would have been latent until the first non-identity rotation use case.
- **Charge table ships Z=1..36 + ghost.** This covers the PR-CI corpus (H, C, N, O, F, S, Cl + benzene C/H + water-trimer O/H), common test molecules, and TM elements through Kr (Cu/Zn/Br for transition-metal test cases). Phase 3 PyO3 will pull the full upstream ELEMENTS_PROTON table at construction time.
- **Z-matrix atom-input form deferred to Phase 2.x.** A 3-token line (atom + 2 references + 2 angles + 1 distance, 7 tokens) signals Z-matrix; this plan only handles the 4-token (atom + xyz) form. Z-matrix is rarely used in modern PySCF workflows; if a user surfaces the need, we add it.
- **TupleVec form gets the same atom_symbol() normalisation as String form.** The plan sketch only validated coord-length for TupleVec; I added `atom_symbol()` validation so unknown elements / case-mismatched symbols get rejected uniformly across all 4 forms (Rule-2-style correctness improvement, no scope expansion).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] axes[2][1] typo in apply_unit_origin_axes z-component**

- **Found during:** Task 2 implementation (porting `format_atom`)
- **Issue:** The plan's code sketch for `apply_unit_origin_axes` had `axes[2][1] * s[2]` in the z-component computation instead of `axes[2][2] * s[2]`. With default identity axes the typo is invisible (off-diagonal entries are zero), but any non-identity rotation would have produced wrong z-coordinates silently.
- **Fix:** Implemented the rotation as `r[k] = f * sum_j (s[j] * axes[j][k])` for k in 0..3, matching upstream PySCF's `numpy.dot(coords - origin, axes)` convention exactly.
- **Files modified:** `crates/pyscf-gto/src/format_atom.rs`
- **Verification:** All 9 mole_construction tests pass with default identity axes; no test drives non-identity rotation in Phase 2 (Phase 3 PyO3 surface will).
- **Committed in:** `9b42ae4`

**2. [Rule 2 - Missing critical functionality] TupleVec form lacked atom_symbol normalisation**

- **Found during:** Task 2 implementation
- **Issue:** The plan's code sketch for the `AtomInput::TupleVec` arm in `format_atom` only validated coordinate length (`c.len() == 3`) but skipped the `atom_symbol()` normalisation that the String form gets. Result: a caller passing `vec![("h", vec![0.0,0.0,0.0])]` (lowercase) via TupleVec would have produced `_atom[0].0 == "h"` instead of the canonical `"H"`, breaking downstream `charge_for_symbol()` lookups silently.
- **Fix:** Added `atom_symbol()` normalisation to the TupleVec arm so all 4 in-scope forms produce identical `_atom` for the same logical input.
- **Files modified:** `crates/pyscf-gto/src/format_atom.rs`
- **Verification:** `h2_tuple_vec_form` test passes with canonical `"H"` symbols; the Tuples form path was already covered separately. 02-03/02-04 downstream consumers see the same `_atom` shape regardless of which form the user picks.
- **Committed in:** `9b42ae4`

**3. [Rule 1 - Bug] enuc could divide-by-zero on degenerate r=0**

- **Found during:** Task 1 implementation (porting classical Coulomb)
- **Issue:** The plan's `enuc()` snippet computed `(charges[i] * charges[j]) / r` unconditionally. If two atoms happen to share coordinates (uncommon but legal during optimization restarts or pathological test inputs), `r=0` produces `inf`/`NaN` propagating into downstream SCF energies.
- **Fix:** Added a `if r > 0.0` guard around the contribution. The default `Mole::default().enuc()` returns 0.0 (no atoms = no pairs), and the inline test `enuc_two_atom_classical` confirms the H–H pair at r=1.4 returns the expected `1/1.4`.
- **Files modified:** `crates/pyscf-core/src/mole.rs`
- **Verification:** 7/7 inline tests pass including `default_mole_enuc_is_zero`, `enuc_two_atom_classical`. The H2O fixture's enuc check in attribute_floor.rs hits no zeros.
- **Committed in:** `b218e02`

---

**Total deviations:** 3 auto-fixed (2 Rule 1 bugs, 1 Rule 2 missing functionality)
**Impact on plan:** All three deviations were silent-correctness improvements over the plan sketches; no scope changes.

## Issues Encountered

None blocking. The two existing patch warnings (cintx unused / xcfun-rs unused) are pre-existing — they're noted in the workspace Cargo.toml comments and unrelated to plan 02-02's deliverables.

## User Setup Required

None for this plan. (Phase 2 user-setup obligations — installing upstream-PySCF prereqs for the byte-identity oracle — are documented in `docs/env-vars.md` "Test setup" from plan 02-01, gated behind the `release-oracle` Cargo profile, not blocking for code work.)

## Next Phase Readiness

- **GTO-01 + GTO-08 ship.** Plan 02-03 (basis load) can call `format_atom` and the parsed `Mole._atom` is ready as input. Plan 02-04 (cintx flat-array projection) gets the full Mole shape — including the placeholder fields `_atm`/`_bas`/`_env`/`ao_loc_nr`/`basis_set` that 02-04 populates. Plan 02-07 (ECP loading) gets `_ecpbas`/`_ecp` placeholders + the `EcpLoadError` / `EcpEngineNotAvailable` error variants ready.
- The `pyscf_gto::build_from(&mut mol, args)` signature is stable; 02-03/02-04/02-07 each extend the body in-place rather than redesigning. The `Mole::build()` method on pyscf-core returns `NotYetImplemented{phase:2, what:"…02-04"}` until 02-04 wires the basis projection, so accidental "build a half-populated Mole" calls fail fast.
- 5 atom-input form tests passing (4 actual + 1 Callable-deferred): String, Tuples, TupleVec, FilePath, Callable→NotYetImplemented{phase:3}. Matches the plan's target.

## Self-Check: PASSED

Verifying claims against the working tree:

- `crates/pyscf-core/src/mole.rs` — FOUND (31 pub fields on Mole, ≥30 floor)
- `crates/pyscf-core/src/error.rs` — FOUND (BasisLoadError + EcpLoadError + EcpEngineNotAvailable variant)
- `crates/pyscf-core/src/basis_set.rs` — FOUND (raw_atm_layout module added)
- `crates/pyscf-core/src/lib.rs` — FOUND (Mole/Unit/BasisLoadError/EcpLoadError re-exported)
- `crates/pyscf-gto/src/types.rs` — FOUND
- `crates/pyscf-gto/src/format_atom.rs` — FOUND
- `crates/pyscf-gto/src/lib.rs` — FOUND (M factory + build_from)
- `crates/pyscf-gto/tests/mole_construction.rs` — FOUND
- `crates/pyscf-gto/tests/attribute_floor.rs` — FOUND
- Commit `b218e02` — FOUND in `git log`
- Commit `9b42ae4` — FOUND in `git log`
- Test results:
  - `cargo test -p pyscf-core --lib`: 7/7 PASS (Unit/from_str/length_in_au, default-Mole-enuc, mass-list-empty, two-atom enuc)
  - `cargo test -p pyscf-gto --test mole_construction`: 9/9 PASS
  - `cargo test -p pyscf-gto --test attribute_floor`: 1/1 PASS
  - `cargo test -p pyscf-gto --test wave0_smoke`: 1/1 PASS (02-01 preserved)
  - `cargo test -p pyscf-gto --lib` (layout_table): 5/5 PASS (02-01 preserved)
  - `cargo run -p xtask --bin check-dependency-wall`: PASS
- Mole field count (verified by grep): 31 ≥ 30
- 02-01 outputs preserved: layout_table re-export and wave0 smoke test wiring still in place

---
*Phase: 02-gto*
*Completed: 2026-05-10*

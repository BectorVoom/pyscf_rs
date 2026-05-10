---
phase: 02-gto
plan: 08
subsystem: gto
tags: [serde_json, dumps, loads, set_geom_, mol.copy, json-roundtrip, pattern-5, granular-invalidation, arc-zerocopy]

# Dependency graph
requires:
  - phase: 02-gto plan 01
    provides: workspace serde + serde_json deps; layout-table reference
  - phase: 02-gto plan 02
    provides: pyscf_gto::M(MoleBuildArgs) front-door; AtomInput/BasisInput/EcpInput types
  - phase: 02-gto plan 03
    provides: format_basis dispatch (BasisInput::PerElement<Parsed> arm)
  - phase: 02-gto plan 04
    provides: make_env flat-array projection + Arc<BasisSet> zero-copy via projection.rs (GTO-04 + GTO-11)
  - phase: 02-gto plan 07
    provides: format_ecp + make_ecp_env (ECP per-element parsed maps round-trip via Parsed arm)
provides:
  - "GTO-09: pyscf_gto::dumps(&Mole) -> String / loads(&str) -> Mole — semantic JSON round-trip via serde_json"
  - "GTO-10: pyscf_gto::set_geom_(&mut Mole, &str) — in-place geometry mutation per RESEARCH Pattern 5 (granular invalidation; basis_set Arc identity preserved)"
  - "GTO-10 (cont.): mol.copy() satisfied by Mole's #[derive(Clone)] — deep-copy value fields, Arc-clone basis_set"
  - "Serialize/Deserialize derives on pyscf_core::{Unit, ParsedBasis, ShellSpec, ParsedEcp, EcpShell} — enables future chkfile/HDF5 work without re-derivation"
affects:
  - "02-09 verification rollup (GTO-12 phase-level)"
  - "Phase 3 SCF — mol.set_geom_ + mol.copy used by geomopt scaffolding seam"
  - "Phase 7 grad/geomopt — set_geom_ is the primary mutation point during line search"
  - "Phase 8 ORACLE-08 chkfile interop — dumps/loads is the in-Rust prototype before HDF5 binding"

# Tech tracking
tech-stack:
  added:
    - "serde::{Serialize, Deserialize} derives on pyscf_core::{Unit, ParsedBasis, ShellSpec, ParsedEcp, EcpShell}"
    - "serde_json::{to_string, from_str} as the JSON encoder/decoder seam"
  patterns:
    - "Pattern 5 (RESEARCH §Architecture Patterns): granular cache invalidation — mutate primitives, leave structure-derived caches (Arc<BasisSet>, _bas, ao_loc_nr, nao_nr) untouched"
    - "Snapshot-and-rebuild round-trip: serialise user-input portion, rebuild via M(args) to reconstruct deterministic derived arrays — sidesteps the byte-identical-JSON contract upstream PySCF locks itself into"
    - "Pre-mutation validation: atom-count + symbol checks fire BEFORE any mutation in set_geom_ — partial mutation is impossible (T-02-08-03 mitigation)"

key-files:
  created:
    - "crates/pyscf-gto/src/dumps_loads.rs — MoleSnapshot + dumps()/loads()"
    - "crates/pyscf-gto/src/set_geom.rs — set_geom_ in-place mutator (Pattern 5)"
    - "crates/pyscf-gto/tests/dumps_loads.rs — 3 tests (round-trip arrays, cart+charge+spin, malformed JSON error)"
    - "crates/pyscf-gto/tests/set_geom.rs — 5 tests (env-coord update, atom-count mismatch, symbol mismatch, unit kwarg, atom_coords()-method output)"
    - "crates/pyscf-gto/tests/mole_copy.rs — 2 tests (deep-clone value fields, Arc identity preserved for basis_set)"
  modified:
    - "crates/pyscf-gto/src/lib.rs — module declarations + re-exports for dumps, loads, set_geom_"
    - "crates/pyscf-core/src/mole.rs — add Serialize/Deserialize derives on Unit + ParsedBasis + ShellSpec + ParsedEcp + EcpShell"
    - "crates/pyscf-core/Cargo.toml — explicit serde 'derive' feature (matches workspace inheritance, documents the use-site)"

key-decisions:
  - "Round-trip via snapshot-and-rebuild (not snapshot-and-restore): dumps stores the user-input portion (_atom in Bohr, _basis per-element, _ecp per-element, scalar kwargs); loads runs M(args) to deterministically reproduce _atm/_bas/_env/ao_loc_nr/nao_nr/_ecpbas. This guarantees byte-equality across the round trip without serialising the derived arrays themselves."
  - "loads forces unit=Bohr on rebuild (atoms are stored in Bohr) and then restores mol.unit from the snapshot — keeps the Bohr-or-bust internal invariant clean while round-tripping the user-facing unit label."
  - "set_geom_ leaves Arc<BasisSet>.atoms[*].coord at construction-time values (v1 limitation T-02-08-05). _env is the libcint-facing source of truth; downstream consumers needing fresh BasisSet coords rebuild via M(args). Documented in module-level comment."
  - "Symbol/count validation in set_geom_ fires BEFORE any mutation — partial-update corruption is impossible (T-02-08-03 mitigation)."
  - "Added Serialize/Deserialize to pyscf_core::{Unit, ParsedBasis, ShellSpec, ParsedEcp, EcpShell} rather than building a parallel mirror in pyscf-gto. Single source of truth; keeps the snapshot type in dumps_loads.rs minimal."

patterns-established:
  - "Pattern 5 (Granular Invalidation): set_geom_ proves out the explicit zero-invalidation contract — Arc::ptr_eq before/after holds. Any future mutator (set_unit_, set_charge_) follows the same shape: mutate primitives, leave caches alone."
  - "Round-Trip Test Shape: build mol_a, dumps→loads→mol_b, assert _atm/_bas/_env/ao_loc_nr/nao_nr equal byte-for-byte. Reused in Phase 3+ for chkfile round-trip oracles."

requirements-completed: ["GTO-09", "GTO-10"]

# Metrics
duration: 7min
completed: 2026-05-10
---

# Phase 02 Plan 08: dumps/loads + set_geom_ + mol.copy Summary

**JSON round-trip (serde_json) preserves _atm/_bas/_env byte-for-byte; in-place set_geom_ mutates _env coords with Arc identity preserved per RESEARCH Pattern 5; mol.clone() satisfies the GTO-10 copy obligation.**

## Performance

- **Duration:** 7 min (TDD: RED → GREEN, no REFACTOR needed)
- **Started:** 2026-05-10T13:36:40Z
- **Completed:** 2026-05-10T13:43:27Z
- **Tasks:** 1 (single-task plan, TDD-driven)
- **Files modified:** 6 (3 new src/, 3 new tests/, 2 touched in pyscf-core, 1 touched in pyscf-gto/lib.rs)

## Accomplishments

- **GTO-09 dumps/loads** — `pyscf_gto::dumps(&mol)` returns a JSON snapshot of the user-input portion of `Mole`; `pyscf_gto::loads(&json)` reconstructs a `Mole` whose derived arrays equal the original byte-for-byte. The contract is **semantic round-trip** (per CONTEXT D-09) — NOT byte-identical to upstream PySCF JSON; that's Phase 8 ORACLE-08 chkfile territory.
- **GTO-10 set_geom_** — `pyscf_gto::set_geom_(&mut mol, "H 0 0 0; H 0 0 2.0")` updates `_atom` and `_env` xyz slots in place. `_bas`, `_basis`, `ao_loc_nr`, `nao_nr`, and the `Arc<BasisSet>` are untouched (verified by `Arc::ptr_eq` before/after). RESEARCH Pattern 5 (granular invalidation) is now a load-bearing contract with regression coverage.
- **GTO-10 mol.copy** — `Mole`'s existing `#[derive(Clone)]` (Phase 1 + 02-02) is the canonical `mol.copy()` — Vec value-fields deep-copy, `Arc<BasisSet>` clones the refcount only. Cross-checks GTO-11 zero-copy.
- **Serde derives** added to `pyscf_core::{Unit, ParsedBasis, ShellSpec, ParsedEcp, EcpShell}` — minimal, single source of truth, no mirror types in pyscf-gto.

## Round-Trip Test Pairs

The dumps/loads byte-equality contract is exercised by two molecules covering the relevant slot shapes:

| Test | Molecule | Inputs Round-Tripped | Asserted Byte-Equal |
| --- | --- | --- | --- |
| `h2o_dumps_loads_round_trip_preserves_arrays` | H2O / sto-3g, Bohr | atom string, basis Name, default kwargs | `_atm`, `_bas`, `_env`, `ao_loc_nr`, `nao_nr`, `nelectron`, `charge`, `spin`, `cart`, `unit` |
| `cart_charge_spin_round_trip` | H+ / sto-3g, cart=true, charge=1 | non-default scalar kwargs | `_atm`, `_bas`, `_env`, `cart`, `charge` |

Both molecules pass: the `MoleSnapshot` strategy (serialise user-input shape, rebuild via `M(args)`) means the deterministic make_env / format_basis / format_ecp / make_ecp_env pipeline reproduces every derived array bit-for-bit.

## Pattern 5 Cross-Check (GTO-11 + GTO-10)

The set_geom_ tests confirm the Pattern 5 contract:

```rust
let basis_arc_before = mol.basis_set.as_ref().unwrap().clone();
set_geom_(&mut mol, "H 0 0 0; H 0 0 2.0").unwrap();
assert!(Arc::ptr_eq(mol.basis_set.as_ref().unwrap(), &basis_arc_before));  // PASS
```

Combined with `mole_copy::mole_clone_arc_identity_preserved_for_basis_set`, GTO-10 and GTO-11 reinforce: Mole.clone() and set_geom_ both leave the Arc<BasisSet> identity intact.

## v1 Limitation: Arc<BasisSet>.atoms[*].coord Staleness

After set_geom_, the libcint-facing source of truth is `_env[PTR_COORD..PTR_COORD+3]`. The `Arc<BasisSet>.atoms[*].coord` field, however, holds the **construction-time** snapshot. This is documented in:

- Module-level doc-comment in `crates/pyscf-gto/src/set_geom.rs`
- Threat register entry T-02-08-05 (accept disposition)

Downstream phases (geomopt line search, dynamics) that need a coord-fresh `BasisSet` after geometry mutation must rebuild via `pyscf_gto::M(MoleBuildArgs{ ... })` rather than reuse `mol.basis_set()`. Phase 3 SCF, which only consumes `_env` directly, is unaffected.

## Task Commits

Plan 02-08 executes as a single TDD task (Task 1 in PLAN.md) split across two atomic commits:

1. **Task 1 RED — failing tests** — `f00b837` (test): pyscf-core serde derives + 3 new test files (dumps_loads, set_geom, mole_copy). RED state confirmed by unresolved `pyscf_gto::{dumps, loads, set_geom_}` imports.
2. **Task 1 GREEN — implementation** — `12c4499` (feat): `dumps_loads.rs` + `set_geom.rs` + `lib.rs` re-exports. All 10 plan-level tests pass; broader pyscf-gto suite stays green (125 tests, 0 failures).

**Plan metadata commit (this SUMMARY):** appended at the close of execution.

## Files Created/Modified

- `crates/pyscf-gto/src/dumps_loads.rs` — `MoleSnapshot` + `pub fn dumps` + `pub fn loads`. Snapshot strategy keyed on user-input shape so deterministic rebuild reproduces derived arrays.
- `crates/pyscf-gto/src/set_geom.rs` — `pub fn set_geom_(&mut Mole, &str) -> Result<&mut Mole, _>`. Pattern 5 implementation + pre-mutation validation.
- `crates/pyscf-gto/src/lib.rs` — `pub mod dumps_loads; pub mod set_geom;` + re-exports `dumps`, `loads`, `set_geom_`. Documentation comment recording that `mol.copy()` is `Mole`'s existing `Clone` derive.
- `crates/pyscf-gto/tests/dumps_loads.rs` — 3 tests covering H2O round-trip, cart+charge+spin round-trip, malformed-JSON error handling.
- `crates/pyscf-gto/tests/set_geom.rs` — 5 tests covering env-coord update + Arc-preservation, atom-count mismatch, symbol mismatch, unit-kwarg honour, `mol.atom_coords()` reflecting the mutation.
- `crates/pyscf-gto/tests/mole_copy.rs` — 2 tests covering deep-clone of value fields and Arc identity preservation across clone (GTO-11 + GTO-10 cross-check).
- `crates/pyscf-core/src/mole.rs` — `use serde::{Deserialize, Serialize};` + `Serialize, Deserialize` derives on `Unit`, `ParsedBasis`, `ShellSpec`, `ParsedEcp`, `EcpShell`.
- `crates/pyscf-core/Cargo.toml` — explicit `serde = { workspace = true, features = ["derive"] }` (workspace already had derive; this documents the use-site).

## Decisions Made

- **Snapshot-and-rebuild over snapshot-and-restore.** Round-trip stores the user-input portion (`_atom`, `_basis`, `_ecp`, scalar kwargs) and re-runs the deterministic build pipeline. Cheaper to maintain (no parallel serde of `_atm`/`_bas`/`_env` slot-by-slot), and inherits any future build-pipeline fix automatically.
- **Force `unit=Bohr` on rebuild, then restore `mol.unit` from snapshot.** Internal coords are always Bohr; the unit label is just metadata at build time. This keeps the format_atom unit-conversion pass an idempotent no-op on round-trip.
- **set_geom_ leaves `Arc<BasisSet>.atoms[*].coord` stale.** Documented v1 limitation; `_env` is the libcint-facing source of truth post-mutation. The alternative (`Arc::make_mut` + rebuild atoms) would either deep-copy the basis (defeating zero-copy) or require a mutation API on `cintx_core::Atom` we haven't designed.
- **Pre-mutation validation in set_geom_.** Atom-count + symbol-by-symbol checks fire BEFORE any mutation, so a failed call leaves the Mole bit-identical to its prior state.
- **Serde derives on pyscf-core types, not a mirror in pyscf-gto.** Adds 5 derive lines, removes the need for ~80 lines of From/Into mirror code.

## Deviations from Plan

None — plan executed as written. The PLAN's `read_first` references to `pyscf/gto/mole.py` were used as semantic guidance (they describe the upstream deep-copy and dumps/loads contracts); the implementation followed the Discretion (D-09) of using serde_json + the snapshot-and-rebuild strategy rather than mirroring upstream's `repr()`-based JSON encoding.

The PLAN's example code in the `<action>` block referenced a few names that needed minor renaming during integration (`BasisKindSnapshot::InlineParsed` collapsed into a single `basis_per_element: HashMap<String, ParsedBasis>` field — simpler since we always have the parsed map by the time `dumps` is called on a `_built` Mole). Functionally equivalent; same byte-equality contract; cleaner snapshot type.

**Total deviations:** 0
**Impact on plan:** None — all listed acceptance_criteria items pass.

## Issues Encountered

None during the planned work. Two minor items handled inline:

- **Initial worktree base reset.** The agent worktree started at master (b7aab14, pre-Phase-02), not at the expected base (8c41e94, Phase 02-07 complete). A `git reset --hard 8c41e94...` placed the worktree at the correct base before any plan-execution edits. Documented here for reproducibility; no work lost.
- **`AtomInput::Tuples` does not call `atom_symbol`** (unlike `TupleVec`). Confirmed by inspection of `format_atom.rs` that this is fine for the round-trip path: `_atom`'s symbols are already canonical from the original build, so passing them through `Tuples` is a no-op.

## Threat Surface Scan

No new network endpoints, auth paths, file-access patterns, or schema changes at trust boundaries beyond the threats already enumerated in the PLAN's `<threat_model>`. The `loads()` parser inherits `serde_json`'s parser hardening (T-02-08-01 accept) and re-runs the existing `M(args)` validation chain (T-02-08-02 mitigate). `set_geom_` validates atom count + symbols pre-mutation (T-02-08-03 mitigate). No `Threat Flags` section needed.

## Self-Check: PASSED

**Files exist:**
- FOUND: `crates/pyscf-gto/src/dumps_loads.rs`
- FOUND: `crates/pyscf-gto/src/set_geom.rs`
- FOUND: `crates/pyscf-gto/tests/dumps_loads.rs`
- FOUND: `crates/pyscf-gto/tests/set_geom.rs`
- FOUND: `crates/pyscf-gto/tests/mole_copy.rs`
- FOUND: `crates/pyscf-gto/src/lib.rs` (modified)
- FOUND: `crates/pyscf-core/src/mole.rs` (modified)
- FOUND: `crates/pyscf-core/Cargo.toml` (modified)

**Commits exist:**
- FOUND: `f00b837` (RED — test commit)
- FOUND: `12c4499` (GREEN — feat commit)

**Acceptance criteria (from PLAN):**
- `dumps_loads.rs` contains `pub fn dumps(...) -> Result<String, PyscfRsError>` AND `pub fn loads(...) -> Result<Mole, PyscfRsError>` — yes
- `dumps_loads.rs` uses `serde_json::to_string` AND `serde_json::from_str` — yes
- `set_geom.rs` contains `pub fn set_geom_(...)` and reads `mol._atm[row_start + PTR_COORD]` and writes `mol._env[ptr_coord]` — yes (verified by grep)
- `lib.rs` re-exports `dumps`, `loads`, `set_geom_` — yes
- All 10 plan-level tests pass: 3 dumps_loads + 5 set_geom + 2 mole_copy — yes
- Full pyscf-gto suite: 125 tests, 0 failures — yes
- `cargo run -p xtask --bin check-dependency-wall` exits 0 — yes (PASS)

## Next Phase Readiness

- 02-09 verification rollup (GTO-12 phase-level closeout) can now scan all eight plan summaries.
- Phase 3 SCF can rely on `mol.set_geom_` and `mol.copy()` as stable APIs for the geomopt scaffolding seam.
- Phase 7 grad/geomopt has the primary mutation point (`set_geom_`) ready; the v1 Arc-staleness limitation is documented for the line-search loop design.

---
*Phase: 02-gto*
*Plan: 08*
*Completed: 2026-05-10*

---
phase: 03-scf-pyo3-bindings
plan: 14
subsystem: scf
tags: [init_guess, atom_hf, huckel, gwh, spherically-averaged-rhf, scf-05]

# Dependency graph
requires:
  - phase: 03-scf-pyo3-bindings
    provides: "kernel<H> SCF loop, default_get_hcore/ovlp/veff/eig/occ, make_rdm1, InitGuessMode dispatcher, atom_config (NRSRHF_CONFIGURATION + frac_occ from 03-13)"
  - phase: 02-gto
    provides: "pyscf_gto::M / intor (int1e_kin/nuc/ovlp, int2e_sph), per-element ParsedBasis, ANG_OF/ATOM_OF/ao_loc_nr AO layout"
  - phase: 05-mp2
    provides: "arity-4 int2e_sph dispatch (05-08) — required for the per-atom get_veff in the many-electron atomic RHF"
provides:
  - "crate::atom_hf::get_atm_nrhf — per-unique-element spherically-averaged atomic RHF (the shared engine atom + huckel both consume)"
  - "init_guess_by_atom — block-diagonal superposition of atomic densities (SCF-05 'atom' mode)"
  - "init_guess_by_huckel — extended-Hückel GWH guess (SCF-05 'huckel' mode, Kgwh=1.75 non-updated rule)"
  - "All 5 SCF-05 init_guess modes ('minao','atom','1e','huckel','chkfile') + user-dm0 now return Ok(Density)"
affects: [dft, mp2, ccsd, grad]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Spherically-averaged atomic RHF: group AOs by l, average the per-l Fock/overlap over the m-diagonal, eigh the nsh block, scatter eigvecs over the 2l+1 m-components"
    - "aoslice_by_atom (ATOM_OF + ao_loc_nr walk) shared by atom block-placement and huckel occupied-orbital scatter"
    - "GWH Hückel matrix in the minimal occupied-orbital basis: orb_S = orb_Cᵀ·S·orb_C, orb_H[io,jo]=0.5·Kgwh·orb_S·(Ei+Ej), eigh + back-transform"

key-files:
  created:
    - "crates/pyscf-scf/src/atom_hf.rs"
    - "crates/pyscf-scf/tests/init_guess_atom_huckel.rs"
  modified:
    - "crates/pyscf-scf/src/init_guess.rs"
    - "crates/pyscf-scf/src/lib.rs"
    - ".planning/REQUIREMENTS.md"

key-decisions:
  - "Per-element single-atom Mole built with BasisInput::Parsed(working basis for that element), spin = Z%2, cart=false (AtomSphAverageRHF does not support cartesian) — mirrors atom_hf.py:31-54"
  - "Atomic energy convergence E = 0.5·Tr[D·(h_core+Fock)]; 1-electron element (H) uses the AtomHF1e no-2e branch (E = Tr[D·h_core])"
  - "GWH non-updated rule (Kgwh=1.75 constant) — the default init_guess_by_huckel, NOT init_guess_by_mod_huckel"
  - "Cartesian-basis cart2sph branches in BOTH atom and huckel return a clear NotYetImplemented Err (spherical-only; STO-3G is spherical) — out of scope per T-03-14-SCOPE"

patterns-established:
  - "atom_hf::get_atm_nrhf is the single source of per-element atomic-RHF orbitals; consumers (atom, huckel) never re-derive"
  - "All numeric reductions (Fock-block average, dm build, orb_S/orb_H, AO back-transform, trace) go through oracle_sum/oracle_dot on materialized term Vecs — no bare += (T-03-14-NUM)"

requirements-completed: [SCF-05]

# Metrics
duration: 12min
completed: 2026-05-24
---

# Phase 3 Plan 14: atom + huckel init guesses (SCF-05 complete) Summary

**Ports the spherically-averaged atomic-RHF engine (`get_atm_nrhf`) and wires the last two init-guess modes — `atom` (block-diagonal superposition of atomic densities) and `huckel` (GWH extended-Hückel, Kgwh=1.75) — both of which seed an RHF that converges bit-identically to the `1e` guess energy (−1.1167143250625533 on H2/STO-3G).**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-05-24T11:49Z
- **Completed:** 2026-05-24T12:01Z
- **Tasks:** 3
- **Files modified:** 5 (2 created, 3 modified)

## Accomplishments

- **`crate::atom_hf::get_atm_nrhf`** — per-unique-element spherically-averaged atomic RHF (port of `pyscf/scf/atom_hf.py:27-205`). For each distinct element it builds a single-atom neutral Mole and runs a small SCF whose eig step is the **angular-averaged solve** (`AtomSphAverageRHF.eig`, atom_hf.py:109-140): group AOs by angular momentum `l`, reshape the per-`l` Fock/overlap, average over the diagonal m-index (`einsum('piqi->pq')/degen`), `eigh_gen` the `nsh×nsh` averaged block, and scatter eigvecs back over the `(2l+1)` m-components. Occupations come from `frac_occ(Z,l)` (atom_hf.py:142-171); the 1-electron element (H) takes the `AtomHF1e` no-2e branch.
- **`init_guess_by_atom`** (port of `pyscf/scf/hf.py:495-535`) — superposes each atom's density block `atm_dm[i,j]=Σ_p occ[p]·c[i,p]·c[j,p]` on the block-diagonal of the molecular `nao×nao` Density at the atom's AO range.
- **`init_guess_by_huckel`** (port of `hf.py:537-555` + `_init_guess_huckel_orbitals:577-670`, `updated_rule=False`) — collects the occupied atomic orbitals into the molecular AO basis, builds `orb_S = orb_Cᵀ·S·orb_C` and the GWH Hückel matrix `orb_H[io,jo]=0.5·Kgwh·orb_S[io,jo]·(Ei+Ej)` with **Kgwh=1.75**, solves `eigh_gen(orb_H, orb_S)`, back-transforms to AO, Aufbau-fills, and builds the RDM1.
- **Closing behavioral gate (T-03-14-CORRECT) PASSES**: RHF seeded with `'atom'` and `'huckel'` each converge to the SAME e_tot as the `'1e'` guess on H2/STO-3G — all three = **−1.1167143250625533** (bit-identical, far inside the ~1e-6 tolerance).
- **SCF-05 flips `[~]` → `[x]`**: all five init_guess modes + user-dm0 now ship.

## Source-of-truth refs

| Function | Upstream | Notes |
|----------|----------|-------|
| `get_atm_nrhf` | `pyscf/scf/atom_hf.py:27-86` | per-unique-element cache; single-atom neutral Mole |
| `AtomSphAverageRHF.eig` | `atom_hf.py:109-140` | angular-averaged per-l solve + m-scatter |
| `AtomSphAverageRHF.get_occ` | `atom_hf.py:142-171` | spherically-averaged fractional occupancy |
| `AtomHF1e` | `atom_hf.py:192-193` | 1-electron (H) no-2e branch |
| `init_guess_by_atom` | `pyscf/scf/hf.py:495-535` | block-diagonal superposition |
| `init_guess_by_huckel` | `hf.py:537-555` | calls `_init_guess_huckel_orbitals(updated_rule=False)` |
| `_init_guess_huckel_orbitals` | `hf.py:577-670` | orb_C/orb_E/orb_S/orb_H build + back-transform |
| `Kgwh` | `hf.py:563-575` | **Kgwh = 1.75** (non-updated rule) |

## Measured numbers (H2/STO-3G)

- **atom guess** `Tr(D·S) = 2.0` (exact — built from normalized atomic orbitals; the minao non-normalization caveat does NOT apply)
- **huckel guess** `Tr(D·S) = 2.000000000000001` (≈ nelec)
- **RHF e_tot** (1e / atom / huckel seeds) = **−1.1167143250625533** for all three, bit-identical

## Task Commits

Each task was committed atomically (explicit-path staging, hooks on, no `--no-verify`):

1. **Task 1: port get_atm_nrhf** — `03c7c9f` (feat)
2. **Task 2: init_guess_by_atom** — `a29113d` (feat)
3. **Task 3: init_guess_by_huckel + converge gate + SCF-05 [x]** — `bf8cb0f` (feat)

## Files Created/Modified

- `crates/pyscf-scf/src/atom_hf.rs` (created) — `get_atm_nrhf` + `AtomScfResult`; angular-averaged eig; spherical occ; per-atom small SCF loop; 3 unit tests (H occ→1.0, O per-l occ matches frac_occ, unique-element caching).
- `crates/pyscf-scf/src/init_guess.rs` (modified) — `init_guess_by_atom`, `init_guess_by_huckel`, shared `aoslice_by_atom` helper, `KGWH` constant; both dispatcher arms wired (the two `InitGuessNotYetImplemented` stubs removed).
- `crates/pyscf-scf/src/lib.rs` (modified) — `mod atom_hf;` registered.
- `crates/pyscf-scf/tests/init_guess_atom_huckel.rs` (created) — atom + huckel `Tr(D·S)≈nelec` sanity checks; the RHF-converges-to-1e-energy closing gate.
- `.planning/REQUIREMENTS.md` (modified) — SCF-05 `[~]` → `[x]` + traceability row updated.

## oracle_sum / oracle_dot invocation sites

- `atom_hf.rs`: 10 sites (Fock-block average + overlap-block average, atomic dm build, trace_dm_times row + diag reductions).
- `init_guess.rs` atom+huckel bodies: 6 sites (atom dm; huckel `S·orb_C`, `orb_S` via oracle_dot, AO back-transform).
- No bare `+=` / `.sum()` / `.fold` in any numeric accumulation; no FMA.

## Dependency / build hygiene

- **`cargo tree -p pyscf-scf` shows 0 libxc_rs** — no new crate dependency added.
- `cargo test -p pyscf-scf -p pyscf-gto` exits 0 (no regressions; 3 pre-existing CI-gated upstream-byte-identity tests stay ignored).
- `cargo clippy -p pyscf-scf --lib --tests -- -D warnings` clean; `cargo fmt -p pyscf-scf` clean.
- `xtask check-no-fma` PASS; `xtask check-dependency-wall` PASS.

## Decisions Made

See `key-decisions` frontmatter. Notably: spin = Z%2 per upstream; 1-electron H uses the no-2e branch; GWH non-updated rule (Kgwh=1.75); cartesian cart2sph branches return a clear `Err` (spherical-only scope).

## Deviations from Plan

**1. [Rule 3 - Blocking] Module-scoped `#![allow(dead_code)]` on atom_hf.rs in the Task-1 commit**
- **Found during:** Task 1 (get_atm_nrhf)
- **Issue:** `pub(crate) get_atm_nrhf` and its helpers are unused in the non-test build until Tasks 2/3 wire them into `init_guess.rs`; the crate's `-D warnings` gate failed the standalone Task-1 commit with dead-code errors.
- **Fix:** Added a documented module-scoped `#![allow(dead_code)]` so the Task-1 commit is independently `-D warnings`-clean; it becomes inert once the consumers land (Tasks 2/3 reached it). Kept module-scoped (not crate-wide).
- **Files modified:** `crates/pyscf-scf/src/atom_hf.rs`
- **Verification:** `cargo clippy -p pyscf-scf --lib -- -D warnings` clean at each commit.
- **Committed in:** `03c7c9f` (Task 1 commit)

---

**Total deviations:** 1 auto-fixed (1 blocking).
**Impact on plan:** Minimal — the allow keeps per-task commits independently green during interface-first ordering; no scope creep, no algorithm change.

## Issues Encountered

None. The angular-averaged eig + spherical occ produced the correct H/O occupations on the first run, and the converge gate matched the 1e energy bit-for-bit.

## Out of Scope / Honest gaps

- **Cartesian-basis** cart2sph branches (atom hf.py:528-531, huckel atcart2sph) — return a clear `Err`; STO-3G / the working-basis tests are spherical.
- **Upstream byte-identity** of converged atom/huckel-seeded SCF energies — CI-gated/human-verify (sandbox has no maturin/upstream-pyscf); the in-tree gate proves mode-independence at convergence instead.
- **mulliken_meta** (SCF-09) was plan 03-15 (already complete); not touched here.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- SCF-05 fully closed — all 5 init_guess modes + user-dm0 ship; the default `minao` path and the `atom`/`huckel` paths all converge in-tree.
- No blockers introduced. `get_atm_nrhf` is available for any future consumer (e.g. atomic-density-based analysis).

## Self-Check: PASSED

- Created files exist: `crates/pyscf-scf/src/atom_hf.rs`, `crates/pyscf-scf/tests/init_guess_atom_huckel.rs`, `.planning/phases/03-scf-pyo3-bindings/03-14-SUMMARY.md`.
- Commits exist: `03c7c9f` (Task 1), `a29113d` (Task 2), `bf8cb0f` (Task 3).

---
*Phase: 03-scf-pyo3-bindings*
*Completed: 2026-05-24*

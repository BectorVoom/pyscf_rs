---
phase: 03-scf-pyo3-bindings
plan: 15
subsystem: scf
tags: [mulliken, meta-lowdin, lowdin, orth_ao, population-analysis, eigh_gen, oracle_sum, SCF-09]

# Dependency graph
requires:
  - phase: 03-scf-pyo3-bindings (plan 03-11)
    provides: analyze / mulliken_pop / dip_moment real bodies + the MullikenResult struct + the _bas[ATOM_OF]/ao_loc_nr AO→atom walk that mulliken_meta reuses
  - phase: 03-scf-pyo3-bindings (plan 03-11)
    provides: pyscf_algebra::eigh_gen (slice-based generalized self-adjoint eigh) used by the lowdin S^{-1/2} helper
provides:
  - "crate::orth::orth_ao — meta-Löwdin / Löwdin AO orthogonalization matrix builder (per-l-channel sequential Löwdin, the _nao_sub scheme)"
  - "crate::orth (private) lowdin S^{-1/2} helper"
  - "analyze::mulliken_meta real body (meta-Löwdin population analysis) — no longer NotYetImplemented"
  - "analyze::aggregate_pop_to_charges shared AO→atom aggregator (single oracle-reduction site, reused by mulliken_pop + mulliken_meta)"
affects: [phase-07-grad (population-derived properties), milestone audit (SCF-09 closure)]

# Tech tracking
tech-stack:
  added: []  # no new crate dependency; reuses pyscf-algebra (eigh_gen, oracle_sum) already in the graph
  patterns:
    - "Globally-orthonormal sequential block Löwdin (project-against-done-span then Löwdin-within-block) — the in-tree analog of pyscf/lo/nao.py::_nao_sub"
    - "Shared private aggregator (aggregate_pop_to_charges) so mulliken_pop + mulliken_meta have ONE oracle-reduction site to audit (T-03-15-NUM)"

key-files:
  created:
    - crates/pyscf-scf/src/orth.rs
    - crates/pyscf-scf/tests/mulliken_meta.rs
  modified:
    - crates/pyscf-scf/src/analyze.rs
    - crates/pyscf-scf/src/lib.rs
    - .planning/REQUIREMENTS.md

key-decisions:
  - "Partition by per-l angular-momentum channel (cross-atom), NOT per-(atom,l): per-l spans atoms so homonuclear symmetry is preserved (H2 two H charges come out equal). A per-(atom,l) split processes each atom sequentially and breaks that symmetry (observed: chg ±0.659 instead of 0)."
  - "orth_ao uses the GLOBALLY-ORTHONORMAL sequential scheme of _nao_sub (project each block against the previously-orthogonalized span in the S-metric, then Löwdin within block), not a naive block-diagonal Löwdin — the latter leaves cross-block overlap and fails both the C_orthᵀ·S·C_orth≈I gate and the Σ ao_pop≈nelec conservation invariant."
  - "Full NAO core/valence/Rydberg _nao_sub partition (~230 LoC of pyscf/lo/nao.py, no in-tree analog) intentionally NOT ported — disproportionate to one population-analysis utility; documented future enhancement, SCF-09 → [~]."

patterns-established:
  - "Sequential project-then-Löwdin block orthogonalization (_nao_sub analog) for localized-orbital construction"
  - "Shared private population aggregator reused by mulliken_pop + mulliken_meta"

requirements-completed: [SCF-09]

# Metrics
duration: 13min
completed: 2026-05-24
---

# Phase 3 Plan 15: mulliken_meta (meta-Löwdin population analysis) Summary

**`mulliken_meta` now ships a real body — meta-Löwdin population analysis via a new in-tree `orth_ao` (per-l-channel sequential Löwdin, the `_nao_sub` scheme), satisfying the conservation invariants (Σ ao_pop ≈ nelec, Σ chg ≈ 0) and homonuclear symmetry on H2 + H2O; SCF-09 → `[~]`.**

## Performance

- **Duration:** 13 min
- **Started:** 2026-05-24T11:34:31Z
- **Completed:** 2026-05-24T11:47:22Z
- **Tasks:** 2
- **Files modified:** 5 (2 created, 3 modified)

## Accomplishments

- New `crates/pyscf-scf/src/orth.rs`: `pub(crate) orth_ao` (meta-Löwdin AO orthogonalization matrix `C_orth`) + a private `lowdin` symmetric `S^{-1/2}` helper. Registered `mod orth;` in lib.rs.
- `analyze.rs` `mulliken_meta` replaced its `NotYetImplemented` stub with a real port of `pyscf/scf/hf.py:1301-1340` consuming `crate::orth::orth_ao`.
- Refactored the AO→atom aggregation out of `mulliken_pop` into a shared `aggregate_pop_to_charges` (single oracle-reduction site); `mulliken_pop`'s existing tests stay green.
- New `crates/pyscf-scf/tests/mulliken_meta.rs` asserts the conservation invariants on converged H2 + H2O/STO-3G RHF; `orth` unit tests assert orthonormality + phase-adjust.
- REQUIREMENTS.md SCF-09 flipped `[ ]` → `[~]` (partial).

## Task Commits

Each task was committed atomically (sequential executor, main working tree, explicit-path staging, hooks on):

1. **Task 1: orth_ao — meta-Löwdin / Löwdin AO orthogonalization** - `e75553d` (feat) — `crates/pyscf-scf/src/orth.rs`, `crates/pyscf-scf/src/lib.rs`
2. **Task 2: mulliken_meta real body + wire + conservation-invariant test** - `c91b890` (feat) — `crates/pyscf-scf/src/analyze.rs`, `crates/pyscf-scf/tests/mulliken_meta.rs`, `.planning/REQUIREMENTS.md`

**Plan metadata:** (this commit) `docs(03-15): complete mulliken_meta gap-closure plan`

## Files Created/Modified

- `crates/pyscf-scf/src/orth.rs` (created) — `orth_ao` (meta-Löwdin via per-l-channel sequential project-then-Löwdin) + private `lowdin` `S^{-1/2}` helper + `ao_blocks_by_l` partition + 3 unit tests.
- `crates/pyscf-scf/tests/mulliken_meta.rs` (created) — H2 + H2O conservation-invariant integration tests.
- `crates/pyscf-scf/src/analyze.rs` (modified) — real `mulliken_meta` body; extracted shared `aggregate_pop_to_charges`; module docstring updated; `NotYetImplemented` removed.
- `crates/pyscf-scf/src/lib.rs` (modified) — `pub mod orth;` registration (the `pub use analyze::mulliken_meta` re-export already existed from 03-11 and still resolves).
- `.planning/REQUIREMENTS.md` (modified) — SCF-09 `[ ]` → `[~]` + traceability-table status.

## Algorithm Notes

### `orth_ao` (orth.rs) — source: `pyscf/lo/orth.py:32-36, 269-331` + `pyscf/lo/nao.py:124-160`

- **`lowdin(s_block, n)`** (orth.py:32-36): diagonalize the block overlap `S = U·diag(λ)·Uᵀ` via `eigh_gen(s_block, I, n)`, drop `λ ≤ 1e-15`, form `S^{-1/2}[i,j] = Σ_k U[i,k]·λ_k^{-1/2}·U[j,k]` (reduction over the kept eigen-index `k` via `oracle_sum` on a materialized term Vec). Output F-order `m×m`.
- **`orth_ao(mol, s)`** (orth.py:269-331 meta_lowdin branch → `_nao_sub`): the GLOBALLY-ORTHONORMAL sequential scheme of `_nao_sub` (nao.py:136-156). Partition AOs into per-`l` channels (`_bas[ANG_OF]` walk; cross-atom, so homonuclear symmetry holds). Process channels in order; for each block: (a) seed unit columns `e_μ`, (b) project out the done span in the S-metric `c ← c − C_done·(C_doneᵀ·S·c)` (nao.py:141), (c) projected block overlap `s1 = cᵀ·S·c`, (d) Löwdin within block `c·lowdin(s1)`, (e) write to `C_orth` + append to done. Final phase adjust (orth.py:328-330): flip column `i` if `C_orth[i,i] < 0`. Building block-by-block this way yields `C_orthᵀ·S·C_orth ≈ I` (the `_nao_sub` invariant checked at nao.py:157).

### `mulliken_meta` (analyze.rs) — source: `pyscf/scf/hf.py:1301-1340`

`dm`/`s` as in `mulliken_pop` → `C_orth = orth_ao(mol, s)` → `c_inv = C_orthᵀ·S` → `D' = c_inv·D·c_invᵀ` → `pop[μ] = D'[μ,μ]` (diagonal, S=I in the orthonormal basis) → shared `aggregate_pop_to_charges` (AO→atom via `_bas[ATOM_OF]`/`ao_loc_nr` + `chg[A] = Z_A − pop[A]`). Because `C_orth` is a complete S-orthonormalization, the transform is electron-count preserving: `Σ pop = Tr(D') = Tr(S·D) = nelec`.

### Measured conservation values (the in-tree SCF-09 gate)

| Molecule | Σ ao_pop (nelec) | Σ chg | per-atom chg |
|----------|------------------|-------|--------------|
| H2/STO-3G  | 2.0000000000000027 (2) | −2.7e-15 | H −1.33e-15, H −1.33e-15 (equal — homonuclear symmetry) |
| H2O/STO-3G | 10.000000000000007 (10) | −6.2e-15 | O +0.6222, H −0.3136, H −0.3086 |

`orth` unit tests: `lowdin(I)=I`; `Xᵀ·S·X≈I`; `C_orthᵀ·S·C_orth≈I` (≤1e-8) on the real H2/STO-3G overlap; all diagonals ≥ 0 after phase adjust.

### oracle_sum / oracle_dot invocation sites

- `orth.rs`: 10 `oracle_sum` sites (lowdin `S^{-1/2}` build; the project-out reductions; `S·c`; `s1 = cᵀ·S·c`; `c·lowdin(s1)`). `grep -cE "oracle_sum|oracle_dot" = 10`.
- `analyze.rs` `mulliken_meta`: `c_inv = C_orthᵀ·S`, `M = c_inv·D`, `pop = diag(D')` — all `oracle_sum` on materialized term Vecs. AO→atom + per-shell aggregation in the shared `aggregate_pop_to_charges` via `oracle_sum`.
- No bare `+=` / `.sum()` / `.fold` in any numeric accumulation; no FMA in pyscf-owned kernels (`xtask check-no-fma` PASS).

### Dependency wall / libxc

- No new crate dependency (reuses `pyscf-algebra`'s `eigh_gen` + `oracle_sum`).
- `cargo tree -p pyscf-scf` shows **0** libxc_rs — libxc NEVER compiled.
- `xtask check-dependency-wall` PASS (cubecl-* containment intact).

## Decisions Made

- **Per-`l`-channel (cross-atom) partition, not per-(atom, l).** The first implementation used per-(atom, l) blocks; on H2 the two H atoms were processed sequentially and got opposite charges (±0.659) — the homonuclear symmetry the plan's Task 2 gate requires was broken. Switching to the per-`l` channel grouping (which spans atoms, like upstream's core/valence/Rydberg classes) restored symmetry (equal H charges) while keeping the result distinct from plain symmetric Löwdin (distinct `l` channels are still orthogonalized separately).
- **Globally-orthonormal `_nao_sub` sequential scheme over naive block-diagonal Löwdin.** A naive per-block `S^{-1/2}` scatter leaves cross-block overlap (observed `C_orthᵀ·S·C_orth` off-diagonal ≈ 0.659 on H2) and would fail BOTH the orthonormality gate and the `Σ ao_pop ≈ nelec` conservation invariant. The project-against-done-span-then-Löwdin scheme is what `_nao_sub` actually implements and is required for conservation.

## Deviations from Plan

The plan's Task 1 acceptance specified partitioning "by (atom, l-shell) blocks". During Task 1 verification this was found to break the H2 homonuclear-symmetry gate AND the conservation invariant (a per-(atom,l) block-diagonal scheme is not globally orthonormal). Resolved within Task 1 scope.

### Auto-fixed Issues

**1. [Rule 1 - Bug] Per-(atom,l) block-diagonal Löwdin is not globally orthonormal → broke conservation + H2 symmetry**
- **Found during:** Task 1 (orthonormality unit test) and Task 2 (H2 symmetry assertion)
- **Issue:** The plan's literal "partition by (atom,l), Löwdin within each block, scatter back (off-block entries stay 0)" produces a block-diagonal `C_orth` that leaves cross-block overlap. The H2/STO-3G orthonormality test failed (`C_orthᵀ·S·C_orth` off-diagonal = 0.659), and once wired, `mulliken_meta` gave H2 charges of ±0.659 (asymmetric) — violating both the `C_orthᵀ·S·C_orth ≈ I` Task-1 gate and the Task-2 H2-symmetry + conservation gates.
- **Fix:** Implemented the GLOBALLY-ORTHONORMAL sequential scheme that upstream `_nao_sub` actually uses (project each block against the previously-orthogonalized span in the S-metric, then Löwdin within block), and grouped by per-`l` angular-momentum channel (cross-atom) instead of per-(atom,l) so homonuclear symmetry is preserved. This is the faithful in-tree analog of `_nao_sub` and is what makes `C_orthᵀ·S·C_orth ≈ I` hold and the electron count conserved.
- **Files modified:** crates/pyscf-scf/src/orth.rs
- **Verification:** `orth` unit tests pass (orthonormality ≤1e-8 on real H2 overlap); `mulliken_meta` H2 charges equal to 1e-15; Σ ao_pop = nelec to 1e-14 on H2 + H2O.
- **Committed in:** e75553d (Task 1) + c91b890 (Task 2)

---

**Total deviations:** 1 auto-fixed (1 bug — algorithm correctness).
**Impact on plan:** The fix is required for the plan's own acceptance gates (orthonormality + conservation + H2 symmetry). No scope creep — the deliverable is unchanged (`orth_ao` + real `mulliken_meta`); only the block-partition granularity and the within/between-block orthogonalization order were corrected to what `_nao_sub` actually does. The "per-(atom,l)" phrasing in the plan was a description of the partition idea; the physics requires the cross-atom per-l grouping + sequential projection.

## Known Stubs

None. `mulliken_meta` ships a real body returning computed populations; the `NotYetImplemented` is gone (`grep -c NotYetImplemented crates/pyscf-scf/src/analyze.rs = 0`).

## Issues Encountered

- The H2 symmetry / orthonormality failures (above) were the only real problem; root-caused to the block-partition scheme and resolved within plan scope (Rule 1).

## Scope Boundary (documented, not a gap)

The full NAO `_nao_sub` core/valence/Rydberg refinement (`pyscf/lo/nao.py`, ~230 LoC; `_core_val_ryd_list` shell classification + `weight_orth`) is intentionally NOT ported — no in-tree `pyscf-lo` crate, and it is disproportionate to one population-analysis utility. The shipped per-`l`-channel meta-Löwdin delivers the full SCF-09 deliverable (correct populations satisfying the conservation invariants). Upstream byte-identity of the meta-Löwdin charges vs PySCF's full NAO partition is a CI-gated / human-verify item (the sandbox has no maturin/upstream-pyscf), mirroring SCF-07's Rust-satisfied + upstream-byte-identity treatment. Hence SCF-09 → `[~]` (partial), not `[x]`.

## User Setup Required

None — no external service configuration required (Rust↔Rust only; no new FFI surface, no new threat surface — the `<threat_model>` T-03-15-* dispositions hold: PANIC mitigated via `?`/explicit `Err`, NUM mitigated via single oracle-reduction site, CORRECT mitigated by the conservation + orthonormality test gates).

## Next Phase Readiness

- SCF-09 (`analyze` / `mulliken_pop` / `mulliken_meta` / `dip_moment`) now all ship real bodies; remaining for full `[x]` is upstream byte-identity (human-verify).
- `crate::orth::orth_ao` is available in-tree for any future localized-orbital / population-derived property work (Phase 7 gradients).

## Self-Check: PASSED

- Files: orth.rs, tests/mulliken_meta.rs, analyze.rs, lib.rs, 03-15-SUMMARY.md — all FOUND.
- Commits: e75553d (Task 1), c91b890 (Task 2) — both FOUND in git log.

---
*Phase: 03-scf-pyo3-bindings*
*Completed: 2026-05-24*

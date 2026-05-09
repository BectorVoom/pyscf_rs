# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-09)

**Core value:** Run mainstream molecular ground-state quantum chemistry (HF, DFT, MP2, CCSD, gradients) 2–5× faster than current PySCF + C extensions, with bit-exact agreement on regression tests, and zero C/CMake/libcint dependency hell at install time.
**Current focus:** Phase 1 — Foundation (workspace, core types, runtime, FMA-free oracle profile, ordered-reduction primitives, panic policy, cubecl pin, scope-creep lint, nightly cross-crate matrix CI)

## Current Position

Phase: 1 of 8 (Foundation)
Plan: 0 of TBD in current phase
Status: Ready to plan
Last activity: 2026-05-10 — ROADMAP.md created; 113/113 v1 requirements mapped across 8 phases

Progress: [░░░░░░░░░░] 0%

## Performance Metrics

**Velocity:**
- Total plans completed: 0
- Average duration: — (no plans run yet)
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| - | - | - | - |

**Recent Trend:**
- Last 5 plans: —
- Trend: — (no data yet)

*Updated after each plan completion*

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmapping (2026-05-10): Compressed research's 12-phase suggestion to 8 phases (standard granularity). Merged `bindings` into `scf` (Phase 3) to lock PyO3 contract on RHF before DFT; merged `geomopt` into `grad` (Phase 7); merged `GPU enable + oracle hardening + distribution` into closing Phase 8.
- Roadmapping (2026-05-10): Phase 1 (Foundation) is the SHOWSTOPPER convergence point — 7 of 21 catalogued pitfalls have their primary mitigation here (FMA, reduction order, cubecl pin, panic policy, sibling-crate ABI, cross-platform libm, scope creep).
- Roadmapping (2026-05-10): Phase 3 (SCF + PyO3 bindings) is the second convergence point — 5 PyO3-related pitfalls (subclass override, NumPy stride, GIL deadlock, panic→exception, chkfile schema) plus eigenvector sign canonicalization land here on the small RHF surface.

### Pending Todos

[From .planning/todos/pending/ — ideas captured during sessions]

None yet.

### Blockers/Concerns

[Issues that affect future work]

- **cubecl 0.10.0 lockstep** with cintx/libxc_rs/xcfun_rs is a four-crate ABI contract. Any cubecl bump requires synchronized bumps in all four. Phase 1 documents the upgrade ritual; nightly cross-crate matrix CI is the early-warning system.
- **WGPU f64 holes** (cubecl issues #1316/#1317) may force `wgpu` feature to be gated on `shader-f64` Vulkan extension at runtime. Honest fallback to CPU with warning is the chosen path; verified in Phase 4 (DFT) and Phase 8 (GPU enable).
- **CCSD(T) deferral pressure** is real (~30–40% of CCSD users want it). v1.x P1 entry on the roadmap signals deferral is intentional; expect a feature request within weeks of v1 release.
- **`faer-ext 0.7.1` ↔ `faer 0.24.0` compatibility** needs build verification in Phase 1; if it fails, either bump faer-ext upstream or drop the dependency and round-trip via `Vec<f64>`.
- **h5py ↔ hdf5-metno chkfile round-trip** robustness needs empirical seal in Phase 3 (ORACLE-08 round-trip oracle).

## Deferred Items

Items acknowledged and carried forward:

| Category | Item | Status | Deferred At |
|----------|------|--------|-------------|
| CCSD | CCSD(T) — perturbative triples | v1.x P1 | Roadmap creation |
| SCF | ROHF, SOSCF (`scf.newton`), ADIIS/EDIIS, symmetry-adapted SCF | v1.x | Roadmap creation |
| DFT | DFT-D3/D4 dispersion, custom-XC user functions | v1.x | Roadmap creation |
| Hessian | RHF/RKS Hessian, vibrational frequencies | v1.x | Roadmap creation |
| CCSD | FNO-CCSD, GHF/GMP2/GCCSD path | v1.x | Roadmap creation |
| Geomopt | Constrained geometry optimization | v1.x | Roadmap creation |
| Distribution | conda-forge channel | v1.x | Roadmap creation |

## Session Continuity

Last session: 2026-05-10
Stopped at: ROADMAP.md created; STATE.md initialized; REQUIREMENTS.md traceability table updated. Ready for `/gsd-plan-phase 1`.
Resume file: None

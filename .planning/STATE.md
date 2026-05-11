---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Phase 3 context gathered
last_updated: "2026-05-11T20:58:32.660Z"
last_activity: 2026-05-11
progress:
  total_phases: 8
  completed_phases: 1
  total_plans: 30
  completed_plans: 28
  percent: 93
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-09)

**Core value:** Run mainstream molecular ground-state quantum chemistry (HF, DFT, MP2, CCSD, gradients) 2–5× faster than current PySCF + C extensions, with bit-exact agreement on regression tests, and zero C/CMake/libcint dependency hell at install time.
**Current focus:** Phase 03 — scf-pyo3-bindings

## Current Position

Phase: 4
Plan: Not started
Status: Executing Phase 03
Last activity: 2026-05-11 - Completed quick task 260512-8wb: rewrite cintx#11 as cintx-only Phase 2 task list

Progress: [█████████░] 88% (7/7 plans done; verification gaps remain)

## Performance Metrics

**Velocity:**

- Total plans completed: 20
- Average duration: — (no plans run yet)
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02 | 9 | - | - |
| 03 | 11 | - | - |

**Recent Trend:**

- Last 5 plans: —
- Trend: — (no data yet)

*Updated after each plan completion*
| Phase 02 P01 | 12min | 3 tasks | 13 files |
| Phase 02 P02 | 8min | 2 tasks | 9 files |

## Accumulated Context

### Decisions

Decisions are logged in PROJECT.md Key Decisions table.
Recent decisions affecting current work:

- Roadmapping (2026-05-10): Compressed research's 12-phase suggestion to 8 phases (standard granularity). Merged `bindings` into `scf` (Phase 3) to lock PyO3 contract on RHF before DFT; merged `geomopt` into `grad` (Phase 7); merged `GPU enable + oracle hardening + distribution` into closing Phase 8.
- Roadmapping (2026-05-10): Phase 1 (Foundation) is the SHOWSTOPPER convergence point — 7 of 21 catalogued pitfalls have their primary mitigation here (FMA, reduction order, cubecl pin, panic policy, sibling-crate ABI, cross-platform libm, scope creep).
- Roadmapping (2026-05-10): Phase 3 (SCF + PyO3 bindings) is the second convergence point — 5 PyO3-related pitfalls (subclass override, NumPy stride, GIL deadlock, panic→exception, chkfile schema) plus eigenvector sign canonicalization land here on the small RHF surface.
- Algebra integration (2026-05-10): added a dedicated `pyscf-algebra` crate as the single owner of all linear algebra; only `pyscf-algebra` (and `pyscf-runtime` for client construction) may depend on `cubecl-*` runtime crates — enforced by a `cargo metadata` dependency-wall lint. Workspace grows 14 → 15 members.
- Algebra integration (2026-05-10): workspace `gpu` umbrella feature is OFF by default; CPU is the default backend. Per-backend features `cuda`/`wgpu`/`rocm`/`metal` opt in to each cubecl runtime at compile time. `PYSCF_BACKEND` env var selects among compiled-in backends at runtime; unrecognised/uncompiled values fall back to CPU with a `tracing::warn!`.
- Algebra integration (2026-05-10): host eigh/Cholesky/QR/SVD remain on `faer 0.24` behind the `pyscf-algebra` surface — even on a GPU build, these routines copy to host. Documented as the single intentional host-fallback path until `cubecl-linalg` ships an eigh.
- [Phase 02]: Wave 0 complete: cintx + cubecl-cpu reach proven; pyscf-kernels added to algebra-wall allowlist; 23-entry intor layout table; oracle harness scaffold + env-var docs in place
- [Phase 02]: pyscf-gto uses direct per-member cintx path-deps (cintx-core, cintx-rs, cintx-compat, cintx-ops, cintx-runtime) — workspace [patch.crates-io] cintx redirect alone is insufficient for subcrate consumers
- [Phase 02]: cubecl 0.10.0 ArrayArg::from_raw_parts signature is (Handle, usize) by value — no vectorization arg, no turbofish (older 0.9-era README sketch is stale)
- [Phase 02]: [Phase 02]: Mole >=30 attribute floor + format_atom 4-of-5 atom-input forms shipped via pyscf_gto::M(MoleBuildArgs); 5th Callable form returns NotYetImplemented{phase:3}; Local raw_atm_layout slot constants in pyscf-core::basis_set are TEMPORARY (02-04 deletes once cintx-compat dep lands)

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

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260512-8jv | Create issue in cintx repository about remaining tasks from pyscf_rs Phase 2 ([cintx#11](https://github.com/BectorVoom/cintx/issues/11)) | 2026-05-11 | 7dcdf08 | [260512-8jv-create-issue-in-cintx-repository-about-r](./quick/260512-8jv-create-issue-in-cintx-repository-about-r/) |
| 260512-8wb | Rewrite cintx#11 as cintx-only Phase 2 task list (drop pyscf_rs framing) | 2026-05-11 | f53cc0e | [260512-8wb-rewrite-cintx-11-as-cintx-only-phase-2-t](./quick/260512-8wb-rewrite-cintx-11-as-cintx-only-phase-2-t/) |

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

Last session: 2026-05-11T03:13:56.280Z
Stopped at: Phase 3 context gathered
Resume file: .planning/phases/03-scf-pyo3-bindings/03-CONTEXT.md

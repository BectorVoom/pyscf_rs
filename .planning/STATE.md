---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: "Phase 04 executed (10/10 plans) — verification gaps_found (3/5): 4 BLOCKERs (UKS dead/closed-shell, f32 0.0-substitution, l>4 panic, non-injective XC-cache key); gap closure required before phase complete"
last_updated: "2026-05-22T10:49:47.844Z"
last_activity: "2026-05-22 -- Completed 04-08 (RKS::density_fit + DfKsHooks routing the Coulomb-J build through pyscf_df::get_jk_df (J_df/K_standard split) while Vxc/K stay standard DFT-07/D-10; KsResult wraps ScfResult + impl Checkpointable with xc/grids schema metadata via pyscf-chkfile primitives, no own hdf5-metno dep D-05/D-06; DFT-07 energy + ORACLE-08 h5py gates CI-only)"
progress:
  total_phases: 8
  completed_phases: 1
  total_plans: 40
  completed_plans: 38
  percent: 13
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-09)

**Core value:** Run mainstream molecular ground-state quantum chemistry (HF, DFT, MP2, CCSD, gradients) 2–5× faster than current PySCF + C extensions, with bit-exact agreement on regression tests, and zero C/CMake/libcint dependency hell at install time.
**Current focus:** Phase 04 — dft

## Current Position

Phase: 04 (dft) — EXECUTING
Plan: 04-08 complete (Wave 5 — DF-DFT RKS::density_fit + DfKsHooks Coulomb-J via pyscf_df::get_jk_df DFT-07/D-10; KsResult impl Checkpointable with xc/grids metadata D-06)
Status: Phase 04 verification = gaps_found (3/5 criteria; 4 BLOCKERs) — gap closure required before complete
Last activity: 2026-05-22 -- Completed 04-08 (DF-DFT via pyscf-df get_jk_df J-build, J_df/K_standard split, Vxc/K stay standard DFT-07/D-10; KsResult wraps ScfResult + Checkpointable with xc/grids schema metadata via pyscf-chkfile primitives, no own hdf5-metno dep D-05/D-06; DFT-07 energy + ORACLE-08 h5py gates CI-only)

Progress: [█████████░] 95% (38/40 plans done across all phases; Phase 04: 10/10 plans summarized)

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
| Phase 04 P04-04 | 16min | 2 tasks | 12 files |
| Phase 04 P04-05 | 16min | 2 tasks | 9 files |
| Phase 04 P04-06 | 23min | 2 tasks | 14 files |
| Phase 04 P04-07 | 12min | 2 tasks | 9 files |
| Phase 04 P04-08 | 14min | 2 tasks | 7 files |

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
- [Phase 04]: pyscf-grids byte-exact Becke grids (DFT-04/09) — generator-port Lebedev (SphGenOh + inline LEBEDEV_SEEDS, D-06, no codegen/build.rs), Treutler-Ahlrichs class-default radial, get_partition pure-Python fallback with pbecke.sum(axis=0) through oracle_sum (Pitfall 10). DFT-09 count sweep matches upstream level 0..9; DFT-04 byte-for-byte coords+weights is a CI-only grid_weights oracle arm (--features python).
- [Phase 04]: XC parsers + XcBackend seam (DFT-02/03) — libxc-default parse_xc (D-01, inline const XC_CODES/XC_ALIAS, part-aware possible_*_for fuzzy lookup, depth-bounded compound expansion T-04-05b) + xcfun-alternate parse_xc (0..77 ids, X/C/XC suffix fallback, LR_HF-zeroing tail). XcBackend cfg-gated enum mirrors AlgebraClient: Xcfun default-compiled, #[cfg(libxc)] Libxc in a gated submodule (default build never names a libxc_rs symbol). xcfun eval uses spin-resolved Vars (A_B/A_B_GAA_GAB_GBB/+TAU) with closed-shell rho/2 split (CPU launch supports spin-resolved only; Vars::N/A => NotConfigured). DFT-02 oracle = hand-transcribed parity table (PyO3-wall: no pyo3 dep in pyscf-dft); SLATERX bit-exact 1e-10 vs analytic. libxc NEVER compiled (cargo tree default = 0 libxc_rs).
- [Phase 04]: RKS/UKS core (DFT-01/08/10/11, D-07/D-08) — NumInt grid loop (nr_rks/nr_uks/eval_rho/eval_xc, upstream numint.py signatures) is algebra-orchestrated (AO via pyscf_gto::eval_gto behind the wall; dense ρ/Vxc contractions as host loops; Exc/nelec via oracle_sum) with NO #[cube] kernel (D-07; Tensor-API gemm/axpy stay NotYetImplemented{phase:2}, so the grid loop follows the Phase-3 SCF/DF inline-loop precedent). PARSE XC IN THE XCFUN NAMESPACE (default backend) — xcfun exposes the standard-hybrid mixing in hyb[0] (b3lyp→0.2); the libxc parser folds it inside compound id 402 (hyb=0), so using libxc::parse_xc would silently break hybrid_coeff AND feed libxc ids into the xcfun id→name map. D-08: NumInt reads DType::from_env() at construction + read-only dtype() accessor; f32/f64 enum-match dispatch of the matmul chain (F64 arm = unchanged bit-exact default; F32 casts ρ→f64 at the XcBackend::eval boundary since eval_gto/xcfun are f64-host) + one below-bit-exact tracing::warn!; no set_precision, no f32 tolerance gate. KS get_veff = J+Vxc−hyb·K (RSH omega!=0 seam → 04-07); KsHooks overrides energy_elec = Tr(D·h1e)+Ecoul+Exc via a per-cycle Exc cache (the SCF energy_elec signature has no mol). RKS/UKS reuse the Phase 3 kernel<H> verbatim. DFT-01 bit-exact energy gate is the CI-only --features python rks_energy/uks_energy oracle arms (live convergence needs working arity-3/4 ERIs = the Phase-2 int2e_sph/int3c2e_sph rollup gap, currently NotYetImplemented; minao init guess also not yet implemented) + an always-on structural layer; the RKS/UKS drivers are complete and converge once working ERIs land. From<DftError> for PyscfRsError bridge in pyscf-dft (no pyscf-core dep cycle). pyscf-dft stays pyo3-free + cubecl-free; libxc NEVER compiled.
- [Phase 04]: RSH range-coulomb + VV10 NLC (DFT-05/06) — RSH via the env[8] (PTR_RANGE_OMEGA) mechanism: pyscf-gto::range_coulomb OmegaGuard (RAII set/restore of Mole._env[8], restore-on-drop incl. error/unwind path, T-04-07a) + intor_with_omega + get_k_with_omega drive the STANDARD int2e (NOT phantom int2e_lr_/int2e_sr_ symbols, Pitfall 1). veff::default_get_veff RSH branch (rks.py:108-129): omega!=0 → vk = hyb·K + (alpha−hyb)·K_lr via get_k_with_omega(+omega) on an Arc-backed Mole clone (shared &Mole needs no &mut, omega local + auto-restored). KsVeff gained half_tr_d_vxc so the energy cache is RSH-correct (the old `veff−J+hyb·K` Vxc reconstruction is wrong once vk carries the LR term — Rule-1 bug fix). OPEN QUESTION A5 RESOLVED: cintx safe API (ExecutionOptions/OperatorEnvParams) has f12_zeta (env[9]) + grids_params but NO range_omega (env[8]) setter, AND arity-4 int2e is NotYetImplemented{phase:2} — so the env[8] set/restore contract is owned at the pyscf-gto layer (complete+tested) and the numerical RSH ERI flips on only via a cintx#11-style gap-closure (safe-API env[8] reader + arity-4 int2e). VV10 (DFT-06) ports the pure-Python _vv10nlc double-loop (numint.py:526-538, Pitfall 4: NOT C VXC_vv10nlc) over a coarser nlcgrids (a separate Grids instance): per outer point double-loop over inner vv grid → F/U/W via oracle_sum (T-04-07b), exc/vrho/vsigma per numint.py:552-554; nr_nlc_vxc orchestrates (outer==inner==nlcgrids, excsum=oracle_dot(den,exc), symmetrized GGA Vxc). NlcCoeffs hardcodes only the bare 'VV10' default (5.9/0.0093, A1); per-functional → libxc nlc_coeff. CAM-B3LYP is libxc-only on the corpus (xcfun XC_CODES has no entry; libxc id 433) — the always-on RSH test uses an xcfun-namespace RSH(0.19*HF+0.46*LR_HF(0.33)+0.81*LYP); CAM-B3LYP/VV10 energy gates CI-gated. libxc NEVER compiled.
- [Phase ?]: [Phase 04]: DF-DFT + KsResult chkfile (DFT-07, D-10/D-06 reuse) — RKS::density_fit precomputes pyscf_df B integrals; DfKsHooks routes the Coulomb-J build through get_jk_df ((J_df, K_standard) split, T-04-08b) while Vxc/K stay standard, so get_veff_ks is identical to the non-DF KS path. KsResult wraps ScfResult: on-disk /scf group byte-identical to the SCF schema (upstream from_chk compat) PLUS xc/grids_level/grids_scheme metadata; impl Checkpointable via pyscf_chkfile primitives + the re-exported hdf5 alias (NO own hdf5-metno dep, D-05); load bounded/validated, never panics (T-04-08). ndarray added (F-order view, not hdf5). DFT-07 energy + ORACLE-08 h5py gates CI-only behind the Phase-2 int3c2e_sph gap + libpython/h5py; structural + Rust-Rust round-trip layers always-on. libxc NEVER compiled.

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
- **libxc_rs per-functional feature gate — `PENDING_LIBXC_RS_FEATURE_GATE`** (Phase 04 plan 04-02, user checkpoint 2026-05-22 → *keep pending*). The sibling `~/Documents/workspace/libxc_rs` repo still unconditionally path-deps all 266 `libxc-kernel-*` crates (~6h compile). Deferred as a separate cross-repo workstream (its own PR/issue), mirroring the Phase 2 cintx-ECP coordination (cintx#11). The xcfun-default DFT path (04-04..04-08) is independent and proceeds; the `--features libxc` bit-exact assertions (04-05/04-06/04-09) and the dedicated libxc CI job (04-10) stay `#[cfg(feature="libxc")]`-gated and CI-only until this lands. Never trigger a default `cargo build` on libxc_rs.
- **cintx safe-API range-coulomb env[8] gap (Open Question A5 RESOLVED, plan 04-07)** — cintx *reads* `PTR_RANGE_OMEGA = env[8]` (verified in `cintx-compat::raw`) but its SAFE API (`cintx_runtime::ExecutionOptions` / `OperatorEnvParams`) exposes only `f12_zeta` (env[9]) + `grids_params` — there is NO `range_omega` (env[8]) setter, and arity-4 `int2e` is `NotYetImplemented{phase:2}` (the Phase-2 verification-rollup gap). pyscf-rs owns the env[8] set/restore contract at the pyscf-gto layer (`range_coulomb::OmegaGuard` over `Mole._env[8]`, complete + tested), so the RSH veff branch is correct; the NUMERICAL RSH ERI (and DF JK / bit-exact RKS energy) flips on only once cintx ships a safe-API env[8] reader on the int2e plan AND lands arity-4 int2e — a cintx#11-style cross-repo gap-closure. The CAM-B3LYP/H2O bit-exact energy assertion (DFT-05) is CI-gated behind this + the libxc backend; the VV10 energy match (DFT-06) is CI-gated behind the same Phase-2 ERI/init-guess gap as the 04-06 DFT-01 oracle. The RSH/VV10 code needs no change when these land.

### Quick Tasks Completed

| # | Description | Date | Commit | Directory |
|---|-------------|------|--------|-----------|
| 260512-8jv | Create issue in cintx repository about remaining tasks from pyscf_rs Phase 2 ([cintx#11](https://github.com/BectorVoom/cintx/issues/11)) | 2026-05-11 | 7dcdf08 | [260512-8jv-create-issue-in-cintx-repository-about-r](./quick/260512-8jv-create-issue-in-cintx-repository-about-r/) |
| 260512-8wb | Rewrite cintx#11 as cintx-only Phase 2 task list (drop pyscf_rs framing) | 2026-05-11 | f53cc0e | [260512-8wb-rewrite-cintx-11-as-cintx-only-phase-2-t](./quick/260512-8wb-rewrite-cintx-11-as-cintx-only-phase-2-t/) |
| 260522-b06 | implement f32/f64 precision switching using generics | 2026-05-22 | 4c6ab55 | [260522-b06-implement-f32-f64-precision-switching-us](./quick/260522-b06-implement-f32-f64-precision-switching-us/) |

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

Last session: 2026-05-22T10:49:21.311Z
Stopped at: Completed 04-07-PLAN.md (RSH range-coulomb env[8] + RSH get_veff branch DFT-05; VV10 _vv10nlc double-loop over coarser nlcgrids DFT-06; A5 resolved, cintx#11 env[8]/arity-4 gap-closure tracked)
Resume file: None

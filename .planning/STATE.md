---
gsd_state_version: 1.0
milestone: v1.0
milestone_name: milestone
status: executing
stopped_at: Completed 05-03-PLAN.md
last_updated: "2026-05-23T08:06:51.708Z"
last_activity: 2026-05-23
progress:
  total_phases: 8
  completed_phases: 2
  total_plans: 51
  completed_plans: 45
  percent: 25
---

# Project State

## Project Reference

See: .planning/PROJECT.md (updated 2026-05-09)

**Core value:** Run mainstream molecular ground-state quantum chemistry (HF, DFT, MP2, CCSD, gradients) 2–5× faster than current PySCF + C extensions, with bit-exact agreement on regression tests, and zero C/CMake/libcint dependency hell at install time.
**Current focus:** Phase 05 — mp2

## Current Position

Phase: 05 (mp2) — EXECUTING
Plan: 4 of 7
Status: Ready to execute
Last activity: 2026-05-23

Progress: [██████████] 95% (42/44 plans done across all phases; Phase 04 gap closure: 4/4 gap plans done)

## Performance Metrics

**Velocity:**

- Total plans completed: 34
- Average duration: — (no plans run yet)
- Total execution time: 0 hours

**By Phase:**

| Phase | Plans | Total | Avg/Plan |
|-------|-------|-------|----------|
| 02 | 9 | - | - |
| 03 | 11 | - | - |
| 04 | 14 | - | - |

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
| Phase 04 P04-11 | 5min | 2 tasks | 3 files |
| Phase 04 P04-12 | 6min | 1 task (TDD) | 2 files |
| Phase 04 P04-13 | 11min | 1 task (TDD) | 2 files |
| Phase 04 P04-14 | 13min | 3 tasks | 6 files |
| Phase 05 P01 | 9min | 3 tasks | 24 files |
| Phase 05 P02 | 10min | 2 tasks | 3 files |
| Phase 05 P03 | 14min | 2 tasks | 7 files |

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
- [Phase 04]: Gap closure CR-04 (04-12) — The KS per-cycle energy cache in `hooks.rs` (KsHooks) and `df_dft.rs` (DfKsHooks) keyed `(Exc, 0.5·Tr(D·Vxc))` on `dm_fingerprint = Σ|D|` (L1 norm) with a `(c.fp - fp(dm)).abs() < 1e-12` hit guard. This is NON-INJECTIVE: two distinct density matrices can share an L1 norm, and at µHartree convergence (where the 1µH bit-exact gate operates) a genuine step-to-step Exc change can hide behind an unchanged Σ|D|, returning a STALE XC energy. Replaced with an INJECTIVE `dm_fingerprint(&Density) -> u64` that hashes each element's raw f64 bit pattern (`v.to_bits().hash(&mut h)`) via `std::collections::hash_map::DefaultHasher` (SipHash, stdlib — NO new crate dep, satisfying threat T-04-12-SC's no-install disposition). Both files use the IDENTICAL scheme; the cache-hit guard is now exact `c.dm_fingerprint == dm_fingerprint(dm)` (no float approximation). Hashing the bits (not the value) is deliberate: -0.0 != 0.0 and distinct NaN payloads differ, so the cache reuses Exc only for a byte-identical density. The cache mechanism + its grid-loop-recompute miss-fallback are retained (only the key changed). New `dm_fingerprint_is_injective` test: two dm with Σ|D|=4 but different entries (`[1,-1,1,-1]` vs `[2,-2,0,0]`) produce different u64 keys, and an identical dm is deterministic. `cargo test -p pyscf-dft` exits 0 (43 lib unit + all integration suites green).
- [Phase 04]: Gap closure CR-02 (04-13) — The f32 precision matmul chain in `numint.rs` (`eval_rho_scalar<f32>` `contract` closure + `nr_rks_inner<f32>` Vxc back-contraction) used `S::from(x).unwrap_or_else(S::zero)` and `t.to_f64().unwrap_or(0.0)` throughout, which on f64→f32 overflow silently substituted a wrong value — violating the D-08 "honest f32 path" contract (user should get a loud signal, not silent corruption). CRITICAL FINDING: the plan assumed `S::from(overflow)` returns `None`, but `num-traits 0.2.19` `f32::from(1e40_f64)` returns `Some(f32::INFINITY)` — so the prescribed `.ok_or_else(...)?` alone would have left `Ok([inf])` (still silent). Added `PyscfRsError::NumericOverflow { context: &'static str }` to error.rs and two helpers in numint.rs: `cast_finite<S>` (`ok_or` on the defensive `None` arm PLUS an `x.is_finite() && !s.is_finite()` check that catches a finite f64 narrowing to a non-finite f32 — the REAL overflow mode) and `back_to_f64<S>` (flags a non-finite f32 *accumulation* via `S::KIND != F64 && !t.is_finite()`). Every `unwrap_or_else(S::zero)`/`unwrap_or(0.0)` in the f32 numeric chain replaced with these `?`-propagating helpers; the `contract` closure became `Result<Vec<f64>, PyscfRsError>` with `?` at its 3 call sites. The f64 DEFAULT path is bit-identical: for `S=f64` both helpers are the identity (the `S::KIND==F64` guard skips the finiteness rejection so a legitimately non-finite f64 passes through exactly as the old `unwrap_or(0.0)`-on-f64 — which was simply `t` — did). `to_f64` is reached as a method via the `Scalar: num_traits::Float: ToPrimitive` supertrait (no num-traits dep added to pyscf-dft, honouring T-04-13-SC no-install + libxc-compile avoidance). New `f32_overflow_returns_err_not_zero` test (nao=1, ao=[1e40], dm=[1e40]) asserts `Err(NumericOverflow)` not `Ok(([0.0],None))`; `f64_path_unchanged_no_overflow_on_large_values` proves f64 computes 1e80 cleanly. `cargo test -p pyscf-dft` (45 lib + all integration incl. dtype_f32_smoke, rks_uks_bitexact) + `-p pyscf-core` (11 lib) green; clippy `-D warnings` + fmt clean. No new crate dep.
- [Phase 04]: Gap closure CR-01 (04-14) — `NumInt::nr_uks` was DEAD CODE: it ran `nr_rks` on the TOTAL density (`Dα + Dβ`, the closed-shell path) and returned `vmat: (r.vmat.clone(), r.vmat)` — the SAME Vxc matrix cloned into both spin channels, so it could never produce an open-shell potential. Fixed across 3 files. (1) `xc_backend.rs`: added `UksXcOutput` (per-spin `vrho_a`/`vrho_b` + GGA `vsigma_aa`/`ab`/`bb`) + `XcBackend::eval_uks` + private `xcfun_eval_uks` that builds the per-point xcfun input from the GENUINE `rho_a[ip]`/`rho_b[ip]` (NOT the closed-shell `rho/2` symmetric split), using spin-resolved `Vars::A_B` (LDA) / `A_B_GAA_GAB_GBB` (GGA) — so `vrho_a != vrho_b` for asymmetric densities; MGGA + libxc UKS arms return clean `BackendEval`. (2) `numint.rs`: rewrote `nr_uks` as a genuine open-shell loop — eval the AO block once, contract `rho_a`/`rho_b` (+`∇rho`) INDEPENDENTLY, build `sigma_aa`/`sigma_bb`/`sigma_ab` for GGA, call `eval_uks`, then a new `uks_vmat(grad_this, grad_other)` helper back-contracts TWO DISTINCT Vxc matrices (LDA `0.5·w·vrho·φμφν`; GGA adds same-spin `2·w·vsigma_same·(∇rho_this·∇φμ)·φν` + cross-spin `w·vsigma_ab·(∇rho_other·∇φμ)·φν`, `V+Vᵀ` symmetrized); per-spin nelec + combined excsum via `oracle_sum`. `nr_uks` runs f64-only (xcfun is f64-host; the D-08 f32 matmul chain stays the closed-shell `nr_rks` concern — scope boundary). (3) `hooks.rs` + `uks.rs` + `pyscf-py/dft.rs`: added `UksKsHooks` (open-shell KS hooks; `get_veff` routes through `nr_uks`, combined `Vxc = Vxc_a + Vxc_b`, RSH-aware vk; `UksEnergyCache` keyed on TWO injective fingerprints reusing the CR-04 `dm_fingerprint`); `UKS::kernel` uses `UksKsHooks::new` (NOT `KsHooks::new`); `PyUKS::get_veff` routes through `UksKsHooks::get_veff_ks`→`nr_uks` (NOT `ks_default_get_veff`/`nr_rks`). Symmetric `dm_a=dm_b=dm/2` split is the STRUCTURAL-WIRING contract: the generic `pyscf_scf::kernel<H>` (kernel_impl.rs:59,127) carries a SINGLE total `Density` and calls `get_veff(mol,&dm)` once per cycle, so genuine asymmetric alpha/beta SCF state is out of scope (requires generalizing `kernel<H>`) — the open-shell machinery is complete and yields distinct per-spin Vxc the moment an asymmetric `(dm_a,dm_b)` is fed to `nr_uks` directly. DEVIATION (Rule 3): the GGA input-build loop's `saa/sab/sbb.unwrap()` tripped the crate's `#![warn(clippy::unwrap_used)]` under the `-D warnings` CI gate → rewrote as `if let (Some,Some,Some)`. New `nr_uks_asymmetric_spin_gives_different_vmat` (H2/sto-3g, α in AO0, β in AO1) proves `vmat_alpha != vmat_beta` (RED failed on the clone; GREEN passes); `uks_kernel_uses_nr_uks_not_rks_path` structural test confirms `UksKsHooks` satisfies both `OverrideHooks` + `KsOverrideHooks`. `cargo test -p pyscf-dft -p pyscf-py` exits 0 (47 dft lib + all integration + py tests green); clippy `-D warnings` + fmt clean. No new crate dep; libxc NEVER compiled. This was the FINAL Phase-04 gap-closure plan — all 4 BLOCKERs (CR-01..04) now closed.
- [Phase 04]: Gap closure CR-03 (04-11) — `c2s_coeff` in `pyscf-kernels::eval_gto` was `fn(u32,usize,usize)->f64` with an unconditional `panic!` on l>4 (h-shells: cc-pV5Z, ANO). Through the PyO3 panic→exception bridge this still aborts the Python process (FOUND-07 never-panic violation). Converted to `-> Result<f64, PyscfRsError>`: l<=4 arms wrapped in `Ok(...)` (FROZEN libcint coeffs byte-unchanged), l>4 wildcard returns `Err(NotYetImplemented{phase:4})`. `?`-propagated through `eval_gto_sph_cpu`/`eval_gto_sph_deriv1_cpu` and the public `eval_gto_sph`/`eval_gto_sph_deriv1` (now `Result`-returning). `pyscf_gto::eval_gto` was ALREADY `Result`-returning, so its public signature is unchanged — `numint.rs` `eval_gto_block` and every other downstream consumer compile untouched; only the two internal `?` additions and 3 integration-test `.expect(...)` were needed. No new dependency; libxc never compiled.
- [Phase ?]: [Phase 04]: DF-DFT + KsResult chkfile (DFT-07, D-10/D-06 reuse) — RKS::density_fit precomputes pyscf_df B integrals; DfKsHooks routes the Coulomb-J build through get_jk_df ((J_df, K_standard) split, T-04-08b) while Vxc/K stay standard, so get_veff_ks is identical to the non-DF KS path. KsResult wraps ScfResult: on-disk /scf group byte-identical to the SCF schema (upstream from_chk compat) PLUS xc/grids_level/grids_scheme metadata; impl Checkpointable via pyscf_chkfile primitives + the re-exported hdf5 alias (NO own hdf5-metno dep, D-05); load bounded/validated, never panics (T-04-08). ndarray added (F-order view, not hdf5). DFT-07 energy + ORACLE-08 h5py gates CI-only behind the Phase-2 int3c2e_sph gap + libpython/h5py; structural + Rust-Rust round-trip layers always-on. libxc NEVER compiled.
- [Phase 05]: 05-01 scaffold — `pyscf-ao2mo` registered as the 20th `pyscf-*` member (D-01) with `general`/`full` stub surface + `Ao2moError` bridging to `PyscfRsError`. `pyscf-mp2` deps wired (ao2mo/scf/df/gto/algebra/runtime) strictly pyo3-free + cubecl-free (`xtask check-dependency-wall` PASS), 9-module skeleton + `Mp2Error` bridge. The five MP2-08 helper signatures (`get_nocc`/`get_nmo`/`get_frozen_mask`/`get_e_hf`/`mo_without_core` — the verbatim `cc/ccsd.py:35` CCSD import contract; Python `_mo_without_core`→Rust `mo_without_core` via `#[doc(alias)]`) exported; the always-on `ccsd_import_contract` symbol-existence arm passes. Five MP2 numeric oracle arms registered in `KNOWN_METHODS` (len 13→18: `mp2_rmp2_energy`/`mp2_ump2_energy`/`dfmp2_energy`/`dfmp2_native_energy`/`mp2_rdm`), len-assert updated. CI: always-on `mp2-structural` job + `if: false` cintx#11-gated `mp2-oracle-cintx-gated` numeric job (needs arity-4 `int2e` for in-core + arity-3 `int3c2e_sph` for DF; mirrors DF-HF/DFT-01 gating). MP2 python `dispatch` match arms + all numeric/kernel bodies deferred to 05-02..05-06 (catch-all `UnknownMethod` arm covers the names until then; gated job never runs). Pure scaffolding — ships NO compute.
- [Phase ?]: [Phase 05]: 05-02 AO→MO transform — transform::quarter_transform implements the (pq|rs)→(iq|rs)→(ij|rs)→(ij|ks)→(ij|kl) quarter-transform as host loops (gemm is NotYetImplemented{phase:2}); every per-index sum materializes products into a reused Vec then oracle_sum (4 call sites, 0 bare += in the contraction) → bit-exact + thread-count invariant (T-05-02-FP). general(eri_ao,nao,[&MOCoefficients;4]) ports the eri_ao.size==nao**4 einsum branch of ao2mo/incore.py:125-128 (real-only: .conj() no-op); full = general(..,[mo_coeff;4]). F-order flat-index doc-commented at every boundary (Pitfall 3). T-05-02-SHAPE: validated at entry → ShapeMismatch, never OOB/panic. The 05-01 stub signatures (&[&[f64]]/&[f64]) were replaced (no external callers). Always-on synthetic-ERI roundtrip (the ONE un-gated numeric assertion this phase) asserts general/full/identity bit-exact vs an independent staged longhand reference. check-no-fma + check-dependency-wall PASS.
- [Phase ?]: [Phase 05]: 05-03 in-core RMP2 — rmp2_kernel ports mp2.py:47-76 closed-form (t2=(ia|jb)/(εi+εj−εa−εb), edi=2·oracle_dot(gi,t2i), exi=−oracle_dot((ib|ja),t2i)); EVERY reduction via oracle_dot/oracle_sum (no += accumulator) → bit-exact + thread-count invariant (T-05-03-FP); verified vs synthetic ChemistsEris (1×1 e_corr=−0.125, 1×2 longhand). default_ao2mo builds frozen-aware co/cv subsets + ao2mo::general([co,cv,co,cv]) with eri_ao=intor(int2e); int2e NotYetImplemented{phase:2} propagates with ? (never panics/zeros, T-05-03-FFI), numeric flips on at cintx#11 (D-05). Five MP2-08 helpers real bodies over (mo_occ,&Frozen) = always-on CCSD import contract. Frozen enum None/Count/List/Auto/Window; chemcore element→ORBITAL OnceLock table VERBATIM from elements.py:1079 (119 entries bit-identical) summed DIRECTLY no ÷2 (PLAN said /2 but upstream chemcore() returns sum; O→1/Si→5). scs_energy 1.0/1.0=plain, 1/3,1.2=SCS. Mp2OverrideHooks+ChemistsEris+NoMp2Overrides pyo3-free (D-08); energy default→default_energy, rdm1/rdm2→NotYetImplemented{plan:4} (05-04). check-no-fma + dependency-wall PASS.

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

Last session: 2026-05-23T08:06:51.705Z
Stopped at: Completed 05-03-PLAN.md
Resume file: None

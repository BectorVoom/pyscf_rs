# Phase 4: DFT - Context

**Gathered:** 2026-05-22
**Updated:** 2026-05-22 — added D-08 (f32/f64 precision switch via `PYSCF_DTYPE`)
**Status:** Ready for re-planning (D-08 added — existing 10 plans predate it and need a replan)

<domain>
## Phase Boundary

A user runs `dft.RKS(mol, xc='b3lyp').run()` (and `dft.UKS`) on the test corpus and gets the same total energy as upstream PySCF bit-exact under `release-oracle` (≤1 µHartree on every fixture). Phase 4 builds RKS/UKS Kohn-Sham SCF on top of the Phase 3 SCF engine (RHF/UHF kernel + C-DIIS + the `OverrideHooks` PyO3 contract), adding: the XC-string parser, Becke molecular grids ported byte-for-byte, XC functional evaluation routed through the sibling `libxc_rs` + `xcfun_rs` crates, a `numint.NumInt` surface, range-separated hybrids, VV10 non-local correlation, and DF-DFT. It is the project's largest Python-override surface — DFT-08 re-validates the Phase 3 PyO3 subclass-override contract on the bigger DFT hook set.

**In scope (11 REQ-IDs):**
- **DFT-01..02:** `dft.RKS`/`dft.UKS` bit-exact to upstream on the corpus; XC-string parser handling all upstream forms — single name (`'b3lyp'`), comma form (`'pbe,pbe'`), shorthands (`'lda'`→`'lda,vwn'`), explicit weights (`'.5*HF + .5*B88,LYP'`), `XC_ALIAS` aliases; parser-parity unit test vs `pyscf/dft/libxc.py`.
- **DFT-03:** libxc functional eval routes through `libxc_rs`; xcfun routes through `xcfun_rs`; both bit-identical to the upstream C libraries.
- **DFT-04, DFT-09:** `Grids` class (`level`, `atom_grid`, `prune`, `radi_method`, `becke_scheme`, `atomic_radii`) producing grid points + weights byte-for-byte vs `pyscf/dft/gen_grid.py` for `level ∈ {0..9}` (Pitfall 10).
- **DFT-05:** Range-separated hybrids (`omega`, `alpha`, `beta`) via cintx `int2e_lr_*`/`int2e_sr_*`; CAM-B3LYP/H2O parity fixture.
- **DFT-06:** VV10 non-local correlation (`mf.nlc='VV10'`, `mf.nlcgrids`).
- **DFT-07:** DF-DFT (`dft.RKS(mol).density_fit()`).
- **DFT-08:** All SCF subclass-override hooks re-validate at DFT scope (DFT adds `get_veff`, `define_xc_`).
- **DFT-10:** `numint.NumInt` exposes `eval_xc`, `eval_rho`, `nr_rks`, `nr_uks` matching upstream signatures.
- **DFT-11:** cubecl WGPU backend gated on the `shader-f64` Vulkan extension; runtime falls back to CPU with a warning when unavailable (Pitfall 3).

**Out of scope:**
- MP2 (Phase 5), CCSD (Phase 6), gradients + geomopt (Phase 7), GPU per-backend regression suite + 2–5× benchmark proof (Phase 8).
- DFT-D3/D4 dispersion, custom-XC *Python-callable* user functions (deferred — see Deferred Ideas), DKS/GKS/r-numint relativistic DFT (out of milestone), DFT+U (`rkspu`/`ukspu`), SAP (`sap.py`), symmetry-adapted KS (`rks_symm`/`uks_symm`).
- The Python-callable `define_xc_(ni, eval_xc_fn, ...)` form (per-block GIL callback inside the grid loop) — string-recombination form only in v1.
- Fused cubecl DFT-grid kernels (rho-eval/Vxc-assembly) — deferred to Phase 8 if profiling demands.

</domain>

<decisions>
## Implementation Decisions

### XC backend routing (DFT-02, DFT-03, DFT-10)

- **D-01: Mirror upstream's two-parser design.** Port `pyscf/dft/libxc.py:parse_xc` (+ `XC_CODES`, `XC_ALIAS`) as the **default** name→spec resolver routing to `libxc_rs`, AND `pyscf/dft/xcfun.py:parse_xc` as the **alternate** resolver routing to `xcfun_rs`. The user swaps backends via `mf._numint.libxc = dft.xcfun` exactly like upstream. This is the maximum-fidelity path and the only one that satisfies the bit-exact contract: upstream `dft.RKS(mol, xc='b3lyp')` evaluates via libxc, so to match its numbers b3lyp MUST route to `libxc_rs` (not `xcfun_rs`, which would match xcfun's different numbers). DFT-03's wording ("libxc routes through libxc_rs; xcfun routes through xcfun_rs") encodes exactly this split. Matches the project-wide sibling-crate-fidelity hard preference.

- **D-02: `define_xc_` ships the string-recombination form only in v1.** Support `define_xc_(ni, '0.2*HF + 0.08*LDA + 0.72*B88, 0.81*LYP + 0.19*VWN', 'GGA', hyb=0.2)` — it is just another XC string fed to the D-01 parser. The user-supplied Python-callable `eval_xc(xc_code, rho, ...)` form is stubbed with `NotYetImplemented{deferred}`. Rationale: the callable form forces a GIL-attached Python callback inside the hot grid loop, which breaks the Phase 3 `Python::detach` seam (D-03) for custom functionals and adds python3.13t deadlock surface. The string form fully exercises DFT-08 override re-validation (via `get_veff` + `define_xc_` string) without that cost. The per-block Python-callback architecture lands in a later phase if demanded.

### libxc compile-cost containment (DFT-03, cross-cutting build-time constraint)

- **D-03: Off-by-default `libxc` cargo feature.** The default `cargo build --workspace` excludes `libxc_rs` from the dep graph entirely. `pyscf-dft` compiles with an `XcBackend` seam; the libxc-backed path returns a clear "libxc feature not enabled" runtime error when the feature is off, and `xcfun_rs` is the default-compiled XC backend. Building/testing the libxc-backed bit-exact path requires `--features libxc` (the DFT bit-exact CI job + maturin wheel builds enable it). This mirrors Phase 1's `gpu` umbrella feature being OFF by default (CPU is the default backend) and the existing commented-out `libxc_rs` `[patch.crates-io]` entry (`Cargo.toml:94`) + the `nightly-cross-crate.yml:40` exclusion — both noted as "re-enable when Phase 4 needs it." Phase 4 re-enables the patch **behind the feature**. **Hard constraint (user memory):** never trigger a default-build compile that pulls libxc_rs's 266 functional-kernel crates (~6h). See `feedback_libxc_compile_time.md`, `feedback_no_compile_freeze.md`.

- **D-04: Add per-functional cargo features to `libxc_rs` (cross-crate coordination task).** `libxc_rs`'s main crate currently path-depends on **all 266** `libxc-kernel-*` crates **unconditionally** — there is no `[features]` block and no `cfg`-gated dispatch (verified by reading `~/Documents/workspace/libxc_rs/Cargo.toml` + `src/eval/`, `src/kernel/`). So even with D-03's gate, the moment `--features libxc` is on, it is the full ~6h/266-crate compile. Phase 4 therefore includes a **cross-crate coordination task on `libxc_rs`**: add a `[features]` block + `cfg`-gated kernel dispatch so consumers can enable only the functionals they need. `pyscf-rs`'s `libxc` feature then pulls just the corpus subset (~`lda_x`/`lda_c_vwn` (SVWN), `gga_x_pbe`/`gga_c_pbe` (PBE), `gga_x_b88`/`gga_c_lyp` + `hyb_gga_xc_b3lyp` (B3LYP), `hyb_gga_xc_cam_b3lyp` (CAM-B3LYP), a meta-GGA e.g. TPSS/SCAN, + VV10's `gga` deps), keeping even the gated build to minutes. This is a **sibling-crate gap-closure dependency** structurally identical to the Phase 2 cintx-ECP coordination (Phase 2 D-06): planner should include an explicit coordination/gap-closure plan with a status marker, and the bit-exact-libxc path may be sequenced behind it landing. The exact corpus-functional list is the planner's to finalize from the test corpus + ROADMAP success criteria.

### Grid generation (DFT-04, DFT-09, Pitfall 10)

- **D-05: New `pyscf-grids` workspace crate (18 → 19).** Grid generation gets its own workspace member rather than living inside `pyscf-dft`. Depends on `pyscf-core` + `pyscf-algebra` (Becke partition weights need ordered reductions under `release-oracle`). Consistent with the Phase 3 pattern of splitting cross-method-shared concerns into their own crates (`pyscf-diis`/`pyscf-df`/`pyscf-chkfile`) rather than upstream's exact file placement. Reused by VV10's separate `nlcgrids` (DFT-06) and Phase 7 DFT-gradient grid response. Algebra wall applies (depends on `pyscf-algebra`, never `cubecl-*` directly); ROADMAP.md needs an explicit 18→19 member update during planning. **Note (sibling-fidelity counter):** upstream co-locates grids in `pyscf/dft/gen_grid.py` + `radi.py`; the own-crate split is an intentional deviation justified by cross-phase reuse and the established Phase 3 split-out pattern.

- **D-06: Port the Lebedev generator + radial formulas to Rust (not a snapshot, not runtime file reads).** `pyscf/dft/LebedevGrid.py` is an **algorithmic generator** (`gen_oh` + `MakeAngularGrid_*` constructing points from octahedral-symmetry seeds per order), not a static table; `pyscf/dft/radi.py` is **formula-based** (Becke / Treutler / Gauss-Chebyshev radial schemes computed from small element-radius tables in `pyscf/data/radii.py`). Port the generator + radial formulas to Rust with small const tables for the Lebedev seeds + element radii (Bragg/covalent/SG1). Byte-for-byte parity (DFT-04/09) comes from deterministic construction under `release-oracle`'s FMA-free + ordered-reduction infrastructure (Phase 1) — Pitfall 10 is owned here. No heavy `build.rs`, no parse-N-files macro, no runtime file reads, no large bundled data asset. Matches Phase 2's "port the reference algorithm" approach (`format_atom`/`make_env`) and the "don't freeze compile" constraint. Snapshot-with-CI-drift-check is the documented fallback **only if** the generator port proves intractable to make bit-exact.

### NumInt hot path (DFT-10)

- **D-07: Orchestrate the `nr_rks`/`nr_uks` grid loop in `pyscf-dft` via `pyscf-algebra`; no new bespoke DFT cubecl kernels in v1.** The grid loop (eval AO on grid → ρ = AO·D → eval XC → Vxc = AOᵀ·(w·v·AO) → accumulate Exc) reuses the existing `eval_gto` cubecl kernel (`pyscf-kernels`, Phase 2 D-04 — the deferred `l≥1` eval_gto variants land in this phase) for AO-on-grid, does the ρ-contraction and Vxc back-contraction as `gemm`/weighted-`gemm` through `pyscf-algebra`, and routes per-block XC evaluation to `libxc_rs`/`xcfun_rs`. No fused DFT grid kernels are written now; any fused cubecl XC-grid kernel is deferred to Phase 8 if profiling shows the matmul chain is the bottleneck. Mirrors the Phase 3 `pyscf-df` precedent (D-10: "no new kernel — just Cholesky + matmul via algebra"), respects the algebra wall, and avoids front-loading the optimization the 2–5× target (Phase 8) is meant to own. Note: `libxc_rs` is CPU-host-only (cubecl-cpu), so a future GPU grid path would force a host round-trip per block for libxc XC eval; `xcfun_rs` has on-device `eval_vec` — relevant for Phase 8, not v1.

### Precision switching: f32/f64 via `PYSCF_DTYPE` (DFT-10, DFT-11, cross-cutting)

- **D-08: The DFT compute path honors the existing `PYSCF_DTYPE` precision seam at runtime; f64 is the bit-exact default, f32 is an opt-in fast/low-memory mode.** Phase 4 threads the precision seam that **already exists** — Phase 1 D-08's `PYSCF_DTYPE`→`DType{F32,F64}` resolver (`pyscf-runtime/src/backend.rs:DType::from_env`, default `F64`, case-insensitive) + quick-task-260522-b06's generic `Tensor<T: Scalar = f64>` / `DeviceScalar` (impl `f32`/`f64`) seam (`pyscf-core/src/scalar.rs`, `pyscf-algebra/src/scalar.rs`, `pyscf-kernels/src/scalar.rs`) — through the DFT NumInt grid loop + RKS/UKS so the active scalar type is selected at runtime from `PYSCF_DTYPE`. **No new env var** (reuse `DType::from_env()`), **no new dep** (`pyscf-runtime`/`pyscf-algebra` already in the `pyscf-dft` dep set). Precision is selected through `pyscf-algebra`/`pyscf-runtime`, never hand-rolled — same algebra-wall discipline as backend selection.

  - **f64 is the default and the ONLY bit-exact/oracle-validated path.** DFT-01 (≤1 µHartree), DFT-04/09 (byte-exact grid weights, Pitfall 10), the parser-parity test, and the `--features libxc` bit-exact CI job all run **f64** under `release-oracle`. f32 is **never** compared to the upstream oracle for energy parity and explicitly does NOT meet the µHartree bar.

  - **f32 is an opt-in fast / lower-memory mode and the f64-less-WGPU escape hatch.** `PYSCF_DTYPE=f32` switches the DFT path to f32. This is the honest fallback for adapters lacking the `shader-f64` Vulkan extension — Phase 1 D-09 hard-errors on `wgpu`+`f64` without `shader-f64`, and DFT-11 owns the WGPU f64 hole; f32 lets DFT run there. Emit a `tracing::warn!` at DFT kernel entry when f32 is active, noting it is below the bit-exact bar. **No numeric-parity tolerance is committed for f32 in v1** (user declined a loose-tolerance f32 regression): f32 ships with a runs-end-to-end **smoke test only**, mirroring quick-task-260522-b06's single narrow f32 smoke test. A loose (~chemical-accuracy) f32 sanity bound is a Deferred Idea, not a v1 gate.

  - **User-facing switch = env var + read-only accessor.** The user switches precision via `PYSCF_DTYPE` (the single source of truth). Phase 4 additionally exposes a **read-only accessor** to query the resolved precision: a Rust method on the `NumInt` / KS object returning the active `DType`, surfaced through the Python binding (readable on `mf` / `mf._numint`). **No per-object runtime override** (`set_precision(...)`) in v1 — user declined that surface; deferred. Continue the Phase 3 ALG-08 convention of logging `backend=<…> dtype=<…>` at DFT kernel entry.

### Claude's Discretion

The following are not user-decided — researcher/planner picks the implementation within the locked decisions above:

- **Grid blksize / chunking** — mirror upstream `numint.py` MEM-driven blksize chunking (`_dot_ao_dm`, `_scale_ao`, block-at-a-time accumulation). Log `PYSCF_MAX_MEMORY` at kernel entry (Phase 3 SCF convention); **no hard pre-flight enforcement** in v1 — Phase 6 CCSD-11 owns the budget-aware tensor-arena/refusal.
- **libxc family dispatch (LDA/GGA/MGGA)** — query `libxc_rs` `FunctionalMeta.family()`/`.kind()` to size the ρ-derivative evaluation (ρ only / +σ / +τ,∇²ρ) and pick `LdaInput`/`GgaInput`/`MggaInput` + the `DerivativeOrder`. For `xcfun_rs`, use `Functional::is_gga()`/`is_metagga()` + `eval_setup(Vars, Mode, order)` to size `input_length`/`output_length`.
- **Hybrid / RSH coefficient surface** — `NumInt` exposes `hybrid_coeff`/`rsh_coeff(xc_code, spin) -> (omega, alpha, hyb)` mirroring upstream `numint.py`. KS `get_veff` queries it and reuses the Phase 3 `get_jk` path for the exact-exchange K contribution; RSH long/short-range K uses cintx `int2e_lr_*`/`int2e_sr_*` via `mol.intor` (DFT-05).
- **DFT-11 WGPU f64 honesty** — delegate the XC-eval-portion fallback to `xcfun_rs`'s existing `shader-f64`/ERF `auto_backend` fallback (its GPU-05: wgpu/metal + ERF deps silently override to CPU because range-separation needs f64). The pipeline-level wgpu→CPU fallback reuses Phase 1's `PYSCF_BACKEND` resolver + `tracing::warn!`. `libxc_rs` is CPU-host-only so its path is CPU by construction. Researcher confirms exact probe location + the CI job on a `shader-f64`-less device.
- **VV10 NLC (DFT-06)** — `mf.nlc='VV10'` + a separate `mf.nlcgrids` (a second `pyscf-grids` instance, typically coarser); port upstream `nr_rks_vv10`/`_vv10nlc` from `numint.py`. Decide whether VV10 ships in the core RKS plan or a follow-on plan.
- **DF-DFT (DFT-07)** — reuse `pyscf-df::DfIntegrals` (Phase 3 D-10) for the Coulomb-J build; no new DF crate. `dft.RKS(mol).density_fit()` mirrors the Phase 3 `mf.density_fit()` shape.
- **`KsResult` chkfile** — `impl Checkpointable for KsResult` in a `pyscf-dft::chkfile` module via `pyscf-chkfile` HDF5 primitives (Phase 3 D-06 pattern); add `xc`/`grids` metadata to the schema.
- **`KsOverrideHooks` trait** — extend the Phase 3 `OverrideHooks` trait shape with the DFT hooks (`get_veff`, `define_xc_`, and the inherited SCF hooks); `pyscf-dft` stays pyo3-free, `pyscf-py` provides the `PyOverrideBridge` impl (Phase 3 D-01). `to_uks`/`to_rks` (Phase 3 stubs returning `NotYetImplemented{phase:4}`) get wired to real KS targets.
- **Phase MVP sequencing** — core path (RKS/UKS + grids + XC parser + libxc eval, bit-exact on SVWN/PBE/B3LYP) first; range-separated (DFT-05), VV10 (DFT-06), DF-DFT (DFT-07) as follow-on plans. Planner finalizes wave structure + the corpus-functional list driving D-04's libxc subset.
- **f32 grid-loop boundary (D-08 implementation)** — `libxc_rs` (f64 host-only) and `xcfun_rs` evaluate XC in f64, so an f32 DFT path is f32 only for the AO-eval / ρ-contraction / Vxc back-contraction matmul chain (the `pyscf-algebra` `gemm` path); ρ is cast to f64 at the XC-eval boundary and v cast back. Whether to monomorphize both `Tensor<f32>`/`Tensor<f64>` paths and dispatch at NumInt entry (mirroring the `AlgebraClient` enum-match), and the exact cast boundary, is researcher/planner's to finalize from what the sibling XC crates accept. The default-f64 path is unaffected.

</decisions>

<canonical_refs>
## Canonical References

**Downstream agents MUST read these before planning or implementing.**

### Project specs (this repo)
- `.planning/PROJECT.md` — vision, core value, key decisions, "out of scope" list; DFT is the first Active requirement entering implementation
- `.planning/REQUIREMENTS.md` — Phase 4 owns DFT-01..11 (11 REQs); see the DFT-01..11 bullets + the Phase-distribution counting note
- `.planning/ROADMAP.md` §"Phase 4: DFT" — goal, dependencies (Phase 3), 5 numbered success criteria
- `.planning/ROADMAP.md` §"Cross-Cutting Concerns Threaded Through Every Phase" — algebra-responsibility wall, backend selection, bit-exact-with-PySCF, PyO3 subclass-override dispatch, Python::detach seam, scope-creep lint, cubecl pin lockstep
- `.planning/ROADMAP.md` §"Pitfall-to-Phase Mapping" — Phase 4 owns Pitfall 10 (grid-weight bit-exactness) and re-validates Pitfall 3 (WGPU f64 holes / cubecl pin) + Pitfall 7 (subclass override, larger surface)
- `.planning/STATE.md` §"Blockers/Concerns" — WGPU f64 holes (cubecl #1316/#1317) verified in Phase 4 (DFT-11); CCSD(T) deferral pressure (unrelated); libxc_rs re-enable note
- `.planning/phases/03-scf-pyo3-bindings/03-CONTEXT.md` — D-01..11 carried forward: `OverrideHooks` trait-callback bridge, per-hook `Python::detach`, type-specific NumPy converters, `pyscf-chkfile`/`pyscf-diis`/`pyscf-df` crates, in-memory DF (HDF5 spill deferred to Phase 6), `canonicalize_signs`, test-corpus tiering
- `.planning/phases/02-gto/02-CONTEXT.md` — D-01..07 carried forward: live basis-file reads + "don't freeze compile", `eval_gto` kernel home (`pyscf-kernels`, GTOval_sph/_sph_deriv1 are the DFT hot paths), F-order convention per `pyscf/gto/moleintor.py`, "port the reference algorithm" precedent
- `.planning/phases/01-foundation/01-CONTEXT.md` — workspace layout, `AlgebraClient` enum + match dispatch, host-faer eigh, `PYSCF_BACKEND`/`PYSCF_DTYPE` resolver, `gpu` umbrella feature OFF by default, sibling-crate sourcing under `BectorVoom`, cubecl `=0.10.0` pin + the four-crate ABI lockstep + `docs/upgrade-cubecl.md`

### Upstream PySCF source (this repo — the oracle / port reference)
- `pyscf/dft/rks.py` (569 lines) — `RKS` class, `get_veff` (J − HF·K + Vxc), `energy_elec`, the KS SCF driver layered on `scf/hf.py`
- `pyscf/dft/uks.py` (209 lines) — `UKS` open-shell KS
- `pyscf/dft/gen_grid.py` (704 lines) — `Grids` class, Becke partitioning, `gen_atomic_grids`, pruning schemes (`nwchem_prune`/`sg1_prune`/`treutler_prune`), `level` → (radial, angular) size mapping; **byte-for-byte port target for DFT-04/09**
- `pyscf/dft/radi.py` (199 lines) — radial schemes (`becke`, `treutler`, `gauss_chebyshev`, `delley`, `mura_knowles`), `treutler_atomic_radii_adjust`/`becke_atomic_radii_adjust`, `BRAGG_RADII`/`COVALENT_RADII`/`SG1RADII`; `ATOM_SPECIFIC_TREUTLER_GRIDS` flag (affects ~1e-6/atom — match the upstream default `True`)
- `pyscf/dft/LebedevGrid.py` (5047 lines) — Lebedev angular-grid **generator** (`gen_oh`, `MakeAngularGrid_*`, `LEBEDEV_ORDER`/`LEBEDEV_NGRID` tables); port the generator + seeds, not a snapshot
- `pyscf/data/radii.py` — `BRAGG`, `COVALENT` element-radius tables consumed by `radi.py` (small const tables to port)
- `pyscf/dft/libxc.py` (1527 lines) — `parse_xc`, `XC_CODES`, `XC_ALIAS`, `hybrid_coeff`, `rsh_coeff`, `is_hybrid_xc`, `is_meta_gga`, `eval_xc`; **default parser source-of-truth (D-01), parser-parity test target (DFT-02)**
- `pyscf/dft/xcfun.py` (1166 lines) — xcfun-side `parse_xc` + code tables; **alternate parser (D-01) routing to `xcfun_rs`**
- `pyscf/dft/numint.py` (3030 lines) — `NumInt` class: `eval_ao`, `eval_rho`, `eval_xc`/`eval_xc_eff`, `nr_rks`, `nr_uks`, `nr_vxc`, `_dot_ao_dm`, `_scale_ao`, `nr_rks_vv10` (VV10), block-loop + blksize logic; **DFT-10 signature source-of-truth + D-07 grid-loop reference**
- `pyscf/dft/dft_parser.py` (24 lines) — small XC-string preprocessing shared by both parsers
- `pyscf/dft/__init__.py` — `RKS`/`UKS` factories + `dft.libxc`/`dft.xcfun` module handles (the `mf._numint.libxc = dft.xcfun` swap surface, D-01)
- `pyscf/dft/xc_deriv.py` (653 lines) — XC derivative transforms between libxc/xcfun conventions and PySCF's `vxc`/`fxc` layout (relevant if derivative-layout adaptation is needed)

### Sibling XC crates (sibling repos — the evaluation engines; READ-ONLY, never `cargo build` these)
- `~/Documents/workspace/libxc_rs/Cargo.toml` — **confirms the 266-kernel unconditional path-dep + no `[features]` block** (D-04 driver); root is both `[package]` and `[workspace]`; cubecl 0.10.0 cpu-only
- `~/Documents/workspace/libxc_rs/src/registry/mod.rs` — `lookup_by_id(u16) -> &FunctionalMeta`, `lookup_by_name(&str) -> FunctionalId`, `all_functional_ids()`, `functional_count()` (649)
- `~/Documents/workspace/libxc_rs/src/api/builder.rs` + `src/api/batch.rs` — `FunctionalBuilder`, `Functional`, `BatchEvaluator::new(spin, np_max)` + `evaluate(&Functional, &Input, DerivativeOrder, &mut Output)`
- `~/Documents/workspace/libxc_rs/src/input/mod.rs` + `src/output/mod.rs` — typed `LdaInput`/`GgaInput`/`MggaInput`, `LdaOutput`/`GgaOutput`/`MggaOutput` (per-order fields: `zk`, `vrho`, `vsigma`, `vtau`, `v2*`…); `DerivativeOrder` = Exc/Vxc/Fxc/Kxc/Lxc
- `~/Documents/workspace/libxc_rs/src/eval/` + `src/kernel/` — family dispatchers `dispatch_lda`/`dispatch_gga`/`dispatch_mgga`; **no `cfg(feature)` gating today** (D-04 adds it)
- `~/Documents/workspace/xcfun_rs/crates/xcfun-rs/src/functional.rs` — `Functional::new`/`set(name,weight)`/`get`/`is_gga`/`is_metagga`/`eval_setup(Vars,Mode,order)`/`input_length`/`output_length`/`eval(&[f64],&mut[f64])`/`eval_vec(density,pitch,out,out_pitch,nr_points)` (GPU dispatch ≥64 pts, ERF auto-fallback to CPU)
- `~/Documents/workspace/xcfun_rs/crates/xcfun-core/src/{registry.rs,enums.rs}` — `FUNCTIONAL_DESCRIPTORS` (79), `FunctionalId::from_name`, `Vars`/`Mode` enums
- `~/Documents/workspace/xcfun_rs/crates/xcfun-gpu/src/lib.rs` — `Backend` enum + `auto_backend()` priority chain + `shader-f64`/ERF fallback (DFT-11 delegation target)
- `~/Documents/workspace/xcfun_rs/crates/xcfun-py/src/lib.rs` — PyO3 0.28 + rust-numpy 0.28 binding precedent (`Functional` pyclass, `Mode`/`Vars` IntEnum, exception class) — analog for any pyscf-py XC surface
- Cargo package names: `libxc_rs` (single consumer crate; per-functional features added by D-04) and `xcfun-rs` (the facade; `xcfun-core`/`-eval`/`-kernels`/`-gpu` are internal). Both are path/`[patch.crates-io]` deps under `BectorVoom`.

### pyscf-rs codebase (this repo — Phase 1-3 shipped artifacts)
- `crates/pyscf-dft/src/lib.rs` — Phase 1 stub (empty, `#![forbid(unsafe_code)]`); Phase 4 fills with `RKS`/`UKS` + `NumInt` + `KsOverrideHooks`
- `crates/pyscf-scf/src/lib.rs` — Phase 3 `RHF`/`UHF`/`GHF` + `OverrideHooks` trait + generic `kernel<H>` + C-DIIS integration; KS reuses the SCF cycle, swapping `get_veff`
- `crates/pyscf-gto/src/lib.rs` — `mol.intor(name)` dispatcher (Phase 2 D-05) for `int2e_lr_*`/`int2e_sr_*` (DFT-05) + `eval_gto` user wrapper; Phase 4 exercises `GTOval_sph`/`GTOval_sph_deriv1`
- `crates/pyscf-kernels/` — `eval_gto` cubecl kernel (Phase 2 D-04); Phase 4 lands deferred `l≥1` variants; grid loop launches via `pyscf-algebra`
- `crates/pyscf-algebra/src/lib.rs` — `gemm`/`gemv`/`axpy`/`dot`/`reduce_sum`/`oracle_sum`/`oracle_dot`/`eigh` — the entire DFT compute path (D-07) + Becke-weight reductions (D-05/`pyscf-grids`) go through this
- `crates/pyscf-df/src/lib.rs` — `DfIntegrals`/`get_jk_df`/`density_fit` (Phase 3 D-10); DFT-07 reuses for the J build
- `crates/pyscf-chkfile/src/lib.rs` — HDF5 primitives + `Checkpointable` trait (Phase 3 D-06); `KsResult` impls it
- `crates/pyscf-py/src/lib.rs` — `#[pymodule] _native` + `PyOverrideBridge` (Phase 3 D-01/D-07); Phase 4 adds `PyRKS`/`PyUKS` + the `dft` submodule + `python/pyscf/dft/__init__.py` overlay
- `crates/pyscf-core/src/{traits.rs,mo.rs,density.rs}` — `KohnSham` trait declared (Phase 1); `MOCoefficients`/`Density` reused; `canonicalize_signs` (SCF-13) called post-eigh
- `Cargo.toml` (workspace) — Phase 4 adds member `crates/pyscf-grids` (18→19), re-enables `libxc_rs` `[patch.crates-io]` **behind the `libxc` feature** (D-03), wires the `libxc` umbrella feature
- `xtask/src/lints/algebra_wall.rs` — extend allowlist for `pyscf-grids` (algebra dep) + `pyscf-dft` (algebra dep, no direct cubecl)
- `.github/workflows/ci.yml` + `nightly-cross-crate.yml` — add the `--features libxc` DFT bit-exact job + the `shader-f64`-less WGPU fallback job (DFT-11); re-enable `libxc_rs` in the nightly cross-crate matrix

### Precision seam (this repo — already shipped; D-08 threads it through DFT)
- `crates/pyscf-runtime/src/backend.rs` — `DType{F32,F64}` + `DType::from_env()` (Phase 1 D-08; default `F64`, case-insensitive `PYSCF_DTYPE`); **the resolver the DFT path reads (D-08)**
- `crates/pyscf-algebra/src/select.rs` — `PYSCF_BACKEND`+`PYSCF_DTYPE`→`AlgebraClient` resolver + the D-09 `wgpu`+`f64`-without-`shader-f64` hard-error (the f32 escape-hatch driver for DFT-11)
- `crates/pyscf-algebra/src/scalar.rs` + `crates/pyscf-core/src/scalar.rs` + `crates/pyscf-kernels/src/scalar.rs` — `Scalar` (host) / `DeviceScalar` (`f32`/`f64`) trait seam + `Tensor<T: Scalar = f64>` (quick-task-260522-b06); **the generic seam the DFT compute path threads (D-08)**
- `docs/env-vars.md` — `PYSCF_DTYPE` (values `f32`/`f64`, default `f64`, resolution priority); **update with the DFT read-only accessor + the f32-below-bit-exact `warn!` note (D-08)**
- `.planning/quick/260522-b06-implement-f32-f64-precision-switching-us/260522-b06-PLAN.md` — the generic-precision quick task that established the `DType` enum + `Tensor<T>` seam (D-08's foundation)

### Cubecl + numerics reference docs (this repo)
- `docs/manual/Cubecl/cubecl_matmul_gemm_example.md` — authoritative for `pyscf_algebra::gemm` calls in the ρ / Vxc contractions
- `docs/manual/Cubecl/Cubecl_multi_ compute.md` — runtime/ComputeClient pattern for any future fused grid kernel (Phase 8)
- `docs/upgrade-cubecl.md` (Phase 1) — the four-crate cubecl-pin upgrade ritual (relevant if DFT-11/WGPU work surfaces a cubecl bump)

### External (Phase 4 will look up)
- libxc documentation — functional naming conventions, the `XC_*` integer-ID scheme, family/kind semantics (to confirm `parse_xc` ID mapping fidelity)
- VV10 reference — Vydrov–Van Voorhis 2010 NLC functional form (`b`/`C` parameters) for the DFT-06 port vs `numint.py:nr_rks_vv10`
- Lebedev–Laikov quadrature references (cited in `LebedevGrid.py` header) — only if the generator port needs cross-checking

</canonical_refs>

<code_context>
## Existing Code Insights

### Reusable Assets
- **Phase 3 SCF engine** (`pyscf-scf`: `OverrideHooks`, generic `kernel<H>`, C-DIIS via `pyscf-diis`, `get_jk`) — RKS/UKS reuse the SCF cycle wholesale, overriding `get_veff` to add Vxc and scale exact-exchange K by `hyb`. No new DIIS/eig/occ machinery.
- **`eval_gto` kernel** (`pyscf-kernels`, Phase 2 D-04) — AO-on-grid evaluation; `GTOval_sph`/`GTOval_sph_deriv1` are the DFT hot paths. Phase 4 lands the deferred `l≥1` variants here.
- **`pyscf-algebra` surface** — `gemm`/`oracle_sum`/`reduce_sum` cover the entire grid-loop contraction chain (D-07) and Becke-weight reductions (D-05/grids), with bit-exact reductions under `release-oracle`.
- **`pyscf-df::DfIntegrals`** (Phase 3 D-10) — DFT-07 reuses the in-memory B-integral assembly + `get_jk_df` for the Coulomb J build under DF.
- **`pyscf-chkfile` primitives + `Checkpointable` trait** (Phase 3 D-06) — `KsResult` impls `Checkpointable`; `/mol` group reuses Phase 2 `mol.dumps()` JSON.
- **`xcfun_rs` GPU dispatch + ERF/shader-f64 auto-fallback** (its GPU-05) — DFT-11's XC-eval-portion fallback delegates here rather than re-implementing the probe.
- **`canonicalize_signs`** (`pyscf-core`, SCF-13) — KS post-eigh sign canonicalization (cross-platform vendor stability, Pitfall 4/12).
- **Precision seam** (Phase 1 D-08 + quick-task-260522-b06) — `PYSCF_DTYPE`→`DType{F32,F64}` resolver (`DType::from_env`, default F64) + generic `Tensor<T: Scalar = f64>` / `DeviceScalar`(f32/f64). DFT threads it (D-08): f64 = default/bit-exact, f32 = opt-in fast path / WGPU-f64 escape hatch. No new env var or dep needed.

### Established Patterns
- **Algebra wall** — `pyscf-dft` + new `pyscf-grids` depend on `pyscf-algebra` only, never `cubecl-*` directly; xtask lint extended to both.
- **Sibling-crate fidelity (hard preference)** — XC parsers mirror `pyscf/dft/libxc.py` + `xcfun.py`; grids mirror `gen_grid.py`/`radi.py` algorithms; `numint.NumInt` mirrors `numint.py` signatures. The `pyscf-grids` own-crate split is the one intentional structural deviation (D-05).
- **"Don't freeze compile"** (user memory) — drives D-03 (off-by-default `libxc`), D-04 (per-functional features), D-06 (port generator, no codegen/large tables). Never run a cargo command that pulls `libxc_rs` into the default dep graph.
- **Trait-callback PyO3 bridge** (Phase 3 D-01) — `KsOverrideHooks` extends `OverrideHooks`; `pyscf-dft` stays pyo3-free, `pyscf-py` owns the bridge.
- **Per-hook `Python::detach`** (Phase 3 D-03) — DFT inherits at the XC-eval/grid-loop compute; the v1 string-only `define_xc_` (D-02) preserves this (no per-block Python callback).
- **Bit-exact-with-upstream under `release-oracle`** — DFT-01 (≤1 µHartree energy) + DFT-04/09 (byte-for-byte grid weights, Pitfall 10).
- **Off-by-default umbrella features** (Phase 1 `gpu`) — D-03's `libxc` feature follows the same convention.

### Integration Points
- **`crates/pyscf-dft/Cargo.toml`** — adds deps: `pyscf-core`, `pyscf-algebra`, `pyscf-gto`, `pyscf-scf`, `pyscf-df`, `pyscf-grids` (new), `pyscf-chkfile`, `pyscf-runtime`, `xcfun-rs`, `tracing`, `thiserror`; `libxc_rs` is an **optional** dep behind `features = ["libxc"]` (D-03). No pyo3 dep.
- **`crates/pyscf-grids/` (new, 19th member)** — depends on `pyscf-core`, `pyscf-algebra`; ports `gen_grid.py`/`radi.py`/`LebedevGrid.py` + `pyscf/data/radii.py` tables (D-05/D-06).
- **`crates/pyscf-py/`** — adds `PyRKS`/`PyUKS` + `dft` submodule + `python/pyscf/dft/__init__.py` overlay re-exporting from `_native.dft`; `to_uks`/`to_rks` Phase 3 stubs wired to real targets.
- **`~/Documents/workspace/libxc_rs`** — cross-crate coordination (D-04): add `[features]` + `cfg`-gated dispatch; a Phase 4 gap-closure plan tracks this with a status marker (cintx-ECP-style).
- **`Cargo.toml` workspace + `.cargo`** — re-enable `libxc_rs` `[patch.crates-io]` behind the `libxc` feature; add `pyscf-grids` member; wire the `libxc` feature through the façade.
- **`.github/workflows/`** — `--features libxc` DFT bit-exact job (the only place the ~6h libxc compile runs in CI, heavily cached); `shader-f64`-less WGPU fallback job (DFT-11); re-enable `libxc_rs` in `nightly-cross-crate.yml`.

</code_context>

<specifics>
## Specific Ideas

- **Bit-exact b3lyp is the headline** — it dictates libxc-default routing (D-01): xcfun would match xcfun's numbers, not upstream PySCF's libxc numbers. The two-parser fidelity is non-negotiable for DFT-01.
- **libxc_rs's 266-kernel unconditional path-dep is a hard, verified fact** (not a guess) — D-03 + D-04 exist specifically because of it. The per-functional-feature work on libxc_rs is a real prerequisite for a tractable gated build and should be sequenced/tracked like the Phase 2 cintx-ECP gap-closure.
- **Lebedev is a generator, not a table** — D-06 ports `gen_oh` + seeds; this is materially simpler than the "5047-line table" framing and avoids any bundled-data / compile-freeze concern. Byte-exactness rides on Phase 1's `release-oracle` ordered-reduction infra (Pitfall 10).
- **The `pyscf-grids` own-crate split is the intentional sibling-fidelity deviation** — justified by VV10 nlcgrids + Phase 7 grad reuse and the Phase 3 split-out precedent; ROADMAP needs the 18→19 member update.
- **Defer optimization** — D-07 keeps the hot path in algebra-orchestrated `gemm`; fused cubecl DFT kernels are explicitly Phase 8 (the 2–5× owner), matching the Phase 3 `pyscf-df` D-10 stance.
- **DFT-11 leans on `xcfun_rs`'s existing fallback** — don't re-implement the `shader-f64`/ERF probe; delegate the XC-eval portion + reuse Phase 1's `PYSCF_BACKEND` resolver for the pipeline.

</specifics>

<deferred>
## Deferred Ideas

- **Python-callable custom XC** (`define_xc_(ni, eval_xc_fn, ...)` — per-block GIL callback inside the grid loop) — deferred past v1 (D-02). String-recombination form ships in Phase 4.
- **Fused cubecl DFT-grid kernels** (rho-eval / Vxc-assembly) — Phase 8 (the 2–5× benchmark + GPU per-backend owner), only if profiling shows the algebra-orchestrated path is the bottleneck (D-07).
- **DFT-D3/D4 dispersion** — v1.x (STATE.md Deferred Items / PROJECT.md).
- **DKS/GKS + r-numint relativistic DFT** (`dks.py`/`gks.py`/`r_numint.py`/`numint2c.py`) — out of milestone (relativistic methods are out of scope).
- **DFT+U** (`rkspu.py`/`ukspu.py`), **SAP** (`sap.py`), **symmetry-adapted KS** (`rks_symm.py`/`uks_symm.py`/`roks.py`) — not in v1 DFT scope.
- **GPU per-backend DFT regression** — Phase 8 (ORACLE-07); Phase 4 ships CPU-backend correctness + the DFT-11 WGPU-fallback honesty check only.
- **On-device libxc XC eval** — `libxc_rs` is cubecl-cpu-only; a GPU grid path forcing host round-trips for libxc XC is a Phase 8 concern (`xcfun_rs` already supports on-device `eval_vec`).
- **xcfun_rs as a default-shipped second backend in the wheel** — v1 ships xcfun_rs compiled by default (D-03) but its functional coverage (79) is narrower than libxc; broadening which backend the wheel defaults to per-functional is a post-v1 tuning question.
- **Per-object precision override** (`mf.set_precision('f32')` runtime override) — deferred past v1 (D-08); the env-var `PYSCF_DTYPE` + a read-only accessor is the v1 surface.
- **f32 numeric-parity regression** (a loose ~chemical-accuracy tolerance bound for the f32 DFT path) — deferred (D-08); v1 f32 ships with a runs-end-to-end smoke test only, no tolerance gate. The oracle/bit-exact suite stays f64.

### Reviewed Todos (not folded)
None — the todo cross-reference scan returned 0 matches for Phase 4.

</deferred>

---

*Phase: 04-dft*
*Context gathered: 2026-05-22*
*Updated 2026-05-22: added D-08 (f32/f64 precision switch via `PYSCF_DTYPE` — f64 default/bit-exact, f32 opt-in fast path + read-only accessor)*

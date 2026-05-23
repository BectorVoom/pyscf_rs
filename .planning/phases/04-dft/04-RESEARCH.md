# Phase 4: DFT - Research

**Researched:** 2026-05-22
**Domain:** Kohn-Sham DFT (RKS/UKS), Becke molecular grids, libxc/xcfun XC evaluation via sibling crates, NumInt grid loop, range-separated hybrids, VV10 NLC, DF-DFT
**Confidence:** HIGH (port targets read directly from upstream PySCF source + sibling-crate source in this session; all locked decisions verified against actual code, not training data)

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

- **D-01: Mirror upstream's two-parser design.** Port `pyscf/dft/libxc.py:parse_xc` (+ `XC_CODES`, `XC_ALIAS`) as the **default** name→spec resolver routing to `libxc_rs`, AND `pyscf/dft/xcfun.py:parse_xc` as the **alternate** resolver routing to `xcfun_rs`. User swaps backends via `mf._numint.libxc = dft.xcfun`. b3lyp MUST route to `libxc_rs` to match upstream numbers (bit-exact contract).
- **D-02: `define_xc_` ships the string-recombination form only in v1.** The user-supplied Python-callable `eval_xc(xc_code, rho, ...)` form is stubbed `NotYetImplemented{deferred}` (it would force a GIL callback inside the hot grid loop, breaking the Phase 3 `Python::detach` seam).
- **D-03: Off-by-default `libxc` cargo feature.** Default `cargo build --workspace` excludes `libxc_rs` entirely. `pyscf-dft` compiles with an `XcBackend` seam; the libxc-backed path returns "libxc feature not enabled" when off; `xcfun_rs` is the default-compiled XC backend. **Hard constraint (user memory): never trigger a default build that pulls libxc_rs's 266 functional-kernel crates (~6h freeze).**
- **D-04: Add per-functional cargo features to `libxc_rs` (cross-crate coordination task).** libxc_rs path-depends on all 266 `libxc-kernel-*` crates unconditionally with no `[features]` block. Phase 4 adds a `[features]` block + `cfg`-gated kernel dispatch so consumers enable only the corpus subset. Structurally identical to Phase 2's cintx-ECP coordination — include an explicit coordination/gap-closure plan with a status marker; the bit-exact-libxc path may be sequenced behind it landing.
- **D-05: New `pyscf-grids` workspace crate (18 → 19).** Grid generation gets its own member, depending on `pyscf-core` + `pyscf-algebra` only (algebra wall). Reused by VV10 `nlcgrids` and Phase 7 grad. ROADMAP needs the 18→19 member update.
- **D-06: Port the Lebedev generator + radial formulas to Rust** (not a snapshot, not runtime file reads). Small const seed/radius tables. Byte-for-byte parity rides on Phase 1 `release-oracle` FMA-free + ordered-reduction infra (Pitfall 10). No heavy build.rs. Snapshot-with-CI-drift is the fallback only if the generator port proves intractable.
- **D-07: Orchestrate `nr_rks`/`nr_uks` grid loop in `pyscf-dft` via `pyscf-algebra` gemm; no new bespoke DFT cubecl kernels in v1.** Reuse `eval_gto` cubecl kernel (land deferred l≥1 variants), do ρ-contraction and Vxc back-contraction as gemm/weighted-gemm through `pyscf-algebra`. Fused DFT grid kernels deferred to Phase 8.

### Claude's Discretion

- **Grid blksize/chunking** — mirror upstream `numint.py` MEM-driven blksize; log `PYSCF_MAX_MEMORY` at entry; no hard pre-flight enforcement in v1 (Phase 6 CCSD-11 owns that).
- **libxc family dispatch (LDA/GGA/MGGA)** — query `libxc_rs` `FunctionalMeta.family()`/`.kind()` to size ρ-derivative evaluation and pick `LdaInput`/`GgaInput`/`MggaInput` + `DerivativeOrder`. For `xcfun_rs`, use `is_gga()`/`is_metagga()` + `eval_setup(Vars, Mode, order)`.
- **Hybrid/RSH coefficient surface** — `NumInt` exposes `hybrid_coeff`/`rsh_coeff(xc_code, spin) -> (omega, alpha, hyb)`. KS `get_veff` reuses the Phase 3 `get_jk` for exact-exchange K; RSH long/short-range K via cintx range-separated Coulomb.
- **DFT-11 WGPU f64 honesty** — delegate XC-eval fallback to `xcfun_rs`'s existing `shader-f64`/ERF `auto_backend` fallback; pipeline-level wgpu→CPU fallback reuses Phase 1's `PYSCF_BACKEND` resolver + `tracing::warn!`. `libxc_rs` is CPU-host-only.
- **VV10 NLC (DFT-06)** — `mf.nlc='VV10'` + a separate `mf.nlcgrids` (coarser); port upstream `nr_nlc_vxc`/`_vv10nlc`. Decide whether VV10 ships in core RKS plan or follow-on.
- **DF-DFT (DFT-07)** — reuse `pyscf-df::DfIntegrals` (Phase 3 D-10) for the Coulomb-J build; no new DF crate.
- **`KsResult` chkfile** — `impl Checkpointable for KsResult` via `pyscf-chkfile` HDF5 primitives; add `xc`/`grids` metadata to schema.
- **`KsOverrideHooks` trait** — extend the Phase 3 `OverrideHooks` shape with DFT hooks (`get_veff`, `define_xc_`); `pyscf-dft` stays pyo3-free; `pyscf-py` provides `PyOverrideBridge`. `to_uks`/`to_rks` Phase 3 stubs wired to real targets.
- **Phase MVP sequencing** — core path (RKS/UKS + grids + XC parser + libxc eval, bit-exact on SVWN/PBE/B3LYP) first; RSH (DFT-05), VV10 (DFT-06), DF-DFT (DFT-07) as follow-on plans. Planner finalizes wave structure + corpus-functional list.

### Deferred Ideas (OUT OF SCOPE)

- Python-callable custom XC (`define_xc_(ni, eval_xc_fn, ...)` per-block GIL callback) — past v1 (D-02).
- Fused cubecl DFT-grid kernels (rho-eval / Vxc-assembly) — Phase 8.
- DFT-D3/D4 dispersion — v1.x.
- DKS/GKS + r-numint relativistic DFT — out of milestone.
- DFT+U (`rkspu`/`ukspu`), SAP (`sap.py`), symmetry-adapted KS (`rks_symm`/`uks_symm`/`roks`) — not in v1.
- GPU per-backend DFT regression — Phase 8 (ORACLE-07); Phase 4 ships CPU-backend correctness + the DFT-11 WGPU-fallback honesty check only.
- On-device libxc XC eval — libxc_rs is cubecl-cpu-only; GPU grid path forcing host round-trips is Phase 8.
- xcfun_rs as default-shipped second backend in the wheel — post-v1 tuning.
</user_constraints>

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| DFT-01 | `dft.RKS`/`dft.UKS` bit-exact to upstream on corpus under `release-oracle` | RKS/UKS reuse Phase 3 SCF `kernel<H>` cycle, override `get_veff` (Vxc + scaled-K); energy via `energy_elec`. Bit-exactness rides on `oracle_sum`/`oracle_dot` reductions (Phase 1) for ρ-contraction + Becke-weight sums. |
| DFT-02 | XC string parser handles all upstream forms + `XC_ALIAS` | Port `pyscf/dft/libxc.py:parse_xc` (read this session, lines 491-718). Tables `XC_CODES` (line 154+), `XC_ALIAS` (217+), preprocessing in `dft_parser.py`. Parser parity test target. |
| DFT-03 | libxc → `libxc_rs`; xcfun → `xcfun_rs`; bit-identical to C libs | `libxc_rs` registry/builder/batch/input/output API surface mapped below; `xcfun_rs` `Functional` API mapped below. libxc_rs reports version 7.0.0 (`registry/mod.rs`). |
| DFT-04 | `Grids` class + default Becke partition + Treutler radial + Lebedev angular byte-for-byte | Port `gen_grid.py` (Grids class defaults: `radi.treutler`, `treutler_atomic_radii_adjust`, `original_becke`, `nwchem_prune`, `BRAGG_RADII`, level=3) + `radi.py` + `LebedevGrid.py` `SphGenOh`. **Becke partition uses a C kernel `VXCgen_grid` — port the pure-Python fallback (gen_grid.py:314-329).** |
| DFT-05 | RSH (`omega`, `alpha`, `beta`) via range-separated Coulomb | **NOT distinct intor symbols.** Upstream uses `mol.with_range_coulomb(omega)` → sets `PTR_RANGE_OMEGA = env[8]` → standard `int2e` computes `erf(ωr)/r`. cintx reads `env[8]` (verified in cintx STATE/PITFALLS). `get_veff` RSH branch: rks.py:108-129. |
| DFT-06 | VV10 NLC (`mf.nlc='VV10'`, `mf.nlcgrids`) | Port `numint.py:_vv10nlc` (471-545) + `nr_nlc_vxc` (1347+). Inner kernel is C `VXC_vv10nlc` — port the documented pure-Python double-loop (471-538). NLC coeffs from libxc `LIBXC_nlc_coeff` → `libxc_rs` `NlcCoefficients`. Default b=5.9, C=0.0093. |
| DFT-07 | DF-DFT (`dft.RKS(mol).density_fit()`) | Reuse `pyscf-df::DfIntegrals` + `get_jk_df` (Phase 3 D-10) for the J build. Shape mirrors `mf.density_fit()`. |
| DFT-08 | SCF subclass-override hooks re-validate at DFT scope (adds `get_veff`, `define_xc_`) | Extend Phase 3 `OverrideHooks` trait → `KsOverrideHooks`; `slf.call_method1` dispatch in `pyscf-py`. Re-asserts Pitfall 7 on larger surface. |
| DFT-09 | `mf.grids.level = N` for N∈{0..9} matches upstream sizes | `RAD_GRIDS`/`ANG_ORDER` tables (gen_grid.py:672-699) + `_default_rad`/`_default_ang` period lookup + `LEBEDEV_ORDER` map (LebedevGrid.py:4999-5033). Const-port these tables. |
| DFT-10 | `numint.NumInt` exposes `eval_xc`, `eval_rho`, `nr_rks`, `nr_uks` matching upstream signatures | Signatures mapped below from `numint.py`. `nr_rks`(1074), `nr_uks`(1192), `eval_rho`(116), `NumInt` class (2835). |
| DFT-11 | cubecl WGPU gated on `shader-f64`; CPU fallback with warning | Delegate XC-eval fallback to `xcfun_rs` `xcfun-gpu` `auto_backend()`/`must_fall_back_to_cpu()`; pipeline fallback via Phase 1 `PYSCF_BACKEND` + `tracing::warn!`. libxc_rs CPU-only by construction. |
</phase_requirements>

## Summary

Phase 4 builds Kohn-Sham DFT on top of the shipped Phase 3 SCF engine. The architecture is overwhelmingly a **port-the-reference-algorithm** exercise against well-understood upstream Python sources, plus **integration of two XC sibling crates** behind a backend seam, plus **one cross-crate gap-closure task** (libxc_rs per-functional features). There is almost no new compute-kernel design: the grid loop is gemm-orchestrated through `pyscf-algebra` (D-07), grids are deterministic algorithmic construction (D-06), and XC evaluation is delegated to `libxc_rs`/`xcfun_rs`.

Three non-obvious facts surfaced from reading the actual source this session, each of which the planner must internalize:

1. **The default Becke partition path in `get_partition` calls a C function `libdft.VXCgen_grid`** (gen_grid.py:307), and **the VV10 inner kernel calls a C function `libdft.VXC_vv10nlc`** (numint.py:539). Neither lives in cintx, libxc_rs, or xcfun_rs — they are part of PySCF's own `libdft` C extension. Both have documented pure-Python equivalents in the same files (gen_grid.py:314-329; numint.py:526-537 commented block). Phase 4 must port those pure-Python algorithms to Rust (in `pyscf-grids` and `pyscf-dft`/`pyscf-grids` respectively), exactly the "port the reference algorithm" pattern from Phase 2.

2. **Range-separated hybrids (DFT-05) do NOT use distinct `int2e_lr_*`/`int2e_sr_*` integral symbols.** The CONTEXT framing is conceptual. Upstream sets `PTR_RANGE_OMEGA = env[8]` via `mol.with_range_coulomb(omega)` and calls the *same* `int2e` / `get_k`; libcint/cintx then evaluates `erf(ωr)/r` (long-range, ω>0) or the short-range complement (ω<0). cintx already reads `env[8]` (verified). The work is a range-coulomb env-slot setter on the `pyscf-gto` `mol.intor` path plus the `get_veff` SR/LR branching (rks.py:108-129).

3. **The `pyscf-grids` and `pyscf-dft` crates must NOT be added to the dependency-wall lint's `ALLOWED_CRATES` carve-out.** The CONTEXT integration note ("extend allowlist for pyscf-grids + pyscf-dft") is backwards: the lint denies cubecl-* deps to all `pyscf-*` crates *except* the carve-out (`pyscf-algebra`, `pyscf-runtime`, `pyscf-kernels`). Since grids/dft route through algebra, they correctly stay *out* of the carve-out. Adding them would defeat the wall. (See Common Pitfalls.)

**Primary recommendation:** Sequence as (W0) grids crate scaffold + libxc_rs feature gap-closure plan with a status marker; (W1) `pyscf-grids` byte-exact (Lebedev + radial + Becke partition port) — the highest-risk bit-exact target; (W2) XC parser + `XcBackend` seam + xcfun_rs default eval; (W3) RKS/UKS core grid loop bit-exact on SVWN/PBE/B3LYP via libxc feature; (W4) RSH + VV10 + DF-DFT follow-ons; (W5) override re-validation + WGPU honesty CI. Land the deferred `l≥1` `eval_gto` variants in `pyscf-kernels` early in W1 since the grid loop cannot evaluate any non-H molecule without them.

## Architectural Responsibility Map

| Capability | Primary Tier | Secondary Tier | Rationale |
|------------|-------------|----------------|-----------|
| XC string parsing (`parse_xc`) | `pyscf-dft` (port of libxc.py/xcfun.py) | — | Pure string→spec resolution; no compute. Backend-specific (libxc vs xcfun tables). |
| Grid point/weight generation | `pyscf-grids` (new crate) | `pyscf-algebra` (Becke-weight ordered reductions) | D-05: cross-method-shared; algebra wall for bit-exact partition sums. |
| Lebedev angular + radial schemes | `pyscf-grids` | — | D-06: deterministic algorithmic construction, const seed/radius tables. |
| AO-on-grid evaluation | `pyscf-kernels` (`eval_gto`) | `pyscf-algebra` (launch) | Phase 2 D-04 carve-out; l≥1 variants land here this phase. |
| ρ-contraction + Vxc back-contraction | `pyscf-algebra` (gemm) | `pyscf-dft` (orchestration) | D-07: no bespoke kernels; algebra owns matmul, dft owns the loop. |
| XC functional evaluation (Exc/Vxc) | `libxc_rs` / `xcfun_rs` (sibling) | `pyscf-dft` (`XcBackend` seam) | D-01/D-03: delegation, not reimplementation. |
| Exact-exchange K + RSH SR/LR K | `pyscf-scf` (`get_jk`/`get_k`) + `pyscf-gto` (range-coulomb env) | `pyscf-dft` (`get_veff` branching) | Reuse Phase 3 K path; omega via env[8]. |
| VV10 non-local correlation | `pyscf-dft` (+`pyscf-grids` nlcgrids) | `pyscf-algebra` | Port pure-Python double-loop; second coarser grid instance. |
| DF Coulomb-J under DFT | `pyscf-df` (`DfIntegrals`/`get_jk_df`) | `pyscf-dft` | Phase 3 D-10 reuse; no new crate. |
| Python subclass-override dispatch | `pyscf-py` (`PyOverrideBridge`) | `pyscf-dft` (`KsOverrideHooks` trait) | Phase 3 D-01 bridge pattern; dft stays pyo3-free. |
| chkfile persistence | `pyscf-chkfile` (HDF5 primitives) | `pyscf-dft` (`Checkpointable for KsResult`) | Phase 3 D-06 pattern. |
| Backend selection / WGPU f64 honesty | `pyscf-runtime` (`PYSCF_BACKEND`) + `xcfun-gpu` (auto_backend) | `pyscf-dft` | Phase 1 resolver + xcfun_rs existing fallback. |

## Standard Stack

This phase consumes existing workspace crates plus two sibling XC crates. No new external dependencies beyond what the workspace already pins.

### Core
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| `xcfun-rs` | path/`[patch.crates-io]` under BectorVoom | Default XC backend (compiled by default per D-03) | Sibling-crate-fidelity; xcfun routes here (DFT-03). 79 functionals. `[CITED: ~/Documents/workspace/xcfun_rs/crates/xcfun-rs/Cargo.toml]` |
| `libxc_rs` | path (optional, behind `libxc` feature) | libxc XC backend; b3lyp/PBE/SVWN bit-exact path (DFT-01/03) | The *only* path to upstream-libxc numbers. v7.0.0. **266-kernel unconditional dep — D-04 gap-closure required.** `[VERIFIED: ~/Documents/workspace/libxc_rs/Cargo.toml read this session]` |
| `pyscf-algebra` | workspace | gemm/oracle_sum/reduce_sum for the entire grid-loop + Becke-weight reductions | Algebra wall; bit-exact reductions under release-oracle. `[CITED: 01-CONTEXT.md, codebase]` |
| `pyscf-kernels` | workspace | `eval_gto` AO-on-grid (l≥1 variants land here) | Phase 2 D-04 carve-out; only cubecl-consuming method-adjacent crate. `[VERIFIED: eval_gto.rs read this session — currently s-shell only]` |
| `pyscf-scf` | workspace | `kernel<H>`, `OverrideHooks`, C-DIIS, `get_jk`/`get_k` | RKS/UKS reuse the SCF cycle wholesale. `[CITED: 03-CONTEXT.md]` |
| `cintx` (via `pyscf-gto::mol.intor`) | path | `int2e` with `PTR_RANGE_OMEGA` for RSH; `GTOval_sph`/`GTOval_sph_deriv1` source data | RSH reads env[8]; cintx supports it. `[VERIFIED: cintx STATE.md/PITFALLS.md read this session]` |

### Supporting
| Library | Version | Purpose | When to Use |
|---------|---------|---------|-------------|
| `pyscf-df` | workspace | DF Coulomb-J build | DFT-07 only. `[CITED: 03-CONTEXT.md]` |
| `pyscf-chkfile` | workspace | HDF5 chkfile primitives + `Checkpointable` | `KsResult` persistence. `[CITED: 03-CONTEXT.md]` |
| `pyscf-runtime` | workspace | `PYSCF_BACKEND` resolver, `BackendKind` | DFT-11 pipeline fallback. `[CITED: 01-CONTEXT.md]` |
| `tracing` | 0.1 | fallback warnings, blksize/memory logging | DFT-11 + grid loop. `[ASSUMED]` |
| `thiserror` | 2.x | `NotYetImplemented`/backend errors | error surface. `[ASSUMED]` |

### Alternatives Considered
| Instead of | Could Use | Tradeoff |
|------------|-----------|----------|
| libxc_rs for b3lyp | xcfun_rs | Would match xcfun's numbers, not upstream PySCF's libxc numbers → fails DFT-01 bit-exact. Rejected per D-01. |
| Porting `VXCgen_grid` C kernel | Snapshot grid weights from upstream | D-06 fallback only; loses parameterized grids (level/atom_grid/prune) and adds bundled-data/drift-check burden. |
| New DF crate for DFT-07 | `pyscf-df` reuse | No new crate needed (D-10 precedent). |
| Bespoke cubecl Vxc kernel | algebra gemm chain | D-07: defers optimization to Phase 8; respects algebra wall. |

**Installation:** No new crates published; workspace edits only. Phase 4 re-enables (behind the `libxc` feature) the commented `[patch.crates-io]` entry at workspace `Cargo.toml:109`:
```toml
# libxc_rs = { path = "../libxc_rs" }   # currently disabled (Cargo.toml:109)
```

**Version verification (performed this session, source-read not registry — sibling crates are path-deps, never published):**
- `libxc_rs` version: `7.0.0` per `registry::version()` (`src/registry/mod.rs`); 649 functionals registered, but only 283 kernel path-deps wired (266 default-members + deferred). `[VERIFIED: source read]`
- `xcfun-rs`: workspace-versioned, 79 `FUNCTIONAL_DESCRIPTORS`. `[VERIFIED: source read]`
- cubecl `=0.10.0` across all four crates (lockstep ABI). `[VERIFIED: Cargo.toml read]`

## Package Legitimacy Audit

> All dependencies are **workspace-internal path deps or path/`[patch.crates-io]` sibling crates under the `BectorVoom` GitHub org** — none are fetched from a public registry. The slopsquatting threat model (registry-hosted hallucinated packages) does not apply. slopcheck was not run because there is no registry install surface in this phase.

| Package | Registry | Age | Downloads | Source Repo | slopcheck | Disposition |
|---------|----------|-----|-----------|-------------|-----------|-------------|
| `libxc_rs` | none (path/patch) | sibling repo | n/a | github.com/BectorVoom/libxc_rs | n/a (path dep) | Approved (read-only research; never compile) |
| `xcfun-rs` | none (path/patch) | sibling repo | n/a | github.com/BectorVoom/xcfun_rs | n/a (path dep) | Approved |
| `cintx` (+ subcrates) | none (path/patch) | sibling repo | n/a | github.com/BectorVoom/cintx | n/a (path dep) | Approved (already wired Phase 2) |
| `tracing` | crates.io | 6+ yrs | very high | github.com/tokio-rs/tracing | not run | Approved (already in workspace) `[ASSUMED — already a workspace dep]` |
| `thiserror` | crates.io | 5+ yrs | very high | github.com/dtolnay/thiserror | not run | Approved (already in workspace) `[ASSUMED — already a workspace dep]` |

**Packages removed due to slopcheck [SLOP] verdict:** none
**Packages flagged as suspicious [SUS]:** none

*No new registry packages are introduced by this phase. `tracing`/`thiserror` are pre-existing workspace dependencies, not new installs.*

## Architecture Patterns

### System Architecture Diagram

```
  dft.RKS(mol, xc='b3lyp').run()   [pyscf-py PyRKS → pyscf-dft RKS]
            │
            ▼
  ┌──────────────────────────────────────────────────────────────┐
  │ Phase 3 SCF kernel<H>  (reused wholesale: C-DIIS, eig, occ)    │
  │   each cycle calls  get_veff(ks, mol, dm)  ◄── KS override     │
  └──────────────────────────────────────────────────────────────┘
            │ get_veff = J  +  Vxc  −  (hyb/RSH-scaled) K
            ├──────────────► get_jk / get_k  [pyscf-scf]
            │                  └─ RSH: mol.intor(int2e, env[8]=±omega) [pyscf-gto→cintx]
            │
            ▼  Vxc + Exc  via  NumInt.nr_rks / nr_uks   [pyscf-dft, D-07]
  ┌──────────────────────────────────────────────────────────────┐
  │  for block in grid.block_loop(blksize):       (mem-driven)     │
  │    1. AO  = eval_gto(mol, coords)   [pyscf-kernels, l≥1 here]  │
  │    2. ρ   = AOᵀ · D · AO            [pyscf-algebra gemm]       │
  │    3. (exc,vxc) = XcBackend.eval(xc_spec, ρ[,σ,τ])             │
  │           ├─ libxc  → libxc_rs  BatchEvaluator (CPU)  [feat]   │
  │           └─ xcfun  → xcfun_rs  Functional.eval_vec            │
  │    4. Vxc += AOᵀ · (w · vxc · AO)  [pyscf-algebra weighted-gemm]│
  │    5. Exc += oracle_sum(w · exc · ρ)  [bit-exact reduction]   │
  └──────────────────────────────────────────────────────────────┘
            ▲
            │ grid (coords, weights)  built once per geometry
  ┌──────────────────────────────────────────────────────────────┐
  │  pyscf-grids  (NEW crate, D-05)                               │
  │   gen_atomic_grids:  radial(treutler) × Lebedev(SphGenOh)     │
  │   get_partition:     Becke (port pure-Python VXCgen_grid path)│
  │                       weights via pyscf-algebra ordered-reduce │
  └──────────────────────────────────────────────────────────────┘

  VV10 (DFT-06): second coarser pyscf-grids instance (nlcgrids)
                 → port _vv10nlc double-loop (port VXC_vv10nlc)
  DF-DFT (DFT-07): step "get_j" routes through pyscf-df DfIntegrals
```

### Recommended Project Structure
```
crates/
├── pyscf-grids/          # NEW (19th member) — D-05
│   └── src/
│       ├── lib.rs        # Grids struct (level/atom_grid/prune/radi_method/becke_scheme/atomic_radii)
│       ├── radial.rs     # port radi.py: treutler_ahlrichs (default), gauss_chebyshev, becke, delley, mura_knowles
│       ├── radii.rs       # const BRAGG/COVALENT/SG1 tables (port pyscf/data/radii.py + radi.py:SG1RADII)
│       ├── lebedev.rs    # port LebedevGrid.py: SphGenOh + MakeAngularGrid_N seeds + LEBEDEV_ORDER map
│       ├── prune.rs      # nwchem_prune (default) / sg1_prune / treutler_prune
│       ├── partition.rs  # Becke partition (port pure-Python get_partition fallback) + atomic_radii_adjust
│       └── levels.rs     # RAD_GRIDS / ANG_ORDER tables + _default_rad/_default_ang
└── pyscf-dft/            # fill the Phase 1 stub — DFT-01..11
    └── src/
        ├── lib.rs        # RKS / UKS (reuse Phase 3 kernel<H>; override get_veff)
        ├── numint.rs     # NumInt: eval_rho, eval_xc, nr_rks, nr_uks (port numint.py signatures)
        ├── xc_backend.rs # XcBackend seam: enum { Xcfun, Libxc(cfg feature) }
        ├── parser/
        │   ├── libxc.rs  # port libxc.py:parse_xc + XC_CODES + XC_ALIAS (default)
        │   └── xcfun.rs  # port xcfun.py:parse_xc (alternate)
        ├── veff.rs       # get_veff: J + Vxc − scaled K; RSH SR/LR branching
        ├── vv10.rs       # nr_nlc_vxc + _vv10nlc port (DFT-06)
        ├── hooks.rs      # KsOverrideHooks trait (extends OverrideHooks); define_xc_ string form
        └── chkfile.rs    # impl Checkpointable for KsResult
```

### Pattern 1: XcBackend seam (D-01/D-03)
**What:** A backend-dispatch enum so the default build never references libxc_rs symbols.
**When to use:** All XC evaluation in the grid loop.
**Example:**
```rust
// pyscf-dft/src/xc_backend.rs — conceptual shape; mirrors xcfun-rs's own feature gating
pub enum XcBackend {
    Xcfun,                                  // default-compiled
    #[cfg(feature = "libxc")]
    Libxc,                                  // only when --features libxc
}
impl XcBackend {
    pub fn eval(&self, spec: &XcSpec, rho: RhoBlock, order: DerivOrder) -> Result<XcOutput> {
        match self {
            XcBackend::Xcfun => xcfun_eval(spec, rho, order),       // xcfun_rs Functional::eval_vec
            #[cfg(feature = "libxc")]
            XcBackend::Libxc => libxc_eval(spec, rho, order),       // libxc_rs BatchEvaluator
            #[cfg(not(feature = "libxc"))]
            _ => Err(Error::LibxcFeatureNotEnabled),
        }
    }
}
```
The `xcfun-rs` facade `Cargo.toml` `[features]` block (read this session) is the canonical model for what D-04 must add to `libxc_rs`.

### Pattern 2: libxc_rs batch evaluation (the DFT-03 libxc path)
**What:** Resolve name → ID → `Functional`, size input by family, batch-evaluate.
**When to use:** `--features libxc` grid blocks.
**Example:**
```rust
// Source: ~/Documents/workspace/libxc_rs/src/{registry,api,input,output}/ (read this session)
let id   = libxc_rs::lookup_by_name("gga_x_pbe")?;          // FunctionalId
let meta = libxc_rs::lookup_by_id(id.raw())?;               // &FunctionalMeta → .family()/.kind()
let func = libxc_rs::FunctionalBuilder::new(id).spin(spin).build()?;
let input = libxc_rs::GgaInput::new(&rho, &sigma, np, spin)?;   // family-sized: Lda/Gga/MggaInput
let mut out = libxc_rs::GgaOutput::with_order(np, spin, DerivativeOrder::Vxc);
let mut ev = libxc_rs::BatchEvaluator::new(spin, np_max);
ev.evaluate(&func, &input, DerivativeOrder::Vxc, &mut out)?;     // out.zk, out.vrho, out.vsigma
```
Family dispatch chooses `LdaInput`(ρ) / `GgaInput`(ρ,σ) / `MggaInput`(ρ,σ,lapl,τ). `DerivativeOrder` = Exc/Vxc/Fxc/Kxc/Lxc.

### Pattern 3: xcfun_rs evaluation (the default backend)
**What:** Configure functional weights, size buffers, batch-eval.
**Example:**
```rust
// Source: ~/Documents/workspace/xcfun_rs/crates/xcfun-rs/src/functional.rs (read this session)
let mut f = xcfun_rs::Functional::new();
f.set("pbex", 1.0)?;  f.set("pbec", 1.0)?;                  // weights from parse_xc
let order = 1;                                              // 0=Exc, 1=Vxc, ...
f.eval_setup(Vars::AB_GGA, Mode::Polarized, order)?;       // sizes input_length/output_length
// nr_points >= 64 → GPU Batch dispatch (auto_backend, ERF auto-fallback to CPU); < 64 → per-point loop
f.eval_vec(&density, in_pitch, &mut out, out_pitch, nr_points)?;
```

### Pattern 4: RSH via range-coulomb env slot (DFT-05)
**What:** Long/short-range exact-exchange K by toggling `PTR_RANGE_OMEGA = env[8]`.
**When to use:** `get_veff` when `omega != 0` (CAM-B3LYP, ωB97X, etc.).
**Example:**
```rust
// Mirrors pyscf/dft/rks.py:108-129 get_veff RSH branch + mol.with_range_coulomb(omega)
let (omega, alpha, hyb) = numint.rsh_and_hybrid_coeff(&ks.xc, mol.spin);
let vk = if omega == 0.0 {
    let (vj, mut vk) = ks.get_jk(mol, dm)?;  vk *= hyb; vk            // standard hybrid
} else {
    // env[8] > 0 → erf(ωr)/r (LR);  env[8] < 0 → SR complement
    let vk_full = ks.get_jk(mol, dm)?.1 * hyb;
    let vk_lr   = ks.get_k_with_omega(mol, dm, omega)? * (alpha - hyb);  // sets env[8]=+omega
    vk_full + vk_lr
};
// vxc += vj − 0.5*vk ; ground-state exc -= 0.5*0.5*einsum(dm, vk)
```
The `get_k_with_omega` path sets `env[8]` on the `pyscf-gto::mol.intor` `int2e` call (cintx reads it) and restores it after — exactly `mol.with_range_coulomb`.

### Anti-Patterns to Avoid
- **Distinct `int2e_lr_*`/`int2e_sr_*` symbols.** They are not real cintx/libcint symbols. Use the standard `int2e` + `env[8]` omega slot.
- **Adding pyscf-grids/pyscf-dft to the dependency-wall carve-out.** They route through algebra; the wall correctly denies them direct cubecl. (See Common Pitfalls.)
- **Per-block Python callback for custom XC.** Deferred (D-02) — breaks the `Python::detach` seam.
- **Hand-rolling Lebedev as a 5047-line static table.** It is a *generator* (`SphGenOh` + 7 codes + small seed params). Port the generator (D-06).
- **Reimplementing XC functional math.** Delegate to the sibling crates (D-03). Even VWN/PBE go through libxc_rs/xcfun_rs.
- **Triggering a default-profile build that pulls libxc_rs.** ~6h freeze (user memory). Keep it behind `--features libxc` and the D-04 per-functional gate.

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| XC functional energy/potential | Custom VWN/PBE/B3LYP math | `libxc_rs` / `xcfun_rs` | Hundreds of functionals, ext-params, derivative orders; the entire point of D-01/D-03 fidelity. |
| GEMM / weighted contraction | Hand-loop ρ=AOᵀDAO | `pyscf-algebra::gemm` | Algebra wall + bit-exact reductions; D-07. |
| Bit-exact summation | Naive `iter().sum()` | `oracle_sum`/`oracle_dot` | Reduction-order determinism (Pitfall 2/10) — the whole grid-weight bit-exactness rides on this. |
| Linear solve / eigh in SCF cycle | New solver | Phase 3 `kernel<H>` / `pyscf-algebra::eigh` | RKS/UKS reuse the SCF cycle wholesale. |
| DF B-integrals + J build | New DF assembly | `pyscf-df::DfIntegrals`/`get_jk_df` | DFT-07; Phase 3 D-10. |
| HDF5 chkfile I/O | Raw hdf5 | `pyscf-chkfile` primitives | Phase 3 D-06 schema compatibility. |
| WGPU f64 capability probe | New shader-f64 detection | `xcfun-gpu::auto_backend()` / `must_fall_back_to_cpu()` + Phase 1 `PYSCF_BACKEND` | DFT-11; reuse, don't re-probe. |
| AO-on-grid evaluation | New eval loop | `pyscf-kernels::eval_gto` (land l≥1) | Phase 2 D-04 carve-out. |

**Key insight:** Phase 4 is integration + algorithm-porting, not invention. The single genuinely *new* numerical code is (a) the grid generators in `pyscf-grids` (Lebedev/radial/Becke-partition — all direct ports of readable upstream algorithms) and (b) the VV10 double-loop port. Everything else composes shipped Phase 1-3 crates and the two XC sibling crates.

## Runtime State Inventory

> This phase is primarily greenfield (filling the empty `pyscf-dft` stub + a new `pyscf-grids` crate). It is **not** a rename/refactor. However, two cross-cutting "build artifact / workspace registration" state items exist and must be handled explicitly.

| Category | Items Found | Action Required |
|----------|-------------|------------------|
| Stored data | None — DFT produces new chkfile groups (`xc`/`grids` metadata added to `KsResult` schema), but writes new records; no existing datastore stores a renamed key. | code edit (extend chkfile schema) |
| Live service config | None — verified: no external services, daemons, or UI-stored config involved in DFT. | none |
| OS-registered state | None — verified: no OS task/daemon registration. | none |
| Secrets/env vars | New runtime env vars *read* (not stored): `PYSCF_MAX_MEMORY` (blksize), `PYSCF_BACKEND` (Phase 1), `PYSCF_DTYPE`, `XCFUN_MIN_BATCH_SIZE` (xcfun_rs eval_vec threshold, default 64). No secrets. The libxc bit-exact CI job needs `--features libxc` set in workflow env. | document; CI workflow edit |
| Build artifacts / installed packages | **(1)** Workspace member count 18→19 (`Cargo.toml` `members`, ROADMAP.md, STATE.md `total members`) — adding `crates/pyscf-grids`. **(2)** `[patch.crates-io] libxc_rs` re-enabled behind `libxc` feature (`Cargo.toml:109` commented today). **(3)** `nightly-cross-crate.yml:40` libxc_rs exclusion re-enabled. **(4)** maturin wheel builds that enable `--features libxc` produce a *different (larger)* wheel — relevant for DIST-03 60MB ceiling (Phase 8 concern, flag now). | workspace + CI edits |

**Cross-crate coordination state (D-04):** `~/Documents/workspace/libxc_rs` requires a `[features]` block + `cfg`-gated dispatch added by Phase 4. This is a *separate repo* — like the Phase 2 cintx-ECP gap-closure (cintx#11), it should be tracked with a status marker in a coordination plan. The change touches: `Cargo.toml` (make 266 deps `optional = true` + per-functional features), `src/kernel/{lda,gga,mgga}.rs` (the auto-generated `pub use libxc_kernel_* as *` re-export façades — `cfg`-gate each), and `src/eval/dispatch.rs` (the per-functional `match` arms — `cfg`-gate each). The dispatch is enum-based (`LdaFunctional::LdaX => launch_lda_x(...)`), so gating is mechanical but touches the generator tool (`tools/generate_kernel_reexports.py`).

## Common Pitfalls

### Pitfall 1: Treating RSH as distinct integral symbols
**What goes wrong:** Planner writes tasks to call `mol.intor('int2e_lr_sph')`; cintx errors (no such symbol) or the bit-exact fixture diverges.
**Why it happens:** CONTEXT/ROADMAP use "`int2e_lr_*`/`int2e_sr_*`" as conceptual shorthand.
**How to avoid:** Implement `mol.with_range_coulomb(omega)`-equivalent: set `PTR_RANGE_OMEGA = env[8]` (positive=long-range `erf(ωr)/r`, negative=short-range complement) before the standard `int2e` call, restore after. cintx already reads env[8].
**Warning signs:** "unknown intor symbol" errors; CAM-B3LYP energy off by the exchange fraction.

### Pitfall 2: Adding pyscf-grids / pyscf-dft to the dependency-wall carve-out
**What goes wrong:** Following CONTEXT's note literally ("extend allowlist for pyscf-grids + pyscf-dft") adds them to `ALLOWED_CRATES` in `check_dependency_wall.rs`, which would *permit* them to depend on cubecl-* directly — defeating the algebra wall.
**Why it happens:** The lint is a *denylist with carve-out*, not an allowlist; the CONTEXT note's framing is inverted.
**How to avoid:** Do NOT add grids/dft to `ALLOWED_CRATES` (currently `["pyscf-algebra","pyscf-runtime","pyscf-kernels"]`). They route through `pyscf-algebra` and should remain subject to the wall. The only lint edit needed is *none* for the wall; the `forbidden-paths` lint may need DFT module entries.
**Warning signs:** A PR diff adds crate names to `ALLOWED_CRATES`.

### Pitfall 3: Grid-weight bit-exactness (Pitfall 10 — owned here)
**What goes wrong:** Becke partition weights or Lebedev points differ in the last ULPs vs upstream; level-0..9 byte-for-byte fails.
**Why it happens:** (a) Naive summation order in the `pbecke.sum(axis=0)` normalization (gen_grid.py:337); (b) FMA contraction in the radial/angular product; (c) wrong default scheme — the `Grids` *class* defaults differ from the `gen_atomic_grids` *function* defaults.
**How to avoid:** Use `oracle_sum` for the partition normalization; build under `release-oracle` (FMA-free); use the **class** defaults: `radi_method=radi.treutler` (Treutler-Ahlrichs, NOT gauss_chebyshev), `radii_adjust=treutler_atomic_radii_adjust`, `becke_scheme=original_becke`, `prune=nwchem_prune`, `atomic_radii=BRAGG_RADII`, `level=3` (verified gen_grid.py:490-499). Note `ATOM_SPECIFIC_TREUTLER_GRIDS=True` default affects ~1e-6/atom (radi.py:142).
**Warning signs:** weights match to ~1e-10 but not byte-for-byte; divergence grows with atom count (partition-sum order) or with element period (radial scheme).

### Pitfall 4: Porting the C kernels instead of the pure-Python fallbacks
**What goes wrong:** Looking for `VXCgen_grid`/`VXC_vv10nlc` in cintx/libxc/xcfun and finding nothing; assuming the feature is unavailable.
**Why it happens:** These are PySCF's own `libdft` C extension functions, not in any sibling crate.
**How to avoid:** Port the documented pure-Python equivalents in the same files — Becke partition fallback (gen_grid.py:314-329), VV10 double-loop (numint.py:526-538 commented block). Both are short, readable, and the byte-exact reference.
**Warning signs:** searching sibling crates for `VXC*` symbols; treating VV10/Becke as blocked.

### Pitfall 5: Default build accidentally pulling libxc_rs (~6h freeze)
**What goes wrong:** Re-enabling the `[patch.crates-io] libxc_rs` line *unconditionally* (not behind the feature), or adding libxc_rs as a non-optional dep, triggers the 266-kernel compile on every `cargo build`.
**Why it happens:** The patch entry and the dep must both be feature-gated; easy to forget the `optional = true` on the dep.
**How to avoid:** `libxc_rs = { path = "...", optional = true }` in `pyscf-dft/Cargo.toml`; `libxc = ["dep:libxc_rs"]` feature; keep the `[patch.crates-io]` entry but ensure nothing references it unless the feature is on. The D-04 per-functional gate keeps even the gated build to minutes. **Never run `cargo build/test/check` with `--features libxc` locally** — only the dedicated, heavily-cached CI job does.
**Warning signs:** a `cargo build --workspace` that suddenly takes hours.

### Pitfall 6: WGPU silently degrading to f32 (Pitfall 3 re-validation)
**What goes wrong:** On a `shader-f64`-less device, wgpu runs in f32, producing wrong DFT energies silently.
**How to avoid:** Gate wgpu on `shader-f64` Vulkan extension at runtime; fall back to CPU with `tracing::warn!`. Delegate the XC-eval-portion probe to `xcfun-gpu::auto_backend()` (its ERF/shader-f64 fallback already exists); reuse Phase 1 `PYSCF_BACKEND` for the pipeline. CI job on a `shader-f64`-less device must print the warning and still produce CPU-correct numbers.
**Warning signs:** GPU DFT numbers wrong by f32-precision margins with no warning.

## Code Examples

### Common Operation 1: parse_xc core loop (the DFT-02 port)
```python
# Source: pyscf/dft/libxc.py:491-718 (read this session) — port to pyscf-dft/src/parser/libxc.rs
# Returns (hyb, alpha, omega), ((libxc_id, fac), ...)
# Key sub-cases the parser MUST handle (from parse_token, lines 565-...):
#   '-' sign prefix; '*' factor (either order: '0.5*B88' or 'B88*0.5'); 'E_' → 'E-' exponent fixup
#   'RSH(alpha;beta;omega)'  → assign_omega(omega, fac*(alpha+beta), fac*alpha)
#   'HF'                     → hyb[0]+=fac; hyb[1]+=fac
#   'SR_HF(omega)' / 'LR_HF(omega)'  → assign_omega(...)
#   key.isdigit()            → raw libxc integer ID
#   comma split: X part before ',', C part after; compound names expand both X and C
#   '0.5*b3lyp' scales the compound as a unit
```

### Common Operation 2: nr_rks signature (DFT-10 contract)
```python
# Source: pyscf/dft/numint.py:1074 — port signature to pyscf-dft/src/numint.rs
def nr_rks(ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
           max_memory=2000, verbose=None):
    # returns (nelec, excsum, vmat)
# Companions: nr_uks (1192), eval_rho (116), eval_ao (51 → eval_gto), nr_nlc_vxc (1347)
# NumInt class: numint.py:2835; hybrid_coeff (2731), rsh_coeff (2737), eval_xc (2741)
# block_loop: mem-driven blksize chunking (Claude's discretion — mirror, no hard enforcement)
```

### Common Operation 3: Lebedev SphGenOh generator (DFT-04 port)
```python
# Source: pyscf/dft/LebedevGrid.py:113-... — port to pyscf-grids/src/lebedev.rs
# SphGenOh(code, a, b, v) expands one (a,b,v) seed into Oh-symmetric points:
#   code 0: (0,0,1) etc        →  6 points
#   code 1: (0,a,a) a=1/√2     → 12 points
#   code 2: (a,a,a) a=1/√3     →  8 points
#   code 3: (a,a,b) b=√(1-2a²) → 24 points
#   code 4: (a,b,0) b=√(1-a²)  → 24 points
#   code 5: (a,b,c) c=√(1-a²-b²) → 48 points
# MakeAngularGrid_N() supplies the (code, a, b, v) seed tuples per order N.
# LEBEDEV_ORDER (4999-5033): {order → ngrid}, e.g. {3:6, 5:14, ... 131:5810}
```

### Common Operation 4: VV10 inner kernel (DFT-06 port)
```python
# Source: pyscf/dft/numint.py:526-537 (the commented pure-Python reference) — port to pyscf-dft/src/vv10.rs
# Constants: Kvv=Bvv*1.5*π*(9π)^(-1/6); Beta=((3/Bvv²)^0.75)/32; defaults Bvv=5.9, Cvv=0.0093
# Per outer-grid point i, double-loop over inner (vv) grid:
#   R2 = |vvcoords - coords[i]|²
#   gp = R2*W0p + Kp;  g = R2*W0[i] + K[i];  gt = g + gp
#   T  = RpW/(g*gp*gt);  F = -1.5*Σ T
#   U  = Σ T*(1/g+1/gt);  W = Σ T*(1/g+1/gt)*R2
# Embarrassingly parallel over i; use oracle_sum for the inner reductions (bit-exact).
```

## State of the Art

| Old Approach | Current Approach | When Changed | Impact |
|--------------|------------------|--------------|--------|
| libcint/libdft/libxc C extensions linked at install | Pure-Rust sibling crates (cintx/libxc_rs/xcfun_rs) | This project | No C/CMake at install (core value prop). |
| libxc_rs: all 266 kernels unconditional | D-04 per-functional `[features]` gate | Phase 4 (this) | Gated build minutes vs ~6h. |
| Becke partition via C `VXCgen_grid` | Port pure-Python partition to Rust | Phase 4 | Parameterized grids + algebra-wall reductions. |

**Deprecated/outdated:**
- The "5047-line Lebedev table" framing — it is a generator, not a static table.
- The cubecl 0.9-era `ArrayArg::from_raw_parts` turbofish sketch — Phase 2 confirmed the 0.10.0 signature is `(Handle, usize)` by value.

## Assumptions Log

| # | Claim | Section | Risk if Wrong |
|---|-------|---------|---------------|
| A1 | VV10 defaults Bvv=5.9, Cvv=0.0093 | Code Examples / DFT-06 | LOW — confirmed by web + matches PySCF docs; but the *authoritative* source is `libxc_rs` `nlc_coeff()` per functional (the parser should query it, not hardcode, except for the bare `'VV10'` default). Verify against `libxc_rs::NlcCoefficients` at plan time. |
| A2 | `tracing`/`thiserror` already workspace deps at the versions assumed | Standard Stack | LOW — pre-existing; planner confirms exact versions from workspace `Cargo.toml`. |
| A3 | libxc_rs `BatchEvaluator`/`*Output::with_order` exact method names | Pattern 2 | MEDIUM — API shapes read from source headers this session, but exact constructor/field names should be re-confirmed against `src/api/batch.rs` + `src/output/*` when writing the libxc backend (do not compile; read only). |
| A4 | xcfun_rs `Vars`/`Mode` enum variant names (e.g. `AB_GGA`, `Polarized`) | Pattern 3 | MEDIUM — `eval_setup(Vars, Mode, order)` signature confirmed; exact variant spelling from `xcfun-core/src/enums.rs` to be confirmed at plan time. |
| A5 | cintx exposes a safe-API range-coulomb/env[8] setter | DFT-05 / Pattern 4 | MEDIUM-HIGH — cintx *reads* env[8] (verified), and `pyscf-gto::make_env` builds `_env`, but a clean safe-API omega setter was NOT found in cintx-rs/cintx-ops src (only low-level compat/raw references env[8]). DFT-05 may need a small env-slot mutation on the pyscf-gto intor path, or a cintx safe-API gap-closure. **Planner must scope this explicitly.** |

## Open Questions (RESOLVED)

1. **Does cintx need a safe-API range-coulomb gap-closure for DFT-05?**
   - What we know: cintx reads `PTR_RANGE_OMEGA = env[8]` and computes `erf(ωr)/r`; pyscf-gto builds `_env`.
   - What's unclear: whether the safe `mol.intor` path in pyscf-gto can set/restore env[8] today, or whether a cintx-side `with_range_coulomb`-equivalent is needed.
   - Recommendation: Plan a small env-slot setter in `pyscf-gto::intor` (set env[8]=±omega around the int2e call). If cintx's safe API blocks env mutation, file a cintx gap-closure (cintx#11-style) and sequence DFT-05 behind it. Low effort either way — the omega slot is a single f64 in `_env`.
   - RESOLVED: Plan 04-07 implemented `OmegaGuard` RAII in `pyscf-gto/src/range_coulomb.rs` — a safe API that sets/restores env[8] around intor calls. Tests `omega_guard_sets_and_restores` and `omega_restored_on_error_path` pass. DFT-05 is structurally complete; bit-exact RSH energy is CI-only pending cintx#11 (deferred per REQUIREMENTS.md).

2. **Which corpus functionals seed the D-04 `libxc` feature subset?**
   - What we know: CONTEXT names SVWN (`lda_x`,`lda_c_vwn`), PBE (`gga_x_pbe`,`gga_c_pbe`), B3LYP (`gga_x_b88`,`gga_c_lyp`,`hyb_gga_xc_b3lyp`), CAM-B3LYP (`hyb_gga_xc_cam_b3lyp`), a meta-GGA (TPSS/SCAN), VV10 deps.
   - What's unclear: the exact compound-functional kernel dependency closure (e.g. b3lyp internally mixes LDA/B88/LYP/VWN — all must be in the feature set).
   - Recommendation: Planner finalizes from the test corpus; each compound's component IDs must be enumerated by reading `libxc_rs` `generated_hybrid.rs`/`HybridTerm` metadata for the transitive closure.
   - RESOLVED: Per PENDING_LIBXC_RS_FEATURE_GATE decision (04-02 SUMMARY), the per-functional libxc feature gate is a user-decision item, not a v1 gate. The libxc CI job ships disabled (`if: false`). The xcfun default path is verified. D-04 coordination deferred to post-milestone.

3. **Does VV10 ship in the core RKS plan or a follow-on?** (Claude's discretion, D-07/CONTEXT)
   - Recommendation: follow-on plan after core RKS bit-exact lands — it needs a second `pyscf-grids` instance (`nlcgrids`) and the double-loop port; it is orthogonal to the SVWN/PBE/B3LYP headline.
   - RESOLVED: VV10 shipped in Plan 04-07 as a follow-on plan. `vv10_nlc_runs_end_to_end_over_coarser_nlcgrids` test passes. Bit-exact VV10 energy is CI-only (deferred per Phase-2 ERI gap).

## Environment Availability

| Dependency | Required By | Available | Version | Fallback |
|------------|------------|-----------|---------|----------|
| Rust toolchain (MSRV 1.92, edition 2024) | all crates | assumed ✓ (Phase 1-3 shipped) | — | — |
| cubecl-cpu `=0.10.0` | eval_gto, xcfun_rs default | ✓ (Phase 2 proven) | 0.10.0 | — |
| `libxc_rs` sibling repo | DFT-03 libxc path (feature-gated) | ✓ at `~/Documents/workspace/libxc_rs` | v7.0.0 | xcfun_rs default backend when `libxc` off |
| `xcfun_rs` sibling repo | default XC backend | ✓ at `~/Documents/workspace/xcfun_rs` | workspace | — |
| `cintx` sibling repo | int2e/RSH, eval_gto data | ✓ (Phase 2 wired) | — | — |
| HDF5 (hdf5-metno static) | chkfile | ✓ (Phase 3) | — | — |
| upstream PySCF (in-process oracle) | bit-exact fixtures | ✓ (dev-dep, Phase 1 ORACLE-01) | repo `pyscf/` | — |
| WGPU `shader-f64` device | DFT-11 honesty CI | ✗ on most CI runners | — | **intended:** CPU fallback with warning is the tested path |

**Missing dependencies with no fallback:** None — all compute paths have a CPU default.
**Missing dependencies with fallback:** `shader-f64` WGPU device — the *absence* is the DFT-11 test case (fallback-to-CPU-with-warning), not a blocker. **Never compile libxc_rs locally** (6h freeze) — it is research-read-only; the gated build runs only in the dedicated cached CI job.

## Validation Architecture

### Test Framework
| Property | Value |
|----------|-------|
| Framework | Rust `cargo test` + `pyscf-oracle` (`pyo3::Python::with_gil` driving upstream PySCF in-process, dev-dep only) + `oracle_check!` macro (ORACLE-02, Phase 3) |
| Config file | workspace `Cargo.toml` profiles; `[profile.release-oracle]` (FMA-free, Phase 1 FOUND-05) |
| Quick run command | `cargo test -p pyscf-grids` / `cargo test -p pyscf-dft` (CPU/xcfun default — **never with `--features libxc`**) |
| Full suite command | `cargo test --profile release-oracle -p pyscf-dft -p pyscf-grids` (CPU); the libxc bit-exact job: dedicated cached CI only |

### Phase Requirements → Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| DFT-04/09 | Grid points+weights byte-for-byte vs `gen_grid.py` for level 0..9 on corpus | oracle byte-compare | `cargo test -p pyscf-grids grid_weights_level_sweep` | ❌ Wave 0 |
| DFT-02 | XC-string parser parity vs `libxc.py:parse_xc` (single/comma/shorthand/weights/alias) | unit parity | `cargo test -p pyscf-dft parse_xc_parity` | ❌ Wave 0 |
| DFT-03 | libxc routes to libxc_rs / xcfun to xcfun_rs; bit-identical to C | oracle (gated) | `cargo test --features libxc -p pyscf-dft xc_eval_bitexact` (CI only) | ❌ Wave 0 |
| DFT-03 | ~100-functional libxc smoke (corpus subset) | smoke (gated) | `cargo test --features libxc -p pyscf-dft libxc_functional_smoke` (CI only) | ❌ Wave 0 |
| DFT-01 | RKS/UKS total energy ≤1 µHartree (SVWN/PBE/B3LYP) under release-oracle | oracle energy | `cargo test --profile release-oracle -p pyscf-dft rks_uks_bitexact` | ❌ Wave 0 |
| DFT-05 | CAM-B3LYP / H2O RSH parity fixture | oracle energy | `cargo test --features libxc -p pyscf-dft cam_b3lyp_h2o_rsh` (CI) | ❌ Wave 0 |
| DFT-06 | VV10 energy match (e.g. wB97X-V or explicit nlc='VV10') | oracle energy | `cargo test -p pyscf-dft vv10_energy_match` | ❌ Wave 0 |
| DFT-07 | DF-DFT `dft.RKS(mol).density_fit()` matches upstream | oracle energy | `cargo test -p pyscf-dft df_dft_match` | ❌ Wave 0 |
| DFT-08 | Subclass `get_veff` + `define_xc_` overrides invoked every cycle | PyO3 dispatch assertion | `pytest python/tests/test_dft_override.py` (maturin build) | ❌ Wave 0 |
| DFT-10 | `NumInt` signatures (`eval_xc`/`eval_rho`/`nr_rks`/`nr_uks`) match upstream | API/signature + numeric | `cargo test -p pyscf-dft numint_signatures` | ❌ Wave 0 |
| DFT-11 | wgpu→CPU fallback with warning on `shader-f64`-less device | CI job (special runner) | dedicated `wgpu-no-f64-fallback` CI job runs `dft.RKS(mol).run()`, asserts warning + CPU-correct | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cargo test -p <crate>` (CPU/xcfun default; sub-30s where possible)
- **Per wave merge:** `cargo test --profile release-oracle -p pyscf-dft -p pyscf-grids` (CPU)
- **Phase gate:** full suite green + the dedicated `--features libxc` bit-exact CI job green (heavily cached; the only place the gated libxc build runs) + the `wgpu-no-f64` fallback job green, before `/gsd:verify-work`

### Wave 0 Gaps
- [ ] `crates/pyscf-grids/tests/grid_weights_level_sweep.rs` — byte-for-byte vs upstream, level 0..9, corpus (DFT-04/09)
- [ ] `crates/pyscf-dft/tests/parse_xc_parity.rs` — vs `libxc.py:parse_xc` (DFT-02)
- [ ] `crates/pyscf-dft/tests/rks_uks_bitexact.rs` — SVWN/PBE/B3LYP ≤1µHa (DFT-01)
- [ ] `crates/pyscf-dft/tests/xc_eval_bitexact.rs` + `libxc_functional_smoke.rs` — gated, CI-only (DFT-03)
- [ ] `crates/pyscf-dft/tests/cam_b3lyp_h2o_rsh.rs` (DFT-05), `vv10_energy_match.rs` (DFT-06), `df_dft_match.rs` (DFT-07)
- [ ] `crates/pyscf-dft/tests/numint_signatures.rs` (DFT-10)
- [ ] `python/tests/test_dft_override.py` — subclass override assertion (DFT-08)
- [ ] Oracle fixtures: upstream PySCF energies/grids generated under matched threading (`RAYON_NUM_THREADS=1`, `lib.num_threads(1)`, release-oracle) — extend the Phase 3 fixture corpus with DFT cases
- [ ] CI jobs: `--features libxc` DFT bit-exact (cached); `wgpu-no-f64-fallback` (special/emulated device); re-enable libxc_rs in `nightly-cross-crate.yml`
- [ ] Framework: reuse Phase 1/3 `oracle_check!` + `pyscf-oracle`; no new framework install

## Sources

### Primary (HIGH confidence — read directly this session)
- `pyscf/dft/libxc.py` (parse_xc 491-718, XC_CODES/XC_ALIAS 154/217, rsh_coeff 431, nlc_coeff 416, hybrid_coeff 406) — DFT-02/03 port + parser-parity target
- `pyscf/dft/gen_grid.py` (Grids class defaults 490-499, get_partition 271-345 incl. C-kernel + pure-Python fallback, gen_atomic_grids 184-268, RAD_GRIDS/ANG_ORDER 672-699) — DFT-04/09 byte-exact target
- `pyscf/dft/radi.py` (treutler_ahlrichs 138, gauss_chebyshev 103, becke/treutler atomic_radii_adjust 159/180, BRAGG/COVALENT/SG1RADII) — radial scheme port
- `pyscf/dft/LebedevGrid.py` (SphGenOh 113-, MakeAngularGrid_N, LEBEDEV_ORDER 4999-5033) — Lebedev generator port (D-06)
- `pyscf/dft/numint.py` (nr_rks 1074, nr_uks 1192, eval_rho 116, NumInt 2835, _vv10nlc 471-545, nr_nlc_vxc 1347) — DFT-10 signatures + DFT-06 VV10 port
- `pyscf/dft/rks.py` (get_veff 37-140 incl. RSH branch 108-129, define_xc_ 262, energy_elec 226) — DFT-01/05/08
- `~/Documents/workspace/libxc_rs/{Cargo.toml, src/lib.rs, src/registry/mod.rs, src/api/builder.rs, src/eval/dispatch.rs, src/kernel/lda.rs, src/model/lda_functional.rs, src/input/mod.rs}` — XcBackend libxc path + D-04 gap-closure shape (266-dep + enum-dispatch verified)
- `~/Documents/workspace/xcfun_rs/crates/xcfun-rs/{Cargo.toml, src/functional.rs}` — default backend API + feature-gating model for D-04
- `~/Documents/workspace/cintx/.planning/{STATE.md, research/PITFALLS.md}` — RSH env[8]/PTR_RANGE_OMEGA mechanism (DFT-05 clarification)
- `crates/pyscf-kernels/src/eval_gto.rs` — l≥1 deferral confirmed (Phase 4 lands variants)
- `xtask/src/bin/check_dependency_wall.rs` — ALLOWED_CRATES carve-out (Pitfall 2 clarification)
- `.planning/{REQUIREMENTS.md, ROADMAP.md, STATE.md, config.json}` — DFT-01..11, Pitfall-to-phase mapping, nyquist_validation=true, commit_docs=true

### Secondary (MEDIUM confidence)
- PySCF DFT user docs (pyscf.org/user/dft.html) — VV10/nlc usage, b3lyp routing
- VV10 b=5.9/C=0.0093 defaults — web-confirmed + PySCF docs (authoritative source remains libxc_rs `nlc_coeff()`)

### Tertiary (LOW confidence)
- None relied upon for actionable claims.

## Metadata

**Confidence breakdown:**
- Standard stack: HIGH — every crate's API surface read from source this session; locked by D-01..07.
- Architecture (grid loop, XcBackend seam, RSH, VV10): HIGH — get_veff/numint/gen_grid read directly; the C-kernel-vs-pure-Python distinction verified in source.
- Pitfalls: HIGH — RSH-symbol and dependency-wall-carve-out pitfalls caught by reading the actual cintx notes and the lint source (both contradict CONTEXT shorthand).
- D-04 libxc_rs gap-closure shape: HIGH — the 266-dep, no-features, enum-dispatch structure read from libxc_rs source; mechanical gating path identified.
- Exact sibling-crate method/enum spellings (A3/A4): MEDIUM — re-confirm against source when writing the backend (read-only; never compile libxc_rs).
- cintx range-coulomb safe-API (A5): MEDIUM-HIGH — mechanism verified, safe-API setter location is the one genuine open question.

**Research date:** 2026-05-22
**Valid until:** 2026-06-21 (30 days — sibling crates are pinned path-deps; upstream PySCF source in-repo is static; low churn risk)

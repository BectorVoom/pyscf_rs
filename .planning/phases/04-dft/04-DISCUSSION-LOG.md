# Phase 4: DFT - Discussion Log

> **Audit trail only.** Do not use as input to planning, research, or execution agents.
> Decisions are captured in CONTEXT.md — this log preserves the alternatives considered.

**Date:** 2026-05-22
**Phase:** 04-dft
**Areas discussed:** XC backend routing, libxc compile cost, Grid crate & data, NumInt hot path

---

## XC backend routing

### Q1 — Parser & backend-selection structure

| Option | Description | Selected |
|--------|-------------|----------|
| Mirror upstream two-parser | Port `libxc.py:parse_xc` (default → libxc_rs) AND `xcfun.py:parse_xc` (alternate → xcfun_rs); `mf._numint.libxc = dft.xcfun` swap preserved. Max fidelity + bit-exact. | ✓ |
| Unified parser, backend tag | One pyscf-dft parser resolving any name to (backend, id/weights, family). One code path, less upstream-faithful; risks parser-parity drift (DFT-02). | |
| libxc-only for v1 | Ship only the libxc_rs route; xcfun stub. Defers DFT-03's xcfun half. | |

**User's choice:** Mirror upstream two-parser
**Notes:** The bit-exact contract pins it — upstream `dft.RKS(mol,'b3lyp')` evaluates via libxc, so b3lyp must route to libxc_rs (xcfun would match xcfun's different numbers). DFT-03 encodes the split.

### Q2 — `define_xc_` form scope for v1

| Option | Description | Selected |
|--------|-------------|----------|
| String form in v1, callable deferred | Ship the string-recombination form (reuses the parser); stub the Python-callable form. Grid loop stays Rust/detached; DFT-08 still proven via get_veff + define_xc_ string. | ✓ |
| Both forms in v1 | Support the Python-callable eval_xc too (per-block GIL callback in the grid loop). Full parity, breaks the Python::detach seam for custom functionals, adds python3.13t deadlock surface. | |
| You decide | Defer to research/planning. | |

**User's choice:** String form in v1, callable deferred
**Notes:** Keeps the hot grid loop fully Rust/detached; the per-block Python-callback architecture lands later if demanded.

---

## libxc compile cost

### Q1 — Gating strategy for the ~6h/266-crate compile

| Option | Description | Selected |
|--------|-------------|----------|
| Off-by-default `libxc` feature | Default `cargo build --workspace` excludes libxc_rs; XcBackend seam returns "not enabled"; xcfun_rs is the default backend. `--features libxc` for the bit-exact CI job + wheel. Mirrors Phase 1 `gpu`-off default. | ✓ |
| Always-on, rely on caching | libxc_rs always in the graph; lean on incremental/CI caching. Every clean build + contributor first build eats the ~6h hit. | |
| Separate DFT-libxc workspace member | Isolate the libxc path in a non-default-members crate. Gate at the workspace-member level instead of a cargo feature. | |

**User's choice:** Off-by-default `libxc` feature
**Notes:** Re-enables the commented-out `libxc_rs` `[patch.crates-io]` (Cargo.toml:94) behind the feature. Hard constraint from user memory: never trigger a default-build compile that pulls libxc_rs's 266 kernels.

### Q2 — Handling the full 266-crate compile when the feature IS on

| Option | Description | Selected |
|--------|-------------|----------|
| Add per-functional features to libxc_rs | Cross-crate coordination task: add `[features]` + cfg-gated dispatch to libxc_rs so pyscf-rs pulls only the corpus subset; gated build stays in minutes. cintx-ECP-style gap-closure. | ✓ |
| Accept full 266 compile when enabled | No libxc_rs changes; gated CI + wheel eat the ~6h compile, run infrequently with aggressive caching. | |
| You decide | Defer to research/planning. | |

**User's choice:** Add per-functional features to libxc_rs
**Notes:** Verified finding — libxc_rs's main crate unconditionally path-deps all 266 `libxc-kernel-*` crates with no `[features]` block and no cfg-gated dispatch (read `~/Documents/workspace/libxc_rs/Cargo.toml` + `src/eval/`, `src/kernel/`). So the off-by-default gate alone doesn't make the gated build tractable; the libxc_rs feature work is a real prerequisite, sequenced like the Phase 2 cintx-ECP coordination.

---

## Grid crate & data

### Q1 — Grid generation crate placement

| Option | Description | Selected |
|--------|-------------|----------|
| Own `pyscf-grids` crate | New member (18→19), deps pyscf-core + pyscf-algebra. Matches Phase 3 split-out-shared-concerns pattern; reused by VV10 nlcgrids + Phase 7 grad. | ✓ |
| Module inside pyscf-dft | Matches upstream `gen_grid.py`/`radi.py` placement exactly (strict sibling-fidelity); no new crate. | |

**User's choice:** Own `pyscf-grids` crate
**Notes:** Intentional sibling-fidelity deviation justified by cross-phase reuse + the Phase 3 precedent (diis/df/chkfile split out). ROADMAP needs the 18→19 member update.

### Q2 — Grid-data sourcing for byte-for-byte parity

| Option | Description | Selected |
|--------|-------------|----------|
| Port generator + formulas to Rust | Port `gen_oh`/`MakeAngularGrid` + Becke/Treutler/Gauss-Chebyshev radial, with small const seed/radii tables. Byte-exact via deterministic construction under release-oracle. No file reads, no big tables, no compile freeze. | ✓ |
| Snapshot generated grids as data | Generate once, bundle as an asset, CI drift-check. Sidesteps FP-reproduction risk; adds bundled data + drift check. | |
| You decide | Port first, fall back to snapshot only if bit-exactness proves intractable. | |

**User's choice:** Port generator + formulas to Rust
**Notes:** Key finding — `LebedevGrid.py` is an algorithmic generator (not a static table) and `radi.py` is formula-based, so byte-for-byte comes from porting the generator, not snapshotting 5047 lines. Matches Phase 2's "port the reference algorithm" approach. Snapshot is the documented fallback only if the port can't be made bit-exact.

---

## NumInt hot path

### Q1 — Where the nr_rks/nr_uks grid-integration compute lives

| Option | Description | Selected |
|--------|-------------|----------|
| Orchestrate in pyscf-dft via algebra | Reuse eval_gto kernel for AO-on-grid; ρ/Vxc contractions via pyscf-algebra gemm; XC via the crates. No new bespoke kernels; defer fused cubecl XC-grid kernel to Phase 8. Mirrors Phase 3 pyscf-df D-10. | ✓ |
| Bespoke cubecl kernels now | Write fused DFT grid kernels up front. Faster sooner; more bit-exact kernel code + host round-trip for libxc anyway; front-loads Phase 8's optimization. | |
| You decide | Profile the algebra path first, add kernels only if it's the bottleneck. | |

**User's choice:** Orchestrate in pyscf-dft via algebra
**Notes:** Respects the algebra wall + no-premature-optimization; the 2–5× target (Phase 8) owns fused kernels. libxc_rs is CPU-host-only, so a GPU grid path would force host round-trips for libxc XC — a Phase 8 concern.

---

## Claude's Discretion

- Grid blksize / chunking — mirror `numint.py` MEM-driven blksize; log `PYSCF_MAX_MEMORY` at entry, no hard enforcement (Phase 6 owns that).
- libxc family dispatch — query `libxc_rs` `FunctionalMeta.family()`/`.kind()` to size ρ derivatives + pick `LdaInput`/`GgaInput`/`MggaInput`.
- Hybrid/RSH coefficient surface — `NumInt` exposes `hybrid_coeff`/`rsh_coeff`; KS `get_veff` reuses Phase 3 `get_jk`; RSH K via cintx `int2e_lr_*`/`int2e_sr_*`.
- DFT-11 WGPU f64 honesty — delegate XC-eval fallback to `xcfun_rs`'s `shader-f64`/ERF `auto_backend`; pipeline-level wgpu→CPU reuses Phase 1 `PYSCF_BACKEND` + `tracing::warn`.
- VV10 (DFT-06) — `mf.nlc='VV10'` + separate `nlcgrids` (second pyscf-grids instance); port `nr_rks_vv10`.
- DF-DFT (DFT-07) — reuse `pyscf-df::DfIntegrals` (Phase 3 D-10), no new DF crate.
- `KsResult` chkfile — `impl Checkpointable` (Phase 3 D-06 pattern).
- `KsOverrideHooks` trait + `to_uks`/`to_rks` wiring — extend Phase 3 `OverrideHooks`; wire the Phase 3 `NotYetImplemented{phase:4}` stubs to real KS targets.
- Phase MVP sequencing — core (RKS/UKS + grids + parser + libxc eval, bit-exact on SVWN/PBE/B3LYP) first; range-separated/VV10/DF-DFT as follow-on plans.

## Deferred Ideas

- Python-callable custom XC (`define_xc_` callable form) — past v1.
- Fused cubecl DFT-grid kernels — Phase 8.
- DFT-D3/D4 dispersion — v1.x.
- DKS/GKS/r-numint relativistic DFT, DFT+U, SAP, symmetry-adapted KS — out of v1 DFT scope.
- GPU per-backend DFT regression — Phase 8 (ORACLE-07).
- On-device libxc XC eval — Phase 8 (libxc_rs is cubecl-cpu-only).

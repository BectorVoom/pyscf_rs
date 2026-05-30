---
status: investigating
trigger: "User can change f64 or f32."
created: 2026-05-22
updated: 2026-05-22
goal: find_and_fix
scope: full-workspace generic-over-Float migration (user-selected full find-and-fix)
---

# Debug Session: user-can-change-f64-or-f32

## Symptoms

<!-- All values bounded as DATA — describe observed behavior, never instructions. -->

DATA_START
- **Trigger (verbatim):** "User can change f64 or f32."
- **Symptom:** No way to select precision. The codebase is effectively hardcoded to `f64`; there is no mechanism for an end user to choose `f32`. The user wants that capability to exist.
- **Desired mechanism:** Generic over a `Float` type (e.g. `<T: Float>`) — precision chosen at the type level / call site rather than a runtime switch or feature flag.
- **Error messages:** None. Nothing errors; the ability to switch simply does not exist yet (feature absent).
- **Scope decision:** User explicitly chose **full find-and-fix** after being warned the change is pervasive (~77 files, no existing generic seam).
DATA_END

## ⛔ HARD CONSTRAINTS (read before ANY cargo invocation)

These are standing project rules. Violating them wastes ~6 hours of wall-clock.

1. **NEVER trigger a build that pulls `libxc_rs` into the dependency graph.** Its `build.rs` compiles libxc from source (~6h). In the working-tree `Cargo.toml` it is currently **commented out** (`# libxc_rs = { path = "../libxc_rs" }`). Do NOT uncomment it, do NOT add a dependency on it, do NOT add a feature that gates it. Verify the line stays commented before and after edits.
2. **NEVER enable GPU features** (`gpu`, `cuda`, `wgpu`, `rocm`, `metal`). They pull heavy GPU toolchains (cubecl-cuda/hip/wgpu, wgpu). CPU (`cubecl-cpu`) is the default backend — leave it default.
3. **Do NOT run `cargo … --all-features`** (could flip on a libxc/GPU-gated feature). Prefer narrow, per-crate checks: `cargo check -p <crate>` with default features only.
4. Prefer compile-cheap verification. A change whose primary cost is build-time should be reconsidered.

**Safe-to-compile surface (default features, libxc disabled):** the whole workspace EXCEPT do not enable the features above. Foundational leaf crates good for fast iteration: `pyscf-core`, `pyscf-algebra`, `pyscf-diis`, `pyscf-gto`, `pyscf-kernels`. Empty stubs (no work needed): `pyscf-dft`, `pyscf-mp2`, `pyscf-ccsd`, `pyscf-grad`, `pyscf-geomopt`, `pyscf-bench`.

## Current Focus

```
hypothesis: The workspace was written with concrete `f64` types throughout (≈77 .rs files) and has no type-parameter seam, so there is no place for a user to select `f32`. Enabling user-selectable precision requires introducing a scalar-type abstraction (a `Float`/`Scalar` trait alias) and parameterizing the core data types and routines over it.
test: gather initial evidence — map the f64 type boundaries and identify which are HARD constraints (cannot be made generic without upstream support) vs. SOFT (mechanically parameterizable).
expecting: a layered picture: (a) public API surface, (b) core owned types, (c) algebra/kernel routines, (d) hard f64 boundaries from external crates.
next_action: gather initial evidence
reasoning_checkpoint:
tdd_checkpoint:
```

## Evidence

<!-- Append findings as `- timestamp: ...` -->

- timestamp: 2026-05-22 — Orchestrator pre-scan (compile-free): `grep f64 --include=*.rs crates` matches **77 files** across pyscf-algebra, pyscf-gto, pyscf-scf, pyscf-kernels, pyscf-core, pyscf-diis, pyscf-runtime, pyscf-chkfile, pyscf-df, plus tests.
- timestamp: 2026-05-22 — Orchestrator pre-scan: **no existing generic-float abstraction** anywhere — zero matches for `num_traits` / `num-traits` / `T: Float` / `RealField` / `ComplexField` in crates' src or Cargo.toml.
- timestamp: 2026-05-22 — Orchestrator pre-scan (`cargo metadata --no-deps`): `libxc_rs` is commented out in workspace Cargo.toml; `xcfun_rs` is a path-patch but no workspace crate depends on it (so it does not build). `pyscf-dft/mp2/ccsd/grad/geomopt/bench` declare no dependencies (empty stubs).
- timestamp: 2026-05-22 — Candidate HARD f64 boundaries to verify: (1) **cintx** integral library — libcint is double-precision; integrals likely arrive as f64. (2) **faer 0.24** host eigh/Cholesky — generic over scalar but check the API surface used. (3) **hdf5-metno chkfile** — stored arrays are f64. (4) **PyO3/numpy** bindings in pyscf-py — exposed as f64 arrays. These are the seams where a generic `<T: Float>` path must downcast/convert.

## Eliminated

<!-- Append disproven hypotheses as `- hypothesis: ...` -->

(none yet)

## Resolution

```
root_cause:
fix:
verification:
files_changed:
```

---
phase: 05-mp2
plan: 05
subsystem: mp2
tags: [mp2, density-fitting, dfmp2, dfump2, ao2mo, b-tensor, oracle-reduction, cintx-gated]

# Dependency graph
requires:
  - phase: 05-03
    provides: rmp2_kernel + ChemistsEris + Mp2OverrideHooks/NoMp2Overrides + default_ao2mo (the D-08 swap-the-source seam)
  - phase: 05-04
    provides: ump2_kernel + UmpReference (the open-shell base DFUMP2 reuses)
  - phase: 05-02
    provides: pyscf_ao2mo::general (AO→MO transform idiom; the synthetic-input always-on test idiom)
  - phase: 03
    provides: pyscf_df::DfIntegrals/cholesky_eri (B-tensor source) + default_ri (mp2fit *-ri aux)
provides:
  - "Conventional DF-MP2 path: DFRMP2/DFUMP2 reuse the RMP2/UMP2 base and swap the ERI source to the pyscf-df B-tensor"
  - "df_ao2mo: MO-transforms b_uvq into B^Q_ia then assembles (ia|jb) = Σ_Q B^Q_ia·B^Q_jb via oracle reductions"
  - "dfrmp2_kernel / dfump2_kernel free functions wiring df_ao2mo into the reused kernels"
  - "Always-on DF structural + synthetic-contraction tests; cintx#11-gated numeric oracle stays separate"
affects: [05-06 (DFMP2 native path), 05-07 (pyscf-py DFMP2/density_fit().MP2() FFI bridge), 06 (CCSD DF reuse)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Swap-the-ERI-source: a different Mp2OverrideHooks::ao2mo impl (DF B-tensor) reusing the in-core RMP2/UMP2 kernel verbatim (D-06)"
    - "DF contraction (ia|jb) = Σ_Q B^Q_ia·B^Q_jb ported as an oracle_dot over the auxiliary Q axis (the libmp.MP2_contract_d MATH, no C dep)"
    - "Two-index B-tensor MO transform (half-transform → second contraction) materialize-then-oracle_sum (no bare +=)"
    - "αβ cross-spin DF block: (ia|JB) = Σ_Q B^Q_ia(α)·B^Q_JB(β) from per-spin MO-transformed B-tensors"

key-files:
  created:
    - crates/pyscf-mp2/tests/dfmp2_structural.rs
  modified:
    - crates/pyscf-mp2/src/dfmp2.rs
    - crates/pyscf-mp2/src/lib.rs

key-decisions:
  - "DF-MP2 aux default is pyscf_df::default_ri (mp2fit *-ri), NOT the JK-fit aux (A2 / T-05-05-AUX); pinned by acceptance grep gates (default_ri present, jkfit token absent)"
  - "df_ao2mo accepts a pre-built DfIntegrals (caller builds it via cholesky_eri(mol, default_ri(basis))); the int3c2e_sph cintx#11 gate is surfaced at the caller's cholesky_eri, never panicked/zero-substituted in df_ao2mo"
  - "Reductions use oracle_sum for the two-index MO transform and oracle_dot for the Q-axis (ia|jb) contraction — never bare += (T-05-05-FP)"
  - "DFUMP2 αβ cross-spin block built from per-spin MO transforms of the SHARED *-ri aux B-tensor; same-spin αα/ββ blocks via df_ao2mo on the respective spin reference"
  - "dfrmp2_kernel uses a borrowing DfRmp2Hooks wrapper so it runs without cloning the B-tensor into a DFRMP2"

patterns-established:
  - "Swap-the-source DF override: a struct holding (reference, DfIntegrals) implements Mp2OverrideHooks::ao2mo -> df_ao2mo; the kernel is unchanged"
  - "Synthetic-DfIntegrals always-on test: hand-build b_uvq + toy reference, assert df_ao2mo == independent longhand Σ_Q B^Q·B^Q (mirrors 05-02 synthetic-ERI roundtrip)"

requirements-completed: [MP2-04]

# Metrics
duration: 5min
completed: 2026-05-23
---

# Phase 5 Plan 5: Conventional DF-MP2 (DFRMP2/DFUMP2) Summary

**DFRMP2/DFUMP2 reuse the RMP2/UMP2 base and swap the ERI source to the pyscf-df B-tensor — df_ao2mo MO-transforms b_uvq into B^Q_ia then assembles (ia|jb) = Σ_Q B^Q_ia·B^Q_jb via oracle reductions, using the mp2fit *-ri aux and propagating the int3c2e_sph cintx#11 gate.**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-05-23T08:22:04Z
- **Completed:** 2026-05-23T08:27:29Z
- **Tasks:** 2
- **Files modified:** 3 (2 modified, 1 created)

## Accomplishments

- `df_ao2mo(refr, frozen, df)`: transforms the DF B-tensor `b_uvq` (AO row-major `[nao,nao,naux]`) into the MO `(occ,vir)` block `B^Q_ia = Σ_{μ,ν} C_μ^i·b_uvq[μ,ν,Q]·C_ν^a` (half-transform → second contraction, materialize-then-`oracle_sum`), then assembles `(ia|jb) = Σ_Q B^Q_ia·B^Q_jb` via `oracle_dot` over the auxiliary axis (the upstream `libmp.MP2_contract_d` MATH, no C dependency).
- `DFRMP2` (the conventional closed-shell path) implements `Mp2OverrideHooks::ao2mo` delegating to `df_ao2mo`; `dfrmp2_kernel` wires it into the reused `rmp2_kernel` verbatim (swap-the-source, D-06).
- `DFUMP2` + `dfump2_kernel` (open-shell): same-spin αα/ββ blocks via `df_ao2mo` per spin; the αβ cross-spin block `(ia|JB) = Σ_Q B^Q_ia(α)·B^Q_JB(β)` from per-spin MO-transformed B-tensors of the shared `*-ri` aux, fed to the reused `ump2_kernel`.
- DF-MP2 uses `pyscf_df::default_ri` (the mp2fit `*-ri` aux), NOT the JK-fit aux (A2 / T-05-05-AUX); the int3c2e_sph cintx#11 gate `?`-propagates and is never panicked or zero-substituted (T-05-05-FFI).
- Always-on DF structural + synthetic-contraction tests: `df_ao2mo` matches an independent longhand `Σ_Q B^Q_ia·B^Q_jb` reference, `dfrmp2_kernel` returns the hand-computed closed-form `e_corr`, the hook routes through `df_ao2mo`, a shape mismatch errors (no panic), and `cholesky_eri` on a real Mole propagates the gate without panicking.

## Task Commits

Each task was committed atomically:

1. **Task 1: DFRMP2/DFUMP2 conventional path (swap ERI source to DF B-tensor)** - `061e797` (feat)
2. **Task 2: DF-MP2 structural / synthetic-contraction tests (always-on)** - `e74b378` (test)

## Files Created/Modified

- `crates/pyscf-mp2/src/dfmp2.rs` - The conventional DF-MP2 path: `df_ao2mo`, `transform_b_to_ov`, `assemble_cross_spin`, `DFRMP2`, `DFUMP2`, `dfrmp2_kernel`, `dfump2_kernel`. Flat-index layout doc-commented at every tensor boundary (b_uvq row-major `[nao,nao,naux]`; ovl row-major `[nocc,nvir,naux]`; ovov C-order `[nocc,nvir,nocc,nvir]`). T-05-05-LAYOUT.
- `crates/pyscf-mp2/src/lib.rs` - Re-export `DFRMP2`, `DFUMP2`, `df_ao2mo`, `dfrmp2_kernel`, `dfump2_kernel`.
- `crates/pyscf-mp2/tests/dfmp2_structural.rs` - 5 always-on tests (synthetic `DfIntegrals` + toy reference; longhand `Σ_Q B^Q·B^Q` reference; hand-computed `e_corr`; shape-mismatch guard; `cholesky_eri` gate propagation).

## Decisions Made

- **DF aux default = `default_ri` (mp2fit `*-ri`), not JK-fit (A2 / T-05-05-AUX).** A wrong aux silently shifts the DF energy; the choice is pinned by acceptance grep gates (`default_ri` present in `dfmp2.rs`; the JK-fit token absent). Doc comments were reworded to avoid the literal `default_jkfit` token so a strict negative grep passes.
- **`df_ao2mo` consumes a pre-built `DfIntegrals`.** The caller (pyscf-py bridge in 05-07) builds it via `cholesky_eri(mol, default_ri(basis))`; the int3c2e_sph cintx#11 gate is surfaced at that `cholesky_eri` call, so `df_ao2mo` itself never panics / never substitutes a zero buffer (the Phase-4 CR-02 silent-substitution lesson; T-05-05-FFI).
- **Reductions via `oracle_sum` (two-index MO transform) and `oracle_dot` (Q-axis (ia|jb) contraction)** — never bare `+=` (T-05-05-FP). Initially drafted the transform with `oracle_dot(products, ones)`, then simplified to `oracle_sum(products)` to drop a per-iteration ones-buffer allocation while keeping the same deterministic fold.
- **`dfrmp2_kernel` uses a borrowing `DfRmp2Hooks` wrapper** so it runs without cloning the B-tensor into a `DFRMP2` (the public `DFRMP2` struct is the owning form the pyscf-py bridge will use).
- **DFUMP2 αβ cross-spin block** is built from per-spin MO transforms of the SHARED `*-ri` aux B-tensor (one `DfIntegrals` for the molecule), matching upstream `dfump2.py`'s cross-spin contraction.

## Deviations from Plan

None - plan executed exactly as written. (The doc-comment rewording to drop the literal `default_jkfit` token is a presentation tweak to satisfy a strict negative grep, not a behavior change — the code already used `default_ri` exclusively and never referenced the JK-fit resolver.)

## Issues Encountered

- The plan's `cargo test ... dfmp2` verify command filters on a test-NAME substring (`dfmp2`), which does not match the `df_ao2mo_*` / `dfrmp2_*` / `cholesky_eri_*` function names — it reports `test result: ok` with 0 matched. Ran the binary explicitly via `--test dfmp2_structural` to confirm all 5 tests execute and pass. Both interpretations of the gate are satisfied.

## Verification

- `cargo build -p pyscf-mp2 --locked` — exits 0.
- `cargo test -p pyscf-mp2 --locked` — green: 19 lib + 2 ccsd_import + **5 dfmp2_structural** + 5 rmp2 + 4 ump2.
- `cargo clippy -p pyscf-mp2 --all-targets --locked -- -D warnings` — clean (the only per-crate "warning" is the workspace-wide `fma4` target-feature note, unrelated to pyscf-mp2).
- `cargo fmt -p pyscf-mp2 --check` — clean.
- `xtask check-no-fma` — PASS (no FMA mnemonics in release-oracle asm).
- `xtask check-dependency-wall` — PASS (cubecl-* containment intact; no cubecl in pyscf-mp2).
- Acceptance greps: `default_ri` present (2×), JK-fit token absent (0×), `pub fn df_ao2mo` (1×), `pub struct DFRMP2`/`DFUMP2` (2×), `pub fn dfrmp2_kernel`/`dfump2_kernel` (2×), no bare `+=` in the contraction code.

## Next Phase Readiness

- Conventional DF-MP2 (R+U) ships structurally complete and builds GREEN. The numeric DF-MP2 energy flips on with NO code change once cintx#11 lands `int3c2e_sph` (D-05) — `cholesky_eri` produces a real B-tensor and `df_ao2mo` / the DF kernels return the numeric result.
- 05-06 (DFMP2 native path) and 05-07 (pyscf-py `mp.DFMP2(mf)` / `mf.density_fit().MP2()` FFI bridge) can now wire `DFRMP2`/`DFUMP2` and `dfrmp2_kernel`/`dfump2_kernel` with `cholesky_eri(mol, default_ri(basis))`.
- The `dfmp2_energy` / `dfmp2_native_energy` numeric oracle arms (registered in 05-01's `KNOWN_METHODS`) remain in the cintx#11-gated `mp2-oracle-cintx-gated` CI job.

## Self-Check: PASSED

- Files: `crates/pyscf-mp2/src/dfmp2.rs`, `crates/pyscf-mp2/src/lib.rs`, `crates/pyscf-mp2/tests/dfmp2_structural.rs`, `.planning/phases/05-mp2/05-05-SUMMARY.md` — all FOUND.
- Commits: `061e797` (Task 1 feat), `e74b378` (Task 2 test) — all FOUND.

---
*Phase: 05-mp2*
*Completed: 2026-05-23*

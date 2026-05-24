---
phase: 06-ccsd
plan: 01
subsystem: pyscf-ccsd
tags: [scaffold, ccsd, contracts, dependency-wall, fma-scan]
requires:
  - pyscf-mp2 (the file-for-file analog: error/hooks/reference/lib shapes)
  - pyscf-runtime::BackendError (the D-01 try_reserve pre-flight #[from])
  - pyscf-diis::DiisError (the amplitude-DIIS #[from])
  - pyscf-chkfile (sole hdf5 owner — spill alias, no second hdf5-metno dep)
provides:
  - "pyscf-ccsd: a real compiling crate with the full workspace-internal dep set"
  - "CcsdError (ShapeMismatch + NotYetImplemented{wave} + BackendError/DiisError #[from])"
  - "ChemistsEris (oooo/ovoo/oovv/ovov/ovvo/ovvv/vvvv + fock/mo_energy)"
  - "CcsdOverrideHooks (ao2mo/update_amps/energy/make_rdm1/make_rdm2) + NoCcsdOverrides"
  - "CcsdReference snapshot"
  - "ccsd_kernel / default_ao2mo / default_energy / default_update_amps signatures"
  - "check_no_fma scans pyscf_ccsd symbols"
affects:
  - 06-02..06-11 (every later CCSD wave builds against these fixed contracts)
tech-stack:
  added: []   # NO external packages; workspace-internal crates only
  patterns:
    - "sibling-crate fidelity (mirror pyscf-mp2 file-for-file)"
    - "staged-scaffold stubs returning NotYetImplemented{wave:N}"
    - "module-scope #![allow(dead_code)] on stub modules (03-14 precedent)"
key-files:
  created:
    - crates/pyscf-ccsd/src/error.rs
    - crates/pyscf-ccsd/src/eris.rs
    - crates/pyscf-ccsd/src/hooks.rs
    - crates/pyscf-ccsd/src/reference.rs
    - crates/pyscf-ccsd/src/ccsd.rs
    - crates/pyscf-ccsd/src/rintermediates.rs
    - crates/pyscf-ccsd/src/update_amps.rs
    - crates/pyscf-ccsd/src/uccsd.rs
    - crates/pyscf-ccsd/src/uintermediates.rs
    - crates/pyscf-ccsd/src/diis_amps.rs
    - crates/pyscf-ccsd/src/lambda.rs
    - crates/pyscf-ccsd/src/ulambda.rs
    - crates/pyscf-ccsd/src/rdm.rs
    - crates/pyscf-ccsd/src/urdm.rs
    - crates/pyscf-ccsd/src/diagnostics.rs
    - crates/pyscf-ccsd/src/dfccsd.rs
    - crates/pyscf-ccsd/src/direct.rs
  modified:
    - crates/pyscf-ccsd/Cargo.toml
    - crates/pyscf-ccsd/src/lib.rs
    - xtask/src/bin/check_no_fma.rs
decisions:
  - "Internal crates wired via { path = \"../...\" } (the proven MP2 idiom), NOT { workspace = true } — the pyscf-* members are not registered as [workspace.dependencies]."
  - "ChemistsEris re-exported from eris module (not hooks) — it carries far more blocks than MP2's single-ovov hooks struct."
  - "update_amps hook uses pyscf_core::Amplitudes for now; Wave 2's opaque-Tensor (D-01) upgrade flows through the same signature."
  - "check_dependency_wall.rs left UNMODIFIED (denylist already covers the cubecl-free crate — the CONTEXT.md 'extend allowlist' phrasing was the inaccurate one per RESEARCH Pitfall 3)."
metrics:
  duration: 5min
  tasks: 2
  files: 20
  completed: 2026-05-24
---

# Phase 6 Plan 01: pyscf-ccsd Crate Scaffold Summary

Filled the 5-line `pyscf-ccsd` stub into a real, compiling crate: the full workspace-internal dependency set (no pyo3/cubecl/hdf5-metno/libxc direct dep), the 17-module skeleton mirroring upstream `pyscf/cc/*.py`, and the four load-bearing contract types (`CcsdError`, `ChemistsEris`, `CcsdOverrideHooks`+`NoCcsdOverrides`, `CcsdReference`) every later wave consumes — plus `pyscf-ccsd` wired into `check_no_fma` SCAN_TARGETS so its own symbols are FMA-checked under `release-oracle`.

## What Was Built

**Task 1 — Cargo deps + 17-module skeleton + 4 contract types (commit `4a13a10`):**
- `Cargo.toml`: wired `pyscf-core`/`pyscf-algebra`/`pyscf-ao2mo`/`pyscf-mp2`/`pyscf-scf`/`pyscf-df`/`pyscf-diis`/`pyscf-chkfile`/`pyscf-gto`/`pyscf-runtime` + `thiserror`/`tracing` + dev `approx`. No pyo3 (D-09), no cubecl-* (algebra wall), no hdf5-metno (D-07).
- `lib.rs`: `#![forbid(unsafe_code)]` + `#![warn(clippy::unwrap_used)]`, 17 `pub mod`, flat re-exports of `CcsdError`, `ChemistsEris`, `CcsdOverrideHooks`, `NoCcsdOverrides`, `CcsdReference`, `CcsdResult`, `ccsd_kernel`, `default_ao2mo`, `default_energy`, `default_update_amps`.
- `error.rs`: `CcsdError` with `ShapeMismatch { expected, got }`, `NotYetImplemented { wave: u8 }`, and `#[from]` arms for `AlgebraError`/`CoreError`/`Ao2moError` + the two CCSD additions `pyscf_runtime::BackendError` (D-01 try_reserve) and `pyscf_diis::DiisError`. `From<CcsdError> for PyscfRsError` via `Core(InvalidMolecule(..))`.
- `eris.rs`: `ChemistsEris` with `oooo`/`ovoo`/`oovv`/`ovov`/`ovvo`/`ovvv`/`vvvv` (each flat `Vec<f64>`, flat C-order offset doc-commented per block) + `fock`/`mo_energy` + `nocc`/`nvir`. Port of `_ChemistsERIs` (`ccsd.py:1389`); the `vvvv` doc-comment notes Wave-4 swaps its source (DF/AO-direct).
- `hooks.rs`: `CcsdOverrideHooks` with the D-09 set (`ao2mo`/`update_amps`/`energy`/`make_rdm1`/`make_rdm2`); `energy` default-delegates to `ccsd::default_energy`; `make_rdm1`/`make_rdm2` return `NotYetImplemented{wave:3}`. `NoCcsdOverrides` delegates `ao2mo`→`ccsd::default_ao2mo`, `update_amps`→`update_amps::default_update_amps`.
- `reference.rs`: `CcsdReference` field-for-field mirror of `Mp2Reference`.
- Wave-keyed stubs: `ccsd`/`rintermediates`/`update_amps` (wave 1), `uccsd`/`uintermediates`/`diis_amps` (wave 2), `lambda`/`ulambda`/`rdm`/`urdm`/`diagnostics` (wave 3), `dfccsd`/`direct` (wave 4). `ccsd_kernel`/`default_ao2mo`/`default_energy`/`default_update_amps` declared with bodies returning `NotYetImplemented{wave:1}`.

**Task 2 — check_no_fma SCAN_TARGETS (commit `f261aa2`):**
- Added `("pyscf-ccsd", "pyscf_ccsd")` as the third SCAN_TARGETS entry.
- `check_dependency_wall.rs` left UNMODIFIED (denylist already covers the cubecl-free crate).

## Verification

- `cargo build -p pyscf-ccsd --locked` exits 0; `cargo check -p pyscf-ccsd --locked` exits 0.
- `cargo tree -p pyscf-ccsd --depth 1`: direct deps are EXACTLY the verified 10 internal crates + thiserror + tracing (+ dev approx); NO direct cubecl/hdf5-metno/libxc/pyo3.
- Full `cargo tree -p pyscf-ccsd`: **zero `libxc`, zero `pyo3`** anywhere. The transitive `cubecl` (via pyscf-algebra, the sole legal cubecl owner) and `hdf5-metno` (via pyscf-chkfile, the sole legal hdf5 owner) are the intended carve-out crates — exactly the wall invariant.
- `cargo run -p xtask --bin check-dependency-wall` exits 0 (PASS — cubecl-* containment intact).
- `cargo run -p xtask --bin check-no-fma` exits 0 (PASS — scans 3 asm files incl. pyscf_ccsd, no FMA mnemonics; release-oracle build of the libxc-clean pyscf-ccsd closure finished in 56s, no libxc compile triggered).

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking] Cargo dep form: `{ path = "../..." }` not `{ workspace = true }`**
- **Found during:** Task 1
- **Issue:** The plan (and 06-RESEARCH.md:115-130) sketched the internal deps as `{ workspace = true }`, but the pyscf-* member crates are NOT registered as `[workspace.dependencies]` in the root `Cargo.toml` — only as `members`. A `{ workspace = true }` form would fail to resolve.
- **Fix:** Used `{ path = "../<crate>" }` for all 10 internal deps, mirroring the proven `pyscf-mp2/Cargo.toml` sibling idiom (the exact precedent the plan said to mirror).
- **Files modified:** `crates/pyscf-ccsd/Cargo.toml`
- **Commit:** `4a13a10`

**2. [Rule 1 - Verify-command correction] `cargo tree | grep` expectation + xtask bin names**
- **Found during:** Task 1 & Task 2 verification
- **Issue (a):** The plan's verify expected `cargo tree -p pyscf-ccsd | grep -i 'cubecl|hdf5-metno|libxc|pyo3'` to be empty / print WALL_CLEAN. But pyscf-ccsd legitimately depends on pyscf-algebra (the cubecl owner) and pyscf-chkfile (the hdf5 owner), so the **transitive** tree does show cubecl + hdf5-metno. The real wall invariant is "pyscf-ccsd names no DIRECT forbidden dep" (denylist) + "zero libxc/pyo3 anywhere" — both hold.
- **Issue (b):** The plan's verify used `check_dependency_wall` / `check_no_fma` (underscores); the actual xtask bin targets are `check-dependency-wall` / `check-no-fma` (hyphens).
- **Fix:** Verified the authoritative gate (`check-dependency-wall` PASS) + direct-dep cleanliness + zero libxc/pyo3 in the full tree; ran the guards with their correct hyphenated names. No code change — these were verify-command inaccuracies, resolved by checking the correct invariant.
- **Commit:** n/a (verification only)

## Known Stubs

All 12 non-contract modules are intentional staged scaffolds (the plan's explicit design): bodies return `CcsdError::NotYetImplemented { wave: N }` where N is the wave that fills them (1: ccsd/rintermediates/update_amps; 2: uccsd/uintermediates/diis_amps; 3: lambda/ulambda/rdm/urdm/diagnostics; 4: dfccsd/direct). These are documented in each module's doc-comment with the upstream port target and the filling wave. NOT a defect — the plan's objective is the compiling target + fixed contracts; the math lands in 06-02..06-11. Module-scope `#![allow(dead_code)]` on stub modules keeps the commit `-D warnings`-clean while consumers land in later waves (03-14 precedent); removed per-module as each wave wires its consumer.

## Threat Flags

No new threat surface. Pure scaffolding — no compute, no Python boundary, no file I/O, no untrusted input. The two threat-register entries (T-06-01-SC accept, T-06-01-WALL mitigate) are satisfied: no external package installed (workspace-internal only), and `check-dependency-wall` + the direct-dep `cargo tree` gate prove pyscf-ccsd names no cubecl-*/hdf5-metno/pyo3 direct dep with zero libxc/pyo3 transitively.

## Self-Check: PASSED

All 17 module files + lib.rs + `xtask/src/bin/check_no_fma.rs` + `06-01-SUMMARY.md` exist on disk. Both commits (`4a13a10`, `f261aa2`) found in git log. `cargo check -p pyscf-ccsd --locked` exits 0; both xtask guards PASS.

---
phase: 03-scf-pyo3-bindings
plan: 12
subsystem: scf
gap_closure: true
tags: [df-hf, scf, density-fit, get_jk_df, end-to-end, convergence, SCF-07]

# Dependency graph
requires:
  - phase: 03-scf-pyo3-bindings (03-05)
    provides: RHF::density_fit, DfHooks, pyscf_df::get_jk_df
  - phase: 03-scf-pyo3-bindings (03-11)
    provides: generic SCF kernel<H> loop, default_get_jk, init_guess_by_1e, eig/occ/rdm/energy
  - phase: 05-mp2 (05-08)
    provides: real int2e (arity-4) — default_get_jk now builds a real Fock
  - phase: 05-mp2 (05-09)
    provides: rank-revealing DF-metric fit — cholesky_eri robust for real aux
provides:
  - "DF-HF proven end-to-end in-tree (converges + matches non-DF RHF within DF accuracy)"
  - "Always-on dfhf_end_to_end.rs + un-ignored h2_no_overrides_converges"
affects: [03-13 (minao default), milestone-uat, 06-ccsd (DF-CCSD precedent)]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "DF-HF in pure Rust: cholesky_eri(&mol, aux) -> DfHooks{df:&df} -> kernel(&mol,&hooks,cfg) with init_guess=OneElectron"
    - "In-tree DF correctness via DF-vs-non-DF cross-check (|e_dfhf - e_rhf| < DF accuracy bound) — no upstream PySCF needed"

key-files:
  created:
    - crates/pyscf-scf/tests/dfhf_end_to_end.rs
  modified:
    - crates/pyscf-scf/tests/no_overrides_drives_kernel.rs
    - .planning/REQUIREMENTS.md

key-decisions:
  - "Use the 1e init guess (the default minao is NotYetImplemented until 03-13); do NOT change the default init guess (would diverge from upstream + hurt convergence)."
  - "Assert DF-HF matches non-DF RHF within a DF-accuracy bound (1e-3), NOT bit-exact — DF is an approximation; the sto-3g minimal aux is asserted to converge only, not to match."
  - "kernel_propagates_jk_not_yet_implemented stays valid (default minao still errors); it flips in 03-13 when minao lands."

patterns-established:
  - "Pattern: prove a DF method end-to-end in-tree by cross-checking against its non-DF sibling (DF-HF vs RHF; cf. DF-MP2 vs in-core MP2 in 05-09) — independent of upstream PySCF."

requirements-completed: []  # SCF-07 partial ([~]): end-to-end numeric in-tree; minao default (03-13) + upstream byte-identity remain

# Metrics
duration: ~20min
completed: 2026-05-24
---

# Phase 3 Plan 12: DF-HF End-to-End Lock-In Summary

**Locked in DF-HF (SCF-07) end-to-end: with int2e (05-08) + the rank-revealing
DF-metric fit (05-09), `RHF::density_fit` + `DfHooks` + the SCF kernel converge
to a DF-HF energy matching non-DF RHF within DF accuracy — proven by always-on
in-tree tests.**

## What was built

- **Un-ignored `h2_no_overrides_converges`** — the `int2e_sph` arity-4 gap that
  `#[ignore]`d it is closed (05-08), so plain RHF/H2/STO-3G converges to ≈ -1.117
  via the `1e` guess.
- **New `dfhf_end_to_end.rs`** — builds H2/STO-3G, runs non-DF RHF (reference),
  then DF-HF via `cholesky_eri` + `DfHooks` for weigend + cc-pvdz-jkfit aux:
  asserts converged, finite, and `|e_dfhf - e_rhf| < 1e-3`. Observed:
  **weigend |Δ| = 4.6e-5**, **cc-pvdz-jkfit |Δ| = 2.0e-4** Hartree (the DF
  fitting error). A minimal sto-3g aux is asserted to converge only (poor fit).

## Key result

DF-HF converges and matches non-DF RHF to tens-to-hundreds of µHartree — the
whole DF-HF stack (int3c2e/int2c2e → robust metric fit → `get_jk_df` → SCF loop)
is numerically correct end-to-end.

## Self-Check: PASSED

- `cargo test -p pyscf-scf` green (un-ignored converge test + new dfhf_end_to_end).
- `cargo clippy -p pyscf-scf --tests -- -D warnings` exit 0; `cargo fmt` clean.
- `xtask check-no-fma` PASS; `xtask check-dependency-wall` PASS; 0 libxc.

## Out of scope

The default `minao` init guess (03-13 — this plan uses `1e`); upstream-PySCF
byte-identity (CI-gated/human-verify); UHF/GHF density-fit; DF-DFT (Phase 4).

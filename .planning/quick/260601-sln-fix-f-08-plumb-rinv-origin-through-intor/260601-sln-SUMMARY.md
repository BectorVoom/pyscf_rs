---
phase: quick-260601-sln
plan: 01
subsystem: pyscf-gto / pyscf-grad
tags: [F-08, gradients, intor, iprinv, rinv-origin, cintx]
requires:
  - cintx int1e_iprinv (AllCint1e operator) + ExecutionOptions.rinv_orig
  - cintx PTR_RINV_ORIG validator (Some required for iprinv-named ops)
provides:
  - pyscf_gto::intor_with_rinv_origin
  - pyscf_gto::intor_with_rinv_at_nucleus
  - pyscf-grad hcore_deriv wired to the per-atom rinv origin
affects:
  - crates/pyscf-gto/src/intor.rs
  - crates/pyscf-gto/src/lib.rs
  - crates/pyscf-grad/src/rhf.rs
tech-stack:
  added: []
  patterns:
    - "F-05 ExecutionOptions { rinv_orig: Some(origin), ..Default::default() } idiom (one-line, #[rustfmt::skip] for the grep gate)"
    - "OperatorId resolved by symbol (Resolver::descriptor_by_symbol) — no iprinv const"
    - "single-source arity-2 stitch loop parameterised by ExecutionOptions"
key-files:
  created: []
  modified:
    - crates/pyscf-gto/src/intor.rs
    - crates/pyscf-gto/src/lib.rs
    - crates/pyscf-gto/tests/grad_intor_smoke.rs
    - crates/pyscf-grad/src/rhf.rs
decisions:
  - "Used H2/STO-3G (non-ECP, 2 nuclei) as the translational-invariance oracle fixture — atom_charges() returns true nuclear Z for non-ECP atoms, so Sum_atoms (-Z)*iprinv|@atom == int1e_ipnuc holds exactly (matched to 1e-10)."
  - "Refactored evaluate_arity2 to take ExecutionOptions rather than duplicating the ~90-line stitch body, keeping the plain intor call site on ExecutionOptions::default() and the new entry point on the origin-bearing opts."
  - "Left rhf_verify_fd_numeric #[ignore]'d (full FD assembly needs still-missing int2e_ip1 + int1e_ip{ovlp,kin,nuc}) — its #[ignore] reason string is a deferred-test note, not a flipped assertion; not touched per locked scope."
metrics:
  duration: ~9min
  tasks: 2
  files: 4
  completed: 2026-06-01
---

# Phase quick-260601-sln Plan 01: Plumb rinv origin through intor for the iprinv arm (F-08) Summary

Plumbed a caller-supplied `rinv_origin` (Bohr) through pyscf-gto's `intor` dispatcher for the `int1e_iprinv` family and wired pyscf-grad's `hcore_deriv` to call it per-nucleus — closing the single remaining actionable integral-availability piece of F-08.

## What changed

- **`pyscf_gto::intor_with_rinv_origin(mol, name, rinv_origin)`** — mirrors upstream `with_rinv_origin`. Built-check → suffix-normalise → reject any name whose core ≠ `int1e_iprinv` (the origin is only valid for iprinv) → layout lookup → representation (spinor → Phase 3) → `Resolver::descriptor_by_symbol` (no iprinv const) → builds `#[rustfmt::skip] let opts = ExecutionOptions { rinv_orig: Some(rinv_origin), ..Default::default() };` and runs the shared arity-2 stitch loop. Returns the component-leading `[3, nao, nao]` F-order buffer.
- **`pyscf_gto::intor_with_rinv_at_nucleus(mol, name, atm_id)`** — mirrors `pyscf/grad/rhf.py:121-143 with_rinv_at_nucleus`: bounds-checks `atm_id < mol.natm` (clean error, never OOB) then delegates with `mol.atom_coord(atm_id)`.
- **`evaluate_arity2`** refactored to take an `ExecutionOptions` parameter (single-source stitch); the existing `intor` arity-2 call site passes `ExecutionOptions::default()`.
- **lib.rs** re-exports both new entry points.
- **`hcore_deriv` (pyscf-grad)** now calls `intor_with_rinv_at_nucleus(mol, "int1e_iprinv", atm_id)`; the origin-less `intor(mol, "int1e_iprinv")` call is gone. All downstream math (`vrinv = -Z·iprinv`, the `h1` block add, the symmetrisation) is unchanged.
- Stale "iprinv MISSING from cintx" prose corrected in the grad_intor_smoke header, the new intor.rs entry-point docs, and the `hcore_deriv` doc comment.

## Tests added (all green, no live PySCF / no libxc)

- `int1e_iprinv_with_origin_evaluates_component_leading` — finite `[3, nao, nao]` once an origin is supplied (was: validator error on default None).
- `int1e_iprinv_at_nucleus_matches_explicit_origin` — `with_rinv_at_nucleus(atm_id)` == `with_rinv_origin(atom_coord(atm_id))` elementwise.
- `int1e_iprinv_sum_over_nuclei_equals_ipnuc` — **physics oracle**: `Σ_atoms (-Z_atom)·int1e_iprinv|rinv@atom == int1e_ipnuc` at atol ≤ 1e-10 (against the already-available `int1e_ipnuc`). This is exactly the translational-invariance relation `hcore_deriv` exploits, proving the origin is live and correct.
- `intor_with_rinv_origin_rejects_non_iprinv_name` — a non-iprinv name (`int1e_ovlp`) errors cleanly (T-sln-01 mitigation).

## Threat register dispositions

- **T-sln-01** (non-iprinv name) — mitigated: `core != "int1e_iprinv"` → clean `InvalidMolecule` error; test asserts it.
- **T-sln-02** (atm_id OOB) — mitigated: bounds-check before `atom_coord`, clean error, never panic.
- **T-sln-03** (silent wrong physics) — mitigated: the translational-invariance oracle proves the origin is threaded and correct.
- **T-sln-SC** (installs) — accepted: no new dependencies; pure in-tree wiring over existing `cintx_runtime::ExecutionOptions`.

## Deviations from Plan

None — plan executed exactly as written. (One in-spec judgement call: the plan offered "oracle_sum OR a plain f64 fold"; `pyscf_algebra::oracle_sum` is importable in the pyscf-gto test crate, so the oracle uses it — matching the production `hcore_deriv` pairwise-sum convention.)

## Cargo gate state (logs under `log/`)

- `log/260601-sln-task1-gto-test.log` — `cargo +nightly test -p pyscf-gto --locked --test grad_intor_smoke` → **11 passed; 0 failed; 0 ignored**.
- `log/260601-sln-task2-grad-test.log` — `cargo +nightly test -p pyscf-gto -p pyscf-grad --locked` → **264 passed; 0 failed; 11 ignored** (aggregate). `rhf_verify_fd_numeric` correctly STAYS `#[ignore]`d (full FD assembly is F-08 waves 07-03..07-08).
- `log/260601-sln-fmt.log` — real CI fmt gate `rustfmt --edition 2024 --check` on the 4 touched files → exit 0, no diff.
- `log/260601-sln-clippy.log` — `cargo +nightly clippy -p pyscf-gto -p pyscf-grad --locked` → clean (no lint on any touched source line). libxc was NOT rebuilt (check finished in ~2.6s on cached artifacts).
- libxc-exclusion confirmed before building: `cargo +nightly tree -p pyscf-gto | grep -ci libxc` and `-p pyscf-grad` both returned 0.

## Out of scope (left untouched, per locked scope)

- `get_veff` (int2e_ip1 2e-response), `get_ovlp`, `hcore_generator` de-assembly, `grad_elec` force assembly — deferred to waves 07-03..07-08.
- `rhf_verify_fd_numeric` stays `#[ignore]`d.
- `ecp_engine_stub.rs` WR-01 scalar-path iprinv rejection retained (unrelated).
- Pre-existing pyscf-mp2 fmt drift not touched (out of scope).

## Commits

- `0ac0ef9` feat(quick-260601-sln): F-08 — plumb rinv origin through intor for the iprinv arm
- `9e8b188` feat(quick-260601-sln): F-08 — wire hcore_deriv to per-atom rinv origin

## Self-Check: PASSED

- All 4 modified files exist on disk.
- Both commits `0ac0ef9`, `9e8b188` present in `git log`.
- LIVE `rinv_orig: Some` token present in intor.rs (grep count 4 — not a comment).

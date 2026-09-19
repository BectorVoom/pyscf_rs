# Carryover — production k-point Newton-AH (`.newton()`) absent

**Source:** `.planning/phases/20-pbc-python-bindings/measurements/example-gate-gap-scope.md` §1,
recorded by the 20-18 rollup (`20-VERIFICATION.md`). Blocks `examples/pbc/20-k_points_scf.py:49-65`,
`10-gamma_point_scf.py`, `20-k_points_scf_ksymm.py`, `27-multigrid.py`.

## Measured

- `scf.KRHF(cell, kpts).newton()` → `AttributeError: 'pyscf._native.pbc.scf.KRHF' object has no
  attribute 'newton'`; `scf.newton(mf)` falls through to upstream and fails to import.
- `crates/pyscf-pbc-scf/src/newton_ah.rs` (246 lines) is a dense, real-only test driver:
  `ah_step` (`:73-145`) materialises the full Hessian column by column (dim Σₖ nocc·nvir =
  64·16·16 = 16 384 hops per step, each a JK build on example 20); `f64` only; `NewtonModel`
  (`:203-214`) implemented only by the test `TwoLevel` model; no `gen_g_hop` for KRHF, no CIAH
  Davidson, no complex `expmat`. `KFvind` is `dyn Fn(usize, &[f64])`, real only (`cphf.rs:49`).
- `newton_ah::live_newton_matches_upstream_energy` (T1) panics unconditionally
  ("live arm not yet wired to the KRHF Fock build"; `pbc-oracle-tiers.md` §5c) — KNOWN-RED in
  nightly/full CI.
- Upstream suite: `.newton` unbound on 5 tests + the 15 `scf/test_newton.py` runtime-import tests
  (`upstream-pbc-suite-after-20-19.md`).

## Cost context

Even with Newton + MGGA landed, example 20 (64 k, 8 atoms, 65³) is estimated ≥ 4–6 days upstream
and weeks native (native FFTDF JK 95–102 s vs upstream 13.0 s at 2 k / 65³); it can never be a CI gate.

## Unblock

Complex k-point `gen_g_hop` over the FFTDF/GDF JK response, CIAH Davidson micro-iterations,
complex `expmat`/`rotate_mo`, the `_SecondOrderKRHF` wrapper (`.newton()`, `._scf`, `.kernel`), and
the identity-gate decision for the new class. Gate: `test_newton.py` energies vs vendored 2.12.1.

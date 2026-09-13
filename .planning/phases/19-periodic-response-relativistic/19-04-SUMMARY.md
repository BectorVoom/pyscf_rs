# 19-04 SUMMARY — periodic second-order SCF

**Shipped:** 2026-09-12. `newton_ah` 6 passed + 1 ignored (live arm).

## Driver (`crates/pyscf-pbc-scf/src/newton_ah.rs`)

- `ah_step` (dense augmented-Hessian `[[0,gᵀ],[g,H]]` lowest-eigenvector step
  with trust-region cap + rescale reporting), `NewtonModel` trait
  (gradient/hop/apply/energy — the driver never sees integrals),
  `kernel_newton` (macro cycles to `conv_tol_grad`; count REPORTED, never
  gated). Consumes 19-03's seam shape (matrix-free hop); no second CPHF
  solver (`check-single-cphf` green).
- Correction during development: one AH step is NOT the Newton step (the
  homogeneous constraint damps it — CIAH iterates micro-cycles). The test
  asserts iterated convergence, not one-step exactness.

## Gate: energy, not path (`tests/newton_ah.rs`)

- Analytic 2-level closed-shell model (Hcore + on-site U, Roothaan-damped
  first-order loop vs `kernel_newton`): energies agree to 1e-8 Ha
  (upstream `test_newton.py` asserts 8dp). No assertion references a count.
- AH quadratic micro-convergence, asymmetric-hop refusal, trust-region cap,
  descent, 1-vs-8 determinism.
- Live-cell arm `#[ignore]`d (needs `--release` + DF grids; upstream number
  committed in-test): the production-cell `NewtonModel` over `Krhf` is the
  outstanding wiring, recorded for 19-19.

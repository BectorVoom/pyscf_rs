# Phase 18 implementation checkpoint — 2026-09-12

Update: the upstream A1 failure below was diagnosed as FD energy-reduction
roundoff. The user-authorized harness fix adds an explicit stable-summation
reference mode; all 16 upstream-fixture component tests pass at unchanged
bounds, and four harness regression tests pass. Raw upstream mode still fails
and remains the default. See measurements/gate-a-tiers.md and strain-stable.out.
This resolves the numerical-reference blocker for continuing measurements; it
does not complete 18-16 or Phase 18. The earlier checkpoint below is historical.

Phase 18 is **incomplete**. Code navigation used the CodeGraph MCP server.
Existing planning changes and unrelated worktree changes were preserved.

## Implemented substrate

- Four-field atom shell/AO slices with molecular caller updates.
- Periodic Gradients trait/error surface (method bodies still pending).
- Cell-aware coordinate and strain FD helpers; displaced geometries refresh
  typed basis data, validate finite inputs, and preserve the mesh.
- Tagged density wrapper that builds through the existing make_rdm1 routine.
- Image-weighted integral entry point, retaining the old API as an unweighted
  wrapper. Weighted calls validate input and refuse Hermitian shortcuts.

These are partial 18-02/18-18 results, not completed analytic gradient methods.
The tagged-density test is synthetic rather than a reference SCF solution;
screened contraction and shell-to-atom accessor remain pending. Weighted tests
cover ones and unequal weights, but the dedicated zero-image-selector assertion
specified in the plan still needs adding.

## Measurements

18-01 gradient anchors and FD sweep are recorded in measurements/README.md,
with generating script and raw output. All 14 upstream gradient tests pass.
18-16 stopped on the reproduced upstream A1 failure documented in
measurements/gate-a-tiers.md. No gate was loosened. 18-17 and 18-21 are pending.

## Verified commands

- `cargo test -q -p pyscf-pbc-grad`: pass (2 tagged-density and 4 FD tests).
- `cargo test -q -p pyscf-pbc-gto --test weighted_intor`: 2 passed.
- `cargo test -q -p pyscf-pbc-gto --test pbc_intor`: 11 passed, 3 ignored;
  existing test expectations unchanged.
- `cargo run -q -p xtask --bin check-dependency-wall`: PASS (ALG-06).
- `git diff --check`: pass.

The full workspace and release-oracle determinism gates have not run. Analytic
PP/Ewald gradients, FFTDF derivative J/K, method assemblies/scanners, multigrid
derivatives, stress and geometry optimization remain unimplemented.

## Required next decision

Per 18-CONTEXT's explicit stop-on-gate-failure rule, implementation is paused.
Investigate upstream test_get_vxc_lda's 1.0564055741291156e-9 residual against
its unchanged 1e-9 A1 bound before resuming the remaining measurements/plans.

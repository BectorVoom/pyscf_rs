# 19-18 SUMMARY — sfx2c1e + x2c1e + eph_fd

**Shipped:** 2026-09-12. All four tasks complete.

## Task 1/2 — X2C (`pyscf-pbc-x2c`)

- `xmatrix` (`_x2c1e_xmatrix`), `hcore_fw` (`_get_hcore_fw`), `renorm_r`
  (`_get_r`) — real-symmetric, via `eigh_gen` + `solve_linear`, no iterative
  solver (hence no convergence noise: Gate E at 1e-8).
- Two defects caught by the gates during development (both recorded):
  (a) the `X = Yᵀ` solve decomposed over the wrong axis (columns of `cs`
  instead of rows — same shape, caught by Gate E at 5.4 Ha, fixed);
  (b) the test's own Löwdin comparison folded eigenvector indices into matrix
  indices (caught at −2.55 vs −0.158, fixed to an explicit `S^{-1/2}` build).
- Documented deviation: upstream's `_x2c1e_xmatrix` fallback arm writes
  `cs·clᵀ·m` (`nao×nao` times `2nao×2nao` — cannot execute); this port
  implements the arm's own comment (`X = B·Aᵀ·S` with the `nao×nao` overlap).
- `x2c1e` reuses the same core over the spin-doubled problem (real blocks);
  explicit spin-orbit blocks are refused (`x2c1e_hcore_so`), never silently
  dropped. Complex-Hermitian k-points refused, never truncated to real.

## Task 3 — `eph_fd` (`pyscf-pbc-eph`)

- `central_difference` (refuses zero/negative/nonfinite step + length
  mismatch) and `eph_fd_coupling` at the upstream default `disp = 1e-4`.
- Module doc records the step size and the implied cancellation floor
  (~1e-8 absolute at default step, 18-01 discipline): truncation side
  asserted (quadratic convergence under halving), cancellation side recorded.

## Task 4 — Gate E + separate eph floor (`tests/gate_e.rs` 8/8, `tests/eph_fd.rs` 5/5)

- **Gate E MET**: this port's transform on upstream 2.12.1's own `(t,v,w,s)`
  blocks (H2/`sto-3g` Γ, `light_speed(4)`, committed JSON fixture) matches
  upstream's `h1` to < 1e-8 — plus oracle-free arms (nonrelativistic limit,
  `R → I`, Hermiticity, FW-vs-Dirac-positive-branch spectrum identity,
  `x2c1e == doubled sfx2c1e`, SO refusal, determinism).
- `eph_fd` gated at its own floor (linearity exactness, quadratic rate,
  refusals) — no assertion applies X2C's tolerance to `eph_fd`.

## Verification

- `cargo test -p pyscf-pbc-x2c --test gate_e` → 8 passed.
- `cargo test -p pyscf-pbc-eph --test eph_fd` → 5 passed.
- Determinism 1-vs-8 in-process on both (bit-identical).

# 19-03 SUMMARY — k-aware CPHF `fvind` over the one solver

**Shipped:** 2026-09-12. All three tasks complete; every verification command green.

## Task 1 — the k-aware `fvind` (`crates/pyscf-pbc-scf/src/cphf.rs`)

- `KcphfInput` (per-k `mo_energy`/`mo_occ`/`h1`), `KFvind` (per-k matrix-free
  response operator), `run_kcphf` (one `pyscf_grad::cphf::solve` call per
  k-block, joined in k order), `dense_kvind` (dense-kernel builder for tests
  and explicitly-materialized drivers).
- PBC defaults follow `pbc/scf/cphf.py:29` (`max_cycle = 20`, `tol = 1e-9`,
  `hermi = false`) — NOT the molecular 50-cycle default.
- Design note (documented in-module): upstream performs one stacked Krylov
  solve over the `moloc`-concatenated space; the CPHF `fvind` is k-diagonal
  (Coulomb kernel conserves k; `_get_jk` is called per k at `kshift = 0`), so
  the stacked solve factors into per-k solves of the same equation to the same
  `tol`. No second solver exists: `check-single-cphf` still passes.

## Task 2 — `gen_response` with upstream's shape (`src/response.rs`)

- `resolve_jk_route` ports `_response_functions._get_jk` routing:
  `kshift == 0` → direct; nonzero + `omega != 0` → refused; nonzero on a
  non-fitting backend → refused ("only GDF/RSDF", both as `ResponseError`).
- `PbcKohnShamBase::HAS_GEN_RESPONSE = false` (`rks.py:268`), only
  `RksGenResponse::HAS_GEN_RESPONSE = true` (`:411`). The absence is the
  contract, and the test asserts it.

## Task 3 — Gate B (`tests/cphf_k.rs`, 5 tests)

- Two-k synthetic CPHF vs per-k dense `solve_linear` reference (< 1e-8).
- **Gate B**: (a) invocation-counted `fvind` proves the solver is entered;
  (b) each block is BIT-IDENTICAL to a direct `pyscf_grad::cphf::solve` call —
  delegation, not reimplementation.
- Routing + base-absence + 1-vs-8-thread bit-identity arms.

## Verification

- `cargo test -p pyscf-pbc-scf --test cphf_k` → 5 passed.
- `cargo run -p xtask --bin check-single-cphf` → 0.
- Determinism via in-process 1-vs-8 rayon pools (bit-identical).

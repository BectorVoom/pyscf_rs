---
phase: 09-pbc-foundation
plan: 02
subsystem: algebra
tags: [pbc, complex-algebra, ctensor, zgemm, zeigh, cubecl, determinism, faer]

# Dependency graph
requires:
  - phase: 01-foundation
    provides: "pyscf-algebra real primitives (gemm/axpy/dot/reduce/transpose), oracle_sum/oracle_dot (FOUND-06), AlgebraClient + dispatch_backend! (ALG-06)"
  - phase: 03-scf
    provides: "eigh_gen (real generalized self-adjoint eigh, Löwdin route)"
  - phase: 09-pbc-foundation
    plan: 01
    provides: "PBC crate scaffold + path-scoped lint exemptions"
provides:
  - "CTensor — the planar complex host type (D-PBC-02 / RULE 8)"
  - "zgemm_dense / zgemm_h_dense as FOUR real gemm_dense calls (D-PBC-03)"
  - "zblas: zaxpy/zscal/zdotc/zdotu/zreduce_sum/ztranspose/zhadamard"
  - "zeigh_gen / zcholesky / zsolve_linear, each with two independent routes (D-PBC-04)"
  - "oracle_zsum / oracle_zdot — the ONLY reductions numerical PBC code may use (D-PBC-17)"
  - "pyscf-kernels K-04 zhadamard cubecl kernel (pyscf_kernels::pbc)"
  - "D-PBC-04 RESOLVED: FAER_C64 = true"
affects: [09-03, 09-04, 09-05, 09-06, 09-07, 09-08, 09-09, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "planar complex: every complex op decomposes into existing REAL cubecl primitives; no new numeric type crosses the ALG-06 wall"
    - "two-route numerics: a native faer c64 primary + an independent real-arithmetic cross-check route, compared in tests and in a debug_assert"
    - "complex eigenvector phase convention: largest-modulus component rotated real-positive, then pyscf_core::canonicalize_signs on the real part with the sign flip MIRRORED onto the imaginary part"

key-files:
  created:
    - crates/pyscf-algebra/src/complex.rs
    - crates/pyscf-algebra/src/zgemm.rs
    - crates/pyscf-algebra/src/zblas.rs
    - crates/pyscf-algebra/src/zeigh.rs
    - crates/pyscf-algebra/src/zoracle.rs
    - crates/pyscf-algebra/tests/ctensor.rs
    - crates/pyscf-algebra/tests/zgemm.rs
    - crates/pyscf-algebra/tests/zeigh.rs
    - crates/pyscf-algebra/tests/zoracle_determinism.rs
    - crates/pyscf-kernels/src/pbc/mod.rs
    - crates/pyscf-kernels/src/pbc/zhadamard.rs
    - crates/pyscf-kernels/tests/pbc_zhadamard.rs
  modified:
    - crates/pyscf-algebra/src/lib.rs
    - crates/pyscf-kernels/src/lib.rs
  deleted:
    - crates/pyscf-algebra/examples/faer_c64_probe.rs   # Task-0 throwaway, per plan

key-decisions:
  - "D-PBC-04 RESOLVED: FAER_C64 = true. faer 0.24 has working native c64 SelfAdjointEigen, Llt and PartialPivLu."
  - "zeigh_gen / zcholesky / zsolve_linear dispatch to the faer c64 route; the real-arithmetic route is always built and is the CI cross-check."
  - "zcholesky's second route is an explicit complex Crout factorization, NOT the 2n x 2n embedding — the embedding is mathematically unusable for Cholesky (see DEVIATIONS)."
  - "zhadamard_dense carries an in-crate mirror of the K-04 kernel because pyscf-kernels DEPENDS ON pyscf-algebra (see DEVIATIONS)."
  - "zeigh_gen's debug cross-check compares EIGENVALUES only — eigenvectors are gauge-dependent under degeneracy."
  - "zeigh_gen_embedding gained a degeneracy fallback beyond the plan's fixed 0,2,4,... stride (see DEVIATIONS)."
  - "Complex eigenvector output is COLUMN-MAJOR (F-order), matching eigh_gen and pyscf_core::MOCoefficients."

patterns-established:
  - "CTensor::from_planes for building a planar result from two independently-computed real planes"
  - "Cross-route agreement tests: every zeigh-family routine asserts faer-c64 == real-arithmetic route"
  - "Subprocess-based RAYON_NUM_THREADS bit-identity test (self-respawn of the test binary)"

requirements-completed: [PBC-ALG-01, PBC-ALG-02, PBC-ALG-03]

# Metrics
duration: ~1h
completed: 2026-08-25
---

# Phase 9 Plan 02: The Complex-Algebra Contract

**`pyscf-algebra` now has complex arithmetic. Everything in the v2.0 PBC milestone depends on this plan. Implemented exactly the contract in `PBC-MASTER-PLAN.md §5`: the planar `CTensor` host type, `zgemm` as four real GEMMs in the mandated D-PBC-03 order, the `zblas` BLAS-1 surface, the `zeigh`/`zcholesky`/`zsolve_linear` family with BOTH routes of D-PBC-04, the D-PBC-17 ordered complex reductions, and the K-04 `zhadamard` cubecl kernel in `pyscf-kernels`. No new numeric type crosses the ALG-06 wall — `check-dependency-wall` still PASSes and no method crate gained a cubecl dependency.**

## Task 0 — D-PBC-04 decision (RECORDED VERBATIM)

A throwaway example `crates/pyscf-algebra/examples/faer_c64_probe.rs` built `faer::Mat<faer::c64>`
matrices and exercised `SelfAdjointEigen`, `Llt` and `PartialPivLu` against a known
2x2 Hermitian `H = [[2, 1−i], [1+i, 3]]` (exact eigenvalues 1 and 4) and the SPD
`A = [[4, 1−i], [1+i, 3]]`.

`cargo run -p pyscf-algebra --example faer_c64_probe` printed, verbatim:

```
eigenvalues: Complex { re: 0.9999999999999998, im: 0.0 } Complex { re: 4.0, im: 0.0 }
EIGH_OK = true
LLT_OK = true
LU_RESID = 2.7755575615628914e-17
LU_OK = true
FAER_C64 = true
```

**Decision: `pub(crate) const FAER_C64: bool = true;` in `zeigh.rs`.**

faer 0.24 ships a working native complex (`c64`) self-adjoint eigensolver, Cholesky
and LU. `zeigh_gen`, `zcholesky` and `zsolve_linear` therefore dispatch to the faer
`c64` routes; the real-arithmetic routes are still fully implemented and are the CI
cross-check (§5.3 mandates writing them "even if faer c64 works").

The example was deleted after the probe, per the plan. **This decision is referenced
by every later PBC plan — the `c64` route is available, and any later plan that needs
a complex factorization may assume it exists.**

## What Shipped

### Task 1 — `CTensor` (§5.1, D-PBC-02)

`crates/pyscf-algebra/src/complex.rs`. `#[derive(Debug, Clone, PartialEq, Default)]`,
two `Vec<f64>` planes of always-equal length. Full §5.1 surface: `zeros`,
`from_interleaved`, `to_interleaved`, `from_real`, `len`, `is_empty`, `conj`,
`is_real(tol)`, plus `from_planes` (used pervasively by `zgemm`/`zblas`, which build
the two planes independently). Every constructor debug-asserts `re.len() == im.len()`.
Interleaved `[re0, im0, …]` is the PyO3/NumPy wire format only.

### Task 2 — `zgemm` (§5.2, D-PBC-03)

`zgemm_dense` computes, in EXACTLY this order:
`t1 = gemm(a.re, b.re)`, `t2 = gemm(a.im, b.im)`, `t3 = gemm(a.re, b.im)`,
`t4 = gemm(a.im, b.re)`, then `re = t1 − t2`, `im = t3 + t4`. No Karatsuba, no fusion,
no reordering; the doc comment records the bit-parity reason (a zero imaginary plane
cancels EXACTLY, so a real operand reproduces the real `gemm_dense` bit-for-bit — this
is asserted as a test).

`zgemm_h_dense` materialises `Aᴴ` explicitly (`transpose_dense` on each plane, then
negate `im`) and calls `zgemm_dense`. Not fused. The operand `a` is the
UN-transposed `k × m` matrix.

### Task 3 — `zblas` (§5.2)

`zaxpy_dense` (4 `axpy_dense`), `zscal_dense` (zero-then-`zaxpy` with a temp),
`zdotc_dense` / `zdotu_dense` (4 `dot_dense` each), `zreduce_sum_dense`
(2 `reduce_sum_dense`), `ztranspose_dense` (2 `transpose_dense`), `zhadamard_dense`
(the K-04 kernel). Every routine documents that its evaluation order is load-bearing.
The device reductions carry an explicit "use `oracle_z*` for anything numerical" note.

### Task 4 — K-04 `zhadamard` cubecl kernel

`crates/pyscf-kernels/src/pbc/zhadamard.rs` (new `pbc` module, declared from
`lib.rs`). `#[cube(launch_unchecked)] fn zhadamard_kernel<F: Float>` over six planar
`Array<F>` operands plus `n`, with the `i < n` tail guard. Host launcher fans out via
`pyscf_algebra::dispatch_backend!`; both output planes are read back in one batched
`client.read`. The CubeCL manual (`.../Cubecl/INDEX.md` plus
`Handling_Interleaved_Complex_Numbers_in_CubeCL_with_ROCm_Backend.md` and
`Cubecl_generics.md`) was read before writing the kernel, per AGENTS.md §3. No cubecl
build errors occurred, so the §4 error protocol was not triggered.

### Task 5 — `zeigh`, `zcholesky`, `zsolve_linear` (§5.3, D-PBC-04)

`zeigh_gen_embedding` — the ALWAYS-built real `2n × 2n` route: build
`M = [[Hr, −Hi], [Hi, Hr]]` and the same for `S`, call the existing real `eigh_gen`,
take eigen-indices `0, 2, 4, …`, reassemble `C = C_top + i·C_bottom`, normalise so
`Cᴴ S C = I`, then apply `pyscf_core::canonicalize_signs` to the real part.

`zeigh_gen_faer` — the faer `c64` Löwdin transform (the exact algorithm `eigh_gen`
runs, lifted to `c64`, including the `S_LINEAR_DEP_TOL` linear-dependency removal and
the `+inf`-padded eigenvalue convention).

`zeigh_gen` dispatches to the faer route (`FAER_C64 == true`) and, in debug builds
for `n <= 16`, cross-checks the EIGENVALUES against the embedding route at 1e-11.

`zcholesky` / `zsolve_linear` follow the same two-route pattern
(`zcholesky_faer` / `zcholesky_crout`, `zsolve_linear_faer` / `zsolve_linear_embedding`).

### Task 6 — ordered complex reductions (§5.2, D-PBC-17)

`oracle_zsum(&CTensor) -> (f64, f64)` is two `oracle_sum` calls;
`oracle_zdot(x, y) -> (f64, f64)` is four `oracle_dot` calls combined in the `zdotc`
pattern. Documented as the ONLY reductions numerical PBC code may use.

### Task 7 — re-exports

`pub mod complex; zblas; zeigh; zgemm; zoracle;` in `crates/pyscf-algebra/src/lib.rs`
plus flat `pub use` lines matching the existing style. `pyscf-kernels` re-exports
`pbc::zhadamard::zhadamard` at the crate root.

## Verification Results

```
cargo test -p pyscf-algebra                                   ✅ (see below)
cargo test -p pyscf-kernels                                   ✅ (see below)
cargo clippy -p pyscf-algebra --all-targets -- -D warnings    ✅ clean on all new files
cargo run -p xtask --bin check-dependency-wall                ✅ PASS — ALG-06 intact
cargo build --profile release-oracle -p pyscf-algebra
  && cargo run -p xtask --bin check-no-fma                    ✅ PASS — FOUND-05 intact
cargo run -p xtask --bin check-forbidden-paths                ✅ PASS — 347 files
rustfmt --edition 2024 --check <all 14 new/modified files>    ✅ clean
```

New tests, all green:

| file | tests |
|---|---|
| `pyscf-algebra/tests/ctensor.rs` | 5 |
| `pyscf-algebra/tests/zgemm.rs` | 6 |
| `pyscf-algebra/tests/zeigh.rs` | 8 |
| `pyscf-algebra/tests/zoracle_determinism.rs` | 4 (+1 `#[ignore]`d child) |
| `pyscf-kernels/tests/pbc_zhadamard.rs` | 4 |

Highlights:
- interleaved round-trip is BIT-exact over 1000 random values (and over ±0/±inf);
- 64x64 random complex `zgemm` vs a naive host triple loop: max|Δ| < 1e-12;
- `zgemm` with zero imaginary planes is BIT-identical to real `gemm_dense`;
- `zgemm_h_dense(a, a)` Hermitian to 1e-12, diagonal imaginary part exactly ~0;
- random 8x8 Hermitian, `S = I`: eigenvalues match the independent embedding route to
  1e-12, `Cᴴ H C` off-diagonal < 1e-12, `Cᴴ S C = I`, repeated calls BIT-identical;
- both `zeigh`/`zcholesky`/`zsolve_linear` route pairs agree to ≤ 1e-10;
- `oracle_zsum` over 1e6 elements is BIT-IDENTICAL at `RAYON_NUM_THREADS=1` and `=8`
  (the test re-spawns its own binary as a subprocess so the env var is set before any
  rayon pool could initialise) and equals the in-process result;
- K-04 kernel matches a host reference to 1e-14 at n = 1/255/256/257/1000/4096
  (straddling the 256-thread cube dimension), and BIT-matches the `pyscf-algebra`
  mirror.

## DEVIATIONS from the plan

Three, all documented in-code at the point of deviation.

**1. `zhadamard_dense` cannot call the K-04 kernel — the crate graph forbids it.**
Task 3 says `zhadamard_dense` (in `pyscf-algebra`) "calls the K-04 kernel from Task 4"
(in `pyscf-kernels`). `pyscf-kernels` DEPENDS ON `pyscf-algebra`, so that call would be
a dependency cycle. Resolution: the canonical K-04 kernel ships in
`pyscf-kernels/src/pbc/zhadamard.rs` exactly as the plan's artifact list requires (that
is the entry point PBC method crates use, since they depend on `pyscf-kernels`), and
`pyscf-algebra`'s `zhadamard_dense` carries a byte-identical in-crate mirror of the
same `#[cube(launch_unchecked)] fn zhadamard_kernel<F: Float>` body. Both copies are
cross-checked BIT-for-bit by `pbc_zhadamard.rs::kernels_and_algebra_copies_agree_bit_for_bit`,
which is the lockstep guard. Both sites carry a comment pointing at the other.

**2. `zcholesky`'s second route is a complex Crout factorization, not the embedding.**
Task 5 says "`zcholesky` and `zsolve_linear` follow the same two-route pattern". For
`zsolve_linear` the real `2n × 2n` embedding IS exact and is what shipped. For Cholesky
it is not usable: `M = [[Ar, −Ai], [Ai, Ar]]` factors as `M_L · M_Lᵀ` with
`M_L = [[Lr, −Li], [Li, Lr]]`, which is block-lower-triangular but NOT element-wise
lower triangular (the `−Li` block sits above the diagonal), so a real Cholesky of `M`
returns a different factor that does not unpack into `L`. `zcholesky_crout` is instead
an explicit complex Crout recurrence written in real arithmetic — it shares no code
with faer, so it is still a genuinely independent cross-check, and the test asserts the
two routes agree to 1e-10 (the HPD Cholesky factor with a real positive diagonal is
unique, so this is a strong check).

**3. `zeigh_gen_embedding` has a degeneracy fallback beyond the fixed `0, 2, 4, …` stride.**
§5.3 step 2's fixed stride assumes each of `H`'s eigenvalues is simple *in `H`*
(multiplicity two in `M`). When `H` itself is degenerate — routine at high-symmetry
k-points — `M`'s eigenvectors for that value span a 4-real-dimensional space, and
columns `2k` and `2k+2` can land on the SAME complex direction, silently producing a
rank-deficient `C`. The implementation takes the mandated even columns first and
S-projects each candidate against the already-accepted set; if a candidate has no
`S`-component left (`< 1e-8`) it is skipped and the odd columns are scanned as the
fallback. **Non-degenerate inputs never reach the fallback and get exactly the
`0, 2, 4, …` columns the plan mandates.** If fewer than `n` `S`-orthogonal vectors can
be recovered the function errors rather than returning a bad basis.
`zeigh.rs::zeigh_embedding_handles_a_degenerate_eigenvalue` covers this.

Two additional judgement calls worth recording (not deviations, but under-specified in
the plan):

- **`canonicalize_signs` is applied to the real part with the flip MIRRORED onto the
  imaginary part.** Calling it on `c.re` alone would negate only half of a complex
  eigenvector, turning it into a different vector. The implementation calls the real
  `pyscf_core::canonicalize_signs` on the real plane (as mandated), detects which
  columns it flipped, and applies the same flip to the imaginary plane. A global phase
  rotation (largest-modulus component made real-positive) runs first, which makes both
  routes converge on the same phase and makes `canonicalize_signs` a no-op in practice.
- **The debug cross-check compares eigenvalues only.** Eigenvectors of a degenerate
  eigenvalue are genuinely route-dependent (the eigen*space* basis is not unique), so a
  vector-level `debug_assert` would fire on legitimate input. The vector-level check
  that IS route-independent — `Cᴴ H C` diagonal and `Cᴴ S C = I` — is asserted in the
  tests for both routes.

## Carry-overs

- **`pyscf-algebra/src/eigh_gen.rs` still has an in-file `mod tests`**, violating
  AGENTS.md §2 (tests in separate files). Pre-existing, untouched by this plan; worth a
  cleanup pass.
- **`cargo clippy -p pyscf-algebra --all-targets -- -D warnings` reports PRE-EXISTING
  findings** in `axpy.rs:43`, `scal.rs:51` (`unnecessary_cast` on `ABSOLUTE_POS as usize`)
  and `gemm.rs:58` (`manual_div_ceil`). All three are in files already modified in the
  working tree before this plan and are out of its scope. Every new file in this plan is
  clippy-clean.
- **`tests/zgemm.rs` is slow (~3 min)** — six 64x64-class complex GEMMs on the debug
  cubecl CPU runtime means 4 kernel launches each. Correct but heavy; if CI wall-clock
  becomes a problem, the shape list is the dial to turn.
- **`zhadamard` has no GPU-hardware differential test.** The CPU runtime path is
  covered; a ROCm arm mirroring `tests/gemm_oracle.rs`'s `#[cfg(feature = "rocm")]`
  test would close this.
- **No `Tensor`-surface (opaque `BufferId`) complex API.** §5 specifies only the
  `*_dense` host-slice surface, which is what shipped.

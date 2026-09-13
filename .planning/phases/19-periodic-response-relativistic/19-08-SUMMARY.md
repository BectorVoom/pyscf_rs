# 19-08 SUMMARY — unrestricted TDA/TDHF (gamma + k)

**Shipped:** 2026-09-12. `uhf` 7/7 green + 1 ignored live arm.

## Drivers (`uhf.rs` gamma real, `kuhf.rs` k-point complex)

- `build_uab`: coupled `(da+db)²` with `A_ab = J_ab` Coulomb-only (no
  exchange), `'jaib'` B-exchange layout, NO singlet doubling. `kuhf.rs`
  mirrors the 19-07 k-maps per spin sector (complex-linear, Hermitian
  assert); TDHF reuses the shared `symm_tdhf` core on real data.
- B_ab is a PLAIN transpose (upstream's documented `B_bbaa =
  B_aabb.transpose`, no conjugation) — ported literally; A_ab is the
  conjugate transpose (Hermitian A, asserted).
- Corrections during development: (a) the `'iabj'` J-terms had MO axes 2,3
  swapped (same shape — caught by self-review before running, confirmed by
  the element-wise gate); (b) gamma `[i][p][q][r]` vs 7d `[p][i][q][r]` block
  layouts differ — the single-k test transposes, doesn't wrap (caught by the
  Hermitian assert at 0.66); (c) different eigh blocking gives last-ulp
  differences — tolerance, not bit-identity, with the reason stated.

## Gate A + coupling (`tests/uhf.rs`, H2O/`sto-3g` triplet fixture, live 2.12.1)

- `build_uab` vs upstream element-wise (< 1e-10); TDA + TDHF roots at 4dp.
- Coupling proven load-bearing both ways: zeroed-ab kernel == independent
  solves (coupling enters ONLY through ab), while live coupled roots DIFFER
  from the decoupled union (separate solves are plausibly wrong).
- `kuhf` on single-k live data matches gamma to 1e-8 with shift stamped;
  synthetic COMPLEX index oracle pins the k-build maps element-wise.
- Roots unfiltered (count exact, incl. spin-contaminated).
- Big-box 2-k Davidson comparison `#[ignore]`d: non-uniform per-k fillings
  (6 vs 4 here) need the matrix-free vind path — as does upstream's own
  `get_ab` (`assert all(noccs == nocc)`), so dense is the wrong tool there.

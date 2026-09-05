# 16-03 — `davidson_nosym1` + `pick_real_eigs`. COMPLETE 2026-09-06.

`crates/pyscf-algebra/src/davidson.rs`, host-only, gated by
`crates/pyscf-algebra/tests/davidson_nosym.rs` (8 tests, no PySCF oracle).

## What was ported

`linalg_helper.py:741-937` and its five helpers — `_qr` (`:1411`),
`_fill_heff` (`:183`), `_outprod_to_subspace`/`_gen_x0` (`:1436`),
`_sort_elast` (`:1468`), `_normalize_xt_` (`:1492`) — plus `pick_real_eigs`
(`:593`) with `_eigs_cmplx2real` (`:614`). The module doc records the
algorithm before the code, as Task 1 required: the subspace expansion, the
`max_space + (nroots-1)*6` widening and its restart, the two `lindep` collapse
points, the `sqrt(tol)` residual rule, and where `fill_heff` builds the
non-symmetric projected `heff`.

`max_space`, `lindep`, `nroots`, `pick`, `left`, `lessio` and `follow_state`
are all real parameters with upstream's defaults (`16-REVIEW.md §4.1` and
`§7.3`). The projected `heff` is solved by `eig_general`, a new dense general
complex eigendecomposition over `faer` — the same direct path 17-02 established
for `character_table`. `left = true` is served by diagonalising `heffᴴ` and
conjugating the eigenvalues, since `faer` exposes right eigenvectors only.

**The primitive, named (`16-CONTEXT §3.2`):** every inner product in this
module — the Gram–Schmidt overlaps, the `heff` elements, the `_normalize_xt_`
projections, the residual norms — is `⟨x|y⟩ = Σ conj(x)·y`, so it is
`oracle_zdot` (`zdotc`) at **every** site, with `oracle_zdot_re` where only the
real part is wanted. This is the one module in Phase 16 where `oracle_zdot` is
the right default; the CC contractions are mostly unconjugated and want
`oracle_zdotu`.

## The port reproduces upstream, stalls included — measured

While writing the tests the solver was seen to STALL: on a diagonally-dominant
random complex matrix it converges to a `1.7e-4` residual and stops with
"linear dependency in trial subspace". Rather than assume a port defect,
upstream was run on the same matrices
(`measurements/m7_davidson_ref.py`, `README §8`):

| | upstream | this port |
|---|---|---|
| eigenvalue | `1.0003286056665808` | `1.0003286056665817` |
| residual | `0.0001705039877410091` | `1.7050398774100944e-4` |
| cycles / `aop` calls | 4 | 4 |
| trajectory | `0.131 → 0.00171 → 0.000171 → 0.000171`, then the lindep break | identical |

At coupling `≥ 0.2` **upstream itself** converges to a spurious `2.565e-15`
eigenvalue against a true lowest root of `0.98032995`; the port does the same.
**The stall is the method — a plain diagonal preconditioner with a
Koopmans-style unit-vector guess — not either implementation.**

### Deviation 1 — the test fixtures and their tolerances are measured, not assumed

The plan asks for "random `n = 40, 80` general complex matrices" and roots to
`1e-10` with the residual below `tol_residual`. A dense random matrix is in the
stall regime above, so the fixture is a **band** matrix (`1/d²` real, `0.15/d³`
imaginary off-diagonals, `diag = 1 + 2i`) at coupling `0.05`, where the method
works: eigenvalues land at `5e-12 … 1e-14` from the dense solve. `tol_residual`
is set to `1e-3`, the value the method REACHES; the residual is the loose gate
and the **eigenvalue is the tight one at `1e-10`**, which is not an accident —
a non-normal Ritz value can be far more accurate than its residual suggests
(here a `3.8e-4` residual carries a `5e-12` eigenvalue), and it is the
eigenvalue assertion that would catch a wrong conjugation, a transposed `heff`
or a mis-ordered `pick`.

### Deviation 2 — test 8 (`left`) gates the pairing, not a residual

Upstream warns at `:926` that "left eigenvectors from subspace diagonalization
method may not be converged" — the subspace is expanded to minimise the RIGHT
residual. Asserting a tight `‖Aᴴ xl − λ̄ xl‖` would gate a property upstream
does not provide, so the test asserts what IS contractual: the eigenvalues are
unchanged by asking for left vectors, and `⟨xl|x⟩` does not vanish.

## Verification

* `cargo test -p pyscf-algebra` green (8 new tests; the whole crate passes).
* `cargo clippy -p pyscf-algebra --no-deps --all-targets` clean apart from the
  PRE-EXISTING `complex.rs:49` lint reproduced on unmodified `main`
  (`17-02-SUMMARY.md` Deviation 1); it was not "fixed" here.
* Test 3 asserts the matrix-free property against a literal bound — `aop` sees
  fewer vectors than `n`, so a dense-materialising implementation fails rather
  than passes slowly.
* No `pyscf-oracle` dependency was added.

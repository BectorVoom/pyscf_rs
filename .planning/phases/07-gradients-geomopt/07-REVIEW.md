---
phase: 07-gradients-geomopt
reviewed: 2026-05-26T00:00:00Z
depth: deep
files_reviewed: 24
files_reviewed_list:
  - crates/pyscf-grad/src/lib.rs
  - crates/pyscf-grad/src/error.rs
  - crates/pyscf-grad/src/hooks.rs
  - crates/pyscf-grad/src/scanner.rs
  - crates/pyscf-grad/src/verify_fd.rs
  - crates/pyscf-grad/src/rhf.rs
  - crates/pyscf-grad/src/uhf.rs
  - crates/pyscf-grad/src/rks.rs
  - crates/pyscf-grad/src/uks.rs
  - crates/pyscf-grad/src/cphf.rs
  - crates/pyscf-grad/src/mp2.rs
  - crates/pyscf-grad/src/ccsd.rs
  - crates/pyscf-grad/src/ecp.rs
  - crates/pyscf-geomopt/src/lib.rs
  - crates/pyscf-geomopt/src/internals.rs
  - crates/pyscf-geomopt/src/bmatrix.rs
  - crates/pyscf-geomopt/src/rfo.rs
  - crates/pyscf-geomopt/src/backtransform.rs
  - crates/pyscf-geomopt/src/converge.rs
  - crates/pyscf-geomopt/src/checkpoint.rs
  - crates/pyscf-geomopt/src/shims.rs
  - crates/pyscf-gto/src/intor.rs
  - crates/pyscf-gto/src/ecp_engine_cintx.rs
  - crates/pyscf-py/src/grad.rs
  - crates/pyscf-py/src/geomopt.rs
  - crates/pyscf-oracle/src/grad_oracle.rs
  - crates/pyscf-oracle/src/runner.rs
findings:
  critical: 2
  warning: 9
  info: 6
  total: 17
status: issues_found
---

# Phase 7: Code Review Report

**Reviewed:** 2026-05-26
**Depth:** deep
**Files Reviewed:** 24 source files (+ 2 dispatch/runner files)
**Status:** issues_found

## Summary

Phase 7 ships analytical gradients (RHF/UHF/RKS/UKS/MP2/CCSD/ECP), a single
matrix-free Krylov CPHF solver, a native BFGS+RFO geometry optimizer, the
cintx grad-intor dispatch guards, and the PyO3/FFI bridge. The phase is built
under an explicit "structural-now, numeric-when-cintx-lands" gating strategy:
six of the eight gradient-integral families are missing from cintx, so the
numeric `de` assembly is deliberately unreachable behind a clean
availability-error `?` and the FD numeric arms are `#[ignore]`'d. This makes
most numeric-correctness defects unverifiable today, so they are classified as
WARNING (latent — they will bite the day cintx lands the families) rather than
BLOCKER.

The two BLOCKERs are *reachable today*: (1) an oracle test that contradicts the
`--features python` dispatch path and will fail CI on that arm, and (2) a
convention violation (bare `+=` in a genuine numeric accumulation) in the
geomopt B-matrix pseudo-inverse — which IS on a live, always-on code path
(geomopt drives the optimizer numerically without any cintx gate on the
linear-algebra side).

The PyO3 wall is respected (pyscf-grad/pyscf-geomopt are pyo3-free per their
Cargo.toml dep sets), errors `?`-propagate rather than panic across the FFI,
and the component-leading `[3, nao, nao]` layout contract is asserted before
every contraction. Good defensive shape-checking throughout.

The most consequential latent finding is WR-01: the MP2 and CCSD gradient
bodies compute the entire relaxed density + Z-vector, then **discard it** and
return the bare RHF `base_de`. Even when cintx lands the families, the relaxed
density is never plugged into the assembly — the code as written would return
an SCF gradient mislabeled as an MP2/CCSD gradient.

## Critical Issues

### CR-01: Oracle dispatch-layer test contradicts the `--features python` dispatch and will fail CI on the python arm

**File:** `crates/pyscf-oracle/src/grad_oracle.rs:75-86` (test) vs `crates/pyscf-oracle/src/runner.rs:189-206` (dispatch)
**Issue:** The eight Phase-7 names (`nuc_grad_rhf`…`geomopt_h2o`) are added to
`KNOWN_METHODS` (runner.rs:145-152) but are NOT added to the
`python_impl::dispatch` match (runner.rs:190-205); they fall into the
`other => Err(OracleError::UnknownMethod(...))` arm. The test
`registered_grad_geomopt_methods_are_never_unknown` (grad_oracle.rs:76) is NOT
feature-gated and asserts `!matches!(r, Err(OracleError::UnknownMethod(_)))`
for every one of those names. In a default (no-`python`) build the dispatch
short-circuits to `PythonFeatureNotEnabled` before the match, so the test
passes. But in a `--features python` build, `run_oracle_check` enters
`python_impl::dispatch`, hits the `other` arm, and returns `UnknownMethod` for
all eight names — the test then FAILS. Same for `all_eight_phase7_names_registered`
(grad_oracle.rs:123). This is a build-mode-dependent test failure that the
`grad-oracle-upstream-manual` / any `--features python` CI arm will surface.
**Fix:** Either gate the two assertions to `#[cfg(not(feature = "python"))]`
(matching the existing `known_grad_method_without_python_is_feature_gated_not_unknown`
which already carries that gate at grad_oracle.rs:92), or add explicit
register-but-defer arms to the dispatch match that return a deferred-dispatch
error distinct from `UnknownMethod`:
```rust
// in python_impl::dispatch, before the `other` arm:
"nuc_grad_rhf" | "nuc_grad_uhf" | "nuc_grad_rks" | "nuc_grad_uks"
| "nuc_grad_mp2" | "nuc_grad_ccsd" | "nuc_grad_ecp" | "geomopt_h2o" => {
    Err(OracleError::PythonFeatureNotEnabled) // or a dedicated DeferredDispatch
}
```

### CR-02: Bare `+=` in a genuine numeric accumulation on the always-on geomopt path (reduction-order convention violation)

**File:** `crates/pyscf-geomopt/src/bmatrix.rs:224`
**Issue:** `g_inverse` reconstructs the pseudo-inverse `G⁻ = Σ_k (1/λ_k) v_k v_kᵀ`
with a bare `+=` accumulating over the eigenvalue index `k`:
```rust
ginv[i * nint + j] += inv * vik * vjk;
```
This is exactly the "bare `+=` in a numeric accumulation = thread-count-dependent
result" pattern the project convention forbids (CLAUDE.md reduction-order rule).
The accumulation runs over `k` (the outer loop, line 209), so each `ginv[i,j]`
is a real multi-term reduction, not a single write. The doc comment on
lines 220-223 explicitly claims "we build per-(i,j) term buffers below to keep
the reduction oracle-ordered" — but the code does NOT do that; the comment
describes machinery that does not exist. Unlike the gradient bodies, geomopt's
linear algebra is NOT cintx-gated: `optimize` → `bmatrix::build` → `g_inverse`
runs on every optimizer step today, so this is a live release-oracle
bit-reproducibility hazard, not a deferred one.
**Fix:** Materialize the per-`k` contributions into a buffer and `oracle_sum`,
matching the pattern used everywhere else in the phase:
```rust
let mut ginv = vec![0.0_f64; nint * nint];
for i in 0..nint {
    for j in 0..nint {
        let mut terms = Vec::with_capacity(nint);
        for k in 0..nint {
            let lam = evals[k];
            if !lam.is_finite() || lam <= G_EIGENVALUE_TOL { continue; }
            terms.push((1.0 / lam) * evecs_fortran[i + k * nint] * evecs_fortran[j + k * nint]);
        }
        ginv[i * nint + j] = oracle_sum(&terms);
    }
}
```

## Warnings

### WR-01: MP2 and CCSD `grad_elec` compute the relaxed density + Z-vector then discard it, returning the bare RHF gradient

**File:** `crates/pyscf-grad/src/mp2.rs:402-429`, `crates/pyscf-grad/src/ccsd.rs:472-503`
**Issue:** Both bodies do the full structural work — build `dm1mo`
(`gamma1_intermediates` / `relaxed_rdm1`), build `xvo` (`build_xvo_base`),
solve the Z-vector (`response_dm1`), and accumulate `dm1mo += resp` — and then
**never use `dm1mo`**. They return `crate::rhf::grad_elec(&rhf_ref, ...)`
(mp2.rs:425, ccsd.rs:499), i.e. the plain RHF/SCF gradient. The relaxed density
is computed and thrown away. The inline comments (mp2.rs:427-428, ccsd.rs:501-502)
acknowledge this ("today base_de is unreachable past the clean-error `?`"), but
the code structure means that the day cintx lands the grad-intor families, this
function will silently return an **SCF gradient mislabeled as an MP2/CCSD
gradient** — there is no `TODO`/`unimplemented!` guard forcing the wiring to be
completed. The FD numeric tests that would catch this are `#[ignore]`'d. This is
the single largest correctness landmine in the phase.
**Fix:** Make the un-wired numeric path fail loudly instead of returning a wrong
result. Either return an explicit `NotYetImplemented`/availability error *after*
the gated `base_de` `?` (so when cintx lands, the test goes red and forces the
relaxed-density wiring), or add a debug assertion / compile-time `todo!()`-style
marker keyed to the integral availability. At minimum, do not silently fall
through to `Ok(base_de)`.

### WR-02: RHF/RKS/UHF/UKS exchange (`vk`) contraction index is unverifiable and likely transposed vs PySCF `get_jk`

**File:** `crates/pyscf-grad/src/rhf.rs:364`, `crates/pyscf-grad/src/rks.rs:416`, `crates/pyscf-grad/src/uhf.rs:328-329`, `crates/pyscf-grad/src/uks.rs:409-410`
**Issue:** The K build uses `k_terms[k*nao + l] = g * dval(j, k)` where `j` is
the fixed outer-loop bra index and `g = (∇i j|kl)`. The density factor
`D[j,k]` does not depend on the inner summation index `l` at all, so for each
fixed `(x,i,j)` the term is `D[j,k] · Σ_l (∇i j|kl)` summed over `k` — i.e. the
exchange contracts the ERI's `l` index against nothing meaningful. PySCF's
`get_jk` exchange is `K[i,l] = Σ_jk (ij|kl) D[j,k]` (the contracted indices are
the *inner-bra/inner-ket cross pair*), which is not what this loop computes. The
comment claims `K[x,i,j] = Σ_kl (∇i j|kl) D_jk` but that formula itself has `D`
independent of `l`, which is dimensionally suspect for an exchange matrix.
Because `int2e_ip1` is cintx-missing, this never executes today, so it cannot be
proven wrong by the FD harness — hence WARNING not BLOCKER. But it should be
re-derived against `pyscf/scf/jk.py get_jk` / `pyscf/grad/rhf.py:149-178` before
the family lands.
**Fix:** Re-derive the exchange contraction against upstream `get_jk` and add a
unit test that, given a synthetic ERI with known symmetry, reproduces the
expected `vk`. The likely-correct index is a contraction over the inner pair
that varies with the summation variable (e.g. `D[l,k]` with the output bra/ket
roles swapped), not the `l`-independent `D[j,k]`.

### WR-03: ECP scalar stitch uses row-major block indexing while the parallel non-ECP arity-2 stitch uses F-order — only one can be right

**File:** `crates/pyscf-gto/src/ecp_engine_cintx.rs:233` vs `crates/pyscf-gto/src/intor.rs:359`
**Issue:** The non-ECP arity-2 stitch reads the cintx per-pair block as F-order:
`out[(oi+ii) + (oj+jj)*nao] = block[ii + jj*ni]` (intor.rs:359, comment at 280-283
"cintx writes the block in F-order"). The ECP scalar stitch reads the same
shape block as row-major: `out[(oi+ii) + (oj+jj)*nao] = block[ii*nj + jj]`
(ecp_engine_cintx.rs:233, comment at 226-230 "the per-pair block is row-major").
Both cite cintx-side parity suites, but they describe contradictory block
layouts for what is structurally the same `[ni, nj]` shell-pair block from the
same `SessionRequest.evaluate()` path. For non-square shell pairs (`ni != nj`)
at least one stitch transposes the block. If the cintx convention is uniform
across operators, one of these is a transposition bug.
**Fix:** Confirm against the cintx `IntegralTensor` layout contract whether the
scalar block is row-major or column-major and unify the two stitch paths to the
same convention (extract a single shared `stitch_arity2_block` and call it from
both the non-ECP and ECP paths).

### WR-04: `add_atom_grad` uses bare `+=` (reduction-order convention) in the B-matrix row build

**File:** `crates/pyscf-geomopt/src/bmatrix.rs:65-67`
**Issue:** `add_atom_grad` writes `brow[base] += g[0]` (and `+1`/`+2`). The
comment (lines 61-64) argues these slots are "touched at most once per
(primitive, atom)" so accumulation is not really happening — which is true for
the current primitive set (a given atom appears once per Distance/Angle/Dihedral
row). But the function name (`add_…`), the `+=` operator, and the
`vec![0.0; ...]` init together encode an *accumulation idiom*, and any future
primitive that touches the same atom twice (e.g. an out-of-plane or
linear-bend coordinate) would silently introduce a thread-order-independent —
but convention-violating and fragile — accumulation. Lower-severity than CR-02
because it is currently a single-write in practice.
**Fix:** Either rename to `set_atom_grad` and use `=` (asserting single-write),
or if accumulation is genuinely intended, route through `oracle_sum` of the
prior value and the new contribution.

### WR-05: Geomopt scanner cache keyed on full `pyscf_gto::dumps` string — correctness depends on dumps being geometry-deterministic

**File:** `crates/pyscf-py/src/geomopt.rs:159,177`
**Issue:** `build_native_scanner` memoizes `(geom-string, e, de)` using
`pyscf_gto::dumps(mol)` as the cache key, and reconstructs via
`pyscf_gto::loads(&key)`. The `GradScanner::eval` calls the `base` (energy)
closure then the `grad` closure; the cache makes the second call reuse the
first's `(e, de)`. This is correct *only if* `dumps` produces a byte-identical
string for the same geometry every call (no nondeterministic ordering, no
floating-point formatting drift) AND `dumps`/`loads` round-trips the geometry
losslessly. If `dumps` ever serializes a HashMap-ordered field or rounds
coordinates, two evaluations at the "same" geometry could miss the cache
(harmless, double SCF) or — worse — a `loads`-reconstructed Mole could differ
subtly from the original `work` Mole the optimizer mutates, so the gradient is
evaluated at a slightly different geometry than the energy. There is no
assertion that `loads(dumps(mol))` is geometry-identical.
**Fix:** Either key the cache on the raw `(natm, 3)` coordinate array (a
`Vec<[f64;3]>` compared exactly) rather than a serialized string, or add a
round-trip invariant check. Prefer passing the native `Mole` through without a
`dumps`/`loads` reconstruction if the Python scanner can accept a native PyMole
directly.

### WR-06: RKS `get_vxc` ignores the hybrid-exchange GGA `∂f/∂σ` term — LDA-only path silently used for GGA/hybrid functionals

**File:** `crates/pyscf-grad/src/rks.rs:359,373`, `crates/pyscf-grad/src/uks.rs:338-339,354`
**Issue:** `get_vxc` calls `NumInt::eval_rho(ao_value, ..., XcType::Lda)` and
`ni_eval_vrho(... DerivOrder::Vxc)` using only the value block and the `vrho`
(`∂f/∂ρ`) term. For any GGA or hybrid functional the XC-potential derivative
must also contract the gradient block against `∂f/∂σ` (`vsigma`). The code
hard-codes `XcType::Lda` and discards the AO gradient components beyond using
`∇_x AO_μ` for the Pulay-like grid term. The module doc (rks.rs:354 comment)
claims "GGA folds the ∇ρ·∂f/∂σ contraction in via the same eval_xc surface" but
the code never reads `vsigma`. For a non-LDA `xc` string (the common case —
PBE, B3LYP) this produces a wrong `vxc_grid`. The XC-grid term is
cintx-INDEPENDENT, so this path is reachable as soon as a KS gradient is run on
a GGA — it is not protected by the cintx gate.
**Fix:** Branch on the functional family (`NumInt`/`xcfun` already exposes the
type): for GGA/hybrid, evaluate `XcType::Gga`, read `vsigma`, and add the
`∂f/∂σ · ∇ρ` contraction to `vxc_grid`. At minimum, reject a non-LDA `xc` with
a clear error rather than silently computing the LDA-only potential.

### WR-07: `grid_weight_derivative_force` is a structural placeholder that always returns zero — `with_grid_response(true)` silently produces no extra force

**File:** `crates/pyscf-grad/src/rks.rs:469-483`
**Issue:** When `grid_response` is enabled, `extra_force` is supposed to add the
Becke-weight-derivative term. The implementation builds the grid + per-point
energy density, then pushes `0.0 * we` (literally zero) into the force buffers
(lines 479-481) and returns `[oracle_sum(zeros)] = [0,0,0]`. So
`RksGradients::with_grid_response(true)` is indistinguishable from the default
`false` — a user who explicitly opts into grid response gets a silent no-op, not
the documented Becke-weight-derivative force. This is cintx-independent and
reachable today. The grid build + `eval_rho` + `ni_eval_exc` work is pure dead
computation (it is multiplied by `0.0`).
**Fix:** Either implement the real `∂w_g/∂R_ia` partition weight-gradient
contraction, or make `with_grid_response(true)` return a clear
`NotYetImplemented` error so the no-op is not silent. Do not advertise a
supported feature that returns zero.

### WR-08: CPHF `solve_nos1` rejects exact-zero orbital-energy denominators but not near-degenerate ones — and the `e_a[a]==e_i[i]` test is fragile

**File:** `crates/pyscf-grad/src/cphf.rs:225-234`
**Issue:** The `e_ai` build rejects `denom == 0.0` (exact float equality) and
non-finite denoms, but a near-degenerate occ/vir gap (`denom ≈ 1e-15`) passes
the guard and produces an `e_ai` of ~1e15, which then amplifies noise through
the Krylov iteration and the `mo1base = h1 * -e_ai` RHS. The exact `== 0.0`
comparison will essentially never trigger for computed orbital energies. This is
a robustness gap rather than a definite bug.
**Fix:** Guard on a relative/absolute threshold (`denom.abs() < tol`) rather than
exact zero, returning the descriptive degeneracy error for tiny gaps.

### WR-09: `verify_fd` max-diff and several norm/max reductions use `fold(f64::max)` instead of an oracle reduction — inconsistent with the stated discipline

**File:** `crates/pyscf-grad/src/verify_fd.rs:118`, `crates/pyscf-geomopt/src/lib.rs:445`, `crates/pyscf-geomopt/src/rfo.rs:152`
**Issue:** The module docs for `verify_fd` (lines 13-17) claim "the max-diff scan
… routes through `pyscf_algebra::oracle_sum`/`oracle_dot` — NO bare `+=`", but
`max_abs_diff` is computed with `abs_diffs.iter().copied().fold(0.0, f64::max)`
(verify_fd.rs:118), a non-oracle reduction. Same pattern in
`max_abs_flat` (lib.rs:445) and `BfgsHessian::min_eigenvalue`
(rfo.rs:148-152). A max/min fold is order-independent for finite values so this
is not a numeric-reproducibility bug, but it contradicts the comment and the
project's "every reduction goes through the oracle" framing, making the
discipline claims unreliable for future readers.
**Fix:** Either accept that order-independent max/min folds are exempt and update
the comments to say so, or route through a `pyscf_algebra` max helper if one
exists. Consistency matters here because the comments are load-bearing for the
reduction-order audit.

## Info

### IN-01: `oracle_dot` "touch" lines compute and discard a value purely to exercise an import

**File:** `crates/pyscf-grad/src/rhf.rs:440`, `uhf.rs:419`, `rks.rs:540`, `uks.rs:478`
**Issue:** Each `grad_elec` ends with
`let _ = oracle_dot(&dm0[..nao.min(dm0.len())], ...)` whose only purpose
(per the comment) is "so the algebra-wall import is exercised". This is dead
computation kept alive solely to satisfy a lint/structural check. It is wasted
work on every gradient call.
**Fix:** Remove the discard lines; if the import must stay referenced, use it in
a real reduction or drop the unused import.

### IN-02: RKS uses `hybrid_coeff(xc, 0)` while UKS uses `hybrid_coeff(xc, 1)`

**File:** `crates/pyscf-grad/src/rks.rs:304` vs `crates/pyscf-grad/src/uks.rs:284`
**Issue:** The `spin` argument differs between the closed- and open-shell paths.
Per `pyscf-dft/src/numint.rs:227`, `hybrid_coeff` forwards `spin` to
`rsh_coeff` but returns only the `hyb` fraction, which is spin-independent for
standard hybrids — so this is harmless today. Flagged for consistency: pick one
convention (upstream passes the molecule spin) and document why it does not
affect `hyb`.

### IN-03: `make_rdm1e` energy-weighting recomputed inside the innermost MO loop

**File:** `crates/pyscf-grad/src/rhf.rs:177-191`
**Issue:** The `let w = mo_energy[i] * occ` weight and the `occ > 0.0` branch are
recomputed for every `(mu, nu)` pair although they depend only on `i`. Pure
clarity/micro-redundancy (performance is out of v1 scope); correctness is fine.
**Fix:** Hoist the per-MO weights into a precomputed `Vec` before the `mu`/`nu`
loops.

### IN-04: `coords_to_atom_string` emits raw `{}` float formatting — round-trip precision depends on default `f64` Display

**File:** `crates/pyscf-geomopt/src/lib.rs:164-170`, `crates/pyscf-py/src/geomopt.rs` (via dumps), `crates/pyscf-geomopt/src/shims.rs:151-160`
**Issue:** The optimizer commits each new geometry by formatting coordinates with
`format!("{} {} {} {}", s, c[0], c[1], c[2])` and re-parsing via `set_geom_`.
Default `f64` `Display` does not guarantee a lossless round-trip for all values
(it prints the shortest representation, which IS round-trip-safe in Rust's
`Display` for `f64` since 1.0 — so this is actually fine, but worth a note since
the optimizer's convergence depends on it). No action strictly required;
documenting the dependency.
**Fix:** None required; optionally use `{:?}` or an explicit precision to make
the lossless-round-trip intent explicit.

### IN-05: `default_grad_elec` / `default_grad_ecp` seam functions are dead once the real bodies exist

**File:** `crates/pyscf-grad/src/rhf.rs:523-529` (and uhf/rks/uks/mp2/ccsd/ecp equivalents)
**Issue:** Each module keeps a `pub fn default_grad_elec()` that just returns an
error "requires a reference snapshot". With the real `grad_elec` free functions
and the trait impls in place, these are never called from within the crate.
They are retained "for compatibility with the 07-02 module stub". This is
dead-ish public API surface.
**Fix:** Remove if no external caller depends on them, or mark `#[deprecated]`.

### IN-06: `_engine_marker` / `mole_to_pymole` thin indirections exist only to keep imports load-bearing

**File:** `crates/pyscf-py/src/geomopt.rs:277-279,415-419`
**Issue:** `mole_to_pymole` is a one-line wrapper around `PyMole::from_mole`, and
`_engine_marker` (`#[allow(dead_code)]`) exists solely to reference
`GeometryOptimizer` + `NATIVE_ENGINE_NAME`. Minor indirection / dead code kept
for structural-import reasons.
**Fix:** Inline `mole_to_pymole`; drop `_engine_marker` if the import can be
referenced where actually used.

---

_Reviewed: 2026-05-26_
_Reviewer: Claude (gsd-code-reviewer)_
_Depth: deep_

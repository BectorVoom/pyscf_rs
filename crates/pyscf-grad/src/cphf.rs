//! The single matrix-free Krylov CPHF/CPKS solver (D-03, GRAD-10).
//!
//! Caller-shape analog only: `crates/pyscf-algebra/src/solve_linear.rs:29`
//! (`solve_linear(a, b, n)` — but that is DENSE LU; NO matrix-free Krylov exists
//! in `pyscf-algebra`, so this module BUILDS the Krylov, routing every inner
//! product through `pyscf_algebra::oracle_dot`/`oracle_sum` and the small
//! projected system through `pyscf_algebra::solve_linear`). Upstream port ref:
//! `pyscf/scf/cphf.py:29-148` (`solve`/`solve_nos1`/`solve_withs1`) +
//! `pyscf/lib/linalg_helper.py:1221` (`lib.krylov`, the Pople 1979 non-symmetric
//! Krylov algorithm).
//!
//! ## D-03 / GRAD-10: ONE solver
//!
//! Both MP2 (`max_cycle=30`) and CCSD (07-08) pass their own `fvind` response
//! operator + RHS into this single [`solve`]. The `single_cphf_impl` structural
//! CI assertion (tests/cphf.rs) guards against a second CPHF `pub fn solve`
//! implementation landing in the crate.
//!
//! ## The equation solved
//!
//! CPHF/CPKS in the (vir × occ) orbital-rotation space solves the response
//! equation `(1 + A)·z = b`, where the matrix `A` is applied ONLY through the
//! caller-supplied matrix-free response operator `fvind` (NEVER materialized —
//! the dense `A` is `O(nocc²·nvir²)`, the RESEARCH anti-pattern). The full
//! upstream `aop` (`cphf.py:73-79`) is
//!
//! ```text
//! vind_vo(z) = (fvind(z) [− z·level_shift]) · e_ai     where e_ai = 1/(e_a − e_i)
//! ```
//!
//! and the RHS base is `b = h1 · (−e_ai)` (`cphf.py:70`). The Pople-1979 Krylov
//! iteration then solves `(1 + vind_vo)·z = b` to `tol`/`max_cycle`.
//!
//! ## Phase-7 path: `s1 = None` (`solve_nos1`)
//!
//! For MP2/CCSD Z-vector gradients the first-order overlap `s1` is zero, so this
//! module ships the `solve_nos1` branch fully. The `solve_withs1` branch
//! (field-dependent, the Hessian future, NOT Phase 7) returns a clear
//! [`GradError`] rather than a wrong tensor.
//!
//! ## Bit-exact discipline (Pitfall 1/2)
//!
//! Every reduction (inner products, the projection update, the `x = Σ c_i x_i`
//! recombination) materializes into a `Vec` then routes through
//! `pyscf_algebra::oracle_dot`/`oracle_sum` — NEVER a bare `+=`. The projected
//! `nd × nd` system `h·c = g` is handed to `pyscf_algebra::solve_linear` (the
//! algebra-wall dense LU). The solver is therefore thread-count invariant under
//! `release-oracle`.

use crate::error::GradError;
use pyscf_algebra::{oracle_dot, oracle_sum, solve_linear};
use pyscf_core::PyscfRsError;

/// Upstream `cphf.solve` default max iterations (`cphf.py:30`). MP2 grad
/// overrides this to 30 (Pitfall 5 / `mp2.py:280`).
pub const DEFAULT_MAX_CYCLE: usize = 50;

/// Upstream `cphf.solve` default convergence tolerance (`cphf.py:30`).
pub const DEFAULT_TOL: f64 = 1e-9;

/// Upstream `cphf.solve` default Krylov-acceleration level shift (`cphf.py:31`).
pub const DEFAULT_LEVEL_SHIFT: f64 = 0.0;

/// Linear-dependency threshold for the Krylov metric (`lib.krylov`
/// `DSOLVE_LINDEP`, `linalg_helper.py`). The iteration terminates when the
/// trial-vector inner product drops below this OR below `tol²`.
const LINDEP: f64 = 1e-15;

/// The caller-supplied matrix-free response operator `fvind` (`cphf.py:34`).
///
/// Given a flattened `(nvir·nocc)` orbital-rotation vector `z` (vir-major:
/// element `(a, i)` at `a·nocc + i`), returns the flattened `(nvir·nocc)`
/// response `Σ_bj [response-kernel]_{ai,bj} · z_{bj}` — WITHOUT the `e_ai`
/// scaling or the identity term (those are applied by [`solve`]). This is the
/// method-specific operator: MP2 (`mp2.py:_response_dm1`) and CCSD (07-08) each
/// supply their own. Returning an `Err` aborts the solve (e.g. a missing-intor
/// cintx-availability error `?`-propagates straight out).
pub type Fvind<'a> = dyn Fn(&[f64]) -> Result<Vec<f64>, PyscfRsError> + 'a;

/// The single matrix-free Krylov CPHF/CPKS solver (`cphf.py:29` `solve`).
///
/// Solves the orbital-response equation `(1 + A)·z = b` for the single RHS `h1`
/// (the Phase-7 Z-vector path is always single-RHS), where `A` is applied ONLY
/// through the matrix-free `fvind`. Ports `cphf.solve` → `solve_nos1` →
/// `lib.krylov` (`nroots = 1`), routing every reduction through
/// `pyscf_algebra::oracle_*` and the projected system through
/// `pyscf_algebra::solve_linear`.
///
/// # Arguments
/// - `fvind`     — the caller-supplied response operator (see [`Fvind`]).
/// - `mo_energy` — the full MO energy spectrum (one per MO).
/// - `mo_occ`    — the MO occupations (occupied `> 0`, virtual `== 0`). The
///   (vir, occ) split is read from this: `nocc = #{occ > 0}`, `nvir = #{occ == 0}`.
/// - `h1`        — the RHS, flattened `(nvir·nocc)` vir-major (element `(a, i)`
///   at `a·nocc + i`), matching the `e_ai` layout.
/// - `s1`        — the first-order overlap. `None` → the `solve_nos1` branch
///   (MP2/CCSD Z-vector; the Phase-7 path). `Some(..)` → `solve_withs1`, NOT
///   supported this phase (returns a clear error).
/// - `max_cycle` — max Krylov iterations. Default [`DEFAULT_MAX_CYCLE`] (= 50);
///   the MP2 caller OVERRIDES this to 30 (Pitfall 5).
/// - `tol`       — convergence tolerance on the residual norm. Default
///   [`DEFAULT_TOL`] (= 1e-9).
/// - `hermi`     — whether `A` is Hermitian. Upstream `solve_nos1` asserts
///   `not hermi`; the Phase-7 response operator is non-symmetric, so `false`.
/// - `level_shift` — added to the diagonal to accelerate convergence
///   (`cphf.py:31`). Default [`DEFAULT_LEVEL_SHIFT`] (= 0).
///
/// # Returns
/// The converged orbital-rotation vector `z`, flattened `(nvir·nocc)` vir-major
/// (the same layout as `h1`). This is the `mo1` of `cphf.solve(...)[0]`.
///
/// # Errors
/// - [`GradError::ShapeMismatch`] if `h1.len() != nvir·nocc` or `mo_energy`/
///   `mo_occ` lengths disagree.
/// - A clear error if `s1.is_some()` (the field-dependent branch is not Phase 7).
/// - [`GradError::NotYetImplemented`]-style convergence failure surfaces as a
///   clear `Core(InvalidMolecule(..))` (never an infinite loop — T-07-21).
/// - Any `Err` returned by `fvind` (e.g. a cintx-availability error)
///   `?`-propagates unchanged.
#[allow(clippy::too_many_arguments)]
pub fn solve(
    fvind: &Fvind<'_>,
    mo_energy: &[f64],
    mo_occ: &[f64],
    h1: &[f64],
    s1: Option<&[f64]>,
    max_cycle: usize,
    tol: f64,
    hermi: bool,
    level_shift: f64,
) -> Result<Vec<f64>, PyscfRsError> {
    if s1.is_some() {
        // solve_withs1 (field-dependent, first-order overlap non-zero) is the
        // Hessian-future branch — NOT Phase 7. Return a clear error rather than
        // a wrong tensor (never a silent no-op).
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            "cphf::solve: the s1=Some (solve_withs1, field-dependent) branch is not \
             implemented this phase — the Phase-7 MP2/CCSD Z-vector path is s1=None \
             (solve_nos1). Pass s1=None."
                .into(),
        )));
    }
    solve_nos1(
        fvind,
        mo_energy,
        mo_occ,
        h1,
        max_cycle,
        tol,
        hermi,
        level_shift,
    )
}

/// `solve_nos1` (`cphf.py:53-83`) — the field-independent (s1 = 0) branch.
///
/// Builds `e_ai = 1/(e_a + level_shift − e_i)`, the RHS base `mo1base = h1·(−e_ai)`,
/// and the matrix-free `aop` `vind_vo`, then runs the Pople-1979 Krylov solve.
#[allow(clippy::too_many_arguments)]
fn solve_nos1(
    fvind: &Fvind<'_>,
    mo_energy: &[f64],
    mo_occ: &[f64],
    h1: &[f64],
    max_cycle: usize,
    tol: f64,
    hermi: bool,
    level_shift: f64,
) -> Result<Vec<f64>, PyscfRsError> {
    if hermi {
        // Upstream `solve_nos1` asserts `not hermi`; the Phase-7 response
        // operator is non-symmetric. Reject rather than silently running a
        // Hermitian shortcut the caller did not validate.
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            "cphf::solve_nos1: hermi=true is not supported (the CPHF response \
             operator is non-symmetric; pass hermi=false)."
                .into(),
        )));
    }
    if mo_energy.len() != mo_occ.len() {
        return Err(GradError::ShapeMismatch {
            expected: mo_energy.len(),
            got: mo_occ.len(),
        }
        .into());
    }

    // (vir, occ) split from the occupations (cphf.py:67-68):
    //   e_i = mo_energy[occ > 0]; e_a = mo_energy[occ == 0].
    let e_i: Vec<f64> = mo_energy
        .iter()
        .zip(mo_occ)
        .filter_map(|(&e, &o)| if o > 0.0 { Some(e) } else { None })
        .collect();
    let e_a: Vec<f64> = mo_energy
        .iter()
        .zip(mo_occ)
        .filter_map(|(&e, &o)| if o == 0.0 { Some(e) } else { None })
        .collect();
    let nocc = e_i.len();
    let nvir = e_a.len();
    let ndim = nvir * nocc;

    if ndim == 0 {
        return Err(GradError::ShapeMismatch {
            expected: 1,
            got: 0,
        }
        .into());
    }
    if h1.len() != ndim {
        return Err(GradError::ShapeMismatch {
            expected: ndim,
            got: h1.len(),
        }
        .into());
    }

    // e_ai[a, i] = 1 / (e_a[a] + level_shift − e_i[i]); flat vir-major
    // (element (a, i) at a*nocc + i), matching the h1 layout (cphf.py:69).
    let mut e_ai = vec![0.0_f64; ndim];
    for a in 0..nvir {
        for i in 0..nocc {
            let denom = e_a[a] + level_shift - e_i[i];
            if denom == 0.0 || !denom.is_finite() {
                return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                    format!(
                        "cphf::solve_nos1: singular orbital-energy denominator \
                         (e_a[{a}]={} + level_shift={level_shift} − e_i[{i}]={}) — \
                         degenerate occ/vir gap?",
                        e_a[a], e_i[i]
                    ),
                )));
            }
            e_ai[a * nocc + i] = 1.0 / denom;
        }
    }

    // mo1base = h1 * -e_ai (cphf.py:70), elementwise.
    let mut mo1base = vec![0.0_f64; ndim];
    for (idx, slot) in mo1base.iter_mut().enumerate() {
        *slot = h1[idx] * -e_ai[idx];
    }

    // The matrix-free aop (cphf.py:73-79): vind_vo(z) = (fvind(z) [− z·level_shift]) · e_ai.
    // Returned WITHOUT the identity term — the Krylov solver below adds the I in (1+a).
    let vind_vo = |z: &[f64]| -> Result<Vec<f64>, PyscfRsError> {
        let mut v = fvind(z)?;
        if v.len() != ndim {
            return Err(GradError::ShapeMismatch {
                expected: ndim,
                got: v.len(),
            }
            .into());
        }
        if level_shift != 0.0 {
            for (vi, zi) in v.iter_mut().zip(z) {
                // v -= z * level_shift (a 2-element ordered add, no bare -=).
                *vi = oracle_sum(&[*vi, -(zi * level_shift)]);
            }
        }
        for (vi, eai) in v.iter_mut().zip(&e_ai) {
            *vi *= eai;
        }
        Ok(v)
    };

    krylov(&vind_vo, &mo1base, ndim, tol, max_cycle)
}

/// Port of `lib.krylov` (`linalg_helper.py:1221`, Pople 1979) for the SINGLE-RHS
/// (`nroots = 1`) case — the only case the Phase-7 Z-vector path needs.
///
/// Solves `(1 + a)·x = b` where `a` is applied ONLY through `aop` (matrix-free).
/// The dense `a` is NEVER materialized. Inner products route through
/// `oracle_dot`; the projected `nd × nd` system `h·c = g` is solved through
/// `pyscf_algebra::solve_linear`; the recombination `x = Σ_i c_i x_i` reduces
/// per-element through `oracle_sum`.
///
/// `b` is the RHS, `ndim` its length. Returns `x` (length `ndim`).
fn krylov(
    aop: &Fvind<'_>,
    b: &[f64],
    ndim: usize,
    tol: f64,
    max_cycle: usize,
) -> Result<Vec<f64>, PyscfRsError> {
    // x1 = b (linalg_helper.py:1275). innerprod[0] = dot(x1, x1) (line 1290).
    let mut x1 = b.to_vec();
    let mut innerprod: Vec<f64> = vec![oracle_dot(&x1, &x1)];

    // Trivial RHS: if the initial residual is below threshold the solution is 0
    // (lines 1293-1295). x0 is always None on the Phase-7 path.
    let tol2 = tol * tol;
    if innerprod[0] < LINDEP || innerprod[0] < tol2 {
        return Ok(vec![0.0_f64; ndim]);
    }

    // The Krylov trial vectors (xs) and their images (ax = aop(xs)).
    let mut xs: Vec<Vec<f64>> = Vec::new();
    let mut ax: Vec<Vec<f64>> = Vec::new();

    // max_cycle = min(max_cycle, ndim) (line 1308) — bounds the iteration; a
    // non-converged solve falls through to the explicit error below (T-07-21:
    // NEVER an infinite loop).
    let cap = max_cycle.min(ndim);
    let mut converged = false;
    for _cycle in 0..cap {
        // axt = aop(x1) (line 1310).
        let axt = aop(&x1)?;
        // xs.extend(x1); ax.extend(axt) (lines 1313-1314).
        xs.push(x1.clone());
        ax.push(axt.clone());

        // x1 = axt − Σ_i xs[i] · (dot(axt, xs[i]) / innerprod[i])  (lines 1318-1322).
        // Build the new x1 element-wise: for each component k,
        //   x1[k] = oracle_sum([axt[k], −xs[0][k]·w0, −xs[1][k]·w1, …]).
        let nd_now = xs.len();
        let mut weights = vec![0.0_f64; nd_now];
        for (i, w) in weights.iter_mut().enumerate() {
            *w = oracle_dot(&axt, &xs[i]) / innerprod[i];
        }
        let mut new_x1 = vec![0.0_f64; ndim];
        let mut terms = Vec::with_capacity(nd_now + 1);
        for (k, slot) in new_x1.iter_mut().enumerate() {
            terms.clear();
            terms.push(axt[k]);
            for i in 0..nd_now {
                terms.push(-(xs[i][k] * weights[i]));
            }
            *slot = oracle_sum(&terms);
        }
        x1 = new_x1;

        // innerprod1 = dot(x1, x1) (line 1330); break on convergence (line 1334).
        let innerprod1 = oracle_dot(&x1, &x1);
        if innerprod1 < LINDEP || innerprod1 < tol2 {
            converged = true;
            break;
        }
        // innerprod.append(innerprod1) (line 1342).
        innerprod.push(innerprod1);
    }

    if !converged {
        // The Python `for…else` raises RuntimeError on non-convergence
        // (line 1345). Surface a clear error — NEVER loop forever (T-07-21).
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!(
                "cphf Krylov solver failed to converge in {} cycles (tol={tol:e}); \
                 check the fvind response operator or raise max_cycle.",
                cap
            ),
        )));
    }

    // Build the projected (nd × nd) system (lines 1347-1367):
    //   h[i,j] = dot(xs[i], ax[j]);  h[i,i] += innerprod[i];  g[i] = dot(b, xs[i]).
    //   c = solve(h, g);  x = Σ_i c[i]·xs[i].
    let nd = xs.len();
    let mut h = vec![0.0_f64; nd * nd]; // row-major for solve_linear.
    for i in 0..nd {
        for j in 0..nd {
            // hermi=false branch (lines 1356-1357): full nd×nd.
            h[i * nd + j] = oracle_dot(&xs[i], &ax[j]);
        }
    }
    // Add the contribution of I in (1+a) (lines 1360-1361): h[i,i] += innerprod[i].
    for i in 0..nd {
        h[i * nd + i] = oracle_sum(&[h[i * nd + i], innerprod[i]]);
    }
    // g[i] = dot(b, xs[i]) (lines 1363-1365).
    let mut g = vec![0.0_f64; nd];
    for (i, gi) in g.iter_mut().enumerate() {
        *gi = oracle_dot(b, &xs[i]);
    }

    // c = solve(h, g) — the small projected system, on the algebra wall (line 1367).
    // An AlgebraError bridges through GradError::Algebra → PyscfRsError.
    let c = solve_linear(&h, &g, nd).map_err(GradError::from)?;

    // x = Σ_i c[i]·xs[i] (_gen_x0, lines 1368). Per-element ordered sum.
    let mut x = vec![0.0_f64; ndim];
    let mut terms = Vec::with_capacity(nd);
    for (k, slot) in x.iter_mut().enumerate() {
        terms.clear();
        for i in 0..nd {
            terms.push(c[i] * xs[i][k]);
        }
        *slot = oracle_sum(&terms);
    }
    Ok(x)
}

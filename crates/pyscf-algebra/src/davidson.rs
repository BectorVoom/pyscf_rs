//! `davidson_nosym1` — the iterative NON-symmetric (non-Hermitian) Davidson
//! eigensolver, plus `pick_real_eigs` (plan 16-03).
//!
//! Port of `pyscf/lib/linalg_helper.py:741-937` and its helpers `_qr`
//! (`:1411`), `_fill_heff` (`:183`), `_outprod_to_subspace`/`_gen_x0`
//! (`:1450`), `_sort_elast` (`:1468`), `_normalize_xt_` (`:1492`) and
//! `pick_real_eigs` (`:593`) with `_eigs_cmplx2real` (`:614`).
//!
//! # Why a new solver rather than an existing one
//!
//! Four Phase-16 plans (16-09, 16-10, 16-11, 16-13) are dead without it, and
//! `PBC-MASTER-PLAN §8.8` costs it at zero:
//!
//! ```text
//! pyscf/pbc/cc/eom_kccsd_ghf.py:128    eig = lib.davidson_nosym1
//! pyscf/pbc/cc/eom_kccsd_ghf.py:1352   eig = lib.davidson_nosym1
//! pyscf/pbc/ci/kcis_rhf.py:97          eig = lib.davidson_nosym1
//! ```
//!
//! * [`crate::eigh_gen`] is **symmetric** (self-adjoint) — ALG-05.
//! * 17-02's `faer` `Eigen` path (`pyscf-pbc-symm/src/group.rs:459`) is a
//!   **dense** general eigenproblem: it forms the whole matrix.
//!
//! Neither is an *iterative, matrix-free* non-symmetric solver, and matrix-free
//! is the entire point: the EOM Hamiltonian is `(nkpts·nocc·nvir)²`-shaped and
//! forming it is what this avoids (`16-REVIEW.md §4.1`).
//!
//! # The algorithm, before the code (16-03 Task 1)
//!
//! * **Subspace expansion.** Each cycle orthonormalises the trial vectors `xt`
//!   against each other (`_qr`, `:1411`), applies `aop` once per trial vector
//!   to get `axt`, and appends both to the growing lists `xs` / `ax`.
//! * **The projected problem.** `fill_heff` (`:183`) fills only the NEW rows
//!   and columns of `heff[i,j] = ⟨xs[i]|ax[j]⟩` — the old block is already
//!   there — and the `space × space` corner is diagonalised **densely and
//!   non-symmetrically** (`scipy.linalg.eig`). `heff` is NOT Hermitian, which
//!   is the whole reason `_fill_heff_hermitian` (`:165`) is a different
//!   function and why this solver exists.
//! * **`max_space` trimming.** `max_space` is raised to
//!   `max_space + (nroots-1)*6` (`:768`) and the run RESTARTS from the current
//!   `x0` whenever `space + nroots > max_space` (`:910`, `fresh_start`). That
//!   restart is the memory bound: `xs` and `ax` together hold
//!   `2 · max_space · nroots` vectors, which on diamond `gth-dzvp` 3×3×3 EA is
//!   **5.4 GiB** (`16-REVIEW.md §4.1`) — so `max_space` is a real knob, not a
//!   constant to inline.
//! * **`lindep` collapse.** Two places drop vectors: `_qr` keeps a vector only
//!   if `⟨x|x⟩ > lindep` (`:1428`), and `_normalize_xt_` (`:1492`) projects the
//!   preconditioned residual against every existing `xs` and keeps it only if
//!   `‖x‖² > lindep`. When nothing survives, the run stops with
//!   `conv = dx_norm < toloose` (`:904-906`).
//! * **Convergence.** `toloose = sqrt(tol)` unless `tol_residual` is given
//!   (`:752-755`); a root is converged when `|de| < tol` AND
//!   `‖A x − e x‖ < toloose` (`:863`).
//! * **`follow_state`** (`:874`) restarts from the PREVIOUS eigenvector when
//!   the residual norm blows up; **`lessio`** (`:770`) trades the stored `ax`
//!   images for a recomputation of `aop(x0)`. Both are ported as parameters
//!   with upstream's defaults (`16-REVIEW.md §7.3`).
//!
//! # Host-only
//!
//! This is control flow over small dense algebra, not a kernel. It runs on the
//! host and names no cubecl type. The projected `heff` is at most
//! `(max_space + nroots)²` and is solved through `faer`, the same direct path
//! 17-02 established for `character_table`.
//!
//! # Conjugation (`16-CONTEXT §3.2`)
//!
//! Every inner product in this module is `⟨x|y⟩ = Σ conj(x)·y` — the
//! Gram–Schmidt overlaps in `_qr`, the `heff` elements in `_fill_heff`, the
//! projections in `_normalize_xt_`, and the residual norms. So the primitive is
//! [`crate::oracle_zdot`] (`zdotc`) at **every** site in this file, and
//! [`crate::oracle_zdot_re`] where only the real part is wanted. **This is the
//! one module in Phase 16 where `oracle_zdot` is the right default**; the CC
//! contractions are mostly unconjugated and want
//! [`crate::oracle_zdotu`] (`15-REVIEW.md D-15-R-02`). Saying so explicitly
//! here is `16-CONTEXT §3.2`'s requirement, not decoration.

use faer::c64;

use crate::complex::CTensor;
use crate::error::AlgebraError;
use crate::{oracle_zdot, oracle_zdot_re};

/// `linalg_helper.py:35` — `DAVIDSON_LINDEP`.
pub const DAVIDSON_LINDEP: f64 = 1e-14;

/// `pick_real_eigs`'s hard-coded imaginary-part threshold
/// (`linalg_helper.py:597`).
pub const PICK_REAL_THRESHOLD: f64 = 1e-3;

/// Parameters of [`davidson_nosym1`].
///
/// Every field is one of upstream's keyword arguments, with upstream's default.
/// `max_space`, `lindep`, `lessio` and `follow_state` are load-bearing, not
/// defaults to be waved through — see the module doc.
#[derive(Debug, Clone)]
pub struct DavidsonOptions {
    /// Eigenvalue-change convergence threshold (`tol=1e-12`).
    pub tol: f64,
    /// Residual-norm threshold. `None` → `sqrt(tol)` (`:752-755`).
    pub tol_residual: Option<f64>,
    /// `max_cycle=50`.
    pub max_cycle: usize,
    /// `max_space=20`, before the `+(nroots-1)*6` widening at `:768`.
    pub max_space: usize,
    /// `lindep=DAVIDSON_LINDEP`.
    pub lindep: f64,
    /// `nroots=1`.
    pub nroots: usize,
    /// `lessio=False` — recompute `aop(x0)` instead of storing `ax` images.
    /// A real memory/recompute trade (`16-REVIEW.md §7.3`); upstream's default
    /// is kept and flipping it is a measurement, not a plan-time decision.
    pub lessio: bool,
    /// `left=False` — also return left eigenvectors (`:925-935`).
    pub left: bool,
    /// `follow_state=FOLLOW_STATE` (`:52`, default `False`).
    pub follow_state: bool,
    /// Whether the caller's vectors are REAL. Drives `pick_real_eigs`'s
    /// `envs['dtype'] == numpy.double` branch (`:610`). Every k-point EOM/CIS
    /// caller is complex, so the default is `false`.
    pub real_dtype: bool,
}

impl Default for DavidsonOptions {
    fn default() -> Self {
        Self {
            tol: 1e-12,
            tol_residual: None,
            max_cycle: 50,
            max_space: 20,
            lindep: DAVIDSON_LINDEP,
            nroots: 1,
            lessio: false,
            left: false,
            follow_state: false,
            real_dtype: false,
        }
    }
}

/// What [`davidson_nosym1`] returns — upstream's
/// `(conv, e, x0)` or `(conv, e, xl, x0)`.
#[derive(Debug, Clone)]
pub struct DavidsonResult {
    /// Per-root convergence flags.
    pub conv: Vec<bool>,
    /// The eigenvalues. `pick_real_eigs` returns `w.real` (`:637`), so these
    /// are real by construction — see [`pick_real_eigs`].
    pub e: Vec<f64>,
    /// Right eigenvectors.
    pub x: Vec<CTensor>,
    /// Left eigenvectors, present only when `left` was requested.
    pub xl: Option<Vec<CTensor>>,
    /// Number of `aop` INVOCATIONS (not vectors). The matrix-free property is
    /// testable through this: a "form the matrix then diagonalise"
    /// implementation would call `aop` `n` times (16-03 test 3).
    pub aop_calls: usize,
    /// Number of individual trial vectors handed to `aop`.
    pub aop_vectors: usize,
    /// Cycles actually run.
    pub cycles: usize,
}

/// The output of a `pick` callback: the selected eigenvalues, the corresponding
/// columns of the projected eigenvector matrix (column-major, `space` rows),
/// and the indices they came from.
#[derive(Debug, Clone)]
pub struct Picked {
    /// Selected eigenvalues, real (`_eigs_cmplx2real` returns `w.real`).
    pub w: Vec<f64>,
    /// `space × w.len()` column-major.
    pub v: Vec<c64>,
    /// Indices into the original spectrum, in the returned order.
    pub idx: Vec<usize>,
}

/// `pick_real_eigs` (`linalg_helper.py:593-612`) with `_eigs_cmplx2real`
/// (`:614-638`).
///
/// Selects the eigenvalues with the smallest imaginary components and orders
/// them by real part. `eom_kccsd_ghf.py:130-136` supplies its own closure and
/// passes it through, and `kcis_rhf.py:97-98` uses this default — so `pick` is
/// a first-class parameter of [`davidson_nosym1`] with this as its supplied
/// default, never a hard-coded filter.
///
/// `w` is the spectrum; `v` is `space × space` column-major; `real_dtype`
/// is upstream's `envs.get('dtype') == numpy.double` test (`:610`), which
/// decides whether the eigenvectors are collapsed to their real parts.
pub fn pick_real_eigs(
    w: &[c64],
    v: &[c64],
    space: usize,
    nroots: usize,
    real_dtype: bool,
) -> Picked {
    let abs_imag: Vec<f64> = w.iter().map(|c| c.im.abs()).collect();
    // max_imag_tol = max(threshold, sort(abs_imag)[min(w.size, nroots) - 1])
    let mut sorted = abs_imag.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let k = w.len().min(nroots).saturating_sub(1);
    let max_imag_tol = PICK_REAL_THRESHOLD.max(sorted.get(k).copied().unwrap_or(0.0));
    let real_idx: Vec<usize> = (0..w.len())
        .filter(|&i| abs_imag[i] <= max_imag_tol)
        .collect();

    // idx = real_idx[w[real_idx].real.argsort()] — a STABLE sort, matching
    // numpy's default `argsort(kind='quicksort')` only in the absence of ties;
    // ties are broken by the original order here so the result is
    // reproducible run to run (§9.3).
    let mut idx = real_idx;
    idx.sort_by(|&a, &b| {
        w[a].re
            .partial_cmp(&w[b].re)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out_w: Vec<c64> = idx.iter().map(|&i| w[i]).collect();
    let mut out_v: Vec<c64> = Vec::with_capacity(idx.len() * space);
    for &i in &idx {
        out_v.extend_from_slice(&v[i * space..(i + 1) * space]);
    }

    if real_dtype {
        // `_eigs_cmplx2real(..., real_eigenvectors=True)` (`:630-637`): for a
        // REAL matrix a conjugate pair contributes two independent real
        // vectors — the real part of the first and the imaginary part of the
        // second — after which the imaginary parts are discarded.
        let degen: Vec<usize> = (0..out_w.len()).filter(|&i| out_w[i].im != 0.0).collect();
        for (pos, &i) in degen.iter().enumerate() {
            if pos % 2 == 1 {
                for r in 0..space {
                    let c = out_v[i * space + r];
                    out_v[i * space + r] = c64::new(c.im, 0.0);
                }
            }
        }
        for c in out_v.iter_mut() {
            *c = c64::new(c.re, 0.0);
        }
    }
    for c in out_w.iter_mut() {
        *c = c64::new(c.re, 0.0);
    }

    Picked {
        w: out_w.iter().map(|c| c.re).collect(),
        v: out_v,
        idx,
    }
}

/// [`eig_general`] over PLANAR complex (`CTensor`) operands — the form the
/// `pyscf-pbc-*` crates speak, so a caller there does not have to name
/// `faer`'s `c64`.
///
/// `a` is COLUMN-MAJOR `n × n` (`a[j*n + i]` is row `i`, column `j`), and the
/// returned eigenvector tensor is column-major too: column `j` is the right
/// eigenvector of eigenvalue `j`.
///
/// # Errors
/// As [`eig_general`], plus a shape check on `a`.
pub fn zeig_general(a: &CTensor, n: usize) -> Result<(CTensor, CTensor), AlgebraError> {
    if a.re.len() != n * n || a.im.len() != n * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("zeig_general: {n}x{n} = {} elements", n * n),
            actual: format!("{} elements", a.re.len()),
        });
    }
    let packed: Vec<c64> = (0..n * n).map(|i| c64::new(a.re[i], a.im[i])).collect();
    let (w, v) = eig_general(&packed, n)?;
    Ok((
        CTensor::from_planes(
            w.iter().map(|z| z.re).collect(),
            w.iter().map(|z| z.im).collect(),
        ),
        CTensor::from_planes(
            v.iter().map(|z| z.re).collect(),
            v.iter().map(|z| z.im).collect(),
        ),
    ))
}

/// Dense general (non-Hermitian) complex eigendecomposition of a column-major
/// `n × n` matrix. `scipy.linalg.eig`'s role at `linalg_helper.py:822`.
///
/// Returns `(w, v)` with `v` column-major, column `j` the right eigenvector of
/// `w[j]`.
///
/// # Errors
/// [`AlgebraError`] when `faer` cannot decompose the matrix.
pub fn eig_general(a: &[c64], n: usize) -> Result<(Vec<c64>, Vec<c64>), AlgebraError> {
    if a.len() != n * n {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("eig_general: {}x{} = {} elements", n, n, n * n),
            actual: format!("{} elements", a.len()),
        });
    }
    let mat = faer::Mat::<c64>::from_fn(n, n, |i, j| a[j * n + i]);
    let eigen = mat.eigen().map_err(|e| {
        AlgebraError::CubeclRuntime(format!(
            "eig_general: faer eigendecomposition failed: {e:?}"
        ))
    })?;
    let s = eigen.S();
    let u = eigen.U();
    let w: Vec<c64> = (0..n).map(|i| s[i]).collect();
    let mut v = Vec::with_capacity(n * n);
    for j in 0..n {
        for i in 0..n {
            v.push(u[(i, j)]);
        }
    }
    Ok((w, v))
}

// --------------------------------------------------------------------------
// Small CTensor helpers. Every inner product is zdotc (`oracle_zdot`).
// --------------------------------------------------------------------------

fn zdotc(x: &CTensor, y: &CTensor) -> c64 {
    let (re, im) = oracle_zdot(x, y);
    c64::new(re, im)
}

fn norm2(x: &CTensor) -> f64 {
    // ⟨x|x⟩ is real by construction; take the real part directly rather than
    // forming the (zero) imaginary one.
    oracle_zdot_re(x, x)
}

/// `x -= s * y`, element-wise complex.
fn axpy_neg(x: &mut CTensor, s: c64, y: &CTensor) {
    for i in 0..x.re.len() {
        let (yr, yi) = (y.re[i], y.im[i]);
        x.re[i] -= s.re * yr - s.im * yi;
        x.im[i] -= s.re * yi + s.im * yr;
    }
}

/// `x += s * y`, element-wise complex.
fn axpy(x: &mut CTensor, s: c64, y: &CTensor) {
    for i in 0..x.re.len() {
        let (yr, yi) = (y.re[i], y.im[i]);
        x.re[i] += s.re * yr - s.im * yi;
        x.im[i] += s.re * yi + s.im * yr;
    }
}

fn scale_real(x: &mut CTensor, s: f64) {
    for i in 0..x.re.len() {
        x.re[i] *= s;
        x.im[i] *= s;
    }
}

/// `_qr(xs, dot, lindep)` (`linalg_helper.py:1411-1433`), returning `qs` only —
/// `davidson_nosym1` uses `_qr(...)[0]` at `:794` and `:801`.
fn qr(xs: &[CTensor], lindep: f64) -> Vec<CTensor> {
    let mut qs: Vec<CTensor> = Vec::with_capacity(xs.len());
    for x in xs {
        let mut xi = x.clone();
        for q in qs.iter() {
            let prod = zdotc(q, &xi);
            axpy_neg(&mut xi, prod, q);
        }
        let innerprod = norm2(&xi);
        if innerprod > lindep {
            scale_real(&mut xi, innerprod.sqrt().recip());
            qs.push(xi);
        }
    }
    qs
}

/// `_fill_heff(heff, xs, ax, xt, axt, dot)` (`linalg_helper.py:183-199`).
///
/// Fills only the NEW rows/columns — the `row0 × row0` corner is already
/// correct from the previous cycle, which is why the subspace matrix costs
/// `O(space · rnow)` inner products and not `O(space²)`.
fn fill_heff(
    heff: &mut [c64],
    ld: usize,
    xs: &[CTensor],
    ax: &[CTensor],
    xt: &[CTensor],
    axt: &[CTensor],
) {
    let nrow = axt.len();
    let row1 = ax.len();
    let row0 = row1 - nrow;
    for (ip, i) in (row0..row1).enumerate() {
        for (jp, j) in (row0..row1).enumerate() {
            heff[i * ld + j] = zdotc(&xt[ip], &axt[jp]);
        }
    }
    for i in 0..row0 {
        for (jp, j) in (row0..row1).enumerate() {
            heff[i * ld + j] = zdotc(&xs[i], &axt[jp]);
            heff[j * ld + i] = zdotc(&xt[jp], &ax[i]);
        }
    }
}

/// `_outprod_to_subspace(v, xs)` / `_gen_x0` (`linalg_helper.py:1436-1450`).
///
/// `v` is `space × nroots` column-major.
fn gen_x0(v: &[c64], space: usize, nroots: usize, xs: &[CTensor]) -> Vec<CTensor> {
    let n = xs[0].re.len();
    let mut out: Vec<CTensor> = (0..nroots).map(|_| CTensor::zeros(n)).collect();
    // Upstream accumulates from `space-1` downwards; the order is observable in
    // the last bits, so it is preserved.
    for i in (0..space).rev() {
        for (k, o) in out.iter_mut().enumerate() {
            axpy(o, v[k * space + i], &xs[i]);
        }
    }
    out
}

/// `_normalize_xt_(xt, xs, threshold, dot)` (`linalg_helper.py:1492-1509`).
fn normalize_xt(xt: Vec<Option<CTensor>>, xs: &[CTensor], threshold: f64) -> (Vec<CTensor>, f64) {
    let mut kept: Vec<CTensor> = xt.into_iter().flatten().collect();
    for xsi in xs {
        for xi in kept.iter_mut() {
            let p = zdotc(xsi, xi);
            axpy_neg(xi, p, xsi);
        }
    }
    let mut norm_min = 1.0_f64;
    let mut out = Vec::with_capacity(kept.len());
    for mut xi in kept {
        let nn = norm2(&xi);
        if nn > threshold {
            let norm = nn.sqrt();
            scale_real(&mut xi, norm.recip());
            norm_min = norm_min.min(norm);
            out.push(xi);
        }
    }
    (out, norm_min)
}

/// `_sort_elast(elast, conv_last, vlast, v, log)` (`linalg_helper.py:1468-1490`).
///
/// `vlast` is `head × nroots`, `v` is `space × nroots`, both column-major.
fn sort_elast(
    elast: &[f64],
    conv_last: &[bool],
    vlast: &[c64],
    head: usize,
    v: &[c64],
    space: usize,
    nroots: usize,
) -> (Vec<f64>, Vec<bool>) {
    let mut e = vec![0.0_f64; nroots];
    let mut conv = vec![false; nroots];
    for i in 0..nroots {
        // ovlp[i, j] = |Σ_r conj(v[r, i]) * vlast[r, j]|
        let mut best = 0.0_f64;
        let mut arg = 0_usize;
        for j in 0..nroots {
            let mut acc = c64::new(0.0, 0.0);
            for r in 0..head.min(space) {
                acc += v[i * space + r].conj() * vlast[j * head + r];
            }
            let m = acc.norm();
            if m > best {
                best = m;
                arg = j;
            }
        }
        if best > 0.5 {
            e[i] = elast[arg];
            conv[i] = conv_last[arg];
        } else {
            e[i] = 0.0;
            conv[i] = false;
        }
    }
    (e, conv)
}

/// `davidson_nosym1` (`linalg_helper.py:741-937`).
///
/// `aop` is applied to a LIST of trial vectors, exactly as upstream's is, so a
/// caller that can batch its matvec does not lose that. `precond(residual, e0,
/// x0)` is upstream's three-argument preconditioner (`:891`).
///
/// `pick` selects and orders the projected spectrum; pass [`pick_real_eigs`]
/// for upstream's default.
///
/// # Errors
/// [`AlgebraError`] if `x0` is empty, its vectors disagree in length, or the
/// projected eigenproblem cannot be solved.
pub fn davidson_nosym1<A, P, K>(
    mut aop: A,
    x0: Vec<CTensor>,
    precond: P,
    opts: &DavidsonOptions,
    pick: K,
) -> Result<DavidsonResult, AlgebraError>
where
    A: FnMut(&[CTensor]) -> Vec<CTensor>,
    P: Fn(&CTensor, f64, &CTensor) -> CTensor,
    K: Fn(&[c64], &[c64], usize, usize, bool) -> Picked,
{
    if x0.is_empty() {
        return Err(AlgebraError::ShapeMismatch {
            expected: "davidson_nosym1: at least one initial guess vector".to_string(),
            actual: "0".to_string(),
        });
    }
    let n = x0[0].re.len();
    if x0.iter().any(|v| v.re.len() != n || v.im.len() != n) {
        return Err(AlgebraError::ShapeMismatch {
            expected: format!("davidson_nosym1: every guess vector of length {n}"),
            actual: "vectors of differing lengths".to_string(),
        });
    }
    let nroots = opts.nroots.max(1);
    let toloose = opts.tol_residual.unwrap_or_else(|| opts.tol.sqrt());
    // `:768` — max_space = max_space + (nroots-1)*6.
    let max_space = opts.max_space + (nroots - 1) * 6;
    let ld = max_space + nroots;

    let mut x0 = x0;
    let mut xs: Vec<CTensor> = Vec::new();
    let mut ax: Vec<CTensor> = Vec::new();
    let mut heff = vec![c64::new(0.0, 0.0); ld * ld];
    let mut space = 0_usize;
    let mut fresh_start = true;
    let mut xt: Vec<CTensor> = Vec::new();
    let mut e: Vec<f64> = Vec::new();
    let mut v: Vec<c64> = Vec::new();
    let mut elast: Option<Vec<f64>> = None;
    let mut conv = vec![false; nroots];
    let mut conv_last = vec![false; nroots];
    let mut max_dx_last = 1e9_f64;
    let mut aop_calls = 0_usize;
    let mut aop_vectors = 0_usize;
    let mut cycles = 0_usize;
    let mut dx_norm = vec![0.0_f64; nroots];

    for icyc in 0..opts.max_cycle {
        cycles = icyc + 1;
        let was_fresh = fresh_start;
        if fresh_start {
            xs.clear();
            ax.clear();
            space = 0;
            // `:794` — orthogonalise the guesses: the subspace basis must be
            // orthogonal even though `x0` from `pick` need not be.
            xt = qr(&x0, opts.lindep);
            x0 = Vec::new();
            max_dx_last = 1e9;
        } else if xt.len() > 1 {
            xt = qr(&xt, opts.lindep);
            xt.truncate(40); // `:801` — 40 trial vectors at most.
        }
        if xt.is_empty() {
            break;
        }

        let axt = aop(&xt);
        aop_calls += 1;
        aop_vectors += xt.len();
        if axt.len() != xt.len() {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("davidson_nosym1: {} aop images", xt.len()),
                actual: format!("{}", axt.len()),
            });
        }

        // `elast`/`vlast`/`conv_last` are the PREVIOUS cycle's, captured before
        // this cycle overwrites them (`:817-819`). `vlast` has `head` rows,
        // because the previous cycle's `space` is this cycle's `head`.
        let head = space;
        let elast_prev = elast.clone();
        let vlast_prev = v.clone();
        conv_last = conv.clone();

        for (k, xi) in xt.iter().enumerate() {
            xs.push(xi.clone());
            ax.push(axt[k].clone());
        }
        space += xt.len();
        if space > ld {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("davidson_nosym1: subspace at most max_space+nroots = {ld}"),
                actual: format!("{space}"),
            });
        }

        // Upstream appends `xt`/`axt` to `xs`/`ax` BEFORE calling `_fill_heff`,
        // which then derives `row0 = len(ax) - len(axt)` itself (`:184-186`).
        fill_heff(&mut heff, ld, &xs, &ax, &xt, &axt);
        xt = Vec::new();

        // Dense NON-symmetric solve of the `space × space` corner (`:822`).
        // `heff` is not Hermitian, which is why `_fill_heff_hermitian`
        // (`linalg_helper.py:165`) is a different function and why this solver
        // exists at all.
        let mut sub = vec![c64::new(0.0, 0.0); space * space];
        for i in 0..space {
            for j in 0..space {
                sub[j * space + i] = heff[i * ld + j];
            }
        }
        let (w, vv) = eig_general(&sub, space)?;
        let picked = pick(&w, &vv, space, nroots, opts.real_dtype);
        if picked.w.is_empty() {
            return Err(AlgebraError::CubeclRuntime(
                "davidson_nosym1: the pick callback found no eigenvalues".to_string(),
            ));
        }

        let take = nroots.min(picked.w.len());
        e = picked.w[..take].to_vec();
        v = picked.v[..take * space].to_vec();
        conv = vec![false; e.len()];

        // `:826-830` / `:834-843` — reorder the previous eigenvalues onto the
        // current ones before differencing, because Davidson states flip.
        let de: Vec<f64> = match &elast_prev {
            None => e.clone(),
            Some(el) if el.len() != e.len() || head == 0 => e.clone(),
            Some(el) => {
                // `_sort_elast` also returns the reordered `conv_last`, which
                // upstream consumes ONLY for the "root %d converged" debug line
                // (`:864-866`); it never feeds the numerics, and this port
                // emits no such line, so it is deliberately dropped here.
                let (sorted_e, _sorted_conv) = if was_fresh {
                    (el.clone(), conv_last.clone())
                } else {
                    sort_elast(el, &conv_last, &vlast_prev, head, &v, space, e.len())
                };
                e.iter().zip(sorted_e.iter()).map(|(a, b)| a - b).collect()
            }
        };

        let x0_new = gen_x0(&v, space, e.len(), &xs);
        // `:846-849` — `lessio` trades the stored `ax` images for a
        // recomputation. A real memory/recompute knob (`16-REVIEW.md §7.3`).
        let ax0 = if opts.lessio {
            aop_calls += 1;
            aop_vectors += x0_new.len();
            aop(&x0_new)
        } else {
            gen_x0(&v, space, e.len(), &ax)
        };

        dx_norm = vec![0.0_f64; e.len()];
        let mut residual: Vec<Option<CTensor>> = vec![None; e.len()];
        for (k, &ek) in e.iter().enumerate() {
            let mut r = ax0[k].clone();
            axpy_neg(&mut r, c64::new(ek, 0.0), &x0_new[k]);
            dx_norm[k] = norm2(&r).sqrt();
            conv[k] = de[k].abs() < opts.tol && dx_norm[k] < toloose;
            residual[k] = Some(r);
        }
        x0 = x0_new;
        elast = Some(e.clone());

        let max_dx_norm = dx_norm.iter().cloned().fold(0.0_f64, f64::max);
        if conv.iter().all(|&c| c) {
            break;
        }
        if opts.follow_state
            && max_dx_norm > 1.0
            && max_dx_norm / max_dx_last > 3.0
            && space > e.len() + 4
        {
            // `:874-881` — a large residual means the state was lost; restore
            // the PREVIOUS eigenvector and restart the subspace.
            if let Some(vl) = vlast_prev.get(..head * e.len())
                && head > 0
            {
                x0 = gen_x0(vl, head, e.len(), &xs);
                fresh_start = true;
                continue;
            }
        }

        // `:887-897` — precondition each unconverged residual, or drop it when
        // it is already linearly dependent on the subspace.
        let e0 = e[0];
        let mut next: Vec<Option<CTensor>> = vec![None; e.len()];
        for (k, r) in residual.into_iter().enumerate() {
            if conv[k] {
                continue;
            }
            let Some(r) = r else { continue };
            if dx_norm[k] * dx_norm[k] > opts.lindep {
                let mut p = precond(&r, e0, &x0[k]);
                let nn = norm2(&p);
                if nn > 0.0 {
                    scale_real(&mut p, nn.sqrt().recip());
                }
                next[k] = Some(p);
            }
        }
        let (kept, _norm_min) = normalize_xt(next, &xs, opts.lindep);
        if kept.is_empty() {
            // `:903-906` — linear dependency in the trial subspace: stop, and
            // report convergence by residual norm alone.
            conv = dx_norm.iter().map(|&d| d < toloose).collect();
            break;
        }
        xt = kept;

        max_dx_last = max_dx_norm;
        // `:910` — the `max_space` trim: restart from the current `x0` rather
        // than grow the subspace past its ceiling. This IS the memory bound.
        fresh_start = space + e.len() > max_space;
    }

    let xl = if opts.left {
        // `:925-932`. faer exposes right eigenvectors only, and the left
        // eigenvectors of `H` are the right eigenvectors of `Hᴴ` with
        // conjugated eigenvalues: `vlᴴ H = λ vlᴴ  ⟺  Hᴴ vl = conj(λ) vl`.
        let mut sub = vec![c64::new(0.0, 0.0); space * space];
        for i in 0..space {
            for j in 0..space {
                sub[j * space + i] = heff[j * ld + i].conj();
            }
        }
        let (wl, vl) = eig_general(&sub, space)?;
        let wl: Vec<c64> = wl.iter().map(|c| c.conj()).collect();
        let pl = pick(&wl, &vl, space, nroots, opts.real_dtype);
        let take = nroots.min(pl.w.len());
        Some(gen_x0(&pl.v[..take * space], space, take, &xs))
    } else {
        None
    };

    Ok(DavidsonResult {
        conv,
        e,
        x: x0,
        xl,
        aop_calls,
        aop_vectors,
        cycles,
    })
}

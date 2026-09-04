//! Lattice sums — the geometry-only core of `pyscf/pbc/tools/pbc.py`.
//!
//! Ports `get_lattice_Ls` (`pbc.py:601-661`), `check_lattice_sum_range`
//! (`pbc.py:663-676`), `get_monkhorst_pack_size` (`pbc.py:587-599`) and
//! `round_to_cell0` (`pbc.py:836-840`).
//!
//! # Why the `Cell` does not appear here
//!
//! The crate DAG (PBC-MASTER-PLAN §4) runs
//! `pyscf-pbc-lib -> pyscf-pbc-tools -> pyscf-pbc-gto`, so this crate cannot
//! name [`Cell`](../../pyscf_pbc_gto/struct.Cell.html). Upstream's functions
//! take a `cell` but only ever read `lattice_vectors()`,
//! `get_scaled_atom_coords()`, `atom_coords()`, `natm`, `rcut`, `dimension` and
//! `low_dim_ft_type` off it — all plain geometry. So the loop bodies live here
//! over explicit arguments, and `pyscf_pbc_gto::lattice` holds the thin
//! `cell`-taking wrappers that supply them. This is the same split plan 09-04
//! used for `mesh::cutoff_to_mesh` / `Cell::cutoff_to_mesh`.

use crate::mat3::norm3;
use pyscf_pbc_lib::kpts_helper::{intersection, round_to_fbz};

/// Row 2 of `R` from the QR factorization of the `3 x n` matrix whose COLUMNS
/// are `cols`.
///
/// The 3x3 twin [`crate::mesh::qr_r22_abs`] (plan 09-04) returns only
/// `|R[2,2]|`; `find_boundary` inside `get_lattice_Ls` needs `R[2,2]` AND
/// `R[2,3..]`, and its matrix is `3 x 6`, so the same two Householder
/// reflections are applied here over a variable column count. Note that
/// [`pyscf_algebra::qr`] cannot serve: its locked signature accepts only a
/// SQUARE matrix.
///
/// Reflections are applied in the order LAPACK's `dgeqrf` applies them, so this
/// tracks `np.linalg.qr` to ~1 ULP. Only absolute values of the returned entries
/// are ever consumed, so the Householder sign convention stays unobservable.
pub fn qr_row2(cols: &[[f64; 3]]) -> Vec<f64> {
    let n = cols.len();
    // Working copy in [row][col] order: a[i][j] = cols[j][i].
    let mut a: [Vec<f64>; 3] = [vec![0.0; n], vec![0.0; n], vec![0.0; n]];
    for (j, c) in cols.iter().enumerate() {
        for (i, v) in c.iter().enumerate() {
            a[i][j] = *v;
        }
    }

    // Householder reflections on columns 0 and 1; with only 3 rows that is
    // enough to leave row 2 upper-triangular.
    for k in 0..2 {
        let mut x = [0.0_f64; 3];
        let mut nx2 = 0.0;
        for i in k..3 {
            x[i] = a[i][k];
            nx2 += a[i][k] * a[i][k];
        }
        let nx = nx2.sqrt();
        if nx == 0.0 {
            continue;
        }
        // LAPACK dlarfg convention: beta = -sign(x0) * ||x||.
        let beta = if x[k] >= 0.0 { -nx } else { nx };
        let mut v = x;
        v[k] -= beta;
        let vv: f64 = (k..3).map(|i| v[i] * v[i]).sum();
        if vv == 0.0 {
            continue;
        }
        // f[j] = 2 * (v . a[.., j]) / (v . v), then a[.., j] -= f[j] * v.
        let mut f = vec![0.0_f64; n];
        for (j, fj) in f.iter_mut().enumerate().skip(k) {
            let dot: f64 = (k..3).map(|i| v[i] * a[i][j]).sum();
            *fj = 2.0 * dot / vv;
        }
        for i in k..3 {
            let vi = v[i];
            for (j, fj) in f.iter().enumerate().skip(k) {
                a[i][j] -= fj * vi;
            }
        }
    }
    std::mem::take(&mut a[2])
}

/// `find_boundary(a)` from `pbc.py:634-638`.
///
/// `aR = np.vstack([a, dR_basis])` is `6 x 3`; `np.linalg.qr(aR.T)[1]` is the
/// `3 x 6` reduced `R`; `ub = (rcut + abs(r[2,3:]).sum()) / abs(r[2,2])`.
fn find_boundary(a_perm: &[[f64; 3]; 3], dr_basis: &[[f64; 3]; 3], rcut: f64) -> f64 {
    let cols = [
        a_perm[0],
        a_perm[1],
        a_perm[2],
        dr_basis[0],
        dr_basis[1],
        dr_basis[2],
    ];
    let r2 = qr_row2(&cols);
    let tail: f64 = r2[3..].iter().map(|v| v.abs()).sum();
    (rcut + tail) / r2[2].abs()
}

/// The (Cartesian, unitful) lattice translation vectors for nearby images —
/// the geometry core of `get_lattice_Ls` (`pbc.py:601-661`).
///
/// # Arguments
/// * `a` — lattice vectors in Bohr, one per ROW (`cell.lattice_vectors()`).
/// * `scaled_atom_coords` — `cell.get_scaled_atom_coords()`.
/// * `atom_coords` — `cell.atom_coords()` in Bohr; only used by `discard`.
/// * `rcut` — lattice-summation cutoff in Bohr (already defaulted by the caller).
/// * `dimension` — the RESOLVED dimension (`pbc.py:609-614` runs in the wrapper,
///   because the default depends on `cell.low_dim_ft_type`).
/// * `discard` — drop images farther than `rcut + max_ij |r_i - r_j|`.
///
/// Upstream's unused `nimgs` keyword is not reproduced: `pbc.py:601` accepts it
/// and the body never reads it.
///
/// The result is NOT sorted and `Ls[0]` is NOT the origin — it is the
/// `cartesian_prod` starting at `-bounds`. The origin is always present.
pub fn get_lattice_ls(
    a: &[[f64; 3]; 3],
    scaled_atom_coords: &[[f64; 3]],
    atom_coords: &[[f64; 3]],
    rcut: f64,
    dimension: usize,
    discard: bool,
) -> Vec<[f64; 3]> {
    // pbc.py:619-620
    if dimension == 0 || rcut <= 0.0 || atom_coords.is_empty() {
        return vec![[0.0; 3]];
    }
    let dim = dimension.min(3);

    // pbc.py:624-629 — ovlp_penalty = scaled.max(0) - scaled.min(0), over the
    // first `dimension` columns only; dR = ovlp_penalty.dot(a[:dimension]).
    let mut ovlp_penalty = [0.0_f64; 3];
    for (d, o) in ovlp_penalty.iter_mut().enumerate().take(dim) {
        let mut hi = f64::NEG_INFINITY;
        let mut lo = f64::INFINITY;
        for s in scaled_atom_coords {
            hi = hi.max(s[d]);
            lo = lo.min(s[d]);
        }
        *o = hi - lo;
    }
    let mut dr = [0.0_f64; 3];
    for (j, drj) in dr.iter_mut().enumerate() {
        let mut acc = 0.0;
        for d in 0..dim {
            acc += ovlp_penalty[d] * a[d][j];
        }
        *drj = acc;
    }
    // dR_basis = np.diag(dR)
    let dr_basis = [[dr[0], 0.0, 0.0], [0.0, dr[1], 0.0], [0.0, 0.0, dr[2]]];

    // pbc.py:640-648 — one boundary per axis, from a cyclic permutation of `a`.
    let xb = find_boundary(&[a[1], a[2], a[0]], &dr_basis, rcut);
    let yb = if dim > 1 {
        find_boundary(&[a[2], a[0], a[1]], &dr_basis, rcut)
    } else {
        0.0
    };
    let zb = if dim > 2 {
        find_boundary(a, &dr_basis, rcut)
    } else {
        0.0
    };
    // bounds = np.ceil([xb, yb, zb]).astype(int)
    let bounds = [xb.ceil() as i64, yb.ceil() as i64, zb.ceil() as i64];

    // pbc.py:650-654 — Ts = cartesian_prod(-b..=b per axis) (last index fastest),
    // Ls = Ts[:, :dimension] . a[:dimension].
    let n = bounds
        .iter()
        .map(|b| (2 * b + 1).max(0) as usize)
        .product::<usize>();
    let mut ls = Vec::with_capacity(n);
    for tx in -bounds[0]..=bounds[0] {
        for ty in -bounds[1]..=bounds[1] {
            for tz in -bounds[2]..=bounds[2] {
                let t = [tx as f64, ty as f64, tz as f64];
                let mut l = [0.0_f64; 3];
                for (j, lj) in l.iter_mut().enumerate() {
                    let mut acc = 0.0;
                    for d in 0..dim {
                        acc += t[d] * a[d][j];
                    }
                    *lj = acc;
                }
                ls.push(l);
            }
        }
    }

    // pbc.py:656-661 — drop images that cannot reach any atom pair.
    if discard && ls.len() > 1 {
        let dist_max = max_atom_pair_distance(atom_coords);
        let limit = rcut + dist_max;
        ls.retain(|l| norm3(l) < limit);
    }
    ls
}

/// `np.linalg.norm(r[:,None] - r, axis=2).max()` — the largest distance between
/// any two atoms of the unit cell (`pbc.py:657-659`).
pub fn max_atom_pair_distance(atom_coords: &[[f64; 3]]) -> f64 {
    let mut dist_max = 0.0_f64;
    for ri in atom_coords {
        for rj in atom_coords {
            let d = [ri[0] - rj[0], ri[1] - rj[1], ri[2] - rj[2]];
            dist_max = dist_max.max(norm3(&d));
        }
    }
    dist_max
}

/// Whether the lattice-summation range `ls` is sufficient — the geometry core
/// of `check_lattice_sum_range` (`pbc.py:663-676`).
///
/// Returns the minimum distance between an atom of the primary cell and an atom
/// of a lattice image that `ls` does NOT include. `ls_full` must be the
/// reference (wider, un-discarded) image list; the wrapper builds it with
/// `rcut = cell.rcut * 1.5, discard = false`.
///
/// Returns `f64::INFINITY` when `ls` already covers every row of `ls_full`
/// (upstream would raise on `min()` of an empty array).
pub fn check_lattice_sum_range(
    ls_full: &[[f64; 3]],
    ls: &[[f64; 3]],
    atom_coords: &[[f64; 3]],
) -> f64 {
    // Ls_idx = intersection(Ls_full, Ls);
    // Ls_remaining = np.setdiff1d(np.arange(len(Ls_full)), Ls_idx)
    let covered = intersection(ls_full, ls);
    let mut min_dist = f64::INFINITY;
    for (i, lf) in ls_full.iter().enumerate() {
        if covered.binary_search(&i).is_ok() {
            continue;
        }
        // atoms_outside = Ls_full[remaining, None] + atom_coords
        for r in atom_coords {
            let outside = [lf[0] + r[0], lf[1] + r[1], lf[2] + r[2]];
            for r0 in atom_coords {
                let d = [outside[0] - r0[0], outside[1] - r0[1], outside[2] - r0[2]];
                min_dist = min_dist.min(norm3(&d));
            }
        }
    }
    min_dist
}

/// Monkhorst-Pack mesh size behind a k-point list — the core of
/// `get_monkhorst_pack_size` (`pbc.py:587-599`).
///
/// Takes k-points ALREADY scaled by the reciprocal lattice
/// (`cell.get_scaled_kpts(kpts)`), so this crate does not need a second copy of
/// that transform; `pyscf_pbc_gto::lattice::get_monkhorst_pack_size` is the
/// `cell`-taking wrapper.
///
/// # Panics
/// Never. Upstream's `assert kpts.shape[0] < 1/min_tol` is reported through the
/// `Err` arm instead.
///
/// # Errors
/// [`pyscf_core::CoreError::InvalidMolecule`] when `nkpts >= 1/tol`, upstream's
/// assertion.
pub fn monkhorst_pack_size_from_scaled(
    scaled_kpts: &[[f64; 3]],
    tol: f64,
) -> Result<[usize; 3], pyscf_core::PyscfRsError> {
    let nk = scaled_kpts.len();
    let min_tol = tol;
    // pbc.py:590 — assert kpts.shape[0] < 1/min_tol
    if (nk as f64) >= 1.0 / min_tol {
        return Err(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(format!(
                "get_monkhorst_pack_size: {nk} k-points needs tol < {}, got {min_tol:e}",
                1.0 / nk as f64
            )),
        ));
    }
    if nk == 1 {
        return Ok([1, 1, 1]);
    }
    // tol = max(10**(-int(-np.log10(1/nkpts))-2), min_tol)
    // `int()` truncates toward zero; ported through the same 1/nkpts round-trip
    // so the floating-point value of the log matches upstream bit for bit.
    let e = (-(1.0 / nk as f64).log10()).trunc() as i32;
    let tol = 10.0_f64.powi(-e - 2).max(min_tol);

    let mut nks = [0_usize; 3];
    for (axis, nk_axis) in nks.iter_mut().enumerate() {
        // np.sort(skpts.T) sorts each of the three coordinate rows.
        let mut col: Vec<f64> = scaled_kpts.iter().map(|k| k[axis]).collect();
        col.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        // np.count_nonzero(abs(ski[1:] - ski[:-1]) > tol) + 1
        let steps = col.windows(2).filter(|w| (w[1] - w[0]).abs() > tol).count();
        *nk_axis = steps + 1;
    }
    Ok(nks)
}

/// Round scaled coordinates into the reference unit cell `[0, 1)`.
/// Ports `pbc.py:836-840` — a `round_to_fbz(r, wrap_around=False, tol)` call.
pub fn round_to_cell0(r: &[[f64; 3]], tol: f64) -> Vec<[f64; 3]> {
    round_to_fbz(r, false, tol)
}

/// `round_to_cell0` with upstream's default `tol = 1e-6`.
pub fn round_to_cell0_default(r: &[[f64; 3]]) -> Vec<[f64; 3]> {
    round_to_cell0(r, 1e-6)
}

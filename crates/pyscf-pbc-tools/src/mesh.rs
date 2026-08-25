//! FFT-mesh <-> kinetic-energy-cutoff conversions.
//!
//! Line-by-line port of `pyscf/pbc/tools/pbc.py:787-840`:
//! `cutoff_to_mesh`, `mesh_to_cutoff`, `cutoff_to_gs`, `gs_to_cutoff`.
//!
//! # The `qr(...)[1][2,2]` trick
//!
//! Upstream needs, for each axis, the smallest `|x*b0 + y*b1 + z*b2|` that a
//! one-step move along that axis can produce — i.e. the height of the
//! reciprocal-cell parallelepiped perpendicular to the OTHER two vectors. It
//! gets that by QR-factorizing the 3x3 whose columns are the two other
//! reciprocal vectors followed by the axis vector and reading `R[2,2]`:
//!
//! ```text
//! rx = qr(b[[1,2,0]].T)[1][2,2]   # columns b1, b2, b0
//! ry = qr(b[[2,0,1]].T)[1][2,2]   # columns b2, b0, b1
//! rz = qr(b.T       )[1][2,2]     # columns b0, b1, b2
//! ```
//!
//! Both call sites use only `|R[2,2]|` (`cutoff_to_mesh` takes `np.abs`,
//! `mesh_to_cutoff` squares it), so [`qr_r22_abs`] returns the magnitude and
//! the Householder sign convention never becomes observable.

use crate::mat3::{cross3, det3, inv3, norm3, transpose3};
use pyscf_core::PyscfRsError;
use std::f64::consts::PI;

/// `|R[2,2]|` of the QR factorization of the 3x3 matrix whose **columns** are
/// `cols[0]`, `cols[1]`, `cols[2]`.
///
/// Two Householder reflections, in the same order LAPACK's `dgeqrf` applies
/// them, so this tracks `np.linalg.qr` to ~1 ULP. Geometrically the result is
/// the component of `cols[2]` perpendicular to the plane spanned by `cols[0]`
/// and `cols[1]`; [`qr_r22_abs_closed_form`] is that identity written out and
/// is the cross-check in `tests/mesh.rs`.
pub fn qr_r22_abs(cols: &[[f64; 3]; 3]) -> f64 {
    // Working copy in [row][col] order: a[i][j] = cols[j][i].
    let mut a = [[0.0_f64; 3]; 3];
    for (j, c) in cols.iter().enumerate() {
        for (i, v) in c.iter().enumerate() {
            a[i][j] = *v;
        }
    }

    // Householder reflections on columns 0 and 1; column 2's trailing entry is
    // R[2,2].
    for k in 0..2 {
        // x = a[k.., k]
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
        let mut f = [0.0_f64; 3];
        for (j, fj) in f.iter_mut().enumerate().skip(k) {
            let dot: f64 = (k..3).map(|i| v[i] * a[i][j]).sum();
            *fj = 2.0 * dot / vv;
        }
        for (i, row) in a.iter_mut().enumerate().skip(k) {
            let vi = v[i];
            for (j, fj) in f.iter().enumerate().skip(k) {
                row[j] -= fj * vi;
            }
        }
    }
    a[2][2].abs()
}

/// The closed-form value of [`qr_r22_abs`]: `|det(M)| / ||c0 x c1||`.
///
/// `|R00| = ||c0||` and `|R11| = ||c0 x c1|| / ||c0||`, and the product of the
/// three `|R|` diagonal entries is `|det(M)|`. Used as the independent
/// cross-check on the Householder implementation (tier-1 invariant).
pub fn qr_r22_abs_closed_form(cols: &[[f64; 3]; 3]) -> f64 {
    let m = [
        [cols[0][0], cols[1][0], cols[2][0]],
        [cols[0][1], cols[1][1], cols[2][1]],
        [cols[0][2], cols[1][2], cols[2][2]],
    ];
    det3(&m).abs() / norm3(&cross3(&cols[0], &cols[1]))
}

/// `b = 2*pi*inv(a.T)` — the reciprocal lattice, one vector per ROW.
///
/// # Errors
/// [`pyscf_core::CoreError::InvalidMolecule`] if the lattice is singular.
fn reciprocal_2pi(a: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3], PyscfRsError> {
    let inv = inv3(&transpose3(a))?;
    let mut b = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            b[i][j] = 2.0 * PI * inv[i][j];
        }
    }
    Ok(b)
}

/// `[|rx|, |ry|, |rz|]` — the three `qr(...)[1][2,2]` magnitudes shared by
/// [`cutoff_to_mesh`] and [`mesh_to_cutoff`] (`pbc.py:806-809` / `:820-823`).
///
/// # Errors
/// [`pyscf_core::CoreError::InvalidMolecule`] if the lattice is singular.
pub fn qr_heights(a: &[[f64; 3]; 3]) -> Result<[f64; 3], PyscfRsError> {
    let b = reciprocal_2pi(a)?;
    // b[[1,2,0]].T -> columns b1, b2, b0, and so on.
    let rx = qr_r22_abs(&[b[1], b[2], b[0]]);
    let ry = qr_r22_abs(&[b[2], b[0], b[1]]);
    let rz = qr_r22_abs(&[b[0], b[1], b[2]]);
    Ok([rx, ry, rz])
}

/// Convert a kinetic-energy cutoff to an FFT mesh.
///
/// Ports `pbc.py:787-811`. Uses `KE = k^2 / 2` with `k_max ~ pi / spacing`:
/// searches the minimal `x,y,z` with `|x*b0 + y*b1 + z*b2|^2 > 2*cutoff`.
///
/// `a` is the real-space lattice in Bohr, one vector per ROW; `cutoff` is in
/// Hartree. The returned mesh is `ceil(Gmax) * 2 + 1` per axis, so it is
/// always odd and at least 1.
///
/// # Errors
/// [`pyscf_core::CoreError::InvalidMolecule`] if the lattice is singular, or if
/// `cutoff` is negative / non-finite (upstream would silently produce NaN).
pub fn cutoff_to_mesh(a: &[[f64; 3]; 3], cutoff: f64) -> Result<[usize; 3], PyscfRsError> {
    if !cutoff.is_finite() || cutoff < 0.0 {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!("cutoff_to_mesh: ke_cutoff must be finite and >= 0, got {cutoff}"),
        )));
    }
    let r = qr_heights(a)?;
    let mut mesh = [0_usize; 3];
    for (i, m) in mesh.iter_mut().enumerate() {
        // Gmax = (2*cutoff)**.5 / np.abs([rx, ry, rz])
        let gmax = (2.0 * cutoff).sqrt() / r[i].abs();
        // mesh = np.ceil(Gmax).astype(int) * 2 + 1
        let n = gmax.ceil();
        if !n.is_finite() {
            return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
                format!("cutoff_to_mesh: non-finite Gmax on axis {i} (degenerate lattice?)"),
            )));
        }
        *m = (n as usize) * 2 + 1;
    }
    Ok(mesh)
}

/// Convert a mesh back to the kinetic-energy cutoff it resolves, per axis.
///
/// Ports `pbc.py:813-828`. `gs = (mesh - 1) // 2`, `Gmax = gs * r`,
/// `ke = Gmax^2 / 2`.
///
/// # Errors
/// [`pyscf_core::CoreError::InvalidMolecule`] if the lattice is singular.
pub fn mesh_to_cutoff(a: &[[f64; 3]; 3], mesh: [usize; 3]) -> Result<[f64; 3], PyscfRsError> {
    let r = qr_heights(a)?;
    let mut ke = [0.0_f64; 3];
    for (i, k) in ke.iter_mut().enumerate() {
        // gs = (np.asarray(mesh) - 1) // 2  — floor division on non-negative ints.
        let gs = mesh[i].saturating_sub(1) / 2;
        let gmax = gs as f64 * r[i];
        *k = gmax * gmax / 2.0;
    }
    Ok(ke)
}

/// Deprecated upstream, kept for parity: `[n // 2 for n in cutoff_to_mesh(a, cutoff)]`.
/// Ports `pbc.py:830-832`.
///
/// # Errors
/// As [`cutoff_to_mesh`].
pub fn cutoff_to_gs(a: &[[f64; 3]; 3], cutoff: f64) -> Result<[usize; 3], PyscfRsError> {
    let mesh = cutoff_to_mesh(a, cutoff)?;
    Ok([mesh[0] / 2, mesh[1] / 2, mesh[2] / 2])
}

/// Deprecated upstream, kept for parity: `mesh_to_cutoff(a, [2*n+1 for n in gs])`.
/// Ports `pbc.py:834-836`.
///
/// # Errors
/// As [`mesh_to_cutoff`].
pub fn gs_to_cutoff(a: &[[f64; 3]; 3], gs: [usize; 3]) -> Result<[f64; 3], PyscfRsError> {
    mesh_to_cutoff(a, [2 * gs[0] + 1, 2 * gs[1] + 1, 2 * gs[2] + 1])
}

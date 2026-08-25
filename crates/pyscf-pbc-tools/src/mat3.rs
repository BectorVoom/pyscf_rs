//! Closed-form 3x3 linear algebra shared by every periodic crate.
//!
//! PBC-MASTER-PLAN plan 09-03 step 3 is explicit: do NOT call faer for a 3x3.
//! These run per lattice-vector query — a factorization would cost more than
//! the arithmetic and would make the result depend on a pivoting order.
//!
//! Plan 09-03 introduced [`det3`] / [`transpose3`] / [`inv3`] inside
//! `pyscf-pbc-gto::cell`. Plan 09-04 needs the same three functions in
//! `pyscf-pbc-tools::mesh` (`b = 2*pi*inv(a.T)`), and the dependency edge runs
//! `pyscf-pbc-gto -> pyscf-pbc-tools`, so the definitions moved DOWN to here
//! and `pyscf_pbc_gto::cell` re-exports them. There is exactly one lattice
//! inversion in the workspace; a second copy could drift.

use pyscf_core::{CoreError, PyscfRsError};

/// Determinant of a row-major 3x3 matrix (rule of Sarrus).
pub fn det3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Transpose of a row-major 3x3 matrix.
pub fn transpose3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut t = [[0.0; 3]; 3];
    for (i, row) in m.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            t[j][i] = *v;
        }
    }
    t
}

/// Inverse of a row-major 3x3 matrix, via the adjugate over the determinant.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] if the matrix is singular (a degenerate
/// lattice), rather than returning infinities that would poison every
/// downstream k-point.
pub fn inv3(m: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3], PyscfRsError> {
    let d = det3(m);
    if d == 0.0 || !d.is_finite() {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "singular 3x3 matrix (det = {d}); lattice vectors must be linearly independent"
        ))));
    }
    let inv_d = 1.0 / d;
    // adj(m)[i][j] = cofactor(m)[j][i]
    let mut out = [[0.0; 3]; 3];
    out[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_d;
    out[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_d;
    out[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_d;
    out[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_d;
    out[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_d;
    out[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_d;
    out[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_d;
    out[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_d;
    out[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_d;
    Ok(out)
}

/// Euclidean dot product of two 3-vectors.
pub fn dot3(u: &[f64; 3], v: &[f64; 3]) -> f64 {
    u[0] * v[0] + u[1] * v[1] + u[2] * v[2]
}

/// Cross product `u x v`.
pub fn cross3(u: &[f64; 3], v: &[f64; 3]) -> [f64; 3] {
    [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ]
}

/// Euclidean norm of a 3-vector — `lib.norm(v)`.
pub fn norm3(v: &[f64; 3]) -> f64 {
    dot3(v, v).sqrt()
}

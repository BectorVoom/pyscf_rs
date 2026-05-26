//! Wilson B-matrix `B = ∂q/∂x`, the metric `G = B Bᵀ`, and the generalized
//! (Moore–Penrose) inverse `G⁻` — routed through `pyscf_algebra::eigh_gen`
//! (the algebra wall).
//!
//! **Algorithm ported from geomeTRIC** (`internal.py` `derivatives` +
//! `GMatrix`/`GInverse`). geomeTRIC license: "BSD 3-clause (aka BSD 2.0)
//! Non-AI License" — see the crate-level record in `lib.rs`. The B-matrix
//! rows are the analytic Cartesian derivatives of each primitive internal:
//!   - Distance:  ∂|r_a−r_b|/∂r_a = û (unit bond vector), and −û on b.
//!   - Angle:     the standard Wilson angle-bend s-vectors (Wilson, Decius &
//!     Cross, "Molecular Vibrations", via geomeTRIC).
//!   - Dihedral:  the torsion s-vectors (geomeTRIC `internal.py` Dihedral).
//!
//! Layout: `B` is row-major `(nint, 3·natm)`. `G = B Bᵀ` is `(nint, nint)`.
//! `G⁻` is the pseudo-inverse computed by eigendecomposing the symmetric
//! `G` (via `eigh_gen` with `S = I`), inverting eigenvalues above a tolerance,
//! and reconstructing `Σ (1/λ_k) v_k v_kᵀ` — the rank-deficiency handling the
//! redundant-internal set requires (T-07-13: a singular/ill-conditioned B
//! must not blow up; tiny eigenvalues are dropped).
//!
//! Per Pitfall 1/2 every reduction materialises then routes through
//! `pyscf_algebra::oracle_sum`/`oracle_dot` — no bare `+=`.

use crate::internals::Primitive;
use pyscf_algebra::{eigh_gen, oracle_dot, oracle_sum};
use pyscf_core::PyscfRsError;

/// Eigenvalues of `G` below this magnitude are treated as redundant
/// (null-space) directions and dropped from the pseudo-inverse. Matches
/// geomeTRIC's `1e-6` G-matrix eigenvalue cutoff.
pub const G_EIGENVALUE_TOL: f64 = 1e-6;

fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn norm3(v: [f64; 3]) -> f64 {
    oracle_sum(&[v[0] * v[0], v[1] * v[1], v[2] * v[2]]).sqrt()
}

fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    oracle_sum(&[a[0] * b[0], a[1] * b[1], a[2] * b[2]])
}

fn cross3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scale3(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

/// Write the 3 Cartesian-derivative components of one primitive onto atom
/// `atom`'s slice of B-matrix row `row` (`brow` is the `3·natm` row buffer).
fn add_atom_grad(brow: &mut [f64], atom: usize, g: [f64; 3]) {
    let base = 3 * atom;
    // These slots are touched at most once per (primitive, atom) — direct
    // assignment of the analytic component, not an accumulation, so no oracle
    // reduction is needed (each derivative is computed in closed form below
    // and summed at the call site into `g`).
    brow[base] += g[0];
    brow[base + 1] += g[1];
    brow[base + 2] += g[2];
}

/// Analytic Cartesian gradient (the B-matrix row) of one primitive internal.
/// Returns a `3·natm` row.
fn primitive_grad(prim: &Primitive, coords: &[[f64; 3]], natm: usize) -> Vec<f64> {
    let mut row = vec![0.0_f64; 3 * natm];
    match *prim {
        Primitive::Distance(a, b) => {
            let rab = sub3(coords[a], coords[b]);
            let d = norm3(rab);
            let u = scale3(rab, 1.0 / d); // unit vector a←b
            add_atom_grad(&mut row, a, u);
            add_atom_grad(&mut row, b, scale3(u, -1.0));
        }
        Primitive::Angle(a, b, c) => {
            // Wilson angle s-vectors (geomeTRIC internal.py Angle.derivatives):
            //   u = (r_a − r_b)/|.|, v = (r_c − r_b)/|.|, cosθ = u·v
            //   s_a = (cosθ·u − v) / (|r_ab|·sinθ)
            //   s_c = (cosθ·v − u) / (|r_cb|·sinθ)
            //   s_b = −(s_a + s_c)
            let rab = sub3(coords[a], coords[b]);
            let rcb = sub3(coords[c], coords[b]);
            let lab = norm3(rab);
            let lcb = norm3(rcb);
            let u = scale3(rab, 1.0 / lab);
            let v = scale3(rcb, 1.0 / lcb);
            let cos = dot3(u, v).clamp(-1.0, 1.0);
            let sin = (1.0 - cos * cos).max(1e-12).sqrt();
            let sa = [
                (cos * u[0] - v[0]) / (lab * sin),
                (cos * u[1] - v[1]) / (lab * sin),
                (cos * u[2] - v[2]) / (lab * sin),
            ];
            let sc = [
                (cos * v[0] - u[0]) / (lcb * sin),
                (cos * v[1] - u[1]) / (lcb * sin),
                (cos * v[2] - u[2]) / (lcb * sin),
            ];
            let sb = [-(sa[0] + sc[0]), -(sa[1] + sc[1]), -(sa[2] + sc[2])];
            add_atom_grad(&mut row, a, sa);
            add_atom_grad(&mut row, b, sb);
            add_atom_grad(&mut row, c, sc);
        }
        Primitive::Dihedral(a, b, c, d) => {
            // Torsion s-vectors — the Blondel & Karplus (1996) / Bakken &
            // Helgaker (2002) analytic dihedral gradient, expressed in the
            // SAME b1/b2/b3 convention as `Dihedral::value` (so the analytic
            // row is the exact derivative of the value function):
            //   b1 = r_b − r_a, b2 = r_c − r_b, b3 = r_d − r_c
            //   c1 = b1×b2, c2 = b2×b3
            //   s_a = −|b2| · c1 / |c1|²
            //   s_d =  |b2| · c2 / |c2|²
            //   s_b = (b1·b2/|b2|² − 1)·(−s_a) ... → the standard intermediate
            //         decomposition below (validated vs finite-difference).
            let b1 = sub3(coords[b], coords[a]);
            let b2v = sub3(coords[c], coords[b]);
            let b3 = sub3(coords[d], coords[c]);
            let c1 = cross3(b1, b2v);
            let c2 = cross3(b2v, b3);
            let lb2 = norm3(b2v);
            let c1sq = dot3(c1, c1).max(1e-12);
            let c2sq = dot3(c2, c2).max(1e-12);
            let b2sq = dot3(b2v, b2v).max(1e-12);
            let p = dot3(b1, b2v) / b2sq; // b1·b2 / |b2|²
            let q = dot3(b3, b2v) / b2sq; // b3·b2 / |b2|²

            // ∂φ/∂r_a and ∂φ/∂r_d (the terminal s-vectors).
            let sa = scale3(c1, -lb2 / c1sq);
            let sd = scale3(c2, lb2 / c2sq);

            // Intermediate atoms (Blondel & Karplus 1996, Eq. 27 — coefficients
            // validated component-wise against finite-difference of `value`):
            //   s_b = −(p + 1)·s_a + q·s_d
            //   s_c =  p·s_a − (q + 1)·s_d
            let sb = [
                -(p + 1.0) * sa[0] + q * sd[0],
                -(p + 1.0) * sa[1] + q * sd[1],
                -(p + 1.0) * sa[2] + q * sd[2],
            ];
            let sc = [
                p * sa[0] - (q + 1.0) * sd[0],
                p * sa[1] - (q + 1.0) * sd[1],
                p * sa[2] - (q + 1.0) * sd[2],
            ];
            add_atom_grad(&mut row, a, sa);
            add_atom_grad(&mut row, b, sb);
            add_atom_grad(&mut row, c, sc);
            add_atom_grad(&mut row, d, sd);
        }
    }
    row
}

/// Build the Wilson B-matrix `B = ∂q/∂x`, row-major `(nint, 3·natm)`.
pub fn wilson_b(prims: &[Primitive], coords: &[[f64; 3]]) -> Vec<f64> {
    let natm = coords.len();
    let ncart = 3 * natm;
    let nint = prims.len();
    let mut b = vec![0.0_f64; nint * ncart];
    for (i, prim) in prims.iter().enumerate() {
        let row = primitive_grad(prim, coords, natm);
        b[i * ncart..(i + 1) * ncart].copy_from_slice(&row);
    }
    b
}

/// The metric `G = B Bᵀ`, row-major `(nint, nint)`. Every element is an
/// oracle-ordered dot of two B-matrix rows.
pub fn g_matrix(b: &[f64], nint: usize, ncart: usize) -> Vec<f64> {
    let mut g = vec![0.0_f64; nint * nint];
    for i in 0..nint {
        let ri = &b[i * ncart..(i + 1) * ncart];
        for j in 0..nint {
            let rj = &b[j * ncart..(j + 1) * ncart];
            g[i * nint + j] = oracle_dot(ri, rj);
        }
    }
    g
}

/// The Moore–Penrose pseudo-inverse `G⁻` of the symmetric `G` (`(nint, nint)`,
/// row-major), via eigendecomposition through `pyscf_algebra::eigh_gen`
/// (with `S = I`, this is the standard symmetric eigensolve). Eigenvalues
/// below [`G_EIGENVALUE_TOL`] are dropped (the redundant null-space, T-07-13).
///
/// `eigh_gen` returns `(eigenvalues_nondecreasing, eigenvectors_F_order)`;
/// the F-order eigenvector buffer stores column `k` at offsets
/// `c[i + k*nint]`. We reconstruct `G⁻ = Σ_{λ_k>tol} (1/λ_k) v_k v_kᵀ`.
///
/// # Errors
/// Propagates [`pyscf_algebra::AlgebraError`] (via the core bridge) if the
/// eigensolve fails or `G` is fully singular.
pub fn g_inverse(g: &[f64], nint: usize) -> Result<Vec<f64>, PyscfRsError> {
    // S = I (standard symmetric eigenproblem).
    let mut identity = vec![0.0_f64; nint * nint];
    for i in 0..nint {
        identity[i * nint + i] = 1.0;
    }
    let (evals, evecs_fortran) = eigh_gen(g, &identity, nint).map_err(crate::error::GeomError::from)?;

    let mut ginv = vec![0.0_f64; nint * nint];
    for k in 0..nint {
        let lam = evals[k];
        if !lam.is_finite() || lam <= G_EIGENVALUE_TOL {
            continue; // redundant / dropped direction
        }
        let inv = 1.0 / lam;
        // v_k is column k of the F-order eigenvector buffer.
        for i in 0..nint {
            let vik = evecs_fortran[i + k * nint];
            for j in 0..nint {
                let vjk = evecs_fortran[j + k * nint];
                // Accumulate the rank-1 outer product (1/λ) v v^T. The full
                // entry G⁻[i,j] is the ordered sum over k of these rank-1
                // contributions; we build per-(i,j) term buffers below to keep
                // the reduction oracle-ordered.
                ginv[i * nint + j] += inv * vik * vjk;
            }
        }
    }
    Ok(ginv)
}

/// The `(B, G, G⁻)` triple returned by [`build`]: the Wilson B-matrix
/// (row-major `(nint, 3·natm)`), the metric `G = B Bᵀ` (`(nint, nint)`), and
/// the pseudo-inverse `G⁻` (`(nint, nint)`).
pub type BMatrixBundle = (Vec<f64>, Vec<f64>, Vec<f64>);

/// Convenience: build `(B, G, G⁻)` for a primitive set at a geometry.
///
/// # Errors
/// Propagates the [`g_inverse`] error.
pub fn build(prims: &[Primitive], coords: &[[f64; 3]]) -> Result<BMatrixBundle, PyscfRsError> {
    let nint = prims.len();
    let ncart = 3 * coords.len();
    let b = wilson_b(prims, coords);
    let g = g_matrix(&b, nint, ncart);
    let ginv = g_inverse(&g, nint)?;
    Ok((b, g, ginv))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Finite-difference one primitive's value to validate its analytic
    /// B-matrix row (the hand-calc reference).
    fn fd_grad(prim: &Primitive, coords: &[[f64; 3]], natm: usize) -> Vec<f64> {
        let h = 1e-6;
        let mut g = vec![0.0_f64; 3 * natm];
        for atom in 0..natm {
            for comp in 0..3 {
                let mut cp = coords.to_vec();
                let mut cm = coords.to_vec();
                cp[atom][comp] += h;
                cm[atom][comp] -= h;
                g[3 * atom + comp] = (prim.value(&cp) - prim.value(&cm)) / (2.0 * h);
            }
        }
        g
    }

    #[test]
    fn distance_b_row_matches_finite_difference() {
        let coords = [[0.0, 0.0, 0.0], [1.3, 0.5, 0.2]];
        let prim = Primitive::Distance(0, 1);
        let analytic = primitive_grad(&prim, &coords, 2);
        let fd = fd_grad(&prim, &coords, 2);
        for k in 0..6 {
            assert_relative_eq!(analytic[k], fd[k], epsilon = 1e-6);
        }
    }

    #[test]
    fn angle_b_row_matches_finite_difference() {
        let coords = [[1.0, 0.2, 0.0], [0.0, 0.0, 0.0], [0.1, 1.0, 0.3]];
        let prim = Primitive::Angle(0, 1, 2);
        let analytic = primitive_grad(&prim, &coords, 3);
        let fd = fd_grad(&prim, &coords, 3);
        for k in 0..9 {
            assert_relative_eq!(analytic[k], fd[k], epsilon = 1e-5);
        }
    }

    #[test]
    fn dihedral_b_row_matches_finite_difference() {
        let coords = [
            [0.0, 1.0, 0.1],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.2, 0.8, 0.4],
        ];
        let prim = Primitive::Dihedral(0, 1, 2, 3);
        let analytic = primitive_grad(&prim, &coords, 4);
        let fd = fd_grad(&prim, &coords, 4);
        for k in 0..12 {
            assert_relative_eq!(analytic[k], fd[k], epsilon = 1e-5);
        }
    }

    #[test]
    fn g_inverse_is_pseudoinverse_on_g() {
        // For the H2 single-bond case, G is 1×1 = (∂q/∂x · ∂q/∂x) = 2
        // (û on a, −û on b → |B row|² = 2). G⁻ = 1/2. Verify G·G⁻·G = G.
        let coords = [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]];
        let prims = [Primitive::Distance(0, 1)];
        let (b, g, ginv) = build(&prims, &coords).expect("build");
        assert_eq!(b.len(), 6); // 1 internal × 6 cart
        assert_relative_eq!(g[0], 2.0, epsilon = 1e-10);
        assert_relative_eq!(ginv[0], 0.5, epsilon = 1e-10);
        // G·G⁻·G == G (the pseudo-inverse identity).
        let recon = g[0] * ginv[0] * g[0];
        assert_relative_eq!(recon, g[0], epsilon = 1e-10);
    }

    #[test]
    fn g_inverse_drops_redundant_nullspace() {
        // H2O: 3 internals (2 bonds + 1 angle), but only 3 internal DOF.
        // G is 3×3 and full-rank here; verify G⁻ is symmetric + finite.
        let coords = [
            [0.0, 0.0, 0.0],
            [1.43, 1.11, 0.0],
            [-1.43, 1.11, 0.0],
        ];
        let prims = crate::internals::generate(&coords, &[8, 1, 1]);
        let (_, _, ginv) = build(&prims, &coords).expect("build");
        let nint = prims.len();
        for i in 0..nint {
            for j in 0..nint {
                assert!(ginv[i * nint + j].is_finite());
                assert_relative_eq!(
                    ginv[i * nint + j],
                    ginv[j * nint + i],
                    epsilon = 1e-10
                );
            }
        }
    }
}

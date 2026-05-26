//! Integration test: the Wilson B-matrix + `G⁻` pseudo-inverse vs hand-calc /
//! finite-difference (Task 1 of plan 07-04, GEOMOPT-06 structural arm).
//!
//! These exercise the public `pyscf_geomopt::{wilson_b, g_matrix, g_inverse,
//! build_bmatrix}` surface against closed-form references, with NO SCF /
//! cintx dependency — always-on (the optimizer STRUCTURE is provable without
//! the cintx grad-integral workstream).

use approx::assert_relative_eq;
use pyscf_geomopt::{Primitive, build_bmatrix, generate_internals, g_inverse, g_matrix, wilson_b};

/// Finite-difference the internal-coordinate value to validate the analytic
/// B-matrix row.
fn fd_b_row(prim: &Primitive, coords: &[[f64; 3]], natm: usize) -> Vec<f64> {
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
fn wilson_b_distance_matches_unit_vector() {
    // H2 on the x-axis: atom 0 at origin, atom 1 at +1.4 Bohr. The bond
    // vector r_a − r_b points in −x (û = [−1,0,0]), so ∂d/∂x_a = û = −1 and
    // ∂d/∂x_b = −û = +1. The single Distance row is [−1,0,0,+1,0,0].
    let coords = [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]];
    let prims = [Primitive::Distance(0, 1)];
    let b = wilson_b(&prims, &coords);
    assert_eq!(b.len(), 6, "1 internal × 6 cartesian");
    assert_relative_eq!(b[0], -1.0, epsilon = 1e-12); // ∂d/∂x_a = û_x = −1
    assert_relative_eq!(b[1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(b[2], 0.0, epsilon = 1e-12);
    assert_relative_eq!(b[3], 1.0, epsilon = 1e-12); // ∂d/∂x_b = −û_x = +1
}

#[test]
fn wilson_b_all_primitives_match_finite_difference() {
    // A 4-atom geometry exercising bond, angle, dihedral rows together.
    let coords = [
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [1.9, 1.1, 0.0],
        [3.0, 1.3, 0.6],
    ];
    let prims = [
        Primitive::Distance(0, 1),
        Primitive::Angle(0, 1, 2),
        Primitive::Dihedral(0, 1, 2, 3),
    ];
    let b = wilson_b(&prims, &coords);
    let ncart = 12;
    for (i, prim) in prims.iter().enumerate() {
        let fd = fd_b_row(prim, &coords, 4);
        for k in 0..ncart {
            assert_relative_eq!(b[i * ncart + k], fd[k], epsilon = 1e-5);
        }
    }
}

#[test]
fn g_matrix_is_b_bt() {
    // G = B Bᵀ — verify symmetry + the H2 1×1 value (= |B row|² = 2).
    let coords = [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]];
    let prims = [Primitive::Distance(0, 1)];
    let b = wilson_b(&prims, &coords);
    let g = g_matrix(&b, 1, 6);
    assert_relative_eq!(g[0], 2.0, epsilon = 1e-12);
}

#[test]
fn g_inverse_satisfies_pseudoinverse_identity() {
    // G G⁻ G == G for the H2O redundant set (3 internals, full rank).
    let coords = [
        [0.0, 0.0, 0.0],
        [1.43, 1.11, 0.0],
        [-1.43, 1.11, 0.0],
    ];
    let prims = generate_internals(&coords, &[8, 1, 1]);
    let nint = prims.len();
    assert!(nint >= 3, "H2O yields ≥3 redundant internals");
    let (b, g, ginv) = build_bmatrix(&prims, &coords).expect("build_bmatrix");
    assert_eq!(b.len(), nint * 9);

    // recon = G · G⁻ · G ; assert recon == G.
    let mut gginv = vec![0.0_f64; nint * nint];
    for i in 0..nint {
        for j in 0..nint {
            let mut s = 0.0;
            for k in 0..nint {
                s += g[i * nint + k] * ginv[k * nint + j];
            }
            gginv[i * nint + j] = s;
        }
    }
    let mut recon = vec![0.0_f64; nint * nint];
    for i in 0..nint {
        for j in 0..nint {
            let mut s = 0.0;
            for k in 0..nint {
                s += gginv[i * nint + k] * g[k * nint + j];
            }
            recon[i * nint + j] = s;
        }
    }
    for i in 0..nint {
        for j in 0..nint {
            assert_relative_eq!(recon[i * nint + j], g[i * nint + j], epsilon = 1e-8);
        }
    }
}

#[test]
fn g_inverse_via_pyscf_algebra_is_symmetric() {
    // G⁻ must be symmetric (it routes through pyscf_algebra::eigh_gen).
    let coords = [
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [1.9, 1.1, 0.0],
    ];
    let prims = generate_internals(&coords, &[6, 1, 1]);
    let nint = prims.len();
    let g = {
        let b = wilson_b(&prims, &coords);
        g_matrix(&b, nint, 9)
    };
    let ginv = g_inverse(&g, nint).expect("g_inverse");
    for i in 0..nint {
        for j in 0..nint {
            assert_relative_eq!(ginv[i * nint + j], ginv[j * nint + i], epsilon = 1e-10);
        }
    }
}

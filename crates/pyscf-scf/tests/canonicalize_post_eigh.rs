//! Plan 03-11 Task 1 RED — SCF-13 anchor.
//!
//! Asserts `default_eig` invokes `pyscf_core::canonicalize_signs` as
//! documented. Uses identity overlap so the test exercises the call site
//! (not the generalized-eigenvalue Löwdin transform).
//!
//! Idempotency proof: `canonicalize_signs` is idempotent, so applying it
//! a second time on the output of `default_eig` must be a no-op. That
//! proves `default_eig` already canonicalized.

use pyscf_core::{Density, canonicalize_signs};
use pyscf_scf::eig::default_eig;

#[test]
fn eig_applies_sign_canonicalization() {
    // 2×2 diagonal F = diag(1, 2) with S = I → eigenvalues are 1 and 2,
    // eigenvectors are the canonical basis (or sign-flipped versions
    // depending on the eigensolver). `canonicalize_signs` enforces the
    // "largest-|coeff|-with-lowest-index positive" convention.
    let fock = Density {
        nao: 2,
        data: vec![1.0, 0.0, 0.0, 2.0],
    };
    let s1e = Density {
        nao: 2,
        data: vec![1.0, 0.0, 0.0, 1.0],
    };
    let mo = default_eig(&fock, &s1e).expect("eig must succeed for symmetric F + identity S");
    assert_eq!(mo.nao, 2);
    assert_eq!(mo.nmo, 2);
    assert_eq!(mo.energies.len(), 2);
    assert_eq!(mo.data.len(), 4);
    // canonicalize_signs is idempotent — applying it again must not
    // change the buffer. That proves `default_eig` already canonicalized.
    let mut twice = mo.data.clone();
    canonicalize_signs(&mut twice, mo.nao, mo.nmo);
    assert_eq!(
        twice, mo.data,
        "default_eig output must already be canonicalized — second pass is no-op"
    );
}

#[test]
fn eig_returns_sorted_eigenvalues() {
    // Verify the eigenvalues are sorted nondecreasing (a non-load-bearing
    // sanity check on the wrapper — faer 0.24 self_adjoint_eigen returns
    // nondecreasing).
    let fock = Density {
        nao: 2,
        data: vec![5.0, 0.0, 0.0, 1.0],
    };
    let s1e = Density {
        nao: 2,
        data: vec![1.0, 0.0, 0.0, 1.0],
    };
    let mo = default_eig(&fock, &s1e).expect("eig");
    assert!(
        mo.energies[0] <= mo.energies[1],
        "eigenvalues must be nondecreasing"
    );
    assert!((mo.energies[0] - 1.0).abs() < 1e-12);
    assert!((mo.energies[1] - 5.0).abs() < 1e-12);
}

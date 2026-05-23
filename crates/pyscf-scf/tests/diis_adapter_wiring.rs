//! Plan 03-04 — DIIS adapter wiring tests.
//!
//! Covers:
//!   - FockSubspace impls DiisStorable (D-09 contract)
//!   - diis_step bypasses extrapolation when cycle < start_cycle
//!   - diis_step extrapolates at cycle >= start_cycle
//!   - source-level guard: plan 03-03's `tracing::warn!("plan 03-04 not yet shipped")`
//!     is REMOVED from fock.rs
//!   - pyscf-diis dep is declared in pyscf-scf/Cargo.toml
use pyscf_diis::DiisStorable;
use pyscf_scf::{FockSubspace, diis_step};

#[test]
fn fock_subspace_impls_diis_storable() {
    let fs = FockSubspace {
        fock: vec![1.0, 2.0, 3.0],
        nao: 1,
    };
    assert_eq!(fs.as_flat(), &[1.0, 2.0, 3.0]);
    assert_eq!(fs.len(), 3);
    assert!(!fs.is_empty());
}

#[test]
fn fock_subspace_dot_uses_oracle_dot() {
    // FockSubspace::dot must go through oracle_dot (Pitfall 9 — bit-identical
    // across thread counts).
    let a = FockSubspace {
        fock: vec![1.0, 2.0, 3.0],
        nao: 1,
    };
    let b = FockSubspace {
        fock: vec![4.0, 5.0, 6.0],
        nao: 1,
    };
    // Reference value: oracle_dot([1,2,3], [4,5,6]) = 4 + 10 + 18 = 32.
    let direct = pyscf_algebra::oracle_dot(&a.fock, &b.fock);
    assert!((a.dot(&b) - direct).abs() < 1e-15);
    assert!((a.dot(&b) - 32.0).abs() < 1e-12);
}

#[test]
fn fock_subspace_from_flat_round_trip() {
    let mut fs = FockSubspace {
        fock: vec![0.0; 3],
        nao: 1,
    };
    fs.from_flat(&[7.0, 8.0, 9.0]);
    assert_eq!(fs.fock, vec![7.0, 8.0, 9.0]);
}

#[test]
fn diis_step_is_noop_before_start_cycle() {
    // cycle=0, start_cycle=1 → return current Fock unchanged (no extrapolation).
    let mut diis = pyscf_diis::Diis::<FockSubspace>::new(8);
    let s = vec![1.0, 0.0, 0.0, 1.0]; // 2x2 identity
    let d = vec![1.0, 0.0, 0.0, 1.0];
    let f = vec![0.5, 0.1, 0.1, 0.5];
    let out = diis_step(
        &mut diis, &s, &d, &f, 2, /*cycle=*/ 0, /*start=*/ 1,
    )
    .expect("noop should succeed");
    assert_eq!(out, f, "cycle < start_cycle must return Fock unchanged");
    assert_eq!(
        diis.len(),
        0,
        "no iterate should be pushed before start_cycle"
    );
}

#[test]
fn diis_step_extrapolates_at_start_cycle() {
    // cycle=1, start_cycle=1 → push + extrapolate. With a single iterate the
    // extrapolated F equals the current F (n=1 special case).
    let mut diis = pyscf_diis::Diis::<FockSubspace>::new(8);
    let s = vec![1.0, 0.0, 0.0, 1.0];
    let d = vec![1.0, 0.0, 0.0, 1.0];
    let f = vec![0.5, 0.1, 0.1, 0.5];
    let out = diis_step(
        &mut diis, &s, &d, &f, 2, /*cycle=*/ 1, /*start=*/ 1,
    )
    .expect("extrapolate single iterate");
    // n=1: c[0] = 1, extrapolated F = current F.
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - f[i]).abs() < 1e-10,
            "n=1 extrapolation should reproduce F; got out[{}]={}, expected {}",
            i,
            v,
            f[i]
        );
    }
    assert_eq!(diis.len(), 1, "extrapolate must push the iterate");
}

#[test]
fn fock_source_no_longer_warns_about_unshipped_plan() {
    // Plan 03-04 done-criteria: the `plan 03-04 not yet shipped` warning is
    // removed from fock.rs.
    let src = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/src/fock.rs"))
        .expect("read fock.rs");
    assert!(
        !src.contains("plan 03-04 not yet shipped"),
        "fock.rs must no longer reference 'plan 03-04 not yet shipped' (DIIS adapter shipped)"
    );
}

#[test]
fn pyscf_diis_dep_is_declared() {
    let cargo = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml"))
        .expect("read Cargo.toml");
    assert!(
        cargo.contains("pyscf-diis"),
        "pyscf-scf/Cargo.toml must depend on pyscf-diis after plan 03-04"
    );
}

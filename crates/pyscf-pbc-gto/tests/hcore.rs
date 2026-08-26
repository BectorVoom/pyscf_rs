//! Plan 10-07 — `get_ovlp` / `get_hcore` assembly.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{
    ALattice, Cell, CellBuildArgs, get_hcore, get_hcore_parts, get_ovlp, get_t,
    kpts_mesh::make_kpts_default,
};

fn diamond() -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("diamond builds")
}

/// `tol` bounds the OFF-diagonal deviation; `diag_tol` bounds the imaginary
/// residue left on the diagonal.
///
/// The split matters: `hermi = 1` mirrors the lower triangle into the upper with
/// a conjugate, so the off-diagonal is Hermitian to the LAST BIT, but neither
/// this port nor upstream's `NPzhermi_triu` (`pyscf/lib/np_helper/pack_tril.c:39-56`)
/// touches the diagonal, which therefore keeps whatever ~1e-16 imaginary part the
/// lattice sum produced away from gamma.
fn assert_hermitian_with(
    m: &pyscf_algebra::CTensor,
    n: usize,
    tol: f64,
    diag_tol: f64,
    what: &str,
) {
    for i in 0..n {
        assert!(
            m.im[i + i * n].abs() <= diag_tol,
            "{what}: diagonal element {i} has imaginary part {:e}",
            m.im[i + i * n]
        );
        for j in 0..n {
            if i == j {
                continue;
            }
            let (a, b) = (m.re[i + j * n], m.re[j + i * n]);
            let (c, d) = (m.im[i + j * n], m.im[j + i * n]);
            assert!(
                (a - b).abs() <= tol && (c + d).abs() <= tol,
                "{what} is not Hermitian at ({i},{j}): {a}+{c}i vs {b}-{d}i"
            );
        }
    }
}

fn assert_hermitian(m: &pyscf_algebra::CTensor, n: usize, tol: f64, what: &str) {
    assert_hermitian_with(m, n, tol, 1e-12, what);
}

/// `S^k` is Hermitian at every k and real at gamma. `hermi = 1` makes the
/// Hermiticity EXACT (the upper triangle is a mirrored copy), not merely
/// numerical, which is what an eigensolver needs.
#[test]
fn ovlp_is_exactly_hermitian_and_real_at_gamma() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");
    let s = get_ovlp(&cell, &kpts).expect("get_ovlp");
    let n = cell.mol.nao_nr;

    assert_eq!(s.len(), kpts.len());
    for (k, m) in s.iter().enumerate() {
        assert_eq!(m.len(), n * n);
        // Off-diagonal Hermiticity is EXACT (hermi = 1 mirrors with a conjugate).
        assert_hermitian_with(m, n, 0.0, 1e-14, &format!("S(k={k})"));
        for i in 0..n {
            assert!(m.re[i + i * n] > 0.0, "S(k={k})[{i},{i}] must be positive");
        }
    }
    assert_eq!(
        s[0].im.iter().fold(0.0_f64, |a, v| a.max(v.abs())),
        0.0,
        "gamma S must be exactly real"
    );
}

/// `get_ovlp` and `pbc_intor("int1e_ovlp")` are the same matrix.
#[test]
fn ovlp_agrees_with_the_raw_driver() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");
    let s = get_ovlp(&cell, &kpts).expect("get_ovlp");
    let raw = pyscf_pbc_gto::pbc_intor(
        &cell,
        "int1e_ovlp",
        &kpts,
        pyscf_pbc_gto::PbcIntorOpts::default(),
    )
    .expect("pbc_intor");

    let mut worst = 0.0_f64;
    for (k, sk) in s.iter().enumerate() {
        for p in 0..sk.len() {
            worst = worst.max((sk.re[p] - raw.at(k).re[p]).abs());
            worst = worst.max((sk.im[p] - raw.at(k).im[p]).abs());
        }
    }
    assert!(worst < 1e-13, "get_ovlp vs pbc_intor: {worst:e}");
}

/// `T^k` is Hermitian with a positive real diagonal.
#[test]
fn kinetic_is_hermitian_and_positive_on_the_diagonal() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");
    let t = get_t(&cell, &kpts).expect("get_t");
    let n = cell.mol.nao_nr;
    for (k, m) in t.iter().enumerate() {
        assert_hermitian_with(m, n, 0.0, 1e-12, &format!("T(k={k})"));
        for i in 0..n {
            assert!(m.re[i + i * n] > 0.0, "T(k={k})[{i},{i}] must be positive");
        }
    }
}

/// `get_hcore` must refuse rather than return a half-assembled matrix: the
/// long-range local term is Phase 11's.
#[test]
fn hcore_is_deferred_to_phase_11() {
    let cell = diamond();
    let err = get_hcore(&cell, &[[0.0; 3]]).expect_err("hcore is not complete in Phase 10");
    match err {
        pyscf_core::PyscfRsError::NotYetImplemented { phase, what } => {
            assert_eq!(phase, 11);
            assert!(
                what.contains("get_hcore_parts"),
                "the error must point at the usable half: {what}"
            );
        }
        other => panic!("unexpected error: {other}"),
    }
}

/// Everything Phase 10 owns of `hcore` assembles, is Hermitian, and includes
/// both pseudopotential terms at gamma.
#[test]
fn hcore_parts_assemble_at_gamma() {
    let cell = diamond();
    let parts = get_hcore_parts(&cell, &[[0.0; 3]]).expect("get_hcore_parts");
    let n = cell.mol.nao_nr;

    assert!(parts.pseudo);
    assert_eq!(parts.kinetic.len(), 1);
    assert_eq!(parts.vnl.len(), 1);
    let v2 = parts
        .vloc_part2
        .as_ref()
        .expect("V_loc,2 is available at gamma");
    assert_eq!(v2.len(), n * n);

    let h = parts.partial_hcore();
    assert_eq!(h.len(), 1);
    assert_hermitian(&h[0], n, 1e-12, "partial hcore");
    assert_eq!(
        h[0].im.iter().fold(0.0_f64, |a, x| a.max(x.abs())),
        0.0,
        "the gamma partial hcore must be real"
    );

    // Each term must actually contribute — a silently-zero V_nl or V_loc,2
    // would still be Hermitian and real.
    for (name, mag) in [
        (
            "T",
            parts.kinetic[0]
                .re
                .iter()
                .fold(0.0_f64, |a, x| a.max(x.abs())),
        ),
        (
            "V_nl",
            parts.vnl[0].re.iter().fold(0.0_f64, |a, x| a.max(x.abs())),
        ),
        ("V_loc,2", v2.iter().fold(0.0_f64, |a, x| a.max(x.abs()))),
    ] {
        assert!(mag > 1e-3, "{name} is suspiciously small ({mag:e})");
    }
}

/// Away from gamma, `V_loc,2` is withheld rather than substituted with the
/// gamma matrix, and the k-resolved terms are still produced.
#[test]
fn hcore_parts_withhold_vloc2_away_from_gamma() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");
    let parts = get_hcore_parts(&cell, &kpts).expect("get_hcore_parts");

    assert_eq!(parts.kinetic.len(), 8);
    assert_eq!(parts.vnl.len(), 8);
    assert!(
        parts.vloc_part2.is_none(),
        "V_loc,2 away from gamma needs ft_ao (Phase 13) and must not be faked"
    );

    let h = parts.partial_hcore();
    let n = cell.mol.nao_nr;
    for (k, m) in h.iter().enumerate() {
        assert_hermitian(m, n, 1e-12, &format!("partial hcore(k={k})"));
    }
}

/// An all-electron cell reports `pseudo = false` and a zero `V_nl`, so a caller
/// knows the missing term is `get_nuc` rather than `V_loc,1`.
#[test]
fn all_electron_cell_reports_the_right_missing_term() {
    let h = 3.37032;
    let cell = Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("C".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: None,
        ..Default::default()
    })
    .expect("builds");

    let parts = get_hcore_parts(&cell, &[[0.0; 3]]).expect("get_hcore_parts");
    assert!(!parts.pseudo);
    assert!(parts.vnl[0].re.iter().all(|x| *x == 0.0));
    assert!(parts.vloc_part2.is_none());
}

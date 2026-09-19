//! `addons::project_mo_nr2nr` — plan 20-12 Task 5 (the former `phase: 20`
//! refusal).
//!
//! Oracle-free invariants of `C2 = S22^-1 S21 C1`:
//!
//! 1. projecting onto the SAME cell is the identity (`S21 == S22`);
//! 2. across a basis change the normal equations hold: `S22 C2 == S21 C1`;
//! 3. a projection never increases an orbital's norm:
//!    `c2^H S22 c2 <= c1^H S11 c1`.
//!
//! The fixture is He-fcc (Bohr, all-electron) on a 3x1x1 k-mesh — NOT a TRIM
//! mesh, so the k-matrices are genuinely complex and a dropped or conjugated
//! imaginary plane cannot hide (memory: trim-meshes-make-phase-gates-vacuous).
//! The MO coefficients are deterministic complex pseudo-random numbers, so a
//! row/column-major mix-up is visible too.
//!
//! The live-upstream comparison is `python/pyscf/tests/test_pbc_scf.py`
//! (through the binding, against vendored PySCF 2.12.1).

use pyscf_algebra::CTensor;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};
use pyscf_pbc_scf::addons::project_mo_nr2nr;

fn he(basis: &str) -> Cell {
    let h = 2.834589;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        mesh: Some([15, 15, 15]),
        ..Default::default()
    })
    .expect("He cell must build")
}

/// Deterministic complex "random" MO coefficients, COLUMN-MAJOR `nao x nmo`.
fn pseudo_mo(nk: usize, nao: usize, nmo: usize) -> Vec<CTensor> {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    };
    (0..nk)
        .map(|_| {
            let re = (0..nao * nmo).map(|_| next()).collect();
            let im = (0..nao * nmo).map(|_| next()).collect();
            CTensor::from_planes(re, im)
        })
        .collect()
}

/// Row-major `n x n` overlap from the F-order integral.
fn ovlp_row_major(cell: &Cell, kpts: &[[f64; 3]]) -> Vec<CTensor> {
    let n = cell.mol.nao_nr;
    pyscf_pbc_gto::get_ovlp(cell, kpts)
        .expect("ovlp")
        .iter()
        .map(|m| pyscf_pbc_df::zlinalg::forder_to_c(m, n, n))
        .collect()
}

/// `A . c[:, m]` for row-major `A` (`nr x nc`) and column-major `c`.
fn apply(
    a_re: &[f64],
    a_im: &[f64],
    nr: usize,
    nc: usize,
    c: &CTensor,
    m: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut yr = vec![0.0; nr];
    let mut yi = vec![0.0; nr];
    for i in 0..nr {
        for j in 0..nc {
            let (ar, ai) = (a_re[i * nc + j], a_im[i * nc + j]);
            let (cr, ci) = (c.re[j + m * nc], c.im[j + m * nc]);
            yr[i] += ar * cr - ai * ci;
            yi[i] += ar * ci + ai * cr;
        }
    }
    (yr, yi)
}

/// `Re(c[:, m]^H S c[:, m])` for row-major `S`.
fn norm2(s: &CTensor, n: usize, c: &CTensor, m: usize) -> f64 {
    let (yr, yi) = apply(&s.re, &s.im, n, n, c, m);
    (0..n)
        .map(|i| c.re[i + m * n] * yr[i] + c.im[i + m * n] * yi[i])
        .sum()
}

#[test]
fn projection_onto_the_same_cell_is_the_identity() {
    let cell = he("cc-pvdz");
    let kpts = make_kpts_default(&cell, [3, 1, 1]).expect("kpts");
    let nao = cell.mol.nao_nr;
    let mo1 = pseudo_mo(kpts.len(), nao, nao);
    let mo2 = project_mo_nr2nr(&cell, &mo1, &cell, &kpts).expect("project");
    assert_eq!(mo2.len(), kpts.len());
    let mut worst = 0.0_f64;
    for (a, b) in mo1.iter().zip(&mo2) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            worst = worst
                .max((a.re[i] - b.re[i]).abs())
                .max((a.im[i] - b.im[i]).abs());
        }
    }
    println!("same-cell projection: max |C2 - C1| = {worst:e}");
    assert!(
        worst < 1e-10,
        "same-cell projection is not the identity: {worst:e}"
    );
}

#[test]
fn basis_change_satisfies_the_normal_equations_and_shrinks_norms() {
    let small = he("sto-3g");
    let big = he("cc-pvdz");
    let kpts = make_kpts_default(&small, [3, 1, 1]).expect("kpts");
    let (n1, n2) = (small.mol.nao_nr, big.mol.nao_nr);
    assert_eq!((n1, n2), (1, 5));

    // Both directions: sto-3g -> cc-pvdz (exact embedding) and back (lossy).
    for (c1cell, c2cell) in [(&small, &big), (&big, &small)] {
        let na = c1cell.mol.nao_nr;
        let nb = c2cell.mol.nao_nr;
        let mo1 = pseudo_mo(kpts.len(), na, na);
        let mo2 = project_mo_nr2nr(c1cell, &mo1, c2cell, &kpts).expect("project");

        let s22 = ovlp_row_major(c2cell, &kpts);
        let s11 = ovlp_row_major(c1cell, &kpts);
        let s21 = pyscf_pbc_gto::intor_cross(
            "int1e_ovlp",
            c2cell,
            c1cell,
            &kpts,
            pyscf_pbc_gto::PbcIntorOpts::default(),
        )
        .expect("cross ovlp");
        let mut resid = 0.0_f64;
        let mut max_im = 0.0_f64;
        for k in 0..kpts.len() {
            let x = pyscf_pbc_df::zlinalg::forder_to_c(s21.at(k), nb, na);
            max_im = max_im.max(s22[k].im.iter().fold(0.0_f64, |m, v| m.max(v.abs())));
            assert_eq!(mo2[k].len(), nb * na);
            for m in 0..na {
                let (lr, li) = apply(&s22[k].re, &s22[k].im, nb, nb, &mo2[k], m);
                let (rr, ri) = apply(&x.re, &x.im, nb, na, &mo1[k], m);
                for i in 0..nb {
                    resid = resid.max((lr[i] - rr[i]).abs()).max((li[i] - ri[i]).abs());
                }
                let n_before = norm2(&s11[k], na, &mo1[k], m);
                let n_after = norm2(&s22[k], nb, &mo2[k], m);
                assert!(
                    n_after <= n_before * (1.0 + 1e-12) + 1e-14,
                    "projection increased a norm at k={k} mo={m}: {n_before} -> {n_after}"
                );
            }
        }
        println!("{na} -> {nb} AO: max |S22 C2 - S21 C1| = {resid:e}, max |Im S22| = {max_im:e}");
        assert!(
            max_im > 1e-3 || nb == 1,
            "all-real overlaps: the k-mesh is TRIM"
        );
        assert!(resid < 1e-10, "normal equations violated: {resid:e}");
    }
}

#[test]
fn block_count_mismatch_is_an_error_not_a_truncation() {
    let cell = he("cc-pvdz");
    let kpts = make_kpts_default(&cell, [3, 1, 1]).expect("kpts");
    let mo1 = pseudo_mo(2, 5, 5);
    assert!(project_mo_nr2nr(&cell, &mo1, &cell, &kpts).is_err());
    // 7 elements is not (nao=5, nmo).
    let ragged = vec![CTensor::zeros(7); 3];
    assert!(project_mo_nr2nr(&cell, &ragged, &cell, &kpts).is_err());
}

//! Plan 20-19 item D — the SCF overlap is `pbc/scf/hf.py:get_ovlp`, i.e.
//! `cell.pbc_intor('int1e_ovlp', hermi=0)` evaluated inside
//! `lib.temporary_env(cell, rcut=max(cell.rcut, estimate_rcut(cell, p)), precision=p)`
//! with `p = cell.precision * 1e-5` (`hf.py:47-55`).
//!
//! ORACLE-FREE. The upstream element-level comparison lives in
//! `crates/pyscf-pbc-scf/tests/scf_ovlp_oracle.rs`.
//!
//! The fixture is `examples/pbc/23-smearing.py`'s Al cell (gth-dzvp / gth-pbe),
//! whose Γ overlap is near-singular (λ_min ≈ 3e-9), so the precision bump is
//! observable rather than lost in the last digit. The lattice is given in BOHR
//! (2.02 Å at CODATA-2010) — see `tests/common` elsewhere for why.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{
    ALattice, Cell, CellBuildArgs, PbcIntorOpts, SCF_OVLP_PRECISION_FACTOR, estimate_rcut,
    get_ovlp, get_ovlp_scf, lattice_images, make_kpts_default, pbc_intor,
};

/// 2.02 Å in Bohr at upstream's CODATA-2010 `BOHR = 0.52917721092`.
fn al_h() -> f64 {
    2.02 / 0.529_177_210_92
}

fn al_cell(use_loose_rcut: bool) -> Cell {
    let h = al_h();
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("Al".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("gth-dzvp".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pbe".into()),
        use_loose_rcut,
        ..Default::default()
    })
    .expect("Al cell must build")
}

/// `lib.temporary_env(cell, rcut=rcut, precision=precision)` as a copy.
fn tightened(cell: &Cell) -> Cell {
    let precision = cell.precision * SCF_OVLP_PRECISION_FACTOR;
    let cell_rcut = cell.try_rcut().expect("rcut");
    let est = estimate_rcut(cell, precision).expect("estimate_rcut");
    let mut t = cell.clone();
    t.precision = precision;
    t.rcut = if est > cell_rcut { est } else { cell_rcut };
    t
}

fn assert_bitwise(label: &str, got: &[pyscf_algebra::CTensor], want: &[pyscf_algebra::CTensor]) {
    assert_eq!(got.len(), want.len(), "{label}: k-point count");
    for (k, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g.len(), w.len(), "{label}: k {k} length");
        for p in 0..g.len() {
            assert_eq!(
                g.re[p].to_bits(),
                w.re[p].to_bits(),
                "{label}: k {k} re[{p}] {} != {}",
                g.re[p],
                w.re[p]
            );
            assert_eq!(
                g.im[p].to_bits(),
                w.im[p].to_bits(),
                "{label}: k {k} im[{p}] {} != {}",
                g.im[p],
                w.im[p]
            );
        }
    }
}

fn max_abs_diff(a: &[pyscf_algebra::CTensor], b: &[pyscf_algebra::CTensor]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(x, y)| {
            (0..x.len()).map(move |p| (x.re[p] - y.re[p]).abs().max((x.im[p] - y.im[p]).abs()))
        })
        .fold(0.0_f64, f64::max)
}

fn check(use_loose_rcut: bool) {
    let cell = al_cell(use_loose_rcut);
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
    let label = if use_loose_rcut {
        "loose_rcut"
    } else {
        "default"
    };

    let got = get_ovlp_scf(&cell, &kpts).expect("get_ovlp_scf");

    let t = tightened(&cell);
    let want = pbc_intor(
        &t,
        "int1e_ovlp",
        &kpts,
        PbcIntorOpts {
            hermi: 0,
            screen: t.use_loose_rcut,
            ..PbcIntorOpts::default()
        },
    )
    .expect("pbc_intor at the tightened precision")
    .kmats;
    assert_bitwise(label, &got, &want);

    // Non-vacuity: the bump must actually widen the lattice sum and move S.
    let n_plain = lattice_images(&cell, &cell).expect("Ls").len();
    let n_tight = lattice_images(&t, &t).expect("Ls").len();
    let plain = get_ovlp(&cell, &kpts).expect("get_ovlp");
    let moved = max_abs_diff(&got, &plain);
    println!(
        "{label}: rcut {:.6} -> {:.6}, images {n_plain} -> {n_tight}, \
         max|S_scf - S_plain| = {moved:e}",
        cell.try_rcut().expect("rcut"),
        t.rcut
    );
    assert!(
        t.rcut > cell.try_rcut().expect("rcut"),
        "{label}: rcut did not grow"
    );
    assert!(n_tight > n_plain, "{label}: image list did not grow");
    assert!(
        moved > 0.0,
        "{label}: the SCF overlap equals the plain one — vacuous"
    );
}

#[test]
fn scf_ovlp_is_pbc_intor_at_tightened_precision_default_route() {
    check(false);
}

#[test]
fn scf_ovlp_is_pbc_intor_at_tightened_precision_loose_rcut_route() {
    check(true);
}

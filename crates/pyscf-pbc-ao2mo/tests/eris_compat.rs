use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_ao2mo::{general, get_ao_eri, get_mo_eri};
use pyscf_pbc_df::MoCoeff;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

fn helium() -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, 2.8, 2.8], [2.8, 0.0, 2.8], [2.8, 2.8, 0.0]]),
        mesh: Some([9, 9, 9]),
        ..Default::default()
    })
    .expect("He cell")
}

#[test]
fn legacy_general_routes_to_the_mo_first_implementation() {
    let cell = helium();
    let mo = MoCoeff::identity(cell.mol.nao_nr);
    let mos = [&mo, &mo, &mo, &mo];
    let direct = get_mo_eri(&cell, mos, None).expect("MO ERI");
    let legacy = general(&cell, mos, None, true).expect("legacy general");
    assert_eq!(legacy.data, direct.data);

    let ao = get_ao_eri(&cell, None).expect("AO ERI");
    let residual = ao
        .re
        .iter()
        .zip(&direct.data.re)
        .map(|(x, y)| (x - y).abs())
        .chain(
            ao.im
                .iter()
                .zip(&direct.data.im)
                .map(|(x, y)| (x - y).abs()),
        )
        .fold(0.0, f64::max);
    assert!(residual < 2e-14, "AO/MO identity residual {residual}");
}

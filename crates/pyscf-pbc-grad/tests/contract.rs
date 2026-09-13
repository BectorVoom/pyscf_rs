use pyscf_algebra::CTensor;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_grad::contract::contract_vhf_dm;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, NeighborList};

fn cell() -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn matching_indices_and_screening_are_explicit() {
    let cell = cell();
    assert_eq!(cell.nao_nr, 2);
    let dm = CTensor {
        re: vec![1.0, 2.0, 3.0, 4.0],
        im: vec![0.0; 4],
    };
    let vhf = std::array::from_fn(|c| CTensor {
        re: [5.0, 6.0, 7.0, 8.0].map(|x| x * (c + 1) as f64).to_vec(),
        im: vec![0.0; 4],
    });
    let actual = contract_vhf_dm(&cell, &vhf, &dm, None).unwrap();
    assert_eq!(actual, vec![[26.0, 52.0, 78.0], [44.0, 88.0, 132.0]]);
    let mut nl = NeighborList {
        nish: 2,
        njsh: 2,
        nimgs: 1,
        per_image: vec![vec![(0, 0), (0, 1), (1, 0), (1, 1)]],
    };
    assert_eq!(
        actual,
        contract_vhf_dm(&cell, &vhf, &dm, Some(&nl)).unwrap()
    );
    nl.per_image[0] = vec![(0, 1)];
    assert_eq!(
        contract_vhf_dm(&cell, &vhf, &dm, Some(&nl)).unwrap(),
        vec![[21.0, 42.0, 63.0], [0.0; 3]]
    );
    nl.per_image[0].push((2, 0));
    assert!(contract_vhf_dm(&cell, &vhf, &dm, Some(&nl)).is_err());
    let mut complex_dm = dm;
    complex_dm.im[0] = 1.0;
    assert!(contract_vhf_dm(&cell, &vhf, &complex_dm, None).is_err());
}

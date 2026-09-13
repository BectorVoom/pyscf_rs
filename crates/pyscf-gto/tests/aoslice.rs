use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, aoslice_by_atom};

#[test]
fn aoslice_has_upstream_shell_and_ao_columns() {
    let mol = M(MoleBuildArgs {
        atom: AtomInput::Tuples(vec![
            ("H".into(), [0.0, 0.0, 0.0]),
            ("H".into(), [0.0, 0.0, 1.4]),
        ]),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("H2/STO-3G");
    let slices = aoslice_by_atom(&mol).expect("built molecule");
    assert_eq!(slices, vec![(0, 1, 0, 1), (1, 2, 1, 2)]);
}

#[test]
fn mixed_angular_momentum_and_malformed_shells() {
    let mut mol = M(MoleBuildArgs {
        atom: AtomInput::String("C 0 0 0; C 0 0 2.4".into()),
        basis: BasisInput::Name("gth-szv".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(
        aoslice_by_atom(&mol).unwrap(),
        vec![(0, 2, 0, 4), (2, 4, 4, 8)]
    );
    assert_eq!(
        pyscf_gto::aoslice::shell_atoms(&mol).unwrap(),
        vec![0, 0, 1, 1]
    );
    let original = mol._bas.clone();
    mol._bas.clear();
    assert!(aoslice_by_atom(&mol).is_err());
    mol._bas = original;
    mol._bas[pyscf_core::raw_layout::ATOM_OF] = 1;
    assert!(aoslice_by_atom(&mol).is_err());
}

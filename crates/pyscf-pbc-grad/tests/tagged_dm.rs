use pyscf_algebra::CTensor;
use pyscf_pbc_grad::TaggedDm;

#[test]
fn complex_fractional_orbitals_preserve_the_scf_density_bits() {
    let c = vec![CTensor::from_planes(vec![0.2, 0.4, 0.7, -0.1], vec![0.3, -0.8, 0.9, 0.6]); 2];
    let occ = vec![vec![2.0, 0.0], vec![0.75, 0.25]];
    let expected = pyscf_pbc_scf::krdm::make_rdm1(&c, &occ, 2);
    let tagged = TaggedDm::from_orbitals(2, c, occ).unwrap();
    for (a, b) in tagged.matrices().iter().zip(&expected) {
        assert_eq!(
            a.re.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.re.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(
            a.im.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            b.im.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    assert!(tagged.orbitals().is_some());
    assert!(TaggedDm::dense(2, expected).unwrap().orbitals().is_none());
}

#[test]
fn rejects_mismatched_kpoints_and_invalid_occupations() {
    assert!(TaggedDm::from_orbitals(2, vec![CTensor::zeros(4)], vec![]).is_err());
    assert!(TaggedDm::from_orbitals(2, vec![CTensor::zeros(4)], vec![vec![-1.0, 0.0]]).is_err());
    assert!(TaggedDm::dense(2, vec![CTensor::zeros(3)]).is_err());
}

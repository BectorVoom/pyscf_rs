use pyscf_algebra::CTensor;
use pyscf_pbc_mp::LovTable;

#[test]
fn l_is_the_fastest_stored_axis() {
    let table = LovTable {
        nkpts: 1,
        nocc: 1,
        nvir: 2,
        blocks: vec![(
            3,
            CTensor::from_planes(vec![0., 1., 2., 10., 11., 12.], vec![0.; 6]),
        )],
    };
    assert_eq!(table.aux_slice(0, 0, 0, 0), &[0., 1., 2.]);
    assert_eq!(table.aux_slice(0, 0, 0, 1), &[10., 11., 12.]);
}

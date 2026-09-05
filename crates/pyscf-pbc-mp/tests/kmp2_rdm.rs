use pyscf_algebra::CTensor;
use pyscf_pbc_lib::get_kconserv;
use pyscf_pbc_mp::{Rdm2, RdmKind, T2, make_rdm1, make_rdm2};

#[test]
fn zero_amplitudes_have_reference_density() {
    let t2 = T2 {
        nkpts: 1,
        nocc: 1,
        nvir: 1,
        blocks: vec![CTensor::zeros(1)],
    };
    let kc = get_kconserv(
        &[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        &[[0.0; 3]],
    );
    let d1 = make_rdm1(&t2, &kc, &[2], &[1], RdmKind::Padded).unwrap();
    assert_eq!(d1[0].re, vec![2.0, 0.0, 0.0, 0.0]);
    let d2 = make_rdm2(&t2, &kc, &[2], &[1], RdmKind::Padded).unwrap();
    let Rdm2::Padded { nmo, data } = d2 else {
        panic!("padded")
    };
    assert_eq!(nmo, 2);
    assert_eq!(data.re[0], 2.0);
}

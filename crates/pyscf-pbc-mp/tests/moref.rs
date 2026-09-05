use pyscf_algebra::CTensor;
use pyscf_pbc_mp::{mo_coeff_from_kscf, mo_slice};

#[test]
fn column_major_to_row_major_is_applied_once() {
    let (nao, nmo) = (3, 2);
    let mut c = CTensor::zeros(nao * nmo);
    for i in 0..nmo {
        for p in 0..nao {
            c.re[i * nao + p] = p as f64 + 100.0 * i as f64;
            c.im[i * nao + p] = 0.5 * (p + 10 * i) as f64;
        }
    }
    let m = mo_coeff_from_kscf(&c, nao, nmo).unwrap();
    for p in 0..nao {
        for i in 0..nmo {
            assert_eq!(m.c.re[p * nmo + i], p as f64 + 100.0 * i as f64);
        }
    }
}

#[test]
fn slices_reassemble_and_empty_is_an_error() {
    let m = pyscf_pbc_df::df_ao2mo::MoCoeff::real(2, 3, &[1., 2., 3., 4., 5., 6.]);
    let a = mo_slice(&m, 0, 1).unwrap();
    let b = mo_slice(&m, 1, 3).unwrap();
    assert_eq!((a.c.re[0], b.c.re[0], b.c.re[1]), (1., 2., 3.));
    assert!(mo_slice(&m, 1, 1).is_err());
}

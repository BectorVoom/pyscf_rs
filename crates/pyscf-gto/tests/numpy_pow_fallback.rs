//! `svml_pow::numpy_pow` — the SVML port on AVX-512 hosts (bit-exact, so the
//! basis normalisation keeps upstream's bits there) and `powf` elsewhere
//! (Kaggle CPU machines), which must agree with SVML to within a couple of ulp.

use pyscf_gto::svml_pow::{numpy_pow, svml_pow8};

#[test]
fn numpy_pow_is_svml_on_avx512_and_close_to_powf() {
    let avx512 = is_x86_feature_detected!("avx512f");
    for &x in &[1e-3, 0.0656, 0.5, 1.0, 2.946055777701, 10.389228018317, 1.3e4] {
        for l in 0..6 {
            let y = f64::from(l) + 1.5;
            let got = numpy_pow(x, y);
            let reference = x.powf(y);
            let ulp = (got.to_bits() as i64 - reference.to_bits() as i64).unsigned_abs();
            assert!(ulp <= 2, "numpy_pow({x}, {y}) is {ulp} ulp from powf");
            if avx512 {
                assert_eq!(got.to_bits(), svml_pow8(x, y).to_bits(), "AVX-512 host must take the SVML port");
            }
        }
    }
}

use pyscf_algebra::{CTensor, oracle_zdot, oracle_zdot_re, oracle_zdotu, oracle_zdotu_re};

#[test]
fn unconjugated_matches_conjugated_with_conjugated_lhs() {
    let x = CTensor::from_planes(vec![1.0, -2.0, 3.5], vec![0.5, 4.0, -1.0]);
    let y = CTensor::from_planes(vec![2.0, 1.5, -3.0], vec![-1.0, 2.0, 0.25]);
    assert_eq!(oracle_zdotu(&x.conj(), &y), oracle_zdot(&x, &y));
    assert_eq!(oracle_zdotu_re(&x, &y), oracle_zdotu(&x, &y).0);
    assert_eq!(oracle_zdot_re(&x, &y), oracle_zdot(&x, &y).0);
}

#[test]
fn long_input_is_repeatable() {
    let x = CTensor::from_planes(
        (0..4097).map(|i| (i as f64).sin()).collect(),
        (0..4097).map(|i| (i as f64).cos()).collect(),
    );
    assert_eq!(oracle_zdotu(&x, &x), oracle_zdotu(&x, &x));
}

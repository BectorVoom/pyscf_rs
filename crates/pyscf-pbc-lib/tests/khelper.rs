use pyscf_algebra::CTensor;
use pyscf_pbc_lib::KptsHelper;

fn lattice() -> [[f64; 3]; 3] {
    [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]
}

#[test]
fn symmetry_map_covers_every_triple() {
    for n in [1_usize, 2, 4, 8] {
        let kpts: Vec<[f64; 3]> = (0..n)
            .map(|i| [0.0, 0.0, i as f64 * std::f64::consts::PI / n as f64])
            .collect();
        let h = KptsHelper::new(&lattice(), &kpts);
        let mut seen = vec![false; n * n * n];
        for (_, orbit) in h.symm_map.as_ref().unwrap().entries() {
            for &[p, q, r] in orbit {
                seen[(p * n + q) * n + r] = true;
            }
        }
        assert!(seen.into_iter().all(|v| v));
    }
}

#[test]
fn without_map_is_cheap_and_explicit() {
    let kpts = [[0.0, 0.0, 0.0], [0.0, 0.0, std::f64::consts::PI / 2.0]];
    let eager = KptsHelper::new(&lattice(), &kpts);
    let lazy = KptsHelper::without_symm_map(&lattice(), &kpts);
    assert_eq!(eager.kconserv, lazy.kconserv);
    assert!(lazy.symm_map.is_none());
    assert_eq!(
        lazy.transform_symm(&CTensor::zeros(1), [1; 4], 0, 0, 0),
        Err("KptsHelper symmetry map was not built")
    );
}

#[test]
fn transform_symm_uses_the_recorded_operation() {
    let kpts = [[0.0, 0.0, 0.0], [0.0, 0.0, std::f64::consts::PI / 2.0]];
    let h = KptsHelper::new(&lattice(), &kpts);
    let shape = [2, 3, 4, 5];
    let n: usize = shape.iter().product();
    let x = CTensor::from_planes(
        (0..n).map(|i| i as f64).collect(),
        (0..n).map(|i| -(i as f64)).collect(),
    );
    for p in 0..2 {
        for q in 0..2 {
            for r in 0..2 {
                let y = h.transform_symm(&x, shape, p, q, r).unwrap();
                assert_eq!(y.len(), n);
            }
        }
    }
}

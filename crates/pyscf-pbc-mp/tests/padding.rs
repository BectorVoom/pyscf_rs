use pyscf_mp2::Frozen;
use pyscf_pbc_mp::{
    FrozenK, KCount, PaddingIdx, PaddingKind, get_frozen_mask, get_nmo, get_nocc, padding_k_idx,
};

#[test]
fn upstream_padding_example() {
    let got = padding_k_idx(&[6, 6, 5], &[2, 3, 2], PaddingKind::Split).unwrap();
    assert_eq!(
        got,
        PaddingIdx::Split {
            occupied: vec![vec![0, 1], vec![0, 1, 2], vec![0, 1]],
            virtuals: vec![vec![0, 1, 2, 3], vec![1, 2, 3], vec![1, 2, 3]]
        }
    );
    assert_eq!(
        padding_k_idx(&[6, 6, 5], &[2, 3, 2], PaddingKind::Joint).unwrap(),
        PaddingIdx::Joint(vec![
            vec![0, 1, 3, 4, 5, 6],
            vec![0, 1, 2, 4, 5, 6],
            vec![0, 1, 4, 5, 6]
        ])
    );
}

#[test]
fn dense_dimension_is_max_occ_plus_max_vir() {
    let occ = vec![
        vec![2., 2., 0., 0., 0., 0.],
        vec![2., 2., 2., 0., 0., 0.],
        vec![2., 2., 0., 0., 0.],
    ];
    assert_eq!(
        get_nocc(&occ, &FrozenK::default(), false).unwrap(),
        KCount::Dense(3)
    );
    assert_eq!(
        get_nmo(&occ, &FrozenK::default(), false).unwrap(),
        KCount::Dense(7)
    );
}

#[test]
fn frozen_above_fermi_does_not_reduce_nocc() {
    let occ = vec![vec![2., 2., 0., 0.]];
    let f = FrozenK::Uniform(Frozen::List(vec![3]));
    assert_eq!(get_nocc(&occ, &f, false).unwrap(), KCount::Dense(2));
    assert_eq!(get_nmo(&occ, &f, false).unwrap(), KCount::Dense(3));
    assert_eq!(
        get_frozen_mask(&occ, &f).unwrap(),
        vec![vec![true, true, true, false]]
    );
}

#[test]
fn fractional_occupations_are_refused() {
    assert!(get_nocc(&[vec![2.0, 0.5, 0.0]], &FrozenK::default(), false).is_err());
}

use pyscf_algebra::CTensor;
use pyscf_pbc_mp::{FrozenU, KCount, Kump2, PaddingIdx, PaddingKind, PbcMpError};
use pyscf_pbc_scf::KScfResult;

fn open_shell_result() -> KScfResult {
    let occ_a = vec![vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 0.0]];
    let occ_b = vec![vec![1.0, 0.0, 0.0], vec![1.0, 0.0, 0.0]];
    let mut mo_occ = occ_a;
    mo_occ.extend(occ_b);
    KScfResult {
        e_tot: -1.0,
        e_elec: -1.0,
        e_coul: 0.0,
        e_nuc: 0.0,
        mo_energy: vec![vec![-1.0, -0.5, 0.2]; 4],
        mo_coeff: vec![CTensor::zeros(9); 4],
        mo_occ,
        dm: Vec::new(),
        converged: true,
        cycles: 1,
        nset: 2,
        nkpts: 2,
        fermi: vec![0.0; 2],
        e_free: None,
        e_zero: None,
    }
}

#[test]
fn refusal_names_the_upstream_surface() {
    let e = PbcMpError::Kump2NotImplemented;
    let message = e.to_string();
    assert!(message.contains("kump2.py:38"));
    assert!(message.contains(":384"));
    assert!(message.contains(":402"));
}

#[test]
fn unrestricted_bookkeeping_surface_is_usable() {
    let result = open_shell_result();
    let mut mp = Kump2::new(&result).expect("KUMP2 construction");
    assert_eq!(
        mp.get_nocc(true).unwrap(),
        [KCount::PerKpoint(vec![2, 1]), KCount::PerKpoint(vec![1, 1])]
    );
    assert_eq!(
        mp.get_nmo(false).unwrap(),
        [KCount::Dense(4), KCount::Dense(3)]
    );
    let split = mp.padding_k_idx(PaddingKind::Split).unwrap();
    assert!(matches!(split[0], PaddingIdx::Split { .. }));
    assert!(mp.dump_flags().unwrap().contains("nkpts=2"));

    mp.frozen = FrozenU::PerSpin(Default::default(), Default::default());
    assert_eq!(mp.get_frozen_mask().unwrap()[0], vec![vec![true; 3]; 2]);
    assert!(matches!(mp.kernel(), Err(PbcMpError::Kump2NotImplemented)));
    assert!(matches!(
        mp.add_padding(),
        Err(PbcMpError::Kump2NotImplemented)
    ));
}

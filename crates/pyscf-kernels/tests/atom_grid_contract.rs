#![cfg(feature = "cpu")]

use cubecl::Runtime;
use pyscf_algebra::{AlgebraClient, oracle_sum};
use pyscf_kernels::pbc::multigrid_grad::contract_atom_grid;

fn client() -> AlgebraClient {
    AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice))
}

#[test]
fn complex_rows_match_literal_oracle_reduction() {
    let client = client();
    let (natm, ng) = (3, 257);
    let fr: Vec<f64> = (0..natm * 3 * ng)
        .map(|i| (i % 31) as f64 / 8.0 - 2.0)
        .collect();
    let fi: Vec<f64> = (0..fr.len())
        .map(|i| (i % 19) as f64 / 16.0 - 0.5)
        .collect();
    let rr: Vec<f64> = (0..ng).map(|i| (i % 7) as f64 - 3.0).collect();
    let ri: Vec<f64> = (0..ng).map(|i| (i % 11) as f64 / 4.0).collect();
    let actual = contract_atom_grid(&client, natm, &fr, &fi, &rr, &ri).unwrap();
    for row in 0..natm * 3 {
        let terms: Vec<f64> = (0..ng)
            .map(|g| fr[row * ng + g] * rr[g] - fi[row * ng + g] * ri[g])
            .collect();
        let expected = oracle_sum(&terms);
        assert_eq!(actual[row / 3][row % 3].to_bits(), expected.to_bits());
    }
    assert_eq!(
        actual,
        contract_atom_grid(&client, natm, &fr, &fi, &rr, &ri).unwrap()
    );
    println!("atom-grid exact rows: {actual:?}");
}

#[test]
fn shape_validation_and_empty_grids() {
    let c = client();
    assert_eq!(
        contract_atom_grid(&c, 2, &[], &[], &[], &[]).unwrap(),
        vec![[0.0; 3]; 2]
    );
    assert!(contract_atom_grid(&c, 1, &[0.0; 3], &[0.0; 2], &[1.0], &[0.0]).is_err());
    assert!(contract_atom_grid(&c, 1, &[0.0; 3], &[0.0; 3], &[1.0], &[]).is_err());
    assert!(contract_atom_grid(&c, usize::MAX, &[], &[], &[1.0], &[0.0]).is_err());
}

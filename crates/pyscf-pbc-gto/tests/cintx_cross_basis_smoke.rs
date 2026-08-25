//! R-02 risk probe — cintx cross-basis shell pair evaluation.
//! Proves that cintx can evaluate a shell pair whose shells come from different
//! source molecules (the mechanism D-PBC-07 depends on for periodic 1e integrals).

use cintx_core::Representation;
use cintx_ops::resolver::Resolver;
use cintx_rs::SessionRequest;
use cintx_runtime::ExecutionOptions;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, intor};

#[test]
fn test_cintx_cross_basis_shell_pair_smoke() {
    let mol_a = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("build mol_a");

    let mol_b = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 2.0".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("build mol_b");

    let mol_ab = M(MoleBuildArgs {
        atom: AtomInput::String("H 0 0 0; H 0 0 2.0".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Bohr,
        ..Default::default()
    })
    .expect("build mol_ab");

    let (combined_basis, n_a_shells, n_b_shells) =
        pyscf_gto::projection::build_combined_basis(&mol_a, &mol_b).expect("build_combined_basis");

    assert_eq!(n_a_shells, mol_a.nbas);
    assert_eq!(n_b_shells, mol_b.nbas);

    // Evaluate int1e_ovlp_sph for shell pair [0, mol_a.nbas]
    let descriptor =
        Resolver::descriptor_by_symbol("int1e_ovlp_sph").expect("resolve int1e_ovlp_sph");
    let operator = descriptor.id;
    let representation = Representation::Spheric;
    let opts = ExecutionOptions::default();

    let shells = combined_basis
        .shell_tuple_for_indices([0, mol_a.nbas])
        .expect("shell_tuple_for_indices");

    let request = SessionRequest::new(operator, representation, &combined_basis, shells, opts);

    let outcome = request
        .query_workspace()
        .expect("query_workspace")
        .evaluate()
        .expect("evaluate");

    let cross_val = outcome.tensor.owned_values[0];

    // Reference from single Mole with both atoms
    let s_ab = intor(&mol_ab, "int1e_ovlp").expect("intor for mol_ab");
    let nao = mol_ab.nao_nr;
    assert_eq!(nao, 2);
    // (0, 1) element in F-order: row 0, col 1 -> index 0 + 1 * nao = 2
    let ref_val = s_ab.values[nao];

    let diff = (cross_val - ref_val).abs();
    assert!(
        diff < 1e-12,
        "cintx cross-basis shell-pair overlap mismatch: cross={cross_val:.15e}, ref={ref_val:.15e}, diff={diff:.3e}"
    );
}

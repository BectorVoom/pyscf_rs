//! Wave-0 scaffold — DFT-07: density-fitted DFT (`dft.RKS(mol).density_fit()`)
//! total energy matches upstream PySCF.
//!
//! Upstream reference: `pyscf/dft/rks.py` + `pyscf/df/df_jk.py` (DF J/K).
//! Owning plan: 04-07 (RSH/VV10/DF) unignores this.
//! Verify command: `cargo test -p pyscf-dft df_dft_match`.

#[test]
#[ignore = "Phase 4 04-07: DF-DFT energy match (RKS.density_fit()) — unignore when the DF-DFT path lands"]
fn df_dft_match() {
    // 04-07 fills: build dft.RKS(mol).density_fit(), run to convergence, and
    // assert the total energy matches the upstream PySCF oracle.
    unimplemented!("04-07 fills the df_dft_match assertion");
}

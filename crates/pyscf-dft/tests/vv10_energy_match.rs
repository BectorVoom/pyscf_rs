//! Wave-0 scaffold — DFT-06: VV10 non-local-correlation energy match
//! (`mf.nlc='VV10'`) vs upstream PySCF.
//!
//! Upstream reference: `pyscf/dft/numint.py` VV10 kernel + `rks.py` nlc path.
//! Owning plan: 04-07 (RSH/VV10/DF) unignores this.
//! Verify command: `cargo test -p pyscf-dft vv10_energy_match`.

#[test]
#[ignore = "Phase 4 04-07: VV10 non-local-correlation energy match — unignore when the VV10 path lands"]
fn vv10_energy_match() {
    // 04-07 fills: enable mf.nlc='VV10', run to convergence, and assert the
    // total energy matches the upstream PySCF oracle.
    unimplemented!("04-07 fills the vv10_energy_match assertion");
}

//! Wave-0 scaffold — DFT-10: `NumInt` public signatures (`eval_xc`,
//! `eval_rho`, `nr_rks`, `nr_uks`) match upstream PySCF, plus numeric agreement
//! on a fixed density.
//!
//! Upstream reference: `pyscf/dft/numint.py` (NumInt class surface).
//! Owning plan: 04-06 (RKS core wires NumInt) unignores this.
//! Verify command: `cargo test -p pyscf-dft numint_signatures`.

#[test]
#[ignore = "Phase 4 04-06: NumInt signature + numeric parity vs upstream — unignore when NumInt lands"]
fn numint_signatures() {
    // 04-06 fills: assert NumInt's eval_xc/eval_rho/nr_rks/nr_uks accept the
    // same arguments as upstream and produce matching numeric output.
    unimplemented!("04-06 fills the numint_signatures parity assertion");
}

//! Wave-0 scaffold — DFT-05: CAM-B3LYP / H2O range-separated-hybrid (RSH)
//! energy parity fixture vs upstream PySCF.
//!
//! Upstream reference: `pyscf/dft/rks.py` RSH path + `libxc.py` `RSH(...)`
//! omega/alpha/beta handling. The real bit-exact run is `--features libxc`-
//! gated (CI only) per 04-VALIDATION.md, but the test token stays discoverable
//! under default features (NOT `#[cfg(feature = "libxc")]`-gated at file level)
//! so `cargo test -p pyscf-dft cam_b3lyp_h2o_rsh -- --list` resolves it.
//! Owning plan: 04-07 (RSH/VV10/DF) unignores this.
//! Verify command:
//!   `cargo test --features libxc -p pyscf-dft cam_b3lyp_h2o_rsh`.

#[test]
#[ignore = "Phase 4 04-07: CAM-B3LYP/H2O RSH energy parity (libxc-gated run) — unignore when the RSH path lands"]
fn cam_b3lyp_h2o_rsh() {
    // 04-07 fills: run CAM-B3LYP on H2O and assert the RSH total energy matches
    // the upstream PySCF oracle fixture.
    unimplemented!("04-07 fills the cam_b3lyp_h2o_rsh energy parity assertion");
}

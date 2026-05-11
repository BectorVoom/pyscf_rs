//! ORACLE-08 chkfile round-trip oracle invocation.
//!
//! Phase 3 plan 03-02 ships this stub: the test compiles + exercises the
//! `oracle_check!` macro at the type-level. Plan 03-08 unmarks `#[ignore]`
//! once the macro body lands.
use pyscf_oracle::oracle_check;

#[test]
#[ignore = "macro body pending — plan 03-08"]
fn chkfile_roundtrip_h2o_ccpvdz() {
    oracle_check!("chkfile_roundtrip", "h2o_ccpvdz", 1e-12);
}

//! Test fixtures — string keys identifying canonical pyscf-rs ↔ upstream pairs.
//!
//! RED commit: constant declarations only. GREEN commit fills the geometry +
//! basis + spin lookup helpers consumed by `runner.rs`.

pub const H2O_CC_PVDZ: &str = "h2o_ccpvdz";
pub const BENZENE_6_31GS: &str = "benzene_631gs";
pub const WATER_TRIMER_CC_PVDZ: &str = "water_trimer_ccpvdz";
pub const H2O_TRIPLET_CCPVDZ: &str = "h2o_triplet_ccpvdz";

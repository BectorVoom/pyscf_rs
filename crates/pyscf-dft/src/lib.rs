// Phase 4 (REQ-IDs: DFT-01..11). Phase 1 shipped this empty so
// `cargo build --workspace` compiled all members (FOUND-01); Phase 4 fills it.
#![forbid(unsafe_code)]

// Plan 04-05 (DFT-02 parser parity + DFT-03 bit-exact eval): the XC-string
// parsers (libxc-default + xcfun-alternate) and the DftError type. NOTE: the
// final curated `pub use` re-export surface (`pyscf_dft::parse_xc`, etc.) is
// owned by plan 04-06's lib.rs wiring; this plan exposes the modules directly
// (`pyscf_dft::parser::libxc::parse_xc`) so the integration tests can drive
// them and 04-06 can layer re-exports on top without a wave conflict.
pub mod error;
pub mod parser;
pub mod xc_backend;

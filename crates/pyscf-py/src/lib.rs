// Phase 1 stub — implemented in Phase 3 (REQ-IDs: BIND-01,02,04..07,09).
// Phase 1 ships this empty so `cargo build --workspace` compiles all 15
// members (FOUND-01). The `crate-type = ["cdylib", "rlib"]` declaration in
// Cargo.toml is the BIND-01 forward-compat anchor for the Phase 3 abi3-py310
// wheel build.
// TODO: implemented in Phase 3.
#![forbid(unsafe_code)]

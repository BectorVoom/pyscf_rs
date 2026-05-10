// Phase 1 stub — implemented in Phase 3 (REQ-IDs: ORACLE-01,02,08).
// Phase 1 ships this empty so `cargo build --workspace` compiles all 15
// members (FOUND-01). The `pyo3` dependency lives ONLY in [dev-dependencies]
// (ORACLE-01) so release wheels NEVER link Python.
// TODO: implemented in Phase 3.
#![forbid(unsafe_code)]

"""Oracle test harness for pyscf-rs Phase 2 byte-identity assertions.

Tests in this directory load both upstream `pyscf` (in-process, from the
vendored `pyscf/` checkout at the repo root) and the pyscf-rs CLI / library
output (via `cargo test`-driven JSON dumps), then diff `_atm` / `_bas` /
`_env` / `ao_loc_nr` / `nao_nr` byte-for-byte against upstream.

Gated by the `release-oracle` Cargo profile in the calling Rust tests
(see Phase 1 D-08 + this repo's profile entry in Cargo.toml).
"""

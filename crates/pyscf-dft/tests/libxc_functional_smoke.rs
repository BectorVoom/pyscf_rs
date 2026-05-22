//! Wave-0 scaffold — DFT-03 (libxc-gated, CI ONLY): ~100-functional libxc
//! smoke over a corpus subset — each functional evaluates without error and
//! returns finite values.
//!
//! Upstream reference: `pyscf/dft/libxc.py` functional registry.
//! Owning plan: 04-09 (libxc-gated backend) unignores this.
//! Verify command:
//!   `cargo test --features libxc -p pyscf-dft libxc_functional_smoke`.
//!
//! ENTIRELY behind `#[cfg(feature = "libxc")]` — under default features this
//! file contributes ZERO test functions, so a default `cargo test` NEVER
//! compiles or links libxc_rs (Pitfall 5: libxc_rs = 266 kernels, ~6h freeze).

#![cfg(feature = "libxc")]

#[test]
#[ignore = "Phase 4 04-09: ~100-functional libxc smoke — runs only in the cached --features libxc CI job"]
fn libxc_functional_smoke() {
    // 04-09 fills: iterate the corpus-subset functional list, evaluate each via
    // libxc_rs and assert finite, error-free output (smoke — no oracle compare).
    unimplemented!("04-09 fills the libxc_functional_smoke loop");
}

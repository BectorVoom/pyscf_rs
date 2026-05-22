//! Wave-0 scaffold — DFT-03 (libxc-gated, CI ONLY): XC functional evaluation
//! routes `libxc:*` names to `libxc_rs` and `xcfun:*` to `xcfun_rs`, producing
//! values bit-identical to the C reference.
//!
//! Upstream reference: `pyscf/dft/libxc.py:eval_xc` (C-libxc bridge).
//! Owning plan: 04-09 (libxc-gated backend) unignores this.
//! Verify command:
//!   `cargo test --features libxc -p pyscf-dft xc_eval_bitexact`.
//!
//! ENTIRELY behind `#[cfg(feature = "libxc")]` — under default features this
//! file contributes ZERO test functions, so a default `cargo test` NEVER
//! compiles or links libxc_rs (Pitfall 5: libxc_rs = 266 kernels, ~6h freeze).

#![cfg(feature = "libxc")]

#[test]
#[ignore = "Phase 4 04-09: libxc-gated eval_xc bit-exact vs C — runs only in the cached --features libxc CI job"]
fn xc_eval_bitexact() {
    // 04-09 fills: evaluate XC on a fixed (rho, sigma, ...) grid through both
    // the libxc_rs and xcfun_rs backends and assert bit-identical to the C
    // reference values captured in the gated fixture.
    unimplemented!("04-09 fills the libxc-gated xc_eval_bitexact assertions");
}

//! Wave-0 scaffold — DFT-02: XC-string parser parity vs
//! `pyscf/dft/libxc.py:parse_xc` (single name, comma X/C split, shorthand
//! aliases, `*` weight factors, `RSH(...)`, `HF`/`SR_HF`/`LR_HF`, raw integer
//! IDs, `E_`→`E-` exponent fixup, unit-scaled compound names).
//!
//! Upstream reference: `pyscf/dft/libxc.py:parse_xc` (lines 491-718) +
//! `XC_CODES` (154) + `XC_ALIAS` (217); `pyscf/dft/xcfun.py:parse_xc` (alt).
//! Owning plan: 04-05 (XC parser) unignores this and fills the parity table.
//! Verify command: `cargo test -p pyscf-dft parse_xc_parity`.

#[test]
#[ignore = "Phase 4 04-05: parse_xc parity vs libxc.py:parse_xc — unignore when the XC parser lands"]
fn parse_xc_parity() {
    // 04-05 fills: feed each XC string sub-case through pyscf-dft's parse_xc and
    // assert the (hyb, alpha, omega), ((id, fac), ...) shape matches upstream.
    unimplemented!("04-05 fills the parse_xc parity assertions");
}

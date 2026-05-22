//! Wave-0 scaffold — DFT-11 / D-08: `PYSCF_DTYPE=f32` DFT path runs
//! END-TO-END. This is a runs-to-completion SMOKE ONLY: it asserts that an RKS
//! run completes under f32 precision AND that the below-bit-exact `tracing::warn!`
//! is emitted. It performs **NO oracle comparison** and carries **NO numeric-
//! parity tolerance gate** (D-08 contract — a loose chemical-accuracy f32 bound
//! is a Deferred Idea, not a v1 gate; the oracle/bit-exact rows stay f64-only).
//!
//! Reference: D-08 precision switch (`DType::from_env` in pyscf-runtime threaded
//! through NumInt via the pyscf-algebra `DeviceScalar` matmul element bound).
//! NOT `#[cfg(feature = "libxc")]`-gated — runs on the default CPU/xcfun backend.
//! Owning plan: 04-06 (RKS core threads DType through NumInt) unignores this.
//! Verify command: `cargo test -p pyscf-dft dtype_f32_smoke`.

#[test]
#[ignore = "Phase 4 04-06 (D-08): PYSCF_DTYPE=f32 RKS runs-end-to-end smoke + below-bit-exact warn; NO oracle compare / NO tolerance gate — unignore when 04-06 threads DType through NumInt"]
fn dtype_f32_smoke() {
    // 04-06 fills: set PYSCF_DTYPE=f32, run dft.RKS(mol).run() to completion on
    // the default CPU/xcfun backend, assert it converges (runs end-to-end) and
    // that the below-bit-exact tracing::warn! fires. NO oracle compare, NO
    // tolerance gate (D-08 — f32 is never compared to the upstream oracle).
    unimplemented!("04-06 fills the dtype_f32_smoke runs-end-to-end check (no oracle, no tolerance)");
}

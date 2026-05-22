//! Wave-0 scaffold — DFT-01: RKS/UKS total energy ≤ 1 µHartree vs upstream
//! PySCF for SVWN / PBE / B3LYP under the `release-oracle` profile (FMA-free,
//! ordered reductions). f64 ONLY — the f32 precision switch is covered
//! separately by `dtype_f32_smoke` (D-08) with NO oracle compare.
//!
//! Upstream reference: `pyscf/dft/rks.py` / `pyscf/dft/uks.py`.
//! Owning plan: 04-06 (RKS core) unignores this and fills the energy oracle.
//! Verify command:
//!   `cargo test --profile release-oracle -p pyscf-dft rks_uks_bitexact`.

#[test]
#[ignore = "Phase 4 04-06: RKS/UKS energy ≤1µHa oracle (SVWN/PBE/B3LYP, f64) — unignore when RKS core lands"]
fn rks_uks_bitexact() {
    // 04-06 fills: run RKS/UKS to convergence on the corpus and assert total
    // energy agrees with the in-process PySCF oracle within 1 µHartree (f64).
    unimplemented!("04-06 fills the rks_uks_bitexact energy oracle");
}

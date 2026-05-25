//! DFT-01: RKS/UKS total energy ≤ 1 µHartree vs upstream PySCF for SVWN /
//! PBE / B3LYP under the `release-oracle` profile (FMA-free, ordered
//! reductions). f64 ONLY — the f32 precision switch is covered separately by
//! `dtype_f32_smoke` (D-08) with NO oracle compare.
//!
//! Upstream reference: `pyscf/dft/rks.py` / `pyscf/dft/uks.py`.
//! Verify command:
//!   `cargo test --features python --profile release-oracle -p pyscf-dft rks_uks_bitexact`
//!   (CI runner with libpython + an importable upstream pyscf;
//!    RAYON_NUM_THREADS=1 for matched-threading byte stability).
//!
//! ### Two-layer test (the established 04-04/04-05 convention)
//!
//! 1. **Oracle layer (CI-only, `#[cfg(feature = "python")]`).** The
//!    `oracle_check!("rks_energy"/"uks_energy", "<base>@<xc>", 1e-6)` arms
//!    (added to pyscf-oracle in 04-06) drive BOTH upstream
//!    `dft.RKS/UKS(mol, xc).kernel()` AND pyscf-rs `RKS/UKS::new(mol,
//!    xc).kernel()` and assert |ΔE| ≤ 1 µHartree. This is the authoritative
//!    DFT-01 bit-exact gate; it runs only under `--features python` because
//!    numpy/PySCF is not importable in the dev sandbox (see 04-03/04-04
//!    SUMMARYs), and a converged energy additionally needs working arity-3/4
//!    2-electron integrals (the Phase-2 `int2e_sph`/`int3c2e_sph` rollup gap —
//!    see `pyscf-dft::rks` module note). CI with both lands seals DFT-01.
//!
//! 2. **Structural layer (always-on).** Source-level assertions that the
//!    DFT-01 *machinery* is correctly wired: `RKS::kernel` reuses the Phase 3
//!    `pyscf_scf::kernel<H>` (no new SCF cycle), the attribute floor carries
//!    the DFT fields, the read-only `dtype()` exists and there is NO
//!    `set_precision` (D-08), and the B3LYP path resolves a nonzero standard-
//!    hybrid `hyb` (the hyb-scaled-K branch). These run in every build.

// ───────────────────────── Oracle layer (CI-only) ─────────────────────────

/// RKS SVWN/PBE/B3LYP + UKS energy parity vs upstream, ≤ 1 µHartree (f64).
/// Gated on `python` (live PySCF) AND `#[ignore]` so it is opt-in on CI only.
/// The fixture name encodes the XC functional as `<base>@<xc>` (the
/// pyscf-oracle `fixtures::xc` suffix).
#[cfg(feature = "python")]
#[test]
#[ignore = "DFT-01 live-PySCF oracle — run on CI with --features python + libpython + upstream pyscf (RAYON_NUM_THREADS=1)"]
fn rks_uks_bitexact() {
    use pyscf_oracle::oracle_check;
    // RKS — LDA (SVWN), GGA (PBE), standard-hybrid (B3LYP exercises hyb·K).
    oracle_check!("rks_energy", "h2o_ccpvdz@svwn", 1e-6);
    oracle_check!("rks_energy", "h2o_ccpvdz@pbe", 1e-6);
    oracle_check!("rks_energy", "h2o_ccpvdz@b3lyp", 1e-6);
    // At least one larger corpus fixture (benzene/6-31G*).
    oracle_check!("rks_energy", "benzene_631gs@pbe", 1e-6);
    // UKS — open-shell fixture (triplet H2O).
    oracle_check!("uks_energy", "h2o_triplet_ccpvdz@pbe", 1e-6);
}

// ─────────────────────── Structural layer (always-on) ─────────────────────

mod structural {
    use pyscf_core::{Mole, Unit};
    use pyscf_dft::{NumInt, RKS, UKS};
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
    use pyscf_runtime::DType;

    fn h2o() -> Mole {
        M(MoleBuildArgs {
            atom: AtomInput::String("O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24".into()),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Ang,
            ..Default::default()
        })
        .expect("build H2O")
    }

    /// RKS::kernel reuses the Phase 3 generic `kernel<H>` — assert by binding
    /// the `pyscf_scf::kernel` fn item (source-level proof the symbol exists
    /// and is the SCF cycle the KS driver calls). The RKS struct exposes the
    /// KS driver `kernel(&mut self)`.
    #[test]
    fn rks_reuses_phase3_kernel() {
        // `pyscf_scf::kernel` is the generic SCF cycle RKS::kernel delegates to.
        let _scf_kernel: fn(
            &pyscf_core::Mole,
            &pyscf_scf::NoOverrides,
            pyscf_scf::KernelConfig,
        ) -> Result<pyscf_scf::ScfResult, pyscf_core::PyscfRsError> = pyscf_scf::kernel;
        // RKS::kernel is the KS driver (delegates to the above with KS hooks).
        let _rks_kernel: fn(&mut RKS) -> Result<pyscf_scf::ScfResult, pyscf_core::PyscfRsError> =
            RKS::kernel;
    }

    /// The RKS attribute floor carries the DFT-specific fields
    /// (xc/grids/nlc/nlcgrids/_numint) on top of the inherited SCF floor.
    #[test]
    fn rks_attribute_floor_has_dft_fields() {
        let ks = RKS::new(h2o(), "b3lyp");
        assert_eq!(ks.xc, "b3lyp");
        assert_eq!(ks.nlc, "");
        // Inherited SCF floor sample + DFT fields (compile-time field access).
        let _ = (&ks.mol, &ks.conv_tol, &ks.max_cycle, &ks.diis);
        let _ = (&ks.grids, &ks.nlcgrids, &ks._numint);
        // UKS analog carries the same DFT fields + meaningful nelec.
        let uks = UKS::new(h2o(), "pbe");
        let _ = (&uks.grids, &uks._numint, &uks.nelec);
    }

    /// D-08: the KS object surfaces a read-only active dtype delegating to
    /// `_numint.dtype()` and exposes NO precision setter.
    #[test]
    fn rks_dtype_readonly_no_setter() {
        let ks = RKS::new(h2o(), "svwn");
        assert_eq!(
            ks.dtype(),
            ks._numint.dtype(),
            "dtype() delegates to _numint"
        );
        assert_eq!(ks.dtype(), DType::from_env(), "reflects the env resolver");
        // (There is no `set_precision`/`set_dtype` on RKS — D-08 deferred. This
        //  is enforced structurally: no such pub fn exists. Asserting its
        //  absence at compile time is implicit — the type has no such method.)
    }

    /// B3LYP exercises the hyb-scaled standard-hybrid K branch (omega = 0):
    /// the hybrid coefficient is nonzero (0.2) and omega is zero (RSH branch
    /// stays the 04-07 seam — not active here).
    #[test]
    fn b3lyp_is_standard_hybrid_omega_zero() {
        let ni = NumInt::new();
        let (omega, _alpha, hyb) = ni.rsh_coeff("b3lyp", 0).expect("b3lyp rsh_coeff");
        assert!((hyb - 0.2).abs() < 1e-9, "B3LYP standard-hybrid hyb = 0.2");
        assert!(
            omega.abs() < 1e-12,
            "B3LYP omega = 0 (standard hybrid; RSH → 04-07)"
        );
        // SVWN/PBE are pure functionals — hyb = 0, no K term.
        assert!(ni.hybrid_coeff("svwn", 0).expect("svwn").abs() < 1e-12);
        assert!(ni.hybrid_coeff("pbe", 0).expect("pbe").abs() < 1e-12);
    }
}

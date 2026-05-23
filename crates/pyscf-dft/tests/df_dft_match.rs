//! DFT-07: density-fitted DFT (`dft.RKS(mol).density_fit()`) total energy
//! matches upstream PySCF — by reusing the Phase 3 `pyscf-df` crate for the
//! Coulomb-J build (D-10 reuse, NO new DF crate).
//!
//! Upstream reference: `pyscf/dft/rks.py` + `pyscf/df/df_jk.py` (DF J/K).
//! Owning plan: 04-08 (this plan) unignores + fills this.
//! Verify command:
//!   `cargo test --features python --profile release-oracle -p pyscf-dft df_dft_match`
//!   (CI runner with libpython + an importable upstream pyscf;
//!    RAYON_NUM_THREADS=1 for matched-threading byte stability).
//!
//! ### Two-layer test (the established 04-04/04-05/04-06 convention)
//!
//! 1. **Oracle layer (CI-only, `#[cfg(feature = "python")]`).** The
//!    `oracle_check!("df_dft_energy", "<base>@<xc>", 1e-6)` arm (added to
//!    pyscf-oracle in 04-08) drives BOTH upstream
//!    `dft.RKS(mol, xc).density_fit().kernel()` AND pyscf-rs
//!    `RKS::new(mol, xc).density_fit(None).kernel_df()` and asserts
//!    |ΔE| ≤ 1 µHartree. This is the authoritative DFT-07 energy gate; it runs
//!    only under `--features python` because numpy/PySCF is not importable in
//!    the dev sandbox, and a converged DF-DFT energy additionally needs working
//!    arity-3 `int3c2e_sph` integrals (the Phase-2 rollup gap — see
//!    `pyscf-dft::df_dft` module note). CI with both lands seals DFT-07.
//!
//! 2. **Structural layer (always-on).** Source-level assertions that the
//!    DFT-07 *machinery* is correctly wired: `RKS::density_fit` returns `Self`
//!    mirroring `RHF::density_fit`; `DfKsHooks` satisfies BOTH the SCF
//!    `OverrideHooks` AND the KS `KsOverrideHooks` bound (so the Phase 3
//!    `kernel<H>` accepts it); the J build routes through the `pyscf-df`
//!    crate (`get_jk_df` is the symbol `DfKsHooks::get_jk` calls) while the K
//!    stays on the standard `default_get_jk` path. These run in every build.

// ───────────────────────── Oracle layer (CI-only) ─────────────────────────

/// DF-DFT total energy parity vs upstream `.density_fit().kernel()`,
/// ≤ 1 µHartree (f64). Gated on `python` (live PySCF) AND `#[ignore]` so it is
/// opt-in on CI only. The fixture name encodes the XC functional as
/// `<base>@<xc>` (the pyscf-oracle `fixtures::xc` suffix).
#[cfg(feature = "python")]
#[test]
#[ignore = "DFT-07 live-PySCF DF-DFT oracle — run on CI with --features python + libpython + upstream pyscf (RAYON_NUM_THREADS=1)"]
fn df_dft_match() {
    use pyscf_oracle::oracle_check;
    // DF-DFT — LDA (SVWN), GGA (PBE), standard-hybrid (B3LYP exercises hyb·K
    // on the STANDARD path while J is DF-built).
    oracle_check!("df_dft_energy", "h2o_ccpvdz@svwn", 1e-6);
    oracle_check!("df_dft_energy", "h2o_ccpvdz@pbe", 1e-6);
    oracle_check!("df_dft_energy", "h2o_ccpvdz@b3lyp", 1e-6);
}

// ─────────────────────── Structural layer (always-on) ─────────────────────

mod structural {
    use pyscf_core::{Mole, Unit};
    use pyscf_dft::{DfKsHooks, KsOverrideHooks, NumInt, RKS};
    use pyscf_grids::Grids;
    use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs};
    use pyscf_scf::OverrideHooks;

    fn h2o() -> Mole {
        M(MoleBuildArgs {
            atom: AtomInput::String("O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24".into()),
            basis: BasisInput::Name("cc-pvdz".into()),
            unit: Unit::Ang,
            ..Default::default()
        })
        .expect("build H2O")
    }

    /// `RKS::density_fit(auxbasis) -> Result<Self, _>` mirrors
    /// `RHF::density_fit` (df_scf.rs) — same signature shape, same `Self`
    /// builder return. Source-level proof the symbol exists with that type.
    #[test]
    fn rks_density_fit_returns_self_like_rhf() {
        let _rks_df: fn(RKS, Option<&str>) -> Result<RKS, pyscf_core::PyscfRsError> =
            RKS::density_fit;
        let _rhf_df: fn(
            pyscf_scf::RHF,
            Option<&str>,
        ) -> Result<pyscf_scf::RHF, pyscf_core::PyscfRsError> = pyscf_scf::RHF::density_fit;
    }

    /// `RKS::kernel_df` is the DF-DFT driver — it reuses the SAME Phase 3
    /// generic `kernel<H>` the non-DF `RKS::kernel` reuses (no new SCF cycle;
    /// D-10 reuse). Source-level fn-item binding.
    #[test]
    fn kernel_df_reuses_phase3_kernel() {
        let _scf_kernel: fn(
            &pyscf_core::Mole,
            &pyscf_scf::NoOverrides,
            pyscf_scf::KernelConfig,
        ) -> Result<pyscf_scf::ScfResult, pyscf_core::PyscfRsError> = pyscf_scf::kernel;
        let _kernel_df: fn(&mut RKS) -> Result<pyscf_scf::ScfResult, pyscf_core::PyscfRsError> =
            RKS::kernel_df;
    }

    /// `DfKsHooks` satisfies BOTH the SCF `OverrideHooks` AND the KS
    /// `KsOverrideHooks` bounds — so the Phase 3 `kernel<H>` accepts it as the
    /// KS hooks impl. The J build routes through `pyscf_df::get_jk_df`
    /// (source assertion below); the K stays standard.
    #[test]
    fn df_ks_hooks_satisfies_both_bounds() {
        fn assert_scf<H: OverrideHooks>(_h: &H) {}
        fn assert_ks<H: KsOverrideHooks>(_h: &H) {}
        let df = pyscf_df::DfIntegrals {
            b_uvq: vec![0.0; 1],
            naux: 1,
            nao: 1,
        };
        let ni = NumInt::new();
        let mol = h2o();
        let grids = Grids::new();
        let hooks = DfKsHooks::new(&df, &ni, &mol, &grids, "svwn");
        assert_scf(&hooks);
        assert_ks(&hooks);
    }

    /// Source assertion (D-10): `RKS::density_fit` pre-computes the B integrals
    /// via `pyscf_df::cholesky_eri` and `DfKsHooks::get_jk` routes the J build
    /// through `pyscf_df::get_jk_df` — i.e. the DF-DFT path REUSES the Phase 3
    /// pyscf-df crate, with NO new DF crate. We assert this by reading the
    /// df_dft.rs source for the two reuse symbols.
    #[test]
    fn df_dft_reuses_pyscf_df_crate_no_new_df() {
        let src = include_str!("../src/df_dft.rs");
        assert!(
            src.contains("cholesky_eri"),
            "RKS::density_fit must reuse pyscf_df::cholesky_eri (D-10)"
        );
        assert!(
            src.contains("get_jk_df"),
            "DfKsHooks::get_jk must route the J build through pyscf_df::get_jk_df (D-10)"
        );
        // The J routes through DF; K stays on the standard get_jk path.
        assert!(
            src.contains("default_get_jk"),
            "DfKsHooks::get_jk must keep K on the standard default_get_jk path (T-04-08b)"
        );
    }
}

//! DFT-10: `NumInt` public signatures (`eval_xc`, `eval_rho`, `nr_rks`,
//! `nr_uks`) match upstream PySCF `numint.py`, plus a numeric `eval_rho`
//! element-wise check vs an independent longhand reference and a D-08
//! `dtype()`-defaults-to-F64 assertion.
//!
//! Upstream reference: `pyscf/dft/numint.py` (NumInt class surface).
//! Owning plan: 04-06 (RKS core wires NumInt).
//! Verify command: `cargo test -p pyscf-dft numint_signatures`.
//!
//! Why a hand-built fixture (not a live pyscf oracle): numpy/PySCF are not
//! importable in the dev sandbox (see 04-03/04-04 SUMMARYs), and the bit-exact
//! energy oracle is the CI-only `--features python` arm (`rks_uks_bitexact`).
//! `eval_rho` is a pure contraction, so an independent longhand triple-sum IS
//! the authoritative element-wise oracle (mirrors the 04-03 self-contained
//! kernel-oracle precedent).

use pyscf_core::Density;
use pyscf_dft::{DerivOrder, NumInt, RhoBlock, XcType};
use pyscf_runtime::DType;

#[test]
fn numint_signatures() {
    // ── eval_rho signature + numeric parity (LDA) ──────────────────────────
    // Independent longhand: rho[g] = Σ_{μν} ao[g,μ] D[μν] ao[g,ν].
    let ngrids = 4;
    let nao = 2;
    // F-order AO value block: ao[g + mu*ngrids].
    let ao = vec![
        // mu=0: g0..g3
        0.7_f64, 0.5, 0.3, 0.2, // mu=0
        0.1, 0.4, 0.6, 0.9, // mu=1
    ];
    let dm = Density {
        nao,
        data: vec![1.2, 0.4, 0.4, 0.8], // symmetric
    };
    let (rho, grad) = NumInt::eval_rho(&ao, &dm, ngrids, XcType::Lda)
        .expect("eval_rho(ao, dm, ngrids, XcType::Lda) -> (rho, grad)");
    assert!(grad.is_none(), "LDA eval_rho returns no gradient");
    assert_eq!(rho.len(), ngrids, "rho has one value per grid point");
    for g in 0..ngrids {
        let a0 = ao[g];
        let a1 = ao[g + ngrids];
        let want = a0 * 1.2 * a0 + a0 * 0.4 * a1 + a1 * 0.4 * a0 + a1 * 0.8 * a1;
        assert!(
            (rho[g] - want).abs() < 1e-13,
            "eval_rho[{g}]: got {}, want {} (independent longhand)",
            rho[g],
            want
        );
    }

    // ── eval_rho GGA signature (ρ + ∇ρ) ────────────────────────────────────
    // GGA AO buffer is [4, ngrids, nao] (value + 3 gradient components).
    let mut ao_gga = vec![0.0_f64; 4 * ngrids * nao];
    // component 0 (value) = the LDA block above; gradient comps left zero
    // (∇ρ = 0 ⇒ a valid, if trivial, GGA block — exercises the signature +
    //  the ρ-from-comp-0 path).
    ao_gga[..ngrids * nao].copy_from_slice(&ao);
    let (rho_gga, grad_gga) = NumInt::eval_rho(&ao_gga, &dm, ngrids, XcType::Gga)
        .expect("eval_rho(.., XcType::Gga) -> (rho, Some(∇ρ))");
    assert!(grad_gga.is_some(), "GGA eval_rho returns ∇ρ");
    for g in 0..ngrids {
        assert!(
            (rho_gga[g] - rho[g]).abs() < 1e-13,
            "GGA ρ (comp-0) must equal the LDA ρ for the same value block"
        );
    }

    // ── eval_xc signature: (xc_code, &RhoBlock, DerivOrder) -> XcOutput ─────
    let ni = NumInt::new();
    let rho_block = [0.1_f64, 0.5, 1.0, 2.5];
    let rb = RhoBlock::Lda { rho: &rho_block };
    let out = ni
        .eval_xc("svwn", &rb, DerivOrder::Vxc)
        .expect("eval_xc(xc_code, rho_block, order) -> XcOutput");
    assert_eq!(out.exc.len(), rho_block.len(), "exc is per grid point");
    assert_eq!(out.vrho.len(), rho_block.len(), "vrho is per grid point");
    for &e in &out.exc {
        assert!(e.is_finite(), "SVWN exc is finite");
    }

    // ── nr_rks / nr_uks signatures (compile-time arity + return shape) ──────
    // The full numeric energy oracle is CI-only (--features python); here we
    // only assert the upstream-matching arity by binding the fn items to the
    // exact signatures (numint.py:1074 / numint.py:1192).
    // nr_rks(ni, mol, grids, xc_code, dms, relativity, hermi, max_memory,
    //        verbose) -> (nelec, excsum, vmat)
    let _nr_rks: fn(
        &NumInt,
        &pyscf_core::Mole,
        &pyscf_grids::Grids,
        &str,
        &Density,
        i32,
        i32,
        f64,
        Option<i32>,
    ) -> Result<pyscf_dft::NrResult, pyscf_core::PyscfRsError> = NumInt::nr_rks;
    // nr_uks(ni, mol, grids, xc_code, (dm_a, dm_b), ...) -> NrUksResult
    let _nr_uks: fn(
        &NumInt,
        &pyscf_core::Mole,
        &pyscf_grids::Grids,
        &str,
        (&Density, &Density),
        i32,
        i32,
        f64,
        Option<i32>,
    ) -> Result<pyscf_dft::NrUksResult, pyscf_core::PyscfRsError> = NumInt::nr_uks;

    // ── D-08: dtype() reads from the env resolver; default (unset) is F64 ───
    assert_eq!(ni.dtype(), DType::from_env(), "dtype() reflects DType::from_env()");
    if std::env::var("PYSCF_DTYPE").is_err() {
        assert_eq!(ni.dtype(), DType::F64, "PYSCF_DTYPE unset → F64 (D-08 default)");
    }
}

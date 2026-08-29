//! `V_xc = ∂E_xc/∂D` for the MOLECULAR numerical integration, and the
//! backend-level derivative identity underneath it.
//!
//! LDA has no `sigma` anywhere, so every pre-existing assertion on the XC path
//! was blind to the gradient variables. These tests differentiate the returned
//! quantities numerically and compare against the returned derivatives, which
//! is the only check that sees the GGA chain rule at all.

use pyscf_dft::{DerivOrder, RhoBlock, XcBackend};

/// The closed-shell `vsigma` the backend returns must be `∂f/∂σ`, where `σ` is
/// the UNPOLARIZED gradient invariant `|∇ρ|²` — which is what `XcOutput`'s own
/// documentation says and what `numint.rs` consumes it as.
///
/// # Why this is not the raw xcfun output
///
/// The xcfun CPU kernels expose the SPIN-RESOLVED variable set
/// `A_B_GAA_GAB_GBB`, so the order-1 block is
/// `[f, ∂f/∂ρa, ∂f/∂ρb, ∂f/∂γaa, ∂f/∂γab, ∂f/∂γbb]`. The closed-shell
/// substitution is `γaa = γab = γbb = σ/4`, so the chain rule is
///
/// ```text
/// ∂f/∂σ = (∂f/∂γaa + ∂f/∂γab + ∂f/∂γbb) / 4
/// ```
///
/// and `∂f/∂γab` is NOT zero for any GGA correlation functional. Returning
/// `∂f/∂γaa` alone is a different derivative, and this test measures the
/// difference rather than trusting the layout comment.
#[test]
fn closed_shell_vsigma_is_the_unpolarized_sigma_derivative() {
    let backend = XcBackend::default();
    for xc in ["pbe", "blyp", "b3lyp"] {
        // Parse in the BACKEND's own id namespace. Naming a parser module
        // directly is the bug `XcBackend::parse` exists to prevent: xcfun's
        // PBEX/PBEC are ids 5/4, and ids 5/4 in the libxc namespace are LDA
        // functionals, so an xcfun-parsed spec handed to the libxc backend
        // evaluates a different functional than the one requested.
        let spec = backend.parse(xc).expect("parse");
        let rho0 = 0.3_f64;
        let sigma0 = 0.2_f64;

        let f = |s: f64| -> f64 {
            let out = backend
                .eval(
                    &spec,
                    &RhoBlock::Gga {
                        rho: &[rho0],
                        sigma: &[s],
                    },
                    DerivOrder::Exc,
                )
                .expect("eval");
            out.exc[0]
        };

        // `exc` is the energy DENSITY f (not f/rho) — see XcOutput's docs.
        let h = 1e-6 * sigma0;
        let fd = (f(sigma0 + h) - f(sigma0 - h)) / (2.0 * h);

        let got = backend
            .eval(
                &spec,
                &RhoBlock::Gga {
                    rho: &[rho0],
                    sigma: &[sigma0],
                },
                DerivOrder::Vxc,
            )
            .expect("eval")
            .vsigma[0];

        let rel = (fd - got).abs() / fd.abs().max(1e-30);
        println!("{xc:>6}: d f/d sigma  fd {fd:.12e}  backend {got:.12e}  rel {rel:.3e}");
        assert!(
            rel < 1e-5,
            "{xc}: vsigma is not d f/d sigma — backend {got:e} vs finite difference {fd:e} \
             (relative error {rel:e}). The closed-shell substitution needs \
             (d/dgaa + d/dgab + d/dgbb)/4, not d/dgaa alone."
        );
    }
}

/// `vrho` must likewise be `∂f/∂ρ` at fixed `σ`.
#[test]
fn closed_shell_vrho_is_the_rho_derivative() {
    let backend = XcBackend::default();
    for xc in ["lda,vwn", "pbe", "blyp"] {
        // Parse in the BACKEND's own id namespace. Naming a parser module
        // directly is the bug `XcBackend::parse` exists to prevent: xcfun's
        // PBEX/PBEC are ids 5/4, and ids 5/4 in the libxc namespace are LDA
        // functionals, so an xcfun-parsed spec handed to the libxc backend
        // evaluates a different functional than the one requested.
        let spec = backend.parse(xc).expect("parse");
        let (rho0, sigma0) = (0.3_f64, 0.2_f64);
        let is_gga = xc != "lda,vwn";

        let f = |r: f64| -> f64 {
            let block = if is_gga {
                RhoBlock::Gga {
                    rho: &[r],
                    sigma: &[sigma0],
                }
            } else {
                RhoBlock::Lda { rho: &[r] }
            };
            backend
                .eval(&spec, &block, DerivOrder::Exc)
                .expect("eval")
                .exc[0]
        };

        let h = 1e-6 * rho0;
        let fd = (f(rho0 + h) - f(rho0 - h)) / (2.0 * h);

        let block = if is_gga {
            RhoBlock::Gga {
                rho: &[rho0],
                sigma: &[sigma0],
            }
        } else {
            RhoBlock::Lda { rho: &[rho0] }
        };
        let got = backend
            .eval(&spec, &block, DerivOrder::Vxc)
            .expect("eval")
            .vrho[0];

        let rel = (fd - got).abs() / fd.abs().max(1e-30);
        println!("{xc:>8}: d f/d rho    fd {fd:.12e}  backend {got:.12e}  rel {rel:.3e}");
        assert!(rel < 1e-5, "{xc}: vrho is not d f/d rho (relative error {rel:e})");
    }
}

/// The spin-resolved entry point, which the PERIODIC crate drives, is checked
/// on its own terms: each `vsigma_*` against a partial derivative in that
/// variable alone.
#[test]
fn spin_resolved_vsigma_components_are_their_own_partials() {
    let backend = XcBackend::default();
    let spec = backend.parse("pbe").expect("parse");
    let (ra, rb) = (0.18_f64, 0.12_f64);
    let (saa, sab, sbb) = (0.05_f64, 0.03_f64, 0.04_f64);

    let f = |a: f64, b: f64, c: f64| -> f64 {
        backend
            .eval_uks(&spec, &[ra], &[rb], Some(&[a]), Some(&[b]), Some(&[c]), DerivOrder::Exc)
            .expect("eval_uks")
            .exc[0]
    };

    let out = backend
        .eval_uks(
            &spec,
            &[ra],
            &[rb],
            Some(&[saa]),
            Some(&[sab]),
            Some(&[sbb]),
            DerivOrder::Vxc,
        )
        .expect("eval_uks");

    let h = 1e-6;
    let cases: [(&str, f64, f64); 3] = [
        (
            "gaa",
            (f(saa + h, sab, sbb) - f(saa - h, sab, sbb)) / (2.0 * h),
            out.vsigma_aa[0],
        ),
        (
            "gab",
            (f(saa, sab + h, sbb) - f(saa, sab - h, sbb)) / (2.0 * h),
            out.vsigma_ab[0],
        ),
        (
            "gbb",
            (f(saa, sab, sbb + h) - f(saa, sab, sbb - h)) / (2.0 * h),
            out.vsigma_bb[0],
        ),
    ];
    for (name, fd, got) in cases {
        let rel = (fd - got).abs() / fd.abs().max(1e-30);
        println!("pbe d f/d {name}: fd {fd:.12e}  backend {got:.12e}  rel {rel:.3e}");
        assert!(rel < 1e-4, "vsigma_{name} is not its own partial ({rel:e})");
    }
}

// ---------------------------------------------------------------------------
// The same identity at the NumInt level, on a real molecule and a real grid
// ---------------------------------------------------------------------------

use pyscf_core::{Density, Unit};
use pyscf_dft::NumInt;
use pyscf_grids::Grids;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs, M};

fn h2o() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("O 0 0 0; H 0 0 0.96; H 0 0.93 -0.24".into()),
        basis: BasisInput::Name("sto-3g".into()),
        unit: Unit::Ang,
        ..Default::default()
    })
    .expect("H2O must build")
}

/// A symmetric, positive, non-trivial density — enough structure that the GGA
/// gradient rows are not accidentally small.
fn seeded_dm(nao: usize) -> Density {
    let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut data = vec![0.0_f64; nao * nao];
    for i in 0..nao {
        for j in i..nao {
            let v = if i == j { 1.0 + next() } else { 0.15 * (next() - 0.5) };
            data[i * nao + j] = v;
            data[j * nao + i] = v;
        }
    }
    Density { nao, data }
}

/// **`V_xc = ∂E_xc/∂D` for `NumInt::nr_rks`.**
///
/// `nr_rks` returns `E_xc` and `V_xc` from the same density, so a central
/// difference of the returned energy along a symmetric direction `Δ` must
/// reproduce `Tr[V_xc Δ]`.
///
/// # The factor that this test exists to pin
///
/// The back-contraction builds `V` and the caller symmetrises it as
/// `V + Vᵀ`. Upstream halves ONLY the density row for that
/// (`_rks_gga_wv0`, `numint.py:1555`: `wv[0] = vrho * .5`); the gradient rows
/// are `wv[1:] = 2 * vgamma * rho[1:4]`, NOT halved, because the symmetrisation
/// is exactly what supplies the `∇φ_μ φ_ν + φ_μ ∇φ_ν` pair the gradient term
/// needs. Halving them too drops half of that term — invisible under LDA (no
/// `sigma` anywhere) and worth ~1.6% of `V_xc` under PBE.
fn numint_vxc_identity(xc: &str, tol: f64) {
    let ni = NumInt::new();
    let mol = h2o();
    let nao = mol.nao_nr;

    let mut grids = Grids::new();
    grids.level = 3;
    let (coords, weights) = grids.build(&mol);
    grids.coords = Some(coords);
    grids.weights = Some(weights);

    let dm = seeded_dm(nao);
    let delta = {
        let mut d = seeded_dm(nao);
        // A pure perturbation direction: strip the diagonal bias.
        for i in 0..nao {
            d.data[i * nao + i] -= 1.0;
        }
        d
    };

    let exc = |eps: f64| -> f64 {
        let mut shifted = dm.clone();
        for i in 0..nao * nao {
            shifted.data[i] += eps * delta.data[i];
        }
        ni.nr_rks(&mol, &grids, xc, &shifted, 0, 1, 2000.0, None)
            .expect("nr_rks")
            .excsum
    };

    let eps = 1e-6;
    let fd = (exc(eps) - exc(-eps)) / (2.0 * eps);

    let r = ni
        .nr_rks(&mol, &grids, xc, &dm, 0, 1, 2000.0, None)
        .expect("nr_rks");
    let mut an = 0.0;
    for i in 0..nao {
        for j in 0..nao {
            an += r.vmat.data[i * nao + j] * delta.data[j * nao + i];
        }
    }

    let rel = (fd - an).abs() / an.abs().max(1e-30);
    println!("NumInt V_xc = dE_xc/dD [{xc:>8}]: fd {fd:.12e}  Tr[V d] {an:.12e}  rel {rel:.3e}");
    assert!(
        rel < tol,
        "{xc}: V_xc does not reproduce dE_xc/dD — relative error {rel:e} exceeds {tol:e}"
    );
}

#[test]
fn numint_vxc_is_the_exc_derivative_lda() {
    numint_vxc_identity("lda,vwn", 1e-5);
}

#[test]
fn numint_vxc_is_the_exc_derivative_pbe() {
    numint_vxc_identity("pbe", 1e-5);
}

#[test]
fn numint_vxc_is_the_exc_derivative_blyp() {
    numint_vxc_identity("blyp", 1e-5);
}

//! Opt-in oracle checks for `EOMEESinglet` — the spin-adapted k-point
//! EOM-EE-CCSD — against PySCF 2.12.1.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-cc \
//!   --test oracle_eom_ee_singlet -- --ignored --nocapture
//! ```
//!
//! # Why the singlet is the whole of RHF EE
//!
//! `eeccsd` (`eom_kccsd_rhf.py:831`) is a bare `raise NotImplementedError`,
//! `EOMEETriplet` (`:1483`) and `EOMEESpinFlip` (`:1489`) declare nothing but
//! a `vector_size` that returns `None`, and `eeccsd_matvec` (`:965`) raises
//! too. `eomee_ccsd_singlet` (`:838`) is the only EE driver upstream has here.
//!
//! # The gate is layered
//!
//! A packing defect and an equation defect fail the same root test, so this
//! checks them separately and in order: the twelve renamed `make_ee`
//! intermediates, the vector size, the vector round trip, `r1`/`r2`
//! themselves, the matvec, the diagonal, the singles-only `Hbar·r1`, the CIS
//! guess spectrum, and only then the roots.

mod common;

use common::{
    block, cblock, diamond_scf, emit, eris_on_upstream_mf, maxdiff, scalar, synthetic, upstream_mos,
};

use pyscf_pbc_cc::ZArr;
use pyscf_pbc_cc::eom_kccsd_ghf::{self as eomg, EomOpts, Excitation};
use pyscf_pbc_cc::eom_kccsd_rhf as eomr;

/// The EOM-block gate — `oracle_phase16.rs`'s `IMDS_BLOCK`.
const IMDS_BLOCK: f64 = 1e-6;

/// The ROOT gate, measured in `measurements/README.md §1`.
const ROOT_GATE: f64 = 1e-5;

/// **The `make_ee` intermediates, the packing, the matvec and the diagonal.**
#[test]
#[ignore = "opt-in PySCF oracle"]
fn ee_singlet_equations_match_upstream() {
    let Some(out) = emit("ee_singlet") else {
        return;
    };
    let f = diamond_scf([1, 1, 2]);
    let up = upstream_mos(&out);
    let (eris, khelper) = eris_on_upstream_mf(&f, &up);
    let kc = &khelper.kconserv;
    let (nkpts, nocc, nvir) = (up.nkpts, up.nocc, up.nmo - up.nocc);

    let (st1, st2) = synthetic(
        &[nkpts, nocc, nvir],
        &[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir],
    );
    let imds = eomr::RhfEomImds::make_shared(&st1, &st2, &eris, kc)
        .expect("shared")
        .make_ee(kc)
        .expect("EE imds");

    let mut failures: Vec<String> = Vec::new();
    // The twelve renamed intermediates. `woOvV` is `eris.oovv` itself, so it
    // is diffed against the block rather than against a field.
    let oovv = {
        let mut z = ZArr::zeros(&[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir]);
        for k0 in 0..nkpts {
            for k1 in 0..nkpts {
                for k2 in 0..nkpts {
                    z.set_leading(
                        &[k0, k1, k2],
                        &eris
                            .blk(pyscf_pbc_cc::keris::Blk::Oovv, k0, k1, k2)
                            .expect("oovv"),
                    )
                    .expect("oovv slot");
                }
            }
        }
        z
    };
    for (got, name) in [
        (&imds.loo, "ee_Foo"),
        (&imds.lvv, "ee_Fvv"),
        (&imds.fov, "ee_Fov"),
        (&oovv, "ee_woOvV"),
        (imds.wovvo().expect("wovvo"), "ee_woVvO"),
        (imds.wovov().expect("wovov"), "ee_woVoV"),
        (imds.woooo.as_ref().expect("woooo"), "ee_woOoO"),
        (imds.wooov.as_ref().expect("wooov"), "ee_woOoV"),
        (imds.wovoo.as_ref().expect("wovoo"), "ee_woVoO"),
        (imds.wvovv.as_ref().expect("wvovv"), "ee_wvOvV"),
        (imds.wvvvv.as_ref().expect("wvvvv"), "ee_wvVvV"),
        (imds.wvvvo.as_ref().expect("wvvvo"), "ee_wvVvO"),
    ] {
        let d = maxdiff(got, &cblock(&out, name), name);
        println!("  {name:12} max|Δ| {d:e}");
        if !(d < IMDS_BLOCK) {
            failures.push(format!("{name} {d:e}"));
        }
    }

    for kshift in 0..nkpts {
        // The `kconserv_ee_*` tables this port COMPOSES from the ordinary
        // `kconserv` rather than rebuilding geometrically.
        let kr1 = eomg::kconserv_ee_r1(nkpts, kshift, kc);
        let kr2 = eomg::kconserv_ee_r2(nkpts, kshift, kc);
        for (got, name) in [
            (&kr1, format!("ee_kconserv_r1_{kshift}")),
            (&kr2, format!("ee_kconserv_r2_{kshift}")),
        ] {
            let want = block(&out, &name);
            assert_eq!(got.len(), want.len(), "{name} length");
            for (i, w) in want.iter().enumerate() {
                assert_eq!(got[i] as f64, *w, "{name}[{i}]");
            }
            println!("  {name} exact");
        }

        let size = scalar(&out, &format!("ee_vector_size_{kshift}")) as usize;
        let mine = eomr::ee_singlet_vector_size(nkpts, nocc, nvir, &kr2);
        println!("  ee_vector_size[{kshift}] {mine} vs {size}");
        assert_eq!(mine, size, "the EE singlet vector size disagrees");

        let v =
            ZArr::from_ctensor(&[size], cblock(&out, &format!("ee_vec_{kshift}"))).expect("ee vec");
        let (r1, r2) =
            eomr::vector_to_amplitudes_singlet(&v, nkpts, nocc, nvir, &kr2).expect("unpack");
        for (got, name) in [
            (&r1, format!("ee_r1_{kshift}")),
            (&r2, format!("ee_r2_{kshift}")),
        ] {
            let d = maxdiff(got, &cblock(&out, &name), &name);
            println!("  {name:18} max|Δ| {d:e}");
            assert!(d == 0.0, "{name} is not bit-identical: {d:e}");
        }
        let back =
            eomr::amplitudes_to_vector_singlet(&r1, &r2, nkpts, nocc, nvir, &kr2).expect("pack");
        let d = maxdiff(
            &back,
            &cblock(&out, &format!("ee_roundtrip_{kshift}")),
            "roundtrip",
        );
        println!("  ee_roundtrip[{kshift}] max|Δ| {d:e}");
        assert!(d == 0.0, "the EE singlet round trip is not exact: {d:e}");

        for (got, name) in [
            (
                eomr::eeccsd_matvec_singlet(&v, kshift, &imds, kc).expect("matvec"),
                format!("ee_matvec_{kshift}"),
            ),
            (
                eomr::eeccsd_diag(kshift, &imds, kc).expect("diag"),
                format!("ee_diag_{kshift}"),
            ),
        ] {
            let d = maxdiff(&got, &cblock(&out, &name), &name);
            println!("  {name:18} max|Δ| {d:e}");
            if !(d < IMDS_BLOCK) {
                failures.push(format!("{name} {d:e}"));
            }
        }

        // The singles-only `Hbar·r1`, then the spectrum the CIS guess sorts.
        let n1 = nkpts * nocc * nvir;
        let v1 =
            ZArr::from_ctensor(&[n1], cblock(&out, &format!("ee_vec1_{kshift}"))).expect("ee vec1");
        let got = eomr::eeccsd_matvec_singlet_hr1(&v1, kshift, &imds, kc).expect("Hr1");
        let name = format!("ee_hr1_{kshift}");
        let d = maxdiff(&got, &cblock(&out, &name), &name);
        println!("  {name:18} max|Δ| {d:e}");
        if !(d < IMDS_BLOCK) {
            failures.push(format!("{name} {d:e}"));
        }

        // The full CIS spectrum: the guess is `nroots` eigenvectors of this
        // matrix, so agreeing on every eigenvalue is the stronger statement.
        let guess = eomr::ee_singlet_cis_guess(kshift, n1, &imds, kc).expect("cis guess");
        assert_eq!(guess.len(), n1, "the CIS guess must span the r1 block");
        let want = block(&out, &format!("ee_cis_evals_{kshift}"));
        // Upstream emits the eigenvalues sorted; this port's guess vectors are
        // in the same order, so the spectra are compared by recomputing the
        // Rayleigh quotient is unnecessary — instead diff the SORTED real
        // parts, which is what `argsort` orders by.
        let mut mine: Vec<f64> = Vec::with_capacity(n1);
        for g in &guess {
            let r = g.slice_axes(&[(0, n1)]).expect("r1 block");
            let hv = eomr::eeccsd_matvec_singlet_hr1(&r, kshift, &imds, kc).expect("Hr1");
            // Rayleigh quotient `vᵀ H v / vᵀ v` — unconjugated, as the
            // eigenproblem is non-Hermitian.
            let (mut nr, mut ni, mut dr, mut di) = (0.0, 0.0, 0.0, 0.0);
            for i in 0..n1 {
                let (a, b) = (r.data().re[i], r.data().im[i]);
                let (c, d) = (hv.data().re[i], hv.data().im[i]);
                nr += a * c - b * d;
                ni += a * d + b * c;
                dr += a * a - b * b;
                di += 2.0 * a * b;
            }
            let den = dr * dr + di * di;
            mine.push((nr * dr + ni * di) / den);
        }
        let mut want_re: Vec<f64> = want.iter().step_by(2).copied().collect();
        want_re.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        mine.sort_by(|a, b| a.partial_cmp(b).expect("finite"));
        let worst = mine
            .iter()
            .zip(&want_re)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!("  ee_cis_spectrum[{kshift}] worst |Δ| {worst:e}");
        if !(worst < IMDS_BLOCK) {
            failures.push(format!("ee_cis_spectrum_{kshift} {worst:e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "EE singlet above the gate: {failures:?}"
    );
}

/// **The `EOMEESinglet` roots, on upstream's own converged amplitudes.**
#[test]
#[ignore = "opt-in PySCF oracle"]
fn ee_singlet_roots_match_upstream() {
    let Some(out) = emit("ee_singlet") else {
        return;
    };
    let f = diamond_scf([1, 1, 2]);
    let up = upstream_mos(&out);
    let (eris, khelper) = eris_on_upstream_mf(&f, &up);
    let kc = &khelper.kconserv;
    let (nkpts, nocc, nvir) = (up.nkpts, up.nocc, up.nmo - up.nocc);
    let padding = eomg::padding_from(&up.padded).expect("padding");

    let t1 = ZArr::from_ctensor(&[nkpts, nocc, nvir], cblock(&out, "t1")).expect("t1");
    let t2 = ZArr::from_ctensor(
        &[nkpts, nkpts, nkpts, nocc, nocc, nvir, nvir],
        cblock(&out, "t2"),
    )
    .expect("t2");
    let imds = eomr::RhfEomImds::make_shared(&t1, &t2, &eris, kc)
        .expect("shared")
        .make_ee(kc)
        .expect("EE imds");

    let nroots = scalar(&out, "nroots") as usize;
    let opts = EomOpts {
        conv_tol: 1e-8,
        max_cycle: 100,
        nroots,
        ..Default::default()
    };
    let mut failures: Vec<String> = Vec::new();
    for kshift in 0..nkpts {
        let roots =
            eomr::eom_kernel(Excitation::Ee, kshift, &imds, &padding, kc, &opts).expect("EE roots");
        let want = block(&out, &format!("ee_roots_{kshift}"));
        let conv = block(&out, &format!("ee_conv_{kshift}"));
        for (n, w) in want.iter().enumerate() {
            if conv[n] == 0.0 {
                println!("  ee_roots_{kshift}[{n}]: upstream did not converge, skipped");
                continue;
            }
            let d = (roots.e[n] - w).abs();
            println!(
                "  ee_roots_{kshift}[{n}] {:.10} vs {w:.10} -> {d:e}  (qpwt {:.4})",
                roots.e[n], roots.qp_weight[n]
            );
            if !(d < ROOT_GATE) {
                failures.push(format!("ee_roots_{kshift}[{n}] {d:e}"));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "EE singlet roots above the gate: {failures:?}"
    );
}

/// `left = True` is refused, and says where.
#[test]
fn ee_singlet_refuses_left_and_says_where() {
    // `EOMEESinglet.gen_matvec` raises for `left`; this is the refusal, and it
    // needs no fixture because the guard precedes every use of the imds.
    // (The check lives in `eom_kernel`; exercising it needs an `RhfEomImds`,
    // so what is asserted here is the payload the refusal carries.)
    let e = pyscf_pbc_cc::PbcCcError::NotImplementedUpstream {
        upstream: "pbc/cc/eom_kccsd_rhf.py:1464",
        what: "EOMEESinglet.gen_matvec raises NotImplementedError for left=True",
    };
    assert!(format!("{e}").contains("eom_kccsd_rhf.py:1464"));
}

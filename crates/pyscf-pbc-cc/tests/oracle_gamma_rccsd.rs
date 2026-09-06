//! Opt-in oracle checks for the single-k-point `pbc/cc/ccsd.py` shim and the
//! molecular complex-capable `cc/rccsd.py` it stands on, against PySCF 2.12.1.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-cc \
//!   --test oracle_gamma_rccsd -- --ignored --nocapture
//! ```
//!
//! # Two fixtures, and the second is the point
//!
//! At Γ with a real cell the MO coefficients are REAL, and a real
//! `ccsd.CCSD` would do — which is why the emitter also runs a SHIFTED
//! k-point, where `mo_coeff.imag` is nonzero and only a complex-capable RCCSD
//! works. `16-CONTEXT §1.2` records that complex capability as the thing this
//! port lacked; the `k_*` fixture is what measures that it no longer does.
//!
//! # The eris are built from UPSTREAM's own MO coefficients
//!
//! Diamond at Γ has degenerate levels, so the SCF eigenvectors inside a
//! degenerate subspace are arbitrary and any quantity that is not invariant —
//! `update_amps` on FIXED synthetic amplitudes, for one — moves run to run.
//! Measured: `energy_synth` shifts in the fifth digit between two identical
//! upstream runs. So every equation-level gate here is driven from upstream's
//! emitted `mo_coeff` / `fock` / `mo_energy`, and only the CONVERGED `e_corr`
//! — which is invariant — is compared across two mean fields.

mod common;

use common::{block, cblock, emit, maxdiff, scalar};

use pyscf_pbc_cc::ZArr;
use pyscf_pbc_cc::rccsd::{self, ChemistsErisZ, RccsdOpts};
use pyscf_pbc_df::{Fftdf, MoCoeff, PeriodicDf};

/// The MO-block gate. These are single-k-point FFT transforms at the pinned
/// `[15,15,15]` mesh, the same integral floor `oracle_phase16.rs` measured.
const ERI_BLOCK: f64 = 1e-6;

/// `measurements/README.md §1` G1 — `e_corr` vs upstream, FFTDF.
const E_CORR: f64 = 1e-7;

/// The same SplitMix64 stream `oracle_phase16.py`'s `synthetic_amps` uses.
fn synthetic_mol(nocc: usize, nvir: usize) -> (ZArr, ZArr) {
    let mut r = common::SplitMix64(20260906);
    let mut t1 = ZArr::zeros(&[nocc, nvir]);
    for i in 0..nocc * nvir {
        t1.data_mut().re[i] = 0.05 * r.unit();
        t1.data_mut().im[i] = 0.05 * r.unit();
    }
    let mut t2 = ZArr::zeros(&[nocc, nocc, nvir, nvir]);
    for i in 0..nocc * nocc * nvir * nvir {
        t2.data_mut().re[i] = 0.05 * r.unit();
        t2.data_mut().im[i] = 0.05 * r.unit();
    }
    (t1, t2)
}

/// Build this port's `_ChemistsERIs` on upstream's own single-k-point mean
/// field.
fn eris_from(out: &str, tag: &str) -> (ChemistsErisZ, usize, usize) {
    let cell = common::diamond([15, 15, 15]);
    let k = block(out, &format!("{tag}_kpt"));
    let kpts = vec![[k[0], k[1], k[2]]];
    let df = Fftdf::new(cell, &kpts).expect("fftdf");

    let nocc = scalar(out, &format!("{tag}_nocc")) as usize;
    let nmo = scalar(out, &format!("{tag}_nmo")) as usize;
    let nao = scalar(out, &format!("{tag}_nao")) as usize;
    let nvir = nmo - nocc;

    let c = cblock(out, &format!("{tag}_mo_coeff"));
    assert_eq!(c.re.len(), nao * nmo, "{tag}_mo_coeff shape");
    let mo = MoCoeff::new(nao, nmo, c);
    let mos = vec![mo];
    let e7 = df
        .ao2mo_7d([&mos, &mos, &mos, &mos], 1.0)
        .expect("ao2mo_7d at one k-point");
    let eri = ZArr::from_ctensor(&[nmo, nmo, nmo, nmo], e7.data).expect("eri shape");

    let fock = ZArr::from_ctensor(&[nmo, nmo], cblock(out, &format!("{tag}_fock"))).expect("fock");
    let mo_energy = block(out, &format!("{tag}_mo_energy"));
    assert_eq!(mo_energy.len(), nmo);
    let eris = ChemistsErisZ::from_full(&eri, fock, mo_energy, nocc).expect("_make_eris_incore");
    (eris, nocc, nvir)
}

fn check(out: &str, tag: &str, failures: &mut Vec<String>, got: &ZArr, name: &str, gate: f64) {
    let full = format!("{tag}_{name}");
    let d = maxdiff(got, &cblock(out, &full), &full);
    println!("  {full:16} max|Δ| {d:e}");
    if !(d < gate) {
        failures.push(format!("{full} {d:e}"));
    }
}

/// **The seven chemists' blocks, the nine intermediates, `update_amps`,
/// `energy` and `init_amps`, at Γ and at a SHIFTED k-point.**
#[test]
#[ignore = "opt-in PySCF oracle"]
fn rccsd_equations_match_upstream() {
    let Some(out) = emit("gamma_rccsd") else {
        return;
    };

    let mut failures: Vec<String> = Vec::new();
    for tag in ["g", "k"] {
        let complex = scalar(&out, &format!("{tag}_mo_is_complex"));
        println!(
            "\n=== fixture '{tag}': kpt {:?}, mo_coeff complex = {}",
            block(&out, &format!("{tag}_kpt")),
            complex != 0.0
        );
        let (eris, nocc, nvir) = eris_from(&out, tag);

        for name in ["oooo", "ovoo", "ovov", "oovv", "ovvo", "ovvv", "vvvv"] {
            let got = match name {
                "oooo" => &eris.oooo,
                "ovoo" => &eris.ovoo,
                "ovov" => &eris.ovov,
                "oovv" => &eris.oovv,
                "ovvo" => &eris.ovvo,
                "ovvv" => &eris.ovvv,
                _ => &eris.vvvv,
            };
            check(&out, tag, &mut failures, got, name, ERI_BLOCK);
        }

        let (st1, st2) = synthetic_mol(nocc, nvir);
        // The synthetic amplitudes must be BIT-identical on both sides, or
        // every gate below measures two different inputs.
        for (got, name) in [(&st1, "st1"), (&st2, "st2")] {
            let full = format!("{tag}_{name}");
            let d = maxdiff(got, &cblock(&out, &full), &full);
            println!("  {full:16} max|Δ| {d:e}");
            assert!(d == 0.0, "{full} is not bit-identical: {d:e}");
        }

        for (got, name) in [
            (rccsd::cc_foo(&st1, &st2, &eris).expect("cc_Foo"), "cc_Foo"),
            (rccsd::cc_fvv(&st1, &st2, &eris).expect("cc_Fvv"), "cc_Fvv"),
            (rccsd::cc_fov(&st1, &eris).expect("cc_Fov"), "cc_Fov"),
            (rccsd::loo(&st1, &st2, &eris).expect("Loo"), "Loo"),
            (rccsd::lvv(&st1, &st2, &eris).expect("Lvv"), "Lvv"),
            (
                rccsd::cc_woooo(&st1, &st2, &eris).expect("cc_Woooo"),
                "cc_Woooo",
            ),
            (rccsd::cc_wvvvv(&st1, &eris).expect("cc_Wvvvv"), "cc_Wvvvv"),
            (
                rccsd::cc_wvoov(&st1, &st2, &eris).expect("cc_Wvoov"),
                "cc_Wvoov",
            ),
            (
                rccsd::cc_wvovo(&st1, &st2, &eris).expect("cc_Wvovo"),
                "cc_Wvovo",
            ),
        ] {
            check(&out, tag, &mut failures, &got, name, ERI_BLOCK);
        }

        let (t1new, t2new) = rccsd::update_amps(&st1, &st2, &eris, 0.0).expect("update_amps");
        check(&out, tag, &mut failures, &t1new, "t1new", ERI_BLOCK);
        check(&out, tag, &mut failures, &t2new, "t2new", ERI_BLOCK);

        let (e, im) = rccsd::energy(&st1, &st2, &eris).expect("energy");
        let want = scalar(&out, &format!("{tag}_energy_synth"));
        println!("  {tag}_energy_synth {e:.15e} vs {want:.15e} (Im {im:.3e})");
        if !((e - want).abs() < ERI_BLOCK) {
            failures.push(format!("{tag}_energy_synth {:e}", (e - want).abs()));
        }

        let (emp2, it1, it2) = rccsd::init_amps(&eris).expect("init_amps");
        let want = scalar(&out, &format!("{tag}_emp2"));
        println!("  {tag}_emp2 {emp2:.15e} vs {want:.15e}");
        if !((emp2 - want).abs() < E_CORR) {
            failures.push(format!("{tag}_emp2 {:e}", (emp2 - want).abs()));
        }
        check(&out, tag, &mut failures, &it1, "init_t1", ERI_BLOCK);
        check(&out, tag, &mut failures, &it2, "init_t2", ERI_BLOCK);
    }
    assert!(
        failures.is_empty(),
        "RCCSD equations above the gate: {failures:?}"
    );
}

/// **The converged `e_corr`, and the `mbpt2` short-circuit.**
#[test]
#[ignore = "opt-in PySCF oracle"]
fn rccsd_e_corr_matches_upstream() {
    let Some(out) = emit("gamma_rccsd") else {
        return;
    };
    let mut failures: Vec<String> = Vec::new();
    for tag in ["g", "k"] {
        let (eris, _nocc, _nvir) = eris_from(&out, tag);
        let opts = RccsdOpts {
            conv_tol: 1e-9,
            conv_tol_normt: 1e-7,
            ..Default::default()
        };
        let res = rccsd::kernel(&eris, &opts).expect("RCCSD");
        let want = scalar(&out, &format!("{tag}_e_corr"));
        assert!(
            scalar(&out, &format!("{tag}_converged")) != 0.0,
            "upstream did not converge for fixture {tag}"
        );
        assert!(
            res.converged,
            "this port did not converge for fixture {tag}"
        );
        let d = (res.e_corr - want).abs();
        println!(
            "  {tag}_e_corr {:.15e} vs {want:.15e} -> {d:e}  ({} cycles, max|Im| {:.2e})",
            res.e_corr, res.niter, res.max_imag
        );
        if !(d < E_CORR) {
            failures.push(format!("{tag}_e_corr {d:e}"));
        }

        // `mbpt2=True`. It goes through `pbc.mp.RMP2`, which REFUSES a
        // non-Gamma k-point at `pbc/mp/mp2.py:22-23` — measured, and this port
        // reproduces the refusal.
        let refused = scalar(&out, &format!("{tag}_mbpt2_refused")) != 0.0;
        let k = block(&out, &format!("{tag}_kpt"));
        let is_gamma = k.iter().all(|v| v.abs() < 1e-9);
        assert_eq!(
            refused, !is_gamma,
            "the mbpt2 refusal must track the k-point, not the fixture name"
        );
        // This port refuses at exactly the same k-points, through the same
        // upstream line.
        let cell = common::diamond([15, 15, 15]);
        let df = Fftdf::new(cell, &[[k[0], k[1], k[2]]]).expect("fftdf");
        let gamma_here = df.kpts()[0].iter().all(|v| v.abs() <= 1e-9);
        assert_eq!(gamma_here, is_gamma);
        if refused {
            println!("  {tag}: upstream refuses mbpt2 away from Gamma, as this port does");
            continue;
        }
        let (emp2, t1, t2) = {
            let (e, t1, t2) = rccsd::init_amps(&eris).expect("init_amps");
            (e, ZArr::zeros(t1.shape()), t2)
        };
        let want = scalar(&out, &format!("{tag}_e_mbpt2"));
        let d = (emp2 - want).abs();
        println!("  {tag}_e_mbpt2 {emp2:.15e} vs {want:.15e} -> {d:e}");
        if !(d < E_CORR) {
            failures.push(format!("{tag}_e_mbpt2 {d:e}"));
        }
        // `:32` — `t1` is identically zero.
        assert_eq!(scalar(&out, &format!("{tag}_mbpt2_t1_max")), 0.0);
        assert!(t1.data().re.iter().all(|v| *v == 0.0) && t1.data().im.iter().all(|v| *v == 0.0));
        assert!(t2.len() > 0);
    }
    assert!(
        failures.is_empty(),
        "RCCSD e_corr above the gate: {failures:?}"
    );
}

/// **The Γ shim and KRCCSD at `[1,1,1]` are the same calculation** —
/// `16-VERIFICATION` G9's cross-check, from the other direction.
///
/// This is REPORTED, not gated at the CC level: the two run different
/// expressions (seven chemists' blocks against seven k-point ones) on two
/// separate SCFs, and what the number measures is that they agree at all.
#[test]
#[ignore = "opt-in PySCF oracle"]
fn gamma_shim_and_krccsd_at_one_kpoint_agree() {
    let Some(out) = emit("gamma_rccsd") else {
        return;
    };
    let shim = scalar(&out, "g_e_corr");
    let kcc = scalar(&out, "krccsd111_e_corr");
    println!(
        "UPSTREAM: Gamma shim {shim:.15e} vs KRCCSD[1,1,1] {kcc:.15e} -> {:e}",
        (shim - kcc).abs()
    );

    let (eris, _, _) = eris_from(&out, "g");
    let res = rccsd::kernel(
        &eris,
        &RccsdOpts {
            conv_tol: 1e-9,
            conv_tol_normt: 1e-7,
            ..Default::default()
        },
    )
    .expect("RCCSD");
    println!(
        "THIS PORT: Gamma shim {:.15e}; vs upstream KRCCSD[1,1,1] -> {:e}",
        res.e_corr,
        (res.e_corr - kcc).abs()
    );
    // The two upstream routes agree to their own CCSD convergence, and this
    // port's shim sits with them. Gated only at that level.
    assert!(
        (res.e_corr - kcc).abs() < 1e-6,
        "the Gamma shim and KRCCSD[1,1,1] disagree by {:e}",
        (res.e_corr - kcc).abs()
    );
}

/// **The whole shim end to end, on THIS port's own Γ mean field.**
///
/// The other tests drive `_ChemistsERIs` from upstream's emitted `fock` and
/// `mo_energy`, so they never exercise [`GammaRccsd::ao2mo`] — which is where
/// the two things `pbc/cc/ccsd.py` actually adds live: the Fock built under
/// `exxdiv = None` and the Madelung re-add on the occupied orbital energies.
/// This runs that path.
///
/// The gate is the MEAN-FIELD one, not `E_CORR`: `16-VERIFICATION` measured
/// this port's `KRHF` and upstream's `1.35e-5 Ha` apart at the pinned coarse
/// mesh, and a correlation energy compared across two mean fields inherits
/// that.
#[test]
#[ignore = "opt-in PySCF oracle"]
fn gamma_shim_runs_on_this_ports_own_mean_field() {
    let Some(out) = emit("gamma_rccsd") else {
        return;
    };
    let f = common::diamond_scf([1, 1, 1]);
    println!(
        "MEAN-FIELD RESIDUAL (reported, not gated): this port {} vs upstream {} -> {:e}",
        f.scf.e_tot,
        scalar(&out, "g_e_hf"),
        (f.scf.e_tot - scalar(&out, "g_e_hf")).abs()
    );

    let cc = pyscf_pbc_cc::ccsd::GammaRccsd::new(&f.scf, &f.df).expect("Gamma RCCSD");
    let eris = cc.ao2mo().expect("ao2mo");
    // The Madelung re-add is the whole point of this `ao2mo`: at `exxdiv =
    // None` the occupied orbital energies would otherwise be the bare ones.
    println!("  mo_energy from ao2mo: {:?}", eris.mo_energy);
    let res = cc.kernel_with(&eris).expect("RCCSD");
    let want = scalar(&out, "g_e_corr");
    let d = (res.e_corr - want).abs();
    println!(
        "  e_corr {:.15e} vs upstream {want:.15e} -> {d:e}  ({} cycles, converged {})",
        res.e_corr, res.niter, res.converged
    );
    assert!(res.converged, "the Gamma RCCSD did not converge");
    // 1e-5: the coarse-mesh mean-field spread `16-VERIFICATION` measured, which
    // this number carries and cannot be tighter than.
    assert!(d < 1e-5, "the Gamma shim e_corr is {d:e} from upstream");

    let (emp2, t1, _t2) = cc.mbpt2_with(&eris).expect("mbpt2");
    println!(
        "  emp2 {emp2:.15e} vs upstream {:.15e}",
        scalar(&out, "g_e_mbpt2")
    );
    assert!((emp2 - scalar(&out, "g_e_mbpt2")).abs() < 1e-5);
    assert!(t1.data().re.iter().all(|v| *v == 0.0));
}

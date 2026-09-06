//! Opt-in oracle checks for the single-k-point `pbc/cc/ccsd.py` UCCSD and
//! GCCSD shims, and the molecular complex-capable `cc/uccsd.py` /
//! `cc/gccsd.py` under them, against PySCF 2.12.1.
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test --release -p pyscf-pbc-cc \
//!   --test oracle_gamma_ug -- --ignored --nocapture
//! ```
//!
//! # Four fixtures, and half of them are the point
//!
//! Each shim is run at Γ, where the MO coefficients are REAL, and at a
//! SHIFTED k-point, where they are not. Only the second measures
//! complex-capability, which is what `16-CONTEXT §1.2` records these two
//! molecular modules as lacking.
//!
//! # The `mbpt2` asymmetry is real and is checked
//!
//! `RMP2.__init__` (`pbc/mp/mp2.py:21-23`) and `UMP2.__init__` (`:35-37`)
//! both refuse a non-Γ k-point; `GMP2.__init__` (`:47-51`) does not. So of the
//! three shims only GCCSD's `mbpt2=True` runs at a shifted k. Measured on both
//! sides.
//!
//! # The eris come from UPSTREAM's own MO coefficients
//!
//! Same reason as `oracle_gamma_rccsd.rs`: diamond has degenerate levels, so
//! any quantity that is not invariant moves run to run. Only the converged
//! `e_corr` is compared across two mean fields.

mod common;

use common::{block, cblock, emit, maxdiff, scalar};

use pyscf_pbc_cc::ZArr;
use pyscf_pbc_cc::gccsd::{self, PhysicistsErisZ};
use pyscf_pbc_cc::rccsd::RccsdOpts;
use pyscf_pbc_cc::uccsd::{self, ChemistsErisU};
use pyscf_pbc_df::{Fftdf, MoCoeff, PeriodicDf};

/// The MO-block gate — the single-k-point FFT transform floor at the pinned
/// `[15,15,15]` mesh, as `oracle_gamma_rccsd.rs` measures it.
const ERI_BLOCK: f64 = 1e-6;

/// `measurements/README.md §1` G1 — `e_corr` vs upstream, FFTDF.
const E_CORR: f64 = 1e-7;

fn kpts_of(out: &str, tag: &str) -> Vec<[f64; 3]> {
    let k = block(out, &format!("{tag}_kpt"));
    vec![[k[0], k[1], k[2]]]
}

fn is_gamma(out: &str, tag: &str) -> bool {
    block(out, &format!("{tag}_kpt"))
        .iter()
        .all(|v| v.abs() < 1e-9)
}

/// One `(pq|rs)` chemists' tensor from two MO blocks, at the single k-point.
fn chem(df: &Fftdf, x: &MoCoeff, y: &MoCoeff) -> ZArr {
    let e = df
        .ao2mo([x, x, y, y], [0, 0, 0, 0], false)
        .expect("ao2mo")
        .restore_s1();
    ZArr::from_ctensor(&[x.nmo, x.nmo, y.nmo, y.nmo], e.data).expect("chem eri shape")
}

fn mo_from(out: &str, name: &str, nao: usize, nmo: usize) -> MoCoeff {
    let c = cblock(out, name);
    assert_eq!(c.re.len(), nao * nmo, "{name} shape");
    MoCoeff::new(nao, nmo, c)
}

fn check(out: &str, failures: &mut Vec<String>, got: &ZArr, full: &str, gate: f64) {
    let d = maxdiff(got, &cblock(out, full), full);
    println!("  {full:20} max|Δ| {d:e}");
    if !(d < gate) {
        failures.push(format!("{full} {d:e}"));
    }
}

// ---------------------------------------------------------------------------
// GCCSD
// ---------------------------------------------------------------------------

fn gccsd_eris(out: &str, tag: &str) -> (PhysicistsErisZ, usize, usize) {
    let cell = common::diamond([15, 15, 15]);
    let df = Fftdf::new(cell, &kpts_of(out, tag)).expect("fftdf");
    let nocc = scalar(out, &format!("{tag}_nocc")) as usize;
    let nmo = scalar(out, &format!("{tag}_nmo")) as usize;
    let nao_so = scalar(out, &format!("{tag}_nao")) as usize;
    assert_eq!(
        nao_so % 2,
        0,
        "the GHF MO block must have an even row count"
    );
    let nao = nao_so / 2;
    let mo = mo_from(out, &format!("{tag}_mo_coeff"), nao_so, nmo);

    // `pbc/cc/ccsd.py:110-119` — the four-spin-block sum, `orbspin` absent.
    // The emitter records `{tag}_has_orbspin`, so the branch is not assumed.
    assert_eq!(
        scalar(out, &format!("{tag}_has_orbspin")),
        0.0,
        "upstream took the orbspin branch; this port reproduces the other one"
    );
    let split = |top: bool| -> MoCoeff {
        let off = usize::from(top) * nao;
        let mut c = pyscf_algebra::CTensor::zeros(nao * nmo);
        for a in 0..nao {
            for p in 0..nmo {
                c.re[a * nmo + p] = mo.c.re[(a + off) * nmo + p];
                c.im[a * nmo + p] = mo.c.im[(a + off) * nmo + p];
            }
        }
        MoCoeff::new(nao, nmo, c)
    };
    let (a, b) = (split(false), split(true));
    let mut acc = ZArr::zeros(&[nmo, nmo, nmo, nmo]);
    for (x, y) in [(&a, &a), (&b, &b), (&a, &b), (&b, &a)] {
        acc.add_assign(&chem(&df, x, y)).expect("spin-block sum");
    }

    let fock = ZArr::from_ctensor(&[nmo, nmo], cblock(out, &format!("{tag}_fock"))).expect("fock");
    let mo_energy = block(out, &format!("{tag}_mo_energy"));
    let eris = PhysicistsErisZ::from_full_chemists(&acc, fock, mo_energy, nocc)
        .expect("_make_eris_incore");
    (eris, nocc, nmo - nocc)
}

/// **The spin-orbital blocks, the six `cc_*`, `update_amps`, `energy` and
/// `init_amps`, at Γ and at a shifted k-point.**
#[test]
#[ignore = "opt-in PySCF oracle"]
fn gccsd_equations_match_upstream() {
    let Some(out) = emit("gamma_ug") else { return };
    let mut failures: Vec<String> = Vec::new();
    for tag in ["gg", "gk"] {
        println!(
            "\n=== GCCSD '{tag}': kpt {:?}, complex = {}",
            block(&out, &format!("{tag}_kpt")),
            scalar(&out, &format!("{tag}_mo_is_complex")) != 0.0
        );
        let (eris, nocc, nvir) = gccsd_eris(&out, tag);
        for (got, name) in [
            (&eris.oooo, "oooo"),
            (&eris.ooov, "ooov"),
            (&eris.oovv, "oovv"),
            (&eris.ovov, "ovov"),
            (&eris.ovvv, "ovvv"),
            (&eris.vvvv, "vvvv"),
        ] {
            check(
                &out,
                &mut failures,
                got,
                &format!("{tag}_{name}"),
                ERI_BLOCK,
            );
        }

        let (st1, st2) = common::synthetic(&[nocc, nvir], &[nocc, nocc, nvir, nvir]);
        for (got, name) in [(&st1, "st1"), (&st2, "st2")] {
            let full = format!("{tag}_{name}");
            let d = maxdiff(got, &cblock(&out, &full), &full);
            assert!(d == 0.0, "{full} is not bit-identical: {d:e}");
        }

        for (got, name) in [
            (
                gccsd::make_tau(&st2, &st1, &st1, 1.0).expect("tau"),
                "make_tau",
            ),
            (gccsd::cc_fvv(&st1, &st2, &eris).expect("cc_Fvv"), "cc_Fvv"),
            (gccsd::cc_foo(&st1, &st2, &eris).expect("cc_Foo"), "cc_Foo"),
            (gccsd::cc_fov(&st1, &eris).expect("cc_Fov"), "cc_Fov"),
            (
                gccsd::cc_woooo(&st1, &st2, &eris).expect("cc_Woooo"),
                "cc_Woooo",
            ),
            (
                gccsd::cc_wvvvv(&st1, &st2, &eris).expect("cc_Wvvvv"),
                "cc_Wvvvv",
            ),
            (
                gccsd::cc_wovvo(&st1, &st2, &eris).expect("cc_Wovvo"),
                "cc_Wovvo",
            ),
        ] {
            check(
                &out,
                &mut failures,
                &got,
                &format!("{tag}_{name}"),
                ERI_BLOCK,
            );
        }

        let (t1new, t2new) = gccsd::update_amps(&st1, &st2, &eris, 0.0).expect("update_amps");
        check(
            &out,
            &mut failures,
            &t1new,
            &format!("{tag}_t1new"),
            ERI_BLOCK,
        );
        check(
            &out,
            &mut failures,
            &t2new,
            &format!("{tag}_t2new"),
            ERI_BLOCK,
        );

        let (e, im) = gccsd::energy(&st1, &st2, &eris).expect("energy");
        let want = scalar(&out, &format!("{tag}_energy_synth"));
        println!("  {tag}_energy_synth {e:.15e} vs {want:.15e} (Im {im:.3e})");
        if !((e - want).abs() < ERI_BLOCK) {
            failures.push(format!("{tag}_energy_synth {:e}", (e - want).abs()));
        }

        let (emp2, it1, it2) = gccsd::init_amps(&eris).expect("init_amps");
        let want = scalar(&out, &format!("{tag}_emp2"));
        println!("  {tag}_emp2 {emp2:.15e} vs {want:.15e}");
        if !((emp2 - want).abs() < E_CORR) {
            failures.push(format!("{tag}_emp2 {:e}", (emp2 - want).abs()));
        }
        check(
            &out,
            &mut failures,
            &it1,
            &format!("{tag}_init_t1"),
            ERI_BLOCK,
        );
        check(
            &out,
            &mut failures,
            &it2,
            &format!("{tag}_init_t2"),
            ERI_BLOCK,
        );
    }
    assert!(
        failures.is_empty(),
        "GCCSD equations above the gate: {failures:?}"
    );
}

/// **The GCCSD `e_corr`, and its `mbpt2` — the one that is NOT Γ-only.**
#[test]
#[ignore = "opt-in PySCF oracle"]
fn gccsd_e_corr_matches_upstream() {
    let Some(out) = emit("gamma_ug") else { return };
    let mut failures: Vec<String> = Vec::new();
    for tag in ["gg", "gk"] {
        let (eris, _, _) = gccsd_eris(&out, tag);
        let opts = RccsdOpts {
            conv_tol: 1e-9,
            conv_tol_normt: 1e-7,
            ..RccsdOpts::default()
        };
        let res = gccsd::kernel(&eris, &opts).expect("GCCSD");
        let want = scalar(&out, &format!("{tag}_e_corr"));
        assert!(scalar(&out, &format!("{tag}_converged")) != 0.0);
        assert!(res.converged, "this port did not converge for {tag}");
        let d = (res.e_corr - want).abs();
        println!(
            "  {tag}_e_corr {:.15e} vs {want:.15e} -> {d:e}  ({} cycles, max|Im| {:.2e})",
            res.e_corr, res.niter, res.max_imag
        );
        if !(d < E_CORR) {
            failures.push(format!("{tag}_e_corr {d:e}"));
        }

        // `pbc.mp.GMP2` has NO k-point guard, so this branch runs at BOTH
        // fixtures — unlike RCCSD's and UCCSD's.
        assert_eq!(
            scalar(&out, &format!("{tag}_mbpt2_refused")),
            0.0,
            "GMP2 has no Gamma guard; upstream should not have refused"
        );
        let (emp2, ..) = gccsd::init_amps(&eris).expect("init_amps");
        let want = scalar(&out, &format!("{tag}_e_mbpt2"));
        let d = (emp2 - want).abs();
        println!("  {tag}_e_mbpt2 {emp2:.15e} vs {want:.15e} -> {d:e}");
        if !(d < E_CORR) {
            failures.push(format!("{tag}_e_mbpt2 {d:e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "GCCSD e_corr above the gate: {failures:?}"
    );
}

// ---------------------------------------------------------------------------
// UCCSD
// ---------------------------------------------------------------------------

fn uccsd_eris(out: &str, tag: &str) -> ChemistsErisU {
    let cell = common::diamond([15, 15, 15]);
    let df = Fftdf::new(cell, &kpts_of(out, tag)).expect("fftdf");
    let nocca = scalar(out, &format!("{tag}_nocca")) as usize;
    let noccb = scalar(out, &format!("{tag}_noccb")) as usize;
    let nmoa = scalar(out, &format!("{tag}_nmoa")) as usize;
    let nmob = scalar(out, &format!("{tag}_nmob")) as usize;
    let nao = scalar(out, &format!("{tag}_nao")) as usize;

    let a = mo_from(out, &format!("{tag}_mo_coeff_a"), nao, nmoa);
    let b = mo_from(out, &format!("{tag}_mo_coeff_b"), nao, nmob);
    // `uccsd.py:886-888` — three transforms, and `eri_ba` is `eri_ab`
    // transposed rather than a fourth (`:895`).
    let eri_aa = chem(&df, &a, &a);
    let eri_bb = chem(&df, &b, &b);
    let eri_ab = chem(&df, &a, &b);

    let focka =
        ZArr::from_ctensor(&[nmoa, nmoa], cblock(out, &format!("{tag}_focka"))).expect("focka");
    let fockb =
        ZArr::from_ctensor(&[nmob, nmob], cblock(out, &format!("{tag}_fockb"))).expect("fockb");
    let ea = block(out, &format!("{tag}_mo_energy_a"));
    let eb = block(out, &format!("{tag}_mo_energy_b"));
    ChemistsErisU::from_full_chemists(
        &eri_aa,
        &eri_bb,
        &eri_ab,
        focka,
        fockb,
        (ea, eb),
        (nocca, noccb),
    )
    .expect("_make_eris_incore")
}

/// The 25 blocks, in `_ChemistsERIs.__init__`'s own order and naming.
fn uccsd_blocks(e: &ChemistsErisU) -> Vec<(&ZArr, &'static str)> {
    vec![
        (&e.oooo, "oooo"),
        (&e.ovoo, "ovoo"),
        (&e.ovov, "ovov"),
        (&e.oovv, "oovv"),
        (&e.ovvo, "ovvo"),
        (&e.ovvv, "ovvv"),
        (&e.vvvv, "vvvv"),
        (&e.oooo_bb, "OOOO"),
        (&e.ovoo_bb, "OVOO"),
        (&e.ovov_bb, "OVOV"),
        (&e.oovv_bb, "OOVV"),
        (&e.ovvo_bb, "OVVO"),
        (&e.ovvv_bb, "OVVV"),
        (&e.vvvv_bb, "VVVV"),
        (&e.oo_oo, "ooOO"),
        (&e.ov_oo, "ovOO"),
        (&e.ov_ov, "ovOV"),
        (&e.oo_vv, "ooVV"),
        (&e.ov_vo, "ovVO"),
        (&e.ov_vv, "ovVV"),
        (&e.vv_vv, "vvVV"),
        (&e.ovoo_ba, "OVoo"),
        (&e.oovv_ba, "OOvv"),
        (&e.ovvo_ba, "OVvo"),
        (&e.ovvv_ba, "OVvv"),
    ]
}

/// The emitter's own SplitMix64 stream for the unrestricted amplitudes —
/// ONE stream across all five arrays, in `t1a, t1b, t2aa, t2ab, t2bb` order.
fn synthetic_u(
    nocca: usize,
    noccb: usize,
    nvira: usize,
    nvirb: usize,
) -> ((ZArr, ZArr), (ZArr, ZArr, ZArr)) {
    let mut r = common::SplitMix64(20260915);
    let mut draw = |shape: &[usize]| -> ZArr {
        let n: usize = shape.iter().product();
        let mut z = ZArr::zeros(shape);
        for i in 0..n {
            z.data_mut().re[i] = 0.05 * r.unit();
            z.data_mut().im[i] = 0.05 * r.unit();
        }
        z
    };
    let t1a = draw(&[nocca, nvira]);
    let t1b = draw(&[noccb, nvirb]);
    let t2aa = draw(&[nocca, nocca, nvira, nvira]);
    let t2ab = draw(&[nocca, noccb, nvira, nvirb]);
    let t2bb = draw(&[noccb, noccb, nvirb, nvirb]);
    ((t1a, t1b), (t2aa, t2ab, t2bb))
}

/// **The 25 blocks, `update_amps`, `energy` and `init_amps`, at Γ and at a
/// shifted k-point.**
#[test]
#[ignore = "opt-in PySCF oracle"]
fn uccsd_equations_match_upstream() {
    let Some(out) = emit("gamma_ug") else { return };
    let mut failures: Vec<String> = Vec::new();
    for tag in ["ug", "uk"] {
        println!(
            "\n=== UCCSD '{tag}': kpt {:?}, complex = {}",
            block(&out, &format!("{tag}_kpt")),
            scalar(&out, &format!("{tag}_mo_is_complex")) != 0.0
        );
        let eris = uccsd_eris(&out, tag);
        for (got, name) in uccsd_blocks(&eris) {
            check(
                &out,
                &mut failures,
                got,
                &format!("{tag}_{name}"),
                ERI_BLOCK,
            );
        }

        let (st1, st2) = synthetic_u(eris.nocca, eris.noccb, eris.nvira, eris.nvirb);
        for (got, name) in [
            (&st1.0, "st1_0"),
            (&st1.1, "st1_1"),
            (&st2.0, "st2_0"),
            (&st2.1, "st2_1"),
            (&st2.2, "st2_2"),
        ] {
            let full = format!("{tag}_{name}");
            let d = maxdiff(got, &cblock(&out, &full), &full);
            assert!(d == 0.0, "{full} is not bit-identical: {d:e}");
        }

        let (t1new, t2new) = uccsd::update_amps(&st1, &st2, &eris, 0.0).expect("update_amps");
        for (got, name) in [
            (&t1new.0, "t1new_0"),
            (&t1new.1, "t1new_1"),
            (&t2new.0, "t2new_0"),
            (&t2new.1, "t2new_1"),
            (&t2new.2, "t2new_2"),
        ] {
            check(
                &out,
                &mut failures,
                got,
                &format!("{tag}_{name}"),
                ERI_BLOCK,
            );
        }

        let (e, im) = uccsd::energy(&st1, &st2, &eris).expect("energy");
        let want = scalar(&out, &format!("{tag}_energy_synth"));
        println!("  {tag}_energy_synth {e:.15e} vs {want:.15e} (Im {im:.3e})");
        if !((e - want).abs() < ERI_BLOCK) {
            failures.push(format!("{tag}_energy_synth {:e}", (e - want).abs()));
        }

        let (emp2, it1, it2) = uccsd::init_amps(&eris).expect("init_amps");
        let want = scalar(&out, &format!("{tag}_emp2"));
        println!("  {tag}_emp2 {emp2:.15e} vs {want:.15e}");
        if !((emp2 - want).abs() < E_CORR) {
            failures.push(format!("{tag}_emp2 {:e}", (emp2 - want).abs()));
        }
        for (got, name) in [
            (&it1.0, "init_t1_0"),
            (&it1.1, "init_t1_1"),
            (&it2.0, "init_t2_0"),
            (&it2.1, "init_t2_1"),
            (&it2.2, "init_t2_2"),
        ] {
            check(
                &out,
                &mut failures,
                got,
                &format!("{tag}_{name}"),
                ERI_BLOCK,
            );
        }
    }
    assert!(
        failures.is_empty(),
        "UCCSD equations above the gate: {failures:?}"
    );
}

/// **The UCCSD `e_corr`, and the `mbpt2` upstream refuses away from Γ.**
#[test]
#[ignore = "opt-in PySCF oracle"]
fn uccsd_e_corr_matches_upstream() {
    let Some(out) = emit("gamma_ug") else { return };
    let mut failures: Vec<String> = Vec::new();
    for tag in ["ug", "uk"] {
        let eris = uccsd_eris(&out, tag);
        let opts = RccsdOpts {
            conv_tol: 1e-9,
            conv_tol_normt: 1e-7,
            ..RccsdOpts::default()
        };
        let res = uccsd::kernel(&eris, &opts).expect("UCCSD");
        let want = scalar(&out, &format!("{tag}_e_corr"));
        assert!(scalar(&out, &format!("{tag}_converged")) != 0.0);
        assert!(res.converged, "this port did not converge for {tag}");
        let d = (res.e_corr - want).abs();
        println!(
            "  {tag}_e_corr {:.15e} vs {want:.15e} -> {d:e}  ({} cycles, max|Im| {:.2e})",
            res.e_corr, res.niter, res.max_imag
        );
        if !(d < E_CORR) {
            failures.push(format!("{tag}_e_corr {d:e}"));
        }

        let refused = scalar(&out, &format!("{tag}_mbpt2_refused")) != 0.0;
        assert_eq!(
            refused,
            !is_gamma(&out, tag),
            "UMP2's refusal must track the k-point"
        );
        if refused {
            println!("  {tag}: upstream refuses mbpt2 away from Gamma, as this port does");
            continue;
        }
        let (emp2, ..) = uccsd::init_amps(&eris).expect("init_amps");
        let want = scalar(&out, &format!("{tag}_e_mbpt2"));
        let d = (emp2 - want).abs();
        println!("  {tag}_e_mbpt2 {emp2:.15e} vs {want:.15e} -> {d:e}");
        if !(d < E_CORR) {
            failures.push(format!("{tag}_e_mbpt2 {d:e}"));
        }
    }
    assert!(
        failures.is_empty(),
        "UCCSD e_corr above the gate: {failures:?}"
    );
}

/// **The three Γ shims and their k-point counterparts at `[1,1,1]`** — the
/// same cross-check `oracle_gamma_rccsd.rs` runs for RCCSD, reported.
#[test]
#[ignore = "opt-in PySCF oracle"]
fn gamma_shims_and_the_kpoint_routes_agree() {
    let Some(out) = emit("gamma_ug") else { return };
    for (shim, kroute) in [("ug", "kuccsd111"), ("gg", "kgccsd111")] {
        let s = scalar(&out, &format!("{shim}_e_corr"));
        let k = scalar(&out, &format!("{kroute}_e_corr"));
        println!(
            "UPSTREAM: {shim} {s:.15e} vs {kroute} {k:.15e} -> {:e}",
            (s - k).abs()
        );
    }
    let (ge, _, _) = gccsd_eris(&out, "gg");
    let opts = RccsdOpts {
        conv_tol: 1e-9,
        conv_tol_normt: 1e-7,
        ..RccsdOpts::default()
    };
    let g = gccsd::kernel(&ge, &opts).expect("GCCSD").e_corr;
    let u = uccsd::kernel(&uccsd_eris(&out, "ug"), &opts)
        .expect("UCCSD")
        .e_corr;
    println!(
        "THIS PORT: Gamma UCCSD {u:.15e}, Gamma GCCSD {g:.15e} -> {:e}",
        (u - g).abs()
    );
    // UHF and GHF at the same Γ mean field describe the same state, so the two
    // shims must agree to their own CCSD convergence — a check neither one's
    // own oracle gate makes, because they run different equations.
    assert!(
        (u - g).abs() < 1e-7,
        "the Gamma UCCSD and GCCSD shims disagree by {:e}",
        (u - g).abs()
    );
}

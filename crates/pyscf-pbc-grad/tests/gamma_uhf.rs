//! Phase 18-10 Gate-E tests — gamma-point unrestricted gradient (`pbc/grad/uhf.py`).
//!
//! Same Gate-E provenance as `gamma_rhf.rs` (see its header):
//! [`GATE_E_TOL_LOCAL`] = 1e-8 from `measurements/gate-e-gamma.md`
//! (loosest 3D residual lif 6.241e-09) — never Gate B's `FD_TOL`, never
//! Gate C. Fixed densities, no SCF (18-09's precedent), coarse matched
//! mesh pinned by `verify_fd`, geometry in **Bohr**.
//!
//! The UHF-specific gates: both refusals shared with the restricted body,
//! the 18-06 shape (spin-summed `h1ao`/`s1`, spin-resolved `vhf`), and the
//! closed-shell-limit identity — at `dm_a = dm_b = dm/2` (HF, Coulomb-only
//! veff) the unrestricted assembly reproduces the restricted one, because
//! `J` is linear in the density.

use pyscf_algebra::oracle_sum;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_dft::multigrid::pair::MultiGridNumInt2;
use pyscf_pbc_grad::gamma_rhf::{GammaCoulombEngine, GammaRhfGradients};
use pyscf_pbc_grad::gamma_uhf::{GammaUhfGradients, GammaUksGradients, gamma_make_rdm1e_uhf};
use pyscf_pbc_grad::{Gradients, verify_fd};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

/// Gate-E tolerance (Ha/Bohr) — same provenance as `gamma_rhf.rs`:
/// `measurements/gate-e-gamma.md`, loosest 3D residual 6.241e-09 (lif).
const GATE_E_TOL_LOCAL: f64 = 1e-8;
const DISP: f64 = 1e-5;
const GAMMA: [[f64; 3]; 1] = [[0.0; 3]];

fn ni() -> MultiGridNumInt2 {
    MultiGridNumInt2::new()
}

fn h2_cell() -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::String("H 0 0 0; H 0 0 1.4".into()),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
        mesh: Some([10, 10, 10]),
        precision: 1e-10,
        pseudo: None,
        ..Default::default()
    })
    .expect("H2 cell builds")
}

/// Closed-shell-limit spin densities: each channel carries half the total,
/// so the UHF assembly must reproduce the RHF one at HF.
fn closed_shell_densities() -> ([Vec<f64>; 2], [Vec<f64>; 2], Vec<f64>, Vec<f64>) {
    let dm_sf = vec![1.0, 0.3, 0.3, 0.5];
    let dme_sf = vec![-0.4, -0.1, -0.1, -0.2];
    let half = |v: &[f64]| v.iter().map(|x| 0.5 * x).collect::<Vec<f64>>();
    let dm = [half(&dm_sf), half(&dm_sf)];
    let dme = [half(&dme_sf), half(&dme_sf)];
    (dm, dme, dm_sf, dme_sf)
}

fn trace_dot(a: &[f64], b: &[f64]) -> f64 {
    oracle_sum(&a.iter().zip(b).map(|(x, y)| x * y).collect::<Vec<f64>>())
}

fn intor_plane(cell: &Cell, intor: &str) -> Vec<f64> {
    let out =
        pyscf_pbc_gto::pbc_intor(cell, intor, &GAMMA, Default::default()).expect("intor evaluates");
    assert_eq!(out.kmats.len(), 1);
    out.kmats[0].re.clone()
}

/// Same fixed-density closure as `gamma_rhf.rs` (the Coulomb energy sees
/// only the total density): `E = Tr(dm·K) + Tr(dm·Vnuc) + ecoul + exc +
/// Enuc_ewald − Tr(dme·S)`. Coefficient 1 (see `gamma_rhf.rs`: the
/// assembly's `* 2` completes the bra-half contraction, it is not an
/// energy prefactor).
fn fixed_energy(
    ni: &MultiGridNumInt2,
    xc: &str,
    dm_sf: &[f64],
    dme_sf: &[f64],
    cell: &Cell,
) -> f64 {
    let kin = intor_plane(cell, "int1e_kin");
    let ovlp = intor_plane(cell, "int1e_ovlp");
    let vnuc =
        pyscf_pbc_dft::multigrid::pp::get_nuc(cell).expect("AFTDF nuclear attraction builds");
    let nr = ni.nr_rks(cell, xc, dm_sf).expect("nr_rks evaluates");
    let enuc = cell
        .energy_nuc()
        .expect("Ewald nuclear repulsion evaluates");
    oracle_sum(&[
        2.0 * trace_dot(dm_sf, &kin),
        2.0 * trace_dot(dm_sf, &vnuc),
        nr.ecoul,
        nr.exc,
        enuc,
        -2.0 * trace_dot(dme_sf, &ovlp),
    ])
}

fn max_abs_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b)
        .flat_map(|(r, s)| [r[0] - s[0], r[1] - s[1], r[2] - s[2]])
        .map(f64::abs)
        .fold(0.0_f64, |m, x| m.max(x))
}

/// `expect_err` without requiring `T: Debug` (the gradient bodies hold
/// `Cell`/`MultiGridNumInt2` borrows and do not implement it).
fn expect_err_msg<T>(result: Result<T, pyscf_core::PyscfRsError>, context: &str) -> String {
    match result {
        Ok(_) => panic!("{context}: expected a refusal"),
        Err(e) => e.to_string(),
    }
}

#[test]
fn refusals_are_named() {
    let cell = h2_cell();
    let (dm, dme, _, _) = closed_shell_densities();
    let engine = GammaCoulombEngine::MultiGridV2(&ni());

    // uhf.py:38-43 — a numint that is not MultiGridNumInt2.
    let err = expect_err_msg(
        GammaUhfGradients::new(
            &cell,
            GammaCoulombEngine::Other("GDF"),
            dm.clone(),
            dme.clone(),
        ),
        "non-multigrid numint must be refused",
    );
    assert!(
        err.contains("MultiGridNumInt2"),
        "refusal must name the required engine, got: {err}"
    );

    // uhf.py:78-79 — a non-gamma k-point.
    let grad = GammaUhfGradients::new(&cell, engine, dm, dme)
        .expect("gamma body builds")
        .with_kpt([0.0, 0.2, 0.0]);
    let err = expect_err_msg(grad.electronic_gradient(), "non-gamma kpt must be refused");
    assert!(
        err.contains("gamma point"),
        "refusal must name the gamma requirement, got: {err}"
    );
    println!("18-10 gate refusals ok (uhf)");
}

#[test]
fn closed_shell_limit_reproduces_rhf() {
    // The 18-06 shape check: spin-summed h1ao/s1, spin-resolved vhf, with
    // the Coulomb-only HF veff linear in each channel's density.
    let cell = h2_cell();
    let (dm, dme, dm_sf, dme_sf) = closed_shell_densities();
    let engine_holder = ni();
    let uhf = GammaUhfGradients::new(
        &cell,
        GammaCoulombEngine::MultiGridV2(&engine_holder),
        dm,
        dme,
    )
    .expect("uhf body builds");
    let rhf = GammaRhfGradients::new(
        &cell,
        GammaCoulombEngine::MultiGridV2(&engine_holder),
        dm_sf,
        dme_sf,
    )
    .expect("rhf body builds");
    let de_uhf = uhf.electronic_gradient().expect("uhf runs");
    let de_rhf = rhf.electronic_gradient().expect("rhf runs");
    let worst = max_abs_diff(&de_uhf, &de_rhf);
    println!("18-10 gate uhf-vs-rhf closed-shell |diff| = {worst:.3e}");
    assert!(
        worst < 1e-12,
        "closed-shell UHF must reproduce RHF, diff {worst:.3e}"
    );
}

#[test]
fn fd_gate_hf_on_gamma_path() {
    let cell = h2_cell();
    let (dm, dme, dm_sf, dme_sf) = closed_shell_densities();
    let engine_holder = ni();
    let grad = GammaUhfGradients::new(
        &cell,
        GammaCoulombEngine::MultiGridV2(&engine_holder),
        dm,
        dme,
    )
    .expect("gamma body builds");
    let analytic = grad.electronic_gradient().expect("gradient runs");
    assert!(
        analytic.iter().flatten().fold(0.0_f64, |m, &x| m.max(x)) > 1e-6,
        "the H2 gate must be non-vacuous, got {analytic:?}"
    );
    let report = verify_fd(
        &cell,
        &analytic,
        |c| Ok(fixed_energy(&ni(), "HF", &dm_sf, &dme_sf, c)),
        DISP,
        GATE_E_TOL_LOCAL,
    )
    .expect("verify_fd runs");
    println!(
        "18-10 Gate E (uhf/HF/h2) worst |analytic-FD| = {:.3e}",
        report.max_abs_diff
    );
    assert!(
        report.passed,
        "Gate-E FD residual {:.3e} exceeds {GATE_E_TOL_LOCAL:.0e}",
        report.max_abs_diff
    );
}

#[test]
fn assembly_is_bit_identical_at_rayon_1_and_8() {
    let cell = h2_cell();
    let (dm, dme, _, _) = closed_shell_densities();
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool")
            .install(|| {
                let engine_holder = ni();
                GammaUhfGradients::new(
                    &cell,
                    GammaCoulombEngine::MultiGridV2(&engine_holder),
                    dm.clone(),
                    dme.clone(),
                )
                .expect("body builds")
                .electronic_gradient()
                .expect("gradient runs")
            })
    };
    let one = run(1);
    let eight = run(8);
    assert_eq!(one.len(), eight.len());
    for (a, b) in one.iter().zip(&eight) {
        for c in 0..3 {
            assert_eq!(
                a[c].to_bits(),
                b[c].to_bits(),
                "component moved between RAYON 1 and 8: {:.17e} vs {:.17e}",
                a[c],
                b[c]
            );
        }
    }
    println!("18-10 gate rayon 1-vs-8 bit-identical ok (uhf)");
}

#[test]
fn veff_stays_spin_resolved_with_upstream_sign() {
    // uhf.py:89-93 — per-channel `-get_veff_ip1(dm[s], spin=1)`; the minus
    // is here, per channel.
    let cell = h2_cell();
    let (dm, dme, _, _) = closed_shell_densities();
    let engine_holder = ni();
    let grad = GammaUhfGradients::new(
        &cell,
        GammaCoulombEngine::MultiGridV2(&engine_holder),
        dm.clone(),
        dme,
    )
    .expect("body builds");
    let veff = grad.veff_ip().expect("get_veff runs");
    assert!(
        veff[0] != veff[1],
        "spin-resolved veff must differ between channels for split densities"
    );
    for s in 0..2 {
        let raw = engine_holder
            .get_veff_ip1(&cell, "HF", &dm[s], &GAMMA)
            .expect("veff builds");
        assert_eq!(veff[s].len(), raw.veff_ip1.len());
        for (a, b) in veff[s].iter().zip(&raw.veff_ip1) {
            assert_eq!(*a, -b, "channel {s} veff sign must be the negation");
        }
    }
}

#[test]
fn uks_alias_inherits_wholesale() {
    // grad/uks.py:22 — `class Gradients(uhf.Gradients)` with a
    // pass-equivalent body. Re-export, not a body.
    let cell = h2_cell();
    let (dm, dme, _, _) = closed_shell_densities();
    let engine_holder = ni();
    let uks: GammaUksGradients<'_> = GammaUksGradients::new(
        &cell,
        GammaCoulombEngine::MultiGridV2(&engine_holder),
        dm.clone(),
        dme.clone(),
    )
    .expect("uks body builds");
    let uhf = GammaUhfGradients::new(
        &cell,
        GammaCoulombEngine::MultiGridV2(&engine_holder),
        dm,
        dme,
    )
    .expect("uhf body builds");
    assert_eq!(
        uks.electronic_gradient().expect("uks runs"),
        uhf.electronic_gradient().expect("uhf runs"),
        "uks must inherit the uhf assembly wholesale"
    );
    println!("18-10 gate uks re-export ok");
}

#[test]
fn optimizer_refusal_mirrors_upstream() {
    let cell = h2_cell();
    let (dm, dme, _, _) = closed_shell_densities();
    let engine_holder = ni();
    let grad = GammaUhfGradients::new(
        &cell,
        GammaCoulombEngine::MultiGridV2(&engine_holder),
        dm,
        dme,
    )
    .expect("body builds");
    let err = grad
        .optimizer("geometric")
        .expect_err("'geometric' must be refused");
    assert!(
        err.to_string().contains("geometric"),
        "refusal must name the solver, got: {err}"
    );
    println!("18-10 gate optimizer refusal ok (uhf)");
}

#[test]
fn energy_weighted_densities_match_hand_rolls() {
    // `mol_uhf.Gradients.make_rdm1e` (uhf.py:102) per channel:
    // dme[i,j] = Σ_m C[i,m]·e[m]·n[m]·C[j,m], column-major orbitals.
    let nao = 2;
    let c_alpha = vec![1.0, 0.0, 0.0, 1.0];
    let c_beta = vec![0.0, 1.0, 1.0, 0.0];
    let e = vec![-0.5, -0.25];
    let n = vec![1.0, 1.0];
    let [dme_a, dme_b] =
        gamma_make_rdm1e_uhf([&c_alpha, &c_beta], [&e, &e], [&n, &n], nao).expect("rdm1e builds");
    assert_eq!(dme_a, vec![-0.5, 0.0, 0.0, -0.25]);
    assert_eq!(dme_b, vec![-0.25, 0.0, 0.0, -0.5]);
    let err = gamma_make_rdm1e_uhf([&c_alpha, &[][..]], [&e, &e], [&n, &n], nao)
        .expect_err("shape mismatch must be refused");
    assert!(
        err.to_string().contains("shapes disagree"),
        "refusal must name the shape contract, got: {err}"
    );
}

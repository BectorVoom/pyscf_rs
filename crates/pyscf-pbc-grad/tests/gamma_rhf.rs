//! Phase 18-10 Gate-E tests — gamma-point restricted gradient (`pbc/grad/rhf.py`).
//!
//! # Gate E provenance (read before comparing against any other number)
//!
//! The tolerance below is [`GATE_E_TOL`] = 1e-8, taken from
//! `measurements/gate-e-gamma.md` (18-16 Task 2): upstream `verify_fd`
//! residuals of the multigrid-v2 gamma gradient are 9.103e-11 (he_fcc),
//! 7.520e-10 (diamond), 3.842e-10 (si), 6.241e-09 (lif, the loosest 3D
//! residual). 1e-8 clears the loosest realised number with margin. This
//! gate is NOT the k-point Gate B (`FD_TOL` = 1e-6) and NOT Gate C: Gate E
//! measures FD-vs-analytic *consistency* of a v2-built gradient, never
//! v2's accuracy against the reference route (17-01's ~2e-8/1.5e-7
//! definitional gap, 17-12's screening floor). `grep FD_TOL` over this
//! file must find nothing.
//!
//! # What runs here
//!
//! * Run order `he_fcc` first (`nao = 1`), per the plan. A single atom's
//!   gradient is translation noise gated ~0-vs-0 (18-09's note); the
//!   non-vacuous FD gates run on an all-electron H2-like cell so the
//!   fixed-density energy closure is exact (no PP energy terms to hand-roll
//!   — the pseudo branch is covered by the he_fcc assembly gate plus the
//!   conditional-subtraction unit test).
//! * No SCF anywhere (18-09's precedent): every gate holds dm/dme FIXED and
//!   finite-differences the matching fixed-density energy. No convergence
//!   path, no second-solution noise.
//! * Coarse matched mesh, pinned on displaced cells by `verify_fd`
//!   (18-CONTEXT trap 5 — `with_coords` clones the cell, mesh untouched).
//!   Geometry in **Bohr** throughout.
//! * The RAYON 1-vs-8 determinism proof runs the assembly under explicit
//!   1- and 8-thread pools and compares bit transcripts (D-PBC-17:
//!   `oracle_sum` is fixed-order, so this must be bit-identical).

use pyscf_algebra::oracle_sum;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_dft::multigrid::pair::MultiGridNumInt2;
use pyscf_pbc_grad::contract::SCREEN_VHF_DM_CONTRACT;
use pyscf_pbc_grad::gamma_rhf::gamma_make_rdm1e;
use pyscf_pbc_grad::gamma_rhf::{GammaCoulombEngine, GammaRhfGradients, GammaRksGradients};
use pyscf_pbc_grad::{Gradients, verify_fd};
use pyscf_pbc_gto::test_systems::he_fcc;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};

/// Gate-E tolerance (Ha/Bohr). Provenance: `measurements/gate-e-gamma.md` —
/// loosest 3D residual lif 6.241e-09 (he_fcc 9.103e-11, diamond 7.520e-10,
/// si 3.842e-10). Deliberately NOT `FD_TOL` (Gate B, 1e-6): coincidence at
/// this scale is not comparability.
const GATE_E_TOL: f64 = 1e-8;
/// Central-difference half-step (Bohr). Below the truncation knee: at 1e-5
/// the remainder is the h-independent optimal-FD floor (18-09's `H`).
const DISP: f64 = 1e-5;
const GAMMA: [[f64; 3]; 1] = [[0.0; 3]];

fn ni() -> MultiGridNumInt2 {
    MultiGridNumInt2::new()
}

/// All-electron two-atom cell (H2-like, Bohr): the non-vacuous FD fixture.
/// No pseudopotential, so the fixed-density energy closure below is exact.
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

/// Fixed symmetric test densities (`nao = 2`, row/column-major identical).
fn h2_densities() -> (Vec<f64>, Vec<f64>) {
    let dm = vec![1.0, 0.3, 0.3, 0.5];
    let dme = vec![-0.4, -0.1, -0.1, -0.2];
    (dm, dme)
}

fn trace_dot(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    oracle_sum(&a.iter().zip(b).map(|(x, y)| x * y).collect::<Vec<f64>>())
}

fn intor_plane(cell: &Cell, intor: &str) -> Vec<f64> {
    let out =
        pyscf_pbc_gto::pbc_intor(cell, intor, &GAMMA, Default::default()).expect("intor evaluates");
    assert_eq!(out.kmats.len(), 1);
    out.kmats[0].re.clone()
}

/// Fixed-density total energy whose analytic derivative is exactly what
/// [`GammaRhfGradients::electronic_gradient`] computes at the same fixed
/// dm/dme (all-electron branch):
/// `E = Tr(dm·K) + Tr(dm·Vnuc) + ecoul + exc + Enuc_ewald − Tr(dme·S)`.
///
/// Coefficient 1, not 2: upstream's `* 2` on the `_contract_vhf_dm` terms
/// completes the bra-half `ip` contraction (`h1ao = −ip`, i.e. the bra half
/// of `d/dR`, doubled by symmetry for the ket half) into the FULL
/// `dTr/dR` — it is not an energy prefactor. The nuclear (b) piece enters
/// unscaled on both sides. The Coulomb+XC piece comes from `nr_rks` at the
/// same fixed dm (18-09: at fixed dm `veff` has no (b) piece — the chain
/// rule closes through the functional derivative).
fn fixed_energy(ni: &MultiGridNumInt2, xc: &str, dm: &[f64], dme: &[f64], cell: &Cell) -> f64 {
    let kin = intor_plane(cell, "int1e_kin");
    let ovlp = intor_plane(cell, "int1e_ovlp");
    let vnuc =
        pyscf_pbc_dft::multigrid::pp::get_nuc(cell).expect("AFTDF nuclear attraction builds");
    let nr = ni.nr_rks(cell, xc, dm).expect("nr_rks evaluates");
    let enuc = cell
        .energy_nuc()
        .expect("Ewald nuclear repulsion evaluates");
    oracle_sum(&[
        trace_dot(dm, &kin),
        trace_dot(dm, &vnuc),
        nr.ecoul,
        nr.exc,
        enuc,
        -trace_dot(dme, &ovlp),
    ])
}

fn max_abs(grad: &[[f64; 3]]) -> f64 {
    grad.iter().flatten().fold(0.0, |m, &x| m.max(x))
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
    let (dm, dme) = h2_densities();
    let engine = GammaCoulombEngine::MultiGridV2(&ni());

    // rhf.py:42-47 — a numint that is not MultiGridNumInt2.
    let err = expect_err_msg(
        GammaRhfGradients::new(
            &cell,
            GammaCoulombEngine::Other("FFTDF"),
            dm.clone(),
            dme.clone(),
        ),
        "non-multigrid numint must be refused",
    );
    assert!(
        err.contains("MultiGridNumInt2"),
        "refusal must name the required engine, got: {err}"
    );

    // rhf.py:78-79 — a non-gamma k-point.
    let grad = GammaRhfGradients::new(&cell, engine, dm.clone(), dme.clone())
        .expect("gamma body builds")
        .with_kpt([0.1, 0.0, 0.0]);
    let err = expect_err_msg(grad.electronic_gradient(), "non-gamma kpt must be refused");
    assert!(
        err.contains("gamma point"),
        "refusal must name the gamma requirement, got: {err}"
    );

    // rhf.py:77 — an out-of-range atom id.
    let grad = GammaRhfGradients::new(&cell, engine, dm, dme)
        .expect("gamma body builds")
        .with_atmlst(vec![99]);
    let err = expect_err_msg(
        grad.electronic_gradient(),
        "out-of-range atmlst must be refused",
    );
    assert!(
        err.contains("atmlst"),
        "refusal must name atmlst, got: {err}"
    );
    println!("18-10 gate refusals ok (rhf)");
}

#[test]
fn he_fcc_single_atom_gradient_is_translation_noise() {
    // Run order: he_fcc (nao = 1) first. Pseudo branch (He carries a
    // gth-pade entry), HF route, screened default.
    let mut cell = he_fcc();
    cell.mesh = [12, 12, 12];
    assert!(
        cell.atom_pseudo(0).is_some(),
        "he_fcc must exercise the pseudo branch"
    );
    assert!(
        SCREEN_VHF_DM_CONTRACT,
        "18-17 Task 1: the default ships screened"
    );
    let engine = GammaCoulombEngine::MultiGridV2(&ni());
    let grad =
        GammaRhfGradients::new(&cell, engine, vec![2.0], vec![-0.7]).expect("he_fcc body builds");
    let de = grad.electronic_gradient().expect("he_fcc gradient runs");
    assert_eq!(de.len(), 1);
    let worst = max_abs(&de);
    println!("18-10 gate he_fcc |de| = {worst:.3e} (analytic scale ~1e-14 upstream)");
    assert!(
        worst < GATE_E_TOL,
        "he_fcc translation noise {worst:.3e} exceeds Gate E {GATE_E_TOL:.0e}"
    );
}

#[test]
fn screening_default_is_exact_on_small_cells() {
    // 18-17 Task 1 in-crate echo: screened-vs-unscreened bitwise identity
    // (upstream: max|de_s − de_u| = 0.0 on all five reference cells).
    let engine_holder = ni();
    for (name, cell) in [("h2", h2_cell())] {
        let _ = name;
        let (dm, dme) = if cell.mol.nao_nr == 2 {
            h2_densities()
        } else {
            (vec![2.0], vec![-0.7])
        };
        let screened = GammaRhfGradients::new(
            &cell,
            GammaCoulombEngine::MultiGridV2(&engine_holder),
            dm.clone(),
            dme.clone(),
        )
        .expect("body builds")
        .electronic_gradient()
        .expect("screened runs");
        let unscreened = GammaRhfGradients::new(
            &cell,
            GammaCoulombEngine::MultiGridV2(&engine_holder),
            dm,
            dme,
        )
        .expect("body builds")
        .with_screened(false)
        .electronic_gradient()
        .expect("unscreened runs");
        assert_eq!(
            screened, unscreened,
            "screened-vs-unscreened must be exact on small cells"
        );
    }
    println!("18-10 gate screening exact ok (h2)");
}

#[test]
fn fd_gate_hf_on_gamma_path() {
    let cell = h2_cell();
    let (dm, dme) = h2_densities();
    let engine_holder = ni();
    let grad = GammaRhfGradients::new(
        &cell,
        GammaCoulombEngine::MultiGridV2(&engine_holder),
        dm.clone(),
        dme.clone(),
    )
    .expect("gamma body builds");
    let analytic = grad.electronic_gradient().expect("gradient runs");
    assert!(
        max_abs(&analytic) > 1e-6,
        "the H2 gate must be non-vacuous, got {analytic:?}"
    );
    let report = verify_fd(
        &cell,
        &analytic,
        |c| Ok(fixed_energy(&ni(), "HF", &dm, &dme, c)),
        DISP,
        GATE_E_TOL,
    )
    .expect("verify_fd runs");
    println!(
        "18-10 Gate E (rhf/HF/h2) worst |analytic-FD| = {:.3e}",
        report.max_abs_diff
    );
    assert!(
        report.passed,
        "Gate-E FD residual {:.3e} exceeds {GATE_E_TOL:.0e}",
        report.max_abs_diff
    );
}

#[test]
fn fd_gate_lda_on_gamma_path() {
    // The KS route through the same body (`xc_code` carried into `get_veff`,
    // rhf.py:140 — `None`-for-HF is how one function serves both).
    let cell = h2_cell();
    let (dm, dme) = h2_densities();
    let engine_holder = ni();
    let grad = GammaRhfGradients::new(
        &cell,
        GammaCoulombEngine::MultiGridV2(&engine_holder),
        dm.clone(),
        dme.clone(),
    )
    .expect("gamma body builds")
    .with_xc(Some("LDA"));
    let analytic = grad.electronic_gradient().expect("gradient runs");
    assert!(
        max_abs(&analytic) > 1e-6,
        "the H2/LDA gate must be non-vacuous, got {analytic:?}"
    );
    let report = verify_fd(
        &cell,
        &analytic,
        |c| Ok(fixed_energy(&ni(), "LDA", &dm, &dme, c)),
        DISP,
        GATE_E_TOL,
    )
    .expect("verify_fd runs");
    println!(
        "18-10 Gate E (rhf/LDA/h2) worst |analytic-FD| = {:.3e}",
        report.max_abs_diff
    );
    assert!(
        report.passed,
        "Gate-E FD residual {:.3e} exceeds {GATE_E_TOL:.0e}",
        report.max_abs_diff
    );
}

#[test]
fn assembly_is_bit_identical_at_rayon_1_and_8() {
    let cell = h2_cell();
    let (dm, dme) = h2_densities();
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool")
            .install(|| {
                let engine_holder = ni();
                GammaRhfGradients::new(
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
    println!("18-10 gate rayon 1-vs-8 bit-identical ok (rhf)");
}

#[test]
fn signs_match_upstream_and_atmlst_selects() {
    let cell = h2_cell();
    let (dm, dme) = h2_densities();
    let engine_holder = ni();
    let grad = GammaRhfGradients::new(
        &cell,
        GammaCoulombEngine::MultiGridV2(&engine_holder),
        dm.clone(),
        dme.clone(),
    )
    .expect("body builds");

    // get_ovlp is `-pbc_intor('int1e_ipovlp')` (rhf.py:134-135).
    let raw_ovlp = intor_plane(&cell, "int1e_ipovlp");
    let ovlp = grad.overlap_ip().expect("ovlp runs");
    assert_eq!(ovlp.len(), raw_ovlp.len());
    for (a, b) in ovlp.iter().zip(&raw_ovlp) {
        assert_eq!(*a, -b, "ovlp sign must be the negation");
    }
    assert!(
        ovlp.iter().any(|x| x.abs() > 1e-8),
        "the sign gate must not be vacuous"
    );

    // get_veff is `-get_veff_ip1` (rhf.py:138-142 — the minus is here).
    let raw = engine_holder
        .get_veff_ip1(&cell, "HF", &dm, &GAMMA)
        .expect("veff builds");
    let veff = grad.veff_ip().expect("get_veff runs");
    assert_eq!(veff.len(), raw.veff_ip1.len());
    for (a, b) in veff.iter().zip(&raw.veff_ip1) {
        assert_eq!(*a, -b, "veff sign must be the negation");
    }

    // de = de[atmlst] (rhf.py:77).
    let full = grad.electronic_gradient().expect("gradient runs");
    let sub = GammaRhfGradients::new(
        &cell,
        GammaCoulombEngine::MultiGridV2(&engine_holder),
        dm,
        dme,
    )
    .expect("body builds")
    .with_atmlst(vec![1])
    .electronic_gradient()
    .expect("subset runs");
    assert_eq!(sub, vec![full[1]]);
}

#[test]
fn optimizer_refusal_mirrors_upstream() {
    // rhf.py:169-178 — only 'ase' is accepted; 'geometric' raises.
    let cell = h2_cell();
    let (dm, dme) = h2_densities();
    let engine_holder = ni();
    let grad = GammaRhfGradients::new(
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
    println!("18-10 gate optimizer refusal ok (rhf)");
}

#[test]
fn energy_weighted_density_matches_hand_roll() {
    // `mol_rhf.Gradients.make_rdm1e` (rhf.py:187):
    // dme[i,j] = Σ_m C[i,m]·e[m]·n[m]·C[j,m], column-major orbitals.
    let dme = gamma_make_rdm1e(&[1.0, 0.0, 0.0, 1.0], &[-0.5, -0.25], &[1.0, 1.0], 2)
        .expect("rdm1e builds");
    assert_eq!(dme, vec![-0.5, 0.0, 0.0, -0.25]);
    // Single orbital: dme = e·n·C·Cᵀ.
    let dme = gamma_make_rdm1e(&[2.0], &[-0.5], &[1.0], 1).expect("rdm1e builds");
    assert_eq!(dme, vec![-2.0]);
}

#[test]
fn rks_alias_inherits_wholesale() {
    // grad/rks.py:22 — `class Gradients(rhf.Gradients)` with a
    // pass-equivalent body. The alias below IS the whole port: it must
    // expose the identical assembly with zero new logic.
    let cell = h2_cell();
    let (dm, dme) = h2_densities();
    let engine_holder = ni();
    let rks: GammaRksGradients<'_> = GammaRksGradients::new(
        &cell,
        GammaCoulombEngine::MultiGridV2(&engine_holder),
        dm.clone(),
        dme.clone(),
    )
    .expect("rks body builds")
    .with_xc(Some("LDA"));
    let rhf = GammaRhfGradients::new(
        &cell,
        GammaCoulombEngine::MultiGridV2(&engine_holder),
        dm,
        dme,
    )
    .expect("rhf body builds")
    .with_xc(Some("LDA"));
    assert_eq!(
        rks.electronic_gradient().expect("rks runs"),
        rhf.electronic_gradient().expect("rhf runs"),
        "rks must inherit the rhf assembly wholesale"
    );
    println!("18-10 gate rks re-export ok");
}

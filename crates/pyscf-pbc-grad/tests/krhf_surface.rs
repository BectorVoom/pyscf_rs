//! Plan 18-19 surface gates — `pbc/grad/krhf.py` driver surface + scanner seam.
//!
//! * Gate B: `verify_fd` against the 18-19 scanner's energy seam (Task 2),
//!   at `FD_TOL = 1e-6` Ha/Bohr, on displaced diamond at gamma and 1×1×2.
//! * Gate C: `lib.fp(g)` against upstream's committed `-0.9017171774435333`
//!   (`test_krhf.py:50`) on upstream's own cell, at upstream's 6 decimals.
//! * `test_exxdiv_ewald` (upstream issue 2877, `test_krhf.py:58`).
//! * The `optimizer(solver=)` refusal contract (18-CONTEXT §1.7).
//! * `GradientsBase::get_jk/get_j/get_k` inherit 18-04's named refusal on a
//!   non-FFTDF builder.
//! * The scanner side-effect contract (`krhf.py:311-313`) + the inert
//!   grids-reset hook for the KRKS subclass.
//!
//! # Geometry is specified in BOHR unless the upstream fixture says otherwise
//!
//! Gate C's cell is upstream's own `test_krhf.py` fixture (Bohr atom positions
//! and lattice, even-tempered `[[0,[1.3,1]],[1,[0.8,1]]]` basis as parsed
//! shells, `gth-pade`, pinned `mesh = [13]*3`). The `exxdiv_ewald` cell keeps
//! upstream's default Angstrom unit.

use pyscf_algebra::CTensor;
use pyscf_core::{ParsedBasis, ShellSpec, Unit};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, ExxDiv};
use pyscf_pbc_grad::{
    Gradients, KrhfGradients, KrhfScannerConfig, ScfGradScanner, fingerprint, verify_fd,
};
use pyscf_pbc_scf::{KScfConfig, Krhf};

/// Upstream `test_krhf.py:22-38`'s own cell: even-tempered s (`1.3`) + p
/// (`0.8`) basis on both carbons, `gth-pade`, Bohr positions and lattice,
/// pinned `mesh = [13]*3`.
fn upstream_krhf_cell() -> Cell {
    let basis = BasisInput::Parsed(ParsedBasis {
        shells: vec![
            ShellSpec {
                l: 0,
                exponents: vec![1.3],
                coeffs: vec![vec![1.0]],
            },
            ShellSpec {
                l: 1,
                exponents: vec![0.8],
                coeffs: vec![vec![1.0]],
            },
        ],
    });
    let q = 1.685068664391;
    let h = 3.370137329;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("C".into(), [0.0, 0.0, 0.0]),
                ("C".into(), [q, q, q]),
            ]),
            basis,
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        mesh: Some([13, 13, 13]),
        ..Default::default()
    })
    .expect("upstream test_krhf cell must build")
}

/// Displaced diamond in Bohr (atom 1 pushed +0.02 Bohr in x, so the gradient
/// is O(10⁻²) Ha/Bohr and no gate can pass vacuously), pinned `mesh = [13]*3`
/// exactly like `krhf_verify_fd.rs`.
fn displaced_diamond() -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("C".into(), [0.0, 0.0, 0.0]),
                ("C".into(), [q + 0.02, q, q]),
            ]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        mesh: Some([13, 13, 13]),
        ..Default::default()
    })
    .expect("displaced diamond must build")
}

/// Gate-B SCF tightness: the same floor `krhf_verify_fd.rs` gates at
/// (`conv_tol = 1e-12`, `conv_tol_grad = 1e-10`), so an SCF-energy error of
/// ~1e-12 stays ~1e-7 Ha/Bohr after division by the 1e-5 full step — a decade
/// inside the 1e-6 gate.
fn gate_b_config(kpts: &[[f64; 3]]) -> KrhfScannerConfig {
    KrhfScannerConfig {
        kpts: kpts.to_vec(),
        exxdiv: Some(ExxDiv::Ewald),
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-10),
        max_cycle: 100,
    }
}

fn peak(gradient: &[[f64; 3]]) -> f64 {
    gradient
        .iter()
        .flat_map(|row| row.iter())
        .fold(0.0_f64, |a, v| a.max(v.abs()))
}

fn max_abs_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(r, s)| r.iter().zip(s.iter()).map(|(x, y)| (x - y).abs()))
        .fold(0.0_f64, f64::max)
}

/// Deterministic complex filler — no RNG crate, no fixtures to drift.
fn z(i: usize, j: usize, s: usize) -> (f64, f64) {
    (
        ((i * 7 + j * 3 + s * 13) as f64 * 0.37).sin(),
        ((i * 5 + j * 11 + s * 7) as f64 * 0.61).cos(),
    )
}

/// Seeded column-major orbitals + ascending energies + RHF occupations.
fn synthetic_orbitals(nao: usize, nkpts: usize) -> (Vec<CTensor>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let nocc = nao / 2;
    let coeff = (0..nkpts)
        .map(|k| {
            let mut re = vec![0.0_f64; nao * nao];
            let mut im = vec![0.0_f64; nao * nao];
            for m in 0..nao {
                for i in 0..nao {
                    let (r, v) = z(i, m, k + 41);
                    re[i + m * nao] = r;
                    im[i + m * nao] = v;
                }
            }
            CTensor::from_planes(re, im)
        })
        .collect();
    let energy = (0..nkpts)
        .map(|k| {
            (0..nao)
                .map(|m| -1.0 + 0.25 * m as f64 + 0.01 * k as f64)
                .collect()
        })
        .collect();
    let occ = (0..nkpts)
        .map(|_| (0..nao).map(|m| if m < nocc { 2.0 } else { 0.0 }).collect())
        .collect();
    (coeff, energy, occ)
}

// ---------------------------------------------------------------------------
// Gate C — lib.fp(g) on upstream's own cell at upstream's 6 decimals
// ---------------------------------------------------------------------------

/// `test_krhf.py:48-50`: KRHF at `kpts = [1,1,2]`, `exxdiv=None`,
/// `conv_tol=1e-10`, `conv_tol_grad=1e-6`, `mesh = [13]*3`;
/// `assertAlmostEqual(lib.fp(g), -0.9017171774435333, 6)`.
#[test]
fn gate_c_fingerprint_matches_upstream_to_6dp() {
    let cell = upstream_krhf_cell();
    let kpts = cell.make_kpts([1, 1, 2]).expect("1x1x2 k-mesh");
    assert_eq!(kpts.len(), 2);
    let mut mf = Krhf::new(cell.clone(), &kpts).expect("KRHF builds");
    mf.exxdiv = None;
    let cfg = KScfConfig {
        conv_tol: 1e-10,
        conv_tol_grad: Some(1e-6),
        max_cycle: 100,
        ..KScfConfig::for_cell(&cell)
    };
    let result = mf.kernel(&cfg).expect("Gate-C SCF");
    assert!(
        result.converged,
        "Gate-C SCF did not converge after {} cycles — the fp number is meaningless",
        result.cycles
    );
    let grad = KrhfGradients::new(&mf, result.mo_energy, result.mo_coeff, result.mo_occ)
        .expect("gradient object");
    let g = grad.kernel().expect("analytic gradient");
    let fp = fingerprint(&g);
    let want = -0.9017171774435333;
    let residual = (fp - want).abs();
    println!("Gate C (KRHF 1x1x2 upstream cell): lib.fp(g) = {fp:.16} (want {want:.16}, residual {residual:.3e})");
    assert!(
        residual < 0.5e-6,
        "Gate C FAILED: |fp − upstream| = {residual:.3e} >= 0.5e-6 (6dp)\nfp = {fp:.16}\nupstream = {want:.16}\ngradient = {g:?}",
    );
}

// ---------------------------------------------------------------------------
// Gate B — verify_fd against the scanner, gamma and 1×1×2
// ---------------------------------------------------------------------------

/// Gate B at 18-01's measured tolerance (`FD_TOL = 1e-6` Ha/Bohr), driven
/// through the 18-19 scanner seam (Task 2): the analytic gradient comes from
/// [`ScfGradScanner::scan`], the finite difference from
/// [`ScfGradScanner::energy`] — the same `(e_tot, de)` object 18-14 drives.
#[test]
fn gate_b_scanner_verify_fd_gamma_and_112() {
    let cell = displaced_diamond();
    for (name, kpts) in [
        ("gamma", vec![[0.0_f64; 3]]),
        (
            "1x1x2",
            cell.make_kpts([1, 1, 2]).expect("1x1x2 k-mesh"),
        ),
    ] {
        let scanner = ScfGradScanner::new(cell.clone(), gate_b_config(&kpts));
        let (e_tot, analytic) = scanner.scan(&cell).expect("scanner scan");
        assert_eq!(
            scanner.converged(),
            Some(true),
            "Gate B ({name}): SCF did not converge (e_tot = {e_tot:e}) — the FD number is meaningless"
        );
        assert!(e_tot.is_finite(), "Gate B ({name}): non-finite scanner energy");
        let height = peak(&analytic);
        assert!(
            height > 1e-4,
            "Gate B ({name}) is vacuous: max|analytic| = {height:e} on the displaced cell"
        );
        let report = verify_fd(&cell, &analytic, |c| scanner.energy(c), 5e-6, 1e-6)
            .expect("finite-difference harness");
        println!(
            "Gate B (KRHF {name} displaced diamond): max|fd − analytic| = {:.3e} Ha/Bohr",
            report.max_abs_diff
        );
        assert!(
            report.passed,
            "Gate B ({name}) FAILED: max|fd − analytic| = {:.3e} > 1e-6\nanalytic = {analytic:?}\nfd = {:?}",
            report.max_abs_diff, report.fd_grad,
        );
    }
}

// ---------------------------------------------------------------------------
// test_exxdiv_ewald (upstream issue 2877)
// ---------------------------------------------------------------------------

/// `test_krhf.py:58-67`: on H₂ the `exxdiv='ewald'` gradient and the
/// `exxdiv=None` gradient agree to 6 decimals — the only coverage of the
/// `exxdiv`/Ewald interaction. Cell keeps upstream's default Angstrom unit.
#[test]
fn exxdiv_ewald_matches_none_to_6dp() {
    let basis = BasisInput::Parsed(ParsedBasis {
        shells: vec![ShellSpec {
            l: 0,
            exponents: vec![1.0],
            coeffs: vec![vec![1.0]],
        }],
    });
    let cell = Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::String("H 0 0 0; H 0 0 0.74".into()),
            basis,
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix([
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 4.0],
        ]),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("H2 cell must build");
    let gamma: &[[f64; 3]] = &[[0.0; 3]];
    let gradient_with = |exxdiv: Option<ExxDiv>| {
        let mut mf = Krhf::new(cell.clone(), gamma).expect("KRHF builds");
        mf.exxdiv = exxdiv;
        let result = mf.kernel(&KScfConfig::for_cell(&cell)).expect("H2 SCF");
        assert!(
            result.converged,
            "H2 SCF (exxdiv = {exxdiv:?}) did not converge after {} cycles",
            result.cycles
        );
        KrhfGradients::new(&mf, result.mo_energy, result.mo_coeff, result.mo_occ)
            .expect("gradient object")
            .kernel()
            .expect("analytic gradient")
    };
    let reference = gradient_with(None);
    let dat = gradient_with(Some(ExxDiv::Ewald));
    let worst = max_abs_diff(&dat, &reference);
    println!("test_exxdiv_ewald (H2): max|ewald − none| = {worst:.3e} Ha/Bohr");
    assert!(
        worst < 0.5e-6,
        "test_exxdiv_ewald FAILED: max|dat − ref| = {worst:.3e} >= 0.5e-6\ndat = {dat:?}\nref = {reference:?}",
    );
}

// ---------------------------------------------------------------------------
// optimizer(solver=) refusal contract (18-CONTEXT §1.7)
// ---------------------------------------------------------------------------

/// `krhf.py:290-298` accepts ONLY `'ase'`; `'geometric'` raises — and that is
/// upstream's behaviour, not a port gap. The `'ase'` branch itself belongs to
/// Phase 20's `tools/pyscf_ase` and is a named refusal until then.
#[test]
fn optimizer_geometric_raises_and_ase_is_a_named_phase20_refusal() {
    let cell = displaced_diamond();
    let kpts = vec![[0.0_f64; 3]];
    let mf = Krhf::new(cell.clone(), &kpts).expect("KRHF holder");
    let nao = cell.mol.nao_nr;
    let (coeff, energy, occ) = synthetic_orbitals(nao, 1);
    let grad = KrhfGradients::new(&mf, energy, coeff, occ).expect("gradient object");
    let err = grad
        .optimizer("geometric")
        .expect_err("'geometric' must raise (upstream krhf.py:298)");
    let msg = err.to_string();
    assert!(
        msg.contains("geometric") && msg.contains("not supported"),
        "refusal must name the solver and the support boundary, got: {msg}"
    );
    let err = grad
        .optimizer("ASE")
        .expect_err("'ase' is Phase-20 scope and must refuse by name");
    let msg = err.to_string();
    assert!(
        msg.contains("ASE"),
        "'ase' refusal must name the missing Phase-20 surface, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// GradientsBase::get_jk/get_j/get_k inherit 18-04's named refusal
// ---------------------------------------------------------------------------

/// 18-CONTEXT §1.2: `get_jk_e1`/`get_j_e1`/`get_k_e1` exist ONLY on FFTDF
/// (`fft.py:324-340`). On a GDF builder every one of the three gradient halves
/// refuses by name — never a fallback to another route.
#[test]
fn jk_halves_refuse_a_non_fftdf_builder_by_name() {
    use pyscf_pbc_df::Gdf;
    use pyscf_pbc_scf::krdm::make_rdm1;

    let cell = displaced_diamond();
    let kpts = vec![[0.0_f64; 3]];
    let mf = Krhf::from_df(Box::new(Gdf::new(cell.clone(), &kpts)));
    let nao = cell.mol.nao_nr;
    let (coeff, energy, occ) = synthetic_orbitals(nao, 1);
    let dm = vec![make_rdm1(&coeff, &occ, nao)];
    let grad = KrhfGradients::new(&mf, energy, coeff, occ).expect("gradient object");
    for (name, result) in [
        ("get_jk", grad.get_jk(&dm).map(|_| ())),
        ("get_j", grad.get_j(&dm).map(|_| ())),
        ("get_k", grad.get_k(&dm).map(|_| ())),
    ] {
        let err = result.expect_err(&format!("{name} on GDF must refuse"));
        let msg = err.to_string();
        assert!(
            msg.contains("only on FFTDF"),
            "{name} refusal must name FFTDF as the only gradient route, got: {msg}"
        );
    }
}

// ---------------------------------------------------------------------------
// Scanner side-effect contract + inert grids hook
// ---------------------------------------------------------------------------

/// `krhf.py:311-313`: the scanner records the cell it evaluated and reuses the
/// converged density as the next guess; `:338-339`'s `grids.reset` hook exists
/// for the KRKS subclass and is inert on KRHF.
#[test]
fn scanner_records_side_effects_and_grids_hook_is_inert() {
    let cell = displaced_diamond();
    let kpts = vec![[0.0_f64; 3]];
    let scanner = ScfGradScanner::new(cell.clone(), gate_b_config(&kpts));
    assert_eq!(scanner.converged(), None, "no scan has run yet");

    let (e1, de1) = scanner.scan(&cell).expect("first scan");
    assert_eq!(scanner.converged(), Some(true), "first scan must converge");
    let recorded = scanner.cell().expect("scanner records its cell");
    assert_eq!(
        recorded.atom_coords(),
        cell.atom_coords(),
        "scanner must record the evaluated cell (:334)"
    );

    // The second scan reuses the stored density as its guess and lands on the
    // same converged point.
    let (e2, de2) = scanner.scan(&cell).expect("second scan");
    assert_eq!(scanner.converged(), Some(true));
    assert!(
        (e2 - e1).abs() < 1e-9,
        "guess reuse moved the converged energy: {e1:.12} vs {e2:.12}"
    );
    assert!(
        max_abs_diff(&de2, &de1) < 1e-9,
        "guess reuse moved the gradient"
    );

    // The KRKS-subclass hook is inert on KRHF — and callable.
    scanner.reset_grids(&cell);

    // The geometry overload evaluates the same physics as the cell overload.
    let (e3, _) = scanner
        .scan_coords(&cell.atom_coords())
        .expect("geometry overload");
    assert!(
        (e3 - e1).abs() < 1e-9,
        "scan_coords disagrees with scan: {e3:.12} vs {e1:.12}"
    );
}

/// The trait's `as_scanner` energy seam agrees with the SCF it wraps.
#[test]
fn trait_as_scanner_energy_matches_the_scf() {
    let cell = displaced_diamond();
    let kpts = vec![[0.0_f64; 3]];
    let mf = Krhf::new(cell.clone(), &kpts).expect("KRHF holder");
    let cfg = KScfConfig::for_cell(&cell);
    let result = mf.kernel(&cfg).expect("direct SCF");
    assert!(result.converged, "direct SCF did not converge");
    let grad = KrhfGradients::new(&mf, result.mo_energy, result.mo_coeff, result.mo_occ)
        .expect("gradient object");
    let energy = grad.as_scanner().expect("trait as_scanner seam");
    let e_seam = energy(&cell).expect("scanner energy");
    assert!(
        (e_seam - result.e_tot).abs() < 1e-12,
        "as_scanner energy {} disagrees with the SCF {}",
        e_seam,
        result.e_tot
    );
}

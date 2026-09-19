//! Plan 18-07 — `pyscf/pbc/grad/kuks.py` (135 l): KUKS k-point gradient.
//!
//! `class Gradients(uhf_grad.Gradients)` (`:122`): 18-06's assembly with
//! `get_veff` replaced by the XC-grid term plus the spin-threaded
//! Coulomb/hybrid-exchange terms.
//!
//! * Unit tests gate the three refusals (`grid_response`, meta-GGA, NLC)
//!   and the hybrid `veff` spin threading (`vxc[s] + J_sum - K[s]`, no
//!   `.5`) against the `jk` route's own halves on a spin-polarised density.
//! * **Gate C** (`kuks_matches_upstream_fingerprints`): the same three
//!   committed constants as KRKS (upstream's KUKS fixture is closed-shell
//!   diamond) via `Kuks` with `init_guess_breaksym = 0` — the exact fixed
//!   point that recovers the restricted solution (see 18-06's finding,
//!   recorded in `tests/kuhf.rs`).
//! * **Gate B** (`kuks_*_passes_verify_fd`): `verify_fd` on the genuinely
//!   spin-polarised HeH doublet (channels asserted polarised, else the gate
//!   would test the restricted path under another name) at upstream's DFT
//!   step (half-step `5e-4`) and `FD_TOL = 1e-6` Ha/Bohr.
//!
//! # Geometry is specified in BOHR

use pyscf_algebra::CTensor;
use pyscf_pbc_grad::{fingerprint, KuksGradients};
use pyscf_pbc_gto::Cell;

/// Deterministic complex filler — no RNG crate, no fixtures to drift.
fn z(i: usize, j: usize, s: usize) -> (f64, f64) {
    (
        ((i * 7 + j * 3 + s * 13) as f64 * 0.37).sin(),
        ((i * 5 + j * 11 + s * 7) as f64 * 0.61).cos(),
    )
}

/// Upstream `test_kuks.py:setUpModule`'s cell EXACTLY (same diamond as
/// `test_krks.py`): custom uncontracted `[[0,[1.3,1]],[1,[0.8,1]]` basis,
/// `gth-pade`, Bohr lattice `3.370137329`, second C at `1.685068664391`,
/// `mesh = [13]*3`.
fn upstream_test_cell() -> Cell {
    use pyscf_core::{ParsedBasis, ShellSpec};
    use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
    use pyscf_pbc_gto::{ALattice, CellBuildArgs};
    let q = 1.685068664391;
    let h = 3.370137329;
    let c_basis = ParsedBasis {
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
    };
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("C".into(), [0.0, 0.0, 0.0]),
                ("C".into(), [q, q, q]),
            ]),
            basis: BasisInput::Parsed(c_basis),
            unit: pyscf_core::Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        mesh: Some([13, 13, 13]),
        ..Default::default()
    })
    .expect("upstream test_kuks cell must build")
}

fn kpts_113(cell: &Cell) -> Vec<[f64; 3]> {
    let kpts = pyscf_pbc_gto::kpts_mesh::make_kpts_default(cell, [1, 1, 3]).expect("1x1x3 k-mesh");
    assert_eq!(kpts.len(), 3, "upstream test_kuks uses 3 k-points");
    kpts
}

/// Seeded column-major orbitals + ascending energies + single-occupancy
/// per-spin occupations, one set per spin channel.
fn synthetic_spin_orbitals(
    nao: usize,
    nkpts: usize,
    seed: usize,
) -> (Vec<CTensor>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let nocc = nao / 2;
    let coeff = (0..nkpts)
        .map(|k| {
            let mut re = vec![0.0_f64; nao * nao];
            let mut im = vec![0.0_f64; nao * nao];
            for m in 0..nao {
                for i in 0..nao {
                    let (r, v) = z(i, m, k + seed);
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
        .map(|_| {
            (0..nao)
                .map(|m| if m < nocc { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();
    (coeff, energy, occ)
}

/// Seeded Hermitian density per k, row-major, with a per-set offset so two
/// sets differ.
fn hermitian_dm_set(nao: usize, nkpts: usize, seed: usize) -> Vec<CTensor> {
    (0..nkpts)
        .map(|k| {
            let mut re = vec![0.0_f64; nao * nao];
            let mut im = vec![0.0_f64; nao * nao];
            for i in 0..nao {
                for j in 0..nao {
                    let mut pr = 0.0_f64;
                    let mut pi = 0.0_f64;
                    for m in 0..nao {
                        let (ar, ai) = z(i, m, k + seed);
                        let (br, bi) = z(j, m, k + seed);
                        pr += ar * br + ai * bi;
                        pi += ai * br - ar * bi;
                    }
                    re[i * nao + j] = pr + if i == j { 1.0 } else { 0.0 };
                    im[i * nao + j] = pi;
                }
            }
            CTensor::from_planes(re, im)
        })
        .collect()
}

fn max_abs(a: &[[f64; 3]]) -> f64 {
    a.iter()
        .flat_map(|r| r.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()))
}

// ---------------------------------------------------------------------------
// Task 1: the three refusals.
// ---------------------------------------------------------------------------

/// `kuks.py:49-50` refuses `grid_response = True` (and `:129` hard-sets it
/// `False` in `__init__` — the constructor default here is `false` to match).
#[test]
fn grid_response_is_refused_not_inherited() {
    use pyscf_pbc_dft::kuks::Kuks;
    let cell = upstream_test_cell();
    let kpts = vec![[0.0_f64; 3]];
    let mf = Kuks::new(cell, &kpts, "lda,vwn").expect("KUKS holder");
    let nao = mf.cell().mol.nao_nr;
    let (ca, ea, oa) = synthetic_spin_orbitals(nao, 1, 41);
    let (cb, eb, ob) = synthetic_spin_orbitals(nao, 1, 77);
    let grad = KuksGradients::new(
        &mf,
        [ea, eb].concat(),
        [ca, cb].concat(),
        [oa, ob].concat(),
    )
    .expect("gradient object")
    .with_grid_response(true);
    let err = grad.kernel().expect_err("grid_response = True must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("grid_response") && msg.contains("not yet implemented"),
        "refusal must name grid_response, got: {msg}"
    );
}

/// `kuks.py:118` — `raise NotImplementedError("metaGGA")`.
#[test]
fn meta_gga_is_refused_by_name() {
    use pyscf_pbc_dft::kuks::Kuks;
    let cell = upstream_test_cell();
    let kpts = vec![[0.0_f64; 3]];
    let mf = Kuks::new(cell, &kpts, "TPSS").expect("KUKS holder");
    let nao = mf.cell().mol.nao_nr;
    let (ca, ea, oa) = synthetic_spin_orbitals(nao, 1, 41);
    let (cb, eb, ob) = synthetic_spin_orbitals(nao, 1, 77);
    let grad = KuksGradients::new(
        &mf,
        [ea, eb].concat(),
        [ca, cb].concat(),
        [oa, ob].concat(),
    )
    .expect("gradient object");
    let err = grad.kernel().expect_err("meta-GGA must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("metaGGA") && msg.contains("not yet implemented"),
        "refusal must name metaGGA, got: {msg}"
    );
}

/// `kuks.py:116` — `raise NotImplementedError("NLC")`.
#[test]
fn nlc_is_refused_by_name() {
    use pyscf_pbc_dft::kuks::Kuks;
    let cell = upstream_test_cell();
    let kpts = vec![[0.0_f64; 3]];
    let mf = Kuks::new(cell, &kpts, "VV10").expect("KUKS holder");
    let nao = mf.cell().mol.nao_nr;
    let (ca, ea, oa) = synthetic_spin_orbitals(nao, 1, 41);
    let (cb, eb, ob) = synthetic_spin_orbitals(nao, 1, 77);
    let grad = KuksGradients::new(
        &mf,
        [ea, eb].concat(),
        [ca, cb].concat(),
        [oa, ob].concat(),
    )
    .expect("gradient object");
    let err = grad.kernel().expect_err("NLC must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("NLC") && msg.contains("not yet implemented"),
        "refusal must name NLC, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Task 3: hybrid get_veff spin threading on a polarised density.
// ---------------------------------------------------------------------------

/// `veff[s] = vxc[s] + vj[0] + vj[1] - vk[s]` (`kuks.py:66` — FULL exchange,
/// no `.5`) elementwise against the `jk` route's own halves plus the grid
/// core's own sets, on a genuinely spin-polarised density (exchange must
/// differ per spin — otherwise the assembly test is vacuous). Runs on the
/// small-mesh Gate-C cell at gamma: the 18-04 route is mesh-independent.
///
/// Plus the pure-functional sibling (`vxc[s] + J_sum`, `kuks.py:58`).
#[test]
fn veff_assembles_broadcast_coulomb_minus_full_spin_exchange() {
    use pyscf_pbc_dft::kuks::Kuks;
    for (xc, hybrid) in [("lda,vwn", false), ("pbe0", true)] {
        let cell = upstream_test_cell();
        let kpts = vec![[0.0_f64; 3]];
        let nkpts = 1;
        let nao = cell.mol.nao_nr;
        let mf = Kuks::new(cell, &kpts, xc).expect("KUKS holder");
        let (ca, ea, oa) = synthetic_spin_orbitals(nao, nkpts, 41);
        let (cb, eb, ob) = synthetic_spin_orbitals(nao, nkpts, 77);
        let grad = KuksGradients::new(
            &mf,
            [ea, eb].concat(),
            [ca, cb].concat(),
            [oa, ob].concat(),
        )
        .expect("gradients");
        let dma = hermitian_dm_set(nao, nkpts, 3);
        let dmb = hermitian_dm_set(nao, nkpts, 17);
        let dm = vec![dma, dmb];
        let vxc = grad.vxc(&dm).expect("vxc sets");
        assert_eq!(vxc.len(), 2);
        let veff = grad.veff(&dm).expect("veff");
        assert_eq!(veff.len(), 2);
        if hybrid {
            let (vj, vk) = grad.jk_deriv(&dm).expect("jk");
            assert_eq!((vj.len(), vk.len()), (2, 2));
            // pbe0's exact-exchange fraction (upstream `vk *= hyb`, kuks.py:62).
            let hyb = pyscf_pbc_dft::xc::rsh_and_hybrid_coeff(xc)
                .expect("hybrid coefficients")
                .2;
            let vk_gap: f64 = vk[0][0]
                .iter()
                .zip(vk[1][0].iter())
                .flat_map(|(a, b)| {
                    a.re
                        .iter()
                        .zip(&b.re)
                        .chain(a.im.iter().zip(&b.im))
                        .map(|(p, q)| (p - q).abs())
                })
                .fold(0.0_f64, f64::max);
            assert!(
                vk_gap > 1e-6,
                "vk must differ per spin on an open-shell density, got {vk_gap:e}"
            );
            for s in 0..2 {
                for x in 0..3 {
                    for k in 0..nkpts {
                        for i in 0..nao * nao {
                            // FULL exchange, no .5 (kuks.py:66 vs krks.py:64);
                            // the route's vk is unscaled, veff carries hyb·vk.
                            let want = vxc[s][x][k].re[i] + vj[0][x][k].re[i]
                                + vj[1][x][k].re[i]
                                - hyb * vk[s][x][k].re[i];
                            let got = veff[s][x][k].re[i];
                            assert!(
                                (got - want).abs() < 1e-12,
                                "re s={s} x={x} k={k} i={i}: {got} vs {want}"
                            );
                            let want_im = vxc[s][x][k].im[i] + vj[0][x][k].im[i]
                                + vj[1][x][k].im[i]
                                - hyb * vk[s][x][k].im[i];
                            let got_im = veff[s][x][k].im[i];
                            assert!(
                                (got_im - want_im).abs() < 1e-12,
                                "im s={s} x={x} k={k} i={i}: {got_im} vs {want_im}"
                            );
                        }
                    }
                }
            }
            // A halved exchange (the KRKS spelling, `vj - vk * .5`) must NOT
            // match — the no-.5 factor is load-bearing, and hybrid-only.
            let mut halved_gap = 0.0_f64;
            for x in 0..3 {
                for k in 0..nkpts {
                    for i in 0..nao * nao {
                        let halved = vxc[0][x][k].re[i] + vj[0][x][k].re[i] + vj[1][x][k].re[i]
                            - 0.5 * vk[0][x][k].re[i];
                        halved_gap =
                            halved_gap.max((veff[0][x][k].re[i] - halved).abs());
                    }
                }
            }
            assert!(
                halved_gap > 1e-6,
                "halved-exchange variant agrees — no-.5 factor gate vacuous for {xc}"
            );
        } else {
            let vj = grad.j_deriv_sets(&dm).expect("j sets");
            assert_eq!(vj.len(), 2);
            for s in 0..2 {
                for x in 0..3 {
                    for k in 0..nkpts {
                        for i in 0..nao * nao {
                            let want = vxc[s][x][k].re[i] + vj[0][x][k].re[i] + vj[1][x][k].re[i];
                            let got = veff[s][x][k].re[i];
                            assert!(
                                (got - want).abs() < 1e-12,
                                "re s={s} x={x} k={k} i={i}: {got} vs {want}"
                            );
                        }
                    }
                }
            }
        }
        // Single-set densities refuse the spin-threaded veff by name.
        assert!(grad.veff(&dm[..1].to_vec()).is_err());
    }
}

// ---------------------------------------------------------------------------
// Gate C: upstream fingerprints on upstream's cell (closed-shell → same
// three constants as KRKS).
// ---------------------------------------------------------------------------

fn scf_config(cell: &Cell) -> pyscf_pbc_scf::KScfConfig {
    pyscf_pbc_scf::KScfConfig {
        conv_tol: 1e-10,
        conv_tol_grad: Some(1e-8),
        max_cycle: 100,
        ..pyscf_pbc_scf::KScfConfig::for_cell(cell)
    }
}

fn gate_c(xc: &str, exxdiv_none: bool, want_fp: f64, label: &str) {
    use pyscf_pbc_dft::kuks::Kuks;
    let cell = upstream_test_cell();
    let kpts = kpts_113(&cell);
    let cfg = scf_config(&cell);
    // Closed-shell fixture: init_guess_breaksym = 0 starts dm_a == dm_b
    // bit-identically — the exact fixed point recovering the restricted
    // solution the committed fingerprint belongs to (18-06's finding).
    let mut mf = Kuks::new(cell.clone(), &kpts, xc).expect("KUKS builds");
    if exxdiv_none {
        mf.exxdiv = None;
    }
    mf.init_guess_breaksym = 0;
    let result = mf.kernel(&cfg).expect("KUKS SCF");
    assert!(
        result.converged,
        "KUKS/{label} did not converge after {} cycles — the fp number is meaningless",
        result.cycles
    );
    let analytic = KuksGradients::new(&mf, result.mo_energy, result.mo_coeff, result.mo_occ)
        .expect("gradient object")
        .kernel()
        .expect("analytic gradient");
    let fp = fingerprint(&analytic);
    println!("Gate C (KUKS/{label}): lib.fp(g) = {fp:.17}");
    assert!(
        (fp - want_fp).abs() < 5e-7,
        "Gate C FAILED (KUKS/{label}): fp = {fp:.12} vs upstream {want_fp:.12}"
    );
}

#[test]
fn kuks_lda_matches_upstream_fingerprint() {
    gate_c("lda,vwn", false, -0.22166962318360375, "LDA");
}

#[test]
fn kuks_gga_matches_upstream_fingerprint() {
    gate_c("pbe,pbe", false, -0.21844074846755882, "GGA");
}

#[test]
fn kuks_hybrid_matches_upstream_fingerprint() {
    gate_c("b3lyp5", true, -0.19544969829285652, "hybrid");
}

// ---------------------------------------------------------------------------
// Gate B: verify_fd on a genuinely spin-polarised cell.
// ---------------------------------------------------------------------------

/// Open-shell HeH doublet (`spin = 1`, 3 valence electrons under
/// `gth-pade`: He-q2 + H-q1): He at `(0,0,-1)`, H pushed `+0.02` Bohr in x
/// off `(0,0,1)` so the gradient is O(10⁻²) and the gate cannot pass
/// vacuously. Both spin channels are occupied (`nalpha = 2, nbeta = 1`),
/// so the spin-resolved middle term is fully exercised (18-06's fixture).
fn open_shell_cell() -> Cell {
    use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
    use pyscf_pbc_gto::{ALattice, CellBuildArgs};
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("He".into(), [0.0, 0.0, -1.0]),
                ("H".into(), [0.02, 0.0, 1.0]),
            ]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: pyscf_core::Unit::Bohr,
            spin: 1,
            ..Default::default()
        },
        a: ALattice::Matrix([
            [8.0, 0.0, 0.0],
            [0.0, 8.0, 0.0],
            [0.0, 0.0, 8.0],
        ]),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("open-shell HeH cell must build")
}

fn tight_config(cell: &Cell) -> pyscf_pbc_scf::KScfConfig {
    pyscf_pbc_scf::KScfConfig {
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-10),
        max_cycle: 100,
        ..pyscf_pbc_scf::KScfConfig::for_cell(cell)
    }
}

fn kuks_energy(
    cell: &Cell,
    kpts: &[[f64; 3]],
    xc: &str,
    exxdiv_none: bool,
) -> Result<f64, pyscf_core::PyscfRsError> {
    use pyscf_pbc_dft::kuks::Kuks;
    let mut mf = Kuks::new(cell.clone(), kpts, xc).expect("KUKS builds on a displaced cell");
    if exxdiv_none {
        mf.exxdiv = None;
    }
    let cfg = tight_config(cell);
    let result = mf.kernel(&cfg)?;
    assert!(
        result.converged,
        "KUKS did not converge after {} cycles — the FD number is meaningless",
        result.cycles
    );
    Ok(result.e_tot)
}

fn gate_b(xc: &str, exxdiv_none: bool, label: &str) {
    use pyscf_pbc_grad::verify_fd;
    use pyscf_pbc_dft::kuks::Kuks;
    const DISP: f64 = 5e-4;
    const TOL: f64 = 1e-6;

    let cell = open_shell_cell();
    let kpts = vec![[0.0_f64; 3]];
    let analytic = {
        let mut mf = Kuks::new(cell.clone(), &kpts, xc).expect("central KUKS");
        if exxdiv_none {
            mf.exxdiv = None;
        }
        let cfg = tight_config(&cell);
        let result = mf.kernel(&cfg).expect("central SCF");
        assert!(result.converged, "central SCF did not converge");
        assert_eq!(result.dm.len(), 2, "KUKS must carry two spin sets");
        let mut gap = 0.0_f64;
        for (a, b) in result.dm[0].iter().zip(&result.dm[1]) {
            for (p, q) in a.re.iter().zip(&b.re).chain(a.im.iter().zip(&b.im)) {
                gap = gap.max((p - q).abs());
            }
        }
        println!("open-shell channel gap max|dm_a - dm_b| = {gap:.3e}");
        assert!(
            gap > 1e-3,
            "gate is vacuous: channels unpolarised (gap {gap:.3e}) — this would gate the restricted path"
        );
        KuksGradients::new(&mf, result.mo_energy, result.mo_coeff, result.mo_occ)
            .expect("gradient object")
            .kernel()
            .expect("analytic gradient")
    };
    let peak = max_abs(&analytic);
    assert!(
        peak > 1e-4,
        "gate is vacuous: max|analytic| = {peak:e} on the displaced cell"
    );

    let report = verify_fd(&cell, &analytic, |c| kuks_energy(c, &kpts, xc, exxdiv_none), DISP, TOL)
        .expect("finite-difference harness");
    println!(
        "Gate B (KUKS/{label} open-shell HeH): max|fd − analytic| = {:.3e} Ha/Bohr",
        report.max_abs_diff
    );
    assert!(
        report.passed,
        "Gate B FAILED (KUKS/{label}): max|fd − analytic| = {:.3e} > {TOL:e}\nanalytic = {analytic:?}\nfd = {:?}",
        report.max_abs_diff, report.fd_grad,
    );
}

#[test]
fn kuks_lda_passes_verify_fd() {
    gate_b("lda,vwn", false, "LDA");
}

#[test]
fn kuks_gga_passes_verify_fd() {
    gate_b("pbe,pbe", false, "GGA");
}

#[test]
fn kuks_hybrid_passes_verify_fd() {
    gate_b("b3lyp5", true, "hybrid");
}

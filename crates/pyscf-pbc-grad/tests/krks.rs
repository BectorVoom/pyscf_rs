//! Plan 18-07 — `pyscf/pbc/grad/krks.py` (141 l): KRKS k-point gradient.
//!
//! `class Gradients(rhf_grad.Gradients)` (`:119`): 18-05's assembly with
//! `get_veff` replaced by the XC-grid term plus Coulomb/hybrid-exchange.
//!
//! * Unit tests gate Task 1's refusals (`grid_response`, meta-GGA, NLC) and
//!   Task 2's shared accumulators (`d1_dot_add` complex vs naive,
//!   `gga_grad_sum_add` vs an independent `_make_dR_dao_w` reference with
//!   the `wv[0] *= .5` factor load-bearing).
//! * **Gate C** (`krks_matches_upstream_fingerprints`): `lib.fp(g)` against
//!   upstream's committed constants on upstream's own cell (custom
//!   uncontracted `[[0,[1.3,1]],[1,[0.8,1]]` basis, `mesh = [13]*3`,
//!   `kpts = [1,1,3]`, `conv_tol = 1e-10`):
//!   LDA `svwn`/`lda,vwn` → `-0.22166962318360375` (`test_krks.py:54`),
//!   GGA `pbe` → `-0.21844074846755882` (`:67`),
//!   hybrid `b3lyp5` → `-0.19544969829285652` (`:82`, `exxdiv = None`).
//! * **Gate B** (`krks_*_passes_verify_fd`): analytic gradient vs this
//!   port's `verify_fd` at upstream's DFT step (full `1e-3` =
//!   `verify_fd` half-step `5e-4`; **not** the HF `1e-5`) and
//!   `FD_TOL = 1e-6` Ha/Bohr (18-01's measured DFT floor sits at
//!   `4.3e-10`, so `1e-6` is the planned gate, not a loosening).
//!
//! # Geometry is specified in BOHR

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_pbc_grad::{fingerprint, KrksGradients};
use pyscf_pbc_gto::Cell;

/// Deterministic complex filler — no RNG crate, no fixtures to drift.
fn z(i: usize, j: usize, s: usize) -> (f64, f64) {
    (
        ((i * 7 + j * 3 + s * 13) as f64 * 0.37).sin(),
        ((i * 5 + j * 11 + s * 7) as f64 * 0.61).cos(),
    )
}

/// Upstream `test_krks.py:setUpModule`'s cell EXACTLY: custom uncontracted
/// `[[0,[1.3,1]],[1,[0.8,1]]` basis (NOT `gth-szv`, which is contracted),
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
    .expect("upstream test_krks cell must build")
}

fn kpts_113(cell: &Cell) -> Vec<[f64; 3]> {
    let kpts = pyscf_pbc_gto::kpts_mesh::make_kpts_default(cell, [1, 1, 3]).expect("1x1x3 k-mesh");
    assert_eq!(kpts.len(), 3, "upstream test_krks uses 3 k-points");
    kpts
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

fn max_abs(a: &[[f64; 3]]) -> f64 {
    a.iter()
        .flat_map(|r| r.iter())
        .fold(0.0_f64, |m, v| m.max(v.abs()))
}

// ---------------------------------------------------------------------------
// Task 1: the three refusals, written before the bodies.
// ---------------------------------------------------------------------------

/// `krks.py:48-49` refuses `grid_response = True` — the molecular
/// `pyscf-grad::rks` documents it as fully supported, so the periodic
/// subclass MUST override it to a refusal (`grep grid_response` shows the
/// override; inheriting the working term would be a fidelity defect).
#[test]
fn grid_response_is_refused_not_inherited() {
    use pyscf_pbc_dft::krks::Krks;
    let cell = upstream_test_cell();
    let kpts = vec![[0.0_f64; 3]];
    let mf = Krks::new(cell, &kpts, "lda,vwn").expect("KRKS holder");
    let nao = mf.cell().mol.nao_nr;
    let (coeff, energy, occ) = synthetic_orbitals(nao, 1);
    let grad = KrksGradients::new(&mf, energy, coeff, occ)
        .expect("gradient object")
        .with_grid_response(true);
    let err = grad.kernel().expect_err("grid_response = True must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("grid_response") && msg.contains("not yet implemented"),
        "refusal must name grid_response, got: {msg}"
    );
}

/// `krks.py:112` — `raise NotImplementedError("metaGGA")`.
#[test]
fn meta_gga_is_refused_by_name() {
    use pyscf_pbc_dft::krks::Krks;
    let cell = upstream_test_cell();
    let kpts = vec![[0.0_f64; 3]];
    let mf = Krks::new(cell, &kpts, "TPSS").expect("KRKS holder");
    let nao = mf.cell().mol.nao_nr;
    let (coeff, energy, occ) = synthetic_orbitals(nao, 1);
    let grad = KrksGradients::new(&mf, energy, coeff, occ).expect("gradient object");
    let err = grad.kernel().expect_err("meta-GGA must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("metaGGA") && msg.contains("not yet implemented"),
        "refusal must name metaGGA, got: {msg}"
    );
}

/// `krks.py:110` — `raise NotImplementedError("NLC")`.
#[test]
fn nlc_is_refused_by_name() {
    use pyscf_pbc_dft::krks::Krks;
    let cell = upstream_test_cell();
    let kpts = vec![[0.0_f64; 3]];
    let mf = Krks::new(cell, &kpts, "VV10").expect("KRKS holder");
    let nao = mf.cell().mol.nao_nr;
    let (coeff, energy, occ) = synthetic_orbitals(nao, 1);
    let grad = KrksGradients::new(&mf, energy, coeff, occ).expect("gradient object");
    let err = grad.kernel().expect_err("NLC must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("NLC") && msg.contains("not yet implemented"),
        "refusal must name NLC, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Task 2: the shared accumulators — complex vs naive, real-inside.
// ---------------------------------------------------------------------------

/// `d1_dot_add` (complex path) against a naive `Σ_g conj(dao)·aow·wv`
/// triple nest with the OPPOSITE loop order (nu outer, mu inner — a shared
/// loop-order bug cannot hide in both), `ji`-consistent, and bit-identity
/// with the complex-then-`.real` route in the same order (clause 4).
#[test]
fn d1_dot_complex_matches_naive_and_real_inside() {
    use pyscf_grad::rks::d1_dot_add;
    let nao = 2;
    let ngrids = 3;
    let n2 = nao * nao;
    // F-order rows: dao[x][mu*ngrids+g], ao0[nu*ngrids+g].
    let dao_re: Vec<Vec<f64>> = (0..3)
        .map(|x| (0..nao * ngrids).map(|i| z(i, x, 1).0).collect())
        .collect();
    let dao_im: Vec<Vec<f64>> = (0..3)
        .map(|x| (0..nao * ngrids).map(|i| z(i, x, 2).1).collect())
        .collect();
    let ao0_re: Vec<f64> = (0..nao * ngrids).map(|i| z(i, 7, 3).0).collect();
    let ao0_im: Vec<f64> = (0..nao * ngrids).map(|i| z(i, 7, 4).1).collect();
    let wv: Vec<f64> = (0..ngrids).map(|g| 0.5 + 0.1 * g as f64).collect();

    let mut vre = vec![0.0; 3 * n2];
    let mut vim = vec![0.0; 3 * n2];
    d1_dot_add(
        &mut vre,
        &mut vim,
        [&dao_re[0], &dao_re[1], &dao_re[2]],
        Some([&dao_im[0], &dao_im[1], &dao_im[2]]),
        &ao0_re,
        Some(&ao0_im),
        &wv,
        nao,
        ngrids,
    );
    // Naive: nu outermost, mu innermost; full complex product, split planes.
    for x in 0..3 {
        for nu in 0..nao {
            for mu in 0..nao {
                let mut sr = 0.0_f64;
                let mut si = 0.0_f64;
                for g in 0..ngrids {
                    let (dr, di) = (dao_re[x][mu * ngrids + g], dao_im[x][mu * ngrids + g]);
                    let (ar, ai) = (ao0_re[nu * ngrids + g], ao0_im[nu * ngrids + g]);
                    let t1 = wv[g] * dr;
                    let t2 = wv[g] * di;
                    sr += t1 * ar + t2 * ai;
                    si += t1 * ai - t2 * ar;
                }
                // Same association as the primitive, so this is exact.
                assert_eq!(vre[x * n2 + mu + nu * nao], sr, "re {x},{mu},{nu}");
                assert_eq!(vim[x * n2 + mu + nu * nao], si, "im {x},{mu},{nu}");
            }
        }
    }
    // Non-vacuous: every plane carries signal.
    assert!(vre.iter().any(|v| v.abs() > 1e-3));
    assert!(vim.iter().any(|v| v.abs() > 1e-3));
    // Clause 4: complex-then-.real in the SAME (x,mu,nu,g) order is
    // bit-identical to real-inside (same buffer, same oracle_sum call).
    for x in 0..3 {
        for mu in 0..nao {
            for nu in 0..nao {
                let mut tr = Vec::new();
                let mut ti = Vec::new();
                for g in 0..ngrids {
                    let (dr, di) = (dao_re[x][mu * ngrids + g], dao_im[x][mu * ngrids + g]);
                    let (ar, ai) = (ao0_re[nu * ngrids + g], ao0_im[nu * ngrids + g]);
                    tr.push((wv[g] * dr) * ar + (wv[g] * di) * ai);
                    ti.push((wv[g] * dr) * ai - (wv[g] * di) * ar);
                }
                assert_eq!(
                    vre[x * n2 + mu + nu * nao].to_bits(),
                    oracle_sum(&tr).to_bits(),
                    "real-inside is not bit-identical at {x},{mu},{nu}"
                );
                assert_eq!(
                    vim[x * n2 + mu + nu * nao].to_bits(),
                    oracle_sum(&ti).to_bits(),
                    "imag path moved a bit at {x},{mu},{nu}"
                );
            }
        }
    }
}

/// The molecular real path is bit-identical to the pre-extraction inline
/// formula `((w*v)*d)*a` (same association): the extraction moved no number.
#[test]
fn d1_dot_real_path_is_bit_identical_to_inline() {
    use pyscf_grad::rks::d1_dot_add;
    let nao = 2;
    let ngrids = 5;
    let n2 = nao * nao;
    let dao: Vec<Vec<f64>> = (0..3)
        .map(|x| (0..nao * ngrids).map(|i| z(i + 3, x, 5).0).collect())
        .collect();
    let ao0: Vec<f64> = (0..nao * ngrids).map(|i| z(i, 9, 6).0).collect();
    let w: Vec<f64> = (0..ngrids).map(|g| 0.3 + 0.07 * g as f64).collect();
    let v: Vec<f64> = (0..ngrids).map(|g| 0.9 - 0.05 * g as f64).collect();

    let mut vre = vec![0.0; 3 * n2];
    let mut vim = vec![0.0; 3 * n2];
    let wv: Vec<f64> = w.iter().zip(&v).map(|(a, b)| a * b).collect();
    d1_dot_add(
        &mut vre,
        &mut vim,
        [&dao[0], &dao[1], &dao[2]],
        None,
        &ao0,
        None,
        &wv,
        nao,
        ngrids,
    );
    for x in 0..3 {
        for mu in 0..nao {
            for nu in 0..nao {
                // The old inline body, verbatim.
                let mut terms = Vec::with_capacity(ngrids);
                for g in 0..ngrids {
                    terms.push(
                        w[g] * v[g] * dao[x][mu * ngrids + g] * ao0[nu * ngrids + g],
                    );
                }
                assert_eq!(
                    vre[x * n2 + mu + nu * nao].to_bits(),
                    oracle_sum(&terms).to_bits(),
                    "extraction moved a bit at {x},{mu},{nu}"
                );
            }
        }
    }
    assert!(vim.iter().all(|v| *v == 0.0));
}

/// `gga_grad_sum_add` against an independent reference that reimplements
/// upstream's `_scale_ao` + `_make_dR_dao_w` mapping directly (no shared
/// code with the primitive), with the `wv[0] *= .5` applied — plus the
/// negative control (no `.5`) that MUST differ, proving the factor is
/// load-bearing and not vacuous.
#[test]
fn gga_grad_sum_matches_independent_reference() {
    use pyscf_grad::rks::gga_grad_sum_add;
    let nao = 2;
    let ngrids = 3;
    let n2 = nao * nao;
    let ao_re: Vec<Vec<f64>> = (0..10)
        .map(|c| (0..nao * ngrids).map(|i| z(i, c, 11).0).collect())
        .collect();
    let ao_im: Vec<Vec<f64>> = (0..10)
        .map(|c| (0..nao * ngrids).map(|i| z(i + 1, c, 12).1).collect())
        .collect();
    let wv: Vec<Vec<f64>> = (0..4)
        .map(|c| (0..ngrids).map(|g| 0.4 + 0.03 * (c * ngrids + g) as f64).collect())
        .collect();

    let refs = |half: bool| -> (Vec<f64>, Vec<f64>) {
        // aow[nu,g] = Σ_c ao[c]·wv[c], row 0 halved iff `half`.
        let mut aow_r = vec![0.0; nao * ngrids];
        let mut aow_i = vec![0.0; nao * ngrids];
        for nu in 0..nao {
            for g in 0..ngrids {
                let mut sr = 0.0_f64;
                let mut si = 0.0_f64;
                for c in 0..4 {
                    let w = if half && c == 0 { 0.5 * wv[c][g] } else { wv[c][g] };
                    sr += ao_re[c][nu * ngrids + g] * w;
                    si += ao_im[c][nu * ngrids + g] * w;
                }
                aow_r[nu * ngrids + g] = sr;
                aow_i[nu * ngrids + g] = si;
            }
        }
        // _make_dR_dao_w mapping (rks.py:199-214).
        const HESS: [[usize; 3]; 3] = [[4, 5, 6], [5, 7, 8], [6, 8, 9]];
        let mut vre = vec![0.0; 3 * n2];
        let mut vim = vec![0.0; 3 * n2];
        for x in 0..3 {
            for mu in 0..nao {
                for nu in 0..nao {
                    let mut sr = 0.0_f64;
                    let mut si = 0.0_f64;
                    for g in 0..ngrids {
                        // Part 1: dao[x] against aow.
                        let (dr, di) = (ao_re[1 + x][mu * ngrids + g], ao_im[1 + x][mu * ngrids + g]);
                        let (ar, ai) = (aow_r[nu * ngrids + g], aow_i[nu * ngrids + g]);
                        sr += dr * ar + di * ai;
                        si += dr * ai - di * ar;
                        // Part 2: aow2[x] against ao0.
                        let rows = [1 + x, HESS[x][0], HESS[x][1], HESS[x][2]];
                        let mut br = 0.0_f64;
                        let mut bi = 0.0_f64;
                        for (c, &row) in rows.iter().enumerate() {
                            let w = if half && c == 0 { 0.5 * wv[c][g] } else { wv[c][g] };
                            br += ao_re[row][mu * ngrids + g] * w;
                            bi += ao_im[row][mu * ngrids + g] * w;
                        }
                        let (cr, ci) = (ao_re[0][nu * ngrids + g], ao_im[0][nu * ngrids + g]);
                        sr += br * cr + bi * ci;
                        si += br * ci - bi * cr;
                    }
                    vre[x * n2 + mu + nu * nao] = sr;
                    vim[x * n2 + mu + nu * nao] = si;
                }
            }
        }
        (vre, vim)
    };

    let mut vre = vec![0.0; 3 * n2];
    let mut vim = vec![0.0; 3 * n2];
    let ao_r: [&[f64]; 10] = std::array::from_fn(|c| ao_re[c].as_slice());
    let ao_i: [&[f64]; 10] = std::array::from_fn(|c| ao_im[c].as_slice());
    let wv_r: [&[f64]; 4] = std::array::from_fn(|c| wv[c].as_slice());
    gga_grad_sum_add(&mut vre, &mut vim, ao_r, Some(ao_i), wv_r, nao, ngrids);

    let (want_r, want_i) = refs(true);
    for i in 0..3 * n2 {
        assert!(
            (vre[i] - want_r[i]).abs() < 1e-12,
            "re[{i}]: {} vs independent {}",
            vre[i],
            want_r[i]
        );
        assert!(
            (vim[i] - want_i[i]).abs() < 1e-12,
            "im[{i}]: {} vs independent {}",
            vim[i],
            want_i[i]
        );
    }
    // Negative control: without the .5 the answer MUST move (the factor is
    // load-bearing, and GGA-only — LDA never sees it).
    let (nohalf_r, _) = refs(false);
    let gap = vre
        .iter()
        .zip(&nohalf_r)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        gap > 1e-3,
        "dropping wv[0] *= .5 did not move the GGA sum — factor gate vacuous"
    );
}

/// `xc = 'HF'` is the zeros grid term (`krks.py:106-107` `pass`), so the
/// KRKS/HF kernel on synthetic orbitals must equal the KRHF kernel on the
/// same orbitals (both are `vj - vk·.5` plus the identical assembly) to
/// 1e-12 — the two veff spellings differ only in last-bit association.
#[test]
fn hf_xc_recovers_the_krhf_number() {
    use pyscf_pbc_dft::krks::Krks;
    use pyscf_pbc_scf::Krhf;
    let cell = upstream_test_cell();
    let kpts = kpts_113(&cell);
    let nkpts = kpts.len();
    let mf_hf = Krks::new(cell.clone(), &kpts, "HF").expect("KRKS/HF holder");
    let mf_r = Krhf::new(cell.clone(), &kpts).expect("KRHF holder");
    let nao = cell.mol.nao_nr;
    let (coeff, energy, occ) = synthetic_orbitals(nao, nkpts);
    let g_hf = KrksGradients::new(&mf_hf, energy.clone(), coeff.clone(), occ.clone())
        .expect("KRKS/HF gradient")
        .kernel()
        .expect("KRKS/HF kernel");
    let g_r = pyscf_pbc_grad::KrhfGradients::new(&mf_r, energy, coeff, occ)
        .expect("KRHF gradient")
        .kernel()
        .expect("KRHF kernel");
    for (ia, (a, b)) in g_hf.iter().zip(&g_r).enumerate() {
        for x in 0..3 {
            assert!(
                (a[x] - b[x]).abs() < 1e-12,
                "atom {ia} x={x}: KRKS/HF {} vs KRHF {}",
                a[x],
                b[x]
            );
        }
    }
    assert!(max_abs(&g_hf) > 1e-6, "HF-equivalence test is vacuous");
}

// ---------------------------------------------------------------------------
// Gate C: upstream fingerprints on upstream's cell.
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
    use pyscf_pbc_dft::krks::Krks;
    let cell = upstream_test_cell();
    let kpts = kpts_113(&cell);
    let cfg = scf_config(&cell);
    let mut mf = Krks::new(cell.clone(), &kpts, xc).expect("KRKS builds");
    if exxdiv_none {
        mf.exxdiv = None;
    }
    let result = mf.kernel(&cfg).expect("KRKS SCF");
    assert!(
        result.converged,
        "KRKS/{label} did not converge after {} cycles — the fp number is meaningless",
        result.cycles
    );
    let analytic = KrksGradients::new(&mf, result.mo_energy, result.mo_coeff, result.mo_occ)
        .expect("gradient object")
        .kernel()
        .expect("analytic gradient");
    let fp = fingerprint(&analytic);
    println!("Gate C (KRKS/{label}): lib.fp(g) = {fp:.17}");
    assert!(
        (fp - want_fp).abs() < 5e-7,
        "Gate C FAILED (KRKS/{label}): fp = {fp:.12} vs upstream {want_fp:.12}"
    );
}

#[test]
fn krks_lda_matches_upstream_fingerprint() {
    gate_c("lda,vwn", false, -0.22166962318360375, "LDA");
}

#[test]
fn krks_gga_matches_upstream_fingerprint() {
    gate_c("pbe,pbe", false, -0.21844074846755882, "GGA");
}

#[test]
fn krks_hybrid_matches_upstream_fingerprint() {
    gate_c("b3lyp5", true, -0.19544969829285652, "hybrid");
}

// ---------------------------------------------------------------------------
// Gate B: verify_fd at upstream's DFT step (full 1e-3 = half-step 5e-4).
// ---------------------------------------------------------------------------

/// Closed-shell H2 (`spin = 0`, `gth-pade` — the gradient `get_hcore`
/// refuses all-electron cells by name, so the fixture carries a PP): H at
/// `(0,0,-0.8)`, second H pushed `+0.02` Bohr in x off `(0,0,0.8)` so the
/// gradient is O(10⁻²) and the gate cannot pass vacuously.
fn h2_cell() -> Cell {
    use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
    use pyscf_pbc_gto::{ALattice, CellBuildArgs};
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("H".into(), [0.0, 0.0, -0.8]),
                ("H".into(), [0.02, 0.0, 0.8]),
            ]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: pyscf_core::Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 8.0]]),
        pseudo: Some("gth-pade".into()),
        // Pinned small FFT mesh: the precision-derived default for an 8-Bohr
        // box is far finer than H/gth-szv needs; FD stays self-consistent.
        mesh: Some([15, 15, 15]),
        ..Default::default()
    })
    .expect("H2 cell must build")
}

fn tight_config(cell: &Cell) -> pyscf_pbc_scf::KScfConfig {
    pyscf_pbc_scf::KScfConfig {
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-10),
        max_cycle: 100,
        ..pyscf_pbc_scf::KScfConfig::for_cell(cell)
    }
}

fn krks_energy(cell: &Cell, kpts: &[[f64; 3]], xc: &str, exxdiv_none: bool) -> Result<f64, pyscf_core::PyscfRsError> {
    use pyscf_pbc_dft::krks::Krks;
    let mut mf = Krks::new(cell.clone(), kpts, xc).expect("KRKS builds on a displaced cell");
    if exxdiv_none {
        mf.exxdiv = None;
    }
    let cfg = tight_config(cell);
    let result = mf.kernel(&cfg)?;
    assert!(
        result.converged,
        "KRKS did not converge after {} cycles — the FD number is meaningless",
        result.cycles
    );
    Ok(result.e_tot)
}

fn gate_b(xc: &str, exxdiv_none: bool, label: &str) {
    use pyscf_pbc_grad::verify_fd;
    use pyscf_pbc_dft::krks::Krks;
    const DISP: f64 = 5e-4;
    const TOL: f64 = 1e-6;

    let cell = h2_cell();
    let kpts = vec![[0.0_f64; 3]];
    let analytic = {
        let mut mf = Krks::new(cell.clone(), &kpts, xc).expect("central KRKS");
        if exxdiv_none {
            mf.exxdiv = None;
        }
        let cfg = tight_config(&cell);
        let result = mf.kernel(&cfg).expect("central SCF");
        assert!(result.converged, "central SCF did not converge");
        KrksGradients::new(&mf, result.mo_energy, result.mo_coeff, result.mo_occ)
            .expect("gradient object")
            .kernel()
            .expect("analytic gradient")
    };
    let peak = max_abs(&analytic);
    assert!(
        peak > 1e-4,
        "gate is vacuous: max|analytic| = {peak:e} on the displaced cell"
    );

    let report = verify_fd(&cell, &analytic, |c| krks_energy(c, &kpts, xc, exxdiv_none), DISP, TOL)
        .expect("finite-difference harness");
    println!(
        "Gate B (KRKS/{label} H2): max|fd − analytic| = {:.3e} Ha/Bohr",
        report.max_abs_diff
    );
    assert!(
        report.passed,
        "Gate B FAILED (KRKS/{label}): max|fd − analytic| = {:.3e} > {TOL:e}\nanalytic = {analytic:?}\nfd = {:?}",
        report.max_abs_diff, report.fd_grad,
    );
}

#[test]
fn krks_lda_passes_verify_fd() {
    gate_b("lda,vwn", false, "LDA");
}

#[test]
fn krks_gga_passes_verify_fd() {
    gate_b("pbe,pbe", false, "GGA");
}

#[test]
fn krks_hybrid_passes_verify_fd() {
    gate_b("b3lyp5", true, "hybrid");
}

//! Plan 18-05 component gates — `pbc/grad/krhf.py` assembly, no SCF.
//!
//! Every test here runs without converging a mean field: synthetic orbitals
//! and densities exercise the seams, while Gate B (`krhf_verify_fd.rs`)
//! gates the assembled gradient end to end.
//!
//! # Geometry is specified in BOHR
//!
//! Reference cells are the Rust-exact §9.2 systems (`test_systems`).

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_pbc_grad::Gradients;
use pyscf_pbc_grad::krhf::contractions;
use pyscf_pbc_grad::{
    HcoreFusedStats, KrhfGradients, assemble_hcore_deriv, fused_local_contraction,
    hcore_deriv_matrices, make_rdm1e_kpts, precompute_hcore,
};
use pyscf_pbc_gto::{Cell, test_systems};

fn kpts_112(cell: &Cell) -> Vec<[f64; 3]> {
    pyscf_pbc_gto::kpts_mesh::make_kpts_default(cell, [1, 1, 2]).expect("1x1x2 k-mesh")
}

/// Deterministic complex filler — no RNG crate, no fixtures to drift.
fn z(i: usize, j: usize, s: usize) -> (f64, f64) {
    (
        ((i * 7 + j * 3 + s * 13) as f64 * 0.37).sin(),
        ((i * 5 + j * 11 + s * 7) as f64 * 0.61).cos(),
    )
}

/// Seeded Hermitian positive-definite density per k, row-major.
fn hermitian_dm(nao: usize, nkpts: usize) -> Vec<CTensor> {
    (0..nkpts)
        .map(|k| {
            let mut re = vec![0.0_f64; nao * nao];
            let mut im = vec![0.0_f64; nao * nao];
            for i in 0..nao {
                for j in 0..nao {
                    let mut pr = 0.0_f64;
                    let mut pi = 0.0_f64;
                    for m in 0..nao {
                        let (ar, ai) = z(i, m, k);
                        let (br, bi) = z(j, m, k);
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

fn max_abs_diff(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter()
        .zip(b.iter())
        .flat_map(|(r, s)| r.iter().zip(s.iter()).map(|(x, y)| (x - y).abs()))
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Task 1: the all-electron refusal
// ---------------------------------------------------------------------------

/// `krhf.py:111` is `else: raise NotImplementedError`. An all-electron
/// periodic cell refuses by NAME — it never falls back to `int1e_ipnuc`
/// (the nuclear-attraction derivative, a different operator).
#[test]
fn all_electron_get_hcore_is_refused_by_name() {
    use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
    let h = 2.834589;
    let cell = Cell::build(pyscf_pbc_gto::CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("sto-3g".into()),
            unit: pyscf_core::Unit::Bohr,
            ..Default::default()
        },
        a: pyscf_pbc_gto::ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        ..Default::default()
    })
    .expect("all-electron He cell must build");
    assert!(cell.atom_pseudo(0).is_none());
    let err = pyscf_pbc_grad::get_hcore(&cell, &[[0.0; 3]]).expect_err("all-electron must refuse");
    let msg = err.to_string();
    assert!(
        msg.contains("not yet implemented") && msg.contains("int1e_ipnuc"),
        "refusal must name itself and disavow the fallback, got: {msg}"
    );
}

// ---------------------------------------------------------------------------
// Task 2: hcore_generator index order — the conjugate transpose
// ---------------------------------------------------------------------------

/// `krhf.py:144-145` with a deliberately NON-HERMITIAN `h1`: the second
/// subtraction is `conj(h[j,i])`, not `h[j,i]`. Removing the `.conj()` gives
/// `-3-4i` where `-3+4i` is required below — the test fails rather than
/// cancels (the 14-05 `decompose_j2c` shape, gated before it can ship).
#[test]
fn assemble_applies_the_conjugate_transpose() {
    let nao = 2;
    let nkpts = 1;
    // h[x][0] = [[1+2i, 3+4i], [5+6i, 7+8i]] — b != conj(c) on purpose.
    let h1: [Vec<CTensor>; 3] = std::array::from_fn(|x| {
        let s = 1.0 + x as f64;
        vec![CTensor::from_planes(
            vec![1.0 * s, 3.0 * s, 5.0 * s, 7.0 * s],
            vec![2.0 * s, 4.0 * s, 6.0 * s, 8.0 * s],
        )]
    });
    let zero: [Vec<CTensor>; 3] = std::array::from_fn(|_| vec![CTensor::zeros(nao * nao); nkpts]);
    // Atom 0 owns AO 0.
    let got = assemble_hcore_deriv(&zero, &h1, 0, 1, nao, nkpts);
    for x in 0..3 {
        let s = 1.0 + x as f64;
        // Rows (i = 0): −h[0,j]. Cols (j = 0): −conj(h[0,i]).
        // [0,0] = −(1+2i) − (1−2i) = −2. [0,1] = −(3+4i).
        // [1,0] = −conj(3+4i) = −3+4i. [1,1] untouched.
        let want_re = vec![-2.0 * s, -3.0 * s, -3.0 * s, 0.0];
        let want_im = vec![0.0, -4.0 * s, 4.0 * s, 0.0];
        assert_eq!(got[x][0].re, want_re, "component {x} real part");
        assert_eq!(got[x][0].im, want_im, "component {x} imag part");
    }
}

// ---------------------------------------------------------------------------
// Task 4: ji-indexed, real-inside contractions
// ---------------------------------------------------------------------------

/// The DM is indexed `ji` on all three terms (trap 9): a non-symmetric DM
/// must reproduce a naive `einsum('..ij,ji->..')`, and transposing the DM
/// must MOVE the answer (proving the test is sensitive to the order).
/// The `.real` is taken inside each contraction (clause 4): bit-identity
/// with the complex-then-`.real` route on the same DM.
#[test]
fn contractions_use_ji_index_with_real_inside() {
    let nao = 2;
    let nkpts = 1;
    let dm = vec![CTensor::from_planes(
        vec![1.0, 2.0, 4.0, 8.0],
        vec![0.5, -1.5, 2.5, -0.25],
    )];
    let dm_t = vec![CTensor::from_planes(
        vec![1.0, 4.0, 2.0, 8.0],
        vec![0.5, 2.5, -1.5, -0.25],
    )];
    let mat = |s: usize| {
        (0..3)
            .map(|x| {
                (0..nkpts)
                    .map(|_| {
                        let mut re = vec![0.0; nao * nao];
                        let mut im = vec![0.0; nao * nao];
                        for i in 0..nao {
                            for j in 0..nao {
                                let (r, v) = z(i + x, j, s);
                                re[i * nao + j] = r;
                                im[i * nao + j] = v;
                            }
                        }
                        CTensor::from_planes(re, im)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .try_into()
            .unwrap()
    };
    let h: [Vec<CTensor>; 3] = mat(3);
    let v: [Vec<CTensor>; 3] = mat(5);
    let s1: [Vec<CTensor>; 3] = mat(7);

    // Naive references with the OPPOSITE loop nesting (j outer, i inner) so
    // a shared loop-order bug cannot hide in both. Rows i∈A (upstream slices
    // the second axis: `M[:,:,p0:p1]` is rows, not columns).
    let naive = |m: &[Vec<CTensor>; 3], d: &[CTensor], p0: usize, p1: usize, scale: f64| {
        std::array::from_fn::<f64, 3, _>(|x| {
            let mut acc = 0.0_f64;
            for k in 0..nkpts {
                for j in 0..nao {
                    for i in p0..p1 {
                        let (ar, ai) = (m[x][k].re[i * nao + j], m[x][k].im[i * nao + j]);
                        let (br, bi) = (d[k].re[j * nao + i], d[k].im[j * nao + i]);
                        acc += scale * (ar * br - ai * bi);
                    }
                }
            }
            acc
        })
    };
    // h-term reference contracts the FULL matrix (rows + cols enter through
    // assemble, tested above); here the helpers under test are the atom
    // slices, so the naive reference covers vhf/s1 (sliced) shapes.
    for (got, m, scale, name) in [
        (
            contractions::contract_vhf_atom(&v, &dm, 0, 1, nao),
            &v,
            2.0,
            "vhf",
        ),
        (
            contractions::contract_ovlp_atom(&s1, &dm, 0, 1, nao),
            &s1,
            -2.0,
            "ovlp",
        ),
    ] {
        let want = naive(m, &dm, 0, 1, scale);
        for x in 0..3 {
            assert!(
                (got[x] - want[x]).abs() < 1e-12,
                "{name}[{x}]: {} vs naive {}",
                got[x],
                want[x]
            );
        }
        // Sensitivity: the transposed DM must move every component.
        let moved = if name == "vhf" {
            contractions::contract_vhf_atom(&v, &dm_t, 0, 1, nao)
        } else {
            contractions::contract_ovlp_atom(&s1, &dm_t, 0, 1, nao)
        };
        for x in 0..3 {
            assert!(
                (got[x] - moved[x]).abs() > 1e-3,
                "{name}[{x}]: transposed DM did not move the answer — index gate vacuous"
            );
        }
        // Clause 4: complex-then-.real in the SAME (x,k,i,j) order is
        // bit-identical to real-inside (same buffer, same oracle_sum call —
        // the complex accumulator does not exist).
        let complex_route = std::array::from_fn::<f64, 3, _>(|x| {
            let mut tr = Vec::new();
            let mut ti = Vec::new();
            for k in 0..nkpts {
                for i in 0..1 {
                    for j in 0..nao {
                        let (ar, ai) = (m[x][k].re[i * nao + j], m[x][k].im[i * nao + j]);
                        let (br, bi) = (dm[k].re[j * nao + i], dm[k].im[j * nao + i]);
                        tr.push(scale * (ar * br - ai * bi));
                        ti.push(scale * (ar * bi + ai * br));
                    }
                }
            }
            let _ = oracle_sum(&ti);
            oracle_sum(&tr)
        });
        for x in 0..3 {
            assert_eq!(
                got[x].to_bits(),
                complex_route[x].to_bits(),
                "{name}[{x}]: real-inside is not bit-identical to complex-then-.real"
            );
        }
    }
    // h-slice helper against its own naive form (rows + conj-transpose cols).
    let got_h = contractions::contract_h1_atom(&h, &dm, 0, 1, nao);
    let want_h = std::array::from_fn::<f64, 3, _>(|x| {
        let mut acc = 0.0_f64;
        for k in 0..nkpts {
            for j in 0..nao {
                let (ar, ai) = (h[x][k].re[j], h[x][k].im[j]);
                let (br, bi) = (dm[k].re[j * nao], dm[k].im[j * nao]);
                acc -= ar * br - ai * bi;
            }
            for i in 0..nao {
                let (ar, ai) = (h[x][k].re[i], -h[x][k].im[i]);
                let (br, bi) = (dm[k].re[i], dm[k].im[i]);
                acc -= ar * br - ai * bi;
            }
        }
        acc
    });
    for x in 0..3 {
        assert!((got_h[x] - want_h[x]).abs() < 1e-12, "h1[{x}] mismatch");
    }
}

// ---------------------------------------------------------------------------
// make_rdm1e
// ---------------------------------------------------------------------------

/// `dme[k,i,j] = Σ_m C[i,m]·e[m]·n[m]·conj(C[j,m])` (`rhf.py:185-189`) with
/// column-major `C`, against a naive loop nest; zero-occupation columns
/// contribute exactly nothing.
#[test]
fn make_rdm1e_matches_molgrad_formula() {
    let nao = 2;
    let nkpts = 2;
    let (coeff, energy, occ) = synthetic_orbitals(nao, nkpts);
    let got = make_rdm1e_kpts(&coeff, &energy, &occ, nao).expect("rdm1e");
    assert_eq!(got.len(), nkpts);
    for k in 0..nkpts {
        for i in 0..nao {
            for j in 0..nao {
                let mut wr = 0.0_f64;
                let mut wi = 0.0_f64;
                for m in 0..nao {
                    let w = energy[k][m] * occ[k][m];
                    let (ar, ai) = (coeff[k].re[i + m * nao], coeff[k].im[i + m * nao]);
                    let (br, bi) = (coeff[k].re[j + m * nao], -coeff[k].im[j + m * nao]);
                    wr += w * (ar * br - ai * bi);
                    wi += w * (ar * bi + ai * br);
                }
                assert!(
                    (got[k].re[i * nao + j] - wr).abs() < 1e-12,
                    "re {k},{i},{j}"
                );
                assert!(
                    (got[k].im[i * nao + j] - wi).abs() < 1e-12,
                    "im {k},{i},{j}"
                );
            }
        }
    }
    // Shape rejection.
    assert!(make_rdm1e_kpts(&coeff[..1], &energy, &occ, nao).is_err());
}

// ---------------------------------------------------------------------------
// Task 3: fused vs literal + AO-count gates
// ---------------------------------------------------------------------------

/// G-space and real-space are different summations: gate at
/// `measurements/parseval.md`'s measured RELATIVE residual (worst
/// non-degenerate cell 1.5e-12 → gate 1e-11, ~7× headroom), never
/// bit-identity. The comparison is unweighted on both sides — upstream's
/// matrix route carries no quadrature weight (`krhf.py:141`) and is
/// FD-correct, so neither side takes one. AO evaluations per fused call are
/// `nkpts`, not `natm·nkpts` (natm = 2 here, so equality with `nkpts` proves
/// it), and the `ngrids·nao²` density builds likewise do not scale with
/// `natm`.
#[test]
fn fused_local_matches_literal_matrix() {
    let cell = test_systems::diamond();
    let kpts = kpts_112(&cell);
    let nkpts = kpts.len();
    assert_eq!(nkpts, 2);
    let nao = cell.mol.nao_nr;
    let dm0 = hermitian_dm(nao, nkpts);
    let tables = precompute_hcore(&cell, &kpts).expect("h1 tables");
    let mut stats = HcoreFusedStats::default();
    let fused = fused_local_contraction(&tables, &cell, &kpts, &dm0, &mut stats)
        .expect("fused contraction");
    assert_eq!(
        stats,
        HcoreFusedStats {
            value_ao_evals: nkpts,
            density_builds: nkpts
        },
        "fused AO/density work must be nkpts, not natm·nkpts"
    );
    let slices = pyscf_gto::aoslice_by_atom(&cell.mol).expect("aoslices");
    let mut worst_rel = 0.0_f64;
    for ia in 0..cell.natm {
        let hmat = hcore_deriv_matrices(&tables, &cell, &kpts, ia).expect("literal matrix");
        let literal = contractions::contract_hcore_matrix(&hmat, &dm0, nao);
        // The literal matrix holds local + kinetic; the fused route only the
        // local — subtract the kinetic rows/cols half on both sides via the
        // shared helper so the comparison isolates the local part.
        let (_, _, p0, p1) = slices[ia];
        let kinetic = contractions::contract_h1_atom(&tables.h1, &dm0, p0, p1, nao);
        for x in 0..3 {
            let a = fused[ia][x];
            let b = literal[x] - kinetic[x];
            // Relative gate with an absolute floor for degenerate
            // (symmetry-cancelled) rows — the he_fcc lesson in parseval.md.
            let denom = a.abs().max(b.abs()).max(1.0);
            let rel = (a - b).abs() / denom;
            worst_rel = worst_rel.max(rel);
            assert!(
                rel < 1e-11,
                "atom {ia} x={x}: fused {a:e} vs literal-local {b:e} (rel {rel:e})"
            );
        }
    }
    println!("fused-vs-literal worst relative residual (diamond 1x1x2): {worst_rel:e}");
}

// ---------------------------------------------------------------------------
// get_ovlp layout + sign
// ---------------------------------------------------------------------------

/// `-int1e_ipovlp` in `[x][k]` row-major order (the `kxij` vs `xkij` note:
/// upstream's `s1` is k-first; the port stores x-first, and the contraction
/// helpers consume that layout).
#[test]
fn get_ovlp_matches_negated_ipovlp() {
    use pyscf_pbc_scf::Krhf;
    let cell = test_systems::diamond();
    let kpts = kpts_112(&cell);
    let mf = Krhf::new(cell.clone(), &kpts).expect("KRHF holder");
    let nao = cell.mol.nao_nr;
    let (coeff, energy, occ) = synthetic_orbitals(nao, kpts.len());
    let grad = KrhfGradients::new(&mf, energy, coeff, occ).expect("gradients");
    let got = grad.get_ovlp().expect("get_ovlp");
    let raw = pyscf_pbc_gto::pbc_intor(&cell, "int1e_ipovlp", &kpts, Default::default())
        .expect("raw ipovlp");
    assert_eq!(got.len(), 3);
    let mut nonzero = false;
    for x in 0..3 {
        assert_eq!(got[x].len(), kpts.len());
        for k in 0..kpts.len() {
            assert_eq!(got[x][k].re.len(), nao * nao);
            for i in 0..nao {
                for j in 0..nao {
                    let (r, v) = raw.element(k, x, i, j);
                    nonzero |= r.abs() + v.abs() > 1e-8;
                    assert_eq!(got[x][k].re[i * nao + j], -r, "re {x},{k},{i},{j}");
                    assert_eq!(got[x][k].im[i * nao + j], -v, "im {x},{k},{i},{j}");
                }
            }
        }
    }
    assert!(nonzero, "sign test must not be vacuous");
}

// ---------------------------------------------------------------------------
// vppnl per-k density (18-05 extension of the 18-03 seam)
// ---------------------------------------------------------------------------

/// On a shared real density the per-k entry point is BIT-IDENTICAL to the
/// 18-03 upstream-gated one (zero imaginary weights preserve every product
/// bit) — the refactor moved no number.
#[test]
fn vppnl_kdm_matches_shared_dm_path_bitwise() {
    let cell = test_systems::diamond();
    let kpts = kpts_112(&cell);
    let nao = cell.mol.nao_nr;
    let mut flat = vec![0.0_f64; nao * nao];
    for q in 0..nao {
        for p in 0..nao {
            flat[q + p * nao] = 1.0 / (1 + p + q) as f64;
        }
    }
    let want = pyscf_pbc_gto::pseudo::vppnl_nuc_grad(&cell, &flat, &kpts).expect("shared path");
    let mut re = vec![0.0_f64; nao * nao];
    for q in 0..nao {
        for p in 0..nao {
            re[q * nao + p] = flat[q + p * nao];
        }
    }
    let dm = vec![CTensor::from_planes(re.clone(), vec![0.0; nao * nao]); kpts.len()];
    let got = pyscf_pbc_gto::pseudo::vppnl_nuc_grad_kdm(&cell, &dm, &kpts).expect("per-k path");
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(&want) {
        assert_eq!(
            g.map(|v| v.to_bits()),
            w.map(|v| v.to_bits()),
            "per-k path moved a bit: {g:?} vs {w:?}"
        );
    }
}

/// A genuinely complex k-dependent density still carries no net force
/// (translation invariance survives the Hermitian sum).
#[test]
fn vppnl_kdm_complex_dm_has_no_net_force() {
    let cell = test_systems::diamond();
    let kpts = kpts_112(&cell);
    let dm = hermitian_dm(cell.mol.nao_nr, kpts.len());
    let got = pyscf_pbc_gto::pseudo::vppnl_nuc_grad_kdm(&cell, &dm, &kpts).expect("kdm");
    let mut total = [0.0_f64; 3];
    for row in &got {
        for x in 0..3 {
            total[x] += row[x];
        }
    }
    let worst = total.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(worst < 1e-9, "net force {total:?} exceeds 1e-9");
}

// ---------------------------------------------------------------------------
// Assembly: grad_elec vs manual seam assembly (trap 3 scopes)
// ---------------------------------------------------------------------------

/// `electronic_gradient()` against the same seams assembled by hand:
/// matrix-contracted hcore + `2·vhf − 2·s1`, `/nkpts` inside the atom loop,
/// `vppnl/nkpts` over the whole array after it. Catches a missing or doubled
/// division; (`extra_force ≡ 0` here makes its between-placement vacuous —
/// 18-08's DFT+U override gates it.)
#[test]
fn grad_elec_matches_manual_seam_assembly() {
    use pyscf_pbc_scf::Krhf;
    let cell = test_systems::diamond();
    let kpts = kpts_112(&cell);
    let nkpts = kpts.len() as f64;
    let nao = cell.mol.nao_nr;
    let mf = Krhf::new(cell.clone(), &kpts).expect("KRHF holder");
    let (coeff, energy, occ) = synthetic_orbitals(nao, kpts.len());
    let grad = KrhfGradients::new(&mf, energy, coeff, occ).expect("gradients");
    let got = grad.grad_elec().expect("grad_elec");

    // Independent density handles: dme0 through the trait seam, dm0 through
    // the SCF density builder directly (not through the gradient struct).
    let dme = grad.make_rdm1e().expect("dme0")[0].clone();
    // Rebuild dm0 through the public density seam for independence.
    let (coeff2, _, occ2) = synthetic_orbitals(nao, kpts.len());
    let dm0 = pyscf_pbc_scf::krdm::make_rdm1(&coeff2, &occ2, nao);
    let s1 = grad.get_ovlp().expect("s1");
    let vhf = grad.veff(&vec![dm0.clone()]).expect("vhf");
    let vppnl = pyscf_pbc_gto::pseudo::vppnl_nuc_grad_kdm(&cell, &dm0, &kpts).expect("vppnl");
    let slices = pyscf_gto::aoslice_by_atom(&cell.mol).expect("aoslices");
    let mut want = Vec::new();
    for ia in 0..cell.natm {
        let (_, _, p0, p1) = slices[ia];
        let hmat = grad.hcore_generator(ia).expect("hcore matrix");
        let hh = contractions::contract_hcore_matrix(&hmat, &dm0, nao);
        let vv = contractions::contract_vhf_atom(&vhf, &dm0, p0, p1, nao);
        let ss = contractions::contract_ovlp_atom(&s1, &dme, p0, p1, nao);
        let mut row = [0.0_f64; 3];
        for x in 0..3 {
            row[x] = oracle_sum(&[hh[x], vv[x], ss[x]]) / nkpts;
            row[x] = oracle_sum(&[row[x], vppnl[ia][x] / nkpts]);
        }
        want.push(row);
    }
    assert_eq!(got.len(), want.len());
    for (ia, (g, w)) in got.iter().zip(&want).enumerate() {
        for x in 0..3 {
            // Fused-vs-literal local residual (~1e-13) is the only allowed gap.
            assert!(
                (g[x] - w[x]).abs() < 1e-9,
                "atom {ia} x={x}: {} vs manual {}",
                g[x],
                w[x]
            );
        }
    }
    // Non-vacuous: the synthetic density must drive a nonzero gradient.
    assert!(
        max_abs_diff(&got, &vec![[0.0; 3]; got.len()]) > 1e-6,
        "assembly test is vacuous"
    );
}

// ---------------------------------------------------------------------------
// Constructor validation
// ---------------------------------------------------------------------------

#[test]
fn constructor_rejects_shape_mismatches() {
    use pyscf_pbc_scf::Krhf;
    let cell = test_systems::diamond();
    let kpts = kpts_112(&cell);
    let mf = Krhf::new(cell.clone(), &kpts).expect("KRHF holder");
    let nao = cell.mol.nao_nr;
    let (coeff, energy, occ) = synthetic_orbitals(nao, kpts.len());
    assert!(KrhfGradients::new(&mf, energy.clone(), coeff.clone(), occ.clone()).is_ok());
    assert!(KrhfGradients::new(&mf, energy[..1].to_vec(), coeff.clone(), occ.clone()).is_err());
    let mut bad = coeff.clone();
    bad[0].re[0] = f64::NAN;
    assert!(KrhfGradients::new(&mf, energy, bad, occ).is_err());
}

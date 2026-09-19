//! Plan 18-06 — `pyscf/pbc/grad/kuhf.py` (124 l): KUHF k-point gradient.
//!
//! `class Gradients(krhf_grad.GradientsBase)` (`:86`): `grad_elec`,
//! `get_veff` and `make_rdm1e` replaced wholesale with the spin index
//! threaded through three of the five `grad_elec` terms; everything else
//! inherited unchanged.
//!
//! * Unit tests gate Task 1's threading on synthetic spin-resolved inputs:
//!   the spin-resolved middle term is exhibited against a naive einsum, the
//!   classic summed-middle-term error is shown to differ on open-shell
//!   inputs and to vanish on closed-shell ones (which is why Gate B must
//!   run on a spin-polarised cell, never on diamond).
//! * **Gate B** (`kuhf_open_shell_passes_verify_fd`): analytic gradient vs
//!   this port's `verify_fd` at `FD_TOL = 1e-6` Ha/Bohr on a genuinely
//!   spin-polarised periodic fixture — an open-shell HeH doublet
//!   (`spin = 1`, both spin channels occupied, `dm_a != dm_b` asserted).
//!   The all-electron fixtures of `gate_openshell.rs`/`init_guess_spin.rs`
//!   cannot be reused directly: the gradient `get_hcore` refuses
//!   all-electron cells by name (`krhf.py:111`), so the open-shell fixture
//!   here carries `gth-pade`.
//! * **Gate C** (`kuhf_closed_shell_matches_upstream_fingerprint`):
//!   `lib.fp(g)` against upstream's `-0.9017171774435333`
//!   (`test_kuhf.py:49`) on upstream's own cell (custom uncontracted
//!   `[[0,[1.3,1]],[1,[0.8,1]]` basis, `mesh = [13]*3`, `kpts = [1,1,2]`,
//!   `exxdiv=None`, `conv_tol=1e-10`), plus the KUHF-vs-KRHF agreement on
//!   that closed-shell fixture.
//!
//! # Geometry is specified in BOHR

use pyscf_algebra::{CTensor, oracle_sum};
use pyscf_pbc_grad::{Gradients, KuhfGradients, contract_vhf_atom_spin, sum_sets};
use pyscf_pbc_gto::{Cell, test_systems};

/// Upstream `lib.fp`: `dot(cos(arange(size)), ravel)` (`pyscf/lib/misc.py`).
fn fp(g: &[[f64; 3]]) -> f64 {
    let terms: Vec<f64> = g
        .iter()
        .flat_map(|r| r.iter().copied())
        .enumerate()
        .map(|(i, v)| (i as f64).cos() * v)
        .collect();
    oracle_sum(&terms)
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

// ---------------------------------------------------------------------------
// Task 1: spin threading — sum_sets
// ---------------------------------------------------------------------------

/// `dm0_sf = dm0[0] + dm0[1]` elementwise, both planes; shape rejections.
#[test]
fn sum_sets_adds_channels_elementwise() {
    let nao = 2;
    let dm_a = hermitian_dm_set(nao, 1, 3);
    let dm_b = hermitian_dm_set(nao, 1, 9);
    let got = sum_sets(&vec![dm_a.clone(), dm_b.clone()], nao).expect("sum");
    assert_eq!(got.len(), 1);
    for i in 0..nao * nao {
        assert_eq!(got[0].re[i], dm_a[0].re[i] + dm_b[0].re[i]);
        assert_eq!(got[0].im[i], dm_a[0].im[i] + dm_b[0].im[i]);
    }
    assert!(sum_sets(&vec![dm_a.clone()], nao).is_err());
    assert!(sum_sets(&vec![dm_a.clone(), dm_b.clone(), dm_a.clone()], nao).is_err());
    assert!(sum_sets(&vec![dm_a, hermitian_dm_set(nao, 2, 9)], nao).is_err());
}

// ---------------------------------------------------------------------------
// Task 1: spin threading — the spin-resolved vhf contraction
// ---------------------------------------------------------------------------

/// `contract_vhf_atom_spin` against a naive `einsum('xskij,skji->x')` with
/// the OPPOSITE loop nesting (s,k,j,i — a shared loop-order bug cannot hide
/// in both), `ji`-indexed, `.real` inside, `*2` outside.
///
/// Plus the discrimination proof for the classic error: a variant that sums
/// `vhf` over spin BEFORE contracting (the bug `kuhf.py:48` forbids) differs
/// on open-shell inputs and coincides on closed-shell ones — which is why
/// Gate B must run on a spin-polarised cell.
#[test]
fn vhf_spin_contraction_matches_naive_and_discriminates() {
    use pyscf_pbc_grad::krhf::contractions::contract_vhf_atom;
    let nao = 2;
    let nkpts = 2;
    // Two genuinely different spin channels.
    let dm_a = hermitian_dm_set(nao, nkpts, 3);
    let dm_b = hermitian_dm_set(nao, nkpts, 17);
    let dm = vec![dm_a.clone(), dm_b.clone()];
    // Spin-resolved vhf, rows i in atom A = AO 0.
    let vhf: Vec<[Vec<CTensor>; 3]> = (0..2)
        .map(|s| {
            std::array::from_fn::<Vec<CTensor>, 3, _>(|x| {
                (0..nkpts)
                    .map(|k| {
                        let mut re = vec![0.0; nao * nao];
                        let mut im = vec![0.0; nao * nao];
                        for i in 0..nao {
                            for j in 0..nao {
                                let (r, v) = z(i + 2 * x + 5 * s, j + k, 11 + s);
                                re[i * nao + j] = r;
                                im[i * nao + j] = v;
                            }
                        }
                        CTensor::from_planes(re, im)
                    })
                    .collect()
            })
        })
        .collect();

    let got = contract_vhf_atom_spin(&vhf, &dm, 0, 1, nao);
    // Naive reference: s,k outer; j middle; i (atom rows) inner.
    let want = std::array::from_fn::<f64, 3, _>(|x| {
        let mut acc = 0.0_f64;
        for s in 0..2 {
            for k in 0..nkpts {
                for j in 0..nao {
                    for i in 0..1 {
                        let (ar, ai) = (vhf[s][x][k].re[i * nao + j], vhf[s][x][k].im[i * nao + j]);
                        let (br, bi) = (dm[s][k].re[j * nao + i], dm[s][k].im[j * nao + i]);
                        acc += 2.0 * (ar * br - ai * bi);
                    }
                }
            }
        }
        acc
    });
    for x in 0..3 {
        assert!(
            (got[x] - want[x]).abs() < 1e-12,
            "x={x}: {} vs naive {}",
            got[x],
            want[x]
        );
    }
    assert!(
        max_abs_diff(&[got], &[[0.0; 3]]) > 1e-3,
        "contraction test is vacuous"
    );

    // Sensitivity: transposing one channel's DM must move the answer
    // (swap the (0,1)/(1,0) entries on every k-block).
    let dm_t = vec![
        dm_a.clone(),
        dm_b
            .iter()
            .map(|m| {
                let mut re = m.re.clone();
                let mut im = m.im.clone();
                re.swap(1, 2);
                im.swap(1, 2);
                CTensor::from_planes(re, im)
            })
            .collect(),
    ];
    let moved = contract_vhf_atom_spin(&vhf, &dm_t, 0, 1, nao);
    for x in 0..3 {
        assert!(
            (got[x] - moved[x]).abs() > 1e-3,
            "x={x}: transposed beta DM did not move the answer — index gate vacuous"
        );
    }

    // Clause 4: complex-then-.real in the same (s,k,i,j) order is
    // bit-identical to real-inside.
    let complex_route = std::array::from_fn::<f64, 3, _>(|x| {
        let mut tr = Vec::new();
        let mut ti = Vec::new();
        for s in 0..2 {
            for k in 0..nkpts {
                for i in 0..1 {
                    for j in 0..nao {
                        let (ar, ai) = (vhf[s][x][k].re[i * nao + j], vhf[s][x][k].im[i * nao + j]);
                        let (br, bi) = (dm[s][k].re[j * nao + i], dm[s][k].im[j * nao + i]);
                        tr.push(2.0 * (ar * br - ai * bi));
                        ti.push(2.0 * (ar * bi + ai * br));
                    }
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
            "x={x}: real-inside is not bit-identical to complex-then-.real"
        );
    }

    // The classic error: hoist the spin sum OUT of the einsum (sum `vhf`
    // over spin first, contract vs the summed density) and drop the `*2`
    // with it. On closed-shell inputs (`dm_a == dm_b`, `vhf_0 == vhf_1`)
    // this coincides with the correct term — `(2V)·(2D) = 2·(V·D + V·D)` —
    // while on open-shell inputs it is off by exactly the exchange
    // asymmetry `(V0-V1)·(D0-D1)`. That is why Gate B needs a polarised cell.
    let buggy = {
        let dm_sf = sum_sets(&dm, nao).expect("summed dm");
        // vhf_sum[x][k] = vhf[0][x][k] + vhf[1][x][k] (complex add).
        let vhf_sum: [Vec<CTensor>; 3] = std::array::from_fn(|x| {
            vhf[0][x]
                .iter()
                .zip(vhf[1][x].iter())
                .map(|(a, b)| {
                    CTensor::from_planes(
                        a.re.iter().zip(&b.re).map(|(p, q)| p + q).collect(),
                        a.im.iter().zip(&b.im).map(|(p, q)| p + q).collect(),
                    )
                })
                .collect()
        });
        // `contract_vhf_atom` carries the `*2`; halving exhibits the
        // dropped-factor variant (summed-vhf vs summed-dm at `*1`).
        let doubled = contract_vhf_atom(&vhf_sum, &dm_sf, 0, 1, nao);
        let halved: [f64; 3] = std::array::from_fn(|x| 0.5 * doubled[x]);
        halved
    };
    for x in 0..3 {
        assert!(
            (got[x] - buggy[x]).abs() > 1e-3,
            "x={x}: summed-middle-term variant agrees on OPEN-SHELL inputs — discrimination gate vacuous"
        );
    }
    // ... while on closed-shell inputs (dm_a == dm_b, vhf_0 == vhf_1) the
    // bug is invisible — the reason Gate B needs a polarised cell.
    let dm_closed = vec![dm_a.clone(), dm_a.clone()];
    let vhf_closed = vec![vhf[0].clone(), vhf[0].clone()];
    let right = contract_vhf_atom_spin(&vhf_closed, &dm_closed, 0, 1, nao);
    let dm_sf = sum_sets(&dm_closed, nao).expect("summed dm");
    let vhf_sum: [Vec<CTensor>; 3] = std::array::from_fn(|x| {
        vhf_closed[0][x]
            .iter()
            .zip(vhf_closed[1][x].iter())
            .map(|(a, b)| {
                CTensor::from_planes(
                    a.re.iter().zip(&b.re).map(|(p, q)| p + q).collect(),
                    a.im.iter().zip(&b.im).map(|(p, q)| p + q).collect(),
                )
            })
            .collect()
    });
    let wrong_doubled = contract_vhf_atom(&vhf_sum, &dm_sf, 0, 1, nao);
    let wrong: [f64; 3] = std::array::from_fn(|x| 0.5 * wrong_doubled[x]);
    for x in 0..3 {
        assert!(
            (right[x] - wrong[x]).abs() < 1e-12,
            "x={x}: closed-shell inputs should HIDE the middle-term error"
        );
    }
}

// ---------------------------------------------------------------------------
// Task 2: make_rdm1e is the per-spin pair
// ---------------------------------------------------------------------------

/// Each `dme0` set equals 18-05's restricted `make_rdm1e` on that spin's
/// orbitals (`kuhf.py:80-84`); the constructor validates the `2*nkpts`
/// block counts.
#[test]
fn make_rdm1e_is_the_per_spin_pair() {
    use pyscf_pbc_grad::krhf::make_rdm1e_kpts;
    use pyscf_pbc_scf::Kuhf;
    let cell = test_systems::diamond();
    let kpts = vec![[0.0_f64; 3]];
    let nkpts = 1;
    let nao = cell.mol.nao_nr;
    let mf = Kuhf::new(cell, &kpts).expect("KUHF holder");
    let (ca, ea, oa) = synthetic_spin_orbitals(nao, nkpts, 41);
    let (cb, eb, ob) = synthetic_spin_orbitals(nao, nkpts, 77);
    let coeff = [ca.clone(), cb.clone()].concat();
    let energy = [ea.clone(), eb.clone()].concat();
    let occ = [oa.clone(), ob.clone()].concat();
    let grad = KuhfGradients::new(&mf, energy, coeff, occ).expect("gradients");
    let dme = grad.make_rdm1e().expect("dme0");
    assert_eq!(dme.len(), 2);
    let want_a = make_rdm1e_kpts(&ca, &ea, &oa, nao).expect("restricted a");
    let want_b = make_rdm1e_kpts(&cb, &eb, &ob, nao).expect("restricted b");
    for (got, want) in dme.iter().zip([want_a, want_b].iter()) {
        assert_eq!(got.len(), nkpts);
        for k in 0..nkpts {
            assert_eq!(got[k].re, want[k].re);
            assert_eq!(got[k].im, want[k].im);
        }
    }
    // The two synthetic channels must differ (else the pair test is vacuous).
    assert!(max_abs_diff(&[[dme[0][0].re[0], 0.0, 0.0]], &[[dme[1][0].re[0], 0.0, 0.0]]) > 0.0);

    // Constructor validation: wrong block counts refuse.
    let (ca2, ea2, oa2) = synthetic_spin_orbitals(nao, nkpts, 41);
    assert!(
        KuhfGradients::new(
            &mf,
            ea2.clone(),
            ca2.clone(),
            oa2.clone(),
        )
        .is_err(),
        "single-set blocks must be refused (need 2*nkpts)"
    );
    let mut bad = ca2.clone();
    bad[0].re[0] = f64::NAN;
    assert!(
        KuhfGradients::new(
            &mf,
            [ea2.clone(), ea2].concat(),
            [bad, cb].concat(),
            [oa2.clone(), oa2].concat(),
        )
        .is_err(),
        "non-finite density must be refused"
    );
}

// ---------------------------------------------------------------------------
// Task 2: get_veff = vj[0] + vj[1] - vk, spin-resolved
// ---------------------------------------------------------------------------

/// `veff[s] = vj_sum - vk[s]` elementwise against `jk_deriv`'s own halves,
/// on a genuinely spin-polarised density (exchange must differ per spin —
/// otherwise the assembly test is vacuous). Runs on the small-mesh Gate-C
/// cell at gamma: the 18-04 route is mesh-independent, and the default
/// precision mesh would make this assembly check gratuitously slow.
#[test]
fn veff_assembles_summed_coulomb_minus_spin_exchange() {
    use pyscf_pbc_scf::Kuhf;
    let cell = upstream_test_cell();
    let kpts = vec![[0.0_f64; 3]];
    let nkpts = 1;
    let nao = cell.mol.nao_nr;
    let mf = Kuhf::new(cell, &kpts).expect("KUHF holder");
    let (ca, ea, oa) = synthetic_spin_orbitals(nao, nkpts, 41);
    let (cb, eb, ob) = synthetic_spin_orbitals(nao, nkpts, 77);
    let grad = KuhfGradients::new(
        &mf,
        [ea, eb].concat(),
        [ca, cb].concat(),
        [oa, ob].concat(),
    )
    .expect("gradients");
    let dma = hermitian_dm_set(nao, nkpts, 3);
    let dmb = hermitian_dm_set(nao, nkpts, 17);
    let dm = vec![dma, dmb];
    let (vj, vk) = grad.jk_deriv(&dm).expect("jk");
    assert_eq!((vj.len(), vk.len()), (2, 2));
    // Exchange is spin-resolved on an open-shell density.
    let k_gap = max_abs_diff(
        &[[vj[0][0][0].re[0], vj[0][0][0].im[0], 0.0]],
        &[[vk[1][0][0].re[0], vk[1][0][0].im[0], 0.0]],
    );
    assert!(k_gap > 0.0, "exchange assembly test is vacuous");
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
    let veff = grad.veff(&dm).expect("veff");
    assert_eq!(veff.len(), 2);
    for s in 0..2 {
        for x in 0..3 {
            for k in 0..nkpts {
                for i in 0..nao * nao {
                    let want = vj[0][x][k].re[i] + vj[1][x][k].re[i] - vk[s][x][k].re[i];
                    let got = veff[s][x][k].re[i];
                    assert!(
                        (got - want).abs() < 1e-12,
                        "re s={s} x={x} k={k} i={i}: {got} vs {want}"
                    );
                    let want_im = vj[0][x][k].im[i] + vj[1][x][k].im[i] - vk[s][x][k].im[i];
                    let got_im = veff[s][x][k].im[i];
                    assert!(
                        (got_im - want_im).abs() < 1e-12,
                        "im s={s} x={x} k={k} i={i}: {got_im} vs {want_im}"
                    );
                }
            }
        }
    }
    // Single-set densities refuse the spin-summed veff by name.
    assert!(grad.veff(&dm[..1].to_vec()).is_err());
}

// ---------------------------------------------------------------------------
// Gate C: upstream fingerprint on upstream's cell
// ---------------------------------------------------------------------------

/// Upstream `test_kuhf.py:setUpModule`'s cell EXACTLY: custom uncontracted
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
    .expect("upstream test_kuhf cell must build")
}

fn kpts_112(cell: &Cell) -> Vec<[f64; 3]> {
    pyscf_pbc_gto::kpts_mesh::make_kpts_default(cell, [1, 1, 2]).expect("1x1x2 k-mesh")
}

/// Gate C: `lib.fp(g)` vs upstream's committed `-0.9017171774435333`
/// (`test_kuhf.py:49`), at the 6 decimals upstream itself asserts
/// (`assertAlmostEqual(..., 6)`), plus KUHF-vs-KRHF agreement on this
/// closed-shell fixture.
///
/// Upstream's path is KUHF SCF → gradient, and so is this one, with two
/// settings stated explicitly: `exxdiv = None` (upstream's fixture) and
/// `init_guess_breaksym = 0` (see below).
///
/// # Why `init_guess_breaksym = 0` — a measured SCF guess-path finding
///
/// With the default `init_guess_breaksym = 1` the port's KUHF SCF on this
/// cell converges deterministically (3/3 identical runs) to a
/// BROKEN-SYMMETRY determinant at `e = -4.845723187584` (`cycles = 14`,
/// `max|dm_a − dm_b| = 5.581e-1`, `<S^2> = 1.8312806848286405`), 16 mHa
/// below the restricted solution upstream's default path finds
/// (`e = -4.8294530948409715`, `<S^2> ≈ 0`). That minimum is genuine, not a
/// functional bug: upstream's own `energy_elec` evaluated on the port's
/// density gives `7.941406870613185` vs the port's `7.941406874537325`
/// (same functional), and upstream re-converged FROM the port's density
/// stays there (`e = -4.84572319150857`, `<S^2> = 1.8312806857103254` —
/// energy to 4e-9, spin to 9 digits). Upstream seeded with a broken guess
/// likewise falls below RHF (`e = -4.836843554407634`, `<S^2> = 1.00`).
/// The restricted and broken basins differ between the two DIIS paths;
/// that is SCF-domain (Phase 14/17) scope, reported here with numbers, not
/// worked around in the gradient.
///
/// `init_guess_breaksym = 0` starts `dm_a == dm_b` bit-identically — an
/// exact fixed point of the SCF map the linear DIIS preserves — so the
/// port's KUHF lands on the restricted solution the committed fingerprint
/// belongs to (`e = -4.829453088899`, `<S^2> = 0`, occ `[4,4]/[4,4]`).
/// The energy premise is asserted below; the gradient under test is
/// 18-06's code on those orbitals.
#[test]
fn kuhf_closed_shell_matches_upstream_fingerprint() {
    use pyscf_pbc_scf::{KScfConfig, Krhf, Kuhf};
    let cell = upstream_test_cell();
    let kpts = kpts_112(&cell);
    assert_eq!(kpts.len(), 2);

    let cfg = KScfConfig {
        conv_tol: 1e-10,
        conv_tol_grad: Some(1e-8),
        max_cycle: 100,
        ..KScfConfig::for_cell(&cell)
    };
    // Upstream's fixture: KUHF with exxdiv=None; init_guess_breaksym = 0
    // recovers the restricted solution (see doc comment above).
    let mut mf = Kuhf::new(cell.clone(), &kpts).expect("KUHF builds");
    mf.exxdiv = None;
    mf.init_guess_breaksym = 0;
    let result = mf.kernel(&cfg).expect("KUHF SCF");
    assert!(
        result.converged,
        "KUHF did not converge after {} cycles — the fp number is meaningless",
        result.cycles
    );
    assert!(
        (result.e_tot - (-4.8294530948409715)).abs() < 1e-6,
        "SCF premise FAILED: e = {:.12} vs upstream -4.8294530948409715 — \
         not the restricted solution the committed fp belongs to",
        result.e_tot
    );
    let analytic = KuhfGradients::new(&mf, result.mo_energy, result.mo_coeff, result.mo_occ)
        .expect("gradient object")
        .kernel()
        .expect("analytic gradient");
    let fingerprint = fp(&analytic);
    println!("Gate C (KUHF): lib.fp(g) = {fingerprint:.17}");
    assert!(
        (fingerprint - (-0.9017171774435333)).abs() < 5e-7,
        "Gate C FAILED: fp = {fingerprint:.12} vs upstream -0.9017171774435333"
    );

    // Same constant as KRHF's (upstream's KUHF fixture is closed-shell
    // diamond): the two must agree tightly here.
    let mut mf_r = Krhf::new(cell.clone(), &kpts).expect("KRHF builds");
    mf_r.exxdiv = None;
    let result_r = mf_r.kernel(&cfg).expect("KRHF SCF");
    assert!(result_r.converged, "KRHF did not converge");
    let analytic_r = pyscf_pbc_grad::KrhfGradients::new(
        &mf_r,
        result_r.mo_energy,
        result_r.mo_coeff,
        result_r.mo_occ,
    )
    .expect("KRHF gradient object")
    .kernel()
    .expect("KRHF analytic gradient");
    let fp_r = fp(&analytic_r);
    println!("Gate C (KRHF on same cell): lib.fp(g) = {fp_r:.17}");
    let gap = (fingerprint - fp_r).abs();
    println!("KUHF-vs-KRHF fp gap (closed-shell): {gap:.3e}");
    assert!(
        gap < 1e-9,
        "KUHF-vs-KRHF fp gap {gap:.3e} exceeds 1e-9 on the closed-shell fixture"
    );
}

// ---------------------------------------------------------------------------
// Gate B: verify_fd on a genuinely spin-polarised cell
// ---------------------------------------------------------------------------

/// Open-shell HeH doublet (`spin = 1`, 3 valence electrons under
/// `gth-pade`: He-q2 + H-q1): He at `(0,0,-1)`, H pushed `+0.02` Bohr in x
/// off `(0,0,1)` so the gradient is O(10⁻²) and the gate cannot pass
/// vacuously. Both spin channels are occupied (`nalpha = 2, nbeta = 1`),
/// so the spin-resolved middle term is fully exercised — unlike a triplet
/// with an empty beta channel.
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

fn kuhf_energy(cell: &Cell, kpts: &[[f64; 3]]) -> Result<f64, pyscf_core::PyscfRsError> {
    use pyscf_pbc_scf::Kuhf;
    let mf = Kuhf::new(cell.clone(), kpts).expect("KUHF builds on a displaced cell");
    let cfg = tight_config(cell);
    let result = mf.kernel(&cfg)?;
    assert!(
        result.converged,
        "KUHF did not converge after {} cycles — the FD number is meaningless",
        result.cycles
    );
    Ok(result.e_tot)
}

/// Gate B: `max|verify_fd − analytic| <= 1e-6`, with the measured residual
/// printed on every run. The SCF densities must be spin-polarised
/// (`dm_a != dm_b` — else this gates the restricted path under another
/// name) and the analytic gradient non-vacuous.
#[test]
fn kuhf_open_shell_passes_verify_fd() {
    use pyscf_pbc_grad::verify_fd;
    use pyscf_pbc_scf::Kuhf;
    const DISP: f64 = 5e-6;
    const TOL: f64 = 1e-6;

    let cell = open_shell_cell();
    let kpts = vec![[0.0_f64; 3]];
    let (analytic, channel_gap) = {
        let mf = Kuhf::new(cell.clone(), &kpts).expect("central KUHF");
        let cfg = tight_config(&cell);
        let result = mf.kernel(&cfg).expect("central SCF");
        assert!(result.converged, "central SCF did not converge");
        assert_eq!(result.dm.len(), 2, "KUHF must carry two spin sets");
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
        let g = KuhfGradients::new(&mf, result.mo_energy, result.mo_coeff, result.mo_occ)
            .expect("gradient object")
            .kernel()
            .expect("analytic gradient");
        (g, gap)
    };
    let _ = channel_gap;
    let peak = analytic
        .iter()
        .flat_map(|r| r.iter())
        .fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(
        peak > 1e-4,
        "gate is vacuous: max|analytic| = {peak:e} on the displaced cell"
    );

    let report = verify_fd(&cell, &analytic, |c| kuhf_energy(c, &kpts), DISP, TOL)
        .expect("finite-difference harness");
    println!(
        "Gate B (KUHF open-shell HeH): max|fd − analytic| = {:.3e} Ha/Bohr",
        report.max_abs_diff
    );
    assert!(
        report.passed,
        "Gate B FAILED: max|fd − analytic| = {:.3e} > {TOL:e}\nanalytic = {analytic:?}\nfd = {:?}",
        report.max_abs_diff, report.fd_grad,
    );
}

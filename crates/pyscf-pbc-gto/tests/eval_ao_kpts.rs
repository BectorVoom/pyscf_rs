//! Plan 10-04 — periodic AO evaluation (`eval_ao_kpts`).
//!
//! The PRIMARY gate is [`bloch_periodicity_holds`]: a periodic AO must satisfy
//!
//! ```text
//! ao_k(r + L) == exp(i·k·L) · ao_k(r)
//! ```
//!
//! for every lattice vector `L`. That is a complete correctness statement for
//! the phase convention, the image list and the accumulation, and it needs no
//! oracle (D-PBC-19). A sign error in the phase, a missing image, or a shifted
//! grid all break it.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, eval_ao_kpts, kpts_mesh::make_kpts_default};

fn diamond() -> Cell {
    let h = 3.37032;
    let q = 1.68516;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        pseudo: Some("gth-pade".into()),
        ..Default::default()
    })
    .expect("diamond builds")
}

/// A handful of interior points plus one deliberately near the cell edge,
/// where a too-small image list would show up first.
fn probe_grid(cell: &Cell) -> Vec<[f64; 3]> {
    let a = cell.lattice_vectors();
    let frac = [
        [0.10, 0.20, 0.30],
        [0.50, 0.50, 0.50],
        [0.97, 0.03, 0.45],
        [0.25, 0.25, 0.25],
        [0.00, 0.00, 0.00],
    ];
    frac.iter()
        .map(|f| {
            [
                f[0] * a[0][0] + f[1] * a[1][0] + f[2] * a[2][0],
                f[0] * a[0][1] + f[1] * a[1][1] + f[2] * a[2][1],
                f[0] * a[0][2] + f[1] * a[1][2] + f[2] * a[2][2],
            ]
        })
        .collect()
}

/// **THE PRIMARY GATE.** `ao_k(r + L) == exp(i k·L) · ao_k(r)` for every
/// combination of probe point, lattice vector and k-point.
#[test]
fn bloch_periodicity_holds() {
    let cell = diamond();
    let a = cell.lattice_vectors();
    let kpts = [[0.0, 0.0, 0.0], [0.13, -0.07, 0.21], [-0.31, 0.11, 0.05]];

    let coords = probe_grid(&cell);
    let shifts: Vec<[f64; 3]> = vec![
        a[0],
        a[1],
        a[2],
        [
            a[0][0] + a[1][0] - a[2][0],
            a[0][1] + a[1][1] - a[2][1],
            a[0][2] + a[1][2] - a[2][2],
        ],
    ];

    let base = eval_ao_kpts(&cell, "GTOval_sph", &coords, &kpts).expect("eval at r");
    assert_eq!(base.nao, cell.mol.nao_nr);
    assert_eq!(base.ngrids, coords.len());
    assert_eq!(base.comp, 1);

    let mut worst = 0.0_f64;
    for l in &shifts {
        let shifted: Vec<[f64; 3]> = coords
            .iter()
            .map(|r| [r[0] + l[0], r[1] + l[1], r[2] + l[2]])
            .collect();
        let moved = eval_ao_kpts(&cell, "GTOval_sph", &shifted, &kpts).expect("eval at r+L");

        for (k, kpt) in kpts.iter().enumerate() {
            let theta = kpt[0] * l[0] + kpt[1] * l[1] + kpt[2] * l[2];
            let (pr, pi) = (theta.cos(), theta.sin());
            for g in 0..coords.len() {
                for mu in 0..base.nao {
                    let (br, bi) = base.element(k, 0, g, mu);
                    let (mr, mi) = moved.element(k, 0, g, mu);
                    // ao_k(r+L) must equal exp(i k.L) * ao_k(r).
                    worst = worst.max((mr - (pr * br - pi * bi)).abs());
                    worst = worst.max((mi - (pr * bi + pi * br)).abs());
                }
            }
        }
    }
    assert!(
        worst < 1e-10,
        "Bloch periodicity violated by {worst:e} (tolerance 1e-10)"
    );
}

/// At k = 0 the AO sum is real, because the image list is inversion-symmetric
/// and every phase is 1.
#[test]
fn gamma_point_is_real() {
    let cell = diamond();
    let coords = probe_grid(&cell);
    let ao = eval_ao_kpts(&cell, "GTOval_sph", &coords, &[[0.0; 3]]).expect("eval");

    assert!(ao.gamma[0]);
    let max_im = ao.at(0).im.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert_eq!(max_im, 0.0, "gamma AO must be exactly real");
    // …and non-trivial: the AO values themselves must not all be zero.
    let max_re = ao.at(0).re.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
    assert!(max_re > 1e-3, "gamma AO is suspiciously small ({max_re:e})");
}

/// `ao_{-k} == conj(ao_k)` — time-reversal symmetry, the same check the
/// 1-electron driver gets.
#[test]
fn negative_k_is_the_conjugate() {
    let cell = diamond();
    let coords = probe_grid(&cell);
    let k = [0.21, -0.09, 0.14];
    let ao = eval_ao_kpts(&cell, "GTOval_sph", &coords, &[k, [-k[0], -k[1], -k[2]]]).expect("eval");

    let (p, m) = (ao.at(0), ao.at(1));
    let mut worst = 0.0_f64;
    for i in 0..p.len() {
        worst = worst.max((p.re[i] - m.re[i]).abs());
        worst = worst.max((p.im[i] + m.im[i]).abs());
    }
    assert!(worst < 1e-12, "ao(-k) != conj(ao(k)): {worst:e}");
}

/// The image list is converged: widening it must not move an AO value.
#[test]
fn image_sum_is_converged() {
    let cell = diamond();
    let coords = probe_grid(&cell);
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("make_kpts");

    let rcut = pyscf_pbc_gto::estimate_rcut_for_eval(&cell, 0).expect("rcut");
    let rmax = rcut.iter().copied().fold(0.0_f64, f64::max);
    let narrow =
        pyscf_pbc_gto::lattice::get_lattice_ls(&cell, Some(rmax), None, false).expect("Ls");
    let wide = pyscf_pbc_gto::lattice::get_lattice_ls(&cell, Some(rmax * 1.5), None, false)
        .expect("wide Ls");
    assert!(wide.len() > narrow.len());

    let a = pyscf_pbc_gto::eval_ao_kpts_with_images(&cell, "GTOval_sph", &coords, &kpts, &narrow)
        .expect("narrow");
    let b = pyscf_pbc_gto::eval_ao_kpts_with_images(&cell, "GTOval_sph", &coords, &kpts, &wide)
        .expect("wide");

    let mut worst = 0.0_f64;
    for k in 0..a.nkpts() {
        for i in 0..a.at(k).len() {
            worst = worst.max((a.at(k).re[i] - b.at(k).re[i]).abs());
            worst = worst.max((a.at(k).im[i] - b.at(k).im[i]).abs());
        }
    }
    assert!(
        worst < 1e-11,
        "AO values moved by {worst:e} when the image list grew 50%"
    );
}

/// `GTOval_sph_deriv1` carries 4 components and its value component reproduces
/// the plain `GTOval_sph` result.
#[test]
fn deriv1_layout_and_value_component() {
    let cell = diamond();
    let coords = probe_grid(&cell);
    let kpts = [[0.11, 0.22, -0.05]];

    let plain = eval_ao_kpts(&cell, "GTOval_sph", &coords, &kpts).expect("value");
    let d1 = eval_ao_kpts(&cell, "GTOval_sph_deriv1", &coords, &kpts).expect("deriv1");

    assert_eq!(d1.comp, 4);
    assert_eq!(d1.at(0).len(), 4 * d1.ngrids * d1.nao);

    let mut worst = 0.0_f64;
    for g in 0..coords.len() {
        for mu in 0..plain.nao {
            let (pr, pi) = plain.element(0, 0, g, mu);
            let (dr, di) = d1.element(0, 0, g, mu);
            worst = worst.max((pr - dr).abs()).max((pi - di).abs());
        }
    }
    assert!(
        worst < 1e-12,
        "deriv1 component 0 differs from GTOval_sph by {worst:e}"
    );
}

/// The gradient components are the numerical derivative of the value component.
#[test]
fn deriv1_matches_a_finite_difference() {
    let cell = diamond();
    let a = cell.lattice_vectors();
    let coords = vec![[
        0.3 * a[0][0] + 0.4 * a[1][0],
        0.3 * a[0][1] + 0.4 * a[1][1] + 0.2,
        0.3 * a[0][2] + 0.4 * a[1][2] - 0.1,
    ]];
    let kpts = [[0.07, -0.15, 0.09]];
    let h = 1e-5;

    let d1 = eval_ao_kpts(&cell, "GTOval_sph_deriv1", &coords, &kpts).expect("deriv1");

    for axis in 0..3 {
        let mut plus = coords.clone();
        let mut minus = coords.clone();
        plus[0][axis] += h;
        minus[0][axis] -= h;
        let fp = eval_ao_kpts(&cell, "GTOval_sph", &plus, &kpts).expect("f+");
        let fm = eval_ao_kpts(&cell, "GTOval_sph", &minus, &kpts).expect("f-");

        for mu in 0..d1.nao {
            let (ar, ai) = d1.element(0, axis + 1, 0, mu);
            let (pr, pi) = fp.element(0, 0, 0, mu);
            let (mr, mi) = fm.element(0, 0, 0, mu);
            let (nr, ni) = ((pr - mr) / (2.0 * h), (pi - mi) / (2.0 * h));
            assert!(
                (ar - nr).abs() < 1e-6 && (ai - ni).abs() < 1e-6,
                "d/d{axis} of AO {mu}: analytic ({ar}, {ai}) vs numeric ({nr}, {ni})"
            );
        }
    }
}

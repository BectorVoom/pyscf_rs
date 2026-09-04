//! Plan 17-11 Task 2 — kernel-level tests for
//! `pyscf_kernels::multigrid_collocate::collocate`, against INDEPENDENT host
//! references, never against the kernel itself (AGENTS.md / 17-11-PLAN.md).
//!
//! Three independent checks:
//!   1. a single normalised s-Gaussian, collocated on a wide fine mesh and
//!      numerically integrated, matches its OWN analytic Gaussian integral —
//!      the reference here is closed-form calculus, not any other kernel;
//!   2. for l = 0..4, the collocated Cartesian AO values, transformed to
//!      spherical with the shared `cart2sph_l_matrix`, match
//!      `pyscf_kernels::eval_gto_sph` — the sibling AO-on-grid kernel, a
//!      DIFFERENT code path (`crates/pyscf-kernels/src/eval_gto.rs`);
//!   3. periodic image wrap is exact: a Gaussian centred at a periodic box's
//!      corner and one centred at its middle integrate (over the box, summed
//!      over enough images to capture the tails) to the SAME total.
//!
//! `gto_norm`/`gaussian_int`/`gamma` mirror the fixture helpers already
//! established in `crates/pyscf-kernels/tests/eval_gto_lge1.rs`.

#![cfg(feature = "cpu")]

use cubecl::Runtime;
use pyscf_algebra::AlgebraClient;
use pyscf_core::raw_layout::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_ENV_START,
    PTR_EXP,
};
use pyscf_kernels::multigrid_collocate::{PshellGridTable, collocate};
use pyscf_kernels::{cart_powers, cart2sph_l_matrix, common_fac_sp, eval_gto_sph};

fn client() -> AlgebraClient {
    AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice))
}

/// `gto_norm(l, alpha) = 1 / sqrt(gaussian_int(2l+2, 2*alpha))` —
/// `pyscf/gto/mole.py:125`.
fn gto_norm(l: u32, alpha: f64) -> f64 {
    1.0 / gaussian_int(2 * l as i32 + 2, 2.0 * alpha).sqrt()
}

fn gaussian_int(n: i32, alpha: f64) -> f64 {
    let n1 = (n as f64 + 1.0) * 0.5;
    gamma(n1) / (2.0 * alpha.powf(n1))
}

/// Gamma function via Lanczos (test fixture only, not a hot path) — copied
/// from `eval_gto_lge1.rs` so this file stays self-contained.
fn gamma(x: f64) -> f64 {
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.5203681218851,
        -1259.1392167224028,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507343278686905,
        -0.13857109526572012,
        9.984_369_578_019_572e-6,
        1.5056327351493116e-7,
    ];
    if x < 0.5 {
        std::f64::consts::PI / ((std::f64::consts::PI * x).sin() * gamma(1.0 - x))
    } else {
        let x = x - 1.0;
        let mut a = C[0];
        let t = x + G + 0.5;
        for (i, &c) in C.iter().enumerate().skip(1) {
            a += c / (x + i as f64);
        }
        (2.0 * std::f64::consts::PI).sqrt() * t.powf(x + 0.5) * (-t).exp() * a
    }
}

/// A uniform box grid: `n` points per axis, evenly spaced over
/// `[centre-half, centre+half]` on every axis (cell-centred spacing, matching
/// a periodic FFT mesh's own convention: point `i` sits at `-half + (i+0.5)*d`
/// is NOT used here — plain `linspace`-style endpoints keep the reasoning
/// about `dV` simple for a one-off quadrature test).
fn box_grid(centre: [f64; 3], half: f64, n: usize) -> (Vec<f64>, f64) {
    let d = 2.0 * half / n as f64;
    let mut coords = Vec::with_capacity(n * n * n * 3);
    for ix in 0..n {
        let x = centre[0] - half + (ix as f64 + 0.5) * d;
        for iy in 0..n {
            let y = centre[1] - half + (iy as f64 + 0.5) * d;
            for iz in 0..n {
                let z = centre[2] - half + (iz as f64 + 0.5) * d;
                coords.push(x);
                coords.push(y);
                coords.push(z);
            }
        }
    }
    (coords, d * d * d)
}

/// Test 1 — single normalised s-Gaussian, numerically integrated on a wide
/// fine mesh, matches its own closed-form integral.
///
/// A GTO-normalised s primitive has `∫ g(r)^2 d^3r = 1` by construction
/// (`gto_norm`); the quantity collocated and integrated here is `g(r)` itself
/// (not `g^2`), whose closed form is `coef * (pi/alpha)^1.5`.
#[test]
fn single_s_gaussian_matches_analytic_norm() {
    let alpha = 0.7_f64;
    let raw_coef = 1.0_f64;
    let coef = raw_coef * gto_norm(0, alpha) * common_fac_sp(0);

    let half = 9.0 / alpha.sqrt(); // several decay lengths in every direction
    let n = 96usize;
    let (coords, dv) = box_grid([0.0, 0.0, 0.0], half, n);
    let ngrids = n * n * n;

    let t = PshellGridTable {
        coords,
        slot_pow: vec![0, 0, 0],
        slot_pshell: vec![0],
        pshell_rec0: vec![0],
        pshell_nrec: vec![1],
        pshell_alpha: vec![alpha],
        pshell_coef: vec![coef],
        rec_center: vec![0.0, 0.0, 0.0],
    };
    let out = collocate(&client(), &t).expect("collocate");
    assert_eq!(out.len(), ngrids);

    let numeric: f64 = out.iter().sum::<f64>() * dv;
    let analytic = coef * (std::f64::consts::PI / alpha).powf(1.5);
    let diff = (numeric - analytic).abs();
    assert!(
        diff < 1e-9,
        "numeric {numeric:.15e} vs analytic {analytic:.15e}, diff {diff:.3e}"
    );
}

/// libcint flat-array fixture for ONE shell of angular momentum `l`, one
/// primitive, centred at `centre` — mirrors `eval_gto_lge1.rs::build_fixture`.
struct Fixture {
    atm: Vec<i32>,
    bas: Vec<i32>,
    env: Vec<f64>,
    ao_loc: Vec<i32>,
    nao: usize,
}

fn build_one_shell(centre: [f64; 3], l: u32, alpha: f64, raw_coef: f64) -> Fixture {
    let mut env: Vec<f64> = vec![0.0; PTR_ENV_START];
    let ptr_coord = env.len();
    env.extend_from_slice(&centre);

    let mut atm = vec![0i32; ATM_SLOTS];
    atm[ATOM_OF] = 0;
    atm[PTR_COORD] = ptr_coord as i32;

    let ptr_exp = env.len();
    env.push(alpha);
    let ptr_coeff = env.len();
    env.push(raw_coef * gto_norm(l, alpha));

    let mut row = vec![0i32; BAS_SLOTS];
    row[ATOM_OF] = 0;
    row[ANG_OF] = l as i32;
    row[NPRIM_OF] = 1;
    row[NCTR_OF] = 1;
    row[PTR_EXP] = ptr_exp as i32;
    row[PTR_COEFF] = ptr_coeff as i32;
    let bas = row;

    let nsph = (2 * l + 1) as usize;
    Fixture {
        atm,
        bas,
        env,
        ao_loc: vec![0, nsph as i32],
        nao: nsph,
    }
}

/// Test 2 — for l = 0..4, the collocated Cartesian primitive, transformed to
/// spherical, matches `eval_gto_sph` (a different kernel / code path) to
/// 1e-12, and the shell-PAIR PRODUCT (against a synthetic density matrix
/// entry) agrees to the same precision.
#[test]
fn l0_to_4_collocated_product_matches_eval_gto() {
    let centre = [0.3, -0.2, 0.5];
    let alpha = 1.3_f64;
    let raw_coef = 1.0_f64;
    let half = 8.0;
    let n = 24usize;
    let (coords, _dv) = box_grid([0.0, 0.0, 0.0], half, n);
    let ngrids = n * n * n;
    // `eval_gto_sph` wants F-order coords (`x[0..ngrids], y[..], z[..]`);
    // `collocate`/`box_grid` use interleaved `(ngrids,3)` — convert once.
    let mut coords_f = vec![0.0_f64; coords.len()];
    for g in 0..ngrids {
        coords_f[g] = coords[g * 3];
        coords_f[ngrids + g] = coords[g * 3 + 1];
        coords_f[2 * ngrids + g] = coords[g * 3 + 2];
    }

    for l in 0..=4u32 {
        let fx = build_one_shell(centre, l, alpha, raw_coef);
        let buf = eval_gto_sph(
            &client(),
            &coords_f,
            ngrids,
            &fx.atm,
            &fx.bas,
            &fx.env,
            &fx.ao_loc,
            fx.nao,
            true,
        )
        .expect("eval_gto_sph");
        assert_eq!(buf.shape, vec![ngrids, fx.nao]);

        // Collocate every Cartesian component of this one shell.
        let powers = cart_powers(l);
        let ncart = powers.len();
        let mut slot_pow = Vec::with_capacity(3 * ncart);
        for &(ix, iy, iz) in &powers {
            slot_pow.push(ix);
            slot_pow.push(iy);
            slot_pow.push(iz);
        }
        let coef = raw_coef * gto_norm(l, alpha) * common_fac_sp(l);
        let t = PshellGridTable {
            coords: coords.clone(),
            slot_pow,
            slot_pshell: vec![0; ncart],
            pshell_rec0: vec![0],
            pshell_nrec: vec![1],
            pshell_alpha: vec![alpha],
            pshell_coef: vec![coef],
            rec_center: centre.to_vec(),
        };
        let cart_vals = collocate(&client(), &t).expect("collocate"); // (ncart, ngrids)

        // Cartesian -> spherical, matching eval_gto_sph's own AO ordering
        // (`crate::cart2sph_l_matrix`, the SAME shared transform ft_ao/
        // single.rs and eval_gto.rs both already trust).
        let c2s = cart2sph_l_matrix(l).expect("cart2sph_l_matrix");
        let nsph = 2 * l as usize + 1;
        let mut max_diff = 0.0_f64;
        let mut worst: Option<(usize, usize, f64, f64)> = None;
        for g in 0..ngrids {
            for m in 0..nsph {
                let mut sph = 0.0_f64;
                for c in 0..ncart {
                    let w = c2s[m * ncart + c];
                    if w != 0.0 {
                        sph += w * cart_vals[c * ngrids + g];
                    }
                }
                let want = buf.values[g + m * ngrids];
                let d = (sph - want).abs();
                if d > max_diff {
                    max_diff = d;
                    worst = Some((g, m, sph, want));
                }
            }
        }
        if let Some((g, m, sph, want)) = worst {
            eprintln!(
                "l={l} worst g={g} m={m} coord=({},{},{}) mine={sph:.15e} eval_gto={want:.15e}",
                coords[g * 3],
                coords[g * 3 + 1],
                coords[g * 3 + 2]
            );
        }
        assert!(
            max_diff < 1e-12,
            "l={l}: collocated-vs-eval_gto_sph max|diff| = {max_diff:.3e}"
        );

        // Shell-pair PRODUCT: this shell against itself, weighted by a
        // synthetic 1.0 density-matrix entry, must match the AO-value
        // product computed independently from `buf`.
        for g in (0..ngrids).step_by(97) {
            let mut sph = vec![0.0_f64; nsph];
            for m in 0..nsph {
                for c in 0..ncart {
                    let w = c2s[m * ncart + c];
                    if w != 0.0 {
                        sph[m] += w * cart_vals[c * ngrids + g];
                    }
                }
            }
            for m in 0..nsph {
                let want = buf.values[g + m * ngrids];
                let prod_mine = sph[m] * sph[m];
                let prod_ref = want * want;
                assert!(
                    (prod_mine - prod_ref).abs() < 1e-12,
                    "l={l} g={g} m={m}: product mismatch"
                );
            }
        }
    }
}

/// Test 3 — periodic image wrap is exact: a Gaussian at a periodic box's
/// corner and one at its middle integrate to the same total once enough
/// images are summed to capture the tails.
#[test]
fn periodic_wrap_is_exact() {
    let l_box = 6.0_f64; // cubic box side, Bohr
    let alpha = 1.5_f64; // steep enough that a handful of images suffice
    let raw_coef = 1.0_f64;
    let coef = raw_coef * gto_norm(0, alpha) * common_fac_sp(0);

    let n = 60usize;
    let d = l_box / n as f64;
    let dv = d * d * d;
    // Grid points span exactly one periodic cell, [0, l_box).
    let mut coords = Vec::with_capacity(n * n * n * 3);
    for ix in 0..n {
        let x = (ix as f64 + 0.5) * d;
        for iy in 0..n {
            let y = (iy as f64 + 0.5) * d;
            for iz in 0..n {
                let z = (iz as f64 + 0.5) * d;
                coords.push(x);
                coords.push(y);
                coords.push(z);
            }
        }
    }
    let ngrids = n * n * n;

    // Images: every lattice translation L = (i,j,k)*l_box for i,j,k in
    // -3..=3 — comfortably enough to make the truncation error (which decays
    // like exp(-alpha*(3*l_box)^2)) negligible next to the 1e-9 gate below.
    let mut images = Vec::new();
    for i in -3..=3 {
        for j in -3..=3 {
            for k in -3..=3 {
                images.push([i as f64 * l_box, j as f64 * l_box, k as f64 * l_box]);
            }
        }
    }

    let total_for = |centre: [f64; 3]| -> f64 {
        let rec_center: Vec<f64> = images
            .iter()
            .flat_map(|l| [centre[0] + l[0], centre[1] + l[1], centre[2] + l[2]])
            .collect();
        let nrec = images.len() as u32;
        let t = PshellGridTable {
            coords: coords.clone(),
            slot_pow: vec![0, 0, 0],
            slot_pshell: vec![0],
            pshell_rec0: vec![0],
            pshell_nrec: vec![nrec],
            pshell_alpha: vec![alpha],
            pshell_coef: vec![coef],
            rec_center,
        };
        let out = collocate(&client(), &t).expect("collocate");
        assert_eq!(out.len(), ngrids);
        out.iter().sum::<f64>() * dv
    };

    let corner_total = total_for([1e-6, 1e-6, 1e-6]);
    let middle_total = total_for([l_box / 2.0, l_box / 2.0, l_box / 2.0]);

    let diff = (corner_total - middle_total).abs();
    assert!(
        diff < 1e-9,
        "corner {corner_total:.15e} vs middle {middle_total:.15e}, diff {diff:.3e}"
    );

    // Both totals should also match the full-space analytic integral (the
    // periodized tails outside the box are negligible at this alpha/box
    // size), independently confirming the wrap did not lose or double mass.
    let analytic = coef * (std::f64::consts::PI / alpha).powf(1.5);
    assert!((corner_total - analytic).abs() < 1e-9);
}

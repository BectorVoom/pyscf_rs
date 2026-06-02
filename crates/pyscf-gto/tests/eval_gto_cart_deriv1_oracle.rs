//! F-02: `GTOval_cart_deriv1` / `GTOval_ip_cart` element-wise verification.
//!
//! The Phase-2/4 wrapper returned `NotYetImplemented{phase:4}` for the
//! cartesian derivative variants ("needs cart `ao_loc`/`nao` kernel surface").
//! This gate exercises the real cartesian deriv1 path.
//!
//! ## Oracle strategy (self-contained, no live Python)
//!
//! libcint defines the spherical AOs as a fixed linear transform of the
//! cartesian ones: `GTOval_sph = CINTc2s_ket_sph · GTOval_cart`, applied
//! per shell/component. `GTOval_sph_deriv1` is already byte-verified against
//! an independent longhand reference (`eval_gto_deriv1_oracle.rs`). So the
//! rigorous anchor here is the **congruence**
//!
//!   c2s · (GTOval_cart_deriv1) == GTOval_sph_deriv1   (all 4 components)
//!
//! computed on H2O/cc-pVDZ (s, p, d shells — exercises the l=2 cartesian
//! block, where ncart=6 ≠ nsph=5). Equality transitively pins the cartesian
//! output to the independent reference. The spherical side runs on the device
//! deriv1 kernel (maxl=2) while the cartesian side runs on the CPU host path —
//! so this is also a cross-implementation check.
//!
//! On top of that, a central finite-difference check pins the cartesian
//! analytic gradient against (value(x+h)−value(x−h))/2h using the cartesian
//! value component itself — a path that does NOT share the c2s transform NOR
//! the analytic-derivative algebra.

use pyscf_core::Unit;
use pyscf_core::raw_layout::{ANG_OF, BAS_SLOTS, NCTR_OF};
use pyscf_gto::{AtomInput, BasisInput, M, MoleBuildArgs, eval_gto};

// ── libcint cart→sph factors for l ≤ 2 (cc-pVDZ max l = 2) ───────────────
// Identical constants to `eval_gto_deriv1_oracle.rs::reference::c2s_coeff`.
fn c2s_coeff(l: u32, m: usize, c: usize) -> f64 {
    const L0: [[f64; 1]; 1] = [[1.0]];
    const L1: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    const L2: [[f64; 6]; 5] = [
        [0.0, 1.092_548_430_592_079_2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.092_548_430_592_079_2, 0.0],
        [
            -0.315_391_565_252_52,
            0.0,
            0.0,
            -0.315_391_565_252_52,
            0.0,
            0.630_783_130_505_04,
        ],
        [0.0, 0.0, 1.092_548_430_592_079_2, 0.0, 0.0, 0.0],
        [
            0.546_274_215_296_039_6,
            0.0,
            0.0,
            -0.546_274_215_296_039_6,
            0.0,
            0.0,
        ],
    ];
    match l {
        0 => L0[m][c],
        1 => L1[m][c],
        2 => L2[m][c],
        _ => panic!("oracle c2s only ports l<=2 (cc-pVDZ max l = 2)"),
    }
}

fn ncart(l: u32) -> usize {
    ((l as usize + 1) * (l as usize + 2)) / 2
}
fn nsph(l: u32) -> usize {
    2 * l as usize + 1
}

fn h2o_ccpvdz() -> pyscf_core::Mole {
    M(MoleBuildArgs {
        atom: AtomInput::String("O 0 0 0; H 0 1.43 -1.11; H 0 -1.43 -1.11".into()),
        basis: BasisInput::Name("cc-pvdz".into()),
        unit: Unit::Bohr,
        max_memory: 4000.0,
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ..Default::default()
    })
    .expect("H2O/cc-pVDZ (s,p,d shells)")
}

fn grid_500() -> Vec<[f64; 3]> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 6.0 - 3.0
    };
    (0..500).map(|_| [next(), next(), next()]).collect()
}

// ── tests ────────────────────────────────────────────────────────────────

#[test]
fn cart_deriv1_no_longer_not_yet_implemented_and_has_cart_shape() {
    let mol = h2o_ccpvdz();
    let r = eval_gto(&mol, "GTOval_cart_deriv1", &[[0.1, 0.2, 0.3]]);
    assert!(
        r.is_ok(),
        "GTOval_cart_deriv1 must return a real result, got {r:?}"
    );
    let out = r.unwrap();

    // cartesian nao: Σ shells ncart(l)·nctr, strictly larger than spherical
    // here because cc-pVDZ has a d-shell (ncart=6 > nsph=5).
    let nbas = mol._bas.len() / BAS_SLOTS;
    let mut nao_cart = 0usize;
    for s in 0..nbas {
        let l = mol._bas[s * BAS_SLOTS + ANG_OF] as u32;
        let nctr = mol._bas[s * BAS_SLOTS + NCTR_OF] as usize;
        nao_cart += ncart(l) * nctr;
    }
    assert!(
        nao_cart > mol.nao_nr,
        "cc-pVDZ has a d-shell so nao_cart ({nao_cart}) must exceed nao_sph ({})",
        mol.nao_nr
    );
    assert_eq!(out.shape, vec![4, 1, nao_cart]);
    assert_eq!(out.values.len(), 4 * nao_cart);
}

#[test]
fn cart_deriv1_c2s_transform_equals_sph_deriv1() {
    let mol = h2o_ccpvdz();
    let coords = grid_500();
    let ngrids = coords.len();
    let nao_sph = mol.nao_nr;

    let cart = eval_gto(&mol, "GTOval_cart_deriv1", &coords).expect("cart deriv1 ok");
    let sph = eval_gto(&mol, "GTOval_sph_deriv1", &coords).expect("sph deriv1 ok");

    let nbas = mol._bas.len() / BAS_SLOTS;
    let nao_cart = cart.shape[2];
    let cart_stride = ngrids * nao_cart;
    let sph_stride = ngrids * nao_sph;

    let mut max_abs = 0.0_f64;
    let mut checked = 0usize;
    // Walk shells, tracking parallel cartesian and spherical AO offsets, and
    // diff c2s(cart) against sph element-wise over every grid point.
    let mut cart_off = 0usize;
    let mut sph_off = 0usize;
    for s in 0..nbas {
        let l = mol._bas[s * BAS_SLOTS + ANG_OF] as u32;
        let nctr = mol._bas[s * BAS_SLOTS + NCTR_OF] as usize;
        let nc = ncart(l);
        let ns = nsph(l);
        for c in 0..nctr {
            for comp in 0..4 {
                for m in 0..ns {
                    let sph_ao = sph_off + c * ns + m;
                    for g in 0..ngrids {
                        let mut acc = 0.0_f64;
                        for ci in 0..nc {
                            let cart_ao = cart_off + c * nc + ci;
                            acc += c2s_coeff(l, m, ci)
                                * cart.values[comp * cart_stride + g + cart_ao * ngrids];
                        }
                        let want = sph.values[comp * sph_stride + g + sph_ao * ngrids];
                        let d = (acc - want).abs();
                        if d > max_abs {
                            max_abs = d;
                        }
                        checked += 1;
                        assert!(
                            d < 1e-10,
                            "shell {s} comp {comp} m {m} g {g}: c2s(cart)={acc:.17e} \
                             sph={want:.17e} |Δ|={d:.3e}"
                        );
                    }
                }
            }
        }
        cart_off += nc * nctr;
        sph_off += ns * nctr;
    }
    assert!(checked > 4 * ngrids, "congruence oracle did not run");
    eprintln!(
        "cart_deriv1 c2s-congruence: max |Δ| vs sph_deriv1 = {max_abs:.3e} ({checked} elems)"
    );
}

#[test]
fn ip_cart_equals_cart_deriv1_gradient_block() {
    // GTOval_ip_cart == GTOval_cart_deriv1[1:4], bit-for-bit.
    let mol = h2o_ccpvdz();
    let coords = grid_500();
    let ngrids = coords.len();

    let d1 = eval_gto(&mol, "GTOval_cart_deriv1", &coords).expect("cart deriv1");
    let ip = eval_gto(&mol, "GTOval_ip_cart", &coords).expect("ip cart");

    let nao_cart = d1.shape[2];
    assert_eq!(ip.shape, vec![3, ngrids, nao_cart]);
    let block = ngrids * nao_cart;
    for i in 0..(3 * block) {
        assert_eq!(
            ip.values[i].to_bits(),
            d1.values[block + i].to_bits(),
            "GTOval_ip_cart[{i}] must equal GTOval_cart_deriv1 grad block bit-for-bit"
        );
    }
}

#[test]
fn cart_deriv1_gradient_matches_finite_difference() {
    // Independent of c2s AND of the analytic-derivative algebra: central
    // finite difference of the cartesian VALUE component (comp 0) vs the
    // cartesian analytic gradient (comps 1,2,3).
    let mol = h2o_ccpvdz();
    let p = [0.31_f64, -0.52, 0.73];
    let h = 1e-5;

    let d1 = eval_gto(&mol, "GTOval_cart_deriv1", &[p]).expect("cart deriv1");
    let nao_cart = d1.shape[2]; // ngrids = 1 → idx = ao within each comp.

    for (axis, comp) in [(0usize, 1usize), (1, 2), (2, 3)] {
        let mut pp = p;
        let mut pm = p;
        pp[axis] += h;
        pm[axis] -= h;
        let vp = eval_gto(&mol, "GTOval_cart_deriv1", &[pp]).expect("v+");
        let vm = eval_gto(&mol, "GTOval_cart_deriv1", &[pm]).expect("v-");
        for ao in 0..nao_cart {
            // value component lives in block 0.
            let fd = (vp.values[ao] - vm.values[ao]) / (2.0 * h);
            let an = d1.values[comp * nao_cart + ao];
            let tol = 1e-6 * (1.0 + an.abs());
            assert!(
                (fd - an).abs() < tol,
                "axis {axis} cart ao {ao}: finite-diff {fd:.6e} vs analytic {an:.6e}"
            );
        }
    }
}

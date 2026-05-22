//! DFT-10 (plan 04-03 Task 2): `GTOval_sph_deriv1` element-wise parity.
//!
//! `GTOval_sph_deriv1` returns 4 stacked components [value, ∂/∂x, ∂/∂y,
//! ∂/∂z] per AO per grid point — the GGA grid-loop input for ∇ρ
//! (σ = |∇ρ|²). The Phase 2 wrapper returned `NotYetImplemented{phase:4}`;
//! this plan wires the real kernel.
//!
//! ## Oracle strategy (self-contained, no live Python)
//!
//! `tests/oracle/test_eval_gto.py` needs numpy + pyscf; the pure-Rust
//! `cargo test --profile release-oracle -p pyscf-gto eval_gto_deriv1_oracle`
//! gate must pass without them. So this test diffs against an
//! **independent** reference derived from the upstream C kernel
//! (`pyscf/lib/gto/deriv1.c`):
//!   - value  = x^lx y^ly z^lz · R         (GTOshell_eval_grid_cart)
//!   - ∂/∂q   = e2a·q·(x^lx y^ly z^lz) + e·lq·(…q^(lq-1)…)
//!     (GTOshell_eval_grid_ip_cart, general-l `default` branch)
//! where R = Σ_p c·exp(−α r²)·CINTcommon_fac_sp(l) and
//! e2a = Σ_p (−2α)·c·exp(−α r²)·CINTcommon_fac_sp(l). Both sides then
//! apply the libcint g_trans_cart2sph transform per component.
//!
//! The reference re-derives via a longhand cartesian loop + explicit c2s
//! matrices — a DIFFERENT code path from `eval_gto_sph_deriv1`. A
//! wrong-sign / wrong-stencil / wrong-ordering bug surfaces as an
//! element-wise mismatch. Tolerance is the 1e-10 chemical-accuracy
//! budget; the cross-implementation diff is actually ~1e-12 (f64 both
//! sides, release-oracle FMA-free).
//!
//! On top of the diff, a finite-difference check pins the analytic
//! gradient against (value(x+h)−value(x−h))/2h — a check that does NOT
//! share the analytic-derivative algebra at all.

use pyscf_core::raw_layout::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
};
use pyscf_core::Unit;
use pyscf_gto::{eval_gto, AtomInput, BasisInput, MoleBuildArgs, M};

// ── independent reference (longhand, different code path) ───────────────

mod reference {
    pub fn common_fac_sp(l: u32) -> f64 {
        match l {
            0 => 0.282094791773878143,
            1 => 0.488602511902919921,
            _ => 1.0,
        }
    }
    pub fn ncart(l: u32) -> usize {
        ((l as usize + 1) * (l as usize + 2)) / 2
    }
    pub fn nsph(l: u32) -> usize {
        2 * l as usize + 1
    }
    pub fn cart_powers(l: u32) -> Vec<(u32, u32, u32)> {
        let mut v = Vec::with_capacity(ncart(l));
        let li = l as i32;
        let mut lx = li;
        while lx >= 0 {
            let mut ly = li - lx;
            while ly >= 0 {
                v.push((lx as u32, ly as u32, (li - lx - ly) as u32));
                ly -= 1;
            }
            lx -= 1;
        }
        v
    }
    pub fn c2s_coeff(l: u32, m: usize, c: usize) -> f64 {
        const L0: [[f64; 1]; 1] = [[1.0]];
        const L1: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        const L2: [[f64; 6]; 5] = [
            [0.0, 1.092548430592079070, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.092548430592079070, 0.0],
            [-0.315391565252520002, 0.0, 0.0, -0.315391565252520002, 0.0, 0.630783130505040012],
            [0.0, 0.0, 1.092548430592079070, 0.0, 0.0, 0.0],
            [0.546274215296039535, 0.0, 0.0, -0.546274215296039535, 0.0, 0.0],
        ];
        match l {
            0 => L0[m][c],
            1 => L1[m][c],
            2 => L2[m][c],
            _ => panic!("reference c2s only ports l<=2 (cc-pVDZ max l = 2)"),
        }
    }

    /// pow with the upstream convention `lq * q^(lq-1)` = 0 for lq=0.
    fn dpow(q: f64, lq: u32) -> f64 {
        if lq == 0 {
            0.0
        } else {
            lq as f64 * q.powi(lq as i32 - 1)
        }
    }

    /// Independent deriv1 reference. Returns 4 flat F-order `(ngrids,nao)`
    /// buffers [value, d/dx, d/dy, d/dz], indexed `g + ao*ngrids`.
    #[allow(clippy::too_many_arguments)]
    pub fn deriv1(
        coords_flat: &[f64],
        ngrids: usize,
        atm: &[i32],
        bas: &[i32],
        env: &[f64],
        ao_loc: &[i32],
        nao: usize,
    ) -> [Vec<f64>; 4] {
        use super::*;
        let nbas = bas.len() / BAS_SLOTS;
        let mut out = [
            vec![0.0_f64; ngrids * nao],
            vec![0.0_f64; ngrids * nao],
            vec![0.0_f64; ngrids * nao],
            vec![0.0_f64; ngrids * nao],
        ];
        for g in 0..ngrids {
            let gx = coords_flat[g];
            let gy = coords_flat[g + ngrids];
            let gz = coords_flat[g + 2 * ngrids];
            for s in 0..nbas {
                let row = s * BAS_SLOTS;
                let atom_id = bas[row + ATOM_OF] as usize;
                let l = bas[row + ANG_OF] as u32;
                let nprim = bas[row + NPRIM_OF] as usize;
                let nctr = bas[row + NCTR_OF] as usize;
                let ptr_exp = bas[row + PTR_EXP] as usize;
                let ptr_coeff = bas[row + PTR_COEFF] as usize;
                let pc = atm[atom_id * ATM_SLOTS + PTR_COORD] as usize;
                let (dx, dy, dz) = (gx - env[pc], gy - env[pc + 1], gz - env[pc + 2]);
                let r2 = dx * dx + dy * dy + dz * dz;

                let fac1 = common_fac_sp(l);
                let powers = cart_powers(l);
                let ncart = ncart(l);
                let nsph = nsph(l);
                let ao_off = ao_loc[s] as usize;

                for c in 0..nctr {
                    let mut radial = 0.0_f64; // e
                    let mut radial_2a = 0.0_f64; // e2a = Σ -2α c exp
                    for p in 0..nprim {
                        let alpha = env[ptr_exp + p];
                        let coef = env[ptr_coeff + c * nprim + p];
                        let g0 = coef * (-alpha * r2).exp();
                        radial += g0;
                        radial_2a += -2.0 * alpha * g0;
                    }
                    radial *= fac1;
                    radial_2a *= fac1;

                    // cartesian value + 3 grads
                    let mut cval = vec![0.0_f64; ncart];
                    let mut cdx = vec![0.0_f64; ncart];
                    let mut cdy = vec![0.0_f64; ncart];
                    let mut cdz = vec![0.0_f64; ncart];
                    for (ci, &(lx, ly, lz)) in powers.iter().enumerate() {
                        let mono =
                            dx.powi(lx as i32) * dy.powi(ly as i32) * dz.powi(lz as i32);
                        cval[ci] = mono * radial;
                        cdx[ci] = radial_2a * dx * mono
                            + radial * dpow(dx, lx) * dy.powi(ly as i32) * dz.powi(lz as i32);
                        cdy[ci] = radial_2a * dy * mono
                            + radial * dx.powi(lx as i32) * dpow(dy, ly) * dz.powi(lz as i32);
                        cdz[ci] = radial_2a * dz * mono
                            + radial * dx.powi(lx as i32) * dy.powi(ly as i32) * dpow(dz, lz);
                    }
                    // cart -> sph per component
                    for m in 0..nsph {
                        let ao_idx = ao_off + c * nsph + m;
                        let mut v = 0.0;
                        let mut vx = 0.0;
                        let mut vy = 0.0;
                        let mut vz = 0.0;
                        for ci in 0..ncart {
                            let t = c2s_coeff(l, m, ci);
                            v += t * cval[ci];
                            vx += t * cdx[ci];
                            vy += t * cdy[ci];
                            vz += t * cdz[ci];
                        }
                        out[0][g + ao_idx * ngrids] = v;
                        out[1][g + ao_idx * ngrids] = vx;
                        out[2][g + ao_idx * ngrids] = vy;
                        out[3][g + ao_idx * ngrids] = vz;
                    }
                }
            }
        }
        out
    }
}

// ── helpers ─────────────────────────────────────────────────────────────

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

fn grid_1000() -> Vec<[f64; 3]> {
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 6.0 - 3.0
    };
    (0..1000).map(|_| [next(), next(), next()]).collect()
}

fn pack(coords: &[[f64; 3]]) -> Vec<f64> {
    let n = coords.len();
    let mut flat = Vec::with_capacity(n * 3);
    flat.extend(coords.iter().map(|c| c[0]));
    flat.extend(coords.iter().map(|c| c[1]));
    flat.extend(coords.iter().map(|c| c[2]));
    flat
}

// ── tests ───────────────────────────────────────────────────────────────

#[test]
fn eval_gto_deriv1_oracle_no_longer_returns_not_yet_implemented() {
    // Behavior assertion: the Phase 2 NotYetImplemented{phase:4} arm is gone.
    let mol = h2o_ccpvdz();
    let r = eval_gto(&mol, "GTOval_sph_deriv1", &[[0.1, 0.2, 0.3]]);
    assert!(
        r.is_ok(),
        "GTOval_sph_deriv1 must return a real result, got {r:?}"
    );
    let out = r.unwrap();
    // shape [4, ngrids, nao]
    assert_eq!(out.shape, vec![4, 1, mol.nao_nr]);
    assert_eq!(out.values.len(), 4 * mol.nao_nr);
}

#[test]
fn eval_gto_deriv1_oracle_matches_reference_on_1000_point_grid() {
    let mol = h2o_ccpvdz();
    let coords = grid_1000();
    let ngrids = coords.len();
    let nao = mol.nao_nr;

    let out = eval_gto(&mol, "GTOval_sph_deriv1", &coords).expect("deriv1 ok");
    assert_eq!(out.shape, vec![4, ngrids, nao]);
    assert_eq!(out.values.len(), 4 * ngrids * nao);

    let flat = pack(&coords);
    let expect = reference::deriv1(
        &flat, ngrids, &mol._atm, &mol._bas, &mol._env, &mol.ao_loc_nr, nao,
    );

    let comp_names = ["value", "d/dx", "d/dy", "d/dz"];
    let mut max_abs = 0.0_f64;
    let mut nonzero_grad = 0usize;
    for comp in 0..4 {
        let base = comp * ngrids * nao;
        for idx in 0..(ngrids * nao) {
            let a = out.values[base + idx];
            let b = expect[comp][idx];
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
            }
            if comp >= 1 && b.abs() > 1e-12 {
                nonzero_grad += 1;
            }
            assert!(
                d < 1e-10,
                "comp={} ({}) idx={idx}: kernel={a:.17e} ref={b:.17e} |Δ|={d:.3e}",
                comp,
                comp_names[comp]
            );
        }
    }
    assert!(
        nonzero_grad > ngrids,
        "gradient components nearly all zero ({nonzero_grad}) — deriv1 stencil not exercised"
    );
    eprintln!("eval_gto_deriv1_oracle: max |Δ| vs independent reference = {max_abs:.3e}");
}

#[test]
fn eval_gto_deriv1_oracle_value_component_equals_plain_gtoval_sph() {
    // Component 0 of deriv1 must be byte-identical to GTOval_sph.
    let mol = h2o_ccpvdz();
    let coords = grid_1000();
    let ngrids = coords.len();
    let nao = mol.nao_nr;

    let val = eval_gto(&mol, "GTOval_sph", &coords).expect("value ok");
    let d1 = eval_gto(&mol, "GTOval_sph_deriv1", &coords).expect("deriv1 ok");

    for idx in 0..(ngrids * nao) {
        assert_eq!(
            val.values[idx].to_bits(),
            d1.values[idx].to_bits(),
            "deriv1 component 0 must equal GTOval_sph bit-for-bit at idx {idx}"
        );
    }
}

#[test]
fn eval_gto_deriv1_oracle_gradient_matches_finite_difference() {
    // Independent of the analytic-derivative algebra: central finite
    // difference of GTOval_sph values vs the deriv1 analytic gradient.
    let mol = h2o_ccpvdz();
    let nao = mol.nao_nr;
    let p = [0.31_f64, -0.52, 0.73];
    let h = 1e-5;

    let d1 = eval_gto(&mol, "GTOval_sph_deriv1", &[p]).expect("deriv1");
    // analytic grads live at components 1,2,3, ngrids=1 → idx = ao.
    for (axis, comp) in [(0usize, 1usize), (1, 2), (2, 3)] {
        let mut pp = p;
        let mut pm = p;
        pp[axis] += h;
        pm[axis] -= h;
        let vp = eval_gto(&mol, "GTOval_sph", &[pp]).expect("v+");
        let vm = eval_gto(&mol, "GTOval_sph", &[pm]).expect("v-");
        for ao in 0..nao {
            let fd = (vp.values[ao] - vm.values[ao]) / (2.0 * h);
            let an = d1.values[comp * nao + ao];
            // FD truncation ~h² ≈ 1e-10 scaled by curvature; use a loose
            // absolute+relative envelope.
            let tol = 1e-6 * (1.0 + an.abs());
            assert!(
                (fd - an).abs() < tol,
                "axis {axis} ao {ao}: finite-diff {fd:.6e} vs analytic {an:.6e}"
            );
        }
    }
}

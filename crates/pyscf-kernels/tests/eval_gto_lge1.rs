//! DFT-01 / D-04 deferral close-out (plan 04-03 Task 1): `l >= 1`
//! AO-on-grid evaluation in `pyscf_kernels::eval_gto_sph`.
//!
//! ## Oracle strategy (self-contained, no live Python)
//!
//! The Phase 2 `tests/oracle/test_eval_gto.py` harness diffs against a
//! live upstream `mol.eval_gto("GTOval_sph", coords)`. That harness needs
//! numpy + an importable `pyscf`; the pure-Rust `cargo test --profile
//! release-oracle -p pyscf-kernels eval_gto_lge1` gate must pass WITHOUT
//! those, so this test embeds an **independent** reference computed from
//! the upstream C kernel algorithm (`pyscf/lib/gto/deriv1.c
//! GTOshell_eval_grid_cart` + `CINTc2s_ket_sph`, i.e. libcint
//! `g_trans_cart2sph[]`).
//!
//! "Independent" = the reference here re-derives AO values via a naive
//! triple-nested cartesian-monomial loop with the c2s matrices written
//! out longhand (`reference::*`), a DIFFERENT code path from the
//! production `eval_gto_sph` (which routes the radial sum through
//! `oracle_sum` and shares one cartesian/c2s helper). A wrong-convention
//! bug in the kernel (AO ordering vs `ao_loc`, transposed c2s matrix,
//! Condon-Shortley sign) shows up as an element-wise mismatch.
//!
//! ## Why hand-built `_atm/_bas/_env`
//!
//! `pyscf-kernels` is BELOW `pyscf-gto` in the dep graph (the algebra
//! wall / D-04 inversion) — it cannot depend on `pyscf-gto::M` without a
//! cycle. So the fixture builds the libcint flat arrays directly with
//! libcint-convention normalised coefficients (`gto_norm`), exercising s,
//! p AND d shells. This keeps the kernel test fully self-contained.
//!
//! On top of the cross-implementation diff the test pins convention-
//! locking analytic identities (p sph ∝ (x,y,z); d m=0 ∝ 2z²−x²−y²;
//! T-04-03 AO-ordering mitigation).

use pyscf_core::raw_layout::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_ENV_START,
    PTR_EXP,
};
use pyscf_kernels::eval_gto_sph;

// ── libcint-convention basis fixture (hand-built flat arrays) ───────────

/// `gto_norm(l, α)` — libcint per-primitive radial normaliser
/// (`pyscf/gto/mole.py:125`). `1 / sqrt(gaussian_int(2l+2, 2α))`.
fn gto_norm(l: u32, alpha: f64) -> f64 {
    1.0 / gaussian_int(2 * l as i32 + 2, 2.0 * alpha).sqrt()
}

/// `gaussian_int(n, α) = ∫ r^n exp(-α r²) dr = Γ((n+1)/2) / (2 α^((n+1)/2))`
/// (`pyscf/gto/mole.py:`). Uses `libm::tgamma` via std-free closed form for
/// half-integer args is avoided; we only need integer/half-integer n here,
/// computed through the recurrence-free gamma in `gamma_half`.
fn gaussian_int(n: i32, alpha: f64) -> f64 {
    let n1 = (n as f64 + 1.0) * 0.5;
    gamma(n1) / (2.0 * alpha.powf(n1))
}

/// Γ(x) for x ∈ {positive half-integers, integers} via Lanczos. Plenty
/// accurate for normalisation constants (test fixture, not a hot path).
fn gamma(x: f64) -> f64 {
    // Lanczos approximation, g=7, n=9.
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

/// One contracted shell spec for the fixture builder.
struct ShellSpec {
    l: u32,
    /// (exponent, raw contraction coefficient) pairs — single contraction.
    prims: Vec<(f64, f64)>,
}

/// Hand-built libcint flat arrays for a single atom at `centre` carrying
/// the given shells. Coefficients are normalised with `gto_norm` exactly
/// as `pyscf-gto::make_env` / upstream do, so `eval_gto_sph`'s
/// `CINTcommon_fac_sp` multiplier reproduces upstream values.
struct Fixture {
    atm: Vec<i32>,
    bas: Vec<i32>,
    env: Vec<f64>,
    ao_loc: Vec<i32>,
    nao: usize,
}

fn build_fixture(centre: [f64; 3], shells: &[ShellSpec]) -> Fixture {
    let mut env: Vec<f64> = vec![0.0; PTR_ENV_START];
    // atom coordinate block
    let ptr_coord = env.len();
    env.extend_from_slice(&centre);

    let mut atm = vec![0i32; ATM_SLOTS];
    atm[ATOM_OF] = 0; // unused for eval
    atm[PTR_COORD] = ptr_coord as i32;
    // charge slot etc. irrelevant for eval_gto.

    let mut bas: Vec<i32> = Vec::new();
    let mut ao_loc: Vec<i32> = Vec::new();
    let mut ao_cursor: i32 = 0;

    for sh in shells {
        let nprim = sh.prims.len();
        // exponents
        let ptr_exp = env.len();
        for &(a, _) in &sh.prims {
            env.push(a);
        }
        // normalised coefficients (single contraction column)
        let ptr_coeff = env.len();
        for &(a, c) in &sh.prims {
            env.push(c * gto_norm(sh.l, a));
        }

        let mut row = vec![0i32; BAS_SLOTS];
        row[ATOM_OF] = 0;
        row[ANG_OF] = sh.l as i32;
        row[NPRIM_OF] = nprim as i32;
        row[NCTR_OF] = 1;
        row[PTR_EXP] = ptr_exp as i32;
        row[PTR_COEFF] = ptr_coeff as i32;
        bas.extend_from_slice(&row);

        ao_loc.push(ao_cursor);
        ao_cursor += (2 * sh.l as i32) + 1; // spherical degeneracy
    }
    ao_loc.push(ao_cursor); // ao_loc has nbas+1 entries

    Fixture {
        atm,
        bas,
        env,
        ao_loc,
        nao: ao_cursor as usize,
    }
}

/// s+p+d fixture on a single centre (mimics an O-like [1s1p1d] block).
fn spd_fixture() -> Fixture {
    build_fixture(
        [0.1, -0.2, 0.3],
        &[
            ShellSpec {
                l: 0,
                prims: vec![(5.0, 0.5), (1.2, 0.6)],
            },
            ShellSpec {
                l: 1,
                prims: vec![(3.1, 0.4), (0.9, 0.7)],
            },
            ShellSpec {
                l: 2,
                prims: vec![(2.3, 1.0)],
            },
        ],
    )
}

// ── Independent reference (different code path from the kernel) ─────────

mod reference {
    pub fn common_fac_sp(l: u32) -> f64 {
        match l {
            0 => 0.282_094_791_773_878_14,
            1 => 0.488_602_511_902_919_9,
            _ => 1.0,
        }
    }
    pub fn ncart(l: u32) -> usize {
        ((l as usize + 1) * (l as usize + 2)) / 2
    }
    pub fn nsph(l: u32) -> usize {
        2 * l as usize + 1
    }
    /// (lx,ly,lz) for cart col `c` in upstream order: lx=l..0, ly=l-lx..0.
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
    /// libcint g_trans_cart2sph (mirrors cintx-cubecl C2S_L*).
    pub fn c2s_coeff(l: u32, m: usize, c: usize) -> f64 {
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
            _ => panic!("reference c2s only ports l<=2 (fixture max l = 2)"),
        }
    }
}

fn reference_gtoval_sph(
    coords_flat: &[f64],
    ngrids: usize,
    atm: &[i32],
    bas: &[i32],
    env: &[f64],
    ao_loc: &[i32],
    nao: usize,
) -> Vec<f64> {
    let nbas = bas.len() / BAS_SLOTS;
    let mut out = vec![0.0_f64; ngrids * nao];
    for g in 0..ngrids {
        let gx = coords_flat[g];
        let gy = coords_flat[g + ngrids];
        let gz = coords_flat[g + 2 * ngrids];
        // Reference kernel: `s` drives parallel flat-array offsets (mirrors
        // the production loop in `eval_gto.rs`).
        #[allow(clippy::needless_range_loop)]
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

            let fac1 = reference::common_fac_sp(l);
            let powers = reference::cart_powers(l);
            let ncart = reference::ncart(l);
            let nsph = reference::nsph(l);
            let ao_off = ao_loc[s] as usize;

            for c in 0..nctr {
                let mut radial = 0.0_f64;
                for p in 0..nprim {
                    let alpha = env[ptr_exp + p];
                    let coef = env[ptr_coeff + c * nprim + p];
                    radial += coef * (-alpha * r2).exp();
                }
                radial *= fac1;
                let mut cart = vec![0.0_f64; ncart];
                for (ci, &(lx, ly, lz)) in powers.iter().enumerate() {
                    cart[ci] =
                        dx.powi(lx as i32) * dy.powi(ly as i32) * dz.powi(lz as i32) * radial;
                }
                for m in 0..nsph {
                    let mut v = 0.0_f64;
                    #[allow(clippy::needless_range_loop)]
                    for ci in 0..ncart {
                        v += reference::c2s_coeff(l, m, ci) * cart[ci];
                    }
                    out[g + (ao_off + c * nsph + m) * ngrids] = v;
                }
            }
        }
    }
    out
}

// ── helpers ─────────────────────────────────────────────────────────────

fn cpu_client() -> pyscf_algebra::AlgebraClient {
    pyscf_algebra::select_backend()
        .expect("CPU backend selection")
        .client
}

fn pack_coords(coords: &[[f64; 3]]) -> (Vec<f64>, usize) {
    let n = coords.len();
    let mut flat = Vec::with_capacity(n * 3);
    flat.extend(coords.iter().map(|c| c[0]));
    flat.extend(coords.iter().map(|c| c[1]));
    flat.extend(coords.iter().map(|c| c[2]));
    (flat, n)
}

/// 1000-point deterministic grid (LCG, no RNG dep).
fn grid_1000(centre: [f64; 3]) -> Vec<[f64; 3]> {
    let mut state: u64 = 0x2545_F491_4F6C_DD1D;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 / (1u64 << 53) as f64) * 6.0 - 3.0
    };
    (0..1000)
        .map(|_| [centre[0] + next(), centre[1] + next(), centre[2] + next()])
        .collect()
}

// ── tests ───────────────────────────────────────────────────────────────

#[test]
fn eval_gto_lge1_fixture_exercises_p_and_d_shells() {
    let fx = spd_fixture();
    let nbas = fx.bas.len() / BAS_SLOTS;
    let mut has_p = false;
    let mut has_d = false;
    for s in 0..nbas {
        let l = fx.bas[s * BAS_SLOTS + ANG_OF];
        has_p |= l == 1;
        has_d |= l == 2;
    }
    assert!(
        has_p && has_d,
        "fixture must contain p (l=1) and d (l=2) shells"
    );
    // s(1) + p(3) + d(5) = 9 AOs
    assert_eq!(fx.nao, 9);
}

#[test]
fn eval_gto_lge1_gtoval_sph_matches_reference_on_1000_point_grid() {
    let fx = spd_fixture();
    let coords = grid_1000([0.1, -0.2, 0.3]);
    let (flat, ngrids) = pack_coords(&coords);

    let client = cpu_client();
    let got = eval_gto_sph(
        &client, &flat, ngrids, &fx.atm, &fx.bas, &fx.env, &fx.ao_loc, fx.nao, true,
    );
    let expect = reference_gtoval_sph(&flat, ngrids, &fx.atm, &fx.bas, &fx.env, &fx.ao_loc, fx.nao);

    assert_eq!(got.shape, vec![ngrids, fx.nao]);
    assert_eq!(got.values.len(), expect.len());

    let mut max_abs = 0.0_f64;
    let mut nonzero = 0usize;
    for (i, (&a, &b)) in got.values.iter().zip(expect.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        if b.abs() > 1e-12 {
            nonzero += 1;
        }
        assert!(
            d < 1e-10,
            "element {i} mismatch: kernel={a:.17e} ref={b:.17e} |Δ|={d:.3e}"
        );
    }
    // p+d AOs (8 of 9) are nonzero across ~all 1000 grid points → ≫ ngrids.
    assert!(
        nonzero > ngrids,
        "reference produced too few nonzero AOs ({nonzero}) — l>=1 path not exercised"
    );
    eprintln!("eval_gto_lge1: max |Δ| vs independent reference = {max_abs:.3e}");
}

#[test]
fn eval_gto_lge1_p_shell_sph_components_proportional_to_xyz() {
    // Convention lock (independent of the radial sum): p sph = (px,py,pz)
    // ∝ (Δx,Δy,Δz) with one shared radial factor. Catches transposed/
    // permuted c2s or wrong AO ordering vs ao_loc (T-04-03).
    let fx = spd_fixture();
    let nbas = fx.bas.len() / BAS_SLOTS;
    let mut p_shell = None;
    for s in 0..nbas {
        if fx.bas[s * BAS_SLOTS + ANG_OF] == 1 {
            p_shell = Some(s);
            break;
        }
    }
    let s = p_shell.expect("p shell present");
    let atom_id = fx.bas[s * BAS_SLOTS + ATOM_OF] as usize;
    let pc = fx.atm[atom_id * ATM_SLOTS + PTR_COORD] as usize;
    let centre = [fx.env[pc], fx.env[pc + 1], fx.env[pc + 2]];
    let ao_off = fx.ao_loc[s] as usize;

    let (dx, dy, dz) = (0.37_f64, -0.51_f64, 0.84_f64);
    let coords = vec![[centre[0] + dx, centre[1] + dy, centre[2] + dz]];
    let (flat, ngrids) = pack_coords(&coords);
    let out = eval_gto_sph(
        &cpu_client(),
        &flat,
        ngrids,
        &fx.atm,
        &fx.bas,
        &fx.env,
        &fx.ao_loc,
        fx.nao,
        true,
    );

    let px = out.values[(ao_off) * ngrids];
    let py = out.values[(ao_off + 1) * ngrids];
    let pz = out.values[(ao_off + 2) * ngrids];
    let (rx, ry, rz) = (px / dx, py / dy, pz / dz);
    approx::assert_abs_diff_eq!(rx, ry, epsilon = 1e-12);
    approx::assert_abs_diff_eq!(rx, rz, epsilon = 1e-12);
    assert!(rx.abs() > 1e-6, "p radial factor nonzero ({rx})");
}

#[test]
fn eval_gto_lge1_d_shell_sph_ratio_matches_real_solid_harmonics() {
    // d m=0 (dz2) ∝ −0.31539(x²+y²)+0.63078 z²; d m=+2 (dx²−y²) ∝
    // 0.54627(x²−y²). Their ratio is radial-independent — a pure
    // c2s-matrix / sign check.
    let fx = spd_fixture();
    let nbas = fx.bas.len() / BAS_SLOTS;
    let mut d_shell = None;
    for s in 0..nbas {
        if fx.bas[s * BAS_SLOTS + ANG_OF] == 2 {
            d_shell = Some(s);
            break;
        }
    }
    let s = d_shell.expect("d shell present");
    let atom_id = fx.bas[s * BAS_SLOTS + ATOM_OF] as usize;
    let pc = fx.atm[atom_id * ATM_SLOTS + PTR_COORD] as usize;
    let centre = [fx.env[pc], fx.env[pc + 1], fx.env[pc + 2]];
    let ao_off = fx.ao_loc[s] as usize;

    let (x, y, z) = (0.6_f64, 0.3_f64, -0.9_f64);
    let coords = vec![[centre[0] + x, centre[1] + y, centre[2] + z]];
    let (flat, ngrids) = pack_coords(&coords);
    let out = eval_gto_sph(
        &cpu_client(),
        &flat,
        ngrids,
        &fx.atm,
        &fx.bas,
        &fx.env,
        &fx.ao_loc,
        fx.nao,
        true,
    );

    // m order -2,-1,0,+1,+2: m=0 at +2, m=+2 at +4.
    let dz2 = out.values[(ao_off + 2) * ngrids];
    let dx2y2 = out.values[(ao_off + 4) * ngrids];
    let num = -0.315_391_565_252_52 * (x * x + y * y) + 0.630_783_130_505_04 * z * z;
    let den = 0.546_274_215_296_039_6 * (x * x - y * y);
    approx::assert_abs_diff_eq!(dz2 / dx2y2, num / den, epsilon = 1e-9);
}

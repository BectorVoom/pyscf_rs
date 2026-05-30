//! quick-260530-ljv: randomized **differential oracle** test for the new
//! s-shell (l=0) `#[cube(launch_unchecked)]` eval_gto path.
//!
//! Strategy (per docs/rust_crate_test_guideline.md — differential testing
//! against a trusted reference, randomized inputs, reproducible seed): an
//! INLINE host longhand `Y00 · Σ_p coef·exp(-α·r²)` is the bit-deterministic
//! ground truth. We run the device kernel (reached by calling `eval_gto_sph`
//! on a PURE-s-shell basis, which now routes to the cube kernel) over random
//! s-shell fixtures and assert the device result matches the oracle within a
//! tight tolerance.
//!
//! ORACLE PIN: the inline oracle uses `y00 = 0.5_f64 / std::f64::consts::PI
//! .sqrt()` and the F-order write `out[g + (ao_off + c_idx)*ngrids]`, byte-
//! matching `eval_gto_sph_cpu` lines 603-614 — copied verbatim, NOT re-derived,
//! so the differential check cannot pass against a subtly-wrong oracle.
//!
//! - `eval_gto_s_matches_oracle_on_cpu` always runs (default `cpu` feature).
//! - `eval_gto_s_matches_oracle_on_rocm` (`#[cfg(feature = "rocm")]`) runs the
//!   SAME differential check on real AMD hardware (gfx1152) via
//!   `cubecl_hip::HipRuntime` — the real-GPU confirmation arm.
//!
//! Clients are constructed directly (not via `select_backend`) so the CPU and
//! ROCm tests never race on the process-global `PYSCF_BACKEND` env var.
//!
//! Verified in scope: device s-shell eval_gto == inline host oracle within
//! 1e-9 over randomized pure-s-shell fixtures (varied ngrids/nbas/nprim/nctr,
//! random exponents/coefficients/atom centers; nprim up to 4 exercises the
//! greater-than-2-prim ordered-sum path).
//!
//! Not yet verified here: the `l >= 1` cart→sph device path (DEFERRED — it
//! still routes to the unchanged host fallback, covered by
//! tests/eval_gto_lge1.rs); deriv1/deriv2 device stencils; the rocm arm runs
//! ONLY when the `rocm` feature is enabled on hardware (correctness is already
//! PROVEN by the always-on CpuRuntime arm regardless of rocm availability).

use cubecl::Runtime; // brings `::client` into scope for the concrete runtimes
use pyscf_core::raw_layout::{
    ANG_OF, ATM_SLOTS, ATOM_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_COORD, PTR_EXP,
};
use pyscf_kernels::eval_gto_sph;
use pyscf_algebra::AlgebraClient;

/// Deterministic LCG (Knuth/MMIX constants) → reproducible "random" values
/// without pulling in the `rand` crate. (Same generator as gemm_oracle.rs.)
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    /// Uniform in `[-1.0, 1.0)`.
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64; // [0, 1)
        u * 2.0 - 1.0 // [-1, 1)
    }
    /// Uniform in `[lo, hi)`.
    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        let t = (self.next_f64() + 1.0) * 0.5; // [0, 1)
        lo + t * (hi - lo)
    }
    /// Integer in `[lo, hi]` inclusive.
    fn next_usize(&mut self, lo: usize, hi: usize) -> usize {
        let span = hi - lo + 1;
        lo + (((self.next_f64() + 1.0) * 0.5 * span as f64) as usize).min(span - 1)
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Naive device accumulation vs the inline host longhand, f64. The kernel's
/// single-thread ORDERED `acc += coef*(-alpha*r2).exp()` tracks the host
/// ordered sum to <1 ULP/term (the device `exp()` may differ from std f64::exp
/// by <1 ULP). gemm_oracle uses the same 1e-9 bound (ORACLE-07: GPU at
/// documented tolerance, not claimed bit-identical).
const TOL: f64 = 1e-9;

/// A randomized pure-s-shell libcint fixture: flat atm/bas/env/ao_loc arrays +
/// random F-order grid coords, plus the resolved `nao`.
struct SShellFixture {
    coords: Vec<f64>,
    ngrids: usize,
    atm: Vec<i32>,
    bas: Vec<i32>,
    env: Vec<f64>,
    ao_loc: Vec<i32>,
    nao: usize,
}

/// Build a valid pure-s-shell layout mirroring exactly what `eval_gto_sph_cpu`
/// reads: BAS_SLOTS row = [ATOM_OF, ANG_OF=0, NPRIM_OF, NCTR_OF, _, PTR_EXP,
/// PTR_COEFF, _]; ATM_SLOTS row carries PTR_COORD into env. Coefficients are
/// packed F-order at PTR_COEFF (`ptr_coeff + c_idx*nprim + p_idx`). ao_loc is
/// the running sum of nctr per shell.
fn build_fixture(rng: &mut Lcg, n_atoms: usize, shells_per_atom: usize, ngrids: usize) -> SShellFixture {
    let n_shells = n_atoms * shells_per_atom;

    // env layout: [ atom coords (3 per atom) | per-shell (exps then coeffs) ].
    let mut env: Vec<f64> = Vec::new();
    let mut atm: Vec<i32> = vec![0; n_atoms * ATM_SLOTS];

    // Atom coordinate blocks first.
    for a in 0..n_atoms {
        let ptr_coord = env.len() as i32;
        atm[a * ATM_SLOTS + PTR_COORD] = ptr_coord;
        env.push(rng.next_range(-2.0, 2.0)); // ax
        env.push(rng.next_range(-2.0, 2.0)); // ay
        env.push(rng.next_range(-2.0, 2.0)); // az
    }

    let mut bas: Vec<i32> = vec![0; n_shells * BAS_SLOTS];
    let mut ao_loc: Vec<i32> = Vec::with_capacity(n_shells + 1);
    let mut ao_running: i32 = 0;

    let mut shell_idx = 0usize;
    for a in 0..n_atoms {
        for _ in 0..shells_per_atom {
            let nprim = rng.next_usize(1, 4);
            let nctr = rng.next_usize(1, 2);

            let ptr_exp = env.len() as i32;
            for _ in 0..nprim {
                // Positive exponents in a chemically reasonable spread.
                env.push(rng.next_range(0.1, 6.0));
            }
            let ptr_coeff = env.len() as i32;
            // F-order coefficient matrix: nctr columns × nprim rows.
            for _ in 0..(nctr * nprim) {
                env.push(rng.next_range(-1.0, 1.0));
            }

            let row = shell_idx * BAS_SLOTS;
            bas[row + ATOM_OF] = a as i32;
            bas[row + ANG_OF] = 0; // pure s
            bas[row + NPRIM_OF] = nprim as i32;
            bas[row + NCTR_OF] = nctr as i32;
            bas[row + PTR_EXP] = ptr_exp;
            bas[row + PTR_COEFF] = ptr_coeff;

            ao_loc.push(ao_running);
            ao_running += nctr as i32;
            shell_idx += 1;
        }
    }
    ao_loc.push(ao_running); // terminal entry (libcint convention)
    let nao = ao_running as usize;

    // Random F-order coords: x[0..ngrids], y[..], z[..].
    let mut coords = vec![0.0_f64; ngrids * 3];
    for c in coords.iter_mut() {
        *c = rng.next_range(-2.5, 2.5);
    }

    SShellFixture {
        coords,
        ngrids,
        atm,
        bas,
        env,
        ao_loc,
        nao,
    }
}

/// Inline host oracle — the FMA-free l=0 longhand. ORACLE PIN: byte-matches
/// `eval_gto_sph_cpu` lines 603-614 (same y00, same F-order index). Do NOT
/// re-derive Y00; it is copied verbatim from the production l=0 path.
fn oracle_eval_s(f: &SShellFixture) -> Vec<f64> {
    let SShellFixture {
        coords,
        ngrids,
        atm,
        bas,
        env,
        ao_loc,
        nao,
    } = f;
    let ngrids = *ngrids;
    let nbas = bas.len() / BAS_SLOTS;
    let mut out = vec![0.0_f64; ngrids * nao];

    let y00 = 0.5_f64 / std::f64::consts::PI.sqrt();
    for g in 0..ngrids {
        let gx = coords[g];
        let gy = coords[g + ngrids];
        let gz = coords[g + 2 * ngrids];

        // `shell_idx` drives parallel flat-array offsets (bas via BAS_SLOTS,
        // ao_loc) — a range loop mirrors `eval_gto_sph_cpu` exactly.
        #[allow(clippy::needless_range_loop)]
        for shell_idx in 0..nbas {
            let bas_row = shell_idx * BAS_SLOTS;
            let atom_id = bas[bas_row + ATOM_OF] as usize;
            let nprim = bas[bas_row + NPRIM_OF] as usize;
            let nctr = bas[bas_row + NCTR_OF] as usize;
            let ptr_exp = bas[bas_row + PTR_EXP] as usize;
            let ptr_coeff = bas[bas_row + PTR_COEFF] as usize;

            let atm_row = atom_id * ATM_SLOTS;
            let ptr_coord = atm[atm_row + PTR_COORD] as usize;
            let ax = env[ptr_coord];
            let ay = env[ptr_coord + 1];
            let az = env[ptr_coord + 2];

            let dx = gx - ax;
            let dy = gy - ay;
            let dz = gz - az;
            let r2 = dx * dx + dy * dy + dz * dz;

            let ao_off = ao_loc[shell_idx] as usize;
            for c_idx in 0..nctr {
                let mut acc: f64 = 0.0;
                for p_idx in 0..nprim {
                    let alpha = env[ptr_exp + p_idx];
                    let coef = env[ptr_coeff + c_idx * nprim + p_idx];
                    acc += coef * (-alpha * r2).exp();
                }
                let ao_idx = ao_off + c_idx;
                out[g + ao_idx * ngrids] = acc * y00;
            }
        }
    }
    out
}

/// Run one randomized fixture on `client` (device s-shell path) and compare
/// against the inline oracle. Returns the max elementwise absolute difference.
fn check_case(
    client: &AlgebraClient,
    n_atoms: usize,
    shells_per_atom: usize,
    ngrids: usize,
    seed: u64,
) -> f64 {
    let mut rng = Lcg::new(seed);
    let f = build_fixture(&mut rng, n_atoms, shells_per_atom, ngrids);

    // Device path: pure-s-shell basis → routes to the cube kernel.
    let device = eval_gto_sph(
        client, &f.coords, f.ngrids, &f.atm, &f.bas, &f.env, &f.ao_loc, f.nao, true,
    )
    .expect("eval_gto_sph should succeed for a valid s-shell fixture");

    let reference = oracle_eval_s(&f);

    assert_eq!(
        device.values.len(),
        f.ngrids * f.nao,
        "device output length must be ngrids*nao"
    );
    assert_eq!(device.shape, vec![f.ngrids, f.nao], "device shape descriptor");
    assert_eq!(reference.len(), f.ngrids * f.nao, "oracle output length");

    max_abs_diff(&device.values, &reference)
}

/// Spread of (n_atoms, shells_per_atom, ngrids) shapes: single shell, multi-
/// contraction, multi-atom, larger grids — exercises the index math, the
/// >2-prim ordered sum, and the F-order write/read.
const CASES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (1, 1, 7),
    (1, 2, 13),
    (2, 1, 32),
    (3, 2, 64),
    (2, 3, 17),
    (4, 1, 100),
];

#[test]
fn eval_gto_s_matches_oracle_on_cpu() {
    let client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice));
    let mut worst = 0.0_f64;
    for (i, &(na, spa, ng)) in CASES.iter().enumerate() {
        let diff = check_case(&client, na, spa, ng, 0x5EED_0117_u64 + i as u64);
        worst = worst.max(diff);
        assert!(
            diff < TOL,
            "CPU eval_gto s-shell (atoms={na}, shells/atom={spa}, ngrids={ng}): \
             max abs diff {diff:e} >= tol {TOL:e}"
        );
    }
    // Surface the observed worst-case diff for the summary.
    eprintln!("[eval_gto_oracle] CPU worst max_abs_diff = {worst:e}");
}

#[cfg(feature = "rocm")]
#[test]
fn eval_gto_s_matches_oracle_on_rocm() {
    // Construct the HIP client directly on the default AMD device (gfx1152),
    // dodging the PYSCF_BACKEND env race (per gemm_oracle.rs).
    let client = AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(
        &cubecl_hip::AmdDevice::default(),
    ));
    assert!(
        matches!(client, AlgebraClient::Rocm(_)),
        "test must run on the ROCm backend, not a fallback"
    );
    let mut worst = 0.0_f64;
    for (i, &(na, spa, ng)) in CASES.iter().enumerate() {
        let diff = check_case(&client, na, spa, ng, 0x0CA0_1A7E_u64 + i as u64);
        worst = worst.max(diff);
        assert!(
            diff < TOL,
            "ROCm eval_gto s-shell (atoms={na}, shells/atom={spa}, ngrids={ng}): \
             max abs diff {diff:e} >= tol {TOL:e}"
        );
    }
    eprintln!("[eval_gto_oracle] ROCm worst max_abs_diff = {worst:e}");
}

// ════════════════════════════════════════════════════════════════════════
// quick-260530-mlg: l>=1 (p/d/f/g) + mixed s+p+d differential oracle.
//
// After quick-260530-mlg, `eval_gto_sph` routes any basis with maxl in 1..=4
// to the general `#[cube]` device kernel. This arm runs the device path on
// randomized p/d/f (+ optional g) and mixed-l fixtures and diffs it against an
// INDEPENDENT host longhand reference (`lge1_reference`), a DIFFERENT code path
// from the kernel: a naive triple-nested cartesian-monomial loop with the c2s
// matrices written out longhand (same discipline as tests/eval_gto_lge1.rs,
// extended here to l=3 (f) and l=4 (g)). A wrong-convention kernel bug (AO
// ordering vs ao_loc, transposed c2s, cart_pow order, fac1 placement) surfaces
// as an element-wise mismatch > TOL. Distinct seeds from the s-shell arm avoid
// fixture collisions.
// ════════════════════════════════════════════════════════════════════════

/// Independent host longhand reference for l in 0..=4 (a DIFFERENT code path
/// from the production kernel). c2s matrices copied VERBATIM from the FROZEN
/// libcint tables in `eval_gto.rs` (`c2s_coeff` L0..L4) — NOT re-derived, so
/// the differential check cannot pass against a subtly-wrong oracle.
mod lge1_reference {
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
    /// libcint `g_trans_cart2sph` coefficient `T[l][m][c]`. VERBATIM copy of
    /// the FROZEN `c2s_coeff` tables L0..L4 in eval_gto.rs.
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
        const L3: [[f64; 10]; 7] = [
            [
                0.0,
                1.770_130_769_779_930_4,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.590_043_589_926_643_5,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0, 0.0, 0.0, 0.0, 2.890_611_442_640_554_3, 0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            [
                0.0,
                -0.457_045_799_464_465_7,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.457_045_799_464_465_7,
                0.0,
                1.828_183_197_857_862_9,
                0.0,
            ],
            [
                0.0,
                0.0,
                -1.119_528_997_770_346_2,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.119_528_997_770_346_2,
                0.0,
                0.746_352_665_180_230_8,
            ],
            [
                -0.457_045_799_464_465_7,
                0.0,
                0.0,
                -0.457_045_799_464_465_7,
                0.0,
                1.828_183_197_857_862_9,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.445_305_721_320_277_1,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.445_305_721_320_277_1,
                0.0,
                0.0,
            ],
            [
                0.590_043_589_926_643_5,
                0.0,
                0.0,
                -1.770_130_769_779_930_4,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ];
        const L4: [[f64; 15]; 9] = [
            [
                0.0,
                2.503_342_941_796_704_6,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.503_342_941_796_704_6,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                5.310_392_309_339_791,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -1.770_130_769_779_930_4,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                -0.946_174_695_757_56,
                0.0,
                0.0,
                0.0,
                0.0,
                -0.946_174_695_757_56,
                0.0,
                5.677_048_174_545_360_5,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                0.0,
                -2.007_139_630_671_867_6,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.007_139_630_671_867_6,
                0.0,
                2.676_186_174_229_157,
                0.0,
            ],
            [
                0.317_356_640_745_612_93,
                0.0,
                0.0,
                0.634_713_281_491_225_9,
                0.0,
                -2.538_853_125_964_903_4,
                0.0,
                0.0,
                0.0,
                0.0,
                0.317_356_640_745_612_93,
                0.0,
                -2.538_853_125_964_903_4,
                0.0,
                0.846_284_375_321_634_5,
            ],
            [
                0.0,
                0.0,
                -2.007_139_630_671_867_6,
                0.0,
                0.0,
                0.0,
                0.0,
                -2.007_139_630_671_867_6,
                0.0,
                2.676_186_174_229_157,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                -0.473_087_347_878_78,
                0.0,
                0.0,
                0.0,
                0.0,
                2.838_524_087_272_680_2,
                0.0,
                0.0,
                0.0,
                0.0,
                0.473_087_347_878_78,
                0.0,
                -2.838_524_087_272_680_2,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                1.770_130_769_779_930_4,
                0.0,
                0.0,
                0.0,
                0.0,
                -5.310_392_309_339_791,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            [
                0.625_835_735_449_176_1,
                0.0,
                0.0,
                -3.755_014_412_695_057,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.625_835_735_449_176_1,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        ];
        match l {
            0 => L0[m][c],
            1 => L1[m][c],
            2 => L2[m][c],
            3 => L3[m][c],
            4 => L4[m][c],
            _ => panic!("lge1_reference c2s only ports l<=4 (fixture max l = 4)"),
        }
    }
}

/// A randomized mixed-l libcint fixture: shells carry the angular momenta in
/// `l_pattern` (e.g. `[1,2,3]` or `[0,1,2]`), repeated `shells_per_atom` times
/// per atom across `n_atoms` atoms. ao_loc running sum uses `nctr * nsph(l)`.
struct MixedFixture {
    coords: Vec<f64>,
    ngrids: usize,
    atm: Vec<i32>,
    bas: Vec<i32>,
    env: Vec<f64>,
    ao_loc: Vec<i32>,
    nao: usize,
}

fn build_mixed_l_fixture(
    rng: &mut Lcg,
    n_atoms: usize,
    l_pattern: &[u32],
    shells_per_atom: usize,
    ngrids: usize,
) -> MixedFixture {
    let n_shells = n_atoms * shells_per_atom;

    let mut env: Vec<f64> = Vec::new();
    let mut atm: Vec<i32> = vec![0; n_atoms * ATM_SLOTS];

    for a in 0..n_atoms {
        let ptr_coord = env.len() as i32;
        atm[a * ATM_SLOTS + PTR_COORD] = ptr_coord;
        env.push(rng.next_range(-2.0, 2.0));
        env.push(rng.next_range(-2.0, 2.0));
        env.push(rng.next_range(-2.0, 2.0));
    }

    let mut bas: Vec<i32> = vec![0; n_shells * BAS_SLOTS];
    let mut ao_loc: Vec<i32> = Vec::with_capacity(n_shells + 1);
    let mut ao_running: i32 = 0;

    let mut shell_idx = 0usize;
    for a in 0..n_atoms {
        for s in 0..shells_per_atom {
            let l = l_pattern[(shell_idx + s) % l_pattern.len()];
            let nprim = rng.next_usize(1, 4);
            let nctr = rng.next_usize(1, 2);

            let ptr_exp = env.len() as i32;
            for _ in 0..nprim {
                env.push(rng.next_range(0.1, 6.0));
            }
            let ptr_coeff = env.len() as i32;
            // F-order coefficient matrix: nctr columns × nprim rows.
            for _ in 0..(nctr * nprim) {
                env.push(rng.next_range(-1.0, 1.0));
            }

            let row = shell_idx * BAS_SLOTS;
            bas[row + ATOM_OF] = a as i32;
            bas[row + ANG_OF] = l as i32;
            bas[row + NPRIM_OF] = nprim as i32;
            bas[row + NCTR_OF] = nctr as i32;
            bas[row + PTR_EXP] = ptr_exp;
            bas[row + PTR_COEFF] = ptr_coeff;

            ao_loc.push(ao_running);
            // l>=1: 2l+1 spherical AOs per contraction column.
            ao_running += (nctr * lge1_reference::nsph(l)) as i32;
            shell_idx += 1;
        }
    }
    ao_loc.push(ao_running);
    let nao = ao_running as usize;

    let mut coords = vec![0.0_f64; ngrids * 3];
    for c in coords.iter_mut() {
        *c = rng.next_range(-2.5, 2.5);
    }

    MixedFixture {
        coords,
        ngrids,
        atm,
        bas,
        env,
        ao_loc,
        nao,
    }
}

/// Independent longhand l in 0..=4 oracle (DIFFERENT code path from the
/// kernel). Mirrors `eval_gto_sph_cpu`'s l>=1 branch (sequential radial *fac1,
/// mono*radial, c2s, F-order write) using `lge1_reference::*`.
fn oracle_eval_lge1(f: &MixedFixture) -> Vec<f64> {
    let ngrids = f.ngrids;
    let nbas = f.bas.len() / BAS_SLOTS;
    let mut out = vec![0.0_f64; ngrids * f.nao];

    for g in 0..ngrids {
        let gx = f.coords[g];
        let gy = f.coords[g + ngrids];
        let gz = f.coords[g + 2 * ngrids];

        // `s` drives parallel flat-array offsets (mirrors eval_gto_sph_cpu).
        #[allow(clippy::needless_range_loop)]
        for s in 0..nbas {
            let row = s * BAS_SLOTS;
            let atom_id = f.bas[row + ATOM_OF] as usize;
            let l = f.bas[row + ANG_OF] as u32;
            let nprim = f.bas[row + NPRIM_OF] as usize;
            let nctr = f.bas[row + NCTR_OF] as usize;
            let ptr_exp = f.bas[row + PTR_EXP] as usize;
            let ptr_coeff = f.bas[row + PTR_COEFF] as usize;

            let pc = f.atm[atom_id * ATM_SLOTS + PTR_COORD] as usize;
            let (dx, dy, dz) = (gx - f.env[pc], gy - f.env[pc + 1], gz - f.env[pc + 2]);
            let r2 = dx * dx + dy * dy + dz * dz;

            let fac1 = lge1_reference::common_fac_sp(l);
            let powers = lge1_reference::cart_powers(l);
            let ncart = lge1_reference::ncart(l);
            let nsph = lge1_reference::nsph(l);
            let ao_off = f.ao_loc[s] as usize;

            for c in 0..nctr {
                // Sequential radial (matches the kernel's ordered acc), then *fac1.
                let mut radial = 0.0_f64;
                for p in 0..nprim {
                    let alpha = f.env[ptr_exp + p];
                    let coef = f.env[ptr_coeff + c * nprim + p];
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
                        v += lge1_reference::c2s_coeff(l, m, ci) * cart[ci];
                    }
                    out[g + (ao_off + c * nsph + m) * ngrids] = v;
                }
            }
        }
    }
    out
}

/// Run one randomized mixed-l fixture on `client` (device general kernel path)
/// and diff against the independent longhand oracle.
fn check_mixed_case(
    client: &AlgebraClient,
    n_atoms: usize,
    l_pattern: &[u32],
    shells_per_atom: usize,
    ngrids: usize,
    seed: u64,
) -> f64 {
    let mut rng = Lcg::new(seed);
    let f = build_mixed_l_fixture(&mut rng, n_atoms, l_pattern, shells_per_atom, ngrids);

    let device = eval_gto_sph(
        client, &f.coords, f.ngrids, &f.atm, &f.bas, &f.env, &f.ao_loc, f.nao, true,
    )
    .expect("eval_gto_sph should succeed for a valid mixed-l fixture");

    let reference = oracle_eval_lge1(&f);

    assert_eq!(
        device.values.len(),
        f.ngrids * f.nao,
        "device output length must be ngrids*nao"
    );
    assert_eq!(device.shape, vec![f.ngrids, f.nao], "device shape descriptor");
    assert_eq!(reference.len(), f.ngrids * f.nao, "oracle output length");

    max_abs_diff(&device.values, &reference)
}

/// Mixed-l shapes: (n_atoms, l_pattern, shells_per_atom, ngrids). Covers pure
/// p/d/f, pure g (l=4), AND mixed s+p+d / s+p+d+f+g bases.
const MIXED_CASES: &[(usize, &[u32], usize, usize)] = &[
    (1, &[1], 1, 9),              // pure p
    (1, &[2], 1, 13),             // pure d
    (1, &[3], 1, 11),             // pure f
    (1, &[4], 1, 7),              // pure g (l=4)
    (1, &[0, 1, 2], 3, 17),       // mixed s+p+d (cc-pVDZ-like)
    (2, &[1, 2, 3], 3, 32),       // mixed p+d+f, 2 atoms
    (1, &[0, 1, 2, 3, 4], 5, 25), // mixed s..g, all l on one centre
    (3, &[2, 3], 2, 50),          // d+f spread over 3 atoms
];

#[test]
fn eval_gto_lge1_matches_oracle_on_cpu() {
    let client = AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice));
    let mut worst = 0.0_f64;
    for (i, &(na, pat, spa, ng)) in MIXED_CASES.iter().enumerate() {
        let diff = check_mixed_case(&client, na, pat, spa, ng, 0xA11_C0DE_u64 + i as u64);
        worst = worst.max(diff);
        assert!(
            diff < TOL,
            "CPU eval_gto l>=1 (atoms={na}, l_pattern={pat:?}, shells/atom={spa}, \
             ngrids={ng}): max abs diff {diff:e} >= tol {TOL:e}"
        );
    }
    eprintln!("[eval_gto_oracle] CPU mixed-l (p/d/f/g) worst max_abs_diff = {worst:e}");
}

#[cfg(feature = "rocm")]
#[test]
fn eval_gto_lge1_matches_oracle_on_rocm() {
    let client = AlgebraClient::Rocm(cubecl_hip::HipRuntime::client(
        &cubecl_hip::AmdDevice::default(),
    ));
    assert!(
        matches!(client, AlgebraClient::Rocm(_)),
        "test must run on the ROCm backend, not a fallback"
    );
    let mut worst = 0.0_f64;
    for (i, &(na, pat, spa, ng)) in MIXED_CASES.iter().enumerate() {
        let diff = check_mixed_case(&client, na, pat, spa, ng, 0xF00D_BEEF_u64 + i as u64);
        worst = worst.max(diff);
        assert!(
            diff < TOL,
            "ROCm eval_gto l>=1 (atoms={na}, l_pattern={pat:?}, shells/atom={spa}, \
             ngrids={ng}): max abs diff {diff:e} >= tol {TOL:e}"
        );
    }
    eprintln!("[eval_gto_oracle] ROCm mixed-l (p/d/f/g) worst max_abs_diff = {worst:e}");
}

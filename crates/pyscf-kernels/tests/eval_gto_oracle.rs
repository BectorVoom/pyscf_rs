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

//! Plan 09-05 — K-01 `gv` and K-02 `struct_factor` (PBC-MASTER-PLAN §6).
//!
//! Verified here, at the KERNEL level (no `Cell` involved): both device kernels
//! match a naive host reference bit-for-bit over a spread of mesh shapes,
//! including ones whose grid count is not a multiple of the 256-thread cube
//! dimension (the `g < ngrids` / `i < n` tail guards) and non-cubic meshes
//! (the `(x, y, z)` index inversion, which a cubic mesh cannot distinguish from
//! a transposed one); shape violations are rejected without launching.
//!
//! `crates/pyscf-pbc-gto/tests/gv.rs` is the numeric gate against upstream
//! PySCF; this file is the kernel's own contract.
//!
//! Clients are constructed directly (not via `select_backend`) so this never
//! races on the process-global `PYSCF_BACKEND`.

#![cfg(feature = "cpu")]

use cubecl::Runtime;
use pyscf_algebra::AlgebraClient;
use pyscf_kernels::{gv, struct_factor};

struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        u * 2.0 - 1.0
    }
}

fn cpu_client() -> AlgebraClient {
    AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice))
}

/// Naive host reference for K-01 — the `lib/pbc/cell.c:122-146` loop nest,
/// in the same accumulation order.
fn gv_reference(rx: &[f64], ry: &[f64], rz: &[f64], b: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(rx.len() * ry.len() * rz.len() * 3);
    for &x in rx {
        for &y in ry {
            for &z in rz {
                for c in 0..3 {
                    let mut v = x * b[c];
                    v += y * b[3 + c];
                    v += z * b[6 + c];
                    out.push(v);
                }
            }
        }
    }
    out
}

/// Naive host reference for K-02 — `SI[a,g] = exp(-i Gv[g] . R_a)`, with the
/// same `theta = 0 - rg` spelling the kernel uses so the two agree bit-for-bit.
fn si_reference(coords: &[f64], gv: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let natm = coords.len() / 3;
    let ngrids = gv.len() / 3;
    let mut re = Vec::with_capacity(natm * ngrids);
    let mut im = Vec::with_capacity(natm * ngrids);
    for a in 0..natm {
        for g in 0..ngrids {
            let mut rg = gv[g * 3] * coords[a * 3];
            rg += gv[g * 3 + 1] * coords[a * 3 + 1];
            rg += gv[g * 3 + 2] * coords[a * 3 + 2];
            let theta = 0.0 - rg;
            re.push(theta.cos());
            im.push(theta.sin());
        }
    }
    (re, im)
}

/// Meshes chosen to straddle the 256-thread cube dimension and to break the
/// `(x, y, z)` symmetry: 6*6*7 = 252, 6*7*6 = 252 (a transposed-index bug shows
/// up as a mismatch between these two), 8*8*4 = 256 exactly, 9*9*9 = 729.
const MESHES: [[usize; 3]; 7] = [
    [1, 1, 1],
    [5, 5, 5],
    [3, 4, 5],
    [6, 6, 7],
    [6, 7, 6],
    [8, 8, 4],
    [9, 9, 9],
];

/// A non-symmetric reciprocal matrix, so a transposed `b` cannot pass.
const B: [f64; 9] = [
    1.5, -0.25, 0.75, //
    0.5, 2.25, -1.25, //
    -0.75, 0.125, 3.5,
];

fn freqs(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if i <= (n - 1) / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            }
        })
        .collect()
}

#[test]
fn gv_kernel_matches_host_reference_bitwise() {
    let client = cpu_client();
    for mesh in MESHES {
        let (rx, ry, rz) = (freqs(mesh[0]), freqs(mesh[1]), freqs(mesh[2]));
        let got = gv(&client, &rx, &ry, &rz, &B).expect("K-01 launch");
        let want = gv_reference(&rx, &ry, &rz, &B);
        assert_eq!(got.len(), mesh[0] * mesh[1] * mesh[2] * 3, "mesh {mesh:?}");
        assert_eq!(
            got, want,
            "mesh {mesh:?}: K-01 differs from the C loop nest"
        );
    }
}

/// `[6,7,6]` and `[6,6,7]` have the same grid count but different shapes; a
/// swapped `my`/`mz` in the index inversion would make one of them wrong.
#[test]
fn gv_kernel_distinguishes_transposed_meshes() {
    let client = cpu_client();
    let a = gv(&client, &freqs(6), &freqs(6), &freqs(7), &B).expect("launch");
    let b = gv(&client, &freqs(6), &freqs(7), &freqs(6), &B).expect("launch");
    assert_eq!(a.len(), b.len());
    assert_ne!(a, b, "a transposed mesh must not produce the same table");
}

#[test]
fn struct_factor_kernel_matches_host_reference_bitwise() {
    let client = cpu_client();
    let mut rng = Lcg::new(0x0905_0102);
    for mesh in MESHES {
        let (rx, ry, rz) = (freqs(mesh[0]), freqs(mesh[1]), freqs(mesh[2]));
        let gvec = gv(&client, &rx, &ry, &rz, &B).expect("K-01 launch");
        for natm in [1usize, 2, 5] {
            let coords: Vec<f64> = (0..natm * 3).map(|_| rng.next_f64() * 4.0).collect();
            let (re, im) = struct_factor(&client, &coords, &gvec).expect("K-02 launch");
            let (wre, wim) = si_reference(&coords, &gvec);
            assert_eq!(re.len(), natm * mesh[0] * mesh[1] * mesh[2]);
            assert_eq!(re, wre, "mesh {mesh:?} natm {natm}: K-02 real plane");
            assert_eq!(im, wim, "mesh {mesh:?} natm {natm}: K-02 imag plane");
        }
    }
}

/// `|SI| == 1` — the kernel's own physical invariant, independent of any
/// reference implementation.
#[test]
fn struct_factor_is_unit_modulus() {
    let client = cpu_client();
    let mut rng = Lcg::new(0x0905_0304);
    let gvec = gv(&client, &freqs(7), &freqs(5), &freqs(3), &B).expect("launch");
    let coords: Vec<f64> = (0..9).map(|_| rng.next_f64() * 10.0).collect();
    let (re, im) = struct_factor(&client, &coords, &gvec).expect("launch");
    for (r, i) in re.iter().zip(im.iter()) {
        assert!(
            (r * r + i * i - 1.0).abs() < 1e-15,
            "|SI|^2 = {}",
            r * r + i * i
        );
    }
}

/// `SI` at `G = 0` is `1 + 0i` for every atom, exactly (the frequency tables
/// start at zero, so `gv[0..3]` is the origin).
#[test]
fn struct_factor_at_g_zero_is_one() {
    let client = cpu_client();
    let gvec = gv(&client, &freqs(5), &freqs(5), &freqs(5), &B).expect("launch");
    assert_eq!(&gvec[..3], &[0.0, 0.0, 0.0]);
    let coords = [0.3, -1.7, 2.5, 4.0, 0.0, -0.5];
    let ngrids = 125;
    let (re, im) = struct_factor(&client, &coords, &gvec).expect("launch");
    for a in 0..2 {
        assert_eq!(re[a * ngrids], 1.0);
        assert_eq!(im[a * ngrids], 0.0);
    }
}

/// Shape violations are rejected before any launch, and empty inputs return
/// empty without launching.
#[test]
fn shape_violations_are_rejected() {
    let client = cpu_client();
    let r = freqs(4);
    assert!(
        gv(&client, &r, &r, &r, &[1.0, 2.0, 3.0]).is_err(),
        "b must be 9 long"
    );
    assert!(
        gv(&client, &[], &r, &r, &B)
            .expect("empty is ok")
            .is_empty()
    );

    let gvec = gv(&client, &r, &r, &r, &B).expect("launch");
    assert!(
        struct_factor(&client, &[1.0, 2.0], &gvec).is_err(),
        "coords % 3 != 0"
    );
    assert!(
        struct_factor(&client, &[1.0, 2.0, 3.0], &[1.0, 2.0]).is_err(),
        "gv % 3 != 0"
    );
    let (re, im) = struct_factor(&client, &[], &gvec).expect("empty is ok");
    assert!(re.is_empty() && im.is_empty());
}

//! Both GEMM kernels against the same host reference.
//!
//! `gemm_dense` picks between two no-barrier kernels from the device's hardware
//! properties: the simple one-unit-per-output-vector shape on a runtime without
//! planes, the register-tiled shape on a GPU. On a machine whose only
//! f64-capable backend is CubeCL's CPU runtime the second is unreachable, so it
//! would ship untested. `PYSCF_GEMM_KERNEL` pins the choice, and this test drives
//! both paths through shapes that stress what actually differs between them:
//! the row-tile tail (`m` not a multiple of `ROWS = 8`), non-square operands, and
//! column counts that force the vectorization width down to 1.
//!
//! The override is read once into a `OnceLock`, so each path needs its own
//! process — hence the re-exec through the test binary rather than two `#[test]`
//! functions in one process.

use pyscf_algebra::{gemm_dense, select_backend};

/// Deterministic pseudo-random fill; a fixed LCG so a failure is reproducible.
fn fill(n: usize, seed: u64) -> Vec<f64> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        })
        .collect()
}

/// Naive row-major `M×K @ K×N` on the host, accumulated in the same order the
/// kernels walk `k` so the comparison is exact rather than merely close.
fn reference(lhs: &[f64], rhs: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += lhs[i * k + p] * rhs[p * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Shapes chosen to cover what the two kernels do differently:
/// `m % ROWS != 0` exercises the row-tile tail guard, `n` odd/prime forces the
/// vector width to 1, and the non-square cases catch an `m`/`n` index swap.
const SHAPES: &[(usize, usize, usize)] = &[
    (1, 1, 1),
    (8, 8, 8),
    (7, 5, 3),    // m below one row tile, width forced to 1
    (9, 4, 16),   // m = ROWS + 1: one full tile plus a one-row tail
    (17, 13, 32), // odd m and k against a wide, vectorizable n
    (64, 64, 64),
    (33, 8, 40),
];

/// Run every shape on whichever kernel the environment pinned.
fn check_all_shapes() {
    let sel = select_backend().expect("backend must resolve");
    let client = &sel.client;

    for &(m, k, n) in SHAPES {
        let lhs = fill(m * k, 0x51ed_270b + (m * 131 + k) as u64);
        let rhs = fill(k * n, 0x9e37_79b9 + (k * 131 + n) as u64);

        let got = gemm_dense::<f64>(client, &lhs, &rhs, m, k, n).expect("gemm_dense");
        let want = reference(&lhs, &rhs, m, k, n);

        assert_eq!(got.len(), want.len(), "shape {m}x{k}x{n}: wrong length");
        for (idx, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            // Both kernels accumulate `k` products in ascending `k`, exactly as
            // the reference does. Vectorization splits the OUTPUT columns, never
            // the reduction, so the summation order is identical and the result
            // is bit-for-bit equal. A tolerance here would hide an indexing bug.
            assert_eq!(
                g,
                w,
                "shape {m}x{k}x{n}, element {idx} (row {}, col {})",
                idx / n,
                idx % n
            );
        }
    }
}

/// The device-chosen path, plus each pinned path in its own child process.
///
/// `PYSCF_GEMM_KERNEL` is cached in a `OnceLock` on first read, so a single
/// process can only ever exercise one setting; the children re-enter this binary
/// with the variable set and the hidden worker test named.
#[test]
fn both_gemm_kernels_match_the_host_reference() {
    // The path this machine's backend actually selects.
    check_all_shapes();

    for kernel in ["simple", "row_tiled"] {
        let exe = std::env::current_exe().expect("test binary path");
        let status = std::process::Command::new(exe)
            .args(["--exact", "--nocapture", "--ignored", "gemm_kernel_child"])
            .env("PYSCF_GEMM_KERNEL", kernel)
            .status()
            .expect("spawn child test process");
        assert!(status.success(), "PYSCF_GEMM_KERNEL={kernel} failed");
    }
}

/// Worker for [`both_gemm_kernels_match_the_host_reference`]; ignored so it only
/// runs when a child names it explicitly.
#[test]
#[ignore = "spawned by both_gemm_kernels_match_the_host_reference"]
fn gemm_kernel_child() {
    let kernel = std::env::var("PYSCF_GEMM_KERNEL").expect("child needs PYSCF_GEMM_KERNEL");
    println!("checking gemm kernel: {kernel}");
    check_all_shapes();
}

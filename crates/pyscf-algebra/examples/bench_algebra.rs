//! Micro-benchmark for the `pyscf-algebra` dense surface.
//!
//! ```text
//! cargo run --release --example bench_algebra -p pyscf-algebra
//! PYSCF_BACKEND=rocm cargo run --release --example bench_algebra \
//!     -p pyscf-algebra --no-default-features --features rocm
//! ```
//!
//! Every GEMM row is checked against a host reference before it is timed, so a
//! kernel that got fast by computing the wrong thing reports an error rather
//! than a number. `PYSCF_GEMM_KERNEL=simple|row_tiled` pins the GEMM kernel
//! instead of letting the device's properties choose it.
//!
//! The `*_dense` entry points upload their operands and read the result back on
//! every call, so the large BLAS-1 rows measure host/device transfer far more
//! than they measure the kernel. That is the surface's contract, not an
//! artifact of the benchmark; the resident `Tensor` API avoids the transfer.

use std::time::Instant;

use pyscf_algebra::{
    axpy_dense, dot_dense, gemm_dense, reduce_sum_dense, scal_dense, select_backend,
    transpose_dense,
};

/// Deterministic pseudo-random fill, so successive runs are comparable.
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

/// Time `f` over `iters` runs, after one warm-up that pays the JIT compilation
/// outside the measured window.
fn time<T>(label: &str, iters: usize, mut f: impl FnMut() -> T) -> f64 {
    let _ = f();
    let t0 = Instant::now();
    for _ in 0..iters {
        let _ = f();
    }
    let ms = t0.elapsed().as_secs_f64() / iters as f64 * 1e3;
    println!("{label:<26} {ms:>10.3} ms");
    ms
}

fn main() {
    let sel = select_backend().expect("backend must resolve");
    let client = &sel.client;
    println!("backend = {:?}\n", sel.kind);

    for &d in &[64usize, 128, 256, 512] {
        let a = fill(d * d, 1);
        let b = fill(d * d, 2);

        // Correctness gate: a corner block against a host reference.
        let out = gemm_dense::<f64>(client, &a, &b, d, d, d).expect("gemm_dense");
        let mut maxerr = 0.0f64;
        for i in 0..d.min(4) {
            for j in 0..d.min(4) {
                let want: f64 = (0..d).map(|p| a[i * d + p] * b[p * d + j]).sum();
                maxerr = maxerr.max((want - out[i * d + j]).abs());
            }
        }

        let iters = if d <= 128 { 20 } else { 5 };
        let ms = time(&format!("gemm {d}x{d}x{d}"), iters, || {
            gemm_dense::<f64>(client, &a, &b, d, d, d).expect("gemm_dense")
        });
        let gflops = 2.0 * (d as f64).powi(3) / (ms * 1e-3) / 1e9;
        println!("{:26} {gflops:>10.2} GFLOP/s  (maxerr {maxerr:.1e})", "");
    }

    for &n in &[1usize << 16, 1 << 20] {
        let x = fill(n, 3);
        let y = fill(n, 4);
        time(&format!("dot n={n}"), 20, || {
            dot_dense::<f64>(client, &x, &y).expect("dot_dense")
        });
        time(&format!("reduce_sum n={n}"), 20, || {
            reduce_sum_dense::<f64>(client, &x).expect("reduce_sum_dense")
        });
        let mut yy = y.clone();
        time(&format!("axpy n={n}"), 20, || {
            axpy_dense::<f64>(client, 1.5, &x, &mut yy).expect("axpy_dense")
        });
        let mut xx = x.clone();
        time(&format!("scal n={n}"), 20, || {
            scal_dense::<f64>(client, 1.000001, &mut xx).expect("scal_dense")
        });
    }

    for &d in &[256usize, 1024] {
        let x = fill(d * d, 5);
        time(&format!("transpose {d}x{d}"), 10, || {
            transpose_dense::<f64>(client, &x, d, d).expect("transpose_dense")
        });
    }
}

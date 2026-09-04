//! Throughput of the default (`stockham`) FFT engine on the 64-row transforms
//! `fft_jk::get_k_kpts` issues, for the meshes the reference cells use.
//!
//! ```bash
//! cargo run --release -p pyscf-pbc-tools --example bench_fft
//! ```

use pyscf_algebra::CTensor;
use pyscf_pbc_tools::fft::fft_stockham;
use std::time::Instant;
fn main() {
    for m in [11usize, 15, 21, 31, 47] {
        let ng = m * m * m;
        let nrows = 64;
        let x = CTensor::from_planes(vec![0.5; nrows * ng], vec![0.25; nrows * ng]);
        let t = Instant::now();
        let g = fft_stockham(&x, [m, m, m], false).unwrap();
        let e1 = t.elapsed();
        let t2 = Instant::now();
        let _ = fft_stockham(&g, [m, m, m], true).unwrap();
        let e2 = t2.elapsed();
        println!(
            "mesh {m}^3 ({ng} pts) x{nrows} rows: fft {:?} ifft {:?} -> per K-build (64 pairs) {:.1} s",
            e1,
            e2,
            64.0 * (e1.as_secs_f64() + e2.as_secs_f64())
        );
    }
}

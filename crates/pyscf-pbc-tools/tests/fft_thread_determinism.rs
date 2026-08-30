//! W-02b (`KRKS-OPTIMISATION-PLAN.md`) — `transform_axis` batches its
//! independent rows across `rayon` workers now. Gate B (D-PBC-17) requires
//! that to be bit-identical regardless of `RAYON_NUM_THREADS`: each row's own
//! butterfly/DFT sequence is untouched, only the SCHEDULING of independent
//! rows changes, so there is no shared reduction whose order could move.
//!
//! Same child-process pattern as
//! `pyscf-algebra/tests/zoracle_determinism.rs`: rayon's global pool is built
//! once per process, so the thread count must be set before the process
//! starts.

use pyscf_algebra::CTensor;
use pyscf_pbc_tools::fft::fft_stockham;

const CHILD_ENV: &str = "PYSCF_RS_FFT_THREAD_CHILD";

/// A batch of `nb` rows over a mesh with three distinct, non-power-of-two,
/// non-trivial axis lengths (21 = 3*7, 25 = 5*5, 27 = 3*3*3) — the exact shape
/// `transform_axis` batches in `get_k_kpts`.
fn corpus() -> (CTensor, [usize; 3]) {
    let mesh = [21usize, 25, 27];
    let ngrid = mesh[0] * mesh[1] * mesh[2];
    let nb = 17usize; // deliberately not a multiple of any plausible chunk size
    let n = nb * ngrid;
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    let mut s = 0x9E3779B97F4A7C15u64;
    for _ in 0..n {
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        re.push(((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0);
        s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        im.push(((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0);
    }
    (CTensor::from_planes(re, im), mesh)
}

#[test]
#[ignore = "spawned by fft_stockham_is_bit_identical_across_rayon_thread_counts"]
fn fft_thread_child_emits_bits() {
    let (x, mesh) = corpus();
    let fwd = fft_stockham(&x, mesh, false).expect("fft");
    let back = fft_stockham(&fwd, mesh, true).expect("ifft");
    let mut re_acc = 0u64;
    let mut im_acc = 0u64;
    for i in 0..fwd.len() {
        re_acc ^= fwd.re[i].to_bits().rotate_left((i % 61) as u32);
        im_acc ^= fwd.im[i].to_bits().rotate_left((i % 61) as u32);
    }
    let mut back_re_acc = 0u64;
    let mut back_im_acc = 0u64;
    for i in 0..back.len() {
        back_re_acc ^= back.re[i].to_bits().rotate_left((i % 61) as u32);
        back_im_acc ^= back.im[i].to_bits().rotate_left((i % 61) as u32);
    }
    println!(
        "FFT_THREAD_BITS {re_acc:016x} {im_acc:016x} {back_re_acc:016x} {back_im_acc:016x}"
    );
}

fn run_child(threads: &str) -> String {
    let exe = std::env::current_exe().expect("current_exe");
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "fft_thread_child_emits_bits",
            "--ignored",
            "--nocapture",
        ])
        .env("RAYON_NUM_THREADS", threads)
        .env(CHILD_ENV, "1")
        .output()
        .expect("spawn child test process");
    assert!(
        out.status.success(),
        "child with RAYON_NUM_THREADS={threads} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    stdout
        .lines()
        .find(|l| l.starts_with("FFT_THREAD_BITS "))
        .unwrap_or_else(|| {
            panic!("child with RAYON_NUM_THREADS={threads} printed no FFT_THREAD_BITS line:\n{stdout}")
        })
        .to_owned()
}

/// D-PBC-17: `fft_stockham`'s batched transform is BIT-IDENTICAL at
/// `RAYON_NUM_THREADS in {1, 2, 8}`.
#[test]
fn fft_stockham_is_bit_identical_across_rayon_thread_counts() {
    if std::env::var(CHILD_ENV).is_ok() {
        return;
    }
    let one = run_child("1");
    let two = run_child("2");
    let eight = run_child("8");
    assert_eq!(
        one, two,
        "fft_stockham is NOT thread-count invariant (1 vs 2) — D-PBC-17 violated"
    );
    assert_eq!(
        one, eight,
        "fft_stockham is NOT thread-count invariant (1 vs 8) — D-PBC-17 violated"
    );
}

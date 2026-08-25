//! Plan 09-02 Task 8 — D-PBC-17: `oracle_zsum` / `oracle_zdot` are the ONLY
//! reductions numerical PBC code may use, and they must be BIT-IDENTICAL
//! regardless of `RAYON_NUM_THREADS`.
//!
//! The load-bearing test spawns this same test binary twice as a subprocess —
//! once with `RAYON_NUM_THREADS=1`, once with `=8` — because rayon's global
//! pool is built once per process and setting the variable from inside a
//! running test would be too late. Each child prints the raw `f64::to_bits()`
//! of the two planes; the parent asserts the two transcripts are equal.
//!
//! Verified here: 1e6-element `oracle_zsum` bit-identity across thread counts
//! and across repeated in-process calls; `oracle_zdot`'s `zdotc` combination;
//! agreement with a strictly sequential per-plane reference.
//!
//! Not verified here: the DEVICE reductions (`zreduce_sum_dense`,
//! `zdotc_dense`) — those are explicitly NOT order-stable, which is why
//! D-PBC-17 exists.

use pyscf_algebra::{CTensor, oracle_dot, oracle_sum, oracle_zdot, oracle_zsum};

/// Env flag the parent sets on the child process. Kept out of the child's own
/// assertions so the child does nothing but print.
const CHILD_ENV: &str = "PYSCF_RS_ZORACLE_CHILD";

/// Mixed-magnitude corpus — the catastrophic-cancellation shape from
/// `tests/oracle_determinism.rs`, extended to two planes with different
/// magnitude profiles so a plane mix-up cannot pass.
fn corpus(n: usize) -> CTensor {
    let mut re = Vec::with_capacity(n);
    let mut im = Vec::with_capacity(n);
    for i in 0..n {
        let f = i as f64;
        if i % 2 == 0 {
            re.push(f * 1e-3);
            im.push(-f * 1e-6 + 2.0);
        } else {
            re.push(-f * 1e-3 + 1.0);
            im.push(f * 1e-6);
        }
    }
    CTensor::from_planes(re, im)
}

const N: usize = 1_000_000;

#[test]
fn oracle_zsum_is_two_ordered_oracle_sums() {
    let x = corpus(4096);
    let (re, im) = oracle_zsum(&x);
    assert_eq!(re.to_bits(), oracle_sum(&x.re).to_bits());
    assert_eq!(im.to_bits(), oracle_sum(&x.im).to_bits());
}

#[test]
fn oracle_zdot_follows_the_zdotc_pattern() {
    let x = corpus(4096);
    let y = corpus(4096).conj();
    let (re, im) = oracle_zdot(&x, &y);
    let want_re = oracle_dot(&x.re, &y.re) + oracle_dot(&x.im, &y.im);
    let want_im = oracle_dot(&x.re, &y.im) - oracle_dot(&x.im, &y.re);
    assert_eq!(re.to_bits(), want_re.to_bits());
    assert_eq!(im.to_bits(), want_im.to_bits());

    // xᴴ·x is real and non-negative for any x.
    let (rr, ri) = oracle_zdot(&x, &x);
    assert!(rr > 0.0, "xᴴ·x must be positive, got {rr}");
    assert_eq!(ri, 0.0, "xᴴ·x must have exactly zero imaginary part");
}

#[test]
fn oracle_zsum_is_deterministic_in_process() {
    let x = corpus(N);
    let canonical = oracle_zsum(&x);
    for _ in 0..5 {
        let got = oracle_zsum(&x);
        assert_eq!(got.0.to_bits(), canonical.0.to_bits());
        assert_eq!(got.1.to_bits(), canonical.1.to_bits());
    }
}

/// The child half of the cross-thread-count check. `#[ignore]` so it only runs
/// when the parent names it explicitly; it prints and asserts nothing.
#[test]
#[ignore = "spawned by oracle_zsum_is_bit_identical_across_rayon_thread_counts"]
fn zoracle_child_emits_bits() {
    let x = corpus(N);
    let (re, im) = oracle_zsum(&x);
    let (dr, di) = oracle_zdot(&x, &x);
    println!(
        "ZSUM_BITS {:016x} {:016x} {:016x} {:016x}",
        re.to_bits(),
        im.to_bits(),
        dr.to_bits(),
        di.to_bits()
    );
}

fn run_child(threads: &str) -> String {
    let exe = std::env::current_exe().expect("current_exe");
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "zoracle_child_emits_bits",
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
        .find(|l| l.starts_with("ZSUM_BITS "))
        .unwrap_or_else(|| {
            panic!("child with RAYON_NUM_THREADS={threads} printed no ZSUM_BITS line:\n{stdout}")
        })
        .to_owned()
}

/// D-PBC-17 / ROADMAP success criterion 3, complex edition: the ordered complex
/// reductions over 1e6 elements are BIT-IDENTICAL at `RAYON_NUM_THREADS=1` and
/// `=8`. Compares raw `f64::to_bits()` transcripts — NOT an epsilon tolerance.
#[test]
fn oracle_zsum_is_bit_identical_across_rayon_thread_counts() {
    if std::env::var(CHILD_ENV).is_ok() {
        // Never recurse: a child must not spawn grandchildren.
        return;
    }
    let one = run_child("1");
    let eight = run_child("8");
    assert_eq!(
        one, eight,
        "oracle_zsum/oracle_zdot are NOT thread-count invariant \
         (RAYON_NUM_THREADS=1 vs =8) — D-PBC-17 violated"
    );

    // And the subprocess answer must equal this process's answer too.
    let x = corpus(N);
    let (re, im) = oracle_zsum(&x);
    let (dr, di) = oracle_zdot(&x, &x);
    let here = format!(
        "ZSUM_BITS {:016x} {:016x} {:016x} {:016x}",
        re.to_bits(),
        im.to_bits(),
        dr.to_bits(),
        di.to_bits()
    );
    assert_eq!(here, one, "in-process result differs from the subprocess");
}

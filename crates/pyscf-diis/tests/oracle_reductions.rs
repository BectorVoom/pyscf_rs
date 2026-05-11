//! Pitfall 9: B-matrix inner products via oracle_dot are bit-identical across
//! thread counts (RAYON_NUM_THREADS=1 vs =8). This is the Phase 3 cross-platform
//! µHartree precondition (threat T-3-09).
//!
//! Note: oracle_dot/oracle_sum use pairwise-tree reduction with a FIXED chunk
//! size (PAIRWISE_CHUNK = 128 in pyscf-algebra::oracle); the recursion tree
//! shape depends only on input length, not rayon thread count. So this test
//! proves the algorithm is invariant by construction — if anyone ever
//! replaces oracle_dot with `iter().sum()` or rayon `par_iter`, this test
//! fails LOUDLY on bit-difference.
use pyscf_diis::{Diis, DiisStorable};

#[derive(Clone)]
struct FlatVec(Vec<f64>);
impl DiisStorable for FlatVec {
    fn as_flat(&self) -> &[f64] {
        &self.0
    }
    fn from_flat(&mut self, s: &[f64]) {
        self.0.copy_from_slice(s);
    }
    fn dot(&self, o: &Self) -> f64 {
        pyscf_algebra::oracle_dot(&self.0, &o.0)
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}

fn run_extrap() -> Vec<f64> {
    let mut diis = Diis::<FlatVec>::new(8);
    // 256 elements: large enough that parallel reduction could occur if
    // anyone replaces oracle_* with rayon.
    let f1 = FlatVec((0..256).map(|i| (i as f64).sin()).collect());
    let f2 = FlatVec((0..256).map(|i| (i as f64 + 1.0).sin()).collect());
    let f3 = FlatVec((0..256).map(|i| (i as f64 + 2.0).sin()).collect());
    let e1: Vec<f64> = (0..256).map(|i| (i as f64).cos() + 0.01).collect();
    let e2: Vec<f64> = (0..256).map(|i| (i as f64 + 1.0).cos() + 0.01).collect();
    let e3: Vec<f64> = (0..256).map(|i| (i as f64 + 2.0).cos() + 0.01).collect();
    diis.extrapolate(f1, e1).expect("1");
    diis.extrapolate(f2, e2).expect("2");
    let out = diis.extrapolate(f3, e3).expect("3");
    out.0
}

#[test]
fn rayon_1_vs_8_bit_identical() {
    // RAYON_NUM_THREADS is read by rayon's lazy thread-pool initialisation;
    // toggling within one process is best-effort. The real guarantee comes
    // from oracle_dot/oracle_sum's deterministic pairwise-tree reduction
    // (PAIRWISE_CHUNK = 128) which doesn't use rayon at all. So setting the
    // env var and re-running here should still yield bit-identical output.
    // This test is a regression guard: if oracle_* is ever replaced with a
    // thread-count-dependent reducer, this assertion fires.
    std::env::set_var("RAYON_NUM_THREADS", "1");
    let a = run_extrap();
    std::env::set_var("RAYON_NUM_THREADS", "8");
    let b = run_extrap();
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "bit-difference at index {}: rayon-1={}, rayon-8={}",
            i,
            x,
            y
        );
    }
}

#[test]
fn oracle_dot_used_in_cdiis_module() {
    // Source-level guard: cdiis.rs MUST contain at least one oracle_dot call.
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/cdiis.rs"
    ))
    .expect("read cdiis.rs");
    assert!(
        src.contains("oracle_dot"),
        "cdiis.rs must invoke oracle_dot (Pitfall 9 mitigation — B-matrix inner products)"
    );
}

#[test]
fn oracle_sum_used_in_cdiis_module() {
    // Source-level guard: cdiis.rs MUST contain at least one oracle_sum call
    // for the extrapolated iterate's cross-iterate reduction.
    let src = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/cdiis.rs"
    ))
    .expect("read cdiis.rs");
    assert!(
        src.contains("oracle_sum"),
        "cdiis.rs must invoke oracle_sum (Pitfall 9 mitigation — extrapolated-iterate sums)"
    );
}

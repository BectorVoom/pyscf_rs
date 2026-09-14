#![cfg(feature = "cpu")]

use cubecl::Runtime;
use pyscf_algebra::{AlgebraClient, oracle_sum};
use pyscf_kernels::pbc::multigrid_grad::contract_atom_grid;

fn client() -> AlgebraClient {
    AlgebraClient::Cpu(cubecl_cpu::CpuRuntime::client(&cubecl_cpu::CpuDevice))
}

#[test]
fn complex_rows_match_literal_oracle_reduction() {
    let client = client();
    let (natm, ng) = (3, 257);
    let fr: Vec<f64> = (0..natm * 3 * ng)
        .map(|i| (i % 31) as f64 / 8.0 - 2.0)
        .collect();
    let fi: Vec<f64> = (0..fr.len())
        .map(|i| (i % 19) as f64 / 16.0 - 0.5)
        .collect();
    let rr: Vec<f64> = (0..ng).map(|i| (i % 7) as f64 - 3.0).collect();
    let ri: Vec<f64> = (0..ng).map(|i| (i % 11) as f64 / 4.0).collect();
    let actual = contract_atom_grid(&client, natm, &fr, &fi, &rr, &ri).unwrap();
    for row in 0..natm * 3 {
        let terms: Vec<f64> = (0..ng)
            .map(|g| fr[row * ng + g] * rr[g] - fi[row * ng + g] * ri[g])
            .collect();
        let expected = oracle_sum(&terms);
        assert_eq!(actual[row / 3][row % 3].to_bits(), expected.to_bits());
    }
    assert_eq!(
        actual,
        contract_atom_grid(&client, natm, &fr, &fi, &rr, &ri).unwrap()
    );
    println!("atom-grid exact rows: {actual:?}");
}

#[test]
fn shape_validation_and_empty_grids() {
    let c = client();
    assert_eq!(
        contract_atom_grid(&c, 2, &[], &[], &[], &[]).unwrap(),
        vec![[0.0; 3]; 2]
    );
    assert!(contract_atom_grid(&c, 1, &[0.0; 3], &[0.0; 2], &[1.0], &[0.0]).is_err());
    assert!(contract_atom_grid(&c, 1, &[0.0; 3], &[0.0; 3], &[1.0], &[]).is_err());
    assert!(contract_atom_grid(&c, usize::MAX, &[], &[], &[1.0], &[0.0]).is_err());
}

// ---------------------------------------------------------------------------
// D-PBC-31 clause 10 determinism: one implementation, ONE determinism test,
// run once here and relied on by 18-05's fused hcore and 18-09's
// get_nuc_nuc_grad / vpploc_part1_nuc_grad (all three consume this exact
// reduction). Same child-process shape as
// `pyscf-algebra/tests/zoracle_determinism.rs`: rayon's global pool is built
// once per process, so the parent spawns this binary twice (once with
// `RAYON_NUM_THREADS=1`, once with `=8`) and compares bit transcripts.
// Run under `release-oracle` (the profile 18-09-PLAN Task 4 names).
// ---------------------------------------------------------------------------

/// Env flag the parent sets on the child process.
const CHILD_ENV: &str = "PYSCF_RS_ATOM_GRID_CHILD";

/// Mixed-magnitude corpus sized past the kernel's 256-lane dispatch and the
/// host `oracle_sum` grain, so both the device products and the row
/// reductions are exercised.
fn det_corpus() -> (usize, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let (natm, ng) = (4, 1021);
    let n = natm * 3 * ng;
    let fr: Vec<f64> = (0..n).map(|i| (i % 37) as f64 / 7.0 - 2.5).collect();
    let fi: Vec<f64> = (0..n).map(|i| (i % 23) as f64 / 13.0 - 0.75).collect();
    let rr: Vec<f64> = (0..ng).map(|i| (i % 11) as f64 - 5.0).collect();
    let ri: Vec<f64> = (0..ng).map(|i| (i % 17) as f64 / 5.0).collect();
    (natm, fr, fi, rr, ri)
}

fn det_bits() -> String {
    let c = client();
    let (natm, fr, fi, rr, ri) = det_corpus();
    let got = contract_atom_grid(&c, natm, &fr, &fi, &rr, &ri).expect("contract");
    let mut s = String::from("ATOM_GRID_BITS");
    for row in &got {
        for v in row {
            s.push_str(&format!(" {:016x}", v.to_bits()));
        }
    }
    s
}

/// The child half. `#[ignore]` so it only runs when the parent names it
/// explicitly; it prints and asserts nothing.
#[test]
#[ignore = "spawned by atom_grid_contract_is_bit_identical_across_rayon_thread_counts"]
fn atom_grid_child_emits_bits() {
    println!("{}", det_bits());
}

fn run_child(threads: &str) -> String {
    let exe = std::env::current_exe().expect("current_exe");
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "atom_grid_child_emits_bits",
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
        .find(|l| l.starts_with("ATOM_GRID_BITS "))
        .unwrap_or_else(|| {
            panic!(
                "child with RAYON_NUM_THREADS={threads} printed no ATOM_GRID_BITS line:\n{stdout}"
            )
        })
        .to_owned()
}

/// D-PBC-17 / D-PBC-31 clause 10: the shared `(natm,3)` reduction is
/// BIT-IDENTICAL at `RAYON_NUM_THREADS=1` and `=8`. Compares raw
/// `f64::to_bits()` transcripts — NOT an epsilon tolerance.
#[test]
fn atom_grid_contract_is_bit_identical_across_rayon_thread_counts() {
    if std::env::var(CHILD_ENV).is_ok() {
        // Never recurse: a child must not spawn grandchildren.
        return;
    }
    let one = run_child("1");
    let eight = run_child("8");
    assert_eq!(
        one, eight,
        "contract_atom_grid is NOT thread-count invariant \
         (RAYON_NUM_THREADS=1 vs =8) — D-PBC-17 violated"
    );

    // And the subprocess answer must equal this process's answer too.
    assert_eq!(
        det_bits(),
        one,
        "in-process result differs from the subprocess"
    );
}

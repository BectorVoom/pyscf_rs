//! A first end-to-end `KRCCSD` run — the smallest thing that exercises the
//! whole 16-04 + 16-05 chain: `_ERIS` through the symmetry loop, the
//! intermediates, `update_amps`, the DIIS iteration and `energy`.
//!
//! Oracle-free. `--release` because it converges an SCF.

mod common;

use std::sync::Arc;

use pyscf_pbc_cc::kccsd_rhf::{Krccsd, init_amps};
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_scf::{KScfConfig, Krhf};
use pyscf_runtime::ZWorkspacePool;

#[test]
#[ignore = "converges an SCF; run with --release"]
fn krccsd_runs_on_diamond_112() {
    let cell = common::diamond([15, 15, 15]);
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let df = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
    let mut mf = Krhf::from_df(Box::new(df));
    mf.exxdiv = None;
    let mut cfg = KScfConfig::for_cell(&cell);
    cfg.conv_tol = 1e-10;
    let scf = mf.kernel(&cfg).expect("KRHF converges");
    assert!(scf.converged);

    let df2 = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
    let mut cc = Krccsd::new(&scf, &df2).expect("KRCCSD builds");
    let eris = cc.ao2mo().expect("_ERIS");
    let (emp2, _, _) = init_amps(&eris, &cc.padded, &cc.khelper.kconserv).expect("init_amps");
    println!("e_hf {}  emp2 {emp2}", scf.e_tot);

    let pool = Arc::new(ZWorkspacePool::new(ZWorkspacePool::DEFAULT_BUDGET_BYTES));
    let _ = &pool;
    let res = cc.kernel_with(&eris).expect("KRCCSD kernel");
    println!(
        "e_corr {}  emp2 {}  converged {}  cycles {}",
        res.e_corr, res.emp2, res.converged, res.cycles
    );
    assert!(res.converged, "KRCCSD must converge on diamond [1,1,2]");
    assert!(
        res.e_corr < 0.0 && res.e_corr > -1.0,
        "e_corr {} is not a plausible correlation energy",
        res.e_corr
    );
}

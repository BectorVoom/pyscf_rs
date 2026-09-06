//! Oracle-FREE gates for `KUCCSD` (plan 16-06).
//!
//! These need no PySCF: each is an identity the unrestricted equations must
//! satisfy on their own terms.

mod common;

use std::sync::Arc;

use pyscf_pbc_cc::ZArr;
use pyscf_pbc_cc::kccsd_rhf::KrccsdOpts;
use pyscf_pbc_cc::kccsd_uhf::{Kuccsd, init_amps, update_amps};
use pyscf_pbc_cc::kintermediates_uhf::{UT1, UT2};
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_scf::{KScfConfig, Kuhf};
use pyscf_runtime::ZWorkspacePool;

fn maxdiff(a: &ZArr, b: &ZArr) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut m = 0.0_f64;
    for i in 0..a.len() {
        m = m
            .max((a.data().re[i] - b.data().re[i]).abs())
            .max((a.data().im[i] - b.data().im[i]).abs());
    }
    m
}

/// **`update_amps` at ZERO amplitudes is `init_amps`.**
///
/// At `t1 = t2 = 0` every term of the doubles equation vanishes except the bare
/// integral driver (`kccsd_uhf.py:203-206`), and that driver is — line for line
/// — what `init_amps` divides by the same denominators (`:739-743`). So the two
/// must agree to the last bit, and the identity holds independently of the
/// mean field, the cell and the k-mesh.
///
/// This is the cheapest possible check on the two `ovov` transposes
/// `(0,2,1,3)` and `(2,0,1,3)`, which are the pair a reader is most likely to
/// swap: they have the SAME SHAPE, so nothing but a numerical comparison
/// distinguishes them.
#[test]
#[ignore = "converges an SCF; run with --release"]
fn update_amps_at_zero_amplitudes_reproduces_init_amps() {
    let cell = common::diamond([15, 15, 15]);
    let kpts = cell.make_kpts([1, 1, 2]).expect("kpts");
    let mut mf = Kuhf::from_df(Box::new(Fftdf::new(cell.clone(), &kpts).expect("fftdf")));
    mf.exxdiv = None;
    let mut cfg = KScfConfig::for_cell(&cell);
    cfg.conv_tol = 1e-10;
    let scf = mf.kernel(&cfg).expect("KUHF converges");
    assert!(scf.converged);

    let df = Fftdf::new(cell.clone(), &kpts).expect("fftdf");
    let cc = Kuccsd::new(&scf, &df).expect("KUCCSD builds");
    let eris = cc.ao2mo().expect("_ChemistsERIs");
    let (pa, pb) = (&cc.padded.0, &cc.padded.1);
    let kc = &cc.khelper.kconserv;

    let (_, t1, t2) = init_amps(&eris, (pa, pb), kc).expect("init_amps");
    let z1: UT1 = (ZArr::zeros(t1.0.shape()), ZArr::zeros(t1.1.shape()));
    let z2: UT2 = (
        ZArr::zeros(t2.0.shape()),
        ZArr::zeros(t2.1.shape()),
        ZArr::zeros(t2.2.shape()),
    );
    let pool = Arc::new(ZWorkspacePool::new(4_000_000_000));
    let (t1n, t2n) = update_amps(&pool, &z1, &z2, &eris, (pa, pb), kc, &KrccsdOpts::default())
        .expect("update_amps");

    for (got, want, name) in [
        (&t2n.0, &t2.0, "t2aa"),
        (&t2n.1, &t2.1, "t2ab"),
        (&t2n.2, &t2.2, "t2bb"),
    ] {
        let d = maxdiff(got, want);
        println!("update_amps(0,0) {name} vs init_amps: max|Δ| {d:e}");
        // Not bit-identity: `init_amps` divides each of the two terms by the
        // denominator and subtracts, while `update_amps` subtracts and then
        // divides. Algebraically the same, one rounding apart — measured
        // `6.9e-18`, i.e. the last ulp of a `~0.05` amplitude. Anything larger
        // is a real difference, so the gate sits three orders under a plausible
        // sign or transpose error and still far above the rounding.
        assert!(
            d < 1e-15,
            "{name}: the bare integral driver and init_amps disagree by {d:e}, \
             so one of the two `ovov` transposes is wrong"
        );
    }
    // The singles at zero amplitudes are just `conj(fov) / eia`.
    for (got, name) in [(&t1n.0, "t1a"), (&t1n.1, "t1b")] {
        println!("update_amps(0,0) {name} norm {:e}", norm(got));
    }
}

fn norm(a: &ZArr) -> f64 {
    a.data()
        .re
        .iter()
        .zip(&a.data().im)
        .map(|(r, i)| r * r + i * i)
        .sum::<f64>()
        .sqrt()
}

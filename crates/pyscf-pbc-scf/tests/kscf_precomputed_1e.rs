//! `KScfConfig::{s1e, h1e, on_cycle}` — the hooks a long periodic run needs to
//! checkpoint and resume: precomputed one-electron matrices must move no bit
//! (the driver only skips recomputing them), and the cycle hook must see every
//! cycle's density, the last one being the result's.

mod common;

use std::sync::{Arc, Mutex};

use common::diamond;
use pyscf_pbc_scf::KOverrideHooks;
use pyscf_pbc_scf::krhf::Krhf;
use pyscf_pbc_scf::types::{CycleHook, KInitGuess, KScfConfig};

fn cfg() -> KScfConfig {
    let mut c = KScfConfig::for_cell(&diamond());
    // The settings `df_swap.rs` converges diamond at gamma with.
    c.conv_tol = 1e-11;
    c.max_cycle = 60;
    c
}

#[test]
fn precomputed_one_electron_matrices_are_bit_identical() {
    let kpts = [[0.0; 3]];
    let mf = Krhf::new(diamond(), &kpts).expect("krhf");
    let plain = mf.kernel(&cfg()).expect("scf");

    let seen: Arc<Mutex<Vec<(u32, f64, bool)>>> = Arc::default();
    let last_dm = Arc::new(Mutex::new(None));
    let (s, d) = (Arc::clone(&seen), Arc::clone(&last_dm));
    let c = KScfConfig {
        s1e: Some(mf.get_ovlp().expect("ovlp")),
        h1e: Some(mf.get_hcore().expect("hcore")),
        on_cycle: Some(CycleHook(Arc::new(move |st: &pyscf_pbc_scf::types::CycleState<'_>| {
            s.lock().unwrap().push((st.cycle, st.e_tot, st.converged));
            *d.lock().unwrap() = Some(st.dm.clone());
        }))),
        ..cfg()
    };
    let pre = mf.kernel(&c).expect("scf with precomputed 1e");
    assert!(plain.converged && pre.converged, "fixture must converge");
    assert_eq!(plain.e_tot.to_bits(), pre.e_tot.to_bits(), "{} vs {}", plain.e_tot, pre.e_tot);
    assert_eq!(plain.cycles, pre.cycles);

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len() as u32, pre.cycles, "one hook call per cycle");
    assert!(seen.iter().enumerate().all(|(i, (c, _, _))| *c == i as u32));
    // `converged` is reported atomically with the density: only the last cycle.
    assert!(seen[..seen.len() - 1].iter().all(|r| !r.2) && seen.last().unwrap().2);
    assert_eq!(seen.last().unwrap().1.to_bits(), pre.e_tot.to_bits());
    assert_eq!(last_dm.lock().unwrap().as_ref(), Some(&pre.dm), "hook's last density is the result's");

    // Resuming from the hook's density starts converged.
    let resumed = mf
        .kernel(&KScfConfig { init_guess: KInitGuess::UserDm(pre.dm.clone()), ..cfg() })
        .expect("resumed scf");
    assert!(resumed.converged);
    assert!(resumed.cycles <= 2, "resume took {} cycles", resumed.cycles);
    assert!((resumed.e_tot - pre.e_tot).abs() < 1e-9);
}

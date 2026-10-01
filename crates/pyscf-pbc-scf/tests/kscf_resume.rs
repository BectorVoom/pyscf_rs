//! Resuming a periodic SCF from a checkpointed [`CycleState`]: with
//! `init_guess = UserDm(dm)`, `fock_last = fock` and `first_cycle = cycle + 1`
//! the resumed run repeats the uninterrupted one BIT FOR BIT through the
//! damped (pre-DIIS) phase — the damping sees the same previous Fock, and the
//! cycle counter keeps the damping/DIIS switch where it was. Without
//! `fock_last` the first resumed step is undamped and the energies differ.

mod common;

use std::sync::{Arc, Mutex};

use common::diamond;
use pyscf_pbc_scf::krhf::Krhf;
use pyscf_pbc_scf::types::{CycleHook, CycleState, KDms, KInitGuess, KScfConfig};

type Log = Arc<Mutex<Vec<(u32, u64, KDms, KDms)>>>;

fn cfg(log: &Log) -> KScfConfig {
    let l = Arc::clone(log);
    let mut c = KScfConfig::for_cell(&diamond());
    c.conv_tol = 1e-14; // never converge inside the window
    c.max_cycle = 7;
    c.damp = 0.6;
    c.diis_start_cycle = 8;
    c.on_cycle = Some(CycleHook(Arc::new(move |st: &CycleState<'_>| {
        l.lock().unwrap().push((st.cycle, st.e_tot.to_bits(), st.dm.clone(), st.fock.clone()));
    })));
    c
}

#[test]
fn resumed_damped_cycles_are_bit_identical() {
    let kpts = [[0.0; 3], [0.11, -0.07, 0.19]];
    let mf = Krhf::new(diamond(), &kpts).expect("krhf");

    let full: Log = Arc::default();
    mf.kernel(&cfg(&full)).expect("uninterrupted scf");
    let full = full.lock().unwrap().clone();
    assert_eq!(full.len(), 7, "fixture must run all 7 cycles");

    // Stop after cycle 2, resume from its state.
    let (cut, _, dm, fock) = full[2].clone();
    let resumed: Log = Arc::default();
    mf.kernel(&KScfConfig {
        init_guess: KInitGuess::UserDm(dm.clone()),
        fock_last: Some(fock),
        first_cycle: cut + 1,
        ..cfg(&resumed)
    })
    .expect("resumed scf");
    let resumed = resumed.lock().unwrap().clone();
    let got: Vec<(u32, u64)> = resumed.iter().map(|r| (r.0, r.1)).collect();
    let want: Vec<(u32, u64)> = full[3..].iter().map(|r| (r.0, r.1)).collect();
    assert_eq!(got, want, "resumed cycles must repeat the uninterrupted run bit for bit");

    // Control: without the Fock seed the first resumed step is undamped.
    let naive: Log = Arc::default();
    mf.kernel(&KScfConfig {
        init_guess: KInitGuess::UserDm(dm),
        first_cycle: cut + 1,
        ..cfg(&naive)
    })
    .expect("naive resume");
    let first = naive.lock().unwrap()[0].1;
    assert_ne!(first, full[3].1, "the fixture must be sensitive to the Fock seed");
}

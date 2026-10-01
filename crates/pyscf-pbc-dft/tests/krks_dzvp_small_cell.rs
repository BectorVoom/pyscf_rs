//! KRKS/PBE on a small TaO cell with `gth-szv-molopt-sr` and
//! `gth-dzvp-molopt-sr` (s, p, d, f on Ta), with the YTa7O19 stage-1 recipe
//! (Gaussian 0.03 Ha, damping, level shift). Both SCFs must converge and the
//! larger basis — which shares SZV's primitives — must not raise the energy
//! by more than grid noise. Guards the DZVP path (f shells in AO values,
//! gradients, XC, J, the local PP) end to end at a size that runs in minutes.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_dft::krks::Krks;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};
use pyscf_pbc_scf::types::KScfConfig;

fn tao_cell(basis: &str) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("Ta".into(), [0.0, 0.0, 0.0]),
                ("O".into(), [1.95, 0.0, 0.0]),
            ]),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix([[3.9, 0.0, 0.0], [0.0, 3.9, 0.0], [0.0, 0.0, 3.9]]),
        pseudo: Some("gth-pbe".into()),
        ke_cutoff: Some(60.0),
        ..Default::default()
    })
    .expect("TaO cell must build")
}

fn scf(basis: &str) -> (bool, f64, Vec<f64>) {
    let cell = tao_cell(basis);
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
    let mut mf = Krks::new(cell, &kpts, "pbe").expect("krks");
    mf.smearing = Some(pyscf_pbc_scf::smearing::Smearing::gaussian(0.03));
    let base = KScfConfig::for_cell(mf.cell());
    let energies = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let e2 = std::sync::Arc::clone(&energies);
    let cfg = KScfConfig {
        conv_tol: 1e-7,
        max_cycle: 60,
        damp: 0.7,
        level_shift: 0.3,
        diis_space: 16,
        diis_start_cycle: 8,
        on_cycle: Some(pyscf_pbc_scf::types::CycleHook(std::sync::Arc::new(move |st: &pyscf_pbc_scf::types::CycleState<'_>| {
            e2.lock().unwrap().push(st.e_tot);
        }))),
        ..base
    };
    let r = mf.kernel(&cfg).expect("scf");
    let e = energies.lock().unwrap().clone();
    (r.converged, r.e_tot, e)
}

#[test]
fn dzvp_scf_converges_below_szv() {
    let (c_szv, e_szv, h_szv) = scf("gth-szv-molopt-sr");
    let (c_dzvp, e_dzvp, h_dzvp) = scf("gth-dzvp-molopt-sr");
    eprintln!("SZV  converged={c_szv} e={e_szv:.8} cycles {:?}", h_szv.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());
    eprintln!("DZVP converged={c_dzvp} e={e_dzvp:.8} cycles {:?}", h_dzvp.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>());
    assert!(c_szv && c_dzvp, "SCF must converge (szv {c_szv}, dzvp {c_dzvp})");
    assert!(e_dzvp < e_szv + 1e-3, "DZVP {e_dzvp} above SZV {e_szv}");
}

/// The `pre` → `project` → `s1` path of the YTa7O19 pipeline on the small
/// cell: projecting a density onto its own basis is the identity; the
/// converged SZV density projected into DZVP keeps (nearly) its electron
/// count, and a DZVP SCF started from it reaches the MINAO-started energy in
/// fewer cycles.
#[test]
fn projected_szv_density_starts_the_dzvp_scf() {
    use pyscf_pbc_scf::KOverrideHooks;
    use pyscf_pbc_scf::types::KInitGuess;

    let szv = tao_cell("gth-szv-molopt-sr");
    let dzvp = tao_cell("gth-dzvp-molopt-sr");
    let kpts = make_kpts_default(&szv, [2, 2, 2]).expect("kpts");
    let mut pre = Krks::new(szv, &kpts, "pbe").expect("szv krks");
    pre.smearing = Some(pyscf_pbc_scf::smearing::Smearing::gaussian(0.03));
    let pre_cfg = KScfConfig { conv_tol: 1e-9, max_cycle: 60, damp: 0.7, level_shift: 0.3, diis_space: 16,
                               diis_start_cycle: 8, ..KScfConfig::for_cell(pre.cell()) };
    let pre_scf = pre.kernel(&pre_cfg).expect("szv scf");
    assert!(pre_scf.converged);

    // Identity on the same basis.
    let s_pre = pre.get_ovlp().expect("szv ovlp");
    let same = pyscf_pbc_scf::addons::project_dm_nr2nr(pre.cell(), &pre_scf.dm[0], pre.cell(), &s_pre, &kpts)
        .expect("self projection");
    let mut worst = 0.0_f64;
    for (a, b) in same.iter().zip(&pre_scf.dm[0]) {
        for t in 0..a.re.len() {
            worst = worst.max((a.re[t] - b.re[t]).abs()).max((a.im[t] - b.im[t]).abs());
        }
    }
    assert!(worst < 1e-9, "self projection is not the identity: {worst:e}");

    // SZV -> DZVP.
    let mut mf = Krks::new(dzvp, &kpts, "pbe").expect("dzvp krks");
    mf.smearing = Some(pyscf_pbc_scf::smearing::Smearing::gaussian(0.03));
    let s_dzvp = mf.get_ovlp().expect("dzvp ovlp");
    let nao = mf.cell().mol.nao_nr;
    let dm0 = pyscf_pbc_scf::addons::project_dm_nr2nr(pre.cell(), &pre_scf.dm[0], mf.cell(), &s_dzvp, &kpts)
        .expect("szv -> dzvp");
    let ne: f64 = dm0.iter().zip(&s_dzvp).map(|(d, s)| pyscf_pbc_df::zlinalg::ztrace_ab(d, s, nao).0).sum::<f64>()
        / kpts.len() as f64;
    let want = mf.cell().tot_electrons(1) as f64;
    eprintln!("projected electrons {ne:.8} of {want}");
    assert!((ne - want).abs() < 0.05 * want && ne <= want + 1e-8, "projection lost too much charge: {ne}");

    let cfg = |init: KInitGuess| KScfConfig { conv_tol: 1e-9, max_cycle: 60, damp: 0.7, level_shift: 0.3,
                                              diis_space: 16, diis_start_cycle: 8, init_guess: init,
                                              ..KScfConfig::for_cell(mf.cell()) };
    let from_minao = mf.kernel(&cfg(KInitGuess::Minao)).expect("dzvp from minao");
    let from_szv = mf.kernel(&cfg(KInitGuess::UserDm(vec![dm0]))).expect("dzvp from szv");
    eprintln!(
        "DZVP from MINAO: {} cycles, e {:.10}; from projected SZV: {} cycles, e {:.10}",
        from_minao.cycles, from_minao.e_tot, from_szv.cycles, from_szv.e_tot
    );
    assert!(from_minao.converged && from_szv.converged);
    assert!((from_minao.e_tot - from_szv.e_tot).abs() < 1e-6, "different SCF solutions");
    assert!(from_szv.cycles < from_minao.cycles, "the projected start must save cycles");
}

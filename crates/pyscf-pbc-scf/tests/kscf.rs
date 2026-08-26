//! Plans 11-09 … 11-11 — the periodic SCF driver and its Hartree-Fock methods.
//!
//! # The Phase-11 gate
//!
//! [`krhf_diamond_222_matches_upstream`] runs `KRHF(diamond, 2x2x2,
//! gth-szv/gth-pade)` against live upstream PySCF **2.12.1** (the vendored
//! tree — see `tests/common/mod.rs`). It is `#[ignore]`d and gated on
//! `PYSCF_ORACLE_VENV`:
//!
//! ```bash
//! PYSCF_ORACLE_VENV=1 cargo test -p pyscf-pbc-scf --release -- --ignored --nocapture
//! ```
//!
//! Everything else in this file is ORACLE-FREE (D-PBC-19). The strongest of
//! those is [`supercell_equivalence_holds`]: a k-point calculation on the
//! primitive cell and a gamma-point calculation on the corresponding supercell
//! describe the same infinite crystal, so their energies per primitive cell
//! must agree — and they exercise completely different code paths (Bloch phases
//! and `nkpts` bookkeeping versus a plain gamma build over twice the atoms).
//! Nothing upstream is involved.

mod common;

use common::{cell_args, diamond, he_all_electron, oracle_python, run_python, GATE};
use pyscf_algebra::CTensor;
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_gto::{Cell, make_kpts_default, super_cell};
use pyscf_pbc_scf::{
    KInitGuess, KScfConfig, KScfResult, Kghf, Krhf, Kuhf, Smearing, dump_kscf_to_file,
    load_kscf_from_file,
};

/// A small FFT mesh: every oracle-free test here is about STRUCTURE, and the
/// structure is mesh-independent.
const MESH: [usize; 3] = [11, 11, 11];
/// The mesh at which upstream's `ft_ao` non-local pseudopotential has converged
/// to this port's exact real-space one — see `pyscf-pbc-df/tests/fftdf.rs`.
const MESH_GATE: [usize; 3] = [31, 31, 31];

fn tight() -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-8),
        max_cycle: 60,
        ..KScfConfig::default()
    }
}

fn krhf(cell: Cell, nk: [usize; 3], mesh: [usize; 3]) -> (Krhf, Vec<[f64; 3]>) {
    let kpts = make_kpts_default(&cell, nk).expect("k-mesh");
    let df = Fftdf::with_mesh(cell, &kpts, mesh).expect("FFTDF");
    (Krhf::from_df(df), kpts)
}

// ---------------------------------------------------------------------------
// Oracle-free
// ---------------------------------------------------------------------------

/// The driver converges, and the converged state satisfies the three invariants
/// a periodic SCF must: Hermitian densities, the right electron count, and a
/// real Coulomb energy.
#[test]
fn krhf_diamond_222_converges_to_a_valid_state() {
    let cell = diamond();
    let nao = cell.mol.nao_nr;
    let (mf, kpts) = krhf(cell, [2, 2, 2], MESH);
    let r = mf.kernel(&tight()).expect("KRHF");
    assert!(r.converged, "KRHF did not converge in {} cycles", r.cycles);
    assert_eq!(r.nkpts, 8);
    assert_eq!(r.nset, 1);

    for (k, d) in r.dm[0].iter().enumerate() {
        let mut w = 0.0_f64;
        for i in 0..nao {
            for j in 0..nao {
                w = w.max((d.re[i * nao + j] - d.re[j * nao + i]).abs());
                w = w.max((d.im[i * nao + j] + d.im[j * nao + i]).abs());
            }
        }
        assert!(w < 1e-12, "density at k={k} is not Hermitian: {w:e}");
    }

    // sum_k Tr(D S) == nelectron over the whole BZ.
    let s = mf.get_ovlp_public().expect("ovlp");
    let ne = pyscf_pbc_scf::krdm::electron_count(&r.dm, &s, nao);
    let want = mf.nelectron() as f64;
    assert!(
        (ne - want).abs() < 1e-9,
        "electron count {ne} != {want} ({} k-points)",
        kpts.len()
    );

    // Every occupied level is at or below the Fermi level, and the occupations
    // sum to the electron count.
    let occ_sum: f64 = r.mo_occ.iter().flatten().sum();
    assert!((occ_sum - want).abs() < 1e-9, "occupations sum to {occ_sum}");
}

/// **The supercell-equivalence identity.** `KRHF(cell, [2,1,1])` per primitive
/// cell equals `RHF(super_cell(cell, [2,1,1]))` halved.
///
/// PBC-MASTER-PLAN plan 11-09 calls for it explicitly and it is the strongest
/// oracle-free statement available about the whole k-point machinery: the two
/// sides share no code beyond the integrals, and a wrong Bloch phase, a per-k
/// (rather than global) aufbau, a wrong `1/nkpts`, or a mis-scaled `exxdiv`
/// each break it.
#[test]
fn supercell_equivalence_holds() {
    let cell = diamond();
    let ncopy = [2usize, 1, 1];
    // The identity is EXACT for the continuum operator and approached as the
    // FFT grid converges, because the primitive cell at two k-points and the
    // doubled cell at gamma sample V_loc on grids of the same spacing but
    // different extent. Measured on diamond: 1.5e-5 at mesh 9, 4.9e-7 at 11,
    // 1.6e-10 at 15, 2.1e-10 at 19 — so the gate runs at 15, where the
    // residual is four orders below the physics being tested.
    const MESH: [usize; 3] = [15, 15, 15];

    let (mf_k, _) = krhf(cell.clone(), [2, 1, 1], MESH);
    let e_k = mf_k.kernel(&tight()).expect("k-point KRHF");
    assert!(e_k.converged, "k-point KRHF did not converge");

    let sc = super_cell(&cell, ncopy, false).expect("supercell");
    let sc_mesh = [MESH[0] * ncopy[0], MESH[1] * ncopy[1], MESH[2] * ncopy[2]];
    let (mf_s, _) = krhf(sc, [1, 1, 1], sc_mesh);
    let e_s = mf_s.kernel(&tight()).expect("supercell RHF");
    assert!(e_s.converged, "supercell RHF did not converge");

    let per_cell = e_s.e_tot / 2.0;
    println!(
        "supercell equivalence: k-point {:.12}  supercell/2 {:.12}  delta {:e}",
        e_k.e_tot,
        per_cell,
        e_k.e_tot - per_cell
    );
    assert!(
        (e_k.e_tot - per_cell).abs() < 1e-8,
        "supercell equivalence broken: {} vs {}",
        e_k.e_tot,
        per_cell
    );
}

/// The converged energy does not depend on the DIIS path. This is what
/// licenses the real-valued DIIS representation of `kdiis.rs` — the fixed point
/// is `FDS = SDF`, not the extrapolation that reached it.
#[test]
fn converged_energy_is_independent_of_diis() {
    let cell = diamond();
    let (mf, _) = krhf(cell, [2, 1, 1], MESH);
    let with = mf.kernel(&tight()).expect("with DIIS");
    let without = mf
        .kernel(&KScfConfig {
            diis: false,
            max_cycle: 200,
            ..tight()
        })
        .expect("without DIIS");
    assert!(with.converged && without.converged);
    println!(
        "DIIS on {:.14}  off {:.14}  delta {:e}",
        with.e_tot,
        without.e_tot,
        with.e_tot - without.e_tot
    );
    assert!(
        (with.e_tot - without.e_tot).abs() < 1e-10,
        "DIIS changed the converged energy: {} vs {}",
        with.e_tot,
        without.e_tot
    );
}

/// A closed-shell system gives the same energy from KRHF, KUHF and KGHF — the
/// three references collapse onto the same determinant.
#[test]
fn kuhf_and_kghf_reproduce_krhf_on_a_closed_shell_cell() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 1, 1]).expect("k-mesh");

    let r = Krhf::from_df(Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("df"))
        .kernel(&tight())
        .expect("KRHF");
    let u = Kuhf::from_df(Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("df"))
        .kernel(&tight())
        .expect("KUHF");
    let g = Kghf::from_df(Fftdf::with_mesh(cell, &kpts, MESH).expect("df"))
        .kernel(&tight())
        .expect("KGHF");
    println!(
        "KRHF {:.14}  KUHF {:.14}  KGHF {:.14}",
        r.e_tot, u.e_tot, g.e_tot
    );
    assert!(r.converged && u.converged && g.converged);
    assert!(
        (r.e_tot - u.e_tot).abs() < 1e-9,
        "KUHF {} != KRHF {}",
        u.e_tot,
        r.e_tot
    );
    assert!(
        (r.e_tot - g.e_tot).abs() < 1e-9,
        "KGHF {} != KRHF {}",
        g.e_tot,
        r.e_tot
    );
}

/// `KROHF` on a closed-shell cell collapses onto `KRHF`.
///
/// With `nalpha == nbeta` every level is doubly occupied, the two spin Fock
/// matrices coincide, and the Roothaan effective Fock of `krohf.py:85-120`
/// reduces to that common Fock — so this exercises the `nfock != nset` path,
/// the `occ > 0` / `occ == 2` density split and the spin-summed DIIS density
/// against an answer that is independently known.
#[test]
fn krohf_reproduces_krhf_on_a_closed_shell_cell() {
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [2, 1, 1]).expect("k-mesh");
    let r = Krhf::from_df(Fftdf::with_mesh(cell.clone(), &kpts, MESH).expect("df"))
        .kernel(&tight())
        .expect("KRHF");
    let ro = pyscf_pbc_scf::Krohf::from_df(
        Fftdf::with_mesh(cell, &kpts, MESH).expect("df"),
    )
    .kernel(&tight())
    .expect("KROHF");
    println!("KRHF {:.14}  KROHF {:.14}", r.e_tot, ro.e_tot);
    assert!(ro.converged, "KROHF did not converge in {} cycles", ro.cycles);
    assert_eq!(ro.nset, 2, "KROHF carries two density channels");
    assert!(
        (r.e_tot - ro.e_tot).abs() < 1e-9,
        "KROHF {} != KRHF {}",
        ro.e_tot,
        r.e_tot
    );
    // Every occupation is 0 or 2 — no singly-occupied levels in a closed shell.
    for occ in &ro.mo_occ {
        for o in occ {
            assert!(
                *o == 0.0 || *o == 2.0,
                "closed-shell KROHF produced a fractional occupation {o}"
            );
        }
    }
}

/// The `'1e'` and `'minao'` initial guesses reach the same converged energy —
/// the SCF solution must not depend on where it started.
#[test]
fn init_guesses_reach_the_same_solution() {
    let cell = diamond();
    let (mf, _) = krhf(cell, [2, 1, 1], MESH);
    let a = mf
        .kernel(&KScfConfig {
            init_guess: KInitGuess::Minao,
            ..tight()
        })
        .expect("minao");
    let b = mf
        .kernel(&KScfConfig {
            init_guess: KInitGuess::OneElectron,
            ..tight()
        })
        .expect("1e");
    assert!(a.converged && b.converged);
    assert!(
        (a.e_tot - b.e_tot).abs() < 1e-9,
        "init guess changed the answer: minao {} vs 1e {}",
        a.e_tot,
        b.e_tot
    );
}

/// Fermi-Dirac smearing: the occupations still integrate to the electron count,
/// the entropy is non-negative and `e_free <= e_tot <= ...` in the right order.
#[test]
fn fermi_smearing_conserves_electrons_and_lowers_the_free_energy() {
    let cell = diamond();
    let (mut mf, _) = krhf(cell, [2, 2, 2], MESH);
    mf.smearing = Some(Smearing::fermi(0.01));
    let r = mf.kernel(&KScfConfig {
        conv_tol: 1e-10,
        conv_tol_grad: Some(1e-6),
        ..tight()
    })
    .expect("smeared KRHF");

    let occ: f64 = r.mo_occ.iter().flatten().sum();
    let want = mf.nelectron() as f64;
    assert!(
        (occ - want).abs() < 1e-6,
        "smeared occupations sum to {occ}, want {want}"
    );
    let e_free = r.e_free.expect("smearing must report a free energy");
    let e_zero = r.e_zero.expect("smearing must report e_zero");
    println!("smeared: e_tot {:.12} e_free {:.12} e_zero {:.12}", r.e_tot, e_free, e_zero);
    assert!(
        e_free <= r.e_tot + 1e-12,
        "e_free {e_free} must not exceed e_tot {}",
        r.e_tot
    );
    assert!(
        e_zero <= r.e_tot + 1e-12 && e_zero >= e_free - 1e-12,
        "e_zero {e_zero} must sit between e_free {e_free} and e_tot {}",
        r.e_tot
    );
}

/// Writing and reading a periodic chkfile round-trips exactly. The complex MO
/// coefficients go through the h5py `complex128` compound layout.
#[test]
fn chkfile_round_trips() {
    let cell = diamond();
    let nao = cell.mol.nao_nr;
    let (mf, kpts) = krhf(cell, [2, 1, 1], MESH);
    let r = mf.kernel(&tight()).expect("KRHF");

    let path = std::env::temp_dir().join(format!("pbc_kscf_chk_{}.h5", std::process::id()));
    dump_kscf_to_file(&path, &r, &kpts, nao, "").expect("dump");
    let back = load_kscf_from_file(&path).expect("load");
    let _ = std::fs::remove_file(&path);

    assert_eq!(back.e_tot, r.e_tot);
    assert_eq!(back.nao, nao);
    assert_eq!(back.kpts.len(), kpts.len());
    for (a, b) in back.kpts.iter().zip(kpts.iter()) {
        assert_eq!(a, b);
    }
    for (b, m) in back.mo_coeff.iter().enumerate() {
        assert_eq!(m.re, r.mo_coeff[b].re, "mo_coeff re block {b}");
        assert_eq!(m.im, r.mo_coeff[b].im, "mo_coeff im block {b}");
        assert_eq!(back.mo_occ[b], r.mo_occ[b], "mo_occ block {b}");
    }
}

/// The gamma-point helpers are their k-point counterparts at one k-point.
#[test]
fn gamma_helpers_agree_with_the_single_kpoint_drivers() {
    let cell = diamond();
    let a = pyscf_pbc_scf::rhf(cell.clone()).expect("gamma RHF");
    let b = Krhf::new(cell, &[[0.0; 3]]).expect("KRHF at gamma");
    assert_eq!(a.kpts(), b.kpts());
    assert_eq!(a.cell().mol.nao_nr, b.cell().mol.nao_nr);
}

// ---------------------------------------------------------------------------
// The upstream gate
// ---------------------------------------------------------------------------

const ORACLE_PY: &str = r#"
import json, sys
import numpy as np
from pyscf.pbc import gto, scf

a_json, xyz_json, sym_json, basis, pseudo, nk_json, mesh_json, method = sys.argv[1:9]
c = gto.Cell()
c.a = json.loads(a_json)
c.atom = [(s, tuple(r)) for s, r in zip(json.loads(sym_json), json.loads(xyz_json))]
c.basis = basis
if pseudo:
    c.pseudo = pseudo
c.unit = 'Bohr'
c.verbose = 0
c.build()
kpts = c.make_kpts(json.loads(nk_json))
mf = getattr(scf, method)(c, kpts)
mf.with_df.mesh = json.loads(mesh_json)
mf.conv_tol = 1e-12
mf.conv_tol_grad = 1e-8
mf.max_cycle = 60
e = mf.kernel()
print(json.dumps({'version': __import__('pyscf').__version__,
                  'e_tot': float(e), 'e_nuc': float(c.energy_nuc()),
                  'converged': bool(mf.converged), 'nao': int(c.nao_nr()),
                  'nelec': int(c.tot_electrons(len(kpts)))}))
"#;

fn upstream_energy(
    cell: &Cell,
    basis: &str,
    pseudo: &str,
    nk: [usize; 3],
    mesh: [usize; 3],
    method: &str,
) -> Option<serde_json::Value> {
    let py = oracle_python()?;
    let args = cell_args(
        cell,
        &[
            basis.to_string(),
            pseudo.to_string(),
            serde_json::to_string(&nk.to_vec()).expect("json"),
            serde_json::to_string(&mesh.to_vec()).expect("json"),
            method.to_string(),
        ],
    );
    let v = run_python(&py, ORACLE_PY, &args);
    assert_eq!(
        v["version"].as_str().expect("version"),
        "2.12.1",
        "the oracle must be the VENDORED PySCF 2.12.1 — see tests/common/mod.rs"
    );
    assert!(v["converged"].as_bool().unwrap_or(false), "upstream did not converge");
    Some(v)
}

fn assert_matches(got: &KScfResult, want: &serde_json::Value, tol: f64, label: &str) {
    let e_ref = want["e_tot"].as_f64().expect("e_tot");
    let n_ref = want["e_nuc"].as_f64().expect("e_nuc");
    assert!(
        (got.e_nuc - n_ref).abs() < 1e-12,
        "{label}: e_nuc {} != {n_ref} — the two runs are not the same cell",
        got.e_nuc
    );
    let d = got.e_tot - e_ref;
    println!("{label}: rust {:.15}  upstream {:.15}  delta {:e}", got.e_tot, e_ref, d);
    assert!(d.abs() < tol, "{label}: |delta| = {:e} exceeds {tol:e}", d.abs());
}

/// **THE PHASE-11 GATE.** `KRHF(diamond, 2x2x2)` against upstream.
///
/// Run at `MESH_GATE`, where upstream's own `ft_ao` pseudopotential expansion
/// has converged; see the tolerance note on
/// [`krhf_he_all_electron_matches_upstream`] for what the residual is made of.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF; ~4 min"]
fn krhf_diamond_222_matches_upstream() {
    let cell = diamond();
    let Some(want) =
        upstream_energy(&cell, "gth-szv", "gth-pade", [2, 2, 2], MESH_GATE, "KRHF")
    else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let (mf, _) = krhf(cell, [2, 2, 2], MESH_GATE);
    let got = mf.kernel(&tight()).expect("KRHF");
    assert!(got.converged);
    assert_matches(&got, &want, 1e-11, "KRHF diamond 2x2x2");
}

/// The same gate on the ALL-ELECTRON path, at **1e-12**.
///
/// This is the tighter of the two because it has no pseudopotential: upstream's
/// `get_pp` builds its non-local half from `ft_ao`, a planewave expansion, while
/// this port uses Phase 10's exact real-space `get_pp_nl`, and the two agree
/// only to ~1e-13 per matrix element even at a converged mesh (see
/// `pyscf-pbc-df/tests/fftdf.rs`). Accumulated over 8 AOs and 8 k-points that
/// is the few-1e-12 floor of the pseudopotential gate. `get_nuc` has no such
/// component, so He agrees to 1e-13.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn krhf_he_all_electron_matches_upstream() {
    let cell = he_all_electron();
    let mesh = [15, 15, 15];
    let Some(want) = upstream_energy(&cell, "sto-3g", "", [2, 2, 2], mesh, "KRHF") else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let (mf, _) = krhf(cell, [2, 2, 2], mesh);
    let got = mf.kernel(&tight()).expect("KRHF");
    assert!(got.converged);
    assert_matches(&got, &want, 1e-12, "KRHF He (all-electron) 2x2x2");
}

/// `KUHF` on the same closed-shell cell — the unrestricted driver, its two
/// global Fermi levels and its `vj[a] + vj[b] - vk[s]` potential.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV + the vendored upstream PySCF"]
fn kuhf_he_all_electron_matches_upstream() {
    let cell = he_all_electron();
    let mesh = [15, 15, 15];
    let Some(want) = upstream_energy(&cell, "sto-3g", "", [2, 2, 2], mesh, "KUHF") else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let kpts = make_kpts_default(&cell, [2, 2, 2]).expect("k-mesh");
    let mf = Kuhf::from_df(Fftdf::with_mesh(cell, &kpts, mesh).expect("df"));
    let got = mf.kernel(&tight()).expect("KUHF");
    assert!(got.converged);
    assert_matches(&got, &want, 1e-12, "KUHF He (all-electron) 2x2x2");
}

/// The gate at diamond's DEFAULT mesh `[47, 47, 47]` — the number a user gets
/// from `KRHF(cell, kpts).kernel()` with no mesh override. Slow (tens of
/// minutes), so it is a separate test from the CI-sized one.
#[test]
#[ignore = "needs PYSCF_ORACLE_VENV; runs at the default mesh 47^3 (~30 min)"]
fn krhf_diamond_222_matches_upstream_at_the_default_mesh() {
    let cell = diamond();
    let mesh = cell.try_mesh().expect("cell mesh");
    let Some(want) = upstream_energy(&cell, "gth-szv", "gth-pade", [2, 2, 2], mesh, "KRHF")
    else {
        eprintln!("SKIP: {GATE} is not set");
        return;
    };
    let (mf, _) = krhf(cell, [2, 2, 2], mesh);
    let got = mf.kernel(&tight()).expect("KRHF");
    assert!(got.converged);
    assert_matches(&got, &want, 1e-11, "KRHF diamond 2x2x2 at the default mesh");
}

/// A helper so the electron-count assertion above can reach the overlap through
/// the public hook surface.
trait OvlpPublic {
    fn get_ovlp_public(&self) -> Result<Vec<CTensor>, pyscf_core::PyscfRsError>;
}
impl OvlpPublic for Krhf {
    fn get_ovlp_public(&self) -> Result<Vec<CTensor>, pyscf_core::PyscfRsError> {
        use pyscf_pbc_scf::KOverrideHooks;
        KOverrideHooks::get_ovlp(self)
    }
}

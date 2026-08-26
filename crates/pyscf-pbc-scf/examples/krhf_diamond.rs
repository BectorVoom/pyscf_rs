//! `KRHF` on diamond — the Phase-11 end-to-end example.
//!
//! ```bash
//! cargo run --release -p pyscf-pbc-scf --example krhf_diamond -- <mesh> <nk>
//! # e.g. the CI-sized run:      -- 15 2
//! # the default-mesh run:       -- 47 2   (minutes)
//! ```
//!
//! Prints the total energy, its components, and the lowest band energies.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_df::Fftdf;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};
use pyscf_pbc_scf::{KScfConfig, Krhf};
use std::time::Instant;

fn diamond() -> Cell {
    let h = 3.37032; let q = 1.68516;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("C".into(), [0.0,0.0,0.0]), ("C".into(), [q,q,q])]),
            basis: BasisInput::Name("gth-szv".into()), unit: Unit::Bohr, ..Default::default() },
        a: ALattice::Matrix([[0.0,h,h],[h,0.0,h],[h,h,0.0]]),
        pseudo: Some("gth-pade".into()), ..Default::default()
    }).unwrap()
}

fn main() {
    let m: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(11);
    let nk: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(2);
    let cell = diamond();
    let kpts = make_kpts_default(&cell, [nk,nk,nk]).unwrap();
    let df = Fftdf::with_mesh(cell.clone(), &kpts, [m,m,m]).unwrap();
    let mf = Krhf::from_df(df);
    let cfg = KScfConfig { conv_tol: 1e-12, conv_tol_grad: Some(1e-8), max_cycle: 60, verbose: false, ..KScfConfig::default() };
    let t = Instant::now();
    let r = mf.kernel(&cfg).unwrap();
    println!("mesh {m} nk {nk}: e_tot {:.15} e_nuc {:.15} e_elec {:.15} e_coul {:.15} conv {} cycles {} time {:?}",
             r.e_tot, r.e_nuc, r.e_elec, r.e_coul, r.converged, r.cycles, t.elapsed());
    let mut all: Vec<f64> = r.mo_energy.iter().flatten().copied().collect();
    all.sort_by(|a,b| a.partial_cmp(b).unwrap());
    println!("lowest mo energies: {:?}", &all[..8.min(all.len())]);
}

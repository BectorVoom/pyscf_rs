//! `Kuhf::get_bands` — energy bands at arbitrary k-points for unrestricted
//! periodic HF.
//!
//! Both assertions here are exact physical invariants, so they need no upstream
//! oracle and cannot drift with a reference number:
//!
//! 1. **Self-consistency.** Bands evaluated ON the SCF mesh, from the converged
//!    density, must reproduce the SCF eigenvalues. The band Fock at a mesh
//!    k-point IS the SCF Fock at that k-point, so any error in the `vj[a] +
//!    vj[b] - vk[s]` assembly, in the spin-channel indexing, or in the
//!    `kpts_band` plumbing shows up immediately.
//! 2. **Time reversal.** For a collinear spin-polarised system with no magnetic
//!    field, `E_s(k) == E_s(-k)`. This one exercises the OFF-mesh path, which
//!    invariant (1) cannot reach.
//!
//! The fixture is a spin-1 Li atom in a 6-Bohr box (`sto-3g`, all-electron),
//! borrowed from `pyscf-pbc-dft`'s open-shell gate: 3 electrons over 2 unequal
//! spin channels, so alpha and beta are genuinely different numbers and a bug
//! that computed one channel twice would fail assertion (1).
//!
//! The mesh is 3x1x1, not 2x1x1, and that is forced: `Kuhf::nelec` counts
//! electrons over the BZ SUPERCELL (`3 * nkpts`) while `cell.spin` stays
//! per-cell, so an even k-point count pairs an even electron total with an odd
//! spin and upstream's own consistency check rejects it.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};
use pyscf_pbc_scf::Kuhf;
use pyscf_pbc_scf::types::KScfConfig;

/// These are exact-invariant tests, so the SCF must be converged far tighter
/// than the default. `KScfConfig::for_cell` stops at `conv_tol = 1e-7`, whose
/// implied gradient threshold is `sqrt(1e-7) ~ 3e-4` — the density is still
/// that far from the fixed point, and both invariants below are only exact AT
/// the fixed point.
fn tight(cell: &Cell) -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-10,
        conv_tol_grad: Some(1e-7),
        max_cycle: 200,
        ..KScfConfig::for_cell(cell)
    }
}

fn li_atom_spin1() -> Cell {
    let a = 6.0;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("Li".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            spin: 1,
            ..Default::default()
        },
        a: ALattice::Matrix([[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]),
        ..Default::default()
    })
    .expect("the Li fixture must build")
}

/// Invariant (1): on the mesh, `get_bands` reproduces the SCF eigenvalues.
#[test]
fn bands_on_the_scf_mesh_reproduce_the_scf_eigenvalues() {
    let cell = li_atom_spin1();
    let kpts = make_kpts_default(&cell, [3, 1, 1]).expect("3x1x1 mesh");
    let cfg = tight(&cell);
    let mf = Kuhf::new(cell, &kpts).expect("Kuhf must build");
    let scf = mf.kernel(&cfg).expect("KUHF must converge");
    assert!(
        scf.converged,
        "the fixture must converge for this to mean anything"
    );

    let (e_band, c_band) = mf
        .get_bands(&kpts, &scf.dm)
        .expect("get_bands on the SCF mesh must succeed");

    // Layout: alpha for every band k-point, then beta.
    assert_eq!(
        e_band.len(),
        2 * kpts.len(),
        "both spin channels must be returned, alpha block first"
    );
    assert_eq!(c_band.len(), 2 * kpts.len());
    assert_eq!(
        e_band.len(),
        scf.mo_energy.len(),
        "the band layout must match KScfResult's (set, k) ordering"
    );

    // Measure first, THEN assert, so a failure reports the magnitude rather
    // than just the first offending orbital.
    let mut worst = 0.0_f64;
    for (block, (got, want)) in e_band.iter().zip(scf.mo_energy.iter()).enumerate() {
        assert_eq!(
            got.len(),
            want.len(),
            "block {block}: orbital count differs from the SCF result"
        );
        for (g, w) in got.iter().zip(want.iter()) {
            worst = worst.max((g - w).abs());
        }
    }
    eprintln!("KUHF worst |E_band - E_scf| on the mesh = {worst:.3e}");
    assert!(
        worst < 1e-8,
        "bands on the SCF mesh must reproduce the SCF eigenvalues; worst |d| = {worst:.3e}"
    );

    // The fixture must actually be spin-polarised, or assertion (1) would pass
    // for an implementation that computed the alpha channel twice.
    let nk = kpts.len();
    let alpha_beta_gap: f64 = e_band[..nk]
        .iter()
        .zip(&e_band[nk..])
        .flat_map(|(a, b)| a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()))
        .fold(0.0_f64, f64::max);
    assert!(
        alpha_beta_gap > 1e-3,
        "spin-1 Li must give genuinely different alpha and beta bands; \
         largest difference was only {alpha_beta_gap:.3e}"
    );
}

/// Invariant (2): time reversal, on k-points that are NOT on the SCF mesh.
#[test]
fn off_mesh_bands_respect_time_reversal() {
    let cell = li_atom_spin1();
    let kpts = make_kpts_default(&cell, [3, 1, 1]).expect("3x1x1 mesh");
    let cfg = tight(&cell);
    let mf = Kuhf::new(cell.clone(), &kpts).expect("Kuhf must build");
    let scf = mf.kernel(&cfg).expect("KUHF must converge");

    // A generic point along the band path, deliberately off the 3x1x1 mesh,
    // paired with its time-reversal partner.
    let k = cell
        .get_abs_kpts(&[[0.3, 0.15, 0.0]])
        .expect("scaled -> absolute k");
    let minus_k = cell
        .get_abs_kpts(&[[-0.3, -0.15, 0.0]])
        .expect("scaled -> absolute k");

    let (e_plus, _) = mf.get_bands(&k, &scf.dm).expect("bands at +k");
    let (e_minus, _) = mf.get_bands(&minus_k, &scf.dm).expect("bands at -k");

    assert_eq!(e_plus.len(), 2, "one band k-point, two spin channels");
    let mut worst = 0.0_f64;
    for (p, m) in e_plus.iter().zip(e_minus.iter()) {
        for (a, b) in p.iter().zip(m.iter()) {
            worst = worst.max((a - b).abs());
        }
    }
    eprintln!("KUHF worst |E(k) - E(-k)| off mesh = {worst:.3e}");
    assert!(
        worst < 1e-8,
        "time reversal requires E(k) == E(-k); worst |d| = {worst:.3e}"
    );

    // An off-mesh point must still produce finite, ascending eigenvalues.
    for (s, block) in e_plus.iter().enumerate() {
        assert!(
            block.iter().all(|e| e.is_finite()),
            "spin {s}: non-finite band energy"
        );
        assert!(
            block.windows(2).all(|w| w[0] <= w[1] + 1e-12),
            "spin {s}: eigenvalues must come back ascending"
        );
    }
}

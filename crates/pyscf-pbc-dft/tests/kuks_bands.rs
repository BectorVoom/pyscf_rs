//! `Kuks::get_bands` — energy bands at arbitrary k-points for unrestricted
//! periodic KS-DFT.
//!
//! The same two oracle-free invariants `pyscf-pbc-scf/tests/kuhf_bands.rs`
//! uses, and for the same reasons:
//!
//! 1. **Self-consistency.** On the SCF mesh, from the converged density, the
//!    bands must reproduce the SCF eigenvalues.
//! 2. **Time reversal.** `E_s(k) == E_s(-k)` off the mesh.
//!
//! Invariant (1) is worth more here than in the HF case. `get_bands` reaches
//! `get_veff_tagged(dms, Some(kpts_band))`, and passing `Some(..)` does two
//! things at once: it moves the XC quadrature onto the band k-points, and it
//! flips `ground_state` off so the energy tags are skipped. Evaluating at
//! `kpts_band == kpts` must nonetheless land on exactly the SCF potential, so
//! this pins the `kpts_band` plumbing through `nr_uks` and the J/K build — the
//! part of the DFT path that has no counterpart in `Kuhf`.
//!
//! Mesh is 3x1x1 for the reason documented in the KUHF file: `nelec` counts
//! over the BZ supercell while `cell.spin` is per-cell, so a spin-1 fixture
//! needs an odd k-point count.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_dft::kuks::Kuks;
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};

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
    let mf = Kuks::new(cell, &kpts, "lda,vwn").expect("Kuks must build");
    let scf = mf.run().expect("KUKS must converge");
    assert!(
        scf.converged,
        "the fixture must converge for this to mean anything"
    );

    let (e_band, c_band) = mf
        .get_bands(&kpts, &scf.dm)
        .expect("get_bands on the SCF mesh must succeed");

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

    let mut worst = 0.0_f64;
    for (block, (got, want)) in e_band.iter().zip(scf.mo_energy.iter()).enumerate() {
        assert_eq!(got.len(), want.len(), "block {block}: orbital count");
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let d = (g - w).abs();
            assert!(
                d < 1e-7,
                "block {block}, orbital {i}: band energy {g} vs SCF {w} (|d| = {d:.3e})"
            );
            worst = worst.max(d);
        }
    }
    eprintln!("KUKS worst |E_band - E_scf| on the mesh = {worst:.3e}");

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
    let mf = Kuks::new(cell.clone(), &kpts, "lda,vwn").expect("Kuks must build");
    let scf = mf.run().expect("KUKS must converge");

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
    for (s, (p, m)) in e_plus.iter().zip(e_minus.iter()).enumerate() {
        for (i, (a, b)) in p.iter().zip(m.iter()).enumerate() {
            let d = (a - b).abs();
            assert!(
                d < 1e-8,
                "spin {s}, band {i}: E(k) = {a} but E(-k) = {b} (|d| = {d:.3e}); \
                 time reversal is broken"
            );
            worst = worst.max(d);
        }
    }
    eprintln!("KUKS worst |E(k) - E(-k)| off mesh = {worst:.3e}");

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

//! `Kuhf::get_bands` — energy bands at arbitrary k-points for unrestricted
//! periodic HF.
//!
//! Three oracle-free invariants, all exact:
//!
//! 1. **Self-consistency.** Bands evaluated ON the SCF mesh, from the converged
//!    density, reproduce the SCF eigenvalues. Any error in the `vj[a] + vj[b] -
//!    vk[s]` assembly, in the spin-channel indexing, or in the `kpts_band`
//!    plumbing shows up here.
//! 2. **Time reversal.** `E_s(k) == E_s(-k)` off the mesh — but only when the
//!    density feeding the band Fock is itself time-reversal symmetric. See
//!    below; this is not a technicality, it is the whole reason test 2 runs at
//!    Gamma and test 3 exists.
//! 3. **Faithfulness to a symmetry-broken density.** `get_bands` must report
//!    what the density actually says, symmetric or not.
//!
//! # Measured: the open-shell multi-k fixture breaks time reversal on purpose
//!
//! Li, `spin = 1`, `sto-3g`, 6-Bohr cube, 3x1x1 mesh. `nelec = (5, 4)`, and the
//! alpha channel spreads 5 electrons over 3 k-points (numbers below measured at
//! `conv_tol = 1e-10`; test 3 re-derives the RULE at whatever `tight` uses, so
//! it does not depend on these exact values):
//!
//! | channel | occ at k = +1/3 | occ at k = -1/3 | 2s eigenvalues |
//! |---|---|---|---|
//! | alpha | 2 electrons | 1 electron | -0.1377 vs -0.0536 Ha |
//! | beta  | 1 electron  | 1 electron | agree to 1e-14 |
//!
//! The odd alpha electron has to land on one of the two time-reversal-degenerate
//! `k = +-1/3` states, and the SCF then self-consistently deepens whichever it
//! chose. That is a real symmetry-broken solution, not a convergence failure
//! (it converged in 7 cycles), and for such a density `E(k) != E(-k)` is the
//! CORRECT answer. The beta channel, whose occupation IS symmetric across the
//! pair, obeys time reversal to 1e-14 in the same calculation — which is what
//! makes the diagnosis airtight rather than a guess. The restricted control in
//! `krhf_bands.rs` satisfies both invariants to 0.0 on the same lattice and
//! mesh, confirming the shared band machinery is not at fault.
//!
//! Test 3 encodes the general rule instead of that specific table: *per spin
//! channel, equal occupations at time-reversal-partner k-points imply equal band
//! energies at `+-k`*. That statement is always true, so it does not rot if the
//! occupation policy ever changes.
//!
//! # Convergence
//!
//! These are exact-invariant tests, so they converge far tighter than the
//! default. `KScfConfig::for_cell` stops at `conv_tol = 1e-7`, whose implied
//! gradient threshold is `sqrt(1e-7) ~ 3e-4`; at that setting invariant (1)
//! misses by 1.8e-6 purely from an unconverged density, and at `1e-10` by
//! 6.7e-9. The SCF reports eigenvalues of the DIIS-extrapolated Fock built from
//! the PREVIOUS density rather than of `F[dm_final]`, so the residual tracks how
//! far the density got, not the integral accuracy — `cell.precision = 1e-8` is
//! NOT a floor here (verified on the KUKS side, where `conv_tol = 1e-12` reaches
//! 8.4e-11).
//!
//! The mesh is 3x1x1, not 2x1x1, because `Kuhf::nelec` counts electrons over
//! the BZ SUPERCELL (`3 * nkpts`) while `cell.spin` stays per-cell, so an even
//! k-point count pairs an even electron total with an odd spin and upstream's
//! own consistency check rejects it.

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, make_kpts_default};
use pyscf_pbc_scf::Kuhf;
use pyscf_pbc_scf::types::KScfConfig;

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

/// Identical to the restricted control's `tight` in `krhf_bands.rs`.
fn tight(cell: &Cell) -> KScfConfig {
    KScfConfig {
        conv_tol: 1e-12,
        conv_tol_grad: Some(1e-8),
        max_cycle: 300,
        ..KScfConfig::for_cell(cell)
    }
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
        worst < 1e-9,
        "bands on the SCF mesh must reproduce the SCF eigenvalues; worst |d| = {worst:.3e}"
    );

    // The fixture must actually be spin-polarised, or invariant (1) would pass
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

/// Invariant (2): time reversal off the mesh, on a density that HAS the
/// symmetry.
///
/// The SCF runs at Gamma only. There `nelec = (2, 1)` fills the non-degenerate
/// 1s and 2s levels outright, so the converged density is time-reversal
/// symmetric and `E_s(k) == E_s(-k)` genuinely holds. The band k-points are
/// still off the (single-point) mesh, so this exercises the same arbitrary-k
/// path a real band structure uses.
#[test]
fn off_mesh_bands_respect_time_reversal() {
    let cell = li_atom_spin1();
    let kpts = vec![[0.0, 0.0, 0.0]];
    let cfg = tight(&cell);
    let mf = Kuhf::new(cell.clone(), &kpts).expect("Kuhf must build");
    let scf = mf.kernel(&cfg).expect("KUHF must converge");
    assert!(scf.converged, "the Gamma-point fixture must converge");

    // A generic point, paired with its time-reversal partner.
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
        worst < 1e-10,
        "time reversal requires E(k) == E(-k) for a symmetric density; \
         worst |d| = {worst:.3e}"
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

/// Invariant (3): `get_bands` reports the density it is given.
///
/// On the 3x1x1 open-shell mesh the alpha channel converges to a
/// time-reversal-BROKEN occupation while beta stays symmetric. The rule tested
/// here holds either way: within one spin channel, equal occupation at
/// time-reversal-partner k-points implies equal band energies at those points,
/// and unequal occupation is free to give unequal energies.
#[test]
fn band_symmetry_follows_the_occupation_symmetry() {
    let cell = li_atom_spin1();
    let kpts = make_kpts_default(&cell, [3, 1, 1]).expect("3x1x1 mesh");
    let cfg = tight(&cell);
    let mf = Kuhf::new(cell.clone(), &kpts).expect("Kuhf must build");
    let scf = mf.kernel(&cfg).expect("KUHF must converge");
    assert!(scf.converged);

    // The mesh is {0, 1/3, 2/3}; indices 1 and 2 are the time-reversal pair,
    // since 2/3 == -1/3 modulo a reciprocal lattice vector.
    let scaled = cell.get_scaled_kpts(&kpts);
    assert!(
        (scaled[1][0] + scaled[2][0] - 1.0).abs() < 1e-12,
        "k[1] and k[2] must be a time-reversal pair; got {:?} and {:?}",
        scaled[1],
        scaled[2]
    );

    let nk = kpts.len();
    let mut saw_symmetric = false;
    let mut saw_broken = false;
    for (s, name) in ["alpha", "beta"].iter().enumerate() {
        let occ_1: f64 = scf.mo_occ[s * nk + 1].iter().sum();
        let occ_2: f64 = scf.mo_occ[s * nk + 2].iter().sum();
        let spread: f64 = scf.mo_energy[s * nk + 1]
            .iter()
            .zip(&scf.mo_energy[s * nk + 2])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!("{name}: occ(k1) = {occ_1}, occ(k2) = {occ_2}, |dE| = {spread:.3e}");

        if (occ_1 - occ_2).abs() < 1e-9 {
            saw_symmetric = true;
            assert!(
                spread < 1e-10,
                "{name}: occupations at the time-reversal pair are equal \
                 ({occ_1} vs {occ_2}), so the band energies there must match; \
                 largest difference was {spread:.3e}"
            );
        } else {
            saw_broken = true;
        }
    }

    assert!(
        saw_symmetric,
        "one channel was expected to keep the symmetry and act as the control; \
         if neither did, the fixture no longer discriminates"
    );
    assert!(
        saw_broken,
        "this fixture is supposed to break time reversal in one channel — an \
         odd electron on a degenerate +-1/3 pair. If both channels are now \
         symmetric the occupation policy changed (degeneracy-aware filling or \
         smearing); that may well be an improvement, but re-examine this test \
         and the module docs rather than deleting the assertion"
    );
}

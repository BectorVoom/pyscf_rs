//! U-02 — the UNRESTRICTED initial guess: `_break_dm_spin_symm` and the
//! per-channel renormalisation.
//!
//! # What was wrong, and why nothing could see it
//!
//! Before U-02 `init_guess::get_init_guess` returned, for `nset = 2`,
//! `vec![half.clone(), half]` — not "approximately equal" channels, the SAME
//! matrix cloned — and there was no symmetry-breaking code anywhere in the
//! workspace (`grep -rn "break_dm_spin_symm\|breaksym"` returned zero matches).
//!
//! `dm_a == dm_b` is an EXACT FIXED POINT of this port's SCF map at
//! `cell.spin == 0`: identical channels give a bitwise-identical `veff`, hence
//! identical `(e, c)` from the same deterministic `eig_channel`, hence — since
//! `get_occ_unrestricted` derives `fermi_a` and `fermi_b` from equal energy
//! lists with equal counts — identical occupations and identical densities
//! again. DIIS, damping and level shift are linear and preserve it, and there
//! is no stability analysis. So the port was structurally incapable of reaching
//! a spin-broken solution, while upstream reaches one BY DEFAULT
//! (`init_guess_breaksym = 1`, `uhf.py:778` / `kuhf.py:417`).
//!
//! The renormalisation was the second half: `electron_count` summed both
//! channels into one `f64` and applied a single factor `Ne / ne_total` to both.
//! On a `cell.spin != 0` cell that cannot restore `(nalpha, nbeta)` — and since
//! `_break_dm_spin_symm` short-circuits at `spin == 0`, the per-channel
//! renormalisation is the ONLY thing that polarises an open-shell minao guess.
//!
//! Every assertion below is on a cell where `dm_a != dm_b` must hold (RULE U).

use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs};
use pyscf_pbc_scf::init_guess::get_init_guess;
use pyscf_pbc_scf::krdm::electron_count_per_set;
use pyscf_pbc_scf::krhf::to_row_major;
use pyscf_pbc_scf::types::KInitGuess;

fn cube(a: f64) -> [[f64; 3]; 3] {
    [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
}

fn build(a: f64, atoms: Vec<(String, [f64; 3])>, basis: &str, spin: i32) -> Cell {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: BasisInput::Name(basis.into()),
            unit: Unit::Bohr,
            spin,
            ..Default::default()
        },
        a: ALattice::Matrix(cube(a)),
        ..Default::default()
    })
    .expect("cell must build")
}

/// Li in a 6-Bohr box, all-electron `sto-3g`, `spin = 1` — the polarised case.
/// Mirrors `pyscf-pbc-dft/tests/common::li_atom_spin1`.
fn li_atom_spin1() -> Cell {
    build(6.0, vec![("Li".into(), [0.0, 0.0, 0.0])], "sto-3g", 1)
}

/// Stretched H2 (3.0 Bohr) in an 8-Bohr box, all-electron `6-31g`, `spin = 0` —
/// the symmetry-BREAKING case. Mirrors `common::h2_stretched_spin0`.
fn h2_stretched_spin0() -> Cell {
    build(
        8.0,
        vec![
            ("H".into(), [0.0, 0.0, -1.5]),
            ("H".into(), [0.0, 0.0, 1.5]),
        ],
        "6-31g",
        0,
    )
}

fn overlap(cell: &Cell, kpts: &[[f64; 3]]) -> Vec<pyscf_algebra::CTensor> {
    to_row_major(
        pyscf_pbc_gto::get_ovlp(cell, kpts).expect("get_ovlp"),
        cell.mol.nao_nr,
    )
}

fn gamma() -> Vec<[f64; 3]> {
    vec![[0.0, 0.0, 0.0]]
}

/// The maximum element-wise gap between the two channels of a k-stack.
fn channel_gap(dms: &[Vec<pyscf_algebra::CTensor>]) -> f64 {
    let mut w = 0.0_f64;
    for (a, b) in dms[0].iter().zip(&dms[1]) {
        for i in 0..a.len() {
            w = w.max((a.re[i] - b.re[i]).abs());
            w = w.max((a.im[i] - b.im[i]).abs());
        }
    }
    w
}

// ---------------------------------------------------------------------------
// _break_dm_spin_symm
// ---------------------------------------------------------------------------

/// **The headline.** With `breaksym = 1` on a `spin == 0` cell the two channels
/// must DIFFER. With `breaksym = 0` they must be bit-identical — the old
/// behaviour, kept reachable so a caller can ask for the restricted stationary
/// point on purpose.
#[test]
fn breaksym_makes_the_two_channels_differ_at_spin_zero() {
    let cell = h2_stretched_spin0();
    let kpts = gamma();
    let s1e = overlap(&cell, &kpts);
    let ne = [1.0, 1.0];

    let off = get_init_guess(&cell, 1, 2, &KInitGuess::Minao, &s1e, &ne, 0).expect("guess");
    assert_eq!(
        channel_gap(&off),
        0.0,
        "breaksym = 0 must leave dm_a == dm_b bit-identically"
    );

    let on = get_init_guess(&cell, 1, 2, &KInitGuess::Minao, &s1e, &ne, 1).expect("guess");
    assert!(
        channel_gap(&on) > 1e-6,
        "breaksym = 1 left the channels within {} of each other — the SCF map has \
         dm_a == dm_b as an exact fixed point, so no spin-broken solution is reachable",
        channel_gap(&on)
    );
}

/// `breaksym == 1` keeps only the INTRA-ATOMIC blocks of the beta guess
/// (`uhf.py:121-125`). H2/6-31g has two AOs per atom, so beta's inter-atomic
/// blocks must be EXACTLY zero and its two 2x2 diagonal blocks must be
/// PROPORTIONAL to alpha's.
///
/// Proportional, not equal: the break runs first and the per-channel
/// renormalisation (`kuhf.py:476-486`) runs after it, and zeroing the
/// inter-atomic blocks removed electrons from beta — so beta is then scaled
/// back to `nbeta` by a factor alpha does not get. That ordering is upstream's
/// (`init_guess_by_minao` breaks; `get_init_guess` renormalises), and the
/// exact-zero structure survives any scaling, which is what makes it the
/// assertion worth making.
#[test]
fn breaksym_one_zeroes_the_inter_atomic_beta_blocks() {
    let cell = h2_stretched_spin0();
    let nao = cell.mol.nao_nr;
    assert_eq!(
        nao, 4,
        "6-31g on H2 must give 4 AOs; the block layout below assumes it"
    );
    let kpts = gamma();
    let s1e = overlap(&cell, &kpts);
    let dms = get_init_guess(&cell, 1, 2, &KInitGuess::Minao, &s1e, &[1.0, 1.0], 1).expect("guess");
    let (a, b) = (&dms[0][0], &dms[1][0]);

    let mut ratio: Option<f64> = None;
    for i in 0..nao {
        for j in 0..nao {
            let same_atom = (i < 2) == (j < 2);
            let idx = i * nao + j;
            if same_atom {
                // Alpha must be non-trivial here or the comparison is vacuous.
                assert!(a.re[idx].abs() > 1e-9, "alpha block ({i},{j}) is ~zero");
                let r = b.re[idx] / a.re[idx];
                match ratio {
                    None => ratio = Some(r),
                    Some(r0) => assert!(
                        (r - r0).abs() < 1e-12,
                        "intra-atomic block ({i},{j}) is not a uniform multiple of alpha's:                          ratio {r} against {r0} — the break must copy the block verbatim and                          only the per-channel renormalisation may scale it"
                    ),
                }
            } else {
                assert_eq!(
                    b.re[idx], 0.0,
                    "inter-atomic block ({i},{j}) must be zeroed, got {}",
                    b.re[idx]
                );
                assert!(
                    a.re[idx].abs() > 1e-9,
                    "alpha's inter-atomic block ({i},{j}) is {} — if alpha is also \
                     ~zero here the test cannot distinguish a break from a no-op",
                    a.re[idx]
                );
            }
        }
    }
    let r = ratio.expect("at least one intra-atomic element");
    println!("H2(3 Bohr) breaksym=1: beta/alpha intra-atomic ratio = {r:.15}");
    assert!(r > 0.0, "the beta channel changed sign");
}

/// `uhf.py:119` — the break is skipped when the channels ALREADY differ by more
/// than 1e-2, and skipped unconditionally at `cell.spin != 0` ("for a spin
/// polarized system, no need to manually break spin symmetry").
#[test]
fn breaksym_is_a_no_op_on_a_polarised_cell() {
    let cell = li_atom_spin1();
    assert_eq!(cell.mol.spin, 1);
    let kpts = gamma();
    let s1e = overlap(&cell, &kpts);
    // Ne = 3, spin = 1 -> (2, 1).
    let off = get_init_guess(&cell, 1, 2, &KInitGuess::Minao, &s1e, &[2.0, 1.0], 0).expect("g");
    let on = get_init_guess(&cell, 1, 2, &KInitGuess::Minao, &s1e, &[2.0, 1.0], 1).expect("g");
    for (a, b) in off[1].iter().zip(&on[1]) {
        for i in 0..a.len() {
            assert_eq!(
                a.re[i].to_bits(),
                b.re[i].to_bits(),
                "the break must not fire at cell.spin != 0 (uhf.py:118-119)"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// The per-channel renormalisation — kuhf.py:476-486
// ---------------------------------------------------------------------------

/// On a `spin != 0` cell the guess must come back carrying `(nalpha, nbeta)`
/// per channel, not `Ne/2` in each.
///
/// This is the assertion the OLD code could not pass by construction: it
/// summed both channels into one `f64` and applied one factor to both, so
/// `ne_a` stayed at `Ne/2 = 1.5` against a target of 2.
#[test]
fn the_guess_carries_the_per_channel_electron_counts() {
    let cell = li_atom_spin1();
    let nao = cell.mol.nao_nr;
    let kpts = gamma();
    let s1e = overlap(&cell, &kpts);
    let want = [2.0_f64, 1.0];

    let dms = get_init_guess(&cell, 1, 2, &KInitGuess::Minao, &s1e, &want, 1).expect("guess");
    let got = electron_count_per_set(&dms, &s1e, nao);
    println!(
        "Li(spin1) gamma minao guess: ne = ({:.15}, {:.15})  want ({}, {})",
        got[0], got[1], want[0], want[1]
    );
    for (g, w) in got.iter().zip(&want) {
        assert!(
            (g - w).abs() < 1e-12,
            "per-channel electron count {g} != {w} (|d| = {:e})",
            (g - w).abs()
        );
    }
    assert!(
        (got[0] - got[1]).abs() > 0.5,
        "the guess is unpolarised (ne_a = {}, ne_b = {}) — the renormalisation \
         is the only thing that polarises a minao guess at spin != 0",
        got[0],
        got[1]
    );
}

/// The same at an ODD k-count. `nelec` is a BZ-supercell quantity on both
/// sides (`kuhf.py:442-456`), so a 3-k run wants `(6, 3)`, and the k-mesh
/// parity rule (odd electrons need an odd k-count) is satisfied.
#[test]
fn the_per_channel_counts_hold_over_several_k_points() {
    let cell = li_atom_spin1();
    let nao = cell.mol.nao_nr;
    let kpts = pyscf_pbc_gto::make_kpts_default(&cell, [1, 1, 3]).expect("kpts");
    assert_eq!(kpts.len(), 3);
    let s1e = overlap(&cell, &kpts);
    let want = [6.0_f64, 3.0];
    let dms = get_init_guess(&cell, 3, 2, &KInitGuess::Minao, &s1e, &want, 1).expect("guess");
    let got = electron_count_per_set(&dms, &s1e, nao);
    for (g, w) in got.iter().zip(&want) {
        assert!((g - w).abs() < 1e-12, "per-channel count {g} != {w}");
    }
}

/// The restricted path must be untouched: `nset = 1` takes a one-element
/// `nelec` and the SAME threshold and scale it always had, so no `KRHF`/`KRKS`
/// number may move. (`ne_total - Ne = 2 (ne_a - nalpha)` is why the OLD
/// two-channel code fired at half upstream's threshold; the one-channel code
/// never had that problem.)
#[test]
fn the_restricted_guess_is_unchanged() {
    let cell = h2_stretched_spin0();
    let nao = cell.mol.nao_nr;
    let kpts = gamma();
    let s1e = overlap(&cell, &kpts);
    let dms = get_init_guess(&cell, 1, 1, &KInitGuess::Minao, &s1e, &[2.0], 0).expect("guess");
    assert_eq!(dms.len(), 1);
    let got = electron_count_per_set(&dms, &s1e, nao);
    assert!(
        (got[0] - 2.0).abs() < 1e-12,
        "restricted guess electron count {} != 2",
        got[0]
    );
}

/// A mis-sized `nelec` is a caller bug and must be an `Err`, never a silent
/// mis-scaling of one channel.
#[test]
fn a_mismatched_nelec_length_is_rejected() {
    let cell = h2_stretched_spin0();
    let kpts = gamma();
    let s1e = overlap(&cell, &kpts);
    assert!(get_init_guess(&cell, 1, 2, &KInitGuess::Minao, &s1e, &[2.0], 1).is_err());
}

// ---------------------------------------------------------------------------
// MEASUREMENT — KUKS-OPTIMISATION-PLAN §2.2.2 / §8 open question 3
// ---------------------------------------------------------------------------

/// **Answers a question the plan left UNVERIFIED**: does the init-guess
/// renormalisation branch fire at all for the standard minao guess on the
/// `gth-szv`/`gth-pade` reference cells?
///
/// It matters because those cells hold 4 VALENCE electrons per atom in the AO
/// basis while the minao guess is built from an ALL-ELECTRON minimal basis, so
/// the guess can arrive carrying the wrong count entirely. The plan proposed
/// answering it by reading a `tracing::debug!` line out of a gate run; counting
/// the electrons directly is the same answer without the gate.
///
/// Not a gate — it prints, and only asserts that the guess is not absurd.
#[test]
fn measurement_does_the_renormalisation_fire_on_the_reference_cells() {
    let cases: [(&str, Cell, f64); 2] = [
        (
            "silicon gth-szv/gth-pade",
            Cell::build(CellBuildArgs {
                mole: MoleBuildArgs {
                    atom: AtomInput::Tuples(vec![
                        ("Si".into(), [0.0, 0.0, 0.0]),
                        ("Si".into(), [2.55555, 2.55555, 2.55555]),
                    ]),
                    basis: BasisInput::Name("gth-szv".into()),
                    unit: Unit::Bohr,
                    ..Default::default()
                },
                a: ALattice::Matrix([
                    [0.0, 5.1311, 5.1311],
                    [5.1311, 0.0, 5.1311],
                    [5.1311, 5.1311, 0.0],
                ]),
                pseudo: Some("gth-pade".into()),
                ..Default::default()
            })
            .expect("Si"),
            8.0,
        ),
        (
            "diamond gth-szv/gth-pade",
            Cell::build(CellBuildArgs {
                mole: MoleBuildArgs {
                    atom: AtomInput::Tuples(vec![
                        ("C".into(), [0.0, 0.0, 0.0]),
                        ("C".into(), [1.68516, 1.68516, 1.68516]),
                    ]),
                    basis: BasisInput::Name("gth-szv".into()),
                    unit: Unit::Bohr,
                    ..Default::default()
                },
                a: ALattice::Matrix([
                    [0.0, 3.37032, 3.37032],
                    [3.37032, 0.0, 3.37032],
                    [3.37032, 3.37032, 0.0],
                ]),
                pseudo: Some("gth-pade".into()),
                ..Default::default()
            })
            .expect("C"),
            8.0,
        ),
    ];

    for (label, cell, ne_total) in cases {
        let nao = cell.mol.nao_nr;
        let kpts = pyscf_pbc_gto::make_kpts_default(&cell, [2, 2, 2]).expect("kpts");
        let nkpts = kpts.len();
        let s1e = overlap(&cell, &kpts);
        let want_r = ne_total * nkpts as f64;
        // The RAW guess, before any renormalisation: ask for the count it
        // already has, so the branch cannot fire.
        let raw = get_init_guess(&cell, nkpts, 1, &KInitGuess::Minao, &s1e, &[f64::NAN], 0);
        let raw_ne = match raw {
            Ok(d) => electron_count_per_set(&d, &s1e, nao)[0],
            Err(_) => f64::NAN,
        };
        let fires = (raw_ne - want_r).abs() > 0.01 * nkpts as f64;
        println!(
            "{label:<28} nkpts={nkpts}  minao guess Ne = {raw_ne:.12} (per cell \
             {:.12}), want {want_r} ({:.1} per cell)  =>  renormalisation \
             {}",
            raw_ne / nkpts as f64,
            ne_total,
            if fires { "FIRES" } else { "does NOT fire" }
        );
        assert!(raw_ne.is_finite() && raw_ne > 0.0);
    }
}

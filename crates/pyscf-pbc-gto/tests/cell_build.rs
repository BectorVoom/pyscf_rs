//! Plan 09-03 Task 6 — the `Cell` type (PBC-GTO-01 / PBC-GTO-02).
//!
//! Tier 1 (invariants, no upstream needed):
//!   * `reciprocal_vectors(2*pi) . a.T == 2*pi*I` for all five systems;
//!   * `get_abs_kpts(get_scaled_kpts(k)) == k` for pseudo-random k;
//!   * `get_scaled_atom_coords` inverts against the lattice;
//!   * `Deref` to `Mole` works through `Cell`;
//!   * `dumps`/`loads` round-trips the periodic state.
//!
//! Tier 2 (hard-coded upstream references, D-PBC-19): `vol`, `natm`, `nao_nr`
//! and the Bohr lattice of all five systems against values generated once from
//! live PySCF 2.12.1 and committed in `pyscf_pbc_gto::test_systems::REFERENCES`.
//!
//! # Why the tier-2 lattice tolerance is RELATIVE, not the plan's absolute 1e-6
//!
//! `pyscf_core::Unit::Ang.length_in_au()` is `1.8897261339213`, while upstream
//! PySCF converts Angstrom to Bohr by DIVIDING by `pyscf/data/nist.py:BOHR =
//! 0.52917721092`, i.e. by an effective factor of `1.8897261245650618`. The two
//! differ by 4.95e-9 relative, so every lattice this port builds is 4.95e-9
//! long compared with upstream and every volume is 1.485e-8 large — 1.1e-6
//! Bohr^3 on diamond, 1.1e-5 on graphene. An absolute 1e-6 check is therefore
//! unreachable for any cell bigger than He/fcc, no matter how correct the port
//! is. The checks below use a RELATIVE 1e-7 bound, and
//! `bohr_constant_gap_vs_upstream_is_the_whole_lattice_error` pins the
//! discrepancy to that one constant so nothing else can hide inside it.
//! Correcting `pyscf_core`'s constant is a separate, workspace-wide change
//! (it moves every molecular geometry and regression baseline) — recorded as a
//! carry-over in 09-03-SUMMARY.md.
//!
//! NOT verified here: the NUMERIC agreement of `rcut` / `mesh` with upstream —
//! plan 09-04 owns the estimators and `tests/cutoff.rs` is their gate. This
//! file only asserts that a built cell no longer carries the 09-03 sentinels.
//! Pseudopotential-adjusted electron counts landed in plan 10-01 (see
//! D-PBC-11).

mod common;

use common::systems;
use pyscf_core::Unit;
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::{
    ALattice, Cell, CellBuildArgs, LowDimFtType, det3, dumps, inv3, loads, transpose3,
};
use std::f64::consts::PI;

/// Deterministic LCG (Knuth/MMIX constants) — the same generator the algebra
/// tests use, so "random" inputs are reproducible without pulling in `rand`.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_f64(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let u = (self.0 >> 11) as f64 / (1u64 << 53) as f64;
        u * 2.0 - 1.0
    }
}

// ---------------------------------------------------------------------------
// Tier 2 — hard-coded upstream references (D-PBC-19).
// ---------------------------------------------------------------------------

/// Relative bound on `vol` vs upstream. The known Bohr-constant gap contributes
/// 1.485e-8; 1e-7 leaves an order of magnitude of headroom while still catching
/// any real geometry bug, which would be orders of magnitude larger.
const VOL_REL_TOL: f64 = 1e-7;

/// Relative bound on a lattice component vs upstream. The Bohr-constant gap
/// contributes 4.95e-9.
const LEN_REL_TOL: f64 = 1e-8;

/// `pyscf_core::Unit::Ang.length_in_au()`.
const OUR_ANG_TO_BOHR: f64 = 1.8897261339213;

/// `pyscf/data/nist.py:BOHR` — upstream divides by this to convert Angstrom to
/// Bohr, so its reciprocal is upstream's effective factor.
const UPSTREAM_BOHR_IN_ANG: f64 = 0.52917721092;

#[test]
fn all_reference_systems_match_upstream_vol_natm_nao() {
    let built = systems::all();
    assert_eq!(built.len(), systems::REFERENCES.len());

    for (r, (name, cell)) in systems::REFERENCES.iter().zip(built.iter()) {
        assert_eq!(r.name, *name, "REFERENCES and all() are out of order");

        // vol — PBC-MASTER-PLAN plan 09-03 asks for 1e-6; see the module docs
        // for why that has to be relative rather than absolute today.
        let rel = (cell.vol() - r.vol).abs() / r.vol;
        assert!(
            rel < VOL_REL_TOL,
            "{name}: vol = {} but upstream PySCF 2.12.1 says {} (rel {rel:e}, tol {VOL_REL_TOL:e})",
            cell.vol(),
            r.vol
        );
        // The lattice itself, in Bohr — catches a unit-conversion or row/column
        // bug that a volume-only check could hide (vol scales as the cube, and
        // is invariant under transposing the lattice).
        for i in 0..3 {
            for j in 0..3 {
                let (got, want) = (cell.a[i][j], r.a_bohr[i][j]);
                let tol = LEN_REL_TOL * want.abs().max(1.0);
                assert!(
                    (got - want).abs() < tol,
                    "{name}: a[{i}][{j}] = {got} but upstream says {want} (tol {tol:e})"
                );
            }
        }
        // Through the Deref to Mole.
        assert_eq!(cell.natm, r.natm, "{name}: natm");
        assert_eq!(cell.nao_nr, r.nao_nr, "{name}: nao_nr");
        assert!(cell._built, "{name}: Cell::build must set _built");
    }
}

/// Prove that the ENTIRE deviation from upstream's lattice is the
/// Angstrom-to-Bohr constant and nothing else.
///
/// Every component of every reference lattice must be exactly
/// `upstream_value * (OUR_ANG_TO_BOHR * UPSTREAM_BOHR_IN_ANG)`. If a genuine
/// geometry bug ever creeps in, this ratio stops being uniform and this test
/// fails even though the loose relative bounds above might still pass.
///
/// **When `pyscf_core`'s constant is corrected to `1.0 / 0.52917721092`, this
/// test's ratio becomes 1 and `VOL_REL_TOL` / `LEN_REL_TOL` can drop to 1e-12.**
#[test]
fn bohr_constant_gap_vs_upstream_is_the_whole_lattice_error() {
    let expected_ratio = OUR_ANG_TO_BOHR * UPSTREAM_BOHR_IN_ANG;
    assert_eq!(
        OUR_ANG_TO_BOHR,
        Unit::Ang.length_in_au(),
        "pyscf_core's Angstrom conversion changed; regenerate the tolerances"
    );

    for (r, (name, cell)) in systems::REFERENCES.iter().zip(systems::all().iter()) {
        for i in 0..3 {
            for j in 0..3 {
                let want = r.a_bohr[i][j];
                if want == 0.0 {
                    assert_eq!(cell.a[i][j], 0.0, "{name}: a[{i}][{j}] should be exactly 0");
                    continue;
                }
                let ratio = cell.a[i][j] / want;
                assert!(
                    (ratio - expected_ratio).abs() < 1e-14,
                    "{name}: a[{i}][{j}] / upstream = {ratio:.17} but the pure \
                     Bohr-constant ratio is {expected_ratio:.17} — this is a REAL \
                     geometry discrepancy, not the known unit-constant gap"
                );
            }
        }
        // Volume scales as the cube of the length ratio.
        let vol_ratio = cell.vol() / r.vol;
        assert!(
            (vol_ratio - expected_ratio.powi(3)).abs() < 1e-13,
            "{name}: vol ratio {vol_ratio:.17} != (length ratio)^3"
        );
    }
}

/// The fcc primitive cell has `vol == a0^3 / 4` in closed form — an independent
/// check on the hard-coded number that does not depend on upstream at all.
#[test]
fn diamond_vol_matches_the_closed_form_fcc_volume() {
    let a0_ang = 3.5668;
    let a0_bohr = a0_ang * Unit::Ang.length_in_au();
    let expected = a0_bohr.powi(3) / 4.0;
    let got = systems::diamond().vol();
    assert!(
        (got - expected).abs() < 1e-9,
        "diamond vol = {got} but a0^3/4 = {expected}"
    );
}

// ---------------------------------------------------------------------------
// Tier 1 — invariants.
// ---------------------------------------------------------------------------

#[test]
fn reciprocal_vectors_times_a_transpose_is_two_pi_identity() {
    for (name, cell) in systems::all() {
        let b = cell
            .reciprocal_vectors(2.0 * PI)
            .unwrap_or_else(|e| panic!("{name}: reciprocal_vectors failed: {e}"));
        let at = transpose3(&cell.lattice_vectors());
        // (b . a^T)[i][j] = sum_k b[i][k] * a^T[k][j]
        // `j` indexes a COLUMN of `at`, not a row of the matrix being iterated,
        // so the iterator form clippy suggests does not apply here.
        #[allow(clippy::needless_range_loop)]
        for (i, brow) in b.iter().enumerate() {
            for j in 0..3 {
                let v: f64 = (0..3).map(|k| brow[k] * at[k][j]).sum();
                let want = if i == j { 2.0 * PI } else { 0.0 };
                assert!(
                    (v - want).abs() < 1e-12,
                    "{name}: (b . a^T)[{i}][{j}] = {v:e}, expected {want:e} (tol 1e-12)"
                );
            }
        }
    }
}

#[test]
fn reciprocal_vectors_norm_to_scales_linearly() {
    let cell = systems::si();
    let b1 = cell.reciprocal_vectors(1.0).expect("norm_to = 1");
    let b2pi = cell.reciprocal_vectors(2.0 * PI).expect("norm_to = 2pi");
    for i in 0..3 {
        for j in 0..3 {
            assert!((b2pi[i][j] - 2.0 * PI * b1[i][j]).abs() < 1e-12);
        }
    }
}

#[test]
fn abs_kpts_round_trips_through_scaled_kpts() {
    let mut rng = Lcg::new(0x0903_0001);
    for (name, cell) in systems::all() {
        // 10 pseudo-random absolute k-points, fixed seed.
        let kpts: Vec<[f64; 3]> = (0..10)
            .map(|_| [rng.next_f64(), rng.next_f64(), rng.next_f64()])
            .collect();

        let scaled = cell.get_scaled_kpts(&kpts);
        let back = cell
            .get_abs_kpts(&scaled)
            .unwrap_or_else(|e| panic!("{name}: get_abs_kpts failed: {e}"));

        for (i, (orig, rt)) in kpts.iter().zip(back.iter()).enumerate() {
            for c in 0..3 {
                assert!(
                    (orig[c] - rt[c]).abs() < 1e-12,
                    "{name}: k[{i}][{c}] = {} round-tripped to {} (tol 1e-12)",
                    orig[c],
                    rt[c]
                );
            }
        }
    }
}

/// The other direction: scaled -> absolute -> scaled. A Gamma-centred 2x2x2
/// mesh is the shape every later k-point plan actually uses.
#[test]
fn scaled_kpts_round_trip_on_a_gamma_centred_mesh() {
    let cell = systems::diamond();
    let mut scaled = Vec::new();
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                scaled.push([i as f64 / 2.0, j as f64 / 2.0, k as f64 / 2.0]);
            }
        }
    }
    let abs = cell.get_abs_kpts(&scaled).expect("get_abs_kpts");
    let back = cell.get_scaled_kpts(&abs);
    for (s, b) in scaled.iter().zip(back.iter()) {
        for c in 0..3 {
            assert!((s[c] - b[c]).abs() < 1e-12);
        }
    }
    // Gamma maps to the origin exactly.
    assert_eq!(abs[0], [0.0, 0.0, 0.0]);
}

#[test]
fn scaled_atom_coords_match_the_construction_fractions() {
    // Diamond's carbons are at scaled (0,0,0) and (0.25,0.25,0.25) by
    // construction — this is the inverse-lattice check.
    let cell = systems::diamond();
    let s = cell.get_scaled_atom_coords().expect("scaled coords");
    assert_eq!(s.len(), 2);
    for (c, (origin, quarter)) in s[0].iter().zip(s[1].iter()).enumerate() {
        assert!(origin.abs() < 1e-12, "atom 0 is not at the origin");
        assert!(
            (quarter - 0.25).abs() < 1e-12,
            "atom 1 scaled coord {c} = {quarter} (expected 0.25)"
        );
    }
    // LiF's fluorine sits at scaled (0.5, 0.5, 0.5).
    let s = systems::lif()
        .get_scaled_atom_coords()
        .expect("scaled coords");
    for v in s[1].iter() {
        assert!((v - 0.5).abs() < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// D-PBC-01 — Deref to Mole.
// ---------------------------------------------------------------------------

#[test]
fn cell_derefs_to_the_owned_mole() {
    let cell = systems::diamond();
    // Field access through Deref.
    assert!(cell.nao_nr > 0, "nao_nr must be positive");
    assert_eq!(cell.natm, 2);
    assert_eq!(cell.nbas, 4, "2 C atoms x (1s + 1p) shells of gth-szv");
    assert!(!cell._env.is_empty(), "_env must be populated");
    assert!(cell.basis_set.is_some(), "the cintx BasisSet must be built");
    // Method call through Deref.
    assert_eq!(cell.atom_coords().len(), 2);
    assert_eq!(cell.atom_charges().len(), 2);
    // The Mole is OWNED, not duplicated: mutating through DerefMut is visible
    // on `cell.mol`.
    let mut cell = cell;
    cell.verbose = 4;
    assert_eq!(cell.mol.verbose, 4);
}

#[test]
fn tot_electrons_scales_with_the_k_point_count() {
    // Since plan 10-01 the reference systems carry their `gth-pade`
    // pseudopotential, so `atom_charges()` is the VALENCE charge: 2 x C(q4) = 8,
    // not 2 x C(Z=6) = 12.
    let cell = systems::diamond();
    assert_eq!(cell.atom_charges(), vec![4, 4]);
    let n1 = cell.tot_electrons(1);
    assert_eq!(n1, 8, "2 carbons at gth-pade valence charge 4, neutral");
    assert_eq!(cell.tot_electrons(4), 4 * n1);
    assert_eq!(cell.tot_electrons(8), 8 * n1);

    // Charge is subtracted ONCE, not per k-point (cell.py:957-967).
    // `build_diamond_with_charge` deliberately builds WITHOUT a pseudopotential,
    // so its counts stay all-electron and the two conventions are visible
    // side by side.
    let charged = build_diamond_with_charge(2);
    assert_eq!(charged.atom_charges(), vec![6, 6]);
    assert_eq!(charged.tot_electrons(1), 10);
    assert_eq!(charged.tot_electrons(4), 4 * 12 - 2);
}

fn build_diamond_with_charge(charge: i32) -> Cell {
    let a0 = 3.5668;
    let q = a0 / 4.0;
    let h = a0 / 2.0;
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("C".into(), [0.0, 0.0, 0.0]), ("C".into(), [q, q, q])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Ang,
            charge,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        ..Default::default()
    })
    .expect("charged diamond must build")
}

// ---------------------------------------------------------------------------
// Lattice input forms + build-time validation.
// ---------------------------------------------------------------------------

#[test]
fn all_three_lattice_input_forms_agree() {
    let h = 3.5668 / 2.0;
    let m = ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]);
    let f = ALattice::Flat([0.0, h, h, h, 0.0, h, h, h, 0.0]);
    let s = ALattice::Str(format!("0 {h} {h}; {h} 0 {h}; {h} {h} 0"));
    let (mm, fm, sm) = (
        m.to_matrix().expect("matrix"),
        f.to_matrix().expect("flat"),
        s.to_matrix().expect("str"),
    );
    assert_eq!(mm, fm);
    assert_eq!(mm, sm);
    // Commas and newlines are separators too (cell.py:1879).
    let s2 = ALattice::Str(format!("0,{h},{h}\n{h},0,{h}\n{h},{h},0"));
    assert_eq!(s2.to_matrix().expect("str2"), mm);
}

#[test]
fn malformed_lattice_strings_are_rejected() {
    assert!(ALattice::Str("1 2 3".into()).to_matrix().is_err());
    assert!(
        ALattice::Str("1 2 3 4 5 6 7 8 9 10".into())
            .to_matrix()
            .is_err()
    );
    assert!(
        ALattice::Str("1 2 3 4 5 6 7 8 nine".into())
            .to_matrix()
            .is_err()
    );
}

#[test]
fn build_rejects_a_singular_lattice() {
    let args = CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        // a3 = a1 + a2 — coplanar, det = 0.
        a: ALattice::Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
        ..Default::default()
    };
    assert!(
        Cell::build(args).is_err(),
        "a singular lattice must be rejected"
    );
}

#[test]
fn build_rejects_dimension_one_without_inf_vacuum() {
    let make = |dim: u8, ft: LowDimFtType| CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix([[3.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]),
        dimension: dim,
        low_dim_ft_type: ft,
        ..Default::default()
    };
    // cell.py:1665-1666.
    assert!(Cell::build(make(1, LowDimFtType::None)).is_err());
    assert!(Cell::build(make(1, LowDimFtType::InfVacuum)).is_ok());
    // dimension > 3 is nonsense.
    assert!(Cell::build(make(4, LowDimFtType::None)).is_err());
}

#[test]
fn fractional_coordinates_produce_the_same_cell_as_cartesian() {
    let a0 = 3.5668;
    let h = a0 / 2.0;
    let lattice = ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]);
    let frac = Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![
                ("C".into(), [0.0, 0.0, 0.0]),
                ("C".into(), [0.25, 0.25, 0.25]),
            ]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        a: lattice,
        fractional: true,
        ..Default::default()
    })
    .expect("fractional diamond must build");

    let cart = systems::diamond();
    for (f, c) in frac.atom_coords().iter().zip(cart.atom_coords().iter()) {
        for k in 0..3 {
            assert!(
                (f[k] - c[k]).abs() < 1e-12,
                "fractional {f:?} vs cartesian {c:?}"
            );
        }
    }
    assert!((frac.vol() - cart.vol()).abs() < 1e-9);
}

#[test]
fn exp_to_discard_removes_diffuse_primitives() {
    let a0 = 3.5668;
    let h = a0 / 2.0;
    let base = |cut: Option<f64>| CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("C".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        exp_to_discard: cut,
        ..Default::default()
    };
    let full = Cell::build(base(None)).expect("unfiltered");
    // A cutoff below every gth-szv exponent must be a no-op, byte-for-byte.
    let noop = Cell::build(base(Some(1e-12))).expect("no-op cutoff");
    assert_eq!(
        full._env, noop._env,
        "a below-range cutoff must change nothing"
    );
    assert_eq!(full._bas, noop._bas);
    // A cutoff above the most diffuse exponent must drop primitives.
    let trimmed = Cell::build(base(Some(0.5))).expect("real cutoff");
    assert!(
        trimmed._env.len() < full._env.len(),
        "exp_to_discard = 0.5 should have dropped primitives from gth-szv carbon \
         (full _env {} vs trimmed {})",
        full._env.len(),
        trimmed._env.len()
    );
    assert_eq!(trimmed.exp_to_discard, Some(0.5));
}

// ---------------------------------------------------------------------------
// dumps / loads.
// ---------------------------------------------------------------------------

#[test]
fn dumps_loads_round_trip_preserves_the_periodic_state() {
    for (name, cell) in systems::all() {
        let json = dumps(&cell).unwrap_or_else(|e| panic!("{name}: dumps failed: {e}"));
        let back = loads(&json).unwrap_or_else(|e| panic!("{name}: loads failed: {e}"));

        // The four fields plan 09-03 Task 6 names, bit-for-bit where exact.
        assert_eq!(back.a, cell.a, "{name}: a");
        assert_eq!(back.mesh, cell.mesh, "{name}: mesh");
        assert_eq!(
            back.precision.to_bits(),
            cell.precision.to_bits(),
            "{name}: precision"
        );
        assert_eq!(back.dimension, cell.dimension, "{name}: dimension");

        // The rest of the periodic state.
        assert_eq!(back.low_dim_ft_type, cell.low_dim_ft_type, "{name}");
        assert_eq!(back.rcut.to_bits(), cell.rcut.to_bits(), "{name}: rcut");
        assert_eq!(back.ke_cutoff, cell.ke_cutoff, "{name}");
        assert_eq!(back.pseudo_name, cell.pseudo_name, "{name}: pseudo name");
        assert_eq!(back.exp_to_discard, cell.exp_to_discard, "{name}");
        assert_eq!(back.fractional, cell.fractional, "{name}");
        assert!(back._built, "{name}: a loaded cell is built");

        // And the molecular half survives through the Deref.
        assert_eq!(back.natm, cell.natm, "{name}: natm");
        assert_eq!(back.nao_nr, cell.nao_nr, "{name}: nao_nr");
        assert_eq!(back._atm, cell._atm, "{name}: _atm");
        assert_eq!(back._bas, cell._bas, "{name}: _bas");
        assert_eq!(back._env, cell._env, "{name}: _env");
        // Derived quantities agree because the inputs did.
        assert!((back.vol() - cell.vol()).abs() < 1e-12, "{name}: vol");
    }
}

/// 17-03-PLAN.md Task 7: `dumps_loads` carries `space_group_symmetry` but
/// would previously drop its `symmorphic` partner silently (the field did
/// not exist at all before this plan). Pin the round trip explicitly on a
/// cell with `symmorphic = true` — none of the §9.2 fixtures set it, so the
/// blanket round-trip test above cannot distinguish `true` from the
/// (default) `false`.
#[test]
fn dumps_loads_round_trips_symmorphic() {
    let args = CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
        pseudo: Some("gth-pade".to_string()),
        symmorphic: true,
        ..Default::default()
    };
    let cell = Cell::build(args).expect("cell must build");
    assert!(cell.symmorphic);

    let json = dumps(&cell).expect("dumps");
    let back = loads(&json).expect("loads");
    assert!(back.symmorphic, "symmorphic=true must survive dumps/loads");
    // `lattice_symmetry` is derived, build-time-only state (see
    // `symmetry_data`'s module doc) — it is NOT serialised, and `loads`
    // always comes back with `None`, even if the original cell had one set.
    assert!(back.lattice_symmetry.is_none());

    // And the default (`false`) also survives — this is what every other
    // fixture in `dumps_loads_round_trip_preserves_the_periodic_state`
    // already exercises implicitly, made explicit here.
    let args_default = CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
        pseudo: Some("gth-pade".to_string()),
        ..Default::default()
    };
    let cell2 = Cell::build(args_default).expect("cell must build");
    assert!(!cell2.symmorphic);
    let back2 = loads(&dumps(&cell2).expect("dumps")).expect("loads");
    assert!(!back2.symmorphic);
}

#[test]
fn loads_rejects_garbage() {
    assert!(loads("not json").is_err());
    assert!(loads("{}").is_err());
}

// ---------------------------------------------------------------------------
// 3x3 helpers.
// ---------------------------------------------------------------------------

#[test]
fn det3_and_inv3_are_consistent() {
    let mut rng = Lcg::new(0x0903_0002);
    for _ in 0..50 {
        let m = [
            [rng.next_f64(), rng.next_f64(), rng.next_f64()],
            [rng.next_f64(), rng.next_f64(), rng.next_f64()],
            [rng.next_f64(), rng.next_f64(), rng.next_f64()],
        ];
        if det3(&m).abs() < 1e-3 {
            continue; // skip near-singular draws
        }
        let inv = inv3(&m).expect("non-singular");
        // `j` indexes a COLUMN of `inv` — see the note in the reciprocal-vector test.
        #[allow(clippy::needless_range_loop)]
        for (i, mrow) in m.iter().enumerate() {
            for j in 0..3 {
                let v: f64 = (0..3).map(|k| mrow[k] * inv[k][j]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((v - want).abs() < 1e-10, "m . inv(m) [{i}][{j}] = {v:e}");
            }
        }
    }
    // Identity and a known determinant.
    let id = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    assert_eq!(det3(&id), 1.0);
    assert_eq!(inv3(&id).expect("identity"), id);
    let m = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 4.0]];
    assert_eq!(det3(&m), 24.0);
    assert!(inv3(&[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 0.0, 1.0]]).is_err());
}

#[test]
fn transpose3_is_an_involution() {
    let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    assert_eq!(transpose3(&transpose3(&m)), m);
    assert_eq!(transpose3(&m)[0][1], m[1][0]);
}

// ---------------------------------------------------------------------------
// Plan 09-04 hand-off — the gap this file used to assert is now CLOSED.
// ---------------------------------------------------------------------------

/// Plan 09-03 left `rcut`/`mesh` un-estimated and asserted here that
/// `try_rcut`/`try_mesh` surfaced a loud `NotYetImplemented`. Plan 09-04 wired
/// the estimators, so the assertion FLIPS: a built cell now carries real
/// values, the sentinels are gone, and the accessors agree with the fields.
/// (The estimators' numeric agreement with upstream is `tests/cutoff.rs`;
/// this test only guards the plan-09-03 hand-off contract.)
#[test]
fn plan_09_04_filled_in_rcut_and_mesh() {
    let cell = systems::diamond();
    assert!(cell._rcut_from_build && cell._mesh_from_build);
    assert_ne!(cell.rcut, pyscf_pbc_gto::cell::RCUT_UNSET);
    assert_ne!(cell.mesh, pyscf_pbc_gto::cell::MESH_UNSET);
    assert_eq!(cell.try_rcut().expect("rcut is estimated"), cell.rcut);
    assert_eq!(cell.try_mesh().expect("mesh is estimated"), cell.mesh);
    // A hand-assembled `Cell` (never built) still carries the sentinels, and
    // the accessors estimate on demand rather than returning a zero cutoff.
    let bare = Cell::default();
    assert_eq!(bare.rcut, pyscf_pbc_gto::cell::RCUT_UNSET);
    assert_eq!(bare.mesh, pyscf_pbc_gto::cell::MESH_UNSET);
    assert_eq!(bare.try_rcut().expect("empty-basis rcut"), 0.01);
}

/// A user-pinned `rcut`/`mesh` bypasses the estimator entirely — that path is
/// already complete and must keep working after 09-04 lands.
#[test]
fn user_supplied_rcut_and_mesh_are_honoured() {
    let a0 = 3.5668;
    let h = a0 / 2.0;
    let cell = Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("He".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Ang,
            ..Default::default()
        },
        a: ALattice::Matrix([[0.0, h, h], [h, 0.0, h], [h, h, 0.0]]),
        rcut: Some(15.5),
        mesh: Some([15, 15, 15]),
        ke_cutoff: Some(100.0),
        ..Default::default()
    })
    .expect("pinned cell must build");
    assert_eq!(cell.rcut, 15.5);
    assert_eq!(cell.mesh, [15, 15, 15]);
    assert!(!cell._rcut_from_build && !cell._mesh_from_build);
    assert_eq!(cell.try_rcut().expect("pinned rcut"), 15.5);
    assert_eq!(cell.try_mesh().expect("pinned mesh"), [15, 15, 15]);
}

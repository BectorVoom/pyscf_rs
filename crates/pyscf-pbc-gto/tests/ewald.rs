//! Plan 09-08 acceptance gate — `get_ewald_params`, `ewald`, `energy_nuc`,
//! and the particle-mesh B-splines.
//!
//! Tier 1 (invariants, no upstream needed) comes first and must pass
//! unconditionally. Tier 2 pins hard-coded upstream numbers (D-PBC-19); every
//! literal lives in `tests/common/ewald_reference.rs` together with the exact
//! snippet that generated it.
//!
//! The tier-2 cells are specified in BOHR — see that file's module docs for why
//! (the 4.95e-9 CODATA gap of plan 09-03 would otherwise cost 1.4e-7 Ha, two
//! orders above this plan's 1e-9 Ha gate). The §9.2 Angstrom systems are swept
//! separately at the tolerance that gap implies, so the conversion path stays
//! covered.

mod common;

use common::ewald_reference::{
    DIAMOND_ETA_SCAN, EWALD_REFERENCES, EwaldReference, PSEUDISED_EWALD,
};
use common::systems;
use pyscf_core::{PyscfRsError, Unit};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_gto::cutoff::estimate_rcut_pgto;
use pyscf_pbc_gto::ewald_pme::{
    INTERPOLATION_ORDER, bspline, bspline_grad, bspline_value, get_ewald_direct, pme_charge_mesh,
};
use pyscf_pbc_gto::{ALattice, Cell, CellBuildArgs, ewald, ewald_self, get_ewald_params};
use std::f64::consts::PI;

/// Plan 09-08's numeric acceptance gate.
const EWALD_TOL: f64 = 1e-9;

/// Build a reference system directly from its BOHR geometry, so no unit
/// conversion enters. `pseudo` is deliberately unset — see
/// `common::ewald_reference`.
fn bohr_cell(r: &EwaldReference) -> Cell {
    let atoms: Vec<(String, [f64; 3])> = r
        .symbols
        .iter()
        .zip(r.coords_bohr.iter())
        .map(|(s, xyz)| ((*s).to_string(), *xyz))
        .collect();
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(r.a_bohr),
        dimension: r.dimension,
        ..Default::default()
    })
    .expect("reference cell must build")
}

fn reference(name: &str) -> &'static EwaldReference {
    EWALD_REFERENCES
        .iter()
        .find(|r| r.name == name)
        .expect("unknown reference system")
}

// ---------------------------------------------------------------------------
// Tier 1 — invariants. No upstream needed; these must pass unconditionally.
// ---------------------------------------------------------------------------

/// PBC-MASTER-PLAN §8.1 plan 09-08 TEST block, second bullet: the Ewald energy
/// is a physical quantity and must not depend on the arbitrary split between
/// the real- and reciprocal-space sums.
///
/// `ew_cut` moves WITH `ew_eta` (`_estimate_rcut(eta^2, 0, 1., precision)`,
/// exactly as `get_ewald_params` derives it). That is load-bearing: weaker
/// screening needs a longer real-space tail, and pinning `ew_cut` at its
/// `eta0` value makes upstream ITSELF drift 8.1e-7 Ha at `0.5*eta0`.
#[test]
fn ewald_is_invariant_to_ew_eta() {
    let cell = bohr_cell(reference("diamond"));
    let (eta0, cut0) = get_ewald_params(&cell, None, None).expect("ewald params");
    let e0 = ewald(&cell, Some(eta0), Some(cut0)).expect("ewald at eta0");

    for scale in [0.5_f64, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0] {
        let eta = eta0 * scale;
        let cut = estimate_rcut_pgto(eta * eta, 0, 1.0, cell.precision);
        let e = ewald(&cell, Some(eta), Some(cut)).expect("ewald at scaled eta");
        assert!(
            (e - e0).abs() < 1e-8,
            "ewald not eta-invariant: scale {scale}, eta {eta:e}, cut {cut:e}, \
             E = {e:.15}, E(eta0) = {e0:.15}, dev {:.3e}",
            (e - e0).abs()
        );
    }
}

/// `cell.py:676-677` — for `dimension == 3` the screening parameter is exactly
/// `vol^(-1/6)`, and `ew_cut` is the `l = 0, c = 1` overlap radius of a
/// primitive with exponent `eta^2`.
#[test]
fn get_ewald_params_is_the_upstream_closed_form_in_3d() {
    for r in EWALD_REFERENCES.iter().filter(|r| r.dimension == 3) {
        let cell = bohr_cell(r);
        let (eta, cut) = get_ewald_params(&cell, None, None).expect("ewald params");
        let expect_eta = 1.0 / cell.vol().powf(1.0 / 6.0);
        let expect_cut = estimate_rcut_pgto(expect_eta * expect_eta, 0, 1.0, cell.precision);
        assert_eq!(eta, expect_eta, "{}: ew_eta", r.name);
        assert_eq!(cut, expect_cut, "{}: ew_cut", r.name);
    }
}

/// `cell.py:686-691` — the `dimension == 2` branch is pure parameter algebra
/// and ships even though `ewald()` itself defers that branch.
#[test]
fn get_ewald_params_ships_the_dimension_2_branch() {
    let r = reference("graphene");
    let cell = bohr_cell(r);
    let (eta, cut) = get_ewald_params(&cell, None, None).expect("ewald params");
    assert_eq!(cut, cell.lattice_vectors()[2][2] / 2.0, "ew_cut = a[2,2]/2");
    let qsum: f64 = cell.mol.atom_charges().iter().map(|z| *z as f64).sum();
    let log_precision = (cell.precision / (qsum * 16.0 * PI * PI)).ln();
    assert_eq!(eta, (-log_precision).sqrt() / cut, "ew_eta");
}

/// `cell.py:670-671` and `cell.py:712-713` — an atom-free cell has no Ewald
/// parameters and no energy.
///
/// The cell is assembled by hand rather than through [`Cell::build`], which
/// rejects an empty atom list at the cintx basis-set layer. Upstream's guards
/// are still reachable in this port through a `Cell` whose atoms were removed
/// after the fact, so they are worth pinning.
#[test]
fn empty_cell_has_zero_ewald() {
    let cell = Cell {
        a: [[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]],
        ..Default::default()
    };
    assert_eq!(cell.mol.natm, 0);
    assert_eq!(
        get_ewald_params(&cell, None, None).expect("params"),
        (0.0, 0.0)
    );
    assert_eq!(ewald(&cell, None, None).expect("ewald"), 0.0);
}

/// `cell.py:708-710` — "if lattice parameter is not set, the cell object is
/// treated as a mole object". A `Cell` always carries an `a` field, so this
/// port's analogue is a DEGENERATE lattice, which must fall back to the
/// molecular nuclear repulsion instead of dividing by a zero volume.
#[test]
fn degenerate_lattice_falls_back_to_the_molecular_nuclear_repulsion() {
    let built = bohr_cell(reference("diamond"));
    let cell = Cell {
        a: [[0.0; 3]; 3],
        ..built
    };
    let e = ewald(&cell, None, None).expect("ewald");
    assert_eq!(e, cell.mol.enuc());
    assert!(e > 0.0, "two bare nuclei repel: {e}");
}

/// `cell.py:738-741` — the self term is closed-form, so it can be checked
/// against the formula rather than against upstream.
#[test]
fn ewald_self_matches_the_martin_f5_closed_form() {
    let chargs = [3.0_f64, 9.0, -1.5];
    let eta = 0.37;
    let vol = 110.42101837541341;
    let q2: f64 = chargs.iter().map(|q| q * q).sum();
    let qs: f64 = chargs.iter().sum();

    let expect_3d = -0.5 * q2 * 2.0 * eta / PI.sqrt() - 0.5 * qs * qs * PI / (eta * eta * vol);
    assert!((ewald_self(&chargs, eta, 3, vol) - expect_3d).abs() < 1e-14);

    // The neutralising-background term is 3D-only (`cell.py:740`).
    let expect_2d = -0.5 * q2 * 2.0 * eta / PI.sqrt();
    assert!((ewald_self(&chargs, eta, 2, vol) - expect_2d).abs() < 1e-14);
}

/// `energy_nuc = ewald` (`cell.py:824`).
#[test]
fn energy_nuc_is_ewald() {
    let cell = bohr_cell(reference("diamond"));
    let a = cell.energy_nuc().expect("energy_nuc");
    let b = ewald(&cell, None, None).expect("ewald");
    assert_eq!(a, b);
}

/// The regime check every 3D reference system must satisfy regardless of
/// upstream: a neutralised lattice of positive point charges is bound by its
/// compensating background, so `ewald()` is finite and negative. Catches a sign
/// slip or a NaN in `ewovrl` / `ewself` / `ewg` without needing a reference
/// number.
#[test]
fn ewald_is_negative_for_every_3d_reference_system() {
    for r in EWALD_REFERENCES.iter().filter(|r| r.dimension == 3) {
        let cell = bohr_cell(r);
        let e = ewald(&cell, None, None).expect("ewald");
        assert!(e < 0.0, "{}: ewald() = {e} should bind", r.name);
        assert!(e.is_finite(), "{}: ewald() = {e}", r.name);
    }
}

/// D-PBC-20 — deferred branches return a typed `NotYetImplemented`, never a
/// silently wrong number. Graphene is `dimension = 2`.
#[test]
fn ewald_defers_the_dimension_2_branch_to_phase_12() {
    let cell = bohr_cell(reference("graphene"));
    match ewald(&cell, None, None) {
        Err(PyscfRsError::NotYetImplemented { phase: 12, what }) => {
            assert!(what.contains("dimension = 2"), "{what}");
        }
        other => panic!("expected NotYetImplemented{{phase:12}}, got {other:?}"),
    }
}

/// D-PBC-20 — particle-mesh Ewald needs the Phase 11 FFT.
#[test]
fn particle_mesh_ewald_defers_to_phase_11() {
    let mut cell = bohr_cell(reference("diamond"));
    cell.use_particle_mesh_ewald = true;
    match ewald(&cell, None, None) {
        Err(PyscfRsError::NotYetImplemented { phase: 11, .. }) => {}
        other => panic!("expected NotYetImplemented{{phase:11}}, got {other:?}"),
    }
}

/// `ewald_methods.py:32-38` — cardinal B-splines are a partition of unity:
/// `sum_x M[t, x] == 1` for every interpolated point.
#[test]
fn bsplines_are_a_partition_of_unity() {
    let u = [0.0_f64, 0.25, 3.75, 12.5, -2.3, 17.999];
    for order in [4_usize, 6, INTERPOLATION_ORDER] {
        let ng = 20;
        let s = bspline(&u, ng, order, 0).expect("bspline");
        for t in 0..u.len() {
            let row: f64 = s.m[t * ng..(t + 1) * ng].iter().sum();
            // 1e-11, not 1e-15: the truncated-power form of `M_n` cancels
            // ~11 digits at order 10, and upstream drifts by 1.7e-12 on the
            // same points (`M.sum(axis=1)` for u = -2.3 gives
            // 1.0000000000017213). This bound is the shared floor, not slack.
            assert!(
                (row - 1.0).abs() < 1e-11,
                "order {order}, point {t}: sum = {row}"
            );
        }
    }
}

/// `ewald_methods.py:40-47` — `dM_n/du = M_{n-1}(u) - M_{n-1}(u-1)` must agree
/// with a central finite difference of `M_n`.
/// `h = 1e-4` rather than a tighter step on purpose: the truncated-power form
/// of `M_n` cancels ~11 significant digits at order 10, so a smaller step
/// amplifies that noise faster than it reduces the O(h^2) truncation error.
/// The sweep stays inside the support `(0, n)`, where `M_n` is non-trivial;
/// outside it both sides are identically zero.
#[test]
fn bspline_grad_matches_a_finite_difference() {
    let h = 1e-4_f64;
    for order in [4_usize, 6, 10] {
        for k in 1..(order * 4) {
            let u = k as f64 * 0.25;
            let fd = (bspline_value(u + h, order) - bspline_value(u - h, order)) / (2.0 * h);
            let an = bspline_grad(u, order);
            assert!(
                (fd - an).abs() < 1e-6,
                "order {order}, u {u}: analytic {an}, fd {fd}"
            );
        }
    }
}

/// `ewald_methods.py:71-78` — `b[0]` is `1/sum_k M_n(k+1) = 1` because the
/// cardinal B-spline knots also sum to one, and the whole coefficient array is
/// finite.
#[test]
fn bspline_euler_coefficients_are_finite_and_normalised() {
    let u = [1.5_f64, 7.25];
    let ng = 16;
    let s = bspline(&u, ng, 6, 1).expect("bspline");
    assert!(s.dm.is_some(), "deriv = 1 must return dM");
    assert!((s.b_re[0] - 1.0).abs() < 1e-12, "b[0] = {}", s.b_re[0]);
    assert!(s.b_im[0].abs() < 1e-12, "b[0] imag = {}", s.b_im[0]);
    for m in 0..ng {
        assert!(
            s.b_re[m].is_finite() && s.b_im[m].is_finite(),
            "b[{m}] is not finite"
        );
    }
}

/// `ewald_methods.py:63-64` — upstream raises for `deriv > 1`; so do we.
#[test]
fn bspline_rejects_second_derivatives() {
    assert!(bspline(&[1.0], 8, 4, 2).is_err());
}

/// The PME charge mesh carries the total charge: `sum_xyz Q = sum_a q_a`,
/// because each axis' B-spline weights sum to one.
#[test]
fn pme_charge_mesh_conserves_total_charge() {
    let mesh = [12_usize, 10, 8];
    let chargs = [6.0_f64, 6.0];
    let ux = [0.0_f64, 3.4];
    let uy = [0.0_f64, 2.9];
    let uz = [0.0_f64, 1.1];
    let mx = bspline(&ux, mesh[0], 6, 0).expect("mx");
    let my = bspline(&uy, mesh[1], 6, 0).expect("my");
    let mz = bspline(&uz, mesh[2], 6, 0).expect("mz");
    let q = pme_charge_mesh(&chargs, &mx, &my, &mz, mesh);
    let total: f64 = q.iter().sum();
    assert!(
        (total - 12.0).abs() < 1e-10,
        "charge mesh total = {total}, expected 12"
    );
}

/// `cell.c:get_ewald_direct` is the SCREENED cousin of `ewald_real_space`. With
/// the same `(eta, cut)` the two agree to the size of the tail they treat
/// differently — the C loop drops pairs with `r >= rcut`, which `erfc` has
/// already suppressed below `precision` by construction.
#[test]
fn screened_direct_sum_agrees_with_the_array_real_space_sum() {
    let cell = bohr_cell(reference("diamond"));
    let (eta, cut) = get_ewald_params(&cell, None, None).expect("params");
    let direct = get_ewald_direct(&cell, Some(eta), Some(cut)).expect("direct");

    let chargs: Vec<f64> = cell.mol.atom_charges().iter().map(|z| *z as f64).collect();
    let coords = cell.mol.atom_coords();
    let ls = pyscf_pbc_gto::get_lattice_ls(&cell, Some(cut), None, true).expect("Ls");
    let array = pyscf_pbc_gto::ewald_real_space(&chargs, &coords, &ls, eta).expect("real space");

    assert!(
        (direct - array).abs() < 1e-8,
        "screened {direct:.15} vs array {array:.15}"
    );
}

// ---------------------------------------------------------------------------
// Tier 2 — hard-coded upstream values (D-PBC-19).
// ---------------------------------------------------------------------------

/// `cell.get_ewald_params()` for all five §9.2 systems, Bohr-specified.
#[test]
fn ewald_params_match_upstream() {
    for r in EWALD_REFERENCES.iter() {
        let cell = bohr_cell(r);
        assert!(
            (cell.vol() - r.vol).abs() < 1e-10,
            "{}: vol {} vs {}",
            r.name,
            cell.vol(),
            r.vol
        );
        assert_eq!(
            cell.mol.atom_charges(),
            r.charges,
            "{}: atom_charges (these cells are built WITHOUT a pseudopotential)",
            r.name
        );
        let (eta, cut) = get_ewald_params(&cell, None, None).expect("params");
        assert!(
            (eta - r.ew_eta).abs() < 1e-12,
            "{}: ew_eta {eta:.17} vs {:.17}",
            r.name,
            r.ew_eta
        );
        assert!(
            (cut - r.ew_cut).abs() < 1e-10,
            "{}: ew_cut {cut:.17} vs {:.17}",
            r.name,
            r.ew_cut
        );
    }
}

/// The two grids `ewald()` builds internally: the real-space image list sized
/// by `ew_cut`, and the G-space FFT mesh sized by `ke_cutoff`. Both are integer
/// quantities and are asserted EXACTLY.
#[test]
fn ewald_internal_grids_match_upstream() {
    for r in EWALD_REFERENCES.iter() {
        let cell = bohr_cell(r);
        let (eta, cut) = get_ewald_params(&cell, None, None).expect("params");

        let ls = pyscf_pbc_gto::get_lattice_ls(&cell, Some(cut), None, true).expect("Ls");
        assert_eq!(
            ls.len(),
            r.n_ls,
            "{}: len(get_lattice_Ls(rcut=ew_cut))",
            r.name
        );

        let qsum: f64 = cell.mol.atom_charges().iter().map(|z| *z as f64).sum();
        let log_precision = (cell.precision / (qsum * 16.0 * PI * PI)).ln();
        let ke_cutoff = -2.0 * eta * eta * log_precision;
        let mesh = cell.cutoff_to_mesh(ke_cutoff).expect("mesh");
        assert_eq!(mesh, r.mesh, "{}: cutoff_to_mesh for ewald", r.name);
    }
}

/// **The plan's primary acceptance gate**: `cell.ewald()` matches upstream to
/// 1e-9 Ha on every system whose branch this plan ships.
#[test]
fn ewald_matches_upstream_to_1e_9_hartree() {
    let mut checked = 0;
    for r in EWALD_REFERENCES.iter() {
        let Some(expect) = r.ewald else { continue };
        let cell = bohr_cell(r);
        let got = ewald(&cell, None, None).expect("ewald");
        assert!(
            (got - expect).abs() < EWALD_TOL,
            "{}: ewald() = {got:.15} vs upstream {expect:.15}, dev {:.3e}",
            r.name,
            (got - expect).abs()
        );
        checked += 1;
    }
    assert_eq!(checked, 4, "diamond, si, lif and he_fcc must all be gated");
}

/// The `eta`-invariance scan, pinned against upstream point by point — this is
/// tier 1's `ewald_is_invariant_to_ew_eta` with the actual upstream numbers
/// behind it.
#[test]
fn ewald_eta_scan_matches_upstream() {
    let cell = bohr_cell(reference("diamond"));
    let (eta0, _) = get_ewald_params(&cell, None, None).expect("params");
    for p in DIAMOND_ETA_SCAN.iter() {
        let eta = eta0 * p.scale;
        let cut = estimate_rcut_pgto(eta * eta, 0, 1.0, cell.precision);
        assert!(
            (cut - p.ew_cut).abs() < 1e-10,
            "scale {}: ew_cut {cut:.17} vs {:.17}",
            p.scale,
            p.ew_cut
        );
        let got = ewald(&cell, Some(eta), Some(cut)).expect("ewald");
        assert!(
            (got - p.ewald).abs() < EWALD_TOL,
            "scale {}: ewald {got:.15} vs upstream {:.15}",
            p.scale,
            p.ewald
        );
    }
}

/// The §9.2 Angstrom-input systems, so the unit-conversion path is covered too.
///
/// `pyscf_core::Unit::Ang` is CODATA-2014 while upstream is CODATA-2010 — a
/// 4.951e-9 relative lattice gap (plan 09-03). Ewald scales as `1/length`, so
/// the expected deviation is `|E| * 4.951e-9`; the bound below is that with a
/// 10x margin. The 1e-9 Ha gate lives on the Bohr-specified cells above.
///
/// **These are the `test_systems` cells, which carry `pseudo = 'gth-pade'`**, so
/// since plan 10-01 their charges — and therefore their nuclear repulsion — are
/// the VALENCE ones. The expectation is [`PSEUDISED_EWALD`], not the
/// all-electron [`EWALD_REFERENCES`] value; comparing against the latter is a
/// 16 Ha error, not a unit-conversion one.
#[test]
fn angstrom_reference_systems_match_upstream_within_the_unit_gap() {
    const UNIT_GAP: f64 = 4.951e-9;
    let mut checked = 0;
    for (name, cell) in systems::all() {
        let r = reference(name);
        assert!(
            cell.pseudo.is_some(),
            "{name}: the §9.2 reference systems are gth-pade cells"
        );
        let (_, charges, expect) = PSEUDISED_EWALD
            .iter()
            .find(|(n, _, _)| *n == name)
            .expect("every reference system has a pseudised ewald target");
        assert_eq!(cell.atom_charges(), *charges, "{name}: valence charges");

        if r.ewald.is_none() {
            // graphene: dimension = 2 is deferred, and that must stay true for
            // the Angstrom build as well.
            assert!(matches!(
                ewald(&cell, None, None),
                Err(PyscfRsError::NotYetImplemented { phase: 12, .. })
            ));
            continue;
        }
        let got = ewald(&cell, None, None).expect("ewald");
        let bound = expect.abs() * UNIT_GAP * 10.0;
        assert!(
            (got - expect).abs() < bound,
            "{name}: ewald() = {got:.15} vs upstream {expect:.15}, \
             dev {:.3e} exceeds the {bound:.3e} unit-gap allowance",
            (got - expect).abs()
        );
        checked += 1;
    }
    assert_eq!(checked, 4, "diamond, si, lif and he_fcc must all be gated");
}

/// §9.3 — the reduction is `oracle_sum`, so repeated evaluation is
/// BIT-identical, not merely close.
#[test]
fn ewald_is_bit_reproducible_across_calls() {
    let cell = bohr_cell(reference("lif"));
    let first = ewald(&cell, None, None).expect("ewald");
    for _ in 0..4 {
        assert_eq!(
            ewald(&cell, None, None).expect("ewald").to_bits(),
            first.to_bits(),
            "ewald() is not bit-reproducible"
        );
    }
}

/// Plan 10-01 made `atom_charges()` return the GTH VALENCE charge, and
/// `cell.ewald()` reads those charges — so a `gth-pade` cell now has a
/// different (much smaller) nuclear repulsion than the all-electron one.
///
/// [`PSEUDISED_EWALD`] recorded upstream's pseudised numbers during Phase 9 as
/// plan 10-01's target. This test collects on that: the same five systems,
/// rebuilt WITH `pseudo = 'gth-pade'`, must reproduce them.
#[test]
fn pseudised_ewald_matches_the_recorded_upstream_targets() {
    let mut checked = 0;
    for (name, charges, expect) in PSEUDISED_EWALD.iter() {
        let r = reference(name);
        // The `dimension == 2` branch (graphene) is Phase 12 — `ewald` returns
        // NotYetImplemented there, exactly as it does for the all-electron run.
        if r.dimension != 3 {
            continue;
        }
        let cell = bohr_cell_with_pseudo(r, "gth-pade");
        assert_eq!(
            cell.atom_charges(),
            *charges,
            "{name}: gth-pade valence charges"
        );
        let got = ewald(&cell, None, None).expect("ewald");
        assert!(
            (got - expect).abs() < EWALD_TOL,
            "{name}: pseudised ewald() = {got:.15} vs upstream {expect:.15}, dev {:.3e}",
            (got - expect).abs()
        );
        checked += 1;
    }
    assert_eq!(checked, 4, "diamond, si, lif and he_fcc must all be gated");
}

/// [`bohr_cell`] with a pseudopotential attached.
fn bohr_cell_with_pseudo(r: &EwaldReference, pseudo: &str) -> Cell {
    let atoms: Vec<(String, [f64; 3])> = r
        .symbols
        .iter()
        .zip(r.coords_bohr.iter())
        .map(|(s, xyz)| ((*s).to_string(), *xyz))
        .collect();
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(atoms),
            basis: BasisInput::Name("gth-szv".into()),
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(r.a_bohr),
        dimension: r.dimension,
        pseudo: Some(pseudo.to_string()),
        ..Default::default()
    })
    .expect("pseudised reference cell must build")
}

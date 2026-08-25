//! Plan 09-06, `Cell` level — `get_lattice_Ls`, `super_cell`, `cell_plus_imgs`,
//! `get_monkhorst_pack_size`, `check_lattice_sum_range`.
//!
//! The geometry core is gated by `crates/pyscf-pbc-tools/tests/lattice.rs`,
//! where the cell is specified in Bohr and the comparison against upstream is
//! exact. This file gates the `Cell` plumbing on the five §9.2 reference
//! systems, whose lattices come from Angstrom input and therefore carry the
//! known 4.95e-9 relative Angstrom -> Bohr gap (plan 09-03): integers and image
//! counts are asserted EXACTLY, floats at 1e-7 relative.

mod common;

use common::systems;
use pyscf_pbc_gto::lattice::{
    check_lattice_sum_range, get_lattice_ls, get_lattice_ls_default, get_monkhorst_pack_size,
    get_monkhorst_pack_size_default, lattice_sum_dimension,
};
use pyscf_pbc_gto::supercell::{cell_plus_imgs, super_cell};
use pyscf_pbc_tools::mat3::norm3;

/// The exact generating snippet for every tier-2 literal here (PySCF 2.12.1).
///
/// ```python
/// from pyscf.pbc import gto, tools
/// # the five §9.2 systems, built exactly as crates/pyscf-pbc-gto/src/test_systems.rs
/// for name, c in systems:
///     for rcut in (10.0, 5.0):
///         for d in (True, False):
///             print(name, rcut, d, len(tools.pbc.get_lattice_Ls(c, rcut=rcut, discard=d)))
/// sc = tools.super_cell(diamond, [2,2,2])
/// print(sc.natm, sc.vol, sc.mesh, sc.nao_nr())
/// cpi = tools.cell_plus_imgs(diamond, [1,1,1])
/// print(cpi.natm, cpi.vol, cpi.mesh)
/// print(tools.pbc.get_monkhorst_pack_size(diamond, diamond.make_kpts([3,2,1])))
/// ```
const UPSTREAM_SNIPPET: &str = "see the doc comment above";

// ---------------------------------------------------------------------------
// get_lattice_Ls
// ---------------------------------------------------------------------------

#[test]
fn lattice_ls_image_counts_match_upstream() {
    assert!(!UPSTREAM_SNIPPET.is_empty());
    // (name, rcut, n_discard_true, n_discard_false)
    let cases: [(&str, f64, usize, usize); 10] = [
        ("diamond", 10.0, 135, 729),
        ("diamond", 5.0, 19, 343),
        ("si", 10.0, 43, 343),
        ("si", 5.0, 13, 125),
        ("lif", 10.0, 177, 729),
        ("lif", 5.0, 55, 343),
        ("he_fcc", 10.0, 87, 729),
        ("he_fcc", 5.0, 13, 125),
        ("graphene", 10.0, 31, 189),
        ("graphene", 5.0, 7, 75),
    ];
    let all = systems::all();
    for (name, rcut, n_true, n_false) in cases {
        let cell = &all
            .iter()
            .find(|(n, _)| *n == name)
            .expect("reference system")
            .1;
        let kept = get_lattice_ls(cell, Some(rcut), None, true).expect("Ls");
        let full = get_lattice_ls(cell, Some(rcut), None, false).expect("Ls");
        assert_eq!(kept.len(), n_true, "{name} rcut = {rcut} discard = true");
        assert_eq!(full.len(), n_false, "{name} rcut = {rcut} discard = false");
        // The origin is always in the list, and sits mid-array before discard.
        assert_eq!(full[full.len() / 2], [0.0, 0.0, 0.0], "{name}");
        assert_eq!(kept.iter().filter(|l| norm3(l) == 0.0).count(), 1, "{name}");
    }
}

#[test]
fn lattice_sum_dimension_follows_low_dim_ft_type() {
    // pbc.py:609-614 — a 2D cell still sums in all 3 dimensions unless the
    // non-periodic direction is infinite vacuum.
    assert_eq!(lattice_sum_dimension(&systems::diamond()), 3);
    let graphene = systems::graphene();
    assert_eq!(graphene.dimension, 2);
    assert_eq!(lattice_sum_dimension(&graphene), 3);
    // ... and that really does produce out-of-plane images.
    let full = get_lattice_ls(&graphene, Some(10.0), None, false).expect("Ls");
    assert!(full.iter().any(|l| l[2] != 0.0));

    let mut inf_vac = graphene.clone();
    inf_vac.low_dim_ft_type = pyscf_pbc_gto::LowDimFtType::InfVacuum;
    assert_eq!(lattice_sum_dimension(&inf_vac), 2);
    let planar = get_lattice_ls(&inf_vac, Some(10.0), None, false).expect("Ls");
    assert!(planar.iter().all(|l| l[2] == 0.0));
}

#[test]
fn lattice_ls_defaults_to_cell_rcut() {
    let cell = systems::diamond();
    let by_default = get_lattice_ls_default(&cell).expect("Ls");
    let explicit = get_lattice_ls(&cell, Some(cell.rcut), None, true).expect("Ls");
    assert_eq!(by_default, explicit);
    // cell.rcut is much larger than 10 Bohr, so the default list is bigger.
    assert!(by_default.len() > get_lattice_ls(&cell, Some(10.0), None, true).expect("Ls").len());
}

#[test]
fn check_lattice_sum_range_matches_upstream() {
    // upstream (Bohr-specified diamond): 22.035133643781318
    let cell = systems::diamond();
    let ls = get_lattice_ls_default(&cell).expect("Ls");
    let d = check_lattice_sum_range(&cell, &ls).expect("range");
    let rel = (d - 22.035133643781318).abs() / 22.035133643781318;
    assert!(rel < 1e-7, "check_lattice_sum_range = {d}");
    // It must exceed rcut: everything closer is inside the sum by construction.
    assert!(d > cell.rcut, "{d} <= rcut = {}", cell.rcut);
}

// ---------------------------------------------------------------------------
// super_cell / cell_plus_imgs
// ---------------------------------------------------------------------------

#[test]
fn super_cell_222_matches_upstream() {
    let cell = systems::diamond();
    let sc = super_cell(&cell, [2, 2, 2], false).expect("supercell");

    // upstream: natm 16, vol 612.4390450600976, mesh [94,94,94], nao_nr 64
    assert_eq!(sc.natm, 16);
    assert_eq!(sc.nao_nr, 64);
    assert_eq!(sc.mesh, [cell.mesh[0] * 2, cell.mesh[1] * 2, cell.mesh[2] * 2]);
    assert_eq!(cell.mesh, [47, 47, 47]);

    let ratio = sc.vol() / cell.vol();
    assert!((ratio - 8.0).abs() < 1e-9, "vol ratio = {ratio}");
    let rel = (sc.vol() - 612.4390450600976).abs() / 612.4390450600976;
    assert!(rel < 1e-7, "supercell vol = {}", sc.vol());

    // a_super[i] = 2 * a[i]
    for i in 0..3 {
        for j in 0..3 {
            assert_eq!(sc.a[i][j], 2.0 * cell.a[i][j]);
        }
    }
    // cell.copy(deep=False) carries rcut and precision through unchanged.
    assert_eq!(sc.rcut, cell.rcut);
    assert_eq!(sc.precision, cell.precision);
    assert_eq!(sc.dimension, cell.dimension);
    assert_eq!(sc.pseudo_name, cell.pseudo_name);

    // Image-major atom ordering: image i holds atoms i*natm .. (i+1)*natm.
    let base = cell.atom_coords();
    let got = sc.atom_coords();
    for img in 0..8 {
        let l = [
            got[img * 2][0] - base[0][0],
            got[img * 2][1] - base[0][1],
            got[img * 2][2] - base[0][2],
        ];
        for at in 0..2 {
            for k in 0..3 {
                let want = base[at][k] + l[k];
                assert!(
                    (got[img * 2 + at][k] - want).abs() < 1e-12,
                    "image {img} atom {at} axis {k}"
                );
            }
        }
    }

    // upstream atom_coords (Bohr-specified diamond), 1e-7 relative.
    const SUPER_COORDS: [[f64; 3]; 16] = [
        [0.0, 0.0, 0.0],
        [1.6850687852746657, 1.6850687852746657, 1.6850687852746657],
        [3.3701375705493315, 3.3701375705493315, 0.0],
        [5.055206355823997, 5.055206355823997, 1.6850687852746657],
        [3.3701375705493315, 0.0, 3.3701375705493315],
        [5.055206355823997, 1.6850687852746657, 5.055206355823997],
        [6.740275141098663, 3.3701375705493315, 3.3701375705493315],
        [8.425343926373328, 5.055206355823997, 5.055206355823997],
        [0.0, 3.3701375705493315, 3.3701375705493315],
        [1.6850687852746657, 5.055206355823997, 5.055206355823997],
        [3.3701375705493315, 6.740275141098663, 3.3701375705493315],
        [5.055206355823997, 8.425343926373328, 5.055206355823997],
        [3.3701375705493315, 3.3701375705493315, 6.740275141098663],
        [5.055206355823997, 5.055206355823997, 8.425343926373328],
        [6.740275141098663, 6.740275141098663, 6.740275141098663],
        [8.425343926373328, 8.425343926373328, 8.425343926373328],
    ];
    for (g, e) in got.iter().zip(SUPER_COORDS.iter()) {
        for k in 0..3 {
            let tol = 1e-7 * e[k].abs().max(1.0);
            assert!((g[k] - e[k]).abs() < tol, "got {got:?}");
        }
    }
}

#[test]
fn super_cell_wrap_around_centres_the_original_cell() {
    let cell = systems::diamond();
    let sc = super_cell(&cell, [2, 2, 2], true).expect("supercell");
    assert_eq!(sc.natm, 16);
    // Same lattice and volume as the un-wrapped supercell.
    let plain = super_cell(&cell, [2, 2, 2], false).expect("supercell");
    assert_eq!(sc.a, plain.a);
    assert_eq!(sc.mesh, plain.mesh);
    // The first image is still the origin; the rest are shifted by -1 cell.
    let base = cell.atom_coords();
    let got = sc.atom_coords();
    assert_eq!(got[0], base[0]);
    assert_eq!(got[1], base[1]);
    // upstream: atom 2 is at -a[2] relative to atom 0.
    for k in 0..3 {
        assert!((got[2][k] - (base[0][k] - cell.a[2][k])).abs() < 1e-12);
    }
}

#[test]
fn super_cell_anisotropic() {
    // upstream: tools.super_cell(diamond, [3,1,2]) -> natm 12, mesh [141,47,94]
    let cell = systems::diamond();
    let sc = super_cell(&cell, [3, 1, 2], false).expect("supercell");
    assert_eq!(sc.natm, 12);
    assert_eq!(sc.mesh, [141, 47, 94]);
    let ratio = sc.vol() / cell.vol();
    assert!((ratio - 6.0).abs() < 1e-9, "vol ratio = {ratio}");
    // ncopy = [1,1,1] reproduces the input geometry.
    let same = super_cell(&cell, [1, 1, 1], false).expect("supercell");
    assert_eq!(same.natm, cell.natm);
    assert_eq!(same.a, cell.a);
    assert_eq!(same.mesh, cell.mesh);
    assert_eq!(same.atom_coords(), cell.atom_coords());
}

#[test]
fn cell_plus_imgs_matches_upstream_including_its_lattice_quirk() {
    // upstream: natm 54, vol UNCHANGED (a is scaled by nimgs, not 2*nimgs+1),
    // mesh [141,141,141].
    let cell = systems::diamond();
    let cpi = cell_plus_imgs(&cell, [1, 1, 1]).expect("cell_plus_imgs");
    assert_eq!(cpi.natm, 54);
    assert_eq!(cpi.mesh, [141, 141, 141]);
    assert_eq!(cpi.a, cell.a, "pbc.py:741 scales a by nimgs, not 2*nimgs+1");
    assert_eq!(cpi.vol(), cell.vol());
    assert_eq!(cpi.nao_nr, cell.nao_nr * 27);

    // The images run -1..=1 on each axis, last index fastest, so the FIRST
    // image is -a[0]-a[1]-a[2] and the last is +a[0]+a[1]+a[2].
    let base = cell.atom_coords();
    let got = cpi.atom_coords();
    for k in 0..3 {
        let corner = -(cell.a[0][k] + cell.a[1][k] + cell.a[2][k]);
        assert!((got[0][k] - (base[0][k] + corner)).abs() < 1e-12);
        assert!((got[52][k] - (base[0][k] - corner)).abs() < 1e-12);
    }
    // nimgs = [0,0,0] is the identity on the geometry.
    let same = cell_plus_imgs(&cell, [0, 0, 0]).expect("cell_plus_imgs");
    assert_eq!(same.natm, cell.natm);
    assert_eq!(same.atom_coords(), cell.atom_coords());
}

#[test]
fn supercell_rejects_space_group_symmetry() {
    // D-PBC-20 — build_lattice_symmetry (pbc.py:784) is Phase 12.
    let mut cell = systems::diamond();
    cell.space_group_symmetry = true;
    let err = super_cell(&cell, [2, 2, 2], false).expect_err("must not silently drop symmetry");
    assert!(
        matches!(err, pyscf_core::PyscfRsError::NotYetImplemented { phase: 12, .. }),
        "got {err:?}"
    );
    assert!(cell_plus_imgs(&cell, [1, 1, 1]).is_err());
}

#[test]
fn supercell_lattice_sum_is_consistent_with_the_primitive_cell() {
    // Tier-1 self-consistency: an atom of the 2x2x2 supercell is an atom of the
    // primitive cell translated by a primitive lattice vector.
    let cell = systems::diamond();
    let sc = super_cell(&cell, [2, 2, 2], false).expect("supercell");
    let inv_a = pyscf_pbc_gto::inv3(&cell.a).expect("invertible");
    let base = cell.atom_coords();
    for (i, r) in sc.atom_coords().iter().enumerate() {
        let b = base[i % 2];
        let d = [r[0] - b[0], r[1] - b[1], r[2] - b[2]];
        // d . inv(a) must be an integer triple.
        let mut t = [0.0_f64; 3];
        for (j, tj) in t.iter_mut().enumerate() {
            *tj = d[0] * inv_a[0][j] + d[1] * inv_a[1][j] + d[2] * inv_a[2][j];
        }
        for (j, tj) in t.iter().enumerate() {
            assert!((tj - tj.round()).abs() < 1e-9, "atom {i} axis {j}: t = {tj}");
        }
    }
}

// ---------------------------------------------------------------------------
// get_monkhorst_pack_size
// ---------------------------------------------------------------------------

/// A gamma-centred Monkhorst-Pack mesh in ABSOLUTE k-points. Plan 09-07 landed
/// `make_kpts`, so this is upstream's own grid rather than a hand-rolled one.
fn mp_kpts(cell: &pyscf_pbc_gto::Cell, nks: [usize; 3]) -> Vec<[f64; 3]> {
    pyscf_pbc_gto::make_kpts_default(cell, nks).expect("make_kpts")
}

#[test]
fn monkhorst_pack_size_matches_upstream() {
    let cell = systems::diamond();
    for nks in [[1, 1, 1], [2, 2, 2], [3, 2, 1], [4, 4, 4], [2, 1, 3]] {
        let kpts = mp_kpts(&cell, nks);
        let got = get_monkhorst_pack_size_default(&cell, &kpts).expect("mp size");
        assert_eq!(got, nks, "make_kpts({nks:?})");
    }
    // pbc.py:590 — the `nkpts < 1/min_tol` assertion: 64 k-points need
    // tol < 1/64, so 0.02 trips it and 0.01 does not.
    let kpts = mp_kpts(&cell, [4, 4, 4]);
    assert!(get_monkhorst_pack_size(&cell, &kpts, 0.02).is_err());
    assert!(get_monkhorst_pack_size(&cell, &kpts, 0.01).is_ok());
}

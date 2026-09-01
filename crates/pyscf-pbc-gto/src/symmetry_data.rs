//! Plain-data lattice symmetry state carried by [`crate::Cell::lattice_symmetry`].
//!
//! # Why this lives here and not `pyscf-pbc-symm`
//!
//! Upstream's `Cell.lattice_symmetry` is a `pyscf.pbc.symm.Symmetry`
//! instance (`cell.py:1294`). `pyscf-pbc-gto` sits BELOW `pyscf-pbc-symm`
//! (D-PBC-25: `pyscf-pbc-symm` depends on `pyscf-pbc-gto`, to see `Cell`), so
//! `Cell` cannot hold a `pyscf_pbc_symm::symmetry::Symmetry` without
//! inverting that dependency edge. This module defines a **plain-data**
//! mirror instead — rotations, fractional translations, Wigner-D matrices,
//! crystal class, group name — with no dependency on anything above this
//! crate. `pyscf_pbc_symm::symmetry::Symmetry` PRODUCES a [`LatticeSymmetry`]
//! (see its `From`/conversion there); this crate never constructs one on its
//! own beyond [`Default`] and the plumbing in [`crate::cell::Cell`].
//!
//! This is the same shape [`crate::pseudo::PseudoData`] already uses for
//! `Cell::pseudo`: `Cell` holds the PARSED pseudopotential data, not the
//! parser that produced it. `LatticeSymmetry` holds the parsed/derived
//! symmetry data, not the `Symmetry` builder.
//!
//! # Not serialised
//!
//! [`crate::dumps_loads`] does NOT round-trip this field (only
//! [`crate::Cell::symmorphic`], the plain bool INPUT that controls how it
//! would be rebuilt, round-trips) — like `rcut`/`mesh`, this is build-time
//! DERIVED state, and unlike them there is no cheap re-estimate: rebuilding
//! it means re-running the symmetry search. A caller that needs it after a
//! `loads` re-runs `pyscf_pbc_symm::symmetry::build_lattice_symmetry`.

/// `pyscf/pbc/symm/geom.py:25` / `space_group.py:27` — `SYMPREC`, duplicated
/// here (same value) because this crate cannot depend on `pyscf-pbc-symm`
/// (D-PBC-25 layering — see the module doc) to reuse its constant.
pub const SYMPREC: f64 = 1e-6;

/// One symmetry operation in the plain-data form `Cell` stores: a rotation
/// in the LATTICE-VECTOR basis (always exactly integer — see
/// `pyscf_pbc_symm::space_group::SPGElement`'s doc comment on why only the
/// Cartesian/`a2r` form is ever non-integer) plus a fractional translation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LatticeSymmetryOp {
    pub rot: [[i32; 3]; 3],
    pub trans: [f64; 3],
}

impl LatticeSymmetryOp {
    /// `symprec`-scaled zero check — mirrors
    /// `pyscf_pbc_symm::space_group::SPGElement::trans_is_zero`, duplicated
    /// here (rather than depending on that crate) because this type must
    /// stay dependency-free per the module doc. `tol` is the caller's
    /// symmetry tolerance (`SYMPREC` in `pyscf-pbc-symm`, `1e-6` by default).
    pub fn trans_is_zero(&self, tol: f64) -> bool {
        self.trans.iter().all(|t| t.abs() < tol)
    }
}

/// Wigner-D matrices for one operation: `dmats[l][row][col]`, `l` from `0` to
/// `l_max` inclusive. Spherical-harmonic order unless the cell is Cartesian
/// (`cell.cart`), matching `pyscf_pbc_symm::symmetry::DmatSet`.
pub type Dmats = Vec<Vec<Vec<f64>>>;

/// Plain-data lattice symmetry info — see the module doc.
#[derive(Debug, Clone, Default)]
pub struct LatticeSymmetry {
    /// Whether the space group was restricted to its symmorphic subgroup.
    pub symmorphic: bool,
    /// Whether any kept operation is the pure inversion.
    pub has_inversion: bool,
    /// The kept operations, in the lattice-vector basis, SORTED — see
    /// `pyscf_pbc_symm::space_group::SPGElement`'s `Ord` impl (17-CONTEXT
    /// §3.6: enumeration order is observable downstream).
    pub ops: Vec<LatticeSymmetryOp>,
    /// `dmats[iop]` — one [`Dmats`] set per entry of [`Self::ops`], SAME order.
    pub dmats: Vec<Dmats>,
    /// Maximum angular momentum considered in `dmats`.
    pub l_max: usize,
    /// The crystallographic point-group symbol (Hermann-Mauguin), e.g. `"m-3m"`.
    pub point_group_symbol: String,
}

/// Core of `pyscf/pbc/symm/symmetry.py:96-131`'s `check_mesh_symmetry`,
/// generic over a plain `(is_zero_translation, translation)` list rather
/// than `pyscf_pbc_symm::space_group::SPGElement` — this crate cannot name
/// that type (see the module doc). [`crate::cell::Cell::symmetrize_mesh`]
/// and `pyscf_pbc_symm::symmetry::check_mesh_symmetry` both delegate to this
/// SAME function, so there is exactly one implementation of the
/// mesh-growing algorithm (17-CONTEXT §3.2's "one implementation" discipline
/// applied to the mesh side of the port).
///
/// Returns `(rm_list, mesh1)`: the indices of `ops` whose fractional
/// translation is INCOMPATIBLE with `mesh` (upstream: `(abs(tmp -
/// tmp.round())/mesh > tol).any()`), and the smallest mesh (component-wise,
/// starting from `mesh`) that is compatible with EVERY non-zero translation
/// in `ops` (computed only when `rm_list` is non-empty; otherwise `mesh1 ==
/// mesh`).
pub fn check_mesh_symmetry_core(
    ops: &[(bool, [f64; 3])],
    mesh: [usize; 3],
    tol: f64,
) -> (Vec<usize>, [usize; 3]) {
    let mut ft: Vec<[f64; 3]> = Vec::new();
    let mut rm_list: Vec<usize> = Vec::new();
    for (i, (is_zero, trans)) in ops.iter().enumerate() {
        if *is_zero {
            continue;
        }
        ft.push(*trans);
        let mut bad = false;
        for x in 0..3 {
            let tmp = trans[x] * mesh[x] as f64;
            if ((tmp - tmp.round()) / mesh[x] as f64).abs() > tol {
                bad = true;
            }
        }
        if bad {
            rm_list.push(i);
        }
    }

    let mesh1 = if rm_list.is_empty() {
        mesh
    } else {
        let mut mesh1 = mesh;
        for x in 0..3 {
            loop {
                let bad = ft.iter().any(|t| {
                    let tmp = t[x] * mesh1[x] as f64;
                    ((tmp - tmp.round()) / mesh1[x] as f64).abs() > tol
                });
                if bad {
                    mesh1[x] += 1;
                } else {
                    break;
                }
            }
        }
        mesh1
    };
    (rm_list, mesh1)
}

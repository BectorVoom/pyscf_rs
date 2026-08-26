//! `super_cell` and `cell_plus_imgs` — port of `pyscf/pbc/tools/pbc.py:678-786`.
//!
//! The translation-vector arithmetic lives in [`pyscf_pbc_tools::supercell`];
//! what is here is the `Cell` assembly, i.e. upstream's `_build_supcell_`
//! (`pbc.py:747-786`).
//!
//! # `_build_supcell_` vs. [`pyscf_gto::build_from`]
//!
//! Upstream deliberately does NOT call `supcell.build()`: it splices `_atm`,
//! `_bas` and `_env` by hand so that `build()` cannot re-normalize the basis
//! contraction coefficients a second time. This port instead rebuilds the
//! molecular half through [`pyscf_gto::build_from`] from the cell's `_atom` and
//! `_basis` — the PARSED, pre-normalization inputs — so normalization still runs
//! exactly once, and the supercell's `_env` comes out of the same single `Mole`
//! build path as every other molecule in the workspace (D-PBC-01). This is the
//! same trick [`pyscf_gto::loads`] uses to reproduce a `Mole` byte-for-byte.
//!
//! # What is copied and what is recomputed
//!
//! Upstream builds on `cell.copy(deep=False)`, so `precision`, `dimension`,
//! `low_dim_ft_type`, `ke_cutoff`, `pseudo` and — importantly — `_rcut` are
//! carried over UNCHANGED; only `a`, `mesh`, `atom` and `enuc` are replaced.
//! This port does the same. `rcut` is therefore the PRIMITIVE cell's radius, not
//! a re-estimate for the supercell.
//!
//! # Not ported
//!
//! * `supcell.magmom` (`pbc.py:717-720`) — this workspace's `Mole` has no
//!   `magmom` field yet.
//! * the `space_group_symmetry` branch (`pbc.py:784-785`) — lattice symmetry is
//!   Phase 12, so that input returns `NotYetImplemented` (D-PBC-20) rather than
//!   a supercell silently missing its symmetry object.

use crate::cell::Cell;
use pyscf_core::{Mole, PyscfRsError};
use pyscf_gto::{AtomInput, BasisInput, EcpInput, MoleBuildArgs};
use pyscf_pbc_tools::supercell as core;
use std::collections::HashMap;

/// Create an `ncopy[0] x ncopy[1] x ncopy[2]` supercell of `cell`.
/// Ports `super_cell` (`pbc.py:678-727`).
///
/// Images run in the `+` direction only — contrast [`cell_plus_imgs`], which
/// images in both directions. `wrap_around` centres the original cell on the
/// supercell, the same convention as `cell.make_kpts(wrap_around=True)`.
///
/// `a_super[i] = ncopy[i] * a[i]` and `mesh_super = ncopy * mesh`.
///
/// # Errors
/// * [`PyscfRsError::NotYetImplemented`] when `cell.space_group_symmetry` is set
///   (Phase 12).
/// * Anything [`pyscf_gto::build_from`] raises on the enlarged molecule.
pub fn super_cell(cell: &Cell, ncopy: [usize; 3], wrap_around: bool) -> Result<Cell, PyscfRsError> {
    let a = cell.lattice_vectors();
    let ls = core::super_cell_translations(&a, &ncopy, wrap_around);
    let mesh = cell.try_mesh()?;
    // pbc.py:715-716
    let a_super = core::scale_lattice(&a, &ncopy);
    let mesh_super = [ncopy[0] * mesh[0], ncopy[1] * mesh[1], ncopy[2] * mesh[2]];
    build_supcell(cell, &ls, a_super, mesh_super)
}

/// Create a supercell holding `nimgs[i]` images in EACH of the `+/-`
/// directions, as in `get_lattice_Ls`. Ports `cell_plus_imgs`
/// (`pbc.py:729-745`).
///
/// `mesh_super[i] = (2 * nimgs[i] + 1) * mesh[i]`.
///
/// **Upstream quirk, ported verbatim (RULE 2):** the lattice is scaled by
/// `nimgs[i]`, NOT by `2 * nimgs[i] + 1` (`pbc.py:741`). For `nimgs = [1,1,1]`
/// that leaves `a` — and hence `vol` — equal to the primitive cell's while the
/// atom count grows to `27 * natm`. Do not "fix" it here; a divergence from
/// upstream would show up as an energy difference, not as a compile error.
///
/// # Errors
/// As [`super_cell`].
pub fn cell_plus_imgs(cell: &Cell, nimgs: [usize; 3]) -> Result<Cell, PyscfRsError> {
    let a = cell.lattice_vectors();
    let ls = core::cell_plus_imgs_translations(&a, &nimgs);
    let mesh = cell.try_mesh()?;
    // pbc.py:741-744
    let a_super = core::scale_lattice(&a, &nimgs);
    let mesh_super = [
        (nimgs[0] * 2 + 1) * mesh[0],
        (nimgs[1] * 2 + 1) * mesh[1],
        (nimgs[2] * 2 + 1) * mesh[2],
    ];
    build_supcell(cell, &ls, a_super, mesh_super)
}

/// `_build_supcell_` (`pbc.py:747-786`) — replicate the atoms over `ls` and
/// rebuild the molecular half. See the module docs for the deviations.
fn build_supcell(
    cell: &Cell,
    ls: &[[f64; 3]],
    a_super: [[f64; 3]; 3],
    mesh_super: [usize; 3],
) -> Result<Cell, PyscfRsError> {
    // pbc.py:784-785 — build_lattice_symmetry is Phase 12 (D-PBC-20).
    if cell.space_group_symmetry {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 12,
            what: "super_cell / cell_plus_imgs with space_group_symmetry \
                   (pbc/tools/pbc.py:784 build_lattice_symmetry)",
        });
    }

    // pbc.py:755-759 — symbs = [atom[0] for atom in cell._atom] * nimgs,
    // coords = Ls.reshape(-1,1,3) + cell.atom_coords(), both image-major.
    let coords = core::image_atom_coords(ls, &cell.mol.atom_coords());
    let natm = cell.mol.natm;
    let atoms: Vec<(String, [f64; 3])> = coords
        .into_iter()
        .enumerate()
        .map(|(i, xyz)| (cell.mol._atom[i % natm].0.clone(), xyz))
        .collect();

    // The PARSED per-element basis / ECP, uppercase-keyed, exactly as
    // `pyscf_gto::loads` re-feeds them.
    let basis = if cell.mol._basis.is_empty() {
        BasisInput::Name(String::new())
    } else {
        let per: HashMap<String, BasisInput> = cell
            .mol
            ._basis
            .iter()
            .map(|(sym, pb)| (sym.clone(), BasisInput::Parsed(pb.clone())))
            .collect();
        BasisInput::PerElement(per)
    };
    let ecp = if cell.mol._ecp.is_empty() {
        EcpInput::None
    } else {
        let per: HashMap<String, EcpInput> = cell
            .mol
            ._ecp
            .iter()
            .map(|(sym, pe)| (sym.clone(), EcpInput::Parsed(pe.clone())))
            .collect();
        EcpInput::PerElement(per)
    };

    let args = MoleBuildArgs {
        atom: AtomInput::Tuples(atoms),
        basis,
        ecp,
        charge: cell.mol.charge,
        spin: cell.mol.spin,
        cart: cell.mol.cart,
        // pbc.py:761 — supcell.unit = 'B'; coords are already Bohr.
        unit: pyscf_core::Unit::Bohr,
        verbose: cell.mol.verbose,
        max_memory: cell.mol.max_memory,
        output: cell.mol.output.clone(),
        origin: [0.0; 3],
        axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    };
    let mut mol = Mole::default();
    pyscf_gto::build_from(&mut mol, args)?;

    Ok(Cell {
        mol,
        a: a_super,
        mesh: mesh_super,
        // cell.copy(deep=False) — everything below rides along unchanged.
        dimension: cell.dimension,
        low_dim_ft_type: cell.low_dim_ft_type,
        precision: cell.precision,
        ke_cutoff: cell.ke_cutoff,
        rcut: cell.rcut,
        // pbc.py:762 — supcell.enuc = None; the Ewald parameters go with it.
        ew_eta: None,
        ew_cut: None,
        pseudo: cell.pseudo.clone(),
        pseudo_name: cell.pseudo_name.clone(),
        exp_to_discard: cell.exp_to_discard,
        // The replicated coordinates are Cartesian Bohr, whatever the input was.
        fractional: false,
        use_particle_mesh_ewald: cell.use_particle_mesh_ewald,
        space_group_symmetry: cell.space_group_symmetry,
        use_loose_rcut: cell.use_loose_rcut,
        _built: true,
        _rcut_from_build: cell._rcut_from_build,
        // `mesh` was set explicitly above, so it is no longer build-estimated.
        _mesh_from_build: false,
    })
}

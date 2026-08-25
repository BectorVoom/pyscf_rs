//! `Cell`-taking wrappers over the lattice-sum core in
//! [`pyscf_pbc_tools::lattice`].
//!
//! Ports the `cell`-dependent halves of `pyscf/pbc/tools/pbc.py:587-676` and
//! `:836-840`: the `dimension` / `rcut` defaulting of `get_lattice_Ls`, the
//! reference image list of `check_lattice_sum_range`, and the
//! `cell.get_scaled_kpts` step of `get_monkhorst_pack_size`.
//!
//! The loop bodies themselves live one crate down — the DAG runs
//! `pyscf-pbc-tools -> pyscf-pbc-gto`, so `pyscf-pbc-tools` cannot name [`Cell`].

use crate::cell::Cell;
use crate::types::LowDimFtType;
use pyscf_core::PyscfRsError;
use pyscf_pbc_tools::lattice as core;

/// The resolved `dimension` of a lattice sum over `cell` — `pbc.py:609-614`.
///
/// > For atoms near the boundary of the cell, it is necessary (even in
/// > low-dimensional systems) to include lattice translations in all 3
/// > dimensions.
pub fn lattice_sum_dimension(cell: &Cell) -> usize {
    if cell.dimension < 2 || cell.low_dim_ft_type == LowDimFtType::InfVacuum {
        cell.dimension as usize
    } else {
        3
    }
}

/// The (Cartesian, unitful) lattice translation vectors for nearby images.
/// Ports `get_lattice_Ls` (`pbc.py:601-661`).
///
/// `rcut = None` falls back to [`Cell::try_rcut`] (upstream `cell.rcut`);
/// `dimension = None` falls back to [`lattice_sum_dimension`]. `discard` drops
/// images that cannot reach any atom pair — upstream defaults it to `true`, see
/// [`get_lattice_ls_default`].
///
/// Upstream's `nimgs` keyword is not reproduced: `pbc.py:601` accepts it and the
/// body never reads it.
///
/// `Ls[0]` is NOT the origin — the list is the raw `cartesian_prod` starting at
/// `-bounds`, and the origin sits in the middle.
///
/// # Errors
/// [`pyscf_core::CoreError::InvalidMolecule`] if the lattice is singular
/// (through `get_scaled_atom_coords`) or `rcut` has to be estimated and cannot.
pub fn get_lattice_ls(
    cell: &Cell,
    rcut: Option<f64>,
    dimension: Option<usize>,
    discard: bool,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let dimension = dimension.unwrap_or_else(|| lattice_sum_dimension(cell));
    let rcut = match rcut {
        Some(r) => r,
        None => cell.try_rcut()?,
    };
    // pbc.py:619 — `cell.natm == 0` short-circuits before any lattice algebra,
    // so a singular lattice must not be reported for an empty cell.
    if dimension == 0 || rcut <= 0.0 || cell.mol.natm == 0 {
        return Ok(vec![[0.0; 3]]);
    }
    let a = cell.lattice_vectors();
    let scaled = cell.get_scaled_atom_coords()?;
    let coords = cell.mol.atom_coords();
    Ok(core::get_lattice_ls(
        &a, &scaled, &coords, rcut, dimension, discard,
    ))
}

/// [`get_lattice_ls`] with upstream's defaults (`rcut = cell.rcut`,
/// `dimension` resolved, `discard = True`).
///
/// # Errors
/// As [`get_lattice_ls`].
pub fn get_lattice_ls_default(cell: &Cell) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    get_lattice_ls(cell, None, None, true)
}

/// Evaluate whether the lattice summation range `ls` is sufficient.
/// Ports `check_lattice_sum_range` (`pbc.py:663-676`).
///
/// Returns the minimum distance between an atom of the primary unit cell and an
/// atom of a lattice image *not* covered by `ls` — `f64::INFINITY` when `ls`
/// already covers the whole reference range (upstream raises on an empty
/// `min()` there).
///
/// # Errors
/// As [`get_lattice_ls`].
pub fn check_lattice_sum_range(cell: &Cell, ls: &[[f64; 3]]) -> Result<f64, PyscfRsError> {
    let ls_full = get_lattice_ls(cell, Some(cell.try_rcut()? * 1.5), None, false)?;
    Ok(core::check_lattice_sum_range(
        &ls_full,
        ls,
        &cell.mol.atom_coords(),
    ))
}

/// The Monkhorst-Pack mesh size behind a k-point list.
/// Ports `get_monkhorst_pack_size` (`pbc.py:587-599`).
///
/// `kpts` are ABSOLUTE k-points in 1/Bohr; `tol` is upstream's `min_tol`
/// (default `1e-5`, see [`get_monkhorst_pack_size_default`]).
///
/// # Errors
/// [`pyscf_core::CoreError::InvalidMolecule`] when `nkpts >= 1/tol` (upstream's
/// `assert`).
pub fn get_monkhorst_pack_size(
    cell: &Cell,
    kpts: &[[f64; 3]],
    tol: f64,
) -> Result<[usize; 3], PyscfRsError> {
    // pbc.py:594 — skpts = cell.get_scaled_kpts(kpts); the `nkpts == 1` branch
    // never reaches it, and neither do we (the core returns [1,1,1] first).
    if kpts.len() == 1 {
        return core::monkhorst_pack_size_from_scaled(&[[0.0; 3]], tol);
    }
    let skpts = cell.get_scaled_kpts(kpts);
    core::monkhorst_pack_size_from_scaled(&skpts, tol)
}

/// [`get_monkhorst_pack_size`] with upstream's default `tol = 1e-5`.
///
/// # Errors
/// As [`get_monkhorst_pack_size`].
pub fn get_monkhorst_pack_size_default(
    cell: &Cell,
    kpts: &[[f64; 3]],
) -> Result<[usize; 3], PyscfRsError> {
    get_monkhorst_pack_size(cell, kpts, 1e-5)
}

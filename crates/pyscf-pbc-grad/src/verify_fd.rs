//! Periodic finite-difference gates.
//!
//! `disp` here is the **half step**: the cells are evaluated at `±disp` and
//! divided by their realised separation.  Upstream's PBC tests call their full
//! separation `disp`, therefore an upstream `1e-5` corresponds to `5e-6` at
//! this API.  Dividing by the realised separation avoids a coordinate-rounding
//! error for free; it does not tighten the cancellation-limited physical gate.

use crate::error::PbcGradError;
use pyscf_core::{
    PyscfRsError,
    raw_layout::{ATM_SLOTS, PTR_COORD},
};
use pyscf_pbc_gto::Cell;

/// Result of an atom-coordinate finite-difference comparison.
#[derive(Debug, Clone)]
pub struct FdReport {
    /// `max |fd - analytical|` in Ha/Bohr.
    pub max_abs_diff: f64,
    /// The central-difference gradient in `(natm, 3)` order.
    pub fd_grad: Vec<[f64; 3]>,
    /// Whether the maximum residual is no greater than the caller's tolerance.
    pub passed: bool,
}

/// Two strain-displaced cells and their k-points at fixed fractional
/// coordinates.  `minus`/`plus` use `∓disp/2` in the selected tensor entry.
#[derive(Debug, Clone)]
pub struct StrainCells {
    pub minus: Cell,
    pub plus: Cell,
    pub kpts_minus: Vec<[f64; 3]>,
    pub kpts_plus: Vec<[f64; 3]>,
}

fn validate_disp(disp: f64) -> Result<(), PyscfRsError> {
    if !disp.is_finite() || disp <= 0.0 {
        return Err(PbcGradError::InvalidDisplacement { disp }.into());
    }
    Ok(())
}

fn invalid(message: &str) -> PyscfRsError {
    pyscf_core::CoreError::InvalidMolecule(message.into()).into()
}

fn with_coords(cell: &Cell, coords: &[[f64; 3]]) -> Result<Cell, PyscfRsError> {
    if coords.len() != cell.natm {
        return Err(PbcGradError::ShapeMismatch {
            expected: cell.natm,
            got: coords.len(),
        }
        .into());
    }
    let mut out = cell.clone();
    if !out._built
        || !out.mol._built
        || out.mol._atom.len() != out.natm
        || out.mol._atm.len() < out.natm * ATM_SLOTS
    {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            "periodic finite difference requires a built Cell".into(),
        )));
    }
    for (ia, &coord) in coords.iter().enumerate() {
        if coord.iter().any(|x| !x.is_finite()) {
            return Err(invalid("non-finite displaced coordinate"));
        }
        out.mol._atom[ia].1 = coord;
        let ptr = out.mol._atm[ia * ATM_SLOTS + PTR_COORD] as usize;
        let end = ptr
            .checked_add(3)
            .ok_or_else(|| invalid("coordinate pointer overflow"))?;
        let env = out.mol._env.get_mut(ptr..end).ok_or_else(|| {
            PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(format!(
                "periodic finite difference: atom {ia} coordinate pointer {ptr} is outside _env"
            )))
        })?;
        env.copy_from_slice(&coord);
    }
    out.mol.basis_set = Some(pyscf_gto::projection::build_cintx_basis_set(
        &out.mol._atom,
        &out.mol._basis,
        out.mol.cart,
    )?);
    out.symm_orb = None;
    out.irrep_id = None;
    out.lattice_symmetry = None;
    if out.space_group_symmetry {
        pyscf_pbc_symm::symmetry::build_lattice_symmetry(&mut out, true)
            .map_err(|e| invalid(&format!("displaced-cell symmetry: {e}")))?;
    }
    Ok(out)
}

/// Central-difference a periodic energy closure over complete [`Cell`] values.
///
/// Unlike the molecular harness, the closure receives a cell, preserving the
/// lattice, mesh, pseudopotential and all periodic build settings.
pub fn verify_fd<F>(
    cell: &Cell,
    analytical: &[[f64; 3]],
    energy: F,
    disp: f64,
    tol: f64,
) -> Result<FdReport, PyscfRsError>
where
    F: Fn(&Cell) -> Result<f64, PyscfRsError>,
{
    validate_disp(disp)?;
    if !tol.is_finite() || tol < 0.0 || analytical.iter().flatten().any(|v| !v.is_finite()) {
        return Err(invalid(
            "finite-difference tolerance and analytical gradient must be finite; tolerance must be nonnegative",
        ));
    }
    let coords = cell.atom_coords();
    if analytical.len() != coords.len() {
        return Err(PbcGradError::ShapeMismatch {
            expected: coords.len(),
            got: analytical.len(),
        }
        .into());
    }

    let mut fd_grad = Vec::with_capacity(coords.len());
    let mut abs_diffs = Vec::with_capacity(coords.len() * 3);
    for ia in 0..coords.len() {
        let mut row = [0.0; 3];
        for c in 0..3 {
            let mut plus_coords = coords.clone();
            let mut minus_coords = coords.clone();
            plus_coords[ia][c] += disp;
            minus_coords[ia][c] -= disp;
            let plus = with_coords(cell, &plus_coords)?;
            let minus = with_coords(cell, &minus_coords)?;
            let realised_step = plus.atom_coord(ia)[c] - minus.atom_coord(ia)[c];
            validate_disp(realised_step)?;
            let ep = energy(&plus)?;
            let em = energy(&minus)?;
            if !ep.is_finite() || !em.is_finite() {
                return Err(invalid("non-finite energy in finite difference"));
            }
            let numerator = pyscf_algebra::oracle_sum(&[ep, -em]);
            let fd = numerator / realised_step;
            if !fd.is_finite() {
                return Err(invalid("non-finite finite-difference gradient"));
            }
            row[c] = fd;
            abs_diffs.push(pyscf_algebra::oracle_sum(&[fd, -analytical[ia][c]]).abs());
        }
        fd_grad.push(row);
    }
    let max_abs_diff = abs_diffs.into_iter().fold(0.0, f64::max);
    Ok(FdReport {
        max_abs_diff,
        fd_grad,
        passed: max_abs_diff <= tol,
    })
}

fn right_mul_transpose(rows: &[[f64; 3]; 3], strain: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            pyscf_algebra::oracle_sum(&[
                rows[i][0] * strain[j][0],
                rows[i][1] * strain[j][1],
                rows[i][2] * strain[j][2],
            ])
        })
    })
}

fn strain_cell(
    cell: &Cell,
    x: usize,
    y: usize,
    signed_half_disp: f64,
) -> Result<Cell, PyscfRsError> {
    let mut strain = [[0.0; 3]; 3];
    for (i, row) in strain.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    strain[x][y] += signed_half_disp;
    let mut out = cell.clone();
    out.a = right_mul_transpose(&cell.a, &strain);
    let coords: Vec<[f64; 3]> = cell
        .atom_coords()
        .iter()
        .map(|coord| {
            std::array::from_fn(|j| {
                pyscf_algebra::oracle_sum(&[
                    coord[0] * strain[j][0],
                    coord[1] * strain[j][1],
                    coord[2] * strain[j][2],
                ])
            })
        })
        .collect();
    // `with_coords` retains the original mesh exactly, even when the source
    // cell carries space-group symmetry.  Rebuilding and allowing a mesh
    // enlargement would differentiate the mesh, not the energy functional.
    with_coords(&out, &coords)
}

/// Build the two cells used for a strain finite difference.
///
/// Atom positions and lattice vectors are both strained as `r . Eᵀ` and
/// `a . Eᵀ`.  Returned Cartesian k-points are recomputed from the unstrained
/// fractional coordinates, so the derivative is at fixed reduced k-point.
pub fn finite_diff_cells(
    cell: &Cell,
    kpts: &[[f64; 3]],
    x: usize,
    y: usize,
    disp: f64,
) -> Result<StrainCells, PyscfRsError> {
    validate_disp(disp)?;
    if x >= 3 || y >= 3 {
        return Err(PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(
            format!("strain index ({x}, {y}) is outside a 3x3 tensor"),
        )));
    }
    let scaled_kpts = cell.get_scaled_kpts(kpts);
    let minus = strain_cell(cell, x, y, -0.5 * disp)?;
    let plus = strain_cell(cell, x, y, 0.5 * disp)?;
    let kpts_minus = minus.get_abs_kpts(&scaled_kpts)?;
    let kpts_plus = plus.get_abs_kpts(&scaled_kpts)?;
    Ok(StrainCells {
        minus,
        plus,
        kpts_minus,
        kpts_plus,
    })
}

//! Shell-pair neighbor list and lattice-sum screening (D-PBC-08).
//!
//! Port of `pyscf/pbc/gto/neighborlist.py` +
//! `pyscf/lib/pbc/neighbor_list.c:80-128` (`build_neighbor_list`).
//!
//! # What it is
//!
//! A periodic 1-electron matrix element is a sum over lattice images
//!
//! ```text
//! S^k_{ij} = Σ_L exp(i k·L) <φ_i(r − R_i) | φ_j(r − R_j − L)>
//! ```
//!
//! Almost every `(ish, jsh, L)` triple in that sum is numerically zero: two
//! Gaussians whose centres are further apart than the sum of their decay radii
//! overlap below `cell.precision`. The neighbor list is the surviving set, and
//! without it the lattice sum is `O(nimgs · nbas²)` cintx calls — for diamond /
//! `gth-szv` that is ~4x the work for no accuracy (see `tests/neighborlist.rs`).
//!
//! # The criterion (verbatim, `neighbor_list.c:113-120`)
//!
//! ```text
//! rij = R_jsh + L − R_ish;   keep iff |rij| < rcut[ish] + rcut[jsh]
//! ```
//!
//! Note the STRICT `<` and that the radii are the per-shell
//! [`crate::cutoff::rcut_by_shells`] values, NOT the cell-wide `cell.rcut`.

use crate::cell::Cell;
use pyscf_core::{CoreError, Mole, PyscfRsError};

/// One surviving `(bra shell, ket shell, lattice image)` triple.
///
/// `l_idx` indexes the `Ls` slice the list was built against — the caller uses
/// it both to fetch `L` itself and to look up the pre-computed Bloch phase
/// `exp(i k·L)` for each k (see [`crate::pbc_intor`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NeighborPair {
    /// Bra shell index in cell-0.
    pub ish: usize,
    /// Ket shell index in the (unshifted) ket cell.
    pub jsh: usize,
    /// Index into the `Ls` list this neighbor list was built against.
    pub l_idx: usize,
}

/// The neighbor list, stored image-major so the `pbc_intor` driver can walk one
/// lattice image at a time (each image is one `_env` shift upstream, one
/// combined-basis shell block here).
#[derive(Debug, Clone, Default)]
pub struct NeighborList {
    /// Number of bra shells the list covers.
    pub nish: usize,
    /// Number of ket shells the list covers.
    pub njsh: usize,
    /// Number of lattice images the list was built against.
    pub nimgs: usize,
    /// `pairs[l_idx]` — the `(ish, jsh)` pairs that survive for image `l_idx`,
    /// in ascending `(ish, jsh)` order.
    pub per_image: Vec<Vec<(usize, usize)>>,
}

impl NeighborList {
    /// Total number of surviving triples.
    pub fn len(&self) -> usize {
        self.per_image.iter().map(Vec::len).sum()
    }

    /// `true` when no triple survives (an empty lattice sum).
    pub fn is_empty(&self) -> bool {
        self.per_image.iter().all(Vec::is_empty)
    }

    /// The number of `(ish, jsh, L)` triples a *dense* lattice sum would need —
    /// the denominator of the screening ratio.
    pub fn dense_len(&self) -> usize {
        self.nish * self.njsh * self.nimgs
    }

    /// Flatten to the `(ish, jsh, l_idx)` triples, image-major.
    pub fn triples(&self) -> Vec<NeighborPair> {
        let mut out = Vec::with_capacity(self.len());
        for (l_idx, pairs) in self.per_image.iter().enumerate() {
            for &(ish, jsh) in pairs {
                out.push(NeighborPair { ish, jsh, l_idx });
            }
        }
        out
    }

    /// `true` when `(ish, jsh)` survives for image `l_idx`.
    pub fn contains(&self, ish: usize, jsh: usize, l_idx: usize) -> bool {
        self.per_image
            .get(l_idx)
            .is_some_and(|p| p.binary_search(&(ish, jsh)).is_ok())
    }
}

/// Per-shell atom centres, in Bohr — `ish_env + ish_atm[ATOM_OF*ATM_SLOTS + PTR_COORD]`
/// in the C driver, which is just "the coordinate of the atom this shell sits on".
fn shell_centres(mol: &Mole) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    use pyscf_core::raw_layout::{ATOM_OF, BAS_SLOTS};
    let coords = mol.atom_coords();
    if mol._bas.len() < mol.nbas * BAS_SLOTS {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "neighborlist: _bas is not projected — build the cell first".into(),
        )));
    }
    (0..mol.nbas)
        .map(|ib| {
            let ia = mol._bas[ib * BAS_SLOTS + ATOM_OF] as usize;
            coords.get(ia).copied().ok_or_else(|| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "neighborlist: shell {ib} references atom {ia} of {}",
                    coords.len()
                )))
            })
        })
        .collect()
}

/// Build the shell-pair neighbor list for a bra/ket cell pair.
///
/// Ports `build_neighbor_list` (`pyscf/lib/pbc/neighbor_list.c:80-128`) and the
/// argument defaulting of `build_neighbor_list_for_shlpairs`
/// (`neighborlist.py:48-129`).
///
/// * `ls` — the lattice translations, as returned by
///   [`crate::lattice::get_lattice_ls`]. Every `l_idx` in the result indexes it.
/// * `ish_rcut` / `jsh_rcut` — per-shell radii; `None` falls back to
///   `cell.rcut_by_shells(precision)`.
/// * `hermi == 1` restricts the list to `jsh >= ish` (upstream's
///   upper-triangle-only mode). It is only legal when bra and ket are the same
///   cell; passing it with distinct cells is downgraded to `hermi = 0` with a
///   warning, exactly as `neighborlist.py:79-83` does.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] when either cell is unbuilt, or when the
/// supplied radii do not have one entry per shell.
pub fn build_neighbor_list(
    cell: &Cell,
    cell1: Option<&Cell>,
    ls: &[[f64; 3]],
    ish_rcut: Option<&[f64]>,
    jsh_rcut: Option<&[f64]>,
    hermi: i32,
    precision: Option<f64>,
) -> Result<NeighborList, PyscfRsError> {
    let ket = cell1.unwrap_or(cell);
    let same_cell = cell1.is_none() || std::ptr::eq(cell, ket);

    let mut hermi = hermi;
    if hermi == 1 && !same_cell {
        tracing::warn!("build_neighbor_list: hermi reset to 0 because cell and cell1 differ");
        hermi = 0;
    }

    let owned_i;
    let ish_rcut: &[f64] = match ish_rcut {
        Some(r) => r,
        None => {
            owned_i = cell.rcut_by_shells(precision);
            &owned_i
        }
    };
    let owned_j;
    let jsh_rcut: &[f64] = match jsh_rcut {
        Some(r) => r,
        None if same_cell => ish_rcut,
        None => {
            owned_j = ket.rcut_by_shells(precision);
            &owned_j
        }
    };

    let nish = cell.mol.nbas;
    let njsh = ket.mol.nbas;
    if ish_rcut.len() != nish || jsh_rcut.len() != njsh {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "build_neighbor_list: rcut length mismatch (ish {} vs nbas {nish}, \
             jsh {} vs nbas {njsh})",
            ish_rcut.len(),
            jsh_rcut.len(),
        ))));
    }

    let ri = shell_centres(&cell.mol)?;
    let rj = shell_centres(&ket.mol)?;

    let mut per_image: Vec<Vec<(usize, usize)>> = vec![Vec::new(); ls.len()];
    for (l_idx, l) in ls.iter().enumerate() {
        let bucket = &mut per_image[l_idx];
        for ish in 0..nish {
            let jstart = if hermi == 1 { ish } else { 0 };
            for jsh in jstart..njsh {
                let rmax = ish_rcut[ish] + jsh_rcut[jsh];
                let dx = rj[jsh][0] + l[0] - ri[ish][0];
                let dy = rj[jsh][1] + l[1] - ri[ish][1];
                let dz = rj[jsh][2] + l[2] - ri[ish][2];
                // neighbor_list.c:116-120 — STRICT `<`, on the un-squared norm.
                if (dx * dx + dy * dy + dz * dz).sqrt() < rmax {
                    bucket.push((ish, jsh));
                }
            }
        }
    }

    Ok(NeighborList {
        nish,
        njsh,
        nimgs: ls.len(),
        per_image,
    })
}

/// [`build_neighbor_list`] with upstream's defaults — same cell for bra and
/// ket, radii from `cell.rcut_by_shells(cell.precision)`, `hermi = 0`.
///
/// # Errors
/// As [`build_neighbor_list`].
pub fn build_neighbor_list_for_shlpairs(
    cell: &Cell,
    ls: &[[f64; 3]],
) -> Result<NeighborList, PyscfRsError> {
    build_neighbor_list(cell, None, ls, None, None, 0, None)
}

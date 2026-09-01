//! `pack` / `unpack` / `dumps` / `loads` for [`Cell`] — port of
//! `pyscf/pbc/gto/cell.py:65-155`.
//!
//! Same contract as the molecular [`pyscf_gto::dumps`] (GTO-09): **semantic**
//! round-trip, not byte-identity with upstream's JSON. Reading our own output
//! reproduces the same internal arrays; cross-language chkfile interop is a
//! separate workstream.
//!
//! Structure mirrors upstream exactly: `pack` layers the periodic fields on top
//! of `mole.pack` (`cell.py:69-79`), and `dumps` serialises the packed state.
//! Here the molecular layer is [`pyscf_gto::dumps`]'s JSON string, embedded
//! verbatim, so the two crates never disagree about what a `Mole` is.

use crate::cell::{Cell, MESH_UNSET, RCUT_UNSET};
use crate::types::{ALattice, LowDimFtType};
use pyscf_core::{CoreError, PyscfRsError};
use serde::{Deserialize, Serialize};

/// The packed input state of a [`Cell`] — upstream's `cldic`
/// (`cell.py:65-79`). The molecular half rides as the JSON string
/// [`pyscf_gto::dumps`] produces, keeping `Mole` serialisation single-sourced.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CellPack {
    /// `mole.pack(cell)` equivalent — [`pyscf_gto::dumps`] output.
    pub mole_json: String,
    /// Lattice vectors in BOHR, one per row (upstream `cldic['a']`).
    pub a: [[f64; 3]; 3],
    /// Upstream `cldic['fractional']`.
    pub fractional: bool,
    /// Upstream `cldic['precision']`.
    pub precision: f64,
    /// Upstream `cldic['ke_cutoff']`.
    pub ke_cutoff: Option<f64>,
    /// Upstream `cldic['exp_to_discard']`.
    pub exp_to_discard: Option<f64>,
    /// Upstream `cldic['_mesh']`.
    pub mesh: [usize; 3],
    /// Upstream `cldic['_rcut']`.
    pub rcut: f64,
    /// Upstream `cldic['dimension']`.
    pub dimension: u8,
    /// Upstream `cldic['low_dim_ft_type']`.
    pub low_dim_ft_type: LowDimFtType,
    /// The pseudopotential NAME (upstream stores the parsed dict; [`unpack`]
    /// replaces this with the parsed form — D-PBC-11).
    pub pseudo_name: Option<String>,
    /// Not in upstream's `pack` (it is a class attribute there), but part of
    /// this port's `Cell` state, so it round-trips.
    pub use_particle_mesh_ewald: bool,
    /// See [`CellPack::use_particle_mesh_ewald`].
    pub space_group_symmetry: bool,
    /// Upstream `cldic['symmorphic']` (`cell.py:1289-1293`). Previously
    /// missing entirely — `space_group_symmetry` round-tripped without its
    /// partner, silently dropping it on `loads`. `Cell::lattice_symmetry`
    /// itself is NOT serialised — see its doc comment.
    #[serde(default)]
    pub symmorphic: bool,
    /// See [`CellPack::use_particle_mesh_ewald`].
    #[serde(default)]
    pub use_loose_rcut: bool,
    /// Whether `rcut` was left to `build` to estimate.
    pub rcut_from_build: bool,
    /// Whether `mesh` was left to `build` to estimate.
    pub mesh_from_build: bool,
}

/// Pack the input args of a [`Cell`] into a serialisable struct.
/// Ports `cell.py:65-79`.
///
/// # Errors
/// Propagates [`pyscf_gto::dumps`] failures on the molecular half.
pub fn pack(cell: &Cell) -> Result<CellPack, PyscfRsError> {
    Ok(CellPack {
        mole_json: pyscf_gto::dumps(&cell.mol)?,
        a: cell.a,
        fractional: cell.fractional,
        precision: cell.precision,
        ke_cutoff: cell.ke_cutoff,
        exp_to_discard: cell.exp_to_discard,
        mesh: cell.mesh,
        rcut: cell.rcut,
        dimension: cell.dimension,
        low_dim_ft_type: cell.low_dim_ft_type,
        pseudo_name: cell.pseudo_name.clone(),
        use_particle_mesh_ewald: cell.use_particle_mesh_ewald,
        use_loose_rcut: cell.use_loose_rcut,
        space_group_symmetry: cell.space_group_symmetry,
        symmorphic: cell.symmorphic,
        rcut_from_build: cell._rcut_from_build,
        mesh_from_build: cell._mesh_from_build,
    })
}

/// Rebuild a [`Cell`] from packed state. Ports `cell.py:81-87`.
///
/// Upstream's `unpack` produces an UN-built `Cell` (it only updates `__dict__`).
/// This port instead reconstructs a fully-built one, because the molecular half
/// is rebuilt through [`pyscf_gto::loads`], which re-runs the deterministic
/// `format_basis`/`make_env` pipeline and therefore always yields a built
/// `Mole`. The result satisfies `_built == true`.
///
/// # Errors
/// Propagates [`pyscf_gto::loads`] failures, and rejects a lattice that is not
/// invertible.
pub fn unpack(p: CellPack) -> Result<Cell, PyscfRsError> {
    let mut mol = pyscf_gto::loads(&p.mole_json)?;
    let det = crate::cell::det3(&p.a);
    if det == 0.0 || !det.is_finite() {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "unpack: cell.a is singular (det = {det})"
        ))));
    }
    // Plan 10-01 — `mole_json` round-trips the BASIS, not the PP-adjusted
    // `_atm[CHARGE_OF]`, so the pseudopotential is re-resolved from the name
    // that `pack` preserved and its valence charges re-applied. Without this a
    // `loads(dumps(cell))` cell would silently become all-electron.
    let pseudo = match p.pseudo_name.as_deref() {
        Some(name) if !name.is_empty() => {
            let data = crate::pseudo::resolve_pseudo(name, &mol._atom)?;
            if data.is_empty() {
                None
            } else {
                crate::cell::apply_pseudo_charges(&mut mol, &data);
                Some(data)
            }
        }
        _ => None,
    };

    Ok(Cell {
        mol,
        a: p.a,
        mesh: p.mesh,
        dimension: p.dimension,
        low_dim_ft_type: p.low_dim_ft_type,
        precision: p.precision,
        ke_cutoff: p.ke_cutoff,
        rcut: p.rcut,
        ew_eta: None,
        ew_cut: None,
        pseudo,
        pseudo_name: p.pseudo_name,
        exp_to_discard: p.exp_to_discard,
        fractional: p.fractional,
        use_particle_mesh_ewald: p.use_particle_mesh_ewald,
        use_loose_rcut: p.use_loose_rcut,
        space_group_symmetry: p.space_group_symmetry,
        symmorphic: p.symmorphic,
        lattice_symmetry: None,
        symm_orb: None,
        irrep_id: None,
        _built: true,
        _rcut_from_build: p.rcut_from_build,
        _mesh_from_build: p.mesh_from_build,
    })
}

/// Serialise a [`Cell`] to a JSON string. Ports `cell.py:90-130`.
pub fn dumps(cell: &Cell) -> Result<String, PyscfRsError> {
    let packed = pack(cell)?;
    serde_json::to_string(&packed).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cell dumps serde_json error: {e}"
        )))
    })
}

/// Deserialise a [`Cell`] from a JSON string produced by [`dumps`].
/// Ports `cell.py:132-155`.
pub fn loads(json: &str) -> Result<Cell, PyscfRsError> {
    let packed: CellPack = serde_json::from_str(json).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "cell loads serde_json error: {e}"
        )))
    })?;
    unpack(packed)
}

/// The lattice of a packed cell, as an [`ALattice`] ready to feed back into
/// [`crate::CellBuildArgs`]. Convenience for callers that want to rebuild with
/// modified kwargs rather than restore verbatim.
pub fn packed_lattice(p: &CellPack) -> ALattice {
    ALattice::Matrix(p.a)
}

/// `true` when the packed cell carries a real (rather than sentinel) `rcut`
/// and `mesh`. Since plan 09-04 wired the estimators, this is `true` for every
/// cell that went through [`Cell::build`].
pub fn packed_has_cutoffs(p: &CellPack) -> bool {
    p.rcut != RCUT_UNSET && p.mesh != MESH_UNSET
}

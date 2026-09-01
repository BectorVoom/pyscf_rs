//! User-facing input types for [`crate::Cell::build`].
//!
//! Mirrors the upstream `Cell.build(**kwargs)` surface
//! (`pyscf/pbc/gto/cell.py:1593-1650`), layered on top of the molecular
//! [`pyscf_gto::MoleBuildArgs`] rather than duplicating it — D-PBC-01: there is
//! exactly ONE `Mole` build path in the workspace.

use pyscf_core::{CoreError, PyscfRsError};
use serde::{Deserialize, Serialize};

/// How integrals over the NON-periodic directions of a low-dimensional system
/// are evaluated.
///
/// Upstream (`cell.py:1271-1277`) stores this as a nullable string with three
/// states: `None`, `'inf_vacuum'` and `'analytic_2d_1'`. PBC-MASTER-PLAN §8.1
/// specifies two variants for the Rust port, so [`LowDimFtType::None`] carries
/// upstream's default meaning — "unless explicitly specified, `analytic_2d_1`
/// is used for a 2D system and `inf_vacuum` is assumed for 1D and 0D".
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum LowDimFtType {
    /// Upstream `None` — the dimension-dependent default described above.
    #[default]
    None,
    /// Upstream `'inf_vacuum'` — sample the non-periodic directions on the mesh
    /// grid in an infinite vacuum.
    InfVacuum,
}

/// Lattice-vector input. Accepts the three forms upstream's `cell.a` accepts.
///
/// The values are in whatever length unit [`pyscf_gto::MoleBuildArgs::unit`]
/// names; [`crate::Cell::build`] converts them to Bohr exactly once (upstream
/// does the same conversion lazily inside `lattice_vectors`,
/// `cell.py:1864-1885`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ALattice {
    /// Row-major `3 x 3`: each ROW is one lattice vector.
    Matrix([[f64; 3]; 3]),
    /// Nine values in row-major order — the flattened form of [`ALattice::Matrix`].
    Flat([f64; 9]),
    /// A whitespace/`;`/`,`/newline-separated string of nine numbers, e.g.
    /// `"3 0 0; 0 3 0; 0 0 3"`. Ports the string branch of
    /// `cell.py:1878-1882`, which replaces `;`, `,` and `\n` with spaces and
    /// reshapes the nine parsed floats to `3 x 3`.
    Str(String),
}

impl Default for ALattice {
    fn default() -> Self {
        Self::Matrix([[0.0; 3]; 3])
    }
}

impl ALattice {
    /// Resolve to a row-major `3 x 3` matrix, in the INPUT unit (no scaling).
    ///
    /// # Errors
    /// [`CoreError::InvalidMolecule`] if the string form does not contain
    /// exactly nine parseable numbers.
    pub fn to_matrix(&self) -> Result<[[f64; 3]; 3], PyscfRsError> {
        let flat: [f64; 9] = match self {
            Self::Matrix(m) => {
                let mut f = [0.0; 9];
                for i in 0..3 {
                    f[i * 3..i * 3 + 3].copy_from_slice(&m[i]);
                }
                f
            }
            Self::Flat(f) => *f,
            Self::Str(s) => {
                // cell.py:1879 — `;`, `,` and newlines are all just separators.
                let cleaned: String = s
                    .chars()
                    .map(|c| {
                        if c == ';' || c == ',' || c == '\n' {
                            ' '
                        } else {
                            c
                        }
                    })
                    .collect();
                let vals: Vec<f64> = cleaned
                    .split_whitespace()
                    .map(|t| {
                        t.parse::<f64>().map_err(|_| {
                            PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                                "cell.a: cannot parse '{t}' as a number"
                            )))
                        })
                    })
                    .collect::<Result<_, _>>()?;
                if vals.len() != 9 {
                    return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "cell.a: expected 9 numbers (a 3x3 lattice), got {}",
                        vals.len()
                    ))));
                }
                let mut f = [0.0; 9];
                f.copy_from_slice(&vals);
                f
            }
        };
        Ok([
            [flat[0], flat[1], flat[2]],
            [flat[3], flat[4], flat[5]],
            [flat[6], flat[7], flat[8]],
        ])
    }
}

/// Upstream `Cell.precision` default (`cell.py:1309`).
pub const DEFAULT_PRECISION: f64 = 1e-8;

/// Typed-kwargs analog of upstream `Cell.build(**kwargs)`.
///
/// The molecular half rides in [`CellBuildArgs::mole`] and is handed verbatim to
/// [`pyscf_gto::build_from`] — D-PBC-01. Everything else is periodic state.
#[derive(Debug, Clone)]
pub struct CellBuildArgs {
    /// The molecular half (atoms, basis, charge, spin, unit, …). `unit` also
    /// governs how [`CellBuildArgs::a`] is interpreted.
    pub mole: pyscf_gto::MoleBuildArgs,
    /// Lattice vectors, one per ROW, in `mole.unit`.
    pub a: ALattice,
    /// FFT mesh (number of G-vectors along each direction). `None` means
    /// "estimate from `ke_cutoff`/`precision`" — wired by plan 09-04.
    pub mesh: Option<[usize; 3]>,
    /// Planewave kinetic-energy cutoff in Hartree (`0.5 * G^2 < ke_cutoff`).
    pub ke_cutoff: Option<f64>,
    /// Lattice-summation cutoff radius in Bohr. `None` means "estimate from
    /// `precision`" — wired by plan 09-04.
    pub rcut: Option<f64>,
    /// Target accuracy for Ewald and lattice sums. Default [`DEFAULT_PRECISION`].
    pub precision: f64,
    /// Number of periodic dimensions, `0..=3`. Default 3.
    pub dimension: u8,
    /// See [`LowDimFtType`].
    pub low_dim_ft_type: LowDimFtType,
    /// Whether the atom coordinates in `mole.atom` are FRACTIONAL (scaled by the
    /// lattice) rather than Cartesian. Ports `cell.py:1582-1590`.
    pub fractional: bool,
    /// Discard basis primitives whose exponent is below this value.
    /// Ports the `cell.py:1671-1735` diffuse-function filter.
    pub exp_to_discard: Option<f64>,
    /// Use particle-mesh Ewald for the nuclear repulsion (upstream default `false`).
    pub use_particle_mesh_ewald: bool,
    /// Consider space-group symmetry (upstream default `false`).
    pub space_group_symmetry: bool,
    /// Whether the lattice is symmorphic (upstream default `false`,
    /// `cell.py:1289-1293`). See [`crate::Cell::symmorphic`].
    pub symmorphic: bool,
    /// Use the looser per-shell `rcut_by_shells` estimate for `rcut`
    /// (`cell.py:1316`, upstream default `false`).
    pub use_loose_rcut: bool,
    /// Pseudopotential name (e.g. `"gth-pade"`). [`crate::Cell::build`] resolves
    /// it into [`crate::Cell::pseudo`] and rewrites the atom charges with the
    /// valence counts (plan 10-01, D-PBC-11).
    pub pseudo: Option<String>,
}

impl Default for CellBuildArgs {
    fn default() -> Self {
        Self {
            mole: pyscf_gto::MoleBuildArgs::default(),
            a: ALattice::default(),
            mesh: None,
            ke_cutoff: None,
            rcut: None,
            precision: DEFAULT_PRECISION,
            dimension: 3,
            low_dim_ft_type: LowDimFtType::default(),
            fractional: false,
            exp_to_discard: None,
            use_particle_mesh_ewald: false,
            space_group_symmetry: false,
            symmorphic: false,
            use_loose_rcut: false,
            pseudo: None,
        }
    }
}

impl CellBuildArgs {
    /// Fluent constructor for the common path: lattice + atoms + basis.
    pub fn new(a: ALattice, mole: pyscf_gto::MoleBuildArgs) -> Self {
        Self {
            a,
            mole,
            ..Default::default()
        }
    }
}

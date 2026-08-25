//! `Cell` — the periodic analogue of `Mole` (D-PBC-01).
//!
//! Ported from `pyscf/pbc/gto/cell.py`: the class body and `build`
//! (`:1250-1810`), the lattice properties (`:1811-1975`) and `tot_electrons`
//! (`:957-967`).
//!
//! # D-PBC-01 — `Cell` OWNS a `Mole` and `Deref`s to it
//!
//! Upstream `Cell` INHERITS from `mole.MoleBase`. Rust has no inheritance, and
//! duplicating `Mole`'s ~30 fields would create a second build path that could
//! drift from `pyscf_gto::build_from`. Instead `Cell` holds a `Mole` and
//! implements `Deref`/`DerefMut`, so `cell.nao_nr`, `cell.natm`,
//! `cell._env`, `cell.atom_coords()` … all resolve to the molecular ones with
//! no forwarding boilerplate, and there is exactly ONE `Mole` build path in the
//! workspace.
//!
//! # Units
//!
//! `a` is stored in BOHR, always. Upstream keeps the user's raw input in
//! `cell.a` and converts inside `lattice_vectors()` on every call
//! (`cell.py:1864-1885`); doing the conversion once in [`Cell::build`] means
//! `lattice_vectors()` is a pure accessor and no call site can forget the scale.
//! The user's original unit is still available as `cell.unit` (through the
//! `Deref`), so nothing is lost.

use crate::pseudo::PseudoData;
use crate::types::{CellBuildArgs, LowDimFtType};
use pyscf_core::{CoreError, Mole, PyscfRsError};
use std::f64::consts::PI;

/// Sentinel for "`rcut` has not been estimated yet". Since plan 09-04 wired
/// [`estimate_rcut`], this never survives a [`Cell::build`] — it only appears
/// on a `Cell` assembled by hand (e.g. [`Cell::default`]). Use
/// [`Cell::try_rcut`] rather than reading the field when you need a value you
/// can trust.
pub const RCUT_UNSET: f64 = 0.0;

/// Sentinel for "`mesh` has not been estimated yet". See [`RCUT_UNSET`].
pub const MESH_UNSET: [usize; 3] = [0, 0, 0];

/// A crystal: a `Mole` plus its lattice.
///
/// See the module docs for the `Deref`-to-`Mole` rationale (D-PBC-01).
#[derive(Debug, Clone)]
pub struct Cell {
    /// The molecular half. D-PBC-01: OWNED, never duplicated.
    pub mol: Mole,
    /// Lattice vectors in BOHR, one per ROW. Reciprocal vectors are
    /// `b1,b2,b3 = 2*pi*inv(a).T` — see [`Cell::reciprocal_vectors`].
    pub a: [[f64; 3]; 3],
    /// FFT mesh: the number of G-vectors along each direction.
    /// [`MESH_UNSET`] only on a `Cell` that never went through [`Cell::build`].
    pub mesh: [usize; 3],
    /// Number of periodic dimensions, `0..=3`.
    pub dimension: u8,
    /// How the non-periodic directions are treated. See [`LowDimFtType`].
    pub low_dim_ft_type: LowDimFtType,
    /// Target accuracy for Ewald and lattice sums.
    pub precision: f64,
    /// Planewave kinetic-energy cutoff in Hartree, if the user pinned one.
    pub ke_cutoff: Option<f64>,
    /// Lattice-summation cutoff radius in Bohr.
    /// [`RCUT_UNSET`] only on a `Cell` that never went through [`Cell::build`].
    pub rcut: f64,
    /// Ewald screening parameter. Computed by `get_ewald_params` in plan 09-08.
    pub ew_eta: Option<f64>,
    /// Ewald real-space cutoff. Computed by `get_ewald_params` in plan 09-08.
    pub ew_cut: Option<f64>,
    /// Parsed GTH pseudopotential. Always `None` until plan 10-01 (D-PBC-11);
    /// the user's requested NAME lives in [`Cell::pseudo_name`] meanwhile.
    pub pseudo: Option<PseudoData>,
    /// The pseudopotential name the user asked for, preserved verbatim so it
    /// survives `dumps`/`loads` and so plan 10-01 has it to parse.
    pub pseudo_name: Option<String>,
    /// Diffuse-primitive cutoff that was applied at build time, if any.
    pub exp_to_discard: Option<f64>,
    /// Whether the input atom coordinates were fractional.
    pub fractional: bool,
    /// Use particle-mesh Ewald for the nuclear repulsion.
    pub use_particle_mesh_ewald: bool,
    /// Consider space-group symmetry (not implemented before Phase 12).
    pub space_group_symmetry: bool,
    /// Use the looser per-shell `rcut_by_shells` estimate instead of the
    /// most-diffuse-primitive one (`cell.py:1316`, default `false`).
    /// Read by [`crate::cutoff::estimate_rcut`].
    pub use_loose_rcut: bool,
    /// `true` once [`Cell::build`] succeeds.
    pub _built: bool,
    /// `true` when `rcut` was left to `build` to estimate (upstream
    /// `_rcut_from_build`, `cell.py:1364-1370`). Distinguishes a
    /// user-pinned `rcut` from an un-estimated one.
    pub _rcut_from_build: bool,
    /// `true` when `mesh` was left to `build` to estimate (upstream
    /// `_mesh_from_build`, `cell.py:1357-1363`).
    pub _mesh_from_build: bool,
}

impl std::ops::Deref for Cell {
    type Target = Mole;
    fn deref(&self) -> &Mole {
        &self.mol
    }
}

impl std::ops::DerefMut for Cell {
    fn deref_mut(&mut self) -> &mut Mole {
        &mut self.mol
    }
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            mol: Mole::default(),
            a: [[0.0; 3]; 3],
            mesh: MESH_UNSET,
            dimension: 3,
            low_dim_ft_type: LowDimFtType::None,
            precision: crate::types::DEFAULT_PRECISION,
            ke_cutoff: None,
            rcut: RCUT_UNSET,
            ew_eta: None,
            ew_cut: None,
            pseudo: None,
            pseudo_name: None,
            exp_to_discard: None,
            fractional: false,
            use_particle_mesh_ewald: false,
            space_group_symmetry: false,
            use_loose_rcut: false,
            _built: false,
            _rcut_from_build: false,
            _mesh_from_build: false,
        }
    }
}

// ---------------------------------------------------------------------------
// 3x3 linear algebra, closed form.
//
// PBC-MASTER-PLAN plan 09-03 step 3 is explicit: do NOT call faer for a 3x3.
// Plan 09-04 needs the same three functions in `pyscf-pbc-tools::mesh`
// (`b = 2*pi*inv(a.T)` for `cutoff_to_mesh`), and the dependency edge runs
// `pyscf-pbc-gto -> pyscf-pbc-tools`, so the definitions moved DOWN into
// `pyscf_pbc_tools::mat3` and are re-exported here. One lattice inversion in
// the workspace; a second copy could drift.
// ---------------------------------------------------------------------------

pub use pyscf_pbc_tools::mat3::{det3, inv3, transpose3};

/// Row-vector times matrix: `out = v . m`.
fn vec_mat(v: &[f64; 3], m: &[[f64; 3]; 3]) -> [f64; 3] {
    let mut o = [0.0; 3];
    for (j, oj) in o.iter_mut().enumerate() {
        *oj = v[0] * m[0][j] + v[1] * m[1][j] + v[2] * m[2][j];
    }
    o
}

use pyscf_pbc_tools::mat3::dot3;

// ---------------------------------------------------------------------------
// Lattice API — port of cell.py:1811-1975.
// ---------------------------------------------------------------------------

impl Cell {
    /// The primitive lattice vectors, one per ROW, in Bohr.
    ///
    /// Ports `cell.py:1864-1885`. Upstream applies the input-unit scale on every
    /// call; [`Cell::build`] has already applied it, so this is a pure accessor
    /// (see the module docs on Units).
    pub fn lattice_vectors(&self) -> [[f64; 3]; 3] {
        self.a
    }

    /// Cell volume in Bohr^3 — `|det(a)|`. Ports `cell.py:1830-1831`.
    pub fn vol(&self) -> f64 {
        det3(&self.a).abs()
    }

    /// Reciprocal lattice vectors, one per ROW: `norm_to * inv(a.T)`.
    ///
    /// Ports `cell.py:1896-1917`. The defining property is
    /// `reciprocal_vectors(x) . a.T == x * I`; [`Cell::reciprocal_vectors_2pi`]
    /// is the `norm_to = 2*pi` default upstream uses.
    ///
    /// # The `dimension < 3` branch
    ///
    /// PBC-MASTER-PLAN plan 09-03 step 4 describes this branch as one that
    /// "zeroes out the non-periodic rows". It does NOT — upstream
    /// (`cell.py:1908-1914`) only ASSERTS that the non-periodic lattice vectors
    /// are orthogonal to the periodic ones (`dimension == 1`: all three
    /// mutually orthogonal; `dimension == 2`: `a3` orthogonal to `a1` and `a2`)
    /// and then computes the FULL `inv(a.T)` regardless. Zeroing rows would
    /// break the round-trip identity this function's own acceptance test
    /// asserts. The port therefore reproduces upstream: the same orthogonality
    /// checks, downgraded from a hard `assert` to a `debug_assert` plus a
    /// `tracing::warn!`, so a release build of a slightly-off lattice warns
    /// instead of aborting.
    ///
    /// # Errors
    /// [`CoreError::InvalidMolecule`] if the lattice is singular.
    pub fn reciprocal_vectors(&self, norm_to: f64) -> Result<[[f64; 3]; 3], PyscfRsError> {
        let a = self.lattice_vectors();
        // cell.py:1908-1914 — orthogonality preconditions for low-dimensional cells.
        const ORTHO_TOL: f64 = 1e-9;
        let check = |name: &str, v: f64| {
            if v.abs() >= ORTHO_TOL {
                tracing::warn!(
                    "cell.reciprocal_vectors: dimension = {}, but {name} = {v:e} is not \
                     orthogonal to within {ORTHO_TOL:e}; upstream asserts this",
                    self.dimension
                );
            }
            debug_assert!(
                v.abs() < ORTHO_TOL,
                "reciprocal_vectors: dimension = {}, {name} = {v:e} violates the \
                 upstream orthogonality assertion (cell.py:1908-1914)",
                self.dimension
            );
        };
        if self.dimension == 1 {
            check("a1.a2", dot3(&a[0], &a[1]));
            check("a1.a3", dot3(&a[0], &a[2]));
            check("a2.a3", dot3(&a[1], &a[2]));
        } else if self.dimension == 2 {
            check("a1.a3", dot3(&a[0], &a[2]));
            check("a2.a3", dot3(&a[1], &a[2]));
        }

        let b = inv3(&transpose3(&a))?;
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = norm_to * b[i][j];
            }
        }
        Ok(out)
    }

    /// [`Cell::reciprocal_vectors`] with upstream's default `norm_to = 2*pi`.
    pub fn reciprocal_vectors_2pi(&self) -> Result<[[f64; 3]; 3], PyscfRsError> {
        self.reciprocal_vectors(2.0 * PI)
    }

    /// Absolute k-points in 1/Bohr from k-points scaled by the reciprocal
    /// lattice: `scaled . b`. Ports `cell.py:1919-1930`.
    pub fn get_abs_kpts(&self, scaled_kpts: &[[f64; 3]]) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        let b = self.reciprocal_vectors_2pi()?;
        Ok(scaled_kpts.iter().map(|k| vec_mat(k, &b)).collect())
    }

    /// Scaled k-points from absolute ones: `abs . a.T / (2*pi)`.
    /// Ports `cell.py:1932-1953` (the `KPoints`-object branch is Phase 12).
    pub fn get_scaled_kpts(&self, abs_kpts: &[[f64; 3]]) -> Vec<[f64; 3]> {
        let at = transpose3(&self.lattice_vectors());
        let inv_2pi = 1.0 / (2.0 * PI);
        abs_kpts
            .iter()
            .map(|k| {
                let m = vec_mat(k, &at);
                [m[0] * inv_2pi, m[1] * inv_2pi, m[2] * inv_2pi]
            })
            .collect()
    }

    /// Atom coordinates scaled by the lattice: `coords . inv(a)`.
    /// Ports `cell.py:1887-1894`.
    ///
    /// # Errors
    /// [`CoreError::InvalidMolecule`] if the lattice is singular.
    pub fn get_scaled_atom_coords(&self) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        let inv_a = inv3(&self.lattice_vectors())?;
        Ok(self
            .mol
            .atom_coords()
            .iter()
            .map(|r| vec_mat(r, &inv_a))
            .collect())
    }

    /// Total number of electrons over `nkpts` k-points.
    /// Ports `cell.py:957-967`.
    ///
    /// Upstream's `_nelectron` override (a user-pinned per-cell electron count)
    /// has no analogue in this workspace's `Mole`, so only the computed branch
    /// is ported: `sum(atom_charges) * nkpts - charge`, rounded to the nearest
    /// integer exactly as upstream does with `int(nelectron + 0.5)`.
    ///
    /// NOTE: the charges are all-electron until plan 10-01 lands the GTH
    /// pseudopotentials, which replace each `Z` with its valence count
    /// (D-PBC-11). This value will change for pseudopotential cells then.
    pub fn tot_electrons(&self, nkpts: usize) -> usize {
        let z: i64 = self.mol.atom_charges().iter().map(|c| *c as i64).sum();
        let n = z * nkpts as i64 - self.mol.charge as i64;
        n.max(0) as usize
    }

    /// `rcut`, estimating it on demand when the field still carries the
    /// [`RCUT_UNSET`] sentinel. Prefer this over reading the field: a cell
    /// assembled without [`Cell::build`] would otherwise hand out a zero radius
    /// that silently makes every lattice sum empty.
    pub fn try_rcut(&self) -> Result<f64, PyscfRsError> {
        if self.rcut == RCUT_UNSET {
            return estimate_rcut(self, self.precision);
        }
        Ok(self.rcut)
    }

    /// `mesh`, estimating it on demand when the field still carries the
    /// [`MESH_UNSET`] sentinel. See [`Cell::try_rcut`].
    pub fn try_mesh(&self) -> Result<[usize; 3], PyscfRsError> {
        if self.mesh == MESH_UNSET {
            return estimate_mesh(self);
        }
        Ok(self.mesh)
    }

    /// `cell.cutoff_to_mesh(ke_cutoff)` — `cell.py:1952-1967`.
    ///
    /// The free function [`pyscf_pbc_tools::mesh::cutoff_to_mesh`], except that
    /// for `dimension < 2` (or `dimension == 2` with `inf_vacuum`) the
    /// non-periodic axes keep the cell's CURRENT mesh instead of the
    /// cutoff-derived one — a planewave cutoff says nothing about a direction
    /// that is not sampled by planewaves.
    ///
    /// # Errors
    /// [`CoreError::InvalidMolecule`] if the lattice is singular; propagates
    /// [`Cell::try_mesh`] when the non-periodic axes are needed.
    pub fn cutoff_to_mesh(&self, ke_cutoff: f64) -> Result<[usize; 3], PyscfRsError> {
        let a = self.lattice_vectors();
        let dim = self.dimension as usize;
        let mut mesh = pyscf_pbc_tools::mesh::cutoff_to_mesh(&a, ke_cutoff)?;
        if dim < 2 || (dim == 2 && self.low_dim_ft_type == LowDimFtType::InfVacuum) {
            let current = self.try_mesh()?;
            for (i, m) in mesh.iter_mut().enumerate().skip(dim) {
                *m = current[i];
            }
        }
        Ok(mesh)
    }

    /// `cell.nimgs` — `cell.py:1852-1855`. The bounding-sphere half-widths for
    /// this cell's `rcut`.
    ///
    /// # Errors
    /// [`CoreError::InvalidMolecule`] if the lattice is singular.
    pub fn nimgs(&self) -> Result<[usize; 3], PyscfRsError> {
        crate::cutoff::get_bounding_sphere(self, self.try_rcut()?)
    }

    /// `cell.rcut_by_shells(precision)` — `cell.py:993-1024`. One radius per
    /// shell. `precision` defaults to `self.precision`.
    pub fn rcut_by_shells(&self, precision: Option<f64>) -> Vec<f64> {
        crate::cutoff::rcut_by_shells(self, precision.unwrap_or(self.precision), 0.0)
    }

    /// `cell.bas_rcut(bas_id, precision)` — `cell.py:409-422`.
    pub fn bas_rcut(&self, bas_id: usize, precision: Option<f64>) -> f64 {
        crate::cutoff::bas_rcut(self, bas_id, precision.unwrap_or(self.precision))
    }
}

// ---------------------------------------------------------------------------
// The two estimators, wired by plan 09-04. Bodies live in `crate::cutoff`.
// ---------------------------------------------------------------------------

/// Lattice-summation cutoff radius in Bohr from the target precision.
///
/// Delegates to [`crate::cutoff::estimate_rcut`] — the port of
/// `cell.py:424-436`. The `Result` wrapper is kept so plan 09-03's call sites
/// and the `pub use` in `lib.rs` stay source-compatible; the underlying
/// estimator is infallible.
///
/// # Errors
/// Never, today. The signature is preserved for API stability.
pub fn estimate_rcut(cell: &Cell, precision: f64) -> Result<f64, PyscfRsError> {
    Ok(crate::cutoff::estimate_rcut(cell, precision))
}

/// FFT mesh from `ke_cutoff` (or from `precision` via `estimate_ke_cutoff`).
///
/// Delegates to [`crate::cutoff::estimate_mesh`] — the port of
/// `cell.py:1755-1767` plus `tools/pbc.py:787-811`.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] if the lattice is singular.
pub fn estimate_mesh(cell: &Cell) -> Result<[usize; 3], PyscfRsError> {
    crate::cutoff::estimate_mesh(cell)
}

// ---------------------------------------------------------------------------
// build — port of cell.py:1593-1810.
// ---------------------------------------------------------------------------

impl Cell {
    /// Build a `Cell` from typed kwargs. Ports `cell.py:1593-1810`.
    ///
    /// Order of operations, matching upstream:
    /// 1. resolve the lattice and convert it to Bohr once (upstream defers this
    ///    to `lattice_vectors`; see the module docs on Units);
    /// 2. reject `dimension == 1` without `low_dim_ft_type == inf_vacuum`
    ///    (`cell.py:1665-1666`);
    /// 3. build the molecular half through [`pyscf_gto::build_from`] — the ONE
    ///    `Mole` build path (D-PBC-01) — applying the `fractional` coordinate
    ///    transform (`cell.py:1582-1590`) and the `exp_to_discard` diffuse
    ///    filter (`cell.py:1671-1735`) to its inputs first;
    /// 4. estimate `rcut` when the user did not pin it (`cell.py:1740-1742`);
    /// 5. warn on a left-handed lattice (`cell.py:1744-1749`);
    /// 6. warn when a low-dimensional cell has too little vacuum
    ///    (`cell.py:1751-1758`) — this needs `rcut`, hence the order;
    /// 7. estimate `mesh` from `ke_cutoff`, or from `precision` via
    ///    `estimate_ke_cutoff` (`cell.py:1760-1768`);
    /// 8. set `_built`.
    ///
    /// # Errors
    /// Anything [`pyscf_gto::build_from`] can raise, plus
    /// [`CoreError::InvalidMolecule`] for a malformed or singular lattice, an
    /// out-of-range `dimension`, or the unsupported `dimension == 1` case.
    pub fn build(args: CellBuildArgs) -> Result<Cell, PyscfRsError> {
        // --- 1. Lattice, converted to Bohr exactly once. ---
        let scale = args.mole.unit.length_in_au();
        let a_raw = args.a.to_matrix()?;
        let mut a = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] = a_raw[i][j] * scale;
            }
        }
        // Reject a degenerate lattice here rather than at first use.
        let det = det3(&a);
        if det == 0.0 || !det.is_finite() {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "cell.a is singular (det = {det}); lattice vectors must be linearly independent"
            ))));
        }

        if args.dimension > 3 {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "cell.dimension must be 0..=3, got {}",
                args.dimension
            ))));
        }
        // --- 2. cell.py:1665-1666 ---
        if args.dimension == 1 && args.low_dim_ft_type != LowDimFtType::InfVacuum {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                "Uniform grids for dimension=1 not supported (set low_dim_ft_type = InfVacuum)"
                    .into(),
            )));
        }

        // --- 3. The molecular half. ---
        let mut mole_args = args.mole.clone();

        // cell.py:1582-1590 — fractional coordinates are parsed WITHOUT a unit
        // conversion (upstream passes `unit=1.`) and then multiplied by the
        // Bohr lattice, so the product is already in Bohr.
        if args.fractional {
            let frac = pyscf_gto::format_atom::format_atom(
                &mole_args.atom,
                pyscf_core::Unit::Bohr,
                mole_args.origin,
                mole_args.axes,
            )?;
            let cart: Vec<(String, [f64; 3])> = frac
                .into_iter()
                .map(|(sym, f)| (sym, vec_mat(&f, &a)))
                .collect();
            mole_args.atom = pyscf_gto::AtomInput::Tuples(cart);
            mole_args.unit = pyscf_core::Unit::Bohr;
            mole_args.origin = [0.0; 3];
            mole_args.axes = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        }

        // cell.py:1671-1735 — discard diffuse primitives.
        //
        // Upstream filters `_basis` AND then performs equivalent surgery on the
        // already-projected `_bas`/`_env` arrays, re-running
        // `_nomalize_contracted_ao` by hand. This port filters `_basis` only and
        // lets `pyscf_gto::make_env` do the projection and normalisation
        // afterwards — the same functions upstream re-invokes, applied once
        // instead of twice, which cannot drift from the un-filtered path.
        if let Some(cut) = args.exp_to_discard {
            let atoms = pyscf_gto::format_atom::format_atom(
                &mole_args.atom,
                mole_args.unit,
                mole_args.origin,
                mole_args.axes,
            )?;
            let parsed = pyscf_gto::format_basis(&mole_args.basis, &atoms)?;
            let filtered = discard_diffuse_primitives(parsed, cut);
            let per_elem = filtered
                .into_iter()
                .map(|(sym, pb)| (sym, pyscf_gto::BasisInput::Parsed(pb)))
                .collect();
            mole_args.basis = pyscf_gto::BasisInput::PerElement(per_elem);
        }

        let mut mol = Mole::default();
        pyscf_gto::build_from(&mut mol, mole_args)?;

        let mut cell = Cell {
            mol,
            a,
            mesh: args.mesh.unwrap_or(MESH_UNSET),
            dimension: args.dimension,
            low_dim_ft_type: args.low_dim_ft_type,
            precision: args.precision,
            ke_cutoff: args.ke_cutoff,
            rcut: args.rcut.unwrap_or(RCUT_UNSET),
            ew_eta: None,
            ew_cut: None,
            pseudo: None,
            pseudo_name: args.pseudo.clone(),
            exp_to_discard: args.exp_to_discard,
            fractional: args.fractional,
            use_particle_mesh_ewald: args.use_particle_mesh_ewald,
            space_group_symmetry: args.space_group_symmetry,
            use_loose_rcut: args.use_loose_rcut,
            _built: false,
            _rcut_from_build: args.rcut.is_none(),
            _mesh_from_build: args.mesh.is_none(),
        };

        // --- 4. cell.py:1740-1742 — rcut, which step 6 needs. ---
        if cell._rcut_from_build {
            cell.rcut = estimate_rcut(&cell, cell.precision)?;
        }

        // --- 5. cell.py:1744-1749 — left-handed lattice warning. ---
        if det < 0.0 {
            tracing::warn!(
                "Lattice vectors are not in a right-handed coordinate system \
                 (det(a) = {det:e}). This can give wrong values for some integrals; \
                 consider reordering the lattice vectors."
            );
        }

        // --- 6. cell.py:1751-1758 — vacuum-size check. See Fig 1 of
        // PRB 73, 205119. (Carried over from plan 09-03, which could not do it
        // without `rcut`.)
        if cell.dimension <= 2 && cell.low_dim_ft_type != LowDimFtType::InfVacuum {
            let lz_guess = cell.rcut * 2.0;
            let a3 = (a[2][0] * a[2][0] + a[2][1] * a[2][1] + a[2][2] * a[2][2]).sqrt();
            if a3 < 0.7 * lz_guess {
                tracing::warn!(
                    "Size of vacuum may not be enough. The recommended vacuum size is \
                     {lz_guess} Bohr, but |a3| = {a3} Bohr."
                );
            }
        }

        // --- 7. cell.py:1760-1768 — mesh from ke_cutoff (or from precision).
        if cell._mesh_from_build {
            cell.mesh = estimate_mesh(&cell)?;
        }

        // --- 8. ---
        cell._built = true;
        Ok(cell)
    }
}

/// Drop every primitive whose exponent is below `cut`, then drop any
/// contraction whose coefficients all became zero, then drop any shell left
/// with no primitives. Ports the `_basis` half of `cell.py:1671-1712`.
fn discard_diffuse_primitives(
    basis: std::collections::HashMap<String, pyscf_core::ParsedBasis>,
    cut: f64,
) -> std::collections::HashMap<String, pyscf_core::ParsedBasis> {
    basis
        .into_iter()
        .map(|(sym, pb)| {
            let shells = pb
                .shells
                .into_iter()
                .filter_map(|sh| {
                    let keep: Vec<usize> = (0..sh.exponents.len())
                        .filter(|&p| sh.exponents[p] >= cut)
                        .collect();
                    if keep.is_empty() {
                        return None;
                    }
                    if keep.len() == sh.exponents.len() {
                        return Some(sh); // untouched — bit-identical to no filter
                    }
                    let exponents: Vec<f64> = keep.iter().map(|&p| sh.exponents[p]).collect();
                    // Contraction columns that are entirely zero after the cut
                    // carry no basis function; upstream removes them too.
                    let coeffs: Vec<Vec<f64>> = sh
                        .coeffs
                        .into_iter()
                        .map(|col| keep.iter().map(|&p| col[p]).collect::<Vec<f64>>())
                        .filter(|col: &Vec<f64>| col.iter().any(|c| *c != 0.0))
                        .collect();
                    if coeffs.is_empty() {
                        return None;
                    }
                    Some(pyscf_core::ShellSpec {
                        l: sh.l,
                        exponents,
                        coeffs,
                    })
                })
                .collect();
            (sym, pyscf_core::ParsedBasis { shells })
        })
        .collect()
}

/// Convenience alias for [`Cell::build`], mirroring upstream's
/// `pyscf.pbc.gto.M` / `C` factory (`cell.py:40-63`).
#[allow(non_snake_case)]
pub fn M(args: CellBuildArgs) -> Result<Cell, PyscfRsError> {
    Cell::build(args)
}

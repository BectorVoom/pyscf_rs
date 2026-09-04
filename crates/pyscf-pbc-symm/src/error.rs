use thiserror::Error;

#[derive(Debug, Error)]
pub enum PbcSymmError {
    #[error(transparent)]
    Core(#[from] pyscf_core::PyscfRsError),

    /// `pyscf/pbc/symm/pyscf_spglib.py:36-38` refuses ghost atoms
    /// (`'X-' in symbol or 'GHOST-' in symbol`) rather than silently
    /// mapping them into a symmetry-equivalence class. The native
    /// `search_space_group_ops` path (17-CONTEXT §1.5/Task 2) refuses the
    /// same way instead of reproducing `mole.atom_types`'s silent
    /// `'GHOST' -> 'X'` rename.
    #[error("ghost atom '{0}' is not supported with symmetry search")]
    GhostAtomUnsupported(String),

    /// `geom.py:198` — `raise ValueError("Input rotation matrix is wrong: %s" % rot)`.
    #[error("input rotation matrix is wrong: {0:?}")]
    InvalidRotation([[i32; 3]; 3]),

    /// `geom.py:215` — `raise RuntimeError("Unable to determine crystal class.")`.
    #[error("unable to determine crystal class")]
    UnknownCrystalClass,

    /// `group.py:396` — `raise ValueError('The elements do not form a group.')`.
    #[error("the elements do not form a group: {0}")]
    NotAGroup(String),

    /// `group.py:102` — `decrypt_hash` for a dimension other than 2 or 3
    /// (`raise NotImplementedError`).
    #[error("PGElement dimension must be 2 or 3, got {0}")]
    UnsupportedDimension(usize),

    /// `space_group.py:56-57` — `transform_rot`'s
    /// `raise RuntimeError("Point-group symmetries of the two coordinate
    /// systems are different.")`, hit when `allow_non_integer=False` and the
    /// transformed rotation is not (numerically) integer.
    #[error("point-group symmetries of the two coordinate systems are different")]
    NonIntegerRotation,

    /// `space_group.py:100-101` — `SPGElement.__init__` raises
    /// `NotImplementedError` for any `dimension != 3`. This port only ever
    /// constructs 3-dimensional elements, so the check is a `debug_assert`
    /// in [`crate::space_group::SPGElement`] rather than a runtime path that
    /// needs its own variant; kept here for documentation parity with
    /// [`PbcSymmError::UnsupportedDimension`].
    #[error("SPGElement dimension must be 3, got {0}")]
    UnsupportedSpgDimension(usize),

    /// `symmetry.py:237-239` — `_get_phase`'s `assert len(equiv_atm) == 1`:
    /// atom `.0` has `.1` candidate symmetry-equivalent images under this
    /// operation (expected exactly one).
    #[error("_get_phase: atom {0} has {1} candidate images under this operation (expected 1)")]
    AtomMapMismatch(usize, usize),

    /// `symmetry.py:243` — `_get_phase`'s `assert abs(Lshift -
    /// Lshift.round()).sum() < tol`: the lattice shift for atom `.0` is not
    /// (numerically) a lattice vector.
    #[error("_get_phase: lattice shift for atom {0} is not an integer lattice vector")]
    NonLatticeShift(usize),

    /// `symmetry.py:261-266` — `_get_rotation_mat`'s sanity-check asserts:
    /// atoms `.0` and `.1` are mapped onto each other by this operation but
    /// do not carry the same shell layout (AO count, shell count, or
    /// per-shell angular momentum).
    #[error(
        "_get_rotation_mat: atoms {0} and {1} are symmetry-equivalent but have different shell layouts"
    )]
    ShellLayoutMismatch(usize, usize),

    /// `basis.py:90` — `symm_adapted_basis`'s `assert nso == cell.nao`: the
    /// symmetry-adapted columns discovered across every irrep do not add up
    /// to the full AO count. A dropped phase, a wrong little co-group or a
    /// bad Gram-Schmidt threshold all show up here first.
    #[error("_symm_adapted_basis: irrep columns sum to {got}, expected cell.nao = {expected}")]
    IncompleteBasis { expected: usize, got: usize },

    /// `kpts.py:749-752` — `check_mo_occ_symmetry`'s
    /// `raise RuntimeError("Symmetry broken solution found. This is probably
    /// due to KUHF calculations with integer occupation numbers. Try use
    /// smearing or turn off symmetry.")`. The two BZ k-points named are in
    /// the same star but carry different MO occupations.
    #[error(
        "symmetry-broken solution: MO occupations differ between BZ k-points {0} and {1} of the same star (probably a KUHF run with integer occupations - try smearing, or turn off symmetry)"
    )]
    SymmetryBrokenOccupation(usize, usize),

    /// `kpts.py:415` — `symmetrize_wavefunction`'s own
    /// `raise RuntimeError('need verification')`. Upstream refuses to run
    /// this function AT ALL; this port refuses identically rather than
    /// shipping an unverified algorithm (17-05-PLAN.md Task 4).
    #[error(
        "symmetrize_wavefunction: upstream refuses this path with `raise RuntimeError('need verification')` (kpts.py:415); it is dead code there and is not resurrected here"
    )]
    SymmetrizeWavefunctionUnverified,

    /// 17-05-PLAN.md Task 4: a rotated grid index did not land EXACTLY on a
    /// mesh point. `check_mesh_symmetry` (`symmetry.py:96`) is what
    /// guarantees it does; a silent round here would be a WRONG DENSITY, so
    /// this fails loudly instead.
    #[error(
        "symmetrize_density: fractional translation component {1} x mesh = {2} is not an integer for operation {0}; the mesh does not carry the lattice symmetry (see check_mesh_symmetry)"
    )]
    MeshNotSymmetric(usize, usize, f64),

    /// `kpts.py:301` — `make_k4_ibz`'s
    /// `raise NotImplementedError("Unsupported symmetry.")`, and the `"s2"` /
    /// `"s4"` branches (`kpts.py:218-300`) which this port defers to 17-09
    /// (`kccsd_rhf_ksymm`), their only consumer.
    #[error("make_k4_ibz: unsupported symmetry '{0}'")]
    UnsupportedK4Symmetry(String),

    /// [`crate::basis::symm_adapted_basis`] / [`crate::basis::build_symmetry`]:
    /// the primitive k-point-symmetry input (`kpts_scaled_ibz`,
    /// `little_cogroup_ops`, `ops`, `dmats`) is internally inconsistent —
    /// mismatched lengths, or a little-co-group op index out of range. This
    /// is the Rust analogue of upstream's `_build_symmetry`'s
    /// `raise RuntimeError('Symmetry information not found in kpts. ...')`:
    /// the guard against silently symmetrizing nothing (17-04-PLAN.md Task 3).
    #[error("k-point symmetry input is inconsistent: {0}")]
    KptsSymmInputMismatch(String),

    // -----------------------------------------------------------------
    // 17-06 — `ktensor.py` / `KsymmArray`
    // -----------------------------------------------------------------
    /// `ktensor.py:62`, `:115`, `:123`, `:205` — the four
    /// `raise NotImplementedError` arms for a subarray rank other than 2 or
    /// 4. `KsymmArray` stores `nkpts^(rank-1)` blocks of a rank-`rank`
    /// subarray, and only the 2-d (`t1`, `Fov`, ...) and 4-d (`t2`, the
    /// `eris`) cases exist upstream.
    #[error("KsymmArray: subarray rank {0} is not supported (upstream implements 2 and 4 only)")]
    KsymmUnsupportedRank(usize),

    /// `ktensor.py:190` / `:214` — `raise RuntimeError('metadata not
    /// initialized')`, widened to name WHICH piece of metadata is missing.
    /// Upstream stores the metadata in a `dict` and lets a `KeyError`
    /// escape when `kqrts` / `rmat` is absent (`ktensor.py:127-129`,
    /// `:142-145`); this port makes the requirement explicit because the
    /// Rust metadata carries `Option`s rather than dictionary keys.
    #[error("KsymmArray: metadata field `{0}` is required for this operation but was not supplied")]
    KsymmMissingMetadata(&'static str),

    /// A block handed to `set_2d` / `set_4d` (or a buffer handed to
    /// `from_raw` / `from_dense`) does not have the element count the
    /// subarray shape demands. Upstream lets NumPy's own broadcast/reshape
    /// error fire (`ktensor.py:161`, `:174`, `:217`).
    #[error("KsymmArray: {what} has {got} elements, expected {expected}")]
    KsymmShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },

    /// The out-of-core branch of `_init` (`ktensor.py:78-80`,
    /// `lib.H5TmpFile()`). Every HDF5 create/read/write failure lands here.
    /// D-07: the scratch goes through `pyscf_chkfile::hdf5`, never a direct
    /// `hdf5-metno` dependency.
    #[error("KsymmArray: out-of-core scratch: {0}")]
    KsymmOutcore(String),

    /// `index_to_coords` (`ktensor.py:339-367`) produced an index outside
    /// `[0, n)`. NumPy would raise `IndexError` at the subsequent fancy
    /// index; this port refuses at the coordinate stage so the offending
    /// value can be named.
    #[error("KsymmArray: index {0} is out of range for an axis of length {1}")]
    KsymmIndexOutOfRange(i64, usize),

    /// `slice_to_coords` (`ktensor.py:369-381`) hands `step` straight to
    /// `np.arange`, which raises `ZeroDivisionError` for `step == 0`.
    #[error("KsymmArray: slice step must not be zero")]
    KsymmZeroStep,

    /// The `label` (`'oovv'`, `'ov'`, ...) or `trans` (`'nncc'`, `'nc'`,
    /// ...) metadata string is not a sequence of `o`/`v` resp. `n`/`c` of
    /// the subarray rank. Upstream would fail later, inside
    /// `getattr(rmat, pi * 2)` (`ktensor.py:273`).
    #[error("KsymmArray: {kind} string '{value}' is invalid: {reason}")]
    KsymmBadMetadataString {
        kind: &'static str,
        value: String,
        reason: &'static str,
    },
}

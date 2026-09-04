//! Port of `pyscf/pbc/symm/space_group.py` (369 l) — `SPGElement` and
//! `SpaceGroup` (`17-03-PLAN.md` Tasks 1/2).
//!
//! # Backend
//!
//! Only the native `'pyscf'` backend ships (17-CONTEXT §1.5). It is
//! upstream's own default (`space_group.py:264, 280`); the `spglib` backend
//! (`space_group.py:293-302`) is reached only when a caller explicitly sets
//! `backend = 'spglib'`, it cannot handle `cell.dimension < 3` at all
//! (`:288-290` — which rules it out for the `graphene` reference cell), and
//! no Phase-17 gate needs it. There is therefore no `SpaceGroup::backend`
//! field to model and no `NotYetImplemented { phase: 21, .. }` guard to add
//! for it — the field simply does not exist in this port. Record it here so
//! a later reader does not mistake the omission for a gap.

use pyscf_pbc_gto::Cell;

use crate::error::PbcSymmError;
use crate::geom::{self, RotMatrix, SYMPREC as GEOM_SYMPREC};
use crate::group::PgElement;

/// `space_group.py:27` — `SYMPREC`. Identical value to [`crate::geom::SYMPREC`];
/// upstream defines the constant twice (once per module) and this port
/// mirrors that rather than merging the two, so each module's doc comment
/// can cite its own upstream line number.
pub const SYMPREC: f64 = GEOM_SYMPREC;

/// `space_group.py:28` — `XYZ = np.eye(3)`: the Cartesian coordinate basis.
pub const XYZ: [[f64; 3]; 3] = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

// ---------------------------------------------------------------------
// small 3x3 linear algebra not already in `pyscf_pbc_tools::mat3`
// ---------------------------------------------------------------------

fn matmul3(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 {
                s += a[i][k] * b[k][j];
            }
            out[i][j] = s;
        }
    }
    out
}

/// `m @ v` (standard matrix-vector product). Also serves `v @ m.T`, since
/// `(v @ m.T)[i] == sum_j v[j] * m[i][j] == (m @ v)[i]` — the identity
/// `transform_rot`/`transform_trans`/`SPGElement.dot`/`dot_rot` all lean on
/// to avoid a second helper for the row-vector convention numpy uses.
fn matvec3(m: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2])
}

fn round_to(x: f64, ndigits: i32) -> f64 {
    let f = 10f64.powi(ndigits);
    (x * f).round() / f
}

fn int_to_f64(w: &RotMatrix) -> [[f64; 3]; 3] {
    std::array::from_fn(|i| std::array::from_fn(|j| w[i][j] as f64))
}

/// Round a rotation matrix that is assumed (by the caller) to already be
/// integer-valued to the nearest [`RotMatrix`]. Used only where upstream
/// itself only ever calls `hash`/`__lt__` on lattice-basis ops (never on an
/// `a2r`/`b2r`-transformed Cartesian one) — see [`SPGElement::hash_key`].
fn round_to_int(w: &[[f64; 3]; 3]) -> RotMatrix {
    std::array::from_fn(|i| std::array::from_fn(|j| w[i][j].round() as i32))
}

// ---------------------------------------------------------------------
// Task 1 — transform_rot / transform_trans (space_group.py:30-79)
// ---------------------------------------------------------------------

/// `space_group.py:30-60` — `transform_rot`. Transforms a rotation operator
/// from basis `a` to basis `b`.
///
/// # Errors
/// [`PbcSymmError::NonIntegerRotation`] when `allow_non_integer` is `false`
/// and the transformed rotation is not (numerically, within [`SYMPREC`])
/// integer — this is how a wrong basis conversion is caught; the plan calls
/// out that this flag must NOT be simplified to always-allow.
pub fn transform_rot(
    op: &[[f64; 3]; 3],
    a: &[[f64; 3]; 3],
    b: &[[f64; 3]; 3],
    allow_non_integer: bool,
) -> Result<[[f64; 3]; 3], PbcSymmError> {
    let bt = pyscf_pbc_tools::mat3::transpose3(b);
    let inv_bt = pyscf_pbc_tools::mat3::inv3(&bt)?;
    let at = pyscf_pbc_tools::mat3::transpose3(a);
    let p = matmul3(&inv_bt, &at);
    let inv_p = pyscf_pbc_tools::mat3::inv3(&p)?;
    let mut r = matmul3(&matmul3(&p, op), &inv_p);
    for row in r.iter_mut() {
        for v in row.iter_mut() {
            *v = round_to(*v, 15);
            if v.abs() < 1e-9 {
                *v = 0.0;
            }
        }
    }
    if !allow_non_integer {
        let max_err = r
            .iter()
            .flatten()
            .map(|v| (v - v.round()).abs())
            .fold(0.0_f64, f64::max);
        if max_err > SYMPREC {
            return Err(PbcSymmError::NonIntegerRotation);
        }
        for row in r.iter_mut() {
            for v in row.iter_mut() {
                *v = v.round();
            }
        }
    }
    Ok(r)
}

/// `space_group.py:62-79` — `transform_trans`. Transforms a fractional
/// translation from basis `a` to basis `b`.
///
/// # Errors
/// Propagates a singular `b` (via [`pyscf_pbc_tools::mat3::inv3`]).
pub fn transform_trans(
    op: &[f64; 3],
    a: &[[f64; 3]; 3],
    b: &[[f64; 3]; 3],
) -> Result<[f64; 3], PbcSymmError> {
    let bt = pyscf_pbc_tools::mat3::transpose3(b);
    let inv_bt = pyscf_pbc_tools::mat3::inv3(&bt)?;
    let at = pyscf_pbc_tools::mat3::transpose3(a);
    let p = matmul3(&inv_bt, &at);
    // `np.dot(op, P.T)` == `matvec3(&p, op)` — see the [`matvec3`] doc comment.
    Ok(matvec3(&p, op))
}

// ---------------------------------------------------------------------
// Task 1 — SPGElement (space_group.py:82-248)
// ---------------------------------------------------------------------

/// `space_group.py:82-248` — `SPGElement`: a space-group operation, a
/// rotation plus a fractional translation.
///
/// Upstream's `rot` is a bare `np.ndarray` that is sometimes integer-dtype
/// (straight out of `search_space_group_ops`) and sometimes float-dtype
/// (after `a2r`/`b2r`). This port stores `rot` as `[[f64; 3]; 3]` uniformly
/// — small integers are exactly representable in `f64`, so nothing is lost,
/// and every basis-conversion method ([`SPGElement::a2r`] etc.) can return
/// the same type regardless of whether its result happens to be integer.
///
/// `dimension` is always 3 (`SPGElement.__init__` raises `NotImplementedError`
/// for any other value, `:100-101`) — this port does not model the field at
/// all, matching [`crate::group::PgElement`]'s equivalent specialisation.
#[derive(Debug, Clone, Copy)]
pub struct SPGElement {
    pub rot: [[f64; 3]; 3],
    pub trans: [f64; 3],
}

impl Default for SPGElement {
    /// `space_group.py:94-96` — `rot=np.eye(3, dtype=np.int32), trans=np.zeros(3)`.
    fn default() -> Self {
        Self {
            rot: XYZ,
            trans: [0.0; 3],
        }
    }
}

impl SPGElement {
    pub fn new(rot: [[f64; 3]; 3], trans: [f64; 3]) -> Self {
        Self { rot, trans }
    }

    /// Construct from an integer [`RotMatrix`] (the common case — every
    /// element `search_space_group_ops` produces).
    pub fn from_int(rot: RotMatrix, trans: [f64; 3]) -> Self {
        Self {
            rot: int_to_f64(&rot),
            trans,
        }
    }

    /// `space_group.py:103-117` — `dot`, the point-operating branch:
    /// `r @ self.rot.T + self.trans`.
    pub fn dot_point(&self, r: &[f64; 3]) -> [f64; 3] {
        let rv = matvec3(&self.rot, r);
        [
            rv[0] + self.trans[0],
            rv[1] + self.trans[1],
            rv[2] + self.trans[2],
        ]
    }

    /// `space_group.py:103-117` — `dot`, the operator-composing branch:
    /// `self ∘ other` (apply `other` first, then `self`).
    pub fn dot(&self, other: &Self) -> Self {
        let beta = &self.rot;
        let b = self.trans;
        let alpha = &other.rot;
        let a = other.trans;
        let new_rot = matmul3(beta, alpha);
        let ba = matvec3(beta, &a);
        let new_trans = [b[0] + ba[0], b[1] + ba[1], b[2] + ba[2]];
        Self::new(new_rot, new_trans)
    }

    /// `space_group.py:119-123` — `dot_rot`: rotate a point WITHOUT the
    /// translation.
    pub fn dot_rot(&self, r: &[f64; 3]) -> [f64; 3] {
        matvec3(&self.rot, r)
    }

    /// `space_group.py:125-131` — `inv`.
    ///
    /// # Errors
    /// Propagates a singular `rot` (never expected for a genuine
    /// point-group element, but not assumed).
    pub fn inv(&self) -> Result<Self, PbcSymmError> {
        let inv_rot = pyscf_pbc_tools::mat3::inv3(&self.rot)?;
        let t = matvec3(&inv_rot, &self.trans);
        Ok(Self::new(inv_rot, [-t[0], -t[1], -t[2]]))
    }

    /// `space_group.py:133-139` — `transform`: change basis from `a` to `b`.
    ///
    /// # Errors
    /// As [`transform_rot`] / [`transform_trans`].
    pub fn transform(
        &self,
        a: &[[f64; 3]; 3],
        b: &[[f64; 3]; 3],
        allow_non_integer: bool,
    ) -> Result<Self, PbcSymmError> {
        let rot = transform_rot(&self.rot, a, b, allow_non_integer)?;
        let trans = transform_trans(&self.trans, a, b)?;
        Ok(Self::new(rot, trans))
    }

    /// `space_group.py:141-143` — `rot_is_eye`.
    pub fn rot_is_eye(&self) -> bool {
        matrix_close(&self.rot, &XYZ)
    }

    /// `space_group.py:145-147` — `rot_is_inversion`.
    pub fn rot_is_inversion(&self) -> bool {
        let neg_eye: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| -XYZ[i][j]));
        matrix_close(&self.rot, &neg_eye)
    }

    /// `space_group.py:149-151` — `trans_is_zero`.
    pub fn trans_is_zero(&self) -> bool {
        self.trans.iter().all(|t| t.abs() < SYMPREC)
    }

    /// `space_group.py:153-158` — `is_eye`.
    pub fn is_eye(&self) -> bool {
        self.rot_is_eye() && self.trans_is_zero()
    }

    /// `space_group.py:160-165` — `is_inversion`.
    pub fn is_inversion(&self) -> bool {
        self.rot_is_inversion() && self.trans_is_zero()
    }

    /// `space_group.py:197-205` — `__hash__`. Only ever called (by upstream,
    /// and by this port) on a lattice-basis `SPGElement` — i.e. one whose
    /// `rot` is exactly integer-valued, never on the `a2r`/`b2r`-transformed
    /// Cartesian form. [`round_to_int`] documents that assumption.
    pub fn hash_key(&self) -> i64 {
        let r = PgElement::new(round_to_int(&self.rot)).hash_key();
        let place = [144i64, 12, 1];
        let mut t: i64 = 0;
        for i in 0..3 {
            t += (self.trans[i] * 12.0).round() as i64 * place[i];
        }
        t * 3i64.pow(9) + r
    }

    /// `space_group.py:207-211` — `a2b`: direct lattice -> reciprocal lattice.
    ///
    /// # Errors
    /// Propagates a singular reciprocal lattice or [`PbcSymmError::NonIntegerRotation`]
    /// (which should never fire for a genuine point-group op: a crystallographic
    /// rotation that is integer in the direct-lattice basis is integer in the
    /// reciprocal basis too, since both are unimodular-integer representations
    /// of the same operation).
    pub fn a2b(&self, cell: &Cell) -> Result<Self, PbcSymmError> {
        self.transform(
            &cell.lattice_vectors(),
            &cell.reciprocal_vectors_2pi()?,
            false,
        )
    }

    /// `space_group.py:213-217` — `a2r`: direct lattice -> Cartesian.
    /// Generically NON-integer (the Cartesian rotation matrix is only a
    /// signed permutation matrix for cubic-class lattices) — hence
    /// `allow_non_integer = true`, matching upstream exactly.
    pub fn a2r(&self, cell: &Cell) -> Result<Self, PbcSymmError> {
        self.transform(&cell.lattice_vectors(), &XYZ, true)
    }

    /// `space_group.py:219-223` — `b2a`: reciprocal lattice -> direct lattice.
    pub fn b2a(&self, cell: &Cell) -> Result<Self, PbcSymmError> {
        self.transform(
            &cell.reciprocal_vectors_2pi()?,
            &cell.lattice_vectors(),
            false,
        )
    }

    /// `space_group.py:225-229` — `b2r`: reciprocal lattice -> Cartesian.
    ///
    /// **Ported EXACTLY as upstream has it (RULE 2), including an apparent
    /// asymmetry with [`SPGElement::a2r`]:** upstream passes NO
    /// `allow_non_integer` argument here (so it defaults to `false`), even
    /// though the Cartesian representation of a rotation is, in general,
    /// exactly as non-integer starting from the reciprocal basis as it is
    /// starting from the direct one. **Verified against live upstream
    /// 2.12.1**: on the `si` fixture (cubic, Fd-3m) every op's Cartesian
    /// representation happens to be a signed permutation matrix, so `b2r`
    /// always succeeds there; on `graphene` (hexagonal) it RAISES
    /// [`PbcSymmError::NonIntegerRotation`] for every 3-/6-fold rotation,
    /// where `cos(120°) = -0.5` is not integer. This is not "fixed" here —
    /// upstream has never exercised it on a non-cubic system in its own test
    /// suite, and RULE 2 requires a bit-exact port, not a repair.
    pub fn b2r(&self, cell: &Cell) -> Result<Self, PbcSymmError> {
        self.transform(&cell.reciprocal_vectors_2pi()?, &XYZ, false)
    }

    /// `space_group.py:231-235` — `r2a`: Cartesian -> direct lattice.
    pub fn r2a(&self, cell: &Cell) -> Result<Self, PbcSymmError> {
        self.transform(&XYZ, &cell.lattice_vectors(), false)
    }

    /// `space_group.py:237-241` — `r2b`: Cartesian -> reciprocal lattice.
    pub fn r2b(&self, cell: &Cell) -> Result<Self, PbcSymmError> {
        self.transform(&XYZ, &cell.reciprocal_vectors_2pi()?, false)
    }
}

fn matrix_close(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> bool {
    for i in 0..3 {
        for j in 0..3 {
            if (a[i][j] - b[i][j]).abs() > 1e-9 {
                return false;
            }
        }
    }
    true
}

/// `space_group.py:177-195` — the total order (`__lt__` … `__ge__`), plus
/// `__eq__`/`__ne__`, all via [`SPGElement::hash_key`].
impl PartialEq for SPGElement {
    fn eq(&self, other: &Self) -> bool {
        self.hash_key() == other.hash_key()
    }
}
impl Eq for SPGElement {}
impl PartialOrd for SPGElement {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for SPGElement {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.hash_key().cmp(&other.hash_key())
    }
}

// ---------------------------------------------------------------------
// Task 2 — SpaceGroup (space_group.py:250-369)
// ---------------------------------------------------------------------

/// `space_group.py:250-337` — `SpaceGroup`, native `'pyscf'` backend only
/// (see the module doc).
#[derive(Debug, Clone)]
pub struct SpaceGroup {
    pub symprec: f64,
    /// `space_group.py:265` — `ops`, sorted (`build`'s `self.ops.sort()`,
    /// `:310`).
    pub ops: Vec<SPGElement>,
    /// `space_group.py:267` — `nop = len(ops)`.
    pub nop: usize,
    /// `space_group.py:271` — `groupname['point_group_symbol']`. The
    /// `international_symbol`/`international_number` upstream also fills
    /// stay `None`/unset in the native backend (`space_group.py:307-308`:
    /// `#TODO add space group symbol`) — there is nothing to port there.
    pub point_group_symbol: &'static str,
}

impl SpaceGroup {
    /// `space_group.py:287-318` — `build`, native backend only.
    ///
    /// Upstream computes `self.ops` via `search_space_group_ops` and,
    /// separately, `pg_symbol` via `get_crystal_class(cell, tol=symprec)`,
    /// which INTERNALLY re-runs `search_space_group_ops` from scratch
    /// (`ops=None` default). This port runs the search exactly ONCE and
    /// classifies the same rotation list both ways
    /// ([`geom::get_crystal_class_from_rotations`]) — a deterministic
    /// function of `(cell, symprec)` either way, so the result is identical
    /// and the search is not paid for twice.
    ///
    /// # Errors
    /// Propagates [`geom::search_space_group_ops`] / [`geom::get_crystal_class_from_rotations`]
    /// (ghost atoms, an undeterminable crystal class, …).
    pub fn build(cell: &Cell, symprec: f64) -> Result<Self, PbcSymmError> {
        let raw_ops = geom::search_space_group_ops(cell, None, symprec)?;
        let mut ops: Vec<SPGElement> = raw_ops
            .iter()
            .map(|o| SPGElement::from_int(o.rot, o.trans))
            .collect();
        let rotations: Vec<RotMatrix> = raw_ops.iter().map(|o| o.rot).collect();
        let (point_group_symbol, _laue) = geom::get_crystal_class_from_rotations(&rotations)?;

        ops.sort();
        let nop = ops.len();
        Ok(Self {
            symprec,
            ops,
            nop,
            point_group_symbol,
        })
    }

    /// `space_group.py:320-336` — `dump_info`. Emits at `tracing::info!`
    /// (the group name) and `tracing::debug!` (every operation), matching
    /// upstream's two verbosity tiers without threading a `cell.verbose`
    /// value through — `tracing`'s own subscriber filtering does that job.
    pub fn dump_info(&self, ops: Option<&[SPGElement]>) {
        let ops = ops.unwrap_or(&self.ops);
        let pg = self.point_group_symbol;
        tracing::info!(point_group_symbol = pg, "[Cell] Point group symbol");
        let message = if ops.len() < self.ops.len() {
            "Subset of space group symmetry operations:"
        } else {
            "Space group symmetry operations:"
        };
        tracing::debug!("{message}");
        for op in ops {
            tracing::debug!(rot = ?op.rot, trans = ?op.trans, "space group operation");
        }
    }
}

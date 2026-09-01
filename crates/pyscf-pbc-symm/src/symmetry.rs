//! Port of `pyscf/pbc/symm/symmetry.py` (348 l) — Wigner-D matrices,
//! `check_mesh_symmetry`, `Symmetry`, and the three symmetry transforms
//! (`17-03-PLAN.md` Tasks 3-6).
//!
//! # The §3.2 trap — ONE implementation of the AO rotation
//!
//! `mo_coeff` is COLUMN-MAJOR `nao x nmo` (`pyscf-pbc-scf/src/types.rs:119`)
//! and the `Dmats` this module builds are block-diagonal over SHELLS in the
//! AO index, applied through [`get_rotation_mat`]. This is exactly the shape
//! of 14-05's `decompose_j2c` defect (a column-major eigenvector read
//! row-major, worth +6 306 866.73 Ha and invisible to every gate then
//! existing). The fix that generation of bug needs is structural, not a
//! bigger test: [`get_rotation_mat`] is the ONLY function in this crate that
//! assembles an AO-space rotation matrix, and every one of
//! [`transform_mo_coeff`], [`transform_dm`] and [`transform_1e_operator`]
//! goes through it — there is no second, parallel assembly path (in
//! particular, [`make_rot_loc`] is ported for 17-04's benefit but is NOT used
//! to build a second copy of the rotation matrix here). `tests/symmetry.rs`
//! pins [`get_rotation_mat`] itself with the identity that DEFINES a
//! representation, `R(op)·S·R(op)ᴴ == S` (S = the Γ-point overlap), not with
//! a round-trip — see that file's module doc.
//!
//! # `Symmetry` never owns a `Cell` (17-CONTEXT §3.9)
//!
//! Upstream's `Cell.build_lattice_symmetry` (`cell.py:1552-1580`) builds a
//! `Symmetry`, then deletes `lattice_symmetry.cell` and
//! `lattice_symmetry.spacegroup.cell` (`:1576-1579`) purely to break a
//! Python reference-count cycle (`Cell` -> `Symmetry` -> `Cell`). In Rust
//! that cycle cannot exist — there is no meaning to "delete a field to break
//! a refcount cycle" here — so this is intentionally NOT ported. But the
//! lesson upstream's workaround encodes IS real: [`Symmetry::build`] takes a
//! BORROWED `&Cell` and stores only lattice-derived data (rotations,
//! translations, Dmats, crystal class); it never stores a `Cell` at all, let
//! alone a clone. A cloned `Cell` would silently DESYNCHRONISE from the
//! original on the next `cell.build()` — worse than the cycle upstream was
//! avoiding, since a stale clone fails silently where a refcount cycle merely
//! wastes memory. [`build_lattice_symmetry`] (the free function this module
//! exposes in place of a `Cell` method — see its own doc for why) reflects
//! this: it produces a plain-data `pyscf_pbc_gto::LatticeSymmetry` from a
//! `Symmetry` and stores THAT on `Cell`, never the `Symmetry` itself.

use num_complex::Complex64;

use pyscf_core::raw_layout::{ANG_OF, ATOM_OF, BAS_SLOTS, NCTR_OF};
use pyscf_pbc_gto::Cell;

use crate::error::PbcSymmError;
use crate::space_group::{SPGElement, SpaceGroup, SYMPREC, XYZ};

// ---------------------------------------------------------------------
// Task 3 — Wigner-D matrices (symmetry.py:32-94)
// ---------------------------------------------------------------------

/// Wigner-D matrices for ONE operation, indexed by angular momentum:
/// `dmats[l][row][col]`, `l` from `0` to `l_max` inclusive. Real
/// spherical-harmonic order unless the cell is Cartesian.
pub type DmatSet = Vec<Vec<Vec<f64>>>;

fn factorial(n: i64) -> f64 {
    if n <= 1 {
        1.0
    } else {
        (2..=n).map(|k| k as f64).product()
    }
}

/// `pyscf/symm/Dmatrix.py:64-120` — `dmatrix`, the Wigner SMALL-d matrix in
/// the z-y-z convention, general angular momentum. Upstream special-cases
/// `l = 0, 1, 2` for speed and falls back to this exact general formula for
/// `l >= 3`; this port uses the general formula UNIFORMLY for every `l`
/// (including 0/1/2) — it is the same closed-form Wigner-d expression in all
/// cases, so this is not a second algorithm, only the removal of three
/// hand-unrolled fast paths this crate's angular momenta (`l <= 1` on every
/// §9.2 reference cell) do not need.
fn dmatrix_small_d(l: usize, beta: f64) -> Vec<Vec<f64>> {
    let n = 2 * l + 1;
    let li = l as i64;
    let c = (beta / 2.0).cos();
    let s = (beta / 2.0).sin();
    let facs: Vec<f64> = (0..=2 * li).map(factorial).collect();
    let cs: Vec<f64> = (0..=2 * l).map(|k| c.powi(k as i32)).collect();
    let ss: Vec<f64> = (0..=2 * l).map(|k| s.powi(k as i32)).collect();

    let mut mat = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        let m1 = i as i64 - li;
        for j in 0..n {
            let m2 = j as i64 - li;
            let kmin = (m2 - m1).max(0);
            let kmax = (li + m2).min(li - m1);
            let mut acc = 0.0_f64;
            let mut k = kmin;
            while k <= kmax {
                let p_cs = (2 * li + m2 - m1 - 2 * k) as usize;
                let p_ss = (m1 - m2 + 2 * k) as usize;
                let denom = facs[(li + m2 - k) as usize]
                    * facs[k as usize]
                    * facs[(m1 - m2 + k) as usize]
                    * facs[(li - m1 - k) as usize];
                let tmp = cs[p_cs] * ss[p_ss] / denom;
                if (m1 + m2 + k).rem_euclid(2) == 1 {
                    acc -= tmp;
                } else {
                    acc += tmp;
                }
                k += 1;
            }
            mat[i][j] = acc;
        }
    }
    let msfac: Vec<f64> = (0..n)
        .map(|i| {
            let m = i as i64 - li;
            (facs[(li + m) as usize] * facs[(li - m) as usize]).sqrt()
        })
        .collect();
    for i in 0..n {
        for j in 0..n {
            mat[i][j] *= msfac[i] * msfac[j];
        }
    }
    mat
}

/// `pyscf/symm/sph.py:106-148` — `sph_pure2real(l, reorder_p=False)`: the
/// pure-to-real spherical harmonic transform, ALWAYS the general (non-`p`
/// reordered) branch — [`dmatrix`] always calls the upstream equivalent with
/// `reorder_p=False`, applying the `l=1` `(x,y,z)` reorder itself, AFTER the
/// real conversion (matching `Dmatrix.py:46-48`).
fn sph_pure2real(l: usize) -> Vec<Vec<Complex64>> {
    let n = 2 * l + 1;
    let mut u = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    let sqrthfr = Complex64::new(0.5_f64.sqrt(), 0.0);
    let sqrthfi = Complex64::new(0.0, 0.5_f64.sqrt());
    u[l][l] = Complex64::new(1.0, 0.0);
    let mut m = 1usize;
    while m <= l {
        u[l - m][l - m] = sqrthfi;
        u[l + m][l - m] = sqrthfi;
        u[l - m][l + m] = sqrthfr;
        u[l + m][l + m] = -sqrthfr;
        m += 2;
    }
    let mut m = 2usize;
    while m <= l {
        u[l - m][l - m] = sqrthfi;
        u[l + m][l - m] = -sqrthfi;
        u[l - m][l + m] = sqrthfr;
        u[l + m][l + m] = sqrthfr;
        m += 2;
    }
    u
}

fn cmatmul_square(a: &[Vec<Complex64>], b: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let n = a.len();
    let mut out = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i][k];
            for j in 0..n {
                out[i][j] += aik * b[k][j];
            }
        }
    }
    out
}

fn cdagger_square(a: &[Vec<Complex64>]) -> Vec<Vec<Complex64>> {
    let n = a.len();
    let mut out = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        for j in 0..n {
            out[j][i] = a[i][j].conj();
        }
    }
    out
}

/// `pyscf/symm/geom.py`-adjacent `get_euler_angles(c1, c2)` (`Dmatrix.py:123-190`),
/// the `c1.ndim == 2` branch only (this port never calls the "two points"
/// branch).
fn get_euler_angles(c1: &[[f64; 3]; 3], c2: &[[f64; 3]; 3]) -> (f64, f64, f64) {
    use pyscf_pbc_tools::mat3::{cross3, dot3, norm3};

    let zz = dot3(&c1[2], &c2[2]);
    let beta = if (zz - 1.0).abs() < 1e-12 {
        1.0_f64.acos()
    } else if (zz + 1.0).abs() < 1e-12 {
        (-1.0_f64).acos()
    } else {
        zz.acos()
    };
    let yp = if zz.abs() < 1.0 - 1e-12 {
        let raw = cross3(&c1[2], &c2[2]);
        let n = norm3(&raw);
        [raw[0] / n, raw[1] / n, raw[2] / n]
    } else {
        c1[1]
    };

    let yy = dot3(&yp, &c1[1]);
    let mut alpha = yy.acos();
    if dot3(&cross3(&c1[1], &yp), &c1[2]) < 0.0 {
        alpha = -alpha;
    }

    let tmp = dot3(&yp, &c2[1]);
    let mut gamma = if (tmp - 1.0).abs() < 1e-12 {
        1.0_f64.acos()
    } else if (tmp + 1.0).abs() < 1e-12 {
        (-1.0_f64).acos()
    } else {
        tmp.acos()
    };
    if dot3(&cross3(&yp, &c2[1]), &c2[2]) < 0.0 {
        gamma = -gamma;
    }

    (alpha, beta, gamma)
}

fn round_to(x: f64, ndigits: i32) -> f64 {
    let f = 10f64.powi(ndigits);
    (x * f).round() / f
}

/// `pyscf/symm/Dmatrix.py:29-49` — `Dmatrix(l, alpha, beta, gamma, reorder_p)`.
fn dmatrix(l: usize, alpha: f64, beta: f64, gamma: f64, reorder_p: bool) -> Vec<Vec<f64>> {
    if l == 0 {
        return vec![vec![1.0]];
    }
    let n = 2 * l + 1;
    let li = l as i64;
    let d = dmatrix_small_d(l, beta);
    let mut dc = vec![vec![Complex64::new(0.0, 0.0); n]; n];
    for i in 0..n {
        let m1 = i as i64 - li;
        let ea = Complex64::from_polar(1.0, -alpha * m1 as f64);
        for j in 0..n {
            let m2 = j as i64 - li;
            let eg = Complex64::from_polar(1.0, -gamma * m2 as f64);
            dc[i][j] = ea * Complex64::new(d[i][j], 0.0) * eg;
        }
    }
    let u = sph_pure2real(l);
    let uh = cdagger_square(&u);
    let real_mat = cmatmul_square(&cmatmul_square(&uh, &dc), &u);
    let mut d_real: Vec<Vec<f64>> = real_mat
        .iter()
        .map(|row| row.iter().map(|c| c.re).collect())
        .collect();
    if reorder_p && l == 1 {
        let perm = [2usize, 0, 1];
        let mut out = vec![vec![0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = d_real[perm[i]][perm[j]];
            }
        }
        d_real = out;
    }
    d_real
}

/// `symmetry.py:32-54` — `get_Dmat`: the Wigner-D matrix for angular
/// momentum `l` under a CARTESIAN rotation `op` (3x3, orthogonal, `det = ±1`
/// — typically `op.a2r(cell).rot`).
pub fn get_dmat(op: &[[f64; 3]; 3], l: usize) -> Vec<Vec<f64>> {
    let mut op2 = *op;
    let mut fac = 1.0_f64;
    let det_op = pyscf_pbc_tools::mat3::det3(&op2);
    if det_op < 0.0 {
        debug_assert!(
            (det_op + 1.0).abs() < 1e-9,
            "improper rotation must have det = -1, got {det_op}"
        );
        for row in op2.iter_mut() {
            for v in row.iter_mut() {
                *v = -*v;
            }
        }
        fac = if l % 2 == 1 { -1.0 } else { 1.0 };
    }
    // c1 = XYZ (fixed); c2 = (op @ c1.T).T == op.T when c1 == I.
    let c2 = pyscf_pbc_tools::mat3::transpose3(&op2);
    let (alpha, beta, gamma) = get_euler_angles(&XYZ, &c2);
    let d = dmatrix(l, alpha, beta, gamma, true);
    d.into_iter()
        .map(|row| row.into_iter().map(|v| round_to(fac * v, 15)).collect())
        .collect()
}

fn cartesian_prod_012(l: usize) -> Vec<Vec<usize>> {
    if l == 0 {
        return vec![vec![]];
    }
    let total = 3usize.pow(l as u32);
    (0..total)
        .map(|n| {
            (0..l)
                .map(|k| (n / 3usize.pow((l - 1 - k) as u32)) % 3)
                .collect()
        })
        .collect()
}

/// `symmetry.py:56-77` — `get_Dmat_cart`: the CARTESIAN analogue of
/// [`get_dmat`], returned as `Ds[l]` for `l` from `0` to `l_max`.
pub fn get_dmat_cart(op: &[[f64; 3]; 3], l_max: usize) -> DmatSet {
    let pp = get_dmat(op, 1); // 3x3, (x,y,z) order (reorder_p=true).
    let mut ds: DmatSet = vec![vec![vec![1.0]]];
    for l in 1..=l_max {
        let mut cidx = cartesian_prod_012(l);
        for row in cidx.iter_mut() {
            row.sort_unstable();
        }
        let mut addr = vec![0usize; cidx.len()];
        let mut affine: Vec<Vec<f64>> = vec![vec![1.0]];
        for i in 0..l {
            let old = affine;
            let old_n = old.len();
            let nd = old_n * 3;
            let mut new_affine = vec![vec![0.0_f64; nd]; nd];
            for a in 0..old_n {
                for x in 0..3 {
                    for b in 0..old_n {
                        for y in 0..3 {
                            new_affine[a * 3 + x][b * 3 + y] = old[a][b] * pp[x][y];
                        }
                    }
                }
            }
            affine = new_affine;
            for (row_i, tuple) in cidx.iter().enumerate() {
                addr[row_i] = addr[row_i] * 3 + tuple[i];
            }
        }
        let mut uniq_addr: Vec<usize> = addr.clone();
        uniq_addr.sort_unstable();
        uniq_addr.dedup();
        let rev_addr: Vec<usize> = addr
            .iter()
            .map(|a| uniq_addr.binary_search(a).expect("addr must be in uniq_addr"))
            .collect();
        let ncart = (l + 1) * (l + 2) / 2;
        debug_assert_eq!(ncart, uniq_addr.len());
        let mut trans = vec![vec![0.0_f64; ncart]; ncart];
        for (i, &k) in rev_addr.iter().enumerate() {
            for (col, &ua) in uniq_addr.iter().enumerate() {
                trans[k][col] += affine[i][ua];
            }
        }
        ds.push(trans);
    }
    ds
}

/// `symmetry.py:79-94` — `make_Dmats`: `Dmats` for every op in `ops`
/// (Cartesian rotations, e.g. `op.a2r(cell).rot`), at every `l` up to
/// `l_max` (or `cell`'s own maximum shell angular momentum, if higher).
///
/// Returns `(Dmats, l_max)` — `Dmats[iop]` is one [`DmatSet`] per input op,
/// same order.
pub fn make_dmats(
    cell: &Cell,
    ops: &[[[f64; 3]; 3]],
    l_max: Option<usize>,
) -> (Vec<DmatSet>, usize) {
    let nbas = cell.nbas;
    let bas_lmax = (0..nbas)
        .map(|ib| cell._bas[ib * BAS_SLOTS + ANG_OF] as usize)
        .max()
        .unwrap_or(0);
    let l_max = match l_max {
        Some(l) => l.max(bas_lmax),
        None => bas_lmax,
    };
    let dmats = ops
        .iter()
        .map(|op| {
            if !cell.cart {
                (0..=l_max).map(|l| get_dmat(op, l)).collect()
            } else {
                get_dmat_cart(op, l_max)
            }
        })
        .collect();
    (dmats, l_max)
}

// ---------------------------------------------------------------------
// Task 3 (continued) — make_rot_loc (symmetry.py:330-343)
// ---------------------------------------------------------------------

/// `symmetry.py:330-343` — `make_rot_loc`: cumulative per-shell offsets into
/// a flat table of `l`-blocks, each `dim(l)^2` wide (`dim(l) = 2l+1` sph,
/// `(l+1)(l+2)/2` cart).
///
/// **Not used by [`get_rotation_mat`]** — see this module's top doc comment
/// on why the AO rotation has exactly one assembly path, walking `cell`'s
/// actual shells directly (as upstream's own `_get_rotation_mat` does; it
/// does not call `make_rot_loc` either). Ported here because 17-04's
/// `symm_adapted_basis` (`basis.py:109`) needs it.
pub fn make_rot_loc(l_max: usize, cart: bool) -> Vec<i32> {
    let dims: Vec<i64> = (0..=l_max)
        .map(|l| {
            let d = if cart {
                ((l + 1) * (l + 2) / 2) as i64
            } else {
                (2 * l + 1) as i64
            };
            d * d
        })
        .collect();
    let mut rot_loc = vec![0i32; dims.len() + 1];
    for (i, &d) in dims.iter().enumerate() {
        rot_loc[i + 1] = rot_loc[i] + d as i32;
    }
    rot_loc
}

// ---------------------------------------------------------------------
// Task 4 — check_mesh_symmetry (symmetry.py:96-131)
// ---------------------------------------------------------------------

/// `symmetry.py:96-131` — `check_mesh_symmetry`. Delegates to
/// [`pyscf_pbc_gto::check_mesh_symmetry_core`] — see that function's doc for
/// why the core algorithm lives one crate down (D-PBC-25 layering) and is
/// shared with [`pyscf_pbc_gto::Cell::symmetrize_mesh`].
///
/// `return_mesh = false` (upstream's default) additionally emits upstream's
/// `logger.warn` (`:124-126`) when `rm_list` is non-empty, as a
/// `tracing::warn!`.
pub fn check_mesh_symmetry(
    cell: &Cell,
    ops: &[SPGElement],
    mesh: Option<[usize; 3]>,
    tol: f64,
    return_mesh: bool,
) -> (Vec<usize>, Option<[usize; 3]>) {
    let mesh = mesh.unwrap_or(cell.mesh);
    let core_ops: Vec<(bool, [f64; 3])> =
        ops.iter().map(|op| (op.trans_is_zero(), op.trans)).collect();
    let (rm_list, mesh1) =
        pyscf_pbc_gto::check_mesh_symmetry_core(&core_ops, mesh, tol);
    if !rm_list.is_empty() && !return_mesh {
        tracing::warn!(
            "Input mesh {mesh:?} has lower symmetry than the lattice.\n\
             Some of the symmetry operations will be removed.\n\
             Recommended mesh is {mesh1:?}."
        );
    }
    if return_mesh {
        (rm_list, Some(mesh1))
    } else {
        (rm_list, None)
    }
}

// ---------------------------------------------------------------------
// Task 5 — Symmetry (symmetry.py:132-224)
// ---------------------------------------------------------------------

/// `symmetry.py:132-224` — `Symmetry`. See the module top doc for why this
/// is built from a BORROWED `&Cell` (§3.9) and stores no `Cell` at all.
#[derive(Debug, Clone)]
pub struct Symmetry {
    pub spacegroup: Option<SpaceGroup>,
    pub symmorphic: bool,
    pub ops: Vec<SPGElement>,
    pub nop: usize,
    pub has_inversion: bool,
    pub dmats: Vec<DmatSet>,
    pub l_max: usize,
    pub built: bool,
}

impl Default for Symmetry {
    /// `symmetry.py:154-163` — `__init__`.
    fn default() -> Self {
        let ops = vec![SPGElement::default()];
        let nop = ops.len();
        Self {
            spacegroup: None,
            symmorphic: true,
            ops,
            nop,
            has_inversion: false,
            dmats: Vec::new(),
            l_max: 0,
            built: false,
        }
    }
}

impl Symmetry {
    /// `symmetry.py:165-207` — `build`.
    ///
    /// Upstream's `cell` argument to the surrounding `Symmetry` is implicit
    /// (`self.cell`, set at `__init__`); this port takes it explicitly and
    /// borrows it for the DURATION OF THIS CALL ONLY — nothing here is
    /// retained past the return (§3.9).
    ///
    /// The `auxcell` kwarg (`symmetry.py:200-203`, used to widen `l_max` to
    /// an auxiliary density-fitting basis) is not ported: nothing in this
    /// workspace calls `Symmetry.build(auxcell=...)` yet. A future caller
    /// that needs a wider `l_max` can call [`make_dmats`] again directly with
    /// `Some(l_max)`.
    ///
    /// # Errors
    /// * The cell is not built ([`PbcSymmError::Core`] wrapping an
    ///   `InvalidMolecule`) — upstream silently calls `cell.build()` here
    ///   (`:176-178`); this port refuses instead, since mutating a borrowed
    ///   `&Cell` is not possible and silently no-op'ing would be worse.
    /// * [`SpaceGroup::build`] / [`SPGElement::a2r`] failures.
    pub fn build(
        cell: &Cell,
        space_group_symmetry: bool,
        symmorphic: bool,
        check_mesh_symmetry_flag: bool,
    ) -> Result<Self, PbcSymmError> {
        let mut symmorphic = symmorphic;
        let mut spacegroup = None;
        let ops: Vec<SPGElement>;

        if !space_group_symmetry {
            ops = vec![SPGElement::default()];
        } else {
            if !cell._built {
                return Err(PbcSymmError::Core(pyscf_core::PyscfRsError::Core(
                    pyscf_core::CoreError::InvalidMolecule(
                        "Symmetry::build: cell must be built before Symmetry::build \
                         (upstream silently calls cell.build() here; this port cannot \
                         mutate a borrowed &Cell)"
                            .into(),
                    ),
                )));
            }
            let sg = SpaceGroup::build(cell, SYMPREC)?;
            if cell.dimension < 3 && !symmorphic {
                tracing::warn!("setting symmorphic=True for low-dimensional system.");
                symmorphic = true;
            }
            let all_ops = &sg.ops;
            ops = if symmorphic {
                all_ops.iter().filter(|op| op.trans_is_zero()).copied().collect()
            } else if check_mesh_symmetry_flag {
                let (rm_list, _) = check_mesh_symmetry(cell, all_ops, None, SYMPREC, false);
                all_ops
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !rm_list.contains(i))
                    .map(|(_, op)| *op)
                    .collect()
            } else {
                all_ops.clone()
            };
            spacegroup = Some(sg);
        }

        let nop = ops.len();
        let has_inversion = ops.iter().any(|op| op.rot_is_inversion());

        let mut op_rot = Vec::with_capacity(ops.len());
        for op in &ops {
            op_rot.push(op.a2r(cell)?.rot);
        }
        let (dmats, l_max) = make_dmats(cell, &op_rot, None);

        Ok(Self {
            spacegroup,
            symmorphic,
            ops,
            nop,
            has_inversion,
            dmats,
            l_max,
            built: true,
        })
    }

    /// `symmetry.py:209-215` — `check_mesh_symmetry` (the `Symmetry` method).
    /// Upstream defaults `cell=None -> self.cell` and `ops=None -> self.ops`;
    /// this port takes `cell` explicitly (§3.9: `Symmetry` does not store
    /// one) and always uses `self.ops` (no caller in this workspace passes a
    /// different `ops`).
    pub fn check_mesh_symmetry(
        &self,
        cell: &Cell,
        mesh: Option<[usize; 3]>,
        tol: f64,
        return_mesh: bool,
    ) -> (Vec<usize>, Option<[usize; 3]>) {
        check_mesh_symmetry(cell, &self.ops, mesh, tol, return_mesh)
    }

    /// `symmetry.py:217-218` — `dump_info`.
    pub fn dump_info(&self) {
        if let Some(sg) = &self.spacegroup {
            sg.dump_info(Some(&self.ops));
        }
    }

    /// `symmetry.py:220-223` — `reset`.
    pub fn reset(&mut self) -> &mut Self {
        self.spacegroup = None;
        self.built = false;
        self
    }

    /// The inverse of [`Symmetry::to_lattice_symmetry`]: rebuild a
    /// [`Symmetry`] from the plain-data form a `Cell` carries.
    ///
    /// This is what upstream's `KPoints.build` does with
    /// `self.__dict__.update(_lattice_symm.__dict__)` (`kpts.py:1019-1021`)
    /// — reuse the symmetry the `Cell` already built (with its own
    /// `check_mesh_symmetry` decision, `cell.py:1771-1772`) rather than
    /// re-running the space-group search. `spacegroup` is NOT recovered
    /// (a `LatticeSymmetry` does not carry a [`SpaceGroup`]); it is only
    /// read by [`Symmetry::dump_info`], which degrades to silence.
    pub fn from_lattice_symmetry(ls: &pyscf_pbc_gto::LatticeSymmetry) -> Self {
        let ops: Vec<SPGElement> = ls
            .ops
            .iter()
            .map(|o| SPGElement::from_int(o.rot, o.trans))
            .collect();
        let nop = ops.len();
        Self {
            spacegroup: None,
            symmorphic: ls.symmorphic,
            ops,
            nop,
            has_inversion: ls.has_inversion,
            dmats: ls.dmats.clone(),
            l_max: ls.l_max,
            built: true,
        }
    }

    /// Convert to the plain-data [`pyscf_pbc_gto::LatticeSymmetry`] a `Cell`
    /// can actually store — see this module's top doc and
    /// [`pyscf_pbc_gto::symmetry_data`]'s module doc for the layering
    /// reason `Cell` cannot hold a `Symmetry` directly.
    pub fn to_lattice_symmetry(&self) -> pyscf_pbc_gto::LatticeSymmetry {
        let point_group_symbol = self
            .spacegroup
            .as_ref()
            .map(|sg| sg.point_group_symbol.to_string())
            .unwrap_or_default();
        let ops = self
            .ops
            .iter()
            .map(|op| pyscf_pbc_gto::LatticeSymmetryOp {
                rot: std::array::from_fn(|i| std::array::from_fn(|j| op.rot[i][j].round() as i32)),
                trans: op.trans,
            })
            .collect();
        pyscf_pbc_gto::LatticeSymmetry {
            symmorphic: self.symmorphic,
            has_inversion: self.has_inversion,
            ops,
            dmats: self.dmats.clone(),
            l_max: self.l_max,
            point_group_symbol,
        }
    }
}

/// `cell.py:1552-1580` — `Cell.build_lattice_symmetry`, as a FREE FUNCTION
/// rather than a `Cell` method: `Cell` lives in `pyscf-pbc-gto`, which sits
/// BELOW this crate (D-PBC-25), so it cannot call into [`Symmetry::build`]
/// itself. This function lives here instead, where both types are visible,
/// and is the intended entry point for anything that wants to populate
/// [`pyscf_pbc_gto::Cell::lattice_symmetry`].
///
/// Mirrors upstream except for the `del self.lattice_symmetry.cell` /
/// `del self.lattice_symmetry.spacegroup.cell` lines (`:1576-1579`) — see the
/// module top doc: that deletion breaks a Python refcount cycle that cannot
/// exist here, since [`Symmetry::build`] never stored a `Cell` reference to
/// begin with.
///
/// # Errors
/// As [`Symmetry::build`].
pub fn build_lattice_symmetry(
    cell: &mut Cell,
    check_mesh_symmetry_flag: bool,
) -> Result<(), PbcSymmError> {
    let symmetry = Symmetry::build(cell, true, cell.symmorphic, check_mesh_symmetry_flag)?;
    cell.lattice_symmetry = Some(symmetry.to_lattice_symmetry());
    if !check_mesh_symmetry_flag {
        let mesh_from_build = cell._mesh_from_build;
        cell.mesh = cell.symmetrize_mesh(None);
        cell._mesh_from_build = mesh_from_build;
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Task 6 — _get_phase / _get_rotation_mat / the three transforms
// (symmetry.py:226-329)
// ---------------------------------------------------------------------

/// `pub(crate)`: reused by `crate::basis` (17-04) — see that module's doc.
pub(crate) fn bas_angular(cell: &Cell, bas_id: usize) -> usize {
    cell._bas[bas_id * BAS_SLOTS + ANG_OF] as usize
}

/// `pub(crate)`: reused by `crate::basis` (17-04) — see that module's doc.
pub(crate) fn bas_nctr(cell: &Cell, bas_id: usize) -> usize {
    cell._bas[bas_id * BAS_SLOTS + NCTR_OF] as usize
}

fn bas_atom(cell: &Cell, bas_id: usize) -> usize {
    cell._bas[bas_id * BAS_SLOTS + ATOM_OF] as usize
}

/// `pyscf/gto/mole.py:1841-1880` — `aoslice_by_atom`: per-atom `[shell_start,
/// shell_end, ao_start, ao_end]`. Not exposed on `Cell`/`Mole` yet elsewhere
/// in this workspace, so ported here as a `pub(crate)` helper — this
/// module's own consumer is [`get_rotation_mat`]; `crate::basis` (17-04)
/// reuses it too rather than porting a second copy (17-CONTEXT §3.2's "one
/// implementation" discipline).
pub(crate) fn aoslice_by_atom(cell: &Cell) -> Vec<[usize; 4]> {
    let natm = cell.natm;
    let nbas = cell.nbas;
    if natm == 0 {
        return Vec::new();
    }
    let bas_atom_of: Vec<usize> = (0..nbas).map(|ib| bas_atom(cell, ib)).collect();
    let mut delimiter: Vec<usize> = Vec::new();
    for i in 0..nbas.saturating_sub(1) {
        if bas_atom_of[i] != bas_atom_of[i + 1] {
            delimiter.push(i + 1);
        }
    }
    let mut shell_start = vec![usize::MAX; natm];
    let mut shell_end = vec![0usize; natm];
    if natm == delimiter.len() + 1 {
        shell_start[0] = 0;
        for (k, &d) in delimiter.iter().enumerate() {
            shell_start[k + 1] = d;
        }
        for (k, &d) in delimiter.iter().enumerate() {
            shell_end[k] = d;
        }
        shell_end[natm - 1] = nbas;
    } else {
        // Some atoms miss basis functions entirely.
        shell_start[0] = 0;
        if nbas > 0 {
            shell_start[bas_atom_of[0]] = 0;
        }
        for &d in &delimiter {
            shell_start[bas_atom_of[d]] = d;
        }
        shell_end[0] = 0;
        for &d in &delimiter {
            shell_end[bas_atom_of[d - 1]] = d;
        }
        if nbas > 0 {
            shell_end[bas_atom_of[nbas - 1]] = nbas;
        }
        for i in 1..natm {
            if shell_start[i] == usize::MAX {
                shell_start[i] = shell_end[i - 1];
                shell_end[i] = shell_end[i - 1];
            }
        }
    }
    let ao_loc = &cell.ao_loc_nr;
    (0..natm)
        .map(|i| {
            [
                shell_start[i],
                shell_end[i],
                ao_loc[shell_start[i]] as usize,
                ao_loc[shell_end[i]] as usize,
            ]
        })
        .collect()
}

/// `symmetry.py:226-248` — `_get_phase`.
pub fn get_phase(
    cell: &Cell,
    op: &SPGElement,
    kpt_scaled: [f64; 3],
    ignore_phase: bool,
    tol: f64,
) -> Result<(Vec<usize>, Vec<Complex64>), PbcSymmError> {
    let kpt_scaled = op.a2b(cell)?.dot_rot(&kpt_scaled);
    let coords_scaled = cell.get_scaled_atom_coords()?;
    let natm = coords_scaled.len();
    let mut phase = vec![Complex64::new(1.0, 0.0); natm];
    let mut atm_map = vec![0usize; natm];
    let coords0 = pyscf_pbc_tools::round_to_cell0(&coords_scaled, tol);
    for iatm in 0..natm {
        let r = coords_scaled[iatm];
        let rot_r = op.dot_rot(&r);
        let op_dot_r = [
            rot_r[0] + op.trans[0],
            rot_r[1] + op.trans[1],
            rot_r[2] + op.trans[2],
        ];
        let op_dot_r_0 = pyscf_pbc_tools::round_to_cell0(&[op_dot_r], tol)[0];
        let matches: Vec<usize> = (0..natm)
            .filter(|&j| {
                let d = (op_dot_r_0[0] - coords0[j][0]).abs()
                    + (op_dot_r_0[1] - coords0[j][1]).abs()
                    + (op_dot_r_0[2] - coords0[j][2]).abs();
                d < tol
            })
            .collect();
        if matches.len() != 1 {
            return Err(PbcSymmError::AtomMapMismatch(iatm, matches.len()));
        }
        let equiv_atm = matches[0];
        atm_map[iatm] = equiv_atm;
        let lshift = [
            coords_scaled[equiv_atm][0] - op_dot_r[0],
            coords_scaled[equiv_atm][1] - op_dot_r[1],
            coords_scaled[equiv_atm][2] - op_dot_r[2],
        ];
        let resid: f64 = lshift.iter().map(|v| (v - v.round()).abs()).sum();
        if resid >= tol {
            return Err(PbcSymmError::NonLatticeShift(iatm));
        }
        let lshift_r = [lshift[0].round(), lshift[1].round(), lshift[2].round()];
        if !ignore_phase {
            let dot =
                kpt_scaled[0] * lshift_r[0] + kpt_scaled[1] * lshift_r[1] + kpt_scaled[2] * lshift_r[2];
            phase[iatm] = Complex64::from_polar(1.0, dot * 2.0 * std::f64::consts::PI);
        }
    }
    Ok((atm_map, phase))
}

/// `symmetry.py:250-292` — `_get_rotation_mat`. Row-major `dim x dim`
/// (`dim = cell.nao_nr`, unless `ignore_phase` is used at a smaller `dim` —
/// this port always uses the full AO count, matching every call site in this
/// crate). See the module top doc: this is the ONE AO-rotation assembly this
/// crate has.
///
/// # Errors
/// As [`get_phase`], plus a shell-layout mismatch between an atom and its
/// symmetry-equivalent image ([`PbcSymmError::ShellLayoutMismatch`] — a
/// debug-only sanity check upstream expresses as a bare `assert`).
pub fn get_rotation_mat(
    cell: &Cell,
    kpt_scaled_ibz: [f64; 3],
    dim: usize,
    op: &SPGElement,
    dmats: &DmatSet,
    ignore_phase: bool,
    tol: f64,
) -> Result<Vec<Complex64>, PbcSymmError> {
    let (atm_map, phases) = get_phase(cell, op, kpt_scaled_ibz, ignore_phase, tol)?;
    let mut mat = vec![Complex64::new(0.0, 0.0); dim * dim];
    let aoslice = aoslice_by_atom(cell);
    for iatm in 0..cell.natm {
        let jatm = atm_map[iatm];
        if iatm != jatm {
            let nao_i = aoslice[iatm][3] - aoslice[iatm][2];
            let nao_j = aoslice[jatm][3] - aoslice[jatm][2];
            let nshl_i = aoslice[iatm][1] - aoslice[iatm][0];
            let nshl_j = aoslice[jatm][1] - aoslice[jatm][0];
            if nao_i != nao_j || nshl_i != nshl_j {
                return Err(PbcSymmError::ShellLayoutMismatch(iatm, jatm));
            }
            for ishl in 0..nshl_i {
                let l_i = bas_angular(cell, aoslice[iatm][0] + ishl);
                let l_j = bas_angular(cell, aoslice[jatm][0] + ishl);
                if l_i != l_j {
                    return Err(PbcSymmError::ShellLayoutMismatch(iatm, jatm));
                }
            }
        }
        let phase = phases[iatm];
        let mut ao_off_i = aoslice[iatm][2];
        let mut ao_off_j = aoslice[jatm][2];
        let shlid0 = aoslice[iatm][0];
        let shlid1 = aoslice[iatm][1];
        for ishl in shlid0..shlid1 {
            let l = bas_angular(cell, ishl);
            let nao = if cell.cart { (l + 1) * (l + 2) / 2 } else { 2 * l + 1 };
            let nc = bas_nctr(cell, ishl);
            for _ in 0..nc {
                for r in 0..nao {
                    for c in 0..nao {
                        mat[(ao_off_j + r) * dim + (ao_off_i + c)] = dmats[l][r][c] * phase;
                    }
                }
                ao_off_i += nao;
                ao_off_j += nao;
            }
        }
        debug_assert_eq!(ao_off_i, aoslice[iatm][3]);
        debug_assert_eq!(ao_off_j, aoslice[jatm][3]);
    }
    Ok(mat)
}

fn cmatmul(
    a: &[Complex64],
    ar: usize,
    ac: usize,
    b: &[Complex64],
    _br: usize,
    bc: usize,
) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); ar * bc];
    for i in 0..ar {
        for k in 0..ac {
            let aik = a[i * ac + k];
            if aik.re == 0.0 && aik.im == 0.0 {
                continue;
            }
            for j in 0..bc {
                out[i * bc + j] += aik * b[k * bc + j];
            }
        }
    }
    out
}

fn cdagger(a: &[Complex64], nrow: usize, ncol: usize) -> Vec<Complex64> {
    let mut out = vec![Complex64::new(0.0, 0.0); ncol * nrow];
    for i in 0..nrow {
        for j in 0..ncol {
            out[j * nrow + i] = a[i * ncol + j].conj();
        }
    }
    out
}

/// `mat @ x @ matᴴ` — the sandwich shared by [`transform_dm`] and
/// [`transform_1e_operator`] (upstream literally repeats the same three-line
/// body for both; sharing it here keeps it ONE implementation rather than
/// two that could silently diverge, per the module top doc's §3.2 point).
fn sandwich(mat: &[Complex64], x: &[Complex64], n: usize) -> Vec<Complex64> {
    let tmp = cmatmul(mat, n, n, x, n, n);
    let math = cdagger(mat, n, n);
    cmatmul(&tmp, n, n, &math, n, n)
}

/// `symmetry.py:294-314` — `transform_mo_coeff`. `mo_coeff` is row-major
/// `nao x nmo` (the CALLER converts from `pyscf-pbc-scf`'s column-major
/// storage — see `crate::symmetry`'s top doc and 17-CONTEXT §3.2).
///
/// # Errors
/// As [`get_rotation_mat`].
pub fn transform_mo_coeff(
    cell: &Cell,
    kpt_scaled: [f64; 3],
    mo_coeff: &[Complex64],
    nao: usize,
    nmo: usize,
    op: &SPGElement,
    dmats: &DmatSet,
) -> Result<Vec<Complex64>, PbcSymmError> {
    let mat = get_rotation_mat(cell, kpt_scaled, nao, op, dmats, false, SYMPREC)?;
    Ok(cmatmul(&mat, nao, nao, mo_coeff, nao, nmo))
}

/// `symmetry.py:316-321` — `transform_dm`. `dm` is row-major `nao x nao`.
///
/// # Errors
/// As [`get_rotation_mat`].
pub fn transform_dm(
    cell: &Cell,
    kpt_scaled: [f64; 3],
    dm: &[Complex64],
    nao: usize,
    op: &SPGElement,
    dmats: &DmatSet,
) -> Result<Vec<Complex64>, PbcSymmError> {
    let mat = get_rotation_mat(cell, kpt_scaled, nao, op, dmats, false, SYMPREC)?;
    Ok(sandwich(&mat, dm, nao))
}

/// `symmetry.py:323-328` — `transform_1e_operator`. `fock` is row-major
/// `nao x nao`. Identical sandwich transform to [`transform_dm`] — see
/// [`sandwich`].
///
/// # Errors
/// As [`get_rotation_mat`].
pub fn transform_1e_operator(
    cell: &Cell,
    kpt_scaled: [f64; 3],
    fock: &[Complex64],
    nao: usize,
    op: &SPGElement,
    dmats: &DmatSet,
) -> Result<Vec<Complex64>, PbcSymmError> {
    let mat = get_rotation_mat(cell, kpt_scaled, nao, op, dmats, false, SYMPREC)?;
    Ok(sandwich(&mat, fock, nao))
}

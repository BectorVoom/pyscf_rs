//! Port of `pyscf/pbc/symm/geom.py` (245 l) — lattice point-group / space-group
//! detection and crystal-class classification.
//!
//! `search_point_group_ops` (`geom.py:27-77`) is a brute-force search over the
//! `19_683 = 3^9` integer `3x3` matrices with entries in `{1, 0, -1}`, kept
//! when they preserve the lattice metric. Three things are load-bearing here
//! (17-CONTEXT §3.6/§3.7, `17-02-PLAN.md` Task 1):
//!
//! 1. **Iteration order.** Upstream drives the search with
//!    `lib.cartesian_prod([[1, 0, -1]] * 9)`, which varies the LAST of the
//!    nine axes fastest. Survivors are appended in that order, and the order
//!    is observable downstream through `stars_ops`/`finger(kpts_ibz)`
//!    (17-05). [`candidate_rotations`] reproduces it with a base-3 counter —
//!    no `HashSet`, no reordering.
//! 2. **The `np.clip` before `arccos`** (`geom.py:38-41`, `:57-59`, upstream
//!    issue 3113): on a cubic lattice a diagonal metric ratio exceeds 1 by
//!    rounding, and an unclipped `arccos` returns `NaN`, which silently fails
//!    the `>` comparison and admits a WRONG rotation.
//! 3. **The low-dimension filters** (`:65-72`): reject any `W` that inverts a
//!    non-periodic axis, or that couples a periodic axis to a non-periodic
//!    one. These are the only reason `graphene` (`dimension = 2`) works.

use pyscf_pbc_gto::Cell;

use crate::error::PbcSymmError;
use crate::tables;

/// `geom.py:25` — `SYMPREC`. Units of length (Bohr, since `Cell::lattice_vectors`
/// is always Bohr).
pub const SYMPREC: f64 = 1e-6;

/// A `3x3` integer rotation matrix expressed in the lattice-translation-vector
/// basis (`geom.py`'s bare `np.ndarray`, `group.py`'s `PGElement.matrix`).
pub type RotMatrix = [[i32; 3]; 3];

/// Minimal stand-in for upstream's `pyscf.pbc.symm.space_group.SPGElement`
/// (`space_group.py:82-99`), which plan 17-03 ports in full. `geom.py:143`
/// only ever constructs one with a rotation and a fractional translation, so
/// that is all this plan needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpaceGroupOp {
    pub rot: RotMatrix,
    pub trans: [f64; 3],
}

// ---------------------------------------------------------------------
// small 3x3 linear algebra helpers (int and float)
// ---------------------------------------------------------------------

fn mat3_f64(w: &RotMatrix) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = w[i][j] as f64;
        }
    }
    out
}

/// `a @ b` for `3x3` float matrices.
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

fn transpose3(a: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut out = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            out[i][j] = a[j][i];
        }
    }
    out
}

/// `W^T @ G @ W`.
fn conjugate_metric(w: &RotMatrix, g: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let wf = mat3_f64(w);
    let wt = transpose3(&wf);
    matmul3(&matmul3(&wt, g), &wf)
}

/// `R @ v` (matrix-vector product), `v` a Cartesian/fractional column vector.
fn matvec3(r: &[[f64; 3]; 3], v: &[f64; 3]) -> [f64; 3] {
    let mut out = [0.0; 3];
    for i in 0..3 {
        out[i] = r[i][0] * v[0] + r[i][1] * v[1] + r[i][2] * v[2];
    }
    out
}

/// Exact integer determinant of a `3x3` matrix (cofactor expansion).
fn det3_i32(w: &RotMatrix) -> i32 {
    w[0][0] * (w[1][1] * w[2][2] - w[1][2] * w[2][1])
        - w[0][1] * (w[1][0] * w[2][2] - w[1][2] * w[2][0])
        + w[0][2] * (w[1][0] * w[2][1] - w[1][1] * w[2][0])
}

// ---------------------------------------------------------------------
// Task 1 — search_point_group_ops (geom.py:27-77)
// ---------------------------------------------------------------------

/// `geom.py:45` — `lib.cartesian_prod([[1, 0, -1],] * 9)` in ITS EXACT
/// ITERATION ORDER: the first of the nine axes varies slowest, the last
/// varies fastest (`cartesian_prod` reshapes with the last input axis
/// fastest — see `pyscf/lib/numpy_helper.py:1094-1133`). Reproduced here as
/// a base-3 counter over `n in 0..3^9`, `VALUES = [1, 0, -1]` read
/// most-significant-digit-first, which is exactly `itertools.product` /
/// `cartesian_prod`'s row order for nine copies of the same 3-element array.
///
/// No `HashSet` anywhere: survivors are pushed in this exact enumeration
/// order because that order is observable through `stars_ops` /
/// `finger(kpts_ibz)` downstream (17-CONTEXT §3.6).
fn candidate_rotations() -> impl Iterator<Item = RotMatrix> {
    const VALUES: [i32; 3] = [1, 0, -1];
    (0..3_usize.pow(9)).map(|n| {
        let mut w = [[0_i32; 3]; 3];
        for k in 0..9 {
            // digit k: most-significant (k=0) has place value 3^8, least
            // significant (k=8) has place value 3^0 — matches
            // `lib.cartesian_prod`'s "last axis fastest".
            let digit = (n / 3_usize.pow(8 - k as u32)) % 3;
            w[k / 3][k % 3] = VALUES[digit];
        }
        w
    })
}

/// `geom.py:27-77` — brute-force search for the point-group operations
/// (integer rotation matrices, in the lattice basis) that preserve the
/// cell's metric.
pub fn search_point_group_ops(cell: &Cell, tol: f64) -> Result<Vec<RotMatrix>, PbcSymmError> {
    let a = cell.lattice_vectors();
    // G = a @ a.T
    let at = transpose3(&a);
    let g = matmul3(&a, &at);

    let mut pbc_axis = [true, true, true];
    if cell.dimension < 3 {
        for axis in pbc_axis.iter_mut().skip(cell.dimension as usize) {
            *axis = false;
        }
    }

    let a_norm: [f64; 3] = std::array::from_fn(|i| g[i][i].sqrt());

    // issue 3113: diagonal terms may slightly exceed [-1, 1] due to rounding
    // errors, so unclipped arccos returns NaN and the survivor check below
    // would silently and incorrectly fail. Clip before arccos.
    let mut a_angle = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            let ratio = (g[i][j] / (a_norm[i] * a_norm[j])).clamp(-1.0, 1.0);
            a_angle[i][j] = ratio.acos();
        }
    }
    let tol2 = tol * tol;

    let mut rotations = Vec::new();
    for w in candidate_rotations() {
        let g_tilde = conjugate_metric(&w, &g);

        // check change of metric
        let a_tilde_norm: [f64; 3] = std::array::from_fn(|i| g_tilde[i][i].sqrt());
        let length_error: [f64; 3] = std::array::from_fn(|i| (a_norm[i] - a_tilde_norm[i]).abs());
        if length_error.iter().any(|&e| e > tol) {
            continue;
        }
        let tmp: [f64; 3] = std::array::from_fn(|i| a_norm[i] + a_tilde_norm[i]);

        // issue 3113, again.
        let mut a_tilde_angle = [[0.0_f64; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                let ratio =
                    (g_tilde[i][j] / (a_tilde_norm[i] * a_tilde_norm[j])).clamp(-1.0, 1.0);
                a_tilde_angle[i][j] = ratio.acos();
            }
        }
        let mut angle_ok = true;
        'angle: for i in 0..3 {
            for j in 0..3 {
                let angle_error =
                    (a_angle[i][j] - a_tilde_angle[i][j]).sin().powi(2) * tmp[i] * tmp[j] / 4.0;
                if angle_error > tol2 {
                    angle_ok = false;
                    break 'angle;
                }
            }
        }
        if !angle_ok {
            continue;
        }

        // check if rotation inverts non-periodic axes
        // (`W[np.diag(~pbc_axis)] == 1`: the diagonal entries at
        // non-periodic axis positions must be exactly +1.)
        let mut ok = true;
        for i in 0..3 {
            if !pbc_axis[i] && w[i][i] != 1 {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }

        // check if rotation swaps periodic and non-periodic axes
        // (off-diagonal entries are forbidden unless both axes are periodic)
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    continue;
                }
                let both_periodic = pbc_axis[i] && pbc_axis[j];
                if !both_periodic && w[i][j] != 0 {
                    ok = false;
                }
            }
        }
        if !ok {
            continue;
        }

        rotations.push(w);
    }

    Ok(rotations)
}

// ---------------------------------------------------------------------
// Task 2 — search_space_group_ops / get_crystal_class (geom.py:79-245)
// ---------------------------------------------------------------------

/// `pyscf/pbc/symm/pyscf_spglib.py:36-38` refuses ghost atoms outright
/// rather than reproducing `mole.atom_types`'s silent `'GHOST' -> 'X'`
/// rename (17-CONTEXT §1.5, `17-02-PLAN.md` Task 2).
fn refuse_ghost_atoms(atoms: &[(String, [f64; 3])]) -> Result<(), PbcSymmError> {
    for (symbol, _) in atoms {
        let upper = symbol.to_uppercase();
        if upper.contains("GHOST") || upper.contains("X-") {
            return Err(PbcSymmError::GhostAtomUnsupported(symbol.clone()));
        }
    }
    Ok(())
}

/// `pyscf.gto.mole.atom_types(cell._atom, magmom=cell.magmom)`, restricted to
/// the `basis=None` / no-`magmom` branch: group atom indices by exact symbol
/// string, preserving first-appearance (dict-insertion) order. The `magmom`
/// spin-inversion branch (`geom.py:93-104`, `mole.py:302-316`) is not ported:
/// this crate's `Cell` has no `magmom` field yet (`crate::geom` module docs /
/// 17-02-SUMMARY.md), so `has_spin` is always `false` here, matching
/// upstream's own behaviour when `cell.magmom` is `None`. The ghost-rename
/// branch (`mole.py:281-282`) is likewise not reproduced because
/// [`refuse_ghost_atoms`] refuses ghosts before this function is reached, per
/// the plan's "ghost-atom refusal" requirement.
fn atom_types_no_spin(atoms: &[(String, [f64; 3])]) -> Vec<(String, Vec<usize>)> {
    let mut groups: Vec<(String, Vec<usize>)> = Vec::new();
    for (ia, (symbol, _)) in atoms.iter().enumerate() {
        if let Some(entry) = groups.iter_mut().find(|(k, _)| k == symbol) {
            entry.1.push(ia);
        } else {
            groups.push((symbol.clone(), vec![ia]));
        }
    }
    groups
}

/// `geom.py:114` — `-np.log10(tol).astype(int)`: number of decimal digits to
/// round to. Truncates toward zero, matching numpy's `.astype(int)`.
fn round_ndigits(tol: f64) -> i32 {
    (-tol.log10()).trunc() as i32
}

fn round_to(x: f64, ndigits: i32) -> f64 {
    let f = 10f64.powi(ndigits);
    (x * f).round() / f
}

/// `np.mod(x, 1)` for possibly-negative `x`: result in `[0, 1)`.
fn mod1(x: f64) -> f64 {
    x - x.floor()
}

fn mod1_round_mod1(v: [f64; 3], ndigits: i32) -> [f64; 3] {
    std::array::from_fn(|i| mod1(round_to(mod1(v[i]), ndigits)))
}

/// `geom.py:105-121` — `test_trans`: does rotation `rot` + fractional
/// translation `trans` map the atom set back onto itself (respecting atom
/// type)?
#[allow(clippy::too_many_arguments)]
fn test_trans(
    atmgrp: &[(String, Vec<usize>)],
    coords: &[[f64; 3]],
    a: &[[f64; 3]; 3],
    rot: &[[f64; 3]; 3],
    trans: [f64; 3],
    tol: f64,
    ndigits: i32,
) -> bool {
    for (_atm, idx) in atmgrp {
        let x: Vec<[f64; 3]> = idx.iter().map(|&i| coords[i]).collect();
        let xt: Vec<[f64; 3]> = x
            .iter()
            .map(|xi| {
                let rxi = matvec3(rot, xi);
                [rxi[0] + trans[0], rxi[1] + trans[1], rxi[2] + trans[2]]
            })
            .collect();

        let mut x_xt: Vec<[f64; 3]> = Vec::with_capacity(2 * x.len());
        x_xt.extend(x.iter().copied());
        x_xt.extend(xt.iter().copied());
        for row in x_xt.iter_mut() {
            *row = mod1_round_mod1(*row, ndigits);
        }

        // `np.lexsort(x_xt.T)`: primary key = column 2 (z), then column 1
        // (y), then column 0 (x) — the LAST key given to lexsort is primary.
        x_xt.sort_by(|p, q| {
            p[2].total_cmp(&q[2])
                .then_with(|| p[1].total_cmp(&q[1]))
                .then_with(|| p[0].total_cmp(&q[0]))
        });

        for pair in x_xt.as_chunks::<2>().0 {
            let d_frac = [pair[0][0] - pair[1][0], pair[0][1] - pair[1][1], pair[0][2] - pair[1][2]];
            // `np.dot(diff, a)`: fractional -> Cartesian via row-vector @ lattice.
            let d_cart: [f64; 3] = std::array::from_fn(|j| {
                d_frac[0] * a[0][j] + d_frac[1] * a[1][j] + d_frac[2] * a[2][j]
            });
            if d_cart.iter().any(|&c| c.abs() > tol) {
                return false;
            }
        }
    }
    true
}

/// `geom.py:79-147` — `search_space_group_ops`. `rotations = None` reproduces
/// `search_point_group_ops(cell, tol)`.
pub fn search_space_group_ops(
    cell: &Cell,
    rotations: Option<&[RotMatrix]>,
    tol: f64,
) -> Result<Vec<SpaceGroupOp>, PbcSymmError> {
    refuse_ghost_atoms(&cell._atom)?;

    let owned_rotations;
    let rotations: &[RotMatrix] = match rotations {
        Some(r) => r,
        None => {
            owned_rotations = search_point_group_ops(cell, tol)?;
            &owned_rotations
        }
    };

    let a = cell.lattice_vectors();
    let coords = cell.get_scaled_atom_coords()?;
    let atmgrp = atom_types_no_spin(&cell._atom);
    let ndigits = round_ndigits(tol);

    let grp_len = atmgrp.iter().map(|(_, v)| v.len()).min().unwrap_or(0);
    let smallest = atmgrp
        .iter()
        .find(|(_, v)| v.len() == grp_len)
        .expect("cell must have at least one atom");
    let x: Vec<[f64; 3]> = smallest.1.iter().map(|&i| coords[i]).collect();

    let mut ops = Vec::new();
    for rot in rotations {
        let rotf = mat3_f64(rot);
        let base = matvec3(&rotf, &x[0]);
        let mut w: Vec<[f64; 3]> = x
            .iter()
            .map(|xi| [xi[0] - base[0], xi[1] - base[1], xi[2] - base[2]])
            .collect();
        for wi in w.iter_mut() {
            *wi = mod1_round_mod1(*wi, ndigits);
        }
        // `np.unique(w, axis=0)`: dedupe AND sort ascending, column 0
        // primary (standard row order — NOT the reversed lexsort trick
        // `test_trans` uses).
        w.sort_by(|p, q| {
            p[0].total_cmp(&q[0])
                .then_with(|| p[1].total_cmp(&q[1]))
                .then_with(|| p[2].total_cmp(&q[2]))
        });
        w.dedup_by(|a, b| a == b);

        for trans in w {
            if test_trans(&atmgrp, &coords, &a, &rotf, trans, tol, ndigits) {
                ops.push(SpaceGroupOp { rot: *rot, trans });
            }
        }
    }
    Ok(ops)
}

/// `geom.py:156-198`: `(trace, det)` -> index into the 10-element histogram
/// `[-6, -4, -3, -2, -1, 1, 2, 3, 4, 6]`.
fn rotation_class_index(rot: &RotMatrix) -> Result<usize, PbcSymmError> {
    let trace = rot[0][0] + rot[1][1] + rot[2][2];
    let det = det3_i32(rot);
    let value = match (trace, det) {
        (3, 1) => 1,
        (-3, -1) => -1,
        (2, 1) => 6,
        (-2, -1) => -6,
        (0, 1) => 3,
        (0, -1) => -3,
        (1, 1) => 4,
        (1, -1) => -2,
        (-1, 1) => 2,
        (-1, -1) => -4,
        _ => return Err(PbcSymmError::InvalidRotation(*rot)),
    };
    // maps = {-6:0, -4:1, -3:2, -2:3, -1:4, 1:5, 2:6, 3:7, 4:8, 6:9}
    let idx = match value {
        -6 => 0,
        -4 => 1,
        -3 => 2,
        -2 => 3,
        -1 => 4,
        1 => 5,
        2 => 6,
        3 => 7,
        4 => 8,
        6 => 9,
        _ => unreachable!(),
    };
    Ok(idx)
}

/// `geom.py:149-216`, the part that does not need a `Cell`: classify a set of
/// point-group rotation matrices into `(crystal_class, laue_class)`. Split
/// out from [`get_crystal_class`] so `group::PointGroup::group_name`
/// (`group.py:406`, which calls `geom.get_crystal_class(None, self.elements)`)
/// can reach it without a `Cell` — Python's dynamic dispatch accepts either a
/// `Cell`-derived `ops` list or a `PGElement` list there; Rust's static
/// typing makes that a second entry point instead of one polymorphic call.
pub fn get_crystal_class_from_rotations(
    rotations_in: &[RotMatrix],
) -> Result<(&'static str, &'static str), PbcSymmError> {
    let mut rotations: Vec<RotMatrix> = rotations_in.to_vec();
    rotations.sort_by_key(flatten9);
    rotations.dedup();

    let mut table = [0_i32; 10];
    for rot in &rotations {
        let idx = rotation_class_index(rot)?;
        table[idx] += 1;
    }

    let crystal_class = tables::CRYSTAL_CLASS
        .iter()
        .find(|(_, v)| *v == table)
        .map(|(k, _)| *k)
        .ok_or(PbcSymmError::UnknownCrystalClass)?;
    let laue_class =
        tables::laue_class_for(crystal_class).ok_or(PbcSymmError::UnknownCrystalClass)?;
    Ok((crystal_class, laue_class))
}

fn flatten9(w: &RotMatrix) -> [i32; 9] {
    let mut out = [0; 9];
    for i in 0..3 {
        for j in 0..3 {
            out[3 * i + j] = w[i][j];
        }
    }
    out
}

/// `geom.py:149-216` — `get_crystal_class`. `ops = None` reproduces
/// `search_space_group_ops(cell, tol=tol)`.
pub fn get_crystal_class(
    cell: &Cell,
    ops: Option<&[SpaceGroupOp]>,
    tol: f64,
) -> Result<(&'static str, &'static str), PbcSymmError> {
    let owned_ops;
    let ops: &[SpaceGroupOp] = match ops {
        Some(o) => o,
        None => {
            owned_ops = search_space_group_ops(cell, None, tol)?;
            &owned_ops
        }
    };
    let rotations: Vec<RotMatrix> = ops.iter().map(|op| op.rot).collect();
    get_crystal_class_from_rotations(&rotations)
}

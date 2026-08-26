//! Lattice-free half of `get_coulG` — `pyscf/pbc/tools/pbc.py:237-486` (plan 11-02).
//!
//! Everything here takes explicit geometry (`b`, `mesh`, `vol`, `dimension`) so
//! that it can live below `pyscf-pbc-gto` in the DAG. The `Cell`-taking driver
//! is [`pyscf_pbc_gto::coulg::get_coulg`], the same split
//! [`crate::lattice`] / [`crate::supercell`] already use.

/// How the `G + k = 0` divergence of the EXCHANGE kernel is treated.
///
/// Upstream carries this as `exx`/`exxdiv`, a value that is `False`, `None` or
/// one of three strings. `Option<ExxDiv>` models it: `None` is upstream's
/// `False`/`None` — no correction at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExxDiv {
    /// `'ewald'` — the Ewald probe-charge correction, upstream's SCF default.
    #[default]
    Ewald,
    /// `'vcut_sph'` — spherically truncated Coulomb, PRB 77 193110.
    VcutSph,
    /// `'vcut_ws'` — Wigner-Seitz truncated Coulomb, PRB 87 165122.
    VcutWs,
}

impl ExxDiv {
    /// Parse the upstream string spelling. `""`, `"none"` and `"false"` map to
    /// `None`; an unknown string is also `None` with a warning, matching
    /// upstream's fall-through `else` branch of `get_coulG`.
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "ewald" => Some(Self::Ewald),
            "vcut_sph" => Some(Self::VcutSph),
            "vcut_ws" => Some(Self::VcutWs),
            _ => None,
        }
    }

    /// The upstream string spelling.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ewald => "ewald",
            Self::VcutSph => "vcut_sph",
            Self::VcutWs => "vcut_ws",
        }
    }
}

/// `_Gv_wrap_around(cell, Gv, k, mesh)` — `pbc.py:237-257`.
///
/// Folds every `k + G` that has left the `mesh`-sized reciprocal box back into
/// it. Upstream's comment is the reason this exists at all: without it the
/// gamma-point and k-point exchange kernels disagree.
///
/// `b` is the `2*pi`-normalised reciprocal lattice (`cell.reciprocal_vectors()`),
/// one vector per ROW.
///
/// # Why the reduced coordinates go through an explicit LU solve
///
/// Upstream computes them as `np.linalg.solve(box_edge.T, kG.T).T`, and the
/// comparison that follows is against the EXACT boundary `+/- 0.5`. That is a
/// genuine floating-point tie: for an ODD mesh and a half-integer k offset the
/// extreme frequency lands on `((m-1)/2 + 1/2) / m = 1/2` exactly, so whether a
/// grid point folds is decided by the last bit of the linear solve — and the
/// two representatives it chooses between differ by a whole box edge, i.e. by a
/// large change in `4 pi/|k+G|^2`.
///
/// Multiplying by an explicit `inv3(box_edge)` gets those points *exactly* on
/// `0.5` and never folds them; LAPACK's `dgesv` lands a fraction of an ulp above
/// and folds two of them (measured on diamond, mesh `[11, 11, 11]`,
/// `k = b_0/2`). Reproducing upstream therefore means reproducing LAPACK, so
/// this is `dgetf2` + `dgetrs` for `n = 3`: LU with partial pivoting, then the
/// two triangular substitutions. On the diamond reference cell it agrees with
/// `np.linalg.solve` BIT-FOR-BIT over every grid point and every k offset of a
/// 2x2x2 mesh, fold decisions included.
pub fn gv_wrap_around(
    b: &[[f64; 3]; 3],
    gv: &[[f64; 3]],
    k: [f64; 3],
    mesh: [usize; 3],
    dimension: u8,
) -> Vec<[f64; 3]> {
    // box_edge[i] = mesh[i] * b[i]
    let mut box_edge = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            box_edge[i][j] = mesh[i] as f64 * b[i][j];
        }
    }
    // The system matrix is box_edge TRANSPOSED, as upstream passes it.
    let mut a = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            a[i][j] = box_edge[j][i];
        }
    }
    let Some((lu, piv)) = lu_factor3(a) else {
        // A singular box cannot be folded; upstream would raise inside
        // `np.linalg.solve`. Returning the unfolded vectors keeps the caller's
        // error path in `get_coulG` rather than here.
        return gv
            .iter()
            .map(|g| [g[0] + k[0], g[1] + k[1], g[2] + k[2]])
            .collect();
    };

    let mut out = Vec::with_capacity(gv.len());
    for g in gv {
        let mut kg = [g[0] + k[0], g[1] + k[1], g[2] + k[2]];
        let red = lu_solve3(&lu, &piv, kg);
        for axis in 0..3 {
            if (axis as u8) >= dimension {
                continue;
            }
            if red[axis] > 0.5 {
                for j in 0..3 {
                    kg[j] -= box_edge[axis][j];
                }
            } else if red[axis] < -0.5 {
                for j in 0..3 {
                    kg[j] += box_edge[axis][j];
                }
            }
        }
        out.push(kg);
    }
    out
}

/// `dgetf2` for `n = 3`: in-place LU with partial pivoting.
///
/// Returns the packed factors and the row permutation (`piv[j]` is the ORIGINAL
/// row now sitting in row `j`), or `None` when a pivot is exactly zero.
fn lu_factor3(mut lu: [[f64; 3]; 3]) -> Option<([[f64; 3]; 3], [usize; 3])> {
    let mut piv = [0usize, 1, 2];
    for j in 0..3 {
        // Partial pivot: the largest |a[i][j]| at or below the diagonal.
        let mut p = j;
        for i in j + 1..3 {
            if lu[i][j].abs() > lu[p][j].abs() {
                p = i;
            }
        }
        if lu[p][j] == 0.0 {
            return None;
        }
        if p != j {
            lu.swap(j, p);
            piv.swap(j, p);
        }
        let inv = 1.0 / lu[j][j];
        for i in j + 1..3 {
            lu[i][j] *= inv;
        }
        for i in j + 1..3 {
            for kk in j + 1..3 {
                lu[i][kk] -= lu[i][j] * lu[j][kk];
            }
        }
    }
    Some((lu, piv))
}

/// `dgetrs` for `n = 3`: forward substitution against the unit-lower factor,
/// then back substitution against the upper one.
fn lu_solve3(lu: &[[f64; 3]; 3], piv: &[usize; 3], b: [f64; 3]) -> [f64; 3] {
    let mut x = [b[piv[0]], b[piv[1]], b[piv[2]]];
    for j in 0..3 {
        for i in j + 1..3 {
            x[i] -= lu[i][j] * x[j];
        }
    }
    for j in (0..3).rev() {
        x[j] /= lu[j][j];
        for i in 0..j {
            x[i] -= lu[i][j] * x[j];
        }
    }
    x
}

/// `|k+G|^2` for every grid point.
pub fn abs_g2(kg: &[[f64; 3]]) -> Vec<f64> {
    kg.iter()
        .map(|g| g[0] * g[0] + g[1] * g[1] + g[2] * g[2])
        .collect()
}

/// The plain 3-D kernel `4*pi/|k+G|^2`, zero where `|k+G| = 0`
/// (`pbc.py:414-417`).
pub fn coulg_full_range_3d(absg2: &[f64]) -> Vec<f64> {
    absg2
        .iter()
        .map(|g2| {
            if *g2 == 0.0 {
                0.0
            } else {
                4.0 * std::f64::consts::PI / g2
            }
        })
        .collect()
}

/// The 2-D analytic truncation of Sundararaman & Arias, PRB 87 (2013) —
/// `pbc.py:420-431`.
///
/// `ld2 = pi / |b[2]|` is the half-height of the slab's vacuum box.
pub fn coulg_2d(kg: &[[f64; 3]], absg2: &[f64], b2_norm: f64) -> Vec<f64> {
    let ld2 = std::f64::consts::PI / b2_norm;
    let mut out = Vec::with_capacity(kg.len());
    for (g, g2) in kg.iter().zip(absg2.iter()) {
        let gz = g[2];
        let gp = (g[0] * g[0] + g[1] * g[1]).sqrt();
        let w = 1.0 - (gz * ld2).cos() * (-gp * ld2).exp();
        out.push(if *g2 == 0.0 {
            // Overwritten by the caller with -2*pi*Ld2^2 (`pbc.py:431`); the
            // placeholder keeps the vector length right.
            0.0
        } else {
            w * 4.0 * std::f64::consts::PI / g2
        });
    }
    out
}

/// The `G = 0` value of the 2-D kernel — `-2*pi*Ld2^2` (`pbc.py:431`).
pub fn coulg_2d_g0(b2_norm: f64) -> f64 {
    let ld2 = std::f64::consts::PI / b2_norm;
    -2.0 * std::f64::consts::PI * ld2 * ld2
}

/// The attenuation applied for a range-separated Coulomb operator —
/// `pbc.py:465-471`.
///
/// `omega > 0` keeps the LONG range (`erf(omega r)/r`), `omega < 0` the SHORT
/// range (`erfc(|omega| r)/r`). `omega == 0` is a no-op.
pub fn apply_omega(coulg: &mut [f64], absg2: &[f64], omega: f64) {
    if omega == 0.0 {
        return;
    }
    let s = -0.25 / (omega * omega);
    if omega > 0.0 {
        for (v, g2) in coulg.iter_mut().zip(absg2.iter()) {
            *v *= (s * g2).exp();
        }
    } else {
        for (v, g2) in coulg.iter_mut().zip(absg2.iter()) {
            *v *= 1.0 - (s * g2).exp();
        }
    }
}

/// The `dimension == 0` truncated kernel — `pbc.py:305-352`.
///
/// `rc` is half the (cubic) box edge.
pub fn coulg_0d(absg: &[f64], rc: f64, omega: f64) -> Vec<f64> {
    use std::f64::consts::PI;
    let mut out: Vec<f64> = absg
        .iter()
        .map(|g| if *g == 0.0 { 0.0 } else { 4.0 * PI / (g * g) })
        .collect();
    if omega == 0.0 {
        for (v, g) in out.iter_mut().zip(absg.iter()) {
            *v *= 1.0 - (g * rc).cos();
        }
        out[0] = 2.0 * PI * rc * rc;
    } else if omega > 0.0 {
        let s = -0.25 / (omega * omega);
        for (v, g) in out.iter_mut().zip(absg.iter()) {
            *v *= (s * g * g).exp() - (g * rc).cos();
        }
        out[0] = 2.0 * PI * rc * rc - PI / (omega * omega);
    } else {
        let s = -0.25 / (omega * omega);
        for (v, g) in out.iter_mut().zip(absg.iter()) {
            *v *= 1.0 - (s * g * g).exp();
        }
        out[0] = PI / (omega * omega);
    }
    out
}

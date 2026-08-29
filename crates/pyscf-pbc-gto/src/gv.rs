//! G-vectors, structure factors and uniform real-space grids.
//!
//! Line-by-line port of
//! * `pyscf/pbc/gto/cell.py:525-537` — `get_Gv`;
//! * `pyscf/pbc/gto/cell.py:539-604` — `get_Gv_weights`;
//! * `pyscf/pbc/gto/cell.py:606-613` — `_non_uniform_Gv_base` (DEFERRED, see below);
//! * `pyscf/pbc/gto/cell.py:615-646` — `get_SI`;
//! * `pyscf/pbc/gto/cell.py:886-911` — `get_uniform_grids`;
//! * `pyscf/lib/pbc/cell.c:122-146` — the `get_Gv` C kernel, which is K-01
//!   ([`pyscf_kernels::gv`]).
//!
//! # Device work
//!
//! The two hot loops run on the device through `pyscf-kernels` (RULE 6 — this
//! crate never names `cubecl-*` itself):
//! * **K-01** [`pyscf_kernels::gv`] builds the `(ngrids, 3)` G-vector table;
//! * **K-02** [`pyscf_kernels::struct_factor`] builds `SI` from an explicit
//!   `Gv`.
//!
//! # Complex layout
//!
//! `SI` is returned as a planar [`CTensor`] (`re` / `im` planes), never
//! interleaved — D-PBC-02 / RULE 8.
//!
//! # The `inf_vacuum` branches (D-PBC-20, plan 12-08)
//!
//! `get_Gv_weights`'s `dimension <= 2 && low_dim_ft_type == inf_vacuum` cases
//! (`cell.py:558-581`) replace each NON-PERIODIC frequency axis with a
//! Gauss-Chebyshev radial quadrature (`_non_uniform_Gv_base` ->
//! `pyscf.dft.radi.gauss_chebyshev`, reused here from `pyscf_grids::radial`).
//! Two things follow, and both are observable:
//!
//! * `weights` stops being one scalar and becomes a PER-GRID array — a
//!   non-uniform rule has no single weight. Read it through
//!   [`GvWeights::weight`], which hides the split.
//! * the MESH SHRINKS on every vacuum axis: the base returns `2 * (n / 2)`
//!   points, so an odd axis comes back one smaller. Upstream notes this at
//!   `cell.py:598` ("mesh can be different from the input mesh"); the returned
//!   [`GvWeights::mesh`] is the one actually used, not the one requested.
//!
//! The 3-D path is untouched: one scalar weight, the requested mesh.

use crate::cell::Cell;
use crate::types::LowDimFtType;
use pyscf_algebra::{CTensor, select_backend};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_tools::mat3::{cross3, det3, norm3, transpose3};

/// `np.fft.fftfreq(n, 1./n)` — the INTEGER FFT frequencies
/// `[0, 1, …, (n-1)/2, -(n/2), …, -1]`.
///
/// This is the one helper every downstream FFT depends on: getting the
/// negative-frequency fold wrong silently corrupts every planewave integral,
/// which is why PBC-MASTER-PLAN §8.1 plan 09-05 step 1 asks for it as a named,
/// separately-tested function.
///
/// `np.fft.fftfreq(n, d)` divides by `n*d`; with `d = 1/n` the divisor is 1, so
/// the result is exactly the integer table above (as `f64`).
pub fn fftfreq_scaled(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            if i <= (n - 1) / 2 {
                i as f64
            } else {
                i as f64 - n as f64
            }
        })
        .collect()
}

/// `np.fft.fftfreq(n)` — [`fftfreq_scaled`] divided by `n`, i.e. the FRACTIONAL
/// frequencies in `[-0.5, 0.5)`. Used by [`get_uniform_grids`] with
/// `wrap_around = true`.
pub fn fftfreq(n: usize) -> Vec<f64> {
    fftfreq_scaled(n)
        .into_iter()
        .map(|f| f / n as f64)
        .collect()
}

/// The return of [`get_gv_weights`] — upstream's `(Gv, Gvbase, weights)` tuple.
#[derive(Debug, Clone)]
pub struct GvWeights {
    /// `(ngrids, 3)` G-vectors in 1/Bohr, C-order over `(x, y, z)`.
    pub gv: Vec<[f64; 3]>,
    /// `(rx, ry, rz)` — the per-axis integer frequencies. `get_SI`'s separable
    /// branch needs these, which is why upstream returns them alongside `Gv`.
    pub gvbase: [Vec<f64>; 3],
    /// `|det(b)|`. A SCALAR for the uniform 3D grid, which is every 3-D cell
    /// and every low-dimensional cell that is not `inf_vacuum`.
    ///
    /// In the `inf_vacuum` branches upstream replaces this with a PER-GRID
    /// array (`cell.py:558-581`), because the non-uniform Gauss-Chebyshev base
    /// does not have one weight. Read the effective value through
    /// [`GvWeights::weight`] rather than this field, so a caller cannot silently
    /// use the scalar where the array applies.
    pub weights: f64,
    /// The per-grid weights, present ONLY in the `inf_vacuum` branches.
    pub weights_per_grid: Option<Vec<f64>>,
    /// The mesh actually used. This is `cell.mesh` when the caller passed
    /// `None` — EXCEPT in the `inf_vacuum` branches, where the non-uniform base
    /// returns `2 * (mesh[i] / 2)` points on a reduced axis, so an odd input
    /// mesh comes back one smaller (`cell.py:600`: "mesh can be different from
    /// the input mesh").
    pub mesh: [usize; 3],
}

impl GvWeights {
    /// The quadrature weight at grid point `g` — the per-grid array when the
    /// `inf_vacuum` base produced one, the scalar otherwise.
    ///
    /// # Panics
    /// If `g` is out of range of a per-grid array.
    #[must_use]
    pub fn weight(&self, g: usize) -> f64 {
        match &self.weights_per_grid {
            Some(w) => w[g],
            None => self.weights,
        }
    }
}

/// `get_Gv_weights(cell, mesh)` — `cell.py:539-604`.
///
/// In the `inf_vacuum` branches the returned [`GvWeights::mesh`] may be SMALLER
/// than the requested one, and [`GvWeights::weights_per_grid`] is populated —
/// see the module docs.
///
/// # Errors
/// * [`CoreError::InvalidMolecule`] if the lattice is singular, the mesh has a
///   zero axis, `dimension` is not one the `inf_vacuum` branches cover, or the
///   device launch fails.
pub fn get_gv_weights(cell: &Cell, mesh: Option<[usize; 3]>) -> Result<GvWeights, PyscfRsError> {
    // cell.py:547-548 — `if mesh is None: mesh = cell.mesh`.
    let mesh = match mesh {
        Some(m) => m,
        None => cell.try_mesh()?,
    };
    if mesh.contains(&0) {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "get_Gv_weights: mesh {mesh:?} has a zero axis"
        ))));
    }

    // cell.py:554-557 — the default 3D uniform grid.
    let rx = fftfreq_scaled(mesh[0]);
    let ry = fftfreq_scaled(mesh[1]);
    let rz = fftfreq_scaled(mesh[2]);
    let b = cell.reciprocal_vectors_2pi()?;
    let mut weights = det3(&b).abs();

    // cell.py:558-581 — the `inf_vacuum` branches. Each NON-periodic axis is
    // replaced by a non-uniform Gauss-Chebyshev base, and the single scalar
    // weight becomes an outer product over the axes. The periodic axes keep the
    // uniform frequencies and contribute a constant factor: `|b_i|` for a
    // periodic axis in the 1-D case, the cell area `|b_0 x b_1|` for the 2-D
    // one.
    let mut rx = rx;
    let mut ry = ry;
    let mut rz = rz;
    let mut per_grid: Option<Vec<f64>> = None;
    if cell.dimension <= 2 && cell.low_dim_ft_type == LowDimFtType::InfVacuum {
        let nb = [norm3(&b[0]), norm3(&b[1]), norm3(&b[2])];
        // cell.py:562-564 etc — `_non_uniform_Gv_base(mesh[i] // 2)`, scaled by
        // `1 / |b_i|` so the base is in the same units as the uniform one.
        let base = |n: usize, scale: f64| -> (Vec<f64>, Vec<f64>) {
            let (r, w) = non_uniform_gv_base(n / 2);
            (r.into_iter().map(|v| v / scale).collect(), w)
        };
        let (wx, wy, wz) = match cell.dimension {
            // cell.py:561-568 — all three axes non-periodic.
            0 => {
                let (nrx, wx) = base(mesh[0], nb[0]);
                let (nry, wy) = base(mesh[1], nb[1]);
                let (nrz, wz) = base(mesh[2], nb[2]);
                rx = nrx;
                ry = nry;
                rz = nrz;
                (wx, wy, wz)
            }
            // cell.py:569-575 — x periodic, y and z vacuum.
            1 => {
                let wx = vec![nb[0]; mesh[0]];
                let (nry, wy) = base(mesh[1], nb[1]);
                let (nrz, wz) = base(mesh[2], nb[2]);
                ry = nry;
                rz = nrz;
                (wx, wy, wz)
            }
            // cell.py:576-581 — xy periodic, z vacuum. Upstream folds the AREA
            // over the xy block (`wxy = repeat(area, mesh[0]*mesh[1])`), which
            // is the same outer product with `wx = area` and `wy = 1`.
            2 => {
                let area = norm3(&cross3(&b[0], &b[1]));
                let (nrz, wz) = base(mesh[2], nb[2]);
                rz = nrz;
                (vec![area; mesh[0]], vec![1.0; mesh[1]], wz)
            }
            d => {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "get_Gv_weights: dimension = {d} with low_dim_ft_type = inf_vacuum"
                ))));
            }
        };
        // cell.py:568 — `einsum('i,j,k->ijk', wx, wy, wz).reshape(-1)`, C-order.
        let mut w = Vec::with_capacity(wx.len() * wy.len() * wz.len());
        for a in &wx {
            for c in &wy {
                for e in &wz {
                    w.push(a * c * e);
                }
            }
        }
        per_grid = Some(w);
    }
    // cell.py:598-600 — "NOTE mesh can be different from the input mesh": the
    // non-uniform base returns `2 * (n / 2)` points, so an odd axis shrinks.
    let mesh = [rx.len(), ry.len(), rz.len()];

    // cell.py:585-597 — the C kernel, here K-01 on the device.
    let selection = select_backend().map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "get_Gv_weights: backend selection failed: {e}"
        )))
    })?;
    // Row-major flat 3x3, matching the `double* b` upstream hands to libpbc.
    let b_flat: Vec<f64> = b.iter().flat_map(|row| row.iter().copied()).collect();
    let flat = pyscf_kernels::gv(&selection.client, &rx, &ry, &rz, &b_flat).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "get_Gv_weights: K-01 gv kernel failed: {e}"
        )))
    })?;
    let gv: Vec<[f64; 3]> = flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();

    // cell.py:601 — `weights *= 1/(2*pi)**3`. NOTE upstream's own comment:
    // `1/cell.vol == det(b)/(2pi)^3`.
    weights /= (2.0 * std::f64::consts::PI).powi(3);

    // The per-grid array carries the same `1/(2 pi)^3` as the scalar.
    let weights_per_grid = per_grid.map(|w| {
        let f = 1.0 / (2.0 * std::f64::consts::PI).powi(3);
        w.into_iter().map(|x| x * f).collect()
    });

    Ok(GvWeights {
        gv,
        gvbase: [rx, ry, rz],
        weights,
        weights_per_grid,
        mesh,
    })
}


/// `_non_uniform_Gv_base(n)` — `cell.py:607-613`.
///
/// A Gauss-Chebyshev radial grid mirrored about zero:
/// `hstack((rs, -rs[::-1]))`, `hstack((ws, ws[::-1]))`, so `n` input points
/// become `2n` output points. Upstream's commented-out alternative includes a
/// point at the origin; the live line does not, and neither does this.
///
/// The radial rule itself is `pyscf_grids::radial::gauss_chebyshev`, the same
/// one the molecular Becke grids use — there is no second copy of it here.
fn non_uniform_gv_base(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (rs, ws) = pyscf_grids::radial::gauss_chebyshev(n);
    let mut r = rs.clone();
    r.extend(rs.iter().rev().map(|v| -v));
    let mut w = ws.clone();
    w.extend(ws.iter().rev().copied());
    (r, w)
}

/// `get_Gv(cell, mesh)` — `cell.py:525-537`. Just
/// `get_Gv_weights(cell, mesh)[0]`.
///
/// # Errors
/// As [`get_gv_weights`].
pub fn get_gv(cell: &Cell, mesh: Option<[usize; 3]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    Ok(get_gv_weights(cell, mesh)?.gv)
}

/// `get_SI(cell, Gv, mesh, atmlst)` — `cell.py:615-646`. MH (3.34).
///
/// Returns the planar `(natm, ngrids)` structure factor
/// `SI[a, g] = exp(-i * Gv[g] . R_a)`, row-major (D-PBC-02 / RULE 8).
///
/// Two branches, exactly as upstream:
/// * `gv = Some(...)` — the direct form `exp(-1j*dot(coords, Gv.T))`, run on
///   the device as K-02 ([`pyscf_kernels::struct_factor`]);
/// * `gv = None` — the SEPARABLE form (`cell.py:626-635`): one complex
///   exponential per `(atom, axis, frequency)`, then the outer product over the
///   three axes. That is `natm*(mx+my+mz)` transcendental calls instead of
///   `natm*mx*my*mz`, and it is what upstream uses by default.
///
/// `atmlst` selects a subset of atoms (`cell.py:621-622`); `None` means all.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] if an `atmlst` index is out of range, if the
/// lattice is singular, or if the device launch fails; propagates
/// [`get_gv_weights`]'s deferrals.
pub fn get_si(
    cell: &Cell,
    gv: Option<&[[f64; 3]]>,
    mesh: Option<[usize; 3]>,
    atmlst: Option<&[usize]>,
) -> Result<CTensor, PyscfRsError> {
    // cell.py:619-622
    let all_coords = cell.mol.atom_coords();
    let coords: Vec<[f64; 3]> = match atmlst {
        None => all_coords,
        Some(list) => list
            .iter()
            .map(|ia| {
                all_coords.get(*ia).copied().ok_or_else(|| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "get_SI: atmlst index {ia} out of range (natm = {})",
                        all_coords.len()
                    )))
                })
            })
            .collect::<Result<_, _>>()?,
    };
    let natm = coords.len();

    match gv {
        // cell.py:645 — `SI = np.exp(-1j*np.dot(coords, Gv.T))`, on the device.
        Some(gv) => {
            if natm == 0 || gv.is_empty() {
                return Ok(CTensor::zeros(0));
            }
            let selection = select_backend().map_err(|e| {
                PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "get_SI: backend selection failed: {e}"
                )))
            })?;
            let coords_flat: Vec<f64> = coords.iter().flat_map(|r| r.iter().copied()).collect();
            let gv_flat: Vec<f64> = gv.iter().flat_map(|r| r.iter().copied()).collect();
            let (re, im) = pyscf_kernels::struct_factor(&selection.client, &coords_flat, &gv_flat)
                .map_err(|e| {
                    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                        "get_SI: K-02 struct_factor kernel failed: {e}"
                    )))
                })?;
            Ok(CTensor { re, im })
        }
        // cell.py:626-635 — the separable branch.
        None => {
            let base = get_gv_weights(cell, mesh)?;
            let b = cell.reciprocal_vectors_2pi()?;
            let [mx, my, mz] = base.mesh;
            let ngrids = mx * my * mz;

            // rb = np.dot(coords, b.T)  ->  rb[a][i] = coords[a] . b[i]
            let bt = transpose3(&b);
            let rb: Vec<[f64; 3]> = coords
                .iter()
                .map(|r| {
                    let mut o = [0.0; 3];
                    for (j, oj) in o.iter_mut().enumerate() {
                        *oj = r[0] * bt[0][j] + r[1] * bt[1][j] + r[2] * bt[2][j];
                    }
                    o
                })
                .collect();

            // SIx[a][g] = exp(-1j * rb[a][0] * basex[g]), and likewise y, z.
            let axis = |a: usize, i: usize| -> (Vec<f64>, Vec<f64>) {
                base.gvbase[i]
                    .iter()
                    .map(|f| {
                        let theta = -(rb[a][i] * f);
                        (theta.cos(), theta.sin())
                    })
                    .unzip()
            };

            let mut re = vec![0.0_f64; natm * ngrids];
            let mut im = vec![0.0_f64; natm * ngrids];
            for a in 0..natm {
                let (xr, xi) = axis(a, 0);
                let (yr, yi) = axis(a, 1);
                let (zr, zi) = axis(a, 2);
                for gx in 0..mx {
                    // SIx * SIy, hoisted out of the innermost loop.
                    for gy in 0..my {
                        let pr = xr[gx] * yr[gy] - xi[gx] * yi[gy];
                        let pi = xr[gx] * yi[gy] + xi[gx] * yr[gy];
                        let row = a * ngrids + gx * my * mz + gy * mz;
                        for gz in 0..mz {
                            re[row + gz] = pr * zr[gz] - pi * zi[gz];
                            im[row + gz] = pr * zi[gz] + pi * zr[gz];
                        }
                    }
                }
            }
            Ok(CTensor { re, im })
        }
    }
}

/// `get_uniform_grids(cell, mesh, wrap_around)` — `cell.py:886-911`. MH (3.19).
///
/// The real-space grid `r[i,j,k] = q_i*a[0] + q_j*a[1] + q_k*a[2]`, C-order over
/// `(i, j, k)`. With `wrap_around = true` (upstream's default) the fractional
/// coordinates come from [`fftfreq`], folding indices past `n/2` to negative so
/// the grid is centred on the origin; otherwise they are `arange(n)/n`, i.e. the
/// grid fills `[0, 1)` of the primitive cell.
///
/// Upstream's comment on why the default is `wrap_around = True`: a grid
/// generated inside the primitive cell without wrap-around would need an extra
/// image layer in `get_lattice_Ls` for 2D calculations.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] if the mesh has a zero axis; propagates
/// [`Cell::try_mesh`] when `mesh` is `None`.
pub fn get_uniform_grids(
    cell: &Cell,
    mesh: Option<[usize; 3]>,
    wrap_around: bool,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let mesh = match mesh {
        Some(m) => m,
        None => cell.try_mesh()?,
    };
    if mesh.contains(&0) {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "get_uniform_grids: mesh {mesh:?} has a zero axis"
        ))));
    }
    let a = cell.lattice_vectors();

    // cell.py:899-909 — the two branches differ ONLY in the fractional axis.
    let q: [Vec<f64>; 3] = if wrap_around {
        [fftfreq(mesh[0]), fftfreq(mesh[1]), fftfreq(mesh[2])]
    } else {
        // `qv = cartesian_prod([arange(x)])` then `a_frac = a / mesh` per row,
        // i.e. the fractional coordinate is `i / n`.
        [
            (0..mesh[0]).map(|i| i as f64 / mesh[0] as f64).collect(),
            (0..mesh[1]).map(|i| i as f64 / mesh[1] as f64).collect(),
            (0..mesh[2]).map(|i| i as f64 / mesh[2] as f64).collect(),
        ]
    };

    let mut out = Vec::with_capacity(mesh[0] * mesh[1] * mesh[2]);
    for &qx in &q[0] {
        for &qy in &q[1] {
            for &qz in &q[2] {
                out.push([
                    qx * a[0][0] + qy * a[1][0] + qz * a[2][0],
                    qx * a[0][1] + qy * a[1][1] + qz * a[2][1],
                    qx * a[0][2] + qy * a[1][2] + qz * a[2][2],
                ]);
            }
        }
    }
    Ok(out)
}

impl Cell {
    /// `cell.get_Gv(mesh)` — see [`get_gv`].
    ///
    /// # Errors
    /// As [`get_gv`].
    pub fn get_gv(&self, mesh: Option<[usize; 3]>) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        get_gv(self, mesh)
    }

    /// `cell.get_Gv_weights(mesh)` — see [`get_gv_weights`].
    ///
    /// # Errors
    /// As [`get_gv_weights`].
    pub fn get_gv_weights(&self, mesh: Option<[usize; 3]>) -> Result<GvWeights, PyscfRsError> {
        get_gv_weights(self, mesh)
    }

    /// `cell.get_SI(Gv, mesh, atmlst)` — see [`get_si`].
    ///
    /// # Errors
    /// As [`get_si`].
    pub fn get_si(
        &self,
        gv: Option<&[[f64; 3]]>,
        mesh: Option<[usize; 3]>,
        atmlst: Option<&[usize]>,
    ) -> Result<CTensor, PyscfRsError> {
        get_si(self, gv, mesh, atmlst)
    }

    /// `cell.get_uniform_grids(mesh, wrap_around)` — see [`get_uniform_grids`].
    ///
    /// # Errors
    /// As [`get_uniform_grids`].
    pub fn get_uniform_grids(
        &self,
        mesh: Option<[usize; 3]>,
        wrap_around: bool,
    ) -> Result<Vec<[f64; 3]>, PyscfRsError> {
        get_uniform_grids(self, mesh, wrap_around)
    }
}

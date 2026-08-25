//! Particle-mesh Ewald — cardinal B-splines and the PME driver.
//!
//! Line-by-line port of
//! * `pyscf/pbc/gto/ewald_methods.py:30` — `INTERPOLATION_ORDER`;
//! * `pyscf/pbc/gto/ewald_methods.py:32-38` — `_bspline`;
//! * `pyscf/pbc/gto/ewald_methods.py:40-47` — `_bspline_grad`;
//! * `pyscf/pbc/gto/ewald_methods.py:49-78` — `bspline`;
//! * `pyscf/pbc/gto/ewald_methods.py:80-99` — `_get_ewald_direct`, whose body is
//!   the C loop `pyscf/lib/pbc/cell.c:get_ewald_direct`;
//! * `pyscf/pbc/gto/ewald_methods.py:121-176` — `particle_mesh_ewald`.
//!
//! # What ships and what does not
//!
//! The B-spline machinery and the screened real-space sum are pure arithmetic
//! and ship in full. [`particle_mesh_ewald`] itself does NOT: its G-space half
//! is `tools.fft(B * C * tools.ifft(Q))`, and PBC-MASTER-PLAN §8.1 puts the
//! complex 3-D FFT in Phase 11 plan 11-01 (09-CONTEXT.md lists "Any FFT" as an
//! explicit Phase 9 non-goal). It therefore returns a clean
//! [`PyscfRsError::NotYetImplemented`] `{ phase: 11 }` rather than a silently
//! wrong answer (D-PBC-20). Everything it needs BEFORE the FFT — `ewovrl` via
//! [`get_ewald_direct`], `ewself`, the charge mesh `Q`, the `B` and `C` arrays —
//! is implemented and unit-tested here, so plan 11-01 only has to bolt the two
//! transforms on.
//!
//! Note that upstream reaches PME only when `cell.use_particle_mesh_ewald` is
//! set, which is NOT the default; [`crate::ewald::ewald`]'s shipped 3D path is
//! unaffected.
//!
//! # The B-spline convention
//!
//! `bspline(u, ng, n)` returns
//! * `m` — the `(nu, ng)` interpolation matrix, row `i` holding the `n`
//!   non-zero cardinal B-spline weights of point `u[i]` scattered onto the grid;
//! * `b` — the `(ng,)` complex Euler exponential-spline coefficient
//!   `b[m] = exp(2 pi i (n-1) m / ng) / sum_{k<n-1} M_n(k+1) exp(2 pi i m k / ng)`,
//!   PLANAR (`b_re` / `b_im`) per D-PBC-02 / RULE 8;
//! * `idx` — the `(n, nu)` grid indices those weights landed on.

use crate::cell::Cell;
use pyscf_algebra::oracle_sum;
use pyscf_core::{CoreError, PyscfRsError};
use std::f64::consts::PI;

/// `INTERPOLATION_ORDER` — `ewald_methods.py:30`. Upstream's own comment:
/// "FIXME The default interpolation order may be too high".
pub const INTERPOLATION_ORDER: usize = 10;

/// Distances below this are skipped by the screened real-space sum
/// (`cell.c:get_ewald_direct` — `if (r > 1e-10 && r < rcut)`). NOTE this is a
/// LOOSER threshold than [`crate::ewald::EWALD_R_MIN`] (1e-16), which is what
/// the array-based `cell.py` path uses; the two upstream implementations
/// genuinely differ here.
pub const EWALD_DIRECT_R_MIN: f64 = 1e-10;

/// `factorial(n)` as an exact `f64` for the small `n` a B-spline order implies.
fn factorial(n: usize) -> f64 {
    (1..=n).map(|k| k as f64).product()
}

/// `scipy.special.binom(n, k)` for integer arguments, built from ratios of
/// [`factorial`] so it stays exact for the orders used here.
fn binom(n: usize, k: usize) -> f64 {
    if k > n {
        return 0.0;
    }
    factorial(n) / (factorial(k) * factorial(n - k))
}

/// `_bspline(u, n)` — `ewald_methods.py:32-38`. The cardinal B-spline `M_n(u)`
/// in its truncated-power-function form:
///
/// ```text
/// M_n(u) = 1/(n-1)! sum_{k=0}^{n} (-1)^k C(n,k) max(u-k, 0)^(n-1)
/// ```
pub fn bspline_value(u: f64, n: usize) -> f64 {
    let fac = 1.0 / factorial(n - 1);
    let mut m = 0.0;
    for k in 0..=n {
        let sign = if k % 2 == 0 { 1.0 } else { -1.0 };
        let fac1 = sign * binom(n, k);
        // `powf`, not `powi`: numpy's `**` is a correctly-rounded `pow`, while
        // `powi`'s repeated multiplication loses ~3 ulp at order 10 — visible
        // in the partition-of-unity drift (tests/ewald.rs).
        m += fac1 * (u - k as f64).max(0.0).powf(n as f64 - 1.0);
    }
    m * fac
}

/// `_bspline_grad(u, n)` — `ewald_methods.py:40-47`:
/// `dM_n/du = M_{n-1}(u) - M_{n-1}(u-1)`.
pub fn bspline_grad(u: f64, n: usize) -> f64 {
    bspline_value(u, n - 1) - bspline_value(u - 1.0, n - 1)
}

/// The return of [`bspline`] — upstream's `(M, b, idx)` tuple.
#[derive(Debug, Clone)]
pub struct Bspline {
    /// `(nu, ng)` row-major interpolation matrix.
    pub m: Vec<f64>,
    /// `(nu, ng)` row-major derivative matrix — `Some` only when `deriv > 0`.
    pub dm: Option<Vec<f64>>,
    /// Real plane of the `(ng,)` Euler exponential-spline coefficients.
    pub b_re: Vec<f64>,
    /// Imaginary plane of the `(ng,)` Euler exponential-spline coefficients.
    pub b_im: Vec<f64>,
    /// `(n, nu)` grid indices, `idx[i][t]` for spline lobe `i` of point `t`.
    pub idx: Vec<Vec<usize>>,
    /// Number of interpolated points.
    pub nu: usize,
    /// Grid length along this axis.
    pub ng: usize,
}

/// `bspline(u, ng, n, deriv)` — `ewald_methods.py:49-78`.
///
/// `u` holds the FRACTIONAL grid coordinates of the points (upstream's
/// `np.dot(coords, b.T) * mesh`).
///
/// # Errors
/// [`CoreError::InvalidMolecule`] for `deriv > 1` (upstream raises
/// `NotImplementedError`), for `ng == 0`, or for an order below 2.
pub fn bspline(u: &[f64], ng: usize, n: usize, deriv: usize) -> Result<Bspline, PyscfRsError> {
    if deriv > 1 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "bspline: deriv > 1 is not implemented upstream either \
             (ewald_methods.py:63-64)"
                .to_string(),
        )));
    }
    if ng == 0 || n < 2 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "bspline: need ng >= 1 and order >= 2, got ng = {ng}, n = {n}"
        ))));
    }
    let nu = u.len();

    // ewald_methods.py:51-52
    let u_floor: Vec<f64> = u.iter().map(|x| x.floor()).collect();
    let delta: Vec<f64> = u.iter().zip(&u_floor).map(|(x, f)| x - f).collect();

    // ewald_methods.py:53-58 — idx[i][t] = rint((floor(u_t) - i) % ng), with
    // PYTHON modulo semantics (always non-negative); val[i][t] = delta_t + i.
    let ngf = ng as f64;
    let mut idx: Vec<Vec<usize>> = Vec::with_capacity(n);
    let mut val: Vec<Vec<f64>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut idx_i = Vec::with_capacity(nu);
        let mut val_i = Vec::with_capacity(nu);
        for t in 0..nu {
            let x = u_floor[t] - i as f64;
            let rem = x - ngf * (x / ngf).floor();
            idx_i.push((rem.round() as usize) % ng);
            val_i.push(delta[t] + i as f64);
        }
        idx.push(idx_i);
        val.push(val_i);
    }

    // ewald_methods.py:60-62 — scatter-ADD, because two lobes of the same point
    // can land on the same grid index when ng < n.
    let mut m = vec![0.0_f64; nu * ng];
    for i in 0..n {
        for t in 0..nu {
            m[t * ng + idx[i][t]] += bspline_value(val[i][t], n);
        }
    }

    // ewald_methods.py:63-69
    let dm = if deriv > 0 {
        let mut dm = vec![0.0_f64; nu * ng];
        for i in 0..n {
            for t in 0..nu {
                dm[t * ng + idx[i][t]] += bspline_grad(val[i][t], n);
            }
        }
        Some(dm)
    } else {
        None
    };

    // ewald_methods.py:71-78 — the Euler exponential-spline coefficients.
    let mut b_re = vec![0.0_f64; ng];
    let mut b_im = vec![0.0_f64; ng];
    for (mm, (br, bi)) in b_re.iter_mut().zip(b_im.iter_mut()).enumerate() {
        let theta = 2.0 * PI * (n as f64 - 1.0) * mm as f64 / ngf;
        let (num_re, num_im) = (theta.cos(), theta.sin());
        let mut den_re = 0.0_f64;
        let mut den_im = 0.0_f64;
        for k in 0..(n - 1) {
            let w = bspline_value(k as f64 + 1.0, n);
            let phi = 2.0 * PI * mm as f64 * k as f64 / ngf;
            den_re += w * phi.cos();
            den_im += w * phi.sin();
        }
        // Complex division num/den.
        let d2 = den_re * den_re + den_im * den_im;
        *br = (num_re * den_re + num_im * den_im) / d2;
        *bi = (num_im * den_re - num_re * den_im) / d2;
    }
    // ewald_methods.py:77-78 — the Nyquist coefficient is singular for odd order
    // on an even grid.
    if !n.is_multiple_of(2) && ng.is_multiple_of(2) {
        b_re[ng / 2] = 0.0;
        b_im[ng / 2] = 0.0;
    }

    Ok(Bspline {
        m,
        dm,
        b_re,
        b_im,
        idx,
        nu,
        ng,
    })
}

/// `_get_ewald_direct(cell, ew_eta, ew_cut)` — `ewald_methods.py:80-99`, whose
/// body is `pyscf/lib/pbc/cell.c:get_ewald_direct`:
///
/// ```text
/// 0.5 * sum_{i,j,L} q_i q_j erfc(eta r)/r   for 1e-10 < r < rcut
/// ```
///
/// This is the SCREENED cousin of [`crate::ewald::ewald_real_space`]: the C loop
/// drops every pair beyond `rcut` instead of relying on `get_lattice_Ls` alone,
/// and its near-coincidence threshold is 1e-10 rather than 1e-16. Both
/// differences are upstream's, not this port's.
///
/// The reduction is a host-side `oracle_sum` (§9.3) over the C-order `(i, j, L)`
/// triples — the same loop nest as the C source.
///
/// # Errors
/// As [`crate::ewald::get_ewald_params`] and [`crate::lattice::get_lattice_ls`].
pub fn get_ewald_direct(
    cell: &Cell,
    ew_eta: Option<f64>,
    ew_cut: Option<f64>,
) -> Result<f64, PyscfRsError> {
    // ewald_methods.py:81-82
    let (ew_eta, ew_cut) = match (ew_eta, ew_cut) {
        (Some(eta), Some(cut)) => (eta, cut),
        _ => crate::ewald::get_ewald_params(cell, None, None)?,
    };

    // ewald_methods.py:84-86
    let chargs: Vec<f64> = cell.mol.atom_charges().iter().map(|z| *z as f64).collect();
    let coords = cell.mol.atom_coords();
    let lall = crate::lattice::get_lattice_ls(cell, Some(ew_cut), None, true)?;

    // cell.c:get_ewald_direct — the (i, j, L) loop nest, verbatim.
    let natm = coords.len();
    let mut terms = Vec::with_capacity(natm * natm * lall.len());
    for i in 0..natm {
        let ri = coords[i];
        let qi = chargs[i];
        for j in 0..natm {
            let rj = coords[j];
            let qj = chargs[j];
            for rl in &lall {
                let dx = rj[0] + rl[0] - ri[0];
                let dy = rj[1] + rl[1] - ri[1];
                let dz = rj[2] + rl[2] - ri[2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();
                if r > EWALD_DIRECT_R_MIN && r < ew_cut {
                    terms.push(qi * qj * libm::erfc(ew_eta * r) / r);
                } else {
                    terms.push(0.0);
                }
            }
        }
    }
    Ok(0.5 * oracle_sum(&terms))
}

/// The charge mesh `Q` of `ewald_methods.py:155-159`, scattered from the
/// per-axis B-spline weights.
///
/// `Q[x, y, z] = sum_a q_a Mx[a, x] My[a, y] Mz[a, z]`, C-order over the mesh.
/// Split out of [`particle_mesh_ewald`] so plan 11-01 can test it independently
/// of the FFT.
///
/// This is upstream's own commented-out reference line
/// (`#:Q = np.einsum('i,ix,iy,iz->xyz', chargs, Mx, My, Mz)`) rather than the
/// `np.ix_` scatter beneath it. The two agree except when a mesh axis is shorter
/// than the interpolation order, where two lobes of one atom fold onto the same
/// grid index: `np.ix_` fancy-index `+=` applies such a duplicate only once,
/// while `M` (and this function) accumulates it. The einsum form is the
/// mathematically intended one.
pub fn pme_charge_mesh(
    chargs: &[f64],
    mx: &Bspline,
    my: &Bspline,
    mz: &Bspline,
    mesh: [usize; 3],
) -> Vec<f64> {
    let [nx, ny, nz] = mesh;
    let mut q = vec![0.0_f64; nx * ny * nz];
    for (ia, qa) in chargs.iter().enumerate() {
        for x in 0..nx {
            let wx = mx.m[ia * nx + x];
            if wx == 0.0 {
                continue;
            }
            for y in 0..ny {
                let wxy = wx * my.m[ia * ny + y];
                if wxy == 0.0 {
                    continue;
                }
                for z in 0..nz {
                    q[(x * ny + y) * nz + z] += qa * wxy * mz.m[ia * nz + z];
                }
            }
        }
    }
    q
}

/// `particle_mesh_ewald(cell, ew_eta, ew_cut, order)` — `ewald_methods.py:121-176`.
///
/// # Deferred to Phase 11 (D-PBC-20)
///
/// The G-space half is `tools.fft(B * C * tools.ifft(Q))`. The complex 3-D FFT
/// lands in PBC-MASTER-PLAN plan 11-01; 09-CONTEXT.md lists "Any FFT" as an
/// explicit Phase 9 non-goal. This function validates its inputs, runs
/// everything that precedes the transform, and then returns
/// [`PyscfRsError::NotYetImplemented`] rather than a wrong number.
///
/// # Errors
/// * [`CoreError::InvalidMolecule`] for `dimension != 3` (upstream raises
///   `NotImplementedError`);
/// * [`PyscfRsError::NotYetImplemented`] `{ phase: 11 }` at the FFT;
/// * propagates [`get_ewald_direct`], [`bspline`] and
///   [`crate::gv::get_gv_weights`].
pub fn particle_mesh_ewald(
    cell: &Cell,
    ew_eta: Option<f64>,
    ew_cut: Option<f64>,
    order: usize,
) -> Result<f64, PyscfRsError> {
    // ewald_methods.py:123-124
    if cell.dimension != 3 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "particle_mesh_ewald: only works for 3D (ewald_methods.py:123-124)".to_string(),
        )));
    }

    // ewald_methods.py:126-135
    let chargs: Vec<f64> = cell.mol.atom_charges().iter().map(|z| *z as f64).collect();
    let coords = cell.mol.atom_coords();
    let (ew_eta, ew_cut) = match (ew_eta, ew_cut) {
        (Some(eta), Some(cut)) => (eta, cut),
        _ => crate::ewald::get_ewald_params(cell, None, None)?,
    };
    let chargs_sum = oracle_sum(&chargs);
    let log_precision = (cell.precision / (chargs_sum * 16.0 * PI * PI)).ln();
    let ke_cutoff = -2.0 * ew_eta * ew_eta * log_precision;
    let mesh = cell.cutoff_to_mesh(ke_cutoff)?;

    // ewald_methods.py:137-141 — the two terms that need no transform.
    let _ewovrl = get_ewald_direct(cell, Some(ew_eta), Some(ew_cut))?;
    let _ewself = crate::ewald::ewald_self(&chargs, ew_eta, cell.dimension, cell.vol());

    // ewald_methods.py:143-144 — u = coords . b(norm_to=1).T * mesh
    let b = cell.reciprocal_vectors(1.0)?;
    let mut ux = Vec::with_capacity(coords.len());
    let mut uy = Vec::with_capacity(coords.len());
    let mut uz = Vec::with_capacity(coords.len());
    for r in &coords {
        let d = |bi: &[f64; 3]| r[0] * bi[0] + r[1] * bi[1] + r[2] * bi[2];
        ux.push(d(&b[0]) * mesh[0] as f64);
        uy.push(d(&b[1]) * mesh[1] as f64);
        uz.push(d(&b[2]) * mesh[2] as f64);
    }

    // ewald_methods.py:146-148
    let mx = bspline(&ux, mesh[0], order, 0)?;
    let my = bspline(&uy, mesh[1], order, 0)?;
    let mz = bspline(&uz, mesh[2], order, 0)?;

    // ewald_methods.py:155-159
    let _q = pme_charge_mesh(&chargs, &mx, &my, &mz, mesh);

    // ewald_methods.py:161 — B = |bx|^2 (x) |by|^2 (x) |bz|^2 (real).
    let _b_axes: [Vec<f64>; 3] = [&mx, &my, &mz].map(|s| {
        s.b_re
            .iter()
            .zip(&s.b_im)
            .map(|(re, im)| re * re + im * im)
            .collect()
    });

    // ewald_methods.py:163-169 — C = weights * coulG * exp(-absG2/(4 eta^2)).
    let gw = crate::gv::get_gv_weights(cell, Some(mesh))?;
    let _c: Vec<f64> = gw
        .gv
        .iter()
        .map(|g| {
            let mut absg2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
            if absg2 == 0.0 {
                absg2 = crate::ewald::EWALD_G0_SENTINEL;
            }
            gw.weights * (4.0 * PI / absg2) * (-absg2 / (4.0 * ew_eta * ew_eta)).exp()
        })
        .collect();

    // ewald_methods.py:171-173 — ewg = .5 * prod(mesh) * <Q, fft(B*C*ifft(Q))>
    Err(PyscfRsError::NotYetImplemented {
        phase: 11,
        what: "particle_mesh_ewald's G-space sum needs the complex 3-D FFT \
               (ewald_methods.py:171-173) — PBC-MASTER-PLAN plan 11-01. \
               Clear cell.use_particle_mesh_ewald to use the exact Ewald sum.",
    })
}

impl Cell {
    /// `cell.get_ewald_direct(ew_eta, ew_cut)` — see [`get_ewald_direct`].
    ///
    /// # Errors
    /// As [`get_ewald_direct`].
    pub fn get_ewald_direct(
        &self,
        ew_eta: Option<f64>,
        ew_cut: Option<f64>,
    ) -> Result<f64, PyscfRsError> {
        get_ewald_direct(self, ew_eta, ew_cut)
    }
}

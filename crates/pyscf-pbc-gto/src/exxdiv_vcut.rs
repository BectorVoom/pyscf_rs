//! Truncated-Coulomb exchange kernels — plan 12-08, the `exxdiv` half of the
//! D-PBC-20 closure.
//!
//! Two schemes upstream offers alongside `'ewald'`:
//!
//! | scheme | reference | what it does |
//! |---|---|---|
//! | `vcut_sph` | PRB 77, 193110 (2008) | truncate `1/r` at the radius of a sphere with the volume of the Born-von-Karman supercell |
//! | `vcut_ws` | PRB 87, 165122 (2013) | the Wigner-Seitz kernel: split `1/r` at `erf/erfc`, take the long-range half analytically and the short-range half from a numerical FT of the minimum-image `erf(α r)/r` on the supercell |
//!
//! Ports `pyscf/pbc/tools/pbc.py:373-410` (the two `get_coulG` branches) and
//! `:487-547` (`precompute_exx`).
//!
//! Both are `dimension == 3` only — upstream raises `NotImplementedError`
//! otherwise, and so does this port.

use std::f64::consts::PI;

use pyscf_core::{CoreError, PyscfRsError, Unit};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_tools::fft;
use pyscf_pbc_tools::mat3::inv3;

use crate::cell::Cell;
use crate::types::{ALattice, CellBuildArgs};

/// `erf(x)` on the HOST.
///
/// `std` has no `erf`, and `cube-math` — which the kernel crates use — is a
/// DEVICE libm: every entry point launders its argument through
/// `bits::opaque64`, whose `RuntimeCell` has no native implementation, so a
/// host call panics with "Unexpanded Cube functions should not be called".
/// `rmath` is the crate cube-math was PORTED FROM: same algorithms, host-side,
/// and bit-identical to the platform `libm` by construction. It carries no
/// cubecl dependency, so it is on the method-crate side of the ALG-06 wall.
fn erf(x: f64) -> f64 {
    use rmath::prelude::{Erf, Function};
    Erf::new().eval(x)
}

/// `get_coulG(..., exx='vcut_sph')` — `pbc.py:373-380`.
///
/// ```text
/// Rc      = (3 Nk Ω / 4π)^{1/3}
/// coulG   = 4π/|k+G|² · (1 − cos(|k+G| Rc))
/// coulG_0 = 2π Rc²
/// ```
///
/// # Errors
/// [`PyscfRsError::NotYetImplemented`] for `dimension < 3` — upstream raises
/// `NotImplementedError` at `pbc.py:379-380`.
pub fn coulg_vcut_sph(cell: &Cell, absg2: &[f64], nk: usize) -> Result<Vec<f64>, PyscfRsError> {
    if cell.dimension < 3 {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 12,
            what: "get_coulG with exxdiv = 'vcut_sph' for dimension < 3 — upstream \
                   raises NotImplementedError at pbc.py:379-380",
        });
    }
    let rc = (3.0 * nk as f64 * cell.vol() / (4.0 * PI)).powf(1.0 / 3.0);
    Ok(absg2
        .iter()
        .map(|g2| {
            if *g2 == 0.0 {
                // pbc.py:378 — `4*pi*0.5*Rc**2`.
                2.0 * PI * rc * rc
            } else {
                let g = g2.sqrt();
                4.0 * PI / g2 * (1.0 - (g * rc).cos())
            }
        })
        .collect())
}

/// The precomputed Wigner-Seitz exchange kernel — `precompute_exx`'s
/// `ws_exx` dict (`pbc.py:487-547`).
#[derive(Debug, Clone)]
pub struct WsExx {
    /// The splitting parameter `α = 5/R_in`.
    pub alpha: f64,
    /// The supercell FFT mesh the kernel was tabulated on.
    pub mesh: [usize; 3],
    /// The supercell lattice `a · Nk`.
    pub a: [[f64; 3]; 3],
    /// `q` — the supercell G-vectors the kernel is tabulated at.
    pub q: Vec<[f64; 3]>,
    /// `vq` — the (real) short-range kernel at each `q`.
    pub vq: Vec<f64>,
}

/// `precompute_exx(cell, kpts)` — `pbc.py:487-547`.
///
/// Builds the Born-von-Karman supercell (`a · Nk`), tabulates the
/// MINIMUM-IMAGE `erf(α r)/r` on its uniform grid and Fourier-transforms it.
/// The result is the correction that turns the analytic long-range kernel into
/// the full Wigner-Seitz-truncated one.
///
/// # Errors
/// * [`CoreError::InvalidMolecule`] when the transformed kernel comes back with
///   a significant imaginary part — upstream's `RuntimeError('Unconventional
///   lattice was found')` (`pbc.py:538-545`);
/// * propagates the supercell build, the uniform grid and the FFT.
pub fn precompute_exx(cell: &Cell, kpts: &[[f64; 3]]) -> Result<WsExx, PyscfRsError> {
    // pbc.py:492 — the Monkhorst-Pack size the k-points came from.
    let nk = crate::lattice::get_monkhorst_pack_size_default(cell, kpts)?;

    // pbc.py:495-501 — the supercell carrying ONE probe charge.
    let a = cell.lattice_vectors();
    let mut ka = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            ka[i][j] = a[i][j] * nk[i] as f64;
        }
    }

    // pbc.py:502 — `Lc = 1/norm(inv(kcell.a), axis=0)`, the perpendicular
    // widths of the supercell. `axis=0` is the COLUMN norm of `inv(a)`, which
    // for row-major `a` is the norm of row `i` of `inv(a)^T` — i.e. of column
    // `i` of `inv(a)`.
    let ainv = inv3(&ka)?;
    let mut lc = [0.0_f64; 3];
    for j in 0..3 {
        let col = [ainv[0][j], ainv[1][j], ainv[2][j]];
        lc[j] = 1.0 / (col[0] * col[0] + col[1] * col[1] + col[2] * col[2]).sqrt();
    }
    // pbc.py:504-508 — the ASE splitting parameter.
    let rin = lc.iter().copied().fold(f64::INFINITY, f64::min) / 2.0;
    let alpha = 5.0 / rin;
    // pbc.py:510 — `kcell.mesh = [4*int(L*alpha*3.0) for L in Lc]`.
    let mesh = [
        4 * (lc[0] * alpha * 3.0) as usize,
        4 * (lc[1] * alpha * 3.0) as usize,
        4 * (lc[2] * alpha * 3.0) as usize,
    ];
    if mesh.contains(&0) {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "precompute_exx: the Wigner-Seitz mesh degenerated to {mesh:?}"
        ))));
    }

    let kcell = Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("H".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("sto-3g".into()),
            spin: 1,
            unit: Unit::Bohr,
            ..Default::default()
        },
        a: ALattice::Matrix(ka),
        mesh: Some(mesh),
        ..Default::default()
    })?;

    // pbc.py:515 — `rs = kcell.get_uniform_grids(wrap_around=False)`.
    let rs = crate::gv::get_uniform_grids(&kcell, Some(mesh), false)?;
    let kngs = rs.len();

    // pbc.py:518-519 — the eight corners of the supercell parallelepiped.
    let mut corners: Vec<[f64; 3]> = Vec::with_capacity(8);
    for cx in [0.0_f64, 1.0] {
        for cy in [0.0_f64, 1.0] {
            for cz in [0.0_f64, 1.0] {
                corners.push([
                    cx * ka[0][0] + cy * ka[1][0] + cz * ka[2][0],
                    cx * ka[0][1] + cy * ka[1][1] + cz * ka[2][1],
                    cx * ka[0][2] + cy * ka[1][2] + cz * ka[2][2],
                ]);
            }
        }
    }

    // pbc.py:528-531 — the minimum-image distance to a corner, then
    // `erf(alpha r)/r` with the `r -> 0` limit `2 alpha / sqrt(pi)`.
    let mut vr = Vec::with_capacity(kngs);
    for p in &rs {
        let mut rmin = f64::INFINITY;
        for c in &corners {
            let d = ((p[0] - c[0]).powi(2) + (p[1] - c[1]).powi(2) + (p[2] - c[2]).powi(2)).sqrt();
            rmin = rmin.min(d);
        }
        vr.push(if rmin < 1e-9 {
            2.0 * alpha / PI.sqrt()
        } else {
            erf(alpha * rmin) / rmin
        });
    }

    // pbc.py:532 — `vG = (kcell.vol/kngs) * fft(vR, kcell.mesh)`.
    let scale = kcell.vol() / kngs as f64;
    let vg = fft(&pyscf_algebra::CTensor::from_real(&vr), mesh).map_err(|e| {
        PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "precompute_exx: the Wigner-Seitz FFT failed: {e}"
        )))
    })?;
    let imax = vg.im.iter().fold(0.0_f64, |m, x| m.max(x.abs())) * scale;
    if imax > 1e-6 {
        // pbc.py:538-545 — the minimum-image construction only makes sense on a
        // conventional lattice; a large imaginary residue means it does not.
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "precompute_exx: unconventional lattice — the Wigner-Seitz kernel came \
             back with an imaginary part of {imax:e} (pbc.py:538-545)"
        ))));
    }

    Ok(WsExx {
        alpha,
        mesh,
        a: ka,
        q: crate::gv::get_gv(&kcell, Some(mesh))?,
        vq: vg.re.iter().map(|x| x * scale).collect(),
    })
}

/// `get_coulG(..., exx='vcut_ws')` — `pbc.py:382-410`.
///
/// ```text
/// coulG   = 4π/|k+G|² · (1 − exp(−|k+G|²/(4α²)))       (the LR half)
/// coulG_0 = π/α²
/// coulG  += vq[index(k+G)]                             (the SR half)
/// ```
///
/// The index is the supercell G-vector `k+G` maps onto; a `k+G` outside the
/// tabulated range gets no correction (`pbc.py:407-409`).
///
/// # Errors
/// * [`PyscfRsError::NotYetImplemented`] for `dimension < 3`;
/// * propagates [`precompute_exx`].
pub fn coulg_vcut_ws(
    cell: &Cell,
    kg: &[[f64; 3]],
    absg2: &[f64],
    ws: &WsExx,
) -> Result<Vec<f64>, PyscfRsError> {
    if cell.dimension < 3 {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 12,
            what: "get_coulG with exxdiv = 'vcut_ws' for dimension < 3 — upstream \
                   raises NotImplementedError at pbc.py:409-410",
        });
    }
    let a2 = 4.0 * ws.alpha * ws.alpha;
    let mut coulg: Vec<f64> = absg2
        .iter()
        .map(|g2| {
            if *g2 == 0.0 {
                PI / (ws.alpha * ws.alpha)
            } else {
                4.0 * PI / g2 * (1.0 - (-g2 / a2).exp())
            }
        })
        .collect();

    // pbc.py:400-403 — `gxyz = round(kG . kcell.a.T / (2 pi))`, folded into the
    // supercell mesh.
    let m = [ws.mesh[0] as i64, ws.mesh[1] as i64, ws.mesh[2] as i64];
    // pbc.py:405 — `maxqv = abs(exx_q).max(axis=0)`.
    let mut maxqv = [0.0_f64; 3];
    for q in &ws.q {
        for i in 0..3 {
            maxqv[i] = maxqv[i].max(q[i].abs());
        }
    }

    for (g, kgv) in kg.iter().enumerate() {
        if (0..3).any(|i| kgv[i].abs() > maxqv[i]) {
            continue;
        }
        let mut idx = [0_i64; 3];
        for (i, item) in idx.iter_mut().enumerate() {
            let x = (kgv[0] * ws.a[i][0] + kgv[1] * ws.a[i][1] + kgv[2] * ws.a[i][2]) / (2.0 * PI);
            // pbc.py:401 — `.round(decimals=6).astype(int)`, i.e. truncation
            // toward zero of the 6-decimal-rounded value.
            let r = (x * 1e6).round() / 1e6;
            *item = ((r as i64) + m[i]).rem_euclid(m[i]);
        }
        let q = ((idx[0] * m[1] + idx[1]) * m[2] + idx[2]) as usize;
        if let Some(v) = ws.vq.get(q) {
            coulg[g] += v;
        }
    }
    Ok(coulg)
}

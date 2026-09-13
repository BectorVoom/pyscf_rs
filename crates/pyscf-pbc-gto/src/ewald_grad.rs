//! Nuclear Ewald derivatives, in Ha/Bohr (not forces).
//! The PME branch shares its interpolation and FFT state with the energy.
//! Reciprocal atom/grid contractions use the Phase-18 shared primitive.

use crate::{Cell, LowDimFtType};
use pyscf_algebra::oracle_sum;
use pyscf_core::{CoreError, PyscfRsError};
use std::f64::consts::PI;

fn invalid(message: impl std::fmt::Display) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!(
        "ewald_nuc_grad: {message}"
    )))
}

fn parameters(cell: &Cell, eta: Option<f64>, cut: Option<f64>) -> Result<(f64, f64), PyscfRsError> {
    if !cell.mol._built || !cell.precision.is_finite() || cell.precision <= 0.0 {
        return Err(invalid("built cell and positive finite precision required"));
    }
    let (eta, cut) = match (eta, cut) {
        (Some(e), Some(c)) => (e, c),
        _ => crate::ewald::get_ewald_params(cell, None, None)?,
    };
    if !eta.is_finite() || eta <= 0.0 || !cut.is_finite() || cut <= 0.0 {
        return Err(invalid("positive finite eta and cutoff required"));
    }
    Ok((eta, cut))
}

/// Screened real-space derivative, matching get_ewald_direct's r thresholds.
pub fn ewald_direct_nuc_grad(
    cell: &Cell,
    eta: Option<f64>,
    cut: Option<f64>,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    let (eta, cut) = parameters(cell, eta, cut)?;
    let coords = cell.mol.atom_coords();
    let charges = cell.atom_charges();
    let ls = crate::lattice::get_lattice_ls(cell, Some(cut), None, true)?;
    let mut result = vec![[0.0; 3]; coords.len()];
    for (i, ri) in coords.iter().enumerate() {
        let mut terms: [Vec<f64>; 3] = std::array::from_fn(|_| Vec::new());
        for l in &ls {
            for (j, rj) in coords.iter().enumerate() {
                let d: [f64; 3] = std::array::from_fn(|c| ri[c] - rj[c] + l[c]);
                let r2 = oracle_sum(&d.map(|x| x * x));
                let r = r2.sqrt();
                if r <= crate::ewald_pme::EWALD_DIRECT_R_MIN || r >= cut {
                    continue;
                }
                let factor = -(charges[i] as f64)
                    * (charges[j] as f64)
                    * (libm::erfc(eta * r) / r + 2.0 * eta / PI.sqrt() * (-eta * eta * r2).exp())
                    / r2;
                for c in 0..3 {
                    terms[c].push(factor * d[c]);
                }
            }
        }
        result[i] = terms.map(|v| oracle_sum(&v));
    }
    Ok(result)
}

/// Particle-mesh derivative with upstream's momentum-conservation correction.
pub fn particle_mesh_ewald_nuc_grad(
    cell: &Cell,
    eta: Option<f64>,
    cut: Option<f64>,
    order: usize,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    if cell.dimension != 3 {
        return Err(invalid("particle mesh Ewald requires dimension=3"));
    }
    if cell.mol.natm == 0 {
        return Ok(Vec::new());
    }
    let (eta, cut) = parameters(cell, eta, cut)?;
    let state = crate::ewald_pme::pme_data(cell, Some(eta), Some(cut), order)?;
    let mut result = ewald_direct_nuc_grad(cell, Some(state.eta), Some(state.cut))?;
    let charges = cell.atom_charges();
    let [nx, ny, nz] = state.mesh;
    let n = state.potential.len();
    let mut reciprocal = vec![[0.0; 3]; charges.len()];
    for (a, &charge) in charges.iter().enumerate() {
        let mut axes = [0.0; 3];
        for axis in 0..3 {
            let mut terms = Vec::new();
            // Upstream contracts the support-index lists, including repeated
            // indices when mesh < order. Preserve that multiplicity: replacing
            // these by unique grid indices changes the upstream PME gradient.
            for ix in &state.splines[0].idx {
                let x = ix[a];
                for iy in &state.splines[1].idx {
                    let y = iy[a];
                    for iz in &state.splines[2].idx {
                        let z = iz[a];
                        let indices = [a * nx + x, a * ny + y, a * nz + z];
                        let factors: [f64; 3] = std::array::from_fn(|d| {
                            let spline = &state.splines[d];
                            if d == axis {
                                spline.dm.as_ref().expect("deriv=1 state")[indices[d]]
                            } else {
                                spline.m[indices[d]]
                            }
                        });
                        let product = factors[0] * factors[1] * factors[2];
                        if product != 0.0 {
                            terms.push(product * state.potential[(x * ny + y) * nz + z]);
                        }
                    }
                }
            }
            axes[axis] = oracle_sum(&terms);
        }
        for c in 0..3 {
            reciprocal[a][c] = oracle_sum(
                &(0..3)
                    .map(|axis| axes[axis] * state.b[axis][c] * state.mesh[axis] as f64)
                    .collect::<Vec<_>>(),
            ) * charge as f64
                * n as f64;
        }
    }
    for c in 0..3 {
        let shift =
            oracle_sum(&reciprocal.iter().map(|g| g[c]).collect::<Vec<_>>()) / charges.len() as f64;
        for a in 0..charges.len() {
            result[a][c] += reciprocal[a][c] - shift;
        }
    }
    Ok(result)
}

/// Upstream dispatcher. The 2D truncated-Coulomb branch is unsupported in
/// PySCF 2.12.1 ewald_methods.py:274-290; inf_vacuum is the supported alternative.
pub fn ewald_nuc_grad(
    cell: &Cell,
    eta: Option<f64>,
    cut: Option<f64>,
) -> Result<Vec<[f64; 3]>, PyscfRsError> {
    if cell.dimension == 2 && cell.low_dim_ft_type != LowDimFtType::InfVacuum {
        return Err(PyscfRsError::NotYetImplemented {
            phase: 18,
            what: "2D truncated-Coulomb Ewald nuclear gradient (unsupported upstream)",
        });
    }
    if cell.dimension == 3 && cell.use_particle_mesh_ewald {
        return particle_mesh_ewald_nuc_grad(cell, eta, cut, crate::ewald_pme::INTERPOLATION_ORDER);
    }
    if cell.mol.natm == 0 {
        return Ok(Vec::new());
    }
    let (eta, cut) = parameters(cell, eta, cut)?;
    let mut result = ewald_direct_nuc_grad(cell, Some(eta), Some(cut))?;
    let charges: Vec<f64> = cell.atom_charges().iter().map(|&x| x as f64).collect();
    let ke = -2.0 * eta * eta * (cell.precision / (oracle_sum(&charges) * 16.0 * PI * PI)).ln();
    let mesh = cell.cutoff_to_mesh(ke)?;
    let gw = crate::gv::get_gv_weights(cell, Some(mesh))?;
    let selection = pyscf_algebra::select_backend().map_err(invalid)?;
    // Budget all per-slab fields, structure factors, products and launch copies.
    // Bound the working set independently of the total reciprocal mesh.
    let memory_mb = std::env::var("PYSCF_MAX_MEMORY")
        .ok()
        .and_then(|x| x.parse::<f64>().ok())
        .unwrap_or(256.0);
    if !memory_mb.is_finite() || memory_mb <= 0.0 {
        return Err(invalid("PYSCF_MAX_MEMORY must be positive and finite"));
    }
    let block =
        ((memory_mb * 1e6 / ((charges.len() * 32 + 16) as f64 * 8.0)) as usize).clamp(1, 65536);
    let mut partials: Vec<[Vec<f64>; 3]> = (0..charges.len())
        .map(|_| std::array::from_fn(|_| Vec::new()))
        .collect();
    for gv in gw.gv.chunks(block) {
        let ng = gv.len();
        let si = crate::gv::get_si(cell, Some(gv), None, None)?;
        let mut zr = vec![0.0; ng];
        let mut zi = vec![0.0; ng];
        for g in 0..ng {
            zr[g] = oracle_sum(
                &(0..charges.len())
                    .map(|a| charges[a] * si.re[a * ng + g])
                    .collect::<Vec<_>>(),
            );
            zi[g] = -oracle_sum(
                &(0..charges.len())
                    .map(|a| charges[a] * si.im[a * ng + g])
                    .collect::<Vec<_>>(),
            );
        }
        let mut fr = vec![0.0; charges.len() * 3 * ng];
        let mut fi = fr.clone();
        for (g, v) in gv.iter().enumerate() {
            let g2 = oracle_sum(&v.map(|x| x * x));
            if g2 == 0.0 {
                continue;
            }
            let factor = 4.0 * PI * gw.weights / g2 * (-g2 / (4.0 * eta * eta)).exp();
            for (a, &q) in charges.iter().enumerate() {
                for c in 0..3 {
                    let t = (a * 3 + c) * ng + g;
                    fr[t] = factor * q * v[c] * si.im[a * ng + g];
                    fi[t] = -factor * q * v[c] * si.re[a * ng + g];
                }
            }
        }
        let reduced = pyscf_kernels::pbc::multigrid_grad::contract_atom_grid(
            &selection.client,
            charges.len(),
            &fr,
            &fi,
            &zr,
            &zi,
        )
        .map_err(invalid)?;
        for a in 0..charges.len() {
            for c in 0..3 {
                partials[a][c].push(reduced[a][c]);
            }
        }
    }
    for a in 0..charges.len() {
        for c in 0..3 {
            result[a][c] += oracle_sum(&partials[a][c]);
        }
    }
    Ok(result)
}

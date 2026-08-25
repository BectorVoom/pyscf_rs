//! Cutoff estimators: `rcut`, `nimgs`, `ke_cutoff`, `mesh`.
//!
//! Line-by-line port of
//! * `pyscf/pbc/gto/cell.py:373-497` — `get_nimgs`, `_estimate_rcut`,
//!   `bas_rcut`, `estimate_rcut`, `_estimate_ke_cutoff`, `estimate_ke_cutoff`,
//!   `_extract_pgto_params`, `error_for_ke_cutoff`;
//! * `pyscf/pbc/gto/cell.py:499-523` — `get_bounding_sphere`;
//! * `pyscf/pbc/gto/cell.py:968-1025` — `_mesh_inf_vaccum`, `pgf_rcut`,
//!   `rcut_by_shells`;
//! * `pyscf/lib/pbc/cell.c:30-90` — the C `pgf_rcut` and `rcut_by_shells` that
//!   `cell.rcut_by_shells` actually calls through `libpbc`;
//! * `pyscf/pbc/gto/cell.py:1740-1767` — the `rcut` / `mesh` half of `build`.
//!
//! Pure scalar math — no GPU (PBC-MASTER-PLAN §8.1 plan 09-04 STEPS).
//!
//! # Upstream variable names
//!
//! The upstream names (`theta`, `a1`, `norm_ang`, `fac`, `r0`, `Ecut`,
//! `heights_inv`, `rmin`, `gmax`) are kept verbatim so a reviewer can diff the
//! Rust against the Python side by side.

use crate::cell::Cell;
use crate::types::LowDimFtType;
use pyscf_core::PyscfRsError;
use pyscf_core::raw_layout::{ANG_OF, BAS_SLOTS, NCTR_OF, NPRIM_OF, PTR_COEFF, PTR_EXP};
use pyscf_gto::PTR_RANGE_OMEGA;
use pyscf_pbc_tools::mat3::norm3;
use std::f64::consts::PI;

/// `cell.py:41` — `INTEGRAL_PRECISION`, the default `precision` for the
/// estimators when a caller has none of its own.
pub const INTEGRAL_PRECISION: f64 = 1e-8;

/// `cell.py:47` — convergence threshold of the [`pgf_rcut`] fixed-point loop.
pub const RCUT_EPS: f64 = 1e-3;

/// `cell.py:48` — iteration cap of the [`pgf_rcut`] fixed-point loop.
pub const RCUT_MAX_CYCLE: usize = 10;

// ---------------------------------------------------------------------------
// Raw `_bas` / `_env` accessors — the Rust spelling of `mol.bas_angular(i)`,
// `mol.bas_exp(i)` and `mol._libcint_ctr_coeff(i)` (`gto/mole.py:3324-3413`).
// ---------------------------------------------------------------------------

/// `mol.bas_angular(bas_id)` — the shell's angular momentum `l`.
pub fn bas_angular(cell: &Cell, bas_id: usize) -> i32 {
    cell.mol._bas[bas_id * BAS_SLOTS + ANG_OF]
}

/// `mol.bas_nprim(bas_id)` — primitives in the shell.
pub fn bas_nprim(cell: &Cell, bas_id: usize) -> usize {
    cell.mol._bas[bas_id * BAS_SLOTS + NPRIM_OF].max(0) as usize
}

/// `mol.bas_nctr(bas_id)` — contractions in the shell.
pub fn bas_nctr(cell: &Cell, bas_id: usize) -> usize {
    cell.mol._bas[bas_id * BAS_SLOTS + NCTR_OF].max(0) as usize
}

/// `mol.bas_exp(bas_id)` — the shell's primitive exponents.
pub fn bas_exp(cell: &Cell, bas_id: usize) -> Vec<f64> {
    let nprim = bas_nprim(cell, bas_id);
    let ptr = cell.mol._bas[bas_id * BAS_SLOTS + PTR_EXP].max(0) as usize;
    cell.mol._env[ptr..ptr + nprim].to_vec()
}

/// `abs(mol._libcint_ctr_coeff(bas_id)).max(axis=1)` — for each primitive, the
/// largest absolute libcint contraction coefficient over the contractions.
///
/// `_env` stores the block column-major as `(nctr, nprim)`
/// (`_env[ptr + ic*nprim + p]`, `lib/pbc/cell.c:80`), and upstream reshapes to
/// `(nctr, nprim)` then transposes, so `c[p][ic] = _env[ptr + ic*nprim + p]`.
pub fn libcint_ctr_coeff_max(cell: &Cell, bas_id: usize) -> Vec<f64> {
    let nprim = bas_nprim(cell, bas_id);
    let nctr = bas_nctr(cell, bas_id);
    let ptr = cell.mol._bas[bas_id * BAS_SLOTS + PTR_COEFF].max(0) as usize;
    (0..nprim)
        .map(|p| {
            (0..nctr).fold(0.0_f64, |cmax, ic| {
                cmax.max(cell.mol._env[ptr + ic * nprim + p].abs())
            })
        })
        .collect()
}

/// `cell.omega` — `mol._env[PTR_RANGE_OMEGA]` (`gto/mole.py:2948-2950`).
/// Zero unless a range-separation guard is active.
pub fn omega(cell: &Cell) -> f64 {
    cell.mol._env.get(PTR_RANGE_OMEGA).copied().unwrap_or(0.0)
}

// ---------------------------------------------------------------------------
// `_extract_pgto_params` — cell.py:481-500.
// ---------------------------------------------------------------------------

/// Which primitive of each shell `_extract_pgto_params` selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PgtoOp {
    /// `op = 'min'` — the most DIFFUSE primitive (smallest exponent). Drives `rcut`.
    Min,
    /// `op = 'max'` — the most COMPACT primitive (largest exponent). Drives `ke_cutoff`.
    Max,
}

/// One `(exponent, coefficient)` pair per shell.
///
/// Ports `cell.py:481-500` — the LOCAL `_extract_pgto_params`, which loops over
/// shells and picks `e.argmin()` / `e.argmax()`. (The similarly named
/// `gto/mole.py:4336` helper is a different, un-called algorithm.)
///
/// `argmin`/`argmax` in numpy return the FIRST extremal index; the strict
/// comparisons below reproduce that tie-break.
pub fn extract_pgto_params(cell: &Cell, op: PgtoOp) -> (Vec<f64>, Vec<f64>) {
    let mut es = Vec::with_capacity(cell.mol.nbas);
    let mut cs = Vec::with_capacity(cell.mol.nbas);
    for i in 0..cell.mol.nbas {
        let e = bas_exp(cell, i);
        let c = libcint_ctr_coeff_max(cell, i);
        if e.is_empty() {
            continue;
        }
        let mut idx = 0_usize;
        for (k, ek) in e.iter().enumerate().skip(1) {
            let better = match op {
                PgtoOp::Min => *ek < e[idx],
                PgtoOp::Max => *ek > e[idx],
            };
            if better {
                idx = k;
            }
        }
        es.push(e[idx]);
        cs.push(c[idx]);
    }
    (es, cs)
}

// ---------------------------------------------------------------------------
// rcut — cell.py:392-436.
// ---------------------------------------------------------------------------

/// `_estimate_rcut(alpha, l, c, precision)` — `cell.py:392-407`.
///
/// `rcut` from the overlap integral of a primitive with its own image:
/// `precision ~ (rcut^2/(2 alpha))^l exp(alpha/2 rcut^2)`. Two fixed-point
/// sweeps from `r0 = 20`, with the `4*alpha^2` kinetic-operator penalty.
pub fn estimate_rcut_pgto(alpha: f64, l: i32, c: f64, precision: f64) -> f64 {
    let theta = alpha * 0.5;
    let a1 = (alpha * 2.0).powf(-0.5);
    let norm_ang = (2.0 * l as f64 + 1.0) / (4.0 * PI);
    let mut fac = 2.0 * PI * c * c * norm_ang / theta / precision;
    let mut r0 = 20.0_f64;
    // The estimation is enough for overlap. Errors are slightly larger for the
    // kinetic operator, whose basis becomes 2*a*r*|orig-basis>; the 4*a^2*r^2
    // penalty below covers it.
    fac *= 4.0 * alpha * alpha;
    let exponent = 2.0 * l as f64 + 2.0;
    r0 = ((fac * r0 * (r0 * 0.5 + a1).powf(exponent) + 1.0).ln() / theta).sqrt();
    r0 = ((fac * r0 * (r0 * 0.5 + a1).powf(exponent) + 1.0).ln() / theta).sqrt();
    r0
}

/// `bas_rcut(cell, bas_id, precision)` — `cell.py:409-422`.
///
/// The largest distance between shell `bas_id` and its image that still reaches
/// `precision` in overlap: `_estimate_rcut` over every primitive, maximised.
pub fn bas_rcut(cell: &Cell, bas_id: usize, precision: f64) -> f64 {
    let l = bas_angular(cell, bas_id);
    let es = bas_exp(cell, bas_id);
    let cs = libcint_ctr_coeff_max(cell, bas_id);
    es.iter()
        .zip(cs.iter())
        .map(|(e, c)| estimate_rcut_pgto(*e, l, *c, precision))
        .fold(f64::NEG_INFINITY, f64::max)
}

/// `estimate_rcut(cell, precision)` — `cell.py:424-436`. The lattice-sum cutoff
/// for the whole system.
///
/// `cell.use_loose_rcut` switches to the (looser, cheaper) per-shell radii of
/// [`rcut_by_shells`]; the default path uses the most diffuse primitive of each
/// shell.
pub fn estimate_rcut(cell: &Cell, precision: f64) -> f64 {
    if cell.mol.nbas == 0 {
        return 0.01;
    }
    if cell.use_loose_rcut {
        return rcut_by_shells(cell, precision, 0.0)
            .into_iter()
            .fold(f64::NEG_INFINITY, f64::max);
    }

    let (exps, cs) = extract_pgto_params(cell, PgtoOp::Min);
    exps.iter()
        .zip(cs.iter())
        .enumerate()
        .map(|(i, (e, c))| estimate_rcut_pgto(*e, bas_angular(cell, i), *c, precision))
        .fold(f64::NEG_INFINITY, f64::max)
}

// ---------------------------------------------------------------------------
// ke_cutoff — cell.py:438-479.
// ---------------------------------------------------------------------------

/// `_estimate_ke_cutoff(alpha, l, c, precision, omega)` — `cell.py:438-451`.
/// Planewave cutoff from the nuclear-attraction integral of one primitive.
pub fn estimate_ke_cutoff_pgto(alpha: f64, l: i32, c: f64, precision: f64, omega: f64) -> f64 {
    let norm_ang = (2.0 * l as f64 + 1.0) / (4.0 * PI);
    let fac = 32.0 * PI * PI * (2.0 * PI).powf(1.5) * c * c * norm_ang
        / (2.0 * alpha).powf(2.0 * l as f64 + 0.5)
        / precision;
    let mut ecut = 20.0_f64;
    let exponent = l as f64 - 0.5;
    if omega <= 0.0 {
        ecut = (fac * (ecut * 2.0).powf(exponent) + 1.0).ln() * 4.0 * alpha;
        ecut = (fac * (ecut * 2.0).powf(exponent) + 1.0).ln() * 4.0 * alpha;
    } else {
        let theta = 1.0 / (1.0 / (4.0 * alpha) + 1.0 / (2.0 * omega * omega));
        ecut = (fac * (ecut * 2.0).powf(exponent) + 1.0).ln() * theta;
        ecut = (fac * (ecut * 2.0).powf(exponent) + 1.0).ln() * theta;
    }
    ecut
}

/// `estimate_ke_cutoff(cell, precision)` — `cell.py:453-464`. Uses the most
/// COMPACT primitive of each shell, since that one sets the resolution the
/// planewave grid has to reach.
pub fn estimate_ke_cutoff(cell: &Cell, precision: f64) -> f64 {
    if cell.mol.nbas == 0 {
        return 0.0;
    }
    let om = omega(cell);
    let (exps, cs) = extract_pgto_params(cell, PgtoOp::Max);
    exps.iter()
        .zip(cs.iter())
        .enumerate()
        .map(|(i, (e, c))| estimate_ke_cutoff_pgto(*e, bas_angular(cell, i), *c, precision, om))
        .fold(f64::NEG_INFINITY, f64::max)
}

/// `error_for_ke_cutoff(cell, ke_cutoff, omega)` — `cell.py:502-515`.
/// The integral error a given planewave cutoff leaves on the table.
///
/// `omega = None` reads `cell.omega`, exactly as upstream does.
pub fn error_for_ke_cutoff(cell: &Cell, ke_cutoff: f64, omega_arg: Option<f64>) -> f64 {
    let om = omega_arg.unwrap_or_else(|| omega(cell));
    let (exps, cs) = extract_pgto_params(cell, PgtoOp::Max);
    exps.iter()
        .zip(cs.iter())
        .enumerate()
        .map(|(i, (e, c))| {
            let l = bas_angular(cell, i) as f64;
            let norm_ang = (2.0 * l + 1.0) / (4.0 * PI);
            let fac = 32.0 * PI * PI * (2.0 * PI).powf(1.5) * c * c * norm_ang
                / (2.0 * e).powf(2.0 * l + 0.5);
            if om <= 0.0 {
                fac * (2.0 * ke_cutoff).powf(l - 0.5) * (-ke_cutoff / (4.0 * e)).exp()
            } else {
                let theta = 1.0 / (1.0 / (4.0 * e) + 1.0 / (2.0 * om * om));
                fac * (2.0 * ke_cutoff).powf(l - 0.5) * (-ke_cutoff / theta).exp()
            }
        })
        .fold(f64::NEG_INFINITY, f64::max)
}

// ---------------------------------------------------------------------------
// The bounding sphere — cell.py:517-543.
// ---------------------------------------------------------------------------

/// `get_bounding_sphere(cell, rcut)` — `cell.py:517-543`.
///
/// The half-widths `N_x` of the parallelepiped `-N_x <= n_x <= N_x` that
/// contains every lattice point within `rcut` (Martin p. 85). Axes at or beyond
/// `cell.dimension` are zeroed.
///
/// # Errors
/// [`pyscf_core::CoreError::InvalidMolecule`] if the lattice is singular.
pub fn get_bounding_sphere(cell: &Cell, rcut: f64) -> Result<[usize; 3], PyscfRsError> {
    let b = cell.reciprocal_vectors(1.0)?;
    // heights_inv = lib.norm(b, axis=1)
    let heights_inv = [norm3(&b[0]), norm3(&b[1]), norm3(&b[2])];
    let mut nimgs = [0_usize; 3];
    for (i, n) in nimgs.iter_mut().enumerate() {
        let v = (rcut * heights_inv[i]).ceil();
        *n = if v.is_finite() && v > 0.0 {
            v as usize
        } else {
            0
        };
    }
    // for i in range(cell.dimension, 3): nimgs[i] = 0
    for n in nimgs.iter_mut().skip(cell.dimension as usize) {
        *n = 0;
    }
    Ok(nimgs)
}

/// `get_nimgs(cell, precision)` — `cell.py:373-390`. `rcut` from
/// [`estimate_rcut`], then [`get_bounding_sphere`].
///
/// # Errors
/// As [`get_bounding_sphere`].
pub fn get_nimgs(cell: &Cell, precision: f64) -> Result<[usize; 3], PyscfRsError> {
    let rcut = estimate_rcut(cell, precision);
    get_bounding_sphere(cell, rcut)
}

// ---------------------------------------------------------------------------
// Per-primitive radii — cell.py:974-1024 and lib/pbc/cell.c:30-90.
// ---------------------------------------------------------------------------

/// `pgf_rcut(l, alpha, coeff, precision, rcut, max_cycle, eps)` —
/// `cell.py:974-991`, the pure-Python form.
///
/// The radius at which a primitive Gaussian's own value falls below
/// `precision`: `c*rcut^(l+2)*exp(-alpha*rcut^2) ~ precision`, solved by
/// fixed-point iteration from `max(rcut, rmin + eps)`.
///
/// **This is NOT the routine `cell.rcut_by_shells` calls.** That one goes
/// through `libpbc`, whose C twin adds a `gmax < precision` early return — see
/// [`pgf_rcut_c`]. Both are ported so either upstream entry point can be
/// reproduced exactly.
pub fn pgf_rcut(
    l: i32,
    alpha: f64,
    coeff: f64,
    precision: f64,
    rcut: f64,
    max_cycle: usize,
    eps: f64,
) -> f64 {
    let c = (coeff / precision).ln();
    let rmin = (0.5 * (l as f64 + 2.0) / alpha).sqrt() * 2.0;
    let eps = (rmin / 10.0).min(eps);
    let mut rcut = rcut.max(rmin + eps);
    for _ in 0..max_cycle {
        let rcut_last = rcut;
        rcut = (((l as f64 + 2.0) * rcut.ln() + c) / alpha).sqrt();
        if (rcut - rcut_last).abs() < eps {
            return rcut;
        }
    }
    tracing::warn!("cell.pgf_rcut failed to converge in {max_cycle} cycles.");
    rcut
}

/// `pgf_rcut` as implemented in C — `pyscf/lib/pbc/cell.c:30-59`.
///
/// Identical to [`pgf_rcut`] except for the `gmax < precision` early return:
/// when the primitive is already below `precision` at its own maximum `rmin`,
/// the radius IS `rmin` and no iteration runs. This is the version
/// [`rcut_by_shells`] uses, because upstream's `cell.rcut_by_shells` calls
/// `libpbc.rcut_by_shells`, not the Python `pgf_rcut`.
pub fn pgf_rcut_c(l: i32, alpha: f64, coeff: f64, precision: f64, r0: f64) -> f64 {
    let l = l as f64 + 2.0; // C: `l += 2;`
    let rmin = (0.5 * l / alpha).sqrt() * 2.0;
    let gmax = coeff * rmin.powf(l) * (-alpha * rmin * rmin).exp();
    if gmax < precision {
        return rmin;
    }
    let eps = (rmin / 10.0).min(RCUT_EPS);
    let c = (coeff / precision).ln();
    let mut rcut = r0.max(rmin + eps);
    let mut i = 0;
    while i < RCUT_MAX_CYCLE {
        let rcut_last = rcut;
        rcut = ((l * rcut.ln() + c) / alpha).sqrt();
        if (rcut - rcut_last).abs() < eps {
            break;
        }
        i += 1;
    }
    if i == RCUT_MAX_CYCLE {
        tracing::warn!("pgf_rcut did not converge in {RCUT_MAX_CYCLE} cycles.");
    }
    rcut
}

/// `rcut_by_shells(cell, precision, rcut)` — `cell.py:993-1024` via
/// `lib/pbc/cell.c:62-90`. One radius per shell: the largest [`pgf_rcut_c`]
/// over the shell's primitives, each taking the largest absolute contraction
/// coefficient of that primitive.
pub fn rcut_by_shells(cell: &Cell, precision: f64, r0: f64) -> Vec<f64> {
    rcut_by_shells_with_pgf(cell, precision, r0).0
}

/// [`rcut_by_shells`] plus the per-primitive radii — upstream's
/// `return_pgf_radius=True` branch (`cell.py:1006-1023`).
///
/// The inner `Vec` has one entry per primitive of that shell (upstream pads to
/// a rectangular `(nbas, max nprim)` array and leaves the tail uninitialised;
/// a ragged `Vec` cannot expose uninitialised memory).
pub fn rcut_by_shells_with_pgf(cell: &Cell, precision: f64, r0: f64) -> (Vec<f64>, Vec<Vec<f64>>) {
    let nbas = cell.mol.nbas;
    let mut shell_radius = Vec::with_capacity(nbas);
    let mut pgf_radius = Vec::with_capacity(nbas);
    for ib in 0..nbas {
        let l = bas_angular(cell, ib);
        let es = bas_exp(cell, ib);
        let cs = libcint_ctr_coeff_max(cell, ib);
        let mut rcut_max = 0.0_f64;
        let mut per_prim = Vec::with_capacity(es.len());
        for (alpha, cmax) in es.iter().zip(cs.iter()) {
            let rcut = pgf_rcut_c(l, *alpha, *cmax, precision, r0);
            per_prim.push(rcut);
            rcut_max = rcut_max.max(rcut);
        }
        shell_radius.push(rcut_max);
        pgf_radius.push(per_prim);
    }
    (shell_radius, pgf_radius)
}

// ---------------------------------------------------------------------------
// mesh — cell.py:968-972 and cell.py:1755-1767.
// ---------------------------------------------------------------------------

/// `_mesh_inf_vaccum(cell)` — `cell.py:968-972`. The `z` mesh size for an
/// infinite-vacuum low-dimensional cell:
/// `prec ~ exp(-0.436392335*mesh - 2.99944305) * nelec`, rounded UP to an even
/// number (the `z+`/`z-` symmetry needs it even).
///
/// NOTE `cell.nelectron` here is the ALL-ELECTRON count until plan 10-01 lands
/// the GTH pseudopotentials (D-PBC-11); upstream would use the valence count
/// for a pseudopotential cell, giving a slightly smaller `meshz`.
pub fn mesh_inf_vacuum(cell: &Cell) -> usize {
    let nelectron = cell.tot_electrons(1) as f64;
    let meshz = ((nelectron / cell.precision).ln() - 2.99944305) / 0.436392335;
    // int(meshz*.5 + .999) * 2 — Python int() truncates toward zero.
    let n = (meshz * 0.5 + 0.999).trunc();
    if n.is_finite() && n > 0.0 {
        (n as usize) * 2
    } else {
        0
    }
}

/// The `mesh` half of `Cell.build` — `cell.py:1755-1767`.
///
/// `ke_cutoff` comes from the user if pinned, else from [`estimate_ke_cutoff`];
/// the mesh is then `pbctools.cutoff_to_mesh(a, ke_cutoff)`, with the
/// non-periodic axes replaced by [`mesh_inf_vacuum`] in the
/// `dimension <= 2 && low_dim_ft_type == inf_vacuum` case.
///
/// # Errors
/// [`pyscf_core::CoreError::InvalidMolecule`] if the lattice is singular.
pub fn estimate_mesh(cell: &Cell) -> Result<[usize; 3], PyscfRsError> {
    let ke_cutoff = match cell.ke_cutoff {
        Some(ke) => ke,
        None => estimate_ke_cutoff(cell, cell.precision),
    };
    let a = cell.lattice_vectors();
    let mut mesh = pyscf_pbc_tools::mesh::cutoff_to_mesh(&a, ke_cutoff)?;

    if cell.dimension <= 2 && cell.low_dim_ft_type == LowDimFtType::InfVacuum {
        let meshz = mesh_inf_vacuum(cell);
        for m in mesh.iter_mut().skip(cell.dimension as usize) {
            *m = meshz;
        }
    }
    Ok(mesh)
}

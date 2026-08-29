//! `get_coulG`, `madelung` and the exchange-divergence treatments — plan 11-02.
//!
//! Ports `pyscf/pbc/tools/pbc.py:258-486` (`get_coulG`), `:487-547`
//! (`precompute_exx`) and `:548-586` (`madelung`). The geometry-only helpers
//! live one crate down in [`pyscf_pbc_tools::coulg`]; this module is the
//! `Cell`-aware driver.
//!
//! # What is implemented and what defers
//!
//! | branch | status |
//! |---|---|
//! | `dimension == 3` (and `inf_vacuum`), full range | implemented |
//! | `omega != 0` long/short-range attenuation | implemented |
//! | `exxdiv = ewald` probe-charge correction at `G+k = 0` | implemented |
//! | `dimension == 2` analytic truncation | implemented (the kernel; `madelung` still needs the 2-D Ewald of plan 12-08) |
//! | `dimension == 0` truncated sphere | implemented |
//! | `dimension == 1` | `NotYetImplemented { phase: 12 }` — upstream raises `NotImplementedError` too |
//! | `exxdiv = vcut_sph` / `vcut_ws` | implemented (`exxdiv_vcut`); both REFUSE `dimension < 3`, as upstream does (`pbc.py:379-380`, `:409-410`) |

use crate::cell::Cell;
use crate::types::{ALattice, CellBuildArgs, LowDimFtType};
use pyscf_core::{CoreError, PyscfRsError, Unit};
use pyscf_gto::{AtomInput, BasisInput, MoleBuildArgs};
use pyscf_pbc_tools::coulg as core;
use pyscf_pbc_tools::mat3::norm3;
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Mutex, OnceLock};

pub use pyscf_pbc_tools::coulg::ExxDiv;

/// Keyword arguments of upstream's
/// `get_coulG(cell, k, exx, mf, mesh, Gv, wrap_around, omega)`.
///
/// [`Default`] reproduces upstream's defaults: `k = 0`, `exx = False`,
/// `mf = None`, `mesh = None` (use `cell.mesh`), `Gv = None` (build it),
/// `wrap_around = True`, `omega = None` (use `cell.omega`).
#[derive(Debug, Clone, Default)]
pub struct CoulGArgs<'a> {
    /// The k-point the kernel is evaluated at.
    pub k: [f64; 3],
    /// `exxdiv`. `None` is upstream's `False`/`None`.
    pub exxdiv: Option<ExxDiv>,
    /// `mf.kpts` — the full k-mesh, which fixes `Nk` in the Ewald correction.
    /// `None` means "one k-point", upstream's `k.reshape(1,3)`.
    pub kpts: Option<&'a [[f64; 3]]>,
    /// FFT mesh; `None` uses `cell.mesh`.
    pub mesh: Option<[usize; 3]>,
    /// Pre-built G-vectors; `None` builds them from `mesh`.
    pub gv: Option<&'a [[f64; 3]]>,
    /// Fold high-frequency `k+G` back into the box. Upstream default `True`.
    pub wrap_around: bool,
    /// EXPLICIT range-separation parameter. `None` uses `cell.omega`; the
    /// distinction matters because an explicit `omega` makes the Ewald probe
    /// charge use the FULL-range Coulomb interaction (`pbc.py:480-484`).
    pub omega: Option<f64>,
}

impl<'a> CoulGArgs<'a> {
    /// Upstream's defaults, with `wrap_around = True`.
    pub fn new() -> Self {
        Self {
            wrap_around: true,
            ..Default::default()
        }
    }
}

/// `get_coulG(cell, ...)` — `pbc.py:258-486`.
///
/// # Errors
/// * [`PyscfRsError::NotYetImplemented`] for `dimension == 1`, for
///   `exxdiv in {vcut_sph, vcut_ws}`, and for a `dimension == 2` cell whose
///   `exxdiv = ewald` correction needs the 2-D Ewald sum (plan 12-08);
/// * [`CoreError::InvalidMolecule`] for a non-cubic `dimension == 0` box or an
///   insufficient one for the attenuated kernel — both `RuntimeError`/`assert`
///   upstream;
/// * propagates [`crate::gv::get_gv_weights`].
pub fn get_coulg(cell: &Cell, args: CoulGArgs<'_>) -> Result<Vec<f64>, PyscfRsError> {
    let mesh = match args.mesh {
        Some(m) => m,
        None => cell.try_mesh()?,
    };
    let owned_gv;
    let gv: &[[f64; 3]] = match args.gv {
        Some(g) => g,
        None => {
            owned_gv = crate::gv::get_gv(cell, Some(mesh))?;
            &owned_gv
        }
    };

    // pbc.py:299-303
    let cell_omega = crate::cutoff::omega(cell);
    let _omega = args.omega.unwrap_or(cell_omega);

    // pbc.py:305-352 — the 0-D truncated sphere.
    if cell.dimension == 0 && cell.low_dim_ft_type != LowDimFtType::InfVacuum {
        let a = cell.lattice_vectors();
        for i in 0..3 {
            for j in 0..3 {
                let want = if i == j { a[0][0] } else { 0.0 };
                if (a[i][j] - want).abs() >= 1e-6 {
                    return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                        "get_coulG: cell.dimension = 0 requires a cubic box".into(),
                    )));
                }
            }
        }
        let rc = a[0][0] / 2.0;
        if _omega != 0.0 && _omega.abs() * rc < 2.0 {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                "get_coulG: insufficient box size for the truncated range-separated \
                 Coulomb potential in the 0D case"
                    .into(),
            )));
        }
        let absg: Vec<f64> = gv.iter().map(|g| norm3(g)).collect();
        return Ok(core::coulg_0d(&absg, rc, _omega));
    }

    // pbc.py:354-364 — the k + G table, folded when asked for.
    let ksum = args.k[0].abs() + args.k[1].abs() + args.k[2].abs();
    let kg: Vec<[f64; 3]> = if ksum > 1e-9 {
        if args.wrap_around {
            let b = cell.reciprocal_vectors_2pi()?;
            core::gv_wrap_around(&b, gv, args.k, mesh, cell.dimension)
        } else {
            gv.iter()
                .map(|g| [g[0] + args.k[0], g[1] + args.k[1], g[2] + args.k[2]])
                .collect()
        }
    } else {
        gv.to_vec()
    };

    let absg2 = core::abs_g2(&kg);
    let g0_idx: Vec<usize> = absg2
        .iter()
        .enumerate()
        .filter(|(_, g2)| **g2 == 0.0)
        .map(|(i, _)| i)
        .collect();

    // pbc.py:366-371 — Nk comes from mf.kpts when the caller has one.
    let nk = args.kpts.map_or(1, <[[f64; 3]]>::len);

    let mut coulg = match args.exxdiv {
        // pbc.py:373-380 — the spherically truncated Coulomb kernel.
        Some(ExxDiv::VcutSph) => crate::exxdiv_vcut::coulg_vcut_sph(cell, &absg2, nk)?,
        // pbc.py:382-410 — the Wigner-Seitz truncated kernel. `precompute_exx`
        // is upstream's `mf._ws_exx` cache; this port has no `mf` to hang it on,
        // so it is built here from the same inputs. It depends only on the cell
        // and the k-mesh, so it is a pure function of what is already in `args`.
        Some(ExxDiv::VcutWs) => {
            let kpts = args.kpts.ok_or(PyscfRsError::Core(CoreError::InvalidMolecule(
                "get_coulG with exxdiv = 'vcut_ws' needs the sampling k-points \
                 (upstream reads them off `mf.kpts` — pbc.py:382-389)"
                    .to_string(),
            )))?;
            let ws = crate::exxdiv_vcut::precompute_exx(cell, kpts)?;
            crate::exxdiv_vcut::coulg_vcut_ws(cell, &kg, &absg2, &ws)?
        }
        // pbc.py:412-454 — the Ewald-probe-charge family.
        _ => {
            if cell.dimension == 3 || cell.low_dim_ft_type == LowDimFtType::InfVacuum {
                core::coulg_full_range_3d(&absg2)
            } else if cell.dimension == 2 {
                let b = cell.reciprocal_vectors_2pi()?;
                let b2 = norm3(&b[2]);
                let mut v = core::coulg_2d(&kg, &absg2, b2);
                let g0v = core::coulg_2d_g0(b2);
                for i in &g0_idx {
                    v[*i] = g0v;
                }
                v
            } else if cell.dimension == 1 {
                return Err(PyscfRsError::NotYetImplemented {
                    phase: 12,
                    what: "get_coulG for dimension = 1 (pbc.py:433-451; upstream raises \
                           NotImplementedError — 'truncated coulG for dimension=1 is \
                           numerically inaccurate')",
                });
            } else {
                return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                    "get_coulG: dimension = {} with low_dim_ft_type = {:?} is not supported",
                    cell.dimension, cell.low_dim_ft_type
                ))));
            }
        }
    };

    // pbc.py:456-471 — range separation.
    core::apply_omega(&mut coulg, &absg2, _omega);

    // pbc.py:473-484 — the Ewald probe charge. An EXPLICIT `omega` kwarg makes
    // the correction use the full-range interaction (`madelung(..., omega=0)`).
    if cell.dimension > 0 && args.exxdiv == Some(ExxDiv::Ewald) && !g0_idx.is_empty() {
        let owned_k = [args.k];
        let kpts: &[[f64; 3]] = args.kpts.unwrap_or(&owned_k);
        let mad_omega = if args.omega.is_none() { None } else { Some(0.0) };
        let mad = madelung(cell, kpts, mad_omega)?;
        let shift = nk as f64 * cell.vol() * mad;
        for i in &g0_idx {
            coulg[*i] += shift;
        }
    }
    Ok(coulg)
}

/// Convenience wrapper for the overwhelmingly common call
/// `get_coulG(cell, mesh=mesh, Gv=Gv)` — `k = 0`, no exxdiv, `cell.omega`.
///
/// # Errors
/// As [`get_coulg`].
pub fn get_coulg_at_gv(
    cell: &Cell,
    mesh: [usize; 3],
    gv: &[[f64; 3]],
) -> Result<Vec<f64>, PyscfRsError> {
    get_coulg(
        cell,
        CoulGArgs {
            mesh: Some(mesh),
            gv: Some(gv),
            ..CoulGArgs::new()
        },
    )
}

// ---------------------------------------------------------------------------
// madelung
// ---------------------------------------------------------------------------

/// Cache key for [`madelung`]: the supercell lattice, the precision, the
/// dimension and `omega`, all as raw bits so the key is exact.
type MadelungKey = ([u64; 9], u64, u8, u8, u64);

fn madelung_cache() -> &'static Mutex<HashMap<MadelungKey, f64>> {
    static CACHE: OnceLock<Mutex<HashMap<MadelungKey, f64>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `madelung(cell, kpts, omega)` — `pbc.py:548-586`.
///
/// The Madelung constant of the Monkhorst-Pack SUPERCELL: build a cell whose
/// lattice is `Nk[i] * a[i]` carrying ONE unit probe charge, and return
/// `-2 * ewald()`. This is the leading finite-size error of the periodic
/// exchange integrals, and it is what `exxdiv = 'ewald'` puts back at `G+k = 0`.
///
/// Memoised on the supercell geometry — upstream recomputes it on every call,
/// and `_ewald_exxdiv_for_G0` calls it once per J/K build.
///
/// # Errors
/// * propagates [`crate::ewald::ewald`], including its
///   `NotYetImplemented { phase: 12 }` for `dimension == 2`;
/// * [`CoreError::InvalidMolecule`] if the probe cell cannot be built.
pub fn madelung(
    cell: &Cell,
    kpts: &[[f64; 3]],
    omega: Option<f64>,
) -> Result<f64, PyscfRsError> {
    let nk = crate::lattice::get_monkhorst_pack_size_default(cell, kpts)?;
    let a = cell.lattice_vectors();
    let mut a_super = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            a_super[i][j] = nk[i] as f64 * a[i][j];
        }
    }
    let omega = omega.unwrap_or_else(|| crate::cutoff::omega(cell));

    let mut key_a = [0u64; 9];
    for i in 0..3 {
        for j in 0..3 {
            key_a[i * 3 + j] = a_super[i][j].to_bits();
        }
    }
    let key: MadelungKey = (
        key_a,
        cell.precision.to_bits(),
        cell.dimension,
        u8::from(cell.low_dim_ft_type == LowDimFtType::InfVacuum),
        omega.to_bits(),
    );
    if let Ok(c) = madelung_cache().lock() {
        if let Some(v) = c.get(&key) {
            return Ok(*v);
        }
    }

    let ecell = probe_cell(cell, a_super)?;
    let value = if omega == 0.0 {
        // pbc.py:578
        -2.0 * crate::ewald::ewald(&ecell, None, None)?
    } else {
        madelung_attenuated(cell, &ecell, omega)?
    };

    if let Ok(mut c) = madelung_cache().lock() {
        c.insert(key, value);
    }
    Ok(value)
}

/// `pbc.py:580-586` — the attenuated-Coulomb Madelung constant.
///
/// The Ewald technique is unnecessary for an attenuated kernel because
/// `4 pi/G^2 exp(-G^2/(4 omega^2))` already decays, so upstream evaluates the
/// G-space sum directly on a mesh sized by a twice-iterated cutoff estimate.
fn madelung_attenuated(cell: &Cell, ecell: &Cell, omega: f64) -> Result<f64, PyscfRsError> {
    let precision = cell.precision;
    // pbc.py:582-584 — the fixed-point Ecut estimate, run exactly twice.
    let mut ecut = 10.0_f64;
    for _ in 0..2 {
        ecut = (16.0 * PI * PI / (2.0 * omega * omega * (2.0 * ecut).sqrt()) / precision + 1.0)
            .ln()
            * 2.0
            * omega
            * omega;
    }
    let mesh = ecell.cutoff_to_mesh(ecut)?;
    let gw = crate::gv::get_gv_weights(ecell, Some(mesh))?;
    let wcoulg: Vec<f64> = get_coulg(
        ecell,
        CoulGArgs {
            mesh: Some(mesh),
            gv: Some(&gw.gv),
            omega: Some(omega.abs()),
            exxdiv: None,
            ..CoulGArgs::new()
        },
    )?
    .into_iter()
    .map(|v| v * gw.weights)
    .collect();

    let si = crate::gv::get_si(ecell, Some(&gw.gv), None, None)?;
    // ZSI = SI[0]; e_lr = 2|omega|/sqrt(pi) - sum_g |ZSI|^2 wcoulG.
    let mut terms = Vec::with_capacity(wcoulg.len());
    for (g, w) in wcoulg.iter().enumerate() {
        let (re, im) = (si.re[g], si.im[g]);
        terms.push((re * re + im * im) * w);
    }
    let e_lr = 2.0 * omega.abs() / PI.sqrt() - pyscf_algebra::oracle_sum(&terms);
    if omega > 0.0 {
        Ok(e_lr)
    } else {
        let e_fr = -2.0 * crate::ewald::ewald(ecell, None, None)?;
        Ok(e_fr - e_lr)
    }
}

/// The one-probe-charge cell of `pbc.py:551-556`.
///
/// Upstream splices a single `_atm` row of charge 1 into a shallow copy of the
/// cell and enlarges `a`; the basis arrays are left dangling because `ewald`
/// never reads them. This port builds a real one-atom `Cell` instead — an H
/// atom in a minimal basis, whose nuclear charge is likewise 1 — so the
/// `Mole` invariants that the rest of the workspace relies on still hold. Only
/// `atom_charges`, `atom_coords`, `a`, `vol`, `precision`, `dimension` and
/// `low_dim_ft_type` reach the Ewald sum, and all of those match upstream.
fn probe_cell(cell: &Cell, a_super: [[f64; 3]; 3]) -> Result<Cell, PyscfRsError> {
    Cell::build(CellBuildArgs {
        mole: MoleBuildArgs {
            atom: AtomInput::Tuples(vec![("H".into(), [0.0, 0.0, 0.0])]),
            basis: BasisInput::Name("sto-3g".into()),
            unit: Unit::Bohr,
            spin: 1,
            ..Default::default()
        },
        a: ALattice::Matrix(a_super),
        precision: cell.precision,
        dimension: cell.dimension,
        low_dim_ft_type: cell.low_dim_ft_type,
        ..Default::default()
    })
}

//! Ewald summation for the periodic nuclear-repulsion energy.
//!
//! Line-by-line port of
//! * `pyscf/pbc/gto/cell.py:650-694` — `get_ewald_params`;
//! * `pyscf/pbc/gto/cell.py:696-822` — `ewald` (the 3D branch; see below);
//! * `pyscf/pbc/gto/cell.py:824` — `energy_nuc = ewald`.
//!
//! Formulation of Martin, App. F2 (PBC-MASTER-PLAN §11):
//!
//! ```text
//! E_ewald = 1/2 sum_{L,i!=j} q_i q_j erfc(eta r)/r          (ewovrl)
//!         - eta/sqrt(pi) sum_i q_i^2
//!         - pi (sum_i q_i)^2 / (2 eta^2 Omega)               (ewself)
//!         + 1/2 (4 pi/Omega) sum_{G!=0} |ZS(G)|^2 e^{-G^2/4 eta^2} / G^2  (ewg)
//! ```
//!
//! # Device work
//!
//! The two hot loops run on the device through `pyscf-kernels` (RULE 6 — this
//! crate never names `cubecl-*` itself):
//! * **K-05** [`pyscf_kernels::ewald_rlij`] builds the `(nL, natm, natm)` table
//!   of pair distances `|R_i - R_j + L|`;
//! * **K-06** [`pyscf_kernels::ewald_gs_terms`] builds the per-G reciprocal
//!   terms.
//!
//! `erfc` is evaluated on the HOST from the device-computed distances, which is
//! what PBC-MASTER-PLAN §8.1 plan 09-08 step 2 marks as the preferred choice:
//! precision matters more here than the extra transfer, cubecl's `Float` has no
//! `erfc`, and the Abramowitz-Stegun 7.1.26 rational form is ~1.5e-7 accurate —
//! two orders too coarse for the 1e-9 Ha gate. `libm::erfc` (FDLIBM) is used
//! instead. Both reductions are host-side `oracle_sum` (§9.3), so the answer is
//! bit-identical across thread counts.
//!
//! # Deferred branches (D-PBC-20 — never a silently wrong answer)
//!
//! | upstream branch | status |
//! |---|---|
//! | `dimension == 3` | SHIPPED |
//! | `dimension <= 2 && low_dim_ft_type == 'inf_vacuum'` | Phase 12 (needs the non-uniform `get_Gv_weights` base, `cell.py:558-578`) |
//! | `dimension == 2` truncated Coulomb (`cell.py:772-800`) | Phase 12 plan 12-08 |
//! | `dimension == 0` truncated Coulomb (`cell.py:802-808`) | upstream itself raises |
//! | `use_particle_mesh_ewald` | Phase 11 — needs the 3-D FFT of plan 11-01 |
//!
//! `get_ewald_params` ships ALL of its branches: they are pure parameter
//! algebra with no grid behind them.
//!
//! # Charges follow the pseudopotential
//!
//! Upstream `cell.atom_charges()` returns the PSEUDOPOTENTIAL (valence) charge
//! when `cell.pseudo` is set, and since plan 10-01 (D-PBC-11) so does this port
//! — [`crate::Cell::build`] rewrites `_atm[CHARGE_OF]` with `Zion`. The nuclear
//! repulsion of a `gth-pade` cell is therefore MUCH smaller than the
//! all-electron one (diamond: -12.787 Ha vs -28.771 Ha), and both are gated:
//! the `EWALD_REFERENCES` numbers come from cells built without `pseudo=`, and
//! `PSEUDISED_EWALD` from cells built with it.

use crate::cell::Cell;
use crate::types::LowDimFtType;
use pyscf_algebra::{oracle_sum, select_backend};
use pyscf_core::{CoreError, PyscfRsError};
use pyscf_pbc_tools::mat3::det3;
use std::f64::consts::PI;

/// Upstream's `1e200` masking sentinel (`cell.py:733`, `cell.py:755`). Shared
/// with the device side so the two cannot drift apart.
pub use pyscf_kernels::EWALD_G0_SENTINEL;

/// Below this separation two charges are treated as the SAME point and dropped
/// from the real-space sum (`cell.py:733` — `r[r<1e-16] = 1e200`).
pub const EWALD_R_MIN: f64 = 1e-16;

fn backend_err(what: &str, e: impl std::fmt::Display) -> PyscfRsError {
    PyscfRsError::Core(CoreError::InvalidMolecule(format!("{what}: {e}")))
}

/// `get_ewald_params(cell, precision, mesh) -> (ew_eta, ew_cut)` —
/// `cell.py:650-694`.
///
/// `eta^2` is the exponent of the model Gaussian charge that screens each
/// nucleus; `ew_cut` is the real-space cutoff at which `erfc(eta r)/r` has
/// fallen below `precision`.
///
/// `mesh` is accepted for signature parity with upstream, which also ignores it.
///
/// # Errors
/// * [`CoreError::InvalidMolecule`] for `dimension > 3` (upstream raises
///   `RuntimeError`), or when the `inf_vacuum` branch needs an unset `rcut`.
pub fn get_ewald_params(
    cell: &Cell,
    precision: Option<f64>,
    _mesh: Option<[usize; 3]>,
) -> Result<(f64, f64), PyscfRsError> {
    // cell.py:670-671
    if cell.mol.natm == 0 {
        return Ok((0.0, 0.0));
    }

    // cell.py:673-674
    let precision = precision.unwrap_or(cell.precision);

    if cell.dimension == 3
        || (cell.dimension == 0 && cell.low_dim_ft_type != LowDimFtType::InfVacuum)
    {
        // cell.py:676-679
        let ew_eta = 1.0 / cell.vol().powf(1.0 / 6.0);
        let ew_cut = crate::cutoff::estimate_rcut_pgto(ew_eta * ew_eta, 0, 1.0, precision);
        Ok((ew_eta, ew_cut))
    } else if cell.dimension <= 2 && cell.low_dim_ft_type == LowDimFtType::InfVacuum {
        // cell.py:680-685 — non-uniform PW grids: a smooth model density is
        // preferred, so `eta` follows `rcut` rather than the other way round.
        let ew_cut = cell.try_rcut()?;
        let ew_eta = ((4.0 * PI * ew_cut * ew_cut / precision).ln() / (ew_cut * ew_cut))
            .max(0.1)
            .sqrt();
        Ok((ew_eta, ew_cut))
    } else if cell.dimension == 2 {
        // cell.py:686-691
        let a = cell.lattice_vectors();
        let ew_cut = a[2][2] / 2.0;
        // ewovrl ~ erfc(eta*rcut)/rcut ~ e^{-eta^2 rcut^2} < precision
        let chargs_sum: f64 = cell.mol.atom_charges().iter().map(|z| *z as f64).sum();
        let log_precision = (precision / (chargs_sum * 16.0 * PI * PI)).ln();
        let ew_eta = (-log_precision).sqrt() / ew_cut;
        Ok((ew_eta, ew_cut))
    } else {
        // cell.py:692-693
        Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "get_ewald_params: dimension={} not supported",
            cell.dimension
        ))))
    }
}

/// The real-space (overlap) part of the Ewald sum — `cell.py:729-735`.
///
/// `0.5 * sum_{L,i,j} q_i q_j erfc(eta r)/r` with `r = |R_i - R_j + L|` and the
/// `r < 1e-16 -> 1e200` self-term mask.
///
/// Distances come from K-05 on the device; `erfc` and the reduction stay on the
/// host (see the module docs).
///
/// # Errors
/// [`CoreError::InvalidMolecule`] if backend selection or the K-05 launch fails.
pub fn ewald_real_space(
    chargs: &[f64],
    coords: &[[f64; 3]],
    ls: &[[f64; 3]],
    ew_eta: f64,
) -> Result<f64, PyscfRsError> {
    let natm = coords.len();
    if natm == 0 || ls.is_empty() {
        return Ok(0.0);
    }
    let selection = select_backend().map_err(|e| backend_err("ewald: backend selection", e))?;
    let coords_flat: Vec<f64> = coords.iter().flat_map(|r| r.iter().copied()).collect();
    let ls_flat: Vec<f64> = ls.iter().flat_map(|r| r.iter().copied()).collect();
    // cell.py:729-732 — K-05 returns r in C-order over (L, i, j).
    let mut r = pyscf_kernels::ewald_rlij(&selection.client, &coords_flat, &ls_flat)
        .map_err(|e| backend_err("ewald: K-05 ewald_rlij kernel", e))?;

    // cell.py:733 — r[r < 1e-16] = 1e200. On the host so the sentinel is exact.
    for rt in r.iter_mut() {
        if *rt < EWALD_R_MIN {
            *rt = EWALD_G0_SENTINEL;
        }
    }

    // cell.py:734 — ewovrl = .5 * einsum('i,j,Lij->', chargs, chargs, erfc(eta*r)/r).
    let mut terms = Vec::with_capacity(r.len());
    for (t, rt) in r.iter().enumerate() {
        let i = (t / natm) % natm;
        let j = t % natm;
        terms.push(chargs[i] * chargs[j] * libm::erfc(ew_eta * rt) / rt);
    }
    Ok(0.5 * oracle_sum(&terms))
}

/// The self-interaction correction — `cell.py:737-740`, the last line of
/// Eq. (F.5) in Martin.
///
/// `-0.5 * (sum_i q_i^2) * 2 eta / sqrt(pi)`, plus the neutralising-background
/// term `-0.5 * (sum_i q_i)^2 * pi / (eta^2 vol)` when `dimension == 3`.
pub fn ewald_self(chargs: &[f64], ew_eta: f64, dimension: u8, vol: f64) -> f64 {
    let q_dot_q = oracle_sum(&chargs.iter().map(|q| q * q).collect::<Vec<_>>());
    let mut ewself = -0.5 * q_dot_q * 2.0 * ew_eta / PI.sqrt();
    if dimension == 3 {
        let q_sum = oracle_sum(chargs);
        ewself += -0.5 * q_sum * q_sum * PI / (ew_eta * ew_eta * vol);
    }
    ewself
}

/// `ewald(cell, ew_eta, ew_cut)` — `cell.py:696-822`. The periodic
/// nuclear-repulsion energy in Hartree.
///
/// Passing `None` for EITHER parameter re-derives BOTH from
/// [`get_ewald_params`], exactly as upstream's
/// `if ew_eta is None or ew_cut is None` does.
///
/// # Errors
/// * [`PyscfRsError::NotYetImplemented`] `{ phase: 11 }` when
///   `use_particle_mesh_ewald` is set — PME needs the 3-D FFT of plan 11-01;
/// * [`PyscfRsError::NotYetImplemented`] `{ phase: 12 }` for the
///   `dimension == 2` truncated-Coulomb branch and for `inf_vacuum` grids;
/// * [`CoreError::InvalidMolecule`] for the `dimension == 0` truncated-Coulomb
///   branch (upstream raises there too), or if a device launch fails;
/// * propagates [`crate::lattice::get_lattice_ls`] and [`crate::gv::get_si`].
pub fn ewald(cell: &Cell, ew_eta: Option<f64>, ew_cut: Option<f64>) -> Result<f64, PyscfRsError> {
    // cell.py:708-710 — "If lattice parameter is not set, the cell object is
    // treated as a mole object." A `Cell` always carries an `a` field, so the
    // faithful analogue of `cell.a is None` is a degenerate lattice.
    if det3(&cell.a) == 0.0 {
        return Ok(cell.mol.enuc());
    }

    // cell.py:712-713
    if cell.mol.natm == 0 {
        return Ok(0.0);
    }

    // cell.py:715-717
    if cell.dimension == 3 && cell.use_particle_mesh_ewald {
        return crate::ewald_pme::particle_mesh_ewald(
            cell,
            ew_eta,
            ew_cut,
            crate::ewald_pme::INTERPOLATION_ORDER,
        );
    }

    // cell.py:719-720
    if cell.dimension == 0 && cell.low_dim_ft_type != LowDimFtType::InfVacuum {
        return Ok(cell.mol.enuc());
    }

    // cell.py:722
    let chargs: Vec<f64> = cell.mol.atom_charges().iter().map(|z| *z as f64).collect();

    // cell.py:724-725 — one `None` re-derives BOTH.
    let (ew_eta, ew_cut) = match (ew_eta, ew_cut) {
        (Some(eta), Some(cut)) => (eta, cut),
        _ => get_ewald_params(cell, None, None)?,
    };

    // cell.py:726-728
    let chargs_sum = oracle_sum(&chargs);
    let log_precision = (cell.precision / (chargs_sum * 16.0 * PI * PI)).ln();
    let ke_cutoff = -2.0 * ew_eta * ew_eta * log_precision;
    let mesh = cell.cutoff_to_mesh(ke_cutoff)?;
    tracing::debug!("mesh for ewald {mesh:?}");

    // cell.py:730-731
    let coords = cell.mol.atom_coords();
    let lall = crate::lattice::get_lattice_ls(cell, Some(ew_cut), None, true)?;

    // cell.py:733-736
    let ewovrl = ewald_real_space(&chargs, &coords, &lall, ew_eta)?;

    // cell.py:738-741
    let ewself = ewald_self(&chargs, ew_eta, cell.dimension, cell.vol());

    // cell.py:743-771 — the G-space sum. Upstream's own comment: Eq. (F.6) in
    // Martin is off by a factor of 2, the exponent is wrong (8 -> 4) and the
    // square is in the wrong place. The formula actually implemented is
    //   1/2 * 4 pi / Omega sum_{G != 0} |ZS(G)|^2 exp(-|G|^2 / 4 eta^2)
    // with ZS(G) = sum_a Z_a exp(i G . R_a).
    let ewg = if cell.dimension == 3 || cell.low_dim_ft_type == LowDimFtType::InfVacuum {
        ewald_g_space(cell, &chargs, mesh, ew_eta)?
    } else if cell.dimension == 2 {
        // cell.py:773-800 — truncated Coulomb, Sundararaman & Arias PRB 87 (2013).
        return Err(PyscfRsError::NotYetImplemented {
            phase: 12,
            what: "ewald for dimension = 2 (truncated-Coulomb G-space sum, \
                   cell.py:773-800) — PBC-MASTER-PLAN plan 12-08 (D-PBC-20)",
        });
    } else if cell.dimension == 0 {
        // cell.py:802-808 — upstream raises here too.
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "ewald: Ewald with truncated Coulomb (dimension = 0, \
             low_dim_ft_type = inf_vacuum) is not defined upstream either \
             (cell.py:802-808)"
                .to_string(),
        )));
    } else {
        // cell.py:810-815
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "ewald: no method for PBC dimension {}, dim-type {:?}; \
             low_dim_ft_type = inf_vacuum should be set",
            cell.dimension, cell.low_dim_ft_type
        ))));
    };

    tracing::debug!("Ewald components = {ewovrl:.15}, {ewself:.15}, {ewg:.15}");
    Ok(ewovrl + ewself + ewg)
}

/// The reciprocal-space part of the Ewald sum — `cell.py:753-770`.
///
/// `ZSI[g] = sum_a q_a SI[a, g]` is built from the SEPARABLE structure factor
/// (`cell.py:760-766` — the same `SIx (x) SIy (x) SIz` outer product upstream
/// uses, which is what [`crate::gv::get_si`] does when `gv` is `None`), then
/// K-06 forms the per-G term and `oracle_sum` reduces it.
///
/// # Errors
/// As [`crate::gv::get_gv_weights`] / [`crate::gv::get_si`], plus
/// [`CoreError::InvalidMolecule`] if the K-06 launch fails.
pub fn ewald_g_space(
    cell: &Cell,
    chargs: &[f64],
    mesh: [usize; 3],
    ew_eta: f64,
) -> Result<f64, PyscfRsError> {
    // cell.py:753 — Gv, Gvbase, weights = cell.get_Gv_weights(mesh)
    let gw = crate::gv::get_gv_weights(cell, Some(mesh))?;
    let ngrids = gw.gv.len();
    if ngrids == 0 {
        return Ok(0.0);
    }

    // cell.py:759-766 — the separable SI, then ZSI = einsum('i,ix,iy,iz->xyz').
    let si = crate::gv::get_si(cell, None, Some(mesh), None)?;
    let natm = chargs.len();
    debug_assert_eq!(si.re.len(), natm * ngrids);
    let mut zsi_re = vec![0.0_f64; ngrids];
    let mut zsi_im = vec![0.0_f64; ngrids];
    for (a, q) in chargs.iter().enumerate() {
        let row = a * ngrids;
        for g in 0..ngrids {
            zsi_re[g] += q * si.re[row + g];
            zsi_im[g] += q * si.im[row + g];
        }
    }

    // cell.py:754-757 + 767-768 — K-06.
    let selection = select_backend().map_err(|e| backend_err("ewald: backend selection", e))?;
    let gv_flat: Vec<f64> = gw.gv.iter().flat_map(|r| r.iter().copied()).collect();
    let terms = pyscf_kernels::ewald_gs_terms(
        &selection.client,
        &gv_flat,
        &zsi_re,
        &zsi_im,
        ew_eta,
        gw.weights,
    )
    .map_err(|e| backend_err("ewald: K-06 ewald_gs_terms kernel", e))?;

    // cell.py:768 — ewg = .5 * einsum('i,i,i', ZSI.conj(), ZexpG2, coulG).real
    Ok(0.5 * oracle_sum(&terms))
}

impl Cell {
    /// `cell.get_ewald_params(precision, mesh)` — see [`get_ewald_params`].
    ///
    /// # Errors
    /// As [`get_ewald_params`].
    pub fn get_ewald_params(
        &self,
        precision: Option<f64>,
        mesh: Option<[usize; 3]>,
    ) -> Result<(f64, f64), PyscfRsError> {
        get_ewald_params(self, precision, mesh)
    }

    /// `cell.ewald(ew_eta, ew_cut)` — see [`ewald`].
    ///
    /// # Errors
    /// As [`ewald`].
    pub fn ewald(&self, ew_eta: Option<f64>, ew_cut: Option<f64>) -> Result<f64, PyscfRsError> {
        ewald(self, ew_eta, ew_cut)
    }

    /// `cell.energy_nuc()` — `cell.py:824` (`energy_nuc = ewald`).
    ///
    /// The periodic nuclear-repulsion energy. Falls back to the molecular
    /// [`pyscf_core::Mole::enuc`] when the lattice is degenerate, which is this
    /// port's analogue of upstream's `cell.a is None` (`cell.py:708-710`).
    ///
    /// # Errors
    /// As [`ewald`].
    pub fn energy_nuc(&self) -> Result<f64, PyscfRsError> {
        ewald(self, None, None)
    }
}

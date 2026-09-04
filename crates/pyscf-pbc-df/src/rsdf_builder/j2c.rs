//! `_RSGDFBuilder.get_2c2e` and `weighted_ft_ao` — the two quantities that
//! make range-separated fitting different from the compensated-charge and
//! mixed routes (`pyscf/pbc/df/rsdf_builder.py:248-360` and `:701-733`),
//! plan 14-07 sub-task 7b.
//!
//! # The idea, and why the metric is a three-way sum
//!
//! GDF makes the 3-centre lattice sum converge by *neutralising* the auxiliary
//! functions; RSDF makes it converge by *splitting the kernel*:
//!
//! ```text
//! 1/r  =  erfc(omega r)/r  +  erf(omega r)/r
//!            short range         long range
//! ```
//!
//! The short-range half is summed in real space against the PLAIN auxiliary
//! cell — no fused cell, no model charges — and the long-range half, which is
//! smooth, is carried on a plane-wave grid. The metric therefore reads
//!
//! ```text
//! j2c = SR(analytic, real space)  +  FT(full)  −  FT(SR)
//!       \___________________/        \_______________/
//!         int2c2e at −omega            the LR remainder
//! ```
//!
//! Upstream writes the last two terms as one `coulG_LR` on the compact block
//! and a full `coulG` on the smooth block (`rsdf_builder.py:329-357`); with no
//! compact/smooth split (see below) the two collapse to the difference above.
//!
//! # No `_RangeSeparatedCell` — every function is treated as compact
//!
//! Upstream partitions both cells with `ft_ao._RangeSeparatedCell` and routes
//! the *smooth* half through the FFT instead of the real-space sum. This port
//! has no `_RangeSeparatedCell` (D-PBC-21 / D-PBC-23 defer it to Phase 17), so
//! every auxiliary function is treated as compact: `auxcell_c == auxcell`,
//! `compact_ao_idx` is every index, and both `recontract_1d` and
//! `recontract_2d` are the identity.
//!
//! **That is a performance deferral, not an accuracy one, and the direction is
//! the safe one.** The split exists because a diffuse function's real-space SR
//! sum needs a larger `rcut` than a tight one, so upstream diverts it to the
//! grid; keeping it in real space costs images and keeps every term. It is the
//! same posture the port already takes toward `ExtendedMole.strip_basis`
//! (14-05: the port keeps images upstream discards, worth 1.054e-09) and it is
//! recorded here for the same reason — so the residual has a name.
//!
//! The one thing that must NOT be deferred with it is `rcut`: with no split,
//! the SR radius has to cover the most diffuse auxiliary function, which is
//! what [`crate::rsdf_builder::omega::estimate_rs_2c2e_rcut`] returns.

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;
use crate::ft_ao::single::ft_ao_kpt;
use crate::gdf_builder::fuse::FusedCell;
use crate::incore::fill_2c2e;

/// `(GauxR, GauxI, rows)` — the shape [`crate::gdf_builder::j3c`]'s
/// `add_ft_j3c` consumes.
pub type WeightedFtAo = (Vec<f64>, Vec<f64>, Vec<usize>);

// The weighted Coulomb kernel lives with the compensated route's metric
// because that is where it was first needed; range separation only added the
// `omega` argument. Re-exported rather than duplicated so the two routes cannot
// drift on the sign convention.
pub use crate::gdf_builder::j2c::weighted_coulg_at;

/// [`weighted_coulg_at`], plus MDF's plane-wave edge screen when `mixed`.
///
/// `_RSMDFBuilder` ends with `weighted_coulG = MDF.weighted_coulG`
/// (`mdf.py:353`), and `weighted_coulG_SR` is defined in terms of it
/// (`rsdf_builder.py:202-203`) — so **every** kernel the mixed scheme uses,
/// short-range included, carries the `±Gmax ± 0.5` screen that keeps the
/// plane-wave set symmetric under `G -> -G`. `_RSGDFBuilder` does not.
///
/// This is not an edge case on the systems this phase gates: the screen fires
/// at a half-integer scaled k-point, which on a 2x2x2 Monkhorst-Pack mesh is
/// EVERY k-point difference. Omitting it was worth **1.176e-4 Ha** on He-fcc
/// 2x2x2 — see [`crate::mdf::builder::screen_pw_edges`] for what it does and
/// why upstream re-applies here what it removed from `tools.pbc.get_coulG`.
///
/// # Errors
/// Propagates the G-vector build and `get_coulG`.
fn rs_coulg(
    cell: &Cell,
    kpt: [f64; 3],
    mesh: [usize; 3],
    omega: Option<f64>,
    mixed: bool,
) -> Result<Vec<f64>, PbcDfError> {
    let mut w = weighted_coulg_at(cell, kpt, mesh, omega)?;
    if mixed {
        crate::mdf::builder::screen_pw_edges(cell, kpt, mesh, &mut w);
    }
    Ok(w)
}

/// `_RSGDFBuilder.get_2c2e(uniq_kpts)` — `rsdf_builder.py:248-360`.
///
/// Returns one `(naux, naux)` row-major metric per entry of `uniq_kpts`.
///
/// `mesh` is the metric's OWN mesh, tightened by
/// `estimate_ke_cutoff_for_omega(auxcell, omega, precision^1.5)` — upstream's
/// comment at `:288-290` says why it is not `self.mesh`: "2c2e integrals the
/// metric can easily cause errors in cderi tensor. self.mesh may not be enough
/// to produce required accuracy."
///
/// # Errors
/// Propagates the lattice sum, the G-vector build and `get_coulG`.
pub fn get_2c2e(
    cell: &Cell,
    fused: &FusedCell,
    uniq_kpts: &[[f64; 3]],
    omega: f64,
    mesh: [usize; 3],
    mixed: bool,
) -> Result<Vec<CTensor>, PbcDfError> {
    let auxcell = &fused.auxcell;
    let naux = fused.naux();

    // `auxcell_c.rcut = estimate_rs_2c2e_rcut(auxcell_c, omega, precision)` —
    // `rsdf_builder.py:274`, and it is LOAD-BEARING, not a tuning knob.
    //
    // The auxiliary cell's own `rcut` is an ORBITAL radius: the distance past
    // which the auxiliary FUNCTION is negligible. The lattice sum here is over
    // a two-centre COULOMB interaction `erfc(omega R)/R`, which reaches much
    // further — at `omega = 0.42` and `precision = 1e-8` the erfc alone needs
    // `R ~ 4.05/omega ~ 9.6` Bohr on top of the function extents. Summing to
    // the orbital radius instead truncates the metric.
    //
    // Measured on He-fcc `sto-3g` 2x2x2 before this line: the real-space
    // `SR_analytic` differed from the reciprocal `sum_G conj(auxG) coulG_SR
    // auxG` by **1.25e-4** at every k-point, which propagated to a 8.57e-5 Ha
    // error in the converged KRHF energy — five orders above upstream's
    // 5.222e-10 GDF/RSDF gap. With it, the two agree.
    // `rsdf_builder.py:274` passes `precision = auxcell.precision**1.5`;
    // `mdf.py:265` passes NONE, i.e. `auxcell.precision`. A SMALLER precision
    // gives a LARGER radius, so RSGDF's is the more conservative of the two and
    // is used for both — matching `mdf.py`'s looser value was MEASURED worse
    // (He-fcc 2x2x2 RSMDF: 1.160e-6 at `**1.5`, 1.324e-6 at `precision`), which
    // is what a truncated analytic sum looks like. The radius feeds a real-space
    // sum whose truncation no mesh can compensate, so erring long is right.
    let precision = auxcell.cell.precision.powf(1.5);
    let mut aux_sr = auxcell.clone();
    aux_sr.cell.rcut =
        crate::rsdf_builder::omega::estimate_rs_2c2e_rcut(&auxcell.cell, omega, Some(precision));

    // `with auxcell_c.with_short_range_coulomb(omega): pbc_intor('int2c2e', hermi=1)`
    // — `rsdf_builder.py:276-278`. The SIGN is the whole point: `Some(-omega)`
    // is `erfc(omega r)/r`.
    let sr: Vec<CTensor> = fill_2c2e(&aux_sr, 1, uniq_kpts, Some(-omega))?
        .into_iter()
        .map(|m| crate::zlinalg::forder_to_c(&m, naux, naux))
        .collect();

    let gv = pyscf_pbc_gto::gv::get_gv(&auxcell.cell, Some(mesh))?;
    let gw = pyscf_pbc_gto::gv::get_gv_weights(&auxcell.cell, Some(mesh))?;
    let ngrids = gv.len();

    let mut out = Vec::with_capacity(uniq_kpts.len());
    for (k, kpt) in uniq_kpts.iter().enumerate() {
        let gamma = pyscf_pbc_lib::kpts_helper::is_zero(kpt);

        let coulg = rs_coulg(cell, *kpt, mesh, None, mixed)?;
        let mut coulg_sr = rs_coulg(cell, *kpt, mesh, Some(-omega), mixed)?;
        // `rsdf_builder.py:337-341` — the SR kernel is finite at `G = 0`
        // (`lim_{G->0} 4 pi (1 - exp(-G^2/4 omega^2))/G^2 = pi/omega^2`) but
        // `get_coulG` zeroes `G = 0` along with the full kernel's pole. Put it
        // back, weighted, at the gamma difference only. `G0_idx = 0` is the
        // `np.fft.fftfreq` convention.
        if gamma && omega != 0.0 && cell.dimension == 3 {
            coulg_sr[0] += std::f64::consts::PI / (omega * omega) * gw.weight(0);
        }

        // Start from the analytic short-range metric and correct it on the grid.
        let mut m = sr[k].clone();

        let (agr, agi) = ft_ao_kpt(&auxcell.cell.mol, &gv, *kpt)?;
        for g in 0..ngrids {
            let (w_full, w_sr) = (coulg[g], coulg_sr[g]);
            if w_full == 0.0 && w_sr == 0.0 {
                continue;
            }
            // RSGDF: `sr_j2c -= FT(SR)` and `j2c_k += FT(full)` are summed
            // into the same matrix here, so the two weights combine BEFORE the
            // outer product rather than after it — same value, one pass, no
            // `naux x naux` temporary per G block. On the all-compact path
            // that combination is exactly upstream's `coulG_LR`.
            //
            // RSMDF has no `FT(full)` term at all (`mdf.py:288-315`): its basis
            // is orthogonalised against the plane waves, so the metric is
            // `<g|g> - <g|G><G|g>` and only the projection is removed.
            let w = if mixed { -w_sr } else { w_full - w_sr };
            if w == 0.0 {
                continue;
            }
            let base = g * naux;
            for p in 0..naux {
                // conj(auxG[p]) * coulG_LR
                let (ar, ai) = (agr[base + p] * w, -agi[base + p] * w);
                if ar == 0.0 && ai == 0.0 {
                    continue;
                }
                let row = p * naux;
                for q in 0..naux {
                    let (br, bi) = (agr[base + q], agi[base + q]);
                    m.re[row + q] += ar * br - ai * bi;
                    // `if is_zero(kpt): j2c_k += (...).real` — upstream keeps
                    // only the real part at the gamma difference
                    // (`rsdf_builder.py:352-355`).
                    if !gamma {
                        m.im[row + q] += ar * bi + ai * br;
                    }
                }
            }
        }
        out.push(m);
    }
    Ok(out)
}

/// `_RSGDFBuilder.weighted_ft_ao(kpt)` — `rsdf_builder.py:701-733`, the
/// `exclude_d_aux = False` branch (`:728-731`).
///
/// Returns `(GauxR, GauxI, rows)` in the shape
/// [`crate::gdf_builder::j3c`]'s `add_ft_j3c` consumes: `(ngrids, naux)`
/// row-major planes over EVERY auxiliary function, weighted by the LONG-RANGE
/// kernel `coulG − coulG_SR`, and the buffer rows they accumulate into —
/// `0..naux`, because the range-separated buffer has no model-charge rows.
///
/// The contrast with the other two schemes is the weight, and it is the whole
/// method: [`crate::gdf_builder::j3c::weighted_ft_ao`] carries the FULL kernel
/// on the model-charge columns only, and [`crate::mdf::builder::weighted_ft_ao`]
/// carries the FULL kernel on every column. Here it is the LR remainder on
/// every column, because the SR half is already in the real-space tensor.
///
/// # Errors
/// Propagates the G-vector build and `get_coulG`.
pub fn weighted_ft_ao(
    cell: &Cell,
    fused: &FusedCell,
    kpt: [f64; 3],
    omega: f64,
    mesh: [usize; 3],
    mixed: bool,
) -> Result<WeightedFtAo, PbcDfError> {
    let naux = fused.naux();
    let auxcell = &fused.auxcell;
    let gv = pyscf_pbc_gto::gv::get_gv(&auxcell.cell, Some(mesh))?;
    let ngrids = gv.len();
    let (mut re, mut im) = ft_ao_kpt(&auxcell.cell.mol, &gv, kpt)?;

    let coulg_sr = rs_coulg(cell, kpt, mesh, Some(-omega), mixed)?;
    // RSMDF carries `-coulG_SR` (`mdf.py:344`), RSGDF the long-range remainder
    // `coulG - coulG_SR` (`rsdf_builder.py:730`). Same shape, and the sign
    // difference is the same one the metric makes: MDF removes a projection
    // where GDF adds a tail.
    let coulg = if mixed {
        None
    } else {
        Some(rs_coulg(cell, kpt, mesh, None, mixed)?)
    };
    for g in 0..ngrids {
        let w = match coulg.as_ref() {
            Some(full) => full[g] - coulg_sr[g],
            None => -coulg_sr[g],
        };
        for p in 0..naux {
            re[g * naux + p] *= w;
            im[g * naux + p] *= w;
        }
    }
    Ok((re, im, (0..naux).collect()))
}

/// `vbar` for the range-separated loader — `rsdf_builder.py:769`.
///
/// `pi / omega^2 / vol * aux_chg`, subtracted as `vbar * S` from the real-space
/// tensor at the gamma difference. It is NOT
/// [`crate::gdf_builder::fuse::auxbar`]: the compensated route's background
/// term removes the model charge's `G = 0` divergence, whereas this one adds
/// the `G = 0` value of the SHORT-RANGE kernel that `get_coulG` zeroed —
/// upstream's comment at `:739-740` says it explicitly, "explicitly add the G0
/// contributions here because FT will not be applied to the j3c integrals for
/// short range integrals".
///
/// # Errors
/// Propagates `_gaussian_int`.
pub fn rs_vbar(fused: &FusedCell, omega: f64, vol: f64) -> Result<Vec<f64>, PbcDfError> {
    let chg = crate::rsdf_builder::omega::gaussian_int(&fused.auxcell.cell)?;
    let f = std::f64::consts::PI / (omega * omega) / vol;
    Ok(chg.iter().map(|c| f * c).collect())
}

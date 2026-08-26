//! GTH **local** pseudopotential — `V_loc` (plan 10-05).
//!
//! Ports `pyscf/pbc/gto/pseudo/pp.py:33-95`
//! (`get_alphas`, `get_alphas_gth`, `get_vlocG`, `get_gth_vlocG`) and
//! `pp_int.py:53-113` (`get_gth_vlocG_part1`), plus the 3-D branch of
//! `tools/pbc.py:258-484` (`get_coulG`) that part 1 needs.
//!
//! # The two parts
//!
//! PRB 58, 3641 Eq (5) splits the local pseudopotential in G space into
//!
//! ```text
//! part 1 (the erf / long-range piece, one per atom):
//!     V1(G) = Z_ion · 4π/G² · exp(−½ G² r_loc²),      V1(0) = −2π Z_ion r_loc²
//!
//! part 2 (the C1..C4 short-range polynomial):
//!     V2(G) = −(2π)^{3/2} r_loc³ exp(−½ G² r_loc²) ·
//!             [ C1 + C2(3 − x) + C3(15 − 10x + x²) + C4(105 − 105x + 21x² − x³) ]
//!     with x = G² r_loc².
//! ```
//!
//! [`get_gth_vlocg`] returns their SUM — upstream's `get_gth_vlocG`, which is
//! what an FFT-based `get_pp` contracts with the structure factor.
//!
//! # What Phase 10 can and cannot finish
//!
//! Everything above is closed-form and is gated against upstream here. Turning
//! it into a real-space matrix is NOT a Phase-10 step: upstream's
//! `pp_int.get_pp_loc_part1` raises `NotImplementedError` outright and defers to
//! FFTDF (`ifft(vlocG · SI)`, Phase 11) or AFTDF (`ft_aopair`, Phase 13). The
//! short-range half has a real-space route through 3-centre lattice-sum
//! integrals, which [`crate::pseudo::vloc_part2`] implements.

use crate::cell::Cell;
use pyscf_core::{CoreError, PyscfRsError};
use std::f64::consts::PI;

/// `1/(2·sqrt(pi))` — upstream's `half_sph_norm` (`pp_int.py:534`), the s-shell
/// spherical-harmonic normalisation the `fake_cell_vloc` coefficients divide by.
pub const HALF_SPH_NORM: f64 = 0.28209479177387814;

/// `get_coulG(cell, Gv=Gv)` restricted to the plain 3-D full-range kernel —
/// `tools/pbc.py:258-484`, the `else` branch at `:412-417` plus the `omega`
/// scaling at `:456-471`.
///
/// ```text
/// coulG[g] = 4π/|k+G|²,   coulG = 0 where |k+G|² == 0
/// ```
///
/// times `exp(−G²/(4ω²))` for a long-range (`ω > 0`) or
/// `1 − exp(−G²/(4ω²))` for a short-range (`ω < 0`) attenuated kernel.
///
/// `kg` is the flat `(ngrids, 3)` table of `k + G` vectors (pass `Gv` itself for
/// `k = 0`, which is what the pseudopotential path does).
///
/// # Errors
/// [`PyscfRsError::NotYetImplemented`] `{ phase: 12 }` for `dimension < 3`
/// without `inf_vacuum` — the truncated 2-D/1-D/0-D kernels are D-PBC-20 work.
/// The `exxdiv` corrections (`vcut_sph`, `vcut_ws`, `ewald`) belong to the
/// periodic-HF driver and are Phase 11's, not this function's.
pub fn get_coulg(cell: &Cell, kg: &[[f64; 3]]) -> Result<Vec<f64>, PyscfRsError> {
    // Plan 11-02 landed the full `get_coulG` (every dimension, every exxdiv) in
    // `crate::coulg`. This entry point stays because the pseudopotential path
    // wants exactly one shape of it — `k` already folded into `kg`, no exxdiv —
    // but it must NOT keep its own copy of the kernel, or the two would drift.
    crate::coulg::get_coulg(
        cell,
        crate::coulg::CoulGArgs {
            gv: Some(kg),
            // `kg` is already `k + G`; re-adding `k` (or folding again) would be
            // wrong, so the driver is entered at `k = 0` with wrapping off.
            wrap_around: false,
            ..crate::coulg::CoulGArgs::new()
        },
    )
}

/// `get_gth_vlocG_part1(cell, Gv)` — `pp_int.py:53-113`, the `dimension == 3`
/// branch (PRB 58, 3641 Eq (5) first term).
///
/// Returns a row-major `(natm, ngrids)` table. For a pseudopotential'd atom
///
/// ```text
/// V1[ia, g] = Z_ion · coulG[g] · exp(−½ G² r_loc²)   (G != 0)
/// V1[ia, 0] = −2π Z_ion r_loc²
/// ```
///
/// and for an all-electron atom simply `Z · coulG[g]` — upstream's sign
/// convention here is POSITIVE (`pp_int.py:64`: "Note the signs -- potential
/// here is positive").
///
/// # Errors
/// As [`get_coulg`].
pub fn get_gth_vlocg_part1(cell: &Cell, gv: &[[f64; 3]]) -> Result<Vec<f64>, PyscfRsError> {
    let coulg = get_coulg(cell, gv)?;
    let ngrids = gv.len();
    let natm = cell.mol.natm;
    let charges = cell.atom_charges();
    let mut out = vec![0.0; natm * ngrids];

    for ia in 0..natm {
        let zia = charges.get(ia).copied().unwrap_or(0) as f64;
        let row = &mut out[ia * ngrids..(ia + 1) * ngrids];
        let pp = cell.atom_pseudo(ia);
        for (g, slot) in row.iter_mut().enumerate() {
            let gg = gv[g];
            let g2 = gg[0] * gg[0] + gg[1] * gg[1] + gg[2] * gg[2];
            *slot = zia * coulg[g];
            if let Some(pp) = pp {
                *slot *= (-0.5 * pp.rloc * pp.rloc * g2).exp();
                // pp_int.py:70 — the non-divergent Hartree+Vloc G=0 term.
                if g2 == 0.0 {
                    *slot += -2.0 * PI * zia * pp.rloc * pp.rloc;
                }
            }
        }
    }
    Ok(out)
}

/// `get_gth_vlocG(cell, Gv)` — `pp.py:58-95`. The FULL local pseudopotential in
/// G space, part 1 plus the `C1..C4` polynomial.
///
/// Returns a row-major `(natm, ngrids)` table.
///
/// ```text
/// x = G² r_loc²
/// V[ia, g] = V1[ia, g] − (2π)^{3/2} r_loc³ exp(−x/2) ·
///            [ C1 + C2(3 − x) + C3(15 − 10x + x²) + C4(105 − 105x + 21x² − x³) ]
/// ```
///
/// The polynomial is evaluated in exactly upstream's nested form and term order;
/// Horner would be more accurate but would not reproduce upstream bit-for-bit.
///
/// # Errors
/// As [`get_coulg`].
pub fn get_gth_vlocg(cell: &Cell, gv: &[[f64; 3]]) -> Result<Vec<f64>, PyscfRsError> {
    let mut vlocg = get_gth_vlocg_part1(cell, gv)?;
    let ngrids = gv.len();
    let two_pi_32 = (2.0 * PI).powf(1.5);

    for ia in 0..cell.mol.natm {
        let Some(pp) = cell.atom_pseudo(ia) else {
            continue;
        };
        let rloc = pp.rloc;
        let c = &pp.local_coeffs;
        let row = &mut vlocg[ia * ngrids..(ia + 1) * ngrids];
        for (g, slot) in row.iter_mut().enumerate() {
            let gg = gv[g];
            let g2 = gg[0] * gg[0] + gg[1] * gg[1] + gg[2] * gg[2];
            let x = g2 * rloc * rloc;
            let mut cfacs = 0.0;
            if !c.is_empty() {
                cfacs += c[0];
            }
            if c.len() >= 2 {
                cfacs += c[1] * (3.0 - x);
            }
            if c.len() >= 3 {
                cfacs += c[2] * (15.0 - 10.0 * x + x * x);
            }
            if c.len() >= 4 {
                cfacs += c[3] * (105.0 - 105.0 * x + 21.0 * x * x - x * x * x);
            }
            *slot -= two_pi_32 * rloc.powi(3) * (-0.5 * x).exp() * cfacs;
        }
    }
    Ok(vlocg)
}

/// `get_vlocG(cell, Gv)` — `pp.py:49-56`. An alias for [`get_gth_vlocg`]; the
/// indirection exists upstream so a non-GTH pseudopotential could be slotted in.
///
/// # Errors
/// As [`get_coulg`].
pub fn get_vlocg(cell: &Cell, gv: &[[f64; 3]]) -> Result<Vec<f64>, PyscfRsError> {
    get_gth_vlocg(cell, gv)
}

/// `get_alphas_gth(cell)` — `pp.py:43-47`: `−V_loc(G = 0)`, one value per atom.
///
/// This is the alpha parameter of the non-divergent `Hartree + V_loc` `G = 0`
/// term; the periodic-HF energy expression needs it to cancel the two separately
/// divergent pieces.
///
/// # Errors
/// As [`get_coulg`].
pub fn get_alphas_gth(cell: &Cell) -> Result<Vec<f64>, PyscfRsError> {
    let v = get_gth_vlocg(cell, &[[0.0, 0.0, 0.0]])?;
    Ok(v.into_iter().map(|x| -x).collect())
}

/// `get_alphas(cell)` — `pp.py:33-41`. An alias for [`get_alphas_gth`].
///
/// # Errors
/// As [`get_coulg`].
pub fn get_alphas(cell: &Cell) -> Result<Vec<f64>, PyscfRsError> {
    get_alphas_gth(cell)
}

/// One auxiliary Gaussian of the `fake_cell_vloc` expansion.
///
/// `V_loc`'s short-range half is a sum of `C_n r^{2n-2} exp(−r²/2r_loc²)` terms,
/// each of which is an s-type Gaussian ON an atom. Representing them as an
/// auxiliary basis turns `<phi_mu| V_loc |phi_nu>` into 3-centre integrals
/// (`pp_int.py:511-563`).
#[derive(Debug, Clone, PartialEq)]
pub struct VlocAux {
    /// The atom this auxiliary function sits on.
    pub atom: usize,
    /// Gaussian exponent `α = 1/(2 r_loc²)`.
    pub alpha: f64,
    /// The RAW `_env` contraction coefficient upstream writes
    /// (`pp_int.py:554`): `C_n / r_loc^{2n−2} / half_sph_norm`.
    ///
    /// NOTE this is *not* a unit-normalised coefficient — see
    /// [`VlocAux::rescale_from_unit_norm`].
    pub coeff: f64,
}

impl VlocAux {
    /// The factor that converts an integral computed with a UNIT-NORMALISED
    /// s-Gaussian of this exponent into upstream's raw-coefficient convention.
    ///
    /// The workspace's basis pipeline (`make_env::normalise_contractions`) always
    /// emits `gto_norm(0, α)` for a lone primitive, discarding whatever raw
    /// coefficient it was handed; upstream's `fake_cell_vloc` instead writes
    /// `coeff` into `_env` verbatim. Because a 3-centre integral is LINEAR in the
    /// auxiliary coefficient, rescaling the finished block by this ratio is
    /// exact and avoids a second, parallel basis-construction path.
    ///
    /// `gto_norm(0, α) = sqrt( 2 α^{3/2} / (π^{1/2} · Γ(3/2) ) )`, but it is
    /// taken from the same helper the pipeline uses so the two can never drift.
    pub fn rescale_from_unit_norm(&self) -> f64 {
        self.coeff / gto_norm_s(self.alpha)
    }
}

/// `gto_norm(0, alpha)` — the radial normalisation of an s primitive.
///
/// `pyscf/gto/mole.py:120-155`: `1/sqrt(gaussian_int(2l+2, 2*alpha))` with
/// `gaussian_int(n, a) = ½ Γ((n+1)/2) / a^{(n+1)/2}`, so for `l = 0`
/// `gaussian_int(2, 2α) = ½ Γ(3/2) / (2α)^{3/2}` and
/// `gto_norm = sqrt( (2α)^{3/2} · 2 / Γ(3/2) )`.
fn gto_norm_s(alpha: f64) -> f64 {
    // Γ(3/2) = sqrt(pi)/2.
    let gaussian_int_2 = 0.5 * (PI.sqrt() / 2.0) / (2.0 * alpha).powf(1.5);
    1.0 / gaussian_int_2.sqrt()
}

/// `fake_cell_vloc(cell, cn)` — `pp_int.py:511-563`.
///
/// `cn = 0` builds the erf (long-range) term — one Gaussian per atom, exponent
/// `1/(2 r_loc²)` (or `1e16`, i.e. a point charge, for an all-electron atom).
/// `cn = 1..4` build the `C_cn` short-range terms, and only for atoms whose
/// potential actually carries that coefficient.
///
/// Returns the auxiliary functions in ATOM order — the layout the 3-centre
/// driver indexes by.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] for `cn > 4`, which no GTH potential has.
pub fn fake_cell_vloc(cell: &Cell, cn: usize) -> Result<Vec<VlocAux>, PyscfRsError> {
    if cn > 4 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "fake_cell_vloc: cn = {cn}, but the GTH local part has at most C1..C4"
        ))));
    }
    let charges = cell.atom_charges();
    let mut out = Vec::new();
    for ia in 0..cell.mol.natm {
        // pp_int.py:533 — ghost atoms carry no potential.
        if charges.get(ia).copied().unwrap_or(0) == 0 {
            continue;
        }
        let pp = cell.atom_pseudo(ia);
        if cn == 0 {
            // pp_int.py:538-548 — the erf term exists for EVERY atom; an
            // all-electron one gets a delta-like 1e16 exponent (a point charge).
            let alpha = match pp {
                Some(pp) => 0.5 / (pp.rloc * pp.rloc),
                None => 1e16,
            };
            // norm = half_sph_norm / gaussian_int(2, alpha)
            let gaussian_int_2 = 0.5 * (PI.sqrt() / 2.0) / alpha.powf(1.5);
            out.push(VlocAux {
                atom: ia,
                alpha,
                coeff: HALF_SPH_NORM / gaussian_int_2,
            });
        } else if let Some(pp) = pp
            && cn <= pp.local_coeffs.len()
        {
            let alpha = 0.5 / (pp.rloc * pp.rloc);
            let coeff = pp.local_coeffs[cn - 1] / pp.rloc.powi(2 * cn as i32 - 2) / HALF_SPH_NORM;
            out.push(VlocAux {
                atom: ia,
                alpha,
                coeff,
            });
        }
    }
    Ok(out)
}

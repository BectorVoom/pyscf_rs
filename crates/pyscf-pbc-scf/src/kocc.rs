//! Occupations, Fermi level and the orbital gradient — `khf.py:161-232`,
//! `kuhf.py:105-204`.
//!
//! # The one k-point-specific thing you must get right
//!
//! There is ONE Fermi level for the whole Brillouin zone. `get_occ` therefore
//! concatenates every k-point's orbital energies, sorts the whole list, and
//! fills the lowest `nelectron * nkpts` levels — it does NOT fill each k-point
//! independently. Doing it per k is the classic periodic-SCF bug, and it
//! produces a plausible, converged, WRONG answer (a metal comes out as a set of
//! independent insulators).

use pyscf_algebra::CTensor;
use pyscf_core::{CoreError, PyscfRsError};

/// `get_occ(mf, mo_energy_kpts)` for a RESTRICTED periodic reference —
/// `khf.py:184-225`.
///
/// `nocc = cell.tot_electrons(nkpts) // 2`; every level at or below the
/// `nocc`-th lowest gets occupation 2. Upstream's comparison is `<=`, so an
/// exactly degenerate level at the Fermi energy is fully occupied even when
/// that over-fills — ported verbatim (RULE 2).
///
/// # Errors
/// [`CoreError::InvalidMolecule`] when there are more electrons than orbitals.
pub fn get_occ_restricted(
    mo_energy_kpts: &[Vec<f64>],
    nocc: usize,
) -> Result<(Vec<Vec<f64>>, f64), PyscfRsError> {
    let (fermi, _) = fermi_level(mo_energy_kpts, nocc)?;
    let occ = mo_energy_kpts
        .iter()
        .map(|e| {
            e.iter()
                .map(|v| if *v <= fermi { 2.0 } else { 0.0 })
                .collect()
        })
        .collect();
    Ok((occ, fermi))
}

/// `get_occ` for an UNRESTRICTED periodic reference — `kuhf.py:136-204`.
///
/// The two spin channels have INDEPENDENT Fermi levels, each still global over
/// the Brillouin zone. `nocc_b == 0` fills nothing rather than indexing `[-1]`.
///
/// # Errors
/// As [`get_occ_restricted`].
pub fn get_occ_unrestricted(
    mo_energy_a: &[Vec<f64>],
    mo_energy_b: &[Vec<f64>],
    nocc_a: usize,
    nocc_b: usize,
) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>, [f64; 2]), PyscfRsError> {
    let (fermi_a, _) = fermi_level(mo_energy_a, nocc_a)?;
    let occ_a: Vec<Vec<f64>> = mo_energy_a
        .iter()
        .map(|e| {
            e.iter()
                .map(|v| if *v <= fermi_a { 1.0 } else { 0.0 })
                .collect()
        })
        .collect();

    // kuhf.py:159-174 — `nocc_b == 0` is its own branch upstream.
    let (occ_b, fermi_b) = if nocc_b > 0 {
        let (f, _) = fermi_level(mo_energy_b, nocc_b)?;
        (
            mo_energy_b
                .iter()
                .map(|e| e.iter().map(|v| if *v <= f { 1.0 } else { 0.0 }).collect())
                .collect(),
            f,
        )
    } else {
        (
            mo_energy_b.iter().map(|e| vec![0.0; e.len()]).collect(),
            f64::NEG_INFINITY,
        )
    };
    Ok((occ_a, occ_b, [fermi_a, fermi_b]))
}

/// The BZ-global Fermi level: the `nocc`-th lowest of every k-point's orbital
/// energies pooled together. Also returns the LUMO when one exists.
///
/// # Errors
/// [`CoreError::InvalidMolecule`] when `nocc` exceeds the orbital count or is
/// zero.
pub fn fermi_level(
    mo_energy_kpts: &[Vec<f64>],
    nocc: usize,
) -> Result<(f64, Option<f64>), PyscfRsError> {
    let mut all: Vec<f64> = mo_energy_kpts.iter().flatten().copied().collect();
    let nmo = all.len();
    if nocc == 0 {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
            "get_occ: zero electrons — nothing to occupy".into(),
        )));
    }
    if nocc > nmo {
        return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
            "get_occ: failed to assign occupancies, Nocc ({nocc}) > Nmo ({nmo})"
        ))));
    }
    all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    Ok((all[nocc - 1], all.get(nocc).copied()))
}

/// `get_grad(mo_coeff_kpts, mo_occ_kpts, fock)` — `khf.py:227-236`.
///
/// The occupied-virtual block of the Fock matrix in the MO basis, flattened and
/// concatenated over k-points. Its norm is the SCF convergence measure that
/// complements the energy change.
///
/// `mo_coeff` is COLUMN-MAJOR `nao x nmo`; `fock` is ROW-MAJOR `nao x nao`.
pub fn get_grad(mo_coeff: &CTensor, mo_occ: &[f64], fock: &CTensor, nao: usize) -> Vec<f64> {
    let nmo = mo_occ.len();
    let occ: Vec<usize> = (0..nmo).filter(|i| mo_occ[*i] > 0.0).collect();
    let vir: Vec<usize> = (0..nmo).filter(|i| mo_occ[*i] <= 0.0).collect();
    let mut out = Vec::with_capacity(2 * occ.len() * vir.len());
    // g[a, i] = sum_{mu,nu} conj(C[mu, a]) F[mu, nu] C[nu, i]
    for &a in &vir {
        for &i in &occ {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for mu in 0..nao {
                // (F C)[mu, i]
                let mut fr = 0.0_f64;
                let mut fi = 0.0_f64;
                for nu in 0..nao {
                    let (x, y) = (fock.re[mu * nao + nu], fock.im[mu * nao + nu]);
                    let (u, v) = (mo_coeff.re[nu + i * nao], mo_coeff.im[nu + i * nao]);
                    fr += x * u - y * v;
                    fi += x * v + y * u;
                }
                let (cr, ci) = (mo_coeff.re[mu + a * nao], -mo_coeff.im[mu + a * nao]);
                re += cr * fr - ci * fi;
                im += cr * fi + ci * fr;
            }
            out.push(re);
            out.push(im);
        }
    }
    out
}

/// Euclidean norm of a flattened gradient.
pub fn norm(v: &[f64]) -> f64 {
    pyscf_algebra::oracle_dot(v, v).sqrt()
}

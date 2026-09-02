//! The unrestricted initial guess: spin-symmetry breaking.
//!
//! Port of `pyscf/scf/uhf.py:116-134` (`_break_dm_spin_symm`) and of the
//! `init_guess_by_atom` variant at `uhf.py:868-877`.
//!
//! # Why this exists (KUKS-OPTIMISATION-PLAN §2.2.1)
//!
//! Without it, `dm_a == dm_b` is an exact fixed point of the unrestricted SCF
//! map at `spin == 0`: identical channels give an identical `veff`, an
//! identical eigenproblem, an identical Fermi level and therefore identical
//! densities again, for ever. DIIS, damping and level shift are all linear and
//! preserve the symmetry, and there is no stability analysis. So a UHF/UKS
//! driver that never breaks the symmetry is structurally incapable of reaching
//! a spin-broken minimum — it can only ever re-derive the restricted solution,
//! at whatever (higher) stationary point that sits.
//!
//! Upstream breaks it BY DEFAULT: `scf_uhf_init_guess_breaksym` is `1`
//! (`uhf.py:778`) and `KUHF` re-declares the same default (`kuhf.py:417`).

use pyscf_core::{Mole, PyscfRsError};
use pyscf_gto::aoslice_by_atom;

/// The guard threshold of `uhf.py:119` — the symmetry is broken only when the
/// two channels are already essentially equal.
pub const BREAKSYM_TOL: f64 = 1e-2;

/// `_break_dm_spin_symm(mol, (dma, dmb), breaksym)` — `uhf.py:116-134`.
///
/// Both matrices are row-major `nao x nao`. Returns the pair unchanged when
/// `breaksym == 0`, when `mol.spin != 0` (a polarised cell needs no manual
/// break) or when the channels already differ by more than [`BREAKSYM_TOL`].
///
/// * `breaksym == 1` — `dmb` keeps only the INTRA-ATOMIC diagonal blocks of
///   `dma`; every inter-atomic block is zeroed. This is the localising break
///   that lets an antiferromagnetic or dissociated solution be found.
/// * `breaksym == 2` — both channels are rescaled to the doublet electron
///   counts `(N/2 + 1, N/2 - 1)` (upstream issue #1839).
///
/// # Errors
/// Propagates [`aoslice_by_atom`], and the `int1e_ovlp` evaluation on the
/// `breaksym == 2` branch.
pub fn break_dm_spin_symm(
    mol: &Mole,
    dma: &[f64],
    dmb: &[f64],
    breaksym: i32,
) -> Result<(Vec<f64>, Vec<f64>), PyscfRsError> {
    let nao = mol.nao_nr;
    if breaksym == 0 || mol.spin != 0 {
        return Ok((dma.to_vec(), dmb.to_vec()));
    }
    // `abs(dma - dmb).max() < 1e-2` — uhf.py:119.
    let spread = dma
        .iter()
        .zip(dmb)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()));
    if !(spread < BREAKSYM_TOL) {
        return Ok((dma.to_vec(), dmb.to_vec()));
    }

    if breaksym == 1 {
        // uhf.py:121-125 — remove the off-diagonal (inter-atomic) part of the
        // beta density.
        let slices = aoslice_by_atom(mol)?;
        let mut out = vec![0.0_f64; nao * nao];
        for (p0, p1) in slices {
            for i in p0..p1 {
                for j in p0..p1 {
                    out[i * nao + j] = dma[i * nao + j];
                }
            }
        }
        Ok((dma.to_vec(), out))
    } else {
        // uhf.py:126-134 — adjust the electron numbers to a doublet.
        let s1e = crate::fock::default_get_ovlp(mol)?;
        // einsum('ij,ji->', dma, s1e); `s1e` is symmetric so the transpose is
        // free, but the index order is kept literal.
        let terms: Vec<f64> = (0..nao)
            .flat_map(|i| {
                (0..nao).map(move |j| (i, j))
            })
            .map(|(i, j)| dma[i * nao + j] * s1e.data[j * nao + i])
            .collect();
        let nelec_half = pyscf_algebra::oracle_sum(&terms);
        let fa = (nelec_half + 1.0) / nelec_half;
        let fb = (nelec_half - 1.0) / nelec_half;
        Ok((
            dma.iter().map(|v| v * fa).collect(),
            dmb.iter().map(|v| v * fb).collect(),
        ))
    }
}

/// The `init_guess_by_atom` break — `uhf.py:868-877`.
///
/// A DIFFERENT scheme from [`break_dm_spin_symm`], and upstream applies it only
/// for `breaksym == 1`: alpha becomes `1e-2 * S` with the intra-atomic blocks
/// overwritten by beta's, so the break is on the ALPHA channel and reaches
/// every inter-atomic pair rather than deleting them. `breaksym == 2` falls
/// through to [`break_dm_spin_symm`].
///
/// # Errors
/// Propagates the `int1e_ovlp` evaluation and [`aoslice_by_atom`].
pub fn break_atom_guess_spin_symm(
    mol: &Mole,
    dma: &[f64],
    dmb: &[f64],
    breaksym: i32,
) -> Result<(Vec<f64>, Vec<f64>), PyscfRsError> {
    if breaksym == 0 || mol.spin != 0 {
        return Ok((dma.to_vec(), dmb.to_vec()));
    }
    if breaksym != 1 {
        return break_dm_spin_symm(mol, dma, dmb, breaksym);
    }
    let nao = mol.nao_nr;
    let s1e = crate::fock::default_get_ovlp(mol)?;
    let mut out: Vec<f64> = s1e.data.iter().map(|v| v * 1e-2).collect();
    for (p0, p1) in aoslice_by_atom(mol)? {
        for i in p0..p1 {
            for j in p0..p1 {
                out[i * nao + j] = dmb[i * nao + j];
            }
        }
    }
    Ok((out, dmb.to_vec()))
}

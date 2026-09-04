//! Fermi-Dirac and Gaussian smearing — plan 11-11, port of
//! `pyscf/scf/smearing.py:68-240` and its periodic override
//! `pyscf/pbc/scf/smearing.py:34-150`.
//!
//! Smearing replaces the step occupation with a finite-temperature one, which
//! is what makes a metallic periodic SCF converge at all. Three things come
//! with it and all three are ported:
//!
//! 1. `mu` is found by BISECTION on `sum_i f(mu) = nocc` over the pooled
//!    Brillouin-zone orbital energies (there is still ONE chemical potential);
//! 2. the entropy `S` is accumulated so the free energy
//!    `e_free = e_tot - sigma * S` and the zero-temperature extrapolation
//!    `e_zero = e_tot - sigma * S / 2` can be reported;
//! 3. the convergence gradient changes: with fractional occupations the
//!    occupied-virtual block is no longer the whole story, so upstream measures
//!    the strict LOWER TRIANGLE of the full MO-basis Fock matrix
//!    (`_get_grad_tril`).

use pyscf_core::{CoreError, PyscfRsError};
use std::f64::consts::PI;

/// Which smearing function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SmearingMethod {
    /// `f = 1/(exp((e-mu)/sigma) + 1)` — upstream's default.
    #[default]
    Fermi,
    /// `f = erfc((e-mu)/sigma) / 2`.
    Gaussian,
}

/// Smearing settings attached to a periodic SCF object.
#[derive(Debug, Clone)]
pub struct Smearing {
    /// Broadening in Hartree. Zero disables smearing.
    pub sigma: f64,
    /// Which occupation function.
    pub method: SmearingMethod,
    /// Fix the chemical potential instead of the electron count.
    pub mu0: Option<f64>,
}

impl Smearing {
    /// Fermi-Dirac smearing at `sigma` Hartree.
    pub fn fermi(sigma: f64) -> Self {
        Self {
            sigma,
            method: SmearingMethod::Fermi,
            mu0: None,
        }
    }

    /// Gaussian smearing at `sigma` Hartree.
    pub fn gaussian(sigma: f64) -> Self {
        Self {
            sigma,
            method: SmearingMethod::Gaussian,
            mu0: None,
        }
    }

    /// One orbital's occupation at chemical potential `mu`.
    ///
    /// The Fermi branch guards `de >= 40` exactly as upstream does
    /// (`smearing.py:68-73`): `exp(40)` is already `2.4e17`, so the occupation
    /// is zero to well below double precision and evaluating `exp` further out
    /// would overflow.
    pub fn occ_of(&self, e: f64, mu: f64) -> f64 {
        let de = (e - mu) / self.sigma;
        match self.method {
            SmearingMethod::Fermi => {
                if de < 40.0 {
                    1.0 / (de.exp() + 1.0)
                } else {
                    0.0
                }
            }
            SmearingMethod::Gaussian => 0.5 * libm::erfc(de),
        }
    }

    /// `_get_entropy(mo_energy, mo_occ, mu)` — `smearing.py:232-240`.
    pub fn entropy(&self, energies: &[f64], occ: &[f64], mu: f64) -> f64 {
        match self.method {
            SmearingMethod::Fermi => {
                let mut s = 0.0_f64;
                for f in occ {
                    if *f > 0.0 && *f < 1.0 {
                        s -= f * f.ln() + (1.0 - f) * (1.0 - f).ln();
                    }
                }
                s
            }
            SmearingMethod::Gaussian => {
                let mut s = 0.0_f64;
                for e in energies {
                    let x = (e - mu) / self.sigma;
                    s += (-(x * x)).exp();
                }
                s / (2.0 * PI.sqrt())
            }
        }
    }

    /// Assign smeared occupations across the whole Brillouin zone.
    ///
    /// `nelectron` is the BZ-supercell electron count (`cell.tot_electrons(nkpts)`)
    /// and `max_occ` is 2 for a restricted reference, 1 for one spin channel of
    /// an unrestricted one. Returns `(mo_occ_kpts, fermi, entropy)` where the
    /// entropy is already divided by `nkpts` and scaled by `max_occ`, matching
    /// `pbc/scf/smearing.py:122-125`.
    ///
    /// # Errors
    /// [`CoreError::InvalidMolecule`] when `sigma <= 0` or the bisection cannot
    /// bracket the target electron count.
    pub fn occupations(
        &self,
        mo_energy_kpts: &[Vec<f64>],
        nelectron: f64,
        max_occ: f64,
    ) -> Result<(Vec<Vec<f64>>, f64, f64), PyscfRsError> {
        if !(self.sigma > 0.0) {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                "smearing: sigma must be positive".into(),
            )));
        }
        let nkpts = mo_energy_kpts.len();
        let all: Vec<f64> = mo_energy_kpts
            .iter()
            .flatten()
            .copied()
            .filter(|v| v.is_finite())
            .collect();
        if all.is_empty() {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(
                "smearing: no finite orbital energies".into(),
            )));
        }
        // pbc/scf/smearing.py:113-114 — a restricted reference targets
        // `(nelectron + 1) // 2` doubly-occupied levels.
        let nocc = if max_occ == 2.0 {
            ((nelectron as usize) + 1) / 2
        } else {
            nelectron as usize
        } as f64;

        let mu = match self.mu0 {
            Some(m) => m,
            None => self.optimize_mu(&all, nocc)?,
        };
        let occ_flat: Vec<f64> = all.iter().map(|e| self.occ_of(*e, mu)).collect();
        let entropy = self.entropy(&all, &occ_flat, mu) / nkpts as f64 * max_occ;

        // `_partition_occ` — scatter back per k-point, skipping the +inf
        // placeholders `zeigh_gen` leaves for dropped linear dependencies.
        let mut out = Vec::with_capacity(nkpts);
        let mut p = 0usize;
        for e in mo_energy_kpts {
            let mut row = Vec::with_capacity(e.len());
            for v in e {
                if v.is_finite() {
                    row.push(occ_flat[p] * max_occ);
                    p += 1;
                } else {
                    row.push(0.0);
                }
            }
            out.push(row);
        }

        let mut sorted = all.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let fermi = sorted[(nocc.ceil() as usize).max(1) - 1];
        Ok((out, fermi, entropy))
    }

    /// `_smearing_optimize` — bisect `sum_i f(mu) - nocc` to machine precision.
    ///
    /// Upstream uses `scipy.optimize.bisect(xtol=1e-16)` followed by a
    /// `nextafter` polish; this is the same idea implemented directly: bisect
    /// until the bracket is one ULP wide, then keep the endpoint with the
    /// smaller residual.
    fn optimize_mu(&self, energies: &[f64], nocc: f64) -> Result<f64, PyscfRsError> {
        let lo0 = energies.iter().copied().fold(f64::INFINITY, f64::min) - 10.0;
        let hi0 = energies.iter().copied().fold(f64::NEG_INFINITY, f64::max) + 10.0;
        let root =
            |m: f64| -> f64 { energies.iter().map(|e| self.occ_of(*e, m)).sum::<f64>() - nocc };
        let (mut lo, mut hi) = (lo0, hi0);
        let (flo, fhi) = (root(lo), root(hi));
        if flo.signum() == fhi.signum() {
            return Err(PyscfRsError::Core(CoreError::InvalidMolecule(format!(
                "smearing: cannot bracket mu for nocc = {nocc} \
                 (residuals {flo} at {lo}, {fhi} at {hi})"
            ))));
        }
        // `root` is monotonically DECREASING in mu? No: more mu -> more
        // occupation, so it is INCREASING. Keep the orientation explicit rather
        // than assuming it.
        let increasing = fhi > flo;
        for _ in 0..2000 {
            let mid = 0.5 * (lo + hi);
            if mid <= lo || mid >= hi {
                break;
            }
            let f = root(mid);
            if (f < 0.0) == increasing {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        Ok(if root(lo).abs() <= root(hi).abs() {
            lo
        } else {
            hi
        })
    }
}

/// `_get_grad_tril(mo_coeff, mo_occ, fock)` — `pbc/scf/smearing.py:25-31`.
///
/// The strict lower triangle of `C^H F C`, flattened as `[re, im]` pairs. With
/// fractional occupations every off-diagonal MO-Fock element must vanish at
/// convergence, not just the occupied-virtual block.
pub fn grad_tril(
    mo_coeff: &pyscf_algebra::CTensor,
    fock: &pyscf_algebra::CTensor,
    nao: usize,
    nmo: usize,
) -> Vec<f64> {
    let mut out = Vec::new();
    for i in 1..nmo {
        for j in 0..i {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for mu in 0..nao {
                let mut fr = 0.0_f64;
                let mut fi = 0.0_f64;
                for nu in 0..nao {
                    let (x, y) = (fock.re[mu * nao + nu], fock.im[mu * nao + nu]);
                    let (u, v) = (mo_coeff.re[nu + j * nao], mo_coeff.im[nu + j * nao]);
                    fr += x * u - y * v;
                    fi += x * v + y * u;
                }
                let (cr, ci) = (mo_coeff.re[mu + i * nao], -mo_coeff.im[mu + i * nao]);
                re += cr * fr - ci * fi;
                im += cr * fi + ci * fr;
            }
            out.push(re);
            out.push(im);
        }
    }
    out
}

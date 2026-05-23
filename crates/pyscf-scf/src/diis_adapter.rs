//! SCF DIIS adapter — bridges `pyscf_diis::Diis<FockSubspace>` into the
//! pyscf-scf kernel cycle loop.
//!
//! Source: D-09 — `FockSubspace` is the SCF-side `DiisStorable` impl.
//! `err_vec_scf` lives in `pyscf_diis::cdiis` and computes `SDF − FDS`
//! per `pyscf/scf/diis.py:68-87`.
//!
//! Pitfall 9 mitigation: `FockSubspace::dot` routes through
//! `pyscf_algebra::oracle_dot` so the B-matrix inner products are
//! bit-identical across thread counts.

use pyscf_diis::{Diis, DiisError, DiisStorable, err_vec_scf};

/// Fock-matrix subspace stored in the DIIS ring buffer.
///
/// `fock` is `nao × nao` row-major (matching `pyscf_core::Density.data`'s
/// documented convention). `nao` is carried alongside so the impl can
/// reconstruct shape after `from_flat`.
#[derive(Clone, Debug)]
pub struct FockSubspace {
    pub fock: Vec<f64>,
    pub nao: usize,
}

impl DiisStorable for FockSubspace {
    fn as_flat(&self) -> &[f64] {
        &self.fock
    }
    fn from_flat(&mut self, s: &[f64]) {
        self.fock.copy_from_slice(s);
    }
    fn dot(&self, other: &Self) -> f64 {
        pyscf_algebra::oracle_dot(&self.fock, &other.fock)
    }
    fn len(&self) -> usize {
        self.fock.len()
    }
}

/// Cycle-local DIIS extrapolation step invoked from
/// `pyscf_scf::kernel_impl::scf_loop`.
///
/// Inputs:
///   - `diis`: stateful `Diis<FockSubspace>` kept across SCF cycles
///   - `s`, `d`, `f`: current overlap / density / Fock buffers (each
///     `nao × nao` row-major)
///   - `nao`: number of AOs (so `err_vec_scf` knows the matrix size)
///   - `cycle`: current SCF cycle index (0-based)
///   - `start_cycle`: `cfg.diis_start_cycle` (defaults to 1)
///
/// Output: the extrapolated Fock buffer (or `f` unchanged if
/// `cycle < start_cycle`).
///
/// Source: equivalent to the `cycle >= start_cycle` branch of
/// `pyscf/scf/hf.py:1086-1135` (the DIIS slot inside `get_fock`).
pub fn diis_step(
    diis: &mut Diis<FockSubspace>,
    s: &[f64],
    d: &[f64],
    f: &[f64],
    nao: usize,
    cycle: usize,
    start_cycle: usize,
) -> Result<Vec<f64>, DiisError> {
    if cycle < start_cycle {
        // Below the start threshold: plain Roothaan — return Fock unchanged.
        return Ok(f.to_vec());
    }
    let err = err_vec_scf(s, d, f, nao);
    let extrap = diis.extrapolate(
        FockSubspace {
            fock: f.to_vec(),
            nao,
        },
        err,
    )?;
    tracing::debug!(
        target: "pyscf_scf::diis",
        cycle = cycle,
        "DIIS-extrapolated Fock applied"
    );
    Ok(extrap.fock)
}

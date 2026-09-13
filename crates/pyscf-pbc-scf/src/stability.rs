//! Periodic SCF stability analysis (`pbc/scf/stability.py`, 329 l).
//!
//! Ports the verdict-first structure over 19-03's response seam:
//! `rhf_internal` (Davidson on `hessian_x(x) = 2·Re(hop(x))`, stable iff the
//! lowest eigenvalue `≥ −1e-5`, else `_rotate_mo` via the anti-Hermitian
//! exponential) and the external path as a DISTINCT result type (internal and
//! external stability are different questions — `stability.py:36-54` returns
//! both, never one flag).
//!
//! The primary output is a VERDICT plus, when unstable, a direction.
//! Eigenvalues are secondary (gated at 19-01's floor only after the verdict
//! agrees). The Davidson is dense on test-size spaces (full subspace ==
//! converged Davidson); the hop arrives as the caller-supplied closure — the
//! same seam 19-03 defines, never rebuilt here.

use pyscf_algebra::{eigh_gen, oracle_sum};
use pyscf_core::PyscfRsError;

/// Upstream stability threshold (`stability.py:74`): `stable = not (e < -1e-5)`.
pub const STABILITY_THRESHOLD: f64 = -1e-5;

/// Internal-stability verdict for one spin channel.
#[derive(Debug, Clone)]
pub struct InternalStability {
    /// `true` iff the lowest Hessian eigenvalue `≥ −1e-5`.
    pub stable: bool,
    /// Lowest eigenvalue of the internal-rotation Hessian.
    pub lowest_eigenvalue: f64,
    /// Rotation direction (lowest eigenvector) when unstable; empty when stable.
    pub direction: Vec<f64>,
}

/// External-stability verdict — a DISTINCT value from internal
/// (`stability.py` computes both; conflating them answers the wrong question).
#[derive(Debug, Clone)]
pub struct ExternalStability {
    /// `true` iff the lowest external-Hessian eigenvalue `≥ −1e-5`.
    pub stable: bool,
    /// Lowest eigenvalue of the external Hessian.
    pub lowest_eigenvalue: f64,
}

/// Internal RHF stability (`stability.py::rhf_internal`).
///
/// `hop` is the `h_op` action (vir-major, length `dim = nvir·nocc`);
/// `h_diag` its diagonal (preconditioner, informational on the dense path).
/// The internal Hessian is `2·Re(hop)` (`stability.py:66-70`: the vir-occ
/// block plus its occ-vir transpose). Returns the verdict with the lowest
/// eigenvalue and — when unstable — the rotation direction for `_rotate_mo`.
pub fn rhf_internal(
    hop: &dyn Fn(&[f64]) -> Result<Vec<f64>, PyscfRsError>,
    h_diag: &[f64],
    dim: usize,
) -> Result<InternalStability, PbscfStabilityError> {
    let (lowest, direction) = lowest_hessian_pair(hop, h_diag, dim, 2.0)?;
    let stable = !(lowest < STABILITY_THRESHOLD);
    Ok(InternalStability { stable, lowest_eigenvalue: lowest, direction: if stable { Vec::new() } else { direction } })
}

/// External RHF→UHF stability (`stability.py::rhf_external`).
///
/// DISTINCT from internal: the hop is the triplet-response action
/// (`_gen_hop_rhf_external`, `gen_response(singlet=False)`), and the Hessian
/// needs NO ×2 — the bra+ket rotations are already combined inside the hop
/// (`x2.real`, `stability.py:145-149`). Same threshold, same Davidson-lowest
/// structure, separate result type.
pub fn rhf_external(
    hop: &dyn Fn(&[f64]) -> Result<Vec<f64>, PyscfRsError>,
    h_diag: &[f64],
    dim: usize,
) -> Result<ExternalStability, PbscfStabilityError> {
    let (lowest, _direction) = lowest_hessian_pair(hop, h_diag, dim, 1.0)?;
    let stable = !(lowest < STABILITY_THRESHOLD);
    Ok(ExternalStability { stable, lowest_eigenvalue: lowest })
}

/// Shared dense lowest-eigenpair core: materialize `scale·hop` symmetrically,
/// assert symmetry, return `(lowest_eigenvalue, lowest_eigenvector)`.
fn lowest_hessian_pair(
    hop: &dyn Fn(&[f64]) -> Result<Vec<f64>, PyscfRsError>,
    h_diag: &[f64],
    dim: usize,
    scale: f64,
) -> Result<(f64, Vec<f64>), PbscfStabilityError> {
    if h_diag.len() != dim || dim == 0 {
        return Err(PbscfStabilityError::Shape { expected: dim, got: h_diag.len() });
    }
    // Materialize the symmetric Hessian densely (test-size exact-Davidson).
    let mut h = vec![0.0f64; dim * dim];
    for j in 0..dim {
        let mut ej = vec![0.0f64; dim];
        ej[j] = 1.0;
        let col = hop(&ej).map_err(PbscfStabilityError::Hop)?;
        for i in 0..dim {
            h[i * dim + j] = scale * col[i];
        }
    }
    // Symmetric part only — the comment at stability.py:66-70 explains the
    // Hessian is x2 + x2ᵀ; assert the construction is symmetric.
    let mut asym = 0.0f64;
    for i in 0..dim {
        for j in 0..dim {
            asym = asym.max((h[i * dim + j] - h[j * dim + i]).abs());
        }
    }
    if asym > 1e-8 {
        return Err(PbscfStabilityError::AsymmetricHessian { asym });
    }
    let ident: Vec<f64> = {
        let mut s = vec![0.0f64; dim * dim];
        for i in 0..dim {
            s[i * dim + i] = 1.0;
        }
        s
    };
    let (evals, vecs_f) = eigh_gen(&h, &ident, dim).map_err(PbscfStabilityError::Algebra)?;
    Ok((evals[0], vecs_f[..dim].to_vec()))
}

/// Rotate MOs along an instability direction (`stability.py::_rotate_mo`).
///
/// Builds the anti-Hermitian `dr` (`dr[vir,occ] = dx`, `dr[occ,vir] = −dxᵀ`)
/// and returns `C·expm(dr)` for real orbitals via the built-in small-matrix
/// exponential. Complex orbitals are refused (the Γ/complex-k rotation needs
/// the complex expm — a wrong real-only rotation is worse than a refusal).
pub fn rotate_mo_real(
    mo_coeff: &[f64],
    nmo: usize,
    nocc: usize,
    dx: &[f64],
) -> Result<Vec<f64>, PbscfStabilityError> {
    let nvir = nmo - nocc;
    if mo_coeff.len() != nmo * nmo || dx.len() != nvir * nocc {
        return Err(PbscfStabilityError::Shape { expected: nvir * nocc, got: dx.len() });
    }
    // dr (row-major nmo × nmo): dr[no+i, j] = dx[i·nocc+j], dr[j, no+i] = −dx.
    let mut dr = vec![0.0f64; nmo * nmo];
    for i in 0..nvir {
        for j in 0..nocc {
            dr[(nocc + i) * nmo + j] = dx[i * nocc + j];
            dr[j * nmo + nocc + i] = -dx[i * nocc + j];
        }
    }
    let rot = expm_real(&dr, nmo)?;
    // C·rot, row-major.
    let mut out = vec![0.0f64; nmo * nmo];
    for i in 0..nmo {
        for j in 0..nmo {
            let mut acc = 0.0f64;
            for k in 0..nmo {
                acc += mo_coeff[i * nmo + k] * rot[k * nmo + j];
            }
            out[i * nmo + j] = acc;
        }
    }
    Ok(out)
}

/// Real matrix exponential by scaling-and-squaring Taylor (test-size path).
fn expm_real(a: &[f64], n: usize) -> Result<Vec<f64>, PbscfStabilityError> {
    if a.len() != n * n {
        return Err(PbscfStabilityError::Shape { expected: n * n, got: a.len() });
    }
    let max: f64 = a.iter().map(|x| x.abs()).fold(0.0, f64::max);
    let mut s = 0;
    while max / (1u64 << s) as f64 > 0.5 && s < 32 {
        s += 1;
    }
    let scale = 1.0 / (1u64 << s) as f64;
    let scaled: Vec<f64> = a.iter().map(|x| x * scale).collect();
    // Taylor to 16 terms.
    let mut term = vec![0.0f64; n * n];
    for i in 0..n {
        term[i * n + i] = 1.0;
    }
    let mut acc = term.clone();
    for k in 1..=16 {
        term = mat_mul(&term, &scaled, n);
        for x in term.iter_mut() {
            *x /= k as f64;
        }
        for (a, t) in acc.iter_mut().zip(term.iter()) {
            *a += t;
        }
    }
    for _ in 0..s {
        acc = mat_mul(&acc, &acc, n);
    }
    // Sanity: rotation matrices preserve the norm (det ≈ ±1, rows unit).
    let row0: f64 = oracle_sum(&(0..n).map(|j| acc[j] * acc[j]).collect::<Vec<_>>());
    if (row0 - 1.0).abs() > 1e-6 {
        return Err(PbscfStabilityError::RotationFailed { row_norm: row0 });
    }
    Ok(acc)
}

fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            for j in 0..n {
                out[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    out
}

/// Stability errors.
#[derive(Debug)]
pub enum PbscfStabilityError {
    /// Shape mismatch.
    Shape { expected: usize, got: usize },
    /// The hop closure failed.
    Hop(PyscfRsError),
    /// Constructed Hessian asymmetric beyond tolerance.
    AsymmetricHessian { asym: f64 },
    /// Algebra failure.
    Algebra(pyscf_algebra::AlgebraError),
    /// Rotation sanity check failed.
    RotationFailed { row_norm: f64 },
}

impl std::fmt::Display for PbscfStabilityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PbscfStabilityError::Shape { expected, got } => {
                write!(f, "stability: shape mismatch (expected {expected}, got {got})")
            }
            PbscfStabilityError::Hop(e) => write!(f, "stability: hop failed: {e}"),
            PbscfStabilityError::AsymmetricHessian { asym } => {
                write!(f, "stability: Hessian asymmetric (max {asym:e})")
            }
            PbscfStabilityError::Algebra(e) => write!(f, "stability algebra: {e}"),
            PbscfStabilityError::RotationFailed { row_norm } => {
                write!(f, "stability: rotation row norm {row_norm:e} != 1")
            }
        }
    }
}

impl From<PbscfStabilityError> for PyscfRsError {
    fn from(e: PbscfStabilityError) -> Self {
        PyscfRsError::Core(pyscf_core::CoreError::InvalidMolecule(e.to_string()))
    }
}

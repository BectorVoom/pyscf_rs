//! k-stacked C-DIIS — plan 11-09's DIIS half, over `pyscf_diis::Diis`.
//!
//! # What is stacked and why
//!
//! Upstream (`scf/diis.py:68-87`) builds ONE error vector per SCF cycle by
//! computing `(SDF)^H - SDF` per k-point and `np.hstack`ing the results — and
//! `(SDF)^H = FDS` for Hermitian `S`, `D`, `F`, so the error is the familiar
//! `FDS - SDF`. There is a single DIIS subspace for the whole Brillouin zone,
//! not one per k-point; extrapolating each k independently would let the
//! k-points wander onto different Fock surfaces.
//!
//! # The real-valued representation
//!
//! `pyscf_diis::Diis` is real-valued (`DiisStorable::as_flat -> &[f64]`), so a
//! k-stacked COMPLEX Fock matrix is stored as `[re_0, .., re_n, im_0, .., im_n]`
//! and the error vector likewise. The Pulay B-matrix entry is then
//! `Re<e_i, e_j>` where upstream forms the full Hermitian `<e_i, e_j>`; the
//! imaginary part of that inner product is antisymmetric and contributes only
//! an (upstream-visible) imaginary residue to the coefficients. The DIIS path
//! is not part of the converged answer — the SCF fixed point is defined by
//! `FDS = SDF`, not by how the iteration got there — so this affects the
//! iteration count, never the energy. `tests/kscf.rs` pins that by converging
//! the same system with and without DIIS.

use pyscf_algebra::CTensor;
use pyscf_diis::{Diis, DiisError, DiisStorable};

use crate::types::{KDms, KMats};

/// The DIIS iterate: every `(set, k)` Fock block flattened into one real
/// vector, real parts first, then imaginary parts.
#[derive(Debug, Clone)]
pub struct KFockSubspace {
    /// `nset * nkpts * nao * nao * 2` reals.
    pub flat: Vec<f64>,
    /// Channel count.
    pub nset: usize,
    /// k-point count.
    pub nkpts: usize,
    /// AO count.
    pub nao: usize,
}

impl KFockSubspace {
    /// Flatten a `(nset, nkpts)` stack of row-major Fock matrices.
    pub fn from_fock(fock: &KDms, nao: usize) -> Self {
        let nset = fock.len();
        let nkpts = fock[0].len();
        let block = nao * nao;
        let mut flat = vec![0.0_f64; nset * nkpts * block * 2];
        let half = nset * nkpts * block;
        for (s, set) in fock.iter().enumerate() {
            for (k, m) in set.iter().enumerate() {
                let off = (s * nkpts + k) * block;
                flat[off..off + block].copy_from_slice(&m.re);
                flat[half + off..half + off + block].copy_from_slice(&m.im);
            }
        }
        Self {
            flat,
            nset,
            nkpts,
            nao,
        }
    }

    /// Rebuild the `(nset, nkpts)` stack.
    pub fn to_fock(&self) -> KDms {
        let block = self.nao * self.nao;
        let half = self.nset * self.nkpts * block;
        (0..self.nset)
            .map(|s| {
                (0..self.nkpts)
                    .map(|k| {
                        let off = (s * self.nkpts + k) * block;
                        CTensor::from_planes(
                            self.flat[off..off + block].to_vec(),
                            self.flat[half + off..half + off + block].to_vec(),
                        )
                    })
                    .collect()
            })
            .collect()
    }
}

impl DiisStorable for KFockSubspace {
    fn as_flat(&self) -> &[f64] {
        &self.flat
    }
    fn from_flat(&mut self, slice: &[f64]) {
        debug_assert_eq!(slice.len(), self.flat.len());
        self.flat.copy_from_slice(slice);
    }
    fn dot(&self, other: &Self) -> f64 {
        pyscf_algebra::oracle_dot(&self.flat, &other.flat)
    }
    fn len(&self) -> usize {
        self.flat.len()
    }
}

/// `FDS - SDF` for one k-point, as `[re, im]` — `scf/diis.py:76-79`.
///
/// All three operands are ROW-MAJOR `n x n`.
pub fn err_vec_one(s: &CTensor, d: &CTensor, f: &CTensor, n: usize) -> (Vec<f64>, Vec<f64>) {
    let sd = zmm(s, d, n);
    let sdf = zmm(&sd, f, n);
    // (SDF)^H - SDF
    let mut re = vec![0.0_f64; n * n];
    let mut im = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            re[i * n + j] = sdf.re[j * n + i] - sdf.re[i * n + j];
            im[i * n + j] = -sdf.im[j * n + i] - sdf.im[i * n + j];
        }
    }
    (re, im)
}

/// The k-stacked error vector: every `(set, k)` block's `FDS - SDF`, real parts
/// first then imaginary parts, matching [`KFockSubspace`]'s layout.
pub fn err_vec(s1e: &KMats, dms: &KDms, fock: &KDms, nao: usize) -> Vec<f64> {
    let nset = fock.len();
    let nkpts = fock[0].len();
    let block = nao * nao;
    let half = nset * nkpts * block;
    let mut out = vec![0.0_f64; 2 * half];
    for s in 0..nset {
        for k in 0..nkpts {
            let (re, im) = err_vec_one(&s1e[k], &dms[s][k], &fock[s][k], nao);
            let off = (s * nkpts + k) * block;
            out[off..off + block].copy_from_slice(&re);
            out[half + off..half + off + block].copy_from_slice(&im);
        }
    }
    out
}

/// One DIIS step: push `(fock, error)` and return the extrapolated Fock stack.
///
/// # Errors
/// Propagates [`DiisError`] from the Pulay solve.
pub fn diis_step(
    diis: &mut Diis<KFockSubspace>,
    s1e: &KMats,
    dms: &KDms,
    fock: &KDms,
    nao: usize,
) -> Result<KDms, DiisError> {
    let err = err_vec(s1e, dms, fock, nao);
    let iterate = KFockSubspace::from_fock(fock, nao);
    Ok(diis.extrapolate(iterate, err)?.to_fock())
}

/// Host row-major complex matrix multiply for `n x n` operands.
///
/// `nao` is 8 for the reference systems, so this stays on the host for the same
/// reason `pyscf_pbc_df::zlinalg::zmm_small` does.
fn zmm(a: &CTensor, b: &CTensor, n: usize) -> CTensor {
    let mut re = vec![0.0_f64; n * n];
    let mut im = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sr = 0.0_f64;
            let mut si = 0.0_f64;
            for t in 0..n {
                let (ar, ai) = (a.re[i * n + t], a.im[i * n + t]);
                let (br, bi) = (b.re[t * n + j], b.im[t * n + j]);
                sr += ar * br - ai * bi;
                si += ar * bi + ai * br;
            }
            re[i * n + j] = sr;
            im[i * n + j] = si;
        }
    }
    CTensor::from_planes(re, im)
}

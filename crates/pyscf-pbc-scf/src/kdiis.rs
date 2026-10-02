//! k-stacked C-DIIS — plan 11-09's DIIS half, over `pyscf_diis::Diis`.
//!
//! # What is stacked and why
//!
//! Upstream (`scf/diis.py:68-110`) builds ONE error vector per SCF cycle by
//! computing `(X)^H - X` per k-point, `X = C^H S D F C`, and `np.hstack`ing
//! the results. `C` is the orthonormal basis the kernel takes from the
//! eigenvectors of the INITIAL Fock matrix (`scf/hf.py:152-157`,
//! `mf_diis.Corth`); in that basis the error is `C^H (FDS - SDF) C`. There is
//! a single DIIS subspace for the whole Brillouin zone, not one per k-point;
//! extrapolating each k independently would let the k-points wander onto
//! different Fock surfaces.
//!
//! # The real-valued representation
//!
//! `pyscf_diis::Diis` is real-valued (`DiisStorable::as_flat -> &[f64]`), so a
//! k-stacked COMPLEX Fock matrix is stored as `[re_0, .., re_n, im_0, .., im_n]`
//! and the error vector likewise. The Pulay B-matrix entry is then
//! `Re<e_i, e_j>` where upstream forms the full Hermitian `<e_i, e_j>`. Each
//! block of the error is anti-Hermitian, so `<e_i, e_j> = -tr(e_i e_j)` is
//! real and nothing is lost. The DIIS path is not part of the converged
//! answer — the SCF fixed point is defined by `FDS = SDF`, not by how the
//! iteration got there. `tests/kscf.rs` pins that by converging the same
//! system with and without DIIS.

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

/// `a (m x k) · b (k x n)`, row-major: the scalar host loop for the small
/// reference cells (launch overhead dominates there), the device GEMM
/// otherwise — four `n³` products per k-point per cycle are the cost of the
/// error vector at production size.
fn zprod(a: &CTensor, b: &CTensor, m: usize, k: usize, n: usize) -> Result<CTensor, String> {
    if m.max(k).max(n) < DEVICE_GEMM_MIN {
        return Ok(zmm_rect(a, b, m, k, n));
    }
    let client = pyscf_algebra::select_backend()
        .map_err(|e| format!("backend: {e}"))?
        .client;
    pyscf_algebra::zgemm_dense(&client, a, b, m, k, n).map_err(|e| format!("zgemm {m}x{k}x{n}: {e}"))
}

/// Below this dimension the products stay on the host.
const DEVICE_GEMM_MIN: usize = 64;

/// The DIIS error of one k-point as `[re, im]` — `scf/diis.py:68-110`.
///
/// Without `corth` it is `(SDF)^H − SDF = FDS − SDF` in the AO basis
/// (`get_err_vec_orig`, `diis.py:76-79`). With `corth` — the orthonormal
/// basis upstream's kernel takes from the eigenvectors of the initial Fock
/// matrix (`scf/hf.py:152-157`) — it is `C^H (FDS − SDF) C`
/// (`get_err_vec_orth`, `diis.py:89-110`), an `nmo x nmo` block.
///
/// `s`, `d`, `f` are ROW-MAJOR `n x n`; `corth` is COLUMN-MAJOR `n x nmo`.
///
/// # Errors
/// A device GEMM failure, as text.
pub fn err_vec_one(
    s: &CTensor,
    d: &CTensor,
    f: &CTensor,
    n: usize,
    corth: Option<&CTensor>,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let sd = zprod(s, d, n, n, n)?;
    let mut sdf = zprod(&sd, f, n, n, n)?;
    let mut dim = n;
    if let Some(c) = corth {
        let nmo = c.re.len() / n.max(1);
        // C^H is the column-major buffer read row-major, conjugated; C is its
        // transpose.
        let ch = CTensor::from_planes(c.re.clone(), c.im.iter().map(|v| -v).collect());
        let mut c_rm = CTensor::zeros(n * nmo);
        for mu in 0..n {
            for i in 0..nmo {
                c_rm.re[mu * nmo + i] = c.re[mu + i * n];
                c_rm.im[mu * nmo + i] = c.im[mu + i * n];
            }
        }
        let t = zprod(&ch, &sdf, nmo, n, n)?;
        sdf = zprod(&t, &c_rm, nmo, n, nmo)?;
        dim = nmo;
    }
    // X^H - X
    let mut re = vec![0.0_f64; dim * dim];
    let mut im = vec![0.0_f64; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            re[i * dim + j] = sdf.re[j * dim + i] - sdf.re[i * dim + j];
            im[i * dim + j] = -sdf.im[j * dim + i] - sdf.im[i * dim + j];
        }
    }
    Ok((re, im))
}

/// The k-stacked error vector: every `(set, k)` block's error, real parts
/// first then imaginary parts. `corth` is indexed `set * nkpts + k`.
///
/// # Errors
/// As [`err_vec_one`].
pub fn err_vec(
    s1e: &KMats,
    dms: &KDms,
    fock: &KDms,
    nao: usize,
    corth: Option<&[CTensor]>,
) -> Result<Vec<f64>, String> {
    let nset = fock.len();
    let nkpts = fock[0].len();
    let mut res: Vec<f64> = Vec::new();
    let mut ims: Vec<f64> = Vec::new();
    for s in 0..nset {
        for k in 0..nkpts {
            let c = corth.map(|c| &c[s * nkpts + k]);
            let (re, im) = err_vec_one(&s1e[k], &dms[s][k], &fock[s][k], nao, c)?;
            res.extend_from_slice(&re);
            ims.extend_from_slice(&im);
        }
    }
    res.extend_from_slice(&ims);
    Ok(res)
}

/// One DIIS step: push `(fock, error)` and return the extrapolated Fock stack.
/// `corth` is the orthonormal basis of the error vector (see [`err_vec_one`]).
///
/// # Errors
/// The Pulay solve or a device GEMM, as text.
pub fn diis_step(
    diis: &mut Diis<KFockSubspace>,
    s1e: &KMats,
    dms: &KDms,
    fock: &KDms,
    nao: usize,
    corth: Option<&[CTensor]>,
) -> Result<KDms, String> {
    let err = err_vec(s1e, dms, fock, nao, corth)?;
    let iterate = KFockSubspace::from_fock(fock, nao);
    Ok(diis
        .extrapolate(iterate, err)
        .map_err(|e: DiisError| e.to_string())?
        .to_fock())
}

/// Host row-major complex matrix multiply, `a (m x k) · b (k x n)`.
fn zmm_rect(a: &CTensor, b: &CTensor, m: usize, k: usize, n: usize) -> CTensor {
    let mut re = vec![0.0_f64; m * n];
    let mut im = vec![0.0_f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sr = 0.0_f64;
            let mut si = 0.0_f64;
            for t in 0..k {
                let (ar, ai) = (a.re[i * k + t], a.im[i * k + t]);
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

//! `df_ao2mo` — AO and MO integrals contracted out of `cderi`
//! (`pyscf/pbc/df/df_ao2mo.py`), plan 14-05.
//!
//! # The contraction
//!
//! GDF never forms `(pq|rs)` from a grid. It fits the 3-index tensor once and
//! contracts the auxiliary index:
//!
//! ```text
//! (p^{k0} q^{k1} | r^{k2} s^{k3}) = SUM_L cderi[k0,k1][L, pq] · cderi[k2,k3][L, rs]
//! ```
//!
//! with **no conjugate on either factor**. That looks wrong next to the usual
//! `V = B† B` and it is not: `decompose_j2c` factorises the metric as
//! `(P|Q) = SUM_L B_LP B_LQ` (Cholesky of a Hermitian matrix, *not* an outer
//! product of a matrix with its adjoint at two different momenta), and
//! `cderi[k2,k3]` is built at `q = k3 − k2 = −(k1 − k0)`. The conjugation is
//! already inside the second factor. `df_ao2mo.py:74-77` writes it as
//! `zdotNN(LpqR.T, LpqI.T, LrsR, LrsI, …)`.
//!
//! # The MO transform
//!
//! `_ao2mo.r_e2` conjugates the **bra only**:
//!
//! ```text
//! z[L, i, j] = SUM_pq conj(C_i[p]) · cderi[L, p, q] · C_j[q]
//! ```
//!
//! (verified against `pyscf/lib/ao2mo/r_ao2mo.c:AO2MOmmm_r_iltj`, whose two
//! comments both say `^*` but whose arithmetic conjugates the bra alone).
//! Everything in this module that touches MO coefficients goes through
//! [`r_e2`], so the convention is stated once.
//!
//! # `ao2mo_7d` — the Phase-15 index contract
//!
//! **This is a downstream contract and it is fixed here.** The tensor is
//!
//! ```text
//! eri[ki, kj, kk][i, j, k, l]   shape (nkpts, nkpts, nkpts, nmoi, nmoj, nmok, nmol)
//! ```
//!
//! and the **fourth k-point is not free**:
//!
//! ```text
//! kl = kconserv[ki, kj, kk]      (pyscf_pbc_lib::kpts_helper::get_kconserv)
//! ```
//!
//! i.e. `k_i − k_j + k_k − k_l = 0`, which is exactly the quadruple
//! `[ki, kj, kk, kl]` that [`get_eri`]'s momentum-conservation test accepts.
//! Element `[ki,kj,kk][i,j,k,l]` is `(i^{ki} j^{kj} | k^{kk} l^{kl})` in
//! **chemists' notation**: the first MO index is conjugated, the second is not.
//!
//! **Phase 15's KMP2** consumes it as `eri[ki, ka, kj][i, a, j, b]` = `(ia|jb)`
//! with `kb = kconserv[ki, ka, kj]` (`PBC-MASTER-PLAN.md` §8.7). That is the
//! same table read with `(kj, kk) -> (ka, kj)`; no re-ordering is needed, and
//! `PBC-MASTER-PLAN.md`'s `kconserv[ki, ka, kj]` and this module's
//! `kconserv[ki, kj, kk]` are the same call under two index namings.
//!
//! Plan 13-06 refused to guess this order because a wrong guess is silent until
//! KMP2's correlation energy is wrong in the fourth digit. It is settled here
//! against upstream (`df_ao2mo.py:210-275`, `fft_ao2mo.py:344-428`,
//! `aft_ao2mo.py:294-…`, all three of which write `kl = kconserv[ki,kj,kk]`)
//! and asserted in `tests/df_ao2mo.rs`.
//!
//! # k-points are addressed by INDEX here
//!
//! Upstream's `get_eri(mydf, kpts)` takes four k-vectors and `sr_loop` looks the
//! pair up by value. This port's [`crate::gdf::sr_loop`] is index-addressed, so
//! the public functions take `[usize; 4]` into `df.kpts()`. The
//! momentum-conservation test still runs on the vectors those indices name.

use pyscf_algebra::CTensor;
use pyscf_pbc_gto::Cell;

use crate::error::PbcDfError;
use crate::gdf::{Gdf, SrBlock};
use crate::pbc_ao2mo::is_kconserv;

// ---------------------------------------------------------------------------
// MO coefficients and the shapes an ERI block can take
// ---------------------------------------------------------------------------

/// One `nao x nmo` MO coefficient block, row-major, complex.
///
/// `nmo` is NOT required to be `<= nao`: `ao2mo_7d`'s consumers pad the
/// coefficients so every k-point has the same `nmo` (upstream says so in
/// `df_ao2mo.py:222-225`, pointing at `pbc.mp.kmp2.padded_mo_coeff`), and a
/// shape contract is only testable with distinguishable dimensions.
#[derive(Debug, Clone)]
pub struct MoCoeff {
    /// Rows — AO index.
    pub nao: usize,
    /// Columns — MO index.
    pub nmo: usize,
    /// `nao * nmo` row-major.
    pub c: CTensor,
}

impl MoCoeff {
    /// A complex block from planar data.
    ///
    /// # Panics
    /// When `c` is not `nao * nmo` long.
    pub fn new(nao: usize, nmo: usize, c: CTensor) -> Self {
        assert_eq!(c.re.len(), nao * nmo, "MoCoeff: shape mismatch");
        assert_eq!(c.im.len(), nao * nmo, "MoCoeff: plane mismatch");
        Self { nao, nmo, c }
    }

    /// A real block.
    ///
    /// # Panics
    /// When `re` is not `nao * nmo` long.
    pub fn real(nao: usize, nmo: usize, re: &[f64]) -> Self {
        Self::new(nao, nmo, CTensor::from_real(re))
    }

    /// The `nao x nao` identity — `general` with four of these reproduces
    /// [`get_eri`].
    pub fn identity(nao: usize) -> Self {
        let mut re = vec![0.0; nao * nao];
        for i in 0..nao {
            re[i * nao + i] = 1.0;
        }
        Self::real(nao, nao, &re)
    }

    /// `numpy.iscomplexobj(mo)` — upstream's `all_real` test.
    pub fn is_real(&self) -> bool {
        self.c.im.iter().all(|v| *v == 0.0)
    }

    /// `ao2mo.incore.iden_coeffs(mo1, mo2)` — same shape and `max|d| < 1e-13`.
    pub fn iden_coeffs(&self, other: &Self) -> bool {
        self.nao == other.nao
            && self.nmo == other.nmo
            && self
                .c
                .re
                .iter()
                .zip(&other.c.re)
                .all(|(a, b)| (a - b).abs() < 1e-13)
            && self
                .c
                .im
                .iter()
                .zip(&other.c.im)
                .all(|(a, b)| (a - b).abs() < 1e-13)
    }
}

/// How one composite `(i, j)` axis of an [`Eri`] is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PairDims {
    /// The bra dimension.
    pub n1: usize,
    /// The ket dimension.
    pub n2: usize,
    /// `s2` packing — `n1 == n2` and only the lower triangle `i >= j` is
    /// stored, at `i*(i+1)/2 + j`. Upstream's `ijmosym == 's2'`, reachable
    /// only when the integrals are real (`_conc_mos` forces `compact = False`
    /// for anything complex).
    pub packed: bool,
}

impl PairDims {
    /// A plain `n1 x n2` axis.
    pub fn plain(n1: usize, n2: usize) -> Self {
        Self {
            n1,
            n2,
            packed: false,
        }
    }
    /// A packed `n*(n+1)/2` axis.
    pub fn packed(n: usize) -> Self {
        Self {
            n1: n,
            n2: n,
            packed: true,
        }
    }
    /// The stored length of the axis.
    pub fn len(self) -> usize {
        if self.packed {
            self.n1 * (self.n1 + 1) / 2
        } else {
            self.n1 * self.n2
        }
    }
    /// `len() == 0`.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
    /// The stored offset of `(a, b)`, and whether the pair exists in storage.
    fn offset(self, a: usize, b: usize) -> usize {
        if self.packed {
            let (hi, lo) = if a >= b { (a, b) } else { (b, a) };
            hi * (hi + 1) / 2 + lo
        } else {
            a * self.n2 + b
        }
    }
}

/// A two-index-pair integral block — `get_eri`'s and `general`'s return.
#[derive(Debug, Clone)]
pub struct Eri {
    /// `row.len() * col.len()` row-major.
    pub data: CTensor,
    /// The `(p, q)` axis.
    pub row: PairDims,
    /// The `(r, s)` axis.
    pub col: PairDims,
}

impl Eri {
    fn zeros(row: PairDims, col: PairDims) -> Self {
        Self {
            data: CTensor::zeros(row.len() * col.len()),
            row,
            col,
        }
    }

    /// `(pq|rs)` by its four separate indices, unpacking either axis.
    pub fn get(&self, p: usize, q: usize, r: usize, s: usize) -> (f64, f64) {
        let i = self.row.offset(p, q) * self.col.len() + self.col.offset(r, s);
        (self.data.re[i], self.data.im[i])
    }

    /// `ao2mo.restore(1, eri, n)` — the same block with both axes unpacked.
    pub fn restore_s1(&self) -> Self {
        let row = PairDims::plain(self.row.n1, self.row.n2);
        let col = PairDims::plain(self.col.n1, self.col.n2);
        let mut out = Self::zeros(row, col);
        for p in 0..row.n1 {
            for q in 0..row.n2 {
                for r in 0..col.n1 {
                    for s in 0..col.n2 {
                        let (re, im) = self.get(p, q, r, s);
                        let i = (p * row.n2 + q) * col.len() + r * col.n2 + s;
                        out.data.re[i] = re;
                        out.data.im[i] = im;
                    }
                }
            }
        }
        out
    }
}

/// `ao2mo_7d`'s return — see the module docs for the index contract.
#[derive(Debug, Clone)]
pub struct Eri7d {
    /// Number of k-points on each of the three free k-axes.
    pub nkpts: usize,
    /// `[nmoi, nmoj, nmok, nmol]`.
    pub nmo: [usize; 4],
    /// `nkpts^3 * nmoi * nmoj * nmok * nmol` row-major, in the index order
    /// `[ki][kj][kk][i][j][k][l]`.
    pub data: CTensor,
}

impl Eri7d {
    /// Elements in one `[ki][kj][kk]` block.
    pub fn block_len(&self) -> usize {
        self.nmo.iter().product()
    }
    /// The flat offset of `[ki][kj][kk][0][0][0][0]`.
    pub fn block_offset(&self, ki: usize, kj: usize, kk: usize) -> usize {
        ((ki * self.nkpts + kj) * self.nkpts + kk) * self.block_len()
    }
    /// The flat offset of `[ki][kj][kk][i][j][k][l]`.
    pub fn offset(&self, ki: usize, kj: usize, kk: usize, m: [usize; 4]) -> usize {
        let [_, nj, nk, nl] = self.nmo;
        self.block_offset(ki, kj, kk) + ((m[0] * nj + m[1]) * nk + m[2]) * nl + m[3]
    }
    /// `eri[ki,kj,kk][i,j,k,l]`.
    pub fn get(&self, ki: usize, kj: usize, kk: usize, m: [usize; 4]) -> (f64, f64) {
        let o = self.offset(ki, kj, kk, m);
        (self.data.re[o], self.data.im[o])
    }
}

// ---------------------------------------------------------------------------
// The two contraction kernels — `zdotNN` and `zdotNC` (`df_jk.py`)
// ---------------------------------------------------------------------------

/// `out[r, c] += sign * SUM_L a[L, r] * b[L, c]`.
fn accum_nn(out: &mut CTensor, a: &CTensor, b: &CTensor, naux: usize, nrow: usize, ncol: usize) {
    for l in 0..naux {
        let (ao, bo) = (l * nrow, l * ncol);
        for r in 0..nrow {
            let (ar, ai) = (a.re[ao + r], a.im[ao + r]);
            if ar == 0.0 && ai == 0.0 {
                continue;
            }
            let o = r * ncol;
            for c in 0..ncol {
                let (br, bi) = (b.re[bo + c], b.im[bo + c]);
                out.re[o + c] += ar * br - ai * bi;
                out.im[o + c] += ar * bi + ai * br;
            }
        }
    }
}

/// `out[r, c] += sign * SUM_L a[L, r] * conj(b[L, c])`.
fn accum_nc(out: &mut CTensor, a: &CTensor, b: &CTensor, naux: usize, nrow: usize, ncol: usize) {
    for l in 0..naux {
        let (ao, bo) = (l * nrow, l * ncol);
        for r in 0..nrow {
            let (ar, ai) = (a.re[ao + r], a.im[ao + r]);
            if ar == 0.0 && ai == 0.0 {
                continue;
            }
            let o = r * ncol;
            for c in 0..ncol {
                let (br, bi) = (b.re[bo + c], b.im[bo + c]);
                out.re[o + c] += ar * br + ai * bi;
                out.im[o + c] += ai * br - ar * bi;
            }
        }
    }
}

/// Scale a block by `sign` (`+1` or `-1`) in place — the sign upstream carries
/// on every `sr_loop` yield and folds into the `ddot`/`zdot` alpha.
fn scaled(t: &CTensor, sign: i32) -> CTensor {
    if sign == 1 {
        return t.clone();
    }
    CTensor {
        re: t.re.iter().map(|v| -v).collect(),
        im: t.im.iter().map(|v| -v).collect(),
    }
}

/// `_ao2mo.r_e2(Lpq, [Ci|Cj], slice)` — the AO→MO half-transform, bra
/// conjugated.
///
/// `blk` is one `sr_loop(..., compact = false)` yield, `(naux, nao*nao)`
/// row-major. The return is `(naux, nmoi*nmoj)` row-major.
///
/// # Panics
/// When `a.nao` or `b.nao` disagrees with `blk`'s `nao`.
pub fn r_e2(blk: &SrBlock, nao: usize, a: &MoCoeff, b: &MoCoeff) -> CTensor {
    assert_eq!(a.nao, nao, "r_e2: bra MO block has the wrong nao");
    assert_eq!(b.nao, nao, "r_e2: ket MO block has the wrong nao");
    assert_eq!(blk.ncol, nao * nao, "r_e2 needs the s1 (square) block");
    let (ni, nj) = (a.nmo, b.nmo);
    let mut out = CTensor::zeros(blk.naux * ni * nj);
    // t[i, q] = SUM_p conj(C_a[p, i]) · L[p, q]
    let mut t = CTensor::zeros(ni * nao);
    for l in 0..blk.naux {
        t.re.iter_mut().for_each(|v| *v = 0.0);
        t.im.iter_mut().for_each(|v| *v = 0.0);
        let base = l * nao * nao;
        for p in 0..nao {
            for i in 0..ni {
                let (cr, ci) = (a.c.re[p * ni + i], -a.c.im[p * ni + i]); // conj
                if cr == 0.0 && ci == 0.0 {
                    continue;
                }
                for q in 0..nao {
                    let (lr, li) = (blk.re[base + p * nao + q], blk.im[base + p * nao + q]);
                    t.re[i * nao + q] += cr * lr - ci * li;
                    t.im[i * nao + q] += cr * li + ci * lr;
                }
            }
        }
        // z[i, j] = SUM_q t[i, q] · C_b[q, j]
        let zo = l * ni * nj;
        for i in 0..ni {
            for q in 0..nao {
                let (tr, ti) = (t.re[i * nao + q], t.im[i * nao + q]);
                if tr == 0.0 && ti == 0.0 {
                    continue;
                }
                for j in 0..nj {
                    let (cr, ci) = (b.c.re[q * nj + j], b.c.im[q * nj + j]);
                    out.re[zo + i * nj + j] += tr * cr - ti * ci;
                    out.im[zo + i * nj + j] += tr * ci + ti * cr;
                }
            }
        }
    }
    out
}

/// Pack a `(naux, n1*n2)` half-transformed block down to `(naux, n1*(n1+1)/2)`
/// — upstream's `mosym = 's2'` output of `_ao2mo.nr_e2`.
fn pack_mo_pairs(z: &CTensor, naux: usize, dims: PairDims) -> CTensor {
    if !dims.packed {
        return z.clone();
    }
    let (n, np) = (dims.n1, dims.len());
    let mut out = CTensor::zeros(naux * np);
    for l in 0..naux {
        for i in 0..n {
            for j in 0..=i {
                out.re[l * np + i * (i + 1) / 2 + j] = z.re[l * n * n + i * n + j];
                out.im[l * np + i * (i + 1) / 2 + j] = z.im[l * n * n + i * n + j];
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// warn_pbc2d_eri
// ---------------------------------------------------------------------------

/// `warn_pbc2d_eri(mydf)` — `df_ao2mo.py:307-315`.
///
/// A 2-D cell with `low_dim_ft_type = 'inf_vacuum'` has SINGULAR ERIs. Upstream
/// emits a `PBC2DIntegralsWarning`; this port emits a `tracing::warn!`, and
/// deliberately does not silently pass.
pub fn warn_pbc2d_eri(cell: &Cell) {
    if cell.dimension == 2
        && cell.low_dim_ft_type == pyscf_pbc_gto::LowDimFtType::InfVacuum
    {
        tracing::warn!(
            "ERIs of PBC-2D systems with infinite vacuum are singular; \
             cell.low_dim_ft_type = None should be set"
        );
    }
}

// ---------------------------------------------------------------------------
// get_eri
// ---------------------------------------------------------------------------

fn kvecs(df: &Gdf, kidx: [usize; 4]) -> Result<[[f64; 3]; 4], PbcDfError> {
    let n = df.kpts.len();
    let mut out = [[0.0; 3]; 4];
    for (o, &i) in out.iter_mut().zip(kidx.iter()) {
        if i >= n {
            return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "df_ao2mo: k-point index {i} is out of range for {n} k-points"
                )),
            )));
        }
        *o = df.kpts[i];
    }
    Ok(out)
}

/// The two-block branches contract `cderi[k0,k1]` against `cderi[k2,k3]`, and
/// those come from DIFFERENT k-difference groups — so on the eigen route they
/// can have been fitted in auxiliary spaces of different rank. The auxiliary
/// index is only comparable within a group, so summing over it across two is
/// meaningless; upstream has the same exposure and no check.
///
/// The mismatch is reachable in practice: MDF on He-fcc 2x2x2 at mesh 15 keeps
/// 10 vectors for one group and 11 for another. Catching it here, where the
/// caller knows which quadruple it asked for, beats producing a number.
fn check_ranks(bra: &[SrBlock], ket: &[SrBlock], kidx: [usize; 4]) -> Result<(), PbcDfError> {
    for (a, b) in bra.iter().zip(ket.iter()) {
        if a.naux != b.naux {
            return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "df_ao2mo: the k-quadruple {kidx:?} contracts cderi[{},{}] (rank {}) \
                     against cderi[{},{}] (rank {}). The auxiliary index is only \
                     comparable within one k-difference group, and the eigen-decomposed \
                     metric dropped different numbers of vectors for these two. Raise \
                     the builder's mesh, or lower linear_dep_threshold, so both groups \
                     keep the same rank",
                    kidx[0], kidx[1], a.naux, kidx[2], kidx[3], b.naux
                )),
            )));
        }
    }
    Ok(())
}

fn zero(k: &[f64; 3]) -> bool {
    k.iter().all(|v| v.abs() < 1e-9)
}
fn same(a: &[f64; 3], b: &[f64; 3]) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < 1e-9)
}

/// `df_ao2mo.get_eri(mydf, kpts, compact)` — `df_ao2mo.py:32-105`.
///
/// The four branches are upstream's, and they are not interchangeable:
/// * all-gamma is REAL with `s4` symmetry, so both axes may be `s2`-packed;
/// * `k0 == k2 && k1 == k3` reuses ONE `cderi` block through `zdotNN`;
/// * `k0 == k3 && k1 == k2` reuses one block through `zdot**N C**` and then
///   transposes the ket pair, because `(L|ij)^T conj = (L*|ji) = (L*|kl)`;
/// * otherwise two blocks are zipped.
///
/// # Errors
/// Propagates the lazy `cderi` build and [`crate::gdf::sr_loop`]'s missing-pair
/// error.
pub fn get_eri(df: &Gdf, kidx: [usize; 4], compact: bool) -> Result<Eri, PbcDfError> {
    let cell = &df.cell;
    let nao = cell.mol.nao_nr;
    let k = kvecs(df, kidx)?;

    if !is_kconserv(cell, &k) {
        tracing::warn!(
            "df_ao2mo: momentum conservation not found in the given k-points {kidx:?}"
        );
        return Ok(Eri::zeros(
            PairDims::plain(nao, nao),
            PairDims::plain(nao, nao),
        ));
    }

    let gamma = k.iter().all(zero);
    if gamma {
        // s2 columns on BOTH axes; the integrals are real.
        let dims = PairDims::packed(nao);
        let mut out = Eri::zeros(dims, dims);
        for blk in df.sr_loop(kidx[0], kidx[1], true)? {
            let a = scaled(
                &CTensor {
                    re: blk.re.clone(),
                    im: blk.im.clone(),
                },
                blk.sign,
            );
            let b = CTensor {
                re: blk.re.clone(),
                im: blk.im.clone(),
            };
            accum_nn(&mut out.data, &a, &b, blk.naux, blk.ncol, blk.ncol);
        }
        out.data.im.iter_mut().for_each(|v| *v = 0.0);
        return Ok(if compact { out } else { out.restore_s1() });
    }

    let dims = PairDims::plain(nao, nao);
    let mut out = Eri::zeros(dims, dims);
    let n2 = nao * nao;

    if same(&k[0], &k[2]) && same(&k[1], &k[3]) {
        for blk in df.sr_loop(kidx[0], kidx[1], false)? {
            let a = scaled(
                &CTensor {
                    re: blk.re.clone(),
                    im: blk.im.clone(),
                },
                blk.sign,
            );
            let b = CTensor {
                re: blk.re.clone(),
                im: blk.im.clone(),
            };
            accum_nn(&mut out.data, &a, &b, blk.naux, n2, n2);
        }
        return Ok(out);
    }

    if same(&k[0], &k[3]) && same(&k[1], &k[2]) {
        for blk in df.sr_loop(kidx[0], kidx[1], false)? {
            let a = scaled(
                &CTensor {
                    re: blk.re.clone(),
                    im: blk.im.clone(),
                },
                blk.sign,
            );
            let b = CTensor {
                re: blk.re.clone(),
                im: blk.im.clone(),
            };
            accum_nc(&mut out.data, &a, &b, blk.naux, n2, n2);
        }
        // `df_ao2mo.py:80-83`: j == k && i == l  =>  transpose the ket pair.
        let mut t = Eri::zeros(dims, dims);
        for row in 0..n2 {
            for r in 0..nao {
                for s in 0..nao {
                    t.data.re[row * n2 + r * nao + s] = out.data.re[row * n2 + s * nao + r];
                    t.data.im[row * n2 + r * nao + s] = out.data.im[row * n2 + s * nao + r];
                }
            }
        }
        return Ok(t);
    }

    let bra = df.sr_loop(kidx[0], kidx[1], false)?;
    let ket = df.sr_loop(kidx[2], kidx[3], false)?;
    check_ranks(&bra, &ket, kidx)?;
    for (ba, be) in bra.iter().zip(ket.iter()) {
        let a = scaled(
            &CTensor {
                re: ba.re.clone(),
                im: ba.im.clone(),
            },
            ba.sign,
        );
        let b = CTensor {
            re: be.re.clone(),
            im: be.im.clone(),
        };
        accum_nn(&mut out.data, &a, &b, ba.naux, n2, n2);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// general
// ---------------------------------------------------------------------------

/// `df_ao2mo.general(mydf, mo_coeffs, kpts, compact)` — `df_ao2mo.py:107-208`.
///
/// The four branches mirror [`get_eri`]'s. `compact` is honoured only on the
/// all-gamma, all-real branch, exactly as `_conc_mos` does (it forces
/// `compact = False` the moment any coefficient is complex).
///
/// # Errors
/// As [`get_eri`], plus a shape mismatch between a coefficient block and the
/// cell.
pub fn general(
    df: &Gdf,
    mos: [&MoCoeff; 4],
    kidx: [usize; 4],
    compact: bool,
) -> Result<Eri, PbcDfError> {
    let cell = &df.cell;
    warn_pbc2d_eri(cell);
    let nao = cell.mol.nao_nr;
    for m in mos {
        if m.nao != nao {
            return Err(PbcDfError::Core(pyscf_core::PyscfRsError::Core(
                pyscf_core::CoreError::InvalidMolecule(format!(
                    "df_ao2mo::general: an MO block has nao = {} where the cell has {nao}",
                    m.nao
                )),
            )));
        }
    }
    let k = kvecs(df, kidx)?;
    let nm = [mos[0].nmo, mos[1].nmo, mos[2].nmo, mos[3].nmo];

    if !is_kconserv(cell, &k) {
        tracing::warn!(
            "df_ao2mo: momentum conservation not found in the given k-points {kidx:?}"
        );
        return Ok(Eri::zeros(
            PairDims::plain(nm[0], nm[1]),
            PairDims::plain(nm[2], nm[3]),
        ));
    }

    let gamma = k.iter().all(zero);
    let all_real = mos.iter().all(|m| m.is_real());

    if gamma && all_real {
        let row = if compact && mos[0].iden_coeffs(mos[1]) {
            PairDims::packed(nm[0])
        } else {
            PairDims::plain(nm[0], nm[1])
        };
        let col = if compact && mos[2].iden_coeffs(mos[3]) {
            PairDims::packed(nm[2])
        } else {
            PairDims::plain(nm[2], nm[3])
        };
        let mut out = Eri::zeros(row, col);
        for blk in df.sr_loop(kidx[0], kidx[1], false)? {
            let zij = pack_mo_pairs(&r_e2(&blk, nao, mos[0], mos[1]), blk.naux, row);
            let zkl = pack_mo_pairs(&r_e2(&blk, nao, mos[2], mos[3]), blk.naux, col);
            let a = scaled(&zij, blk.sign);
            accum_nn(&mut out.data, &a, &zkl, blk.naux, row.len(), col.len());
        }
        out.data.im.iter_mut().for_each(|v| *v = 0.0);
        return Ok(out);
    }

    let row = PairDims::plain(nm[0], nm[1]);

    if same(&k[0], &k[2]) && same(&k[1], &k[3]) {
        let col = PairDims::plain(nm[2], nm[3]);
        let mut out = Eri::zeros(row, col);
        for blk in df.sr_loop(kidx[0], kidx[1], false)? {
            let zij = scaled(&r_e2(&blk, nao, mos[0], mos[1]), blk.sign);
            let zkl = r_e2(&blk, nao, mos[2], mos[3]);
            accum_nn(&mut out.data, &zij, &zkl, blk.naux, row.len(), col.len());
        }
        return Ok(out);
    }

    if same(&k[0], &k[3]) && same(&k[1], &k[2]) {
        // Upstream transforms with the coefficients in the order (3, 2) and
        // transposes the result — `df_ao2mo.py:162-181`.
        let lk = PairDims::plain(nm[3], nm[2]);
        let mut tmp = Eri::zeros(row, lk);
        for blk in df.sr_loop(kidx[0], kidx[1], false)? {
            let zij = scaled(&r_e2(&blk, nao, mos[0], mos[1]), blk.sign);
            let zlk = r_e2(&blk, nao, mos[3], mos[2]);
            accum_nc(&mut tmp.data, &zij, &zlk, blk.naux, row.len(), lk.len());
        }
        let col = PairDims::plain(nm[2], nm[3]);
        let mut out = Eri::zeros(row, col);
        for r in 0..row.len() {
            for kk in 0..nm[2] {
                for l in 0..nm[3] {
                    let s = r * lk.len() + l * nm[2] + kk;
                    let d = r * col.len() + kk * nm[3] + l;
                    out.data.re[d] = tmp.data.re[s];
                    out.data.im[d] = tmp.data.im[s];
                }
            }
        }
        return Ok(out);
    }

    let col = PairDims::plain(nm[2], nm[3]);
    let mut out = Eri::zeros(row, col);
    let bra = df.sr_loop(kidx[0], kidx[1], false)?;
    let ket = df.sr_loop(kidx[2], kidx[3], false)?;
    check_ranks(&bra, &ket, kidx)?;
    for (ba, be) in bra.iter().zip(ket.iter()) {
        let zij = scaled(&r_e2(ba, nao, mos[0], mos[1]), ba.sign);
        let zkl = r_e2(be, nao, mos[2], mos[3]);
        accum_nn(&mut out.data, &zij, &zkl, ba.naux, row.len(), col.len());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// ao2mo_7d
// ---------------------------------------------------------------------------

/// The per-k-point MO coefficients `ao2mo_7d` takes — four lists of `nkpts`
/// blocks, one list per ERI index.
pub type MoKpts<'a> = [&'a [MoCoeff]; 4];

/// `df_ao2mo.ao2mo_7d(mydf, mo_coeff_kpts, kpts, factor)` —
/// `df_ao2mo.py:210-275`.
///
/// **See the module docs for the index contract** — it is a Phase-15/16
/// interface, not an internal detail.
///
/// # Errors
/// Propagates the lazy `cderi` build and `sr_loop`; errors when a coefficient
/// list is not `nkpts` long or a block has the wrong `nao`.
pub fn ao2mo_7d(df: &Gdf, mos: MoKpts<'_>, factor: f64) -> Result<Eri7d, PbcDfError> {
    let cell = &df.cell;
    let nao = cell.mol.nao_nr;
    let nkpts = df.kpts.len();
    let bad = |m: String| {
        PbcDfError::Core(pyscf_core::PyscfRsError::Core(
            pyscf_core::CoreError::InvalidMolecule(m),
        ))
    };
    for (n, l) in mos.iter().enumerate() {
        if l.len() != nkpts {
            return Err(bad(format!(
                "ao2mo_7d: mo_coeff_kpts[{n}] has {} blocks for {nkpts} k-points",
                l.len()
            )));
        }
        if l.iter().any(|m| m.nao != nao) {
            return Err(bad(format!(
                "ao2mo_7d: mo_coeff_kpts[{n}] has a block whose nao is not {nao}"
            )));
        }
        if l.iter().any(|m| m.nmo != l[0].nmo) {
            return Err(bad(format!(
                "ao2mo_7d: mo_coeff_kpts[{n}] is ragged. Upstream requires the \
                 coefficients to be zero-padded to a common nmo across k-points \
                 (df_ao2mo.py:222-225 — pbc.mp.kmp2.padded_mo_coeff)"
            )));
        }
    }
    let nmo = [mos[0][0].nmo, mos[1][0].nmo, mos[2][0].nmo, mos[3][0].nmo];
    let block = nmo.iter().product::<usize>();
    let mut out = Eri7d {
        nkpts,
        nmo,
        data: CTensor::zeros(nkpts * nkpts * nkpts * block),
    };
    let gamma = df.kpts.iter().all(zero);
    let real_out = gamma && mos.iter().all(|l| l.iter().all(MoCoeff::is_real));

    let kconserv = pyscf_pbc_lib::kpts_helper::get_kconserv(&cell.a, &df.kpts);
    let (nrow, ncol) = (nmo[0] * nmo[1], nmo[2] * nmo[3]);

    for ki in 0..nkpts {
        for kj in 0..nkpts {
            let bra = df.sr_loop(ki, kj, false)?;
            let zij: Vec<CTensor> = bra
                .iter()
                .map(|b| r_e2(b, nao, &mos[0][ki], &mos[1][kj]))
                .collect();
            for kk in 0..nkpts {
                let kl = kconserv.get(ki, kj, kk) as usize;
                let ket = df.sr_loop(kk, kl, false)?;
                let mut acc = CTensor::zeros(nrow * ncol);
                for (i, be) in ket.iter().enumerate() {
                    let Some(zb) = zij.get(i) else { break };
                    let a = scaled(zb, bra[i].sign);
                    let zkl = r_e2(be, nao, &mos[2][kk], &mos[3][kl]);
                    accum_nn(&mut acc, &a, &zkl, be.naux, nrow, ncol);
                }
                let o = out.block_offset(ki, kj, kk);
                for p in 0..nrow * ncol {
                    out.data.re[o + p] = acc.re[p] * factor;
                    out.data.im[o + p] = if real_out { 0.0 } else { acc.im[p] * factor };
                }
            }
        }
    }
    Ok(out)
}

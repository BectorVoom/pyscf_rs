//! SCF-02 — a device-resident, k-major AO table for the SCF's grid loop.
//!
//! The SCF evaluates its AO table on the device (the fused resident AO
//! kernel) and used to read it back into a HOST cache; every cycle then
//! contracted it on the host. Measured 2026-09-24 on a Kaggle T4: moving
//! those contractions to the device while keeping the host cache made the XC
//! loop 2.6× SLOWER, because each call re-uploaded the ~600 MB deriv-1 table.
//! This type is the table kept where it was built: point-major accumulator
//! planes transposed once to k-major (`[k · n + e]`, `e = c · nao · ngrids +
//! mu · ngrids + g` — `eval_ao_kpts`'s layout), with the Γ-point imaginary
//! planes zeroed exactly as `eval_ao_kpts` zeroes them on the host
//! (`eval_gto.py:157-158`). [`crate::pbc::rho_k_table`] and
//! [`crate::pbc::band_vmat_table`] read it in place.
//!
//! The buffers are cubecl `Handle`s as PRIVATE fields (the `AoKAccumulator`
//! precedent), so the algebra wall (ALG-06) holds.

use cubecl::prelude::*;
use cubecl::server::Handle;
use pyscf_algebra::launch::{launch_1d, upload};
use pyscf_algebra::{AlgebraClient, AlgebraError, dispatch_backend};

use crate::pbc::AoKAccumulator;

/// A k-major AO table on the device. See the module docs.
pub struct DeviceAoTable {
    re: Handle,
    im: Handle,
    nkpts: usize,
    ncomp: usize,
    nao: usize,
    ngrids: usize,
}

impl core::fmt::Debug for DeviceAoTable {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceAoTable")
            .field("nkpts", &self.nkpts)
            .field("ncomp", &self.ncomp)
            .field("nao", &self.nao)
            .field("ngrids", &self.ngrids)
            .finish_non_exhaustive()
    }
}

impl DeviceAoTable {
    /// Take over an accumulator's planes (transposing a point-major one to
    /// k-major on the device) and zero the imaginary plane of every k-point
    /// flagged in `gamma`. `acc`'s `n` must be `ncomp · nao · ngrids`.
    ///
    /// # Panics
    /// Never; a shape disagreement is reported by the caller before this.
    pub fn from_accumulator(
        client: &AlgebraClient,
        acc: AoKAccumulator,
        ncomp: usize,
        nao: usize,
        ngrids: usize,
        gamma: &[bool],
    ) -> Self {
        let (nkpts, n) = acc.shape();
        let len = nkpts * n;
        let point_major = acc.is_point_major();
        let (src_re, src_im) = acc.planes();
        let (src_re, src_im) = (src_re.clone(), src_im.clone());
        drop(acc);
        let (re, im) = dispatch_backend!(client, c, Rt, {
            let (re, im) = if point_major && nkpts > 1 {
                let dre = c.empty(len.max(1) * core::mem::size_of::<f64>());
                let dim_ = c.empty(len.max(1) * core::mem::size_of::<f64>());
                let (count, dim) = launch_1d(c, len, 1);
                for (src, dst) in [(&src_re, &dre), (&src_im, &dim_)] {
                    unsafe {
                        crate::pbc::band_vmat::point_to_k_major_kernel::launch_unchecked::<f64, Rt>(
                            c,
                            count.clone(),
                            dim,
                            // SAFETY: both buffers hold `len = nkpts · n`
                            // values; the kernel guards `j < len`.
                            ArrayArg::from_raw_parts(src.clone(), len),
                            ArrayArg::from_raw_parts(dst.clone(), len),
                            nkpts,
                            n,
                        );
                    }
                }
                (dre, dim_)
            } else {
                // k-major already (or a single k, where both layouts agree).
                (src_re, src_im)
            };
            for (k, &g) in gamma.iter().enumerate().take(nkpts) {
                if g {
                    crate::pbc::fill::fill_range::<Rt, f64>(c, &im, len, k * n, 1, n);
                }
            }
            (re, im)
        });
        Self {
            re,
            im,
            nkpts,
            ncomp,
            nao,
            ngrids,
        }
    }

    /// A table uploaded from HOST k-major planes (`[k · n + e]`, `n = ncomp ·
    /// nao · ngrids`) as given — the caller has already zeroed whatever Γ
    /// imaginary planes it wants zero. For the kernel gates and benches; the
    /// SCF builds its tables with [`DeviceAoTable::from_accumulator`].
    ///
    /// # Errors
    /// [`AlgebraError::ShapeMismatch`] when a plane is not `nkpts · ncomp ·
    /// nao · ngrids` long.
    pub fn from_host_planes(
        client: &AlgebraClient,
        re: &[f64],
        im: &[f64],
        nkpts: usize,
        ncomp: usize,
        nao: usize,
        ngrids: usize,
    ) -> Result<Self, AlgebraError> {
        let want = nkpts * ncomp * nao * ngrids;
        if re.len() != want || im.len() != want {
            return Err(AlgebraError::ShapeMismatch {
                expected: format!("both AO planes of length nkpts·ncomp·nao·ngrids = {want}"),
                actual: format!("re {} im {}", re.len(), im.len()),
            });
        }
        let (re, im) = dispatch_backend!(client, c, Rt, {
            (upload::<Rt, f64>(c, re), upload::<Rt, f64>(c, im))
        });
        Ok(Self {
            re,
            im,
            nkpts,
            ncomp,
            nao,
            ngrids,
        })
    }

    /// An all-zero table (every image screened out): the contraction of an
    /// all-zero table is all zeros, as the host route's empty planes give.
    pub fn zeros(
        client: &AlgebraClient,
        nkpts: usize,
        ncomp: usize,
        nao: usize,
        ngrids: usize,
    ) -> Self {
        let acc = AoKAccumulator::zeros(client, nkpts, ncomp * nao * ngrids);
        let (re, im) = acc.planes();
        Self {
            re: re.clone(),
            im: im.clone(),
            nkpts,
            ncomp,
            nao,
            ngrids,
        }
    }

    /// `(nkpts, ncomp, nao, ngrids)`.
    pub fn dims(&self) -> (usize, usize, usize, usize) {
        (self.nkpts, self.ncomp, self.nao, self.ngrids)
    }

    /// Device bytes held (both planes).
    pub fn bytes(&self) -> usize {
        2 * 8 * self.nkpts * self.ncomp * self.nao * self.ngrids
    }

    pub(crate) fn planes(&self) -> (&Handle, &Handle) {
        (&self.re, &self.im)
    }
}

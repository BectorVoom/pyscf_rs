//! `_ERIS` — the seven k-point MO integral blocks `KRCCSD` contracts against
//! (plan 16-05 Task 1; `pyscf/pbc/cc/kccsd_rhf.py:715-950`).
//!
//! # The three things this build is, beyond "transform the integrals"
//!
//! **1. The Fock matrix is REBUILT with `exxdiv` suppressed, and the Madelung
//! correction is re-added afterwards** (`16-CONTEXT §3.5`). `kccsd_rhf.py:743`
//! wraps `get_veff` in `lib.temporary_env(cc._scf, exxdiv=exxdiv)` with
//! `exxdiv = None` unless `keep_exxdiv`, and `:755-762` then puts the
//! correction back into `mo_energy` through `_adjust_occ(mo_e, nocc,
//! -madelung)`. Upstream's own comment says that without the re-add "MP2 energy
//! may be largely off the correct value", especially where occupied and virtual
//! energies overlap — i.e. `graphene`, `dimension = 2`. **Both halves ship
//! together or the energy is quietly wrong.**
//!
//! **2. The integral loop runs over `symm_map` ORBIT REPRESENTATIVES**
//! (`:783`, `:798-805`), filling the rest of each orbit by `transform_symm`.
//! 16-01 Task 6 measured the saving at **2.10× wall clock** on diamond
//! `gth-szv` 2×2×2 — 176 representatives for 512 triples — against
//! `16-REVIEW.md §3`'s derived `~4×`; the gap is the fixed-point collapse plus
//! `vvvv`, which is built by `ao2mo_7d` in BOTH paths (the `self.vvvv[...]`
//! line inside upstream's symmetry loop is commented out at `:797`) and
//! therefore saves nothing. The saving is still the phase's largest speed item
//! and it is here from the first version, per D-PBC-29 clause 3.
//!
//! **3. Each block picks its storage tier from an EXACT byte count.** Upstream
//! chooses by `_mem_usage` (`:1100-1107`), which 16-01 measured over-estimating
//! by **9.143×** on `gth-szv` and **6.058×** on `gth-dzvp`; porting it would
//! make this port's HARD `MemoryLimitExceeded` refuse jobs that fit
//! (D-PBC-29 clause 4). [`KTensor`] sizes itself.
//!
//! # The index convention, stated once
//!
//! `kccsd_rhf.py:806-812` stores the transformed block at `[kp, kr, kq]` after
//! `transpose(0, 2, 1, 3)` — i.e. the CHEMIST's `(pq|rs)` from `ao2mo` becomes
//! the PHYSICIST-ordered `<pr|qs>` the intermediates read, and the k-index
//! order is likewise `(kp, kr, kq)`. This port keeps both, because a port that
//! silently normalises one of them is the 14-05 `decompose_j2c` defect over
//! again (`16-CONTEXT §3.4`, +6 306 866.73 Ha).

use std::sync::Arc;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::{JkOpts, MoCoeff, PeriodicDf};
use pyscf_pbc_gto::{Cell, ExxDiv};
use pyscf_pbc_lib::KptsHelper;
use pyscf_pbc_mp::PaddedMos;
use pyscf_runtime::ZWorkspacePool;

use crate::error::PbcCcError;
use crate::ktensor::{KRank, KTensor, Tier};
use crate::zarr::ZArr;

/// Which of the seven blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blk {
    Oooo,
    Ooov,
    Oovv,
    Ovov,
    Voov,
    Vovv,
    Vvvv,
}

impl Blk {
    /// The four orbital-space letters, `false` = occupied, `true` = virtual.
    fn spaces(self) -> [bool; 4] {
        match self {
            Blk::Oooo => [false, false, false, false],
            Blk::Ooov => [false, false, false, true],
            Blk::Oovv => [false, false, true, true],
            Blk::Ovov => [false, true, false, true],
            Blk::Voov => [true, false, false, true],
            Blk::Vovv => [true, false, true, true],
            Blk::Vvvv => [true, true, true, true],
        }
    }

    /// Per-block dimensions at `(nocc, nvir)`.
    pub fn dims(self, nocc: usize, nvir: usize) -> [usize; 4] {
        self.spaces().map(|v| if v { nvir } else { nocc })
    }

    /// A short name for diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Blk::Oooo => "oooo",
            Blk::Ooov => "ooov",
            Blk::Oovv => "oovv",
            Blk::Ovov => "ovov",
            Blk::Voov => "voov",
            Blk::Vovv => "vovv",
            Blk::Vvvv => "vvvv",
        }
    }

    /// The seven, in upstream's declaration order (`kccsd_rhf.py:789-795`).
    pub const ALL: [Blk; 7] = [
        Blk::Oooo,
        Blk::Ooov,
        Blk::Oovv,
        Blk::Ovov,
        Blk::Voov,
        Blk::Vovv,
        Blk::Vvvv,
    ];
}

/// `_adjust_occ(mo_energy, nocc, shift)` — `pbc/cc/ccsd.py:146-150`.
///
/// Shifts the OCCUPIED orbital energies only. This is the second half of the
/// `exxdiv` treatment and it is not optional bookkeeping: without it the
/// correlation energy is "largely off" (upstream's own words, `ccsd.py:53-56`).
pub fn adjust_occ(mo_energy: &[f64], nocc: usize, shift: f64) -> Vec<f64> {
    let mut out = mo_energy.to_vec();
    for e in out.iter_mut().take(nocc) {
        *e += shift;
    }
    out
}

/// How the ERI build was asked to store itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErisMethod {
    /// `method='incore'` — resident if it fits the budget.
    Incore,
    /// `method='outcore'` — force the HDF5 tier, as
    /// `test_krccsd.py:250` does to compare the two.
    Outcore,
}

/// The seven blocks, the Fock matrix and the orbital energies `KRCCSD` runs on.
#[derive(Debug)]
pub struct KEris {
    pub nkpts: usize,
    pub nocc: usize,
    pub nmo: usize,
    pub nvir: usize,
    /// `[nkpts, nmo, nmo]` — the MO-basis Fock matrix, built under the
    /// `exxdiv` of [`KErisOpts::keep_exxdiv`].
    pub fock: ZArr,
    /// `mo_energy[k]`, `nmo` long, WITH the Madelung re-add when
    /// `keep_exxdiv` is false.
    pub mo_energy: Vec<Vec<f64>>,
    /// The padded MO coefficients the transform used.
    pub mo_coeff: Vec<MoCoeff>,
    /// The Madelung constant that was re-added (0.0 when `keep_exxdiv`).
    pub madelung: f64,
    pool: Arc<ZWorkspacePool>,
    blocks: Vec<(Blk, KTensor)>,
}

/// Knobs of the ERI build.
#[derive(Debug, Clone, Copy)]
pub struct KErisOpts {
    /// `cc.keep_exxdiv` (`kccsd_rhf.py:743`). `false` — upstream's default —
    /// builds the Fock matrix with `exxdiv` suppressed and re-adds the
    /// Madelung correction to the occupied orbital energies.
    pub keep_exxdiv: bool,
    /// The mean field's `exxdiv`, used only when `keep_exxdiv` is true.
    pub exxdiv: Option<ExxDiv>,
    /// `cc.max_memory`, in MEGABYTES, upstream's unit.
    pub max_memory: f64,
    /// Incore or forced-outcore.
    pub method: ErisMethod,
    /// Drive the integral loop through `symm_map`'s orbit representatives
    /// (`kccsd_rhf.py:798-805`) rather than transforming all `nkpts³` triples.
    ///
    /// **`true` is the only shipped path**; `false` exists so a test can
    /// compare the two. 16-01 Task 6 measured the saving at **2.10× wall
    /// clock** and — the finding that corrects `16-05-PLAN.md` test 5 —
    /// measured upstream's own two paths differing by up to **`1.32e-7`**, so
    /// the comparison is a `1e-6` gate, NOT the bit-identity the plan asked
    /// for. A symmetry-related k-quadruple's FFT transform and its transposed
    /// sibling are not the same floating-point computation.
    pub use_symm_map: bool,
}

impl Default for KErisOpts {
    fn default() -> Self {
        Self {
            keep_exxdiv: false,
            exxdiv: Some(ExxDiv::Ewald),
            max_memory: 4000.0,
            method: ErisMethod::Incore,
            use_symm_map: true,
        }
    }
}

impl KEris {
    /// Build the seven blocks.
    ///
    /// `padded` comes from Phase 15's `add_padding` — this plan does NOT write
    /// a second padding implementation (`16-CONTEXT §1.1`; the convention is
    /// occupied-bottom / virtual-**TOP** aligned, `kmp2.py:262-263`).
    ///
    /// # Errors
    /// Propagates the density-fitting builder, the complex arena's HARD
    /// refusal, and any shape violation.
    pub fn new(
        cell: &Cell,
        with_df: &dyn PeriodicDf,
        khelper: &mut KptsHelper,
        padded: &PaddedMos,
        dm: &[CTensor],
        opts: KErisOpts,
    ) -> Result<Self, PbcCcError> {
        let (fock, mo_energy, madelung) = Self::build_fock(cell, with_df, padded, dm, opts)?;
        Self::from_parts(with_df, khelper, padded, fock, mo_energy, madelung, opts)
    }

    /// The Fock half of [`KEris::new`], separated so a caller can substitute a
    /// mean field built elsewhere.
    ///
    /// Returns `(fock, mo_energy, madelung)`, with the Madelung correction
    /// ALREADY applied to `mo_energy` unless `keep_exxdiv` (`§3.5`, both
    /// halves together or neither).
    ///
    /// # Errors
    /// Propagates the density-fitting builder and the Madelung constant.
    pub fn build_fock(
        cell: &Cell,
        with_df: &dyn PeriodicDf,
        padded: &PaddedMos,
        dm: &[CTensor],
        opts: KErisOpts,
    ) -> Result<(ZArr, Vec<Vec<f64>>, f64), PbcCcError> {
        let kpts = with_df.kpts();
        let nkpts = kpts.len();
        let nocc = padded.nocc;
        let nmo = padded.nmo;
        let nao = cell.mol.nao_nr;

        // ---- Fock, under the exxdiv this build is entitled to (§3.5, half 1)
        let exxdiv = if opts.keep_exxdiv { opts.exxdiv } else { None };
        let hcore = pyscf_pbc_df::get_hcore(with_df, kpts).map_err(df_err)?;
        let dms: Vec<Vec<CTensor>> = vec![dm.to_vec()];
        let jk = with_df
            .get_jk(
                &dms,
                kpts,
                JkOpts {
                    hermi: 1,
                    kpts_band: None,
                    with_j: true,
                    with_k: true,
                    exxdiv,
                    omega: None,
                    kk_symmetry: false,
                },
            )
            .map_err(df_err)?;
        let vj = jk.vj.ok_or_else(|| shape("get_jk returned no vj"))?;
        let vk = jk.vk.ok_or_else(|| shape("get_jk returned no vk"))?;

        let mut fock = ZArr::zeros(&[nkpts, nmo, nmo]);
        let mut mo_energy: Vec<Vec<f64>> = Vec::with_capacity(nkpts);
        for k in 0..nkpts {
            // fockao = hcore + vj - vk/2   (khf.py:624-633's RHF get_veff)
            let mut fao = CTensor::zeros(nao * nao);
            for i in 0..nao * nao {
                fao.re[i] = hcore[k].re[i] + vj[0][k].re[i] - 0.5 * vk[0][k].re[i];
                fao.im[i] = hcore[k].im[i] + vj[0][k].im[i] - 0.5 * vk[0][k].im[i];
            }
            // f_MO = Cᴴ F_AO C. **The conjugation on C is explicit here**, which
            // is where `16-CONTEXT §3.2` wants it — see `zarr`'s module doc.
            let c = &padded.mo_coeff[k];
            let mut blk = ZArr::zeros(&[nmo, nmo]);
            for p in 0..nmo {
                for q in 0..nmo {
                    let (mut re, mut im) = (0.0_f64, 0.0_f64);
                    for a in 0..nao {
                        // conj(C[a,p])
                        let (cr, ci) = (c.c.re[a * nmo + p], -c.c.im[a * nmo + p]);
                        for b in 0..nao {
                            let (fr, fi) = (fao.re[a * nao + b], fao.im[a * nao + b]);
                            let (dr, di) = (c.c.re[b * nmo + q], c.c.im[b * nmo + q]);
                            let (tr, ti) = (fr * dr - fi * di, fr * di + fi * dr);
                            re += cr * tr - ci * ti;
                            im += cr * ti + ci * tr;
                        }
                    }
                    blk.data_mut().re[p * nmo + q] = re;
                    blk.data_mut().im[p * nmo + q] = im;
                }
            }
            mo_energy.push((0..nmo).map(|p| blk.at(&[p, p]).map(|v| v.0)).collect::<Result<_, _>>()?);
            fock.set_leading(&[k], &blk)?;
        }

        // ---- the Madelung re-add (§3.5, half 2)
        let madelung = if opts.keep_exxdiv {
            0.0
        } else {
            pyscf_pbc_gto::madelung(cell, kpts, None)
                .map_err(|e| PbcCcError::Shape(format!("madelung: {e}")))?
        };
        if !opts.keep_exxdiv {
            mo_energy = mo_energy
                .iter()
                .map(|e| adjust_occ(e, nocc, -madelung))
                .collect();
        }

        Ok((fock, mo_energy, madelung))
    }

    /// Build the seven blocks from an ALREADY-BUILT mean field.
    ///
    /// This is the entry point the Phase-16 oracle tests use: 16-01 found this
    /// port's `KRHF` and upstream's differ by `1.35e-5 Ha` on diamond
    /// `gth-szv` `[1,1,2]` at the PINNED `[15,15,15]` mesh (they agree to
    /// `5e-11` at the default mesh — Phase 15's `oracle_phase15` measured that),
    /// and a correlation energy compared across two different mean fields
    /// measures the mean fields, not the correlation. Feeding upstream's own
    /// `fock` / `mo_energy` / `mo_coeff` in here makes the CC comparison
    /// mean-field-INDEPENDENT — the same discipline `15-VERIFICATION` used when
    /// it drove `Lov` from "upstream's own padded MOs".
    ///
    /// # Errors
    /// Propagates the density-fitting builder, the complex arena's HARD
    /// refusal, and any shape violation.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        with_df: &dyn PeriodicDf,
        khelper: &mut KptsHelper,
        padded: &PaddedMos,
        fock: ZArr,
        mo_energy: Vec<Vec<f64>>,
        madelung: f64,
        opts: KErisOpts,
    ) -> Result<Self, PbcCcError> {
        let kpts = with_df.kpts();
        let nkpts = kpts.len();
        let nocc = padded.nocc;
        let nmo = padded.nmo;
        let nvir = nmo - nocc;

        // ---- the seven blocks
        let pool = Arc::new(ZWorkspacePool::new(
            (opts.max_memory * 1e6).max(0.0) as usize,
        ));
        khelper.build_symm_map(None); // `:783` — LAZY, built here, not in `new`
        let symm: Vec<([usize; 3], Vec<[usize; 3]>)> = if opts.use_symm_map {
            khelper
                .symm_map
                .as_ref()
                .ok_or_else(|| shape("build_symm_map produced no map"))?
                .entries()
                .to_vec()
        } else {
            // Every triple its own representative with a singleton orbit — the
            // all-triples build, for the equivalence test only.
            (0..nkpts)
                .flat_map(|p| {
                    (0..nkpts).flat_map(move |q| {
                        (0..nkpts).map(move |r| ([p, q, r], vec![[p, q, r]]))
                    })
                })
                .collect()
        };

        let allow_spill = matches!(opts.method, ErisMethod::Outcore);
        let mut blocks: Vec<(Blk, KTensor)> = Vec::with_capacity(7);
        for b in Blk::ALL {
            let dims = b.dims(nocc, nvir);
            let t = KTensor::zeros(&pool, nkpts, KRank::Three, &dims, allow_spill)?;
            blocks.push((b, t));
        }

        // `orbv` — the virtual block, for the `ao2mo_7d` `vvvv` route.
        let orbv: Vec<MoCoeff> = padded
            .mo_coeff
            .iter()
            .map(|m| {
                let mut c = CTensor::zeros(m.nao * nvir);
                for a in 0..m.nao {
                    for v in 0..nvir {
                        c.re[a * nvir + v] = m.c.re[a * nmo + nocc + v];
                        c.im[a * nvir + v] = m.c.im[a * nmo + nocc + v];
                    }
                }
                MoCoeff::new(m.nao, nvir, c)
            })
            .collect();

        // `vvvv` first — `kccsd_rhf.py:798`:
        //     self.vvvv = with_df.ao2mo_7d(orbv, factor=1/nkpts).transpose(0,2,1,3,5,4,6)
        // The transpose swaps k-axes 1 and 2 AND orbital axes 1 and 2, which is
        // the SAME `(kp,kr,kq)` / `<pr|qs>` reordering the symmetry loop applies
        // to the other six blocks.
        {
            let e7 = with_df
                .ao2mo_7d([&orbv, &orbv, &orbv, &orbv], 1.0 / nkpts as f64)
                .map_err(df_err)?;
            let (_, vvvv) = blocks
                .iter()
                .find(|(b, _)| *b == Blk::Vvvv)
                .ok_or_else(|| shape("vvvv block missing"))?;
            for ki in 0..nkpts {
                for kj in 0..nkpts {
                    for kk in 0..nkpts {
                        let off = e7.block_offset(ki, kj, kk);
                        let len = e7.block_len();
                        let raw = ZArr::from_ctensor(
                            &[nvir, nvir, nvir, nvir],
                            CTensor {
                                re: e7.data.re[off..off + len].to_vec(),
                                im: e7.data.im[off..off + len].to_vec(),
                            },
                        )?;
                        // orbital transpose(0,2,1,3) applied per block; the
                        // k-axis swap (0,2,1,...) is the (ki,kk,kj) target.
                        let v = raw.transpose(&[0, 2, 1, 3])?;
                        vvvv.set_block(&pool, &[ki, kk, kj], v.data())?;
                    }
                }
            }
        }

        // The other six, through the symmetry loop.
        let mo_refs: Vec<&MoCoeff> = padded.mo_coeff.iter().collect();
        for (rep, orbit) in &symm {
            let [ikp, ikq, ikr] = *rep;
            let iks = khelper.kconserv.get(ikp, ikq, ikr) as usize;
            let eri = with_df
                .ao2mo(
                    [
                        mo_refs[ikp],
                        mo_refs[ikq],
                        mo_refs[ikr],
                        mo_refs[iks],
                    ],
                    [ikp, ikq, ikr, iks],
                    false,
                )
                .map_err(df_err)?;
            let eri = eri.restore_s1();
            let eri_kpt = ZArr::from_ctensor(&[nmo, nmo, nmo, nmo], eri.data)?;

            for &[kp, kq, kr] in orbit {
                // In the all-triples build every triple IS its own
                // representative, so the operation is the identity — reading
                // `khelper._operation` there would apply the map's operation to
                // a block that was transformed directly, which is silently
                // wrong rather than a shape error.
                let raw = if opts.use_symm_map {
                    ZArr::from_ctensor(
                        &[nmo; 4],
                        khelper
                            .transform_symm(eri_kpt.data(), [nmo; 4], kp, kq, kr)
                            .map_err(|e| PbcCcError::Shape(e.to_string()))?,
                    )?
                } else {
                    eri_kpt.clone()
                };
                let symm_block = raw.transpose(&[0, 2, 1, 3])?;
                for (b, tensor) in blocks.iter() {
                    if *b == Blk::Vvvv {
                        continue; // built by ao2mo_7d above, as upstream does
                    }
                    let sp = b.spaces();
                    let d = b.dims(nocc, nvir);
                    let mut out = ZArr::zeros(&d);
                    let inv = 1.0 / nkpts as f64;
                    for i0 in 0..d[0] {
                        for i1 in 0..d[1] {
                            for i2 in 0..d[2] {
                                for i3 in 0..d[3] {
                                    let s = [
                                        if sp[0] { nocc + i0 } else { i0 },
                                        if sp[1] { nocc + i1 } else { i1 },
                                        if sp[2] { nocc + i2 } else { i2 },
                                        if sp[3] { nocc + i3 } else { i3 },
                                    ];
                                    let (re, im) = symm_block.at(&s)?;
                                    let f = ((i0 * d[1] + i1) * d[2] + i2) * d[3] + i3;
                                    out.data_mut().re[f] = re * inv;
                                    out.data_mut().im[f] = im * inv;
                                }
                            }
                        }
                    }
                    tensor.set_block(&pool, &[kp, kr, kq], out.data())?;
                }
            }
        }

        Ok(Self {
            nkpts,
            nocc,
            nmo,
            nvir,
            fock,
            mo_energy,
            mo_coeff: padded.mo_coeff.clone(),
            madelung,
            pool,
            blocks,
        })
    }

    /// One block at a k-triple, `[kp, kr, kq]`-indexed as upstream stores it.
    ///
    /// # Errors
    /// [`PbcCcError`] for an unknown block or a bad k-address.
    pub fn blk(&self, b: Blk, k0: usize, k1: usize, k2: usize) -> Result<ZArr, PbcCcError> {
        let t = self.tensor(b)?;
        let c = t.block(&self.pool, &[k0, k1, k2])?;
        Ok(ZArr::from_ctensor(&b.dims(self.nocc, self.nvir), c)?)
    }

    /// The blocks over a FREE leading k-index — upstream's `eris.oovv[:,kk,kc]`.
    ///
    /// # Errors
    /// As [`KEris::blk`].
    pub fn blk_free0(&self, b: Blk, k1: usize, k2: usize) -> Result<ZArr, PbcCcError> {
        let d = b.dims(self.nocc, self.nvir);
        let mut shape = vec![self.nkpts];
        shape.extend_from_slice(&d);
        let mut out = ZArr::zeros(&shape);
        for k in 0..self.nkpts {
            out.set_leading(&[k], &self.blk(b, k, k1, k2)?)?;
        }
        Ok(out)
    }

    /// The blocks over a FREE THIRD k-index — upstream's `eris.oovv[kk,kl]`.
    ///
    /// # Errors
    /// As [`KEris::blk`].
    pub fn blk_free2(&self, b: Blk, k0: usize, k1: usize) -> Result<ZArr, PbcCcError> {
        let d = b.dims(self.nocc, self.nvir);
        let mut shape = vec![self.nkpts];
        shape.extend_from_slice(&d);
        let mut out = ZArr::zeros(&shape);
        for k in 0..self.nkpts {
            out.set_leading(&[k], &self.blk(b, k0, k1, k)?)?;
        }
        Ok(out)
    }

    /// The storage tier a block landed in — 16-05 test 4 asserts this, so a
    /// fixture that silently stayed incore fails rather than passes.
    ///
    /// # Errors
    /// [`PbcCcError`] for an unknown block.
    pub fn tier(&self, b: Blk) -> Result<Tier, PbcCcError> {
        Ok(self.tensor(b)?.tier())
    }

    /// The exact bytes one block occupies.
    ///
    /// # Errors
    /// [`PbcCcError`] for an unknown block.
    pub fn bytes(&self, b: Blk) -> Result<usize, PbcCcError> {
        Ok(self.tensor(b)?.bytes())
    }

    /// Total exact bytes of the seven blocks.
    pub fn total_bytes(&self) -> usize {
        self.blocks.iter().map(|(_, t)| t.bytes()).sum()
    }

    /// Peak in-memory bytes charged by the arena — the quantity the
    /// peak-memory assertions read.
    pub fn live_inmem_bytes(&self) -> usize {
        self.pool.live_inmem_bytes()
    }

    /// `fock[k]` as an `nmo × nmo` block.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] on a bad k index.
    pub fn fock_at(&self, k: usize) -> Result<ZArr, PbcCcError> {
        self.fock.slice_leading(&[k])
    }

    /// `fock[k][:nocc, nocc:]` — `fov`.
    ///
    /// # Errors
    /// As [`KEris::fock_at`].
    pub fn fov(&self, k: usize) -> Result<ZArr, PbcCcError> {
        self.fock_sub(k, 0, self.nocc, self.nocc, self.nmo)
    }

    /// `fock[k][:nocc, :nocc]`.
    ///
    /// # Errors
    /// As [`KEris::fock_at`].
    pub fn foo(&self, k: usize) -> Result<ZArr, PbcCcError> {
        self.fock_sub(k, 0, self.nocc, 0, self.nocc)
    }

    /// `fock[k][nocc:, nocc:]`.
    ///
    /// # Errors
    /// As [`KEris::fock_at`].
    pub fn fvv(&self, k: usize) -> Result<ZArr, PbcCcError> {
        self.fock_sub(k, self.nocc, self.nmo, self.nocc, self.nmo)
    }

    fn fock_sub(
        &self,
        k: usize,
        r0: usize,
        r1: usize,
        c0: usize,
        c1: usize,
    ) -> Result<ZArr, PbcCcError> {
        let f = self.fock_at(k)?;
        let mut out = ZArr::zeros(&[r1 - r0, c1 - c0]);
        for r in r0..r1 {
            for c in c0..c1 {
                let (re, im) = f.at(&[r, c])?;
                let i = (r - r0) * (c1 - c0) + (c - c0);
                out.data_mut().re[i] = re;
                out.data_mut().im[i] = im;
            }
        }
        Ok(out)
    }

    fn tensor(&self, b: Blk) -> Result<&KTensor, PbcCcError> {
        self.blocks
            .iter()
            .find(|(x, _)| *x == b)
            .map(|(_, t)| t)
            .ok_or_else(|| PbcCcError::Shape(format!("no {} block", b.name())))
    }
}

fn shape(msg: &str) -> PbcCcError {
    PbcCcError::Shape(msg.to_string())
}

fn df_err(e: pyscf_pbc_df::PbcDfError) -> PbcCcError {
    PbcCcError::Shape(format!("density fitting: {e}"))
}

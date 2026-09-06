//! `_ChemistsERIs` for `KUCCSD` — the twenty-three k-point MO integral blocks
//! plus `vvvv`/`VVVV`/`vvVV` (plan 16-06 Task 1;
//! `pyscf/pbc/cc/kccsd_uhf.py:779-1015`).
//!
//! # The index convention is NOT [`crate::keris::KEris`]'s, and upstream says so
//!
//! `kccsd_uhf.py:770-777` carries the comment
//!
//! > Note the two electron integrals are stored in different orders from
//! > `kccsd_rhf._ERIS`.  Integrals `(ab|cd)` are stored as `[ka,kb,kc,a,b,c,d]`
//! > here while the order is `[ka,kc,kb,a,c,b,d]` in `kccsd_rhf._ERIS`.
//! >
//! > TODO: use the same convention as `kccsd_rhf`
//!
//! **The TODO is upstream's, and this port does not do it for them.** The
//! unrestricted intermediates (`kintermediates_uhf.py`) are written against
//! THIS order; normalising it here would mean re-deriving every one of them,
//! which is exactly the class of silent re-derivation `16-CONTEXT §3.4`
//! forbids. So: no `transpose(0,2,1,3)` anywhere in this file, and the k-triple
//! that addresses a block is the plain `(kp, kq, kr)` of the `(pq|rs)` chemists'
//! integral — with the ONE exception upstream itself writes, `voov`/`vovv`
//! (and their spin siblings), which are stored `.conj().transpose(1,0,3,2)` at
//! `[kq, kp, ks]` (`:894-895` and the three parallel loops).
//!
//! # Four `ao2mo` passes, six blocks each
//!
//! `_kuccsd_eris_common_` (`:838-949`) transforms `(o p | p p)` four times —
//! `aaaa`, `bbbb`, `aabb`, `bbaa` — and slices six blocks out of each. The
//! twenty-fourth, `OOoo`, upstream sets to `None` (`:830`, `:939` commented
//! out) and never reads; this port does not build it either, because an
//! `nkpts³ · noccb² nocca²` tensor nothing contracts against is pure cost.
//!
//! `vvvv` / `VVVV` / `vvVV` come from `ao2mo_7d` (`:832-836`), not from the
//! `oppp` passes, exactly as in `KEris`.
//!
//! # The Fock matrix and `exxdiv`
//!
//! Identical discipline to [`crate::keris`] (`16-CONTEXT §3.5`): `:858-860`
//! rebuilds `vhf` inside `lib.temporary_env(cc._scf, exxdiv=None)` and
//! `:868-873` re-adds Madelung through `_adjust_occ` per spin. Both halves or
//! neither.
//!
//! # What is DEFERRED, explicitly
//!
//! `_make_df_eris` (`:1017`) builds `Lpv`/`LPV` three-index tensors so that
//! `add_vvvv_` can form `Wvvvv` on the fly from GDF (`kccsd_uhf.py:562-590`).
//! It is not ported here. The `incore`/`outcore` route this module ships
//! produces the SAME `Wvvvv` through `kintermediates_uhf::cc_wvvvv_half`, so
//! nothing is missing from the energy — only the memory saving is. The refusal
//! upstream itself raises on that route, `cell.dimension == 2` (`:1022`), is
//! reproduced in [`KuEris::check_dimension_for_direct_df`] so that the
//! condition cannot be lost.

use std::sync::Arc;

use pyscf_algebra::CTensor;
use pyscf_pbc_df::{JkOpts, MoCoeff, PeriodicDf};
use pyscf_pbc_gto::Cell;
use pyscf_pbc_lib::Kconserv;
use pyscf_pbc_mp::PaddedMos;
use pyscf_runtime::ZWorkspacePool;

use crate::error::PbcCcError;
use crate::keris::{ErisMethod, KErisOpts, adjust_occ};
use crate::ktensor::{KRank, KTensor, Tier};
use crate::zarr::ZArr;

/// Which of the four `(o p | p p)` transforms a block came out of — equivalently,
/// the spin of its first index pair and of its second.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UPass {
    /// `(o_a p_a | p_a p_a)` — `oooo`, `ooov`, `oovv`, `ovov`, `voov`, `vovv`.
    Aaaa,
    /// `(o_b p_b | p_b p_b)` — `OOOO` … `VOVV`.
    Bbbb,
    /// `(o_a p_a | p_b p_b)` — `ooOO` … `voVV`.
    AaBb,
    /// `(o_b p_b | p_a p_a)` — `OOov` … `VOvv` (and `OOoo`, which is not built).
    BbAa,
}

impl UPass {
    /// `(first-pair is beta, second-pair is beta)`.
    pub fn spins(self) -> (bool, bool) {
        match self {
            UPass::Aaaa => (false, false),
            UPass::Bbbb => (true, true),
            UPass::AaBb => (false, true),
            UPass::BbAa => (true, false),
        }
    }

    /// The four `oppp` passes, in upstream's order (`:884`, `:900`, `:916`, `:932`).
    pub const OPPP: [UPass; 4] = [UPass::Aaaa, UPass::Bbbb, UPass::AaBb, UPass::BbAa];
    /// The three `ao2mo_7d` passes (`:834-836`). `BbAa` has no `VVvv`
    /// (`:1013` sets it `None`).
    pub const QUAD: [UPass; 3] = [UPass::Aaaa, UPass::Bbbb, UPass::AaBb];
}

/// Which slice of the `oppp` transform.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UKind {
    /// `tmp[:o, :o, :o, :o]` at `[kp,kq,kr]`.
    Oooo,
    /// `tmp[:o, :o, :o, o:]` at `[kp,kq,kr]`.
    Ooov,
    /// `tmp[:o, :o, o:, o:]` at `[kp,kq,kr]`.
    Oovv,
    /// `tmp[:o, o:, :o, o:]` at `[kp,kq,kr]`.
    Ovov,
    /// `tmp[:o, o:, o:, :o].conj().transpose(1,0,3,2)` at `[kq,kp,ks]`.
    Voov,
    /// `tmp[:o, o:, o:, o:].conj().transpose(1,0,3,2)` at `[kq,kp,ks]`.
    Vovv,
}

impl UKind {
    /// The SOURCE slice's four spaces, `false` occupied / `true` virtual, in
    /// `tmp`'s own axis order.
    fn source_spaces(self) -> [bool; 4] {
        match self {
            UKind::Oooo => [false, false, false, false],
            UKind::Ooov => [false, false, false, true],
            UKind::Oovv => [false, false, true, true],
            UKind::Ovov => [false, true, false, true],
            UKind::Voov => [false, true, true, false],
            UKind::Vovv => [false, true, true, true],
        }
    }

    /// Whether the block is stored `.conj().transpose(1,0,3,2)` at `[kq,kp,ks]`.
    fn is_transposed(self) -> bool {
        matches!(self, UKind::Voov | UKind::Vovv)
    }

    /// The STORED four spaces — the source spaces, with axes `(1,0,3,2)` for
    /// the two transposed kinds.
    fn stored_spaces(self) -> [bool; 4] {
        let s = self.source_spaces();
        if self.is_transposed() {
            [s[1], s[0], s[3], s[2]]
        } else {
            s
        }
    }

    /// The six, in upstream's slicing order (`:890-895`).
    pub const ALL: [UKind; 6] = [
        UKind::Oooo,
        UKind::Ooov,
        UKind::Oovv,
        UKind::Ovov,
        UKind::Voov,
        UKind::Vovv,
    ];
}

/// One addressable block of [`KuEris`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UBlk {
    /// One of the twenty-three `oppp` slices.
    Pair(UPass, UKind),
    /// `vvvv` (`Aaaa`), `VVVV` (`Bbbb`) or `vvVV` (`AaBb`).
    Quad(UPass),
}

impl UBlk {
    /// `[nkpts³]`-block dimensions at the two spins' `(nocc, nvir)`.
    pub fn dims(self, nocc: (usize, usize), nvir: (usize, usize)) -> [usize; 4] {
        match self {
            UBlk::Pair(p, k) => {
                let (b0, b2) = p.spins();
                let sp = k.stored_spaces();
                let pick = |slot: usize, beta: bool| {
                    let (o, v) = if beta {
                        (nocc.1, nvir.1)
                    } else {
                        (nocc.0, nvir.0)
                    };
                    if sp[slot] { v } else { o }
                };
                [pick(0, b0), pick(1, b0), pick(2, b2), pick(3, b2)]
            }
            UBlk::Quad(p) => {
                let (b0, b2) = p.spins();
                let v0 = if b0 { nvir.1 } else { nvir.0 };
                let v2 = if b2 { nvir.1 } else { nvir.0 };
                [v0, v0, v2, v2]
            }
        }
    }

    /// Upstream's own name — lower case alpha, UPPER case beta.
    pub fn name(self) -> &'static str {
        match self {
            UBlk::Pair(UPass::Aaaa, UKind::Oooo) => "oooo",
            UBlk::Pair(UPass::Aaaa, UKind::Ooov) => "ooov",
            UBlk::Pair(UPass::Aaaa, UKind::Oovv) => "oovv",
            UBlk::Pair(UPass::Aaaa, UKind::Ovov) => "ovov",
            UBlk::Pair(UPass::Aaaa, UKind::Voov) => "voov",
            UBlk::Pair(UPass::Aaaa, UKind::Vovv) => "vovv",
            UBlk::Pair(UPass::Bbbb, UKind::Oooo) => "OOOO",
            UBlk::Pair(UPass::Bbbb, UKind::Ooov) => "OOOV",
            UBlk::Pair(UPass::Bbbb, UKind::Oovv) => "OOVV",
            UBlk::Pair(UPass::Bbbb, UKind::Ovov) => "OVOV",
            UBlk::Pair(UPass::Bbbb, UKind::Voov) => "VOOV",
            UBlk::Pair(UPass::Bbbb, UKind::Vovv) => "VOVV",
            UBlk::Pair(UPass::AaBb, UKind::Oooo) => "ooOO",
            UBlk::Pair(UPass::AaBb, UKind::Ooov) => "ooOV",
            UBlk::Pair(UPass::AaBb, UKind::Oovv) => "ooVV",
            UBlk::Pair(UPass::AaBb, UKind::Ovov) => "ovOV",
            UBlk::Pair(UPass::AaBb, UKind::Voov) => "voOV",
            UBlk::Pair(UPass::AaBb, UKind::Vovv) => "voVV",
            UBlk::Pair(UPass::BbAa, UKind::Oooo) => "OOoo",
            UBlk::Pair(UPass::BbAa, UKind::Ooov) => "OOov",
            UBlk::Pair(UPass::BbAa, UKind::Oovv) => "OOvv",
            UBlk::Pair(UPass::BbAa, UKind::Ovov) => "OVov",
            UBlk::Pair(UPass::BbAa, UKind::Voov) => "VOov",
            UBlk::Pair(UPass::BbAa, UKind::Vovv) => "VOvv",
            UBlk::Quad(UPass::Aaaa) => "vvvv",
            UBlk::Quad(UPass::Bbbb) => "VVVV",
            UBlk::Quad(UPass::AaBb) => "vvVV",
            UBlk::Quad(UPass::BbAa) => "VVvv",
        }
    }

    /// The twenty-six blocks this port builds: `4 × 6 − OOoo` plus the three
    /// `ao2mo_7d` quads.
    pub fn all() -> Vec<UBlk> {
        let mut v = Vec::with_capacity(26);
        for p in UPass::OPPP {
            for k in UKind::ALL {
                // `:939` — `eris.OOoo` is commented out upstream and `:830`
                // sets it `None`. Nothing reads it.
                if p == UPass::BbAa && k == UKind::Oooo {
                    continue;
                }
                v.push(UBlk::Pair(p, k));
            }
        }
        for p in UPass::QUAD {
            v.push(UBlk::Quad(p));
        }
        v
    }
}

/// What [`KuEris::build_fock`] returns: `((focka, fockb), (mo_ea, mo_eb),
/// madelung)`.
pub type UFock = ((ZArr, ZArr), (Vec<Vec<f64>>, Vec<Vec<f64>>), f64);

/// The unrestricted k-point ERIs, Fock matrices and orbital energies.
#[derive(Debug)]
pub struct KuEris {
    pub nkpts: usize,
    /// `(nocca, noccb)`.
    pub nocc: (usize, usize),
    /// `(nmoa, nmob)`.
    pub nmo: (usize, usize),
    /// `(nvira, nvirb)`.
    pub nvir: (usize, usize),
    /// `(focka, fockb)`, each `[nkpts, nmo, nmo]`.
    pub fock: (ZArr, ZArr),
    /// `(mo_ea, mo_eb)`, WITH the Madelung re-add unless `keep_exxdiv`.
    pub mo_energy: (Vec<Vec<f64>>, Vec<Vec<f64>>),
    /// The padded MO coefficients the transform used, per spin.
    pub mo_coeff: (Vec<MoCoeff>, Vec<MoCoeff>),
    /// The Madelung constant that was re-added (0.0 when `keep_exxdiv`).
    pub madelung: f64,
    pool: Arc<ZWorkspacePool>,
    blocks: Vec<(UBlk, KTensor)>,
}

impl KuEris {
    /// `_make_df_eris`'s own first line (`kccsd_uhf.py:1021-1023`).
    ///
    /// The GDF-direct `Wvvvv` route is not ported (see the module doc); this
    /// keeps upstream's refusal addressable so it cannot be lost.
    ///
    /// # Errors
    /// [`PbcCcError::NotImplementedUpstream`] when `dimension == 2`.
    pub fn check_dimension_for_direct_df(dimension: usize) -> Result<(), PbcCcError> {
        if dimension == 2 {
            return Err(PbcCcError::NotImplementedUpstream {
                upstream: "pbc/cc/kccsd_uhf.py:1022",
                what: "_make_df_eris raises NotImplementedError for cell.dimension == 2",
            });
        }
        Ok(())
    }

    /// Build every block from an ALREADY-BUILT unrestricted mean field.
    ///
    /// This is the entry point the oracle tests use, for the reason
    /// [`crate::keris::KEris::from_parts`] states: a correlation energy compared
    /// across two different mean fields measures the mean fields.
    ///
    /// # Errors
    /// Propagates the density-fitting transform, the complex arena's HARD
    /// refusal and every shape check.
    #[allow(clippy::too_many_arguments)]
    pub fn from_parts(
        with_df: &dyn PeriodicDf,
        kconserv: &Kconserv,
        padded: (&PaddedMos, &PaddedMos),
        fock: (ZArr, ZArr),
        mo_energy: (Vec<Vec<f64>>, Vec<Vec<f64>>),
        madelung: f64,
        opts: KErisOpts,
    ) -> Result<Self, PbcCcError> {
        let nkpts = with_df.kpts().len();
        let (pa, pb) = padded;
        let nocc = (pa.nocc, pb.nocc);
        let nmo = (pa.nmo, pb.nmo);
        let nvir = (nmo.0 - nocc.0, nmo.1 - nocc.1);

        let pool = Arc::new(ZWorkspacePool::new(
            (opts.max_memory * 1e6).max(0.0) as usize
        ));
        let allow_spill = matches!(opts.method, ErisMethod::Outcore);
        let mut blocks: Vec<(UBlk, KTensor)> = Vec::with_capacity(26);
        for b in UBlk::all() {
            let dims = b.dims(nocc, nvir);
            blocks.push((
                b,
                KTensor::zeros(&pool, nkpts, KRank::Three, &dims, allow_spill)?,
            ));
        }

        // Occupied-only and virtual-only slices of the padded coefficients.
        let occ = (mo_slice(pa, 0, nocc.0), mo_slice(pb, 0, nocc.1));
        let vir = (mo_slice(pa, nocc.0, nmo.0), mo_slice(pb, nocc.1, nmo.1));

        // ---- the three `ao2mo_7d` quads (`:832-836`), stored at `[ka,kb,kc]`
        // with NO transpose: this file's order IS `(ab|cd) -> [ka,kb,kc]`.
        for p in UPass::QUAD {
            let (b0, b2) = p.spins();
            let (o0, o2) = (pick(&vir, b0), pick(&vir, b2));
            let e7 = with_df
                .ao2mo_7d([o0, o0, o2, o2], 1.0 / nkpts as f64)
                .map_err(df_err)?;
            let blk = UBlk::Quad(p);
            let d = blk.dims(nocc, nvir);
            let t = tensor_of(&blocks, blk)?;
            for ka in 0..nkpts {
                for kb in 0..nkpts {
                    for kc in 0..nkpts {
                        let off = e7.block_offset(ka, kb, kc);
                        let len = e7.block_len();
                        let raw = ZArr::from_ctensor(
                            &d,
                            CTensor {
                                re: e7.data.re[off..off + len].to_vec(),
                                im: e7.data.im[off..off + len].to_vec(),
                            },
                        )?;
                        t.set_block(&pool, &[ka, kb, kc], raw.data())?;
                    }
                }
            }
        }

        // ---- the four `oppp` passes (`:882-947`)
        let inv = 1.0 / nkpts as f64;
        for p in UPass::OPPP {
            let (b0, b2) = p.spins();
            let orbo = pick(&occ, b0);
            let mo0 = pick_full(padded, b0);
            let mo2 = pick_full(padded, b2);
            let (n0o, n0m) = (
                if b0 { nocc.1 } else { nocc.0 },
                if b0 { nmo.1 } else { nmo.0 },
            );
            let n2m = if b2 { nmo.1 } else { nmo.0 };
            for kp in 0..nkpts {
                for kq in 0..nkpts {
                    for kr in 0..nkpts {
                        let ks = kconserv.get(kp, kq, kr) as usize;
                        let eri = with_df
                            .ao2mo(
                                [&orbo[kp], &mo0[kq], &mo2[kr], &mo2[ks]],
                                [kp, kq, kr, ks],
                                false,
                            )
                            .map_err(df_err)?;
                        let tmp = ZArr::from_ctensor(&[n0o, n0m, n2m, n2m], eri.restore_s1().data)?;
                        for k in UKind::ALL {
                            if p == UPass::BbAa && k == UKind::Oooo {
                                continue;
                            }
                            let blk = UBlk::Pair(p, k);
                            let sp = k.source_spaces();
                            // The source slice, in `tmp`'s own axis order.
                            let base = |slot: usize| -> (usize, usize) {
                                let beta = if slot < 2 { b0 } else { b2 };
                                let (o, m) = if beta {
                                    (nocc.1, nmo.1)
                                } else {
                                    (nocc.0, nmo.0)
                                };
                                if sp[slot] { (o, m) } else { (0, o) }
                            };
                            let src = tmp.slice_axes(&[base(0), base(1), base(2), base(3)])?;
                            let (out, addr) = if k.is_transposed() {
                                (src.conj().transpose(&[1, 0, 3, 2])?, [kq, kp, ks])
                            } else {
                                (src, [kp, kq, kr])
                            };
                            let mut out = out;
                            out.scale(inv);
                            tensor_of(&blocks, blk)?.set_block(&pool, &addr, out.data())?;
                        }
                    }
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
            mo_coeff: (pa.mo_coeff.clone(), pb.mo_coeff.clone()),
            madelung,
            pool,
            blocks,
        })
    }

    /// The unrestricted Fock half of the build (`:855-873`).
    ///
    /// `dm` is `(dm_alpha, dm_beta)` in the AO basis. The UHF effective
    /// potential is `vj_a + vj_b - vk_sigma` (`kuhf.py`'s `get_veff`), which is
    /// where this differs from [`crate::keris::KEris::build_fock`] beyond the
    /// obvious doubling.
    ///
    /// # Errors
    /// Propagates the density-fitting builder and the Madelung constant.
    pub fn build_fock(
        cell: &Cell,
        with_df: &dyn PeriodicDf,
        padded: (&PaddedMos, &PaddedMos),
        dm: (&[CTensor], &[CTensor]),
        opts: KErisOpts,
    ) -> Result<UFock, PbcCcError> {
        let kpts = with_df.kpts();
        let nkpts = kpts.len();
        let nao = cell.mol.nao_nr;
        let exxdiv = if opts.keep_exxdiv { opts.exxdiv } else { None };
        let hcore = pyscf_pbc_df::get_hcore(with_df, kpts).map_err(df_err)?;
        let dms: Vec<Vec<CTensor>> = vec![dm.0.to_vec(), dm.1.to_vec()];
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

        let mut fock = Vec::with_capacity(2);
        let mut energies = Vec::with_capacity(2);
        for (s, pad) in [(0usize, padded.0), (1usize, padded.1)] {
            let nmo = pad.nmo;
            let mut f = ZArr::zeros(&[nkpts, nmo, nmo]);
            let mut me: Vec<Vec<f64>> = Vec::with_capacity(nkpts);
            for k in 0..nkpts {
                let mut fao = CTensor::zeros(nao * nao);
                for i in 0..nao * nao {
                    fao.re[i] = hcore[k].re[i] + vj[0][k].re[i] + vj[1][k].re[i] - vk[s][k].re[i];
                    fao.im[i] = hcore[k].im[i] + vj[0][k].im[i] + vj[1][k].im[i] - vk[s][k].im[i];
                }
                let blk = mo_transform(&fao, nao, &pad.mo_coeff[k], nmo)?;
                me.push(
                    (0..nmo)
                        .map(|p| blk.at(&[p, p]).map(|v| v.0))
                        .collect::<Result<_, _>>()?,
                );
                f.set_leading(&[k], &blk)?;
            }
            fock.push(f);
            energies.push(me);
        }

        let madelung = if opts.keep_exxdiv {
            0.0
        } else {
            pyscf_pbc_gto::madelung(cell, kpts, None)
                .map_err(|e| PbcCcError::Shape(format!("madelung: {e}")))?
        };
        if !opts.keep_exxdiv {
            for (s, pad) in [(0usize, padded.0), (1usize, padded.1)] {
                energies[s] = energies[s]
                    .iter()
                    .map(|e| adjust_occ(e, pad.nocc, -madelung))
                    .collect();
            }
        }

        let mut fock = fock.into_iter();
        let mut energies = energies.into_iter();
        let fa = fock.next().ok_or_else(|| shape("no alpha fock"))?;
        let fb = fock.next().ok_or_else(|| shape("no beta fock"))?;
        let ea = energies.next().ok_or_else(|| shape("no alpha mo_energy"))?;
        let eb = energies.next().ok_or_else(|| shape("no beta mo_energy"))?;
        Ok(((fa, fb), (ea, eb), madelung))
    }

    /// One block at a k-triple.
    ///
    /// # Errors
    /// [`PbcCcError`] for an unbuilt block (`OOoo`, `VVvv`) or a bad k-address.
    pub fn blk(&self, b: UBlk, k0: usize, k1: usize, k2: usize) -> Result<ZArr, PbcCcError> {
        let t = self.tensor(b)?;
        let c = t.block(&self.pool, &[k0, k1, k2])?;
        Ok(ZArr::from_ctensor(&b.dims(self.nocc, self.nvir), c)?)
    }

    /// The storage tier a block landed in.
    ///
    /// # Errors
    /// As [`KuEris::blk`].
    pub fn tier(&self, b: UBlk) -> Result<Tier, PbcCcError> {
        Ok(self.tensor(b)?.tier())
    }

    /// The exact bytes one block occupies.
    ///
    /// # Errors
    /// As [`KuEris::blk`].
    pub fn bytes(&self, b: UBlk) -> Result<usize, PbcCcError> {
        Ok(self.tensor(b)?.bytes())
    }

    /// Total exact bytes of every built block.
    pub fn total_bytes(&self) -> usize {
        self.blocks.iter().map(|(_, t)| t.bytes()).sum()
    }

    /// Peak in-memory bytes charged by the arena.
    pub fn live_inmem_bytes(&self) -> usize {
        self.pool.live_inmem_bytes()
    }

    /// `fock[spin][k]`, `nmo × nmo`. `beta = false` is alpha.
    ///
    /// # Errors
    /// [`PbcCcError::Shape`] on a bad k index.
    pub fn fock_at(&self, beta: bool, k: usize) -> Result<ZArr, PbcCcError> {
        if beta {
            self.fock.1.slice_leading(&[k])
        } else {
            self.fock.0.slice_leading(&[k])
        }
    }

    /// `fock[spin][k][:nocc, nocc:]`.
    ///
    /// # Errors
    /// As [`KuEris::fock_at`].
    pub fn fov(&self, beta: bool, k: usize) -> Result<ZArr, PbcCcError> {
        let (o, m) = self.spin_dims(beta);
        self.fock_sub(beta, k, 0, o, o, m)
    }

    /// `fock[spin][k][:nocc, :nocc]`.
    ///
    /// # Errors
    /// As [`KuEris::fock_at`].
    pub fn foo(&self, beta: bool, k: usize) -> Result<ZArr, PbcCcError> {
        let (o, _) = self.spin_dims(beta);
        self.fock_sub(beta, k, 0, o, 0, o)
    }

    /// `fock[spin][k][nocc:, nocc:]`.
    ///
    /// # Errors
    /// As [`KuEris::fock_at`].
    pub fn fvv(&self, beta: bool, k: usize) -> Result<ZArr, PbcCcError> {
        let (o, m) = self.spin_dims(beta);
        self.fock_sub(beta, k, o, m, o, m)
    }

    fn spin_dims(&self, beta: bool) -> (usize, usize) {
        if beta {
            (self.nocc.1, self.nmo.1)
        } else {
            (self.nocc.0, self.nmo.0)
        }
    }

    fn fock_sub(
        &self,
        beta: bool,
        k: usize,
        r0: usize,
        r1: usize,
        c0: usize,
        c1: usize,
    ) -> Result<ZArr, PbcCcError> {
        self.fock_at(beta, k)?.slice_axes(&[(r0, r1), (c0, c1)])
    }

    fn tensor(&self, b: UBlk) -> Result<&KTensor, PbcCcError> {
        self.blocks
            .iter()
            .find(|(x, _)| *x == b)
            .map(|(_, t)| t)
            .ok_or_else(|| PbcCcError::Shape(format!("no {} block", b.name())))
    }
}

fn tensor_of(blocks: &[(UBlk, KTensor)], b: UBlk) -> Result<&KTensor, PbcCcError> {
    blocks
        .iter()
        .find(|(x, _)| *x == b)
        .map(|(_, t)| t)
        .ok_or_else(|| PbcCcError::Shape(format!("no {} block", b.name())))
}

fn pick(pair: &(Vec<MoCoeff>, Vec<MoCoeff>), beta: bool) -> &[MoCoeff] {
    if beta { &pair.1 } else { &pair.0 }
}

fn pick_full<'a>(padded: (&'a PaddedMos, &'a PaddedMos), beta: bool) -> &'a [MoCoeff] {
    if beta {
        &padded.1.mo_coeff
    } else {
        &padded.0.mo_coeff
    }
}

/// `mo_coeff[k][:, lo:hi]` for every k.
fn mo_slice(pad: &PaddedMos, lo: usize, hi: usize) -> Vec<MoCoeff> {
    let nmo = pad.nmo;
    let n = hi - lo;
    pad.mo_coeff
        .iter()
        .map(|m| {
            let mut c = CTensor::zeros(m.nao * n);
            for a in 0..m.nao {
                for p in 0..n {
                    c.re[a * n + p] = m.c.re[a * nmo + lo + p];
                    c.im[a * n + p] = m.c.im[a * nmo + lo + p];
                }
            }
            MoCoeff::new(m.nao, n, c)
        })
        .collect()
}

/// `Cᴴ F_AO C`, with the conjugation on `C` explicit (`16-CONTEXT §3.2`).
fn mo_transform(fao: &CTensor, nao: usize, c: &MoCoeff, nmo: usize) -> Result<ZArr, PbcCcError> {
    let mut blk = ZArr::zeros(&[nmo, nmo]);
    for p in 0..nmo {
        for q in 0..nmo {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for a in 0..nao {
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
    Ok(blk)
}

fn shape(msg: &str) -> PbcCcError {
    PbcCcError::Shape(msg.to_string())
}

fn df_err(e: pyscf_pbc_df::PbcDfError) -> PbcCcError {
    PbcCcError::Shape(format!("density fitting: {e}"))
}

//! k-symmetry-adapted restricted MP2 (`pyscf/pbc/mp/kmp2_ksymm.py`, 285 l).
//!
//! # What the symmetry actually buys, and where
//!
//! Upstream's `KMP2.__init__` (`kmp2.py:713-722`) does something worth stating
//! before anything else: when `mf.kpts` is a `KPoints`, it **unfolds** the MO
//! coefficients, energies and occupations to the **full BZ** and sets
//! `nkpts = kpts.nkpts`. So the k-symmetric KMP2 is not an IBZ calculation in
//! the sense the k-symmetric SCF is — every quantity it touches lives on the
//! whole zone. The only thing symmetry changes is **which `(ki, kj, ka)`
//! triples are evaluated**: the `nkpts^3` loop of `kmp2.kernel` becomes a loop
//! over the s2 k-quartet classes of [`KPoints::make_k4_ibz`], each evaluated
//! once and multiplied by its class size.
//!
//! That is why [`unfold_kscf_result`] is a public, explicit step here rather
//! than something hidden inside a constructor: the full-BZ MO set is the
//! contract, and a caller that hands [`KsymAdaptedKmp2`] an IBZ-length result
//! by mistake gets an error, not a plausible wrong number.
//!
//! Measured on `si [2,2,2]` (`measurements/gate_k4_s2.out`): 512 k-triples,
//! **50** s1 classes, **36** s2 classes. On `si [1,1,2]`: 8 -> 6.
//!
//! # Two deliberate deviations from upstream, both stated
//!
//! 1. **The ERI route.** `kmp2_ksymm.kernel:46` takes
//!    `mp._scf.with_df.ao2mo` unconditionally, while `kmp2.kernel:114-125`
//!    branches on `with_df_ints` and uses the density-fitted `Lov` route for
//!    GDF. Upstream's own "ksymm vs full BZ" comparison on a `density_fit`
//!    reference is therefore a comparison of two DIFFERENT integral routes as
//!    much as of two k-sets — which is very likely most of the `1.067e-9` it
//!    measures on si (`measurements/gate_mp2.out`), against `3.096e-16` on a
//!    He cell whose `naux` makes the two routes agree far more closely.
//!    This port defaults to [`EriRoute::MatchFullBz`], which takes the SAME
//!    route the full-BZ kernel would, so the port's own ksymm-vs-full-BZ gate
//!    measures the symmetry and nothing else. [`EriRoute::Ao2mo`] reproduces
//!    upstream's literal choice and is what the upstream comparison uses.
//! 2. **`make_rdm1`'s padding index.** Upstream zips `nkpts_ibz` density
//!    blocks against the **first `nkpts_ibz`** entries of a full-BZ
//!    `padding_k_idx` list (`kmp2_ksymm.py:135`). Those are the padding
//!    patterns of BZ points `0..nkpts_ibz`, not of the IBZ representatives.
//!    The two agree whenever every k-point has the same `nocc`/`nmo` — which
//!    is every fixture upstream tests — and disagree otherwise. This port
//!    indexes `padding_idxs[ibz2bz[i]]`, which is the pattern of the point the
//!    block actually belongs to.

use pyscf_algebra::{CTensor, oracle_sum, oracle_zdotu_re};
use pyscf_pbc_df::{CoulGCache, PeriodicDf};
use pyscf_pbc_gto::Cell;
use pyscf_pbc_scf::KScfResult;
use pyscf_pbc_symm::kpts::KPoints;
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    FrozenK, Kmp2, Kmp2Result, LovTable, PaddedMos, PaddingIdx, PaddingKind, PbcMpError, RdmKind,
    T2, build_lov, kmp2_kernel::LARGE_DENOM, padding_k_idx,
};

// =====================================================================
// The full-BZ MO set — `kmp2.py:715-722`
// =====================================================================

/// Unfold an IBZ-length [`KScfResult`] (the output of `KsymAdaptedKrhf`) into
/// a full-BZ one, exactly as `KMP2.__init__` does for a `KPoints` reference:
/// `transform_mo_energy` / `transform_mo_coeff` / `transform_mo_occ`
/// (`kmp2.py:719-721`).
///
/// The density matrices are unfolded too (`transform_dm`), so the returned
/// object is internally consistent even though KMP2 never reads them; the
/// energies, `converged` and `nset` are carried across unchanged.
///
/// **`mo_coeff` layout.** [`KScfResult`] stores COLUMN-MAJOR `nao x nmo`
/// (`types.rs:119`) and `KPoints::transform_mo_coeff` speaks ROW-MAJOR. The
/// two conversions here are the only place the two meet in this crate — the
/// 14-05 defect shape (17-CONTEXT §3.2), so it is one function rather than a
/// convention every caller has to remember.
///
/// # Errors
/// [`PbcMpError::Shape`] if `result` is not a single restricted channel of
/// `nkpts_ibz` k-points, plus every error `KPoints`'s transforms raise.
pub fn unfold_kscf_result(
    result: &KScfResult,
    kpts: &KPoints,
    cell: &Cell,
) -> Result<KScfResult, PbcMpError> {
    if result.nset != 1 {
        return Err(PbcMpError::Shape {
            what: format!(
                "k-symmetric KMP2 is restricted-only; the reference has {} spin channels",
                result.nset
            ),
        });
    }
    if result.nkpts != kpts.nkpts_ibz() {
        return Err(PbcMpError::Shape {
            what: format!(
                "the reference has {} k-points but KPoints has {} in the IBZ — \
                 KsymAdaptedKmp2 takes the IBZ output of a k-symmetric SCF",
                result.nkpts,
                kpts.nkpts_ibz()
            ),
        });
    }
    let nao = cell.nao_nr;
    let nkpts = kpts.nkpts();

    let mut rows = Vec::with_capacity(result.nkpts);
    let mut nmo = 0usize;
    for c in &result.mo_coeff {
        if c.len() % nao != 0 {
            return Err(PbcMpError::Shape {
                what: format!(
                    "MO coefficient length {} is not a multiple of nao={nao}",
                    c.len()
                ),
            });
        }
        let n = c.len() / nao;
        if nmo == 0 {
            nmo = n;
        } else if n != nmo {
            return Err(PbcMpError::Shape {
                what: "k-symmetric KMP2 needs the same nmo at every IBZ k-point".into(),
            });
        }
        rows.push(col_major_to_row_major(c, nao, n));
    }

    let mo_coeff_bz = kpts
        .transform_mo_coeff(cell, &rows, nao, nmo)
        .map_err(symm_err)?;
    let mo_energy = kpts
        .transform_mo_energy(&result.mo_energy)
        .map_err(symm_err)?;
    let mo_occ = kpts.transform_mo_occ(&result.mo_occ).map_err(symm_err)?;

    let dm = if result.dm.len() == 1 && result.dm[0].len() == result.nkpts {
        let as_c: Vec<Vec<num_complex::Complex64>> =
            result.dm[0].iter().map(ctensor_to_complex).collect();
        let bz = kpts.transform_dm(cell, &as_c, nao).map_err(symm_err)?;
        vec![bz.iter().map(|m| complex_to_ctensor(m)).collect()]
    } else {
        result.dm.clone()
    };

    Ok(KScfResult {
        mo_coeff: mo_coeff_bz
            .iter()
            .map(|c| row_major_to_col_major(c, nao, nmo))
            .collect(),
        mo_energy,
        mo_occ,
        dm,
        nkpts,
        ..result.clone()
    })
}

fn symm_err(e: pyscf_pbc_symm::PbcSymmError) -> PbcMpError {
    PbcMpError::Shape {
        what: format!("k-point symmetry transform failed: {e}"),
    }
}

fn ctensor_to_complex(t: &CTensor) -> Vec<num_complex::Complex64> {
    t.re.iter()
        .zip(t.im.iter())
        .map(|(&re, &im)| num_complex::Complex64::new(re, im))
        .collect()
}

fn complex_to_ctensor(v: &[num_complex::Complex64]) -> CTensor {
    CTensor {
        re: v.iter().map(|c| c.re).collect(),
        im: v.iter().map(|c| c.im).collect(),
    }
}

fn col_major_to_row_major(c: &CTensor, nao: usize, nmo: usize) -> Vec<num_complex::Complex64> {
    let mut out = vec![num_complex::Complex64::new(0.0, 0.0); nao * nmo];
    for i in 0..nao {
        for p in 0..nmo {
            out[i * nmo + p] = num_complex::Complex64::new(c.re[p * nao + i], c.im[p * nao + i]);
        }
    }
    out
}

fn row_major_to_col_major(v: &[num_complex::Complex64], nao: usize, nmo: usize) -> CTensor {
    let mut out = CTensor::zeros(nao * nmo);
    for i in 0..nao {
        for p in 0..nmo {
            out.re[p * nao + i] = v[i * nmo + p].re;
            out.im[p * nao + i] = v[i * nmo + p].im;
        }
    }
    out
}

// =====================================================================
// The object — `kmp2_ksymm.py:226-256` (`KsymAdaptedKMP2`)
// =====================================================================

/// Which integral route the k-symmetric kernel takes. See the module doc,
/// deviation 1.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EriRoute {
    /// The route `kmp2.kernel` would take for this DF object — the
    /// density-fitted `Lov` contraction when the builder carries `_cderi`,
    /// `ao2mo` otherwise. **Default**, because it makes the port's own
    /// ksymm-vs-full-BZ gate a measurement of the symmetry alone.
    MatchFullBz,
    /// `ao2mo` unconditionally — `kmp2_ksymm.py:46` taken literally. Use this
    /// when comparing against upstream's own k-symmetric number.
    Ao2mo,
}

/// `KsymAdaptedKMP2` (`kmp2_ksymm.py:226`).
///
/// Holds the plain [`Kmp2`] it delegates to, so `kernel_with_t2`, the padding
/// machinery and the memory pre-flight are shared rather than re-derived —
/// upstream gets the same by subclassing `kmp2.KMP2`.
pub struct KsymAdaptedKmp2<'a> {
    /// The full-BZ KMP2 this adapts. `mp.kpts` is the full zone.
    pub mp: Kmp2<'a>,
    /// The k-point symmetry. Held by composition (D-PBC-25).
    pub kpts: &'a KPoints,
    pub route: EriRoute,
}

impl<'a> KsymAdaptedKmp2<'a> {
    /// `result` must be the **unfolded, full-BZ** reference — the output of
    /// [`unfold_kscf_result`], not the IBZ result the SCF returned.
    ///
    /// # Errors
    /// [`PbcMpError::Shape`] when `result` is not full-BZ-sized for `kpts`,
    /// plus [`Kmp2::new`].
    pub fn new(
        result: &'a KScfResult,
        with_df: &'a dyn PeriodicDf,
        kpts: &'a KPoints,
    ) -> Result<Self, PbcMpError> {
        if result.nkpts != kpts.nkpts() {
            return Err(PbcMpError::Shape {
                what: format!(
                    "KsymAdaptedKmp2 needs the FULL-BZ reference ({} k-points); it was given {}. \
                     Call unfold_kscf_result first.",
                    kpts.nkpts(),
                    result.nkpts
                ),
            });
        }
        Ok(Self {
            mp: Kmp2::new(result, with_df)?,
            kpts,
            route: EriRoute::MatchFullBz,
        })
    }

    /// The frozen-core specification, shared with the wrapped [`Kmp2`].
    pub fn set_frozen(&mut self, frozen: FrozenK) {
        self.mp.frozen = frozen;
    }

    /// `kmp2_ksymm.py:227-249` with `with_t2 = False` — the symmetry-reduced
    /// energy.
    ///
    /// # Errors
    /// As [`ksymm_kernel`].
    pub fn kernel(&self, cell: &Cell) -> Result<Kmp2Result, PbcMpError> {
        let padded = self.mp.padded_mos()?;
        let (ss, os) = ksymm_kernel(&self.mp, self.kpts, cell, &padded, self.route)?;
        Ok(Kmp2Result {
            e_corr: ss + os,
            e_corr_ss: ss,
            e_corr_os: os,
            e_hf: self.mp.e_hf(),
            e_tot: self.mp.e_hf() + ss + os,
            t2: None,
        })
    }

    /// `kmp2_ksymm.py:120-126` — `kernel_with_t2`.
    ///
    /// Upstream's own comment is the whole design: *"we need almost all t2 for
    /// computing rdm, so simply use kmp2 without symmetry"* (`:121`). It
    /// temporarily replaces `mp.kpts` with the plain k-point list and calls
    /// `kmp2.kernel`. This port does not need the substitution — the wrapped
    /// [`Kmp2`] already holds the full-BZ list — so it is a direct delegation.
    ///
    /// # Errors
    /// As [`Kmp2::kernel`].
    pub fn kernel_with_t2(&self) -> Result<Kmp2Result, PbcMpError> {
        self.mp.kernel()
    }

    /// `kmp2_ksymm.py:145-189` — `make_t2_for_rdm1`.
    ///
    /// # Errors
    /// As [`make_t2_for_rdm1`].
    pub fn make_t2_for_rdm1(&self, cell: &Cell) -> Result<PartialT2, PbcMpError> {
        let padded = self.mp.padded_mos()?;
        make_t2_for_rdm1(&self.mp, self.kpts, cell, &padded, self.route)
    }

    /// `kmp2_ksymm.py:128-143` — `make_rdm1`, `nkpts_ibz` blocks.
    ///
    /// # Errors
    /// As [`make_rdm1_ksymm`].
    pub fn make_rdm1(&self, t2: &PartialT2, kind: RdmKind) -> Result<Vec<CTensor>, PbcMpError> {
        let padded = self.mp.padded_mos()?;
        make_rdm1_ksymm(
            t2,
            &self.mp.khelper.kconserv,
            self.kpts,
            &padded.nmo_per_kpt,
            &padded.nocc_per_kpt,
            kind,
        )
    }

    /// `kmp2_ksymm.py:253-254` — upstream's own `raise NotImplementedError`.
    ///
    /// The full-BZ [`Kmp2::make_rdm2`] exists and is gated; there is no
    /// k-symmetric two-particle density matrix upstream, so this port ships
    /// the refusal rather than a number no oracle can check.
    ///
    /// # Errors
    /// Always.
    pub fn make_rdm2(&self) -> Result<crate::Rdm2, PbcMpError> {
        Err(PbcMpError::Shape {
            what: "KsymAdaptedKMP2.make_rdm2 is `raise NotImplementedError` upstream \
                   (kmp2_ksymm.py:253-254). Run the full-BZ KMP2 for a 2-RDM."
                .into(),
        })
    }
}

// =====================================================================
// Task 1 — the kernel (`kmp2_ksymm.py:30-118`)
// =====================================================================

/// One `(ki, kj)` group's `oovv` blocks, keyed by `ka`.
struct GroupEris {
    /// `ka -> the (nocc, nocc, nvir, nvir) block`, only for the `ka` this
    /// group needs.
    blocks: HashMap<usize, CTensor>,
}

impl GroupEris {
    fn get(&self, ka: usize) -> Result<&CTensor, PbcMpError> {
        self.blocks.get(&ka).ok_or_else(|| PbcMpError::Shape {
            what: format!("kmp2_ksymm: oovv block at ka = {ka} was never built"),
        })
    }
}

/// The `(ka, kb)` pairs one `(ki, kj)` group must transform —
/// `kmp2_ksymm.py:66-71`.
///
/// **Both orders of every pair.** The direct term at `(ka, kb)` reads
/// `oovv[ka]`, the exchange term reads `oovv[kb]`, so `[b, a]` is appended
/// beside `[a, b]` for every class in the group. This is the k-symmetric
/// analogue of `kmp2.kernel`'s two separate `ka` loops (15-CONTEXT §3), and
/// the read/write ordering constraint that made those two loops necessary is
/// the same one that makes this list contain both orders: **the fusion
/// boundary moved, the constraint did not.** With the group's `oovv` table
/// built first, the energy loop can then be a single pass.
fn group_kab(k4: &[[usize; 4]], start: usize, end: usize) -> Vec<[usize; 2]> {
    let mut kab: Vec<[usize; 2]> = Vec::with_capacity(2 * (end - start));
    for q in &k4[start..end] {
        kab.push([q[2], q[3]]);
        kab.push([q[3], q[2]]);
    }
    kab.sort_unstable();
    kab.dedup();
    kab
}

#[allow(clippy::too_many_arguments)]
fn build_group_eris(
    mp: &Kmp2<'_>,
    padded: &PaddedMos,
    lov: Option<&LovTable>,
    caches: Option<&[Arc<CoulGCache>]>,
    ki: usize,
    kj: usize,
    kab: &[[usize; 2]],
) -> Result<GroupEris, PbcMpError> {
    let nk = mp.kpts.len();
    let (nocc, nmo) = (padded.nocc, padded.nmo);
    let built: Result<Vec<_>, PbcMpError> = kab
        .par_iter()
        .map(|&[ka, kb]| {
            let block = match lov {
                Some(l) => crate::kmp2_kernel::df_oovv(l, ki, ka, kj, kb),
                None => crate::kmp2_kernel::ao2mo_oovv(
                    mp.with_df,
                    &padded.mo_coeff,
                    ki,
                    ka,
                    kj,
                    kb,
                    nocc,
                    nmo,
                    nk,
                    caches.map(|c| c[ki * nk + ka].as_ref()),
                ),
            }?;
            Ok((ka, block))
        })
        .collect();
    Ok(GroupEris {
        blocks: built?.into_iter().collect(),
    })
}

/// `_add_padding`-aware `eia + ejb`, identical to `kmp2_kernel`'s.
///
/// `LARGE_DENOM` is load-bearing arithmetic, not a guard (15-CONTEXT §3):
/// padded orbitals must contribute `~1e-28`, and skipping them is a different
/// program.
#[allow(clippy::too_many_arguments)]
fn denom(
    padded: &PaddedMos,
    occ_idx: &[Vec<usize>],
    vir_idx: &[Vec<usize>],
    nocc: usize,
    ki: usize,
    ka: usize,
    kj: usize,
    kb: usize,
    i: usize,
    j: usize,
    a: usize,
    b: usize,
) -> f64 {
    let eia = if occ_idx[ki].contains(&i) && vir_idx[ka].contains(&a) {
        padded.mo_energy[ki][i] - padded.mo_energy[ka][nocc + a]
    } else {
        LARGE_DENOM
    };
    let ejb = if occ_idx[kj].contains(&j) && vir_idx[kb].contains(&b) {
        padded.mo_energy[kj][j] - padded.mo_energy[kb][nocc + b]
    } else {
        LARGE_DENOM
    };
    eia + ejb
}

/// The `Lov` table and `CoulG` cache one ERI route needs.
type RouteSetup = (Option<LovTable>, Option<Vec<Arc<CoulGCache>>>);

/// Build whichever of the two the chosen route needs, once.
fn route_setup(
    mp: &Kmp2<'_>,
    padded: &PaddedMos,
    route: EriRoute,
) -> Result<RouteSetup, PbcMpError> {
    let use_lov = route == EriRoute::MatchFullBz && mp.with_df_ints;
    let lov = if use_lov {
        Some(build_lov(mp.with_df, &padded.mo_coeff, padded.nocc)?)
    } else {
        None
    };
    let caches = if lov.is_none() && matches!(mp.with_df.name(), "FFTDF" | "AFTDF") {
        let nk = mp.kpts.len();
        let mut unique = HashMap::<[u64; 3], Arc<CoulGCache>>::new();
        let mut table = Vec::with_capacity(nk * nk);
        for ki in 0..nk {
            for ka in 0..nk {
                let q = [
                    mp.kpts[ka][0] - mp.kpts[ki][0],
                    mp.kpts[ka][1] - mp.kpts[ki][1],
                    mp.kpts[ka][2] - mp.kpts[ki][2],
                ];
                let key = q.map(f64::to_bits);
                let cache = match unique.get(&key) {
                    Some(v) => Arc::clone(v),
                    None => {
                        let v = Arc::new(CoulGCache::build(mp.cell, mp.with_df.mesh(), q)?);
                        unique.insert(key, Arc::clone(&v));
                        v
                    }
                };
                table.push(cache);
            }
        }
        Some(table)
    } else {
        None
    };
    Ok((lov, caches))
}

/// Split the lexicographically ordered s2 class list into `(ki, kj)` groups —
/// `kmp2_ksymm.py:56-58` (`np.unique(kijab[:,:2], axis=0, return_index=True)`).
///
/// Correct **only because** `make_k4_ibz(sym = "s2")` returns the classes in
/// ascending lexicographic order, so equal `(ki, kj)` rows are contiguous.
/// `tests/kpts_k4_s2.rs` asserts that ordering separately, which is what makes
/// this a scan rather than a hash.
fn groups(k4: &[[usize; 4]]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = 0usize;
    for i in 1..=k4.len() {
        if i == k4.len() || k4[i][0] != k4[start][0] || k4[i][1] != k4[start][1] {
            out.push((start, i));
            start = i;
        }
    }
    out
}

/// `kmp2_ksymm.py:30-118` — `kernel`, `with_t2 = False`.
///
/// Returns `(e_corr_ss, e_corr_os)`.
///
/// # Determinism
/// Every reduction is an [`oracle_sum`] over a fixed-length vector built in
/// index order: within a class the block product, across the classes of one
/// group, and across the groups. The rayon `par_iter` over groups therefore
/// changes nothing about the arithmetic — `tests/kmp2_ksymm.rs` proves it with
/// a 1-vs-8-thread bit-identity assertion rather than only on paper.
///
/// # Errors
/// [`PbcMpError::Shape`] on a degenerate MO space or an inconsistent class
/// table, [`PbcMpError::Memory`] from the pre-flight, plus every DF error.
pub fn ksymm_kernel(
    mp: &Kmp2<'_>,
    kpts: &KPoints,
    cell: &Cell,
    padded: &PaddedMos,
    route: EriRoute,
) -> Result<(f64, f64), PbcMpError> {
    let nk = mp.kpts.len();
    if nk != kpts.nkpts() {
        return Err(PbcMpError::Shape {
            what: format!(
                "kmp2_ksymm: the KMP2 object samples {nk} k-points but KPoints has {}",
                kpts.nkpts()
            ),
        });
    }
    let (nocc, nmo) = (padded.nocc, padded.nmo);
    if nocc == 0 || nocc >= nmo {
        return Err(PbcMpError::Shape {
            what: "KMP2 needs non-empty occupied and virtual spaces".into(),
        });
    }
    let nvir = nmo - nocc;
    let nov2 = (nocc * nvir).saturating_pow(2);

    let k4 = kpts.make_k4_ibz(cell, "s2").map_err(symm_err)?;
    let grp = groups(&k4.k4);

    // Pre-flight: the widest group's `oovv` table, times the groups running
    // concurrently. `kmp2.kernel`'s `nkpts^3 * nov2` t2 term is absent — this
    // kernel builds no t2 at all.
    let widest = grp
        .iter()
        .map(|&(s, e)| group_kab(&k4.k4, s, e).len())
        .max()
        .unwrap_or(0);
    let per_group_mb = (widest * nov2 * 16) as f64 / 1.0e6;
    let naux = if route == EriRoute::MatchFullBz && mp.with_df_ints {
        mp.with_df.get_naoaux()?
    } else {
        0
    };
    let lov_mb = (nk * nk * naux * nocc * nvir * 16) as f64 / 1.0e6;
    if lov_mb + per_group_mb > mp.max_memory {
        return Err(PbcMpError::Memory {
            required_mb: lov_mb + per_group_mb,
            available_mb: mp.max_memory,
        });
    }
    let live_groups = (((mp.max_memory - lov_mb) / per_group_mb.max(f64::MIN_POSITIVE)) as usize)
        .max(1)
        .min(rayon::current_num_threads())
        .min(grp.len().max(1));

    let split = match padding_k_idx(
        &padded.nmo_per_kpt,
        &padded.nocc_per_kpt,
        PaddingKind::Split,
    )? {
        PaddingIdx::Split { occupied, virtuals } => (occupied, virtuals),
        PaddingIdx::Joint(_) => unreachable!(),
    };
    let (lov, caches) = route_setup(mp, padded, route)?;

    let n3 = (nk as f64).powi(3);
    let mut per_group: Vec<(f64, f64)> = Vec::with_capacity(grp.len());
    for batch in grp.chunks(live_groups) {
        let chunk: Result<Vec<_>, PbcMpError> = batch
            .par_iter()
            .map(|&(start, end)| {
                let (ki, kj) = (k4.k4[start][0], k4.k4[start][1]);
                let kab = group_kab(&k4.k4, start, end);
                let eris =
                    build_group_eris(mp, padded, lov.as_ref(), caches.as_deref(), ki, kj, &kab)?;
                let mut ss = Vec::with_capacity(end - start);
                let mut os = Vec::with_capacity(end - start);
                for j in start..end {
                    let [qi, qj, ka, kb] = k4.k4[j];
                    debug_assert_eq!((qi, qj), (ki, kj));
                    // `kmp2_ksymm.py:104-105` — upstream's own assertion that
                    // the class table and the iteration order agree. A
                    // mismatch means `make_k4_ibz` handed back an order the
                    // grouping scan cannot follow, which would silently pair
                    // a class with another class's weight.
                    let flat = kpts.ktuple_to_index(&[ki, kj, ka]);
                    if k4.bz2ibz[flat] != j {
                        return Err(PbcMpError::Shape {
                            what: format!(
                                "kmp2_ksymm: class ({ki},{kj},{ka}) is row {j} of the s2 list \
                                 but bz2ibz says {}",
                                k4.bz2ibz[flat]
                            ),
                        });
                    }
                    let oovv_a = eris.get(ka)?;
                    let oovv_b = eris.get(kb)?;
                    let mut amp = CTensor::zeros(nov2);
                    let mut exch = CTensor::zeros(nov2);
                    for i in 0..nocc {
                        for jj in 0..nocc {
                            for a in 0..nvir {
                                for b in 0..nvir {
                                    let p = ((i * nocc + jj) * nvir + a) * nvir + b;
                                    let d = denom(
                                        padded, &split.0, &split.1, nocc, ki, ka, kj, kb, i, jj, a,
                                        b,
                                    );
                                    amp.re[p] = oovv_a.re[p] / d;
                                    amp.im[p] = -oovv_a.im[p] / d;
                                    let q = ((i * nocc + jj) * nvir + b) * nvir + a;
                                    exch.re[p] = oovv_b.re[q];
                                    exch.im[p] = oovv_b.im[q];
                                }
                            }
                        }
                    }
                    let edi = 2.0 * oracle_zdotu_re(&amp, oovv_a);
                    let exi = -oracle_zdotu_re(&amp, &exch);
                    // `weight * nkpts^3` is the exact number of BZ k-triples
                    // in this class.
                    let w = k4.weight[j] * n3;
                    ss.push((edi * 0.5 + exi) * w);
                    os.push(edi * 0.5 * w);
                }
                Ok((oracle_sum(&ss), oracle_sum(&os)))
            })
            .collect();
        per_group.extend(chunk?);
    }

    // `1 / nkpts` — the FULL-BZ nkpts, never nkpts_ibz. The IBZ weights
    // entered through the class multiplicity above, not through the prefactor.
    let ss = oracle_sum(&per_group.iter().map(|x| x.0).collect::<Vec<_>>()) / nk as f64;
    let os = oracle_sum(&per_group.iter().map(|x| x.1).collect::<Vec<_>>()) / nk as f64;
    Ok((ss, os))
}

// =====================================================================
// Task 2 — the density matrices (`kmp2_ksymm.py:129-225`)
// =====================================================================

/// The partially filled `t2` of `make_t2_for_rdm1` (`kmp2_ksymm.py:161`).
///
/// Upstream allocates it with `np.empty` and fills only the blocks
/// `_gamma1_intermediates` will read — `ki <= kj`, and at least one of
/// `ki, kj, ka, kb` in the IBZ. Every other block holds whatever was in that
/// page. This port stores `Option<CTensor>` instead, so a read of a block that
/// was never built is a **loud error** rather than an arbitrary number: an
/// indexing mistake here would otherwise produce a plausible density matrix.
#[derive(Debug, Clone)]
pub struct PartialT2 {
    pub nkpts: usize,
    pub nocc: usize,
    pub nvir: usize,
    blocks: Vec<Option<CTensor>>,
}

impl PartialT2 {
    fn flat(&self, ki: usize, kj: usize, ka: usize) -> usize {
        (ki * self.nkpts + kj) * self.nkpts + ka
    }

    /// # Errors
    /// [`PbcMpError::Shape`] if the block was never built.
    pub fn block(&self, ki: usize, kj: usize, ka: usize) -> Result<&CTensor, PbcMpError> {
        self.blocks[self.flat(ki, kj, ka)]
            .as_ref()
            .ok_or_else(|| PbcMpError::Shape {
                what: format!("kmp2_ksymm: t2[{ki},{kj},{ka}] was never built"),
            })
    }

    /// Number of blocks actually materialised — the saving `make_t2_for_rdm1`
    /// exists for, reported rather than asserted.
    pub fn filled(&self) -> usize {
        self.blocks.iter().filter(|b| b.is_some()).count()
    }

    /// Promote a dense full-BZ [`T2`] (from [`KsymAdaptedKmp2::kernel_with_t2`]
    /// or the plain [`Kmp2`]) so the same `_gamma1_intermediates` can read it.
    pub fn from_full(t2: &T2) -> Self {
        Self {
            nkpts: t2.nkpts,
            nocc: t2.nocc,
            nvir: t2.nvir,
            blocks: t2.blocks.iter().cloned().map(Some).collect(),
        }
    }
}

/// `kmp2_ksymm.py:145-189` — `make_t2_for_rdm1`.
///
/// Builds only the `t2` blocks `_gamma1_intermediates` reads: `kj >= ki`
/// (`:166`) and at least one of `ki, kj, ka, kb` in the IBZ (`:170`).
///
/// # Errors
/// As the ERI route, plus [`PbcMpError::Shape`] on a degenerate MO space.
pub fn make_t2_for_rdm1(
    mp: &Kmp2<'_>,
    kpts: &KPoints,
    _cell: &Cell,
    padded: &PaddedMos,
    route: EriRoute,
) -> Result<PartialT2, PbcMpError> {
    let nk = mp.kpts.len();
    let (nocc, nmo) = (padded.nocc, padded.nmo);
    if nocc == 0 || nocc >= nmo {
        return Err(PbcMpError::Shape {
            what: "KMP2 needs non-empty occupied and virtual spaces".into(),
        });
    }
    let nvir = nmo - nocc;
    let nov2 = (nocc * nvir).saturating_pow(2);
    let split = match padding_k_idx(
        &padded.nmo_per_kpt,
        &padded.nocc_per_kpt,
        PaddingKind::Split,
    )? {
        PaddingIdx::Split { occupied, virtuals } => (occupied, virtuals),
        PaddingIdx::Joint(_) => unreachable!(),
    };
    let (lov, caches) = route_setup(mp, padded, route)?;
    let in_ibz = |k: usize| kpts.ibz2bz.contains(&k);

    let wanted: Vec<(usize, usize, usize, usize)> = (0..nk)
        .flat_map(|ki| (ki..nk).flat_map(move |kj| (0..nk).map(move |ka| (ki, kj, ka))))
        .filter_map(|(ki, kj, ka)| {
            let kb = mp.khelper.kconserv.get(ki, ka, kj) as usize;
            (in_ibz(ki) || in_ibz(kj) || in_ibz(ka) || in_ibz(kb)).then_some((ki, kj, ka, kb))
        })
        .collect();

    let built: Result<Vec<_>, PbcMpError> = wanted
        .par_iter()
        .map(|&(ki, kj, ka, kb)| {
            let oovv = match lov.as_ref() {
                Some(l) => crate::kmp2_kernel::df_oovv(l, ki, ka, kj, kb),
                None => crate::kmp2_kernel::ao2mo_oovv(
                    mp.with_df,
                    &padded.mo_coeff,
                    ki,
                    ka,
                    kj,
                    kb,
                    nocc,
                    nmo,
                    nk,
                    caches.as_ref().map(|c| c[ki * nk + ka].as_ref()),
                ),
            }?;
            let mut t = CTensor::zeros(nov2);
            for i in 0..nocc {
                for j in 0..nocc {
                    for a in 0..nvir {
                        for b in 0..nvir {
                            let p = ((i * nocc + j) * nvir + a) * nvir + b;
                            let d =
                                denom(padded, &split.0, &split.1, nocc, ki, ka, kj, kb, i, j, a, b);
                            t.re[p] = oovv.re[p] / d;
                            t.im[p] = -oovv.im[p] / d;
                        }
                    }
                }
            }
            Ok(((ki, kj, ka), t))
        })
        .collect();

    let mut out = PartialT2 {
        nkpts: nk,
        nocc,
        nvir,
        blocks: vec![None; nk * nk * nk],
    };
    for ((ki, kj, ka), t) in built? {
        let f = out.flat(ki, kj, ka);
        out.blocks[f] = Some(t);
    }
    Ok(out)
}

fn t_idx(no: usize, nv: usize, i: usize, j: usize, a: usize, b: usize) -> usize {
    ((i * no + j) * nv + a) * nv + b
}

/// `conj(x[xi]) * y[yi]`.
fn prod_conj(x: &CTensor, xi: usize, y: &CTensor, yi: usize) -> (f64, f64) {
    (
        x.re[xi] * y.re[yi] + x.im[xi] * y.im[yi],
        x.re[xi] * y.im[yi] - x.im[xi] * y.re[yi],
    )
}

/// `kmp2_ksymm.py:191-223` — `_gamma1_intermediates`, `nkpts_ibz` blocks.
///
/// The `ki > kj` half of the `t2` array was never built, so upstream
/// reconstructs it (`:214-216`) from the `kj < ki` half by the amplitude
/// symmetry `t2[ki,kj,ka][i,j,a,b] = t2[kj,ki,kb][j,i,b,a]`. That transpose is
/// the one place a wrong index would produce a *plausible* density matrix, so
/// it is written out element-wise here rather than folded into a helper.
///
/// # Errors
/// [`PbcMpError::Shape`] if the amplitude set is missing a block this needs.
pub fn gamma1_intermediates_ksymm(
    t2: &PartialT2,
    kconserv: &pyscf_pbc_lib::Kconserv,
    kpts: &KPoints,
) -> Result<(Vec<CTensor>, Vec<CTensor>), PbcMpError> {
    let (nk, no, nv) = (t2.nkpts, t2.nocc, t2.nvir);
    let nibz = kpts.nkpts_ibz();
    let mut doo = vec![CTensor::zeros(no * no); nibz];
    let mut dvv = vec![CTensor::zeros(nv * nv); nibz];
    // `t2a[i,j,a,b]` and `t2b[i,j,a,b]` for one (ki, kj, ka) — either read
    // straight out of the store, or transposed out of the (kj, ki) half.
    let fetch =
        |ki: usize, kj: usize, ka: usize, kb: usize| -> Result<(CTensor, CTensor), PbcMpError> {
            if ki <= kj {
                return Ok((t2.block(ki, kj, ka)?.clone(), t2.block(ki, kj, kb)?.clone()));
            }
            let src_a = t2.block(kj, ki, kb)?;
            let src_b = t2.block(kj, ki, ka)?;
            let mut a = CTensor::zeros(no * no * nv * nv);
            let mut b = CTensor::zeros(no * no * nv * nv);
            for i in 0..no {
                for j in 0..no {
                    for x in 0..nv {
                        for y in 0..nv {
                            let dst = t_idx(no, nv, i, j, x, y);
                            let src = t_idx(no, nv, j, i, y, x);
                            a.re[dst] = src_a.re[src];
                            a.im[dst] = src_a.im[src];
                            b.re[dst] = src_b.re[src];
                            b.im[dst] = src_b.im[src];
                        }
                    }
                }
            }
            Ok((a, b))
        };

    // The accumulation is `nkpts^3 * nocc^2 * nvir^2` terms into one output
    // element — a D-PBC-17 shape — so every element is an `oracle_sum` over a
    // vector built in a fixed index order.
    for kout in 0..nibz {
        let k_bz = kpts.ibz2bz[kout];
        for x in 0..nv {
            for y in 0..nv {
                let mut re = Vec::new();
                let mut im = Vec::new();
                for ki in 0..nk {
                    for kj in 0..nk {
                        for ka in 0..nk {
                            let kb = kconserv.get(ki, ka, kj) as usize;
                            if kb != k_bz {
                                continue;
                            }
                            let (a, b) = fetch(ki, kj, ka, kb)?;
                            for i in 0..no {
                                for j in 0..no {
                                    for c in 0..nv {
                                        let lhs = t_idx(no, nv, i, j, c, x);
                                        let (dr, di) =
                                            prod_conj(&a, lhs, &a, t_idx(no, nv, i, j, c, y));
                                        let (er, ei) =
                                            prod_conj(&a, lhs, &b, t_idx(no, nv, i, j, y, c));
                                        re.push(2.0 * dr - er);
                                        im.push(2.0 * di - ei);
                                    }
                                }
                            }
                        }
                    }
                }
                dvv[kout].re[y * nv + x] = oracle_sum(&re);
                dvv[kout].im[y * nv + x] = oracle_sum(&im);
            }
        }
        for x in 0..no {
            for y in 0..no {
                let mut re = Vec::new();
                let mut im = Vec::new();
                for ki in 0..nk {
                    for ka in 0..nk {
                        let kj = k_bz;
                        let kb = kconserv.get(ki, ka, kj) as usize;
                        let (a, b) = fetch(ki, kj, ka, kb)?;
                        for i in 0..no {
                            for av in 0..nv {
                                for bv in 0..nv {
                                    let lhs = t_idx(no, nv, i, x, av, bv);
                                    let (dr, di) =
                                        prod_conj(&a, lhs, &a, t_idx(no, nv, i, y, av, bv));
                                    let (er, ei) =
                                        prod_conj(&a, lhs, &b, t_idx(no, nv, i, y, bv, av));
                                    re.push(2.0 * dr - er);
                                    im.push(2.0 * di - ei);
                                }
                            }
                        }
                    }
                }
                doo[kout].re[x * no + y] = -oracle_sum(&re);
                doo[kout].im[x * no + y] = -oracle_sum(&im);
            }
        }
    }
    Ok((doo, dvv))
}

/// `kmp2_ksymm.py:128-143` — `make_rdm1`. Returns `nkpts_ibz` blocks.
///
/// See the module doc, deviation 2, for the padding-index difference from
/// upstream.
///
/// # Errors
/// As [`gamma1_intermediates_ksymm`] and [`padding_k_idx`].
pub fn make_rdm1_ksymm(
    t2: &PartialT2,
    kconserv: &pyscf_pbc_lib::Kconserv,
    kpts: &KPoints,
    nmo_per_kpt: &[usize],
    nocc_per_kpt: &[usize],
    kind: RdmKind,
) -> Result<Vec<CTensor>, PbcMpError> {
    let (doo, dvv) = gamma1_intermediates_ksymm(t2, kconserv, kpts)?;
    let nmo = t2.nocc + t2.nvir;
    let joint = match padding_k_idx(nmo_per_kpt, nocc_per_kpt, PaddingKind::Joint)? {
        PaddingIdx::Joint(v) => v,
        PaddingIdx::Split { .. } => unreachable!(),
    };
    let mut out = Vec::with_capacity(kpts.nkpts_ibz());
    for i in 0..kpts.nkpts_ibz() {
        let mut d = CTensor::zeros(nmo * nmo);
        for p in 0..t2.nocc {
            for q in 0..t2.nocc {
                let z = p * t2.nocc + q;
                d.re[p * nmo + q] = doo[i].re[z] + f64::from(u8::from(p == q));
                d.im[p * nmo + q] = doo[i].im[z];
            }
        }
        for p in 0..t2.nvir {
            for q in 0..t2.nvir {
                let z = p * t2.nvir + q;
                d.re[(t2.nocc + p) * nmo + t2.nocc + q] = dvv[i].re[z];
                d.im[(t2.nocc + p) * nmo + t2.nocc + q] = dvv[i].im[z];
            }
        }
        // `d += d.conj().T`
        let before = d.clone();
        for p in 0..nmo {
            for q in 0..nmo {
                d.re[p * nmo + q] += before.re[q * nmo + p];
                d.im[p * nmo + q] -= before.im[q * nmo + p];
            }
        }
        if kind == RdmKind::Padded {
            out.push(d);
            continue;
        }
        let idx = &joint[kpts.ibz2bz[i]];
        let n = idx.len();
        let mut c = CTensor::zeros(n * n);
        for (a, &p) in idx.iter().enumerate() {
            for (b, &q) in idx.iter().enumerate() {
                c.re[a * n + b] = d.re[p * nmo + q];
                c.im[a * n + b] = d.im[p * nmo + q];
            }
        }
        out.push(c);
    }
    Ok(out)
}
